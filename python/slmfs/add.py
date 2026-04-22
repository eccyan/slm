"""Online bulk ingestion into a running engine.

Usage: python -m slmfs add <file.md> [file2.md ...] [--shm-path=path]
       python -m slmfs add --stop-fuse ~/docs/**/*.md

Bypasses FUSE — streams chunks directly via shared memory with
backpressure handling when the slab pool is temporarily full.

The --stop-fuse flag automatically stops the FUSE service before
ingestion and restarts it afterward (requires launchctl on macOS
or systemctl on Linux).
"""

import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

from .config import SlmfsConfig
from .cooker import cook_write, CMD_WRITE_COMMIT
from .embedder import MiniLMEmbedder
from .init import parse_markdown, Chunk
from .shm_client import ShmClient

_PLIST_NAME = "com.eccyan.slmfs-fuse.plist"
_SYSTEMD_UNIT = "slmfs-fuse"


def _fuse_plist() -> Path:
    return Path.home() / "Library/LaunchAgents" / _PLIST_NAME


def _is_fuse_mounted(mount_point: str = "/.agent_memory") -> bool:
    """Check if the FUSE mount is active."""
    try:
        result = subprocess.run(
            ["mount"], capture_output=True, text=True, timeout=5,
        )
        return mount_point in result.stdout
    except Exception:
        return False


def _wait_for_fuse_unmount(timeout: float = 10.0) -> bool:
    """Poll until FUSE is no longer mounted."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _is_fuse_mounted():
            return True
        time.sleep(0.5)
    return False


def _wait_for_fuse_mount(timeout: float = 10.0) -> bool:
    """Poll until FUSE is mounted."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _is_fuse_mounted():
            return True
        time.sleep(0.5)
    return False


def _stop_fuse() -> bool:
    """Stop the FUSE service. Returns True on success."""
    if platform.system() == "Darwin":
        plist = _fuse_plist()
        if not plist.exists():
            print("  WARNING: FUSE plist not found, skipping stop")
            return False
        result = subprocess.run(
            ["launchctl", "unload", str(plist)],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            print(f"  WARNING: launchctl unload failed: {result.stderr.strip()}")
            return False
        if not _wait_for_fuse_unmount():
            print("  WARNING: FUSE did not unmount within timeout")
            return False
        return True
    elif shutil.which("systemctl"):
        result = subprocess.run(
            ["systemctl", "--user", "stop", _SYSTEMD_UNIT],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            print(f"  WARNING: systemctl stop failed: {result.stderr.strip()}")
            return False
        if not _wait_for_fuse_unmount():
            print("  WARNING: FUSE did not unmount within timeout")
            return False
        return True
    else:
        print("  WARNING: no service manager found (launchctl/systemctl)")
        return False


def _start_fuse() -> bool:
    """Restart the FUSE service. Returns True on success."""
    if platform.system() == "Darwin":
        plist = _fuse_plist()
        if not plist.exists():
            print("  WARNING: FUSE plist not found, cannot restart")
            return False
        result = subprocess.run(
            ["launchctl", "load", str(plist)],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            print(f"  WARNING: launchctl load failed: {result.stderr.strip()}")
            return False
        if not _wait_for_fuse_mount():
            print("  WARNING: FUSE did not mount within timeout")
            return False
        return True
    elif shutil.which("systemctl"):
        result = subprocess.run(
            ["systemctl", "--user", "start", _SYSTEMD_UNIT],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            print(f"  WARNING: systemctl start failed: {result.stderr.strip()}")
            return False
        if not _wait_for_fuse_mount():
            print("  WARNING: FUSE did not mount within timeout")
            return False
        return True
    else:
        print("  WARNING: no service manager found (launchctl/systemctl)")
        return False


def add_file(
    path: Path,
    embedder: MiniLMEmbedder,
    shm: ShmClient,
    max_retries: int = 100,
    retry_delay: float = 0.05,
):
    """Parse, embed, and stream a Markdown file into the running engine."""
    chunks = parse_markdown(path)
    if not chunks:
        print(f"  No chunks found in {path}")
        return 0

    texts = [c.text for c in chunks]
    embeddings = embedder.embed_batch(texts)

    ingested = 0
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        payload = cook_write(
            chunk.text, embedding, parent_id=0, depth=chunk.depth
        )

        slab_idx = None
        for attempt in range(max_retries):
            slab_idx = shm.acquire_slab()
            if slab_idx is not None:
                break
            time.sleep(retry_delay)

        if slab_idx is None:
            print(
                f"  WARNING: slab pool exhausted after {max_retries} retries, "
                f"skipping chunk {i}"
            )
            continue

        shm.write_to_slab(slab_idx, payload)
        handle = (CMD_WRITE_COMMIT << 24) | slab_idx
        if not shm.push_handle_blocking(handle):
            print(f"  WARNING: ring buffer full, releasing slab for chunk {i}")
            shm.release_slab(slab_idx)
            continue
        ingested += 1

    return ingested


def main():
    """Entry point: python -m slmfs add <file.md> [--stop-fuse] [--shm-path=path]"""
    config = SlmfsConfig()
    paths: list[Path] = []
    stop_fuse = False

    for arg in sys.argv[1:]:
        if arg.startswith("--shm-path="):
            config.shm_path = Path(arg.split("=", 1)[1]).expanduser()
        elif arg == "--stop-fuse":
            stop_fuse = True
        else:
            paths.append(Path(arg))

    if not paths:
        print("Usage: python -m slmfs add <file.md> [file2.md ...]")
        print("       python -m slmfs add --stop-fuse ~/docs/**/*.md")
        sys.exit(1)

    fuse_stopped = False
    if stop_fuse:
        print("Stopping FUSE service...")
        fuse_stopped = _stop_fuse()

    try:
        print(f"Connecting to engine via shm: {config.shm_path}")
        embedder = MiniLMEmbedder()
        shm = ShmClient(config)

        total = 0
        for p in paths:
            print(f"  Ingesting {p}...")
            count = add_file(p, embedder, shm)
            print(f"    {count} chunks streamed")
            total += count

        shm.close()
        print(f"Done. {total} chunks ingested into running engine.")
    finally:
        if stop_fuse:
            # Always attempt restart when --stop-fuse was requested,
            # even if _stop_fuse failed (best-effort recovery).
            print("Restarting FUSE service...")
            if not _start_fuse():
                print("  ERROR: FUSE service failed to restart. "
                      "Run manually: launchctl load ~/Library/LaunchAgents/"
                      f"{_PLIST_NAME}")


if __name__ == "__main__":
    main()
