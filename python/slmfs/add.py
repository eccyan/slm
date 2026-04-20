"""Online bulk ingestion into a running engine.

Usage: python -m slmfs.add <file.md> [file2.md ...] [--shm-name=name]

Bypasses FUSE — streams chunks directly via shared memory with
backpressure handling when the slab pool is temporarily full.
"""

import sys
import time
from pathlib import Path

from .config import SlmfsConfig
from .cooker import cook_write, CMD_WRITE_COMMIT
from .embedder import MiniLMEmbedder
from .init import parse_markdown, Chunk
from .shm_client import ShmClient


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
        shm.push_handle(handle)
        ingested += 1

    return ingested


def main():
    """Entry point: python -m slmfs.add <file.md> [--shm-name=name]"""
    config = SlmfsConfig()
    paths: list[Path] = []

    for arg in sys.argv[1:]:
        if arg.startswith("--shm-name="):
            config.shm_name = arg.split("=", 1)[1]
        else:
            paths.append(Path(arg))

    if not paths:
        print("Usage: python -m slmfs.add <file.md> [file2.md ...]")
        print("       python -m slmfs.add --shm-name=slmfs_shm *.md")
        sys.exit(1)

    print(f"Connecting to engine via shm: {config.shm_name}")
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


if __name__ == "__main__":
    main()
