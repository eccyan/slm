"""FUSE filesystem layer for SLMFS."""

import errno
import os
import stat
import sys
import time
from pathlib import Path

import numpy as np
from fuse import FUSE, FuseOSError, Operations

from .config import SlmfsConfig
from .cooker import cook_write, cook_read, CMD_READ, CMD_READ_ACTIVE, CMD_WRITE_COMMIT
from .embedder import MiniLMEmbedder
from .shm_client import ShmClient


class SlmfsFS(Operations):
    """FUSE filesystem presenting .agent_memory/ with active.md and search/."""

    def __init__(self, config: SlmfsConfig):
        self.config = config
        self.embedder = MiniLMEmbedder()
        self.shm = ShmClient(config)
        self._heading_map: dict[int, int] = {}
        self._uid = os.getuid()
        self._gid = os.getgid()
        self._mount_time = time.time()

    def destroy(self, path):
        self.shm.close()

    def _base_stat(self, mode, size=0, nlink=1):
        now = self._mount_time
        return dict(
            st_mode=mode, st_nlink=nlink, st_size=size,
            st_uid=self._uid, st_gid=self._gid,
            st_atime=now, st_mtime=now, st_ctime=now,
        )

    def getattr(self, path, fh=None):
        if path == "/":
            return self._base_stat(stat.S_IFDIR | 0o755, nlink=2)
        if path == "/active.md":
            return self._base_stat(stat.S_IFREG | 0o644, size=4096)
        if path == "/search":
            return self._base_stat(stat.S_IFDIR | 0o555, nlink=2)
        if path.startswith("/search/") and path.endswith(".md"):
            return self._base_stat(stat.S_IFREG | 0o444, size=4096)
        raise FuseOSError(errno.ENOENT)

    def readdir(self, path, fh):
        if path == "/":
            return [".", "..", "active.md", "search"]
        if path == "/search":
            return [".", ".."]
        raise FuseOSError(errno.ENOENT)

    def open(self, path, flags):
        return 0

    def truncate(self, path, length, fh=None):
        pass

    def read(self, path, size, offset, fh):
        if path == "/active.md":
            return self._read_active(size, offset)
        if path.startswith("/search/"):
            query = path[len("/search/"):]
            if query.endswith(".md"):
                query = query[:-3]
            return self._read_search(query, size, offset)
        raise FuseOSError(errno.ENOENT)

    def _read_active(self, size, offset):
        # Passive read — no search, no activation, no mutation
        return self._submit_command(CMD_READ_ACTIVE, b"", size, offset)

    def _read_search(self, query: str, size, offset):
        # Search read — top-k + activate matched nodes
        embedding = self.embedder.embed(query.replace("_", " "))
        payload = cook_read(embedding)
        return self._submit_command(CMD_READ, payload, size, offset)

    def _submit_command(self, cmd, payload, size, offset):
        slab_idx = self.shm.acquire_slab()
        if slab_idx is None:
            raise FuseOSError(errno.ENOMEM)
        # Always clear the slab header to prevent stale DONE_MAGIC
        # from a previous read causing wait_for_done to return immediately
        self.shm.write_to_slab(slab_idx, b"\x00" * 64)
        if payload:
            self.shm.write_to_slab(slab_idx, payload)
        handle = (cmd << 24) | slab_idx
        if not self.shm.push_handle_blocking(handle):
            self.shm.release_slab(slab_idx)
            return b""
        result = self.shm.wait_for_done(slab_idx)
        if result is None:
            return b""
        return result[offset : offset + size]

    def write(self, path, data, offset, fh):
        if path != "/active.md":
            raise FuseOSError(errno.EACCES)
        text = data.decode("utf-8", errors="replace")
        parent_id, depth = self._parse_heading_context(text)
        embedding = self.embedder.embed(text)
        payload = cook_write(text, embedding, parent_id, depth)
        slab_idx = self.shm.acquire_slab()
        if slab_idx is None:
            raise FuseOSError(errno.ENOMEM)
        self.shm.write_to_slab(slab_idx, payload)
        handle = (CMD_WRITE_COMMIT << 24) | slab_idx
        if not self.shm.push_handle_blocking(handle):
            self.shm.release_slab(slab_idx)
            raise FuseOSError(errno.EBUSY)
        return len(data)

    def create(self, path, mode, fi=None):
        if path.startswith("/search/"):
            raise FuseOSError(errno.EACCES)
        raise FuseOSError(errno.ENOENT)

    def _parse_heading_context(self, text: str) -> tuple[int, int]:
        lines = text.strip().split("\n")
        for line in lines:
            if line.startswith("#"):
                depth = len(line) - len(line.lstrip("#"))
                return self._heading_map.get(depth - 1, 0), depth
        return 0, 0


def main():
    """Entry point: python -m slmfs.fuse_layer [--mount=path] [--shm-name=name]"""
    config = SlmfsConfig()
    for arg in sys.argv[1:]:
        if arg.startswith("--mount="):
            config.mount_point = Path(arg.split("=", 1)[1])
        elif arg.startswith("--shm-name="):
            config.shm_name = arg.split("=", 1)[1]

    config.mount_point.mkdir(parents=True, exist_ok=True)
    print(f"SLMFS mounting at {config.mount_point}")
    print(f"  shm_name: {config.shm_name}")

    FUSE(
        SlmfsFS(config),
        str(config.mount_point),
        foreground=True,
        nothreads=True,
        allow_other=False,
        volname="SLMFS",
        noappledouble=True,
        noapplexattr=True,
    )


if __name__ == "__main__":
    main()
