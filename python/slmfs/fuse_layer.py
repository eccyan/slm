"""FUSE filesystem layer for SLMFS."""

import errno
import sys
from pathlib import Path

import numpy as np
from fuse import FUSE, FuseOSError, Operations

from .config import SlmfsConfig
from .cooker import cook_write, cook_read, CMD_READ, CMD_WRITE_COMMIT
from .embedder import MiniLMEmbedder
from .shm_client import ShmClient


class SlmfsFS(Operations):
    """FUSE filesystem presenting .agent_memory/ with active.md and search/."""

    def __init__(self, config: SlmfsConfig):
        self.config = config
        self.embedder = MiniLMEmbedder()
        self.shm = ShmClient(config)
        self._heading_map: dict[int, int] = {}

    def destroy(self, path):
        self.shm.close()

    def getattr(self, path, fh=None):
        if path == "/":
            return dict(st_mode=0o40755, st_nlink=2)
        if path == "/active.md":
            return dict(st_mode=0o100644, st_nlink=1, st_size=4096)
        if path == "/search":
            return dict(st_mode=0o40555, st_nlink=2)
        if path.startswith("/search/") and path.endswith(".md"):
            return dict(st_mode=0o100444, st_nlink=1, st_size=4096)
        raise FuseOSError(errno.ENOENT)

    def readdir(self, path, fh):
        if path == "/":
            return [".", "..", "active.md", "search"]
        if path == "/search":
            return [".", ".."]
        raise FuseOSError(errno.ENOENT)

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
        zero_query = np.zeros(self.embedder.dim, dtype=np.float32)
        return self._submit_read(zero_query, size, offset)

    def _read_search(self, query: str, size, offset):
        embedding = self.embedder.embed(query.replace("_", " "))
        return self._submit_read(embedding, size, offset)

    def _submit_read(self, embedding, size, offset):
        payload = cook_read(embedding)
        slab_idx = self.shm.acquire_slab()
        if slab_idx is None:
            raise FuseOSError(errno.ENOMEM)
        self.shm.write_to_slab(slab_idx, payload)
        handle = (CMD_READ << 24) | slab_idx
        self.shm.push_handle(handle)
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
        self.shm.push_handle(handle)
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
    )


if __name__ == "__main__":
    main()
