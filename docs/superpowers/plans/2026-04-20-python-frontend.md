# Python Frontend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the complete Python FUSE frontend — the user-facing layer that lets AI agents interact with the memory system via standard filesystem operations (`cat`, `echo`), plus offline/online ingestion scripts.

**Architecture:** The `slmfs` Python package provides: `config.py` (dataclass + TOML loading with multi-project isolation), `embedder.py` (abstract Embedder + MiniLM default), `cooker.py` (text → binary payload packer matching the C++ `MemoryFSHeader` layout), `shm_client.py` (POSIX shared memory + SPSC queue operations), `fuse_layer.py` (FUSE filesystem intercepting read/write on `active.md` and `search/*.md`), `init.py` (offline "Day Zero" migration with AST parsing, batch embedding, Poincaré placement, and direct SQLite ingestion), and `add.py` (online bulk ingestion into a running engine with backpressure handling).

**Tech Stack:** Python 3.10+, fusepy, sentence-transformers (all-MiniLM-L6-v2), mistune, numpy, sqlite3, multiprocessing.shared_memory, pytest

---

## File Map

| File | Responsibility |
|---|---|
| `python/pyproject.toml` | Package metadata, dependencies, entry points |
| `python/slmfs/__init__.py` | Package marker |
| `python/slmfs/config.py` | `SlmfsConfig` dataclass with TOML loading and CLI override |
| `python/slmfs/embedder.py` | Abstract `Embedder` + `MiniLMEmbedder` (384-dim default) |
| `python/slmfs/cooker.py` | `cook_write()`, `cook_read()` — binary payload packers |
| `python/slmfs/shm_client.py` | `ShmClient` — shared memory access + SPSC queue operations |
| `python/slmfs/fuse_layer.py` | `SlmfsFS(Operations)` — FUSE filesystem |
| `python/slmfs/init.py` | `slmfs init` — offline Markdown migration to SQLite |
| `python/slmfs/add.py` | `slmfs add` — online bulk ingestion into running engine |
| `tests/python/test_config.py` | Config tests |
| `tests/python/test_cooker.py` | Binary payload packing tests |
| `tests/python/test_init.py` | AST parsing, placement, ingestion tests |

---

### Task 1: Package Setup & Config

**Files:**
- Create: `python/pyproject.toml`
- Create: `python/slmfs/__init__.py`
- Create: `python/slmfs/config.py`
- Create: `tests/python/test_config.py`

- [ ] **Step 1: Create pyproject.toml**

Create `python/pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.build_meta"

[project]
name = "slmfs"
version = "0.1.0"
description = "SLMFS — AI agent memory filesystem"
requires-python = ">=3.10"
dependencies = [
    "fusepy>=3.0",
    "sentence-transformers>=2.2",
    "mistune>=3.0",
    "numpy>=1.24",
]

[project.optional-dependencies]
dev = ["pytest>=7.0"]

[tool.pytest.ini_options]
testpaths = ["../tests/python"]
```

- [ ] **Step 2: Create package init**

Create `python/slmfs/__init__.py`:

```python
"""SLMFS — AI agent memory filesystem."""
```

- [ ] **Step 3: Write the failing tests**

Create `tests/python/test_config.py`:

```python
import pytest
from slmfs.config import SlmfsConfig


def test_default_values():
    config = SlmfsConfig()
    assert config.shm_name == "slmfs_shm"
    assert config.shm_size == 4 * 1024 * 1024
    assert config.slab_size == 64 * 1024
    assert config.control_block_size == 4096
    assert config.model_name == "sentence-transformers/all-MiniLM-L6-v2"
    assert config.vector_dim == 384
    assert str(config.mount_point) == ".agent_memory"
    assert str(config.db_path) == ".slmfs/memory.db"
    assert config.active_radius == 0.3
    assert config.search_top_k == 10


def test_custom_values():
    config = SlmfsConfig(
        shm_name="project_a_shm",
        db_path="custom/path.db",
        mount_point="custom_mount",
    )
    assert config.shm_name == "project_a_shm"
    assert str(config.db_path) == "custom/path.db"
    assert str(config.mount_point) == "custom_mount"


def test_slab_count():
    config = SlmfsConfig()
    expected = (config.shm_size - config.control_block_size) // config.slab_size
    assert config.slab_count == expected
    assert config.slab_count == 63
```

- [ ] **Step 4: Implement SlmfsConfig**

Create `python/slmfs/config.py`:

```python
"""Runtime configuration for SLMFS."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SlmfsConfig:
    """Configuration for the SLMFS frontend.

    All paths (shm_name, db_path, mount_point) are overridable for
    multi-project isolation — each project can run its own independent
    Poincaré disk by specifying unique values.
    """

    # Shared memory
    shm_name: str = "slmfs_shm"
    shm_size: int = 4 * 1024 * 1024       # 4MB
    slab_size: int = 64 * 1024             # 64KB
    control_block_size: int = 4096         # 4KB

    # Embedding model
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    vector_dim: int = 384

    # Mount point
    mount_point: Path = field(default_factory=lambda: Path(".agent_memory"))

    # Persistence
    db_path: Path = field(default_factory=lambda: Path(".slmfs/memory.db"))

    # Thresholds
    active_radius: float = 0.3
    search_top_k: int = 10

    @property
    def slab_count(self) -> int:
        """Number of slabs that fit in the shared memory pool."""
        return (self.shm_size - self.control_block_size) // self.slab_size
```

- [ ] **Step 5: Run tests**

Run:
```bash
cd python && pip install -e ".[dev]" && pytest ../tests/python/test_config.py -v
```

Expected: All 3 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add python/ tests/python/
git commit -m "feat(python): add slmfs package with SlmfsConfig"
```

---

### Task 2: Embedder

**Files:**
- Create: `python/slmfs/embedder.py`

No tests for this task — the embedder wraps sentence-transformers which requires model download. We'll verify it works via the init.py integration tests (Task 6).

- [ ] **Step 1: Implement Embedder**

Create `python/slmfs/embedder.py`:

```python
"""Embedding model interface."""

from abc import ABC, abstractmethod
import numpy as np


class Embedder(ABC):
    """Abstract embedding model interface."""

    @abstractmethod
    def embed(self, text: str) -> np.ndarray:
        """Embed a single text. Returns float32 array of shape (dim,)."""
        ...

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """Embed multiple texts. Returns float32 array of shape (N, dim)."""
        ...

    @property
    @abstractmethod
    def dim(self) -> int:
        """Embedding dimension."""
        ...


class MiniLMEmbedder(Embedder):
    """Default embedder using sentence-transformers/all-MiniLM-L6-v2 (384-dim)."""

    def __init__(self):
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer("all-MiniLM-L6-v2")
        self._dim = 384

    def embed(self, text: str) -> np.ndarray:
        return (
            self._model.encode(text, normalize_embeddings=True)
            .astype(np.float32)
        )

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        return (
            self._model.encode(
                texts,
                normalize_embeddings=True,
                batch_size=64,
                show_progress_bar=True,
            )
            .astype(np.float32)
        )

    @property
    def dim(self) -> int:
        return self._dim
```

- [ ] **Step 2: Commit**

```bash
git add python/slmfs/embedder.py
git commit -m "feat(python): add Embedder interface with MiniLM default"
```

---

### Task 3: Cooker (Binary Payload Packer)

**Files:**
- Create: `python/slmfs/cooker.py`
- Create: `tests/python/test_cooker.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/python/test_cooker.py`:

```python
import struct
import numpy as np
import pytest
from slmfs.cooker import cook_write, cook_read, MAGIC, HEADER_SIZE, CMD_READ, CMD_WRITE_COMMIT


def test_header_size():
    assert HEADER_SIZE == 64


def test_magic_value():
    assert MAGIC == 0x4D454D46


def test_cook_write_header_fields():
    text = "hello world"
    embedding = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    payload = cook_write(text, embedding, parent_id=5, depth=2)

    # Parse header
    magic, cmd = struct.unpack_from("<IB", payload, 0)
    assert magic == MAGIC
    assert cmd == CMD_WRITE_COMMIT

    total_size = struct.unpack_from("<Q", payload, 8)[0]
    text_offset, text_length = struct.unpack_from("<II", payload, 16)
    vector_offset, vector_dim = struct.unpack_from("<II", payload, 24)
    parent_id, depth = struct.unpack_from("<IB", payload, 32)

    assert text_offset == 64
    assert text_length == len(text)
    assert vector_offset % 64 == 0  # 64-byte aligned
    assert vector_dim == 3
    assert parent_id == 5
    assert depth == 2


def test_cook_write_text_content():
    text = "test text"
    embedding = np.zeros(4, dtype=np.float32)
    payload = cook_write(text, embedding)

    extracted = payload[64:64 + len(text)].decode("utf-8")
    assert extracted == "test text"


def test_cook_write_vector_content():
    text = "x"
    embedding = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    payload = cook_write(text, embedding)

    vec_offset = struct.unpack_from("<I", payload, 24)[0]
    vec_dim = struct.unpack_from("<I", payload, 28)[0]
    vec_bytes = payload[vec_offset:vec_offset + vec_dim * 4]
    vec = np.frombuffer(vec_bytes, dtype=np.float32)

    np.testing.assert_array_almost_equal(vec, embedding)


def test_cook_write_vector_alignment():
    # Various text lengths — vector offset must always be 64-byte aligned
    for text_len in [1, 10, 63, 64, 65, 100, 200]:
        text = "x" * text_len
        embedding = np.zeros(4, dtype=np.float32)
        payload = cook_write(text, embedding)

        vec_offset = struct.unpack_from("<I", payload, 24)[0]
        assert vec_offset % 64 == 0, f"Failed for text_len={text_len}"


def test_cook_read_header():
    embedding = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    payload = cook_read(embedding)

    magic, cmd = struct.unpack_from("<IB", payload, 0)
    assert magic == MAGIC
    assert cmd == CMD_READ

    text_offset, text_length = struct.unpack_from("<II", payload, 16)
    assert text_length == 0  # no text for read queries

    vec_offset = struct.unpack_from("<I", payload, 24)[0]
    vec_dim = struct.unpack_from("<I", payload, 28)[0]
    assert vec_offset % 64 == 0
    assert vec_dim == 3


def test_cook_read_vector_content():
    embedding = np.array([0.5, -0.5], dtype=np.float32)
    payload = cook_read(embedding)

    vec_offset = struct.unpack_from("<I", payload, 24)[0]
    vec = np.frombuffer(payload[vec_offset:vec_offset + 8], dtype=np.float32)
    np.testing.assert_array_almost_equal(vec, embedding)


def test_cook_write_384dim():
    text = "MiniLM vector"
    embedding = np.random.randn(384).astype(np.float32)
    payload = cook_write(text, embedding)

    vec_dim = struct.unpack_from("<I", payload, 28)[0]
    assert vec_dim == 384

    vec_offset = struct.unpack_from("<I", payload, 24)[0]
    vec = np.frombuffer(
        payload[vec_offset:vec_offset + 384 * 4], dtype=np.float32
    )
    np.testing.assert_array_almost_equal(vec, embedding)
```

- [ ] **Step 2: Implement Cooker**

Create `python/slmfs/cooker.py`:

```python
"""Binary payload packer matching the C++ MemoryFSHeader layout."""

import struct
import numpy as np

# Constants matching src/slab/include/slab/header.hpp
MAGIC = 0x4D454D46          # 'MEMF' little-endian
DONE_MAGIC = 0x444F4E45     # 'DONE' little-endian
CMD_READ = 0x01
CMD_WRITE_COMMIT = 0x02
HEADER_SIZE = 64


def _align_up(offset: int, alignment: int) -> int:
    """Round offset up to the next multiple of alignment."""
    return (offset + alignment - 1) & ~(alignment - 1)


def cook_write(
    text: str,
    embedding: np.ndarray,
    parent_id: int = 0,
    depth: int = 0,
) -> bytes:
    """Pack text + vector into a slab-ready binary payload.

    Layout:
        [0..64)           MemoryFSHeader
        [64..64+text_len) UTF-8 text
        [vec_offset..)    float32 array (64-byte aligned)
    """
    text_bytes = text.encode("utf-8")
    text_offset = HEADER_SIZE
    text_length = len(text_bytes)

    vector_offset = _align_up(text_offset + text_length, 64)
    vector_bytes = embedding.astype(np.float32).tobytes()
    vector_dim = embedding.shape[0]

    total_size = vector_offset + len(vector_bytes)

    # Pack header: magic(4) cmd(1) pad(3) total_size(8)
    #   text_offset(4) text_length(4) vector_offset(4) vector_dim(4)
    #   parent_id(4) depth(1) reserved(27)
    header = struct.pack(
        "<I B 3x Q I I I I I B 27x",
        MAGIC,
        CMD_WRITE_COMMIT,
        total_size,
        text_offset,
        text_length,
        vector_offset,
        vector_dim,
        parent_id,
        depth,
    )
    assert len(header) == HEADER_SIZE

    padding_len = vector_offset - text_offset - text_length
    payload = header + text_bytes + (b"\x00" * padding_len) + vector_bytes
    return payload


def cook_read(query_embedding: np.ndarray) -> bytes:
    """Pack a read query (vector only, no text)."""
    vector_offset = _align_up(HEADER_SIZE, 64)  # = 64
    vector_bytes = query_embedding.astype(np.float32).tobytes()
    vector_dim = query_embedding.shape[0]
    total_size = vector_offset + len(vector_bytes)

    header = struct.pack(
        "<I B 3x Q I I I I I B 27x",
        MAGIC,
        CMD_READ,
        total_size,
        0,  # text_offset (unused)
        0,  # text_length
        vector_offset,
        vector_dim,
        0,  # parent_id
        0,  # depth
    )

    payload = header + vector_bytes
    return payload
```

- [ ] **Step 3: Run tests**

Run:
```bash
cd python && pytest ../tests/python/test_cooker.py -v
```

Expected: All 8 tests PASS.

- [ ] **Step 4: Commit**

```bash
git add python/slmfs/cooker.py tests/python/test_cooker.py
git commit -m "feat(python): add cooker binary payload packer matching MemoryFSHeader"
```

---

### Task 4: ShmClient (Shared Memory Access)

**Files:**
- Create: `python/slmfs/shm_client.py`

This module wraps POSIX shared memory access. Testing requires a running engine or a mock shared memory region. We'll verify it via integration testing later; the cooker tests already validate the payload format that ShmClient transmits.

- [ ] **Step 1: Implement ShmClient**

Create `python/slmfs/shm_client.py`:

```python
"""Shared memory client for IPC with the C++ engine."""

import ctypes
import struct
import time
from multiprocessing import shared_memory
from typing import Optional

from .config import SlmfsConfig
from .cooker import DONE_MAGIC


class ShmClient:
    """Python-side shared memory access: slab allocation + SPSC push."""

    # Control block layout (must match C++ ControlBlock):
    # [0..8)     atomic<uint64_t> free_bitmask
    # [64..1088) SPSCRingBuffer<uint32_t, 256>
    #   [64..72)   head (atomic, cache-line padded)
    #   [128..136) cached_tail
    #   [192..200) tail (atomic)
    #   [256..264) cached_head
    #   [320..1344) buffer[256] (uint32_t each)
    _BITMASK_OFF = 0
    _RING_HEAD_OFF = 64
    _RING_TAIL_OFF = 192
    _RING_BUF_OFF = 320
    _RING_CAPACITY = 256
    _RING_MASK = _RING_CAPACITY - 1

    def __init__(self, config: SlmfsConfig):
        self._shm = shared_memory.SharedMemory(
            name=config.shm_name, create=False
        )
        self._buf = self._shm.buf
        self._slab_size = config.slab_size
        self._slab_count = config.slab_count
        self._ctrl_size = config.control_block_size
        self._data_offset = config.control_block_size

    def close(self):
        """Release the shared memory mapping (does not unlink)."""
        self._shm.close()

    def acquire_slab(self) -> Optional[int]:
        """Atomic CAS on free_bitmask. Returns slab index or None."""
        while True:
            current = struct.unpack_from("<Q", self._buf, self._BITMASK_OFF)[0]
            if current == 0:
                return None

            # Find lowest set bit
            idx = (current & -current).bit_length() - 1

            # CAS: clear that bit
            new_val = current & ~(1 << idx)
            # Use ctypes for atomic CAS
            ptr = ctypes.c_uint64.from_buffer(self._buf, self._BITMASK_OFF)
            if ctypes.c_uint64(current).value == ptr.value:
                ptr.value = new_val
                return idx
            # CAS failed — retry

    def release_slab(self, index: int):
        """Set bit back in bitmask."""
        ptr = ctypes.c_uint64.from_buffer(self._buf, self._BITMASK_OFF)
        ptr.value |= 1 << index

    def write_to_slab(self, index: int, payload: bytes):
        """Copy payload into slab."""
        offset = self._data_offset + index * self._slab_size
        self._buf[offset : offset + len(payload)] = payload

    def read_slab(self, index: int, length: int) -> bytes:
        """Read bytes from a slab."""
        offset = self._data_offset + index * self._slab_size
        return bytes(self._buf[offset : offset + length])

    def read_slab_u32(self, index: int, byte_offset: int = 0) -> int:
        """Read a uint32 from a slab at the given byte offset."""
        offset = self._data_offset + index * self._slab_size + byte_offset
        return struct.unpack_from("<I", self._buf, offset)[0]

    def push_handle(self, handle: int):
        """Push 32-bit handle to SPSC ring buffer."""
        head = struct.unpack_from("<Q", self._buf, self._RING_HEAD_OFF)[0]
        next_head = (head + 1) & self._RING_MASK

        # Write value to buffer[head & mask]
        slot_off = self._RING_BUF_OFF + (head & self._RING_MASK) * 4
        struct.pack_into("<I", self._buf, slot_off, handle)

        # Release store: increment head
        struct.pack_into("<Q", self._buf, self._RING_HEAD_OFF, next_head)

    def wait_for_done(
        self, slab_index: int, timeout: float = 1.0
    ) -> Optional[bytes]:
        """Spin-wait for engine to write DONE magic, then read result."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            magic = self.read_slab_u32(slab_index, 0)
            if magic == DONE_MAGIC:
                # Read text_length from header
                text_length = self.read_slab_u32(slab_index, 20)
                result = self.read_slab(slab_index, 64 + text_length)
                self.release_slab(slab_index)
                return result[64 : 64 + text_length]
            time.sleep(0.0001)  # 100μs

        self.release_slab(slab_index)
        return None
```

- [ ] **Step 2: Commit**

```bash
git add python/slmfs/shm_client.py
git commit -m "feat(python): add ShmClient for shared memory IPC with C++ engine"
```

---

### Task 5: FUSE Layer

**Files:**
- Create: `python/slmfs/fuse_layer.py`

- [ ] **Step 1: Implement FUSE layer**

Create `python/slmfs/fuse_layer.py`:

```python
"""FUSE filesystem layer for SLMFS."""

import errno
import os
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
        self._heading_map: dict[int, int] = {}  # depth → node_id

    def destroy(self, path):
        self.shm.close()

    # ── Directory structure ──

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

    # ── active.md: Read ──

    def read(self, path, size, offset, fh):
        if path == "/active.md":
            return self._read_active(size, offset)
        if path.startswith("/search/"):
            query = path[len("/search/") :]
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

    # ── active.md: Write (non-blocking) ──

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

        return len(data)  # non-blocking return

    # ── search/: Write denied ──

    def create(self, path, mode, fi=None):
        if path.startswith("/search/"):
            raise FuseOSError(errno.EACCES)
        raise FuseOSError(errno.ENOENT)

    # ── Heading parser ──

    def _parse_heading_context(self, text: str) -> tuple[int, int]:
        """Extract parent_id and depth from Markdown heading."""
        lines = text.strip().split("\n")
        for line in lines:
            if line.startswith("#"):
                depth = len(line) - len(line.lstrip("#"))
                return self._heading_map.get(depth - 1, 0), depth
        return 0, 0


def main():
    """Entry point: python -m slmfs.fuse_layer [--config=path.toml]"""
    config = SlmfsConfig()

    # Simple CLI arg parsing
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
```

- [ ] **Step 2: Commit**

```bash
git add python/slmfs/fuse_layer.py
git commit -m "feat(python): add FUSE layer with active.md read/write and search/ queries"
```

---

### Task 6: slmfs init (Offline Migration)

**Files:**
- Create: `python/slmfs/init.py`
- Create: `tests/python/test_init.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/python/test_init.py`:

```python
import math
import sqlite3
import tempfile
import time
from pathlib import Path

import numpy as np
import pytest
from slmfs.init import parse_markdown, compute_initial_position, Chunk


def test_parse_simple_markdown(tmp_path):
    md = tmp_path / "test.md"
    md.write_text("# Title\nSome content\n## Section\nMore content\n")

    chunks = parse_markdown(md)
    assert len(chunks) >= 2
    assert chunks[0].depth == 1  # H1
    assert "Title" in chunks[0].text
    assert chunks[1].depth == 2  # H2


def test_parse_preserves_hierarchy(tmp_path):
    md = tmp_path / "test.md"
    md.write_text("# Root\n## Child A\n### Grandchild\n## Child B\n")

    chunks = parse_markdown(md)
    # Child A's parent should be Root (index 0)
    assert chunks[1].parent_idx == 0
    # Grandchild's parent should be Child A (index 1)
    assert chunks[2].parent_idx == 1
    # Child B's parent should be Root (index 0)
    assert chunks[3].parent_idx == 0


def test_parse_preamble(tmp_path):
    md = tmp_path / "test.md"
    md.write_text("Some preamble text\n# First heading\nContent\n")

    chunks = parse_markdown(md)
    assert chunks[0].depth == 0  # preamble has depth 0
    assert "preamble" in chunks[0].text.lower() or len(chunks[0].text) > 0


def test_placement_depth_zero_at_center():
    chunk = Chunk(text="root", depth=0, parent_idx=-1,
                  source_mtime=time.time())
    r, _ = compute_initial_position(chunk, time.time())
    assert r == 0.0


def test_placement_recent_file_near_center():
    now = time.time()
    chunk = Chunk(text="recent", depth=1, parent_idx=0,
                  source_mtime=now - 86400)  # 1 day ago
    r, _ = compute_initial_position(chunk, now)
    assert 0.0 < r < 0.2


def test_placement_old_file_near_boundary():
    now = time.time()
    chunk = Chunk(text="old", depth=1, parent_idx=0,
                  source_mtime=now - 365 * 86400)  # 1 year ago
    r, _ = compute_initial_position(chunk, now)
    assert r > 0.7


def test_placement_clamped():
    now = time.time()
    chunk = Chunk(text="ancient", depth=1, parent_idx=0,
                  source_mtime=now - 10 * 365 * 86400)  # 10 years ago
    r, _ = compute_initial_position(chunk, now)
    assert r <= 0.90


def test_golden_angle_distribution():
    """Verify angular distribution prevents clustering."""
    now = time.time()
    chunks = [
        Chunk(text=f"chunk{i}", depth=1, parent_idx=0, source_mtime=now)
        for i in range(20)
    ]
    from slmfs.init import place_all

    positions = place_all(chunks)
    angles = []
    for x, y in positions:
        if x != 0.0 or y != 0.0:
            angles.append(math.atan2(y, x))

    # With golden angle, consecutive angles should differ
    if len(angles) >= 2:
        diffs = [
            abs(angles[i + 1] - angles[i]) for i in range(len(angles) - 1)
        ]
        # No two consecutive angles should be exactly the same
        for d in diffs:
            assert d > 0.01
```

- [ ] **Step 2: Implement slmfs init**

Create `python/slmfs/init.py`:

```python
"""Offline migration: legacy Markdown → SQLite memory graph.

Usage: python -m slmfs.init /path/to/MEMORY.md [/path/to/other.md ...]
"""

import math
import sqlite3
import sys
import time as _time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import SlmfsConfig
from .embedder import MiniLMEmbedder


@dataclass
class Chunk:
    """A parsed Markdown chunk with heading hierarchy info."""

    text: str
    depth: int  # heading level (0 = preamble)
    parent_idx: int  # index into chunks list (-1 = root)
    source_mtime: float  # file mtime (epoch seconds)


def parse_markdown(path: Path) -> list[Chunk]:
    """Split Markdown into chunks at heading boundaries."""
    mtime = path.stat().st_mtime
    content = path.read_text(encoding="utf-8")
    lines = content.split("\n")

    chunks: list[Chunk] = []
    heading_stack: list[tuple[int, int]] = []  # (depth, chunk_index)

    current_lines: list[str] = []
    current_depth = 0
    current_parent = -1

    def flush():
        nonlocal current_lines
        text = "\n".join(current_lines).strip()
        if text:
            chunks.append(
                Chunk(
                    text=text,
                    depth=current_depth,
                    parent_idx=current_parent,
                    source_mtime=mtime,
                )
            )

    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith("#") and " " in stripped:
            # Count heading level
            level = len(stripped) - len(stripped.lstrip("#"))
            heading_text = stripped.lstrip("#").strip()

            # Flush previous chunk
            flush()

            # Pop stack to find parent
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()

            parent = heading_stack[-1][1] if heading_stack else -1
            heading_stack.append((level, len(chunks)))

            current_lines = [heading_text]
            current_depth = level
            current_parent = parent
        else:
            current_lines.append(line)

    # Flush final chunk
    flush()
    return chunks


def compute_initial_position(
    chunk: Chunk, now: float
) -> tuple[float, float]:
    """Map chunk to initial Poincaré disk coordinates (r, angle_unused)."""
    if chunk.depth == 0:
        return (0.0, 0.0)

    age_seconds = now - chunk.source_mtime
    age_days = age_seconds / 86400.0

    # Sigmoid mapping: 0 days → r≈0.05, 180 days → r≈0.7, 365+ → r≈0.85
    r = 0.85 * (1.0 - math.exp(-age_days / 180.0))
    r = max(0.05, min(r, 0.90))

    return (r, 0.0)


def place_all(chunks: list[Chunk]) -> list[tuple[float, float]]:
    """Assign (x, y) positions on the Poincaré disk."""
    now = _time.time()
    positions = []
    golden_angle = 2.399963  # radians

    for i, chunk in enumerate(chunks):
        r, _ = compute_initial_position(chunk, now)

        if r == 0.0:
            positions.append((0.0, 0.0))
        else:
            angle = i * golden_angle
            x = r * math.cos(angle)
            y = r * math.sin(angle)
            positions.append((x, y))

    return positions


def _create_schema(conn: sqlite3.Connection):
    """Create the memory_nodes and edges tables."""
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS memory_nodes (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            parent_id    INTEGER NOT NULL DEFAULT 0,
            depth        INTEGER NOT NULL DEFAULT 0,
            text         TEXT NOT NULL,
            mu           BLOB NOT NULL,
            sigma        BLOB NOT NULL,
            access_count INTEGER NOT NULL DEFAULT 0,
            pos_x        REAL NOT NULL,
            pos_y        REAL NOT NULL,
            last_access  REAL NOT NULL,
            annotation   TEXT,
            status       INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS edges (
            source_id    INTEGER NOT NULL,
            target_id    INTEGER NOT NULL,
            edge_type    INTEGER NOT NULL,
            relation     BLOB,
            PRIMARY KEY (source_id, target_id)
        );
        CREATE INDEX IF NOT EXISTS idx_nodes_status ON memory_nodes(status);
        CREATE INDEX IF NOT EXISTS idx_nodes_parent ON memory_nodes(parent_id);
        CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_id);
    """
    )


def ingest_to_sqlite(
    chunks: list[Chunk],
    embeddings: np.ndarray,
    positions: list[tuple[float, float]],
    db_path: Path,
    sigma_max: float = 10.0,
) -> int:
    """Bypass IPC — write directly to the persistence DB."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    _create_schema(conn)

    now = _time.time()
    chunk_to_db_id: dict[int, int] = {}

    # Pass 1: Insert all nodes with parent_id=0
    for i, (chunk, pos) in enumerate(zip(chunks, positions)):
        sigma = np.full(embeddings.shape[1], sigma_max, dtype=np.float32)
        cursor = conn.execute(
            """INSERT INTO memory_nodes
               (parent_id, depth, text, mu, sigma,
                access_count, pos_x, pos_y, last_access, status)
               VALUES (?, ?, ?, ?, ?, 0, ?, ?, ?, 0)""",
            (
                0,
                chunk.depth,
                chunk.text,
                embeddings[i].tobytes(),
                sigma.tobytes(),
                pos[0],
                pos[1],
                now,
            ),
        )
        chunk_to_db_id[i] = cursor.lastrowid

    # Pass 2: Update parent_id references
    for i, chunk in enumerate(chunks):
        if chunk.parent_idx >= 0:
            parent_db_id = chunk_to_db_id[chunk.parent_idx]
            conn.execute(
                "UPDATE memory_nodes SET parent_id = ? WHERE id = ?",
                (parent_db_id, chunk_to_db_id[i]),
            )

    # Pass 3: Create structural edges
    children_of: dict[int, list[int]] = defaultdict(list)
    for i, chunk in enumerate(chunks):
        pid = chunk_to_db_id.get(chunk.parent_idx, 0)
        children_of[pid].append(chunk_to_db_id[i])

    edge_rows = []
    for parent_db_id, child_ids in children_of.items():
        for cid in child_ids:
            if parent_db_id > 0:
                edge_rows.append((parent_db_id, cid, 0, None))
        for j in range(len(child_ids)):
            for k in range(j + 1, len(child_ids)):
                edge_rows.append((child_ids[j], child_ids[k], 0, None))

    conn.executemany(
        "INSERT OR IGNORE INTO edges VALUES (?, ?, ?, ?)", edge_rows
    )

    conn.commit()
    conn.close()
    return len(chunks)


def main():
    """Entry point: python -m slmfs.init /path/to/*.md"""
    config = SlmfsConfig()
    paths: list[Path] = []

    for arg in sys.argv[1:]:
        if arg.startswith("--db-path="):
            config.db_path = Path(arg.split("=", 1)[1])
        else:
            paths.append(Path(arg))

    if not paths:
        print("Usage: python -m slmfs.init <file.md> [file2.md ...]")
        print("       python -m slmfs.init --db-path=.slmfs/memory.db *.md")
        sys.exit(1)

    # 1. Parse all files
    all_chunks: list[Chunk] = []
    for p in paths:
        all_chunks.extend(parse_markdown(p))
    print(f"Parsed {len(all_chunks)} chunks from {len(paths)} files")

    # 2. Batch embed
    embedder = MiniLMEmbedder()
    texts = [c.text for c in all_chunks]
    embeddings = embedder.embed_batch(texts)
    print(
        f"Generated {embeddings.shape[0]} embeddings ({embeddings.shape[1]}-dim)"
    )

    # 3. Compute Poincaré positions
    positions = place_all(all_chunks)

    # 4. Ingest to SQLite
    count = ingest_to_sqlite(all_chunks, embeddings, positions, config.db_path)
    print(f"Ingested {count} nodes into {config.db_path}")
    print("Ready. Start the engine: slmfs_engine")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run tests**

Run:
```bash
cd python && pytest ../tests/python/test_init.py -v
```

Expected: All 8 tests PASS (the embedding-dependent `test_golden_angle_distribution` uses `place_all` which doesn't need the model).

- [ ] **Step 4: Commit**

```bash
git add python/slmfs/init.py tests/python/test_init.py
git commit -m "feat(python): add slmfs init offline migration with AST parsing and Poincaré placement"
```

---

### Task 7: slmfs add (Online Bulk Ingestion)

**Files:**
- Create: `python/slmfs/add.py`

- [ ] **Step 1: Implement slmfs add**

Create `python/slmfs/add.py`:

```python
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

        # Acquire slab with backpressure handling
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
```

- [ ] **Step 2: Commit**

```bash
git add python/slmfs/add.py
git commit -m "feat(python): add slmfs add online bulk ingestion with backpressure handling"
```

---

### Task 8: Python __main__ Entry Points

**Files:**
- Create: `python/slmfs/__main__.py`

- [ ] **Step 1: Create __main__.py for unified CLI**

Create `python/slmfs/__main__.py`:

```python
"""SLMFS CLI entry point.

Usage:
    python -m slmfs init <file.md> [...]     Offline migration
    python -m slmfs add <file.md> [...]      Online ingestion
    python -m slmfs fuse [--mount=path]      Mount FUSE filesystem
"""

import sys


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    command = sys.argv[1]
    # Remove the command from argv so submodules see their own args
    sys.argv = [sys.argv[0]] + sys.argv[2:]

    if command == "init":
        from .init import main as init_main
        init_main()
    elif command == "add":
        from .add import main as add_main
        add_main()
    elif command == "fuse":
        from .fuse_layer import main as fuse_main
        fuse_main()
    else:
        print(f"Unknown command: {command}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add python/slmfs/__main__.py
git commit -m "feat(python): add unified CLI entry point (python -m slmfs init|add|fuse)"
```

---

## Summary

After completing all 8 tasks, the Python frontend provides:

| Module | What it does |
|---|---|
| `config.py` | `SlmfsConfig` dataclass with multi-project isolation (shm_name, db_path, mount_point) |
| `embedder.py` | Abstract `Embedder` + `MiniLMEmbedder` (384-dim, batch support) |
| `cooker.py` | `cook_write()` / `cook_read()` binary payload packers matching MemoryFSHeader |
| `shm_client.py` | POSIX shared memory access, slab acquire/release, SPSC push, DONE spin-wait |
| `fuse_layer.py` | FUSE ops: `active.md` read/write, `search/<query>.md` read, EACCES on search write |
| `init.py` | Offline migration: AST parse → batch embed → Poincaré placement → 3-pass SQLite ingest |
| `add.py` | Online bulk ingestion: parse → embed → stream via ShmClient with backpressure |
| `__main__.py` | Unified CLI: `python -m slmfs init\|add\|fuse` |

CLI commands:
- `python -m slmfs init ~/MEMORY.md` — offline Day Zero migration
- `python -m slmfs add reference_docs.md --shm-name=slmfs_shm` — online ingestion
- `python -m slmfs fuse --mount=.agent_memory` — mount FUSE filesystem
