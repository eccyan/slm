"""Shared memory client for IPC with the C++ engine.

Safety model:
- Single Python producer assumed (FUSE runs with nothreads=True, or slmfs add
  runs standalone). Multiple concurrent Python producers are NOT supported.
- The C++ engine is the sole consumer of the SPSC queue.
- Slab acquire: only Python acquires (clears bits). C++ only releases (sets bits
  via atomic fetch_or). This means the Python load-check-store on the bitmask is
  safe because there is no concurrent acquirer.
- Slab release: Python sets bits (for READ response acknowledgement). C++ also
  sets bits. Both are idempotent OR operations on disjoint bits, so no conflict.
"""

import fcntl
import os
import struct
import tempfile
import time
from multiprocessing import shared_memory
from pathlib import Path
from typing import Optional

from .config import SlmfsConfig
from .cooker import DONE_MAGIC


class ShmClient:
    """Python-side shared memory access: slab allocation + SPSC push.

    IMPORTANT: Only one Python process should use this at a time.
    Running FUSE and slmfs-add simultaneously against the same shared
    memory is not supported (would require a real interprocess mutex).
    """

    # Control block layout (must match C++ ControlBlock):
    # [0..8)     atomic<uint64_t> free_bitmask
    # [64..72)   head (atomic, cache-line padded)
    # [128..136) cached_tail
    # [192..200) tail (atomic)
    # [256..264) cached_head
    # [320..1344) buffer[256] (uint32_t each)
    _BITMASK_OFF = 0
    _RING_HEAD_OFF = 64
    _RING_TAIL_OFF = 192
    _RING_BUF_OFF = 320
    _RING_CAPACITY = 256
    _RING_MASK = _RING_CAPACITY - 1

    def __init__(self, config: SlmfsConfig):
        # Enforce single-producer: acquire an exclusive file lock.
        # If another Python producer (FUSE or slmfs-add) is already running
        # against this shm_name, this will raise immediately.
        lock_path = Path(tempfile.gettempdir()) / f".slmfs_{config.shm_name}.lock"
        self._lock_file = open(lock_path, "w")
        try:
            fcntl.flock(self._lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self._lock_file.write(str(os.getpid()))
            self._lock_file.flush()
        except OSError:
            self._lock_file.close()
            raise RuntimeError(
                f"Another SLMFS producer is already connected to '{config.shm_name}'. "
                f"Only one Python producer (FUSE or slmfs-add) may run at a time."
            )

        self._shm = shared_memory.SharedMemory(
            name=config.shm_name, create=False
        )
        self._buf = self._shm.buf
        self._slab_size = config.slab_size
        self._slab_count = config.slab_count
        self._ctrl_size = config.control_block_size
        self._data_offset = config.control_block_size

    def close(self):
        """Release the shared memory mapping and producer lock."""
        self._shm.close()
        fcntl.flock(self._lock_file, fcntl.LOCK_UN)
        self._lock_file.close()

    def acquire_slab(self) -> Optional[int]:
        """Acquire a free slab. Returns slab index or None if pool exhausted.

        Safety: only one Python process may call this concurrently.
        The C++ engine never acquires slabs (only releases), so the
        single-producer load-check-store is race-free.
        """
        current = struct.unpack_from("<Q", self._buf, self._BITMASK_OFF)[0]
        if current == 0:
            return None

        # Find lowest set bit = first free slab
        idx = (current & -current).bit_length() - 1

        # Clear that bit (mark as in-use)
        new_val = current & ~(1 << idx)
        struct.pack_into("<Q", self._buf, self._BITMASK_OFF, new_val)
        return idx

    def release_slab(self, index: int):
        """Return a slab to the free pool.

        Safe for concurrent use: setting a bit is idempotent and the
        C++ engine sets disjoint bits (for WRITE slabs it released).
        """
        current = struct.unpack_from("<Q", self._buf, self._BITMASK_OFF)[0]
        new_val = current | (1 << index)
        struct.pack_into("<Q", self._buf, self._BITMASK_OFF, new_val)

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

    def push_handle(self, handle: int) -> bool:
        """Push 32-bit handle to SPSC ring buffer.

        Returns False if the ring buffer is full (caller should retry).
        Matches the C++ SPSC protocol: check tail before advancing head.
        """
        head = struct.unpack_from("<Q", self._buf, self._RING_HEAD_OFF)[0]
        next_head = (head + 1) & self._RING_MASK

        # Check if full: next_head == tail means the ring is full
        tail = struct.unpack_from("<Q", self._buf, self._RING_TAIL_OFF)[0]
        if next_head == tail:
            return False

        # Write value to buffer[head & mask]
        slot_off = self._RING_BUF_OFF + (head & self._RING_MASK) * 4
        struct.pack_into("<I", self._buf, slot_off, handle)

        # Release store: increment head
        struct.pack_into("<Q", self._buf, self._RING_HEAD_OFF, next_head)
        return True

    def push_handle_blocking(
        self, handle: int, timeout: float = 5.0, retry_delay: float = 0.001
    ) -> bool:
        """Push handle with retry on full ring. Returns False on timeout."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.push_handle(handle):
                return True
            time.sleep(retry_delay)
        return False

    def wait_for_done(
        self, slab_index: int, timeout: float = 1.0
    ) -> Optional[bytes]:
        """Spin-wait for engine to write DONE magic, then read and release slab."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            magic = self.read_slab_u32(slab_index, 0)
            if magic == DONE_MAGIC:
                text_length = self.read_slab_u32(slab_index, 20)
                result = self.read_slab(slab_index, 64 + text_length)
                self.release_slab(slab_index)
                return result[64 : 64 + text_length]
            time.sleep(0.0001)
        self.release_slab(slab_index)
        return None
