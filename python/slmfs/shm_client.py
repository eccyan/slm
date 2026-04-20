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
            idx = (current & -current).bit_length() - 1
            new_val = current & ~(1 << idx)
            ptr = ctypes.c_uint64.from_buffer(self._buf, self._BITMASK_OFF)
            if ctypes.c_uint64(current).value == ptr.value:
                ptr.value = new_val
                return idx

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
        slot_off = self._RING_BUF_OFF + (head & self._RING_MASK) * 4
        struct.pack_into("<I", self._buf, slot_off, handle)
        struct.pack_into("<Q", self._buf, self._RING_HEAD_OFF, next_head)

    def wait_for_done(
        self, slab_index: int, timeout: float = 1.0
    ) -> Optional[bytes]:
        """Spin-wait for engine to write DONE magic, then read result."""
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
