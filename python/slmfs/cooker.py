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
