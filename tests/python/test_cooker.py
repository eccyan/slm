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

    magic, cmd = struct.unpack_from("<IB", payload, 0)
    assert magic == MAGIC
    assert cmd == CMD_WRITE_COMMIT

    total_size = struct.unpack_from("<Q", payload, 8)[0]
    text_offset, text_length = struct.unpack_from("<II", payload, 16)
    vector_offset, vector_dim = struct.unpack_from("<II", payload, 24)
    parent_id, depth = struct.unpack_from("<IB", payload, 32)

    assert text_offset == 64
    assert text_length == len(text)
    assert vector_offset % 64 == 0
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
    assert text_length == 0

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
