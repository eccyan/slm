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
