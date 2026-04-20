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
