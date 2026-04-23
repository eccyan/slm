"""Tests for SLMFS MCP server tools."""

import math
import sqlite3
import struct
import tempfile
from pathlib import Path

import numpy as np
import pytest

SAMPLE_NODES = [
    {
        "text": "Deployment uses proxy at 10.0.0.5",
        "mu": [0.1] * 384,
        "sigma": [1.0] * 384,
        "pos_x": 0.05,
        "pos_y": 0.0,
        "access_count": 3,
        "status": 0,
    },
    {
        "text": "GSeurat collision spec committed",
        "mu": [0.2] * 384,
        "sigma": [1.0] * 384,
        "pos_x": 0.25,
        "pos_y": 0.1,
        "access_count": 1,
        "status": 0,
    },
    {
        "text": "Old migration notes from Q1",
        "mu": [0.3] * 384,
        "sigma": [1.0] * 384,
        "pos_x": 0.7,
        "pos_y": 0.3,
        "access_count": 0,
        "status": 0,
    },
    {
        "text": "Archived deployment log",
        "mu": [0.4] * 384,
        "sigma": [1.0] * 384,
        "pos_x": 0.0,
        "pos_y": 0.0,
        "access_count": 0,
        "status": 1,
    },
]


def _create_test_db(db_path: Path, nodes=None):
    """Create a test SQLite database with the memory_nodes schema."""
    if nodes is None:
        nodes = SAMPLE_NODES

    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """
        CREATE TABLE memory_nodes (
            id INTEGER PRIMARY KEY,
            parent_id INTEGER,
            depth INTEGER,
            text TEXT,
            mu BLOB,
            sigma BLOB,
            access_count INTEGER,
            pos_x REAL,
            pos_y REAL,
            last_access REAL,
            annotation TEXT,
            status INTEGER
        )
        """
    )
    for i, node in enumerate(nodes):
        dim = len(node["mu"])
        mu_blob = struct.pack(f"{dim}f", *node["mu"])
        sigma_blob = struct.pack(f"{dim}f", *node["sigma"])
        conn.execute(
            """
            INSERT INTO memory_nodes
                (id, parent_id, depth, text, mu, sigma, access_count,
                 pos_x, pos_y, last_access, annotation, status)
            VALUES (?, 0, 0, ?, ?, ?, ?, ?, ?, 0.0, NULL, ?)
            """,
            (
                i + 1,
                node["text"],
                mu_blob,
                sigma_blob,
                node["access_count"],
                node["pos_x"],
                node["pos_y"],
                node["status"],
            ),
        )
    conn.commit()
    conn.close()


# ── Task 2: read_active ──────────────────────────────────────────────


class TestReadActive:
    def test_returns_only_active_within_radius(self, tmp_path):
        db_path = tmp_path / "memory.db"
        _create_test_db(db_path)

        from slmfs.mcp_server import _read_active

        result = _read_active(str(db_path), active_radius=0.3)

        # Node 1 (r=0.05) and Node 2 (r~0.269) are within 0.3
        # Node 3 (r~0.762) is outside 0.3
        # Node 4 is archived (status=1)
        assert "Deployment uses proxy" in result
        assert "GSeurat collision spec" in result
        assert "Old migration notes" not in result
        assert "Archived deployment log" not in result

    def test_spatial_metadata_format(self, tmp_path):
        db_path = tmp_path / "memory.db"
        _create_test_db(db_path)

        from slmfs.mcp_server import _read_active

        result = _read_active(str(db_path), active_radius=0.3)

        assert "[r=0.05 | 3 accesses]" in result
        # Node 2: r = sqrt(0.25^2 + 0.1^2) = sqrt(0.0725) ~ 0.269
        assert "accesses]" in result

    def test_empty_db(self, tmp_path):
        db_path = tmp_path / "memory.db"
        _create_test_db(db_path, nodes=[])

        from slmfs.mcp_server import _read_active

        result = _read_active(str(db_path), active_radius=0.3)
        assert "No active memories" in result

    def test_read_only_mode(self, tmp_path):
        db_path = tmp_path / "memory.db"
        _create_test_db(db_path)

        from slmfs.mcp_server import _read_active

        # Should not raise — opens in read-only mode
        result = _read_active(str(db_path), active_radius=0.3)
        assert isinstance(result, str)


# ── Task 3: search_memory ────────────────────────────────────────────


class TestSearchMemory:
    def test_returns_ranked_results(self, tmp_path):
        db_path = tmp_path / "memory.db"
        _create_test_db(db_path)

        from slmfs.mcp_server import _search_memory

        # Query vector close to node 1 (mu=[0.1]*384)
        query_vec = np.array([0.1] * 384, dtype=np.float32)
        result = _search_memory(str(db_path), query_vec, top_k=3)

        lines = result.strip().split("\n")
        # First result should be node 1 (exact match, distance=0)
        assert "Deployment uses proxy" in lines[0]

    def test_fisher_rao_distance_ordering(self, tmp_path):
        db_path = tmp_path / "memory.db"
        _create_test_db(db_path)

        from slmfs.mcp_server import _search_memory

        query_vec = np.array([0.15] * 384, dtype=np.float32)
        result = _search_memory(str(db_path), query_vec, top_k=4)

        # Distance from [0.15] to [0.1] = sqrt(384*(0.05/1.0)^2) = sqrt(0.96) ~ 0.98
        # Distance from [0.15] to [0.2] = sqrt(384*(0.05/1.0)^2) = sqrt(0.96) ~ 0.98
        # Distance from [0.15] to [0.3] = sqrt(384*(0.15/1.0)^2) = sqrt(8.64) ~ 2.94
        # Distance from [0.15] to [0.4] = sqrt(384*(0.25/1.0)^2) = sqrt(24.0) ~ 4.90
        lines = [l for l in result.strip().split("\n") if l.strip()]
        assert "Deployment uses proxy" in lines[0] or "GSeurat collision spec" in lines[0]

    def test_active_vs_archived_labels(self, tmp_path):
        db_path = tmp_path / "memory.db"
        _create_test_db(db_path)

        from slmfs.mcp_server import _search_memory

        query_vec = np.array([0.4] * 384, dtype=np.float32)
        result = _search_memory(str(db_path), query_vec, top_k=4)

        assert "Active" in result
        assert "Archived" in result

    def test_archived_shows_r_na(self, tmp_path):
        db_path = tmp_path / "memory.db"
        _create_test_db(db_path)

        from slmfs.mcp_server import _search_memory

        query_vec = np.array([0.4] * 384, dtype=np.float32)
        result = _search_memory(str(db_path), query_vec, top_k=4)

        assert "r=N/A" in result

    def test_top_k_limits_results(self, tmp_path):
        db_path = tmp_path / "memory.db"
        _create_test_db(db_path)

        from slmfs.mcp_server import _search_memory

        query_vec = np.array([0.1] * 384, dtype=np.float32)
        result = _search_memory(str(db_path), query_vec, top_k=2)

        lines = [l for l in result.strip().split("\n") if l.strip()]
        assert len(lines) == 2


# ── Task 4: brain_status ─────────────────────────────────────────────


class TestBrainStatus:
    def test_total_counts(self, tmp_path):
        db_path = tmp_path / "memory.db"
        _create_test_db(db_path)

        from slmfs.mcp_server import _brain_status

        result = _brain_status(str(db_path))

        assert "Total: 4" in result
        assert "Active: 3" in result
        assert "Archived: 1" in result

    def test_header_format(self, tmp_path):
        db_path = tmp_path / "memory.db"
        _create_test_db(db_path)

        from slmfs.mcp_server import _brain_status

        result = _brain_status(str(db_path))
        assert result.startswith("## Brain Status")

    def test_spatial_distribution(self, tmp_path):
        db_path = tmp_path / "memory.db"
        _create_test_db(db_path)

        from slmfs.mcp_server import _brain_status

        result = _brain_status(str(db_path))

        # Node 1 (r=0.05) and Node 2 (r~0.269) are working (r<0.3)
        # Node 3 (r~0.762) is drifting (0.3<=r<0.8)
        assert "Working" in result
        assert "Drifting" in result

    def test_avg_radius(self, tmp_path):
        db_path = tmp_path / "memory.db"
        _create_test_db(db_path)

        from slmfs.mcp_server import _brain_status

        result = _brain_status(str(db_path))
        assert "Avg radius" in result

    def test_friction_ratio(self, tmp_path):
        db_path = tmp_path / "memory.db"
        _create_test_db(db_path)

        from slmfs.mcp_server import _brain_status

        result = _brain_status(str(db_path))
        assert "Friction" in result

    def test_empty_db(self, tmp_path):
        db_path = tmp_path / "memory.db"
        _create_test_db(db_path, nodes=[])

        from slmfs.mcp_server import _brain_status

        result = _brain_status(str(db_path))
        assert "Total: 0" in result
