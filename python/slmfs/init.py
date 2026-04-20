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
    depth: int
    parent_idx: int
    source_mtime: float


def parse_markdown(path: Path) -> list[Chunk]:
    """Split Markdown into chunks at heading boundaries."""
    mtime = path.stat().st_mtime
    content = path.read_text(encoding="utf-8")
    lines = content.split("\n")

    chunks: list[Chunk] = []
    heading_stack: list[tuple[int, int]] = []

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
            level = len(stripped) - len(stripped.lstrip("#"))
            heading_text = stripped.lstrip("#").strip()

            flush()

            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()

            parent = heading_stack[-1][1] if heading_stack else -1
            heading_stack.append((level, len(chunks)))

            current_lines = [heading_text]
            current_depth = level
            current_parent = parent
        else:
            current_lines.append(line)

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

    r = 0.85 * (1.0 - math.exp(-age_days / 180.0))
    r = max(0.05, min(r, 0.90))

    return (r, 0.0)


def place_all(chunks: list[Chunk]) -> list[tuple[float, float]]:
    """Assign (x, y) positions on the Poincaré disk."""
    now = _time.time()
    positions = []
    golden_angle = 2.399963

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

    for i, chunk in enumerate(chunks):
        if chunk.parent_idx >= 0:
            parent_db_id = chunk_to_db_id[chunk.parent_idx]
            conn.execute(
                "UPDATE memory_nodes SET parent_id = ? WHERE id = ?",
                (parent_db_id, chunk_to_db_id[i]),
            )

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

    all_chunks: list[Chunk] = []
    for p in paths:
        all_chunks.extend(parse_markdown(p))
    print(f"Parsed {len(all_chunks)} chunks from {len(paths)} files")

    embedder = MiniLMEmbedder()
    texts = [c.text for c in all_chunks]
    embeddings = embedder.embed_batch(texts)
    print(
        f"Generated {embeddings.shape[0]} embeddings ({embeddings.shape[1]}-dim)"
    )

    positions = place_all(all_chunks)

    count = ingest_to_sqlite(all_chunks, embeddings, positions, config.db_path)
    print(f"Ingested {count} nodes into {config.db_path}")
    print("Ready. Start the engine: slmfs_engine")


if __name__ == "__main__":
    main()
