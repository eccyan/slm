"""SLMFS MCP Server — read-only memory introspection tools.

Provides Claude Code with native tools for querying the SLMFS memory
engine via SQLite (no shared memory or FUSE required).
"""

import math
import os
import sqlite3
import struct
import numpy as np
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("slmfs")

# ── Helpers ───────────────────────────────────────────────────────────


def _get_db_path() -> str:
    """Resolve database path from env or default."""
    return os.path.expanduser(
        os.environ.get("SLMFS_DB", "~/.slmfs/memory.db")
    )


def _connect_ro(db_path: str) -> sqlite3.Connection:
    """Open SQLite in read-only mode."""
    uri = f"file:{db_path}?mode=ro"
    return sqlite3.connect(uri, uri=True)


# ── Task 2: read_active ──────────────────────────────────────────────


def _read_active(db_path: str, active_radius: float = 0.3) -> str:
    """Return active nodes within the given Poincare-disk radius.

    Testable internal function — does not depend on MCP framework.
    """
    conn = _connect_ro(db_path)
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT text, pos_x, pos_y, access_count "
            "FROM memory_nodes WHERE status = 0"
        )
        rows = cur.fetchall()
    finally:
        conn.close()

    lines = []
    for text, px, py, access_count in rows:
        r = math.sqrt(px * px + py * py)
        if r < active_radius:
            lines.append(f"- {text}  [r={r:.2f} | {access_count} accesses]")

    if not lines:
        return "No active memories in working range."

    return "\n".join(lines)


@mcp.tool()
def read_active() -> str:
    """Read active memories in the Poincare-disk working region (r < 0.3)."""
    return _read_active(_get_db_path())


# ── Task 3: search_memory ────────────────────────────────────────────

_embedder = None


def _get_embedder():
    """Lazy-load the MiniLM embedder."""
    global _embedder
    if _embedder is None:
        from slmfs.embedder import MiniLMEmbedder

        _embedder = MiniLMEmbedder()
    return _embedder


def _search_memory(db_path: str, query_vec: np.ndarray, top_k: int = 10) -> str:
    """Search all nodes by Fisher-Rao distance to the query vector.

    Distance = sqrt(sum((q - mu)^2 / sigma^2))
    """
    conn = _connect_ro(db_path)
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT text, mu, sigma, pos_x, pos_y, access_count, status "
            "FROM memory_nodes"
        )
        rows = cur.fetchall()
    finally:
        conn.close()

    if not rows:
        return "No memories found."

    scored = []
    for text, mu_blob, sigma_blob, px, py, access_count, status in rows:
        dim = len(mu_blob) // 4  # float32 = 4 bytes
        mu = np.array(struct.unpack(f"{dim}f", mu_blob), dtype=np.float32)
        sigma = np.array(struct.unpack(f"{dim}f", sigma_blob), dtype=np.float32)

        diff = query_vec - mu
        dist = float(np.sqrt(np.sum((diff ** 2) / (sigma ** 2))))

        r = math.sqrt(px * px + py * py)
        scored.append((dist, text, r, access_count, status))

    scored.sort(key=lambda x: x[0])
    scored = scored[:top_k]

    lines = []
    for dist, text, r, access_count, status in scored:
        if status == 0:
            label = "Active"
            r_str = f"r={r:.2f}"
        else:
            label = "Archived"
            r_str = "r=N/A"
        lines.append(
            f"- {text}  [{r_str} | {label} | {access_count} accesses]"
        )

    return "\n".join(lines)


@mcp.tool()
def search_memory(query: str, top_k: int = 10) -> str:
    """Search memories by semantic similarity using Fisher-Rao distance."""
    embedder = _get_embedder()
    query_vec = embedder.embed(query)
    return _search_memory(_get_db_path(), query_vec, top_k)


# ── Task 4: brain_status ─────────────────────────────────────────────


def _brain_status(db_path: str) -> str:
    """Generate a brain-wave status dashboard."""
    conn = _connect_ro(db_path)
    try:
        cur = conn.cursor()

        # Node counts
        cur.execute("SELECT COUNT(*) FROM memory_nodes WHERE status = 0")
        active_count = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM memory_nodes WHERE status != 0")
        archived_count = cur.fetchone()[0]
        total = active_count + archived_count

        # Spatial distribution of active nodes
        cur.execute("SELECT pos_x, pos_y FROM memory_nodes WHERE status = 0")
        active_nodes = cur.fetchall()

        # Cohomology frictions
        cur.execute(
            "SELECT COUNT(*) FROM memory_nodes "
            "WHERE status = 0 AND annotation IS NOT NULL AND annotation != ''"
        )
        friction_count = cur.fetchone()[0]
    finally:
        conn.close()

    working = 0   # r < 0.3
    drifting = 0  # 0.3 <= r < 0.8
    nearing = 0   # 0.8 <= r < 0.95
    beyond = 0    # r >= 0.95
    radii = []

    for px, py in active_nodes:
        r = math.sqrt(px * px + py * py)
        radii.append(r)
        if r < 0.3:
            working += 1
        elif r < 0.8:
            drifting += 1
        elif r < 0.95:
            nearing += 1
        else:
            beyond += 1

    avg_r = sum(radii) / len(radii) if radii else 0.0

    lines = [
        f"## Brain Status\n",
        f"Total: {total} | Active: {active_count} | Archived: {archived_count}",
        "",
        "### Spatial Distribution",
        f"  Working  (r < 0.30): {working}",
        f"  Drifting (r < 0.80): {drifting}",
        f"  Nearing  (r < 0.95): {nearing}",
        f"  Beyond   (r >= 0.95): {beyond}",
        f"  Avg radius: {avg_r:.4f}",
        "",
        "### Cohomology",
        f"  Friction nodes: {friction_count}",
    ]

    if active_count > 0:
        pct = (friction_count / active_count) * 100
        lines.append(f"  Friction ratio: {pct:.1f}%")
    else:
        lines.append("  Friction ratio: N/A")

    return "\n".join(lines)


@mcp.tool()
def brain_status() -> str:
    """Show brain-wave status dashboard with spatial and cohomology metrics."""
    return _brain_status(_get_db_path())


# ── Entry point ───────────────────────────────────────────────────────

if __name__ == "__main__":
    mcp.run()
