import math
import time
from pathlib import Path

import pytest
from slmfs.init import parse_markdown, compute_initial_position, place_all, Chunk


def test_parse_simple_markdown(tmp_path):
    md = tmp_path / "test.md"
    md.write_text("# Title\nSome content\n## Section\nMore content\n")

    chunks = parse_markdown(md)
    assert len(chunks) >= 2
    assert chunks[0].depth == 1
    assert "Title" in chunks[0].text
    assert chunks[1].depth == 2


def test_parse_preserves_hierarchy(tmp_path):
    md = tmp_path / "test.md"
    md.write_text("# Root\n## Child A\n### Grandchild\n## Child B\n")

    chunks = parse_markdown(md)
    assert chunks[1].parent_idx == 0  # Child A → Root
    assert chunks[2].parent_idx == 1  # Grandchild → Child A
    assert chunks[3].parent_idx == 0  # Child B → Root


def test_parse_preamble(tmp_path):
    md = tmp_path / "test.md"
    md.write_text("Some preamble text\n# First heading\nContent\n")

    chunks = parse_markdown(md)
    assert chunks[0].depth == 0


def test_placement_depth_zero_at_center():
    chunk = Chunk(text="root", depth=0, parent_idx=-1,
                  source_mtime=time.time())
    r, _ = compute_initial_position(chunk, time.time())
    assert r == 0.0


def test_placement_recent_file_near_center():
    now = time.time()
    chunk = Chunk(text="recent", depth=1, parent_idx=0,
                  source_mtime=now - 86400)
    r, _ = compute_initial_position(chunk, now)
    assert 0.0 < r < 0.2


def test_placement_old_file_near_boundary():
    now = time.time()
    chunk = Chunk(text="old", depth=1, parent_idx=0,
                  source_mtime=now - 365 * 86400)
    r, _ = compute_initial_position(chunk, now)
    assert r > 0.7


def test_placement_clamped():
    now = time.time()
    chunk = Chunk(text="ancient", depth=1, parent_idx=0,
                  source_mtime=now - 10 * 365 * 86400)
    r, _ = compute_initial_position(chunk, now)
    assert r <= 0.90


def test_golden_angle_distribution():
    now = time.time()
    chunks = [
        Chunk(text=f"chunk{i}", depth=1, parent_idx=0, source_mtime=now)
        for i in range(20)
    ]

    positions = place_all(chunks)
    angles = []
    for x, y in positions:
        if x != 0.0 or y != 0.0:
            angles.append(math.atan2(y, x))

    if len(angles) >= 2:
        diffs = [
            abs(angles[i + 1] - angles[i]) for i in range(len(angles) - 1)
        ]
        for d in diffs:
            assert d > 0.01
