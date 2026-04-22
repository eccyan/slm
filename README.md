# SLMFS

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![C++](https://img.shields.io/badge/C++-23-blue.svg)](https://isocpp.org/)
[![Python](https://img.shields.io/badge/Python-3.10+-yellow.svg)](https://www.python.org/)

**Giving AI memory the laws of physics and geometry.**

SLMFS (Superlocal Memory FileSystem) is a zero-copy, mathematically rigorous long-term memory engine for autonomous AI agents (like Claude Code). Instead of relying on ad-hoc RAG pipelines, hardcoded forgetting thresholds, or simple cosine similarities, it grounds agent memory in **Information Geometry**, **Algebraic Topology**, and **Stochastic Differential Equations (SDE)**.

To the AI agent, it looks like a standard, transparent Markdown file system. Under the hood, it's a high-performance C++23 engine running a continuous physical simulation of memory.

## Theoretical Origin

This implementation is based on the visionary concepts described at **[superlocalmemory.com](https://www.superlocalmemory.com/)**.

## The Mathematical Foundations

* **Fisher-Rao Retrieval Metric (Information Geometry):** Memories are not points, but probability distributions (diagonal Gaussian families). Retrieval scoring is derived from the Fisher information structure, smoothly transitioning from linear cosine similarity to geodesic distance over the first few accesses.
* **Sheaf Cohomology for Consistency (Algebraic Topology):** Ad-hoc contradiction checks are replaced with algebraic guarantees. By computing the coboundary operator over the local memory graph, the system mathematically detects logical contradictions and gracefully self-corrects the state.
* **Riemannian Langevin Lifecycle (Non-Euclidean SDE):** Memory nodes exist on a Poincare disk. Frequently accessed memories are pulled to the center. Neglected memories undergo Langevin diffusion, naturally drifting toward the hyperbolic boundary where space compresses. There are no "delete after 30 days" rules -- memories simply self-archive through thermodynamic drift.

## Architecture

SLMFS splits the workload into an offline "Cooker" and a zero-copy "Loader", connected via a lock-free shared memory architecture.

```
Agent (cat/echo/rg)
  |
  v
Python FUSE Frontend ---- Shared Memory (lock-free) ----> C++23 Engine
  |                         |                                |
  +- Text parsing           +- ControlBlock (4KB)            +- Tier 1: I/O drain
  +- Embedding (MiniLM)     +- SPSC ring buffer              +- Tier 2: Cohomology
  +- Binary cooking         +- Slab pool (64KB blocks)       +- Tier 3: Langevin SDE
  +- FUSE VFS intercept                                      +- SQLite persistence
```

### C++23 Backend Engine

A user-space daemon running a three-tier game-engine-style scheduler. It uses `std::span` to achieve zero-copy deserialization directly from shared memory. It manages the slab allocator, executes SIMD-optimized vector math (NEON/AVX2), and runs the continuous Langevin physics simulation in the background without blocking the AI's I/O.

```
src/
+-- slab/       -> libslab      (shared memory, SPSC queue, slab allocator)
+-- metric/     -> libmetric    (Fisher-Rao distance, SIMD kernels)
+-- langevin/   -> liblangevin  (Poincare disk, SDE integrator)
+-- sheaf/      -> libsheaf     (coboundary operator, annotations)
+-- persist/    -> libpersist   (abstract Store, SQLite V1)
+-- engine/     -> engine lib   (MemoryGraph, Scheduler) + slmfs_engine binary
```

### Python FUSE Frontend

Intercepts standard file system calls. It parses AI text inputs, generates embedding vectors via a local model (all-MiniLM-L6-v2, 384-dim), and packs them into strictly aligned binary payloads.

```
python/slmfs/
+-- config.py       Runtime configuration (multi-project isolation)
+-- embedder.py     Abstract Embedder + MiniLM default
+-- cooker.py       Text -> binary payload packer
+-- shm_client.py   Shared memory + SPSC queue operations
+-- fuse_layer.py   FUSE filesystem (active.md + search/)
+-- init.py         Offline "Day Zero" migration
+-- add.py          Online bulk ingestion
+-- analyze.py      Brain-wave dashboard (read-only SQLite)
```

## The Agent Experience

AI agents interact with the memory space purely through standard CLI tools (`cat`, `rg`, `echo`).

**The Working Memory**
```bash
# The agent reads its current active context (memories near the Poincare center)
cat .agent_memory/active.md
```

**On-Demand Vector Search via Magic Paths**
```bash
# The agent searches for a concept.
# The FUSE layer intercepts the path, vectorizes the query, pulls relevant memories
# from the archive back to the center, and streams the result with zero-copy.
cat .agent_memory/search/AWS_closed_network_deployment.md
```

**Ingestion and Consistency**
```bash
# The agent writes a new memory.
# The C++ engine ingests it, computes the Sheaf Cohomology coboundary norm,
# and places it at the active center.
echo "The new proxy IP is 10.0.0.5" >> .agent_memory/active.md
```

## Getting Started

### Prerequisites

* Linux or macOS (WSL2 supported)
* CMake 3.20+
* GCC 13+ or Clang 16+ (C++23 support)
* SQLite3
* Python 3.10+ (with `fusepy`, `sentence-transformers`)
* FUSE support (for the filesystem mount):
  * **Linux:** `sudo apt install libfuse3-dev`
  * **macOS:** [FUSE-T](https://github.com/macos-fuse-t/fuse-t) (recommended) — `brew install macos-fuse-t/homebrew-cask/fuse-t`
    * FUSE-T implements libfuse via a local NFSv4 loopback — no kernel extensions, no Recovery Mode, works on Apple Silicon out of the box
    * macFUSE also works but requires kernel extension approval and reboots

> **Note:** FUSE is only needed for the transparent filesystem mount (`cat .agent_memory/active.md`). The engine, `slmfs init`, and `slmfs add` all work without FUSE installed.

### Building the C++ Engine

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

### Installing the Python Frontend

```bash
cd python && pip install -e ".[dev]"
```

### Running Tests

```bash
# C++ tests (140 tests)
cd build && ctest --output-on-failure

# Python tests (20 tests)
cd python && pytest ../tests/python/ -v
```

### Quick Install

```bash
# One-command install (builds, tests, installs service)
./install.sh
```

This will:
1. Install dependencies (cmake, sqlite3, FUSE-T on macOS / libfuse3 on Linux)
2. Build the C++23 engine
3. Run all tests (140 C++ + 20 Python)
4. Install the engine binary to `~/.local/bin/`
5. Register background services (launchd on macOS, systemd on Linux)

### Usage

```bash
# 1. Migrate existing notes into the Poincare memory space
python -m slmfs init --db-path=~/.slmfs/memory.db ~/MEMORY.md ~/notes/*.md

# 2. Start the C++ engine daemon (physics params are tunable via CLI)
slmfs_engine --db-path=~/.slmfs/memory.db \
  --lambda-decay=5e-6 --noise-scale=2e-4 \
  --thermal-kick-radius=0.01 --archive-threshold=0.95

# 3. Mount the FUSE filesystem
python -m slmfs fuse --mount=~/.agent_memory

# 4. Agent interacts normally
cat ~/.agent_memory/active.md
echo "New memory" > ~/.agent_memory/active.md
cat ~/.agent_memory/search/deployment_notes.md

# 5. Bulk ingest a large reference document into the running engine
python -m slmfs add reference_docs.md

# 6. Bulk ingest an entire docs folder (--stop-fuse handles the lock automatically)
python -m slmfs add --stop-fuse ~/projects/my-app/docs/**/*.md

# 7. Observe the brain — statistical dashboard of Poincaré disk state
python -m slmfs analyze
```

> **Note:** `slmfs add` and the FUSE layer share a single-producer lock on the shared memory file. Use `--stop-fuse` to automatically stop and restart the FUSE service around the ingestion. Without this flag, you must manually stop FUSE first if it's running.

> **Note:** If you ran `install.sh`, steps 2-3 are already running as background services.

### Integrating with Claude Code

SLMFS is designed to be a drop-in long-term memory for Claude Code sessions. After installing, add the following to your project's `CLAUDE.md` (or `~/.claude/CLAUDE.md` for global use):

```markdown
## Agent Memory (SLMFS)

SLMFS is running as a background service. Use it to persist and retrieve
long-term memories across sessions.

**Read active memories** (passive, no side effects):
\`\`\`bash
cat ~/.agent_memory/active.md
\`\`\`

**Search for relevant context** (activates matched nodes):
\`\`\`bash
cat ~/.agent_memory/search/<query>.md
\`\`\`
Use underscores for spaces: `cat ~/.agent_memory/search/deployment_config.md`

**Write a new memory**:
\`\`\`bash
echo "learned something important" > ~/.agent_memory/active.md
\`\`\`

Before starting work, read `~/.agent_memory/active.md` to check for relevant
context from prior sessions. When you learn something that would be useful in
future sessions, write it to active.md.
```

This works because Claude Code can run `cat` and `echo` — the FUSE layer transparently handles embedding, retrieval, and the Langevin physics underneath.

#### Automatic Memory Loading via Hook

For a fully hands-free experience, add a `SessionStart` hook to your `~/.claude/settings.json`. This automatically injects active SLMFS memories into every new Claude Code session — no manual `cat` needed:

```json
{
  "permissions": {
    "allow": [
      "Bash(cat ~/.agent_memory:*)",
      "Bash(echo:*)",
      "Read(//Users/YOU/.agent_memory/**)",
      "Write(//Users/YOU/.agent_memory/**)"
    ]
  },
  "hooks": {
    "SessionStart": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "CONTENT=$(cat ~/.agent_memory/active.md 2>/dev/null | head -100); if [ -n \"$CONTENT\" ]; then jq -n --arg ctx \"$CONTENT\" '{\"hookSpecificOutput\":{\"hookEventName\":\"SessionStart\",\"additionalContext\":(\"## Active SLMFS Memories\\n\\n\" + $ctx)}}'; fi",
            "timeout": 5,
            "statusMessage": "Loading SLMFS memories..."
          }
        ]
      }
    ],
    "PostToolUse": [
      {
        "matcher": "Bash",
        "hooks": [
          {
            "type": "command",
            "if": "Bash(gh pr:*)",
            "command": "COMMITS=$(git log --oneline --since=\"12 hours ago\" --no-merges 2>/dev/null | head -20); if [ -z \"$COMMITS\" ]; then exit 0; fi; BRANCH=$(git branch --show-current 2>/dev/null); REPO=$(basename \"$(git rev-parse --show-toplevel 2>/dev/null)\" 2>/dev/null); SUMMARY=\"Session in ${REPO} (${BRANCH}): ${COMMITS}\"; echo \"$SUMMARY\" > ~/.agent_memory/active.md; COUNT=$(echo \"$COMMITS\" | wc -l | tr -d ' '); jq -n --arg msg \"Wrote session summary to SLMFS ($COUNT commits)\" '{\"systemMessage\": $msg}'",
            "timeout": 5,
            "statusMessage": "Writing session to SLMFS..."
          },
          {
            "type": "command",
            "if": "Bash(git commit:*)",
            "command": "CMD=$(jq -r '.tool_input.command' 2>/dev/null); MSG=$(echo \"$CMD\" | sed -n \"s/.*git commit.*-m.*[\\\"']\\(.*\\)[\\\"'].*/\\1/p\"); if [ -z \"$MSG\" ]; then exit 0; fi; if echo \"$MSG\" | grep -qiE '(spec|plan|design)'; then REPO=$(basename \"$(git rev-parse --show-toplevel 2>/dev/null)\" 2>/dev/null); BRANCH=$(git branch --show-current 2>/dev/null); FILES=$(git diff --name-only HEAD~1 HEAD 2>/dev/null | head -5 | tr '\\n' ', '); SUMMARY=\"Spec/plan committed in ${REPO} (${BRANCH}): ${MSG} | Files: ${FILES}\"; echo \"$SUMMARY\" > ~/.agent_memory/active.md; jq -n --arg msg \"Wrote spec/plan to SLMFS\" '{\"systemMessage\": $msg}'; fi",
            "timeout": 5,
            "statusMessage": "Checking for spec/plan commit..."
          }
        ]
      }
    ]
  }
}
```

> **Requires:** `jq` (`brew install jq` / `apt install jq`)
>
> Replace `YOU` in the Read/Write permissions with your macOS username.

The hooks form a complete memory loop:

| Hook | Trigger | What it does |
|------|---------|-------------|
| `SessionStart` | Session opens | Reads `active.md` and injects it as context |
| `PostToolUse` | After `gh pr` commands | Writes recent commits as a session summary to `active.md` |
| `PostToolUse` | After `git commit` with "spec", "plan", or "design" in message | Writes spec/plan details + file list to `active.md` |

This gives Claude continuity across sessions — it remembers what it did last time and picks up where it left off. Spec and plan commits are captured automatically, so the next session knows what was designed before implementation begins. The FUSE layer handles embedding and the Langevin physics ensures old memories naturally drift to the archive while recent work stays in focus.

### Observing the Brain

The `analyze` command provides a real-time dashboard of the Poincaré disk's state:

```bash
python -m slmfs analyze
```

```
──────────────────────────────────────────────────
  Node Population
──────────────────────────────────────────────────
  Total nodes:    110
  Active:         20
  Archived:       90

──────────────────────────────────────────────────
  Spatial Distribution (Langevin Drift)
──────────────────────────────────────────────────
  Working Memory  (r < 0.30):     20  nodes
  Drifting        (r < 0.80):      0  nodes
  Nearing Archive (r < 0.95):      0  nodes
  Beyond boundary (r >= 0.95):     0  nodes

──────────────────────────────────────────────────
  Poincare Disk Map (Active Nodes)
──────────────────────────────────────────────────
          .........
         ....   ....
       ...         ...
      ..             ..
     ..               ..
    .         *         .
     ..               ..
      ..             ..
       ...         ...
         ....   ....
          .........
```

Metrics include:
- **Node Population** — Total, active, and archived node counts
- **Spatial Distribution** — How nodes cluster across Langevin drift zones (working memory → drifting → nearing archive)
- **Fisher-Rao Certainty** — Average access count and sigma (lower sigma = higher confidence)
- **Cohomology Frictions** — Nodes where topological contradictions were detected and resolved
- **ASCII Disk Map** — 2D scatter plot of active node positions on the Poincaré disk

### Tuning Memory Retention

The Langevin physics parameters control how fast memories drift toward the archive boundary. All are configurable via CLI flags — no recompilation needed:

| Flag | Default | Effect |
|------|---------|--------|
| `--lambda-decay` | `5e-6` | Outward drift rate. Higher = faster forgetting |
| `--noise-scale` | `2e-4` | Brownian noise intensity. Higher = more random drift |
| `--thermal-kick-radius` | `0.01` | Initial offset on activation (avoids origin singularity) |
| `--archive-threshold` | `0.95` | Poincaré disk radius at which nodes are archived |
| `--friction-penalty-radius` | `0.6` | Where contradicted nodes land after sheaf cohomology detection |

With the defaults, an unaccessed memory takes ~10 days to reach the archive boundary (r=0.95). Accessing a memory resets it to the disk center. Contradicted memories (detected by sheaf cohomology) are scattered isotropically at r=0.6, giving them ~5 days of grace before archival. Tune `--lambda-decay` to control the "forgetfulness" of the agent's working memory.

### Multi-Project Isolation

Each project can run its own independent "brain" by specifying unique file paths:

```bash
# Project A
slmfs_engine --shm-path=~/work/projA/.slmfs/ipc_shm.bin --db-path=~/work/projA/.slmfs/memory.db

# Project B (completely independent Poincare disk)
slmfs_engine --shm-path=~/work/projB/.slmfs/ipc_shm.bin --db-path=~/work/projB/.slmfs/memory.db
```

### Service Management

**macOS (launchd)** — installed automatically by `install.sh`:
```bash
# Status
launchctl list | grep slmfs

# Stop
launchctl unload ~/Library/LaunchAgents/com.eccyan.slmfs.plist
launchctl unload ~/Library/LaunchAgents/com.eccyan.slmfs-fuse.plist

# Restart
launchctl unload ~/Library/LaunchAgents/com.eccyan.slmfs.plist && launchctl load ~/Library/LaunchAgents/com.eccyan.slmfs.plist
```

**Linux (systemd)**:
```bash
# Install the user service
cp config/slmfs-engine.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now slmfs-engine
```

## License

This project is licensed under the **Apache License 2.0** - see the [LICENSE](LICENSE) file for details.
