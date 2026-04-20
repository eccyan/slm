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
# C++ tests (133 tests)
cd build && ctest --output-on-failure

# Python tests (20 tests)
cd python && pytest ../tests/python/ -v
```

### Usage

```bash
# 1. Migrate existing notes into the Poincare memory space
python -m slmfs init ~/MEMORY.md ~/notes/*.md

# 2. Start the C++ engine daemon
./build/src/engine/slmfs_engine --db-path=.slmfs/memory.db --shm-name=slmfs_shm

# 3. Mount the FUSE filesystem
python -m slmfs fuse --mount=.agent_memory

# 4. Agent interacts normally
cat .agent_memory/active.md
echo "New memory" >> .agent_memory/active.md
cat .agent_memory/search/deployment_notes.md

# 5. Bulk ingest a large reference document into the running engine
python -m slmfs add reference_docs.md
```

### Multi-Project Isolation

Each project can run its own independent "brain" by specifying unique shared memory and database paths:

```bash
# Project A
slmfs_engine --shm-name=projA_shm --db-path=~/work/projA/.slmfs/memory.db

# Project B (completely independent Poincare disk)
slmfs_engine --shm-name=projB_shm --db-path=~/work/projB/.slmfs/memory.db
```

### Service Management (systemd)

```bash
# Install the user service
cp config/slmfs-engine.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now slmfs-engine
```

## License

This project is licensed under the **Apache License 2.0** - see the [LICENSE](LICENSE) file for details.
