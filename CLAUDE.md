# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SLMFS (Superlocal Memory FileSystem) — a zero-copy, mathematically rigorous long-term memory engine for AI agents. It presents a standard Markdown filesystem interface (via FUSE) while running a continuous physical simulation of memory underneath.

Mathematical foundations: Fisher-Rao retrieval metric (information geometry), sheaf cohomology for consistency (algebraic topology), and Riemannian Langevin lifecycle on a Poincaré disk (SDE).

## Architecture

Two-layer design connected via file-backed mmap (`~/.slmfs/ipc_shm.bin`):

1. **Python FUSE Frontend** (I/O & Cooker) — Intercepts filesystem calls, parses AI text inputs, generates embedding vectors via local models, packs strictly aligned binary payloads into shared memory.
2. **C++23 Backend Engine** (Loader & Simulator) — User-space daemon with independent event loop. Zero-copy deserialization via `std::span`, slab allocator, SIMD-optimized vector math, continuous Langevin physics simulation.

Agents interact through standard CLI tools (`cat`, `echo`, `rg`) on the `.agent_memory/` mount point.

## Build

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

The engine binary is at `build/src/engine/slmfs_engine`. Run with:
```bash
./slmfs_engine --db-path=.slmfs/memory.db
```

## Test

```bash
cmake --build build && cd build && ctest --output-on-failure
```

Run a specific test suite: `ctest -R metric` (or `slab`, `langevin`, `sheaf`, `engine`, `persist`).

**Requirements:** CMake 3.20+, GCC 13+ or Clang 16+ (C++23), SQLite3, Python 3.10+ with `fusepy`.

## C++ Module Structure

```
src/
├── slab/       → libslab      (shared memory, SPSC queue, slab allocator)
├── metric/     → libmetric    (Fisher-Rao distance, SIMD kernels)
├── langevin/   → liblangevin  (Poincaré disk, SDE integrator)
├── sheaf/      → libsheaf     (coboundary operator, annotations)
├── persist/    → libpersist   (abstract Store, SQLite V1)
└── engine/     → engine lib   (MemoryGraph, Scheduler) + slmfs_engine binary
```

All libraries are C++ static libraries with public include directories. Each is independently testable via Google Test.

## Agent Memory (SLMFS)

SLMFS is running as a background service on this machine. Use it to persist and retrieve long-term memories across sessions.

**Read active memories** (passive, no side effects):
```bash
cat ~/.agent_memory/active.md
```

**Search for relevant context** (activates matched nodes):
```bash
cat ~/.agent_memory/search/<query>.md
```
Use underscores for spaces: `cat ~/.agent_memory/search/deployment_config.md`

**Write a new memory**:
```bash
echo "learned something important" > ~/.agent_memory/active.md
```

Before starting work, read `~/.agent_memory/active.md` to check for relevant context from prior sessions. When you learn something that would be useful in future sessions, write it to active.md.

## Naming Conventions

- C++ namespaces: `slm::slab`, `slm::metric`, `slm::langevin`, `slm::sheaf`, `slm::engine`, `slm::persist`
- Shared memory file: `~/.slmfs/ipc_shm.bin` (file-backed mmap)
- Data directory default: `.slmfs/`
- Python package: `slmfs`
- Binary: `slmfs_engine`
