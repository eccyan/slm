# SLMFS — Architecture Specification

> **Note:** This document was authored under the original project name "Superlocal MemoryFS". The project has since been renamed to **SLMFS**. Code references (namespaces, binary names, Python package) use the new names. Some prose may still reference "Superlocal" for historical context.

**Date:** 2026-04-20
**Status:** Approved
**Version:** 1.1

---

## 1. Overview

SLMFS (Superlocal Memory FileSystem) is a zero-copy, mathematically rigorous long-term memory engine for autonomous CLI-based AI agents. It presents a transparent Markdown filesystem interface (via FUSE) while running a continuous physical simulation of memory on a non-Euclidean manifold.

**Core philosophy:** No hardcoded forgetting rules. Memory self-organizes through physics (Langevin SDE), retrieval adapts through geometry (Fisher-Rao), and consistency is guaranteed through topology (sheaf cohomology).

### Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Embedding model coupling | Dimension-agnostic engine | C++ reads `vector_dim` from header; Python defaults to `all-MiniLM-L6-v2` (384-dim) |
| Persistence (V1) | SQLite | Debug-friendly, single-file; internal structures designed for V2 mmap swap |
| Contradiction handling | Annotated expulsion | HTML comments in `active.md` give the agent a "cognitive flash" without breaking Markdown |
| Architecture | Modular monolith (Approach B) | Single binary with static library boundaries; game-engine DNA |
| Data layout | Structure of Arrays (SoA) | Hot math data (NodeState, GaussianNode) separated from cold text payloads |
| Bootstrap | Offline `slm init` | Direct SQLite ingestion before engine starts; no IPC needed |

---

## 2. System Architecture

Two-tier architecture: Python FUSE frontend (I/O & Cooker) and C++23 backend engine (Loader & Simulator), cooperating via zero-copy IPC over shared memory.

```
Agent (cat/echo)
  │
  ▼
Python FUSE Frontend ──── Shared Memory (lock-free) ────► C++23 Engine
  │                         │                                │
  ├─ Text parsing           ├─ ControlBlock (4KB)            ├─ Tier 1: I/O drain
  ├─ Embedding (MiniLM)     ├─ SPSC ring buffer              ├─ Tier 2: Cohomology
  ├─ Binary cooking         └─ Slab pool (64KB blocks)       ├─ Tier 3: Langevin SDE
  └─ FUSE VFS intercept                                      └─ SQLite persistence
```

### C++ Module Structure (Approach B — Modular Monolith)

```
src/
├── slab/          → libslab      (shared memory, SPSC queue, slab allocator)
├── metric/        → libmetric    (Fisher-Rao distance, SIMD kernels)
├── sheaf/         → libsheaf     (coboundary operator, contradiction annotations)
├── langevin/      → liblangevin  (Poincaré disk coordinates, SDE integrator)
├── persist/       → libpersist   (abstract Store interface, SQLite V1)
└── engine/        → main binary  (three-tier scheduler, MemoryGraph registry)
```

Each library is a CMake `STATIC` library with `PUBLIC` include directories. Single binary deployment; no inter-process complexity. Each math subsystem is independently unit-testable with Google Test.

### Python Package Structure

```
python/
├── pyproject.toml
└── slmfs/
    ├── __init__.py
    ├── config.py          # Runtime configuration (dataclass + TOML)
    ├── embedder.py        # Abstract Embedder + MiniLMEmbedder default
    ├── cooker.py          # Text → binary payload packer
    ├── shm_client.py      # Shared memory + SPSC queue operations
    ├── fuse_layer.py      # FUSE operations (main entry point)
    ├── init.py            # slmfs init bootstrap script
    └── add.py             # Online bulk ingestion
```

### Build System

Root `CMakeLists.txt` sets `CMAKE_CXX_STANDARD 23`, detects SIMD capability (SSE4.2/AVX2/NEON), and finds SQLite3. Test framework: Google Test via `FetchContent`. Python tests: pytest.

**Requirements:** CMake 3.20+, GCC 13+ or Clang 16+ (C++23), Python 3.10+ with `fusepy`, `sentence-transformers`, `mistune`.

---

## 3. Virtual Filesystem Specification

### 3.1 `active.md` (Working Memory)

**Read:** Extracts nodes with Poincaré radius $r < 0.3$ (active threshold), assembles as Markdown. Includes cohomology annotations as HTML comments.

**Write/Append:** Ingests text chunks, embeds via MiniLM, packs binary payload into shared memory slab, pushes handle to SPSC queue. Returns immediately (non-blocking I/O). Engine places new node at $r = 0$.

**Overwrite/Delete:** Applies strong outward drift penalty to corresponding nodes.

### 3.2 `search/<query>.md` (On-Demand Search)

**Read:** Vectorizes the filename as a search query, retrieves relevant memories from the archive layer via Fisher-Rao `top_k`. Retrieved nodes are activated (pulled to $r = 0$) and output as Markdown.

**Write:** Read-only. Returns `EACCES` (Permission denied).

---

## 4. Shared Memory & IPC (libslab)

### 4.1 Memory Layout

Total shared memory: 4MB. Divided into a control block and a slab pool.

```
┌─────────────────────────────────────────────────┐
│  ControlBlock (4KB, page-aligned)               │
│  ┌─────────────────────────────────────────────┐│
│  │ atomic<uint64_t> free_bitmask               ││  1 bit per slab (up to 64 slabs)
│  │ SPSCRingBuffer<uint32_t, 256> cmd_queue     ││  lock-free, cache-line padded
│  │ atomic<uint32_t> engine_status              ││  IDLE / BUSY / SHUTDOWN
│  │ uint32_t slab_count                         ││
│  │ uint32_t slab_size                          ││
│  └─────────────────────────────────────────────┘│
├─────────────────────────────────────────────────┤
│  Slab[0]  (64KB, 64-byte aligned)              │
│  Slab[1]  (64KB, 64-byte aligned)              │
│  ...                                           │
│  Slab[62] (64KB, 64-byte aligned)              │
└─────────────────────────────────────────────────┘
```

63 slabs x 64KB = 4,032KB. Each slab holds `MemoryFSHeader` (64B) + text (variable) + padding + vector payload. Accommodates up to 1536-dim models.

### 4.2 Binary Payload (MemoryFSHeader)

```cpp
struct alignas(64) MemoryFSHeader {
    uint32_t    magic;           // 0x4D454D46 ('MEMF')
    uint8_t     command;         // 0x01: READ, 0x02: WRITE_COMMIT
    uint8_t     padding[3];
    uint64_t    total_size;
    uint32_t    text_offset;     // always 64 (immediately after header)
    uint32_t    text_length;
    uint32_t    vector_offset;   // always a multiple of 64
    uint32_t    vector_dim;      // dimension-agnostic
    // reserved[32]: first 5 bytes used for scene graph
    uint32_t    parent_id;       // scene graph parent (0 = root)
    uint8_t     depth;           // heading level (0 = preamble)
    uint8_t     reserved_rest[27];
};
```

Layout within a slab:

1. `MemoryFSHeader` (64 bytes fixed)
2. Text Payload (UTF-8, variable length)
3. Padding (for 64-byte boundary alignment of vector data)
4. Vector Payload (float32 array, variable length)

### 4.3 SPSC Ring Buffer

```cpp
template <typename T, std::size_t Capacity>
struct alignas(64) SPSCRingBuffer {
    alignas(64) std::atomic<std::size_t> head_{0};   // writer (Python)
    alignas(64) std::size_t cached_tail_{0};          // local cache for producer
    alignas(64) std::atomic<std::size_t> tail_{0};   // reader (C++ engine)
    alignas(64) std::size_t cached_head_{0};          // local cache for consumer
    alignas(64) std::array<T, Capacity> buffer_{};

    bool try_push(T value);  // release on write
    bool try_pop(T& value);  // acquire on read
};
```

`head_` and `tail_` on separate cache lines to prevent false sharing. `Capacity` must be power of 2 (modulo via bitmask). Cached counters reduce atomic read frequency.

### 4.4 Handle Encoding (32-bit)

```
┌──────────┬────────────────────────┐
│ cmd (8b) │ slab_index (24b)       │
└──────────┴────────────────────────┘
```

Python side: acquire slab (atomic CAS on bitmask) → write payload → push 32-bit handle. Returns immediately.

### 4.5 Slab Allocator

```cpp
class SlabAllocator {
public:
    SlabAllocator(void* shm_base, uint32_t slab_count, uint32_t slab_size);
    std::optional<uint32_t> acquire();          // atomic CAS on free_bitmask
    void release(uint32_t index);               // atomic OR on free_bitmask
    std::span<std::byte> get(uint32_t index);   // zero-copy span into slab
    const MemoryFSHeader& header(uint32_t index);
};
```

### 4.6 Read Response Protocol

For READ commands, the engine reuses the same slab to write results:

```
Engine writes:
  magic = 0x444F4E45 ('DONE')
  text_offset = 64
  text_length = len(markdown_result)
  [64..] = UTF-8 Markdown with cohomology annotations
```

FUSE spin-waits (100us sleep, 1s timeout) for magic to flip to `DONE`, then reads and returns the Markdown.

---

## 5. Mathematical Models

### 5.1 Fisher-Rao Retrieval Metric (libmetric)

Each memory node is modeled as a diagonal Gaussian distribution $\mathcal{N}(\mu, \text{diag}(\sigma^2))$.

**Graduated ramp (access count 0–10):**

$$\alpha = \min\left(\frac{\text{access\_count}}{10},\ 1.0\right)$$

$$\sigma_i = \sigma_{\max} \cdot (1 - \alpha) + \sigma_{\min} \cdot \alpha$$

At $\alpha = 0$ (fresh node, $\sigma \to \infty$): degenerates to cosine similarity.
At $\alpha = 1$ (10+ accesses, $\sigma = \sigma_{\min}$): full geodesic distance.

**Geodesic distance on diagonal Gaussians:**

$$d_{\text{FR}}^2 = \sum_{i=1}^{D} \left[ 2\ln\frac{\sigma_{Q,i}}{\sigma_{P,i}} \right]^2 + \sum_{i=1}^{D} \frac{(\mu_{P,i} - \mu_{Q,i})^2}{\sigma_{P,i}\,\sigma_{Q,i}}$$

**C++ interface:**

```cpp
struct GaussianNode {
    std::span<const float> mu;       // embedding vector (zero-copy from slab)
    std::span<const float> sigma;    // per-dimension std dev
    uint32_t access_count;
};

class FisherRaoMetric {
public:
    float distance(const GaussianNode& p, const GaussianNode& q) const;
    std::vector<uint32_t> top_k(
        const GaussianNode& query,
        std::span<const GaussianNode> candidates,
        uint32_t k
    ) const;
};
```

**SIMD kernels (simd_ops.hpp):**

```cpp
// AVX2: 8 floats/cycle, SSE4.2/NEON: 4 floats/cycle, scalar fallback
float simd_weighted_sq_diff(
    const float* mu_p, const float* mu_q,
    const float* sigma_p, const float* sigma_q,
    uint32_t dim
);

float simd_variance_divergence(
    const float* sigma_p, const float* sigma_q,
    uint32_t dim
);
```

Compile-time dispatch via `#ifdef __AVX2__` / `__SSE4_2__` / `__ARM_NEON`. The 64-byte slab alignment guarantees aligned SIMD loads.

### 5.2 Sheaf Cohomology for Consistency (libsheaf)

**Graph structure:** Memory nodes form a local knowledge graph. Edges carry relation vectors representing expected semantic differences.

**Edge construction (dual topology):**

1. **Primary — structural edges:** Siblings (same `parent_id`) and parent-child relationships from the scene graph. Deterministic, weighted higher in the coboundary norm.
2. **Secondary — k-NN cross-branch:** Smaller k-NN query (k=3–5) via Fisher-Rao to catch contradictions across different branches of the heading tree.

**Coboundary operator $\delta$:**

For a 0-cochain $f$ assigning an embedding to each node, the coboundary on edge $(i, j)$:

$$(\delta f)(i, j) = f(j) - f(i) - r_{ij}$$

Computed locally over the 1-ring neighborhood of the newly ingested node: $O(k)$ per ingestion, not $O(N)$.

**Contradiction detection flow:**

1. New node $B$ ingested under `parent_id = X`
2. Pull siblings (same parent) and parent as primary neighborhood
3. Optionally find k-NN cross-branch neighbors
4. Compute $\|\delta\|$ over the union of edge sets
5. If $\|\delta\| > \text{threshold}$: conflicting old nodes receive massive Langevin drift penalty (pushed to $r = 0.93$), annotation attached to new node

**Annotation format (Annotated Expulsion):**

```html
<!-- cohomology: superseded "proxy IP is 10.0.0.1" by "proxy IP is 10.0.0.5", δ-norm=0.73 -->
```

Injected into the Markdown stream when FUSE reads `active.md`. Provides the agent with a "cognitive flash" of what changed without breaking Markdown rendering.

```cpp
struct CoboundaryResult {
    float norm;
    std::vector<uint32_t> conflicting;
};

class CoboundaryOperator {
public:
    CoboundaryResult compute_local(
        uint32_t new_node,
        const MemoryGraph& graph,
        float contradiction_threshold
    ) const;
};

std::string format_annotation(const Annotation& ann);
```

### 5.3 Riemannian Langevin Lifecycle (liblangevin)

Memory nodes live on a Poincaré disk (radius $< 1$). Active memories sit near the center ($r \approx 0$). Unaccessed memories drift outward via Langevin dynamics.

**Poincaré disk metric tensor:**

$$g_{ij}(p) = \frac{4}{(1 - r^2)^2}\,\delta_{ij}$$

Space compresses exponentially near the boundary. A small Euclidean step near $r = 0.99$ covers vastly more "conceptual distance" than near $r = 0$.

**Riemannian Langevin SDE:**

$$dp = -g^{-1}(p)\,\nabla U(p)\,dt + \sqrt{2\,g^{-1}(p)\,dt}\;\xi$$

where $g^{-1}(p) = \frac{(1-r^2)^2}{4}$, $U(p) = -\lambda_{\text{decay}} \cdot \Delta t_{\text{since\_last\_access}} \cdot r$, $\xi \sim \mathcal{N}(0, I)$.

**Discrete integration (Euler-Maruyama):**

```cpp
struct DiskPosition { float x, y; };

struct NodeState {
    DiskPosition pos;
    double last_access_time;
    uint32_t access_count;
};

class LangevinStepper {
public:
    struct Config {
        float dt;                  // tick interval (5.0s)
        float lambda_decay;        // outward drift rate
        float noise_scale;         // diffusion intensity
        float archive_threshold;   // 0.95
    };

    std::vector<uint32_t> step(
        std::span<NodeState> nodes,
        double current_time,
        std::mt19937& rng
    ) const;

    static void activate(NodeState& node, double current_time);

private:
    Config config_;
    static DiskPosition project_to_disk(DiskPosition p);  // clamp r < 1
};
```

**`project_to_disk` safety:** If Euler-Maruyama overshoots past $r = 1$, scale to $r = 0.999$. Preserves drift direction, prevents numerical explosion.

**Cohomology integration:** When $\delta \neq 0$, superseded nodes are instantly repositioned to $(0.0, 0.93)$ — near the archive threshold. The next Langevin tick finishes the expulsion.

**Activation:** When a node is read or retrieved via search, `activate()` resets it to $r = 0$ and refreshes `last_access_time`. Access count increments, driving the Fisher-Rao $\sigma$ ramp.

---

## 6. Engine Scheduler & In-Memory Registry

### 6.1 Three-Tier Scheduler (Game Engine Frame Model)

Single `std::jthread` main loop. Tiers execute in priority order with time budgets.

```cpp
class Scheduler {
public:
    struct Config {
        std::chrono::microseconds tier1_poll_interval{100};
        std::chrono::milliseconds tier2_time_budget{50};
        std::chrono::seconds      tier3_tick_interval{5};
        std::chrono::seconds      checkpoint_interval{60};
    };

    Scheduler(
        SlabAllocator& slab,
        SPSCRingBuffer<uint32_t, 256>& queue,
        MemoryGraph& graph,
        FisherRaoMetric& metric,
        CoboundaryOperator& sheaf,
        LangevinStepper& langevin,
        Store& persist
    );

    void run(std::stop_token token);
};
```

**Frame structure:**

| Priority | Tier | Trigger | Budget | Yields to |
|---|---|---|---|---|
| 1 | I/O Command Drain | Every frame, first | Unbounded (drain all) | Nothing |
| 2 | Cohomology Check | Pending queue non-empty | 50ms per frame | Tier 1 (on `queue_.peek()`) |
| 3 | Langevin Drift | Fixed Δt = 5s | Full scene sweep | Nothing (fast with SoA) |

**Tier 1 — Command Processing:**

- Pop handle from SPSC queue
- READ: construct GaussianNode from slab, run `top_k`, activate retrieved nodes, write Markdown result back to same slab, set magic to `DONE`
- WRITE_COMMIT: parse header (including `parent_id`, `depth`), insert node at $r = 0$, enqueue for Tier 2

**Tier 2 — Amortized Cohomology:**

- Process nodes from `cohomology_pending_` deque
- Time-budgeted: stops after 50ms or yields if Tier 1 queue has new items
- On contradiction: apply drift penalty + attach annotation

**Tier 3 — Langevin Tick:**

- Fires every 5 seconds regardless of I/O load
- Iterates contiguous `std::vector<NodeState>` (SoA hot data)
- Nodes exceeding $r > 0.95$ are archived via `Store::archive_node()` and removed from the graph

**Idle backpressure:** Sleeps 100us only when all queues are empty. No busy-waiting under load.

**Graceful shutdown:** `std::jthread` destructor calls `request_stop()` → scheduler drains remaining commands → `persist_.flush()` → exit.

### 6.2 MemoryGraph — SoA In-Memory Registry

```cpp
class MemoryGraph {
public:
    // ── Hot arrays (contiguous, Tier 2/3 iterate these) ──
    std::vector<NodeState>    states;
    std::vector<GaussianNode> gaussians;
    std::vector<uint32_t>     parent_ids;
    std::vector<uint8_t>      depths;

    // ── Cold storage (Tier 1 only, indexed by same slot) ──
    std::vector<std::string>  texts;
    std::vector<std::string>  annotations;

    // ── Bookkeeping ──
    std::vector<uint32_t>     ids;
    std::unordered_map<uint32_t, size_t> id_to_index;
    uint32_t next_id{1};

    uint32_t insert(GaussianNode g, std::string text,
                    uint32_t parent_id, uint8_t depth, NodeState state);
    void remove(uint32_t id);

    // Scene graph queries
    std::vector<uint32_t> siblings(uint32_t parent_id) const;
    std::optional<uint32_t> parent(uint32_t node_id) const;
};
```

**Why SoA:** The Langevin stepper iterates every node every 5 seconds. With AoS, `std::string` heap pointers pollute cache lines. SoA keeps `NodeState` at 16 bytes per element — tight, linear, cache-friendly. Cold text/annotation vectors are never touched during Tier 3.

---

## 7. Persistence Layer (libpersist)

### 7.1 Abstract Interface

```cpp
class Store {
public:
    virtual ~Store() = default;
    virtual void checkpoint(const MemoryGraph& graph) = 0;
    virtual void flush(const MemoryGraph& graph) = 0;
    virtual void load(MemoryGraph& graph) = 0;
    virtual void archive_node(const MemoryGraph::Node& node) = 0;
    virtual std::vector<MemoryGraph::Node> retrieve_archived(
        const GaussianNode& query, const FisherRaoMetric& metric, uint32_t k
    ) = 0;
};
```

Designed for V2 swap: replace `SqliteStore` with a binary mmap implementation without touching any other module. The internal SoA MemoryGraph is already contiguous and cleanly isolated from DB logic.

### 7.2 SQLite V1 Schema

```sql
CREATE TABLE memory_nodes (
    id           INTEGER PRIMARY KEY,
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
    status       INTEGER NOT NULL DEFAULT 0  -- 0=active, 1=archived
);

CREATE TABLE edges (
    source_id    INTEGER NOT NULL,
    target_id    INTEGER NOT NULL,
    edge_type    INTEGER NOT NULL,     -- 0=structural, 1=knn
    relation     BLOB,
    PRIMARY KEY (source_id, target_id)
);

CREATE INDEX idx_nodes_status ON memory_nodes(status);
CREATE INDEX idx_nodes_parent ON memory_nodes(parent_id);
CREATE INDEX idx_edges_source ON edges(source_id);
```

### 7.3 Checkpoint Strategy

- `checkpoint()`: Every 60s. Single SQLite transaction with `INSERT OR REPLACE` for all active nodes.
- `flush()`: On SIGTERM. Same as checkpoint + `sqlite3_wal_checkpoint(TRUNCATE)` for durability.
- `load()`: On startup. Reads `status=0` rows into MemoryGraph. Archived rows stay on disk, queried on demand.

---

## 8. Bootstrap Process (`slm init`)

Offline migration of legacy Markdown files into the Poincaré memory space. Runs before the C++ engine starts. Bypasses shared memory and IPC — writes directly to SQLite.

### 8.1 CLI Entry Point

```
python -m slmfs init /path/to/MEMORY.md [/path/to/other.md ...]
```

### 8.2 Pipeline

```
Legacy .md files
  │
  ▼
1. AST Parse (mistune) ── chunk at heading boundaries
  │                        extract heading hierarchy → (parent_id, depth)
  ▼
2. Batch Embed ── MiniLMEmbedder.embed_batch(chunks) → float32[N, 384]
  │               single batched forward pass
  ▼
3. Poincaré Placement ── map each chunk to initial (x, y)
  │                       depth=0 → r=0 (center)
  │                       others → exponential decay of file mtime
  │                       golden angle (2.399963 rad) angular distribution
  ▼
4. SQLite Ingest ── 3-pass: insert nodes → fix parent_ids → create edges
```

### 8.3 Poincaré "Big Bang" Placement Rules

| Node type | Radial position | Rationale |
|---|---|---|
| `depth=0` (preamble, system prompts) | $r = 0$ | Always active, maximum visibility |
| Recent notes (modified yesterday) | $r \approx 0.05$ | Near center, readily accessible |
| Moderate age (modified ~6 months ago) | $r \approx 0.7$ | Mid-distance, retrievable via search |
| Old notes (modified 1+ year ago) | $r \approx 0.85$ | Near archive boundary |

Radial mapping: $r = 0.85 \cdot (1 - e^{-\text{age\_days}/180})$, clamped to $[0.05, 0.90]$.

Angular distribution: golden angle prevents spatial clustering regardless of input count.

Initial $\sigma = \sigma_{\max}$ for all imported nodes (maximum uncertainty — the Fisher-Rao ramp will tighten as the agent accesses them).

### 8.4 Three-Pass SQLite Ingestion

1. **Pass 1:** Insert all chunks with `parent_id=0` to obtain auto-increment IDs
2. **Pass 2:** Map `chunk.parent_idx` to actual DB IDs, update `parent_id` references
3. **Pass 3:** Generate structural edges (parent-child + sibling pairs)

### 8.5 Startup Sequence

```
1. python -m slmfs init ~/MEMORY.md ~/notes/*.md
     → parse, embed, place, write to .slmfs/memory.db

2. systemctl --user start slmfs-engine
     → SqliteStore::load() populates MemoryGraph
     → Langevin SDE + cohomology begin immediately

3. python -m slmfs fuse
     → mounts .agent_memory/
     → agent can cat/echo as normal
```

---

## 9. Service Configuration

### 9.1 systemd Unit (Linux / WSL2)

Location: `~/.config/systemd/user/slmfs-engine.service`

```ini
[Unit]
Description=SLMFS Memory C++ Backend Engine
Before=slmfs-fuse.service

[Service]
Type=simple
ExecStart=%h/.local/bin/slmfs_engine --db-path=%h/.slmfs/memory.db --shm-name=slmfs_shm
Restart=on-failure
LimitMEMLOCK=infinity
KillSignal=SIGTERM
TimeoutStopSec=10

[Install]
WantedBy=default.target
```

### 9.2 SIGTERM Handling

`std::jthread` cooperative cancellation: SIGTERM → `request_stop()` → scheduler drains command queue → `persist_.flush()` → clean exit within `TimeoutStopSec`.

---

## 10. Test Strategy

| Library | Framework | What to test |
|---|---|---|
| libslab | Google Test | SPSC correctness under concurrent push/pop, slab acquire/release, bitmask wraparound |
| libmetric | Google Test | Fisher-Rao distance against known analytical values, SIMD vs scalar parity, graduated ramp behavior |
| libsheaf | Google Test | Coboundary norm for known contradiction pairs, annotation formatting, structural vs k-NN edge priority |
| liblangevin | Google Test | SDE integration stability, `project_to_disk` boundary behavior, activate/archive lifecycle |
| libpersist | Google Test | SQLite round-trip (checkpoint → load), archive/retrieve, schema creation |
| engine | Google Test | Scheduler tier ordering, cohomology queueing, shutdown drain |
| cooker.py | pytest | Binary payload alignment verification, header packing correctness |
| fuse_layer.py | pytest | Virtual directory structure, EACCES on search writes, heading parsing |
| init.py | pytest | AST chunk extraction, golden angle placement, 3-pass ingestion correctness |

---

## 11. V2 Migration Path (Future)

The V1 SQLite persistence is explicitly designed to be replaced. The migration to zero-copy binary mmap:

1. Serialize the SoA `MemoryGraph` vectors (states, gaussians, parent_ids, depths) as a single contiguous binary file
2. On startup, `mmap()` the file and construct `std::span` views directly over it
3. Cold data (texts, annotations) moves to a separate index file or remains in SQLite
4. Implement `MmapStore : public Store` — swap in `CMakeLists.txt`, no other code changes

The SoA layout and abstract `Store` interface make this a frictionless transition.
