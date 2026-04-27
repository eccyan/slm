# Cognitive Ticks: Event-Driven Time for Langevin SDE

**Date:** 2026-04-27
**Status:** Approved

## Problem

The Langevin SDE physics engine uses wall-clock time (`steady_clock`) to compute memory age and drift. If the agent stops working on Friday and returns Monday, 72 hours of drift have occurred — memories archive even though the agent was "asleep." Wall-clock time is fundamentally wrong for an asynchronous AI agent.

## Solution

Replace wall-clock time with **cognitive ticks** — an event-driven counter that only advances when the engine processes a FUSE command (READ, WRITE_COMMIT, READ_ACTIVE). Memories fade through interference (the agent doing other things), not passive clock time.

## Design

### 1. Global Tick Counter

`Scheduler` owns a `std::atomic<uint64_t> global_tick_{0}`. Every Tier 1 command handler increments it by 1 after processing. This is the single source of cognitive time.

The scheduler also tracks `uint64_t last_tier3_tick_{0}` to compute elapsed ticks between physics steps.

**Restart safety:** On engine startup, the persistence layer queries `SELECT MAX(last_access_tick) FROM memory_nodes` and returns it via `Store::max_tick()`. Both `global_tick_` and `last_tier3_tick_` are initialized to this value before any nodes are loaded or physics runs. This prevents unsigned underflow in the age calculation (`current_tick - node.last_access_tick`) which would otherwise wrap to ~2^64 and instantly archive every loaded memory.

### 2. NodeState Changes

```cpp
struct NodeState {
    std::array<float, 2> pos{0.0f, 0.0f};
    uint64_t last_access_tick{0};  // was: double last_access_time
};
```

Remove `SECONDS_PER_DAY` constant — no longer needed.

### 3. SDE Stepper Interface

**Config:**
- `dt` remains as base SDE integration step size (dimensionless)
- `lambda_decay` recalibrated for tick-based age (target: ~3000 ticks to archive)

**step() signature:**
```cpp
auto step(std::span<NodeState> nodes, uint64_t current_tick,
          uint64_t delta_ticks, std::mt19937& rng) -> std::vector<uint32_t>;
```

**Euler-Maruyama scaling** (critical for burst correctness):
- Effective drift step: `effective_dt = base_dt * delta_ticks`
- Brownian noise: `noise_scale * sqrt(2 * g_inv * effective_dt)` — the `sqrt(delta_ticks)` scaling is captured inside `sqrt(effective_dt)`, not applied separately

Without this scaling, a 100-tick burst produces the same displacement as a 1-tick event.

**activate() signature:**
```cpp
void activate(NodeState& node, uint64_t current_tick, std::mt19937& rng) const;
```

### 4. Scheduler Tier 3 Gate

```cpp
void Scheduler::process_tier3() {
    uint64_t current = global_tick_.load();
    uint64_t delta = current - last_tier3_tick_;
    if (delta == 0) return;  // brain is asleep — skip physics
    last_tier3_tick_ = current;
    auto archived = langevin_.step(graph_.all_states(), current, delta, rng_);
    // ... archive processing unchanged
}
```

The 5s wall-clock loop continues for checkpoints and housekeeping. Only the SDE physics step is gated by cognitive activity.

### 5. Persistence Layer

SQLite column change: `last_access REAL` → `last_access_tick INTEGER`.

Since this is v0.1 pre-release, bump the schema version. Existing DBs are wiped on schema mismatch — no migration needed.

The `reactivate_node()` method changes accordingly:
```cpp
void reactivate_node(uint32_t id, float pos_x, float pos_y, uint64_t tick);
```

### 6. Lambda Decay Calibration

With `dt=1.0` (dimensionless base step) and `lambda_decay=5e-6`:
- Simulated archival at tick 3039 (from r=0.01 to r=0.95)
- With ~200 interactions/day: ~15 days to archive
- Tunable via `--lambda-decay` CLI flag

### 7. Unchanged Components

| Component | Why unchanged |
|-----------|--------------|
| FUSE frontend | No timestamps in wire protocol (already server-side) |
| Cooker / SHM header | No time fields in MemoryFSHeader |
| MCP server | Reads SQLite directly; column rename is transparent |
| Checkpoint cadence | Stays at 60s wall-clock (crash recovery) |
| Tier 1 poll interval | Stays at 100µs wall-clock |

## Files to Modify

| File | Change |
|------|--------|
| `src/langevin/include/langevin/poincare_disk.hpp` | `last_access_time` → `last_access_tick`, remove `SECONDS_PER_DAY` |
| `src/langevin/include/langevin/sde_stepper.hpp` | Update signatures for tick-based time |
| `src/langevin/src/sde_stepper.cpp` | Euler-Maruyama with `delta_ticks`, tick-based age |
| `src/engine/include/engine/scheduler.hpp` | Add `global_tick_`, `last_tier3_tick_` |
| `src/engine/src/scheduler.cpp` | Increment tick on commands, gate Tier 3 |
| `src/engine/src/main.cpp` | Recalibrate default `lambda_decay` |
| `src/engine/src/memory_graph.cpp` | Snapshot serialization uses tick |
| `src/persist/include/persist/store.hpp` | Update `reactivate_node` signature, add `max_tick()` |
| `src/persist/src/sqlite_store.cpp` | `last_access_tick INTEGER`, bind as int64, implement `max_tick()` |
| `python/slmfs/mcp_server.py` | Read `last_access_tick` column |
| `tests/test_langevin.cpp` | All tests updated for tick-based API |
| `tests/test_engine.cpp` | If exists, update for tick-based scheduler |
| `tests/python/test_mcp_server.py` | Update test DB schema |

## Risks

- **Tick starvation:** An agent that does very few operations will have very slow drift. This is by design — infrequent use means memories persist longer.
- **Burst flooding:** A bulk `slmfs add` of 1000 chunks produces 1000 ticks in seconds. The `delta_ticks` Euler-Maruyama scaling handles this correctly.
- **Schema break:** v0.1 pre-release, acceptable to wipe DB.
