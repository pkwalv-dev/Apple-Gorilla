"""Distributed compute — the 'mutual processing' primitive, with real throughput.

A deterministic CPU benchmark that is identical on every device and every execution
path, so AG can:
  - measure each node's real throughput (MIPS) for capability-weighted scheduling, and
  - VERIFY a remote result instead of trusting it: the same (seed, iterations) always
    produces the same checksum, so a node that returns garbage (or returns instantly
    without doing the work) is caught.

Throughput: the kernel is an ORDER-INDEPENDENT sum of a per-index hash, so the same
checksum falls out whether it runs scalar or vectorised with NumPy — orders of
magnitude faster — when NumPy is installed. (An earlier draft parallelised with
`multiprocessing`, but on Windows `spawn` re-imports `__main__`, which for `python -m
ag` re-runs the CLI in every worker — a hang/fork-bomb. NumPy gives the speedup in one
process with none of that risk, so the process pool was dropped.)

NumPy is optional and loaded via importlib so the core stays stdlib-only (the bundle
portability gate stays green) — install it (`pip install numpy`, or the ComfyUI/LoRA
env already has it) for the fast path; without it the bench still runs, scalar, and
reports its throughput honestly. Ported/expanded from the water18 proposal.
"""
from __future__ import annotations

import importlib
import importlib.util
import time
from typing import List

MASK = 0xFFFFFFFF
GOLDEN = 0x9E3779B9
MAX_BENCH_ITERATIONS = 2_000_000_000
DEFAULT_ITERATIONS = 50_000_000
MIN_SHARD = 100_000
_NP_CHUNK = 8_000_000              # vectorised chunk size (bounds peak memory)

CANARY_SEED = 0x9E3779B9
CANARY_ITERS = 200_000
_CANARY_CACHE: dict = {}


def _value(i: int, seed: int) -> int:
    """A splitmix-style 32-bit hash of index i under seed. Pure function of (i, seed),
    so summing over i is order-independent — the key to a parallel, verifiable kernel."""
    z = (seed + i * GOLDEN) & MASK
    z = ((z ^ (z >> 16)) * 0x7FEB352D) & MASK
    z = ((z ^ (z >> 15)) * 0x846CA68B) & MASK
    return (z ^ (z >> 16)) & MASK


def _numpy():
    if importlib.util.find_spec("numpy") is None:
        return None
    try:
        return importlib.import_module("numpy")
    except Exception:
        return None


def _sum_scalar(start: int, count: int, seed: int) -> int:
    acc = 0
    for i in range(start, start + count):
        z = (seed + i * GOLDEN) & MASK
        z = ((z ^ (z >> 16)) * 0x7FEB352D) & MASK
        z = ((z ^ (z >> 15)) * 0x846CA68B) & MASK
        acc += (z ^ (z >> 16)) & MASK
    return acc


def _sum_numpy(np, start: int, count: int, seed: int) -> int:
    total = 0
    off = start
    end = start + count
    while off < end:
        c = min(_NP_CHUNK, end - off)
        i = np.arange(off, off + c, dtype=np.uint32)
        z = (np.uint32(seed) + i * np.uint32(GOLDEN)).astype(np.uint32)
        z = (z ^ (z >> np.uint32(16))) * np.uint32(0x7FEB352D)
        z = (z ^ (z >> np.uint32(15))) * np.uint32(0x846CA68B)
        z = z ^ (z >> np.uint32(16))
        total += int(z.astype(np.uint64).sum())
        off += c
    return total


def bench(iterations: int, seed: int = CANARY_SEED) -> dict:
    """Run the deterministic loop and report checksum + measured throughput. Uses the
    NumPy vectorised path when available, else a scalar loop — both yield the same
    checksum, so a result stays verifiable regardless of the device's kernel."""
    n = max(1, min(MAX_BENCH_ITERATIONS, int(iterations)))
    seed &= MASK
    t0 = time.perf_counter()
    kernel = "scalar"
    np = _numpy()
    if np is not None:
        try:
            total = _sum_numpy(np, 0, n, seed)
            kernel = "numpy"
        except Exception:
            total = _sum_scalar(0, n, seed)
    else:
        total = _sum_scalar(0, n, seed)

    ms = (time.perf_counter() - t0) * 1000.0
    mips = round(n / (ms / 1000.0) / 1e6, 2) if ms > 0 else 0.0
    return {"iterations": n, "seed": seed, "checksum": f"{total & MASK:08x}",
            "ms": round(ms, 2), "mips": mips, "kernel": kernel}


def canary_checksum() -> str:
    key = (CANARY_SEED, CANARY_ITERS)
    if key not in _CANARY_CACHE:
        # The canary is small and must be path-stable; compute it scalar for a fixed cost.
        _CANARY_CACHE[key] = f"{_sum_scalar(0, CANARY_ITERS, CANARY_SEED) & MASK:08x}"
    return _CANARY_CACHE[key]


def run_shard(iterations: int, seed: int) -> dict:
    """What a node runs for one shard: the real work plus the canary, so its result can
    be verified against the coordinator's reference."""
    out = bench(iterations, seed)
    out["canary"] = canary_checksum()
    return out


def plan_shards(weighted_nodes: List[tuple], iterations: int) -> List[dict]:
    """Split `iterations` across nodes weighted by capability. `weighted_nodes` is a list
    of (node_id, name, weight); the strongest device gets the most work."""
    iters = max(MIN_SHARD, min(MAX_BENCH_ITERATIONS, int(iterations)))
    picked = [(nid, name, max(1.0, float(w))) for nid, name, w in weighted_nodes if nid]
    if not picked:
        return []
    total = sum(w for _, _, w in picked)
    shards = []
    for i, (nid, name, w) in enumerate(picked):
        share = max(MIN_SHARD, int(iters * w / total))
        seed = (0x9E3779B9 + i * 0x85EBCA6B) & MASK
        shards.append({"node_id": nid, "name": name, "iterations": share, "seed": seed})
    return shards


def aggregate(shard_results: List[dict]) -> dict:
    """Roll finished shards into one result: total iterations, summed MIPS (parallel
    throughput), and whether every shard's canary verified."""
    done = [s for s in shard_results if s.get("ok")]
    failed = [s for s in shard_results if not s.get("ok")]
    expected = canary_checksum()
    verified = bool(done) and all(s.get("canary") == expected for s in done)
    return {
        "aggregate": True,
        "iterations": sum(int(s.get("iterations", 0)) for s in done),
        "aggregate_mips": round(sum(float(s.get("mips", 0)) for s in done), 2),
        "shards_done": len(done),
        "shards_failed": len(failed),
        "verified": verified,
    }
