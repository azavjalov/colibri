# NUMA / thread-pinning model for glmrt decode on aibox101b (SNC3)

> Companion doc for `tools/numa_preflight.sh` and `tools/perf_decode_probe.sh`.
> Explains WHY decode single-stream leaves ~690 GB/s of DRAM on the floor and
> what the two scripts check / measure. See project `AGENTS.md` for the settled
> ground-truth constants this builds on.

## Box topology (verified `numactl --hardware` / `lscpu`)

- **Single socket** Xeon 6980P (Granite Rapids-AP), 128 physical cores / 256 threads.
- **3 NUMA nodes = SNC3 (Sub-NUMA Clustering, 3-way) enabled in BIOS.** The 3 GNR-AP
  compute dies each front ~1/3 of the package's memory controllers. This is NOT
  dual-socket; node distances 10/15/17 are on-package die-to-die, not cross-socket UPI,
  so the cross-node penalty is real but modest.
- Node CPUs: n0 = 0-42,128-170 · n1 = 43-85,171-213 · n2 = 86-127,214-255
  (each node = 42-43 physical cores + their HT siblings). ~257 GB/node, ~769 GB total.

## The decode gap (what these scripts exist to catch)

Two independent pinning concerns — do not conflate them:

1. **Are threads pinned to cores?** YES. The tuned recipe + glm.c startup set
   `OMP_PROC_BIND=close`, `OMP_PLACES=cores`, `OMP_DYNAMIC=FALSE`,
   `OMP_WAIT_POLICY=active`, `GOMP_SPINCOUNT=200000`, and glm.c re-`execv`s once so
   these bind before the OMP runtime initializes. **The OS scheduler is out of the
   picture** — threads do not migrate.

2. **Are threads pinned NUMA-locally to the weights they compute on?** NO, in decode.
   - Resident expert weights are `mbind`'d round-robin: `expert_node(eid) = eid % 3`
     (glm.c:2320), and the recipe adds `numactl --interleave=all`.
   - The NUMA-aware compute path `numa_bind_pool_to_node(n)` (glm.c:2425) — which sizes
     the pool to one die's cores and pins them so die-n cores read die-n-resident
     weights — is called ONLY at glm.c:4290, gated by `_nc_on` (`NUMA_COMPUTE=1` **and**
     `S >= NUMA_COMPUTE_SMIN=8`).
   - The tuned **decode** recipe runs `NUMA_COMPUTE=0`, and decode is S=1 < 8 anyway.
     So decode runs ONE full pool spanning all 3 dies, reading interleaved/round-robin
     weights → routine **cross-die loads**.

**Consequence:** a single S=1 decode stream (a) issues too few concurrent loads to fill
the memory pipe (low memory-level parallelism) and (b) pulls a large fraction of its
bytes cross-die. Measured effect: ~63 GB/s of the 753 GB/s the DIMMs can deliver.
`NUMA_COMPUTE=1` is S-gated OFF for decode because its 3-pass-per-node fork/join+rebind
overhead (3× per 64-expert block) dwarfs the single-token work (decode collapsed to
0.05 tok/s when it was forced on at S=1).

## `tools/numa_preflight.sh` (read-only validator, no sudo)

Run before any decode benchmark. It:
- Confirms SNC (1 socket + >1 NUMA node) and prints per-node core lists + free memory.
- Validates the pinning env (`OMP_PROC_BIND` / `OMP_PLACES` / `OMP_DYNAMIC` /
  `OMP_NUM_THREADS` / `COLI_MMAP` / `RESIDENT`).
- **WARNS** when `NUMA_COMPUTE=0` and `OMP_NUM_THREADS` > cores-per-node — the exact
  cross-die decode misconfiguration above.
- Emits two ready-to-run A/B launch lines: the baseline (`--interleave=all`, full pool)
  vs a single-die-confined run (`numactl --cpunodebind=0 --membind=0`, team sized to one
  die's cores) — the cleanest test of the cross-die penalty on a single stream.

## `tools/perf_decode_probe.sh <SNAP> <NTH> [tag]` (needs sudo for uncore)

Runs `perf` around a REPLAY decode and captures, into `/tmp/perf_<tag>.*`:
- **Block A (core PMU):** cycles, instructions → IPC; `cycle_activity.stalls_total`,
  `stalls_mem_any`, `stalls_l3_miss`, `stalls_l2_miss`; `mem_load_retired.l3_miss` /
  `l2_miss`; `LLC-load-misses` / `LLC-loads`. Answers *are the cores computing or
  waiting on memory?*
- **Block B (uncore_imc):** `cas_count_read` / `cas_count_write` → actual DRAM GB/s
  pulled during decode (64 B/CAS, system-wide `-a`). Answers *how much of 753 GB/s are
  we really using?*
- **Block C (A/B):** single-die-confined vs `--interleave=all` at matched thread counts,
  comparing IPC + `stalls_mem_any` + tok/s. Answers *does removing cross-die traffic
  raise per-stream throughput?*

Run under `ssh_exec sudo:true` (root overrides `perf_event_paranoid`). Run on a CLEAN
box — perf timing is corrupted by other heavy jobs (e.g. a running conversion).

## Baseline / A-B recipes

See `AGENTS.md` → TUNED RECIPES. For the NUMA A/B, hold everything constant except
`OMP_NUM_THREADS` and the `numactl` binding; keep `TEMP=0 REPLAY=1 REF=ref_bench.json
REF_FORCE=1` so the decode trace is identical across runs.

## Open experiment: SNC3 vs SNC-off (flat) — BIOS

Whether SNC3 helps or hurts single-stream decode is untested. SNC3 gives 3 smaller,
lower-latency local domains (good IF compute is pinned node-local); a flat single-node
interleaves all controllers under one domain (removes the cross-die-placement problem at
the cost of higher average latency). Proposed A/B: capture the SNC3 baseline with the
probe above, then disable SNC in BIOS (needs reboot via BMC/DCUI), re-run identically,
compare tok/s + DRAM GB/s + IPC. Do NOT flip the BIOS bit without capturing the SNC3
numbers first.
