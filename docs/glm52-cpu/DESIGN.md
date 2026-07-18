# GLM-5.2 Purpose-Built Runtime — Design (v2)

**Codename:** working title `glmrt` (GLM runtime). Forked from colibri (`glm.c`), harvested down to a
purpose-built serving runtime for **GLM-5.2 only**, on **our target hardware configs only**.

## 0. Intent (the pivot)

colibri's philosophy = run a 744B MoE on a RAM-starved consumer box by **streaming experts from disk**.
That is the *opposite* of our deployment. We keep colibri as a **base building block** (its correct GLM-5.2
forward pass, tokenizer, MLA/DSA/MoE data flow, int4 weight format, our AMX kernel, the token-exact oracle)
and **replace its execution model** with one built for a big-RAM AMX Xeon:

> **Whole int4 model resident in RAM, no disk streaming, AMX compute, all cores pinned, continuous-batching
> multi-stream server, NUMA-partitioned per SNC node.**

Empirical justification (measured this project):
- colibri streaming on GLM-5.2 (int8, 744 GB > RAM): **0.14 tok/s decode, 69% of prefill = disk-WAIT.**
- DeepSeek-V4 (150 GB, fits RAM) via the resident+NUMA+AMX SGLang golden config: **11.4 tok/s.**
- **The difference is residency + execution model, not the kernel.** Our AMX GEMM already matches oneDNN
  (100-120% at batch, §7 of CHECKPOINT.md).

## 1. Target configs

**CPU-ONLY across all configs for now** (GPU tiers explicitly deferred — no GPU backend seam in MVP):
1. GNR 6980P, RAM-resident, NUMA-partitioned, AMX — **the base + MVP focus**.
2. (future) GNR + 2× B70 hot-expert tier.
3. (future) GNR + RTX 6000 Blackwell hot-expert tier.
Configs 2/3 are CPU-identical; only a future optional GPU expert-tier differs. Design the CPU core cleanly;
add a GPU seam later, not now.

**Box facts:** GNR 6980P, 128 physical cores / 256 threads, **SNC=3** (node0 cores 0-42, node1 43-85,
node2 86-127; HT siblings 128-255), 751 GiB RAM, 7 GB swap. AMX int8/bf16. gcc 15.2.

## 2. Model / memory model

**Serving target = `/srv/models/glm52-i4x`** (true grouped-int4, `--xbits 4 --group-size 128`, ≈375 GB).
NOT `/srv/models/glm52-i4` (that one stored experts at int8 = 744 GB, doesn't fit with KV headroom — the
default `--xbits` didn't propagate; fixed with explicit `--xbits 4`, verified 6.29 MB/expert int4).

- GLM-5.2 = glm_moe_dsa: 78 layers (first 3 dense, 75 MoE), 256 routed experts, top-8, 1 shared expert,
  hidden 6144, moe_intermediate 2048, MLA attention + DSA lightning-indexer sparse attention.
- Per expert (int4): gate/up [2048,6144] ≈6.3 MB, down [6144,2048] ≈6.3 MB → ≈18.9 MB/expert.
  256 × 75 MoE layers ≈ **363-375 GB experts** + ≈19 GB dense/attention/shared/io ≈ **~395 GB resident**.
- **Fits 751 GB with ~350 GB headroom** for KV caches (multi-stream!), activations, scratch, OS.

**Residency:** hold ALL experts resident as directly-indexed QT, wired once at load. Two impl options:
- (a) `mmap(MAP_SHARED)` the shards + prefault (MADV_WILLNEED + touch) → QT are zero-copy views, page cache
  holds them; simplest, and colibri's `COLI_MMAP=1` path already does exactly this per-expert.
- (b) malloc-resident + `mlock` (needs raised `ulimit -l`) → guaranteed no eviction.
  MVP uses (a) with prefault (proven pattern, no ulimit fuss); revisit (b) if page-cache eviction ever bites.

## 3. Resident-expert conversion (CONFIRMED clean — subagent code analysis)

The streaming machinery is **not woven through** the forward pass; it hangs off a **single seam** in `moe()`
(glm.c:3013-3555): `(layer,eid) → ESlot* → ->g/->u/->d (QT)`. `expert_gate_up`/`matmul_qt` take `QT*` and
never inspect where it came from. So:
- **Change in moe():** ~25-30 lines — delete the pin/ecache resolve scan (3288-3296), the miss/`expert_load`/
  PIPE-dispatch block (3358-3369), and the LRU-promotion swap (3508-3512). Replace with a **direct index into
  a resident `QT experts[layer][eid].{g,u,d}` array**.
- **Wire once at load:** loop the existing `expert_load()` over all (layer,eid) with `#pragma omp parallel for`
  (this is exactly what `pin_load()` already does for a ranked subset — make it total/unconditional).
- **Delete ~1000 lines** of streaming plumbing (all self-contained, no tendrils into attention/dense/step):
  `uring_*`, PIPE thread pool, `pilot_*`, `couple_*`, `repin_*`, `pin_load/pin_wire`, `g_expert_budget`,
  `g_cache_route`, `g_direct`/O_DIRECT, `ehit_mark`/telemetry, `.coli_usage`/`.coli_kv` persist,
  CUDA/Metal/ARM backend hooks (already compiled out by CUDA?=0/METAL?=0).
- **Verdict: (A) clean/localized.** Risks: (1) use the i4x checkpoint not the int8 one; (2) hybrid GPU
  overlap logic is entangled with miss/hit split — irrelevant for CPU-only MVP; (3) eager whole-model load =
  one big blocking startup (~19200 expert_load calls) — fan out with OMP, expect a minutes-long load.

## 4. Threading / core pinning (requirement)

- **Pin all worker threads to PHYSICAL cores only** (no HT siblings) — matches the golden bind that gave
  11.4 tok/s. Explicit `pthread_setaffinity_np` per thread, not left to OMP defaults.
- **Reserve a few cores per NUMA node** for OS / orchestration / web-streaming (HTTP, tokenize, detokenize,
  scheduler). E.g. per SNC node use ~40 of 43 phys cores for compute, leave ~3 for the runtime's non-compute
  work. Compute threads = one pool per NUMA node bound to that node's physical cores.
- Baked into the runtime (not env vars): a startup topology detect (lscpu -p) → compute-core set + reserved
  set, like the golden `detect-topology.sh` but internal.

## 5. Multi-stream continuous-batching scheduler (requirement — production, and it makes AMX pay in decode)

**Why it matters for the kernel:** single-stream decode = S=1 GEMV = AMX no-win (bandwidth-bound). **N concurrent
streams → decode batches to S=N → GEMM → AMX 2-10×.** Multi-streaming is what turns our kernel into a decode
win, not just prefill. (Anchors: SGLang DeepSeek-V4 on GNR+1GPU = 33 tok/s single → 218 aggregate @ N=12; CPU
path scales the same in aggregate.)

**Design:** persistent server holding N sessions, each with its own KV cache (MLA-compressed, ~cheap). Core loop
= **continuous batching**:
1. Scheduler gathers all sessions with a ready token into one batch (prefill chunks + decode steps mixed or
   phase-separated).
2. Attention: per-sequence (each has own KV / position) — MLA + DSA per row.
3. **MoE: batch-union routing** — across the whole batch, for each unique expert gather its assigned token rows
   and do **one AMX GEMM over all those rows** (this is exactly the Intel `moe_align_block_size` 32-row-block
   pattern, and colibri's `moe()` *already* groups rows per expert — so batching layers on naturally). At S≥8-32
   per expert the AMX kernel is in its 2-10× regime.
4. Scatter results, sample per sequence, stream tokens out per session.

**KV headroom:** ~350 GB free after weights → plenty for many concurrent MLA KV caches (MLA's 57× compression
helps). Cap concurrency by KV budget.

## 6. Milestones

- **MVP-1 (correctness + single-stream floor):** resident int4 experts (single NUMA, all resident via mmap+
  prefault), AMX compute, strip streaming. Run GLM-5.2 end-to-end. Validate **token-exact vs oracle**, measure
  prefill/decode tok/s vs the 0.14 streaming baseline (expect prefill disk-wait → ~0; decode still S=1 so
  modest, but no disk stall).
- **MVP-2 (multi-stream, the real win):** continuous-batching scheduler, N sessions, batched decode → AMX GEMM.
  Measure **aggregate tok/s scaling with N** (this is where AMX shows in decode). AMX=1 vs AMX=0 A/B.
- **MVP-3 (NUMA-partition + pinning):** one expert-set per SNC node, first-touch local alloc, pinned compute
  threads + reserved OS cores. Addresses the measured single-SNC-domain AMX scaling ceiling.

## 7. Keep / Delete / Build summary

**KEEP:** GLM-5.2 forward (`step`, `attention_rows`, `dense_mlp`, MLA, DSA indexer), MoE routing math + shared
expert, tokenizer (`tok.h`), int4/int8 QT weight format + `qt_load`/`expert_load` (loader reused for resident
wiring), our AMX GEMM (`matmul_q_idot_mm_amx` 2×2), VNNI fallback, oracle (`ref_glm.json` / TF mode).
**DELETE:** disk-streaming (ESlot slab/pread, cap/LRU/hit-miss, uring/PIPE/pilot/couple/repin/pin, EXPERT_BUDGET,
CACHE_ROUTE, O_DIRECT, .coli_kv/.coli_usage), CUDA/Metal/ARM backends + portability shims, telemetry dashboard.
**BUILD:** resident `QT experts[layer][eid]` wired once; continuous-batching multi-stream scheduler w/ per-session
KV + batch-union MoE; topology-aware thread pinning w/ reserved cores; NUMA-partitioned expert placement +
first-touch; a lean OpenAI-ish server loop for the web-streaming front end.

## 8. Status

- int4 re-convert (`glm52-i4x`) in progress (~128 GB/48 shards of 141, ~375 GB target), detached
  (`conv-glm52-i4.sh`, log `conv-glm52-i4.log`, pid 152291). Verified experts = true int4.
- Next after convert: `coli plan` on i4x to confirm 0-cold residency + KV headroom; then start MVP-1
  (fork the tree, strip streaming, resident wiring, run + validate).

## 9. Progress + a NUMA finding that reorders the milestones

**int4 re-convert DONE:** `/srv/models/glm52-i4x` = 141 shards, 369 GB true grouped-int4 (config+tokenizer copied
from glm52-i4). This is the serving target.

**MVP-1 DONE** (fork `/home/intel/b70-sglang-xpu/glmrt/c/`): resident experts wired once via `expert_load()` over
all (sparse-layer, eid), forced mmap zero-copy; `moe()` resolve-scan direct-indexes `m->experts[layer][eid]` under
`RESIDENT=1` (default). **Disk-WAIT 60.0s→0.0s**, prefill 87s→65s, decode **0.14→2.7-2.9 tok/s**, token-exact vs
streaming (byte-identical IDs, TEMP=0). `amx: on`, builds clean.

**NUMA finding (reorders milestones):** the box is **3 NUMA nodes × ~257 GB**, not one 751 GB pool. The 369 GB
resident model does NOT fit one node → spills cross-socket → **expert-matmul rose 18.8s→42.7s** (cross-node reads).
So **MVP-3 (NUMA-partition) is now REQUIRED and next**, before MVP-2: shard experts across the 3 SNC nodes, first-
touch local, compute pinned to the node holding its experts, reserved OS cores. Then MVP-2 (multi-stream) builds on
a NUMA-correct base and routes each expert's batched rows to its owning node.

**Revised milestone order: MVP-1 (done) → MVP-3 (NUMA-partition) → MVP-2 (multi-stream).**
