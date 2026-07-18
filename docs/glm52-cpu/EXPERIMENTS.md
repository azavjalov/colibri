# GLM-5.2 CPU Runtime (glmrt) — Experiments Ledger

Quick-reference tracker for every optimization experiment: what was tried, the measured
result, and whether it's worth **re-trying under multi-stream** (the next phase). The
detailed narrative lives in `CHECKPOINT.md`; this file is the scannable index.

**Host:** aibox101b — single-socket Granite Rapids Xeon 6980P, 128 physical cores, 3 SNC
NUMA nodes, 751 GB DDR5-8800 MRDIMM (~753 GB/s aggregate, ~252 GB/s/node).
**Engine:** `glmrt` (lean pure-C fork of colibri), `/home/intel/b70-sglang-xpu/glmrt/c/`.
**Model:** GLM-5.2 (744B MoE, glm_moe_dsa: MLA + DSA sparse attn + MoE 256 experts top-8, 78 layers).
**Serving target:** `/srv/models/glm52-mxfp4` (369.9 GB, whole model RAM-resident via mmap).

---

## Current baseline (single-stream) — THE BANKED RESULT

| metric | value | vs int4 |
|---|---|---|
| decode | **2.99 tok/s** | +33% |
| prefill S=642 expert-matmul | **12 s** | vs 16.6 s |
| accuracy (rel-err vs fp8) | **10.7%** | vs 12.8% |

**Recipe:** `run-mxfp4.sh` = MXFP4 model + single NUMA node (40 threads, `numactl
--cpunodebind=0`), NUMA 3-pass OFF, MTP OFF. `AMX=1 COLI_MMAP=1 RESIDENT=1
NUMA_PARTITION=0 NUMA_COMPUTE=0 COLI_NO_OMP_TUNE=1 MTP=0 OMP_NUM_THREADS=40`.

**Kernel:** MXFP4 experts (E2M1 nibbles + E8M0 gs=32 group scales) → BF16 → AVX512-BF16
`_mm512_dpbf16_ps`, 4×4 register-blocked (`matmul_mxfp4_bf16` in glm.c, fmt=5). NOT AMX tiles.

**Measured bottleneck:** single-stream decode is **latency / fork-join / per-expert-small-GEMV
bound**, NOT bandwidth-bound (runs at ~10% of 753 GB/s peak). Top-8/256 sparsity → even S=642
prefill gives only ~20 rows/expert. ~83k OMP parallel-region launches per prefill token.

---

## Experiments

Legend: ✅ win (kept) · ❌ ruled out · ⏸ parked (unresolved) · 🔁 = re-try under multi-stream

| # | Experiment | Result | Status | Multi-stream? |
|---|---|---|---|---|
| E1 | **RAM-resident experts** (MVP-1): mmap whole int4 model, no disk streaming | disk-wait 60s→0; decode 0.14→2.8 tok/s (~20×) | ✅ | keep |
| E2 | **MXFP4 kernel** (E2M1+E8M0, AVX512-BF16 dpbf16, ported from kt-kernel GemmKernel224) | +33% decode, +37% prefill, better accuracy vs int4; kernel A/B 1.5-3.8× at S≥8 | ✅ | keep — wins bigger at batch |
| E3 | **NUMA 3-pass compute** (MVP-3b): per-node expert shard + pinned teams, 3 sequential passes | prefill matmul 42.7→16.6s for int4; but per-block rebind CHURN makes MXFP4 ~10× slower (127s) → single-node wins (12s) for single-stream | ❌ single-stream | 🔁 **YES** — was the point of MVP-3b; fix nesting (pass-outside-block, rebind once/pass/layer) for batched all-3-node compute |
| E4 | **MTP speculative decode** (native head, layer 78, draft=3) | head converts + activates, but **0-4% draft acceptance → net slowdown** (1.4 vs 2.99). draft0 (g=0) itself wrong. | ⏸ parked | 🔁 maybe — needs the g=0 bug found first (see MTP sub-ledger) |
| E5 | **Hot-expert domain tier** (pin/replicate experts specialized to coding/ESXi domains) | **NO exploitable skew:** 0.1% specialists (vs ref 7.9%), leave-one-out 8.3% WORSE than chance. GLM router doesn't separate technical sub-domains. | ❌ closed | no |
| E6 | **Kernel tiling for small-nr** (parallelize over O vs O/4; ldexpf precompute; batch-union) | ldexpf-precompute kept (correct); the 8× slowdown was NOT tiling — it was E3's rebind churn | (subsumed by E3) | — |

### MTP sub-ledger (all applied, none fixed 0% acceptance — g=0 draft still wrong)
| fix | source | result |
|---|---|---|
| Bug A: chain `shared_head.norm(hx)=row` not raw `hx` between draft steps | ref-diff vs DeepseekModelNextN | applied, still 0% |
| eh_proj at F32 (full precision, vs int4) | precision hypothesis | applied, still 0% |
| Bug B: position `p=kv` not `kv-1`; mtp_absorb `pos_base+1` | ref position semantics | applied, still 0% |
| **Proven correct** (numerical debug): eh_proj weights (cosine 0.98-0.995, not transposed), RMSNorm `x*w` not `1+w`, norms/eps/lm_head, hlast, full projection chain reproduces reference logit dist | Sonnet5 numerical dump | H1/H2/H3 refuted |
| **Remaining suspect:** the MTP **transformer block** (`layer_forward(mtpL)`) or its KV bookkeeping — needs a numerical dump of the MTP layer's intermediate output vs a reference PyTorch forward | — | TODO if MTP revisited |
Note: this checkpoint self-EOSes after ~4 tokens on most prompts, structurally limiting MTP's value regardless.

---

## Next flows (from CPU-opt research — ranked, for future work)

Full report in CHECKPOINT.md. Lead candidates:

**Single-stream (current regime):**
- **F1. Finish MTP + batch-union verify** — converts S=1 GEMV → batched verify GEMM, directly
  attacks the bottleneck. Blocked on the g=0 bug (E4). Caveat: colibri#8 + ktransformers#2088
  found MTP net-negative on *bandwidth*-bound CPU boxes; we're *latency*-bound so it may transfer
  — must measure, and must dedupe experts across the draft window into one GEMM/expert.
- **F2. Routing-lookahead software prefetch (L2/L3)** 🌟 — colibri PILOT: next-layer routing
  ~71.6% predictable. Adapt its disk-readahead to `_mm_prefetch` of next expert-block weights
  during current compute. Attacks latency independent of BW (we have 90% BW headroom). Low risk
  (a wrong prefetch just wastes a hint). NOT shipped anywhere in this form. **Best new single-stream bet.**
- **F3. MTP-guided prefetch** — use the draft head only to *warm* experts (not accept tokens),
  sidesteps F1's net-negative risk; needs only directionally-sane routing (lower bar than E4).

**Multi-stream (next phase):**
- **F4. Grouped/variable-M GEMM as first-class primitive** — batch at the EXPERT level (gather all
  tokens routing to expert E across streams into one GEMM), same batch-union as F1. Build once.
- **F5. Continuous batching + chunked prefill** — SGLang CPU pattern: 1 TP rank = 1 SNC via
  per-SNC core binding (their Xeon 6980P example matches our chip).
- **F6. Persistent NUMA-pinned worker pools** (bind once at load, lock-free queue) — fixes E3's
  rebind churn without llama.cpp's `--numa mirror` (infeasible: 382-460GB × replication > 751GB).
- **F7. AMX-tile kernel for wide-M** (oneDNN PR#5495 K-chunking-residency: keep AMX C-accumulator
  resident across the K loop) — only pays once F1/multi-stream produce wide-M. Sequel, not standalone.

**Closed / not applicable:** DSpark/ds4#468 (GPU-only ds4; DSpark = a DeepSeek-V4-specific trained
draft artifact, no GLM version, adopting = training a draft model, out of scope — fixing our own
MTP captures the same class). CACHE_ROUTE (no cache-miss cost when RAM-resident). GPU/CPU expert
tiering (no GPU). Intel IPEX (archived). Global hot-expert freq caching (aux-loss-free routing
defeats it).

**Framework status:** ktransformers GLM support WIP/unstable; SGLang DeepSeek-V4 CPU PR#24976 still
unmerged, no GLM-5.2 CPU; llama.cpp runs GLM/DeepSeek MoE GGUF generically but untuned; oneDNN
MoE-on-CPU immature (grouped-GEMM experimental, GNR AMX-gap PR#5495 draft). **Nobody has a mature
solution for our exact shape+hardware.**

---

## Reproducibility

- **Fast single-stream:** `cd /home/intel/b70-sglang-xpu/glmrt/c && ./run-mxfp4.sh "prompt"`
- **Kernel A/B:** `./mxfp4bench 6144 2048 40` (I=6144 O=2048, 40 threads)
- **Expert-atlas domain sweep:** `./atlas-sweep-tech.sh` then `python3 tools/expert_atlas/analyze.py --stats atlas-tech/stats ...`
- **Microarch profiling:** `sudo perf stat -a -M tma_* -- sleep 5` during a detached run (in-window verified via `[prefill] layer N/78`).
- **NUMA/MLC baseline:** `/home/intel/tools/mlc/Linux/mlc`.
- Backups: `glm.c.bak-{mxfp4,mtpfix,bugB,instr}`. Model backups: index `.bak-pre-mtp`.
