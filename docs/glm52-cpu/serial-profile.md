# GLM-5.2 CPU runtime — single-thread serial cost profile (the fundamental bottleneck map)

This is the authoritative per-phase **serial compute cost** of one decode token, measured
at `OMP_NUM_THREADS=1`, N=1 (no batching), over the full 80-step `ref_bench.json` replay on
the Granite Rapids Xeon 6980P. Running single-threaded removes all fork/join, barrier,
NUMA-rebind and thread-imbalance noise, so the reported per-phase seconds are the *true*
serial arithmetic cost — the correct denominator for deciding which optimizations can matter.

Binary: `glm.fmt4vec.bin` (session source: vectorized fmt=4 absorb helpers + INSTR harness).
Recipe: `SNAP=/srv/models/glm52-mxfp4 CTX=4096 TEMP=0 AMX=1 COLI_MMAP=1 RESIDENT=1
NUMA_PARTITION=1 NUMA_COMPUTE=0 COLI_NO_OMP_TUNE=1 MTP=0 OMP_NUM_THREADS=1
OMP_PROC_BIND=close OMP_PLACES=cores REPLAY=1 REF=ref_bench.json REF_FORCE=1 INSTR=1
numactl --interleave=all ./glm.fmt4vec.bin 256`

## Headline result

```
REPLAY decode: 80 tokens in 746.658s | 0.11 tok/s | expert hit 100.0%
```

| Phase (PROFILE bucket)        | seconds | % decode | notes |
|-------------------------------|--------:|---------:|-------|
| **MoE expert-matmul**         | 471.28  |  63.1%   | dominant; MXFP4 fmt=5 experts, top-8 over 75 sparse layers |
| **Attention (total)**         | 189.99  |  25.4%   | breakdown below |
| &nbsp;&nbsp;output projection `t_aout` (o_proj)         |  83.57  |  11.2%   | fmt=4 grouped-int4, **no AMX** |
| &nbsp;&nbsp;projection/RoPE `t_aproj` (q_a/q_b/kv_a)    |  81.32  |  10.9%   | fmt=4 grouped-int4, **no AMX** |
| &nbsp;&nbsp;score-softmax-value `t_acore` (absorb core) |  25.10  |   3.4%   | the (s,h) collapse(2) region |
| other (unattributed)          |  79.93  |  10.7%   | rmsnorm, router, embed, residual, alloc, glue |
| lm_head                       |   5.46  |   0.7%   | fmt=1 int8 (the one tensor that *can* hit AMX) |
| **total**                     | 746.66  | 100.0%   | buckets sum exactly to wall |

Absorb-core sub-split (`ACORE-SPLIT`, within the 25.10s `t_acore`):
`qabs/W_K 7.13s | score+rope 7.52s | softmax+clat 1.12s | value/W_V 9.27s`.

### Serial cost hierarchy
`MoE (471s) ≫ o_proj (84s) ≈ q/kv-proj (81s) ≫ absorb-core (25s) ≫ lm_head (5s)`

MoE FFN is ~5.6× the next-largest phase and ~94× lm_head. The two dense attention
projections together (o_proj + q/kv = **165s, 22%**) are the entire rest of the meaningful
compute; the attention *absorb core* everyone reaches for first is only 25s (3.4%).

## Why this reorders the optimization levers

The prior session identified two levers from N=1 latency / N=16 throughput symptoms:
- **Lever 1** — parallelize the absorb-core `collapse(2)` over (s,h) to expose >64 granules.
- **Lever 2** — get the attention projections onto AMX at batch.

The serial profile shows their relative *ceilings*:

- **Lever 1 targets `t_acore` = 25s / 3.4%.** Even a perfect fix recovers a fraction of 3.4%
  of decode. It is a latency/utilization fix (N=1 tail), not a throughput mover.
- **Lever 2 targets o_proj + q/kv = 165s / 22%.** ~9× the serial prize of Lever 1, and the
  mechanism (AMX at S≥16) directly attacks dense-matmul cost.
- **MoE = 471s / 63% is untouched by either lever** and is the real dominant cost.

## The fmt=4 AMX-unreachability finding (fundamental)

All five attention projection weights are loaded by `qt_load` at `dbits`:
`q_a, q_b, kv_a` (`t_aproj`), `kv_b` (absorb reconstruction), and `o` (`t_aout`)
— glm.c ~1668–1674. On this snapshot they are all **fmt=4 grouped-int4** (gs=128).

`matmul_qt_ex` dispatches fmt=4 at its very top:

```c
if(w->fmt==4){ matmul_i4_grouped(y,x,w->q4,w->s,S,w->I,w->O,w->gs); return; }   // ~glm.c:1209
```

This `return` fires **before** the idot-eligible block (`allow_idot && g_idot &&
(fmt==1||fmt==2)`, ~1488) and therefore before the AMX gate nested inside it
(`g_amx && w->amx_eligible && S>=g_amx_smin`, ~1497). So for a fmt=4 tensor the AMX path is
**structurally unreachable**, regardless of S. `g_amx=1`, `amx_eligible=1`, `S=16≥smin=16`
are all individually true (verified by instrumentation) yet moot — confirmed empirically:

```
[dbg-oproj] S=16 g_amx=1 amx_elig=1 smin=16 fmt=4 amx_packed=0 cuda_projected=0 pipe_done=0
```

Only `lm_head` (fmt=1 int8) reaches the AMX gate today. Consequently the AMX tile GEMM
(`matmul_q_idot_mm_amx`, validated ~6–8× over VNNI at S=32–128) is doing **nothing** for the
attention projections at any batch size. Getting them onto AMX requires either:
1. an fmt=4 grouped-int4 AMX tile path (dequant-to-int8 per 128-group into the tile, or a
   grouped-scale AMX kernel), or
2. requantizing o_proj / q/kv to fmt=1 int8 (hits the existing gate immediately, at some
   quality cost to be SCORE-gated).

MoE experts are `amx_eligible=0` (streaming slots, set only by `qt_load`) AND MXFP4 fmt=5,
so they are doubly off the AMX path — the largest prize (63%) needs its own investigation.

## Caveat on MOE-SPLIT

`MOE-SPLIT: kernel 941.73s ...` — the "kernel" sub-timer (941s) exceeds its parent
`expert-matmul` bucket (471s). With 288000 parallel-region launches recorded even at 1
thread, this is a nested / per-launch-summed timer, not wall-exclusive. Trust
`expert-matmul = 471.28s` as the real MoE cost; treat MOE-SPLIT internal figures as
relative-only (kernel ≫ bf16-conv 0.13s, orchestration 0.25s). (The double-count is
mechanical: `t_moe_kernel` is incremented once inside the fmt==5 branch spans and again
unconditionally after the branch closes; `t_emm = t_moe_kernel/2 + conv + orch` verifies
exactly against captured logs.)

## Research conclusions — three optimization directions

Each direction was investigated against this snapshot's source
(`c/glm.c.SESSION-full-featured`). Verdicts:

### Direction 1 — absorb-core granularity (Lever 1): **NO-GO**

The absorb core is a `#pragma omp parallel for collapse(2)` over `(s,h)`, H=64, so at N=1
it exposes only `S*H = 64` work-granules for ~120 threads (measured util ~35% vs a 53%
structural ceiling). But it is only **3.4% of serial decode**. The four phases have a strict
data-dependency chain (qabs → score → softmax → clat → vproj); softmax is over the full
score vector, so any kv-dimension split needs a 2-phase (partial-scores → softmax →
weighted-value) restructure with a barrier and, for the reduction-style variants, a
floating-point summation-order change that risks the SCORE 0-argmax-flip gate.

Ceiling arithmetic: even a *perfect, zero-overhead* redesign recovers **~1.5% of total
decode wall-time at most** (the kv-scan slice is 34.7% of a 3.4% region; the fixed-cost
qabs/vproj phases the other 65%). With real barrier/reduction overhead subtracted, under 1%
— at or below the N=1 noise floor (±0.3 tok/s). The only correctness-safe split (by
independent output index, no reduction) is also the lowest-value one.

The one cheap, zero-risk thing worth trying (not a redesign): **NUMA-bind the existing
absorb loop to one node**. The MoE path already does this (`numa_bind_pool_to_node` +
`omp_set_num_threads(node_ncpu)`, gated `S>=g_numa_smin=8`); `attention_rows` never does, at
any S. On a 3-SNC-domain die the default static schedule's 64 threads plausibly straddle
NUMA domains, which would explain 35%-vs-53% util. Wrapping the unchanged loop with the
existing bind precedent is zero INSTR/SCORE risk, but still bounded well under 1% of decode.

### Direction 2 — attention projections onto AMX (Lever 2): **GO, tightly scoped**

Targets o_proj + kv_b (**not** q_a/q_b/kv_a). Key facts:
- fmt=4 today is **weight-only-quantized, activations stay FP32** (`matmul_i4_grouped` takes
  `const float *x`). Every AMX/IDOT path requires int8 activation quant (`qrow_i8`).
- A code comment (matmul_qt_ex ~1432–1438) records that int8 activation quant on
  q_a/q_b/kv_a costs **+0.117 nats/tok (~+12% perplexity), measured** — which is why they run
  `allow_idot=0`. Moving them to AMX reopens that regression (activation-side; weight
  grouping does not fix it). **Exclude q/kv from the initial rollout.**
- o_proj and kv_b already tolerate int8-activation IDOT (the codebase's own trusted set).
- This is a **batch-only lever** (S≥16); S=1 always stays VNNI. At exactly S=16 the 2×2
  register-blocked bulk path (source of the "6–8×" figure) does **not** fire (nSt=1, nSb=0)
  — it falls to the single-tile ragged path. **Batch in multiples of 32** to get the 2×2
  path; must benchmark S=16 vs S=32.
- kv_b caveat: when `kvs!=NULL` (mux/batch-decode) `absorb` is always true, so kv_b runs
  qt_addrow/qt_matvec_rows and never reaches `matmul_qt_ex` — only the S>4 single-seq-prefill
  reconstruct path (`matmul_qt(kvb_all, &l->kv_b, ...)`) is AMX-reachable.

**Plan:** probe first with **Option B(b1)** — load-time dequant fmt4 → `quantize_rows()` →
fmt1 int8 for `l->o` + `l->kv_b`, behind an env flag (e.g. `AMX_REQUANT=1`), applied after
their `qt_load` calls. This needs **zero new kernel code** (the existing fmt=1 AMX path
works) and is the fastest falsification test. If SCORE passes, build the production
**Option A** — a dedicated fmt=4-grouped AMX kernel (gs=128 = 2·KT aligns cleanly; widen
`amx_prepack_i4`'s guard to accept fmt==4; restructure the K-loop into group-nested form:
zero C-tiles per 128-col group, `_tile_stored`+scale with `scl[o,g]`, `+=` into float scratch
across groups). Make the new AMX branch respect `allow_idot` so it auto-scopes to
o_proj/kv_b (allow_idot=1) and skips q/kv (allow_idot=0). SCORE-gate **per tensor**.

### Direction 3 — MoE experts onto AMX (63%, the big prize): **NO-GO for AMX**

Three stacked blockers plus one that survives fixing them:
1. fmt=5 (MXFP4) returns at `matmul_qt_ex:1473`, before the AMX gate.
2. Experts are `amx_eligible=0` by design (streaming slots via `qt_from_disk`).
3. **No AMX-BF16 kernel exists** in the file (only int8 `_tile_dpbssd`); routing MXFP4
   through it needs MXFP4→int8 weight requant (~19 MiB/expert, redone per touch or doubles
   resident RAM), plus double-quantization error.
4. **Arithmetic blocker:** batch-union dedups rows-per-expert. With diverse requests at N=16,
   expected mean rows/expert ≈ **1.27** (128 activations over ~101 distinct experts).
   Reaching mean=16 (to fill an AMX M-tile) needs ~512 diverse concurrent rows — 8× past the
   hard `S<=64` cap. At nr≈1–2, AMX runs 6–12% tile util, slower than the current kernel.

The MXFP4 kernel (`matmul_mxfp4_bf16`) is **already** AVX-512-BF16 (`_mm512_dpbf16_ps`), 4×4
register-blocked, with a SIMD nibble→bf16 LUT — there is no "just vectorize it" win.

**Harness artifact (important):** `run_replay_batch` replays *identical* content across all N
streams, so routing collapses to nu=8 experts/layer at any N (proven: `moe_pcalls=288000`
invariant across N=1 and N=16). The harness's N=16 speedup measures "stop re-reading the same
8 experts 16×," **not** diverse batching — any future MoE-batch experiment needs per-stream
input diversity first.

**Real MoE levers** (the regime is bandwidth/latency-bound: N=16 MoE sustains only ~42 GB/s,
far below socket peak; 90→129 cores *regresses* 4.02→3.54 tok/s at S=1):
- **(safe)** validate + enable `PILOT_CACHE_XE` (`g_pilot_cache_xe`, default-OFF;
  `pilot_cache_eslot`) — software-prefetch the next expert's weights one ahead. Pure
  prefetch = zero correctness risk; targets the diagnosed DRAM→cache latency. Then extend
  prefetch depth 1→N / overlap 2+ expert streams for more memory-level parallelism.
- **(big/speculative)** real continuous/cross-request batching to raise diverse
  rows-per-expert (needs the `S<=64` cap lifted; large cross-cutting serving change).
