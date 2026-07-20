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

---

## Results — what was implemented and what was rejected

Two experiments were run to conclusion off the back of the research above.

### LANDED — Direction 2, Option B: requant o_proj + kv_b fmt=4 → int8 (`AMX_REQUANT`, default ON)

Commit `d5b6e2f` (`GLM-5.2: requant o_proj/kv_b fmt=4->int8 to reach AMX tile path`).

At load time, both the MLA output-projection (`o_proj`) and `kv_b` weights are dequantized
from fmt=4 grouped-int4 back to fp32 and re-quantized to fmt=1 int8 (per-row scale) via the
existing `quantize_rows()`. This is a **load-time plumbing change only — no new kernel** —
reusing the fmt=1 AMX path that already exists. Gated by `AMX_REQUANT` (default ON;
`AMX_REQUANT=0` disables). A one-line `[amx-requant] requantized 156 MLA proj tensors …`
summary prints at model init (156 = 78 layers × {o_proj, kv_b}).

Two independently-confirmed mechanisms:
- **o_proj genuinely reaches AMX** — with fmt=1 it now passes `amx_prepack_q8` +
  `matmul_q_idot_mm_amx` (confirmed AMX-packed on 78/78 layers). ~2.7–2.9× on the o_proj op.
- **kv_b's absorb-path dequant gets cheaper** — per-row int8 vs per-128-group int4. Note
  kv_b is **not** AMX (in batch decode `kvs!=NULL` forces `absorb=true`, so kv_b runs
  `qt_addrow`/`qt_matvec_rows` and never enters `matmul_qt_ex`); the win is purely
  simpler-per-element dequant. (A dedicated fmt=4 AMX kernel — "Option A" — would therefore
  gain nothing on kv_b.)

**Throughput:** ~**1.3–1.4× aggregate** decode at N=16/N=32 (conservative, cross-validated;
the raw 1.9× seen at N=16 was noise-inflated by unrelated buckets — the clean N=32 number is
1.35× with o_proj+kv_b accounting for 94% of the step-time reduction). **S=1 unaffected**
(stays VNNI — AMX only gates in at S≥16).

**Quality (SCORE gate, ABSORB=1, 36 probes, on the landed committed build):**
`0/36 argmax/greedy flips`, nat/tok delta **+0.0033** (baseline 2.7941 → 2.7974). ~35× under
the +0.117 nat/tok that disqualified q_a/q_b/kv_a from int8 activations — those attention
input projections are deliberately **left as fmt=4** (their loss comes from int8 *activation*
quantization, which requant does not fix).

**Cost:** resident +3.94 GB (+1.07%, 377316→381348 MB); model load +11.5 s one-time
(14.0→25.5 s). Both trivial / off the decode critical path.

### REJECTED — Direction 3 (safe lever): `PILOT_CACHE_XE` expert-weight prefetch

Clean negative; **left at its default (OFF), nothing landed.** Triangulated across N=1, N=16,
and a patched lookahead-depth sweep (1/3/7):
- N=1: OFF 3.996 vs ON 3.874 tok/s (−3.1%, inside noise). N=16: 16.230 vs 16.174 (−0.3%).
- Depth sweep showed **no dose-response** (depths 1/3/7 statistically indistinguishable at
  N=1; depth 7 was −1.8% *worse* at N=16) — so it is not a "needs a longer shadow window"
  problem.
- Consistent with the bandwidth diagnosis: the regime runs at ~5% of peak DRAM bandwidth, so
  weight-fill latency is not contention-bound, and seeding ~0.1% of an expert's block adds
  nothing the HW L2 streamer wasn't already getting from the matmul's natural access pattern.
  Root-causing the MoE bottleneck further needs stall-cycle profiling of the kernel itself,
  not more prefetch tuning.

---

## Per-function hot-spot profile (`perf record`, independent verification)

The INSTR phase timers above are wall-clock brackets around code regions. To (a) confirm them
against an independent method and (b) get *function*- and *instruction*-level attribution, a
`sudo perf record -F 999 --call-graph dwarf` was run on a disposable
`-g -fno-omit-frame-pointer` build (`glm.perf.bin`, numerically identical to the committed
binary; note the Makefile `LDFLAGS -lm -pthread -lnuma` must be added to the literal compile
line or the link fails). 20-step N=1 single-thread decode (`ref_bench_20steps.json`; there is
**no `STEPS` env knob**, so a truncated ref file is the only way to cap step count without
editing source).

**Cold-cache caveat that had to be handled:** on a cold run, model load (~172 s) is roughly
equal to decode (~175 s), so a naive whole-run flat profile misattributes ~50% of samples to
load. The decode window was isolated with `perf report --time <start>,<stop>`, derived from the
program's own reported decode duration (174 K decode-window samples, 0 lost — matches
999 Hz × 175 s).

### Decode-only flat profile (self %)

| Self % | Symbol | Phase |
|-------:|--------|-------|
| 66.70% | `matmul_mxfp4_bf16._omp_fn.0` | MoE routed/sparse experts (MXFP4 × BF16) |
| 19.17% | `matmul_i4_grouped._omp_fn.0` | **three call sites** — see split below |
| 6.30%  | `matmul_q_idot._omp_fn.0` | 5.67% o_proj (VNNI `dot_i8i8`) + 0.63% lm_head |
| 5.88%  | `attention_rows._omp_fn.1` (self) | absorb-core (inlined `qt_matvec_rows`/`qt_addrow`, kv_b) |
| 1.50%  | `matmul._omp_fn.0` | MoE router / gate logits (unquantized) |
| ≤0.04% | `moe.constprop.0`, `__expf_fma`, `matmul_qt_ex`, `f32_to_bf16_buf`, memmove | orchestration / softmax exp / dispatch / activation conv / memcpy |

Top 5 = **99.55%** of decode self-time. `rmsnorm`/`rope` are fully inlined and never appear as
distinct symbols; `qrow_i8` (int8 activation quant) rounds to 0.00%. Everything past the top ~9
userspace symbols is sub-0.02% OS scheduler-tick noise.

**Cross-validation vs the INSTR timers — agreement within ~1 pp on every phase:**

| perf-sampled | value | INSTR timer | value |
|--------------|------:|-------------|------:|
| o_proj (`matmul_q_idot` attn slice) | 5.67% | output projection | 5.84% |
| absorb-core (`attention_rows` self) | 5.88% | score-softmax-value | 5.48% |
| q/kv-proj (`matmul_i4_grouped` attn slice) | 10.38% | projection/RoPE | 9.68% |
| lm_head | 0.63% | lm_head | 0.64% |
| routed experts (`matmul_mxfp4_bf16`) | 66.70% | expert-matmul | 68.7% |

Two independent methods (statistical sampling vs instrumented wall-clock) agree to ~1 pp — high
confidence in the attribution.

### `matmul_i4_grouped` is three things (call-graph attribution)

The single 19.17% symbol splits across three distinct callers — this is the key refinement the
phase view hid:

| Sub-slice | % decode | Caller | `allow_idot` |
|-----------|---------:|--------|:------------:|
| q_a / q_b / kv_a down-projections (+ DSA `ix_wk`) | 10.38% | `attention_rows → matmul_qt_ex` | **0** |
| MoE **shared expert** (always-on `sh_gate`/`sh_up`/`sh_down`) | 7.25% | `moe → matmul_qt → matmul_qt_ex` | 1 |
| dense MLP (layers 0–2) | 1.54% | `dense_mlp → matmul_qt_ex` | 1 |

All three go through the **unmodified AVX2** `matmul_i4_grouped` at S=1. (o_proj + kv_b escaped
this kernel via the landed `AMX_REQUANT` requant.)

### Instruction-level annotate — the two hot kernels

**`matmul_mxfp4_bf16`** (66.7%): 93.75% of samples in one 12-instruction inner loop —
FMA 43.5% (`vdpbf16ps` 27.8% + `vfmadd231ps` scale 15.7%), weight nibble loads 26.2%,
nibble→bf16 unpack (LUT shuffle chain) 22.0%, loop overhead 4.3%. Activation f32→bf16 conv is
0% (done once upfront). **Balanced — no single bottleneck; compute edges out load+unpack.** This
is already good AVX-512-BF16 code; there is no easy win here.

**`matmul_i4_grouped`** (19.2%, AVX2/256-bit): unpack/dequant 31.1%, memory load 21.6%
(`vmovq`, only **8 B/iteration** — narrow), FMA **13.0%**, loop overhead 21.7%, epilogue
horizontal-reduce 6.6%. **Overhead-bound per useful FLOP** — the actual arithmetic is only 13%;
the rest is unpack + narrow loads + a horizontal reduce done *every* 128-element group. This is
the optimization target below.

### Cold-start surprise (not decode, but worth recording)

The load-phase profile is dominated by an entirely different set: `expert_load`'s E8M0→float
scale-table precompute (`scalbnf`/`ldexpf` libm) = **~54% of the 172 s load** (~93 s). It calls
general-purpose libm for what is mathematically "build a power-of-2 float from an 8-bit
exponent" — a bit-manipulation trick could do it far cheaper. It's a one-time cost (the
precompute exists specifically to kill per-call `ldexpf` during decode, and `scalbnf` is indeed
absent from the decode window), correctly amortized for real multi-token serving — but a
legitimate, sizeable cold-start latency finding no decode-only timer would surface.

## `matmul_i4_grouped` optimization plan (researched, not yet implemented)

The 19.2% `matmul_i4_grouped` slice is the highest-value CPU-decode target left after the
requant win (MoE is bandwidth-bound and already optimally vectorized; o_proj is done). Kernel at
`c/glm.c:612–646`, signature
`(float *y, const float *x, const uint8_t *q4, const float *scale, int S, int I, int O, int gs)`.
Current structure: `#pragma omp parallel for` over `O` → `s<S` → groups `g` (gs=128) → AVX2
inner loop with an 8 B `_mm_loadl_epi64`, nibble unpack, 2× `_mm256_fmadd_ps` (16 elem/iter),
and an `hsum256` **every group** then scalar scale-accumulate. It is AVX2 only because it was
never ported — `dot_i4f_avx512`/`axpy_i4f_avx512` (`glm.c:369–395`) already implement the
AVX-512 version of this exact unpack (validated by `i4_acc512_selftest`, `I4_ACC512_TEST=1`).

**All target tensors have `I` an exact multiple of 128** (6144, 2048, 12288) → **no ragged tail
in production.**

Microbenchmark (built in `/tmp`, deleted, `glm.c` untouched; all validated ~1e-7 rel err vs f64
reference, on real model shapes, single-thread):

| Variant | Technique | Speed vs current |
|---------|-----------|-----------------:|
| baseline (today's AVX2) | — | 1.00× |
| naive AVX-512 | reuse `dot_i4f_avx512` per group | 1.44–1.49× |
| **v_def4** | AVX-512 + **deferred per-group reduction** (keep the unreduced partial-sum vector across K-groups, scale-FMA each group into a persistent accumulator, reduce **once** at end-of-row) + **4-row register block** (share activation loads across 4 weight rows, like `matmul_mxfp4_bf16`) | **1.67–1.73×** |
| VNNI/int8 | weight→int8 per-row + activation→int8 + `dot_i8i8` | 2.85–3.21× |

Why v_def4 is the pick: the deferred reduction directly attacks the profile's two biggest
non-arithmetic buckets (loop overhead 21.7% + per-group hsum 6.6%), the wider loads attack the
narrow-8 B-load 21.6%, and it is **fp32-math-unchanged → ~zero quality risk** (pure
reassociation; the same class of change as commit `2aff633` which showed 0 SCORE flips).

### Ranked plan

| Rank | Action | Applies to | Risk | Kernel × | Est. decode gain* |
|-----:|--------|------------|------|---------:|------------------:|
| **1** | AVX-512 port, deferred-reduction + 4-row block (v_def4) | all 3 sub-slices (19.2%) | ~zero (fp32 unchanged) | 1.7× | 19.2·(1−1/1.7) ≈ **7.9 pts → ~+8.6% tok/s** |
| 1a | fallback: naive AVX-512 (call `dot_i4f_avx512` per group) | all 3 | ~zero, less work | 1.45× | ~6.0 pts → ~+6.3% |
| **2** | VNNI/int8 requant, **only if SCORE-gate passes** | shared-expert + dense-MLP (8.8%) — **q/kv stays on #1** | moderate, must be gated | 2.85–3.21× | +5.9 pts more → ~+11.3% combined |
| 3 | fuse shared-expert gate+up into one OMP region | shared-expert | ~zero, bit-identical | — | OMP-launch reduction (invisible at 1 thread) |
| — | AMX grouped kernel / LUT unpack / streaming prefetch | — | — | — | **not applicable** at S=1 |

\* first-order ceiling arithmetic (`saved% = self% · (1 − 1/×)`), single-thread microbench — not
a substitute for the real A/B.

**Why q_a/q_b/kv_a stay off VNNI (#2):** the same measured +0.117 nat/tok / +12% perplexity
precedent (`allow_idot=0` at every call site). Shared-expert + dense-MLP are *always-active*
(no routing sparsity → structurally closer to o_proj, which is proven safe on the weight-requant
axis) but their int8-*activation* risk is genuinely untested — resolvable **only** by the SCORE
gate, not by assumption. AMX grouped kernel is irrelevant here: it only gates at S≥16 and loses
to VNNI at S=1 (perf confirmed AMX is completely idle — 0 `_tile_dpbssd` samples — during N=1
decode).

**Validation harness for the eventual implementation:** (1) bit-exact selftest vs scalar
reference (extend the `i4_acc512_selftest` pattern, or the `amx_idot_selftest.c`
`#define main …`/`#include "glm.c"` trick to reach static functions); (2) SCORE gate (0 flips),
run **twice** — once for #1 fp32-alone (expect noise-level delta) and once for #2's
shared/dense requant specifically (the real unknown); (3) perf A/B via INSTR — but note **no
existing timer cleanly isolates the shared expert from the routed fmt=5 experts** (the
`t_moe_kernel` "fmt=4" comment at ~:3995 is stale), so a clean shared-expert A/B likely needs a
small dedicated timer added, mirroring the existing `t_aproj`/`t_aout` pattern.

---

### REJECTED — MoE MXFP4 unpack/scale codegen from SGLang (`MXFP4_LUT`)

Clean negative; **left at its default (OFF), reverted — nothing landed.** This ported the
unmerged Intel SGLang `intel_dev` MXFP4→bf16 codegen (`cvt_mxfp4_e2m1_bf16_intrinsic_lut`,
`sgl-kernel/csrc/cpu/vec.h`) into `matmul_mxfp4_bf16` / `matmul_mxfp4`, replacing two hot
sequences at once:
- **Unpack:** one `_mm512_permutexvar_epi16` against a 16-entry bf16 LUT, instead of glmrt's
  ~13-op 128-bit SSE shuffle chain (`_mxfp4_to_bf16_32`, the 22% "nibble→bf16" bucket).
- **Scale:** fold the E8M0 group scale into the bf16 weight as an exponent add
  (`_mm512_add_epi16(w, (int16_t)(e8m0−127)<<7)`), so the `_dpbf16_ps` output is already
  scaled — dropping glmrt's per-group `_mm512_set1_ps(scale)` + `vfmadd231ps` (the 15.7%
  scale-FMA bucket) and collapsing to one `reduce_add` per output (same win class as v_def4).

**Compute win is real; wall-clock win is not.** Standalone microbench (single thread, real
expert shapes I=6144→2048 and 2048→6144, gs=32, vs an f64 MXFP4 oracle): **~1.5× GF/s**
(43.3 vs ~29 gate/up; 42.7 vs 27.8 down) at **accuracy parity** (LUT rel-err 1.12e-7 vs
glmrt 1.07e-7 — both at the bf16 noise floor; the two kernels agree to 7.75e-8). Bit-safety of
the exponent-add scaling was verified across E8M0 exponents 120..134, both signs (the E8M0
scale is exactly 2^(b−127), a pure power of two, so the fold is mathematically exact absent
bf16 overflow/denormal).

But in-model the gain does not convert:
- **N=1 (120T):** expert-matmul **14.051 vs 14.052 s** — flat.
- **N=16 (120T):** 13.944 → 13.680 s (−1.9%, +1.2% tok/s) — single-sample, opposite tiny sign
  from N=1, inside run-to-run variance; not a reproducible separation.
- Same bandwidth diagnosis as `PILOT_CACHE_XE`: the routed-expert MXFP4 path is
  DRAM-bandwidth-bound (~42 GB/s, ~5% of peak; N=16 expert-matmul ≈ N=1 because the replay
  reads the same 8 experts/layer regardless of batch), so a compute-side unpack/scale
  reduction has no wall-clock headroom to recover.

**And it carries a small but real quality cost.** SCORE gate (36 probes / 562 tok, 40T
deterministic): **0/36 argmax flips**, but **+0.0104 nat/tok** perplexity. A baseline-vs-baseline
control (LUT=0 run twice) came back at **exactly 0.000000** nat/tok delta and identical total
lp to the last digit — the harness is fully deterministic at this config, so the +0.0104 is a
**genuine** shift, not reassociation noise. The exponent-add scaling is exact for the scale
itself, but the permutexvar bf16 LUT rounds the E2M1 code values slightly differently than
glmrt's shuffle-LUT byte pair, which is the likely source.

**Verdict:** two independent disqualifiers — no reproducible wall-clock benefit (bandwidth
wall) **and** a real +0.0104 nat/tok regression paid for nothing. Reverted; the SGLang
CPU MXFP4 codegen is AOT-compiled C++ templates (no runtime JIT), and its cleverness is
purely compute-side, which this workload cannot cash in. Recorded so the idea is not re-tried
without first attacking the bandwidth bound (weight footprint / prefetch-into-the-matmul /
NUMA placement), not the unpack instruction count.

---

## B70 (Intel Arc Pro, Battlemage) XPU offload — CLEAN NEGATIVE (shelved)

Task 3: research an optional `#ifdef COLI_XPU` GPU-offload backend for the 2× Arc Pro B70
(64 GB VRAM total) in aibox101b, mirroring colibri's `COLI_CUDA` seam. Full write-up:
`docs/glm52-cpu/b70-xpu-offload-plan.md`. Outcome: **SHELVE — no backend justified.**

**Infra proven (kept for any future experiment):** both B70s operational (xe driver, 2×32 GB);
one-line libumf `LD_LIBRARY_PATH` fix enables SYCL/L0 enumeration. Trivial GEMM PASSes on B70 via
BOTH SYCL (icpx, JIT — AOT `-device bmg` hangs) and Level-Zero (SPIR-V via `ocloc`, needs explicit
`zeCommandListAppendBarrier` between H2D→kernel→D2H). **Chose Level-Zero** if ever pursued: 7 vs 16
`ldd` libs, zero oneAPI runtime dep, works without setvars — matches glmrt-lean ethos.

**Why shelved (three converging measurements):**
1. **Routing is near-uniform, model overflows VRAM.** Per-expert MXFP4 = 20.05 MB; only ~2,792
   fit in 56 GB usable of 19,200 total. N=1 STATS replay: 99.4% of experts hit in 80 tokens, mean
   142.9 vs 142.0 uniform. Top 56 GB hottest carry only 52.7% of selections → static hot-set caps
   at ~53% hit; the other ~47% still bottleneck host DRAM (and per-token latency gates on the
   slowest path).
2. **Batching doesn't concentrate access.** Batch-union reuse = 1.27 rows/expert at N=16 (E=256,
   K=8 combinatorics; matches the recorded MoE-AMX figure). Wall-clock cross-check: expert-matmul
   −8.9% only. Weight bandwidth does not amortize with batch.
3. **Ceiling is tiny + glmrt has no overlap pipeline.** The always-hit VRAM-fitting set (shared
   expert ×75 + dense layers 0-2) = 1.87 GB/token = **only 13.4%** of MoE-weight bytes → sequential
   offload ceiling ~9.8% of decode, ≤~10% even with overlap (routed = 86.6% dominates).

**Reconciliation with the production NVIDIA+AMX hybrid** (which DOES boost a VRAM-overflow model):
its win is **disk-elimination (already banked here via RESIDENT=1) + CPU/GPU OVERLAP on a
deliberately-low-active-param model** (DeepSeek-V4-Flash, 13B active) — NOT GPU expert placement
(its own tuning record: `--kt-num-gpu-experts` sweep = "wash"; lesson = "increase overlap, don't
move experts to GPU"). colibri's `COLI_CUDA` path is strictly SEQUENTIAL (every call
`cudaStreamSynchronize`s; pthread overlap was tried on the 6×5090 rig and abandoned), and its
CUDA routed-expert tier is inert for native MXFP4 (`row_bytes()` returns 0 for fmt=5). GLM-5.2
(37B active) is 3× the CPU feed of V4-Flash. So: to capture a ≤10% bandwidth-bound ceiling on B70
we'd first have to build an async overlap pipeline glmrt lacks — not worth it. Same root cause as
MXFP4-LUT / MoE-AMX: perf is gated by memory bandwidth, not by where compute runs.

---

## Task 4 — 2-bit routed-expert format (fmt=6) — ATTACK THE BANDWIDTH WALL (PLANNED)

Rationale: every prior lever (MXFP4-LUT, MoE-AMX, B70 offload) died on the SAME cause — routed-
expert decode reads **12.03 GB/token** from DRAM at ~42 GB/s (86.6% of MoE-weight bytes). Those were
all compute-side or placement-side and had ZERO headroom. **Shrinking the weight FORMAT is the first
lever that converts LINEARLY to tok/s in a bandwidth-bound regime.** A 2-bit routed format ≈ halves
the feed. Model validated: 41.1B active / 743.2B total (reproduces published GLM-5.2).

**Projection (expert-matmul 15.22s of 20.84s N=1 decode scales with feed bytes):**
- MXFP4 4.25b (now): 3.84 tok/s.
- ~2.06b routed: em 8.06s → decode 13.68s → **5.85 tok/s (~1.52×)**.
- ~2.5b routed (if dynamic alloc needs it): decode 14.57s → **5.49 tok/s (~1.43×)**.

**QUALITY GATE — DELIBERATELY RELAXED (user decision, m00130):** The prior HARD rule was
zero-argmax-flips on the 36-probe/562-tok SCORE gate. Uniform 2-bit CANNOT meet that — Unsloth's own
GLM-5.2 UD-IQ2_M reports **82% top-1 accuracy = ~18% argmax divergence vs BF16** (they argue it's
filler/stop-word variation, not wrong answers; mean-KLD ~99.9%). So for Task 4 ONLY, the landing bar
becomes **bounded-KLD/PPL**: mean-KLD ≥ ~99.9% and nat/tok (PPL) delta under a small threshold on the
40T deterministic harness, allowing minor argmax divergence on low-signal tokens. This is a
documented, intentional change to the landing bar for low-bit quant work — NOT a general relaxation.
Precedent contrast: MXFP4-LUT was rejected at +0.0104 nat/tok under the OLD bar; under the NEW bar a
few-× larger PPL delta for a 1.4-1.5× real speedup is acceptable.

**Reference (Unsloth UD-IQ2_M):** NOT uniform 2-bit — it's DYNAMIC: attention, shared expert, router,
embed/lm_head, and first/last blocks stay at 4-8 bit; only the least-important routed-expert tensors
drop to ~2b (net ~245 GB, ~2.6b avg). Dynamic 1-bit = 76.2% top-1, dynamic 2-bit = 82% top-1, 4-bit
UD-Q4_K_XL ≈ lossless. This maps PERFECTLY onto our need: the routed experts (86.6% of bandwidth) are
ALSO the least argmax-sensitive per their layer-importance analysis — quantize those, keep the rest.

**IMPLEMENTATION PLAN:**
- **Scope:** 2-bit ONLY the 256×75 routed experts (fmt 5→6). Keep attention (fmt=1 int8), shared
  expert (fmt=5 MXFP4), router/embed/lm_head as-is. This is the "dynamic" split, done at glmrt's
  tensor granularity via the per-tensor `fmt` field (autodetected from `.qs` sidecar byte size,
  glm.c:2046-2065 qt_from_disk).
- **fmt=6 = next free** (verified: fmts 0-5 in use). Format candidates in preference order:
  1. **IQ2-style: 2-bit codes + small importance codebook + per-group fp8 scale** (~2.3b effective) —
     the proven Unsloth/llama.cpp path; best quality/bit. Dequant = 256-entry codebook lookup
     (port `dequantize_row_iq2_*` from llama.cpp, then AVX-512 `vpermw`/gather version).
  2. Fallback: naive per-group 2-bit + fp8 E8M0 scale (~2.25b) — trivial dequant (like MXFP4) but
     worst quality; use only if IQ2 kernel proves too costly and the PPL gate still passes.
- **Produce the checkpoint — via the EXISTING glmrt converter, NOT llama.cpp** (verified this
  session): glmrt's loader (`qt_from_disk` glm.c:2043; `expert_load` glm.c:2641) reads glmrt's OWN
  flat container — per-expert `...{gate,up,down}_proj.weight` (packed quant bytes) + a sidecar
  (`.qs` = f32 scales for fmt≤4, or `_scale` = E8M0 bytes for fmt=5 MXFP4). **There is NO GGUF
  ingest path**; llama.cpp's IQ2 super-block layout is incompatible, so llama-quantize is the WRONG
  tool. Instead extend `/home/intel/b70-sglang-xpu/conv-glm52-mxfp4.py` (the proven FP8→MXFP4
  producer; source `/srv/models/glm52-fp8`, 704 GB, 141 shards). It already: reads FP8-block source
  (`dequant_fp8_block`, 128×128), dequantizes each expert-proj to f32 (`convert_one_expert_proj`),
  and emits the flat container (`save_file` per `model-layer-NNN.safetensors` + index). Add a
  `quantize_iq2(w_f32)` + `dequantize_iq2` pair mirroring `quantize_mxfp4`/`dequantize_mxfp4`
  (glm.c-side names/round-trip already there for validate mode), plus a `--fmt {mxfp4,iq2}` arg. If
  IQ2 uses an importance codebook, derive per-tensor codebook from the FP8 weights directly (k-means
  on |w|), OR add an optional imatrix from the deterministic replay harness (ref_bench.json) — but
  first try weight-only (no calibration) since Unsloth's static codebooks already do well.
- **fmt=6 loader hook (exact, verified):** detection goes in `expert_load` at glm.c:2687-2689,
  right after the `is_mx` (fmt=5) branch. fmt=5 is detected by presence of the `_scale` sidecar
  (glm.c:2664-2665) → `fmt=5, gs=32`. For fmt=6, add: if the `_scale` sidecar is present AND weight
  bytes `nb == O*(I/4)` (2-bit ⇒ I/4 bytes/row, vs I/2 for MXFP4) → `fmt=6, gs=32`. Store the fp32
  codebook in a third sidecar `..._cb` (tiny, per-tensor). Mirror the fmt=5 E8M0→f32 precompute at
  glm.c:2696-2699. The non-mmap path and `qt_from_disk` (glm.c:2043) need the parallel branch.
- **fmt=6 dequant kernel:** new `matmul_iq2_bf16` cloned from `matmul_mxfp4_bf16` (glm.c:565) and
  `matmul_iq2` from `matmul_mxfp4` (glm.c:520); dispatch at glm.c:1685 (`if(w->fmt==6)`). Unpack
  2-bit code (4/byte) → codebook lookup (16-entry `_mm512_permutexvar_epi16` like the MXFP4 LUT, if
  codebook ≤16 entries) → E8M0 group scale → `_mm512_dpbf16_ps`. Note `MXFP4_SMIN` analog
  (`g_mxfp4_smin`, glm.c:2188) may want an `IQ2_SMIN`.
- **Quality harness (mostly EXISTS):** SCORE `logprob_target()` (glm.c:5379) ALREADY sums
  continuation log-prob + tracks greedy match; SCORE `.out` triple = `<logprob> <contlen> <greedy>`.
  So nat/tok (PPL) delta is available with NO new infra — extend `score_compare.py` to gate on the
  logprob-sum delta (bounded-PPL) instead of requiring zero greedy flips. True token-KLD would need
  full logit-vector dump (glm.c has DEBUG_LOGITS top-5 at :7165) — nat/tok delta is the practical
  proxy and is the primary gate.
- **Validation:** wall-clock N=1 AND N=16 on the tuned REPLAY recipe (hard rule stands for PERF) +
  bounded-KLD/PPL gate. Expect a per-tensor bit-allocation sweep (some experts may need 3-4b to hold
  PPL) — realizable speedup likely 1.3-1.5×.
- **Lean-build hygiene:** fmt=6 dequant is pure CPU C in glm.c (no new deps), guarded like the
  existing AVX-512 paths. No impact on the lean default link.

**Open risk:** zero-flip is off the table by design; the real question is whether ~2.3b routed holds
mean-KLD ≥99.9% / small nat/tok delta, or whether dynamic allocation pushes the average toward ~2.6b
(Unsloth's actual UD-IQ2_M avg) — which softens the win toward ~1.4×. Still the first bandwidth-
cashing lever in this investigation.

**Environment verified this session (ready to start):** source `/srv/models/glm52-fp8` (704 GB, 141
shards) present; 3.8 TB free on `/`; converter `conv-glm52-mxfp4.py` present; two llama.cpp trees
built (`/home/intel/llama.cpp`, `/home/intel/b70-llamacpp/llama.cpp`) — retained only as an IQ2
codebook/algorithm REFERENCE, not in the production path. Loader hooks and converter template fully
mapped (glm.c:565/520/1685/2043/2641/2687; conv-glm52-mxfp4.py quantize_mxfp4/save_file).

**First concrete step:** add `quantize_iq2` to `conv-glm52-mxfp4.py`, run `--mode validate` on a
handful of experts to measure rel-err vs FP8 f32 oracle (mirrors the existing MXFP4 validate path)
BEFORE any full conversion or glm.c work — cheap go/no-go on the format's quality/bit tradeoff.

### Task 4 — STEP 1 RESULT: weight-only (RTN, no calibration) 2-bit is a NO-GO

Built `conv-glm52-iq2.py` (copy of the MXFP4 converter + `quantize_iq2`/`dequantize_iq2` and four
design-probe variants). Ran `--mode validate` on routed experts sampled across layers 3/20/40/60/77.
Reconstruction error vs the FP8 f32 oracle (60-tensor aggregate for the primary run; per-row cosine
is the matmul-relevant metric):

| Format | bits/wt | mean rel-err | mean row-cos |
|---|---|---|---|
| MXFP4 (fmt=5, today)            | ~4.25 | **0.108** | **0.9934** |
| int4 gs128 (fmt=4)              | ~4.5  | 0.130 | ~0.99  |
| gA — per-tensor 4-cb + E8M0     | ~2.25 | 0.376 | 0.933  |
| gB — per-group 4-means + fp16 s | ~2.5  | **0.305** | **0.954** |
| gC — per-group linear 2b + fp16 | ~2.25 | 0.508 | 0.881  |
| gD — per-row 4-cb + amax scale  | ~2.5  | 0.364 | 0.938  |

**Every weight-only 2-bit variant tops out at cos ≈ 0.954 / rel ≈ 0.30 (gB, the best) — ~2.8× worse
rel-err than MXFP4 and clearly worse than int4.** `down_proj` is consistently the worst tensor
(rel 0.36–0.47). Root cause: GLM-5.2 expert weights are near-Gaussian with no structure a cheap
1-D per-group codebook exploits; round-to-nearest 2-bit simply has too few levels. This would fail
the bounded-PPL gate. **Recorded as a clean negative: weight-only RTN 2-bit ≠ Unsloth UD-IQ2.**

**Why Unsloth UD-IQ2 actually works (the missing ingredient):** it is NOT weight-only RTN. Its
quality comes from (1) an **imatrix** — activation-importance weights per input channel, so the
quantizer minimizes `Σ_i importance_i·(w_i − q_i)²` and pushes error onto channels the activations
rarely excite — and (2) a large **shared multi-dimensional codebook** (llama.cpp IQ2_XXS/IQ2_S use
256-entry 8-D lattice grids), not a per-group 1-D 4-level fit.

### Task 4 — DECISION (user, this session): pursue the IMATRIX path

Weight-only is abandoned. Next: build the imatrix pipeline.
- **Capture:** run the deterministic replay harness (`ref_bench.json`, 80-step) with per-input-
  channel activation-importance accumulation into the routed experts (sum of squared — or |x| —
  activations feeding each expert's gate/up/down input dim). Deterministic ⇒ reproducible imatrix.
- **Quantizer:** importance-weighted 2-bit — minimize `Σ_i imp_i·(w_i − q_i)²` per group (weighted
  k-means / weighted nearest-centroid), optionally a shared codebook. Re-run `--mode validate`
  reporting the **importance-WEIGHTED** rel/cos (the metric that actually predicts PPL), compared to
  MXFP4 under the same weighting.
- Only if weighted-validate closes the gap to near-MXFP4 do we proceed to full conversion + the
  glm.c fmt=6 loader/kernel + the wall-clock N=1/N=16 + bounded-PPL gate.

Probe tooling kept: `conv-glm52-iq2.py` (on box + local scratch
`C:\Users\azavjalo\AppData\Local\Temp\opencode\conv-glm52-iq2.py`); report `/tmp/iq2-validate-report.json`.

### Task 4 — STEP 2 RESULT: imatrix (calibration) does NOT rescue uniform ~2.25b either

Built the imatrix pipeline end-to-end and measured it:
- **glm.c instrumentation** (`IMATRIX=<file>` env): added globals + `imat_init`/`imat_accum`/`imat_dump`,
  a per-expert accumulation hook in `moe()`'s CPU loop (right after the gate+up+down compute, covering
  both the fmt=5 and fallback branches at the shared `xg`/`gg` buffers), and `atexit(imat_dump)`.
  Patch script `c/patch_imatrix.py`; backup `/tmp/glm.c.bak.imatrix`. Dump format: header
  `{int32 magic 0x54414D49 'IMAT', L, E, D, I}` + `f64[L*E*D]` gate/up-input Σx² + `f64[L*E*I]`
  down-input Σx² (per input CHANNEL, per (layer,expert)). Built clean (only the pre-existing
  snprintf warnings).
- **Calibration run:** `IMATRIX=/srv/models/glm52-mxfp4/imatrix.bin SCORE=score_probes.txt ... ./glm 256`
  — all 36 probes scored, dumped `imatrix.bin` = 1,308,622,868 B (matches 20 + 78·256·(6144+2048)·8).
  Retained at `/srv/models/glm52-mxfp4/imatrix.bin`.
- **Importance-weighted quantizer** (`quantize_iq2_imat` in conv-glm52-iq2.py): per-group E8M0 scale +
  4-level signed codebook fit by **importance-weighted** 1-D k-means (error `Σ imp_col·(w−q)²`), 2-bit
  codes. Validate now reports the **weighted** rel-err `Σ imp·|Δw| / Σ imp·|w|` — the PPL-predictive
  metric — for IQ2 vs MXFP4.

**Result (N=60 routed-expert tensors, layers 3/20/40/60/77):**

| Metric | MXFP4 (~4.25b) | IMAT-IQ2 (~2.25b) | ratio |
|---|---|---|---|
| WEIGHTED mean rel-err | 0.1068 | 0.3667 | **3.43×** |
| unweighted mean rel-err | 0.108 | 0.386 | 3.5× |
| mean row-cosine | 0.9934 | 0.931 | — |

**Imatrix weighting bought essentially nothing: weighted ≈ unweighted per-tensor** (e.g. L20 E192
gate wrel 0.3725 vs urel 0.3726), and the ratio is flat across gate/up/down (all 3.43×). Root cause:
GLM-5.2's per-input-channel activation importance is **too flat to relocate quantization error** —
there are no low-importance channels to dump error into. This is exactly why Unsloth's UD-IQ2 works
on *some* models (skewed importance + never-activated channels) but not here. The binding constraint
is representational: **true 2-bit = 4 scalar levels cannot fit near-Gaussian expert weights near
4-bit fidelity**, and calibration cannot manufacture skew that isn't in the activations.

**VERDICT: uniform ~2.25b routed-expert quant is a NO-GO — twice-disconfirmed (weight-only RTN AND
imatrix-weighted).** The 2-bit hypothesis as a *uniform routed-expert format* is closed.

**What remains genuinely untested (did NOT probe):**
1. **True vector/lattice codebook (llama.cpp IQ2_XXS/IQ2_S-style, ~2.06–2.31b):** 8 weights share a
   256-entry signed 8-D grid — encodes *shape correlations* a 1-D 4-level scalar codebook cannot. This
   is what actually ships at ~2.3b in llama.cpp. Substantial kernel effort (grid-lookup dequant), and
   real risk it still trails 4-bit given the 3.4× scalar gap — but it is the only 2-bit variant with a
   fundamentally different representational basis.
2. **Mixed ~3-bit allocation:** keep down_proj + most-sensitive experts at 3–4b, drop only the
   well-behaved gate/up of low-importance experts to ~2b → ~2.6–3.0b avg, softer **~1.2–1.3×** win,
   far likelier to pass bounded-PPL. Aligns with Unsloth's *actual* UD-IQ2_M average (~2.6b) rather
   than a true 2b.

Artifacts kept: `conv-glm52-iq2.py` (+ imatrix loader/weighted quantizer), `c/patch_imatrix.py`,
`imatrix.bin` (1.31 GB), `iqw-report.json`. glm.c IMATRIX instrumentation is LANDED-IN-TREE (useful
for any future calibrated-quant work) — decide whether to keep or revert it with the path choice.

### Task 4 — STEP 3 RESULT: 3-bit IS viable — and uniform 3b beats a mixed scheme

Probed 3-bit variants (imatrix-weighted rel-err, the PPL-predictive metric; N=60 routed tensors,
layers 3/20/40/60/77 × 4 experts):

| Format | bits/wt | WEIGHTED rel-err | cos | ratio vs MXFP4 |
|---|---|---|---|---|
| MXFP4 (fmt=5)                          | ~4.25 | 0.107 | 0.993 | 1.00× |
| **h3e — per-TENSOR 8-lvl k-means cb + E8M0 gs32** | **~3.06** | **0.196** | **0.981** | **1.83×** |
| IMAT-IQ2 (2-bit)                       | ~2.25 | 0.367 | 0.931 | 3.43× |

**3-bit clears the bar the way 2-bit couldn't, and it's UNIFORM across gate/up/down (all 1.83×, no
outlier tensor)** — unlike 2-bit where down_proj blew up. cos 0.981 vs MXFP4's 0.993. 1.83× weighted
rel-err at 0.72× the bits is a far better quality/bit slope than 2-bit's 3.43× at 0.53×.

Two design lessons from the probe:
- **A FIXED 3-bit grid fails** (`{-6,-3,-1.5,-.5,.5,1.5,3,6}/6` gave wrel 0.62 — worse than 2-bit).
  The 8 levels MUST be **data-fit by k-means**, not hardcoded.
- **Per-TENSOR codebook = per-ROW** (0.1955 vs ~0.192) at 1/30th the cost → the kernel needs only an
  **8-entry per-tensor LUT** + E8M0 group scale. A cheap clone of the MXFP4 LUT kernel.

**PIVOT — go UNIFORM 3b, not mixed.** Since 3b quality is uniform (no sensitive-tensor outliers), a
mixed 3b/4b scheme only *dilutes* the speedup. Wall-clock projection (expert-matmul 15.22s→scales
with feed bytes):

| Scheme | avg bits | decode | tok/s | speedup |
|---|---|---|---|---|
| all MXFP4 (now)        | 4.25 | 20.8s | 12.3 | 1.00× |
| **all 3b (fmt=6)**     | 3.06 | 16.6s | 15.4 | **1.26×** |
| 75% 3b + 25% MXFP4     | 3.36 | 17.6s | 14.5 | 1.18× |
| 50/50                  | 3.66 | 18.7s | 13.7 | 1.11× |

So: convert ALL 256×75 routed experts to fmt=6 (3b). Keep the mixed option ONLY as a fallback — if
uniform-3b fails bounded-PPL, selectively bump the worst experts back to MXFP4 (fmt=5), which glmrt
supports for free at per-tensor granularity.

**fmt=6 FORMAT (decided):** per-expert-proj emits THREE sidecars:
`...weight` = 3-bit codes packed (8 codes / 3 bytes, or simpler 1 code/byte-with-waste? NO — pack
tight: I codes × 3b = 3I/8 bytes/row), `..._scale` = E8M0 uint8 [O, I/32] (reuse fmt=5's sidecar
name + loader), `..._cb` = f32[8] per-tensor codebook (tiny). Kernel `matmul_iq3`/`matmul_iq3_bf16`
= clone of `matmul_mxfp4`/`_bf16` (glm.c:520/565) with the fixed E2M1 LUT replaced by the 8-entry
per-tensor `_cb` loaded into the `_mm512_permutexvar` table; unpack 3-bit codes instead of 4-bit
nibbles; same E8M0 group-scale fold. Dispatch `if(w->fmt==6)` at glm.c:1685. Loader hook at
glm.c:2687-2689: `_scale` present AND `nb == O*ceil(3*I/8)` → fmt=6.

**NEXT:** implement fmt=6 (converter emit + glm.c loader + kernel), convert routed experts, gate on
wall-clock N=1 AND N=16 + bounded-PPL (extend score_compare.py to logprob-sum delta). This is the
first lever in the whole investigation that both attacks the bandwidth wall AND has a viable
quality basis.

---

## Task-4 RESULT (2026-07-20): IQ3 fmt=6 MEASURED — NO-GO for S=1 decode

The projection above (1.26× from feed-byte scaling) was WRONG. It assumed decode scales
with weight-feed bytes — but the clean fifo-fenced perf proof (this session) established
decode is **memory-LATENCY / low-MLP bound, NOT DRAM-bandwidth-bound** (only 0.72% of loads
reach DRAM; ~12× BW headroom). Shrinking feed bytes therefore cannot help, and ADDING
decode-path compute hurts.

**Wall-clock A/B (N=1, canonical tuned recipe: 128T, NUMA_PARTITION=1, numactl --interleave=all,
REPLAY, 80 decode tokens):**

| model | decode | tok/s | expert-matmul |
|---|---|---|---|
| glm52-mxfp4 (baseline) | 19.931s | **4.01** | 14.470s |
| glm52-iq3 (fmt=6)      | 29.727s | **2.69** | 23.927s |
| delta                  | +9.80s  | **1.49× SLOWER** | **+9.46s (+65%)** |

Entire regression is in expert-matmul (+65%). attention (3.56→3.73s), lm_head (0.12s),
other (1.78→1.95s) unchanged — as expected, only routed experts became fmt=6.

**Quality gate (36 probes / 562 tok, TEMP=0, single-die 40T, ABSORB=1, SCORE):**
- mxfp4 nat/tok 2.787485 → iq3 nat/tok 2.845367 = **+0.057882 nat/tok (~6% PPL, exp(0.0579)=1.060)**
- **ARGMAX/GREEDY FLIPS: 0/36.** Quality is fine. Quality was never the problem.

**Root cause:** `matmul_iq3` must, per weight, unpack a 3-bit code + do an 8-entry per-tensor
LUT lookup + fold the E8M0 group scale — extra work injected onto the already-serialized
dependent-load chain that IS the decode bottleneck. Cutting a resource with 12× surplus
(DRAM bytes) at the cost of the resource we're bound on (near-cache load latency / ILP) is a
net loss. **Generalizes: quantizing the weight feed cannot speed S=1 decode on this box.**

**N=16 note:** current glm.c REPLAY is hardcoded S=1 (`step(m,full+i,1,i)`); `BATCH` env is
dead (not read). True batched decode only via `run_serve_mux`/`openai_server.py`. N=16 would
not rescue IQ3 anyway — at S≥8/16 the mxfp4 path engages AMX tiles, so a scalar 3-bit-unpack
kernel is even more disadvantaged. Only reopens if a batched AMX fmt=6 kernel is written AND
MTP raises S.

**Disposition:** IQ3 → settled-negatives in AGENTS.md. fmt=6 kernel/loader/converter KEPT in
tree as verified, env-gated, opt-in (may be reused if a batched AMX variant is ever built).
Model `/srv/models/glm52-iq3` retained for reference. **Quant thrust (2-bit + 3-bit) fully
closed.** Next lever: MTP / speculative decode (raise S).
