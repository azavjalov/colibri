# GLM-5.2 CPU runtime — single-stream decode data/execution flow

This documents the CPU-only decode path in `c/glm.c` for the GLM-5.2 (744B MoE,
MXFP4) runtime, as measured on a Granite Rapids Xeon 6980P (single socket, SNC-3,
one NUMA node bound: 40 physical cores / `OMP_NUM_THREADS=40`).

The `COLI_CUDA` and `COLI_METAL` blocks in the source are compiled out in this
build; the trace below is the pure CPU path.

## Model shape

- hidden = 6144, vocab = 154880
- 78 layers; `first_k_dense_replace = 3` ⇒ layers 0–2 dense FFN, layers 3–77 sparse MoE
- MoE: 256 experts, top-8, moe_intermediate = 2048, 1 shared expert
- MLA attention: n_heads = 64, kv_lora = 512, qk_nope = 192, qk_rope = 64, v_head = 256
- Experts stored MXFP4 (`fmt=5`); MLA `kv_b` weight `[28672,512]` is grouped-int4 (`fmt=4`, gs=128, ng=4)

## Top-level: one decode token

`step_all(m, ids, S=1, pos)` — glm.c:4691

```
step_all(m, ids, S=1, pos)                                              [:4691]
 │
 ├─ embed_row(m, token, x)     token id → x[6144] (dequant embed row)   [:4694 / :2297]
 │
 ├─ layers_forward(m, x, S=1, pos)   loop i = 0 .. 77 (78 layers)       [:4623 → :4555]
 │    │   per layer: layer_forward_rows()                               [:4475]
 │    ├─(A) rmsnorm(nrm, x, l->in_ln)          pre-attention norm       [:4536]
 │    ├─(B) attention_rows(m,l,i, nrm,S,pos, tmp)   MLA attention → tmp [:4537 / :3091]
 │    ├─(C) x += tmp                           residual add             [:4538]
 │    ├─(D) rmsnorm(nrm, x, l->post_ln)        pre-MoE norm             [:4549]
 │    ├─(E) if sparse: moe(m,l,i, nrm,S, tmp, 1)   MoE (layers 3..77)   [:4549 / :3421]
 │    │     else:      dense_mlp(...)          dense FFN (layers 0..2)
 │    └─(F) x += tmp                           residual add             [:4550]
 │
 ├─ rmsnorm(row, x, m->final_norm)             final norm (last pos)    [:4699]
 └─ matmul_qt(lo, row, m->lm_head, 1)          → logits[vocab=154880]   [:4700]
```

## (B) Attention — MLA absorb path (S=1 decode)

Per layer, under one `#pragma omp parallel for collapse(2)` over (s, h):

1. rmsnorm the latent; project **q** (`q_a` → `q_a_ln` → `q_b`); project/norm **kv**
   latent (`kv_a` → `kv_a_ln`, kv_lora=512).  [glm.c:3160, :3177]
2. **qabs / W_K** — reconstruct per-head keys from the `kv_b` weight (fmt=4 grouped-int4)
   via `qt_addrow`.
3. **score + RoPE** — q·k dot over kvl=512 + qk_rope=64, per (s, h).
4. **softmax** over KV positions.
5. **value / W_V** — weighted sum through `kv_b` via `qt_matvec_rows`.
6. **o-proj** — project the 256-dim per-head output back to hidden=6144.

Steps 2 and 5 are the two fmt=4 grouped-int4 inner loops. They were originally
scalar and dominated the batched-decode profile (94% of attention thread-seconds);
they are now AVX-512 vectorized (commit `2aff633`), reusing `dot_i4f_avx512` and a
new `axpy_i4f_avx512` dequant-AXPY helper.

## (E) MoE — routed experts (sparse layers)

- Router matmul → top-8 of 256 experts; resolve residency. With `RESIDENT=1` all 256
  experts are pre-loaded ⇒ 100% hit, no disk/streaming.  [glm.c:3468]
- Each chosen expert: gate / up / down matmul on the MXFP4 (`fmt=5`) weights via the
  AMX bf16 kernel; plus one shared expert. Accumulate → `tmp`.
- Largest single wall-clock cost at S=1.

## Per-token cost profile (S=1, after vectorization, 3.02 tok/s)

```
expert-matmul  17.4s   ← dominant (MoE weights, AMX)
attention       5.8s   ← was 8.8s before vectorization
  ├ proj/RoPE   2.4s
  ├ score-s-v   0.9s   ← was 4.0s (vectorized fmt=4)
  └ o-proj      2.5s   ← now the attention hot spot
other/norms     2.8s
lm_head        0.35s
```

## Parallelism / synchronization

- Every matmul/proj primitive opens its own OMP parallel region: ~46 fork/join
  barriers per layer, 288,000 per 80-token run (independent of batch size).
- Measured fork-join + load-imbalance stall in the dominant attention region:
  **0.2%** of wall (97% thread utilization) — the region is **compute-bound**, not
  sync-bound, so region-fusion is not worth pursuing (< 3% ceiling).
- Multi-stream (N=16) decode saturates the 40 bound cores: process 3995% CPU
  (≈ 40 cores × 100%), all 40 OMP threads in R state.
- DRAM bandwidth during decode ≈ 43–65 GB/s vs a ~844 GB/s socket ceiling (~5–8%):
  **not memory-bandwidth-bound** at any batch size.

## Throughput (single node, 40 cores, MXFP4, RESIDENT=1)

| Config | tok/s aggregate | tok/s per stream |
|---|---|---|
| Single-stream (N=1) | 3.02 | 3.02 |
| Batched (N=16)      | 8.85 | 0.55 |

The numbers above are the single-node baseline. Spreading across all three SNC nodes
(next section) roughly doubles N=16 throughput.

## Core allocation — use the whole socket (full-pool span)

The baseline recipe above pins the runtime to one SNC node (`--cpunodebind=0`,
`OMP_NUM_THREADS=40`), leaving ~90 of the box's 129 usable physical cores idle. The
earlier bandwidth measurement showed decode used only ~5–8% of the ~844 GB/s socket
ceiling at 40 cores, i.e. large headroom. Spreading the resident experts across all
three SNC nodes (`NUMA_PARTITION=1`, `expert_node = eid % 3`) and running **one OMP
pool across all three nodes' cores** (`numactl --interleave=all`,
`OMP_NUM_THREADS=120`) consumes that headroom:

| Config | cores / threads | N=1 tok/s | N=16 tok/s (agg) |
|---|---|---|---|
| Baseline (1 node) | node0, 40T, `--cpunodebind=0` | 3.17 | 8.98 |
| **Full-pool span** | all 3 nodes, 120T, `--interleave=all`, `NUMA_COMPUTE=0` | **4.16 (+31%)** | **16.47 (+84%)** |
| NUMA 3-pass | all nodes, `NUMA_COMPUTE=1` | 3.35 | pathological (killed) |

Tuned recipe (full-pool span — the recommended default):

```
NUMA_PARTITION=1 NUMA_COMPUTE=0 OMP_NUM_THREADS=120 \
  OMP_PROC_BIND=close OMP_PLACES=cores \
  numactl --interleave=all ./glm 256
```

This is a **runtime/env change, not a code change**.

### Thread-count tuning

- **Sweet spot ≈ 108–120 physical threads**, and the curve is flat across it:
  N=16 gave 15.69 (102T), 16.57 (108T), 16.47 (120T), 15.97 (126T) tok/s — 108 and
  120 are within run-to-run noise (~0.6%). Default kept at 120.
- **Do not exceed ~120**: using all 129 physical cores starves the OS-reserved cores
  and regresses (N=1 3.54 at 129T vs ~4.1 at ~108T).
- **Do not use hyperthreading**: `OMP_PLACES=threads` with 240 logical threads gave
  N=1 2.88 tok/s (−30%). Stick to physical cores (`OMP_PLACES=cores`).
- N=1 is latency-bound and noisy (±0.3 tok/s), so the thread sweet spot is best read
  from the throughput-bound N=16 sweep.

## Remaining levers

- **o-proj** — the largest attention sub-phase after the fmt=4 vectorization.
- **expert-matmul** — already AMX; hard to beat.
- **NUMA_COMPUTE 3-pass** — confirmed a dead end: it runs each node's cores
  sequentially (to keep the shared `out[]` accumulation race-free) plus a per-pass
  affinity-rebind tax, so it cannot exploit multiple nodes concurrently and is far
  slower than the full-pool span. A genuinely concurrent multi-node design would need
  per-node partial output buffers + a final reduce; not pursued.
