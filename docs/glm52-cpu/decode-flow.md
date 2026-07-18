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

## Throughput (this recipe: single node, 40 cores, MXFP4, RESIDENT=1)

| Config | tok/s aggregate | tok/s per stream |
|---|---|---|
| Single-stream (N=1) | 3.02 | 3.02 |
| Batched (N=16)      | 8.85 | 0.55 |

## Remaining levers

- **o-proj** — now the largest attention sub-phase.
- **expert-matmul** — already AMX; hard to beat.
- **More cores** — the box has 128 physical / 256 logical cores across 3 SNC nodes,
  but the current NUMA compute design runs node passes sequentially (one node's ~40
  cores active at a time), so it cannot exploit the other nodes concurrently. A
  genuinely concurrent design would need per-node partial output buffers + a final
  reduce.
