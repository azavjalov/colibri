# colibri × AMX — Investigation Checkpoint

**Host:** aibox101b = intel@<LAB-HOST> (pw <REDACTED-PW>, shared lab cred — never leaks off-box).
Single-socket Intel Granite Rapids Xeon **6980P**, 128 physical cores / 256 threads, **SNC=3**
(NUMA node0 = cores 0-42, node1 = 43-85, node2 = 86-127; +HT siblings 128-255), 751 GiB RAM,
5.9 TB free on `/`. AMX (amx_bf16 / amx_tile / amx_int8), AVX-512 + AVX-512-VNNI + AVX-VNNI all present.
gcc 15.2, oneAPI 2025.3 + 2026.1 side-by-side. 2× Intel Arc B70 (Battlemage) also in box (unused here).

**Access rules:** ssh_* tools only (Read/Edit/Glob/Grep are LOCAL-only, remote inspected via `ssh_exec`
sed/grep and edited via ssh_upload / python string-replace). ssh_exec runs under dash, channel dies ~30s
(launch long ops detached `setsid nohup … </dev/null &`, poll one short cmd/turn, no bashisms).
Web via web-cache MCP only.

**Work dir on box:** `/home/intel/b70-sglang-xpu/colibri/c/` (colibri repo clone, HEAD d4b4f33).
`PY=/home/intel/b70-sglang-xpu/.venv/bin/python`.

---

## 1. What colibri is (and why it's the right substrate)

`github.com/JustVugg/colibri` — a **pure-C, zero-dependency, single-file** inference engine (`glm.c`, ~6000 lines)
hard-specialized for **one model: GLM-5.2 (744B MoE)**. Its premise: run a frontier MoE on RAM-starved
consumer machines by **streaming experts from NVMe** (~11 GB random reads/token), keeping only ~10-26 GB resident.
Kernels: int8/int4/int2, **AVX2 + AVX-VNNI + AVX-512-VNNI** (compiled by `-march=native`; the README's
"AVX2-only" is outdated — on GNR it builds the `idot: avx512-vnni` path). GPU tier is **CUDA-only + experimental
Metal**; **no SYCL/Level-Zero/Intel-GPU path**. GLM-5.2 arch = DeepSeek-V3-style sigmoid router + MLA attention
+ DSA sparse attention + native MTP speculative head — token-exact-validated against HF `transformers`.

**User's thesis (the whole point of this work):** *"We don't need heavy frameworks like SGLang if we know the
specific model. A lightweight runner with a pre-built, optimized harness is enough — and porting/writing an AMX
kernel is faster than optimizing SGLang/vLLM because of their generalized, complex architecture."*

Colibri is the ideal vehicle to test this because it's a clean, readable engine with a **token-exact oracle**,
and its x86 batch GEMM path was **completely unoptimized** (see §3).

---

## 2. Kernel A/B — does AMX beat colibri's existing VNNI? (standalone micro-bench)

Built `amxbench.c` on the box: A/B of `C[S][O]=A[S][I]·Bᵀ[O][I]`, int8→int32 (per-row scales applied
outside), validated **bit-exact vs a scalar reference** (`match=ok`). Kernels: `gemm_scalar` (oracle),
`gemm_vnni` (= colibri's `dot_i8i8` AVX-512-VNNI row loop, the baseline), `gemm_amx` (single 16×16 C tile).

**Fair (O-parallel, 43 threads = 1 SNC node), I=4096 O=2048 (DeepSeek-V4 MoE shape):**

| S (batch) | VNNI GF/s | AMX GF/s | AMX/VNNI |
|---|---|---|---|
| 1 (decode)      | 1230 | 1062 | **0.86× (AMX slightly slower)** |
| 8               | 3338 | 7110 | 2.1× |
| 32              | 3866 | 24036 | 6.2× |
| 128 (prefill/MTP) | 4013 | 39056 | 9.7× |

**Boundary (physics, not tuning):** AMX **loses at S=1** — single-stream decode is a GEMV, one activation row
can't fill a 16-row tile and it's memory-bandwidth-bound (VNNI already ~BW ceiling). AMX **wins big at batch**
(prefill, MTP speculative verification, batch-union MoE). On-the-fly B-packing kills the win at small S
(0.29× at S=1, break-even S=8) → **pre-pack resident weights once**; only worth packing streaming/transient
weights at S≥32.

---

## 3. The x86 hole in colibri (why there was room at all)

`glm.c` has a **tiled batch GEMM only for ARM**: `matmul_q_idot_mm` / `matmul_i4_idot_mm` gated behind
`#if defined(__ARM_NEON) && defined(__ARM_FEATURE_MATMUL_INT8)` (uses ARM `vmmlaq_s32` i8mm 2×2 tile op).
On **x86 the batched (S≥2) path falls through to the per-row VNNI dot loop** — no tiling, AMX unused.
The ARM tiled kernel is the structural reference; we filled the x86 hole with AMX.

**Layout facts (established, not re-derived):** `QT` struct (glm.c:105) `{fmt; qf; q8; uint8_t*q4; float*s; O,I,gs; …}`.
fmt=1 INT8 (q8 row-major [O][I]), fmt=2 INT4 packed (2/byte), scale `s` **per-row** (per output channel),
gs=0 (grouped quant is a separate fmt=4 that bypasses idot — untouched). Activations quantized per-row to int8
by `qrow_i8` → `xq`[S][I] + per-row `sx[s]`. Output `y[s*O+o]=dot(q8[o],xq[s])*s[o]*sx[s]`. `qalloc` (glm.c:1162)
= resident-tensor allocator; `qt_load` sets resident tensors (dense MLA proj, dense MLP, shared expert);
streaming experts use reused slots (`ESlot`, `qt_from_disk`, not resident).

---

## 4. Integrated AMX into glm.c (Phase 1 — int8 + int4)

Delegated to Sonnet 5 (`general`). Backups on box: `glm.c.bak-amx`, `glm.c.bak-2x2port`.
- Added `amx_enable()` (arch_prctl XTILEDATA) once at startup; global `g_amx` (+ env `AMX=0/1`, `AMX_SMIN`,
  default threshold 16); banner shows `amx: on/off`. All AMX guarded `#if defined(__AMX_INT8__) && defined(__linux__)`
  so the portable `ARCH=x86-64-v3` build compiles it out cleanly.
- `QT` gained `int8_t *amx_q8; int amx_packed, amx_eligible;`. **`amx_eligible` (not `cuda_eligible`)** — the
  latter is only set inside `#ifdef COLI_CUDA` which the CPU build never compiles, so it'd be permanently 0.
  `amx_eligible=1` set unconditionally in `qt_load` (resident tensors only; streaming slots bypass it).
- `pack_amx`/`pack_amx_full`/`amx_prepack_q8`/`amx_prepack_i4` (lazy pre-pack of resident B into AMX VNNI layout,
  int4 unpacked→int8 once). Dispatch in `matmul_qt_ex`: if `g_amx && amx_eligible && S>=g_amx_smin` → AMX on the
  full-tile part + VNNI tail for the ragged <16 rows; **S=1 always stays VNNI**.
- Validated: `amx_idot_selftest.c` (#includes glm.c) — **126 checks all bit-exact** vs scalar/VNNI, int8+int4,
  S∈{1,8,15,16,17,32,37,64,100,128} incl ragged S/O; `make test-c` idot exactness ok; native+portable builds clean.

---

## 5. Borrowing Intel's optimization logic (not code)

**Key reframing discovery:** Intel's own DeepSeek-V4 CPU kernels (blog: "Run DeepSeek V4 on Intel CPUs and GPUs",
SGLang PR #24976) — source on box at `/home/intel/sglang-src/sgl-kernel/csrc/cpu/` (`gemm_int8.cpp`, `moe_*.cpp`, …) —
**do NOT hand-write AMX tile ops at all.** Every hot path bottoms out in `at::native::cpublas::brgemm(...)`
= PyTorch/oneDNN's **JIT'd** AMX kernel (`gemm_int8.cpp:229`). Zero `_tile_*` intrinsics in the tree. So the
"engineer-hours" split: (a) the actual AMX tile microkernel lives in **oneDNN** (a shipped lib you call, not
re-derive); (b) the readable, borrowable IP is the **wrapper logic** (packing, blocking, threading, MoE fusion).

Design digest mined from Intel's source (`general` subagent):
- **block_size_m = block_size_n = 2×TILE = 32** → **4 live C accumulator tiles** per macro-block (the #1 thing our
  single-tile kernel was missing).
- L2-resident weight-panel loop order (GEBP); `BLOCK_K=128`; per-dtype `can_use_brgemm<T>(M)` threshold ("brgemm"
  = batch-reduce GEMM, oneDNN pattern); `convert_weight_packed` VNNI pre-pack; U8×S8 + per-channel compensation
  (because VNNI `dpbusd` is u8×s8; AMX `tdpbssd` is s8×s8 native → no compensation needed there).
- FP8: dequant FP8→bf16 then **bf16 AMX brgemm** (not FP8→int8); block scales folded into the unpack.
- MoE: token bucket-sort into 32-row blocks (`moe_align_block_size`), one brgemm per (block × 32-col), gate+up
  fused (silu/mul between the two GEMMs, no intermediate materialized), down-proj reads the fused output directly.
- NUMA handled once at init (`numa_migrate_pages` + `numa_set_membind` + strict), not per-call; per-thread affinity.

**On disassembly (user asked):** oneDNN's brgemm is **JIT** (Xbyak), no static symbol to `objdump`. You CAN capture
it with `ONEDNN_JIT_DUMP=1` (dumps generated kernels to `.bin`, then `objdump -D -b binary -m i386:x86-64`) — did
this, see §7 — but the *logic* is cleaner from oneDNN's open-source JIT generator + Intel's C++ wrapper than from
raw AMX asm.

---

## 6. Applied the borrowed logic — 2×2 tile register blocking (Path A)

Delegated to Sonnet 5. New `gemm_amx_2x2` in `amxbench.c`, then ported into `glm.c`'s `matmul_q_idot_mm_amx`
(replacing the single-tile body; dispatch/scaling unchanged). Design: **all 8 AMX tile regs** — tmm0-3 = C00,C01,C10,C11;
tmm4,5 = A0,A1 (rows 0-15/16-31); tmm6,7 = B0,B1 (cols 0-15/16-31); 32×32 macro-block; **4 independent
`_tile_dpbssd` per 4 tile-loads** (hides TMUL latency). Ragged S/O fall back to single-tile. Tile-config hoisted
once per thread per parallel region.

**Result (43 threads = 1 SNC node, I=4096 O=2048):**

| S | VNNI GF/s | single-tile AMX | **2×2 AMX** | 2×2/VNNI | 2×2/single-tile |
|---|---|---|---|---|---|
| 32  | 3875 | 24102 | **29668** | 7.7× | +23% |
| 128 | 4028 | 39422 | **57131** | **14.2×** | **+45%** |

**Peak ~57 TFLOP/s int8 = ~66-70% of theoretical AMX peak** (43 cores × ~2 TOPS). Ported into glm.c: **126 checks
bit-exact** (int8+int4, ragged), both builds clean, +39% over single-tile in-engine.

**Rejected (tested honestly, reverted):**
- **OPT2 L2 GEBP loop reorder: −30%** — the B weight panel already fits GNR's per-core L2 at these sizes;
  coarsening the parallel grain (nSb·nOb → nOb) just cost parallelism.
- **OPT3 software prefetch: noise** (flipped sign between reruns) — matches Intel's own evidence (they ship
  `PREFETCH_SIZE_K=0` in the tile-multiply path, enable it only in fp8-unpack streaming loops).

**NUMA ceiling (important, structural):** the kernel does **NOT scale past one SNC domain** — 43T@57 TFLOP/s >
128T@42 TFLOP/s, because the bench allocs A/B from one thread (cross-domain latency). This is a **data-placement**
problem, not a kernel problem — and it's exactly why colibri's (and the golden SGLang config's) **one-rank-per-NUMA-node
+ local weights** deployment shape is correct.

---

## 7. The ceiling check — us vs oneDNN (Path B, the definitive result)

Linked **the exact function SGLang calls**: `at::native::cpublas::brgemm` (int8 s8s8→i32 overload) from the
torch-bundled oneDNN inside `libtorch_cpu.so` (no standalone libdnnl; raw `dnnl_brgemm_create` symbols not exported →
used the ATen wrapper, same entry point Intel uses). B pre-packed via ATen `cpublas::pack()`. New kernel `gemm_onednn`
in `onednn_probe.cpp` (C++), linked into `amxbench.c` under `#ifdef USE_ONEDNN` (plain build unaffected).
Link needed `-L$VENVLIB -Wl,-rpath,$VENVLIB` for Intel compiler-runtime deps (libsvml/libirng/libimf/libintlc).

**Our 2×2 vs oneDNN brgemm (all `match=ok`, bit-exact int32):**

| Shape | Thr | S | VNNI | 2×2 AMX | oneDNN | **2×2 / oneDNN** |
|---|---|---|---|---|---|---|
| 4096×2048 | 43 | 32  | 3847 | 26795 | 24622 | **109%** |
| 4096×2048 | 43 | 128 | 4033 | **57004** | 53718 | **106%** |
| 4096×2048 | 43 | 1   | 1384 | 1030  | 1270  | 81% |
| 4096×2048 | 43 | 8   | 3349 | 8902  | 9794  | 91% |
| 5120×1536 | 43 | 32  | 3681 | 27747 | 23059 | **120%** |
| 5120×1536 | 43 | 128 | 3855 | 56111 | 56062 | **100%** |

**Our hand-written ~40-line kernel MATCHES/slightly BEATS Intel's production oneDNN at the batch sizes that matter
(S≥32: 100-120%).** oneDNN wins only at small S (81-96%).

**JIT-dump smoking gun** (`ONEDNN_JIT_DUMP=1` → disassembled `dnnl_dump_cpu_jit_brgemm_amx_uker_base_t.*.bin`):
oneDNN's M=32 kernel is **instruction-for-instruction our design** — `tilezero tmm0-3`; K-loop of 4 tile-loads
(2 A + 2 B, with `tileloaddt1` streaming hint on B) feeding **4 independent `tdpbssd`**; `tilestored`. Confirms the
2×2/4-accumulator schedule. At M=1 oneDNN uses a **1×2** fallback (2 tdpbssd) while ours drops to **1×1** — this
exactly explains the small-S gap (→ cheap future polish: give colibri a 1×2 ragged-M fallback).

---

## 8. Thesis verdict

**Validated, with a precise boundary.**
- **Effort:** ~40 lines of C, informed by their design, reaches production parity — vs fighting SGLang's abstraction
  layers (the ccs-fault saga on the GPU side is the cautionary counter-example). Colibri is genuinely the better
  substrate to *write* a kernel for.
- **"We couldn't beat Intel's kernels":** you *can match* them at batch — the moat is **not** the tile math
  (~250 lines of well-known technique both implement identically, JIT-confirmed), it's oneDNN's **breadth**
  (every shape/dtype/small-M edge tuned) + framework **orchestration** (NUMA, threading, MoE fusion).
- **The AMX win regime:** prefill / MTP speculative verification / batched throughput (S≥32). **NOT** single-stream
  cold decode — that's disk-streaming-bound (~11 GB/token NVMe) for the real 744B model, so an end-to-end decode
  tok/s number measures the SSD, not the kernel.
- **colibri now has a production-grade x86 AMX int8/int4 GEMM** where it had zero x86 tiling → a real, upstreamable
  contribution.

---

## 9. End-to-end GLM-5.2 on CPU — measured (Option 1 baseline)

**Model ready:** `/srv/models/glm52-i4` — real GLM-5.2 (glm_moe_dsa, 78 layers, 256 experts, top-8, hidden 6144),
downloaded FP8 (704 GB, 141 shards, `/srv/models/glm52-fp8`) → converted to int4 (`conv-glm52.sh`, rc=0). On disk
**694 GB / 744.4 GB logical**. Tokenizer fetched from HF (`tokenizer.json` etc). **MTP head NOT produced** — the
converter's `--mtp` branch uses `a.repo` (HF download) not `--indir`, so with our local-dir convert it silently
no-op'd (rc=0). MTP is optional (speculative-decode only); to get it: re-run `convert_fp8_to_int4.py --repo
zai-org/GLM-5.2-FP8 --outdir /srv/models/glm52-i4 --ebits 8 --mtp` with proxy.

**`coli plan`:** 744.4 GB logical > 751 GB RAM. Budget 700.5 GB = 18.8 dense + 7.3 runtime + **674.4 warm experts**;
**~51 GB cold experts spill to disk**. "warn: cold expert misses may reach disk."

**BENCH (warm-ish page cache, AMX=1, 1 SNC node = numactl node0, 43 threads, prompt 11 tok, NGEN 64):**
Command `SNAP=/srv/models/glm52-i4 PROMPT="…" NGEN=64 AMX=1 OMP_NUM_THREADS=43 numactl --cpunodebind=0 --membind=0
./glm 237` (script `glmbench.sh`, log `glmbench-amx1.log`). Banner: `experts@8-bit dense@8-bit | idot: avx512-vnni
| amx: on | MTP absent`. Loaded in 9.9s, resident dense 17.5 GB.

| Phase | Result |
|---|---|
| **PREFILL (11 tok)** | **87.48 s total** |
| ├ expert-disk | **0.000 s service / 60.037 s WAIT** ← 69% of prefill = disk I/O wait |
| ├ expert-matmul (AMX) | 18.80 s (~21%) |
| ├ attention | 6.76 s |
| └ other | 1.78 s |
| **DECODE** | **~0.14 tok/s**, expert-cache hit **72%** (28% miss→disk), RSS climbing 242 GB+ |

### THE CENTRAL FINDING — colibri on this 744B model is DISK-BOUND, not compute-bound
- **~69% of prefill and the entire slow decode are dominated by `expert-disk WAIT`**, because the model does NOT
  stay resident in page cache (RAM during run: used 189 GB, buff/cache 557 GB — int4 shards fault in and get
  evicted; 72% hit / 28% disk-miss on decode). Our AMX kernel (18.8 s matmul, ~21% of prefill) is doing its job
  but is NOT the bottleneck here — **residency is the bigger lever than the kernel for this model/box.**
- **`experts@8-bit` (surprise):** despite `--ebits 4`, the engine loaded experts as **int8** (routed experts =
  `classify→"x"`, width from `--xbits` which defaults to `ebits`=4, yet banner shows 8-bit — the runtime chose
  int8 residency). int8 experts ≈ 2× int4 size → this is WHY the model is 744 GB and doesn't fit warm. Getting
  experts genuinely to **int4** (explicit `--xbits 4`) — or **int2 for cold experts** — is the lever to make the
  model fit fully in RAM with KV headroom, which would remove the disk-wait entirely.

This is the empirical motivation for the RAM-resident work (user: "Option 1 now, Option 3 future").

## 10. Next steps

1. **Make it fit in RAM (Option 1 completion):** re-convert (or a targeted re-quant) with explicit `--xbits 4`
   (int4 routed experts) — projected ~370-400 GB, fits 751 GB with big KV headroom → page-cache the whole model,
   disk-wait → 0. Then re-run the same bench: **cold vs warm × AMX=1 vs AMX=0**, and NOW the AMX win in prefill
   (expert-matmul) becomes the visible lever instead of being masked by disk. This is the run that actually
   measures our kernel end-to-end. (Consider `--xbits 2` on cold/rare experts if int4 still tight.)
2. **AMX on/off A/B** once RAM-resident: `AMX=0` vs `AMX=1`, compare `expert-matmul` seconds in the PREFILL PROFILE
   line (that isolates the kernel cleanly — same disk/attention, only the GEMM changes).
3. Optional: 1×2 ragged-M fallback in the colibri AMX kernel (close small-S gap vs oneDNN).
4. **FUTURE (Option 3):** re-architect to fully-resident NUMA-partitioned (all experts resident, one rank per SNC
   node, AMX on resident weights) — like the golden SGLang config. MTP head via `--repo`+proxy for speculative decode.

**Reference perf anchors:** DeepSeek-V4-Flash on this box — SGLang CPU-AMX golden container `v4-cpu-snc`
(SNC bind 0-42|43-85|86-127) measured **11.4-11.8 tok/s** single-stream decode (that model is ~150 GB and FITS
in RAM — which is exactly why it hits ~11 tok/s while GLM-5.2 at 744 GB is disk-bound at 0.14 tok/s; **the
difference is residency, not the kernel**).

**Artifacts on box** (`/home/intel/b70-sglang-xpu/colibri/c/`): `glm.c` (AMX-integrated, backups `.bak-amx`
/`.bak-2x2port`), `amxbench.c` (+ `.bak-2x2`/`.bak-onednn`; kernels vnni/amx/amx_2x2/onednn), `onednn_probe.cpp`,
`amx_idot_selftest.c`, `amxbench`/`amxbench_dnn` binaries, `glmbench.sh`/`glmbench-amx1.log`, `dl-glm52.sh`,
`conv-glm52.sh`/`conv-glm52.log`. Models: `/srv/models/glm52-fp8` (704 GB FP8 source), `/srv/models/glm52-i4`
(694 GB int4, experts loaded int8). Nothing committed to git.

---

## 11. PROJECT PIVOT — purpose-built GLM-5.2 runtime (`glmrt`), see DESIGN.md

User reframed the goal (m0938): colibri is a **base building block**, not the end product. Build a **small
runtime optimized for GLM-5.2 only, on our target configs only** — harvest colibri's correct forward pass +
AMX kernel, **replace its disk-streaming execution model** with RAM-resident. Requirements (m0938-m0947):
**(1) CPU-only** all configs for now (GPU tiers deferred); **(2)** model fully resident in RAM; **(3)** pin all
worker threads to physical cores, reserve a few per NUMA node for OS/orchestration/web-streaming;
**(4) multi-stream** continuous batching for production (this is what makes AMX pay in *decode*: N streams →
decode batches to S=N → GEMM → 2-10×). Full design in `colibri-ai/DESIGN.md`. Serving target =
`/srv/models/glm52-i4x` (true grouped-int4, `--xbits 4 --group-size 128`, **369 GB**, verified 6.29 MB/expert;
NOT `glm52-i4` which was accidentally int8/744 GB — the `--xbits` default didn't propagate in the `--indir` path).

### MVP-1 DONE — RAM-resident experts (fork `/home/intel/b70-sglang-xpu/glmrt/c/`)
Delegated to Sonnet 5. Forked colibri→glmrt (original untouched, md5-verified). Backup `glm.c.bak-mvp1`.
- **Change:** added `ESlot **experts; int resident;` to Model; at load, wire ALL experts resident via the
  EXISTING `expert_load()` looped over all (sparse-layer, eid) with OMP + forced `g_mmap=1` (zero-copy mmap
  views = page cache residency); in `moe()` resolve-scan, `RESIDENT=1` (default-on) path = `use[j] =
  &m->experts[layer][uniq[base+j]]` direct index, streaming miss/load/LRU-swap left dormant behind `!resident`.
  Net +80 lines. Compute region (expert_gate_up/matmul_qt) untouched. Also short-circuited PIN/AUTOPIN/
  cap_for_ram/CACHE_ROUTE/EXPERT_BUDGET under resident so the resident wire is the single source.
- **Results (real model, 1 SNC node, numactl --cpunodebind=0, AMX=1):** wired 19200 ESlots / 385 GB in 48s,
  loaded in 54s. **PROFILE PREFILL: expert-disk WAIT 60.037s → 0.000s** (the whole point). Prefill 87s→65s.
  Decode **0.14 → 2.7-2.9 tok/s** (~20×, TEMP=0 greedy, 100% hit). **Correctness: byte-identical token IDs**
  vs streaming (`RESIDENT=0` vs `=1`, TEMP=0, same prompt → same 7 token IDs) — proves the seam changed
  nothing computed. `amx: on`. Builds clean native + portable.

### NEW FINDING (reshapes MVP-3 priority): box is NUMA 3× ~257 GB, not one 751 GB pool
The 369 GB resident set **cannot fit in one NUMA node** → spills cross-socket. With compute pinned to node 0 but
~⅔ of experts on other nodes, **expert-matmul ROSE 18.8s → 42.7s** (cross-node reads during the GEMM) — which is
why prefill only hit 65s not ~25s. Same single-SNC-domain ceiling as the kernel A/B (§6/§7), now at model level.
Ran with `--cpunodebind=0` only (dropped `--membind=0`, which would hard-fail — a node holds only ~257 GB;
dmesg showed a historical `CONSTRAINT_MEMORY_POLICY` OOM at ~251 GB corroborating). **⇒ NUMA-partitioning is now
REQUIRED, not optional:** shard experts across the 3 SNC nodes, first-touch local, compute pinned to the node
holding its experts. This is MVP-3 and it directly recovers the 42.7s→~18s matmul.

### MVP-3a DONE — data placement (mbind) — and its lesson
Each expert's g/u/d weight+scale ranges `mbind`'d (MPOL_BIND, MPOL_MF_MOVE) to `expert_node(eid)=eid%3` in the
resident-wire loop (gated `NUMA_PARTITION`, default on); `-lnuma` in Makefile. Placed 115200 tensors, 0 failures.
**But data-placement ALONE gave only −3.8%** (60.05 vs 62.41s expert-matmul, 120 threads spread). And 120T-spread
(60s) was *slower* than MVP-1 node0-only 43T (42.7s) — cross-socket OMP fork/join sync dominates. **⇒ the NUMA win
needs COMPUTE-side pinning matched to the data**, same lesson as the kernel A/B (naive all-core spread loses to
per-node discipline).

### MVP-3b DONE — compute-side per-node passes (the real NUMA win) — done DIRECTLY (not delegated)
Two subagent attempts stalled on the long edit→build→multi-min-run→poll loop; did it directly via ssh_exec.
- **Infra** (glm.c, `#ifdef __linux__`): `NUMA_COMPUTE` (default on), `RESERVE_CORES` (default 3),
  `numa_topo_init()` builds per-node PHYSICAL-core `cpu_set_t` (keeps only ids < ncpu/2 = lower thread of each
  core, drops HT siblings; reserves 3/node → **n0=40 n1=40 n2=39**), `numa_bind_pool_to_node(n)` rebinds the OMP
  pool's threads to node n's cores via `#pragma omp parallel { pthread_setaffinity_np }`, `numa_pool_unbind()`
  restores all-cores after.
- **moe() restructure:** the CPU expert compute loop wrapped in **3 sequential per-node passes** (pass n∈{0,1,2}):
  before each, rebind pool to node n + `omp_set_num_threads(g_node_ncpu[n])`; inside, `continue` experts where
  `expert_node(eid)!=n`. So each expert's inner parallel-for runs node-local on node-local (mbind'd) weights.
  Sequential passes keep the shared `out[]`/scratch race-free. After passes, unbind + restore thread count.
  Only engages when `m->resident && g_nnodes>1 && S>=NUMA_COMPUTE_SMIN` (default **8**).
- **S-gating was essential:** first version (no S gate) ran the 3-pass path in decode too → decode **0.05 tok/s**
  (3× fork/join + rebind overhead per 64-expert block dwarfs the S=1 bandwidth-bound work). Gating to S≥8 sends
  prefill (S=11) through the NUMA path and decode (S=1) back to the fast single full-pool loop.
- **RESULTS (real model glm52-i4x, TEMP=0, drop_caches, AMX=1):**

| Config | prefill expert-matmul | prefill total | decode tok/s |
|---|---|---|---|
| MVP-1 (node0-only, 43T) | 42.7 s | 65 s | 2.7-2.9 |
| **NUMA_COMPUTE=0** (same binary, 120T spread, data placed) | **60.26 s** | 84.95 s | ~0.076 |
| **MVP-3b NUMA_COMPUTE=1** (data + 3-pass compute) | **16.6 s** | **31.4 s** | **2.25** |

  Clean same-binary A/B: **expert-matmul 60.26s → 16.6s = 3.6×** (NUMA_COMPUTE off vs on), and vs MVP-1's
  single-node 42.7s it's **2.6×** (all 3 nodes now compute in parallel on node-local data). Prefill total
  65/85s → 31.4s. Decode unaffected (2.25 tok/s, S-gated to the full-pool path; NUMA_COMPUTE=0's 0.076 is the
  spread-pool decode penalty which the S-gate avoids). **Token-exact:** output byte-identical
  (" The9.5; calledComments", same 7 IDs) across NUMA_COMPUTE=0/1 → restructure changed nothing computed.
  Topology auto-detected (`[numa-compute] 3 nodes n0=40 n1=40 n2=39`). Builds clean native + portable.
  Backup `glm.c.bak-mvp3b`. Note: wire+mbind MOVE from cold cache is slow (~135s vs ~50s warm) but one-time.

### Roadmap
- **MVP-2 (NEXT):** continuous-batching multi-stream scheduler (N sessions, per-session KV, batch-union MoE →
  per-expert AMX GEMM over batched rows). At S=N the AMX + 3-pass NUMA path pays in DECODE too (not just prefill).
  Route each expert's rows to the node holding it (reuses the MVP-3b per-node pass structure — that's why NUMA
  landed first). Measure aggregate tok/s scaling with N, AMX on/off. Anchor: SGLang DeepSeek-V4 GNR+1GPU 33 SS→218 agg@N=12.
- **HOT-EXPERT DOMAIN TIER (after MVP-2, user m0970):** colibri already has ROUTE_TRACE + `tools/expert_atlas/`
  (domain-specialist affinity w/ cross-prompt replication + leave-one-out validation) + eusage/pin_load/repin.
  Measure whether GLM-5.2's load-balanced router concentrates routing on the user's domains (coding / platform &
  software architecture / ESXi OS code analysis) enough to exploit (DeepSeek-V4 was ~uniform — open empirical Q).
  If skew is real: tier-0 hot domain-specialists **replicated across all 3 nodes** (every per-node pass serves them
  locally), tier-1 cold sharded by eid%3 as now.
- **Optional:** 1×2 ragged-M fallback in the colibri AMX kernel (close small-S gap vs oneDNN).

**glmrt artifacts** (`/home/intel/b70-sglang-xpu/glmrt/c/`): `glm.c` (resident, `.bak-mvp1`), `tiny/glm_bench_tiny/`
(regression fixture), `/tmp/glm-*.log`. Model `/srv/models/glm52-i4x` (369 GB true int4, 141 shards; config+
tokenizer copied from glm52-i4). Original `/home/intel/b70-sglang-xpu/colibri` tree untouched.

---

## 12. Single-stream decode is NOT memory-bandwidth-bound — PROVEN (user m1042/m1050 demanded proof)

I had *asserted* decode was BW-bound; the user rightly demanded measurement. **The assertion was wrong.**

**Machine peak (Intel MLC `/home/intel/tools/mlc/Linux/mlc`, cached `mlc-baremetal.txt`):** DDR5-**8800 MT/s** MRDIMM,
**753 GB/s aggregate all-reads**, **~252 GB/s per-node local**, ~210 GB/s cross-node. 3 nodes ~257 GB each.

**Bytes moved per decode token (computed from config):** grouped-int4 expert = 3·MI·D/2 nibbles + F32 group scales
(gs=128) ≈ **20.05 MB/expert**; (8 routed + 1 shared) × 75 sparse layers = **13.54 GB/token** of expert weights.

**Achieved vs peak (MVP-3b decode, 7 tok/3.11s = 444 ms/tok; expert-matmul 169 ms/tok):**
- expert-matmul phase: 13.54 GB / 169 ms = **80 GB/s = 10.7% of peak** (and only 32% of even *one* node's 252 GB/s).
- whole token: 13.54 GB / 444 ms = **30 GB/s = 4% of peak**.

**Thread-count sweep (node0, NUMA_COMPUTE off, decode is S=1 so single full-pool loop; grouped-int4 kernel):**

| OMP threads | decode tok/s |
|---|---|
| 8   | 0.54 |
| 40 (1 node) | 2.10 |
| 120 (3 nodes) | 2.25 |

Scales ~4× from 8→40 threads, then **plateaus 40→120** (basically flat). A BW-bound kernel would keep scaling to the
BW knee; a per-core-compute-bound one would scale with all 120 cores. **Plateau at ~1 node's cores while at 10.7% of
BW ⇒ decode is LATENCY / FORK-JOIN-OVERHEAD / MEMORY-LEVEL-PARALLELISM bound, NOT bandwidth-bound.**

**Root-cause suspects (evidence-based):**
1. **Fork/join overhead:** fmt=4 experts take the `else` branch in `expert_gate_up` (fused pair is `fmt==2` only) →
   gate + up + down = **3 separate `#pragma omp parallel for` per expert × 9 × 75 ≈ 2025 parallel regions/token**.
2. **Low memory-level parallelism** in `matmul_i4_grouped` (per-output-row dependent reduction, 16-wide SSE, group
   scale reset every 128) — not enough outstanding loads/thread to hide DRAM latency → 10% of BW.
3. **Decode not NUMA-distributed:** single parallel-for over all experts; other-node experts are remote reads and
   only one node's local BW is in play (adding nodes 2/3 cores did nothing → the 40→120 plateau).

**Re-ranked single-stream levers (by evidence, not assumption):**
- **A. Fuse gate+up for fmt=4** (currently fmt=2 only) → ~⅓ fewer parallel regions + one activation read instead of two.
- **B. Batch a layer's experts into ONE parallel region** (instead of 3/expert) → collapse ~2025 → ~75-225 launches/token.
- **C. Raise memory-level parallelism in the grouped-int4 GEMV** (more independent accumulators / prefetch) to climb
  from 10% toward the BW ceiling.
- **D. NUMA-distribute decode** so all 3 nodes' local BW (3×252) is usable — but only helps once A/B/C lift the per-node
  ceiling above one node's BW.
- **E. MTP speculative decode** (multiplicative, orthogonal — fewer forwards per token).

Pipeline (user m1050: "put all 3 in pipeline + measure actual memory transaction speed, I don't believe we are
memory bandwidth bound, need proofs" — proof delivered above): do A+B+C (kernel/launch restructure), then D, then E.
Logs `/tmp/sweep-{8t,40t}.log`, `/tmp/mvp3b-on2.log`.

---

## 13. MXFP4 kernel ported into colibri (fmt=5) — the single-stream WIN

Decision (user m1116/m1131): match the golden DeepSeek-V4 config, which uses **MXFP4 (E2M1 nibbles + E8M0
power-of-2 group microscales) → BF16 → AVX512-BF16 `dpbf16`** for experts (kt-kernel `GemmKernel224MXFP4SmallKGroup`
— **NOT AMX tiles**; `grep -c _tile_ = 0`). Confirms our earlier `tma_amx_busy=0%` was expected. Converter (Sonnet 5)
→ `/srv/models/glm52-mxfp4` (369.9 GB, gs=32, E8M0 byte scales; MXFP4 **10.7%** rel-err **beat** int4gs128 **12.8%**
on 288/288 tensors). Scale sidecar name = `<weight_name> + "_scale"` (append, no dot), U8 `[O, I/32]`.

**Kernel port (I did this directly):** new `fmt=5` in glm.c — `QT.e8` field, `matmul_mxfp4` + `matmul_mxfp4_bf16`
(4×4 register-blocked dpbf16, ported from fp4-moe.hpp), `_mxfp4_to_bf16_32` PSHUFB LUT unpack, f32→bf16 linear-order
convert, `expert_load` mmap fmt=5 detection, batch-union (convert activations bf16 once/expert, fused gate+up+down),
E8M0→f32 scale precompute at wire time, dispatch in `matmul_qt_ex`. Backup `glm.c.bak-mxfp4`. **3 bugs fixed:**
(1) `.qs`-only gate routed MXFP4 to the broken f32 fallback → `st_read_f32` segfault; (2) sidecar name was `_scale`
append not `.weight_scale`; (3) mmap 4-align gate rejected byte-scale offset → relaxed for MXFP4 (E8M0 read unaligned).
**Correctness VALIDATED:** "The capital of France is **Paris**".

**The perf saga (microbench-lied trap, then root-caused):**
- Microbench (`colibri/c/mxfp4bench.c`, isolated, nr=20, 40T): MXFP4 **1.65× FASTER** than int4.
- First in-engine run: MXFP4 **8× SLOWER** (127s vs int4 16.6s expert-matmul @ S=642). Batch-union + ldexpf-precompute
  did NOT fix it.
- **ROOT CAUSE (the real bug):** the **MVP-3b NUMA 3-pass**. Its `numa_bind_pool_to_node()` (`#pragma omp parallel {
  pthread_setaffinity_np }`) + `omp_set_num_threads()` fire **per 64-expert-block × per node-pass × per layer** (the
  pass loop cycles node0→1→2 *inside* each block, so the guard never skips). This rebind/thread-count **churn** is
  benign for int4's simple f32-FMA loop but ~10× catastrophic for the fork-heavier MXFP4 dpbf16 kernel.
- **MXFP4 S=642 prefill expert-matmul by config:** NUMA-3-pass churn **127s** | 120T-spread (data-placed, no 3-pass)
  **32s** | **40T single-node (NUMA off) 12s** ← winner. The fork-heavy per-expert kernel is fastest on ONE node's
  40 cores; thread-spread pays cross-socket fork sync; the 3-pass rebind is pure poison.

**SINGLE-STREAM RESULT (validated, this is the win):** MXFP4 on one node/40 threads, NUMA 3-pass OFF, BEATS int4:

| single-stream metric | int4 (best) | **MXFP4 (recipe)** |
|---|---|---|
| decode tok/s | 2.25 | **2.99 (+33%)** |
| prefill S=642 expert-matmul | 16.6s (NUMA) | **12s** (single-node) |
| accuracy (rel-err vs fp8) | 12.8% | **10.7%** |

**Recipe** = `/home/intel/b70-sglang-xpu/glmrt/c/run-mxfp4.sh`: MXFP4 model + `NUMA_PARTITION=0 NUMA_COMPUTE=0
OMP_NUM_THREADS=40 numactl --cpunodebind=0` + `AMX=1 COLI_MMAP=1 RESIDENT=1 COLI_NO_OMP_TUNE=1`. The MXFP4 kernel
uses AVX512-BF16 (`dpbf16`), NOT AMX — AMX stays 0% (correct for this kernel family; AMX tiles were 3.5-3.8× slower
at decode M=1 per the golden config's own dead-end).

**Key lesson:** the NUMA 3-pass (MVP-3b) is a **batched/multi-stream** optimization; for single-stream + the
fork-heavy MXFP4 kernel it is actively harmful. Single-node is the single-stream sweet spot. The 3-pass's per-block
rebind nesting (pass-inside-block) is the structural flaw — a proper fix (pass-outside-block, rebind once/pass/layer)
is deferred to MVP-2 where multi-stream makes all-3-node compute worthwhile.

**glmrt artifacts added:** `glm.c` (+`.bak-mxfp4`), `mxfp4bench.c`/`mxfp4bench` (kernel A/B harness), `run-mxfp4.sh`
(the recipe), logs `/tmp/mxfp4-*.log`. Model `/srv/models/glm52-mxfp4`. Converter `conv-glm52-mxfp4.py` in
`/home/intel/b70-sglang-xpu/`.

---

## 14. MTP speculative decode — head converted + activates, but drafts don't work (0% acceptance) — PARKED

Goal (user m1327): enable GLM-5.2's native MTP (multi-token-prediction / "nextn") head for a multiplicative
single-stream speedup. Status: **head converted, engine activates it, but draft acceptance is 0-4% → net slowdown;
root cause not fully found; PARKED. The MXFP4 win (§13) stands independent of this.**

**Conversion (Sonnet 5):** GLM-5.2's MTP head = model **layer 78** (1569 tensors: full MLA+DSA+MoE layer + the
MTP-specific `eh_proj`/`enorm`/`hnorm`/`shared_head.norm`). Converted + merged into `/srv/models/glm52-mxfp4` as
`model-layer-078.safetensors` (5.3 GB): 256 experts → MXFP4 (`_scale`, 10.7% rel-err), non-expert weights →
grouped-int4 (`.qs`, gs=128), norms/router → F32 raw. Index updated (116915→118478 tensors, backup
`.index.json.bak-pre-mtp`). All 16 engine-required `has_mtp` tensors present. (Subagent corrected 2 of my spec
errors: shard naming is `model-layer-NNN.safetensors`; `mlp.gate` router is F32-raw not int4. Also: `st_init`
dir-scans shards, doesn't need the index — but index updated anyway.)

**Engine:** MTP machinery was ALREADY fully built in colibri (`mtp_draft`/`mtp_absorb`/`spec_decode`/`has_mtp`/
`spec_pinned`/`mtp_norm`). It auto-activates when the layer-78 tensors are present: `[MTP] active: native
speculative decoding (draft=3)`. Loads fine (76 rows / 390 GB).

**The problem: 0-4% draft acceptance → MTP is a NET SLOWDOWN** (1.36 tok/s w/ MTP vs 2.99 without — failed drafts
add forward passes). Base model is CORRECT (predicts " Rome" for "capital of Italy is"), so it's the MTP HEAD
mispredicting, not model quality.

**Debugging done (all ruled out):**
- `MTP_SWAP` (concat order) and `MTP_PRENORM` env toggles: both still 0%.
- **Reference wiring research (Sonnet 5, via SGLang source):** GLM-5.2 (`GlmMoeDsaForCausalLMNextN`) inherits
  DeepSeek-V3's `DeepseekModelNextN` MTP forward verbatim. Found **Bug A**: colibri chained the RAW post-layer
  hidden `hx` between draft steps instead of the `shared_head.norm`-normalized `row` (reference feeds
  `spec_info.hidden_states` = post-shared_head.norm). Also **Bug B**: a position off-by-one (`p=kv-1`), argued
  mathematically inert under relative RoPE + self-referential KV. Everything else (concat order/content, embed
  token, hidden pre/post-final-norm on g=0, eh_proj [D,2D] orientation, lm_head sharing, 1 MTP layer autoregressed
  for depth) matches the reference at colibri's defaults.
- **Applied Bug A fix** (glm.c ~4758: `memcpy(h,hx)` → `memcpy(h,row)`, backup `.bak-mtpfix`): still 0%, and
  critically **draft0 (g=0) itself is wrong** (`draft0=13 verified=17`), so it's not just the chaining.
- **Applied eh_proj at F32** (full precision — re-emitted `model.layers.78.eh_proj.weight.f32` shard, 302 MB;
  engine loads it as fmt=0 when present, logs `[MTP] eh_proj loaded at F32`): still 0%, draft0 still wrong. So
  NOT eh_proj int4 lossiness.
- **Diagnostic (MTP_DEBUG=2):** `pre_blk` (the MTP prediction from just eh_proj→mtp_norm→lm_head, BEFORE the
  transformer layer) is ~RANDOM every step (92219, 18, 84821...). Since pre_blk involves no attention/position/MoE,
  this means **`eh_proj(cat)` output is not in the right space for lm_head** — even with f32 eh_proj + verified-
  correct norm/lm_head loads + correct concat order. Genuinely puzzling; would need a numerical dump of the MTP
  intermediate vectors compared against a reference PyTorch forward of the layer-78 head to locate.

**Why PARKED (honest):** (1) two concrete identified bugs fixed without moving acceptance off 0; the remaining cause
needs deep numerical archaeology (dump/compare intermediates vs a reference forward) with uncertain payoff; (2) this
checkpoint SELF-EOSes after ~4 tokens on every prompt, so even a working MTP would help little for these short
generations; (3) the banked single-stream win is MXFP4 @ 2.99 tok/s regardless. MTP default stays OFF (`MTP=0`, or
just don't need it — but note the head IS in glm52-mxfp4 now so it auto-activates; **use `MTP=0` to disable** for the
fast 2.99 tok/s path, else it drops to ~1.4). Head + f32-eh_proj shards left in place for future debugging.
Artifacts: `glm.c.bak-mtpfix`, `/srv/models/glm52-mxfp4/model-layer-078.safetensors` + `model-ehproj-f32.safetensors`,
`conv-glm52-mtp-layer78.py`, logs `/tmp/mtp-*.log`.

**IMPORTANT run-recipe note:** since the MTP head now auto-activates in glm52-mxfp4 and is a slowdown, the fast
single-stream recipe should add `MTP=0`. Update run-mxfp4.sh accordingly (TODO) or pass `MTP=0`.

**UPDATE — MTP numerical debug (Sonnet 5, deeper):** the eh_proj→norm→lm_head projection chain is **PROVEN
CORRECT** — dequantized colibri's on-disk eh_proj vs pristine bf16 source: per-row cosine 0.98-0.995 (int4-typical,
not transposed); RMSNorm convention is plain `x*w` matching `GlmMoeDsaRMSNorm` (NOT `1+w`); norms/eps/lm_head all
match the reference `DeepseekModelNextN.forward` (GLM-5.2 = `GlmMoeDsaForCausalLMNextN` inherits DeepSeek-V3 MTP
verbatim); hlast is provably the correct hidden. Ran the full chain with real weights → reproduces the reference
logit distribution almost exactly. So H1(double-norm)/H2(1+w)/H3(orientation) all **refuted**. The `pre_blk`
"randomness" is EXPECTED (it queries lm_head with the pre-transformer-block vector = off-manifold) and clusters
non-uniformly (~1000× enrichment) = an attractor, not scrambled data. **⇒ the real bug is in the MTP TRANSFORMER
BLOCK** (`layer_forward(m->mtpL, hx, 1, pos)`, glm.c ~4750) or its **position/KV bookkeeping** (`pos=(kv-1)+g`;
`kv_start`/`mtp_absorb` position consistency) — i.e. the researcher's earlier **"Bug B" (position off-by-one)** is
now the prime suspect, NOT the projection. `run-mxfp4.sh` updated with `MTP=0`. run.mxfp4.sh done.

If MTP is revisited: apply Bug B (position: `p=kv` not `kv-1`; +1 in mtp_absorb) and instrument `pos` per draft step
vs a non-speculative replay; that's the last unverified seam.

---

## 15. Hot-expert domain-tier — MEASURED, NOT VIABLE for GLM-5.2 (routing doesn't concentrate)

User's original idea (m0970, revisited m1388): pin/replicate a hot subset of experts specialized to the user's
domains (coding / platform & software architecture / ESXi OS-code analysis) to exploit routing skew. Open empirical
question: does GLM-5.2's aux-loss-balanced router concentrate routing on those domains enough?

**Method:** built a 9-category × 4-prompt tech-domain probe set (`glmrt/c/probes-tech.json`: code_c_systems,
code_kernel_driver, esxi_hypervisor, software_architecture, code_python, code_sql + contrast: prose_story,
math_proof, translation_de). Ran the atlas sweep on the MXFP4 model via `glmrt/c/atlas-sweep-tech.sh` (glm binary,
`STATS=` per-run histogram dump, the 4 confounds controlled: TOPP=0, MTP=0, DRAFT=0, greedy, fresh routing history,
single-node recipe). 36 stats files in `glmrt/c/atlas-tech/stats/`. Analyzed with `tools/expert_atlas/analyze.py`
+ `validate.py` (leave-one-prompt-out).

**RESULT — decisive negative:**
- **Strong specialists (spec≥0.5): 28 / 18,865 = 0.1%** (vs the reference general-topic int4 sweep's **7.9%**).
- **Specialization vs depth: 0.007-0.026 across ALL layers** (vs reference's 0.19-0.27 at mid layers) — ~10-30× lower.
- **Every prompt touches ~19,000 of 19,200 experts (~99%)** — near-total spread.
- **Leave-one-prompt-out: 3/36 = 8.3% — WORSE THAN CHANCE (11.1%).** The domain-specialist sets do NOT generalize;
  routing follows raw expert popularity (nearly every held-out prompt misclassifies as `code_c_systems`, the
  highest-count category; own-set match ~0.05-0.14% vs best-other ~1.1-1.3%).

**VERDICT: GLM-5.2's routing is NOT skewed by these technical sub-domains — a hot-expert domain tier would give
essentially nothing.** Same conclusion class as DeepSeek-V4 (~uniform routing). Why the contrast with the reference
atlas's 96.7% leave-one-out + 7.9% specialists: that reference sweep used BROAD, cross-lingual, cross-modal topics
(chinese/german/poetry/law/medicine); our probe set is all narrow technical/coding domains which are routing-wise
near-identical to GLM's router — they hit the same generalist experts. Coding/systems/ESXi are simply not a routing
axis GLM-5.2 separates on. The hot-expert-tier idea is **empirically closed** for this workload. Artifacts:
`glmrt/c/probes-tech.json`, `atlas-sweep-tech.sh`, `atlas-tech/{stats,experts.json,validate.log}`.
