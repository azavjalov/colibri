# Task 3 — Intel Arc Pro B70 (Battlemage) GPU offload: research plan

**Status:** RESEARCH ONLY. No backend code committed yet. Final host-API choice
(Level-Zero vs SYCL) is **deferred behind a spike** — see §5. This doc realizes
the GPU seam that `DESIGN.md §1` deferred ("add a GPU seam later, not now"; config 2 =
"GNR + 2× B70 hot-expert tier").

Substrate constraint (`glmrt`-is-lean, per DESIGN §0): the default build stays
single-file `c/glm.c` + in-tree headers, links only libm/libgomp/libnuma/libc. The
B70 backend is opt-in `#ifdef COLI_XPU` with a **single** `-lze` **or** `-lsycl` link,
mirroring how `COLI_CUDA`/`COLI_METAL` are absent from the default build.

---

## 1. Hardware & toolchain inventory (aibox101b — verified this session)

**GPUs — two B70s present, fully operational:**
- lspci: `66:00.0` and `8c:00.0`, both `Battlemage G31 [Intel Graphics] [8086:e223]`.
- Both bound to the modern `xe` DRM driver (not legacy i915).
- Render nodes: `/dev/dri/renderD128` (66:00.0), `/dev/dri/renderD129` (8c:00.0); group `render`.
- **32 GB VRAM each** (lspci BAR = 32G), **64 GB total**. Single tile each (tile0). Both on NUMA node 0.

**Toolchain (Ubuntu/dpkg):**
- Level-Zero: `libze_loader.so.1.28.2` + `libze_intel_gpu.so.1.15.38646` in
  `/usr/lib/x86_64-linux-gnu/`. Packages: `libze1` 1.28.2-2, `libze-dev` 1.28.2-2,
  `libze-intel-gpu1` 26.22.38646.4-0.
- OpenCL: `intel-opencl-icd` 26.22.38646.4-0 **working** — `clinfo` enumerates
  "Intel(R) Arc(TM) Pro B70 Graphics" (both GPUs) + the Xeon 6980P CPU device.
- oneAPI: full toolkit at `/opt/intel/oneapi/`. Compilers `icpx`/`dpcpp`/`sycl-ls` at
  `/opt/intel/oneapi/compiler/{2026.1,2025.3}/bin/`. UMF at `/opt/intel/oneapi/umf/{1.0,1.1}/lib/`.

**CRITICAL GAP — SYCL broken out of the box (fixable, env only):**
`sycl-ls` → "No platforms found" because the loader can't load the level_zero/opencl UR
adapters: `libumf.so.1: cannot open shared object file`. The lib EXISTS at
`/opt/intel/oneapi/umf/1.1/lib/libumf.so.1` but is not on the linker search path.
**Fix (one line):**
```
export LD_LIBRARY_PATH=/opt/intel/oneapi/umf/1.1/lib:/opt/intel/oneapi/compiler/2026.1/opt/compiler/lib:$LD_LIBRARY_PATH
# or: source /opt/intel/oneapi/setvars.sh
```
Then `sycl-ls` enumerates both B70s. This is an env-setup issue, NOT missing
HW/driver/runtime. Level-Zero needs the same fix (both L0 and SYCL depend on libumf).

---

## 2. How colibri's GPU backend is structured (the pattern to mirror)

**The abstraction seam is scattered inline `#ifdef` dispatch, NOT a function-pointer /
backend-struct table.** Each op self-gates its own GPU calls inline, then falls through to
CPU. 49 `COLI_CUDA` blocks + 19 `COLI_METAL` blocks in `c/glm.c` (~7198 lines).

**Per-op dispatch pattern:**
```c
#ifdef COLI_CUDA
  if (g_cuda_enabled && <tensor eligible checks> && !omp_in_parallel()) {
      if (coli_cuda_op(...)) return;   // GPU took it
      w->cuda_failed = 1;              // else mark + fall through
  }
#endif
// CPU fallback below
```

**Kernel source layout (`c/`):**
- `backend_cuda.cu` (~1024 lines): ALL CUDA kernels inline. quant_matmul (main matmul),
  silu_mul, w4a16_matmul, w4a16_gate_up, grouped_s4_wmma (tensor-core expert GEMM),
  grouped_hidden/_w4/_w4_dual, grouped_down_w4, attention_absorb_kernel/_batch,
  pipe_rmsnorm_rows, pipe_rope_rows, pipe_add_n, pipe_rows_add, + ~15 `coli_cuda_*` C wrappers.
- `backend_cuda.h` (~143 lines): public API — `coli_cuda_init/_shutdown/_matmul/_expert_mlp/
  _expert_group/_attention_absorb{,_batch}/_project_batch{,_dev}/_shared_mlp_w4a16` +
  resident-pipe primitives (`_pipe_scratch/_upload/_download/_rmsnorm/_rope`).
- `backend_metal.mm` (~786) + `backend_metal.h`; `backend_loader.c` (~445, Windows DLL loader).
- Tests: `tests/test_backend_cuda.cu`, `tests/bench_tensor_core.cu`, `tests/test_pipe_cuda.cu`.

**glm.c dispatch/integration points (exact lines):**
- L62 `#ifdef COLI_CUDA` include; L151-156 QT field `ColiCudaTensor *cuda`;
  L187-188 QT flags cuda_eligible/cuda_failed/cuda_device; L289-313 globals
  (g_cuda_enabled, g_cuda_devices[], g_cuda_dense/_pipe/_expert_gb + qt_cuda_reset/_upload/_update).
- `matmul_qt_ex()` L1655-1720 — DISPATCH: Metal gate L1664-1669, CUDA gate+fallback L1669-1678,
  CPU L1679+.
- `qt_load()` L2069-2087 sets cuda_eligible, round-robin device L2079-2082.
- `qt_cuda_colocate()` L2154-2161 pins attn chain (q_a,q_b,kv_a,o_proj) to kv_b's device;
  called L2357-2360 (attn) & L2378-2380 (shared expert).
- `attention_rows()` L3357-3600: Metal decode L3362-3396 (S<=4, absorb); CUDA PIPE L3414-3438
  (S>=8, `_attention_project_batch_dev`); CUDA absorb L3524-3585 (S<=4, env COLI_CUDA_ATTN=1,
  `_attention_absorb_batch`); CPU L3600+.
- `moe()` L3717-4305: routing L3717-3755 (CPU routes for both backends); CUDA grouped experts
  L4138-4165 (≤64 experts/block, `_expert_group`), CUDA single L4160-4165 (`_expert_mlp`),
  CPU expert L4166-4189; shared expert L4276-4290 (CUDA `_shared_mlp_w4a16` if S>=shared_min &
  env COLI_CUDA_SHARED_W4A16=1).
- `main()` init L6981-7006 (env COLI_CUDA, devices, g_cuda_enabled); shutdown L6562-6594; stats L6326-6351.

**Memory model:** host weights via `qalloc()` (~L1426). Device = opaque
`ColiCudaTensor {void *weights; float *scales; int device;}` (backend_cuda.cu L11-18),
lazy upload on first `coli_cuda_matmul` via `coli_cuda_tensor_upload` (L453); host copy freed
if `CUDA_RELEASE_HOST=1`. Resident pipe (`_pipe_upload/_download/_scratch`) keeps multi-layer
ops device-resident.

**Build gating (`c/Makefile`):** `CUDA=1` → `-DCOLI_CUDA`, links `backend_cuda.o` via
`-L$(CUDA_HOME)/lib64 -lcudart -lstdc++`, CUDA_ARCH=native; `CUDA_DLL=1` (Win) → `backend_loader.o`;
`METAL=1` → `-DCOLI_METAL`, `backend_metal.o`, `-framework Metal -framework Foundation`. Default CPU-only.

**KEY GAP inherited from CUDA:** CUDA has **NO MXFP4 kernel** — fmt=5 experts are CPU-only
(`matmul_mxfp4()` AVX-512, glm.c ~L1692-1695). This is the crux for B70 (see §4).

---

## 3. Proposed COLI_XPU integration surface

Mirror the CUDA seam exactly:
- New `c/backend_xpu.h` + `c/backend_xpu.cpp` (or `.dp.cpp` for SYCL): ~15 `coli_xpu_*`
  functions 1:1 with the `coli_cuda_*` API, opaque `ColiXpuTensor`.
- ~30 lines added to `glm.c` at the ~10 dispatch sites listed in §2 (include L62, QT field
  L151, globals L289, matmul gate L1664-1678, qt_load eligibility L2079, attention L3414/L3524,
  moe L4138/L4276, main init L6981).
- `c/Makefile`: `XPU=1` → `-DCOLI_XPU` + Intel compiler + single link.
- Dispatch precedence: Metal → CUDA → **XPU** → CPU (mutually exclusive at runtime).
- Zero impact on the lean default build (guarded, opt-in).

---

## 4. What to offload first (informed by the CPU perf profile)

Settled fact (serial-profile.md): **the MoE MXFP4 decode path is DRAM-bandwidth-bound
(~42 GB/s, ~5% of peak) at N=1.** Implications for B70 offload strategy:
- **Expert weights are the bandwidth problem, not the compute problem.** B70 wins only if the
  hot experts live in the 64 GB of B70 VRAM so their bandwidth comes off the GPU's HBM/GDDR
  instead of host DRAM — i.e. this must be a **hot-expert residency tier** (DESIGN §1 config 2),
  not a per-op compute offload of a host-resident weight.
- 377 GB model ≫ 64 GB VRAM → only a **subset of experts** can be resident. Needs a
  hot-set / routing-frequency policy (open research item — measure expert hit distribution).
- CUDA lacks an MXFP4 kernel; a B70 MXFP4 E2M1+E8M0 dequant kernel would be **net-new**
  (can reuse the LUT technique from the closed CPU negative — see serial-profile.md — since the
  DRAM-bandwidth disqualifier does NOT apply once weights are in VRAM).
- **Validation gate is unchanged and HARD (user rule): wall-clock decode tok/s at N=1 AND N=16.**
  A microbench/compute win alone is meaningless. Plus the SCORE gate (0 argmax flips) for quality.

### 4a. EXPERT HIT-FREQUENCY MEASUREMENT (run this session — STRATEGY-CHANGING RESULT)

Measured with the existing built-in instrumentation (NO patch needed): the runtime already
tracks `m->eusage[layer][expert]` and dumps `layer expert count` triples when `STATS=<file>`
is set (glm.c:6420 `stats_dump_q`, called at end of run L7196; the HOT-STORE `EXPERT_BUDGET`
loader at glm.c:6439 already consumes this same file to pick top experts by frequency — that IS
the residency mechanism a backend would reuse).

Run: deterministic REPLAY harness (`REPLAY=1 REF=ref_bench.json REF_FORCE=1`, TEMP=0, 80 tokens,
OMP=86) → `[STATS] 2,726,400 selections across 19,084 distinct experts`.

**Per-expert MXFP4 VRAM footprint = 20.05 MB** (gate+up+down = 37,748,736 elems; nibbles/2 +
E8M0 group bytes/32). → only **~3,191 experts fit in 64 GB** (2,792 in ~56 GB usable) out of
**19,200 total** (75 MoE layers × 256).

**The routing distribution is NEARLY FLAT — this undercuts the naive hot-set strategy:**
- **99.4%** of all (layer,expert) slots were hit at least once in just 80 tokens.
- **mean 142.9 selections/slot vs 142.0 uniform** — essentially uniform. top=2,225, median=70, min=1.
- **Cumulative mass:** top 2,792 slots (56 GB) carry only **52.7%** of selections; top 3,191
  (64 GB) = **56.5%**. Filling ALL VRAM with the hottest experts still misses ~44% of expert reads.
- **Within-layer concentration weak:** top-8/256 experts = ~19.6% of a layer's traffic (min 14.4,
  max 28.9); need top-64/256 to reach ~69%.

**Implication (revises §4 strategy):** a STATIC hot-expert VRAM tier caps at ~53% expert-read
hit-rate at 56 GB. Since decode is DRAM-bandwidth-bound and per-token latency is gated by the
slowest path, offloading only ~53% of expert bandwidth to the GPU leaves the other ~47% still
bottlenecked on host DRAM — at best a partial win, likely far below linear. **A static hot-set is
NOT clearly worth it.** Re-frame the B70 role around one of:
  1. **Full-model streaming through B70** — stage ALL experts' bandwidth through GPU GDDR (needs
     the model resident in VRAM, which 64 GB can't hold → doesn't work for 375 GB). Rejected.
  2. **Dynamic per-token residency / prefetch** — upload the top-8×75 = 600 selected experts for
     the *next* token to VRAM ahead of compute. But that's 600×20 MB = 12 GB/token over PCIe
     each step — PCIe 5.0 x16 ≈ 64 GB/s → ~190 ms/token just for transfer, worse than the current
     ~260 ms/token CPU decode only if it overlaps poorly. Needs measurement, likely a loss.
  3. **Compute-tier only for the SHARED expert + dense layers** (always-hit, small, fit easily in
     VRAM) — bounded, cheap, but small share of total expert bandwidth.
  4. **Batch/throughput regime (N≫1)** where expert reuse across the batch amortizes weight
     bandwidth — the flat single-stream distribution matters less when many tokens share experts.

**Caveats:** single 80-token replay on one prompt set (ref_bench.json); real multi-user serving
may concentrate differently, and the N≫1 continuous-batching regime (DESIGN's actual deployment)
was NOT measured here — expert reuse across a batch could restore concentration. **Before writing
any backend, measure the batched (N=16) expert-reuse rate** — that is the regime that matters for
the serving target, and it may reverse this negative.

### 4b. BATCHED (N≫1) EXPERT-REUSE — the caveat is RESOLVED, and it's still a NEGATIVE

The N=16 caveat above is now closed. Two independent confirmations that batching does NOT restore
concentration enough to amortize expert-weight bandwidth:

1. **Combinatorics of near-uniform top-8/256 routing** (batch-union distinct experts per
   layer-step, E=256, K=8):
   - N=1  → 8 requests → ~7.9 distinct, reuse **1.01** rows/expert
   - N=16 → 128 requests → ~100.9 distinct, reuse **1.27** rows/expert  *(matches the
     independently-recorded MoE-AMX finding: "~1.27 rows/expert at N=16, need ~512 to fill a tile")*
   - N=32 → reuse 1.58 ; N=64 → 2.31 ; N=128 → 4.07. Reuse only becomes meaningful at batch
     sizes far beyond the decode serving regime, and even then a loaded expert weight serves only
     a handful of tokens.
2. **Wall-clock** (REPLAY 80-tok, identical trace — note BATCH=16 is a NO-OP under REPLAY+REF_FORCE,
   which replays one fixed single-stream trace, so this is a timing cross-check not a true 16-seq
   batch): expert-matmul N=1 = 15.220s vs "N=16" = 13.861s (**−8.9% only**). Expert-weight traffic
   does not amortize with batch because each distinct expert still gets streamed from DRAM ~once
   per token that selects it.

**CONCLUSION — recommend SHELVING the B70 hot-expert offload as a documented clean negative.**
The workload is (a) DRAM-bandwidth-bound, (b) near-uniform in expert access (flat distribution,
1.27× batch reuse at N=16), and (c) far too large for VRAM (375 GB ≫ 64 GB, static hot-set caps
at ~53% hit). A per-op or hot-set GPU offload cannot help a bandwidth-bound, near-uniform-access,
VRAM-overflowing workload — the ~47% of expert reads that miss VRAM still bottleneck host DRAM and
gate per-token latency. This mirrors the MXFP4-LUT and MoE-AMX negatives: real perf is gated by
memory bandwidth, not by where the compute runs.

**What COULD still be worth a bounded follow-up (NOT the main backend):** offload only the
always-hit, VRAM-fitting pieces — the **shared expert + the 3 dense layers (0-2)** — as a small
`COLI_XPU` compute tier. These are hit every token, fit trivially in VRAM, and their bandwidth
would come off GPU GDDR. But this is a small share of total expert bandwidth; measure the ceiling
before investing. The Level-Zero infrastructure decision (§5a) stands if this narrow path is pursued.

### 4c. RECONCILIATION vs the NVIDIA+AMX hybrid ("why does *that* boost a VRAM-overflow model?") + ceiling

Prompted by the observation that our production NVIDIA-GPU + Granite-Rapids-CPU hybrid gets a
strong boost on DeepSeek-V4-Flash despite the model not fitting VRAM. Investigated two systems:

**(A) colibri's own `COLI_CUDA` path (glm.c) — measured on the 6×RTX 5090 rig, NOT this box:**
- The ~48× (0.12→5.77 tok/s) win was **disk elimination / full residency**, not GPU compute
  placement. That lever is **already banked on aibox101b** (RESIDENT=1, zero disk, 3.0-4.2 tok/s
  N=1 with NO GPU).
- GPU/CPU expert compute is **strictly sequential** — every `coli_cuda_*` ends in
  `cudaStreamSynchronize` (backend_cuda.cu); the moe() loop computes cold experts on CPU inline,
  then calls `coli_cuda_expert_group` only after. NO overlap. The 5090 experiment log explicitly
  records "CPU/GPU expert overlap via pthreads: no stable gain."
- Structural dead-end for OUR checkpoint: `row_bytes()` returns 0 for fmt=5, so
  `coli_cuda_tensor_upload` **fails for native MXFP4 routed experts** — the CUDA routed-expert
  tier is architecturally inert here; the 5090 numbers used a separately int4-quantized artifact.

**(B) The production SGLang + KTransformers kt-kernel hybrid (DeepSeek-V4-Flash, RTX 6000 Blackwell
+ dual Xeon 6972P) — the "good boost" system:** ~30 tok/s single-stream, 120 @6, 219 @12. Its
mechanism, from its own tuning record (perf_skills/sglang_deepseek_v4_flash_gpu_hybrid/SKILL.md):
- **The win is CPU/GPU OVERLAP + disk-free residency, NOT putting experts on GPU.** Its own
  DEAD-ENDS: `--kt-num-gpu-experts` sweep (32/48/56) = **"wash"**; explicit lesson #2 =
  **"Increase CPU/GPU overlap, don't move experts to GPU."** The 40/256 GPU-resident experts exist
  mainly to free the deferral pipeline, not for hit-rate. Its GPU util is ~88-91% via a **1-layer
  expert-deferral pipeline** (GPU runs dense/attn/shared while CPU chews the deferred experts).
- V4-Flash was CHOSEN for this role precisely because it has the **lowest active params (13B →
  lowest CPU feed bytes/token)** and a shared expert — i.e. it minimizes the exact
  DRAM-bandwidth-bound routed-expert feed that dominates GLM-5.2 (37B active, 8/256 experts).
- Notably, its CPU MXFP4→BF16 unpack patch (the SAME vpermw+E8M0 technique we closed as a glmrt
  CPU negative) DID win there (+6-7% single, +26-48% concurrent) — because kt-kernel's AMX BRGEMM
  path is more compute-bound at concurrency; it does NOT contradict our N=1 bandwidth-bound finding.

**Ceiling measurement (this box, from weight sizes; bandwidth-bound ⇒ time ∝ bytes):**
- Per-token MoE-weight reads: routed experts = 8×75×20.05 MB = **12.03 GB (86.6%)**; shared expert
  = 75×20.05 MB = 1.50 GB; dense MLP layers 0-2 = 3×120.32 MB = 0.36 GB.
- **Always-hit, VRAM-fitting set (shared + dense 0-2) = 1.87 GB/token = only 13.4%** of MoE-weight
  bytes; persistent VRAM footprint 1.87 GB (fits 64 GB trivially).
- Mapped to measured decode (expert-matmul 15.22s = 73% of 20.84s N=1 decode): a GPU tier that
  removed the always-hit bytes from the CPU critical path has a **sequential ceiling of ~9.8% of
  decode (~2.04s of 20.84s).** Under an overlap model the realizable gain is bounded by the longer
  path (CPU routed = 86.6%), so overlapping away the always-hit 13.4% yields **≤~10% end-to-end**.

**FINAL VERDICT — SHELVE.** To capture even that ≤10% ceiling on B70 we would first have to build
an **async CPU/GPU overlap pipeline that glmrt does not have** (its GPU path is fully synchronous),
which is the bulk of the effort — for a bandwidth-bound ≤10% ceiling, on the wrong model (GLM-5.2's
37B active dwarfs V4-Flash's 13B). The production hybrid's boost comes from (i) disk elimination
(already banked here) and (ii) CPU/GPU overlap on a deliberately-low-active-param model — neither
transfers to a B70 hot-expert/compute tier for GLM-5.2 CPU-resident decode. The Level-Zero
infrastructure finding (§5a) is preserved for any future narrow experiment, but no backend is
justified now.

---

## 5. Level-Zero vs SYCL — decision matrix (choice DEFERRED behind a spike)

Per user: **compare both in the plan, commit to neither yet.** Gate the choice on a small
spike that (a) applies the libumf fix, (b) enumerates both B70s, (c) runs a hello-world GEMM.

| Axis | Level-Zero (`-lze`) | SYCL / icpx (`-lsycl`) |
|---|---|---|
| Leanness (link) | Leanest: single `-lze`, thin loader | `-lsycl` + fatter oneAPI runtime |
| Structural fit to backend_cuda.cu | Explicit device/queue/memory — maps well to the resident-pipe model, but more boilerplate | Closest analog to CUDA runtime API; backend_xpu mirrors backend_cuda.cu shape most directly |
| Kernel language | SPIR-V / OpenCL C (separate kernel build) | Single-source C++ (`icpx`); MXFP4 intrinsics inline in C++ |
| Toolchain readiness on box | Runtime installed; needs libumf path fix | All present; blocked only by libumf path fix |
| MXFP4 dequant kernel effort | More manual (explicit SPIR-V/OCL-C) | Easier to express in C++ w/ sub-group intrinsics |
| Debug/tooling | Lower-level, sparser | icpx + oneAPI tooling richer |

**Provisional lean toward SYCL** for structural fit + MXFP4 kernel ergonomics, **but not
committed** — the spike decides. If the spike shows `-lsycl` runtime bloat violates the lean
constraint unacceptably, fall back to Level-Zero.

### 5a. SPIKE RESULTS (run this session — DECISION MADE: Level-Zero)

Both APIs were proven end-to-end on a B70 with a trivial GEMM. **Recommendation flipped from
the provisional SYCL to Level-Zero**, driven by the lean-build constraint.

**libumf fix confirmed:** `export LD_LIBRARY_PATH=/opt/intel/oneapi/umf/1.1/lib:/opt/intel/oneapi/compiler/2026.1/opt/compiler/lib:$LD_LIBRARY_PATH`
→ `sycl-ls` enumerates both B70s via Level-Zero AND OpenCL + the Xeon. (Env-only; no HW/driver issue.)

**SYCL (icpx):** GEMM 256³ fp16→fp32, **max|gpu-cpu| = 0, PASS**. BUT:
- **AOT `-fsycl-targets=spir64_gen -Xs "-device bmg"` HANGS** at kernel execution (process alive,
  both render nodes open, no progress >60s — had to `pkill -9`).
- **JIT (plain `-fsycl`) works instantly and correctly.** So on this stack, prefer JIT/online
  compile over AOT-for-bmg. Host code ~62 lines (kernel inlined as lambda).
- `ldd`: **16 shared libs** — pulls the full Intel runtime: `libsycl.so.9`, `libur_loader.so.0`,
  `libimf`, `libsvml`, `libirng`, `libintlc.so.5`, libz, libpthread, librt, libdl, + system.

**Level-Zero (`-lze_loader`):** GEMM 256³ fp32, **max err = 5.72e-6, PASS**. Details:
- Kernel written in OpenCL C → SPIR-V via **`ocloc compile -file k.cl -device bmg -spv_only`**
  (2528-byte .spv, genuine SPIR-V magic). Host: `gcc -O2 -o spike_ze_gemm spike_ze_gemm.c -lze_loader -lm`.
- **NO hang** — SPIR-V is JIT'd by the L0 driver at module-create, ran in ~1s every time.
- **BUG found + fixed (important for the real backend):** Level-Zero does **NOT** auto-insert
  completion dependencies between commands on the same command list. First attempt raced — D2H
  copy of C ran ahead of the kernel, ~97% of C read back zero, nondeterministic zero-count across
  runs. **Fix: `zeCommandListAppendBarrier()` after the H2D copies and after the kernel launch.**
  Then deterministic PASS. (SYCL's queue model inserts these deps implicitly — a real ergonomic
  cost of L0 we must budget for.)
- `ldd`: **7 shared libs** — `libze_loader.so.1` (system-packaged) + libc/libstdc++/libgcc_s/libm.
  **ZERO oneAPI runtime libs.**
- **Bonus:** the L0 binary + `ocloc` both work in a **stripped env** (`env -i`, no setvars, no
  libumf path). The libumf fix is only needed for the SYCL/UR/icpx path, NOT for plain
  Level-Zero + system `ocloc`.
- Host code ~224 lines + 17-line kernel = **~3.6× more boilerplate than SYCL** (manual driver/
  device enum, command-queue-group discovery, explicit context/queue/list/module/kernel lifecycle,
  arg-binding by index, the barrier above).

**DECISION: Level-Zero for COLI_XPU.** Rationale (lean wins, per DESIGN §0):
- Link footprint 7 vs 16 libs; **zero oneAPI runtime dependency** — matches "links only
  libm/libgomp/libnuma/libc" ethos. A single `-lze_loader` (system pkg `libze1`) is the leanest
  possible GPU link, cleaner than CUDA's `-lcudart -lstdc++`.
- No fragile env setup (works without setvars/libumf) → robust default for anyone building `XPU=1`.
- AOT-for-bmg instability on the SYCL path is avoided entirely (L0 JITs SPIR-V reliably).
- Cost accepted: ~3.6× host boilerplate + must manage command-list barriers/deps manually. This is
  a one-time backend cost, contained in `backend_xpu.cpp`, and the explicit device/memory/queue
  control actually maps well to colibri's resident-pipe model (§2).
- MXFP4 dequant kernel goes in OpenCL C → SPIR-V via `ocloc` (or hand SPIR-V); the E2M1+E8M0 LUT
  technique ports fine to OpenCL C.

---

## 6. Next concrete steps (in order)

1. **Spike (blocks the decision):** apply libumf `LD_LIBRARY_PATH` fix; confirm `sycl-ls`
   + a Level-Zero `zeInit`/device-enum both see the two B70s; run a trivial FP16/BF16 GEMM on
   one B70 via each API; record link footprint of each. → picks L0 vs SYCL.
   **DONE (see §5a): both PASS; chose Level-Zero.**
2. Measure **expert hit-frequency distribution** on the deterministic REPLAY harness to size the
   hot-expert VRAM residency set (what fits in 64 GB, expected hit-rate).
   **DONE (see §4a): distribution nearly FLAT — static 56 GB hot-set caps at ~53% hit-rate.
   This casts doubt on the whole hot-set strategy.**
3. **Batched (N=16) expert-reuse.** DONE (see §4b): batch-union reuse is only **1.27 rows/expert**
   at N=16 (combinatorics + wall-clock, expert-matmul −8.9% only). Concentration is NOT restored by
   batching → the caveat is resolved as a NEGATIVE.
5. **Ceiling + hybrid reconciliation.** DONE (see §4c): always-hit VRAM-fitting set (shared +
   dense 0-2) = 13.4% of MoE-weight bytes → sequential ceiling ~9.8% of decode, ≤~10% under
   overlap. The production NVIDIA+AMX hybrid's boost = disk-elimination (already banked here) +
   CPU/GPU overlap on a low-active-param model (V4-Flash 13B); glmrt has NO overlap pipeline
   (synchronous GPU path). Verdict stands: **SHELVE.**
6. **DONE-DECISION: shelve.** No `COLI_XPU` backend justified for GLM-5.2 CPU-resident decode.
   Preserve the Level-Zero infra finding (§5a) for any future narrow experiment. Record the
   negative in `docs/glm52-cpu/serial-profile.md` alongside MXFP4-LUT / MoE-AMX.

---

## 7. Don'ts (carried from settled decisions)
- Do NOT re-litigate the MXFP4 LUT CPU negative — but note its disqualifier (DRAM-bandwidth-bound)
  does not apply in VRAM, so the LUT dequant technique is reusable on-GPU.
- Any GPU compute win MUST be validated by wall-clock at N=1 AND N=16 (hard user rule); microbench
  alone is meaningless.
- Keep the lean default build untouched: `#ifdef COLI_XPU`, single link, opt-in only.
