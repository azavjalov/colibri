/* onednn_probe.cpp -- calls oneDNN's brgemm (Intel AMX int8 production
 * GEMM), exactly the ATen entry point Intel's sgl-kernel uses
 * (at::native::cpublas::brgemm, csrc/cpu/gemm_int8.cpp), so amxbench.c can
 * compare our hand-written AMX kernels against the production ceiling.
 *
 * Kept as a separate C++ translation unit (extern "C" ABI) because
 * ATen/native/CPUBlas.h is a C++ header; amxbench.c stays plain C and links
 * against this .o only when built with -DUSE_ONEDNN (see amxbench.c).
 *
 * oneDNN is not shipped as a standalone libdnnl.so here -- it's compiled
 * into libtorch_cpu.so. The four symbols we need are all exported with
 * default visibility (confirmed via `nm -D libtorch_cpu.so`):
 *   at::native::cpublas::brgemm(M,N,K,ld_a,ld_b,ld_c,add_C,
 *                                const signed char* A, const signed char* B,
 *                                int32_t* C, bool is_vnni)      -- s8s8->i32
 *   at::native::cpublas::pack(K,N,ld_in,ld_out,dt_in,dt_out,in,out)
 *   at::native::cpublas::could_pack(ScalarType)
 *   at::native::cpublas::brgemm_release(bool is_vnni)
 * The raw dnnl_brgemm_create/dnnl_transform_create C ukernel symbols are
 * NOT exported from libtorch_cpu.so (checked, absent from `nm -D`), so the
 * "OPTION 2" raw ukernel path is not linkable here -- this file uses the
 * ATen wrapper (OPTION 1) exclusively, including for weight packing.
 *
 * We deliberately use the *signed*-char/*signed*-char overload (true s8s8),
 * matching amxbench's other kernels (gemm_amx/gemm_amx_2x2 use _tile_dpbssd,
 * true s8*s8, no sign trick). sgl-kernel's own gemm_int8.cpp instead calls
 * the unsigned/signed (u8s8) overload and subtracts a per-channel int32
 * "Bcomp" compensation it fuses into weight packing -- that's an AVX512-VNNI
 * sign-trick artifact (u8*b - 128*b) that AMX's native s8s8 dpbssd doesn't
 * need. Using s8s8i32 directly gives us the same int32 dot products as
 * gemm_scalar with zero compensation bookkeeping.
 */

#include <ATen/native/CPUBlas.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>

extern "C" {

int gemm_onednn_available = 0;

/* Block size for both the brgemm call grid AND the weight packing grid.
 * 32 = 2*16 matches Intel's own sgl-kernel gemm.h block_size_m()/block_size_n()
 * (2*TILE_M, 2*TILE_N) -- i.e. the same 32x32 / four-AMX-tile macro-block our
 * own gemm_amx_2x2 uses, just with oneDNN's JIT'd microkernel body instead of
 * our hand-written one. */
#define DNN_BLOCK_M 32
#define DNN_BLOCK_N 32

/* One-time probe: does this build/CPU actually get a working AMX ukernel
 * out of libtorch_cpu, or would brgemm/pack silently be unusable? Both
 * cpublas::could_pack(Char) and the int8 brgemm's internal device_check()
 * gate on the same oneDNN ukernel + ISA requirement, so this is a faithful
 * stand-in probe for "will brgemm(is_vnni=true) actually work". */
void gemm_onednn_init(void) {
  gemm_onednn_available = 0;
  try {
    gemm_onednn_available = at::native::cpublas::could_pack(at::kChar) ? 1 : 0;
  } catch (const std::exception &e) {
    fprintf(stderr, "[onednn] could_pack(Char) probe threw: %s\n", e.what());
  } catch (...) {
    fprintf(stderr, "[onednn] could_pack(Char) probe threw unknown exception\n");
  }
  if (!gemm_onednn_available) {
    fprintf(stderr,
            "[onednn] could_pack(Char) == false: AMX ukernel path unavailable "
            "(needs oneDNN ukernel build + avx512_core_amx) -- onednn column "
            "will be skipped\n");
  }
}

/* Pack B[O][I] (our row-major weight storage: O rows/output-channels, I
 * contiguous input-channels per row) into oneDNN's VNNI/AMX-blocked layout
 * for brgemm(..., is_vnni=true).
 *
 * ATen's cpublas::pack() always uses dnnl::ukernel::pack_type::no_trans
 * internally (hard-coded in CPUBlas.cpp's Pack::call) -- i.e. it expects the
 * *input* in canonical [K,N] row-major form (N contiguous). Our B is stored
 * [N,K]=[O,I] (K/I contiguous) -- the transposed convention -- so we
 * materialize a one-time [I][O] transpose (Bt, ld=O) first, exactly
 * analogous to how prepack_B() does a one-time, untimed setup pass for our
 * own AMX kernels.
 *
 * We then pack in independent DNN_BLOCK_N-wide column blocks (K=I, N=32,
 * ld_in=O, ld_out=32 per call) rather than one call for the full O: this
 * matches sgl-kernel's own observed convention exactly (each brgemm call
 * later uses ld_b == block width, e.g. gemm_int8.cpp's
 * `ldb=nb_size`/`cpublas::brgemm(M,N,K,lda,ldb,BLOCK_N,...)`), so each
 * block's packed bytes are self-contained/contiguous (32*I bytes, blocks
 * concatenated by nb) and the per-block ld_b at brgemm-call time is
 * unambiguous -- no assumption needed about whether a single big pack()
 * call auto-chunks internally. All calls for a given (K,N=32,ld_in,ld_out)
 * share one cached JIT'd pack kernel (ATen's PackKey cache), so looping
 * O/32 times here costs one JIT + O/32 fast executes, not O/32 JITs.
 *
 * Bp_out must be a caller-allocated buffer of exactly O*I bytes (packing is
 * a pure rearrangement -- no padding, since I%4==0 and O%32==0 for our test
 * shapes).
 */
void gemm_onednn_prepack_B(int8_t *Bp_out, const int8_t *B, int I, int O) {
  if (O % DNN_BLOCK_N != 0) {
    fprintf(stderr, "[onednn] prepack_B: O=%d not a multiple of %d, aborting pack\n", O, DNN_BLOCK_N);
    gemm_onednn_available = 0;
    return;
  }
  int8_t *Bt = (int8_t *)malloc((size_t)I * (size_t)O);
  if (!Bt) {
    fprintf(stderr, "[onednn] prepack_B: Bt alloc failed\n");
    gemm_onednn_available = 0;
    return;
  }
  for (int o = 0; o < O; o++) {
    const int8_t *row = B + (int64_t)o * I;
    for (int i = 0; i < I; i++) Bt[(int64_t)i * O + o] = row[i];
  }

  int nNb = O / DNN_BLOCK_N;
  try {
    for (int nb = 0; nb < nNb; nb++) {
      const int8_t *in = Bt + nb * DNN_BLOCK_N;
      int8_t *out = Bp_out + (int64_t)nb * DNN_BLOCK_N * I;
      at::native::cpublas::pack(
          /* K */ I, /* N */ DNN_BLOCK_N, /* ld_in */ O, /* ld_out */ DNN_BLOCK_N,
          at::kChar, at::kChar, in, out);
    }
  } catch (const std::exception &e) {
    fprintf(stderr, "[onednn] pack() threw: %s\n", e.what());
    gemm_onednn_available = 0;
  } catch (...) {
    fprintf(stderr, "[onednn] pack() threw unknown exception\n");
    gemm_onednn_available = 0;
  }
  free(Bt);
}

/* GEMM: acc[S][O] = A[S][I] . B^T[O][I], via oneDNN brgemm (s8s8->i32).
 *
 * brgemm is a *ukernel*: single-threaded per call, caller supplies the
 * threading -- exactly like our own gemm_amx_2x2's #pragma omp parallel
 * around raw _tile_* intrinsics. We tile the (S,O) output into the same
 * 32x32 macro-blocks used for packing and let OpenMP spread them across
 * threads; each call covers the FULL K=I reduction internally (oneDNN's
 * JIT'd kernel loops over K itself, the same role our own `for(ko...)`
 * loop plays), so there is no K-blocking across separate brgemm calls.
 *
 * No S-padding needed (unlike gemm_amx/gemm_amx_2x2): brgemm takes the
 * real per-call M (m_size, clipped on the ragged S edge), so we operate
 * directly on the caller's real A/S -- no Ap/Sp scratch required.
 */
void gemm_onednn(int32_t *acc, const int8_t *A, const int8_t *Bp, int S, int I, int O) {
  const int nMb = (S + DNN_BLOCK_M - 1) / DNN_BLOCK_M;
  const int nNb = O / DNN_BLOCK_N;

#pragma omp parallel
  {
    int32_t Ctmp[DNN_BLOCK_M * DNN_BLOCK_N];
#pragma omp for schedule(static) collapse(2)
    for (int mb = 0; mb < nMb; mb++) {
      for (int nb = 0; nb < nNb; nb++) {
        int m0 = mb * DNN_BLOCK_M;
        int m_size = (m0 + DNN_BLOCK_M <= S) ? DNN_BLOCK_M : (S - m0);
        int n0 = nb * DNN_BLOCK_N;
        const int8_t *Arow = A + (int64_t)m0 * I;
        const int8_t *Bblk = Bp + (int64_t)n0 * I;
        try {
          at::native::cpublas::brgemm(
              m_size, DNN_BLOCK_N, I,
              /* ld_a */ I, /* ld_b */ DNN_BLOCK_N, /* ld_c */ DNN_BLOCK_N,
              /* add_C */ false, Arow, Bblk, Ctmp, /* is_vnni */ true);
        } catch (const std::exception &e) {
          fprintf(stderr, "[onednn] brgemm() threw: %s\n", e.what());
        } catch (...) {
          fprintf(stderr, "[onednn] brgemm() threw unknown exception\n");
        }
        for (int m = 0; m < m_size; m++)
          memcpy(acc + (int64_t)(m0 + m) * O + n0, Ctmp + m * DNN_BLOCK_N, DNN_BLOCK_N * sizeof(int32_t));
      }
    }
    /* Per-thread AMX hw-context teardown, mirroring sgl-kernel's own usage
     * (called once per thread, inside the parallel region, right after that
     * thread's last brgemm call) and our own kernels' per-thread
     * _tile_release(). */
    at::native::cpublas::brgemm_release(true);
  }
}

} /* extern "C" */
