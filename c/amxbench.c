/* amxbench.c — A/B micro-benchmark of colibri's int8 GEMM path on x86.
 *
 * Models colibri's matmul_q_idot: C[S][O] = A[S][I] . B^T[O][I], int8->int32,
 * with per-row activation scale sx[s] and per-out-row weight scale scale[o].
 *   A = activations xq  [S][I] int8 (row-major)
 *   B = weights     q   [O][I] int8 (row-major; each row is one output channel)
 *   y[s][o] = dot(B[o], A[s]) * scale[o] * sx[s]
 *
 * Three kernels, all producing identical int32 dot products (scales applied after):
 *   1. scalar reference          (correctness oracle)
 *   2. AVX-512-VNNI row-loop     (== colibri's current x86 kernel: dot_i8i8 per (s,o))
 *   3. AMX-tile (_tile_dpbssd)   (the ported tiled batch kernel)
 *
 * Signed x signed int8. colibri's VNNI uses a sign-trick because vpdpbusd is u8*s8;
 * AMX _tile_dpbssd is s8*s8 directly, so no trick needed.
 *
 * Build: gcc -O3 -march=native -fopenmp amxbench.c -o amxbench -lm
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <immintrin.h>
#include <sys/syscall.h>
#include <unistd.h>

#ifdef USE_ONEDNN
/* oneDNN brgemm comparison kernel -- see onednn_probe.cpp (separate C++ TU,
 * extern "C" linkage, linked in only for this build). Mirrors the
 * gemm_amx/gemm_amx_2x2(acc,A,Bp,S,I,O) call shape: Bp is pre-packed once by
 * gemm_onednn_prepack_B(), reused across the timing loop. */
extern int gemm_onednn_available;
void gemm_onednn_init(void);
void gemm_onednn_prepack_B(int8_t *Bp_out, const int8_t *B, int I, int O);
void gemm_onednn(int32_t *acc, const int8_t *A, const int8_t *Bp, int S, int I, int O);
#endif

/* ---- AMX tile config + XFEATURE enable ---- */
#define ARCH_REQ_XCOMP_PERM 0x1023
#define XFEATURE_XTILEDATA 18
static int amx_enable(void){
    if(syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA)!=0){
        perror("arch_prctl XTILEDATA"); return -1;
    }
    return 0;
}

typedef struct __attribute__((packed)) {
    uint8_t palette; uint8_t start_row; uint8_t reserved[14];
    uint16_t colsb[16]; uint8_t rows[16];
} tilecfg_t;

static double now_s(void){
    struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t);
    return t.tv_sec + t.tv_nsec*1e-9;
}

/* ---------- 1. scalar reference ---------- */
static void gemm_scalar(int32_t *acc, const int8_t *A, const int8_t *B, int S, int I, int O){
    for(int s=0;s<S;s++)
        for(int o=0;o<O;o++){
            int32_t d=0; const int8_t *a=A+(int64_t)s*I, *b=B+(int64_t)o*I;
            for(int i=0;i<I;i++) d += (int32_t)a[i]*b[i];
            acc[(int64_t)s*O+o]=d;
        }
}

/* ---------- 2. AVX-512-VNNI, row-by-row (colibri's dot_i8i8, x86 path) ---------- */
static inline int32_t dot_i8i8_vnni(const int8_t *w, const int8_t *x, int I){
    int32_t sum=0; int i=0;
    __m512i a=_mm512_setzero_si512();
    for(;i+64<=I;i+=64){
        __m512i wv=_mm512_loadu_si512((const void*)(w+i));
        __m512i xv=_mm512_loadu_si512((const void*)(x+i));
        __mmask64 neg=_mm512_movepi8_mask(wv);
        __m512i xs=_mm512_mask_sub_epi8(xv,neg,_mm512_setzero_si512(),xv);
        a=_mm512_dpbusd_epi32(a,_mm512_abs_epi8(wv),xs);
    }
    sum=_mm512_reduce_add_epi32(a);
    for(;i<I;i++) sum+=(int32_t)w[i]*x[i];
    return sum;
}
static void gemm_vnni(int32_t *acc, const int8_t *A, const int8_t *B, int S, int I, int O){
    #pragma omp parallel for schedule(static)
    for(int o=0;o<O;o++){
        const int8_t *b=B+(int64_t)o*I;
        for(int s=0;s<S;s++)
            acc[(int64_t)s*O+o]=dot_i8i8_vnni(b, A+(int64_t)s*I, I);
    }
}

/* ---------- 3. AMX tiles ----------
 * _tile_dpbssd computes C[m][n] += sum_k A_tile[m][k]*B_tile[n... ] with the
 * AMX layout: A tile = M rows x (K*4 bytes as int8, K groups of 4), B tile =
 * (K) rows x (N*4 bytes) with 4 int8 packed along the row per N. To multiply
 * A[S][I] by B^T[O][I] we tile: M<=16 rows of A, N<=16 output channels, K=64
 * contraction chunk. B must be repacked into the AMX "VNNI" layout:
 * Bp[o_tile][k/4][n][4] i.e. for a 16xK B block, layout is [K/4][16][4].
 *
 * We pre-pack the whole B[O][I] once into Bp so the inner loop is pure tile ops
 * (matches how a real engine would store weights). A is packed per M-block.
 */

/* pack a [rows x K] int8 block (row-major, rows<=16, K mult of 4) into AMX
 * layout [K/4][rows][4] flattened. Used for both A (M rows) and B (N rows). */
static void pack_amx(int8_t *dst, const int8_t *src, int rows, int K, int64_t src_stride){
    /* dst laid out as K/4 groups, each 'rows' entries of 4 bytes = rows*4 bytes,
     * total (K/4)*rows*4 = rows*K bytes. dst[(kg*rows + r)*4 + b] = src[r][kg*4+b] */
    for(int r=0;r<rows;r++){
        const int8_t *sr=src+(int64_t)r*src_stride;
        for(int kg=0; kg<K/4; kg++){
            int8_t *d=dst + ((int64_t)kg*rows + r)*4;
            d[0]=sr[kg*4+0]; d[1]=sr[kg*4+1]; d[2]=sr[kg*4+2]; d[3]=sr[kg*4+3];
        }
    }
}

/* pack a full [rows x I] int8 block (rows<=16) into AMX layout for the whole
 * K dimension: output chunks [I/KT][KT/4][rows][4]. */
static void pack_amx_full(int8_t *dst, const int8_t *src, int rows, int I){
    const int KT=64;
    for(int ko=0; ko<I/KT; ko++)
        pack_amx(dst + (int64_t)ko*(KT/4)*rows*4, src + ko*KT, rows, KT, I);
}

/* AMX GEMM: acc[S][O] = A[S][I] . B^T[O][I], with Bp pre-packed per 16-col tile. */
static void gemm_amx(int32_t *acc, const int8_t *A, const int8_t *Bp_all,
                     int S, int I, int O){
    const int MT=16, NT=16, KT=64;      /* tile dims: 16x64 int8 (A), 16x64 (B), 16x16 int32 (C) */
    tilecfg_t cfg; memset(&cfg,0,sizeof(cfg)); cfg.palette=1;
    /* tile 0 = C (16 rows x 16 int32 = 64 colsb); tile 1 = A (16 x 64 int8);
     * tile 2 = B packed (16 x 64 int8). */
    cfg.rows[0]=MT; cfg.colsb[0]=NT*4;
    cfg.rows[1]=MT; cfg.colsb[1]=KT;
    cfg.rows[2]=KT/4; cfg.colsb[2]=NT*4;   /* B packed: K/4 rows x (N*4) bytes */
    _tile_loadconfig(&cfg);

    int nOt = O/NT;                 /* assume O, I multiples of tile dims for the bench */
    int nKt = I/KT;
    int nSt = (S+MT-1)/MT;
    #pragma omp parallel
    {
        _tile_loadconfig(&cfg);
        /* parallelize over the (S-tile, O-tile) grid so all cores stay busy even
         * when S is small: at S<=16 the work is nOt output tiles across threads. */
        #pragma omp for schedule(static) collapse(2)
        for(int so=0; so<nSt; so++){
            for(int no=0; no<nOt; no++){
                int s0=so*MT;
                const int8_t *Arow = A+(int64_t)s0*I;   /* A row-major, no packing */
                const int8_t *Bp = Bp_all + (int64_t)no*I*NT;
                _tile_zero(0);
                for(int ko=0; ko<nKt; ko++){
                    _tile_loadd(1, Arow + ko*KT, I);                     /* A block: 16 x 64, stride I */
                    _tile_loadd(2, Bp + (int64_t)ko*(KT/4)*NT*4, NT*4);  /* B block VNNI: (K/4) x (N*4) */
                    _tile_dpbssd(0,1,2);
                }
                int32_t tmp[16*16];
                _tile_stored(0, tmp, NT*4);
                int mrows = (s0+MT<=S)?MT:(S-s0);
                for(int m=0;m<mrows;m++)
                    memcpy(acc+(int64_t)(s0+m)*O+no*NT, tmp+m*NT, NT*sizeof(int32_t));
            }
        }
        _tile_release();
    }
}

/* pre-pack full B[O][I] into per-16-col-tile AMX layout:
 * Bp_all[no][ko][kg_in_tile][16][4]  == for each 16-output-channel tile 'no',
 * store as nKt chunks each [KT/4][16][4]. */
static void prepack_B(int8_t *Bp_all, const int8_t *B, int I, int O){
    const int NT=16, KT=64;
    int nKt=I/KT;
    for(int no=0; no<O/NT; no++){
        int8_t *dst = Bp_all + (int64_t)no*I*NT;
        for(int ko=0; ko<nKt; ko++){
            /* pack 16 output-channels (rows of B), K=64 slice -> [KT/4][16][4] */
            pack_amx(dst + (int64_t)ko*(KT/4)*NT*4,
                     B + (int64_t)no*NT*I + ko*KT, NT, KT, I);
        }
    }
}

/* AMX GEMM variant that packs B on the fly (per output-tile, thread-local) —
 * models the in-engine reality where weights arrive row-major and there's no
 * persistent pre-packed copy (e.g. streaming experts). Pack cost O(N*I) vs
 * GEMM O(S*N*I) MACs, so this is only amortized at larger S. */
static void gemm_amx_packB(int32_t *acc, const int8_t *A, const int8_t *B,
                           int S, int I, int O){
    const int MT=16, NT=16, KT=64;
    tilecfg_t cfg; memset(&cfg,0,sizeof(cfg)); cfg.palette=1;
    cfg.rows[0]=MT; cfg.colsb[0]=NT*4;
    cfg.rows[1]=MT; cfg.colsb[1]=KT;
    cfg.rows[2]=KT/4; cfg.colsb[2]=NT*4;
    int nOt=O/NT, nKt=I/KT, nSt=(S+MT-1)/MT;
    #pragma omp parallel
    {
        _tile_loadconfig(&cfg);
        int8_t *Bp=(int8_t*)aligned_alloc(64,(size_t)NT*I);   /* one 16-channel packed tile */
        #pragma omp for schedule(static)
        for(int no=0; no<nOt; no++){
            /* pack this 16-output-channel block of B once, reuse across all S-tiles */
            pack_amx_full(Bp, B+(int64_t)no*NT*I, NT, I);
            for(int so=0; so<nSt; so++){
                int s0=so*MT; const int8_t *Arow=A+(int64_t)s0*I;
                _tile_zero(0);
                for(int ko=0; ko<nKt; ko++){
                    _tile_loadd(1, Arow+ko*KT, I);
                    _tile_loadd(2, Bp+(int64_t)ko*(KT/4)*NT*4, NT*4);
                    _tile_dpbssd(0,1,2);
                }
                int32_t tmp[16*16]; _tile_stored(0,tmp,NT*4);
                int mrows=(s0+MT<=S)?MT:(S-s0);
                for(int m=0;m<mrows;m++)
                    memcpy(acc+(int64_t)(s0+m)*O+no*NT, tmp+m*NT, NT*sizeof(int32_t));
            }
        }
        free(Bp);
        _tile_release();
    }
}

/* ---------- 4. AMX tiles, 2x2 register-blocked (OPT1) ----------
 * Single-tile gemm_amx above keeps only 1 C accumulator live, so each K-step
 * is 2 tile loads (A,B) feeding exactly 1 tdpbssd -> the TMUL pipe stalls
 * waiting on tileload latency between multiplies. Intel's oneDNN/sgl-kernel
 * AMX GEMM instead blocks 2x2 in tile-register space: a 32-row x 32-col
 * macro-block held as FOUR live C tiles, fed by 2 A row-tiles and 2 B
 * col-tiles, so every 4 tile loads feed 4 INDEPENDENT tdpbssd ops (4x the
 * useful work per load pair), which hides tileload->tdpbssd latency far
 * better because the 4 dpbssd's have no data dependency on each other.
 *
 * All 8 AMX tile registers are used:
 *   tile0..3 = C00,C01,C10,C11  (16x16 int32 each; rows 0-15/16-31 x cols 0-15/16-31)
 *   tile4,5  = A0,A1            (16x64 int8; M rows 0-15 / 16-31 of the macro-block)
 *   tile6,7  = B0,B1            (16x64 packed VNNI; N cols 0-15 / 16-31 of the macro-block)
 *
 * Bp_all is prepack_B's per-16-col-tile layout, so B0/B1 for macro-col-block
 * 'nb' are simply the two ADJACENT 16-col tiles Bp_all + (2*nb)*I*NT and
 * + (2*nb+1)*I*NT — no repacking needed.
 *
 * Ragged edges (S or O not a multiple of 32) fall back to a single-tile path
 * that reuses tile0 (C), tile4 (A), tile6 (B) — their cfg shapes are IDENTICAL
 * to gemm_amx's tile0/1/2, so no second _tile_loadconfig is needed; one config
 * serves both the 2x2 bulk path and the ragged fallback (OPT3: config load is
 * hoisted to once per thread per parallel region either way).
 */
static void gemm_amx_2x2(int32_t *acc, const int8_t *A, const int8_t *Bp_all,
                          int S, int I, int O){
    const int MT=16, NT=16, KT=64;
    tilecfg_t cfg; memset(&cfg,0,sizeof(cfg)); cfg.palette=1;
    for(int t=0;t<4;t++){ cfg.rows[t]=MT; cfg.colsb[t]=NT*4; }  /* C00,C01,C10,C11 */
    cfg.rows[4]=MT;    cfg.colsb[4]=KT;                          /* A0 */
    cfg.rows[5]=MT;    cfg.colsb[5]=KT;                          /* A1 */
    cfg.rows[6]=KT/4;  cfg.colsb[6]=NT*4;                        /* B0 */
    cfg.rows[7]=KT/4;  cfg.colsb[7]=NT*4;                        /* B1 */

    int nOt=O/NT, nKt=I/KT, nSt=(S+MT-1)/MT;
    int nOb=nOt/2, nOt_rem=nOt%2;
    int nSb=nSt/2, nSt_rem=nSt%2;

    #pragma omp parallel
    {
        _tile_loadconfig(&cfg);

        /* --- bulk: 32x32 macro-blocks, 4 live C tiles, 4 dpbssd per K-step ---
         * Guard with a (loop-invariant, thread-uniform) runtime if: when S<32
         * nSb==0 and this work-sharing construct would have zero trip count
         * anyway, but the omp-for dispatch + implicit barrier still costs a
         * few us — skip it entirely rather than pay that on every ragged-only
         * call (matters at S=1/8 where the whole kernel call is a few 10s of us). */
        if(nSb>0 && nOb>0){
        #pragma omp for schedule(static) collapse(2)
        for(int sb=0; sb<nSb; sb++){
            for(int nb=0; nb<nOb; nb++){
                int s0=sb*2*MT;
                const int8_t *A0row=A+(int64_t)s0*I;
                const int8_t *A1row=A+(int64_t)(s0+MT)*I;
                const int8_t *B0=Bp_all+(int64_t)(2*nb)*I*NT;
                const int8_t *B1=Bp_all+(int64_t)(2*nb+1)*I*NT;
                _tile_zero(0); _tile_zero(1); _tile_zero(2); _tile_zero(3);
                for(int ko=0; ko<nKt; ko++){
                    int64_t boff=(int64_t)ko*(KT/4)*NT*4;
                    _tile_loadd(4, A0row+ko*KT, I);
                    _tile_loadd(5, A1row+ko*KT, I);
                    _tile_loadd(6, B0+boff, NT*4);
                    _tile_loadd(7, B1+boff, NT*4);
                    _tile_dpbssd(0,4,6);   /* C00 += A0.B0 */
                    _tile_dpbssd(1,4,7);   /* C01 += A0.B1 */
                    _tile_dpbssd(2,5,6);   /* C10 += A1.B0 */
                    _tile_dpbssd(3,5,7);   /* C11 += A1.B1 */
                }
                int32_t t00[16*16],t01[16*16],t10[16*16],t11[16*16];
                _tile_stored(0,t00,NT*4); _tile_stored(1,t01,NT*4);
                _tile_stored(2,t10,NT*4); _tile_stored(3,t11,NT*4);
                int no0=2*nb*NT, no1=(2*nb+1)*NT;
                for(int m=0;m<MT;m++){
                    memcpy(acc+(int64_t)(s0+m)*O+no0,    t00+m*NT, NT*sizeof(int32_t));
                    memcpy(acc+(int64_t)(s0+m)*O+no1,    t01+m*NT, NT*sizeof(int32_t));
                    memcpy(acc+(int64_t)(s0+MT+m)*O+no0, t10+m*NT, NT*sizeof(int32_t));
                    memcpy(acc+(int64_t)(s0+MT+m)*O+no1, t11+m*NT, NT*sizeof(int32_t));
                }
            }
        }
        }

        /* --- ragged O edge (O/16 odd): leftover 16-col tile x ALL S 16-row tiles --- */
        if(nOt_rem){
            int no=2*nOb;
            const int8_t *Bp=Bp_all+(int64_t)no*I*NT;
            #pragma omp for schedule(static)
            for(int so=0; so<nSt; so++){
                int s0=so*MT;
                const int8_t *Arow=A+(int64_t)s0*I;
                _tile_zero(0);
                for(int ko=0; ko<nKt; ko++){
                    _tile_loadd(4, Arow+ko*KT, I);
                    _tile_loadd(6, Bp+(int64_t)ko*(KT/4)*NT*4, NT*4);
                    _tile_dpbssd(0,4,6);
                }
                int32_t tmp[16*16]; _tile_stored(0,tmp,NT*4);
                int mrows=(s0+MT<=S)?MT:(S-s0);
                for(int m=0;m<mrows;m++)
                    memcpy(acc+(int64_t)(s0+m)*O+no*NT, tmp+m*NT, NT*sizeof(int32_t));
            }
        }

        /* --- ragged S edge (S/16 odd): leftover 16-row tile x macro-block O cols
               only (the leftover-O corner, if any, was already covered above) --- */
        if(nSt_rem){
            int so=nSb*2, s0=so*MT;
            const int8_t *Arow=A+(int64_t)s0*I;
            #pragma omp for schedule(static)
            for(int no=0; no<2*nOb; no++){
                const int8_t *Bp=Bp_all+(int64_t)no*I*NT;
                _tile_zero(0);
                for(int ko=0; ko<nKt; ko++){
                    _tile_loadd(4, Arow+ko*KT, I);
                    _tile_loadd(6, Bp+(int64_t)ko*(KT/4)*NT*4, NT*4);
                    _tile_dpbssd(0,4,6);
                }
                int32_t tmp[16*16]; _tile_stored(0,tmp,NT*4);
                int mrows=(s0+MT<=S)?MT:(S-s0);
                for(int m=0;m<mrows;m++)
                    memcpy(acc+(int64_t)(s0+m)*O+no*NT, tmp+m*NT, NT*sizeof(int32_t));
            }
        }

        _tile_release();
    }
}

static int64_t checksum(const int32_t *a, int64_t n){
    int64_t s=0; for(int64_t i=0;i<n;i++) s+= (int64_t)a[i]*(1+(i&7)); return s;
}

int main(int argc, char**argv){
    int I = argc>1?atoi(argv[1]):4096;
    int O = argc>2?atoi(argv[2]):2048;
    int reps = argc>3?atoi(argv[3]):50;
    int Ss[] = {1,8,32,128};
    if(amx_enable()!=0){ fprintf(stderr,"AMX enable failed\n"); return 1; }

#ifdef USE_ONEDNN
    gemm_onednn_init();
#endif

    printf("GEMM C[S][O]=A[S][I].B^T[O][I]  I=%d O=%d  reps=%d\n", I,O,reps);
#ifdef USE_ONEDNN
    printf("%-6s %-11s %-11s %-11s %-11s %-11s %-9s %-9s %-11s %-9s %-8s %-8s\n",
           "S","vnni_GF","amx_GF","amx2x2_GF","amxPk_GF","onednn_GF","2x2/vnni","2x2/amx","amxPk/vnni","2x2/dnn%","match","dnn");
#else
    printf("%-6s %-11s %-11s %-11s %-11s %-9s %-9s %-11s %-8s\n",
           "S","vnni_GF","amx_GF","amx2x2_GF","amxPk_GF","2x2/vnni","2x2/amx","amxPk/vnni","match");
#endif

    int8_t *B = aligned_alloc(64,(size_t)O*I);
    int8_t *Bp= aligned_alloc(64,(size_t)O*I);
    for(int64_t i=0;i<(int64_t)O*I;i++) B[i]=(int8_t)((i*131+7)%255-127);
    prepack_B(Bp,B,I,O);
#ifdef USE_ONEDNN
    int8_t *BpDnn = aligned_alloc(64,(size_t)O*I);
    if(gemm_onednn_available) gemm_onednn_prepack_B(BpDnn,B,I,O);
#endif

    for(int si=0; si<4; si++){
        int S=Ss[si];
        int8_t *A=aligned_alloc(64,(size_t)S*I);
        for(int64_t i=0;i<(int64_t)S*I;i++) A[i]=(int8_t)((i*197+3)%255-127);
        int32_t *r0=malloc((size_t)S*O*sizeof(int32_t));
        int32_t *r1=malloc((size_t)S*O*sizeof(int32_t));
        int32_t *r2=malloc((size_t)S*O*sizeof(int32_t));
        int32_t *r3=malloc((size_t)S*O*sizeof(int32_t));
#ifdef USE_ONEDNN
        int32_t *r4=malloc((size_t)S*O*sizeof(int32_t));
#endif

        /* scalar once (correctness + timing on small S only) */
        double t=now_s(); gemm_scalar(r0,A,B,S<=8?S:8,I,O); double sc_ms=(now_s()-t)*1e3/(S<=8?S:8)*S;

        gemm_vnni(r1,A,B,S,I,O);
        /* AMX needs S multiple of 16; pad S up for the tiled path, compare only valid rows */
        int Sp = ((S+15)/16)*16;
        int8_t *Ap=aligned_alloc(64,(size_t)Sp*I); memset(Ap,0,(size_t)Sp*I);
        memcpy(Ap,A,(size_t)S*I);
        int32_t *r2p=malloc((size_t)Sp*O*sizeof(int32_t));
        int32_t *r3p=malloc((size_t)Sp*O*sizeof(int32_t));
        gemm_amx(r2p,Ap,Bp,Sp,I,O);
        memcpy(r2,r2p,(size_t)S*O*sizeof(int32_t));
        gemm_amx_2x2(r3p,Ap,Bp,Sp,I,O);
        memcpy(r3,r3p,(size_t)S*O*sizeof(int32_t));

        /* correctness: compare vnni & amx & amx2x2 vs scalar on the rows we computed scalar for */
        int scr=S<=8?S:8; int match=1;
        for(int64_t k=0;k<(int64_t)scr*O;k++){ if(r1[k]!=r0[k]||r2[k]!=r0[k]||r3[k]!=r0[k]){match=0;break;} }

#ifdef USE_ONEDNN
        /* oneDNN operates on the real (unpadded) A/S directly -- brgemm takes
         * a real per-call M, no Sp/Ap tile-multiple padding needed. Reported
         * separately from `match` since a layout/compensation mismatch here
         * shouldn't be conflated with our own kernels' correctness. */
        int dnn_match=0;
        if(gemm_onednn_available){
            gemm_onednn(r4,A,BpDnn,S,I,O);
            dnn_match=1;
            for(int64_t k=0;k<(int64_t)scr*O;k++){ if(r4[k]!=r0[k]){dnn_match=0;break;} }
        }
#endif

        double flops = 2.0*(double)S*O*I;
        t=now_s(); for(int r=0;r<reps;r++) gemm_vnni(r1,A,B,S,I,O); double vnni_ms=(now_s()-t)*1e3/reps;
        t=now_s(); for(int r=0;r<reps;r++) gemm_amx(r2p,Ap,Bp,Sp,I,O); double amx_ms=(now_s()-t)*1e3/reps;
        t=now_s(); for(int r=0;r<reps;r++) gemm_amx_2x2(r3p,Ap,Bp,Sp,I,O); double amx2_ms=(now_s()-t)*1e3/reps;
        t=now_s(); for(int r=0;r<reps;r++) gemm_amx_packB(r2p,Ap,B,Sp,I,O); double amxp_ms=(now_s()-t)*1e3/reps;
        double vnni_gf=flops/(vnni_ms*1e6), amx_gf=flops/(amx_ms*1e6), amx2_gf=flops/(amx2_ms*1e6), amxp_gf=flops/(amxp_ms*1e6);
#ifdef USE_ONEDNN
        double onednn_ms=0.0, onednn_gf=0.0;
        if(gemm_onednn_available){
            t=now_s(); for(int r=0;r<reps;r++) gemm_onednn(r4,A,BpDnn,S,I,O); onednn_ms=(now_s()-t)*1e3/reps;
            onednn_gf=flops/(onednn_ms*1e6);
        }
#endif

#ifdef USE_ONEDNN
        printf("%-6d %-11.1f %-11.1f %-11.1f %-11.1f %-11.1f %-9.2f %-9.2f %-11.2f %-9.2f %-8s %-8s\n",
               S, vnni_gf, amx_gf, amx2_gf, amxp_gf, onednn_gf,
               amx2_gf/vnni_gf, amx2_gf/amx_gf, amxp_gf/vnni_gf,
               onednn_gf>0.0 ? amx2_gf/onednn_gf*100.0 : 0.0,
               match?"ok":"MISMATCH",
               gemm_onednn_available ? (dnn_match?"ok":"MISMATCH") : "skip");
#else
        printf("%-6d %-11.1f %-11.1f %-11.1f %-11.1f %-9.2f %-9.2f %-11.2f %-8s\n",
               S, vnni_gf, amx_gf, amx2_gf, amxp_gf, amx2_gf/vnni_gf, amx2_gf/amx_gf, amxp_gf/vnni_gf, match?"ok":"MISMATCH");
#endif
        (void)checksum(r1,(int64_t)S*O);
        free(A);free(r0);free(r1);free(r2);free(r3);free(Ap);free(r2p);free(r3p);
#ifdef USE_ONEDNN
        free(r4);
#endif
    }
    free(B);free(Bp);
#ifdef USE_ONEDNN
    free(BpDnn);
#endif
    return 0;
}
