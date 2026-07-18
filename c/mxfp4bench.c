/* mxfp4bench.c — A/B microbench: colibri's current fmt=4 grouped-int4 GEMV (AVX2 FP)
 * vs the ported kt-kernel MXFP4 (E2M1 + E8M0) BF16-dot kernel (GemmKernel224MXFP4SmallKGroup).
 *
 * The two kernels solve the SAME MoE expert GEMM but with different weight formats:
 *   - int4 grouped (fmt=4): weight nibble in [-8,7], per-group f32 scale, gs=128, ACT f32.
 *   - MXFP4:                weight E2M1 nibble {0,.5,1,1.5,2,3,4,6}(+sign), per-group E8M0
 *                           power-of-2 scale, gs=32, ACT bf16.
 * So they are NOT bit-comparable; each is validated against ITS OWN f32 scalar reference.
 * We compare GF/s (compute throughput) at S = 1,8,32,128,642 on the two GLM shapes.
 *
 * Build: gcc -O3 -march=native -fopenmp -pthread mxfp4bench.c -o mxfp4bench -lm
 * (native on GNR gives AVX-512 + AVX512-BF16; the MXFP4 kernel uses _mm512_dpbf16_ps.)
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#include <immintrin.h>

static double now_s(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }

/* ---------------- E2M1 magnitude LUT (verified 1:1 vs ml_dtypes.float4_e2m1fn) ------------- */
static const float E2M1[16] = {0.0f,0.5f,1.0f,1.5f,2.0f,3.0f,4.0f,6.0f,
                               -0.0f,-0.5f,-1.0f,-1.5f,-2.0f,-3.0f,-4.0f,-6.0f};

/* ============================================================================
 * BASELINE: colibri's current fmt=4 grouped-int4 GEMV kernel (copied verbatim from
 * glm.c matmul_i4_grouped). Weight int4 nibble - 8, per-group f32 scale, f32 activations.
 * y[S,O] = x[S,I] @ W^T ; W packed 2 nibbles/byte, scale[o][g].
 * ============================================================================ */
static inline float hsum256(__m256 v){
    __m128 lo=_mm256_castps256_ps128(v), hi=_mm256_extractf128_ps(v,1); lo=_mm_add_ps(lo,hi);
    __m128 sh=_mm_movehdup_ps(lo); __m128 s=_mm_add_ps(lo,sh); sh=_mm_movehl_ps(sh,s); s=_mm_add_ss(s,sh);
    return _mm_cvtss_f32(s);
}
static void gemm_i4_grouped(float *y, const float *x, const uint8_t *q4, const float *scale,
                            int S, int I, int O, int gs){
    int rb=(I+1)/2; int ng=(I+gs-1)/gs;
    #pragma omp parallel for schedule(static)
    for(int o=0;o<O;o++){
        const uint8_t *w=q4+(int64_t)o*rb;
        const float *scl=scale+(int64_t)o*ng;
        for(int s=0;s<S;s++){
            const float *xs=x+(int64_t)s*I; float a=0;
            for(int g=0; g*gs<I; g++){
                int base=g*gs; int glen=gs; if(base+glen>I) glen=I-base;
                float sc=scl[g]; int i=base;
                const __m128i m4=_mm_set1_epi8(0x0F); const __m256i b8=_mm256_set1_epi32(8);
                __m256 acc=_mm256_setzero_ps();
                for(; i+16<=base+glen; i+=16){ __m128i by=_mm_loadl_epi64((const __m128i*)(w+(i>>1)));
                    __m128i lo=_mm_and_si128(by,m4),hi=_mm_and_si128(_mm_srli_epi16(by,4),m4);
                    __m128i nib=_mm_unpacklo_epi8(lo,hi);
                    __m256 w0=_mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_cvtepu8_epi32(nib),b8));
                    __m256 w1=_mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_srli_si128(nib,8)),b8));
                    acc=_mm256_fmadd_ps(_mm256_loadu_ps(xs+i),   w0, acc);
                    acc=_mm256_fmadd_ps(_mm256_loadu_ps(xs+i+8), w1, acc); }
                a+=hsum256(acc)*sc;
                for(; i<base+glen; i+=2){
                    if(i+1<base+glen){ uint8_t byte=w[i>>1];
                        a+=(xs[i]*(float)((int)(byte&0xF)-8)+xs[i+1]*(float)((int)(byte>>4)-8))*sc; }
                    else { uint8_t byte=w[i>>1]; a+=xs[i]*(float)((int)(byte&0xF)-8)*sc; }
                }
            }
            y[(int64_t)s*O+o]=a;
        }
    }
}

/* ============================================================================
 * MXFP4 kernel — ported from kt-kernel GemmKernel224MXFP4SmallKGroup (fp4-moe.hpp).
 * Weight: E2M1 nibble (low=even-K, high=odd-K), per-32-group E8M0 byte scale (2^(b-127)).
 * Activations: BF16. Dot via _mm512_dpbf16_ps. gs=32.
 * Layout matches converter: q4[o][I/2] nibbles, e8m0[o][I/32] bytes.
 * y[S,O] = x[S,I] @ W^T.  x given as bf16 rows (S x I).
 * ============================================================================ */
alignas(16) static const uint8_t fp4_bf16_lo[16] = {
    0x00,0x00,0x80,0xC0,0x00,0x40,0x80,0xC0, 0x00,0x00,0x80,0xC0,0x00,0x40,0x80,0xC0};
alignas(16) static const uint8_t fp4_bf16_hi[16] = {
    0x00,0x3F,0x3F,0x3F,0x40,0x40,0x40,0x40, 0x80,0xBF,0xBF,0xBF,0xC0,0xC0,0xC0,0xC0};

/* 16 packed FP4 bytes (32 vals) -> 32 BF16 in K/column order [lo0,hi0,lo1,hi1,...] */
static inline __m512i mxfp4_to_bf16_32(__m128i packed){
    __m128i m=_mm_set1_epi8(0x0F);
    __m128i lo=_mm_and_si128(packed,m), hi=_mm_and_si128(_mm_srli_epi16(packed,4),m);
    __m128i Ll=_mm_load_si128((__m128i*)fp4_bf16_lo), Lh=_mm_load_si128((__m128i*)fp4_bf16_hi);
    __m128i l_lo=_mm_shuffle_epi8(Ll,lo), l_hi=_mm_shuffle_epi8(Lh,lo);
    __m128i b0=_mm_unpacklo_epi8(l_lo,l_hi), b1=_mm_unpackhi_epi8(l_lo,l_hi);
    __m128i h_lo=_mm_shuffle_epi8(Ll,hi), h_hi=_mm_shuffle_epi8(Lh,hi);
    __m128i c0=_mm_unpacklo_epi8(h_lo,h_hi), c1=_mm_unpackhi_epi8(h_lo,h_hi);
    __m128i p0=_mm_unpacklo_epi16(b0,c0), p1=_mm_unpackhi_epi16(b0,c0);
    __m128i p2=_mm_unpacklo_epi16(b1,c1), p3=_mm_unpackhi_epi16(b1,c1);
    __m256i q0=_mm256_inserti128_si256(_mm256_castsi128_si256(p0),p1,1);
    __m256i q1=_mm256_inserti128_si256(_mm256_castsi128_si256(p2),p3,1);
    return _mm512_inserti64x4(_mm512_castsi256_si512(q0),q1,1);
}

static inline float bench_hsum(__m512 v){ return _mm512_reduce_add_ps(v); }

/* mat-vec inner: one token, N rows 4-at-a-time. Used as the S=1 path and M-tail. */
static inline void mxfp4_row_novec(float *yrow, const uint16_t *ar, const uint8_t *q4,
                                   const float *scltab, int kg, int I, int O, int o0, int o1){
    const int rb=I/2, ng=I/32;
    int o=o0;
    for(; o+4<=o1; o+=4){
        const uint8_t *w0=q4+(int64_t)(o+0)*rb,*w1=q4+(int64_t)(o+1)*rb,*w2=q4+(int64_t)(o+2)*rb,*w3=q4+(int64_t)(o+3)*rb;
        const float *s0=scltab+(int64_t)(o+0)*ng,*s1=scltab+(int64_t)(o+1)*ng,*s2=scltab+(int64_t)(o+2)*ng,*s3=scltab+(int64_t)(o+3)*ng;
        __m512 a0=_mm512_setzero_ps(),a1=_mm512_setzero_ps(),a2=_mm512_setzero_ps(),a3=_mm512_setzero_ps();
        for(int g=0;g<kg;g++){
            __m512bh av=(__m512bh)_mm512_loadu_si512((const void*)(ar+g*32));
            a0=_mm512_fmadd_ps(_mm512_set1_ps(s0[g]),_mm512_dpbf16_ps(_mm512_setzero_ps(),av,(__m512bh)mxfp4_to_bf16_32(_mm_loadu_si128((const __m128i*)(w0+g*16)))),a0);
            a1=_mm512_fmadd_ps(_mm512_set1_ps(s1[g]),_mm512_dpbf16_ps(_mm512_setzero_ps(),av,(__m512bh)mxfp4_to_bf16_32(_mm_loadu_si128((const __m128i*)(w1+g*16)))),a1);
            a2=_mm512_fmadd_ps(_mm512_set1_ps(s2[g]),_mm512_dpbf16_ps(_mm512_setzero_ps(),av,(__m512bh)mxfp4_to_bf16_32(_mm_loadu_si128((const __m128i*)(w2+g*16)))),a2);
            a3=_mm512_fmadd_ps(_mm512_set1_ps(s3[g]),_mm512_dpbf16_ps(_mm512_setzero_ps(),av,(__m512bh)mxfp4_to_bf16_32(_mm_loadu_si128((const __m128i*)(w3+g*16)))),a3);
        }
        yrow[o+0]=bench_hsum(a0); yrow[o+1]=bench_hsum(a1); yrow[o+2]=bench_hsum(a2); yrow[o+3]=bench_hsum(a3);
    }
    for(; o<o1; o++){ const uint8_t*w=q4+(int64_t)o*rb; const float*s=scltab+(int64_t)o*ng; __m512 acc=_mm512_setzero_ps();
        for(int g=0;g<kg;g++){ __m512bh av=(__m512bh)_mm512_loadu_si512((const void*)(ar+g*32));
            acc=_mm512_fmadd_ps(_mm512_set1_ps(s[g]),_mm512_dpbf16_ps(_mm512_setzero_ps(),av,(__m512bh)mxfp4_to_bf16_32(_mm_loadu_si128((const __m128i*)(w+g*16)))),acc); }
        yrow[o]=bench_hsum(acc); }
}

/* full kernel: 4x4 register-blocked mat_mat (MB tokens x NB rows) — decodes each weight
 * row's FP4->BF16 ONCE and reuses across MB tokens (the "224" amortization). M/N tails -> mat_vec. */
static void gemm_mxfp4(float *y, const uint16_t *xbf16, const uint8_t *q4, const uint8_t *e8m0,
                       int S, int I, int O){
    const int kg=I/32, rb=I/2, ng=I/32;
    #pragma omp parallel
    {
      /* per-thread decoded-scale table for the O rows this thread owns is rebuilt inline;
       * to keep it simple we compute scale on the fly from e8m0 per (o,g). */
      #pragma omp for schedule(static)
      for(int o4=0;o4<O;o4+=4){
        int nb = (o4+4<=O)?4:(O-o4);
        /* precompute decoded scales for these up-to-4 rows */
        static _Thread_local float scl[4][8192/32];
        for(int j=0;j<nb;j++){ const uint8_t*e=e8m0+(int64_t)(o4+j)*ng; for(int g=0;g<kg;g++) scl[j][g]=ldexpf(1.0f,(int)e[g]-127); }
        const uint8_t *w[4]; for(int j=0;j<nb;j++) w[j]=q4+(int64_t)(o4+j)*rb;
        int s=0;
        if(nb==4){
          for(; s+4<=S; s+=4){
            const uint16_t *a0=xbf16+(int64_t)(s+0)*I,*a1=xbf16+(int64_t)(s+1)*I,*a2=xbf16+(int64_t)(s+2)*I,*a3=xbf16+(int64_t)(s+3)*I;
            __m512 acc[4][4]; for(int i=0;i<4;i++)for(int j=0;j<4;j++) acc[i][j]=_mm512_setzero_ps();
            for(int g=0;g<kg;g++){
              __m512bh d0=(__m512bh)mxfp4_to_bf16_32(_mm_loadu_si128((const __m128i*)(w[0]+g*16)));
              __m512bh d1=(__m512bh)mxfp4_to_bf16_32(_mm_loadu_si128((const __m128i*)(w[1]+g*16)));
              __m512bh d2=(__m512bh)mxfp4_to_bf16_32(_mm_loadu_si128((const __m128i*)(w[2]+g*16)));
              __m512bh d3=(__m512bh)mxfp4_to_bf16_32(_mm_loadu_si128((const __m128i*)(w[3]+g*16)));
              const uint16_t *ar[4]={a0,a1,a2,a3};
              for(int i=0;i<4;i++){ __m512bh av=(__m512bh)_mm512_loadu_si512((const void*)(ar[i]+g*32));
                acc[i][0]=_mm512_fmadd_ps(_mm512_set1_ps(scl[0][g]),_mm512_dpbf16_ps(_mm512_setzero_ps(),av,d0),acc[i][0]);
                acc[i][1]=_mm512_fmadd_ps(_mm512_set1_ps(scl[1][g]),_mm512_dpbf16_ps(_mm512_setzero_ps(),av,d1),acc[i][1]);
                acc[i][2]=_mm512_fmadd_ps(_mm512_set1_ps(scl[2][g]),_mm512_dpbf16_ps(_mm512_setzero_ps(),av,d2),acc[i][2]);
                acc[i][3]=_mm512_fmadd_ps(_mm512_set1_ps(scl[3][g]),_mm512_dpbf16_ps(_mm512_setzero_ps(),av,d3),acc[i][3]); }
            }
            for(int i=0;i<4;i++){ float*yr=y+(int64_t)(s+i)*O+o4;
              yr[0]=bench_hsum(acc[i][0]); yr[1]=bench_hsum(acc[i][1]); yr[2]=bench_hsum(acc[i][2]); yr[3]=bench_hsum(acc[i][3]); }
          }
        }
        /* S tail (or nb<4): per-token mat_vec inner over these rows */
        for(; s<S; s++){
          /* rebuild a scltab view for mxfp4_row_novec covering [o4,o4+nb) — reuse scl[][] */
          const uint16_t *ar=xbf16+(int64_t)s*I; float *yr=y+(int64_t)s*O;
          for(int j=0;j<nb;j++){ const uint8_t*ww=w[j]; __m512 acc=_mm512_setzero_ps();
            for(int g=0;g<kg;g++){ __m512bh av=(__m512bh)_mm512_loadu_si512((const void*)(ar+g*32));
              acc=_mm512_fmadd_ps(_mm512_set1_ps(scl[j][g]),_mm512_dpbf16_ps(_mm512_setzero_ps(),av,(__m512bh)mxfp4_to_bf16_32(_mm_loadu_si128((const __m128i*)(ww+g*16)))),acc); }
            yr[o4+j]=bench_hsum(acc); }
        }
      }
    }
    (void)mxfp4_row_novec;
}

/* ---------------- scalar references (each format its own oracle) ---------------- */
static void ref_i4(float *y,const float*x,const uint8_t*q4,const float*scale,int S,int I,int O,int gs){
    int rb=(I+1)/2,ng=(I+gs-1)/gs;
    for(int o=0;o<O;o++)for(int s=0;s<S;s++){ double a=0;
        for(int i=0;i<I;i++){ uint8_t by=q4[(int64_t)o*rb+(i>>1)]; int nb=(i&1)?(by>>4):(by&0xF);
            a+=(double)x[(int64_t)s*I+i]*(double)(nb-8)*(double)scale[(int64_t)o*ng+i/gs]; }
        y[(int64_t)s*O+o]=(float)a; }
}
static float bf16f(uint16_t b){ uint32_t u=((uint32_t)b)<<16; float f; memcpy(&f,&u,4); return f; }
static void ref_mxfp4(float *y,const uint16_t*xb,const uint8_t*q4,const uint8_t*e8,int S,int I,int O){
    int rb=I/2,ng=I/32;
    for(int o=0;o<O;o++)for(int s=0;s<S;s++){ double a=0;
        for(int i=0;i<I;i++){ uint8_t by=q4[(int64_t)o*rb+(i>>1)]; int nb=(i&1)?(by>>4):(by&0xF);
            float sc=ldexpf(1.0f,(int)e8[(int64_t)o*ng+i/32]-127);
            a+=(double)bf16f(xb[(int64_t)s*I+i])*(double)E2M1[nb]*(double)sc; }
        y[(int64_t)s*O+o]=(float)a; }
}
static double relerr(const float*a,const float*b,int64_t n){ double e=0,m=0; for(int64_t i=0;i<n;i++){ e+=fabs(a[i]-b[i]); m+=fabs(b[i]); } return m>0?e/m:0; }

static uint16_t f2bf16(float f){ uint32_t u; memcpy(&u,&f,4); return (uint16_t)((u+0x8000)>>16); }

int main(int argc,char**argv){
    int I = argc>1?atoi(argv[1]):6144;   /* GLM gate/up: I=6144 O=2048 ; down: I=2048 O=6144 */
    int O = argc>2?atoi(argv[2]):2048;
    int reps = argc>3?atoi(argv[3]):50;
    int Ss[]={16,20,24,32}; int nS=4;
    srand(1234);
    /* build random int4 + mxfp4 weights of matching shape (I mult of 32) */
    int rb=I/2, ng128=(I+127)/128, ng32=I/32;
    uint8_t *q4=malloc((int64_t)O*rb); uint8_t *mx=malloc((int64_t)O*rb);
    float *sc128=malloc((int64_t)O*ng128*sizeof(float)); uint8_t *e8=malloc((int64_t)O*ng32);
    for(int64_t i=0;i<(int64_t)O*rb;i++){ q4[i]=rand()&0xFF; mx[i]=rand()&0xFF; }
    for(int64_t i=0;i<(int64_t)O*ng128;i++) sc128[i]=0.01f+0.001f*(rand()%50);
    for(int64_t i=0;i<(int64_t)O*ng32;i++) e8[i]=120+(rand()%12);   /* ~2^-7..2^4 */
    int Smax=642;
    float *xf=malloc((int64_t)Smax*I*sizeof(float)); uint16_t *xb=malloc((int64_t)Smax*I*sizeof(uint16_t));
    for(int64_t i=0;i<(int64_t)Smax*I;i++){ float v=((rand()%2001)-1000)/1000.0f; xf[i]=v; xb[i]=f2bf16(v); }
    float *y1=malloc((int64_t)Smax*O*sizeof(float)),*y2=malloc((int64_t)Smax*O*sizeof(float));
    float *r1=malloc((int64_t)8*O*sizeof(float)),*r2=malloc((int64_t)8*O*sizeof(float));

    printf("# I=%d O=%d reps=%d threads=%d\n",I,O,reps,omp_get_max_threads());
    /* validate both kernels vs their own scalar refs at S=4 */
    gemm_i4_grouped(y1,xf,q4,sc128,4,I,O,128); ref_i4(r1,xf,q4,sc128,4,I,O,128);
    gemm_mxfp4(y2,xb,mx,e8,4,I,O); ref_mxfp4(r2,xb,mx,e8,4,I,O);
    printf("# validate: i4 relerr=%.2e  mxfp4 relerr=%.2e\n", relerr(y1,r1,4*O), relerr(y2,r2,4*O));

    printf("%-6s %12s %12s %10s\n","S","i4grp_GF","mxfp4_GF","mxfp4/i4");
    for(int si=0;si<nS;si++){ int S=Ss[si];
        double flops=2.0*S*O*I;
        double t=now_s(); for(int r=0;r<reps;r++) gemm_i4_grouped(y1,xf,q4,sc128,S,I,O,128); double i4=(now_s()-t)*1e3/reps;
        t=now_s(); for(int r=0;r<reps;r++) gemm_mxfp4(y2,xb,mx,e8,S,I,O); double mxf=(now_s()-t)*1e3/reps;
        double g1=flops/(i4*1e6), g2=flops/(mxf*1e6);
        printf("%-6d %12.1f %12.1f %9.2fx\n",S,g1,g2,g2/g1);
    }
    return 0;
}
