/* amx_idot_selftest.c -- standalone validation of the Phase 1 AMX int8 tile
 * GEMM path added to glm.c: amx_prepack_q8/amx_prepack_i4 + matmul_q_idot_mm_amx,
 * and their wiring into matmul_qt_ex's dispatch. Modeled on tests/test_idot.c's
 * "#define main / #include glm.c / #undef main" trick to reach glm.c's static
 * functions and globals directly. NOT part of the Makefile test-c suite --
 * ad hoc, like amxbench.c/iobench.c. Run manually:
 *
 *   gcc -O3 -march=native -fopenmp -pthread -Wall -Wextra -Wno-unused-parameter \
 *       -Wno-misleading-indentation -Wno-unused-function \
 *       amx_idot_selftest.c -o amx_idot_selftest -lm -fopenmp -pthread
 *   ./amx_idot_selftest
 *
 * Two independent checks, for both fmt=1 (int8) and fmt=2 (int4) tensors built
 * with amx_eligible=1 (i.e. shaped as if they came from qt_load):
 *
 *  1. check_kernel_exact: calls matmul_q_idot_mm_amx directly (after
 *     amx_prepack_q8/_i4) and compares its int32 dot product / final float
 *     against a plain scalar int64 reference -- the same "match=ok" contract
 *     amxbench.c already validated. S is restricted to multiples of 16 here
 *     (matmul_q_idot_mm_amx's documented contract -- the ragged S tail is the
 *     CALLER's job in matmul_qt_ex, exercised separately below).
 *
 *  2. check_dispatch_matches_vnni: calls the full matmul_qt_ex dispatch twice
 *     on the same tensor and input, once with g_amx=0 (pure VNNI/scalar-idot
 *     path, today's behavior) and once with g_amx=1 (AMX path, including the
 *     ragged S-tail VNNI fallback and the O%16!=0/I%64!=0 ineligible-shape
 *     fallback), and requires the output floats to be bit-for-bit identical.
 *     S values include 1 and values below g_amx_smin (must stay on VNNI) and
 *     non-multiples of 16 (must hit the tail fallback).
 */
#define main coli_glm_main_unused
#include "glm.c"
#undef main

#include <stdio.h>

static uint32_t rng_state=0xC0FFEEu;
static uint32_t xr(void){ rng_state^=rng_state<<13; rng_state^=rng_state>>17; rng_state^=rng_state<<5; return rng_state; }

static int32_t ref_dot_i8(const int8_t *w, const int8_t *x, int I){
    int64_t s=0; for(int i=0;i<I;i++) s+=(int32_t)w[i]*x[i]; return (int32_t)s;
}

/* Build a resident-shaped QT (amx_eligible=1, as if it came from qt_load). */
static void fill_qt_resident(QT *w, int fmt, int O, int I){
    memset(w,0,sizeof *w);
    w->fmt=fmt; w->O=O; w->I=I; w->amx_eligible=1;
    w->s=malloc((size_t)O*sizeof(float));
    for(int o=0;o<O;o++) w->s[o]=0.0005f+(float)(xr()%2000)*1e-6f;
    if(fmt==1){
        w->q8=malloc((size_t)O*I);
        for(int64_t i=0;i<(int64_t)O*I;i++) w->q8[i]=(int8_t)((int)(xr()%256)-128);
    } else {
        size_t nb=(size_t)O*((I+1)/2);
        w->q4=malloc(nb);
        for(size_t i=0;i<nb;i++) w->q4[i]=(uint8_t)(xr()&0xFF);
    }
}
static void free_qt(QT *w){
    free(w->s); free(w->q8); free(w->q4); if(w->amx_q8) free(w->amx_q8);
}

/* ---- 1. Direct kernel check: matmul_q_idot_mm_amx's int32 dot vs scalar.
 * S must be a multiple of 16 (the function's documented contract). */
static int check_kernel_exact(int fmt, int O, int I, int S){
    QT w; fill_qt_resident(&w,fmt,O,I);
    if(fmt==1) amx_prepack_q8(&w); else amx_prepack_i4(&w);
    if(w.amx_packed!=1){
        fprintf(stderr,"SKIP kernel fmt=%d O=%d I=%d: not AMX-packable (ragged shape)\n",fmt,O,I);
        free_qt(&w); return 0;
    }
    int8_t *xq=malloc((size_t)S*I);
    for(int64_t i=0;i<(int64_t)S*I;i++) xq[i]=(int8_t)((int)(xr()%255)-127);
    float *sx=malloc((size_t)S*sizeof(float));
    for(int s=0;s<S;s++) sx[s]=0.25f+(float)(xr()%100)*0.01f;   /* nontrivial per-row scale too */
    float *y=malloc((size_t)S*O*sizeof(float));
    memset(y,0,(size_t)S*O*sizeof(float));

    matmul_q_idot_mm_amx(y,xq,sx,w.amx_q8,w.s,S,I,O);

    int rc=0;
    int8_t *wrow=malloc((size_t)I);
    for(int o=0;o<O && !rc;o++){
        if(fmt==1) memcpy(wrow, w.q8+(int64_t)o*I, I);
        else{   /* reconstruct the raw int8 row the same way amx_prepack_i4 did */
            int rb=(I+1)/2; const uint8_t *row4=w.q4+(int64_t)o*rb; int i=0;
            for(;i+1<I;i+=2){ uint8_t b=row4[i>>1];
                wrow[i]=(int8_t)((int)(b&0xF)-8); wrow[i+1]=(int8_t)((int)(b>>4)-8); }
            if(i<I){ uint8_t b=row4[i>>1]; wrow[i]=(int8_t)((int)(b&0xF)-8); }
        }
        for(int s=0;s<S;s++){
            int32_t want=ref_dot_i8(wrow, xq+(int64_t)s*I, I);
            float wantf=(float)want*w.s[o]*sx[s];
            if(memcmp(&y[(int64_t)s*O+o],&wantf,sizeof(float))!=0){
                fprintf(stderr,"FAIL kernel fmt=%d O=%d I=%d S=%d o=%d s=%d: got %.9g want %.9g (dot=%d)\n",
                        fmt,O,I,S,o,s,(double)y[(int64_t)s*O+o],(double)wantf,want);
                rc=1; break;
            }
        }
    }
    free(wrow); free(xq); free(sx); free(y); free_qt(&w);
    return rc;
}

/* ---- 2. Integrated dispatch check: matmul_qt_ex AMX-on vs AMX-off. S may be
 * anything (S=1, S<g_amx_smin, non-multiple-of-16, ragged O/I all included by
 * the caller) -- matmul_qt_ex must produce IDENTICAL floats either way. */
static int check_dispatch_matches_vnni(int fmt, int O, int I, int S){
    QT w; fill_qt_resident(&w,fmt,O,I);
    float *x=malloc((size_t)S*I*sizeof(float));
    for(int64_t i=0;i<(int64_t)S*I;i++) x[i]=((float)(xr()%4001)-2000.f)/500.f;
    float *y_amx=malloc((size_t)S*O*sizeof(float));
    float *y_vnni=malloc((size_t)S*O*sizeof(float));

    int saved_amx=g_amx, saved_smin=g_amx_smin;
    g_amx=0; matmul_qt_ex(y_vnni,x,&w,S,1);            /* pure VNNI/scalar-idot reference */
    g_amx=1; g_amx_smin=16; matmul_qt_ex(y_amx,x,&w,S,1); /* AMX path (fresh QT: amx_packed was 0) */
    g_amx=saved_amx; g_amx_smin=saved_smin;

    int rc=0;
    for(int64_t i=0;i<(int64_t)S*O && !rc;i++)
        if(memcmp(&y_amx[i],&y_vnni[i],sizeof(float))!=0){
            fprintf(stderr,"FAIL dispatch fmt=%d O=%d I=%d S=%d idx=%lld: amx=%.9g vnni=%.9g\n",
                    fmt,O,I,S,(long long)i,(double)y_amx[i],(double)y_vnni[i]);
            rc=1;
        }
    free(x); free(y_amx); free(y_vnni); free_qt(&w);
    return rc;
}

int main(void){
#if !defined(__AMX_INT8__) || !defined(__linux__)
    printf("amx_idot_selftest: built without __AMX_INT8__/__linux__ -- nothing to test "
           "(rebuild with -march=native on a GNR+ x86 Linux host)\n");
    return 0;
#else
    if(amx_enable()!=0){ fprintf(stderr,"amx_enable failed -- no AMX tile permission on this kernel/CPU\n"); return 1; }
    g_amx=1; g_amx_smin=16;

    static const int shapes[][2]={ {256,512}, {128,256}, {16,64}, {512,1024} };
    static const int Ss_kernel[]={16,32,48,64,128};
    static const int Ss_dispatch[]={1,8,15,16,17,32,37,64,100,128};
    int fails=0, ran=0;

    for(unsigned sh=0; sh<sizeof(shapes)/sizeof(shapes[0]); sh++){
        int O=shapes[sh][0], I=shapes[sh][1];
        for(int fmt=1; fmt<=2; fmt++){
            for(unsigned si=0; si<sizeof(Ss_kernel)/sizeof(Ss_kernel[0]); si++){
                fails+=check_kernel_exact(fmt,O,I,Ss_kernel[si]); ran++;
            }
            for(unsigned si=0; si<sizeof(Ss_dispatch)/sizeof(Ss_dispatch[0]); si++){
                fails+=check_dispatch_matches_vnni(fmt,O,I,Ss_dispatch[si]); ran++;
            }
        }
    }
    /* Ragged shapes: O%16!=0 and, separately, I%64!=0 -- amx_prepack_* must
     * cleanly mark them ineligible (amx_packed=-1) and matmul_qt_ex must fall
     * back to plain VNNI, still bit-exact vs the AMX-off reference. */
    static const int ragged[][2]={ {200,512}, {256,500}, {201,513} };
    for(unsigned sh=0; sh<sizeof(ragged)/sizeof(ragged[0]); sh++)
        for(int fmt=1; fmt<=2; fmt++){
            fails+=check_dispatch_matches_vnni(fmt, ragged[sh][0], ragged[sh][1], 64); ran++;
        }

    if(fails){ fprintf(stderr,"amx_idot_selftest: %d/%d checks FAILED\n",fails,ran); return 1; }
    printf("amx_idot_selftest: %d checks, all bit-exact vs scalar/VNNI (ok)\n",ran);
    return 0;
#endif
}
