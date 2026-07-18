#!/bin/sh
# GLM-5.2 MXFP4 single-stream launch recipe (the validated fast config).
# MXFP4 experts (fmt=5) run the ported kt-kernel dpbf16 kernel. Single-stream winner:
# ONE NUMA node (40 phys cores), NO NUMA 3-pass (its per-block rebind churn is ~10x slower
# for the fork-heavy per-expert MXFP4 kernel). Beats int4 on decode (2.99 vs 2.25 tok/s),
# prefill (12s vs 16.6s expert-matmul @ S=642) and accuracy (10.7% vs 12.8% relerr).
#
# Usage: ./run-mxfp4.sh "your prompt"   [NGEN default 64]
set -e
cd /home/intel/b70-sglang-xpu/glmrt/c
PROMPT="${1:-The capital of France is}"
NGEN="${NGEN:-64}"
SNAP=/srv/models/glm52-mxfp4 \
PROMPT="$PROMPT" NGEN="$NGEN" CTX="${CTX:-4096}" TEMP="${TEMP:-0}" \
AMX=1 COLI_MMAP=1 RESIDENT=1 \
NUMA_PARTITION=0 NUMA_COMPUTE=0 COLI_NO_OMP_TUNE=1 MTP=0 \
OMP_NUM_THREADS=40 OMP_PROC_BIND=close OMP_PLACES=cores \
numactl --cpunodebind=0 ./glm 256
