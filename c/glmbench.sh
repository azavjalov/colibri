#!/bin/sh
# args: AMXVAL(0/1) label
cd /home/intel/b70-sglang-xpu/colibri/c
export SNAP=/srv/models/glm52-i4
export PROMPT="Explain in three sentences why the sky is blue."
export NGEN=64
export CTX=4096
export AMX=$1
export OMP_NUM_THREADS=43
export OMP_PROC_BIND=close
export OMP_PLACES=cores
echo "=== RUN label=$2 AMX=$1 $(date) ==="
numactl --cpunodebind=0 --membind=0 ./glm 237 2>&1
echo "=== END label=$2 rc=$? $(date) ==="
