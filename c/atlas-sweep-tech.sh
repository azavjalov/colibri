#!/bin/sh
# Expert-atlas sweep driven by the glmrt `glm` binary (MXFP4 model), for the tech-domain probe set.
# Controls the 4 atlas confounds: TOPP=0, MTP=0, DRAFT=0, greedy TEMP=0, and STATS dumped per-run
# from an empty routing history (glm's eusage starts fresh each process; no .coli_usage persistence
# is used here since RESIDENT mode doesn't write one, but we pass a throwaway path to be safe).
# Uses the validated single-stream recipe (1 NUMA node, no 3-pass churn).
set -u
cd /home/intel/b70-sglang-xpu/glmrt/c
PROBES=probes-tech.json
OUT=atlas-tech
NGEN=64
mkdir -p "$OUT/stats"

# expand probes.json -> runlist (cat, idx, prompt)
/home/intel/b70-sglang-xpu/.venv/bin/python - "$PROBES" > "$OUT/runlist.tsv" <<'PY'
import json,sys
for cat,prompts in json.load(open(sys.argv[1])).items():
    if cat.startswith('_'): continue
    for i,p in enumerate(prompts): print(f"{cat}\t{i}\t{p}")
PY

n=$(wc -l < "$OUT/runlist.tsv"); i=0
echo "$n probes -> $OUT/stats"
while IFS=$(printf '\t') read -r cat idx prompt; do
  i=$((i+1))
  dst="$OUT/stats/${cat}_${idx}.txt"
  [ -s "$dst" ] && { echo "  [$i/$n] $cat/$idx (cached)"; continue; }
  SNAP=/srv/models/glm52-mxfp4 PROMPT="$prompt" NGEN=$NGEN CTX=4096 TEMP=0 \
  AMX=1 COLI_MMAP=1 RESIDENT=1 NUMA_PARTITION=0 NUMA_COMPUTE=0 COLI_NO_OMP_TUNE=1 MTP=0 DRAFT=0 TOPP=0 \
  STATS="$dst" OMP_NUM_THREADS=40 numactl --cpunodebind=0 \
    ./glm 256 > "$OUT/stats/${cat}_${idx}.log" 2>&1
  sel=$(grep -aoE '[0-9]+ selections across [0-9]+ distinct experts' "$OUT/stats/${cat}_${idx}.log" | head -1)
  echo "  [$i/$n] $cat/$idx  $sel"
done < "$OUT/runlist.tsv"
echo "=== SWEEP DONE ==="
