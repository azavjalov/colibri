#!/usr/bin/env bash
# proof_mlp.sh <SNAP> <NTH> [tag]
# Decode-ONLY (perf --control=fifo fence) proof of memory-LATENCY-bound-with-low-MLP.
# Measures: (1) MLP = avg outstanding L1D-miss loads in flight (l1d_pend_miss.pending
#           / pending_cycles) + fill-buffer-full pressure;
#           (2) load-latency breakdown = mem_load_retired.* histogram (L1/FB/L2/L3/DRAM)
#           => proves WHERE stalled loads resolve (near-cache vs DRAM);
#           (3) offcore outstanding demand reads (L2-miss MLP).
# Run under sudo (uncore/offcore need root). Box must be idle.
set -u
SNAP="${1:?usage: proof_mlp.sh <SNAP> <NTH> [tag]}"
NTH="${2:?need NTH}"
TAG="${3:-proof}"
CD=/home/intel/b70-sglang-xpu/glmrt/c
BIN=./glm
OUT=/tmp/proof_${TAG}
CTL=/tmp/proof_${TAG}.ctl
ACKF=/tmp/proof_${TAG}.ack

# Common glm env: deterministic decode replay, resident, mmap, no OMP self-reexec so
# perf attaches the real process, fifo fence so counters cover ONLY the 80-tok decode.
COMMON="SNAP=$SNAP CTX=4096 TEMP=0 AMX=1 COLI_MMAP=1 RESIDENT=1 NUMA_PARTITION=1 \
NUMA_COMPUTE=0 COLI_NO_OMP_TUNE=1 COLI_OMP_TUNED=1 MTP=0 \
OMP_NUM_THREADS=$NTH OMP_PROC_BIND=close OMP_PLACES=cores \
REPLAY=1 REF=ref_bench.json REF_FORCE=1"

# 3 event groups (multiplexed; each ~<=8 core events to avoid PMU overflow):
# G1 MLP + fill buffers + cycles baseline
G1="cycles,instructions,l1d_pend_miss.pending,l1d_pend_miss.pending_cycles,l1d_pend_miss.fb_full,l1d_pend_miss.fb_full_periods"
# G2 load-latency histogram: where do retired loads resolve?
G2="mem_load_retired.l1_hit,mem_load_retired.fb_hit,mem_load_retired.l2_hit,mem_load_retired.l2_miss,mem_load_retired.l3_hit,mem_load_retired.l3_miss"
# G3 offcore outstanding demand-data-reads (L2-miss MLP) + hierarchical stall cycles
G3="cycles,offcore_requests_outstanding.cycles_with_demand_data_rd,offcore_requests_outstanding.demand_data_rd,memory_activity.stalls_l2_miss,memory_activity.stalls_l3_miss"

rm -f "$OUT".g1.perf "$OUT".g2.perf "$OUT".g3.perf "$OUT".done "$CTL" "$ACKF"
RUN=/tmp/proof_${TAG}_run.sh
cat > "$RUN" <<RUNNER
#!/usr/bin/env bash
set -u
cd $CD
run_one(){
  local grp="\$1" evs="\$2"
  rm -f "$CTL" "$ACKF"; mkfifo "$CTL" "$ACKF"
  env $COMMON PERF_FIFO="$CTL,$ACKF" \
    perf stat --delay=-1 --control=fifo:"$CTL","$ACKF" -e "\$evs" \
    numactl --interleave=all $BIN 256 > "$OUT".\$grp.out 2> "$OUT".\$grp.perf
  rm -f "$CTL" "$ACKF"
}
run_one g1 "$G1"
run_one g2 "$G2"
run_one g3 "$G3"
echo ALL_DONE > "$OUT".done
RUNNER
chmod +x "$RUN"
nohup bash "$RUN" >/tmp/proof_${TAG}_launch.log 2>&1 &
echo "launched proof_mlp $TAG pid $! ; poll $OUT.done ; groups g1(MLP) g2(latency-hist) g3(offcore)"
