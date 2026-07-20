#!/usr/bin/env bash
# perf_decode_probe.sh — settle "core/compute-bound vs memory-stall vs cross-die" for glmrt
# decode with HW counters. Requires root (uncore). Run under ssh_exec sudo:true on a CLEAN box.
#
# Writes a self-contained runner to /tmp and nohup's it (survives the ssh timeout);
# poll /tmp/perf_<tag>.done for ALL_DONE. Blocks:
#   A) TMA top-down + core PMU, real baseline (interleave=all @ NTH)  -> core vs mem bound
#   B) uncore IMC CAS (sch0/sch1 sum) -> actual DRAM GB/s during decode
#   C) single-die-confined (cpunodebind=0 membind=0) -> cross-die penalty vs A
#   D) thread-scaling sweep at N=1 -> per-thread IPC + tok/s vs thread count
#      (disentangles over-threading/barrier dilution from a real execution wall)
#
# GNR-AP notes: no cycle_activity.stalls_mem_any; IMC CAS split per sub-channel
# (uncore_imc/cas_count_{read,write}_sch{0,1}/); TMA metric groups available.
#
# Usage: perf_decode_probe.sh <SNAP_dir> <OMP_THREADS> [tag]   (SWEEP env overrides D list)
set -u
SNAP="${1:?need model dir}"; NTH="${2:?need thread count}"; TAG="${3:-decode}"
BIN=/home/intel/b70-sglang-xpu/glmrt/c/glm
CD=/home/intel/b70-sglang-xpu/glmrt/c
OUT=/tmp/perf_${TAG}
SWEEP="${SWEEP:-1 4 8 16 21 42}"
CPN=$(( $(lscpu | awk -F: '/^Core\(s\) per socket/{gsub(/ /,"",$2);print $2}') \
       / $(numactl --hardware | awk '/available:/{print $2}') ))

# CAS sub-channel event list across every IMC PMU.
EVB=""
for p in $(ls -d /sys/bus/event_source/devices/uncore_imc_* 2>/dev/null | xargs -n1 basename); do
  for e in cas_count_read_sch0 cas_count_read_sch1 cas_count_write_sch0 cas_count_write_sch1; do
    [ -f "/sys/bus/event_source/devices/$p/events/$e" ] && EVB="$EVB$p/$e/,"
  done
done
EVB="${EVB%,}"

# COLI_OMP_TUNED=1 => glm does NOT re-exec (perf must attach to the real process).
COMMON="TEMP=0 AMX=1 COLI_MMAP=1 RESIDENT=1 NUMA_PARTITION=1 NUMA_COMPUTE=0 COLI_NO_OMP_TUNE=1 COLI_OMP_TUNED=1 MTP=0 OMP_WAIT_POLICY=active OMP_PROC_BIND=close OMP_PLACES=cores REPLAY=1 REF=ref_bench.json REF_FORCE=1 INSTR=1"
COREV="cycles,instructions,cycle_activity.stalls_total,cycle_activity.stalls_l3_miss,cycle_activity.stalls_l2_miss,cycle_activity.stalls_l1d_miss,exe_activity.bound_on_loads,exe.amx_busy,uops_executed.thread,uops_executed.stalls,fp_arith_dispatched.port_0,fp_arith_dispatched.port_1,fp_arith_dispatched.port_5,mem_load_retired.l3_miss,mem_load_retired.l2_miss,LLC-load-misses,LLC-loads"
TMAM="tma_retiring,tma_frontend_bound,tma_backend_bound,tma_core_bound,tma_memory_bound"

# Generate a self-contained runner (no nested escaping) and launch it detached.
RUN=/tmp/perf_${TAG}_run.sh
cat > "$RUN" <<RUNNER
#!/usr/bin/env bash
set -u
cd "$CD" || exit 1
export $COMMON

# perf --control=fifo plumbing: glm writes "enable"/"disable" (PERF_FIFO) to
# bracket ONLY its decode loop; perf starts with counters off (--delay=-1).
CTL=/tmp/perf_${TAG}.ctl; ACKF=/tmp/perf_${TAG}.ack
setup_fifo(){ rm -f "\$CTL" "\$ACKF"; mkfifo "\$CTL" "\$ACKF"; }
FIFO="--delay=-1 --control=fifo:\$CTL,\$ACKF"

# A) TMA + core PMU, real baseline interleave=all @ $NTH (decode-only via fifo)
setup_fifo
env SNAP="$SNAP" OMP_NUM_THREADS=$NTH PERF_FIFO="\$CTL,\$ACKF" \
  perf stat \$FIFO -M $TMAM -e "$COREV" -- numactl --interleave=all "$BIN" 256 \
  > "$OUT.A.out" 2> "$OUT.A.perf"

# B) uncore IMC CAS -> DRAM GB/s @ $NTH (decode-only; -a system-wide but fenced)
setup_fifo
env SNAP="$SNAP" OMP_NUM_THREADS=$NTH PERF_FIFO="\$CTL,\$ACKF" \
  perf stat -a \$FIFO -e "$EVB" -- numactl --interleave=all "$BIN" 256 \
  > /dev/null 2> "$OUT.B.perf"

# C) single-die-confined @ ${CPN}T (decode-only via fifo)
setup_fifo
env NUMA_PARTITION=0 SNAP="$SNAP" OMP_NUM_THREADS=$CPN PERF_FIFO="\$CTL,\$ACKF" \
  perf stat \$FIFO -M tma_retiring,tma_core_bound,tma_memory_bound \
  -e cycles,instructions,cycle_activity.stalls_l3_miss,exe_activity.bound_on_loads \
  -- numactl --cpunodebind=0 --membind=0 "$BIN" 256 \
  > "$OUT.C.out" 2> "$OUT.C.perf"

# D) thread-scaling sweep at N=1, single-die (decode-only via fifo)
{
  echo "# NTH ipc amx_busy% core_bound mem_bound stalls_l3% tok/s"
  for t in $SWEEP; do
    rm -f "\$CTL" "\$ACKF"; mkfifo "\$CTL" "\$ACKF"
    env NUMA_PARTITION=0 SNAP="$SNAP" OMP_NUM_THREADS=\$t PERF_FIFO="\$CTL,\$ACKF" \
      perf stat --delay=-1 --control=fifo:"\$CTL","\$ACKF" -M tma_core_bound,tma_memory_bound \
      -e cycles,instructions,cycle_activity.stalls_l3_miss,exe.amx_busy \
      -- numactl --cpunodebind=0 --membind=0 "$BIN" 256 \
      > "$OUT.D.t\$t.out" 2> "$OUT.D.t\$t.perf"
    ipc=\$(awk '/insn per cycle/{print \$(NF-3);exit}' "$OUT.D.t\$t.perf")
    cyc=\$(awk '/ cycles /{gsub(/[^0-9]/,"",\$1);print \$1;exit}' "$OUT.D.t\$t.perf")
    l3=\$(awk '/stalls_l3_miss/{gsub(/[^0-9]/,"",\$1);print \$1;exit}' "$OUT.D.t\$t.perf")
    amx=\$(awk '/amx_busy/{gsub(/[^0-9]/,"",\$1);print \$1;exit}' "$OUT.D.t\$t.perf")
    cb=\$(awk '/tma_core_bound/{print \$(NF-1);exit}' "$OUT.D.t\$t.perf")
    mb=\$(awk '/tma_memory_bound/{print \$(NF-1);exit}' "$OUT.D.t\$t.perf")
    tok=\$(grep -ioE '[0-9.]+ tok/s' "$OUT.D.t\$t.out" | tail -1)
    l3p=\$(awk -v l=\$l3 -v c=\$cyc 'BEGIN{if(c>0)printf "%.1f",100*l/c;else print "NA"}')
    amxp=\$(awk -v a=\$amx -v c=\$cyc 'BEGIN{if(c>0)printf "%.1f",100*a/c;else print "NA"}')
    echo "\$t ipc=\$ipc amx=\$amxp% cb=\$cb mb=\$mb l3=\$l3p% \$tok"
  done
} > "$OUT.D.out" 2>&1

rm -f "\$CTL" "\$ACKF"
echo ALL_DONE > "$OUT.done"
RUNNER
chmod +x "$RUN"
rm -f "$OUT.done"
nohup bash "$RUN" >/dev/null 2>&1 &

echo "launched $RUN  (SNAP=$SNAP NTH=$NTH CPN=$CPN IMC_events=$(printf '%s' "$EVB" | tr ',' '\n' | grep -c /) SWEEP='$SWEEP')"
echo "poll: cat $OUT.done   (expect ALL_DONE; A+B+C+sweep = several decode runs)"
echo "parse:"
echo "  A: grep -E 'tma_|insn per cycle|stalls|bound_on_loads|miss|elapsed' $OUT.A.perf"
echo "  B: grep -v 'Unexpected IMC' $OUT.B.perf | awk '/cas_count/{g+=\$1}/elapsed/{t=\$1}END{print g*64/1e9/t\" GB/s over \"t\"s\"}'"
echo "  C: grep -E 'tma_|insn per cycle|stalls_l3|bound_on_loads|elapsed' $OUT.C.perf"
echo "  D: cat $OUT.D.out"
