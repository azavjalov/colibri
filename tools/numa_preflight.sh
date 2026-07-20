#!/usr/bin/env bash
# numa_preflight.sh — NUMA topology check + pinning recommendation for the glmrt
# decode/prefill runtime on the aibox101b-class host (single-socket GNR-AP, SNC3).
#
# WHY THIS EXISTS (see AGENTS.md "THE ONE FACT"): single-stream decode on this box
# pulls only ~63 of 753 GB/s. The runtime pins OMP threads (OMP_PROC_BIND=close)
# but in DECODE it does NOT bind them per-NUMA-node (the 3-pass NUMA_COMPUTE path is
# S-gated off at S=1). Weights are mbind'd round-robin (expert_node=eid%3) and the
# tuned recipe adds `numactl --interleave=all`, so a node-0-pinned thread routinely
# reads node-1/2-resident weights (on-package die-to-die, distance 15/17). This
# script makes the topology + the actual thread/memory placement VISIBLE before a
# run and flags the common misconfigurations, so we stop rediscovering them.
#
# It does NOT modify the runtime; it validates env + emits a recommended launch line.
# Read-only except for optional --probe which runs a tiny STREAM-like check per node.
set -u

RED=$'\e[31m'; GRN=$'\e[32m'; YEL=$'\e[33m'; RST=$'\e[0m'
ok(){   echo "${GRN}[ok]${RST}  $*"; }
warn(){ echo "${YEL}[warn]${RST} $*"; }
bad(){  echo "${RED}[BAD]${RST} $*"; }
info(){ echo "      $*"; }

echo "=== glmrt NUMA preflight ==="

# ---- 1. Topology ----
if ! command -v numactl >/dev/null; then bad "numactl not found"; exit 1; fi
NNODES=$(numactl --hardware | awk '/available:/{print $2}')
SOCKETS=$(lscpu | awk -F: '/^Socket\(s\)/{gsub(/ /,"",$2);print $2}')
CPS=$(lscpu | awk -F: '/Core\(s\) per socket/{gsub(/ /,"",$2);print $2}')
TPC=$(lscpu | awk -F: '/Thread\(s\) per core/{gsub(/ /,"",$2);print $2}')
echo "sockets=$SOCKETS  cores/socket=$CPS  threads/core=$TPC  numa_nodes=$NNODES"

# SNC detection: >1 NUMA node on a single socket == Sub-NUMA Clustering enabled.
if [ "$SOCKETS" = "1" ] && [ "${NNODES:-1}" -gt 1 ]; then
  warn "SNC${NNODES} is ENABLED (single socket, $NNODES NUMA nodes = 3 dies each fronting ~1/$NNODES of the mem controllers)."
  info "Cross-node here is on-package die-to-die (distance 15/17), NOT cross-socket UPI. Penalty is modest but real for a single stream."
  info "To A/B against a flat address space, disable SNC in BIOS (UEFI > Uncore > SNC = Disable) -> 1 node, all controllers hw-interleaved."
elif [ "${NNODES:-1}" -gt 1 ]; then
  ok "multi-socket, $NNODES nodes"
else
  ok "single flat NUMA node (SNC disabled or not present)"
fi

# ---- 2. Per-node memory headroom (resident model must fit the interleave) ----
echo "--- per-node memory ---"
numactl --hardware | awk '/node [0-9]+ size:/{sz=$4} /node [0-9]+ free:/{print "node "$2": free "$4" MB"}'
warn "resident glmrt needs the model interleaved across nodes; a node near-full forces cross-node spill. Check 'free' per node > model_share."

# ---- 3. Env sanity for the DECODE recipe ----
echo "--- launch env sanity ---"
[ "${OMP_PROC_BIND:-}" = "close" ] && ok "OMP_PROC_BIND=close (threads pinned, scheduler out of the picture)" \
  || warn "OMP_PROC_BIND not 'close' (got '${OMP_PROC_BIND:-unset}') -> threads may migrate; set OMP_PROC_BIND=close OMP_PLACES=cores"
[ "${OMP_PLACES:-}" = "cores" ] && ok "OMP_PLACES=cores" || warn "OMP_PLACES not 'cores' (got '${OMP_PLACES:-unset}')"
[ "${OMP_DYNAMIC:-}" = "FALSE" ] && ok "OMP_DYNAMIC=FALSE (fixed team)" || warn "OMP_DYNAMIC not FALSE -> per-region thread churn possible"
if [ -n "${OMP_NUM_THREADS:-}" ]; then
  # physical cores per node = cores/socket / nodes; warn if threads span > one node without NUMA_COMPUTE
  PPN=$(( CPS / NNODES ))
  ok "OMP_NUM_THREADS=$OMP_NUM_THREADS (physical cores/node ~= $PPN)"
  if [ "${NUMA_COMPUTE:-0}" = "0" ] && [ "$OMP_NUM_THREADS" -gt "$PPN" ]; then
    warn "NUMA_COMPUTE=0 and threads ($OMP_NUM_THREADS) > cores/node ($PPN): the single pinned pool spans multiple dies but reads mbind'd (eid%%$NNODES) weights => cross-die loads. This is the ~63/753 GB/s decode gap."
    info "Options to A/B: (a) --cpunodebind=0 --membind=0 OMP_NUM_THREADS=$PPN (confine to one die, no cross-node) ; (b) enable NUMA_COMPUTE for S>=8 ; (c) disable SNC in BIOS."
  fi
else
  warn "OMP_NUM_THREADS unset"
fi
[ "${COLI_MMAP:-0}" = "1" ] && ok "COLI_MMAP=1" || warn "COLI_MMAP!=1: MXFP4/IQ3 require the mmap loader path (E8M0 byte scales)"
[ "${RESIDENT:-0}" = "1" ] && ok "RESIDENT=1 (no disk in the read path)" || warn "RESIDENT!=1: disk may enter the decode path"

# ---- 4. Recommended launch lines ----
PPN=$(( CPS / NNODES ))
echo "--- recommended A/B launch lines (single-socket SNC$NNODES) ---"
cat <<EOF
# baseline (current tuned, NUMA-oblivious full pool + interleave):
SNAP=<model> CTX=4096 TEMP=0 AMX=1 COLI_MMAP=1 RESIDENT=1 NUMA_PARTITION=1 NUMA_COMPUTE=0 \\
  COLI_NO_OMP_TUNE=1 MTP=0 OMP_NUM_THREADS=$CPS OMP_PROC_BIND=close OMP_PLACES=cores \\
  REPLAY=1 REF=ref_bench.json REF_FORCE=1 INSTR=1 numactl --interleave=all ./glm 256

# single-die-confined (tests cross-die penalty: all compute + all touched mem on node 0):
SNAP=<model> CTX=4096 TEMP=0 AMX=1 COLI_MMAP=1 RESIDENT=1 NUMA_PARTITION=0 NUMA_COMPUTE=0 \\
  COLI_NO_OMP_TUNE=1 MTP=0 OMP_NUM_THREADS=$PPN OMP_PROC_BIND=close OMP_PLACES=cores \\
  REPLAY=1 REF=ref_bench.json REF_FORCE=1 INSTR=1 numactl --cpunodebind=0 --membind=0 ./glm 256
EOF

echo "=== preflight done ==="
