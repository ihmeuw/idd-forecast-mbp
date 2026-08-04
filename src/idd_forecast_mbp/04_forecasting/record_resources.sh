#!/usr/bin/env bash
# Append Slurm resource usage (one row per task) to forecast_resource_log.csv, so
# we can tune --mem / -t for this and future forecast runs from measured data.
#
# RUN FROM A LOGIN NODE: sacct is a Slurm client and is NOT present inside the
# rstudio singularity image, so the finalize job cannot self-record. After any
# run (probe or real) completes, run this with the array (or single) job id(s).
#
# Usage: record_resources.sh <run_label> <jobid> [jobid ...]
#   e.g. record_resources.sh forecast_2026_06_02_real 47891234
set -euo pipefail

LOG="$(cd "$(dirname "$0")" && pwd)/forecast_resource_log.csv"
LABEL="${1:?usage: record_resources.sh <run_label> <jobid> [jobid ...]}"; shift
NOW="$(date '+%Y-%m-%d %H:%M:%S')"

[ -f "$LOG" ] || echo "recorded_at,run_label,jobid,jobname,state,elapsed,totalcpu,maxrss_gb,reqmem,alloccpus" > "$LOG"

for J in "$@"; do
  # sacct emits a main row (metadata) then a .batch step (carries MaxRSS/TotalCPU)
  # for each task; pair them in awk. Array jobs expand to <id>_<idx> tasks.
  sacct -j "$J" -P --noheader \
    --format=JobID,JobName,State,Elapsed,TotalCPU,MaxRSS,ReqMem,AllocCPUS \
  | awk -v now="$NOW" -v label="$LABEL" -F'|' '
      $1 !~ /\./        { id=$1; name=$2; state=$3; elapsed=$4; reqmem=$7; cpus=$8; have=1; next }
      $1 ~ /\.batch$/ && have {
        tcpu=$5; mr=$6; gb=mr
        if      (mr ~ /K$/) { sub(/K$/,"",mr); gb=sprintf("%.1f", mr/1e6) }
        else if (mr ~ /M$/) { sub(/M$/,"",mr); gb=sprintf("%.1f", mr/1e3) }
        else if (mr ~ /G$/) { sub(/G$/,"",mr); gb=sprintf("%.1f", mr+0)   }
        printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n", now, label, id, name, state, elapsed, tcpu, gb, reqmem, cpus
        have=0
      }
    ' >> "$LOG"
done
echo "Appended to $LOG:"
column -t -s',' "$LOG"
