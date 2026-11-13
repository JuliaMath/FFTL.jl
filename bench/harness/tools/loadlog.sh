#!/bin/bash
# Load history of a shared Linux machine (sysstat collection may be off): every <step>
# seconds one line with the time, the 1-min load average and the busy fraction (from /proc/stat
# deltas, user+nice+system+irq+softirq+steal over all) of each block of 16 consecutive CPUs.
#   harness/tools/loadlog.sh <out.csv> [step = 60] [hours = 48]
# Columns: time,load1,b0,b16,…,b240 (bK = CPUs K…K+15). Pick quiet windows and blocks with
# harness/tools/quiet.jl.
out=$1; step=${2:-60}; hours=${3:-48}
snap() { awk '/^cpu[0-9]/{id=substr($1,4); busy=$2+$3+$4+$7+$8+$9; tot=busy+$5+$6; print id, busy, tot}' /proc/stat; }
nb=$(( $(getconf _NPROCESSORS_ONLN) / 16 ))
[ -f "$out" ] || { printf "time,load1"; for b in $(seq 0 $((nb - 1))); do printf ",b%d" $((16 * b)); done; echo; } > "$out"
prev=$(snap); end=$(( $(date +%s) + 3600 * hours ))
while [ "$(date +%s)" -lt "$end" ]; do
  sleep "$step"
  cur=$(snap)
  line=$(paste -d' ' <(echo "$prev") <(echo "$cur") | awk -v nb="$nb" '
    {b=int($1/16); db[b]+=$5-$2; dt[b]+=$6-$3}
    END{for(i=0;i<nb;i++) printf ",%.3f", (dt[i]>0 ? db[i]/dt[i] : 0)}')
  echo "$(date +%FT%T),$(cut -d' ' -f1 /proc/loadavg)$line" >> "$out"
  prev=$cur
done
