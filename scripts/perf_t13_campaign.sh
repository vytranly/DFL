#!/usr/bin/env bash
# Chiến dịch đo T13 (docs/PERF.md §3, §4). Chạy từ gốc repo, ghi ra build/perf/ (không vào results/):
#   bash scripts/perf_t13_campaign.sh > build/perf/campaign.log 2>&1
set -u
PY=.venv/bin/python
P=build/perf
mkdir -p $P/cfg  # cấu hình tạm, kế thừa profile của repo
for k in 0 1 2 3 4 5 6; do printf 'extends: ../../../configs/profiles/fast.yaml\nseed: %s\n' $k > $P/cfg/fast_seed$k.yaml; done
printf 'extends: ../../../configs/base.yaml\n' > $P/cfg/full.yaml
printf 'extends: ../../../configs/profiles/fast.yaml\nnum_nodes: 8\nlocal_steps: 200\nruntime: processes\n' > $P/cfg/proc8.yaml
printf 'extends: ../../../configs/profiles/fast.yaml\nnum_nodes: 8\nlocal_steps: 200\n' > $P/cfg/inproc8.yaml
cores() { $PY scripts/perf_t13.py cores --seconds 30; }
batch() {  # batch <tên> <số lượt song song>
  name=$1; k=$2; rm -rf $P/$name; start=$(date +%s); cores
  for i in $(seq 0 $((k - 1))); do $PY -m dfl run --config $P/cfg/fast_seed$i.yaml --out $P/$name/s$i > $P/$name-s$i.log 2>&1 & done
  wait; echo "BATCH $name workers=$k wall=$(( $(date +%s) - start )) s"; cores
}
batch fast_w1 1
batch fast_w4 4
batch fast_w7 7
rm -rf $P/full; cores; start=$(date +%s)
/usr/bin/time -v $PY -m dfl run --config $P/cfg/full.yaml --out $P/full > $P/full.log 2>&1
echo "FULL wall=$(( $(date +%s) - start )) s"; grep -E 'Maximum resident' $P/full.log; cores
rm -rf $P/proc8 $P/inproc8; start=$(date +%s)
$PY -m dfl run --config $P/cfg/inproc8.yaml --out $P/inproc8 > $P/inproc8.log 2>&1; echo "INPROC8 wall=$(( $(date +%s) - start )) s"
start=$(date +%s)
$PY -m dfl run --config $P/cfg/proc8.yaml --out $P/proc8 > $P/proc8.log 2>&1 & pid=$!
peak=0; while kill -0 $pid 2>/dev/null; do s=$(ps -eo rss,args | awk '/dfl node/ && !/awk/ {t+=$1} END {print t+0}'); [ "$s" -gt "$peak" ] && peak=$s; sleep 1; done
echo "PROC8 wall=$(( $(date +%s) - start )) s; RSS đỉnh tổng của 8 tiến trình node = $((peak / 1024)) MiB"
cmp $(ls $P/inproc8/runs/*/attempt-1/rounds.jsonl) $(ls $P/proc8/runs/*/attempt-1/rounds.jsonl) && echo "PROC8 rounds.jsonl trùng bản inprocess"
cores
echo DONE
