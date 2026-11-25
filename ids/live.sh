#!/usr/bin/bash
# Capture packet headers for N seconds, turn them into flow records, score them on the NPU, list the oddest flows.
# usage: live.sh [seconds] [interface ...]      (default 30 s on loopback and the default-route interface)
set -eu; cd "$(dirname "$0")"; N=${1:-30}; shift || true
IF=("$@"); [ ${#IF[@]} -gt 0 ] || IF=(lo $(ip -o route show default | awk '{print $5; exit}'))
args=(); for i in "${IF[@]}"; do args+=(-i "$i"); done
[ "$(cat /sys/class/remoteproc/remoteproc1/state)" = running ] || { echo "compute DSP is not running"; exit 1; }
echo "capturing $N s of packet headers on ${IF[*]} ..."
dumpcap -q -s 96 "${args[@]}" -a duration:"$N" -w cap/live.pcapng >/dev/null 2>&1
python3 -I ids.py features cap/live.pcapng cap/live.npz
python3 -I ids.py prepare cap/live.npz live-model.bin
MODEL=$PWD/live-model.bin OUT=$PWD/live-out.bin CLIENT=$PWD/../npu/npu_client ../run.sh 30
grep -q 'npu: network' ../last-run.log || { echo "accelerator run failed (after a reboot run ../cdsp78.sh first):"; tail -3 ../last-run.log; exit 1; }
grep -o 'npu: time per image.*' ../last-run.log | sed 's/image/flow record/'
python3 -I ids.py report cap/live.npz live-model.bin live-out.bin
