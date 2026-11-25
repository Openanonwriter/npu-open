#!/usr/bin/bash
# Live flow scoring with the improved method: capture packet headers for N seconds, turn them into flow
# records, and score them on the NPU against the real-traffic baseline (whitened distance, no trained model).
# Needs the NPU server (./serve.sh start) and a baseline: ids/cap/real-baseline.f32 (docs/SETUP.md).
# usage: live2.sh [seconds] [interface ...]      default: 30 s on the default-route interface
set -eu; cd "$(dirname "$0")"; N=${1:-30}; shift || true
IF=("$@"); [ ${#IF[@]} -gt 0 ] || IF=($(ip -o route show default | awk '{print $5; exit}'))
args=(); for i in "${IF[@]}"; do args+=(-i "$i"); done
echo "capturing $N s of packet headers on ${IF[*]} ..."
dumpcap -q -s 96 "${args[@]}" -a duration:"$N" -w cap/live.pcapng >/dev/null 2>&1
python3 -I ids.py features cap/live.pcapng cap/live.npz
python3 -I ids.py export cap/live.npz cap/live.f32
NPU_OPEN="$(cd .. && pwd)" ../rust/target/release/usecases 209
