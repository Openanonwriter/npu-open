#!/usr/bin/bash
# Keep scoring live traffic: run live2.sh every N seconds until interrupted, printing only the summary
# line and any alarms. Needs the NPU server and a baseline (see live2.sh).
# usage: watch.sh [seconds per capture]      default 30
cd "$(dirname "$0")"; N=${1:-30}
while true; do
	./live2.sh "$N" 2>&1 | grep -E '^209|ALARM' | sed "s/^/$(date +%H:%M:%S) /"
done
