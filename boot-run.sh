#!/usr/bin/bash
# Runs once at boot (npu-open-boot-test.service): load the build-00078 DSP firmware into RAM, run the digit
# classifier on the accelerator, and write the outcome to boot-run.log. It disables itself before doing anything,
# so a crash or hang here cannot repeat on the following boot.
systemctl disable npu-open-boot-test.service >/dev/null 2>&1
D="$(cd "$(dirname "$0")" && pwd)"; LOG=$D/boot-run.log; OWNER=$(stat -c %U "$D"); R=/sys/class/remoteproc/remoteproc1
pause() { read -t "$1" < /dev/zero 2>/dev/null; true; }
{
	echo "=== boot test $(date '+%Y-%m-%d %H:%M:%S'), kernel $(uname -r), up $(cut -d' ' -f1 /proc/uptime)s"
	for i in $(seq 1 60); do [ "$(cat $R/state 2>/dev/null)" = running ] && [ -e /dev/fastrpc-cdsp ] && break; pause 1; done
	echo "cdsp before: $(cat $R/state 2>/dev/null)"
	$D/cdsp78.sh
	CLIENT=$D/nn/nn_client runuser -u $OWNER -- $D/run.sh 20
	grep -E 'nn: [0-9]+ images|nn: accelerator time|nn: .*failed|INIT_CREATE|Could not' $D/last-run.log
	grep -q 'nn: 1024 images' $D/last-run.log && echo "RESULT: classifier ran on the accelerator" || echo "RESULT: FAILED (see last-run.log)"
	echo "cdsp after: $(cat $R/state 2>/dev/null); DSP crashes this boot: $(journalctl -k -b --no-pager | grep -c 'crash detected')"
} >> $LOG 2>&1
chown $OWNER: $LOG $D/last-run.log 2>/dev/null
