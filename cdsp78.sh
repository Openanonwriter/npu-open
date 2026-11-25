#!/usr/bin/bash
# Restart the compute DSP on the firmware pair held in ./fw78 (qccdsp8380.mbn + cdsp_dtb.mbn under
# fw78/qcom/x1e80100/microsoft/Denali/). RAM only: nothing under /lib/firmware is replaced, and a reboot
# or a DSP crash recovery goes back to the installed firmware. Needs sudo.
ROOT="$(cd "$(dirname "$0")" && pwd)"
R=/sys/class/remoteproc/remoteproc1; FP=/sys/module/firmware_class/parameters/path
[ "$(cat $R/name)" = cdsp ] || { echo "remoteproc1 is not the cdsp; refusing"; exit 1; }
pause() { read -t "$1" < /dev/zero 2>/dev/null; true; }
for i in 1 2 3 4 5 6 7 8 9 10; do [ "$(cat $R/state)" = running ] && break; pause 1; done
echo -n "$ROOT/fw78" | sudo -n tee $FP >/dev/null
echo stop | sudo -n tee $R/state >/dev/null; pause 2
echo start | sudo -n tee $R/state >/dev/null
echo | sudo -n tee $FP >/dev/null        # a newline clears the path; writing nothing leaves it set
for i in 1 2 3 4 5 6 7 8 9 10; do [ "$(cat $R/state)" = running ] && [ -e /dev/fastrpc-cdsp ] && break; pause 1; done
pause 2
echo "cdsp: $(cat $R/state), firmware size loaded: $(journalctl -k --since -1min --no-pager | grep -o 'qccdsp8380.mbn, size [0-9]*' | tail -1)"
