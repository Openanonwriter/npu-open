#!/usr/bin/bash
# Start the patched open daemon with an unsigned DSP process and run one client program against the cDSP.
# usage: CLIENT=path/to/client [other client settings] run.sh [timeout seconds]     output: last-run.log
#
# NPU_SHELL: the FastRPC shell ELF matching the running cDSP firmware build (default shell/fastrpc_shell_unsigned_3).
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT" || exit 1
export NPU_OPEN="$ROOT"
SHELL_ELF="${NPU_SHELL:-$ROOT/shell/fastrpc_shell_unsigned_3}"
CLIENT="${CLIENT:-$ROOT/add/client}"
[ -r "$SHELL_ELF" ] || { echo "missing $SHELL_ELF (see docs/SETUP.md)" | tee last-run.log; exit 1; }
CMD="timeout -s TERM ${1:-8} stdbuf -o0 -e0 hexagonrpc/build/hexagonrpcd/hexagonrpcd -f /dev/fastrpc-cdsp -d cdsp -u -c $SHELL_ELF -R $ROOT/root -p $CLIENT"
if id -Gn | tr ' ' '\n' | grep -qx fastrpc; then $CMD > last-run.log 2>&1; else sg fastrpc -c "$CMD" > last-run.log 2>&1; fi
true
