#!/usr/bin/bash
# Start, stop or query the NPU server (rust/target/release/server behind the patched hexagonrpcd).
# usage: serve.sh start | stop | status        log: server.log   socket: npu.sock
ROOT="$(cd "$(dirname "$0")" && pwd)"; cd "$ROOT" || exit 1
export NPU_OPEN="$ROOT"
SHELL_ELF="${NPU_SHELL:-$ROOT/shell/fastrpc_shell_unsigned_3}"
BIN="$ROOT/rust/target/release/server"
pause() { read -t "$1" < /dev/zero 2>/dev/null; true; }
running() { [ -f server.pid ] && kill -0 "$(cat server.pid)" 2>/dev/null; }
case "${1:-}" in
start)
	running && { echo "already running (pid $(cat server.pid))"; exit 0; }
	[ -x "$BIN" ] || { echo "build first: (cd rust && cargo build --release)"; exit 1; }
	[ -r "$SHELL_ELF" ] || { echo "missing $SHELL_ELF (see docs/SETUP.md)"; exit 1; }
	[ "$(cat /sys/class/remoteproc/remoteproc1/state)" = running ] || { echo "compute DSP is not running"; exit 1; }
	rm -f npu.sock
	CMD="stdbuf -o0 -e0 hexagonrpc/build/hexagonrpcd/hexagonrpcd -f /dev/fastrpc-cdsp -d cdsp -u -c $SHELL_ELF -R $ROOT/root -p $BIN"
	if id -Gn | tr ' ' '\n' | grep -qx fastrpc; then setsid nohup $CMD > server.log 2>&1 &
	else setsid nohup sg fastrpc -c "$CMD" > server.log 2>&1 & fi
	echo $! > server.pid
	for i in $(seq 1 30); do [ -S npu.sock ] && { echo "NPU server ready (pid $(cat server.pid))"; exit 0; }; running || break; pause 0.5; done
	echo "server did not come up; after a reboot or DSP crash run ./cdsp78.sh first. Last log lines:"; grep -v '^\(read\|lseek\|openat\|close\)' server.log | tail -5; exit 1 ;;
stop)
	[ -S npu.sock ] && "$ROOT/rust/target/release/usecases" quit >/dev/null 2>&1
	pause 1
	running && kill -TERM "$(cat server.pid)" 2>/dev/null   # the daemon is idle here; never SIGKILL mid-call
	rm -f server.pid npu.sock; echo "stopped" ;;
status)
	running && [ -S npu.sock ] && echo "running (pid $(cat server.pid))" || { echo "not running"; exit 1; } ;;
*) echo "usage: $0 start|stop|status"; exit 1 ;;
esac
