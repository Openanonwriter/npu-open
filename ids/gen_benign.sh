#!/usr/bin/bash
# Ordinary-looking local traffic for the baseline capture: a web server on loopback and a client fetching
# files of varied sizes at irregular intervals, plus a few idle keep-alive connections. Loopback only.
cd "$(dirname "$0")"; D=${1:-150}
for s in 1 4 16 64 256 1024 4096; do head -c $((s * 1024)) /dev/urandom > www/f$s.bin; done
python3 -m http.server 8099 --bind 127.0.0.1 --directory www >/dev/null 2>&1 & SRV=$!
read -t 1 < /dev/zero 2>/dev/null
end=$((SECONDS + D))
( while [ $SECONDS -lt $end ]; do ncat 127.0.0.1 8099 -i 3 </dev/null >/dev/null 2>&1; read -t 2 < /dev/zero 2>/dev/null; done ) &
while [ $SECONDS -lt $end ]; do
	s=$(shuf -n1 -e 1 1 1 4 4 16 16 64 256 1024 4096)
	curl -s -o /dev/null "http://127.0.0.1:8099/f$s.bin"
	[ $((RANDOM % 5)) = 0 ] && curl -s -o /dev/null "http://127.0.0.1:8099/missing-$RANDOM"
	read -t "0.$((RANDOM % 9 + 1))" < /dev/zero 2>/dev/null
done
kill $SRV 2>/dev/null; wait 2>/dev/null
