#!/usr/bin/bash
# Attack-like traffic for evaluation, against this machine's loopback address only:
# port scans (connect, and SYN if run with sudo), a UDP probe sweep, and rapid repeated logins-style connections.
cd "$(dirname "$0")"
python3 -m http.server 8099 --bind 127.0.0.1 --directory www >/dev/null 2>&1 & SRV=$!
read -t 1 < /dev/zero 2>/dev/null
nmap -sT -n -Pn -p 1-3000 --max-retries 0 -T4 127.0.0.1 >/dev/null 2>&1
sudo -n nmap -sS -n -Pn -p 3001-6000 --max-retries 0 -T4 127.0.0.1 >/dev/null 2>&1
sudo -n nmap -sU -n -Pn -p 1-300 --max-retries 0 -T4 127.0.0.1 >/dev/null 2>&1
for i in $(seq 1 400); do ncat -w 1 127.0.0.1 8099 </dev/null >/dev/null 2>&1; done
for i in $(seq 1 150); do printf 'GET /login?u=admin&p=%s HTTP/1.0\r\n\r\n' $RANDOM | ncat -w 1 127.0.0.1 8099 >/dev/null 2>&1; done
kill $SRV 2>/dev/null; wait 2>/dev/null
