#!/usr/bin/bash
# Build one probe library per name in names.txt (gen.sh) and the client that reports which ones the DSP resolves.
set -eu
cd "$(dirname "$0")"; H="$(cd .. && pwd)/hexagonrpc"
./gen.sh
gcc -O2 -Wall -I$H/include probe_client.c -o probe_client -L$H/build/libhexagonrpc -lhexagonrpc -Wl,-rpath,$H/build/libhexagonrpc
