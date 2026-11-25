#!/usr/bin/bash
set -eu; cd "$(dirname "$0")"; H="$(cd .. && pwd)/hexagonrpc"; T="--target=hexagon-unknown-none-elf -mv73"
clang $T -O2 -fPIC -G0 -ffreestanding -fno-builtin -c mm_skel.c -o mm_skel.o
clang $T -c hmxops.S -o hmxops.o
ld.lld -shared --hash-style=sysv -z max-page-size=4096 -z norelro --no-rosegment -z separate-loadable-segments -Bsymbolic -o libmm_skel.so mm_skel.o hmxops.o
cp libmm_skel.so ../root/dsp/cdsp/
gcc -O2 -Wall -I$H/include mm_client.c -o mm_client -L$H/build/libhexagonrpc -lhexagonrpc -Wl,-rpath,$H/build/libhexagonrpc
gcc -O2 -Wall -I$H/include mm2_client.c -o mm2_client -L$H/build/libhexagonrpc -lhexagonrpc -Wl,-rpath,$H/build/libhexagonrpc
