#!/usr/bin/bash
set -eu; cd "$(dirname "$0")"; H="$(cd .. && pwd)/hexagonrpc"
clang --target=hexagon-unknown-none-elf -mv73 -O2 -fPIC -G0 -ffreestanding -fno-builtin -c hmx_skel.c -o hmx_skel.o
ld.lld -shared --hash-style=sysv -z max-page-size=4096 -z norelro --no-rosegment -z separate-loadable-segments -Bsymbolic -o libhmx_skel.so hmx_skel.o
cp libhmx_skel.so ../root/dsp/cdsp/
gcc -O2 -Wall -I$H/include hmx_client.c -o hmx_client -L$H/build/libhexagonrpc -lhexagonrpc -Wl,-rpath,$H/build/libhexagonrpc
