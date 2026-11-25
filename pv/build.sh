#!/usr/bin/bash
set -eu; cd "$(dirname "$0")"; H="$(cd .. && pwd)/hexagonrpc"
clang --target=hexagon-unknown-none-elf -mv73 -O2 -fPIC -G0 -ffreestanding -fno-builtin -Wall -c pv_skel.c -o pv_skel.o
ld.lld -shared --hash-style=sysv -z max-page-size=4096 -z norelro --no-rosegment -z separate-loadable-segments -Bsymbolic -o libpv_skel.so pv_skel.o
cp libpv_skel.so ../root/dsp/cdsp/
gcc -O2 -Wall -I$H/include pv_client.c -o pv_client -L$H/build/libhexagonrpc -lhexagonrpc -Wl,-rpath,$H/build/libhexagonrpc
