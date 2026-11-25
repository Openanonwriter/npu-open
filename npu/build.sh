#!/usr/bin/bash
set -eu; cd "$(dirname "$0")"; H="$(cd .. && pwd)/hexagonrpc"; T="--target=hexagon-unknown-none-elf -mv73"
clang $T -O2 -fPIC -G0 -ffreestanding -fno-builtin -Wall -c npu_skel.c -o npu_skel.o
clang $T -c hmxops.S -o hmxops.o
clang $T -mhvx -mhvx-length=128B -O2 -fPIC -G0 -ffreestanding -fno-builtin -c hvxcopy.c -o hvxcopy.o
ld.lld -shared --hash-style=sysv -z max-page-size=4096 -z norelro --no-rosegment -z separate-loadable-segments -Bsymbolic -o libnpu_skel.so npu_skel.o hmxops.o hvxcopy.o
cp libnpu_skel.so ../root/dsp/cdsp/
gcc -O3 -Wall -I$H/include npu_client.c -o npu_client -lm -L$H/build/libhexagonrpc -lhexagonrpc -Wl,-rpath,$H/build/libhexagonrpc
