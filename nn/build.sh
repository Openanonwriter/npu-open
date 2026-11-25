#!/usr/bin/bash
set -eu; cd "$(dirname "$0")"; H="$(cd .. && pwd)/hexagonrpc"; T="--target=hexagon-unknown-none-elf -mv73"
clang $T -O2 -fPIC -G0 -ffreestanding -fno-builtin -c nn_skel.c -o nn_skel.o
clang $T -c hmxops.S -o hmxops.o
ld.lld -shared --hash-style=sysv -z max-page-size=4096 -z norelro --no-rosegment -z separate-loadable-segments -Bsymbolic -o libnn_skel.so nn_skel.o hmxops.o
cp libnn_skel.so ../root/dsp/cdsp/
gcc -O2 -Wall -I$H/include nn_client.c -o nn_client -lm -L$H/build/libhexagonrpc -lhexagonrpc -Wl,-rpath,$H/build/libhexagonrpc
