#!/usr/bin/bash
# Build the DSP-side library with the open LLVM toolchain, stage it where the daemon serves it, and build the client.
# The cDSP loader needs: exactly two LOAD segments, the second starting on a 4K page boundary in file and memory.
set -eu
cd "$(dirname "$0")"; H="$(cd .. && pwd)/hexagonrpc"
clang --target=hexagon-unknown-none-elf -mv73 -O2 -fPIC -G0 -ffreestanding -fno-builtin -c add_skel.c -o add_skel.o
ld.lld -shared --hash-style=sysv -z max-page-size=4096 -z norelro --no-rosegment \
       -z separate-loadable-segments -Bsymbolic -o libadd_skel.so add_skel.o
mkdir -p ../root/dsp/cdsp && cp libadd_skel.so ../root/dsp/cdsp/
gcc -O2 -Wall -I$H/include client.c -o client -L$H/build/libhexagonrpc -lhexagonrpc -Wl,-rpath,$H/build/libhexagonrpc
# Emulator check of the same C code (needs qemu-user-static-hexagon): exits with 42 for 2 + 40.
clang --target=hexagon-unknown-linux-musl -mv73 -O2 -G0 -ffreestanding -fno-builtin -c emu_test.c -o emu_test.o
clang --target=hexagon-unknown-linux-musl -mv73 -O2 -G0 -ffreestanding -fno-builtin -c add_skel.c -o add_static.o
ld.lld -static -e _start -o emu_test emu_test.o add_static.o
