#!/usr/bin/bash
# Build the C++ operation-sweep library for the cDSP (stock clang++ and lld; no C++ runtime) and stage it.
set -eu; cd "$(dirname "$0")"
T="--target=hexagon-unknown-none-elf -mv73 -std=c++17 -fPIC -G0 -ffreestanding -fno-builtin -fno-exceptions -fno-rtti -fno-threadsafe-statics -fno-use-cxa-atexit -Wall"
clang++ $T -O2 -c ops_skel.cpp -o ops_skel.o
clang++ $T -O3 -fno-vectorize -fno-slp-vectorize -c ops_scalar.cpp -o ops_scalar.o
clang++ $T -O3 -mhvx -mhvx-length=128B -fvectorize -mllvm -hexagon-autohvx -c ops_hvx.cpp -o ops_hvx.o
ld.lld -shared --hash-style=sysv -z max-page-size=4096 -z norelro --no-rosegment -z separate-loadable-segments -Bsymbolic -o libops_skel.so ops_skel.o ops_scalar.o ops_hvx.o
llvm-readelf -lW libops_skel.so | grep -c LOAD
llvm-nm -u libops_skel.so | tr '\n' ' '; echo
llvm-objdump -d ops_hvx.o | grep -c -i -E "vmem|v[0-9]+\.[a-z]+ *=" || true  # HVX instructions in the vector build (0 = not vectorised)
cp libops_skel.so ../root/dsp/cdsp/
