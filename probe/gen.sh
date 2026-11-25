#!/usr/bin/bash
# For each name in names.txt build libp<N>_skel.so that needs that symbol from the DSP shell at load time.
set -eu; cd "$(dirname "$0")"; i=0
while read -r sym; do
  printf 'extern void %s(void);\nvoid (*const volatile probe_ref)(void) = %s;\nint p%d_skel_invoke(unsigned sc, void *pra) { return probe_ref ? 0 : 1; }\n' "$sym" "$sym" $i > p$i.c
  clang --target=hexagon-unknown-none-elf -mv73 -O2 -fPIC -G0 -ffreestanding -fno-builtin -c p$i.c -o p$i.o
  ld.lld -shared --hash-style=sysv -z max-page-size=4096 -z norelro --no-rosegment -z separate-loadable-segments -Bsymbolic -o libp${i}_skel.so p$i.o
  cp libp${i}_skel.so ../root/dsp/cdsp/; i=$((i+1))
done < names.txt
echo "built $i probe libraries"
