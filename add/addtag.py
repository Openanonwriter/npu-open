#!/usr/bin/python3
"""Rewrite entries in a 32-bit little-endian ELF's .dynamic: addtag.py file OLDTAG:NEWTAG=VALUE ..."""
import struct, sys
path = sys.argv[1]
b = bytearray(open(path, 'rb').read())
phoff, = struct.unpack_from('<I', b, 28); phentsize, phnum = struct.unpack_from('<HH', b, 42)
for i in range(phnum):
    t, off, _, _, filesz = struct.unpack_from('<IIIII', b, phoff + i * phentsize)
    if t == 2: break
else: sys.exit('no PT_DYNAMIC')
for a in sys.argv[2:]:
    old, rest = a.split(':'); new, val = rest.split('=')
    old, new, val = int(old, 0), int(new, 0), int(val, 0)
    for e in range(off, off + filesz, 8):
        if struct.unpack_from('<I', b, e)[0] == old:
            struct.pack_into('<II', b, e, new, val); break
    else: sys.exit('tag %#x not found' % old)
open(path, 'wb').write(b)
