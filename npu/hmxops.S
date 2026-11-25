/* HMX FP16 instructions emitted as raw words. Encodings from LLVM's Apache-licensed
 * HexagonDepInstrInfo.td (M8_* definitions); bits 15:14 are the packet parse bits. */
	.text
	.macro FN name
	.globl \name
	.type \name, @function
	.p2align 4
\name:
	.endm

FN hmx_clracc              /* mxclracc.hf */
	.word 0xa6e0c013
	jumpr r31

FN hmx_bias_load           /* bias = mxmem2(r0) */
	.word 0x9200c3fe
	jumpr r31

FN hmx_mac                 /* { activation.hf = mxmem(r0,r1); weight.hf = mxmem(r2,r3) } */
	.word 0x920041e4
	.word 0x9202e3ef
	jumpr r31

FN hmx_cvt                 /* cvt.hf = acc(r0) */
	.word 0xa6e0da10
	jumpr r31

FN hmx_store               /* mxmem(r0,r1) = cvt */
	.word 0xa6e0c118
	jumpr r31

FN hmx_mac_deep            /* { activation.hf = mxmem(r0,r1):deep; weight.hf = mxmem(r2,r3) } : up to 32 contiguous tiles */
	.word 0x920041e2
	.word 0x9202e3ef
	jumpr r31
