/* Clean-room FastRPC skeleton for the Hexagon cDSP. No Qualcomm headers or code.
 * The DSP shell resolves "<name>_skel_invoke" in "lib<name>_skel.so" (documented in
 * the hexagonrpc README) and calls it with the scalars word and the argument array. */
typedef unsigned int u32;
typedef union { struct { void *pv; u32 len; } buf; u32 h; } remote_arg;

#define METHOD(sc) (((sc) >> 24) & 0x1f)
#define N_IN(sc)   (((sc) >> 16) & 0xff)
#define N_OUT(sc)  (((sc) >> 8) & 0xff)

int add_skel_invoke(u32 sc, remote_arg *pra)
{
	const u32 *in;
	u32 *out;

	if (METHOD(sc) != 0 || N_IN(sc) != 1 || N_OUT(sc) != 1)
		return 0x14; /* unsupported */
	if (pra[0].buf.len < 8 || pra[1].buf.len < 4)
		return 0x0e; /* bad parameter */
	in = pra[0].buf.pv;
	out = pra[1].buf.pv;
	out[0] = in[0] + in[1];
	return 0;
}
