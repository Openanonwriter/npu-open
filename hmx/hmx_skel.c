/* Clean-room HMX bring-up on the Hexagon cDSP.
 * Function names were found by black-box symbol probing (see ../probe); argument shapes follow the
 * usage visible in llama.cpp's MIT-licensed ggml-hexagon backend. Instruction encodings come from
 * LLVM's Apache-licensed HexagonDepInstrInfo.td. No Qualcomm headers or code are used. */
typedef unsigned int u32;
typedef union { struct { void *pv; u32 len; } buf; u32 h; } remote_arg;

extern int compute_resource_attr_init(void *attr);
extern int compute_resource_attr_set_hmx_param(void *attr, unsigned char enable);
extern u32 compute_resource_acquire(void *attr, u32 timeout_us);
extern int compute_resource_release(u32 ctx);
extern int compute_resource_hmx_lock(u32 ctx);
extern int compute_resource_hmx_unlock(u32 ctx);
extern int HAP_power_set(void *context, void *request);
extern void *memset(void *, int, unsigned);

static unsigned long long attr_buf[64];
static unsigned long long req_buf[64];   /* power request; layout unknown, probed */ /* opaque attribute block; real size unknown, 512 bytes is generous */

/* mxclracc.hf : Inst{31-16}=0xA6E0, Inst{13-0}=0x0013, parse bits 15:14 = 11 (end of packet) */
__attribute__((noinline)) static void hmx_clracc(void) { __asm__ volatile(".word 0xa6e0c013"); }

#define METHOD(sc) (((sc) >> 24) & 0x1f)

int hmx_skel_invoke(u32 sc, remote_arg *pra)
{
	const u32 *in = pra[0].buf.pv;
	u32 *out = pra[1].buf.pv;
	u32 stage, ctx;

	if (METHOD(sc) != 0 || pra[0].buf.len < 12 || pra[1].buf.len < 28)
		return 0x0e;
	stage = in[0];
	out[0] = out[1] = out[2] = out[3] = out[4] = out[5] = out[6] = 0xdeadbeef;

	if (in[1] != 0xffffffff) {
		/* Guess: { u32 type; union { struct { u8 power_up; } ... } } with the union at offset 4 or 8. */
		u32 *r = (u32 *)req_buf;
		memset(req_buf, 0, sizeof(req_buf));
		r[0] = in[1];
		r[1] = in[2] & 1;        /* candidate field at offset 4 */
		r[2] = (in[2] >> 1) & 1; /* candidate field at offset 8 */
		out[6] = (u32)HAP_power_set(attr_buf, req_buf);
		if (out[6] != 0) return 0;
	}

	out[0] = (u32)compute_resource_attr_init(attr_buf);
	if (stage < 1) return 0;
	out[1] = (u32)compute_resource_attr_set_hmx_param(attr_buf, 1);
	if (stage < 2) return 0;
	out[2] = ctx = compute_resource_acquire(attr_buf, 100000);
	if (stage < 3 || !ctx) return 0;
	out[3] = (u32)compute_resource_hmx_lock(ctx);
	if (stage >= 4 && out[3] == 0) {
		hmx_clracc();
		out[5] = 0x600d;
	}
	if (out[3] == 0) out[4] = (u32)compute_resource_hmx_unlock(ctx);
	compute_resource_release(ctx);
	return 0;
}
