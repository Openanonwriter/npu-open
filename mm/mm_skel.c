/* Clean-room HMX FP16 matrix multiply test for the Hexagon cDSP. See ../hmx/hmx_skel.c for provenance notes;
 * data layouts and register fields follow Qualcomm's public HMX Programmer's Reference Manual (80-N2040-62). */
typedef unsigned int u32;
typedef unsigned short u16;
typedef union { struct { void *pv; u32 len; } buf; u32 h; } remote_arg;

extern int compute_resource_attr_init(void *attr);
extern int compute_resource_attr_set_hmx_param(void *attr, unsigned char enable);
extern int compute_resource_attr_set_vtcm_param_v2(void *attr, u32 size, u32 min_page, u32 min_size);
extern int compute_resource_attr_get_vtcm_ptr_v2(void *attr, void **ptr, u32 *size);
extern u32 compute_resource_acquire(void *attr, u32 timeout_us);
extern int compute_resource_release(u32 ctx);
extern int compute_resource_hmx_lock(u32 ctx);
extern int compute_resource_hmx_unlock(u32 ctx);
extern int HAP_power_set(void *context, void *request);
extern void *memset(void *, int, unsigned);

void hmx_clracc(void);
void hmx_bias_load(u32 rs);
void hmx_mac(u32 act_rs, u32 act_rt, u32 wt_rs, u32 wt_rt);
void hmx_cvt(u32 rs);
void hmx_store(u32 rs, u32 rt);

#define POWER_TYPE_HMX 13            /* found by probing; request = { type, power_up } */
#define VTCM_SIZE (1024 * 1024)
#define TILE 2048

static unsigned long long attr_buf[64];
static unsigned long long req_buf[64];

/* method 0 in: { stage, act_fp16, wt_fp16, scale_fp16 } (uniform tiles)
 * method 1 in: same four words (act/wt ignored) + a 2 KB activation tile + a 2 KB weight tile
 * out: 16 status words + 2048 bytes of output tile */
int mm_skel_invoke(u32 sc, remote_arg *pra)
{
	const u32 *in = pra[0].buf.pv;
	u32 *st = pra[1].buf.pv;
	u16 *res = (u16 *)(st + 16);
	u32 stage, ctx, vsize = 0, i, *r = (u32 *)req_buf;
	void *vtcm = 0;
	u16 *act, *wt, *out;
	u32 *bias;

	u32 method = (sc >> 24) & 0x1f, n_in = (sc >> 16) & 0xff;
	const u16 *src_act = 0, *src_wt = 0;

	if (method > 1 || pra[0].buf.len < 16 || pra[n_in].buf.len < 64 + TILE)
		return 0x0e;
	st = pra[n_in].buf.pv; res = (u16 *)(st + 16);
	if (method == 1) {
		/* caller-supplied tiles: either separate buffers or packed after the four words */
		if (n_in >= 3 && pra[1].buf.len >= TILE && pra[2].buf.len >= TILE) { src_act = pra[1].buf.pv; src_wt = pra[2].buf.pv; }
		else if (pra[0].buf.len >= 16 + 2 * TILE) { src_act = (const u16 *)(in + 4); src_wt = src_act + TILE / 2; }
		else return 0x0e;
	}
	stage = in[0];
	for (i = 0; i < 16; i++) st[i] = 0xdeadbeef;

	st[0] = (u32)compute_resource_attr_init(attr_buf);
	st[1] = (u32)compute_resource_attr_set_hmx_param(attr_buf, 1);
	st[2] = (u32)compute_resource_attr_set_vtcm_param_v2(attr_buf, VTCM_SIZE, VTCM_SIZE, VTCM_SIZE);
	if (stage < 1) return 0;
	st[3] = ctx = compute_resource_acquire(attr_buf, 100000);
	if (!ctx) return 0;
	st[4] = (u32)compute_resource_attr_get_vtcm_ptr_v2(attr_buf, &vtcm, &vsize);
	st[5] = (u32)vtcm; st[6] = vsize;
	if (stage < 2 || st[4] != 0 || !vtcm || vsize < 8 * TILE || ((u32)vtcm & (TILE - 1))) goto release;

	act = vtcm; wt = act + TILE / 2; bias = (u32 *)(wt + TILE / 2); out = (u16 *)((char *)vtcm + 3 * TILE);
	for (i = 0; i < TILE / 2; i++) {
		act[i] = src_act ? src_act[i] : (u16)in[1];
		wt[i] = src_wt ? src_wt[i] : (u16)in[2];
		out[i] = 0xaaaa;
	}
	for (i = 0; i < 32; i++) { bias[i] = in[3]; bias[32 + i] = method == 1 ? in[1] : 0; }  /* low word: scale + output bias; high word: shape etc. */
	st[7] = 1;
	if (stage < 3) goto release;

	memset(req_buf, 0, sizeof(req_buf));
	r[0] = POWER_TYPE_HMX; r[1] = 1; r[2] = 1;
	st[8] = (u32)HAP_power_set(attr_buf, req_buf);
	if (st[8] != 0) goto release;
	st[9] = (u32)compute_resource_hmx_lock(ctx);
	if (st[9] != 0) goto release;

	if (method == 1) {
		/* Observed on this v73 part: the convert output is clamped at zero after scale and output bias.
		 * Two passes over the same accumulator, with scale +1 and -1, give max(x,0) and max(-x,0);
		 * exactly one of them is non-zero per cell, so the signed result is rebuilt with integer ops. */
		u32 *bias_neg = bias + 64;
		u16 *out_neg = out + TILE / 2;
		for (i = 0; i < 32; i++) { bias[i] = 0x3c00; bias[32 + i] = 0; bias_neg[i] = 0xbc00; bias_neg[32 + i] = 0; }
		/* Only one bias set appears to exist here: the set index in the load is ignored, so reload between passes. */
		hmx_clracc();
		hmx_mac((u32)act, 2047, (u32)wt, 2047);
		hmx_bias_load((u32)bias);            /* scale +1.0 */
		hmx_cvt(1);                          /* convert, retain accumulator */
		hmx_store((u32)out, 0x782);
		hmx_bias_load((u32)bias_neg);        /* scale -1.0 */
		hmx_cvt(0);                          /* convert, clear accumulator */
		hmx_store((u32)out_neg, 0x782);
		st[10] = 8;
		st[11] = (u32)compute_resource_hmx_unlock(ctx);
		for (i = 0; i < TILE / 2; i++)
			res[i] = out[i] ? out[i] : (out_neg[i] ? (u16)(out_neg[i] | 0x8000) : 0);
		goto release;
	}

	if (stage >= 4) { hmx_bias_load((u32)bias); st[10] = 4; }
	if (stage >= 5) { hmx_clracc(); hmx_mac((u32)act, 2047, (u32)wt, 2047); st[10] = 5; }
	if (stage >= 6) { hmx_cvt(0); st[10] = 6; }
	if (stage >= 7) { hmx_store((u32)out, 0x782); st[10] = 7; }

	st[11] = (u32)compute_resource_hmx_unlock(ctx);
	for (i = 0; i < TILE / 2; i++) res[i] = out[i];
release:
	st[12] = (u32)compute_resource_release(ctx);
	return 0;
}
