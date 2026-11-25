/* Clean-room linear layer on the Hexagon HMX accelerator (FP16):  out[s][o] = f(sum_i act[s][i] * wt[o][i] + bias[o])
 * for 32 samples (s), 32 outputs (o) and 32*nk inputs (i), accumulated over nk tiles.
 * mode 0: f = max(x, 0), one convert pass (the chip clamps at zero by itself).
 * mode 1: f = identity, rebuilt from two convert passes with scale +1 and -1.
 * Provenance as in ../mm/mm_skel.c: no Qualcomm headers or code. */
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
extern void *memcpy(void *, const void *, unsigned);

void hmx_clracc(void);
void hmx_bias_load(u32 rs);
void hmx_mac(u32 act_rs, u32 act_rt, u32 wt_rs, u32 wt_rt);
void hmx_cvt(u32 rs);
void hmx_store(u32 rs, u32 rt);

#define POWER_TYPE_HMX 13
#define VTCM_SIZE (1024 * 1024)
#define TILE 2048
#define MAX_NK 64

static unsigned long long attr_buf[64];
static unsigned long long req_buf[64];

/* in: words { nk, mode, 0, 0 }, activation tiles (nk * 2 KB), weight tiles (nk * 2 KB), 32 FP16 biases
 * out: 16 status words + one 2 KB output tile */
int nn_skel_invoke(u32 sc, remote_arg *pra)
{
	u32 n_in = (sc >> 16) & 0xff, nk, mode, ctx, vsize = 0, i, t, *st, *r = (u32 *)req_buf, *bias, *bias_neg;
	const u32 *in = pra[0].buf.pv;
	const unsigned char *src_act, *src_wt;
	const u16 *src_bias;
	u16 *res, *out, *out_neg;
	char *act, *wt;
	void *vtcm = 0;

	if (((sc >> 24) & 0x1f) != 0 || pra[0].buf.len < 16 || pra[n_in].buf.len < 64 + TILE)
		return 0x0e;
	nk = in[0]; mode = in[1];
	if (nk < 1 || nk > MAX_NK) return 0x0e;
	if (n_in >= 4 && pra[1].buf.len >= nk * TILE && pra[2].buf.len >= nk * TILE && pra[3].buf.len >= 64) {
		src_act = pra[1].buf.pv; src_wt = pra[2].buf.pv; src_bias = pra[3].buf.pv;
	} else if (pra[0].buf.len >= 16 + 2 * nk * TILE + 64) {
		src_act = (const unsigned char *)(in + 4); src_wt = src_act + nk * TILE; src_bias = (const u16 *)(src_wt + nk * TILE);
	} else return 0x0e;
	st = pra[n_in].buf.pv; res = (u16 *)(st + 16);
	for (i = 0; i < 16; i++) st[i] = 0xdeadbeef;
	st[13] = n_in;

	st[0] = (u32)compute_resource_attr_init(attr_buf);
	st[1] = (u32)compute_resource_attr_set_hmx_param(attr_buf, 1);
	st[2] = (u32)compute_resource_attr_set_vtcm_param_v2(attr_buf, VTCM_SIZE, VTCM_SIZE, VTCM_SIZE);
	st[3] = ctx = compute_resource_acquire(attr_buf, 100000);
	if (!ctx) return 0;
	st[4] = (u32)compute_resource_attr_get_vtcm_ptr_v2(attr_buf, &vtcm, &vsize);
	if (st[4] != 0 || !vtcm || vsize < (2 * nk + 3) * TILE || ((u32)vtcm & (TILE - 1))) goto release;

	act = vtcm; wt = act + nk * TILE;
	bias = (u32 *)(wt + nk * TILE); bias_neg = bias + 64;
	out = (u16 *)((char *)bias + TILE); out_neg = out + TILE / 2;
	memcpy(act, src_act, nk * TILE);
	memcpy(wt, src_wt, nk * TILE);
	for (i = 0; i < 32; i++) {
		bias[i] = 0x3c00 | ((u32)src_bias[i] << 16);                 /* scale +1.0, output bias b */
		bias_neg[i] = 0xbc00 | ((u32)(src_bias[i] ^ 0x8000) << 16);   /* scale -1.0, output bias -b */
		bias[32 + i] = bias_neg[32 + i] = 0;
	}

	memset(req_buf, 0, sizeof(req_buf));
	r[0] = POWER_TYPE_HMX; r[1] = 1; r[2] = 1;
	st[8] = (u32)HAP_power_set(attr_buf, req_buf);
	if (st[8] != 0) goto release;
	st[9] = (u32)compute_resource_hmx_lock(ctx);
	if (st[9] != 0) goto release;

	hmx_clracc();
	for (t = 0; t < nk; t++)
		hmx_mac((u32)(act + t * TILE), 2047, (u32)(wt + t * TILE), 2047);
	hmx_bias_load((u32)bias);
	if (mode == 0) {
		hmx_cvt(0);
		hmx_store((u32)out, 0x782);
	} else {
		hmx_cvt(1);                    /* retain accumulator for the second pass */
		hmx_store((u32)out, 0x782);
		hmx_bias_load((u32)bias_neg);
		hmx_cvt(0);
		hmx_store((u32)out_neg, 0x782);
	}
	st[11] = (u32)compute_resource_hmx_unlock(ctx);
	for (i = 0; i < TILE / 2; i++)
		res[i] = (mode == 0 || out[i]) ? out[i] : (out_neg[i] ? (u16)(out_neg[i] | 0x8000) : 0);
	st[10] = 1;
release:
	st[12] = (u32)compute_resource_release(ctx);
	return 0;
}
