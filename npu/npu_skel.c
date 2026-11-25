/* Clean-room fully connected network runner for the Hexagon HMX accelerator (FP16).
 * The accelerator is acquired and powered once (open), weights are loaded once into VTCM (load), and each
 * run call pushes one or more batches of 32 samples through every layer on the DSP.
 *
 * A layer has nk input tiles (32 input channels each) and ng output groups (32 outputs each); group g of
 * layer l writes output tile g, which is input tile g of layer l+1, so ng(l) must equal nk(l+1).
 * mode 0 = ReLU (the chip clamps at zero by itself), mode 1 = signed output (two convert passes).
 * Provenance as in ../mm/mm_skel.c: no Qualcomm headers or code. */
typedef unsigned int u32;
typedef unsigned short u16;
typedef unsigned long long u64;
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
extern u64 HAP_perf_get_time_us(void);
extern int qurt_hvx_lock(int mode);      /* mode 1 = 128-byte vectors (value assumed; return code is checked) */
extern int qurt_hvx_unlock(void);
void hvx_copy(void *dst, const void *src, unsigned n);
extern void *memset(void *, int, unsigned);
extern void *memcpy(void *, const void *, unsigned);

void hmx_clracc(void);
void hmx_bias_load(u32 rs);
void hmx_mac(u32 act_rs, u32 act_rt, u32 wt_rs, u32 wt_rt);
void hmx_mac_deep(u32 act_rs, u32 act_rt, u32 wt_rs, u32 wt_rt);
void hmx_cvt(u32 rs);
void hmx_store(u32 rs, u32 rt);

#define POWER_TYPE_HMX 13
#define TILE 2048
#define MAX_LAYERS 8
#define E_PARM 0x0e
#define E_STATE 0x0d

struct layer { u32 nk, ng, mode; char *wt; u32 *bias; char *in; };

static unsigned long long attr_buf[64];
static unsigned long long req_buf[64];
static struct { u32 ctx, vsize, n_layers, used; char *vtcm; char *scratch; char *out; char *outneg; struct layer l[MAX_LAYERS]; } S;

static char *valloc(u32 bytes)
{
	char *p;
	bytes = (bytes + TILE - 1) & ~(TILE - 1);
	if (!S.vtcm || S.used + bytes > S.vsize) return 0;
	p = S.vtcm + S.used; S.used += bytes;
	return p;
}

/* Blob k of a call: either its own buffer (index 1 + k) or packed after the leading words. */
static const void *blob(remote_arg *pra, u32 n_in, u32 n_words, u32 k, const u32 *sizes)
{
	const char *p = (const char *)pra[0].buf.pv + 4 * n_words;
	u32 i, off = 0;
	if (n_in >= 2) return pra[1 + k].buf.len >= sizes[k] ? pra[1 + k].buf.pv : 0;
	for (i = 0; i < k; i++) off += sizes[i];
	return pra[0].buf.len >= 4 * n_words + off + sizes[k] ? p + off : 0;
}

/* method 0: open.  in { vtcm_bytes, n_layers, nk[0..7], ng[0..7], mode[0..7], mips, latency_us }  out { status[8] }
 * mips: if non-zero, a MIPS performance request. Any accepted vote (including the HMX power-on) otherwise drops
 * the DSP to a low clock: a fixed scalar loop went from 8.5 ms to 22.7 ms, and to 7.1 ms with mips >= 1000.
 * Request layout found by test: { u32 type = 1; u32 unused; u8 set_mips @8; u32 mips_per_thread @12; u32 mips_total @16 }.
 * latency_us: if non-zero, also sets the guessed sleep-latency fields { u8 set_latency @34; i32 latency @36 }. */
static int do_open(const u32 *in, u32 *st)
{
	u32 i, *r = (u32 *)req_buf, vt = in[0];
	void *p = 0;

	if (S.ctx) return E_STATE;
	if (in[1] < 1 || in[1] > MAX_LAYERS) return E_PARM;
	st[0] = (u32)compute_resource_attr_init(attr_buf);
	st[1] = (u32)compute_resource_attr_set_hmx_param(attr_buf, 1);
	st[2] = (u32)compute_resource_attr_set_vtcm_param_v2(attr_buf, vt, vt, vt);
	st[3] = S.ctx = compute_resource_acquire(attr_buf, 200000);
	if (!S.ctx) return 0;
	st[4] = (u32)compute_resource_attr_get_vtcm_ptr_v2(attr_buf, &p, &S.vsize);
	S.vtcm = p; S.used = 0;
	st[5] = (u32)p; st[6] = S.vsize;
	memset(req_buf, 0, sizeof(req_buf));
	r[0] = POWER_TYPE_HMX; r[2] = 1;   /* request = { type, (unused here), power_up at offset 8 }: isolated by test */
	st[7] = (u32)HAP_power_set(attr_buf, req_buf);
	if (st[4] || !p || ((u32)p & (TILE - 1)) || st[7]) goto fail;
	if (in[26]) {
		memset(req_buf, 0, sizeof(req_buf));
		r[0] = 1; r[2] = 1; r[3] = in[26]; r[4] = in[26];
		if (in[27]) { r[8] = 1 << 16; r[9] = in[27]; }
		st[0] = (u32)HAP_power_set(attr_buf, req_buf);   /* reported in status[0] */
	}

	S.n_layers = in[1];
	for (i = 0; i < S.n_layers; i++) {
		struct layer *l = &S.l[i];
		l->nk = in[2 + i]; l->ng = in[10 + i]; l->mode = in[18 + i];
		if (!l->nk || !l->ng || (i && l->nk != S.l[i - 1].ng)) goto fail;
		l->wt = valloc(l->nk * l->ng * TILE);
		l->bias = (u32 *)valloc(l->ng * 512);            /* per group: 256 B scale +1 block, 256 B scale -1 block */
		l->in = i ? 0 : valloc(l->nk * TILE);
		if (!l->wt || !l->bias || (!i && !l->in)) goto fail;
		memset(l->wt, 0, l->nk * l->ng * TILE);
	}
	for (i = 0; i < S.n_layers; i++) {                       /* each layer's output area is the next layer's input */
		char *o = valloc(S.l[i].ng * TILE);
		if (!o) goto fail;
		if (i + 1 < S.n_layers) S.l[i + 1].in = o; else S.out = o;
	}
	S.scratch = valloc(TILE);
	S.outneg = valloc(S.l[S.n_layers - 1].ng * TILE);      /* second-pass tiles of the last layer, for flag bit 1 */
	if (!S.scratch || !S.outneg) goto fail;
	return 0;
fail:
	compute_resource_release(S.ctx);
	memset(&S, 0, sizeof(S));
	st[3] = 0;
	return 0;
}

/* method 1: load.  in { layer, group } + nk weight tiles + 32 FP16 biases */
static int do_load(remote_arg *pra, u32 n_in)
{
	const u32 *in = pra[0].buf.pv;
	struct layer *l;
	u32 sizes[2], i, *b;
	const void *w; const u16 *sb;

	if (!S.ctx || in[0] >= S.n_layers) return E_STATE;
	l = &S.l[in[0]];
	if (in[1] >= l->ng) return E_PARM;
	sizes[0] = l->nk * TILE; sizes[1] = 64;
	w = blob(pra, n_in, 2, 0, sizes); sb = blob(pra, n_in, 2, 1, sizes);
	if (!w || !sb) return E_PARM;
	memcpy(l->wt + in[1] * l->nk * TILE, w, l->nk * TILE);
	b = l->bias + in[1] * 128;
	for (i = 0; i < 32; i++) {
		b[i] = 0x3c00 | ((u32)sb[i] << 16);                  /* scale +1.0, output bias b */
		b[64 + i] = 0xbc00 | ((u32)(sb[i] ^ 0x8000) << 16);  /* scale -1.0, output bias -b */
		b[32 + i] = b[96 + i] = 0;
	}
	return 0;
}

/* method 2: run.  in { n_batches, flags } + n_batches * nk[0] activation tiles.  flags bit 0: one "deep" multiply
 * per up-to-32 tiles instead of one multiply per tile.  flags bit 2: copy tiles with HVX vectors (falls back to
 * memcpy if the HVX lock is refused; status[2] = lock return code).  flags bit 1: for a signed last layer return both convert
 * passes (ng tiles of max(x,0), then ng tiles of max(-x,0)) and let the caller merge them.
 * out { status[8], padding to 128 bytes } + n_batches * ng[last] output tiles.  status[1] = total microseconds,
 * status[4..7] = microseconds in copy-in, multiply+convert, sign merge, copy-out. */
static int do_run(remote_arg *pra, u32 n_in)
{
	const u32 *in = pra[0].buf.pv;
	u32 *st = pra[n_in].buf.pv, nb = in[0], b, li, g, t, i, sizes[1], in_bytes, out_bytes;
	const char *src; char *dst = (char *)st + 128;   /* 128-byte header keeps the tiles vector-aligned */
	u64 t0, t1, ta, tb, us_in = 0, us_mul = 0, us_merge = 0, us_out = 0;
	u32 deep = in[1] & 1, raw = (in[1] >> 1) & 1, hvx = (in[1] >> 2) & 1;

	if (!S.ctx) return E_STATE;
	in_bytes = S.l[0].nk * TILE; out_bytes = S.l[S.n_layers - 1].ng * TILE;
	if (raw && S.l[S.n_layers - 1].mode == 1) out_bytes *= 2; else raw = 0;
	sizes[0] = nb * in_bytes;
	src = blob(pra, n_in, 2, 0, sizes);
	if (!nb || !src || pra[n_in].buf.len < 128 + nb * out_bytes) return E_PARM;
	st[2] = 0xffff;
	if (hvx) {
		st[2] = (u32)qurt_hvx_lock(1);
		if (st[2] || (nb * in_bytes) % 128 || ((u32)src & 127) || ((u32)dst & 127)) hvx = 0;
	}
	st[0] = (u32)compute_resource_hmx_lock(S.ctx);
	if (st[0]) { if (st[2] == 0) qurt_hvx_unlock(); return 0; }
	t0 = HAP_perf_get_time_us();
	for (b = 0; b < nb; b++) {
		ta = HAP_perf_get_time_us();
		if (hvx) hvx_copy(S.l[0].in, src + b * in_bytes, in_bytes); else memcpy(S.l[0].in, src + b * in_bytes, in_bytes);
		tb = HAP_perf_get_time_us(); us_in += tb - ta;
		for (li = 0; li < S.n_layers; li++) {
			struct layer *l = &S.l[li];
			char *o = li + 1 < S.n_layers ? S.l[li + 1].in : S.out;
			for (g = 0; g < l->ng; g++) {
				char *w = l->wt + g * l->nk * TILE, *og = o + g * TILE;
				u32 *bias = l->bias + g * 128;
				ta = HAP_perf_get_time_us();
				hmx_clracc();
				if (deep) {
					for (t = 0; t < l->nk; t += 32) {
						u32 n = l->nk - t < 32 ? l->nk - t : 32;
						hmx_mac_deep((u32)(l->in + t * TILE), ((n - 1) << 11) | 2047, (u32)(w + t * TILE), n * TILE - 1);
					}
				} else {
					for (t = 0; t < l->nk; t++)
						hmx_mac((u32)(l->in + t * TILE), 2047, (u32)(w + t * TILE), 2047);
				}
				hmx_bias_load((u32)bias);
				if (l->mode == 0) {
					hmx_cvt(0);
					hmx_store((u32)og, 0x782);
					us_mul += HAP_perf_get_time_us() - ta;
				} else {
					u16 *pos = (u16 *)og, *neg = (u16 *)((raw && li + 1 == S.n_layers) ? S.outneg + g * TILE : S.scratch);
					hmx_cvt(1);
					hmx_store((u32)og, 0x782);
					hmx_bias_load((u32)(bias + 64));
					hmx_cvt(0);
					hmx_store((u32)neg, 0x782);
					tb = HAP_perf_get_time_us(); us_mul += tb - ta;
					if (!(raw && li + 1 == S.n_layers))
						for (i = 0; i < TILE / 2; i++)
							if (!pos[i] && neg[i]) pos[i] = (u16)(neg[i] | 0x8000);
					us_merge += HAP_perf_get_time_us() - tb;
				}
			}
		}
		ta = HAP_perf_get_time_us();
		if (raw) {
			if (hvx) { hvx_copy(dst + b * out_bytes, S.out, out_bytes / 2); hvx_copy(dst + b * out_bytes + out_bytes / 2, S.outneg, out_bytes / 2); }
			else { memcpy(dst + b * out_bytes, S.out, out_bytes / 2); memcpy(dst + b * out_bytes + out_bytes / 2, S.outneg, out_bytes / 2); }
		} else {
			if (hvx) hvx_copy(dst + b * out_bytes, S.out, out_bytes); else memcpy(dst + b * out_bytes, S.out, out_bytes);
		}
		us_out += HAP_perf_get_time_us() - ta;
	}
	t1 = HAP_perf_get_time_us();
	st[1] = (u32)(t1 - t0);
	if (st[2] == 0) qurt_hvx_unlock();
	st[4] = (u32)us_in; st[5] = (u32)us_mul; st[6] = (u32)us_merge; st[7] = (u32)us_out;
	st[3] = (u32)compute_resource_hmx_unlock(S.ctx);
	return 0;
}

int npu_skel_invoke(u32 sc, remote_arg *pra)
{
	u32 method = (sc >> 24) & 0x1f, n_in = (sc >> 16) & 0xff, n_out = (sc >> 8) & 0xff, i;
	u32 *st = n_out ? pra[n_in].buf.pv : 0;

	switch (method) {
	case 0:
		if (pra[0].buf.len < 28 * 4 || !st || pra[n_in].buf.len < 32) return E_PARM;
		for (i = 0; i < 8; i++) st[i] = 0xdeadbeef;
		return do_open(pra[0].buf.pv, st);
	case 1:
		return pra[0].buf.len < 8 ? E_PARM : do_load(pra, n_in);
	case 2:
		if (pra[0].buf.len < 8 || !st || pra[n_in].buf.len < 32) return E_PARM;
		for (i = 0; i < 8; i++) st[i] = 0xdeadbeef;
		return do_run(pra, n_in);
	case 3:
		if (S.ctx) compute_resource_release(S.ctx);
		memset(&S, 0, sizeof(S));
		return 0;
	}
	return 0x14;
}
