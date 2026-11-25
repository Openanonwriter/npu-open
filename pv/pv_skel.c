/* Power-vote probe: time a fixed scalar loop on the DSP, send one HAP_power_set request, time the loop again.
 * A vote that raises the DSP clock shows up as a shorter second time. No Qualcomm headers or code. */
typedef unsigned int u32;
typedef unsigned long long u64;
typedef union { struct { void *pv; u32 len; } buf; u32 h; } remote_arg;
extern int HAP_power_set(void *context, void *request);
extern u64 HAP_perf_get_time_us(void);
extern void *memset(void *, int, unsigned);
extern void *memcpy(void *, const void *, unsigned);

static unsigned long long req_buf[64];
static int ctx_token;

__attribute__((noinline)) static u32 spin(u32 n)
{
	u32 x = 12345, i;
	for (i = 0; i < n; i++) x = x * 1664525u + 1013904223u + (x >> 7);
	return x;
}
static u32 timed(u32 n, u32 *sink)
{
	u64 t0 = HAP_perf_get_time_us();
	*sink += spin(n);
	return (u32)(HAP_perf_get_time_us() - t0);
}
/* in: { iterations, do_vote, 16 request words }   out: { us_before, vote_ret, us_after, us_after2, sink } */
int pv_skel_invoke(u32 sc, remote_arg *pra)
{
	const u32 *in = pra[0].buf.pv;
	u32 *out = pra[1].buf.pv;
	if (((sc >> 24) & 0x1f) != 0 || pra[0].buf.len < 18 * 4 || pra[1].buf.len < 20) return 0x0e;
	out[4] = 0;
	out[0] = timed(in[0], &out[4]);
	out[1] = 0xdeadbeef;
	if (in[1]) {
		memset(req_buf, 0, sizeof(req_buf));
		memcpy(req_buf, in + 2, 16 * 4);
		out[1] = (u32)HAP_power_set(&ctx_token, req_buf);
	}
	out[2] = timed(in[0], &out[4]);
	out[3] = timed(in[0], &out[4]);
	return 0;
}
