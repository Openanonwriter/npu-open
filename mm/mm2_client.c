/* Sends a 32x32 activation and weight matrix with distinct values to the accelerator and compares the
 * result with a CPU reference:  out[s][o] = sum_i act[s][i] * wt[o][i].
 * Tile element order (hypothesis from the public HMX manual): 128-byte vectors hold two interleaved rows.
 *   activation index p: s = 2*(p/64) + p%2, i = (p%64)/2
 *   weight index q:     i = 2*(q/64) + q%2, o = (q%64)/2
 *   output index r:     s = 2*(r/64) + r%2, o = (r%64)/2 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>
#include <libhexagonrpc/hexagonrpc.h>
#include <libhexagonrpc/session.h>
#include <libhexagonrpc/interface/remotectl.h>
struct tile { uint16_t v[1024]; };
struct res { uint32_t st[16]; uint16_t tile[1024]; };
static struct hrpc_arg_def_interp4 args[] = { { HRPC_ARG_WORD, 4 }, { HRPC_ARG_WORD, 4 }, { HRPC_ARG_WORD, 4 }, { HRPC_ARG_WORD, 4 },
	{ HRPC_ARG_BLOB, sizeof(struct tile) }, { HRPC_ARG_BLOB, sizeof(struct tile) }, { HRPC_ARG_OUT_BLOB, sizeof(struct res) } };
static const struct hrpc_method_def_interp4 def = { .msg_id = 1, .n_args = 7, .args = args };
static uint16_t h(float f) { _Float16 x = (_Float16)f; uint16_t u; memcpy(&u, &x, 2); return u; }
static float f(uint16_t u) { _Float16 x; memcpy(&x, &u, 2); return (float)x; }
int main(void)
{
	static struct tile ta, tw; static struct res o; static float A[32][32], W[32][32], R[32][32];
	char err[256] = ""; uint32_t hdl = 0, dlerr = 0; int ret, s, i, oc, bad = 0, shown = 0;
	int fd = hexagonrpc_fd_from_env();
	if (fd < 0) return 1;
	srand(getenv("SEED") ? atoi(getenv("SEED")) : 1);
	for (s = 0; s < 32; s++) for (i = 0; i < 32; i++) { A[s][i] = (rand() % 9) - 4; W[s][i] = ((rand() % 17) - 8) * 0.25f; }
	for (s = 0; s < 32; s++) for (oc = 0; oc < 32; oc++) { R[s][oc] = 0; for (i = 0; i < 32; i++) R[s][oc] += A[s][i] * W[oc][i]; }
	for (int p = 0; p < 1024; p++) {
		ta.v[p] = h(A[2 * (p / 64) + p % 2][(p % 64) / 2]);
		tw.v[p] = h(W[(p % 64) / 2][2 * (p / 64) + p % 2]);
	}
	sleep(2);
	ret = remotectl_open(fd, strlen("mm") + 1, "mm", &hdl, sizeof(err), err, &dlerr);
	if (ret || dlerr) { fprintf(stderr, "mm2: open failed ret=0x%x dlerr=0x%x\n", ret, dlerr); return 2; }
	ret = hexagonrpc(&def, fd, hdl, (uint32_t)7, (uint32_t)(getenv("HI") ? strtoul(getenv("HI"), NULL, 0) : 0), (uint32_t)(getenv("CVT") ? strtoul(getenv("CVT"), NULL, 0) : 0), (uint32_t)(getenv("LO") ? strtoul(getenv("LO"), NULL, 0) : 0x3c00), &ta, &tw, &o);
	fprintf(stderr, "mm2: invoke=0x%x power=%x lock=%x reached=%x unlock=%x\n", ret, o.st[8], o.st[9], o.st[10], o.st[11]);
	for (int r = 0; r < 1024; r++) {
		s = 2 * (r / 64) + r % 2; oc = (r % 64) / 2;
		if (f(o.tile[r]) != R[s][oc]) { bad++; if (shown++ < 6) fprintf(stderr, "mm2: mismatch r=%d (s=%d,o=%d): npu=%g cpu=%g\n", r, s, oc, f(o.tile[r]), R[s][oc]); }
	}
	fprintf(stderr, "mm2: %d of 1024 cells match the CPU reference (sample: npu %g %g %g | cpu %g %g %g)\n", 1024 - bad,
		f(o.tile[0]), f(o.tile[1]), f(o.tile[2]), R[0][0], R[1][0], R[0][1]);
	return 0;
}
