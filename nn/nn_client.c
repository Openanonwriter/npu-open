/* Runs the 256-32-10 digit classifier from model.bin on the HMX accelerator, 32 images per batch,
 * and compares every prediction and logit with a float32 CPU reference. */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include <libhexagonrpc/hexagonrpc.h>
#include <libhexagonrpc/session.h>
#include <libhexagonrpc/interface/remotectl.h>
#define TILE 2048
struct res { uint32_t st[16]; uint16_t tile[1024]; };
static uint16_t h(float f) { _Float16 x = (_Float16)f; uint16_t u; memcpy(&u, &x, 2); return u; }
static float f16(uint16_t u) { _Float16 x; memcpy(&x, &u, 2); return (float)x; }
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static int fd; static uint32_t hdl;

/* One accelerator layer call: nk tiles of activations and weights, 32 biases, mode 0 = ReLU, 1 = signed. */
static int layer(uint32_t nk, uint32_t mode, const uint16_t *act, const uint16_t *wt, const uint16_t *bias, struct res *o)
{
	struct hrpc_arg_def_interp4 args[] = { { HRPC_ARG_WORD, 4 }, { HRPC_ARG_WORD, 4 }, { HRPC_ARG_WORD, 4 }, { HRPC_ARG_WORD, 4 },
		{ HRPC_ARG_BLOB, nk * TILE }, { HRPC_ARG_BLOB, nk * TILE }, { HRPC_ARG_BLOB, 64 }, { HRPC_ARG_OUT_BLOB, sizeof(struct res) } };
	struct hrpc_method_def_interp4 def = { .msg_id = 0, .n_args = 8, .args = args };
	int ret = hexagonrpc(&def, fd, hdl, nk, mode, (uint32_t)0, (uint32_t)0, act, wt, bias, o);
	if (ret || o->st[10] != 1) { fprintf(stderr, "nn: layer failed ret=0x%x power=%x lock=%x done=%x bufs=%u\n", ret, o->st[8], o->st[9], o->st[10], o->st[13]); return -1; }
	return 0;
}
/* Tile element order: index p holds row 2*(p/64)+p%2 and column (p%64)/2 (rows = samples for activations and
 * outputs; for weights the roles swap: row = input channel, column = output channel). */
static void pack_weights(uint16_t *t, const float *W, int n_out, int n_in, int nk)
{
	for (int k = 0; k < nk; k++) for (int q = 0; q < 1024; q++) {
		int i = 32 * k + 2 * (q / 64) + q % 2, o = (q % 64) / 2;
		t[k * 1024 + q] = (o < n_out && i < n_in) ? h(W[o * n_in + i]) : 0;
	}
}
int main(void)
{
	int32_t hd[4]; char err[256] = ""; uint32_t dlerr = 0; int ret;
	char path[512];
	snprintf(path, sizeof(path), "%s/nn/model.bin", getenv("NPU_OPEN") ? getenv("NPU_OPEN") : ".");
	FILE *f = fopen(path, "rb");
	fd = hexagonrpc_fd_from_env();
	if (fd < 0 || !f || fread(hd, 4, 4, f) != 4) return 1;
	int NI = hd[0], NH = hd[1], NO = hd[2], NT = hd[3], nk = (NI + 31) / 32;
	float *W1 = malloc(4 * NH * NI), *b1 = malloc(4 * NH), *W2 = malloc(4 * NO * NH), *b2 = malloc(4 * NO), *X = malloc(4 * NT * NI);
	uint8_t *y = malloc(NT);
	if (NH > 32 || NO > 32 || fread(W1, 4, NH * NI, f) != (size_t)(NH * NI) || fread(b1, 4, NH, f) != (size_t)NH || fread(W2, 4, NO * NH, f) != (size_t)(NO * NH) ||
	    fread(b2, 4, NO, f) != (size_t)NO || fread(X, 4, NT * NI, f) != (size_t)(NT * NI) || fread(y, 1, NT, f) != (size_t)NT) return 1;
	uint16_t *w1t = calloc(nk, TILE), *w2t = calloc(1, TILE), *act = calloc(nk, TILE), bias1[32] = { 0 }, bias2[32] = { 0 };
	pack_weights(w1t, W1, NH, NI, nk); pack_weights(w2t, W2, NO, NH, 1);
	for (int o = 0; o < NH; o++) bias1[o] = h(b1[o]);
	for (int o = 0; o < NO; o++) bias2[o] = h(b2[o]);

	sleep(2);
	ret = remotectl_open(fd, strlen("nn") + 1, "nn", &hdl, sizeof(err), err, &dlerr);
	if (ret || dlerr) { fprintf(stderr, "nn: open failed ret=0x%x dlerr=0x%x\n", ret, dlerr); return 2; }

	static struct res hid, logits;
	int ok_npu = 0, ok_cpu = 0, agree = 0, shown = 0; double maxdiff = 0, t_npu = 0, t0;
	int limit = getenv("N") ? atoi(getenv("N")) : NT;
	for (int base = 0; base + 32 <= limit; base += 32) {
		for (int k = 0; k < nk; k++) for (int p = 0; p < 1024; p++) {
			int s = 2 * (p / 64) + p % 2, i = 32 * k + (p % 64) / 2;
			act[k * 1024 + p] = i < NI ? h(X[(base + s) * NI + i]) : 0;
		}
		t0 = now();
		if (layer(nk, 0, act, w1t, bias1, &hid)) return 3;           /* hidden = ReLU(X W1' + b1) */
		if (layer(1, 1, hid.tile, w2t, bias2, &logits)) return 3;    /* logits = hidden W2' + b2; the output tile feeds straight in */
		t_npu += now() - t0;
		for (int s = 0; s < 32; s++) {
			float hcpu[32], lc[32]; int pn = 0, pc = 0;
			for (int o = 0; o < NH; o++) { float a = b1[o]; for (int i = 0; i < NI; i++) a += X[(base + s) * NI + i] * W1[o * NI + i]; hcpu[o] = a > 0 ? a : 0; }
			for (int o = 0; o < NO; o++) { float a = b2[o]; for (int i = 0; i < NH; i++) a += hcpu[i] * W2[o * NH + i]; lc[o] = a; }
			for (int o = 0; o < NO; o++) {
				float ln = f16(logits.tile[64 * (s / 2) + 2 * o + s % 2]);
				if (fabsf(ln - lc[o]) > maxdiff) maxdiff = fabsf(ln - lc[o]);
				if (ln > f16(logits.tile[64 * (s / 2) + 2 * pn + s % 2])) pn = o;
				if (lc[o] > lc[pc]) pc = o;
			}
			ok_npu += pn == y[base + s]; ok_cpu += pc == y[base + s]; agree += pn == pc;
			if (base == 0 && shown++ < 16) fprintf(stderr, "nn: image %2d label %d  npu %d  cpu %d\n", s, y[s], pn, pc);
		}
	}
	int n = limit / 32 * 32;
	fprintf(stderr, "nn: %d images, accelerator accuracy %.2f%%, CPU accuracy %.2f%%, same prediction on %d of %d, largest logit difference %.4f\n",
		n, 100.0 * ok_npu / n, 100.0 * ok_cpu / n, agree, n, maxdiff);
	fprintf(stderr, "nn: accelerator time %.1f ms per batch of 32 (two calls, each re-acquiring and powering the accelerator)\n", 1000 * t_npu / (n / 32));
	return 0;
}
