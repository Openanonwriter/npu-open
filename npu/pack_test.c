/* Checks npu_pack_batch against the straightforward per-cell formula, including padded shapes, and times it. */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "npu_pack.h"
static uint16_t h(float f) { _Float16 x = (_Float16)f; uint16_t u; memcpy(&u, &x, 2); return u; }
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
int main(void)
{
	int shapes[][2] = { { 32, 784 }, { 32, 64 }, { 32, 10 }, { 16, 784 }, { 7, 100 }, { 1, 33 } }, bad = 0;
	for (unsigned t = 0; t < sizeof(shapes) / sizeof(shapes[0]); t++) {
		int nr = shapes[t][0], nf = shapes[t][1], nk = (nf + 31) / 32;
		float *x = malloc(sizeof(float) * nr * nf); uint16_t *a = malloc(nk * 2048), *b = malloc(nk * 2048);
		for (int i = 0; i < nr * nf; i++) x[i] = (rand() % 20001 - 10000) / 313.0f;
		npu_pack_batch(a, x, nr, nf);
		for (int k = 0; k < nk; k++) for (int p = 0; p < 1024; p++) {
			int s = 2 * (p / 64) + p % 2, i = 32 * k + (p % 64) / 2;
			b[k * 1024 + p] = (s < nr && i < nf) ? h(x[s * nf + i]) : 0;
		}
		int diff = memcmp(a, b, nk * 2048) != 0; bad += diff;
		printf("shape %2d x %3d: %s\n", nr, nf, diff ? "MISMATCH" : "identical");
		free(x); free(a); free(b);
	}
	int nf = 784, nk = 25, reps = 20000; float *x = malloc(sizeof(float) * 32 * nf); uint16_t *a = aligned_alloc(128, nk * 2048);
	for (int i = 0; i < 32 * nf; i++) x[i] = i % 255 / 255.0f;
	double t0 = now(); for (int r = 0; r < reps; r++) { npu_pack_batch(a, x, 32, nf); x[r % 100] += a[5] * 1e-9f; }
	printf("%s path: %.3f us per image (784 features)\n",
#ifdef NPU_PACK_NEON
		"NEON",
#else
		"portable",
#endif
		1e6 * (now() - t0) / reps / 32);
	return bad;
}
