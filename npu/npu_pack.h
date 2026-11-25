/* CPU-side conversion of float32 data into the accelerator's FP16 activation tiles.
 *
 * One call fills the tiles for one batch of 32 samples. Tile k covers features 32k..32k+31; inside a tile,
 * each 128-byte row holds two samples interleaved feature by feature:
 *   row j = { s[2j][0], s[2j+1][0], s[2j][1], s[2j+1][1], ... , s[2j][31], s[2j+1][31] }   (FP16)
 * Samples past n_rows and features past n_feat are zero. Uses NEON half-precision conversion when the
 * compiler targets it (-march=armv8.2-a+fp16), otherwise a portable loop. */
#ifndef NPU_PACK_H
#define NPU_PACK_H
#include <stdint.h>
#include <string.h>
#if defined(__ARM_NEON) && defined(__ARM_FP16_FORMAT_IEEE)
#include <arm_neon.h>
#define NPU_PACK_NEON 1
#endif

/* x: n_rows samples of n_feat float32 each, row-major, starting at the first sample of this batch.
 * tiles: ((n_feat + 31) / 32) * 2048 bytes. */
static inline void npu_pack_batch(uint16_t *tiles, const float *x, int n_rows, int n_feat)
{
	static const float zero_row[32] = { 0 };
	int nk = (n_feat + 31) / 32;
	if (n_rows > 32) n_rows = 32;
	for (int j = 0; j < 16; j++) {
		const float *ra = 2 * j < n_rows ? x + (size_t)(2 * j) * n_feat : NULL;
		const float *rb = 2 * j + 1 < n_rows ? x + (size_t)(2 * j + 1) * n_feat : NULL;
		for (int k = 0; k < nk; k++) {
			float ta[32], tb[32];
			const float *a = zero_row, *b = zero_row;
			int have = n_feat - 32 * k;            /* features available in this tile */
			if (ra) { if (have >= 32) a = ra + 32 * k; else { memset(ta, 0, sizeof(ta)); memcpy(ta, ra + 32 * k, have * sizeof(float)); a = ta; } }
			if (rb) { if (have >= 32) b = rb + 32 * k; else { memset(tb, 0, sizeof(tb)); memcpy(tb, rb + 32 * k, have * sizeof(float)); b = tb; } }
			uint16_t *out = tiles + (size_t)k * 1024 + 64 * j;
#ifdef NPU_PACK_NEON
			for (int c = 0; c < 32; c += 4) {
				float16x4_t ha = vcvt_f16_f32(vld1q_f32(a + c)), hb = vcvt_f16_f32(vld1q_f32(b + c));
				vst1q_f16((float16_t *)(out + 2 * c), vcombine_f16(vzip1_f16(ha, hb), vzip2_f16(ha, hb)));
			}
#else
			for (int c = 0; c < 32; c++) {
				_Float16 ha = (_Float16)a[c], hb = (_Float16)b[c];
				memcpy(out + 2 * c, &ha, 2); memcpy(out + 2 * c + 1, &hb, 2);
			}
#endif
		}
	}
}
#endif
