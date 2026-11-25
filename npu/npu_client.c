/* Loads a fully connected network (model file from train_mnist.py), runs its test set on the HMX accelerator
 * through one persistent session, and compares predictions, logits and speed with a float32 CPU reference.
 * usage (through run.sh):  MODEL=path [NB=batches per call] [OUT=file for raw outputs] npu_client */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <misc/fastrpc.h>
#include <libhexagonrpc/hexagonrpc.h>
#include <libhexagonrpc/session.h>
#include <libhexagonrpc/interface/remotectl.h>
#include "npu_pack.h"
#define TILE 2048
#define MAXL 8
#define W4 { HRPC_ARG_WORD, 4 }
static uint16_t h(float f) { _Float16 x = (_Float16)f; uint16_t u; memcpy(&u, &x, 2); return u; }
static float f16(uint16_t u) { _Float16 x; memcpy(&x, &u, 2); return (float)x; }
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
/* A dma-buf from the FastRPC driver, mapped here. Passed to a call by fd, the kernel maps it for the DSP
 * instead of copying its contents. persist: also ask for a mapping that outlives each call. */
static void *shared_alloc(int fd, size_t bytes, int *dfd, int persist)
{
	struct fastrpc_alloc_dma_buf db = { .fd = -1, .flags = 0, .size = (bytes + 4095) & ~(size_t)4095 };
	void *p;
	if (ioctl(fd, FASTRPC_IOCTL_ALLOC_DMA_BUFF, &db)) { perror("npu: ALLOC_DMA_BUFF"); return NULL; }
	p = mmap(NULL, db.size, PROT_READ | PROT_WRITE, MAP_SHARED, db.fd, 0);
	if (p == MAP_FAILED) { perror("npu: mmap dma-buf"); return NULL; }
	if (persist) {
		struct fastrpc_mem_map mm = { .version = 0, .fd = db.fd, .offset = 0, .flags = FASTRPC_MAP_STATIC, .vaddrin = (uintptr_t)p, .length = db.size };
		int r = ioctl(fd, FASTRPC_IOCTL_MEM_MAP, &mm);
		fprintf(stderr, "npu: persistent map of %zu KB returned %d (0x%x), DSP address 0x%llx\n", (size_t)db.size / 1024, r, r, (unsigned long long)mm.vaddrout);
	}
	*dfd = db.fd;
	return p;
}
static void die(const char *m) { fprintf(stderr, "npu: %s\n", m); exit(1); }
#define RD(p, n) do { if (fread(p, 1, n, f) != (size_t)(n)) die("short model file"); } while (0)

int main(void)
{
	char dflt[512];
	snprintf(dflt, sizeof(dflt), "%s/npu/mnist-128.bin", getenv("NPU_OPEN") ? getenv("NPU_OPEN") : ".");
	const char *path = getenv("MODEL") ? getenv("MODEL") : dflt;
	int NB = getenv("NB") ? atoi(getenv("NB")) : 32, fd = hexagonrpc_fd_from_env(), ret;
	char magic[4], err[256] = ""; int32_t nl, sz[MAXL + 1], nt; float *W[MAXL], *B[MAXL];
	uint32_t hdl = 0, dlerr = 0, nk[MAXL] = { 0 }, ng[MAXL] = { 0 }, mode[MAXL] = { 0 }, st[8];
	FILE *f = fopen(path, "rb");
	if (fd < 0 || !f) die("no FastRPC fd or model file");
	RD(magic, 4); RD(&nl, 4);
	if (memcmp(magic, "NPU2", 4) || nl < 1 || nl > MAXL) die("bad model file");
	RD(sz, 4 * (nl + 1)); RD(&nt, 4);
	for (int l = 0; l < nl; l++) {
		W[l] = malloc(4 * sz[l] * sz[l + 1]); B[l] = malloc(4 * sz[l + 1]);
		RD(W[l], 4 * sz[l] * sz[l + 1]); RD(B[l], 4 * sz[l + 1]);
		nk[l] = (sz[l] + 31) / 32; ng[l] = (sz[l + 1] + 31) / 32; mode[l] = l + 1 == nl;
	}
	int NI = sz[0], NO = sz[nl], OG = ng[nl - 1];
	float *X = malloc(4 * (size_t)nt * NI); uint8_t *y = malloc(nt);
	RD(X, 4 * (size_t)nt * NI); RD(y, nt);

	sleep(2);
	ret = remotectl_open(fd, strlen("npu") + 1, "npu", &hdl, sizeof(err), err, &dlerr);
	if (ret || dlerr) { fprintf(stderr, "npu: open failed ret=0x%x dlerr=0x%x\n", ret, dlerr); return 2; }

	/* open: acquire + power the accelerator once and describe the layers */
	{
		struct hrpc_arg_def_interp4 a[29] = { W4, W4, W4, W4, W4, W4, W4, W4, W4, W4, W4, W4, W4, W4, W4, W4, W4, W4, W4, W4, W4, W4, W4, W4, W4, W4, W4, W4, { HRPC_ARG_OUT_BLOB, 32 } };
		struct hrpc_method_def_interp4 d = { .msg_id = 0, .n_args = 29, .args = a };
		ret = hexagonrpc(&d, fd, hdl, (uint32_t)(1 << 20), (uint32_t)nl,
			nk[0], nk[1], nk[2], nk[3], nk[4], nk[5], nk[6], nk[7], ng[0], ng[1], ng[2], ng[3], ng[4], ng[5], ng[6], ng[7],
			mode[0], mode[1], mode[2], mode[3], mode[4], mode[5], mode[6], mode[7],
			(uint32_t)(getenv("MIPS") ? atoi(getenv("MIPS")) : 2000), (uint32_t)(getenv("LATENCY") ? atoi(getenv("LATENCY")) : 100), st);
		if (ret || !st[3] || st[3] == 0xdeadbeef) { fprintf(stderr, "npu: session open failed ret=0x%x acquire=%x getptr=%x power=%x vtcm=%x\n", ret, st[3], st[4], st[7], st[6]); return 3; }
		fprintf(stderr, "npu: session open, fast memory %u KB at 0x%x, performance request returned 0x%x\n", st[6] / 1024, st[5], st[0]);
	}
	/* load: weights for every layer and output group, once */
	double t0 = now(); size_t wbytes = 0;
	for (int l = 0; l < nl; l++) for (uint32_t g = 0; g < ng[l]; g++) {
		uint16_t *wt = calloc(nk[l], TILE), bias[32] = { 0 };
		for (uint32_t k = 0; k < nk[l]; k++) for (int q = 0; q < 1024; q++) {
			int i = 32 * k + 2 * (q / 64) + q % 2, o = 32 * g + (q % 64) / 2;
			wt[k * 1024 + q] = (o < sz[l + 1] && i < sz[l]) ? h(W[l][o * sz[l] + i]) : 0;
		}
		for (int o = 0; o < 32; o++) if (32 * (int)g + o < sz[l + 1]) bias[o] = h(B[l][32 * g + o]);
		struct hrpc_arg_def_interp4 a[] = { W4, W4, { HRPC_ARG_BLOB, nk[l] * TILE }, { HRPC_ARG_BLOB, 64 } };
		struct hrpc_method_def_interp4 d = { .msg_id = 1, .n_args = 4, .args = a };
		ret = hexagonrpc(&d, fd, hdl, (uint32_t)l, g, wt, bias);
		if (ret) { fprintf(stderr, "npu: weight load failed layer %d group %u ret=0x%x\n", l, g, ret); return 3; }
		wbytes += nk[l] * TILE; free(wt);
	}
	fprintf(stderr, "npu: loaded %zu KB of weights in %.1f ms\n", wbytes / 1024, 1000 * (now() - t0));

	/* run: NB batches of 32 images per call */
	int nbatch = (nt + 31) / 32; size_t in_b = nk[0] * TILE, out_b = OG * TILE * (((getenv("FLAGS") ? atoi(getenv("FLAGS")) : 7) & 2) ? 2 : 1);
	int shared = getenv("SHARED") ? atoi(getenv("SHARED")) : 1, persist = getenv("PERSIST") ? atoi(getenv("PERSIST")) : 1, fd_in = -1, fd_out = -1;
	uint16_t *act = NULL; uint8_t *res = NULL;
	if (shared) {
		act = shared_alloc(fd, NB * in_b, &fd_in, persist);
		res = shared_alloc(fd, 128 + NB * out_b, &fd_out, persist);
		if (!act || !res) { fprintf(stderr, "npu: shared buffers unavailable, falling back to copied buffers\n"); shared = 0; }
	}
	if (!shared) { act = calloc(NB, in_b); res = malloc(128 + NB * out_b); }
	float *Ln = malloc(4 * (size_t)nbatch * 32 * NO);
	double t_wall = 0, t_dsp = 0, t_pack = 0, part[4] = { 0 }; int calls = 0;
	uint32_t flags = getenv("FLAGS") ? (uint32_t)atoi(getenv("FLAGS")) : 7;   /* bit 0: deep multiply, bit 1: merge sign passes here, bit 2: HVX copies */
	int raw = (flags >> 1) & 1;
	for (int b0 = 0; b0 < nbatch; b0 += NB) {
		int nb = nbatch - b0 < NB ? nbatch - b0 : NB;
		t0 = now();
		for (int b = 0; b < nb; b++) {
			int first = (b0 + b) * 32;
			npu_pack_batch(act + (size_t)b * nk[0] * 1024, X + (size_t)first * NI, nt - first, NI);
		}
		t_pack += now() - t0;
		struct hrpc_arg_def_interp4 a[] = { W4, W4, { HRPC_ARG_BLOB, nb * in_b }, { HRPC_ARG_OUT_BLOB, 128 + nb * out_b } };
		struct hrpc_method_def_interp4 d = { .msg_id = 2, .n_args = 4, .args = a };
		t0 = now();
		if (shared) {
			uint32_t words[2] = { (uint32_t)nb, flags };
			struct fastrpc_invoke_args ia[3] = {
				{ .ptr = (uintptr_t)words, .length = sizeof(words), .fd = -1 },
				{ .ptr = (uintptr_t)act, .length = nb * in_b, .fd = fd_in },
				{ .ptr = (uintptr_t)res, .length = 128 + nb * out_b, .fd = fd_out },
			};
			struct fastrpc_invoke inv = { .handle = hdl, .sc = REMOTE_SCALARS_MAKE(2, 2, 1), .args = (uintptr_t)ia };
			ret = ioctl(fd, FASTRPC_IOCTL_INVOKE, &inv);
		} else {
			ret = hexagonrpc(&d, fd, hdl, (uint32_t)nb, flags, act, res);
		}
		t_wall += now() - t0; calls++;
		memcpy(st, res, 32);
		if (ret || st[0] || st[3]) { fprintf(stderr, "npu: run failed ret=0x%x lock=%x unlock=%x\n", ret, st[0], st[3]); return 3; }
		t_dsp += st[1] * 1e-6;
		for (int i = 0; i < 4; i++) part[i] += st[4 + i] * 1e-6;
		const uint16_t *tiles = (const uint16_t *)(res + 128);
		int per = raw ? 2 * OG : OG;
		for (int b = 0; b < nb; b++) for (int s = 0; s < 32; s++) for (int o = 0; o < NO; o++) {
			int cell = 64 * (s / 2) + 2 * (o % 32) + s % 2;
			uint16_t v = tiles[(b * per + o / 32) * 1024 + cell];
			if (raw && !v) { uint16_t n = tiles[(b * per + OG + o / 32) * 1024 + cell]; if (n) v = n | 0x8000; }
			Ln[((size_t)(b0 + b) * 32 + s) * NO + o] = f16(v);
		}
	}
	{ struct hrpc_method_def_interp4 d = { .msg_id = 3, .n_args = 0, .args = NULL }; hexagonrpc(&d, fd, hdl); }

	/* CPU reference: plain float32 loops, one thread */
	int mx = 0; for (int l = 0; l <= nl; l++) if (sz[l] > mx) mx = sz[l];
	float *a0 = malloc(4 * mx), *a1 = malloc(4 * mx), *Lc = malloc(4 * (size_t)nt * NO);
	t0 = now();
	for (int n = 0; n < nt; n++) {
		const float *in = X + (size_t)n * NI; float *o = a0;
		for (int l = 0; l < nl; l++) {
			o = l + 1 == nl ? Lc + (size_t)n * NO : (l % 2 ? a1 : a0);
			for (int j = 0; j < sz[l + 1]; j++) {
				float acc = B[l][j]; const float *w = W[l] + (size_t)j * sz[l];
				for (int i = 0; i < sz[l]; i++) acc += in[i] * w[i];
				o[j] = (l + 1 < nl && acc < 0) ? 0 : acc;
			}
			in = o;
		}
	}
	double t_cpu = now() - t0;

	if (getenv("OUT")) {   /* raw outputs for callers that are not classifiers: n, width, accelerator outputs, CPU outputs */
		FILE *o = fopen(getenv("OUT"), "wb"); int32_t hdr[2] = { nt, NO };
		if (!o) die("cannot write OUT file");
		fwrite(hdr, 4, 2, o); fwrite(Ln, 4, (size_t)nt * NO, o); fwrite(Lc, 4, (size_t)nt * NO, o); fclose(o);
	}
	int ok_n = 0, ok_c = 0, agree = 0; double maxdiff = 0;
	for (int n = 0; n < nt; n++) {
		int pn = 0, pc = 0;
		for (int o = 0; o < NO; o++) {
			float dn = Ln[(size_t)n * NO + o], dc = Lc[(size_t)n * NO + o];
			if (fabsf(dn - dc) > maxdiff) maxdiff = fabsf(dn - dc);
			if (dn > Ln[(size_t)n * NO + pn]) pn = o;
			if (dc > Lc[(size_t)n * NO + pc]) pc = o;
		}
		ok_n += pn == y[n]; ok_c += pc == y[n]; agree += pn == pc;
	}
	fprintf(stderr, "npu: network");
	for (int l = 0; l <= nl; l++) fprintf(stderr, "%s%d", l ? "-" : " ", sz[l]);
	fprintf(stderr, ", %d test images, %d batches per call, %s buffers%s\n", nt, NB, shared ? "shared" : "copied", shared && persist ? " (persistent map)" : "");
	fprintf(stderr, "npu: accuracy  accelerator %.2f%%  CPU %.2f%%  same prediction on %d of %d  largest logit difference %.4f\n",
		100.0 * ok_n / nt, 100.0 * ok_c / nt, agree, nt, maxdiff);
	fprintf(stderr, "npu: time per image  accelerator compute %.2f us  accelerator incl. FastRPC call %.2f us  packing on CPU %.2f us  CPU reference %.2f us\n",
		1e6 * t_dsp / nt, 1e6 * t_wall / nt, 1e6 * t_pack / nt, 1e6 * t_cpu / nt);
	fprintf(stderr, "npu: accelerator breakdown per image  copy-in %.2f us  multiply+convert %.2f us  sign merge %.2f us  copy-out %.2f us  (flags=%u)\n",
		1e6 * part[0] / nt, 1e6 * part[1] / nt, 1e6 * part[2] / nt, 1e6 * part[3] / nt, flags);
	fprintf(stderr, "npu: %d run calls, %.2f ms per call, HVX lock returned 0x%x\n", calls, 1000 * t_wall / calls, st[2]);
	return 0;
}
