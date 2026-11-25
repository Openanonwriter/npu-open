// Operation sweep for the Hexagon cDSP, in freestanding C++ (no exceptions, no RTTI, no global
// constructors, no C++ runtime library). Clean-room: no Qualcomm headers. The functions declared below are
// exported by the DSP shell (found by name with probe/; signatures are the standard C and POSIX ones,
// except HAP_perf_get_time_us and qurt_hvx_lock, which are documented in TECHNICAL.md section 5).
//   method 0: every operation of kern.inc, scalar and HVX builds, timed on the DSP clock
//   method 1: N POSIX threads each doing the same scalar work (do they run in parallel?)
//   method 2: C library calls from the shell: malloc, qsort, sqrtf, sinf, expf, 64-bit and double maths
#include "types.h"
typedef union { struct { void *pv; u32 len; } buf; u32 h; } remote_arg;
extern "C" {
u64 HAP_perf_get_time_us(void);
int qurt_hvx_lock(int mode);
int qurt_hvx_unlock(void);
void *memalign(u32 align, u32 size);
void *malloc(u32 size);
void free(void *p);
void qsort(void *base, u32 n, u32 size, int (*cmp)(const void *, const void *));
float sqrtf(float);
float sinf(float);
float expf(float);
int pthread_attr_init(void *attr);
int pthread_attr_setstacksize(void *attr, u32 size);
int pthread_create(void *thread, const void *attr, void *(*fn)(void *), void *arg);
int pthread_join(u32 thread, void **ret);
int qurt_thread_get_id(void);
int ops_skel_invoke(u32 sc, remote_arg *pra);
}
namespace scalar { u32 run(u32, const u8 *, const u8 *, void *, u32, const u8 *); }
namespace hvx { u32 run(u32, const u8 *, const u8 *, void *, u32, const u8 *); }

enum { N_OPS = 13, REPEAT = 16 };

static u32 fnv(const void *p, u32 n)
{
	const u8 *b = static_cast<const u8 *>(p);
	u32 h = 2166136261u;
	for (u32 i = 0; i < n; i++) h = (h ^ b[i]) * 16777619u;
	return h;
}

// A small C++ class to show the language works: fills buffers with the generator the client also uses.
class Lcg {
	u32 x;
public:
	explicit Lcg(u32 seed) : x(seed) {}
	u8 next() { x = x * 1664525u + 1013904223u; return (u8)(x >> 24); }
};

struct Job { u32 iters, result, tid; };
static void *worker(void *p)
{
	Job *j = static_cast<Job *>(p);
	u32 x = 1;
	for (u32 i = 0; i < j->iters; i++) x = x * 1664525u + 1013904223u + (x >> 7);
	j->result = x;
	j->tid = (u32)qurt_thread_get_id();
	return nullptr;
}

static int cmp_u32(const void *a, const void *b)
{
	u32 x = *static_cast<const u32 *>(a), y = *static_cast<const u32 *>(b);
	return x < y ? -1 : x > y;
}

int ops_skel_invoke(u32 sc, remote_arg *pra)
{
	u32 method = (sc >> 24) & 0x1f;
	const u32 *in = static_cast<const u32 *>(pra[0].buf.pv);
	u32 *out = static_cast<u32 *>(pra[1].buf.pv);
	if (((sc >> 16) & 0xff) != 1 || ((sc >> 8) & 0xff) != 1 || pra[0].buf.len < 8) return 0x0e;

	if (method == 0) {
		u32 n = in[0];
		if (pra[1].buf.len < (4 * N_OPS + 1) * 4 || n > (1u << 20)) return 0x0e;
		u8 *a = static_cast<u8 *>(memalign(128, n)), *b = static_cast<u8 *>(memalign(128, n));
		void *o = memalign(128, 4 * n);
		if (!a || !b || !o) return 0x02;
		u8 lut[256];
		Lcg g(in[1]);
		for (u32 i = 0; i < n; i++) { a[i] = g.next(); b[i] = g.next(); }
		for (u32 i = 0; i < 256; i++) lut[i] = (u8)(i * 7 + 3);
		int lock = qurt_hvx_lock(1);
		out[4 * N_OPS] = (u32)lock;
		for (u32 op = 0; op < N_OPS; op++) {
			u32 bytes = 0;
			u64 t0 = HAP_perf_get_time_us();
			for (u32 r = 0; r < REPEAT; r++) bytes = scalar::run(op, a, b, o, n, lut);
			u64 t1 = HAP_perf_get_time_us();
			out[4 * op] = fnv(o, bytes);
			out[4 * op + 2] = (u32)(t1 - t0);
			out[4 * op + 1] = 0;
			out[4 * op + 3] = 0;
			if (lock == 0) {
				t0 = HAP_perf_get_time_us();
				for (u32 r = 0; r < REPEAT; r++) bytes = hvx::run(op, a, b, o, n, lut);
				t1 = HAP_perf_get_time_us();
				out[4 * op + 1] = fnv(o, bytes);
				out[4 * op + 3] = (u32)(t1 - t0);
			}
		}
		if (lock == 0) qurt_hvx_unlock();
		free(a); free(b); free(o);
		return 0;
	}
	if (method == 1) {
		u32 k = in[0], iters = in[1];
		if (k < 1 || k > 8 || pra[1].buf.len < (2 + 2 * 8) * 4) return 0x0e;
		Job jobs[8];
		u64 ids[8];                      // room for a thread id of either width
		u8 *attr = static_cast<u8 *>(malloc(512)); // attribute object: layout unknown, only passed to the library
		if (!attr) return 0x02;
		u32 err = 0;
		u64 t0 = HAP_perf_get_time_us();
		for (u32 i = 0; i < k; i++) {
			jobs[i].iters = iters; jobs[i].result = 0; jobs[i].tid = 0; ids[i] = 0;
			err |= (u32)pthread_attr_init(attr);
			err |= (u32)pthread_attr_setstacksize(attr, 16384);
			err |= (u32)pthread_create(&ids[i], attr, worker, &jobs[i]);
		}
		if (!err) for (u32 i = 0; i < k; i++) err |= (u32)pthread_join(*reinterpret_cast<u32 *>(&ids[i]), nullptr);
		u64 t1 = HAP_perf_get_time_us();
		out[0] = (u32)(t1 - t0);
		out[1] = err;
		for (u32 i = 0; i < 8; i++) { out[2 + 2 * i] = i < k ? jobs[i].result : 0; out[3 + 2 * i] = i < k ? jobs[i].tid : 0; }
		free(attr);
		return 0;
	}
	if (method == 2) {
		if (pra[1].buf.len < 8 * 4) return 0x0e;
		u32 n = 4096, *v = static_cast<u32 *>(malloc(n * 4));
		if (!v) return 0x02;
		Lcg g(in[1]);
		for (u32 i = 0; i < n; i++) v[i] = (u32)g.next() << 16 | (u32)g.next() << 8 | g.next();
		u64 t0 = HAP_perf_get_time_us();
		qsort(v, n, 4, cmp_u32);
		out[1] = (u32)(HAP_perf_get_time_us() - t0);
		out[0] = fnv(v, n * 4);
		free(v);
		union { float f; u32 u; } s, si, e;
		volatile float two = 2.0f, one = 1.0f;
		s.f = sqrtf(two); si.f = sinf(one); e.f = expf(one);
		out[2] = s.u; out[3] = si.u; out[4] = e.u;
		volatile u64 big = 0x123456789abcdefULL; volatile u32 d = in[0] ? in[0] : 7;
		out[5] = (u32)(big / d);        // 64-bit divide: a compiler support routine the shell must supply
		volatile double x = 1.5, y = 2.25;
		out[6] = (u32)(x * y * 1000.0); // double precision is software on this DSP
		out[7] = (u32)qurt_thread_get_id();
		return 0;
	}
	return 0x14;
}
