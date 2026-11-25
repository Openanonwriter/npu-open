/* Copy with 128-byte HVX vectors. Scalar memcpy from a shared (dma-buf) mapping into VTCM manages about
 * 750 MB/s on this DSP; whole-vector loads and stores move the same data in far fewer memory accesses.
 * Both pointers must be 128-byte aligned and n a multiple of 128. Compiled with -mhvx -mhvx-length=128B. */
typedef int hvx_vec __attribute__((__vector_size__(128), __aligned__(128)));

void hvx_copy(void *dst, const void *src, unsigned n)
{
	hvx_vec *d = dst;
	const hvx_vec *s = src;
	unsigned i, k = n / 128;
	for (i = 0; i < k; i++)
		d[i] = s[i];
}
