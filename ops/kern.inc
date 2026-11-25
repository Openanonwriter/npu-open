// The operations of the sweep, written once and compiled twice: ops_scalar.cpp (no vector unit) and
// ops_hvx.cpp (-mhvx, so the compiler turns the loops into HVX instructions). Included inside a namespace.
// Each returns the number of output bytes. The Rust client (rust/src/bin/ops.rs) repeats them on the CPU.
template <typename Out, typename F> static u32 map2(const u8 *a, const u8 *b, void *out, u32 n, F f)
{
	Out *o = static_cast<Out *>(out);
	for (u32 i = 0; i < n; i++) o[i] = f(a[i], b[i], i);
	return n * sizeof(Out);
}
template <typename F> static u32 fold(const u8 *a, const u8 *b, void *out, u32 n, F f)
{
	u32 s = 0;
	for (u32 i = 0; i < n; i++) s += f(a[i], b[i]);
	*static_cast<u32 *>(out) = s;
	return 4;
}
u32 run(u32 op, const u8 *a, const u8 *b, void *out, u32 n, const u8 *lut)
{
	switch (op) {
	case 0: return map2<u8>(a, b, out, n, [](u32 x, u32 y, u32) { u32 s = x + y; return (u8)(s > 255 ? 255 : s); });
	case 1: return map2<u8>(a, b, out, n, [](u32 x, u32 y, u32) { return (u8)(x > y ? x - y : y - x); });
	case 2: return map2<u8>(a, b, out, n, [](u32 x, u32 y, u32) { return (u8)((x + y + 1) >> 1); });
	case 3: return map2<u8>(a, b, out, n, [](u32 x, u32 y, u32) { return (u8)((x ^ y) | (x & 0x0f)); });
	case 4: return map2<u8>(a, b, out, n, [](u32 x, u32 y, u32) { return (u8)((x << 3) | (y >> 5)); });
	case 5: return map2<u16>(a, b, out, n, [](u32 x, u32 y, u32) { return (u16)(x * y); });
	case 6: return fold(a, b, out, n, [](u32 x, u32 y) { return x * y; });
	case 7: return fold(a, b, out, n, [](u32 x, u32 y) { return x > y ? x - y : y - x; });
	case 8: return map2<u8>(a, b, out, n, [](u32 x, u32 y, u32) { return (u8)(x > 128 ? y : 0); });
	case 9: return map2<u8>(a, b, out, n, [](u32 x, u32 y, u32) { return (u8)__builtin_popcount(x ^ y); });
	case 10: return map2<u32>(a, b, out, n, [](u32 x, u32 y, u32 i) { return (x * 65537u) * (y * 257u) + i; });
	case 11: return map2<float>(a, b, out, n, [](u32 x, u32 y, u32) { return (float)x * 0.5f + (float)y * 1.5f; });
	case 12: return map2<u8>(a, b, out, n, [lut](u32 x, u32 y, u32) { return (u8)(lut[x] ^ y); });
	}
	return 0;
}
