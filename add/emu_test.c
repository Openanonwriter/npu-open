/* Freestanding harness: runs add_skel_invoke under qemu-hexagon and exits with the sum. */
typedef unsigned int u32;
typedef union { struct { void *pv; u32 len; } buf; u32 h; } remote_arg;
int add_skel_invoke(u32 sc, remote_arg *pra);

static void sys_exit(int code)
{
	register int r0 __asm__("r0") = code;
	register int r6 __asm__("r6") = 94; /* exit_group */
	__asm__ volatile("trap0(#1)" : : "r"(r0), "r"(r6));
	for (;;) ;
}

void _start(void)
{
	u32 in[2] = { 2, 40 }, out[1] = { 0 };
	remote_arg a[2] = { { .buf = { in, 8 } }, { .buf = { out, 4 } } };
	int ret = add_skel_invoke((0u << 24) | (1 << 16) | (1 << 8), a);
	sys_exit(ret ? 200 + ret : (int)out[0]);
}
