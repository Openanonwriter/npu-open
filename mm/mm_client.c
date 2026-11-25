#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>
#include <libhexagonrpc/hexagonrpc.h>
#include <libhexagonrpc/session.h>
#include <libhexagonrpc/interface/remotectl.h>
struct res { uint32_t st[16]; uint16_t tile[1024]; };
static struct hrpc_arg_def_interp4 args[] = { { HRPC_ARG_WORD, 4 }, { HRPC_ARG_WORD, 4 }, { HRPC_ARG_WORD, 4 }, { HRPC_ARG_WORD, 4 }, { HRPC_ARG_OUT_BLOB, sizeof(struct res) } };
static const struct hrpc_method_def_interp4 def = { .msg_id = 0, .n_args = 5, .args = args };
static uint32_t envu(const char *n, uint32_t d) { return getenv(n) ? strtoul(getenv(n), NULL, 0) : d; }
int main(void)
{
	static struct res o; char err[256] = ""; uint32_t hdl = 0, dlerr = 0; int ret, i, n = 0, first_bad = -1;
	int fd = hexagonrpc_fd_from_env();
	uint32_t stage = envu("STAGE", 0), a = envu("ACT", 0x3c00), w = envu("WT", 0x3800), s = envu("SCALE", 0x3c00), want = envu("WANT", 0x4c00);
	if (fd < 0) return 1;
	sleep(2);
	ret = remotectl_open(fd, strlen("mm") + 1, "mm", &hdl, sizeof(err), err, &dlerr);
	if (ret || dlerr) { fprintf(stderr, "mm: open failed ret=0x%x dlerr=0x%x\n", ret, dlerr); return 2; }
	ret = hexagonrpc(&def, fd, hdl, stage, a, w, s, &o);
	fprintf(stderr, "mm: stage=%u invoke=0x%x init=%x hmx=%x vtcmparam=%x ctx=%x getptr=%x vtcm=%x size=%x filled=%x power=%x lock=%x reached=%x unlock=%x release=%x\n",
		stage, ret, o.st[0], o.st[1], o.st[2], o.st[3], o.st[4], o.st[5], o.st[6], o.st[7], o.st[8], o.st[9], o.st[10], o.st[11], o.st[12]);
	for (i = 0; i < 1024; i++) { if (o.tile[i] == want) n++; else if (first_bad < 0) first_bad = i; }
	fprintf(stderr, "mm: output cells equal to 0x%04x: %d of 1024; first eight: %04x %04x %04x %04x %04x %04x %04x %04x", want, n,
		o.tile[0], o.tile[1], o.tile[2], o.tile[3], o.tile[4], o.tile[5], o.tile[6], o.tile[7]);
	if (first_bad >= 0) fprintf(stderr, "; first mismatch at %d = %04x", first_bad, o.tile[first_bad]);
	fprintf(stderr, "\n");
	return 0;
}
