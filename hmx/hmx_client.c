#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>
#include <libhexagonrpc/hexagonrpc.h>
#include <libhexagonrpc/session.h>
#include <libhexagonrpc/interface/remotectl.h>
struct out6 { uint32_t v[7]; };
static struct hrpc_arg_def_interp4 args[] = { { HRPC_ARG_WORD, 4 }, { HRPC_ARG_WORD, 4 }, { HRPC_ARG_WORD, 4 }, { HRPC_ARG_OUT_BLOB, sizeof(struct out6) } };
static const struct hrpc_method_def_interp4 def = { .msg_id = 0, .n_args = 4, .args = args };
int main(void)
{
	char err[256] = ""; uint32_t hdl = 0, dlerr = 0; struct out6 o; int ret;
	int fd = hexagonrpc_fd_from_env();
	uint32_t stage = getenv("STAGE") ? atoi(getenv("STAGE")) : 0;
	uint32_t ptype = getenv("PTYPE") ? strtoul(getenv("PTYPE"), NULL, 0) : 0xffffffff;
	if (fd < 0) return 1;
	sleep(2);
	ret = remotectl_open(fd, strlen("hmx") + 1, "hmx", &hdl, sizeof(err), err, &dlerr);
	if (ret || dlerr) { fprintf(stderr, "hmx: open failed ret=0x%x dlerr=0x%x\n", ret, dlerr); return 2; }
	memset(&o, 0, sizeof(o));
	ret = hexagonrpc(&def, fd, hdl, stage, ptype, (uint32_t)(getenv("FIELDS") ? atoi(getenv("FIELDS")) : 3), &o);
	fprintf(stderr, "hmx: stage=%u invoke=0x%x attr_init=0x%x set_hmx=0x%x ctx=0x%x lock=0x%x unlock=0x%x hmx_insn=0x%x | power(type=%d)=0x%x\n",
		stage, ret, o.v[0], o.v[1], o.v[2], o.v[3], o.v[4], o.v[5], (int)ptype, o.v[6]);
	return 0;
}
