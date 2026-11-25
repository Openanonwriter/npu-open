/* Linux-side client: asks the DSP to load libadd_skel.so and add two numbers. */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <libhexagonrpc/hexagonrpc.h>
#include <libhexagonrpc/handle.h>
#include <libhexagonrpc/session.h>
#include <libhexagonrpc/interface/remotectl.h>
#include <string.h>

static struct hrpc_arg_def_interp4 add_args[] = {
	{ HRPC_ARG_WORD, sizeof(uint32_t) },
	{ HRPC_ARG_WORD, sizeof(uint32_t) },
	{ HRPC_ARG_OUT_BLOB, sizeof(uint32_t) },
};
static const struct hrpc_method_def_interp4 add_def = {
	.msg_id = 0, .n_args = 3, .args = add_args, .n_inner_types = 0, .inner_types = NULL,
};

int main(void)
{
	char err[256] = "";
	uint32_t hdl = 0, sum = 0;
	int fd = hexagonrpc_fd_from_env(), ret;

	if (fd < 0) { fprintf(stderr, "client: no HEXAGONRPC_FD\n"); return 1; }
	sleep(2); /* let the daemon register its file-serving listener */
	uint32_t dlerr = 0;
	const char *name = getenv("NPU_LIB") ? getenv("NPU_LIB") : "add";
	ret = remotectl_open(fd, strlen(name) + 1, name, &hdl, sizeof(err), err, &dlerr);
	fprintf(stderr, "client: open(%s) ret=0x%x dlerr=0x%x handle=%u\n", name, ret, dlerr, hdl);
	if (getenv("NPU_LIB")) return 0;
	if (ret || dlerr) { fprintf(stderr, "client: open failed: ret=0x%x dlerr=0x%x msg='%s'\n", ret, dlerr, err); return 2; }
	fprintf(stderr, "client: DSP loaded our library, handle %u\n", hdl);
	ret = hexagonrpc(&add_def, fd, hdl, (uint32_t)1234, (uint32_t)4321, &sum);
	fprintf(stderr, "client: invoke ret=%d, 1234 + 4321 = %u (computed on the DSP)\n", ret, sum);
	hexagonrpc_close(fd, hdl);
	return ret;
}
