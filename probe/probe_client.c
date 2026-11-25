/* Opens libp<N>_skel.so for every line of names.txt and reports whether the DSP could resolve that symbol. */
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>
#include <libhexagonrpc/hexagonrpc.h>
#include <libhexagonrpc/handle.h>
#include <libhexagonrpc/session.h>
#include <libhexagonrpc/interface/remotectl.h>
int main(void)
{
	char sym[128], name[16], err[256];
	int fd = hexagonrpc_fd_from_env(), i = 0;
	char path[512];
	snprintf(path, sizeof(path), "%s/probe/names.txt", getenv("NPU_OPEN") ? getenv("NPU_OPEN") : ".");
	FILE *f = fopen(path, "r");
	if (fd < 0 || !f) return 1;
	sleep(2);
	while (fgets(sym, sizeof(sym), f)) {
		uint32_t hdl = 0, dlerr = 0; int ret;
		sym[strcspn(sym, "\n")] = 0; err[0] = 0;
		snprintf(name, sizeof(name), "p%d", i++);
		ret = remotectl_open(fd, strlen(name) + 1, name, &hdl, sizeof(err), err, &dlerr);
		fprintf(stderr, "probe: %-40s %s (ret=0x%x dlerr=0x%x)\n", sym, (!ret && !dlerr) ? "EXPORTED" : "missing", ret, dlerr);
	}
	fprintf(stderr, "probe: done\n");
	return 0;
}
