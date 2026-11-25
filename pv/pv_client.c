/* usage (through run.sh): REQ="type w1 w2 ..." (up to 16 words, hex or decimal) [ITER=n] pv_client
 * Runs the timing loop, sends the request, runs the loop twice more, and repeats the timing once more after 1 s. */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>
#include <libhexagonrpc/hexagonrpc.h>
#include <libhexagonrpc/session.h>
#include <libhexagonrpc/interface/remotectl.h>
#define W4 { HRPC_ARG_WORD, 4 }
int main(void)
{
	static struct hrpc_arg_def_interp4 a[] = { W4, W4, W4, W4, W4, W4, W4, W4, W4, W4, W4, W4, W4, W4, W4, W4, W4, W4, { HRPC_ARG_OUT_BLOB, 20 } };
	static const struct hrpc_method_def_interp4 d = { .msg_id = 0, .n_args = 19, .args = a };
	char err[256] = "", *req = getenv("REQ"), *tok; uint32_t hdl = 0, dlerr = 0, w[16] = { 0 }, o[5], o2[5];
	uint32_t iter = getenv("ITER") ? strtoul(getenv("ITER"), NULL, 0) : 2000000;
	int fd = hexagonrpc_fd_from_env(), ret, n = 0;
	if (fd < 0) return 1;
	if (req) { req = strdup(req); for (tok = strtok(req, " "); tok && n < 16; tok = strtok(NULL, " ")) w[n++] = strtoul(tok, NULL, 0); }
	sleep(2);
	ret = remotectl_open(fd, strlen("pv") + 1, "pv", &hdl, sizeof(err), err, &dlerr);
	if (ret || dlerr) { fprintf(stderr, "pv: open failed ret=0x%x dlerr=0x%x\n", ret, dlerr); return 2; }
#define CALL(vote, out) hexagonrpc(&d, fd, hdl, iter, (uint32_t)(vote), w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7], w[8], w[9], w[10], w[11], w[12], w[13], w[14], w[15], out)
	ret = CALL(n > 0, o);
	sleep(1);
	CALL(0, o2);
	fprintf(stderr, "pv: req [%s] ret=0x%x vote=0x%x loop us: before %u, after %u %u, 1 s later %u %u %u\n",
		getenv("REQ") ? getenv("REQ") : "none", ret, o[1], o[0], o[2], o[3], o2[0], o2[2], o2[3]);
	return 0;
}
