# Patches

`hexagonrpcd-npu-open.patch` applies to [linux-msm/hexagonrpc](https://github.com/linux-msm/hexagonrpc)
at commit `da7a374` (`scripts/fetch-deps.sh` does this).

**Licence:** this patch modifies GPL-3.0-or-later code (`hexagonrpcd`), so the patch itself is
GPL-3.0-or-later. It is the one file in this repository not under the MIT licence.

What it changes:

| Change | Why |
|---|---|
| `-u` option | Create an unsigned DSP process (`FASTRPC_MODE_UNSIGNED_MODULE`) |
| Report the real result of `INIT_CREATE` | DSP errors are the ioctl's return value; the stock code printed a stale `errno` |
| `apps_mem` method 3 (`request_unmap64`) | The DSP calls it when releasing heap; without it large calls hang |
| Return the full mapped address from `request_map64` | Truncating it to 32 bits made DSP heap growth fail (`0x80000468`) on this machine; `HRPC_VADSP32=1` restores the old behaviour |
