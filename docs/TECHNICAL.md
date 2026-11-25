# Technical record

Everything done and learned, in reference form. First written 2026-11-24; see [JOURNAL.md](JOURNAL.md) for
the order things happened in and [SETUP.md](SETUP.md) for reproducing it. The goal was to run our own code on the
Hexagon compute DSP (cDSP) and its HMX matrix accelerator ("the NPU") using only open tools,
without Qualcomm's SDK, headers, toolchain or licence click-throughs.

**Status:** a fully connected MNIST classifier (784-128-10) runs on the accelerator at 97.75%,
giving the same prediction as the CPU on all 10,000 test images. With shared buffers and vector copies it
takes about 1 µs per image including the call from Linux, close to numpy on all ten CPU cores.

## 1. Ground rules

Set by the owner of the machine:

- No EULA click-throughs. The QAIRT / QNN SDK and the Hexagon SDK were not downloaded.
- Do not use Qualcomm's code and do not reverse engineer Qualcomm's binaries.
- Running Qualcomm's unmodified firmware and FastRPC shell from the owner's own Windows driver
  backup is accepted: the DSP cannot run anything without them.

Method used instead:

- Open-licensed source as reference: LLVM (Apache-2.0), llama.cpp (MIT), qualcomm/fastrpc (BSD-3),
  linux-msm/hexagonrpc (GPL-3).
- Qualcomm's public documentation (HMX Programmer's Reference Manual 80-N2040-62).
- Black-box experiments on the hardware: return codes, which symbol names resolve, what a
  computation returns.

Two places where the line was approached, both disclosed at the time:

- I read the ELF header and dynamic-section tags (metadata, no code) of one Qualcomm example
  library before asking. Not repeated.
- With explicit permission, two unmodified Qualcomm example libraries were loaded once as a
  control test, to tell a file-format problem from a loading-path problem.

## 2. Machine

| Item | Value |
|---|---|
| Device | Microsoft Surface Pro 11 (LCD), `microsoft,denali-lcd`, Snapdragon X Plus `x1p64100` |
| OS | Fedora 45 aarch64, kernel `7.3.5-300.sp11.1.fc45` (sp11 spin) |
| cDSP | `remoteproc1`, Hexagon v73, device nodes `/dev/fastrpc-cdsp`, `/dev/fastrpc-cdsp-secure` |
| aDSP | `remoteproc0` (audio, sensors, charging). **Never stop it**: reported to crash the SoC |
| GPU | Adreno X1-85, working through Mesa `turnip` (not used here) |
| Toolchain | Fedora `clang`/`lld`/`llvm` 23.1.5, `qemu-user-static-hexagon` 11.1.3 |

## 3. Quick start

After every reboot, and after every DSP crash, the DSP is back on the installed firmware (build
00046), whose shell we do not have. Reload build 00078 into RAM first:

```bash
./cdsp78.sh
```

Then run the MNIST network on the accelerator:

```bash
CLIENT=$PWD/npu/npu_client ./run.sh 30
```

Results are written to `last-run.log` (lines starting `npu:`). Other clients: `add/client`,
`hmx/hmx_client`, `mm/mm_client`, `mm/mm2_client`, `nn/nn_client`, `probe/probe_client`.

Rebuild a component with its `build.sh` (`add/`, `hmx/`, `mm/`, `nn/`, `npu/`); each one compiles the
DSP library, copies it to `root/dsp/cdsp/`, and builds the Linux client.

## 4. Directory map

| Path | What it is |
|---|---|
| `run.sh` | Starts the patched daemon with the unsigned shell in `shell/` and one client program |
| `shell/` | The FastRPC shell matching the firmware (not in git; see SETUP.md) |
| `docs/`, `scripts/`, `systemd/` | Documentation, dependency fetch script, boot-test service template |
| `cdsp78.sh` | Restarts the cDSP on the 00078 firmware pair from `fw78/` (RAM only) |
| `boot-run.sh`, `boot-run.log` | One-shot boot test and its log (service is disabled) |
| `add/` | Step 1: add two numbers on the DSP |
| `probe/` | Symbol probing: one tiny library per candidate function name |
| `hmx/` | Step 2a: reserve, power, lock, execute one HMX instruction |
| `mm/` | Step 2b: one 32x32 FP16 multiply, uniform (`mm_client`) and random (`mm2_client`) |
| `nn/` | Step 3: 256-32-10 classifier on font-rendered digits, one call per layer |
| `npu/` | Step 4: persistent session, any layer widths, MNIST, benchmark |
| `ids/` | First application: network flow anomaly scoring (section 18) |
| `pv/` | Power-vote probe: times a fixed loop on the DSP around one request |
| `root/dsp/cdsp/` | Directory the daemon serves to the DSP as its library path |
| `fw78/` | Build-00078 cDSP firmware and device-tree file, copied from the Windows backup (not in git) |
| `fw/` | Generic build-00094 pair from Fedora's `qcom-firmware` (rejected by this machine; not in git) |
| `hexagonrpc/` | linux-msm/hexagonrpc at `da7a374` plus our patch; built in `hexagonrpc/build` (fetched by `scripts/fetch-deps.sh`) |
| `patches/hexagonrpcd-npu-open.patch` | That patch as a diff |
| `qcom-fastrpc/` | qualcomm/fastrpc at `1d099a6`, used only for its open error-code header |
| `hexagon-dsp-binaries/` | linux-msm clone at `862bb2b` (1.7 GB); its shells do not match, can be deleted |
| `ref/llvm/`, `ref/llamacpp/`, `ref/manual/` | Reference material fetched from public sources |
| `data/mnist/` | The four standard MNIST files (11.6 MB) from `ossci-datasets.s3.amazonaws.com/mnist/` |

## 5. Firmware and shell

- A user process on the DSP needs a "shell" ELF (`fastrpc_shell_unsigned_3`), passed in by the
  Linux side when the process is created.
- **The shell must match the firmware build.** With any other build, process creation
  (`FASTRPC_IOCTL_INIT_CREATE`) returns `0x8000054f` (`AEE_EQURTBADASID`). A Qualcomm maintainer
  describes a hash check between shell and firmware
  ([qualcomm/fastrpc#400](https://github.com/qualcomm/fastrpc/issues/400)).

| Firmware build | Where it is | Result |
|---|---|---|
| `CDSP.HT.2.9.c1-00046` | Installed in `/lib/firmware/qcom/x1e80100/microsoft/Denali/` (Dell copy from Fedora) | Boots; no matching shell available |
| `CDSP.HT.2.9.c1-00078` | Windows DriverStore package `qcnspmcdm_ext_cdsp8380.inf_arm64_4a8c3ebe3aad408a/` | Boots; matching shell in the same package's `CDSP/` folder. **This is the one used** |
| `CDSP.HT.2.9.c3-00004` | Windows DriverStore package `qcsubsys_ext_cdsp8380.inf_arm64_9ed31fd1359980a9/` | Not tried |
| `CDSP.HT.2.9.c1-00094` | Fedora `qcom-firmware` generic `cdsp.mbn.xz` | Rejected: `error -22 initializing firmware` |

- Public shells in `linux-msm/hexagon-dsp-binaries` are builds 00069, 00082 and 00094; all three
  were refused against the 00046 firmware.
- Firmware is swapped without replacing files: set
  `/sys/module/firmware_class/parameters/path` to `fw78`, stop and start `remoteproc1`, then clear
  the path. The kernel's crash recovery reloads from the installed path, hence `cdsp78.sh`.
- To clear the path write a newline (`echo | tee`); writing zero bytes leaves it set.

## 6. Linux side: the daemon

`hexagonrpcd` (open source) creates the DSP process, then serves the DSP's file requests. Our
patch (`patches/hexagonrpcd-npu-open.patch`) adds:

- `-u`: create an unsigned process (`FASTRPC_MODE_UNSIGNED_MODULE`).
- Error reporting on `INIT_CREATE`. The stock code prints a stale `errno`; DSP errors arrive as
  the ioctl's positive return value.
- `apps_mem` method 3 (`request_unmap64`), which the DSP calls and stock `hexagonrpcd` lacks.
- `apps_mem` map64 now returns the full mapped address. Stock `hexagonrpcd` truncates it to 32 bits,
  and on this machine the DSP's heap growth then fails (`0x80000468`), which broke every call
  larger than about 100 KB. Set `HRPC_VADSP32=1` to get the old behaviour.
- It is built with `-Dhexagonrpcd_verbose=true`, which logs every file operation the DSP asks for.
  That log is how the loader's requirements were found.

Other facts:

- `-p PROGRAM` starts a client that inherits the FastRPC fd in `HEXAGONRPC_FD`; clients use
  `libhexagonrpc`.
- The daemon maps `<root>/dsp/<dsp name>/` as the DSP's library directory (`-R root -d cdsp`).
- Closing the last library handle ends the DSP process and the daemon's listener
  ("Could not fetch next FastRPC message" at exit is that, and harmless).
- Device access needs group `fastrpc` (udev rule from the `hexagonrpc` package).
- Do not SIGKILL a client mid-call. A failed process creation followed by exit crashed the cDSP
  once (`sleep_statsi.c:537`).

## 7. DSP side: building a loadable library

```bash
clang --target=hexagon-unknown-none-elf -mv73 -O2 -fPIC -G0 -ffreestanding -fno-builtin -c x.c
ld.lld -shared --hash-style=sysv -z max-page-size=4096 -z norelro --no-rosegment \
       -z separate-loadable-segments -Bsymbolic -o libNAME_skel.so x.o
```

- The loader accepts **exactly two loadable segments, the second starting on a 4 KB page boundary
  in both file and memory**. Anything else fails with `0x80000406` (`AEE_EUNABLETOLOAD`).
- Opening interface `NAME` loads `libNAME_skel.so` and calls
  `int NAME_skel_invoke(uint32_t sc, remote_arg *pra)`.
- `remote_arg` is `union { struct { void *pv; uint32_t len; } buf; uint32_t h; }`.
- `sc`: method in bits 28:24, input buffer count in 23:16, output buffer count in 15:8.
- A missing imported symbol also gives `0x80000406`; a missing `_skel_invoke` gives `0x80000442`.
- Undefined symbols are resolved against the shell at load time, so libc-style functions
  (`memset`, `memcpy`, `malloc`) can simply be declared and used.
- The same C code can be checked first under `qemu-hexagon-static` (see `add/emu_test.c`).

## 8. Finding function names by probing

`probe/gen.sh` builds one library per name in `probe/names.txt`, each needing that symbol;
`probe_client` reports which ones load.

| Exported by the shell | Not exported |
|---|---|
| `HAP_power_set`, `HAP_power_get` | `HAP_compute_res_*` (these are header-only wrappers) |
| `compute_resource_attr_init`, `_attr_set_hmx_param`, `_attr_set_vtcm_param`, `_attr_set_vtcm_param_v2`, `_attr_get_vtcm_ptr`, `_attr_get_vtcm_ptr_v2`, `_attr_set_serialize`, `_attr_set_cache_mode` | `qurt_hmx_lock`, `qurt_hmx_unlock` |
| `compute_resource_acquire`, `_release`, `_acquire_cached`, `_release_cached`, `_hmx_lock`, `_hmx_unlock`, `compute_resource_query_VTCM` | |
| `HAP_query_total_VTCM`, `HAP_query_avail_VTCM`, `HAP_request_VTCM`, `HAP_release_VTCM` | |
| `qurt_hvx_lock`, `qurt_hvx_unlock`, `qurt_hvx_get_units`, `qurt_thread_get_id` | |
| `HAP_mem_get_stats`, `HAP_debug_v2`, `HAP_perf_get_time_us`, `memset`, `malloc` | |

Signatures that work (argument shapes taken from how llama.cpp calls the wrappers):

```c
int  compute_resource_attr_init(void *attr);                 /* attr: opaque; 512 bytes reserved */
int  compute_resource_attr_set_hmx_param(void *attr, unsigned char enable);
int  compute_resource_attr_set_vtcm_param_v2(void *attr, u32 size, u32 min_page, u32 min_size);
int  compute_resource_attr_get_vtcm_ptr_v2(void *attr, void **ptr, u32 *size);
u32  compute_resource_acquire(void *attr, u32 timeout_us);   /* returns a context id, 0 on failure */
int  compute_resource_hmx_lock(u32 ctx);
int  compute_resource_hmx_unlock(u32 ctx);
int  compute_resource_release(u32 ctx);
int  HAP_power_set(void *context, void *request);
u64  HAP_perf_get_time_us(void);
```

- Requesting 1 MB of VTCM with all three size arguments equal returns `0xff000000`, size 1 MB.
- The acquired context survives across FastRPC calls; lock and unlock are done per call.

## 9. Powering the accelerator

- **Locking HMX without powering it resets the DSP** (watchdog, "SFR Init: wdog or kernel error
  suspected") a few seconds later, even though lock and unlock return 0.
- No public description of the power request exists, so it was probed.

`HAP_power_set` return codes by request type, with a zeroed request plus 1 at offsets 4 and 8:

| Type | Return |
|---|---|
| 1, 2, 3, 7, 10, 12, 13 | 0 |
| 8, 9, 11 | `0x80000454` (`AEE_EBADCONTEXT`) |
| 0, 4, 6 | -3 |
| 5 | -2 |
| 14 and above | -3 |

- **Type 13 powers HMX.** Types 10 and 12 return 0 but do not (lock still resets the DSP).
- **Layout: `{ u32 type = 13; u32 unused; u8 power_up at offset 8 }`.** Setting only offset 4, or
  neither, leaves it unpowered.
- Types 8, 9 and 11 probably need a real context; untested.

### Clock and performance requests

Measured with `pv/` (a fixed 2,000,000-iteration scalar loop on the DSP, timed before and after
one request):

| State | Loop time |
|---|---|
| No request from our process | 8.54 ms |
| After any accepted request (types 1, 2, 3, 7, 10, 12, 13 with small values) | 22.66 ms |
| After a MIPS request of 500 | 8.54 ms |
| After a MIPS request of 1000, 2000 or 4000 | 7.14 ms |

- **Any accepted vote, including the HMX power-on, drops the DSP to a low clock** unless a
  performance request is also made. Everything before this finding ran about 2.65x slower than
  the DSP's default.
- **MIPS request (type 1), layout found by test:**
  `{ u32 type = 1; u32 unused; u8 set_mips @8; u32 mips_per_thread @12; u32 mips_total @16 }`.
  It plateaus from 1000 upward, so it probably cannot reach the highest operating point.
- **Sleep-latency fields (guess):** `{ u8 set_latency @34; i32 latency_us @36 }` in the same
  request. With 100 µs the cost per call fell about 14% in two measurements; not otherwise
  verified.
- These are requests to the DSP's power manager, which chooses among its own operating points.
  Highest temperature on any sensor stayed at 37-38 °C throughout; runs last a few seconds.
- No DSP reset occurred in any of these tests.

## 10. HMX instructions

Encodings from LLVM `HexagonDepInstrInfo.td` (`ref/llvm/`). Fedora's assembler does not know
them, so they are emitted as raw words (`mm/hmxops.S`). Bits 15:14 are the packet parse bits
(`01` = more instructions follow in the packet, `11` = last).

| Instruction | Fixed bits | Word used (registers) |
|---|---|---|
| `mxclracc.hf` | `Inst{31-16}=0xA6E0`, `Inst{13-0}=0x0013` | `0xa6e0c013` |
| `bias = mxmem2(Rs)` | `Inst{31-21}=0x490`, `Inst{13-0}=0x03FE` | `0x9200c3fe` (r0) |
| `activation.hf = mxmem(Rs,Rt)` | `Inst{31-21}=0x490`, `Inst{13}=0`, `Inst{7-0}=0xE4` | `0x920041e4` (r0, r1; first in packet) |
| `weight.hf = mxmem(Rs,Rt)` | `Inst{31-21}=0x490`, `Inst{13}=1`, `Inst{7-0}=0xEF` | `0x9202e3ef` (r2, r3; last in packet) |
| `cvt.hf = acc(Rs)` | `Inst{31-21}=0x537`, `Inst{13-0}=0x1A10` | `0xa6e0da10` (r0) |
| `mxmem(Rs,Rt) = cvt` | `Inst{31-21}=0x537`, `Inst{13}=0`, `Inst{7-0}=0x18` | `0xa6e0c118` (r0, r1) |

`Rs` sits in bits 20:16 and `Rt` in bits 12:8. Not yet used: `:deep` and `:single` activation
(`Inst{7-0}` = `0xE2`, `0xF8`), `:deep` weight (`0xF4`), `mxmem2(Rs) = bias` (`Inst{13-0}=0x0016`).

- The activation and weight instructions must be in one packet.
- All data the instructions touch must be in VTCM.

## 11. Data layouts and register values (FP16)

One multiply computes, for 32 "spatials" `s`, 32 output channels `o`, 32 input channels `i`:

`acc[s][o] += sum_i act[s][i] * wt[o][i]`

The accumulator persists across multiplies until cleared, which is how more than 32 input
channels are handled (one multiply per 32-channel tile).

| Tile (2 KB, 1024 FP16 cells) | Cell index `p` holds |
|---|---|
| Activation | `s = 2*(p/64) + p%2`, `i = (p%64)/2` |
| Weight | `i = 2*(p/64) + p%2`, `o = (p%64)/2` |
| Output | `s = 2*(p/64) + p%2`, `o = (p%64)/2` |

- An output tile therefore has exactly the layout of an activation tile and can be fed to the
  next layer unchanged.

| Operand | Value used | Meaning |
|---|---|---|
| activation `Rs` | tile address (2 KB aligned) | offset and channel-start fields zero |
| activation `Rt` | `2047` | no next tile, spatial mask all ones, channel stop 31 |
| weight `Rs` | weight address (128 B aligned) | no negate |
| weight `Rt` | `2047` | distance to last 128 B vector (`0x780`) plus low 7 bits set |
| bias `Rs` | address (256 B aligned) | set index in bits 1:0 (see deviations) |
| convert `Rs` | `0`, or `1` to keep the accumulator | bias select in bits 13:12 |
| store `Rs`, `Rt` | 2 KB aligned address, `0x782` | spatial mask all ones |

Bias block: 256 bytes, 32 low words then 32 high words, one 64-bit entry per output channel.

| Bits | Field |
|---|---|
| 15:0 | scale (FP16) |
| 31:16 | output bias (FP16) |
| 35:32, 39:36 | extra mantissa bits for scale and output bias |
| 42:40 | shaping function: 0 `x`, 1 `min(x,0)`, 2 `max(x,0)`, 3 `abs`, 4-7 the negatives of those |
| 63:43 | input bias |

Convert computes `shape(acc + input_bias) * scale + output_bias`.

## 12. Where this chip differs from the manual

The manual is written for v81; this part is v73.

- **Output is clamped at zero.** The stored result is `max(0, shape(acc)*scale + output_bias)`
  for every shaping code. Tested with all eight shapes, scale -1, output bias +2 and -2, and
  convert bits 6 and 7. Cause unknown.
- **Only one bias set seems to exist.** Loading with set index 1 overwrote the set in use.
- **Workaround for signed results:** convert twice from the same accumulator (first with `Rs=1`
  to keep it), with scale +1 then scale -1 and output bias negated; each cell is non-zero in at
  most one pass, so the signed value is rebuilt with integer operations.
- **ReLU is therefore free**: a hidden layer needs a single convert pass.
- A linear layer's bias goes in the output-bias field.

## 13. Results

| Test | Result |
|---|---|
| `add`: 1234 + 4321 on the DSP | 5555 |
| `mm_client`: uniform tiles, three value sets | 1024 of 1024 cells correct each |
| `mm2_client`: random signed 32x32, seeds 1-3 | 1024 of 1024 cells match the CPU each |
| `nn`: 256-32-10, digits rendered from installed fonts, 38 held-out fonts | 96.39% accelerator and CPU, 1024 of 1024 agree |
| Same test run by a boot service 10 s after a cold boot | passed |
| `npu`: MNIST 784-128-10, 10,000 test images | 97.75% accelerator and CPU, 10,000 of 10,000 agree, largest logit difference 0.016 |

MNIST timing per image (accelerator time measured on the DSP clock):

| Configuration | On the DSP | Including the FastRPC call |
|---|---|---|
| First version: sign merge on the DSP, low clock, 2 batches per call | 9.3 µs | about 24 µs |
| Sign merge moved to the CPU | 3.4 µs | about 18 µs |
| Plus MIPS request 2000 | 1.1 µs | about 11 µs |
| Plus latency hint, 8 batches per call | 1.4 µs | about 8 µs |
| Plus latency hint, 32 batches per call | 2.5 µs | about 7 µs |
| Shared dma-buf buffers, 8 batches per call | 2.7 µs | 4.4 µs |
| Shared buffers, persistent mapping, 32 batches per call | 2.7 µs | 3.2 µs |
| Plus HVX vector copies, 8 batches per call | 0.33 µs | 1.4 µs |
| **Plus HVX vector copies, 32 batches per call (current default)** | **0.39 µs** | **1.0 µs** |

| Reference | Time per image |
|---|---|
| Packing inputs into tiles on the CPU | about 3.5 µs |
| CPU, plain C loops, one core | about 80 µs |
| CPU, numpy with BLAS, 10 cores, whole test set at once | 0.74 µs |

Shared buffers (`SHARED=1`): the client allocates dma-bufs with `FASTRPC_IOCTL_ALLOC_DMA_BUFF`,
maps them, optionally asks for a mapping that outlives each call (`FASTRPC_IOCTL_MEM_MAP`,
`PERSIST=1`), and passes them to `FASTRPC_IOCTL_INVOKE` by fd so the kernel maps instead of
copying. `libhexagonrpc` always passes fd -1, so the run call uses the ioctl directly. This also
removes the DSP heap growth that large copied calls trigger.

HVX copies (`FLAGS` bit 2): scalar `memcpy` on the DSP reads a shared mapping at about 750 MB/s
and reads VTCM even more slowly; copying whole 128-byte vectors (`npu/hvxcopy.c`, built with
`-mhvx -mhvx-length=128B`) took copy-in from 2.1 to 0.3 µs per image and copy-out from 0.55 to
0.01 µs. `qurt_hvx_lock(1)` returns 0 and is taken around each run; the mode value 1 for
128-byte vectors is an assumption that worked. Buffers must be 128-byte aligned, which is why
the output header is 128 bytes.

The CPU-side cost outside these figures is converting float32 inputs to FP16 tiles. A plain C loop
took about 3.6 µs per image. `npu/npu_pack.h` does it with NEON (convert four floats at a time,
interleave two samples per 128-byte row): 0.06 µs per image in isolation (`npu/pack_test.c`, which
also checks it against the per-cell formula for padded shapes) and about 0.43 µs when writing
straight into the shared buffer. End to end MNIST is therefore about 1.4 µs per image.

Where the DSP time went at MIPS 2000, copied buffers, 2 batches per call: copy-in 0.54 µs,
multiply and convert 0.01 µs, copy-out 0.53 µs. The timer counts whole microseconds per interval, so the multiply
figure is a lower bound, but the matrix work is clearly not the bottleneck.

- Reading VTCM from scalar code is slow: the 4 KB copy-out costs as much as the 50 KB copy-in,
  and the sign-merge loop (which reads VTCM) took 6.6 µs per image before it was moved.
- A call costs roughly 0.45 ms fixed plus the time to move its data (about 1.5 KB per image).
- One "deep" multiply per up-to-32 tiles replaces one multiply per tile (`FLAGS` bit 0); results
  are identical.

Weights: 208 KB of tiles, loaded once in about 4 ms.

## 14. Open problems

- **What is left of the time** is split between the call itself (about 0.6 µs per image at 32
  batches per call), CPU-side packing into the shared buffer (0.43 µs) and the DSP's copy into
  VTCM (0.36 µs). The matrix work is about 0.01 µs.
- **Highest operating point not reached.** The MIPS request plateaus; the DCVS-style request
  types are not understood.
- **Convolution** is not started. The manual gives the building blocks (spatial offset,
  "single" activation, channel start/stop, 64 filters with "deep" weights) but no recipe.
- **Zero clamp and single bias set** are unexplained.
- **Rounding** is uncharacterised beyond "largest logit difference 0.016 on MNIST".
- **Layer limits:** fully connected only; widths are padded to multiples of 32; 1 MB of VTCM
  bounds the model (weights are `nk * ng` tiles of 2 KB per layer).
- **Firmware:** depends on build 00078 being reloaded after each boot or crash.
- Possible kernel-side issue seen by others on X1E laptops ("No context ID matches response",
  [qualcomm/fastrpc#418](https://github.com/qualcomm/fastrpc/issues/418)) was not observed here.

## 15. Crashes seen

| Cause | Effect |
|---|---|
| Exit after failed process creation (mismatched shell) | cDSP fatal `sleep_statsi.c:537`, auto-recovered |
| HMX locked without power (first attempt) | cDSP watchdog reset |
| Power request types 10 and 12, then lock | cDSP watchdog reset each |
| Power-field test with the flag at offset 4 only, and with neither | cDSP watchdog reset each (deliberate) |

Every one recovered automatically within seconds. No kernel BUG, no whole-machine hang, and the
aDSP and sensor service were never affected.

## 16. Changes made to the system

- Packages installed: `clang`, `lld`, `llvm`, `qemu-user-static-hexagon` (and dependencies).
- The user was added to group `fastrpc`.
- `/etc/systemd/system/npu-open-boot-test.service` created; it ran once and is now **disabled**.
- Nothing under `/lib/firmware` was changed. Firmware swaps are RAM-only.

## 17. Sources and licences

| Source | Licence | Used for |
|---|---|---|
| [llvm/llvm-project](https://github.com/llvm/llvm-project) `HexagonDepInstrInfo.td`, `hmx_hexagon_protos.h` | Apache-2.0 with LLVM exception | HMX instruction encodings and names |
| [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) `ggml/src/ggml-hexagon/htp/` | MIT | Call sequence for power, resource and lock; operand values such as 2047 |
| [qualcomm/fastrpc](https://github.com/qualcomm/fastrpc) `inc/AEEStdErr.h`, `src/apps_mem_imp.c` | BSD-3-Clause | Error-code names, `apps_mem` method order |
| [linux-msm/hexagonrpc](https://github.com/linux-msm/hexagonrpc) | GPL-3 (library LGPL-3) | Daemon and client library, patched |
| [Qualcomm HMX Programmer's Reference Manual 80-N2040-62](https://docs.qualcomm.com/doc/80-N2040-62/80-N2040-62_REV_AA_Qualcomm_Hexagon_V81_HMX_Programmers_Reference_Manual.pdf) | Public document | Data layouts, register fields, convert equation |
| [linux-msm/hexagon-dsp-binaries](https://github.com/linux-msm/hexagon-dsp-binaries) | Qualcomm redistributable binary licence | Tried; shells did not match |
| Owner's Windows driver backup | Microsoft / Qualcomm, shipped with the device | Build-00078 firmware and shell, run unmodified |
| MNIST | Public dataset | Training and test data |

Background reading: [Surface Pro 11 NPU guide](https://github.com/denisix/ubuntu-surface-pro-11),
[Yoga Slim 7x NPU guide](https://github.com/hastyheat-com/x1e-yoga-slim7x-npu-linux).

Our own code in this directory has no licence chosen yet. The patched `hexagonrpcd` is a
derivative of GPL-3 code.

## 18. First application: network flow anomaly scoring (`ids/`)

A small intrusion-scoring pipeline that runs its model on the accelerator. It is a demonstrator, not a
validated intrusion detection system.

| Stage | What happens | Where |
|---|---|---|
| Capture | `dumpcap -s 96`: packet headers only, no payloads | `ids/live.sh`, `ids/cap/` |
| Features | One record per flow per 5-second window, 32 features (packet and byte counts, sizes, gaps, TCP flag shares, port class, and how many flows, ports and hosts the same source touched) | `ids.py features` |
| Model | Autoencoder 32-32-8-32-32, trained with numpy on baseline flows; the score is the reconstruction error | `ids.py train` |
| Scoring | Both encoder and decoder run on the accelerator through `npu/npu_client` (`OUT=` saves raw outputs) | `ids.py eval`, `ids.py report` |

Addresses are never used as features. Features are standardised with the baseline's mean and
deviation and clipped to +-8.

Data used on 2026-11-24:

- Baseline: 150 s on loopback and Wi-Fi while `gen_benign.sh` fetched files from a local web server:
  584 flow records (350 train, 116 validation, 118 test).
- Attack-like: `gen_attack.sh` against 127.0.0.1 only (connect scan of 3,000 ports, SYN scan of 3,000,
  UDP sweep of 300, 550 rapid short connections): 6,835 flow records.

| Result | Accelerator | CPU |
|---|---|---|
| Area under the ROC curve | 1.0000 | 1.0000 |
| Attack records above the threshold | 100% of 6,835 | 100% |
| Benign test records above the threshold | 5.1% of 118 | 5.1% |
| Same alarm decision | 6,953 of 6,953 | |

- Largest score difference between accelerator and CPU: 0.003, against a threshold of 0.199.
- Time: 0.19 µs per flow record including the call (7 calls for 6,953 records).
- Live test (`live.sh 25` with a 200-port connect scan in the middle): every scan connection scored
  7-9; ordinary flows scored below 1.
- Quiet live test (`live.sh 30`, no scan): 6 of 25 ordinary flow records exceeded the threshold, with
  scores of 0.2-0.9.

Limits:

- **The baseline is far too short.** 150 seconds of mostly local web fetches does not cover normal
  traffic, which is why a quarter of ordinary flows in the quiet test raise alarms. Hours of real
  traffic are needed before the threshold means anything.
- **The attacks are the easy kind.** Port scans and connection bursts look nothing like ordinary
  flows. The perfect separation says little about subtler behaviour.
- **Source-context features leak across flows:** an ordinary loopback flow in the same window as a
  scan from the same address inherits the scan's counts.
- The first baseline capture kept full loopback packets (the snap length option was placed after the
  interfaces); it was trimmed to 96 bytes afterwards and the scripts now put `-s 96` first.

To score live traffic:

```bash
ids/live.sh 30
```
