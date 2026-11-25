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
| `rust/` | Rust crate: FastRPC layer, tile conversion, session, `server` and `usecases` binaries (sections 19-20) |
| `serve.sh` | Start, stop or query the NPU server |
| `CLAUDE.md` | Working rules: prefer Rust, document everything, clean-room and safety rules |
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

Our own code is under the MIT licence (`LICENSE`). The `hexagonrpcd` patch modifies GPL-3 code and is
GPL-3.0-or-later (`patches/README.md`).

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

## 19. The NPU server (`rust/`)

Written in Rust with the standard library only (no crates). 1,727 lines including the
demonstrations. Rust 1.100 from Fedora; `cargo build --release` needs no network.

| Module | Role |
|---|---|
| `rust/src/fastrpc.rs` | The kernel's FastRPC ioctls, called directly through `extern "C" ioctl/mmap` |
| `rust/src/tiles.rs` | float32 to FP16 tiles and back |
| `rust/src/session.rs` | One persistent accelerator session on `libnpu_skel.so` |
| `rust/src/proto.rs` | Unix-socket protocol and the client |
| `rust/src/bin/server.rs` | `server`: accepts one client at a time on `npu.sock` |
| `rust/src/bin/usecases/` | `usecases`: the 25 demonstrations (section 20) |

DSP library methods (`npu/npu_skel.c`): 0 open, 1 load one group, 2 run, 3 close, 4 load many groups.

Why a server: the FastRPC file descriptor only exists inside a process started by
`hexagonrpcd -p`, and the accelerator session is tied to it. The server holds both and lets any
number of short-lived programs use the accelerator through a socket.

```bash
./serve.sh start
```

```bash
rust/target/release/usecases all
```

```bash
./serve.sh stop
```

### FastRPC details the Rust code relies on

All confirmed by the server working, unless marked.

| Fact | Value | Known from |
|---|---|---|
| ioctl number | `(3 << 30) \| (size << 16) \| ('R' << 8) \| nr` | `_IOWR` in the kernel UAPI |
| `INVOKE` | nr 3, `struct { u32 handle; u32 sc; u64 args; }` (16 bytes) | `misc/fastrpc.h` |
| `ALLOC_DMA_BUFF` | nr 1, `struct { s32 fd; u32 flags; u64 size; }` (16 bytes) | `misc/fastrpc.h` |
| `MEM_MAP` | nr 10, `struct fastrpc_mem_map`, 64 bytes with padding | `misc/fastrpc.h` |
| Argument | `struct { u64 ptr; u64 length; s32 fd; u32 attr; }`, inputs first, then outputs | `misc/fastrpc.h` |
| Return value | the DSP's return code; -1 with errno when the kernel refuses | experiment |
| Opening a library | handle 0, method 0, 2 inputs and 2 outputs: `{ name_len, err_len }`, name with NUL, `{ handle, dlerr }`, error text | libhexagonrpc's definition; the hand-built call works |
| Our own methods | words in buffer 0, each blob in its own buffer (`npu_skel.c` `blob()`) | our design |
| Close call | sent with one dummy input word; a call with no buffers was not tried | assumption avoided |

### Tile conversion in Rust

- `fcvtn v.4h, v.4s` and `zip1` through `core::arch::asm!` convert four floats from each of two
  rows and interleave them. These are base ARMv8 NEON, so no target feature flag is needed.
  Rust's `f16` type and the NEON FP16 intrinsics are not stable, which is why it is inline
  assembly.
- FP16 to float32 for results: `unpack_row` handles one 128-byte row at a time with `uzp1`/`uzp2` (split the
  two interleaved samples), `fcvtl` (to float32) and, for signed outputs, `cmeq`/`cmtst`/`orr`/`bsl` to take the
  negated second pass where the first is zero. A scalar exact conversion (`f16_to_f32`) remains for tests.
- `cargo test` runs four unit tests on this module: conversions and round trips, packing against the
  per-cell formula for six padded shapes, unpacking including the sign merge, and weight tile roles.

### Session behaviour

| Item | Value |
|---|---|
| VTCM requested | 4 MB first, then 1 MB. **4 MB was granted** (the C client only ever asked for 1 MB). A 16 MB request was refused |
| VTCM page sizes | assumed to be powers of four (1, 4, 16 MB), so 2 and 8 MB are not tried; unverified |
| Shared input buffer | 4 MB dma-buf, persistent mapping |
| Shared output buffer | 8 MB dma-buf, persistent mapping, 128-byte header (2 MB until 2026-11-25; it limited a 2,048-output job to 7 batches per call) |
| Batches per call | up to 32, limited by the shared buffer sizes |
| Layers | up to 8; each ReLU or signed; a signed layer that is not last is merged on the DSP (slow path) |
| Redefining the model | close (method 3), open again, load weights; the shared buffers are reused |
| Loading weights | DSP method 4: all output groups of a layer in one call (as many as fit the 4 MB shared input buffer), copied to VTCM with HVX. A 2,000 x 784 layer loads in 5.6 ms (0.5 close and open, 3.9 packing on the CPU, 0.9 the call). Before, one call per group: 63 calls, 25-35 ms |
| Timing output | start the server with `NPU_SERVER_TIMING=1` to log each model load's breakdown in `server.log` |
| Power | MIPS request 2000 and latency hint 100 µs at every open |

### Socket protocol (`npu.sock`, little-endian)

| Request | Body | Reply after the status word |
|---|---|---|
| `M` define | u32 layers; per layer u32 n_in, n_out, relu; then per layer f32 weights `[n_out][n_in]`, f32 bias | u32 VTCM kilobytes |
| `R` run | u32 rows; f32 `x[rows][n_in]` | u64 dsp_us, u64 wall_us, u64 pack_us, u32 calls, u32 n_out, f32 `y[rows][n_out]` |
| `A` argmax | as `R` | u64 dsp_us, u64 wall_us, u64 pack_us, u32 calls, then per row u32 index of the largest output and f32 value |
| `Q` quit | none | none |

`A` exists because search-style uses need only the best match per row: the server takes the maximum while
unpacking and returns two numbers per row, where `R` returns `rows x n_out` floats.

Every reply starts with u32 status: 0, or 1 followed by u32 length and a UTF-8 message.

### Limits

- One client at a time; a second connection waits.
- The model must fit in VTCM: `nk * ng` tiles of 2 KB per layer (`nk` = inputs / 32, `ng` = outputs
  / 32, both rounded up) plus the activation areas. With 4 MB that is roughly 2 million weights.
- The socket has no authentication; it is a file in the project directory with the user's
  permissions.
- `serve.sh stop` sends `Q`, then SIGTERM to the idle daemon. It never sends SIGKILL.

## 20. Twenty-five demonstrations (`rust/src/bin/usecases/`)

Asked for by the owner: math, cyber defence, networking and radio, with no trained models and the
work done on the accelerator. Nothing here uses gradient descent. Weights are random, analytic
(Fourier transform, templates, steering vectors), counted (letter pairs, centroids) or fitted in
closed form (PCA by Jacobi rotations, least squares by Cholesky), all hand-written in
`usecases/util.rs`.

Each line reports the accelerator's time per row on the DSP clock and the total including packing
and the call. "verified" means the accelerator's output was compared with the same matrix product
computed in float32 on the CPU and the largest difference was under 2-5% of the output's RMS; the
CPU figure itself is not shown. Run of 2026-11-25:

```
 1 | Batched matrix multiply      | 4096x256 times 256x128             | 268.4 billion multiply-adds/s, max error 0.22% of RMS |    0.12 us/row NPU,    0.63 us/row total | verified
 2 | Random projection            | 1024 dims to 64, 1024 vectors      | pairwise distances kept within 7.2% on average       |    0.42 us/row NPU,    1.31 us/row total | verified
 3 | PCA scoring                  | 4096 rows, 64 dims to 8 components | 99.7% of the variance captured                       |    0.04 us/row NPU,    0.26 us/row total | verified
 4 | Similarity search            | 4 queries against 182 text lines   | top hit for query 1: "- **Locking HMX without powering it rese..." |    2.25 us/row NPU,   42.25 us/row total | verified
 5 | Kernel random features       | 512 vectors to 1024 ReLU features  | kernel values reproduced within 7.9%                 |    0.16 us/row NPU,    0.86 us/row total | verified
 6 | Regression inference         | 8192 rows, 32 inputs, 16 targets   | R squared 0.9986 against held-out targets            |    0.03 us/row NPU,    0.24 us/row total | verified
 7 | Function surrogate           | 2 layers: 3 to 256 ReLU to 1       | RMS error 0.124 against sin(3a)+b*c                  |    0.06 us/row NPU,    0.29 us/row total | verified
 8 | Intrusion scoring (PCA)      | 7419 flow records, 32 features     | 100.0% of scan flows flagged, 2.4% false alarms      |    0.03 us/row NPU,    0.24 us/row total | verified
 9 | Log anomaly                  | 2102 kernel log lines              | 22 unusual lines; oddest: "random: crng init done"   |    0.08 us/row NPU,    0.53 us/row total | verified
10 | Lookalike domains            | 16 names against 16 protected      | 8 flagged, e.g. paypa1.com resembles paypal.com      |    0.50 us/row NPU,   13.88 us/row total | verified
11 | File triage                  | 120 files, byte histograms         | 120 of 120 sorted into program, text or compressed   |    0.09 us/row NPU,    1.02 us/row total | verified
12 | Generated domain names       | 1000 names, 729 letter-pair weights | 98.4% of word-like and random names told apart       |    0.17 us/row NPU,    0.67 us/row total | verified
13 | Packet-size patterns         | 19985 windows against 8 patterns   | 78 of 78 embedded patterns found, 0 false hits       |    0.02 us/row NPU,    0.21 us/row total | verified
14 | Behaviour baseline           | 7419 flow records whitened         | median distance 2.8 for baseline, 42.7 for scan traffic |    0.02 us/row NPU,    0.24 us/row total | verified
15 | Traffic clustering           | 7419 flows, 6 centres, 8 rounds    | 92.1% of flows in clusters that are all-scan or all-baseline |    0.03 us/row NPU,    0.23 us/row total | not checked
16 | First-packet signatures      | 4000 connections against 64 signatures | 100.0% matched to the right signature                |    0.03 us/row NPU,    0.30 us/row total | verified
17 | Link quality prediction      | 8192 samples, 8 features (synthetic) | R squared 0.9464 on unseen samples                   |    0.02 us/row NPU,    0.28 us/row total | verified
18 | Latency forecasting          | 6000 one-step forecasts, 16 taps (synthetic) | RMS error 0.215 against 0.220 for repeating the last value |    0.03 us/row NPU,    0.25 us/row total | verified
19 | Telemetry sketching          | 7419 flow records, 32 numbers to 8 | distances between records kept within 39% on average |    0.03 us/row NPU,    0.28 us/row total | verified
20 | Modulation recognition       | 1200 spectra of 128 points (synthetic) | 398 of 400 bursts named correctly; per type [100, 100, 100, 98] of 100 (2-tone FSK, BPSK, QPSK, 8PSK) |    0.14 us/row NPU,
21 | Spectrum occupancy           | 1024 frames, 128-point transform (synthetic) | occupied bins found: [10, 37, 90]                    |    0.15 us/row NPU,    0.62 us/row total | verified
22 | Transmitter fingerprinting   | not run                            | needs recordings from real transmitters; no radio was attached
23 | Matched-filter bank          | 29937 windows against 32 sequences | 66 of 66 buried sequences found at 0 dB; 13 extra hits, 12 beside a real one |    0.04 us/row NPU,    0.28 us/row total | verif
24 | Beamforming                  | 2048 snapshots, 16 antennas, 91 directions | sources located at [-20.0, 36.0] degrees (true -20 and 36) |    0.06 us/row NPU,    0.46 us/row total | verified
25 | Denoising                    | 4096 frames of 64 samples (synthetic) | signal-to-noise ratio raised from 5.2 dB to 15.4 dB  |    0.04 us/row NPU,    0.35 us/row total | verified
```

Where the data comes from:

| Cases | Data |
|---|---|
| 1-3, 5-7 | Generated with a fixed-seed xorshift generator |
| 4 | The lines of this document |
| 8, 14, 15, 19 | The flow records captured for section 18 (`ids/ids.py export` writes them as `.f32`) |
| 9 | This boot's kernel log (`journalctl -k -b`) |
| 10 | A built-in list of 16 protected names and 16 candidates |
| 11 | Files under `/usr/bin`, `/usr/share/man/man1`, `/usr/share/doc` |
| 12 | `/usr/share/dict/words` (half for counting, half for testing) and random strings |
| 13, 16-18, 20-25 | Synthetic; no radio was attached and no link statistics were recorded |

What the results do and do not show:

- **Weak results.** 18 (forecasting) barely beats repeating the last value (0.215 against 0.220):
  the synthetic bursts are unpredictable by a linear filter. 19 (sketching to 8 numbers) distorts
  distances by 39%, which is what 8 dimensions give. 15 (clustering) has no reference check.
- **Hand-set thresholds.** 10, 13, 20, 21 and 23 use thresholds chosen by looking at these inputs.
- **22 is not run.** Transmitter fingerprinting needs recordings of real transmitters.
- **Synthetic is not real.** The radio and several networking cases show the arithmetic works on
  the accelerator, not that the method survives real signals.
- **Convolution is still absent.** 13 and 23 are one-dimensional convolutions run as a dense layer
  over sliding windows prepared on the CPU, which multiplies the input data by the window length.

Facts learned while building them:

| Finding | Detail |
|---|---|
| Peak throughput | Case 1 reads about 270 billion multiply-adds per second, but that job is too small for the DSP's 1 µs timer. The verified large job (case 301, section 21) gives 2.3 trillion per second |
| Small jobs are dominated by the call | Cases 4 and 10 (4 and 16 rows) cost 9-42 µs per row; cases with thousands of rows cost 0.2-0.4 µs |
| Lookalike domains | Comparing whole names flagged three unrelated `.org` sites; comparing only the label before the last dot fixed it |
| Modulation rules | With one threshold of 30, 92 of 100 FSK bursts were called BPSK. Median peak-to-average of x, x^2, x^4: FSK 18, 37, 31; BPSK 12, 118, 95; QPSK 14, 11, 95; 8PSK 13, 13, 9. Rules now: x^2 > 70 BPSK, else x^4 > 60 QPSK, else x^2 > 22 FSK, else 8PSK |
| Matched filter extras | 13 hits beyond the 66 buried sequences; 12 are windows partly overlapping a buried sequence, 1 is a true false alarm |
| Complex numbers | Carried as `[re..., im...]` rows with a real block matrix `[[Wr, -Wi], [Wi, Wr]]`; a 128-point transform is a 256 x 256 real matrix |

## 21. The same ideas on real data (2026-11-25)

One hour of work with real datasets in place of synthetic ones. The raw notes are in
[LAB-NOTES.md](LAB-NOTES.md). Cases are numbered 100, 200 or 300 plus the original case and live in
`rust/src/bin/usecases/real.rs`; run them with `rust/target/release/usecases real`.

### Data

Nothing below is in the repository; `data/` and `ids/cap/` are ignored.

| Data | Source | Size | Used for |
|---|---|---|---|
| NSL-KDD | `github.com/defcom17/NSL_KDD`, `KDDTrain+.txt` and `KDDTest+.txt` | 19.1 MB + 3.4 MB; 125,973 and 22,544 connections | Intrusion scoring and classification |
| Generated and legitimate domains | `github.com/chrmor/DGA_domains_dataset`, `dga_domains_full.csv` | 20.0 MB; 337,500 generated (by family) and 337,398 legitimate | Generated-domain detection |
| Phishing URLs | OpenPhish community feed (`github.com/openphish/public_feed`) | 300 URLs, 286 distinct hosts | Lookalike and host scoring |
| Popular domains | Tranco top 1M (`tranco-list.eu/top-1m.csv.zip`) | 9.7 MB zip | False-alarm measurement |
| Off-air radio recordings | `github.com/merbanan/rtl_433_tests` (regression corpus of rtl_433) | 196 recordings, 57 MB, unsigned 8-bit I/Q | Radio cases |
| MNIST | as in section 13 | 30,000 training and 10,000 test images used | Classification and search |
| Man-page summaries | `man -k .` on this machine | 20,646 entries | Search |
| Own traffic | 25-minute header-only capture on the default-route interface | see table below | Network cases |
| Own latency | `ping -D` to the gateway at 5 per second and to 1.1.1.1 at 2 per second | see table below | Forecasting |
| Own Wi-Fi link | `iw dev <if> station dump` twice a second | see table below | Bitrate prediction |

- **Radio labels.** The corpus has no modulation field. Labels come from the open rtl_433 source: each
  device file's `.modulation = OOK_...` or `FSK_...` and the model names it emits. 196 of 239 picked
  recordings (one per device folder, 64-512 KB) got one unambiguous label: 130 OOK, 66 FSK.
- **Sample rate.** Only the 164 recordings at 250 kS/s are used. 24 of the 66 FSK recordings are at
  1000k/1024k against 8 of 130 OOK, so sample rate would otherwise leak the label.
- **RadioML 2016.10a was not obtained.** Its official URL now returns an HTML page and no open mirror was
  found. Had it been, its pickle file would have needed a restricted unpickler.
- **NSL-KDD encoding** (`scripts/prep_real.py`): 38 numeric columns (log1p on the seven heavy-tailed ones),
  3 protocol flags, 11 connection-state flags and the 20 most common services, 72 columns padded to 96.
  Standardised with the mean and deviation of normal training connections, clipped to +-8.

### Method

No gradient descent anywhere. Three kinds of model:

| Kind | How the weights are obtained | Cases |
|---|---|---|
| Analytic or averaged | Fourier matrix, templates, class means, whitening from a covariance (Jacobi), or the data itself as weights (nearest neighbour) | 103, 104, 108, 114, 120 rule, 121, 204, 208, 209, 215, 223, 229, 323 |
| Linear readout | Ridge regression solved by Cholesky; sparse accumulation where inputs are sparse | 112, 217, 218, 227, 312 |
| Two layers | A random ReLU layer whose outputs are computed **on the accelerator**, then a ridge readout (`random_feature_net`) | 120 readout, 126, 203, 212, 226, 228 |

### Results

Scored against real labels or real held-out data. "not checked" in the last column only means there is no
second implementation to compare with; the arithmetic path is the one verified in sections 20 and case 301.

Final run, 2026-11-25, with the full 25-minute capture (62 thousand packets, 1,805 flow records), 7,000 ping
samples and 2,798 Wi-Fi samples:

```
103 | MNIST nearest centroid       | 5000 real digit images, 10 class means | 80.4% correct with no trained model                  |    0.20 us/row NPU,    0.71 us/row total | not checked
104 | Man-page search              | 6 queries against 20646 real entries | top hits: dir, gunzip, chmod, df, curl, xzegrep      |   27.50 us/row NPU,  231.50 us/row total | not checked
108 | Intrusion scoring, NSL-KDD   | 22544 real test connections, PCA   | attacks flagged 65.1% (DoS 80, probe 60, R2L 32, U2R 63), false alarms 9.8% |    0.05 us/row NPU,    0.25 us/row total | not checked
110 | Lookalikes, live phishing feed | 300 phishing hosts, 20000 popular sites | 7.3% of phishing hosts flagged ("github" ~ github); 3.44% of popular sites |    0.07 us/row NPU,    0.32 us/row total | not checked
112 | Generated domains, real set  | 40000 real names, 732 features     | 86.9% correct; hardest families: nymaim 6%, suppobox 11%, gozi 55% |    0.16 us/row NPU,    0.68 us/row total | not checked
114 | Behaviour baseline, NSL-KDD  | 22544 real test connections, whitened | attacks flagged 78.0% (DoS 89, probe 89, R2L 40, U2R 75), false alarms 4.4% |    0.05 us/row NPU,    0.26 us/row total | not checked
126 | Intrusion classifier, NSL-KDD | 22544 real test connections, 2 layers | 78.8% correct on KDDTest+ (68.5% of attacks caught, 7.5% false alarms) |    0.09 us/row NPU,    0.29 us/row total | not checked
203 | MNIST, random features       | 10000 real test digits, 784-1024-10 | 94.14% correct, fitted in closed form (no gradient descent) |    0.37 us/row NPU,    0.97 us/row total | not checked
212 | Generated domains, 2 layers  | 40000 real names, 732-1024-1       | 85.4% correct; hardest families: nymaim 12%, suppobox 16%, gozi 55% |    0.33 us/row NPU,    0.91 us/row total | not checked
226 | NSL-KDD, 1024 random features | 22544 real test connections        | 77.4% correct on KDDTest+ (65.8% of attacks caught, 7.2% false alarms) |    0.14 us/row NPU,    0.33 us/row total | not checked
227 | Phishing host scoring        | 143 unseen phishing hosts, 10000 popular sites | 69.9% of phishing hosts caught at 1% false alarms    |    0.11 us/row NPU,    0.52 us/row total | not checked
120 | OOK or FSK, real recordings  | 164 off-air recordings, 146253 spectra | rule: 143 of 164 right; fitted readout on unseen half: FSK 17/23, OOK 54/59 |    0.12 us/row NPU,    0.59 us/row total | not checked
121 | Occupancy, real recordings   | 164 off-air recordings, 146253 spectra | signal found in 142; median 16 bursts, 6 kHz occupied |    0.14 us/row NPU,    0.73 us/row total | not checked
204 | Nearest neighbour, MNIST     | 10000 digits against 30000 real images | 96.2% correct; 235 billion multiply-adds in total    |   11.40 us/row NPU,   27.42 us/row total | not checked
228 | NSL-KDD, classifier or distance | 22544 real test connections        | 84.4% correct; attacks 79.1% (DoS 95, probe 90, R2L 28, U2R 79), false alarms 8.6% |    0.07 us/row NPU,    0.27 us/row total | not checked
312 | Generated domains, richer inputs | 60000 real names, 1764 features    | 88.7% correct, 7.3% false alarms; hardest: nymaim 6%, suppobox 17%, gozi 65% |    0.41 us/row NPU,    1.24 us/row total | not checked
217 | Wi-Fi bitrate, real link     | 1120 real samples of this link     | RMS error 151 Mbit/s; repeat-last 168; the rate changed in 130 samples |    0.02 us/row NPU,    0.45 us/row total | not checked
218 | Latency forecast, real pings | 2800 real round-trip times, 16 taps | RMS error 0.452 (log ms); repeat-last 0.616, long-run mean 0.621 |    0.02 us/row NPU,    0.33 us/row total | not checked
301 | Throughput check             | 2048x800 times 800x2048            | 2269 billion multiply-adds/s on the DSP clock, 220.4 billion/s end to end |    0.72 us/row NPU,    1.56 us/row total | verified
208 | Intrusion scoring, own traffic | 1805 real flow records + 6835 scan records | 1.1% false alarms on unseen real traffic, 100.0% of scan flows flagged |    0.02 us/row NPU,    0.24 us/row total | not checked
215 | Clusters, own traffic        | 1805 real flow records, 6 centres  | sizes: 634 (34% udp, 19% dns, 64% https); 326 (15% udp, 0% dns, 100% https); 338 (100% udp, 0% dns, 0% https); 106 (15% udp, 0% dns, 100% https); 124 (0% ud
223 | Matched filter, carrier      | 24560 windows of 24 joined real excerpts | 21 of 24: best other match is in the same device's excerpt (median score 1.00; mostly matches carrier frequency) |    0.12 us/row NPU,    0.54 us/row 
323 | Matched filter, packet pattern | 24320 envelope windows, 24 real devices | 16 of 24: best other match is in the same device's excerpt (median score 0.87, chance is 1 in 24) |    0.06 us/row NPU,    0.29 us/row total | not ch
216 | First packets, own connections | not run                            | too few new TCP connections per server in the capture (needs 6 or more to one endpoint)
229 | NSL-KDD nearest neighbour    | 22544 test against 125973 training connections | 78.8% correct; attacks 68.4% (DoS 86, probe 82, R2L 11, U2R 36), false alarms 7.5% |   34.76 us/row NPU,   72.04 us/row total | not checked
```

Case 203 with 2,048 random features (`NPU_FEATS=2048`) gives 95.56%; 4,096 do not fit in VTCM.

### Findings

| Finding | Detail |
|---|---|
| Accelerator throughput | **2.3 trillion FP16 multiply-adds per second** on the DSP clock: 2048x800 times 800x2048 (3.36 billion) in 1.43 ms, output verified. One 128-byte weight vector serves 2 input channels x 32 outputs x 32 samples = 2,048 multiply-adds, so this is one vector per cycle at about 1.1 GHz |
| End to end | The same job runs at 213 billion per second through the socket (90 billion before the NEON unpacking and the larger output buffer). The copies between client, server and shared buffers are now 10 times the accelerator's time |
| Tables larger than VTCM | Paging works: 30,000 x 784 image table in 15 blocks, 20,646 x 256 text table in 5 blocks. Loading weights per block dominated (1.3 s for the image search, of which 23 ms is DSP run time) until loading became one call per layer: now 0.65 s, and the text search 0.05 s |
| A FastRPC call costs about 0.35 ms | Whatever it carries. 63 per-group load calls cost 20-29 ms; packing the weights was only 4.7 ms, so vectorising the packing alone changed nothing |
| Search without any model | Exact nearest neighbour through request `A`: MNIST, 10,000 test digits against 30,000 images, 96.2% in 1.97 s wall. NSL-KDD, 22,544 test against all 125,973 training connections, 78.8% in 7.8 s wall (0.79 s DSP) |
| Narrow inputs are convert-limited | The NSL-KDD search runs at 346 billion multiply-adds per second on the DSP clock, not 2.3 trillion: with 3 input tiles per output group the fixed work per group (clear, bias load, two converts, two stores) dominates |
| Whitening beats PCA for anomalies | NSL-KDD: 78.0% of attacks at 4.4% false alarms against 65.1% at 9.8% |
| Fusion helps on NSL-KDD | Classifier OR whitened distance: 84.4% correct, up from 78.8%. More random features alone do not help (77.4% with 1,024 on 60,000 rows): KDDTest+ has attack types missing from training. R2L stays at 28-40% |
| A real baseline fixes the false alarms | With this machine's own traffic as baseline (1,805 flow records), 1.1% false alarms on unseen real traffic and every scan flagged; a quiet 20 s live capture gave 0 alarms in 19 records, where the section 18 tool gave 6 in 25 |
| Lookalike matching is the wrong tool for today's phishing | 187 of 300 feed hosts are `*.pages.dev`; only 7.3% resemble a brand, and 3.44% of popular sites do too (brand-owned names such as amazonaws). Scoring the whole hostname catches 69.9% of unseen hosts at 1% false alarms, mostly by learning hosting platforms |
| Word-built generated names pass | 87-89% overall, but nymaim 6%, suppobox 17%, gozi 65%: names made of dictionary words look like legitimate ones to letter statistics, and adding dictionary-word counts moved the total only from 87.8% to 88.7% |
| Closed-form MNIST | 94.14% with 1,024 random features fitted on 30,000 images in 6.3 s, 95.56% with 2,048 in 40 s; nearest centroid 80.4%; exact nearest neighbour over 30,000 images 94.2% |
| Real radio | OOK or FSK by a no-fit rule: 143 of 164 (87.2%; majority class 74%). The strongest bin moving between frames marks FSK. Bin 0 must be ignored (receiver DC offset) |
| Matched filters on real signals | A 1 ms complex template matches carrier frequency, not packets (score 1.00, a warning sign). The envelope pattern over 16 ms finds a device's repeat packet for 16 of 24 devices |
| Real latency is partly predictable | 16-tap predictor on 7,000 gateway pings (fit on the first 60%): RMS 0.452 log-ms against 0.616 for repeating the last value and 0.621 for the long-run mean. On the first 8 minutes alone it was 0.250 against 0.335 |
| Wi-Fi rate barely is | RMS 151 against 168 Mbit/s for repeating the last value on 1,120 test samples; the rate changed in 130 of them. `signal avg` and `tx retries` parse as zero on this driver |
| First-packet signatures need more traffic | At most 5 new TCP connections to one server in 25 minutes: long-lived HTTP/2 and QUIC. Not evaluable |

### Limits

- Fits and tests come from the same sources (same dataset, same day, same machine). None of this shows
  how the detectors age.
- The phishing feed is 300 URLs from one day; the split is 143 fit and 143 test hosts.
- NSL-KDD is a 1999-derived benchmark; it is real traffic of its time, not of today.
- The radio labels are inferred from decoder source, one recording per device type.
- Thresholds: 95th-99th percentile of the fitting data's own scores, stated per case in the source.
- Highest temperature seen during the hour was 52 °C, during CPU-side fitting. No DSP resets.

### Live flow scoring, improved (`ids/live2.sh`)

Replaces the autoencoder of section 18 for live use. Needs the server running and a baseline file.

| Step | Command |
|---|---|
| Baseline, once | capture headers for 20 minutes or more, then `ids.py features` and `ids.py export` to `ids/cap/real-baseline.f32` |
| Score | `ids/live2.sh 30` |

It fits the whitened distance on the baseline, scores the capture on the accelerator, and for the ten oddest
flows prints the score and the single feature furthest from the baseline in standard deviations.
