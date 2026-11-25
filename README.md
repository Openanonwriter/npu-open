# npu-open

Running your own code on the Hexagon NPU of a Snapdragon X laptop under Linux, with open tools only.

No Qualcomm SDK, headers, compiler or runtime are used. The DSP-side code is built with stock
`clang` and `lld`, the Linux side uses the open `hexagonrpcd` daemon (with a small patch), and
everything about the accelerator was learned from open-licensed source, Qualcomm's public
manual, and black-box experiments on the hardware.

Developed on a Microsoft Surface Pro 11 (Snapdragon X Plus, `x1p64100`) running Fedora 45.

## Status

| What | Result |
|---|---|
| Own code on the compute DSP | Works (`add/`) |
| HMX matrix accelerator powered, locked and executing | Works (`hmx/`) |
| 32x32 FP16 matrix multiply, checked against the CPU | 1024 of 1024 cells match (`mm/`) |
| MNIST 784-128-10 on the accelerator | 97.75%, same prediction as the CPU on all 10,000 test images (`npu/`) |
| Speed, MNIST, per image | about 1.0 µs including the call from Linux, about 1.4 µs end to end (numpy on ten cores: 0.74 µs) |
| First application | Network flow anomaly scoring (`ids/`), a demonstrator |
| Convolution | Not started |

This is research code from a single machine. Expect rough edges.

## Read this first

- **You need your own firmware and shell.** The DSP only runs a user process inside Qualcomm's
  "shell" binary, which must be the same build as the DSP firmware. Neither is in this
  repository and neither may be redistributed here. They come from your own device's Windows
  driver package. See [docs/SETUP.md](docs/SETUP.md).
- **It can reset the DSP.** Mistakes such as locking the accelerator before powering it cause a
  DSP watchdog reset. The kernel recovers it within seconds and nothing persistent is changed,
  but it is unsupported use of closed firmware. The audio DSP (`remoteproc0`) must never be
  stopped.
- **Firmware is loaded into RAM only.** `cdsp78.sh` restarts the compute DSP on a firmware pair
  from `fw78/` without touching `/lib/firmware`. It has to be rerun after every boot and after
  every DSP crash.
- **Unofficial.** This project is not affiliated with or endorsed by Qualcomm or Microsoft.

## Quick start

After setup ([docs/SETUP.md](docs/SETUP.md)):

```bash
./cdsp78.sh
```

```bash
CLIENT=$PWD/npu/npu_client ./run.sh 30
```

```bash
grep 'npu:' last-run.log
```

## How it fits together

| Layer | Piece | Source |
|---|---|---|
| Model and data | `npu/train_mnist.py`, `ids/ids.py` | ours (numpy) |
| Linux client | `npu/npu_client.c`, `npu/npu_pack.h` | ours; talks FastRPC through `libhexagonrpc` and direct ioctls |
| Daemon | `hexagonrpcd` | linux-msm/hexagonrpc plus `patches/hexagonrpcd-npu-open.patch` |
| Kernel | `fastrpc`, `qcom_q6v5_pas` | mainline |
| DSP process | `npu/npu_skel.c`, `hmxops.S`, `hvxcopy.c` | ours; built with `clang --target=hexagon` |
| DSP shell and firmware | `fastrpc_shell_unsigned_3`, `qccdsp8380.mbn` | Qualcomm, unmodified, from your device |
| Accelerator | HMX (matrix), HVX (vector) | hardware |

## Repository layout

| Path | Contents |
|---|---|
| `add/` | Smallest example: add two numbers on the DSP; also runs under `qemu-hexagon` |
| `probe/` | Finds which function names the DSP shell exports |
| `hmx/` | Reserve, power and lock the accelerator; execute one instruction |
| `pv/` | Measures the effect of power and clock requests |
| `mm/` | One 32x32 FP16 matrix multiply, uniform and random |
| `nn/` | 256-32-10 classifier on digits rendered from installed fonts |
| `npu/` | Persistent-session runner for fully connected networks, MNIST, benchmark |
| `ids/` | Network flow anomaly scoring on the accelerator |
| `patches/` | Our changes to `hexagonrpcd` |
| `scripts/` | `fetch-deps.sh`: clone, patch and build `hexagonrpcd`; fetch references and MNIST |
| `systemd/` | Template for the one-shot boot test |
| `run.sh`, `cdsp78.sh`, `boot-run.sh` | Run a client; reload DSP firmware; boot test |
| `docs/` | Setup, full technical record, journal |

## Documentation

- [docs/SETUP.md](docs/SETUP.md): reproduce it on your own machine.
- [docs/TECHNICAL.md](docs/TECHNICAL.md): every finding in reference form: firmware and shell
  matching, the loader's requirements, probed function names and signatures, power and clock
  requests, instruction encodings, data layouts, where the chip differs from the manual,
  results, open problems, crashes seen, and sources.
- [docs/JOURNAL.md](docs/JOURNAL.md): what was tried in what order, including the dead ends.

## Rules this project follows

- No licence click-throughs, and no Qualcomm SDK.
- No use of Qualcomm's code and no reverse engineering of Qualcomm's binaries. Their firmware
  and shell are run unmodified, as the device already does.
- Knowledge comes from open-licensed source (LLVM, llama.cpp, qualcomm/fastrpc, hexagonrpc),
  Qualcomm's public HMX manual, and experiments on our own hardware.
- Every finding is written down, including the ones that did not work.

## Licence

MIT, see [LICENSE](LICENSE). One exception: `patches/hexagonrpcd-npu-open.patch` modifies GPL-3 code and
is GPL-3.0-or-later (see [patches/README.md](patches/README.md)). No Qualcomm or Microsoft files are included.

## Sources

| Source | Licence | Used for |
|---|---|---|
| [llvm/llvm-project](https://github.com/llvm/llvm-project) | Apache-2.0 with LLVM exception | HMX instruction encodings |
| [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) Hexagon backend | MIT | Call sequence for power, resource and lock |
| [qualcomm/fastrpc](https://github.com/qualcomm/fastrpc) | BSD-3-Clause | Error-code names, daemon protocol details |
| [linux-msm/hexagonrpc](https://github.com/linux-msm/hexagonrpc) | GPL-3 (library LGPL-3) | Daemon and client library |
| [Qualcomm HMX Programmer's Reference Manual 80-N2040-62](https://docs.qualcomm.com/doc/80-N2040-62/80-N2040-62_REV_AA_Qualcomm_Hexagon_V81_HMX_Programmers_Reference_Manual.pdf) | Public document | Data layouts and register fields |
