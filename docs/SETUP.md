# Setup

How to reproduce this on your own machine. It was developed on one Surface Pro 11 (Snapdragon X
Plus, Fedora 45, kernel 7.3.5 from the sp11 spin). Other Snapdragon X laptops should be similar
but are untested; paths that mention `microsoft/Denali` are specific to this model.

## 1. What you need

| Requirement | How to check |
|---|---|
| The compute DSP boots under Linux | `cat /sys/class/remoteproc/remoteproc1/name` prints `cdsp` and `state` is `running` |
| FastRPC device nodes | `ls /dev/fastrpc-cdsp` |
| Kernel FastRPC driver | `lsmod \| grep fastrpc` (`CONFIG_QCOM_FASTRPC`) |
| Access to the device nodes | you are in group `fastrpc` (udev rule from the `hexagonrpc` package) |
| `sudo` | only for `cdsp78.sh`, which restarts the DSP |

Check which remoteproc is which before running anything. On this machine `remoteproc0` is the
audio DSP and `remoteproc1` the compute DSP. **Never stop the audio DSP**: it also handles
charging and sensors, and stopping it is reported to crash the whole SoC.

## 2. Packages (Fedora)

```bash
sudo dnf install clang lld llvm qemu-user-static-hexagon meson ninja-build gcc git python3-numpy rust cargo
```

Optional, by component:

| Component | Extra packages |
|---|---|
| `nn/train.py` (font-rendered digits) | `python3-pillow`, some fonts |
| `ids/` | `wireshark-cli` (`dumpcap`, `tshark`, `editcap`), `nmap`, `nmap-ncat`, `curl`; membership of group `wireshark` |

Fedora's `clang` includes the Hexagon target; check with `clang -print-targets | grep hexagon`.

## 3. Open-source dependencies

```bash
scripts/fetch-deps.sh --refs --mnist
```

This clones linux-msm/hexagonrpc at commit `da7a374`, applies
`patches/hexagonrpcd-npu-open.patch`, and builds it into `hexagonrpc/build`. `--refs` also
fetches qualcomm/fastrpc (for its error-code header) and LLVM's Hexagon instruction tables;
`--mnist` downloads the four MNIST files (11.6 MB).

## 4. Firmware and shell (you must supply these)

The DSP refuses to create a user process unless the shell binary is the same build as the
running firmware (error `0x8000054f`). You need a matched pair from your own device.

Where they are: the Windows driver store of your device, or the manufacturer's driver package.
On the Surface Pro 11 the package folder is named `qcnspmcdm_ext_cdsp8380.inf_arm64_<hash>` and
contains:

| File in the package | Copy to |
|---|---|
| `qccdsp8380.mbn` | `fw78/qcom/x1e80100/microsoft/Denali/qccdsp8380.mbn` |
| `cdsp_dtbs.elf` | `fw78/qcom/x1e80100/microsoft/Denali/cdsp_dtb.mbn` (renamed) |
| `CDSP/fastrpc_shell_unsigned_3` | `shell/fastrpc_shell_unsigned_3` |

- The destination path under `fw78/` must be the firmware path your device tree asks for
  (`cat /sys/class/remoteproc/remoteproc1/firmware`).
- Check that both are the same build:

```bash
strings fw78/qcom/x1e80100/microsoft/Denali/qccdsp8380.mbn | grep -m1 QC_IMAGE_VERSION
```

```bash
strings shell/fastrpc_shell_unsigned_3 | grep -m1 -o 'CDSP.HT[^ _/]*'
```

- If the firmware already installed in `/lib/firmware` is the same build as your shell, you do
  not need `fw78/` or `cdsp78.sh` at all.
- These files are excluded by `.gitignore`. Do not commit or redistribute them.

## 5. Load the firmware and run

```bash
./cdsp78.sh
```

It should print `cdsp: running` and the size of the firmware it loaded. Then build and run the
smallest example:

```bash
add/build.sh
```

```bash
./run.sh 8
```

```bash
grep 'client:' last-run.log
```

Expected: `client: invoke ret=0, 1234 + 4321 = 5555 (computed on the DSP)`.

## 6. The other components

Each folder has a `build.sh` that builds the DSP library, stages it in `root/dsp/cdsp/`, and
builds the Linux client. Run a client with `CLIENT=$PWD/<folder>/<client> ./run.sh <seconds>` and
read `last-run.log`.

| Component | Prepare | Run |
|---|---|---|
| Symbol probe | edit `probe/names.txt`, `probe/build.sh` | `CLIENT=$PWD/probe/probe_client ./run.sh 60` |
| Accelerator bring-up | `hmx/build.sh` | `STAGE=4 PTYPE=13 FIELDS=2 CLIENT=$PWD/hmx/hmx_client ./run.sh 9` |
| Power request probe | `pv/build.sh` | `REQ="1 0 1 2000 2000" CLIENT=$PWD/pv/pv_client ./run.sh 9` |
| Matrix multiply | `mm/build.sh` | `CLIENT=$PWD/mm/mm2_client ./run.sh 9` |
| Font digits | `python3 nn/train.py` (run inside `nn/`), `nn/build.sh` | `CLIENT=$PWD/nn/nn_client ./run.sh 15` |
| MNIST | `python3 -I npu/train_mnist.py data/mnist npu/mnist-128.bin 128`, `npu/build.sh` | `CLIENT=$PWD/npu/npu_client ./run.sh 30` |
| Flow anomaly scoring | see below | `ids/live.sh 30` |

`npu_client` settings (environment variables): `MODEL` (model file), `NB` (batches of 32 per call,
default 32), `FLAGS` (bit 0 multi-tile multiply, bit 1 merge sign passes on the CPU, bit 2 HVX
copies; default 7), `SHARED` (dma-buf buffers, default 1), `PERSIST` (persistent mapping,
default 1), `MIPS` (performance request, default 2000, 0 to skip), `LATENCY` (sleep-latency hint
in µs, default 100, 0 to skip), `OUT` (file for raw outputs).

Flow anomaly scoring needs a baseline of your own traffic first:

```bash
cd ids && dumpcap -q -s 96 -i lo -i "$(ip -o route show default | awk '{print $5; exit}')" -a duration:1800 -w cap/benign.pcapng
```

```bash
cd ids && python3 -I ids.py features cap/benign.pcapng cap/benign.npz
```

For evaluation, `ids/gen_attack.sh` generates scans against 127.0.0.1 only; capture loopback
while it runs, extract features to `cap/attack.npz`, then `python3 -I ids.py train cap/benign.npz
cap/attack.npz`. Put `-s 96` before the `-i` options so it applies to every interface.

## 7. The NPU server and the demonstrations

The server is Rust with no external crates, so the build needs no network:

```bash
(cd rust && cargo build --release)
```

It needs the DSP library from `npu/build.sh` in `root/dsp/cdsp/`. Then:

```bash
./serve.sh start
```

```bash
rust/target/release/usecases all
```

```bash
./serve.sh stop
```

- `usecases` takes `math`, `cyber`, `net`, `radio`, `all`, or case numbers. `NPU_DEBUG=1` prints
  extra diagnostics.
- Cases 8, 14, 15 and 19 need flow records: after capturing (section 6),
  `python3 -I ids/ids.py export ids/cap/benign.npz ids/cap/benign.f32`, and the same for `attack`.
- Case 12 needs `/usr/share/dict/words` (package `words`).
- The socket is `npu.sock` in the project folder (`NPU_SOCK` overrides it); the log is `server.log`.
- To use the accelerator from your own Rust program, depend on the `npu` crate by path and use
  `npu::proto::Client` (`define`, `linear`, `run`).

## 8. Real data for the demonstrations

`usecases real` needs data that is not in the repository. Sources and sizes are in TECHNICAL.md section 21.

```bash
mkdir -p data/nslkdd data/dga data/phish data/tranco data/rtl433 data/live
```

| Data | Get it | Convert |
|---|---|---|
| NSL-KDD | download `KDDTrain+.txt`, `KDDTest+.txt` into `data/nslkdd/` | `python3 -I scripts/prep_real.py nslkdd data/nslkdd` |
| MNIST | `scripts/fetch-deps.sh --mnist` | `python3 -I scripts/prep_real.py mnist data/mnist data/mnist/test.f32` and `... data/mnist/train.f32 train 30000` |
| Domains | `dga_domains_full.csv` into `data/dga/` | read directly |
| Phishing feed | `feed.txt` into `data/phish/` | read directly |
| Tranco | `top-1m.csv.zip` into `data/tranco/`, then `unzip` | read directly |
| Man pages | `man -k . \| sort -u > data/live/whatis.txt` | read directly |
| Radio | recordings from rtl_433_tests into `data/rtl433/` with an `index.txt` of lines `OOK\|FSK 250k file.cu8` | read directly |
| Ping | `ping -i 0.2 -D <gateway> > data/live/ping-gateway.txt` | `python3 -I scripts/prep_real.py ping data/live/ping-gateway.txt data/live/ping-gateway.f32` |
| Wi-Fi | loop of `date +%s.%N; iw dev <if> station dump` into `data/live/wifi-station.txt` | `python3 -I scripts/prep_real.py wifi data/live/wifi-station.txt data/live/wifi.f32` |
| Radio identification | `scripts/fetch-radioid.py` (389 MB from rtl_433_tests into `data/rtl433id/`) | read directly; `usecases radioid` |
| Own traffic | `dumpcap -q -s 96 -i <if> -a duration:1500 -w ids/cap/real-baseline.pcapng` | `python3 -I ids/ids.py features ...` then `ids.py export ... ids/cap/real-baseline.f32` |

- Treat downloads as untrusted: keep each in its own folder and read them with `python3 -I`.
- `cargo test --release` in `rust/` runs the unit tests; they need no DSP.
- Radio identification, NPU against CPU: `usecases radioid` (cases 130-133). The in-process run (case 134) needs
  the server stopped: `./serve.sh stop`, `NPU_DIRECT=1 CLIENT=$PWD/rust/target/release/usecases ./run.sh 25`, read
  `last-run.log`, `./serve.sh start`.
- Live scoring with the real baseline: `./serve.sh start`, then `ids/live2.sh 30`.

## 9. When something fails

| Symptom | Cause |
|---|---|
| `INIT_CREATE ... 0x8000054f` in `last-run.log` | Shell and firmware builds differ, or the DSP went back to the installed firmware after a reboot or crash: run `./cdsp78.sh` |
| `dlerr=0x80000406` opening a library | The library's ELF layout is wrong (needs two LOAD segments, the second page-aligned) or it imports a symbol the shell does not export |
| `dlerr=0x80000442` | The library has no `<name>_skel_invoke` |
| `cdsp78.sh` reports the DSP not running and the kernel log shows `error -22 initializing firmware` | The firmware in `fw78/` is not signed for this device |
| DSP watchdog reset a few seconds after a run | The accelerator was locked without the power-on request |
| A call with large buffers fails with `0x80000468` | Stock `hexagonrpcd` without our patch, or `HRPC_VADSP32=1` set |
| `Could not fetch next FastRPC message` at the end of a run | Normal: the DSP process exited |
| `serve.sh start` says the server did not come up | Usually the shell and firmware no longer match after a reboot or crash: run `./cdsp78.sh`, then start again |
| `usecases`: cannot reach the NPU server | Run `./serve.sh start` |

Error-code names are in `qcom-fastrpc/inc/AEEStdErr.h` (DSP errors are `0x80000400` plus the
code).
