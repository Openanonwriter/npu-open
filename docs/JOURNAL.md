# Journal

What was tried, in order, including what failed and why. Reference details for each finding are
in [TECHNICAL.md](TECHNICAL.md). All of this happened on 2026-11-24 on one Surface Pro 11.

## 1. Is the NPU usable at all?

- The compute DSP was already booted by the kernel, the `fastrpc` driver was loaded and
  `/dev/fastrpc-cdsp` existed. `hexagonrpcd` was installed but only serving the sensor DSP.
- Missing: any runtime able to run a model. The only supported route is Qualcomm's QAIRT SDK,
  which needs a licence acceptance.
- Decision by the owner: no licence click-throughs, no Qualcomm code, no reverse engineering of
  Qualcomm's binaries. Build a clean-room path.

## 2. Step 1: our own code on the DSP

| Attempt | Outcome |
|---|---|
| Build `add_skel.c` with Fedora `clang`/`lld`, test under `qemu-hexagon` | Returned 42 for 2 + 40 |
| Patch `hexagonrpcd` to create an unsigned process (`-u`); use the public build-00069 shell with the installed build-00046 firmware | `INIT_CREATE` failed. The daemon printed "No such file or directory": a stale `errno`. The real result was the ioctl's return value, `0x8000054f` |
| Same with the build-00082 and build-00094 shells | Same error |
| Load the generic build-00094 firmware from Fedora into RAM to match a public shell | Rejected by the device: `error -22 initializing firmware`. Restored the installed firmware |
| Owner pointed to a Windows driver backup on disk holding firmware build 00078 and its shell | Loaded 00078 into RAM; process creation succeeded |
| Load `libadd_skel.so` | `0x80000406` (unable to load). Three layout variants (4K pages, Hexagon version tag, classic symbolic tag) all failed |
| Control: load an unmodified Qualcomm example library (with permission) | It loaded (then failed only on a missing symbol). The daemon's log showed its second segment starting on a page boundary in the file |
| Relink with `-z separate-loadable-segments` | Loaded. 1234 + 4321 = 5555 on the DSP |

Lessons: the shell must match the firmware build; the loader wants two segments with the second
page-aligned; DSP errors come back as positive ioctl return values.

Mistake recorded: I read the ELF header and dynamic tags of a Qualcomm example library before
asking. It was metadata, not code, but closer to the rules than it should have been.

## 3. Step 2: the accelerator

| Attempt | Outcome |
|---|---|
| Find instruction encodings | Already public in LLVM's `HexagonDepInstrInfo.td` (the `M8_*` definitions) |
| Find declarations for power and lock calls | None under an open licence. llama.cpp shows how they are called, not their layouts |
| Probe which names the shell exports (one tiny library per name) | `HAP_compute_res_*` do not exist; `compute_resource_*` do. `HAP_power_set` exists |
| Closing a probe library after loading it | Ended the DSP process: closing the last handle exits it. Kept handles open |
| Reserve and lock HMX without powering it | Calls returned 0, then the DSP watchdog reset it |
| Call `HAP_power_set` with each type 0-20, no lock, a guessed layout | Types 1-13 exist. Seven return 0 |
| Lock after types 12, 13, 10 | 12 and 10: reset. 13: no reset. Type 13 powers HMX |
| Execute `mxclracc.hf` with power and lock | Ran |
| Read Qualcomm's public HMX manual for layouts | Activation, weight, bias and convert fields; data must be in VTCM |
| Probe VTCM function names | All candidates exist |
| Multiply uniform tiles (1.0 x 0.5 over 32 channels) | 16.0 in all 1024 cells; two more value sets also correct |
| Multiply random signed matrices | 491 of 1024 matched: every negative result came back as 0 |
| Try all eight shaping codes, two convert bits, negative scale, output bias | Output is `max(0, shape(acc)*scale + bias)` in every case |
| Two convert passes using bias sets 0 and 1 | Both passes used the same set: the set index is ignored |
| Two passes, reloading the bias between them (scale +1 then -1), merge with integer ops | 1024 of 1024 cells match for three seeds |

## 4. Step 3: a classifier

- No dataset on disk and no download approved, so digits were rendered from 190 installed fonts
  (152 for training, 38 held out). A 256-32-10 network trained with numpy reached 96.4%.
- Hidden layer: one convert pass, because the chip's zero clamp is a ReLU. Output layer: the
  two-pass signed method. A layer's output tile has the layout of an input tile, so it feeds the
  next layer unchanged.
- Result: 96.39% on accelerator and CPU, the same prediction on all 1024 test images.
- Boot test: a one-shot systemd service ran the same test 10 seconds after a cold boot and
  passed. The service disables itself first so a crash cannot repeat at the next boot.

## 5. Step 4: persistent session, wider layers, MNIST

- `npu_skel.c`: acquire and power once, load weights once, run the whole network per call, any
  multiple of 32 outputs per layer.
- MNIST 784-128-10 trained with numpy: 97.75%. On the accelerator: 97.75%, identical to the CPU
  on all 10,000 images, first try.
- More than two batches per call hung. The DSP was asking the daemon to grow its heap and then
  calling an unmap method the daemon lacked. Adding the method turned the hang into
  `0x80000468`. The real cause, found later, was the daemon truncating the mapped address to 32
  bits.
- Power request field isolated: the flag at offset 8 powers HMX; the one at offset 4 does
  nothing (two deliberate DSP resets to prove it).

## 6. Speed

| Change | DSP time per image | Total including the call |
|---|---|---|
| Starting point | 9.3 µs | about 24 µs |
| Timing breakdown showed the multiply at about 0.04 µs and my sign-merge loop at 6.6 µs. Moved the merge to the CPU | 3.4 µs | about 18 µs |
| Found that any accepted power request drops the DSP to a low clock (a fixed loop: 8.5 ms to 22.7 ms). A MIPS request of 1000 or more brings it to 7.1 ms | 1.1 µs | about 11 µs |
| Fixed the daemon's address truncation, allowing 32 batches per call; added a sleep-latency hint | 2.5 µs | about 7 µs |
| Shared dma-buf buffers passed by fd, persistent mapping | 2.7 µs | 3.2 µs |
| HVX vector copies in place of `memcpy` (first attempt silently fell back because the output buffer was 32 bytes off alignment) | 0.39 µs | about 1.0 µs |
| NEON conversion of inputs to FP16 tiles on the CPU (3.6 to 0.43 µs) | 0.37 µs | about 1.0 µs, 1.4 µs end to end |

- The owner asked whether a clock request could harm the DSP. Answer given: a correct request
  only selects among the firmware's own operating points, but guessing an unknown layout is the
  risky part. The test was done in order of increasing risk: first only requests already sent
  earlier, then the standard MIPS performance request with small values. No resets;
  temperatures stayed at 37-38 °C.
- One multiply instruction can take up to 32 tiles ("deep" activation). Results are identical
  and it is slightly faster.

## 7. First application: flow anomaly scoring

- Chosen from a list of 25 ideas in math, cyber defence, networking and radio. Twenty of them
  need only dense layers; all of them need fast input conversion, which is why that was done
  first.
- Pipeline: header-only capture, 32 features per flow per 5-second window, autoencoder
  32-32-8-32-32 trained on baseline flows, both halves run on the accelerator.
- Result: all 6,835 scan and burst records flagged; accelerator and CPU agree on all 6,953
  decisions; 0.19 µs per record.
- Honest limits: the baseline was 150 seconds, so 6 of 25 ordinary flows in a quiet capture also
  alarmed; the attacks were local port scans, the easy case.
- Mistake recorded: the first baseline capture kept full loopback packets because `-s 96` was
  placed after the interface options. Trimmed afterwards; scripts fixed.

## 8. Preparing for git

- Scripts and clients made location-independent (`NPU_OPEN`, `NPU_SHELL`).
- `.gitignore` keeps Qualcomm and Microsoft binaries, third-party checkouts, datasets, trained
  models and all captured traffic out of the repository.
- `scripts/fetch-deps.sh` rebuilds the open-source dependencies from scratch.
- Licence chosen by the owner: MIT for our own code. The `hexagonrpcd` patch stays GPL-3.0-or-later
  because it modifies GPL-3 code.

## 9. Rust server and 25 demonstrations (2026-11-25)

- The owner asked to get every use case from the 25-idea list working without training, on the
  accelerator, and said the CPU comparison lines were not wanted: "same score, just time
  difference". The comparison is kept as a silent check and no longer printed.
- I had started designing a C server; the owner asked for Rust and for the preference to be
  written down. `CLAUDE.md` now holds the rules: prefer Rust, document meticulously, the
  clean-room rules, and the DSP safety rules.
- `rust/`: FastRPC ioctls called directly, tile conversion with inline NEON, a persistent session,
  a Unix-socket server and client. Standard library only. It compiled and served on the first
  build; one borrow error in the demonstration code.
- New fact: asking for 4 MB of VTCM succeeds (the C client only asked for 1 MB).
- 24 of the 25 demonstrations run. Fixes along the way:

| Case | Problem | Fix |
|---|---|---|
| 10 lookalike domains | 11 flagged instead of 8: unrelated `.org` names resembled each other | Compare only the label before the last dot |
| 20 modulation | 306 of 400: FSK was called BPSK | Looked at the median line strengths per type and set three rules; now 398 of 400 |
| 23 matched filters | 13 extra hits | Not a bug: 12 are partial overlaps with a real sequence. Reported as such |
| 22 fingerprinting | Cannot be done without real recordings | Printed as not run |

- No DSP resets. Highest temperature 40 °C.

## DSP resets, in total

| When | Cause |
|---|---|
| Step 1 | Exit after a failed process creation with a mismatched shell |
| Step 2 | HMX locked without power |
| Step 2 | Power request types 12 and 10 followed by a lock (two resets) |
| Step 4 | Power-field test with the flag at offset 4 only, and with neither field (two resets, deliberate) |

Six in total. Each recovered automatically within seconds. No kernel errors, no whole-machine
hang, and the audio DSP was never affected.
