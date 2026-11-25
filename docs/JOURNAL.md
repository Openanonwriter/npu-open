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

## 10. One hour on real data (2026-11-25)

The owner set a one-hour limit: test everything on real data, improve it any way possible, take notes,
stop at the end and record the findings. Timestamped notes: [LAB-NOTES.md](LAB-NOTES.md). Reference
form: TECHNICAL.md section 21.

| When | What | Outcome |
|---|---|---|
| 0 min | Started a 25-minute header-only capture, pings and Wi-Fi sampling in the background; downloaded NSL-KDD, a labelled generated-domain set, a phishing feed and the Tranco list | All arrived. RadioML's official link is dead |
| 2 min | First real-data run of seven cases | Man-page search and NSL-KDD scoring good; phishing lookalikes poor (7.3% caught, 3.44% false alarms) |
| 4 min | Closed-form two-layer models with the feature layer on the accelerator | MNIST 94.14%; phishing hosts 69.9% at 1% false alarms; no gain for generated domains or NSL-KDD |
| 6 min | Found real radio data: rtl_433's test corpus; derived OOK/FSK labels from the decoder source | 164 usable recordings; rule 87.2%, readout 86.6% |
| 7 min | Nearest neighbour over 30,000 images by paging; NSL-KDD fusion; richer domain inputs | 94.2%; 84.4% (from 78.8%); 87.8% |
| 9 min | Throughput check after an implausible figure | Confirmed: 2.3 trillion multiply-adds per second on the DSP clock, verified output |
| 10 min | NEON unpacking and an 8 MB output buffer in the server | End to end 90 to 213 billion per second; all 23 checkable original cases still verify |
| 11 min | Own-traffic baseline from the capture so far | 1.1% false alarms on unseen real traffic (was about 24%) |
| 12 min | `ids/live2.sh` on the improved method | Scan of the gateway flagged with the reason named; quiet capture 0 alarms |
| 13 min | Unit tests for the Rust conversion code | 4 pass |
| 13 min | Matched filter on real recordings | Carrier form misleading (score 1.00); envelope-pattern form finds repeats for 16 of 24 devices |
| 14 min | First-packet signatures on own traffic | Not evaluable: at most 5 new connections per server |
| 16 min | Wider MNIST model; larger VTCM request | 2,048 features 95.56%; 4,096 do not fit; 16 MB of VTCM refused |
| 25 min | Capture finished; full real suite rerun | 23 results in TECHNICAL.md section 21 |
| 27 min | Vectorised weight packing to speed up paged searches | No change: timing showed 63 load calls at 0.35 ms each were the cost, not packing |
| 28 min | DSP method 4: load a whole layer in one call through the shared buffer | Model load 25-35 ms to 5.6 ms; image search 1.3 s to 0.65 s; everything still verifies |
| 29 min | Server request `A`: best match per row only | Nearest neighbour for all 10,000 MNIST digits: 96.2% in 1.97 s |
| 30 min | NSL-KDD nearest neighbour over all 125,973 training rows, no fitted model | 78.8%; showed that narrow-input layers are limited by convert, not multiply |
| 32 min | Generated domains by nearest neighbour | 69.1% overall but suppobox 78% (readout: 17%) |
| 33 min | Combined readout and close neighbour | First try used a weak readout and looked bad; redone with the strong one: 89.3%, suppobox 69%, false alarms 9.7% |
| 34 min | Soak test, 5 minutes of continuous runs | 28 passes, 672 of 672 checks verified, 49 °C steady, no resets |
| 39 min | Three-way vote on NSL-KDD | Any one 85.0%; all three 73.3% with 1.7% false alarms |

Things that did not work, kept on record:

- Lookalike matching against a live phishing feed (wrong tool for free-hosting subdomains).
- A nonlinear layer for generated domains (85.4%, below the linear 86.9%); word-built families stay at
  6-17% however the inputs are enriched.
- More capacity on NSL-KDD (77.4% with 1,024 features on 60,000 rows, below 78.8% with 512 on 24,000).
- The first matched-filter design, which matched carrier frequency instead of packets.
- RadioML could not be downloaded.

A light scan was run against this network's own gateway (101 TCP ports, connect scan) to test the live
tool. No DSP resets during the hour. Highest temperature 53 °C, during CPU-side fitting.

## 11. Radio identification against the CPU (2026-11-25)

Asked for 30 minutes of radio identification compared with the CPU, on free samples from the internet. About 14
minutes of experiments, the rest documentation. Full numbers: TECHNICAL.md section 22.

In order:

1. Chose rtl_433_tests again as the source (RadioML is still unavailable). First selection rule was too strict
   (only files with `_250k` in the name, at most 1.5 MB): 8 devices. Relaxed it: 46 devices, 776 recordings.
2. Wrote `radioid.rs` with one `Engine` trait so the same pipeline runs on the NPU and on the CPU. Features:
   the 100 spectrum numbers from the earlier hour plus a new timing spectrum of bursts (envelope and frequency).
3. First run: 96.9% / 97.7% on 46 devices, same on both. The NPU was slower (matrix work 2.94 s against
   2.26 s). Mistake found: the timed section included the closed-form fit, which runs on the CPU in both
   paths. Fixed to time only the matrix calls: 1.71 s against 1.06 s. Still slower.
4. Suspected the 97% was inflated by same-session recordings. Built a second set split by recording folder:
   79.4%. Three protocols failed completely. The file names show at least one of them (ambient_weather: 915 MHz
   against 433.92 MHz) is two different products, so my assumption that numbered folders are the same protocol
   was wrong for some families. Added leave-one-session-out: 82.3%, or 95.7% without the two mismatched classes.
5. Measured the spectrum stage alone at 128, 256 and 512 points. The NPU through the server only overtakes ten
   threads at 512 points. Added a plain FFT on one core for honesty: it beat every matrix path at that point.
6. Tried to read battery draw for an energy comparison: on mains power and full, nothing to read. Used
   processor time from `/proc` as a stand-in: the NPU path uses 9 to 25 times less.
7. Broke down the NPU time: the accelerator needs about 6% of it; about 60% is the server socket.
8. Ran the session in-process (server stopped): 0.47-0.52 s, faster than ten threads at every size.
9. Call count was now the limit (539 calls of about 0.35 ms). Queued 16 recordings per request: little gain
   at 128 points because of a cap of 32 batches per call in `session.rs`. Raised the cap to 256 (client side
   only). Result 0.21-0.30 s: faster than the one-core FFT.
10. Re-ran cases 1-3 and the unit tests after the cap change: verified, 4 of 4 tests pass.

Not done: shared memory between client and server (the obvious next step), a multi-threaded FFT reference,
decoding the mismatched sessions to confirm they are different protocols, DSP-side energy.

## 12. Operation sweep (2026-11-25)

Asked: what operations can it do, can a custom C++ or Rust kernel be written, can threads be dispatched to it.
Ten minutes. Numbers: TECHNICAL.md section 23.

1. Probed 32 runtime names (threads, mutexes, memory, C maths, compiler support): all exported by the shell.
2. Wrote `ops/` in freestanding C++ with 13 operations compiled twice (scalar and HVX). First build had no
   vector instructions at all: `-mhvx` does not enable auto-vectorisation; `-fvectorize -mllvm -hexagon-autohvx`
   does (287 HVX instructions). My first check for vector instructions also used a wrong search pattern.
3. Ran it: 13 of 13 correct in both builds, HVX 29-132 times faster than the scalar cores on integer work;
   float and table lookup are not vectorised.
4. Threads through `pthread_create`: up to 8 created, all results correct; time flat up to 4.
5. C library: qsort, sqrtf, sinf, expf, 64-bit divide, double multiply all correct.
6. Matrix accelerator sweep (case 140): 18 of 18 operation kinds correct.
7. Rust for the DSP: the compiler knows the target but has no core library for it here; not attempted.

Not done: HVX float, HVX or HMX inside spawned threads, the `qurt_*` thread calls (signatures unknown), a clock
request in the `ops` library, Rust on the DSP.

## DSP resets, in total

| When | Cause |
|---|---|
| Step 1 | Exit after a failed process creation with a mismatched shell |
| Step 2 | HMX locked without power |
| Step 2 | Power request types 12 and 10 followed by a lock (two resets) |
| Step 4 | Power-field test with the flag at offset 4 only, and with neither field (two resets, deliberate) |

Six in total. Each recovered automatically within seconds. No kernel errors, no whole-machine
hang, and the audio DSP was never affected.
