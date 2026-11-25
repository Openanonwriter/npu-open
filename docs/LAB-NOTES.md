# Lab notes: one hour on real data (2026-11-25)

Timestamped as work proceeds. Folded into TECHNICAL.md and JOURNAL.md at the end of the hour.

- 00:29 Start. Goal: rerun the demonstrations on real data and improve them.
- 00:29 Background: 25-minute header-only capture on the default-route interface; ping to the gateway (5 per second) and to 1.1.1.1 (2 per second); Wi-Fi link statistics twice a second.
- 00:30 Downloads arrived: NSL-KDD (train 19.1 MB, test 3.4 MB, github.com/defcom17/NSL_KDD), DGA dataset (20.0 MB, 675k rows, github.com/chrmor/DGA_domains_dataset), OpenPhish community feed (300 URLs), Tranco top 1M (9.7 MB zip). RadioML 2016.10a URL on opendata.deepsig.io now returns an HTML page, not the dataset.
- 00:32 First real-data run (rust usecases "real", cases 103-126). No CPU comparison is printed.
  - 103 MNIST nearest centroid (class means of 5,000 test images as weights): 80.4% on the other 5,000.
  - 104 man-page search: 20,646 entries paged through VTCM in blocks of 5,120; top hits for six queries: dir, gunzip, chmod, df, curl, xzegrep. All sensible. 286 us per query row in total: reloading weights per block dominates.
  - 108 NSL-KDD PCA anomaly (fit on 20,000 normal training rows, 10 components, threshold = 95th percentile of the fit rows): 65.1% of test attacks flagged (DoS 80, probe 60, R2L 32, U2R 63), 9.8% false alarms.
  - 114 NSL-KDD whitening (Mahalanobis): 78.0% flagged (DoS 89, probe 89, R2L 40, U2R 75), 4.4% false alarms. Better than PCA on both counts.
  - 126 NSL-KDD closed-form classifier (512 random ReLU features computed on the NPU, ridge readout, fit on 24,000 rows): 78.8% on KDDTest+ (68.5% of attacks, 7.5% false alarms).
  - 112 DGA real set (ridge over 729 letter-pair frequencies + length + digit share + constant; 60,000 fit, 40,000 test): 86.9%. Hardest families: nymaim 6%, suppobox 11%, gozi 55% (suppobox and gozi build names from dictionary words).
  - 110 lookalikes on the OpenPhish feed: only 7.3% of 300 phishing hosts flagged, and 3.44% of the top 20,000 popular sites. Poor: most phishing hosts do not imitate a brand in the hostname, and brand-owned names (amazonaws, googleapis) count as lookalikes.
- 00:34 Phishing feed structure: 187 of 300 hosts are *.pages.dev, 14 vercel.app, 11 weebly.com, 7 github.io. Free hosting with arbitrary subdomains, so "resembles a brand" is the wrong test for most of them.
- 00:34 RadioML 2016.10a: the official URL is dead and a web search found no open mirror (IEEE DataPort entry has no files). Not obtained.
- 00:34 Improved models, all fitted in closed form with the feature layer computed on the NPU (rust real.rs random_feature_net: random ReLU layer, ridge readout by Cholesky):
  - 203 MNIST 784-1024-10, fit on 30,000 training images: 94.14% on the 10,000 test images (nearest centroid was 80.4%). 6.3 s wall including the fit.
  - 227 phishing host scoring (512 hashed trigrams of the whole hostname, ridge readout, fit on half the feed and half of the top 20,000 sites, classes balanced by weight): 69.9% of 143 unseen phishing hosts caught at 1% false alarms. It mostly learns hosting-platform suffixes.
  - 212 DGA 732-1024-1: 85.4%, slightly worse than the linear readout (86.9%). nymaim 12%, suppobox 16%, gozi 55%.
  - 226 NSL-KDD 96-1024-1 fit on 60,000 rows: 77.4% (512 features on 24,000 rows gave 78.8%). More capacity does not help: the test set contains attack types absent from training.
- 00:35 Real radio data found: github.com/merbanan/rtl_433_tests, 3,724 off-air IQ recordings (unsigned 8-bit I/Q, mostly 250 kS/s, 433/868/315 MHz devices). Picked one recording per device folder between 64 KB and 512 KB (239). The expected-output JSON has no modulation field, so labels were derived from the rtl_433 source (GPL, read for facts only): each device file's `.modulation = OOK_... | FSK_...` and the model strings it emits. 196 recordings got one unambiguous label: 130 OOK, 66 FSK.
- 00:36 Real radio on the NPU (164 recordings at 250 kS/s; the 32 at 1000k/1024k are left out so sample rate cannot leak the label; 146,253 frames of 128 samples transformed on the accelerator, 0.15 us per spectrum):
  - 120 OOK or FSK. Rule with no fitting ("in more than 30% of signal frames the strongest bin is 2 or more bins away from its usual place" means FSK): 143 of 164 right (87.2%). Closed-form readout over 100 spectral features, fitted on the even-numbered recordings, tested on the odd: FSK 17/23, OOK 54/59 (86.6%). Majority-class baseline is 74%.
  - 121 occupancy: a transmission (4 or more frames above six times the median frame energy) found in 142 of 164; median 16 bursts and 6 kHz occupied. Bin 0 is excluded: it holds the receiver's DC offset.
- 00:37 More real-data results:
  - 204 exact nearest neighbour, 2,000 MNIST test digits against 30,000 training images paged through VTCM in 15 blocks of 2,000 (each block is 1,575 weight tiles, 3.1 MB): 94.2% correct. 47 billion multiply-adds; 23 ms of DSP time in the run calls and 1.8 s wall including loading 15 blocks of weights. The DSP figure implies about 2 trillion multiply-adds per second, seven times the earlier 270 billion measurement; not cross-checked, treat as unverified.
  - 228 NSL-KDD, alarm if the closed-form classifier OR the whitened distance (97th percentile of normal training rows) says so: 84.4% correct on KDDTest+, 79.1% of attacks (DoS 95, probe 90, R2L 28, U2R 79), 8.6% false alarms. Up from 78.8% for the classifier alone.
  - 312 DGA with 1,759 inputs (letter pairs, 1,024 hashed triples, six shape numbers), sparse ridge fit on 120,000 names, 60,000 tested: 87.8%, 7.3% false alarms. nymaim 7%, suppobox 17%, gozi 61%: word-built names still pass as legitimate. Character statistics cannot fix that; it needs word-level inputs.
  - Highest temperature rose to 52 C during the closed-form fits (CPU work in Rust), from 40 C.
- 00:38 Real networking data from this machine (first 8 minutes of collection):
  - 218 gateway round-trip time, 5 samples per second, 2,420 samples (mean 5.8 ms, spikes to 176 ms). 16-tap linear predictor of log(1 + ms), fitted on the first 60%: RMS error 0.250 on the rest, against 0.335 for repeating the last value and 0.272 for the long-run mean.
  - 217 Wi-Fi link, 2 samples per second, 964 samples (signal -43 dBm +- 1.4, transmit rate mean 2,345 Mbit/s). Predicting transmit rate from signal, receive rate, traffic and the previous sample: RMS 169 Mbit/s against 190 for repeating the last value. The rate changed in only 41 of 386 test samples. "signal avg" and "tx retries" parsed as zero from `iw station dump` on this driver, so they are not used.
- 00:39 301 throughput check, 2048x800 times 800x2048 (3.36 billion multiply-adds), output verified against the CPU on a sample of rows: 1.43 ms on the DSP clock, twice in a row = 2.3 trillion multiply-adds per second. This confirms the nearest-neighbour figure. It matches the architecture in the public manual: one 128-byte weight vector (2 input channels x 32 outputs) applied to 32 samples is 2,048 multiply-adds, so one vector per cycle at about 1.1 GHz gives 2.3e12/s. The earlier 270 billion figure came from a job too small to fill the timer's 1 us resolution per interval.
  - End to end the same job runs at 90 billion/s: 25 times slower than the accelerator. The time goes to the socket copies (6.5 MB in, 16.8 MB out), scalar FP16-to-float unpacking in the server, and 10 calls because the 2 MB output buffer holds only 7 batches.
- 00:39 Server improvement: result unpacking rewritten with NEON (uzp1/uzp2 to split the two interleaved samples, fcvtl to float32, and a bsl select to merge the two sign passes), and the shared output buffer raised from 2 MB to 8 MB so more batches fit per call. Case 301 end to end: 90 -> 213 billion multiply-adds/s. Cases 1, 7 and 301 still verify against the CPU; 104 and 126 give the same answers as before.
- 00:39 312 with five dictionary-word inputs added (greedy longest-word split against the system word list): 88.7% (from 87.8%), false alarms 7.3%. nymaim 6%, suppobox 17%, gozi 65%. Legitimate names are also made of words, so a linear readout cannot tell them from word-built generated names. This is the ceiling for character and word statistics here.
- 00:40 Own-traffic baseline (first 11 minutes of the header-only capture, 905 flow records): 208 whitened-distance scoring fitted on the first 70% in time order gives 1.1% false alarms on the unseen 30% at the 99th-percentile threshold, with 100% of the 6,835 local scan records flagged. The 150-second baseline of section 18 gave 6 alarms in 25 ordinary flows. 215 clustering of the same traffic separates plain UDP, QUIC (UDP to 443), HTTPS over TCP and DNS-heavy groups without labels.
- 00:41 Live tool on the improved method: ids/live2.sh (capture, features, export, rust usecases case 209). Test 1: 20 s of real traffic during a 101-port connect scan of this network's own gateway: 118 flow records, 103 alarms; the scan connections score 28-35 against a threshold of 12.0, and the report names the deviating feature ("ports touched by this source" at +8 deviations). Test 2, quiet, in the line below.
  - Quiet test: 19 flow records, 0 alarms (highest score 9.3 against 12.0). The section 18 tool raised 6 alarms on 25 quiet records.
- 00:43 Unit tests added to rust/src/tiles.rs (cargo test): FP16 conversions and round trips, pack_batch against the cell formula for six padded shapes, unpack_row including the sign-pass merge, weight tile roles. 4 pass.
- 00:43 223 matched filter on real recordings, first form (256-sample complex template, about 1 ms): for 21 of 24 devices the best match away from the template is in the same device's excerpt, but the median normalised score is 1.00. That is too perfect: a 1 ms piece of an on-off-keyed burst is mostly bare carrier, so this matches the carrier frequency of each recording, not the packet. Recorded as a weak result; an envelope-pattern version follows.
- 00:43 323 matched filter on the on/off pattern (magnitude averaged over 16 samples, 256-point template spanning about 16 ms, zero-mean unit-length windows): for 16 of 24 real devices the best match away from the template is another packet in the same device's excerpt (median score 0.87; chance is 1 in 24). This is the meaningful form: it finds a device's repeat transmissions among 23 other devices' signals.
- 00:44 216 first-packet signatures on own traffic: not evaluable. In 14 minutes this machine opened at most 5 new TCP connections to any one server (5, 4, 4, 2, 2, then ones): modern traffic reuses long-lived HTTP/2 connections and QUIC. The extractor (ids.py firstpackets) and the case exist; they need a busier or longer capture.
- 00:45 203 with wider models: 2,048 random features give 95.56% on MNIST (1,024 gave 94.14%), 40 s wall for the closed-form fit on 30,000 images. 4,096 features do not fit: 784 -> 4,096 needs 25 x 128 = 3,200 weight tiles (6.4 MB) and VTCM is 4 MB; the server reports it cleanly. Trying a 16 MB request next.
- 00:46 VTCM: a 16 MB request is refused and the server falls back to 4 MB. 4 MB is the most this DSP has granted. Reverted to asking for 4 MB first.
- 00:55 Capture finished: 25 minutes, 62 thousand packets, 1,805 flow records. Full real suite rerun; final table in TECHNICAL.md section 21. With all the data: 218 ping forecast RMS 0.452 against 0.616 (repeat last) and 0.621 (mean) on 2,800 test samples; 217 Wi-Fi 151 against 168 Mbit/s; 208 own-traffic scoring 1.1% false alarms, 100% of scans; 216 still not evaluable (at most 5 new connections per server).
- 00:57 Weight packing rewritten with NEON (two values per output, gathered per tile row): unit tests pass, but paged search time did not change (1.28 s vs 1.31 s). Server timing (NPU_SERVER_TIMING=1) for one 2,000 x 784 block: 25-35 ms, of which 63 load calls take 20-29 ms (about 0.35 ms each: the fixed cost of a FastRPC call), packing 4.7 ms, close+open 0.6 ms. The calls are the cost, not the conversion. Next: load all groups of a layer in one call through the shared buffer.
- 00:58 DSP method 4 added to npu/npu_skel.c: load any number of consecutive output groups of a layer in one call, weights taken from the shared input buffer and copied to VTCM with HVX (falls back to memcpy if the HVX lock is refused). The Rust server now uses it for every model. Loading one 2,000 x 784 block: 25-35 ms -> 5.6 ms (0.5 close+open, 3.9 packing, 0.9 for the single load call). Nearest-neighbour search over 30,000 images: 1.3 s -> 0.65 s wall. Man-page search: 0.22 s -> 0.05 s. All 23 checkable original cases still verify and all 23 real-data results are unchanged. Older DSP libraries without method 4 make the server report the load error and ask for a rebuild.
- 00:59 New server request 'A' (argmax): returns only the largest output per row and its index, computed in the server from the unpacked tiles. With it, nearest neighbour for all 10,000 MNIST test digits against 30,000 images runs in 1.97 s wall (235 billion multiply-adds, 112 ms of DSP time, 15 weight blocks): 96.2% correct. Before, the full score matrix (10,000 x 2,000 floats per block) would have crossed the socket 15 times.
- 01:00 229 NSL-KDD nearest neighbour with no fitted model: every one of 22,544 test connections compared with all 125,973 training connections (272 billion multiply-adds) in 16 blocks of 8,000: 78.8% correct, attacks 68.4% (DoS 86, probe 82, R2L 11, U2R 36), false alarms 7.5%. 0.79 s of DSP time, 7.8 s wall. That is 346 billion multiply-adds per second on the DSP clock, far below the 2.3 trillion of case 301: with only 3 input tiles per output group, the per-group fixed work (clear, bias load, two converts, two stores) dominates. Wide-output, narrow-input layers are convert-limited.
- 01:01 412 generated domains by nearest neighbour (cosine of 1,024 hashed letter triples, 10,000 test names against 60,000 labelled names in 60 blocks, 6.3 s wall): only 69.1% overall with 8.1% false alarms, but the word-built families jump: suppobox 78% (17% with the readout), nymaim 36% (6%), gozi 68% (65%). Random-string families have no close neighbour, so their nearest name is arbitrary. The two methods fail on opposite kinds of name; combining them next.
- 01:02 512 combination for generated domains, done twice. First attempt paired the neighbour rule with a weak readout (hashed triples only, 77.8%) and looked bad (78.9%, 22.5% false alarms): not a fair test. Second attempt uses the strong 1,764-input readout (88.2% alone on this split of 10,000 names). "Readout OR nearest neighbour is a generated name with cosine above c":
  - c = 0.5: 89.3% correct, 9.7% false alarms, suppobox 69%
  - c = 0.6: 88.7% correct, 7.9% false alarms, suppobox 35%
  - c = 0.7: 88.3% correct, 7.5% false alarms, suppobox 17%
  So neighbours rescue the word-built family (17% -> 69%) at a cost of about 2 points of false alarms. No threshold was tuned on the test names; all three are reported.
- 01:08 Soak test: every original case plus 301, 204 and 229 in a loop for 306 s: 28 passes, 672 of 672 "verified" lines, 0 failures, highest temperature 49 C throughout (it was 49 C at the start), no DSP reset, no kernel errors from fastrpc.
- 01:09 328 NSL-KDD with three opinions (closed-form classifier, whitened distance, label of the nearest training connection): alarm on any one: 85.0% correct, 8.9% false alarms; any two: 79.8%, 7.4%; all three: 73.3%, 1.7%. "Any one" is the best accuracy of the hour on KDDTest+; "all three" is a low-false-alarm setting.
- 01:10 Final quiet live test with the full baseline (1,805 flow records): 17 records, 0 alarms (threshold 14.2). Unit tests pass. Server stopped so the DSP is not held at a raised clock.

## Findings at the end of the hour (stopped at 41 minutes)

What works well on real data:

| Use | Result |
|---|---|
| Search with no model | MNIST nearest neighbour 96.2% (10,000 against 30,000 images, 1.97 s); man-page search over 20,646 entries in 0.05 s |
| Intrusion benchmark (NSL-KDD) | 85.0% with any of three opinions; 73.3% with 1.7% false alarms when all three agree; 78.8% by nearest neighbour alone |
| Live flow scoring on this machine | 1.1% false alarms on unseen real traffic, every scan flagged, quiet captures 0 alarms |
| Closed-form MNIST | 94.14% (1,024 features), 95.56% (2,048) |
| Radio, real recordings | OOK or FSK 87% by a rule with no fitting; repeat packets found for 16 of 24 devices |
| Phishing hosts | 69.9% of unseen hosts at 1% false alarms |
| Latency forecast | 27% lower error than repeating the last value |

What does not work, or only partly:

| Use | Why |
|---|---|
| Lookalike matching on a phishing feed | Today's phishing sits on free hosting; hostnames rarely imitate brands |
| Word-built generated domains | Statistics see ordinary words; neighbours help (suppobox 17% -> 69%) at about 2 points of false alarms |
| R2L attacks in NSL-KDD | 11-40% whatever the method; they look like normal sessions |
| First-packet signatures | Too few new connections per server in real modern traffic |
| Wi-Fi rate prediction | Barely better than repeating the last value |
| Transmitter fingerprinting, beamforming, denoising on real signals | No suitable real data (no array recordings, no clean references, RadioML link dead) |

Engine facts learned:

- The accelerator does 2.3 trillion FP16 multiply-adds per second on its own clock (verified output).
- A FastRPC call costs about 0.35 ms whatever it carries. Fewer, larger calls is the main lever: loading a
  layer in one call cut model loading from 25-35 ms to 5.6 ms.
- Layers with few input tiles per output group are limited by convert and store, not by multiply (346 billion/s).
- 4 MB of VTCM is granted; 16 MB is refused. A 784 -> 4,096 layer does not fit.
- End to end the server reaches 200-240 billion multiply-adds per second; the copies between client, server and
  shared buffers are the remaining cost.
- 306 s of continuous load: 672 of 672 checks correct, 49 C, no DSP reset. No DSP reset in the whole hour.

Server and tool changes made during the hour: NEON result unpacking; 8 MB shared output buffer; NEON weight
packing; DSP method 4 (load a layer in one call); request 'A' (best match per row); unit tests; ids/live2.sh and
ids/watch.sh; scripts/prep_real.py.

## Radio identification against the CPU (2026-11-25, 07:08 start, 30 minutes allowed)

| Time | Note |
|---|---|
| 07:08 | Start. DSP running on build 00078, server stopped. 164 recordings from the earlier hour; need labelled sets with many recordings per device |
| 07:09 | Selected 46 devices / 776 recordings from rtl_433_tests, downloading (about 225 MB) |
| 07:11 | First run: 96.9% nearest, 97.7% readout, NPU = CPU answers on 383/384. NPU slower. Timing included the CPU fit by mistake |
| 07:13 | Cross-session set (11 protocols, 299 recordings): 79.4%. ambient_weather, radiohead_ask, alecto fail |
| 07:15 | Spectra scaling with FFT reference and processor time. FFT on one core beats all matrix paths via the server |
| 07:16 | Leave one session out: 82.3% |
| 07:17 | Time breakdown: DSP 0.012 s of 0.20 s; socket and copies 0.125 s |
| 07:18 | In-process session: 0.47 / 0.52 / 0.51 s for 128 / 256 / 512-point |
| 07:19 | 16 recordings per request: 0.53 / 0.40 / 0.36 s; 128-point limited by the 32-batch cap |
| 07:20 | Cap raised to 256: 0.21 / 0.27 / 0.30 s, 67 calls. Cases 1-3 still verified. Highest temperature 34.8 C |
| 07:21 | Stopped experiments; writing up |

Findings: same accuracy on NPU and CPU; NPU via the server is socket-bound; in-process with full calls it is
2.7-5.1 times faster than ten CPU threads on the same matrices and about twice as fast as a one-core FFT; it
uses 9-25 times less processor time. No DSP reset.
