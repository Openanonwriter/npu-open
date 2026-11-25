//! 130. Radio device identification on real off-air recordings, run twice with the same code: once with
//! the matrix work on the NPU and once on the CPU (all cores). Recordings are from merbanan/rtl_433_tests
//! (`data/rtl433id/index.txt`: lines of `device file.cu8`, 250 kS/s unsigned 8-bit I/Q).
//!
//! Every stage is a matrix product, so one `Engine` trait covers both paths:
//!   A. 128-point spectra of every frame (256 -> 256 matrix): where the energy sits and how it moves.
//!   B. timing spectra of bursts (1024 -> 512 matrix): how fast the envelope and the frequency switch.
//!   C. identification: nearest fitted recording, and a random-feature layer with a closed-form readout.
//! Nothing is trained by gradient descent.
use crate::real::{dft128, radio_features};
use crate::util::*;
use crate::Case;
use npu::proto::Client;
use npu::session::Layer;
use std::time::Instant;

pub const CASES: [Case; 3] = [(130, "radioid", device_identification), (131, "radioid", protocol_identification), (132, "radioid", spectrum_scaling)];

trait Engine {
    fn define(&mut self, layers: &[Layer]) -> Result<(), String>;
    fn run(&mut self, x: &[f32], rows: usize) -> Result<Vec<f32>, String>;
}

struct Npu<'a>(&'a mut Client);
impl Engine for Npu<'_> {
    fn define(&mut self, layers: &[Layer]) -> Result<(), String> {
        self.0.define(layers).map(|_| ())
    }
    fn run(&mut self, x: &[f32], rows: usize) -> Result<Vec<f32>, String> {
        self.0.run(x, rows).map(|r| r.0)
    }
}

/// The same layers in float32 on the CPU, rows split over `threads` threads.
struct Cpu {
    layers: Vec<Layer>,
    threads: usize,
}

/// Dot product with 16 independent sums so the compiler can use NEON (a single running sum cannot be
/// vectorised because float addition is not associative).
fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0f32; 16];
    let (ca, cb) = (a.chunks_exact(16), b.chunks_exact(16));
    let tail: f32 = ca.remainder().iter().zip(cb.remainder()).map(|(x, y)| x * y).sum();
    for (x, y) in ca.zip(cb) {
        for i in 0..16 {
            acc[i] += x[i] * y[i];
        }
    }
    acc.iter().sum::<f32>() + tail
}

impl Engine for Cpu {
    fn define(&mut self, layers: &[Layer]) -> Result<(), String> {
        self.layers = layers.to_vec();
        Ok(())
    }
    fn run(&mut self, x: &[f32], rows: usize) -> Result<Vec<f32>, String> {
        let mut cur = x.to_vec();
        for l in &self.layers {
            let mut out = vec![0f32; rows * l.n_out];
            let per = rows.div_ceil(self.threads).max(1);
            std::thread::scope(|s| {
                for (o, i) in out.chunks_mut(per * l.n_out).zip(cur.chunks(per * l.n_in)) {
                    s.spawn(move || {
                        for (orow, irow) in o.chunks_exact_mut(l.n_out).zip(i.chunks_exact(l.n_in)) {
                            for j in 0..l.n_out {
                                let v = dot(&l.w[j * l.n_in..(j + 1) * l.n_in], irow) + l.b[j];
                                orow[j] = if l.relu { v.max(0.0) } else { v };
                            }
                        }
                    });
                }
            });
            cur = out;
        }
        Ok(cur)
    }
}

const WIN: usize = 1024; // decimated samples per burst window (16.4 ms at 62.5 kS/s)
const BINS: usize = 256; // timing-spectrum bins kept (61 Hz apart, up to 15.6 kHz)
const MAXWIN: usize = 8;
const DIM: usize = 100 + 2 * BINS;

fn timing_matrix() -> Vec<f32> {
    let s = 1.0 / (WIN as f32).sqrt();
    let mut w = vec![0f32; 2 * BINS * WIN];
    for k in 0..BINS {
        for t in 0..WIN {
            let ph = 2.0 * std::f32::consts::PI * (((k + 1) * t) % WIN) as f32 / WIN as f32;
            w[k * WIN + t] = s * ph.cos();
            w[(BINS + k) * WIN + t] = s * ph.sin();
        }
    }
    w
}

fn read_iq(file: &str) -> Result<Vec<f32>, String> {
    let b = std::fs::read(data_path("data/rtl433id").join(file)).map_err(|e| format!("{file}: {e}"))?;
    Ok(b.iter().take(2 << 20).map(|&u| (u as f32 - 127.5) / 127.5).collect())
}

/// Burst windows of one recording: (envelope, frequency) pairs of WIN samples each, decimated by 4.
/// Envelope is amplitude relative to the window's peak; frequency is the phase step per sample in units
/// of pi, zeroed where there is no signal and with its mean (the tuning offset) removed.
fn burst_windows(iq: &[f32]) -> Vec<f32> {
    let n = iq.len() / 8;
    let (mut env, mut fm) = (vec![0f32; n], vec![0f32; n]);
    for i in 0..n {
        let (mut e, mut re, mut im) = (0f32, 0f32, 0f32);
        for t in 4 * i..4 * i + 4 {
            let (a, b) = (iq[2 * t], iq[2 * t + 1]);
            e += a * a + b * b;
            if t > 0 {
                let (c, d) = (iq[2 * t - 2], iq[2 * t - 1]);
                re += a * c + b * d;
                im += b * c - a * d;
            }
        }
        env[i] = (e / 4.0).sqrt();
        fm[i] = im.atan2(re) / std::f32::consts::PI;
    }
    let (med, top) = (quantile(&env, 0.5), quantile(&env, 0.999));
    let thr = med + 0.25 * (top - med);
    let mut starts = Vec::new();
    let mut i = 0;
    while i + WIN <= n && starts.len() < MAXWIN {
        if env[i] > thr {
            starts.push(i.saturating_sub(8));
            i += WIN;
        } else {
            i += 1;
        }
    }
    if starts.is_empty() {
        starts.push(0);
    }
    let mut out = Vec::with_capacity(starts.len() * 2 * WIN);
    for s in starts {
        let e = &env[s..s + WIN];
        let peak = e.iter().cloned().fold(1e-6f32, f32::max);
        out.extend(e.iter().map(|v| v / peak));
        let on: Vec<bool> = e.iter().map(|&v| v > thr).collect();
        let cnt = on.iter().filter(|&&b| b).count().max(1) as f32;
        let mean = (0..WIN).filter(|&t| on[t]).map(|t| fm[s + t]).sum::<f32>() / cnt;
        out.extend((0..WIN).map(|t| if on[t] { fm[s + t] - mean } else { 0.0 }));
    }
    out
}

fn normalise(v: &mut [f32]) {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
    v.iter_mut().for_each(|x| *x /= n);
}

/// Feature rows (DIM each) for every recording; returns the seconds spent in stage A and stage B.
fn features(e: &mut dyn Engine, files: &[String]) -> Result<(Vec<f32>, [f64; 2], f64), String> {
    let mut secs = [0f64; 2];
    let mut macs = 0f64;
    let mut feats = vec![0f32; files.len() * DIM];
    let (mut wins, mut count) = (Vec::new(), Vec::new());
    e.define(&[Layer { n_in: 256, n_out: 256, relu: false, w: dft128(), b: vec![0.0; 256] }])?;
    for (r, f) in files.iter().enumerate() {
        let iq = read_iq(f)?;
        let frames = iq.len() / 256;
        let mut x = vec![0f32; frames * 256];
        for fr in 0..frames {
            for t in 0..128 {
                x[fr * 256 + t] = iq[2 * (fr * 128 + t)];
                x[fr * 256 + 128 + t] = iq[2 * (fr * 128 + t) + 1];
            }
        }
        let t0 = Instant::now();
        let y = e.run(&x, frames)?;
        secs[0] += t0.elapsed().as_secs_f64();
        macs += (frames * 256 * 256) as f64;
        let p: Vec<f32> = y.chunks_exact(256).flat_map(|row| (0..128).map(move |k| row[k] * row[k] + row[128 + k] * row[128 + k])).collect();
        let (mut v, _) = radio_features(&p);
        normalise(&mut v);
        feats[r * DIM..r * DIM + 100].copy_from_slice(&v);
        let w = burst_windows(&iq);
        count.push(w.len() / WIN);
        wins.extend(w);
    }
    e.define(&[Layer { n_in: WIN, n_out: 2 * BINS, relu: false, w: timing_matrix(), b: vec![0.0; 2 * BINS] }])?;
    let rows = wins.len() / WIN;
    let t0 = Instant::now();
    let y = e.run(&wins, rows)?;
    secs[1] = t0.elapsed().as_secs_f64();
    macs += (rows * WIN * 2 * BINS) as f64;
    let mut row = 0;
    for (r, &c) in count.iter().enumerate() {
        for w in 0..c {
            let o = &y[(row + w) * 2 * BINS..(row + w + 1) * 2 * BINS];
            let base = r * DIM + 100 + (w % 2) * BINS; // windows alternate: envelope, frequency
            for k in 0..BINS {
                feats[base + k] += (o[k] * o[k] + o[BINS + k] * o[BINS + k]).sqrt().sqrt();
            }
        }
        row += c;
        normalise(&mut feats[r * DIM + 100..r * DIM + 100 + BINS]);
        normalise(&mut feats[r * DIM + 100 + BINS..(r + 1) * DIM]);
    }
    Ok((feats, secs, macs))
}

struct Outcome {
    nearest: Vec<usize>,
    readout: Vec<usize>,
    secs: [f64; 3],
    macs: f64,
}

type Split = (Vec<usize>, Vec<usize>); // (fitted recordings, tested recordings)

fn identify(e: &mut dyn Engine, files: &[String], label: &[usize], classes: usize, splits: &[Split]) -> Result<Outcome, String> {
    let (feats, s, mut macs) = features(e, files)?;
    let (mut nearest, mut readout, mut secs) = (Vec::new(), Vec::new(), 0f64);
    for (fit, test) in splits {
    let (fit, test) = (&fit[..], &test[..]);
    let pick = |idx: &[usize]| -> Vec<f32> { idx.iter().flat_map(|&i| feats[i * DIM..(i + 1) * DIM].to_vec()).collect() };
    let (xf, xt) = (pick(fit), pick(test));
    // `secs` is matrix time only; the closed-form fit below runs on the CPU for both engines and is not counted
    let mut timed = |e: &mut dyn Engine, x: &[f32], rows: usize| -> Result<Vec<f32>, String> {
        let t0 = Instant::now();
        let y = e.run(x, rows);
        secs += t0.elapsed().as_secs_f64();
        y
    };
    // Nearest fitted recording: rows are unit length per block, so the largest product is the nearest.
    e.define(&[Layer { n_in: DIM, n_out: fit.len(), relu: false, w: xf.clone(), b: vec![0.0; fit.len()] }])?;
    let y = timed(e, &xt, test.len())?;
    nearest.extend(argmax_rows(&y, fit.len()).iter().map(|p| label[fit[p.0]]));
    // Random ReLU features, readout fitted in closed form (ridge regression by Cholesky on the CPU).
    let feat_n = 1024usize;
    let mut r = Rng(130);
    let l1 = Layer { n_in: DIM, n_out: feat_n, relu: true, w: r.normals(feat_n * DIM, 1.0), b: (0..feat_n).map(|_| r.uniform() - 0.5).collect() };
    e.define(std::slice::from_ref(&l1))?;
    let phi = timed(e, &xf, fit.len())?;
    let (mut pp, mut pt) = (vec![0f64; feat_n * feat_n], vec![0f64; feat_n * classes]);
    for (row, &i) in fit.iter().enumerate() {
        let p = &phi[row * feat_n..(row + 1) * feat_n];
        for a in 0..feat_n {
            if p[a] != 0.0 {
                pt[a * classes + label[i]] += p[a] as f64;
                for b in a..feat_n {
                    pp[a * feat_n + b] += (p[a] * p[b]) as f64;
                }
            }
        }
    }
    for a in 0..feat_n {
        for b in 0..a {
            pp[a * feat_n + b] = pp[b * feat_n + a];
        }
    }
    let sol = solve_spd(&pp, feat_n, &pt, classes, 1e-2 * fit.len() as f64);
    let w2: Vec<f32> = (0..classes).flat_map(|j| (0..feat_n).map(move |a| (j, a))).map(|(j, a)| sol[a * classes + j] as f32).collect();
    e.define(&[l1, Layer { n_in: feat_n, n_out: classes, relu: false, w: w2, b: vec![0.0; classes] }])?;
    let y = timed(e, &xt, test.len())?;
    readout.extend(argmax_rows(&y, classes).iter().map(|p| p.0));
    macs += (test.len() * DIM * fit.len() + (fit.len() + test.len()) * DIM * feat_n + test.len() * feat_n * classes) as f64;
    }
    Ok(Outcome { nearest, readout, secs: [s[0], s[1], secs], macs })
}

fn threads() -> usize {
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1)
}

/// Run the whole identification on the NPU, then on the CPU, and print both.
fn compare(c: &mut Client, id: u32, title: &str, files: &[String], label: &[usize], names: &[String], splits: &[Split]) -> Result<(), String> {
    let test: Vec<usize> = splits.iter().flat_map(|s| s.1.clone()).collect();
    let (test, fit) = (&test[..], &splits[0].0[..]);
    let radio: f64 = files.iter().map(|f| data_path("data/rtl433id").join(f).metadata().map(|m| m.len().min(2 << 20)).unwrap_or(0) as f64 / 500_000.0).sum();
    let score = |p: &[usize]| 100.0 * p.iter().zip(test).filter(|(a, &i)| **a == label[i]).count() as f32 / test.len() as f32;
    println!("{id} | {title} | {} real recordings ({radio:.0} s of radio), {} classes; {} fitted, {} tested", files.len(), names.len(), fit.len(), test.len());
    let line = |who: &str, o: &Outcome, total: f64| {
        let m: f64 = o.secs.iter().sum();
        println!("    | {who:<16} | nearest {:.1}%, readout {:.1}% | matrix work {:.2} s (spectra {:.2}, timing {:.2}, identify {:.3}) = {:.0} billion multiply-adds/s | whole run {:.1} s, {:.0} times faster than real time",
            score(&o.nearest), score(&o.readout), m, o.secs[0], o.secs[1], o.secs[2], o.macs / m / 1e9, total, radio / total);
    };
    let t0 = Instant::now();
    let n = identify(&mut Npu(c), files, label, names.len(), splits)?;
    line("NPU", &n, t0.elapsed().as_secs_f64());
    if std::env::var("NPU_ONLY").is_err() {
        let t0 = Instant::now();
        let o = identify(&mut Cpu { layers: Vec::new(), threads: threads() }, files, label, names.len(), splits)?;
        line(&format!("CPU, {} threads", threads()), &o, t0.elapsed().as_secs_f64());
        let same = |a: &[usize], b: &[usize]| a.iter().zip(b).filter(|(x, y)| x == y).count();
        println!("    | same answer on NPU and CPU: nearest {} of {}, readout {} of {}", same(&n.nearest, &o.nearest), test.len(), same(&n.readout, &o.readout), test.len());
    }
    if std::env::var("NPU_DEBUG").is_ok() {
        for (k, name) in names.iter().enumerate() {
            let idx: Vec<usize> = (0..test.len()).filter(|&j| label[test[j]] == k).collect();
            let ok = idx.iter().filter(|&&j| n.nearest[j] == k).count();
            let ok2 = idx.iter().filter(|&&j| n.readout[j] == k).count();
            println!("    |   {name:<40} nearest {ok:>2}/{:<2} readout {ok2:>2}/{:<2}", idx.len(), idx.len());
        }
    }
    Ok(())
}

fn usable(file: &str) -> bool {
    data_path("data/rtl433id").join(file).metadata().map(|m| m.len() >= 60_000).unwrap_or(false)
}

fn class_of(names: &mut Vec<String>, name: &str) -> usize {
    names.iter().position(|n| n == name).unwrap_or_else(|| {
        names.push(name.to_string());
        names.len() - 1
    })
}

/// 130. Which device made this recording? Recordings of each device alternate between fitting and testing.
/// Easy version: the fitted and tested recordings of a device come from the same folder, usually the same
/// transmitter, receiver and day.
fn same_session_set() -> Result<(Vec<String>, Vec<usize>, Vec<String>, Split), String> {
    let index = std::fs::read_to_string(data_path("data/rtl433id/index.txt")).map_err(|e| format!("data/rtl433id/index.txt: {e}"))?;
    let (mut files, mut label, mut names): (Vec<String>, Vec<usize>, Vec<String>) = (Vec::new(), Vec::new(), Vec::new());
    for l in index.lines() {
        let p: Vec<&str> = l.split_whitespace().collect();
        if p.len() == 2 && usable(p[1]) {
            label.push(class_of(&mut names, p[0]));
            files.push(p[1].to_string());
        }
    }
    let (mut fit, mut test, mut seen) = (Vec::new(), Vec::new(), vec![0usize; names.len()]);
    for i in 0..files.len() {
        if seen[label[i]] % 2 == 0 { fit.push(i) } else { test.push(i) }
        seen[label[i]] += 1;
    }
    Ok((files, label, names, (fit, test)))
}

fn device_identification(c: &mut Client) -> Result<(), String> {
    let (files, label, names, split) = same_session_set()?;
    compare(c, 130, "Radio device identification, same session", &files, &label, &names, &[split])
}

/// The accelerator used from inside this process, with no server and no socket in between.
struct Direct(npu::session::Session);
impl Engine for Direct {
    fn define(&mut self, layers: &[Layer]) -> Result<(), String> {
        self.0.define(layers)
    }
    fn run(&mut self, x: &[f32], rows: usize) -> Result<Vec<f32>, String> {
        self.0.run(x, rows).map(|r| r.0)
    }
}

/// 134. Cases 130 and 132 again with the session in-process (run through `run.sh` with NPU_DIRECT=1 and the
/// server stopped). Shows what the server's socket costs.
pub fn direct() -> Result<(), String> {
    let mut e = Direct(npu::session::Session::connect()?);
    let (files, label, names, split) = same_session_set()?;
    let test = split.1.clone();
    let t0 = Instant::now();
    let o = identify(&mut e, &files, &label, names.len(), &[split])?;
    let total = t0.elapsed().as_secs_f64();
    let score = |p: &[usize]| 100.0 * p.iter().zip(&test).filter(|(a, &i)| **a == label[i]).count() as f32 / test.len() as f32;
    let m: f64 = o.secs.iter().sum();
    println!("134 | Device identification, NPU in-process | nearest {:.1}%, readout {:.1}% | matrix work {:.2} s (spectra {:.2}, timing {:.2}, identify {:.3}) = {:.0} billion multiply-adds/s | whole run {:.1} s",
        score(&o.nearest), score(&o.readout), m, o.secs[0], o.secs[1], o.secs[2], o.macs / m / 1e9, total);
    let half: Vec<&String> = files.iter().step_by(2).collect();
    for n in [128usize, 256, 512] {
        e.define(&[Layer { n_in: 2 * n, n_out: 2 * n, relu: false, w: dft(n), b: vec![0.0; 2 * n] }])?;
        let (mut secs, mut rows, mut st) = (0f64, 0usize, [0u64; 4]);
        let (mut queue, mut secs_b, mut calls, mut calls_b) = (Vec::new(), 0f64, 0u32, 0u32);
        for (i, f) in half.iter().enumerate() {
            let iq = read_iq(f)?;
            let frames = iq.len() / (2 * n);
            let mut x = vec![0f32; frames * 2 * n];
            for fr in 0..frames {
                for t in 0..n {
                    x[fr * 2 * n + t] = iq[2 * (fr * n + t)];
                    x[fr * 2 * n + n + t] = iq[2 * (fr * n + t) + 1];
                }
            }
            let t0 = Instant::now();
            let (_, s) = e.0.run(&x, frames)?;
            secs += t0.elapsed().as_secs_f64();
            rows += frames;
            st = [st[0] + s.dsp_us, st[1] + s.wall_us, st[2] + s.pack_us, st[3] + s.unpack_us];
            calls += s.calls;
            // The same frames again, queued and sent 16 recordings at a time: fewer, fuller FastRPC calls.
            queue.extend_from_slice(&x);
            if (i + 1) % 16 == 0 || i + 1 == half.len() {
                let t0 = Instant::now();
                let (_, s) = e.0.run(&queue, queue.len() / (2 * n))?;
                secs_b += t0.elapsed().as_secs_f64();
                calls_b += s.calls;
                queue.clear();
            }
        }
        println!("    |   {n}-point, 16 recordings per request: {secs_b:.2} s in {calls_b} calls (one recording per request: {calls} calls)");
        let g = (rows * 4 * n * n) as f64 / 1e9;
        println!("    | {n:>3}-point, {rows:>6} frames, {g:>4.0} billion multiply-adds | NPU in-process {secs:.2} s ({:.0} billion/s) = pack {:.2} + call {:.2} (on the DSP {:.2}) + unpack {:.2}",
            g / secs, st[2] as f64 / 1e6, st[1] as f64 / 1e6, st[0] as f64 / 1e6, st[3] as f64 / 1e6);
    }
    e.0.close();
    Ok(())
}

/// 131. The hard version: fit on one recording session (folder) of each protocol, test on the other sessions.
/// `index-sessions.txt` holds lines of `protocol session file.cu8`; session 0 is fitted.
fn protocol_identification(c: &mut Client) -> Result<(), String> {
    let index = std::fs::read_to_string(data_path("data/rtl433id/index-sessions.txt")).map_err(|e| format!("data/rtl433id/index-sessions.txt: {e}"))?;
    let (mut files, mut label, mut names): (Vec<String>, Vec<usize>, Vec<String>) = (Vec::new(), Vec::new(), Vec::new());
    let (mut fit, mut test, mut session) = (Vec::new(), Vec::new(), Vec::new());
    for l in index.lines() {
        let p: Vec<&str> = l.split_whitespace().collect();
        if p.len() == 3 && usable(p[2]) {
            if p[1] == "0" { fit.push(files.len()) } else { test.push(files.len()) }
            session.push(p[1].to_string());
            label.push(class_of(&mut names, p[0]));
            files.push(p[2].to_string());
        }
    }
    compare(c, 131, "Radio protocol identification, one session fitted", &files, &label, &names, &[(fit, test)])?;
    // 133. Leave one session out: every session is tested against a model fitted on all the other sessions.
    let splits: Vec<Split> = ["0", "1", "2", "3"].iter().map(|s| ((0..files.len()).filter(|&i| session[i] != *s).collect(), (0..files.len()).filter(|&i| session[i] == *s).collect())).filter(|(_, t): &Split| !t.is_empty()).collect();
    compare(c, 133, "Radio protocol identification, leave one session out", &files, &label, &names, &splits)
}

fn dft(n: usize) -> Vec<f32> {
    let s = 1.0 / (n as f32).sqrt();
    let mut w = vec![0f32; 4 * n * n];
    for k in 0..n {
        for t in 0..n {
            let ph = -2.0 * std::f32::consts::PI * ((k * t) % n) as f32 / n as f32;
            let (a, b) = (s * ph.cos(), s * ph.sin());
            w[k * 2 * n + t] = a;
            w[k * 2 * n + n + t] = -b;
            w[(n + k) * 2 * n + t] = b;
            w[(n + k) * 2 * n + n + t] = a;
        }
    }
    w
}

/// In-place radix-2 FFT (n a power of two), the method a CPU would normally use for spectra.
fn fft(re: &mut [f32], im: &mut [f32], tw: &[(f32, f32)]) {
    let n = re.len();
    let mut j = 0;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j |= bit;
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }
    let mut len = 2;
    while len <= n {
        let step = n / len;
        for s in (0..n).step_by(len) {
            for k in 0..len / 2 {
                let (c, d) = tw[k * step];
                let (a, b) = (s + k, s + k + len / 2);
                let (xr, xi) = (re[b] * c - im[b] * d, re[b] * d + im[b] * c);
                re[b] = re[a] - xr;
                im[b] = im[a] - xi;
                re[a] += xr;
                im[a] += xi;
            }
        }
        len *= 2;
    }
}

/// CPU seconds (user + system, all threads) used so far by a process, from /proc/<pid>/stat (10 ms steps).
fn cpu_secs(pid: &str) -> f64 {
    let s = std::fs::read_to_string(format!("/proc/{pid}/stat")).unwrap_or_default();
    let after = s.rsplit(')').next().unwrap_or("");
    let v: Vec<f64> = after.split_whitespace().skip(11).take(2).filter_map(|x| x.parse().ok()).collect();
    v.iter().sum::<f64>() / 100.0
}

/// 132. Where the NPU wins and where it does not: the spectrum stage alone at three frame lengths, on the
/// NPU, on all CPU cores, on one CPU core, and by FFT on one core. Same recordings; the first three use the
/// same matrices. "CPU-s" is processor time consumed (client plus NPU server), a stand-in for energy: this
/// machine was on mains power, so battery draw could not be read.
fn spectrum_scaling(c: &mut Client) -> Result<(), String> {
    let index = std::fs::read_to_string(data_path("data/rtl433id/index.txt")).map_err(|e| format!("data/rtl433id/index.txt: {e}"))?;
    let files: Vec<String> = index.lines().filter_map(|l| l.split_whitespace().nth(1)).filter(|f| usable(f)).step_by(2).map(String::from).collect();
    let server = std::fs::read_to_string(data_path("server.pid")).unwrap_or_default().trim().to_string();
    let used = || cpu_secs("self") + cpu_secs(&server);
    println!("132 | Spectra of {} real recordings at three frame lengths | seconds in the transform (processor seconds consumed)", files.len());
    for n in [128usize, 256, 512] {
        let layer = [Layer { n_in: 2 * n, n_out: 2 * n, relu: false, w: dft(n), b: vec![0.0; 2 * n] }];
        let frames_of = |f: &String| -> Result<(Vec<f32>, usize), String> {
            let iq = read_iq(f)?;
            let frames = iq.len() / (2 * n);
            let mut x = vec![0f32; frames * 2 * n];
            for fr in 0..frames {
                for t in 0..n {
                    x[fr * 2 * n + t] = iq[2 * (fr * n + t)];
                    x[fr * 2 * n + n + t] = iq[2 * (fr * n + t) + 1];
                }
            }
            Ok((x, frames))
        };
        let (mut secs, mut cpu) = ([0f64; 4], [0f64; 4]);
        if n != 256 {
            // Where the NPU path spends its time, on the first 64 recordings (server-side counters).
            c.define(&layer)?;
            let (mut st, mut wall, mut nrow) = ([0u64; 4], 0f64, 0usize);
            for f in files.iter().take(64) {
                let (x, frames) = frames_of(f)?;
                let t0 = Instant::now();
                let (_, s) = c.run(&x, frames)?;
                wall += t0.elapsed().as_secs_f64();
                nrow += frames;
                st = [st[0] + s.dsp_us, st[1] + s.wall_us, st[2] + s.pack_us, st[3] + s.unpack_us];
            }
            println!("    |   NPU path, {n}-point, {nrow} frames: {:.3} s seen by the client = pack to FP16 {:.3} + call {:.3} (of which on the DSP {:.3}) + unpack {:.3} + socket and copies {:.3}",
                wall, st[2] as f64 / 1e6, st[1] as f64 / 1e6, st[0] as f64 / 1e6, st[3] as f64 / 1e6, wall - (st[1] + st[2] + st[3]) as f64 / 1e6);
        }
        let (mut rows, mut worst, mut worst_fft) = (0usize, 0f32, 0f32);
        let mut engines: Vec<Box<dyn Engine>> = vec![Box::new(Npu(c)), Box::new(Cpu { layers: Vec::new(), threads: threads() }), Box::new(Cpu { layers: Vec::new(), threads: 1 })];
        let mut keep: Vec<Vec<f32>> = Vec::new(); // first 16 recordings' NPU output, to compare
        for (k, e) in engines.iter_mut().enumerate() {
            e.define(&layer)?;
            for (i, f) in files.iter().enumerate() {
                let (x, frames) = frames_of(f)?;
                let (t0, u0) = (Instant::now(), used());
                let y = e.run(&x, frames)?;
                secs[k] += t0.elapsed().as_secs_f64();
                cpu[k] += used() - u0;
                if k == 0 {
                    rows += frames;
                    if i < 16 {
                        keep.push(y);
                    }
                } else if k == 1 && i < 16 {
                    worst = worst.max(rel_err(&keep[i], &y));
                }
            }
        }
        drop(engines);
        let tw: Vec<(f32, f32)> = (0..n / 2).map(|k| (-2.0 * std::f32::consts::PI * k as f32 / n as f32).sin_cos()).map(|(s, co)| (co, s)).collect();
        let s = 1.0 / (n as f32).sqrt();
        for (i, f) in files.iter().enumerate() {
            let (x, _) = frames_of(f)?;
            let (t0, u0) = (Instant::now(), used());
            let mut y = vec![0f32; x.len()];
            for (o, row) in y.chunks_exact_mut(2 * n).zip(x.chunks_exact(2 * n)) {
                o.copy_from_slice(row);
                let (re, im) = o.split_at_mut(n);
                fft(re, im, &tw);
                o.iter_mut().for_each(|v| *v *= s);
            }
            secs[3] += t0.elapsed().as_secs_f64();
            cpu[3] += used() - u0;
            if i < 16 {
                worst_fft = worst_fft.max(rel_err(&keep[i], &y));
            }
        }
        let g = (rows * 4 * n * n) as f64 / 1e9;
        println!("    | {n:>3}-point, {rows:>6} frames, {g:>4.0} billion multiply-adds | NPU {:.2} s ({:.2} CPU-s, {:.0} billion/s) | CPU {} threads {:.2} s ({:.2} CPU-s, {:.0} billion/s) | CPU 1 thread {:.2} s ({:.0} billion/s) | FFT 1 thread {:.2} s | NPU differs from CPU matrix by {:.1}%, from FFT by {:.1}% of signal RMS at worst",
            secs[0], cpu[0], g / secs[0], threads(), secs[1], cpu[1], g / secs[1], secs[2], g / secs[2], secs[3], 100.0 * worst, 100.0 * worst_fft);
    }
    Ok(())
}
