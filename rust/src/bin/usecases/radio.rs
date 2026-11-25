//! Radio demonstrations (20-25) on synthetic signals: no SDR was attached. Complex samples are carried
//! as real pairs: a row is [re..., im...], and a complex matrix W becomes [[Wr, -Wi], [Wi, Wr]].
use crate::util::*;
use crate::Case;
use npu::proto::Client;
use std::f32::consts::PI;

pub const CASES: [Case; 6] = [(20, "radio", modulation), (21, "radio", spectrum), (22, "radio", fingerprinting), (23, "radio", matched_filters), (24, "radio", beamforming), (25, "radio", denoising)];

/// Real form of a complex matrix (n_out x n_in), for rows laid out as [re..., im...].
fn complex_weights(wr: &[f32], wi: &[f32], n_out: usize, n_in: usize) -> Vec<f32> {
    let mut w = vec![0f32; 4 * n_out * n_in];
    for o in 0..n_out {
        for i in 0..n_in {
            let (a, b) = (wr[o * n_in + i], wi[o * n_in + i]);
            w[o * 2 * n_in + i] = a;
            w[o * 2 * n_in + n_in + i] = -b;
            w[(n_out + o) * 2 * n_in + i] = b;
            w[(n_out + o) * 2 * n_in + n_in + i] = a;
        }
    }
    w
}

/// Discrete Fourier transform as a matrix: bin k = sum_n x[n] exp(-2 pi i k n / N) / sqrt(N).
fn dft_weights(n: usize) -> Vec<f32> {
    let s = 1.0 / (n as f32).sqrt();
    let (mut wr, mut wi) = (vec![0f32; n * n], vec![0f32; n * n]);
    for k in 0..n {
        for t in 0..n {
            let ph = -2.0 * PI * ((k * t) % n) as f32 / n as f32;
            wr[k * n + t] = s * ph.cos();
            wi[k * n + t] = s * ph.sin();
        }
    }
    complex_weights(&wr, &wi, n, n)
}

/// Power per bin from rows of [re..., im...].
fn power(y: &[f32], n: usize) -> Vec<f32> {
    y.chunks_exact(2 * n).flat_map(|r| (0..n).map(|k| r[k] * r[k] + r[n + k] * r[n + k]).collect::<Vec<_>>()).collect()
}

/// 20. Modulation recognition without a trained model: raising a phase-keyed signal to the 2nd or 4th
/// power turns its hidden carrier into a spectral line. The three spectra per burst come from the accelerator.
fn modulation(c: &mut Client) -> Result<(), String> {
    let (n, sps, per_class) = (128usize, 4usize, 100usize);
    let names = ["2-tone FSK", "BPSK", "QPSK", "8PSK"];
    let mut r = Rng(20);
    let (mut x, mut labels) = (Vec::new(), Vec::new());
    for class in 0..4 {
        for _ in 0..per_class {
            let f0 = (3 + r.next() % 10) as f32 / n as f32;
            let (mut re, mut im, mut phase, mut sym) = (vec![0f32; n], vec![0f32; n], 0f32, 0f32);
            for t in 0..n {
                if t % sps == 0 {
                    sym = match class { 0 => if r.next() % 2 == 0 { 8.0 / n as f32 } else { -8.0 / n as f32 }, 1 => PI * (r.next() % 2) as f32, 2 => PI / 2.0 * (r.next() % 4) as f32, _ => PI / 4.0 * (r.next() % 8) as f32 };
                }
                let ph = if class == 0 { phase += 2.0 * PI * (f0 + sym); phase } else { 2.0 * PI * f0 * t as f32 + sym };
                re[t] = ph.cos() + r.normal() * 0.1;
                im[t] = ph.sin() + r.normal() * 0.1;
            }
            // rows: the signal, its square and its fourth power (complex), computed here; the transforms run on the NPU
            let (mut re2, mut im2, mut re4, mut im4) = (vec![0f32; n], vec![0f32; n], vec![0f32; n], vec![0f32; n]);
            for t in 0..n {
                re2[t] = re[t] * re[t] - im[t] * im[t];
                im2[t] = 2.0 * re[t] * im[t];
                re4[t] = re2[t] * re2[t] - im2[t] * im2[t];
                im4[t] = 2.0 * re2[t] * im2[t];
            }
            for (a, b) in [(&re, &im), (&re2, &im2), (&re4, &im4)] {
                x.extend_from_slice(a);
                x.extend_from_slice(b);
            }
            labels.push(class);
        }
    }
    let w = dft_weights(n);
    c.linear(w.clone(), 2 * n, 2 * n, false)?;
    let rows = 3 * labels.len();
    let (y, st) = c.run(&x, rows)?;
    let p = power(&y, n);
    let peak = |row: usize| { let s = &p[row * n..(row + 1) * n]; s.iter().cloned().fold(0.0, f32::max) / (s.iter().sum::<f32>() / n as f32) };
    // Threshold chosen by hand on these synthetic signals: a spectral line stands well above the average bin.
    const T: f32 = 22.0;
    let mut per_class = [0usize; 4];
    let mut med = [[0f32; 3]; 4];
    for class in 0..4 {
        for (j, m) in med[class].iter_mut().enumerate() {
            *m = quantile(&(0..labels.len()).filter(|&i| labels[i] == class).map(|i| peak(3 * i + j)).collect::<Vec<_>>(), 0.5);
        }
    }
    if std::env::var("NPU_DEBUG").is_ok() {
        eprintln!("median peak-to-average of x, x^2, x^4 per class: {med:?}");
    }
    for i in 0..labels.len() {
        let (p1, p2, p4) = (peak(3 * i), peak(3 * i + 1), peak(3 * i + 2));
        // BPSK: one strong line in x^2 (median 118 here). QPSK: none in x^2, strong in x^4 (95).
        // 2-tone FSK: weaker lines in both (37 and 31). 8PSK: no line in either (13 and 9).
        // The first rule set (a single threshold of 30) called 92 of 100 FSK bursts BPSK.
        let _ = p1;
        let guess = if p2 > 70.0 { 1 } else if p4 > 60.0 { 2 } else if p2 > T { 0 } else { 3 };
        per_class[labels[i]] += (guess == labels[i]) as usize;
    }
    let right: usize = per_class.iter().sum();
    let err = rel_err(&y, &matmul_ref(&x, rows, 2 * n, &w, 2 * n));
    report(20, "Modulation recognition", "1200 spectra of 128 points (synthetic)", &format!("{right} of {} bursts named correctly; per type {per_class:?} of 100 ({})", labels.len(), names.join(", ")), rows, st, Some(err < 0.05));
    Ok(())
}

/// 21. Spectrum occupancy: a 128-point Fourier transform of every frame as one matrix multiply.
fn spectrum(c: &mut Client) -> Result<(), String> {
    let (n, frames) = (128usize, 1024usize);
    let carriers = [(10usize, 1.0f32), (37, 0.5), (90, 0.25)];
    let mut r = Rng(21);
    let mut x = vec![0f32; frames * 2 * n];
    for f in 0..frames {
        let ph0: Vec<f32> = carriers.iter().map(|_| r.uniform() * 2.0 * PI).collect();
        for t in 0..n {
            let (mut re, mut im) = (r.normal() * 0.3, r.normal() * 0.3);
            for (ci, &(bin, amp)) in carriers.iter().enumerate() {
                let ph = 2.0 * PI * (bin * t) as f32 / n as f32 + ph0[ci];
                re += amp * ph.cos();
                im += amp * ph.sin();
            }
            x[f * 2 * n + t] = re;
            x[f * 2 * n + n + t] = im;
        }
    }
    let w = dft_weights(n);
    c.linear(w.clone(), 2 * n, 2 * n, false)?;
    let (y, st) = c.run(&x, frames)?;
    let p = power(&y, n);
    let avg: Vec<f32> = (0..n).map(|k| (0..frames).map(|f| p[f * n + k]).sum::<f32>() / frames as f32).collect();
    let floor = quantile(&avg, 0.5);
    let found: Vec<usize> = (0..n).filter(|&k| avg[k] > 6.0 * floor).collect();
    let ok = found == carriers.iter().map(|c| c.0).collect::<Vec<_>>();
    let err = rel_err(&y, &matmul_ref(&x, frames, 2 * n, &w, 2 * n));
    report(21, "Spectrum occupancy", "1024 frames, 128-point transform (synthetic)", &format!("occupied bins found: {found:?}"), frames, st, Some(err < 0.05 && ok));
    Ok(())
}

/// 22. Transmitter fingerprinting needs recordings of real transmitters; there is nothing honest to run without them.
fn fingerprinting(_c: &mut Client) -> Result<(), String> {
    println!("22 | Transmitter fingerprinting   | not run                            | needs recordings from real transmitters; no radio was attached");
    Ok(())
}

/// 23. Matched-filter bank: every window of a noisy stream correlated with 32 known sequences at once.
fn matched_filters(c: &mut Client) -> Result<(), String> {
    let (len, n_tpl, stream_len) = (64usize, 32usize, 30000usize);
    let mut r = Rng(23);
    let tpl: Vec<f32> = (0..n_tpl * len).map(|_| if r.next() % 2 == 0 { 1.0 } else { -1.0 }).collect();
    let mut stream = r.normals(stream_len, 1.0);
    let (mut truth, mut pos) = (Vec::new(), 200);
    while pos + len < stream_len {
        let k = (r.next() % n_tpl as u64) as usize;
        (0..len).for_each(|i| stream[pos + i] += tpl[k * len + i]);
        truth.push((pos, k));
        pos += 300 + (r.next() % 300) as usize;
    }
    let w: Vec<f32> = tpl.iter().map(|v| v / 8.0).collect();
    let rows = stream_len - len + 1;
    let x: Vec<f32> = (0..rows).flat_map(|i| stream[i..i + len].to_vec()).collect();
    c.linear(w.clone(), n_tpl, len, false)?;
    let (y, st) = c.run(&x, rows)?;
    let best = argmax_rows(&y, n_tpl);
    // a match scores 64/8 = 8 with noise 1; 5 is five noise deviations up
    let hits: Vec<usize> = (0..rows).filter(|&i| best[i].1 > 5.0).collect();
    let found = truth.iter().filter(|&&(p, k)| hits.contains(&p) && best[p].0 == k).count();
    let extra = hits.iter().filter(|&&i| !truth.iter().any(|&(p, _)| p == i)).count();
    // how many of the extra hits are windows that partly overlap a buried sequence (its energy raises every correlation)
    let near = hits.iter().filter(|&&i| truth.iter().any(|&(p, _)| p != i && (i as i64 - p as i64).abs() < len as i64)).count();
    if std::env::var("NPU_DEBUG").is_ok() {
        eprintln!("matched filters: {extra} extra hits, {near} of them within one sequence length of a buried sequence");
    }
    let err = rel_err(&y, &matmul_ref(&x, rows, len, &w, n_tpl));
    report(23, "Matched-filter bank", &format!("{rows} windows against 32 sequences"), &format!("{found} of {} buried sequences found at 0 dB; {extra} extra hits, {near} beside a real one", truth.len()), rows, st, Some(err < 0.05));
    Ok(())
}

/// 24. Beamforming: 16 antennas, 91 look directions; the steering matrix is applied to every snapshot.
fn beamforming(c: &mut Client) -> Result<(), String> {
    let (m, snaps) = (16usize, 2048usize);
    let angles: Vec<f32> = (0..91).map(|i| -90.0 + 2.0 * i as f32).collect();
    let sources = [(-20.0f32, 1.0f32), (36.0, 0.7)];
    let steer = |deg: f32, e: usize| PI * e as f32 * (deg * PI / 180.0).sin();
    let mut r = Rng(24);
    let mut x = vec![0f32; snaps * 2 * m];
    for s in 0..snaps {
        let ph: Vec<f32> = sources.iter().map(|_| r.uniform() * 2.0 * PI).collect();
        for e in 0..m {
            let (mut re, mut im) = (r.normal() * 0.3, r.normal() * 0.3);
            for (si, &(deg, amp)) in sources.iter().enumerate() {
                re += amp * (steer(deg, e) + ph[si]).cos();
                im += amp * (steer(deg, e) + ph[si]).sin();
            }
            x[s * 2 * m + e] = re;
            x[s * 2 * m + m + e] = im;
        }
    }
    let (mut wr, mut wi) = (vec![0f32; angles.len() * m], vec![0f32; angles.len() * m]);
    for (a, &deg) in angles.iter().enumerate() {
        for e in 0..m {
            wr[a * m + e] = steer(deg, e).cos() / m as f32;
            wi[a * m + e] = -steer(deg, e).sin() / m as f32;
        }
    }
    let w = complex_weights(&wr, &wi, angles.len(), m);
    c.linear(w.clone(), 2 * angles.len(), 2 * m, false)?;
    let (y, st) = c.run(&x, snaps)?;
    let p = power(&y, angles.len());
    let avg: Vec<f32> = (0..angles.len()).map(|a| (0..snaps).map(|s| p[s * angles.len() + a]).sum::<f32>() / snaps as f32).collect();
    let mut peaks: Vec<usize> = (1..angles.len() - 1).filter(|&a| avg[a] > avg[a - 1] && avg[a] > avg[a + 1]).collect();
    peaks.sort_by(|&a, &b| avg[b].partial_cmp(&avg[a]).unwrap());
    let mut got: Vec<f32> = peaks.iter().take(2).map(|&a| angles[a]).collect();
    got.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let ok = got.len() == 2 && (got[0] + 20.0).abs() <= 2.0 && (got[1] - 36.0).abs() <= 2.0;
    let err = rel_err(&y, &matmul_ref(&x, snaps, 2 * m, &w, 2 * angles.len()));
    report(24, "Beamforming", "2048 snapshots, 16 antennas, 91 directions", &format!("sources located at {got:?} degrees (true -20 and 36)"), snaps, st, Some(err < 0.05 && ok));
    Ok(())
}

/// 25. Denoising by subspace projection: the signal's principal subspace is estimated from the noisy
/// data itself, and every frame is projected onto it on the accelerator.
fn denoising(c: &mut Client) -> Result<(), String> {
    let (d, frames) = (64usize, 4096usize);
    let freqs = [3.0f32, 7.0, 11.0];
    let mut r = Rng(25);
    let (mut clean, mut noisy) = (vec![0f32; frames * d], vec![0f32; frames * d]);
    for f in 0..frames {
        let par: Vec<(f32, f32)> = freqs.iter().map(|_| (0.5 + r.uniform(), r.uniform() * 2.0 * PI)).collect();
        for t in 0..d {
            let v: f32 = freqs.iter().zip(&par).map(|(fr, (a, ph))| a * (2.0 * PI * fr * t as f32 / d as f32 + ph).sin()).sum();
            clean[f * d + t] = v;
            noisy[f * d + t] = v + r.normal() * 0.7;
        }
    }
    let p = pca_projector(&covariance(&noisy, frames, d), d, 6);
    c.linear(p.clone(), d, d, false)?;
    let (y, st) = c.run(&noisy, frames)?;
    let snr = |v: &[f32]| { let (s, e) = clean.iter().zip(v).fold((0f32, 0f32), |(s, e), (c, x)| (s + c * c, e + (x - c) * (x - c))); 10.0 * (s / e).log10() };
    let err = rel_err(&y, &matmul_ref(&noisy, frames, d, &p, d));
    report(25, "Denoising", "4096 frames of 64 samples (synthetic)", &format!("signal-to-noise ratio raised from {:.1} dB to {:.1} dB", snr(&noisy), snr(&y)), frames, st, Some(err < 0.05));
    Ok(())
}
