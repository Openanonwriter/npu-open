//! Networking demonstrations (15-19). Models are centroids, tables or closed-form fits.
use crate::util::*;
use crate::Case;
use npu::proto::Client;
use npu::session::Layer;

pub const CASES: [Case; 5] = [(15, "net", traffic_clustering), (16, "net", first_packets), (17, "net", link_quality), (18, "net", forecasting), (19, "net", sketching)];

/// 15. Traffic clustering: k-means where the expensive step, comparing every flow with every centre,
/// runs on the accelerator (argmax of x.c - |c|^2/2 picks the nearest centre). Centres are updated as means.
fn traffic_clustering(c: &mut Client) -> Result<(), String> {
    let (b, nb, d) = load_rows("ids/cap/benign.f32")?;
    let (a, na, _) = load_rows("ids/cap/attack.f32")?;
    let (mean, sd) = mean_std(&b, nb, d);
    let x: Vec<f32> = [standardise(&b, d, &mean, &sd), standardise(&a, d, &mean, &sd)].concat();
    let (n, k) = (nb + na, 6);
    let mut r = Rng(15);
    let mut centres: Vec<f32> = (0..k).flat_map(|_| { let i = (r.next() % n as u64) as usize; x[i * d..(i + 1) * d].to_vec() }).collect();
    let (mut assign, mut st) = (vec![0usize; n], Default::default());
    for _ in 0..8 {
        let bias: Vec<f32> = (0..k).map(|j| -0.5 * centres[j * d..(j + 1) * d].iter().map(|v| v * v).sum::<f32>()).collect();
        c.define(&[Layer { n_in: d, n_out: k, relu: false, w: centres.clone(), b: bias }])?;
        let (y, s) = c.run(&x, n)?;
        st = s;
        assign = argmax_rows(&y, k).iter().map(|p| p.0).collect();
        let (mut sum, mut cnt) = (vec![0f32; k * d], vec![0f32; k]);
        for i in 0..n {
            cnt[assign[i]] += 1.0;
            (0..d).for_each(|j| sum[assign[i] * d + j] += x[i * d + j]);
        }
        for j in 0..k {
            if cnt[j] > 0.0 {
                (0..d).for_each(|q| centres[j * d + q] = sum[j * d + q] / cnt[j]);
            }
        }
    }
    let attack_in: Vec<usize> = (0..k).map(|j| (nb..n).filter(|&i| assign[i] == j).count()).collect();
    let size: Vec<usize> = (0..k).map(|j| assign.iter().filter(|&&q| q == j).count()).collect();
    let pure: usize = (0..k).map(|j| attack_in[j].max(size[j] - attack_in[j])).sum();
    report(15, "Traffic clustering", &format!("{n} flows, 6 centres, 8 rounds"), &format!("{:.1}% of flows in clusters that are all-scan or all-baseline", 100.0 * pure as f32 / n as f32), n, st, None);
    Ok(())
}

/// 16. First-packet signatures: the first eight packet sizes and directions of a connection looked up
/// against a table of known signatures.
fn first_packets(c: &mut Client) -> Result<(), String> {
    let (n_sig, d, rows) = (64, 16, 4000);
    let mut r = Rng(16);
    let unit = |v: &[f32]| -> Vec<f32> { let n = v.iter().map(|a| a * a).sum::<f32>().sqrt().max(1e-6); v.iter().map(|a| a / n).collect() };
    let sigs: Vec<Vec<f32>> = (0..n_sig).map(|_| (0..d).map(|i| if i < 8 { 60.0 + (r.next() % 1400) as f32 } else { if r.next() % 2 == 0 { 700.0 } else { -700.0 } }).collect()).collect();
    let w: Vec<f32> = sigs.iter().flat_map(|s| unit(s)).collect();
    let (mut x, mut labels) = (Vec::new(), Vec::new());
    for _ in 0..rows {
        let s = (r.next() % n_sig as u64) as usize;
        let noisy: Vec<f32> = sigs[s].iter().enumerate().map(|(i, v)| if i < 8 { v + r.normal() * 40.0 } else { *v }).collect();
        x.extend(unit(&noisy));
        labels.push(s);
    }
    c.linear(w.clone(), n_sig, d, false)?;
    let (y, st) = c.run(&x, rows)?;
    let right = argmax_rows(&y, n_sig).iter().zip(&labels).filter(|(p, l)| p.0 == **l).count();
    let err = rel_err(&y, &matmul_ref(&x, rows, d, &w, n_sig));
    report(16, "First-packet signatures", "4000 connections against 64 signatures", &format!("{:.1}% matched to the right signature", 100.0 * right as f32 / rows as f32), rows, st, Some(err < 0.05));
    Ok(())
}

/// 17. Link quality prediction. Synthetic data: throughput is a made-up function of signal, noise,
/// retries and channel load. Closed-form least squares, predictions on the accelerator.
fn link_quality(c: &mut Client) -> Result<(), String> {
    let (n_fit, rows, d) = (3000, 8192, 8);
    let mut r = Rng(17);
    let sample = |r: &mut Rng| -> (Vec<f32>, f32) {
        let (sig, snr, retry, load) = (r.uniform(), r.uniform(), r.uniform(), r.uniform());
        let y = (2.0 * snr + sig - 1.5 * retry * load - 0.8 * load * load + r.normal() * 0.05).max(0.0);
        (vec![1.0, sig, snr, retry, load, load * load, retry * load, snr * snr], y)
    };
    let (mut xtx, mut xty) = (vec![0f64; d * d], vec![0f64; d]);
    for _ in 0..n_fit {
        let (f, y) = sample(&mut r);
        for i in 0..d {
            xty[i] += f[i] as f64 * y as f64;
            (0..d).for_each(|j| xtx[i * d + j] += f[i] as f64 * f[j] as f64);
        }
    }
    let w: Vec<f32> = solve_spd(&xtx, d, &xty, 1, 1e-6).iter().map(|&v| v as f32).collect();
    let (mut x, mut truth) = (Vec::new(), Vec::new());
    for _ in 0..rows {
        let (f, y) = sample(&mut r);
        x.extend(f);
        truth.push(y);
    }
    c.linear(w.clone(), 1, d, false)?;
    let (y, st) = c.run(&x, rows)?;
    let mean = truth.iter().sum::<f32>() / rows as f32;
    let r2 = 1.0 - y.iter().zip(&truth).map(|(a, b)| (a - b).powi(2)).sum::<f32>() / truth.iter().map(|b| (b - mean).powi(2)).sum::<f32>();
    let err = rel_err(&y, &matmul_ref(&x, rows, d, &w, 1));
    report(17, "Link quality prediction", "8192 samples, 8 features (synthetic)", &format!("R squared {r2:.4} on unseen samples"), rows, st, Some(err < 0.05));
    Ok(())
}

/// 18. Latency forecasting. Synthetic series (two rhythms plus bursts). A 16-tap linear predictor is
/// fitted in closed form; every one-step forecast is a row on the accelerator.
fn forecasting(c: &mut Client) -> Result<(), String> {
    let (len, taps, n_fit) = (12000usize, 16usize, 6000usize);
    let mut r = Rng(18);
    let mut burst = 0f32;
    let series: Vec<f32> = (0..len).map(|t| { burst = burst * 0.9 + if r.next() % 97 == 0 { 2.0 } else { 0.0 }; 1.0 + 0.5 * (t as f32 * 0.07).sin() + 0.3 * (t as f32 * 0.31).sin() + burst + r.normal() * 0.05 }).collect();
    let (mut xtx, mut xty) = (vec![0f64; taps * taps], vec![0f64; taps]);
    for t in taps..n_fit {
        for i in 0..taps {
            xty[i] += series[t - taps + i] as f64 * series[t] as f64;
            (0..taps).for_each(|j| xtx[i * taps + j] += series[t - taps + i] as f64 * series[t - taps + j] as f64);
        }
    }
    let w: Vec<f32> = solve_spd(&xtx, taps, &xty, 1, 1e-3).iter().map(|&v| v as f32).collect();
    let rows = len - n_fit;
    let x: Vec<f32> = (n_fit..len).flat_map(|t| series[t - taps..t].to_vec()).collect();
    c.linear(w.clone(), 1, taps, false)?;
    let (y, st) = c.run(&x, rows)?;
    let rms = |f: &dyn Fn(usize) -> f32| ((0..rows).map(|i| (f(i) - series[n_fit + i]).powi(2)).sum::<f32>() / rows as f32).sqrt();
    let (model, naive) = (rms(&|i| y[i]), rms(&|i| series[n_fit + i - 1]));
    let err = rel_err(&y, &matmul_ref(&x, rows, taps, &w, 1));
    report(18, "Latency forecasting", "6000 one-step forecasts, 16 taps (synthetic)", &format!("RMS error {model:.3} against {naive:.3} for repeating the last value"), rows, st, Some(err < 0.05));
    Ok(())
}

/// 19. Telemetry sketching: flow records squeezed from 32 numbers to 8 with a random matrix, keeping
/// distances between records roughly intact.
fn sketching(c: &mut Client) -> Result<(), String> {
    let (b, nb, d) = load_rows("ids/cap/benign.f32")?;
    let (a, na, _) = load_rows("ids/cap/attack.f32")?;
    let (mean, sd) = mean_std(&b, nb, d);
    let x: Vec<f32> = [standardise(&b, d, &mean, &sd), standardise(&a, d, &mean, &sd)].concat();
    let (n, k) = (nb + na, 8);
    let mut r = Rng(19);
    let w = r.normals(k * d, 1.0 / (k as f32).sqrt());
    c.linear(w.clone(), k, d, false)?;
    let (y, st) = c.run(&x, n)?;
    let dist = |v: &[f32], w: usize, p: usize, q: usize| (0..w).map(|j| (v[p * w + j] - v[q * w + j]).powi(2)).sum::<f32>().sqrt();
    let (mut ratio, mut pairs) = (0f32, 0);
    for _ in 0..2000 {
        let (p, q) = ((r.next() % n as u64) as usize, (r.next() % n as u64) as usize);
        let full = dist(&x, d, p, q);
        if full > 0.5 {
            ratio += (dist(&y, k, p, q) / full - 1.0).abs();
            pairs += 1;
        }
    }
    let err = rel_err(&y, &matmul_ref(&x, n, d, &w, k));
    report(19, "Telemetry sketching", &format!("{n} flow records, 32 numbers to 8"), &format!("distances between records kept within {:.0}% on average", 100.0 * ratio / pairs as f32), n, st, Some(err < 0.05));
    Ok(())
}
