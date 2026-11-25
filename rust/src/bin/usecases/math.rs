//! Math demonstrations (1-7).
use crate::util::*;
use crate::Case;
use npu::proto::Client;
use npu::session::Layer;

pub const CASES: [Case; 7] = [
    (1, "math", matmul), (2, "math", random_projection), (3, "math", pca_scoring), (4, "math", similarity_search),
    (5, "math", kernel_features), (6, "math", regression), (7, "math", surrogate),
];

/// 1. Batched matrix multiply: 4096 rows of 256 values times a 128 x 256 matrix.
fn matmul(c: &mut Client) -> Result<(), String> {
    let (rows, n_in, n_out) = (4096, 256, 128);
    let mut r = Rng(1);
    let (w, x) = (r.normals(n_out * n_in, 0.25), r.normals(rows * n_in, 1.0));
    let vtcm_kb = c.linear(w.clone(), n_out, n_in, false)?;
    if std::env::var("NPU_DEBUG").is_ok() {
        eprintln!("accelerator fast memory (VTCM) granted: {vtcm_kb} KB");
    }
    let (y, st) = c.run(&x, rows)?;
    let err = rel_err(&y, &matmul_ref(&x, rows, n_in, &w, n_out));
    let gmacs = (rows * n_in * n_out) as f64 / st.dsp_us.max(1) as f64 / 1e3;
    report(1, "Batched matrix multiply", "4096x256 times 256x128", &format!("{gmacs:.1} billion multiply-adds/s, max error {:.2}% of RMS", 100.0 * err), rows, st, Some(err < 0.02));
    Ok(())
}

/// 2. Random projection: 1024 dimensions down to 64 with a Gaussian matrix; distances are roughly preserved.
fn random_projection(c: &mut Client) -> Result<(), String> {
    let (rows, n_in, n_out) = (1024, 1024, 64);
    let mut r = Rng(2);
    let (w, x) = (r.normals(n_out * n_in, 1.0 / (n_out as f32).sqrt()), r.normals(rows * n_in, 0.25));
    c.linear(w.clone(), n_out, n_in, false)?;
    let (y, st) = c.run(&x, rows)?;
    let dist = |v: &[f32], n: usize, a: usize, b: usize| (0..n).map(|k| (v[a * n + k] - v[b * n + k]).powi(2)).sum::<f32>().sqrt();
    let mut distortion = 0.0;
    for _ in 0..500 {
        let (a, b) = ((r.next() % rows as u64) as usize, (r.next() % rows as u64) as usize);
        if a != b {
            distortion += (dist(&y, n_out, a, b) / dist(&x, n_in, a, b) - 1.0).abs() / 500.0;
        }
    }
    let err = rel_err(&y, &matmul_ref(&x, rows, n_in, &w, n_out));
    report(2, "Random projection", "1024 dims to 64, 1024 vectors", &format!("pairwise distances kept within {:.1}% on average", 100.0 * distortion), rows, st, Some(err < 0.02));
    Ok(())
}

/// 3. PCA scoring: components fitted once on the CPU (Jacobi), every row projected on the accelerator.
fn pca_scoring(c: &mut Client) -> Result<(), String> {
    let (rows, d, latent, keep) = (4096, 64, 8, 8);
    let mut r = Rng(3);
    let mix = r.normals(latent * d, 1.0);
    let mut x = vec![0f32; rows * d];
    for row in 0..rows {
        let z = r.normals(latent, 1.0);
        for k in 0..d {
            x[row * d + k] = (0..latent).map(|l| z[l] * mix[l * d + k]).sum::<f32>() * 0.3 + r.normal() * 0.05 + 0.5;
        }
    }
    let mean: Vec<f64> = (0..d).map(|k| (0..rows).map(|row| x[row * d + k] as f64).sum::<f64>() / rows as f64).collect();
    let mut cov = vec![0f64; d * d];
    for row in 0..rows {
        for i in 0..d {
            let xi = x[row * d + i] as f64 - mean[i];
            for j in i..d {
                cov[i * d + j] += xi * (x[row * d + j] as f64 - mean[j]) / rows as f64;
            }
        }
    }
    for i in 0..d {
        for j in 0..i {
            cov[i * d + j] = cov[j * d + i];
        }
    }
    let (vals, vecs) = jacobi_eigen(&cov, d);
    let w: Vec<f32> = vecs[..keep * d].iter().map(|&v| v as f32).collect();
    let b: Vec<f32> = (0..keep).map(|o| -(0..d).map(|k| w[o * d + k] as f64 * mean[k]).sum::<f64>() as f32).collect();
    c.define(&[Layer { n_in: d, n_out: keep, relu: false, w: w.clone(), b: b.clone() }])?;
    let (y, st) = c.run(&x, rows)?;
    let captured: f64 = (0..keep).map(|o| (0..rows).map(|row| (y[row * keep + o] as f64).powi(2)).sum::<f64>() / rows as f64).sum::<f64>() / vals.iter().sum::<f64>();
    let mut yref = matmul_ref(&x, rows, d, &w, keep);
    for row in 0..rows {
        for o in 0..keep {
            yref[row * keep + o] += b[o];
        }
    }
    let err = rel_err(&y, &yref);
    report(3, "PCA scoring", "4096 rows, 64 dims to 8 components", &format!("{:.1}% of the variance captured", 100.0 * captured), rows, st, Some(err < 0.03));
    Ok(())
}

/// 4. Similarity search: every line of the technical record is a row of the weight matrix; a query is one multiply.
fn similarity_search(c: &mut Client) -> Result<(), String> {
    let dim = 256;
    let path = std::path::PathBuf::from(std::env::var("NPU_OPEN").unwrap_or_else(|_| ".".into())).join("docs/TECHNICAL.md");
    let text = std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    let lines: Vec<&str> = text.lines().filter(|l| l.len() > 40 && !l.starts_with('|') && !l.starts_with('#')).take(480).collect();
    let w: Vec<f32> = lines.iter().flat_map(|l| trigram_vector(l, dim)).collect();
    c.linear(w.clone(), lines.len(), dim, false)?;
    let queries = ["watchdog reset when the accelerator is locked without power", "second segment must start on a page boundary", "shell must match the firmware build", "output is clamped at zero"];
    let x: Vec<f32> = queries.iter().flat_map(|q| trigram_vector(q, dim)).collect();
    let (y, st) = c.run(&x, queries.len())?;
    let yref = matmul_ref(&x, queries.len(), dim, &w, lines.len());
    let mut agree = 0;
    let mut shown = String::new();
    for (qi, _) in queries.iter().enumerate() {
        let top = |v: &[f32]| (0..lines.len()).max_by(|&a, &b| v[qi * lines.len() + a].partial_cmp(&v[qi * lines.len() + b]).unwrap()).unwrap();
        let t = top(&y);
        agree += (t == top(&yref)) as usize;
        if qi == 0 {
            shown = lines[t].chars().take(40).collect();
        }
    }
    report(4, "Similarity search", &format!("{} queries against {} text lines", queries.len(), lines.len()), &format!("top hit for query 1: \"{shown}...\""), queries.len(), st, Some(agree == queries.len()));
    Ok(())
}

/// 5. Random-feature kernel approximation: mean of relu(w.x) relu(w.y) over random w approximates the arc-cosine kernel.
fn kernel_features(c: &mut Client) -> Result<(), String> {
    let (rows, d, feats) = (512, 32, 1024);
    let mut r = Rng(5);
    let (w, x) = (r.normals(feats * d, 1.0), r.normals(rows * d, 0.3));
    c.linear(w, feats, d, true)?;
    let (phi, st) = c.run(&x, rows)?;
    let (mut err, mut scale) = (0f64, 0f64);
    for _ in 0..300 {
        let (a, b) = ((r.next() % rows as u64) as usize, (r.next() % rows as u64) as usize);
        let dot = |u: &[f32], n: usize| (0..n).map(|k| u[a * n + k] as f64 * u[b * n + k] as f64).sum::<f64>();
        let (na, nb) = ((0..d).map(|k| (x[a * d + k] as f64).powi(2)).sum::<f64>().sqrt(), (0..d).map(|k| (x[b * d + k] as f64).powi(2)).sum::<f64>().sqrt());
        let theta = (dot(&x, d) / (na * nb)).clamp(-1.0, 1.0).acos();
        let exact = na * nb * (theta.sin() + (std::f64::consts::PI - theta) * theta.cos()) / (2.0 * std::f64::consts::PI);
        err += (dot(&phi, feats) / feats as f64 - exact).abs();
        scale += exact.abs();
    }
    report(5, "Kernel random features", "512 vectors to 1024 ReLU features", &format!("kernel values reproduced within {:.1}%", 100.0 * err / scale), rows, st, Some(err / scale < 0.1));
    Ok(())
}

/// 6. Bulk regression inference: coefficients from a closed-form least-squares fit, predictions on the accelerator.
fn regression(c: &mut Client) -> Result<(), String> {
    let (n_fit, rows, d, targets) = (2000, 8192, 32, 16);
    let mut r = Rng(6);
    let truth = r.normals(targets * d, 0.5);
    let gen = |r: &mut Rng, n: usize| {
        let x = r.normals(n * d, 1.0);
        let mut y = matmul_ref(&x, n, d, &truth, targets);
        y.iter_mut().for_each(|v| *v += r.normal() * 0.1);
        (x, y)
    };
    let (xf, yf) = gen(&mut r, n_fit);
    let (mut xtx, mut xty) = (vec![0f64; d * d], vec![0f64; d * targets]);
    for row in 0..n_fit {
        for i in 0..d {
            for j in 0..d {
                xtx[i * d + j] += xf[row * d + i] as f64 * xf[row * d + j] as f64;
            }
            for t in 0..targets {
                xty[i * targets + t] += xf[row * d + i] as f64 * yf[row * targets + t] as f64;
            }
        }
    }
    let beta = solve_spd(&xtx, d, &xty, targets, 1e-6);
    let w: Vec<f32> = (0..targets).flat_map(|t| (0..d).map(move |i| (t, i))).map(|(t, i)| beta[i * targets + t] as f32).collect();
    let (x, y_true) = gen(&mut r, rows);
    c.linear(w.clone(), targets, d, false)?;
    let (y, st) = c.run(&x, rows)?;
    let mean = y_true.iter().sum::<f32>() / y_true.len() as f32;
    let r2 = 1.0 - y.iter().zip(&y_true).map(|(a, b)| (a - b).powi(2)).sum::<f32>() / y_true.iter().map(|b| (b - mean).powi(2)).sum::<f32>();
    let err = rel_err(&y, &matmul_ref(&x, rows, d, &w, targets));
    report(6, "Regression inference", "8192 rows, 32 inputs, 16 targets", &format!("R squared {r2:.4} against held-out targets"), rows, st, Some(err < 0.02));
    Ok(())
}

/// 7. Surrogate model: a two-layer network (random ReLU features + closed-form readout) standing in for a function.
fn surrogate(c: &mut Client) -> Result<(), String> {
    let (n_fit, rows, d, feats) = (4000, 4096, 3, 256);
    let f = |x: &[f32]| (3.0 * x[0]).sin() + x[1] * x[2];
    let mut r = Rng(7);
    let (w1, b1): (Vec<f32>, Vec<f32>) = (r.normals(feats * d, 1.5), (0..feats).map(|_| r.uniform() * 2.0 - 1.0).collect());
    let phi = |x: &[f32]| -> Vec<f64> { (0..feats).map(|k| ((0..d).map(|i| w1[k * d + i] * x[i]).sum::<f32>() + b1[k]).max(0.0) as f64).collect() };
    let (mut pp, mut py) = (vec![0f64; feats * feats], vec![0f64; feats]);
    for _ in 0..n_fit {
        let x: Vec<f32> = (0..d).map(|_| r.uniform() * 2.0 - 1.0).collect();
        let p = phi(&x);
        let y = f(&x) as f64;
        for i in 0..feats {
            py[i] += p[i] * y;
            for j in 0..feats {
                pp[i * feats + j] += p[i] * p[j];
            }
        }
    }
    let w2: Vec<f32> = solve_spd(&pp, feats, &py, 1, 1e-2 * n_fit as f64).iter().map(|&v| v as f32).collect();
    let x: Vec<f32> = (0..rows * d).map(|_| r.uniform() * 2.0 - 1.0).collect();
    c.define(&[Layer { n_in: d, n_out: feats, relu: true, w: w1.clone(), b: b1.clone() }, Layer { n_in: feats, n_out: 1, relu: false, w: w2.clone(), b: vec![0.0] }])?;
    let (y, st) = c.run(&x, rows)?;
    let (mut se, mut se_ref) = (0f64, 0f64);
    for row in 0..rows {
        let xr = &x[row * d..(row + 1) * d];
        let exact = f(xr) as f64;
        let cpu: f64 = phi(xr).iter().zip(&w2).map(|(p, w)| p * *w as f64).sum();
        se += (y[row] as f64 - exact).powi(2);
        se_ref += (y[row] as f64 - cpu).powi(2);
    }
    let (rmse, dev) = ((se / rows as f64).sqrt(), (se_ref / rows as f64).sqrt());
    report(7, "Function surrogate", "2 layers: 3 to 256 ReLU to 1", &format!("RMS error {rmse:.3} against sin(3a)+b*c"), rows, st, Some(dev < 0.05));
    Ok(())
}
