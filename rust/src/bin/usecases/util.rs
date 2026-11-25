//! Shared helpers for the demonstrations: random numbers, small dense linear algebra, checking, reporting.
use npu::session::Stats;

/// xorshift64* generator: deterministic, no dependencies.
pub struct Rng(pub u64);
impl Rng {
    pub fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    pub fn uniform(&mut self) -> f32 {
        ((self.next() >> 40) as f32 + 0.5) / (1u64 << 24) as f32
    }
    /// Standard normal (Box-Muller).
    pub fn normal(&mut self) -> f32 {
        let (u, v) = (self.uniform(), self.uniform());
        (-2.0 * u.ln()).sqrt() * (2.0 * std::f32::consts::PI * v).cos()
    }
    pub fn normals(&mut self, n: usize, scale: f32) -> Vec<f32> {
        (0..n).map(|_| self.normal() * scale).collect()
    }
}

/// Reference `y = x W^T` on the CPU (used only to verify the accelerator's output).
pub fn matmul_ref(x: &[f32], rows: usize, n_in: usize, w: &[f32], n_out: usize) -> Vec<f32> {
    let mut y = vec![0f32; rows * n_out];
    for r in 0..rows {
        for o in 0..n_out {
            y[r * n_out + o] = x[r * n_in..(r + 1) * n_in].iter().zip(&w[o * n_in..(o + 1) * n_in]).map(|(a, b)| a * b).sum();
        }
    }
    y
}

/// Largest absolute difference relative to the reference's RMS value.
pub fn rel_err(y: &[f32], yref: &[f32]) -> f32 {
    let rms = (yref.iter().map(|v| v * v).sum::<f32>() / yref.len().max(1) as f32).sqrt().max(1e-6);
    y.iter().zip(yref).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max) / rms
}

/// Eigen-decomposition of a symmetric n x n matrix by cyclic Jacobi rotations.
/// Returns (eigenvalues, eigenvectors as rows), sorted by decreasing eigenvalue.
pub fn jacobi_eigen(a: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut a = a.to_vec();
    let mut v = vec![0f64; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }
    for _ in 0..60 {
        let off: f64 = (0..n).flat_map(|i| (0..n).filter(move |&j| j != i).map(move |j| (i, j))).map(|(i, j)| a[i * n + j] * a[i * n + j]).sum();
        if off < 1e-18 {
            break;
        }
        for p in 0..n {
            for q in p + 1..n {
                if a[p * n + q].abs() < 1e-300 {
                    continue;
                }
                let theta = (a[q * n + q] - a[p * n + p]) / (2.0 * a[p * n + q]);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let (c, s) = (1.0 / (t * t + 1.0).sqrt(), t / (t * t + 1.0).sqrt());
                for k in 0..n {
                    let (akp, akq) = (a[k * n + p], a[k * n + q]);
                    a[k * n + p] = c * akp - s * akq;
                    a[k * n + q] = s * akp + c * akq;
                }
                for k in 0..n {
                    let (apk, aqk) = (a[p * n + k], a[q * n + k]);
                    a[p * n + k] = c * apk - s * aqk;
                    a[q * n + k] = s * apk + c * aqk;
                    let (vpk, vqk) = (v[p * n + k], v[q * n + k]);
                    v[p * n + k] = c * vpk - s * vqk;
                    v[q * n + k] = s * vpk + c * vqk;
                }
            }
        }
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| a[j * n + j].partial_cmp(&a[i * n + i]).unwrap());
    (order.iter().map(|&i| a[i * n + i]).collect(), order.iter().flat_map(|&i| v[i * n..(i + 1) * n].to_vec()).collect())
}

/// Solves (A + ridge I) X = B for symmetric positive definite A (n x n) and B with m columns, by Cholesky.
pub fn solve_spd(a: &[f64], n: usize, b: &[f64], m: usize, ridge: f64) -> Vec<f64> {
    let mut l = a.to_vec();
    for i in 0..n {
        l[i * n + i] += ridge;
    }
    for j in 0..n {
        for k in 0..j {
            let ljk = l[j * n + k];
            for i in j..n {
                l[i * n + j] -= l[i * n + k] * ljk;
            }
        }
        let d = l[j * n + j].max(1e-12).sqrt();
        for i in j..n {
            l[i * n + j] /= d;
        }
    }
    let mut x = b.to_vec();
    for c in 0..m {
        for i in 0..n {
            let mut s = x[i * m + c];
            for k in 0..i {
                s -= l[i * n + k] * x[k * m + c];
            }
            x[i * m + c] = s / l[i * n + i];
        }
        for i in (0..n).rev() {
            let mut s = x[i * m + c];
            for k in i + 1..n {
                s -= l[k * n + i] * x[k * m + c];
            }
            x[i * m + c] = s / l[i * n + i];
        }
    }
    x
}

/// Hashes character trigrams of `text` into `dim` buckets and L2-normalises (a simple text fingerprint).
pub fn trigram_vector(text: &str, dim: usize) -> Vec<f32> {
    let b: Vec<u8> = format!("  {}  ", text.to_lowercase()).into_bytes();
    let mut v = vec![0f32; dim];
    for w in b.windows(3) {
        let h = (w[0] as u32).wrapping_mul(0x9E37_79B1) ^ (w[1] as u32).wrapping_mul(0x85EB_CA6B) ^ (w[2] as u32).wrapping_mul(0xC2B2_AE35);
        v[(h.wrapping_mul(0x27D4_EB2F) >> 16) as usize % dim] += 1.0;
    }
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-6);
    v.iter_mut().for_each(|x| *x /= norm);
    v
}

/// One result line: id, name, what the accelerator did, the outcome, its time, and whether it was verified.
pub fn report(id: u32, name: &str, did: &str, outcome: &str, rows: usize, st: Stats, verified: Option<bool>) {
    let v = match verified {
        Some(true) => "verified",
        Some(false) => "CHECK FAILED",
        None => "not checked",
    };
    println!("{id:>2} | {name:<28} | {did:<34} | {outcome:<52} | {:>7.2} us/row NPU, {:>7.2} us/row total | {v}",
        st.dsp_us as f64 / rows.max(1) as f64, (st.wall_us + st.pack_us) as f64 / rows.max(1) as f64);
}

/// Flow feature rows written by `ids/ids.py export` (u32 rows, u32 width, float32 data).
pub fn load_rows(rel: &str) -> Result<(Vec<f32>, usize, usize), String> {
    let path = std::path::PathBuf::from(std::env::var("NPU_OPEN").unwrap_or_else(|_| ".".into())).join(rel);
    let b = std::fs::read(&path).map_err(|e| format!("{} missing ({e}); create it with ids/ids.py export, see docs/SETUP.md", path.display()))?;
    let (n, d) = (u32::from_le_bytes(b[0..4].try_into().unwrap()) as usize, u32::from_le_bytes(b[4..8].try_into().unwrap()) as usize);
    Ok((b[8..8 + 4 * n * d].chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect(), n, d))
}

/// Column means and standard deviations of `x` (n rows, d columns).
pub fn mean_std(x: &[f32], n: usize, d: usize) -> (Vec<f32>, Vec<f32>) {
    let mean: Vec<f32> = (0..d).map(|k| (0..n).map(|r| x[r * d + k]).sum::<f32>() / n as f32).collect();
    let sd = (0..d).map(|k| ((0..n).map(|r| (x[r * d + k] - mean[k]).powi(2)).sum::<f32>() / n as f32).sqrt() + 1e-3).collect();
    (mean, sd)
}

/// (x - mean) / sd, clipped to +-8.
pub fn standardise(x: &[f32], d: usize, mean: &[f32], sd: &[f32]) -> Vec<f32> {
    x.iter().enumerate().map(|(i, v)| ((v - mean[i % d]) / sd[i % d]).clamp(-8.0, 8.0)).collect()
}

/// Covariance of zero-mean rows, as f64 (n rows, d columns).
pub fn covariance(x: &[f32], n: usize, d: usize) -> Vec<f64> {
    let mut c = vec![0f64; d * d];
    for r in 0..n {
        for i in 0..d {
            let xi = x[r * d + i] as f64;
            for j in 0..d {
                c[i * d + j] += xi * x[r * d + j] as f64 / n as f64;
            }
        }
    }
    c
}

/// Projector onto the top `k` principal directions: P = V_k^T V_k (d x d, row-major f32).
pub fn pca_projector(cov: &[f64], d: usize, k: usize) -> Vec<f32> {
    let (_, vecs) = jacobi_eigen(cov, d);
    let mut p = vec![0f32; d * d];
    for c in 0..k {
        for i in 0..d {
            for j in 0..d {
                p[i * d + j] += (vecs[c * d + i] * vecs[c * d + j]) as f32;
            }
        }
    }
    p
}

/// Mean squared difference between each row of `x` and of `y`.
pub fn row_mse(x: &[f32], y: &[f32], d: usize) -> Vec<f32> {
    x.chunks_exact(d).zip(y.chunks_exact(d)).map(|(a, b)| a.iter().zip(b).map(|(p, q)| (p - q).powi(2)).sum::<f32>() / d as f32).collect()
}

pub fn quantile(v: &[f32], q: f32) -> f32 {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    s[((s.len() - 1) as f32 * q) as usize]
}

/// Largest value in each row of `y` and its column.
pub fn argmax_rows(y: &[f32], width: usize) -> Vec<(usize, f32)> {
    y.chunks_exact(width).map(|r| r.iter().cloned().enumerate().fold((0, f32::MIN), |m, (i, v)| if v > m.1 { (i, v) } else { m })).collect()
}

/// Row file plus its one-byte-per-row label file (`.u8` beside the `.f32`), from `scripts/prep_real.py`.
pub fn load_labelled(rel: &str) -> Result<(Vec<f32>, Vec<u8>, usize, usize), String> {
    let (x, n, d) = load_rows(rel)?;
    let p = std::path::PathBuf::from(std::env::var("NPU_OPEN").unwrap_or_else(|_| ".".into())).join(rel).with_extension("u8");
    let y = std::fs::read(&p).map_err(|e| format!("{}: {e}", p.display()))?;
    Ok((x, y, n, d))
}

pub fn data_path(rel: &str) -> std::path::PathBuf {
    std::path::PathBuf::from(std::env::var("NPU_OPEN").unwrap_or_else(|_| ".".into())).join(rel)
}

/// Rows of `x` whose label satisfies `keep`, at most `max` of them, evenly spaced.
pub fn select(x: &[f32], y: &[u8], d: usize, max: usize, keep: impl Fn(u8) -> bool) -> Vec<f32> {
    let idx: Vec<usize> = (0..y.len()).filter(|&i| keep(y[i])).collect();
    let step = (idx.len() / max.max(1)).max(1);
    idx.iter().step_by(step).take(max).flat_map(|&i| x[i * d..(i + 1) * d].to_vec()).collect()
}
