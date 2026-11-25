//! Cyber defence demonstrations (8-14). Nothing here is trained by gradient descent.
use crate::util::*;
use crate::Case;
use npu::proto::Client;

pub const CASES: [Case; 7] = [
    (8, "cyber", intrusion_pca), (9, "cyber", log_anomaly), (10, "cyber", lookalike_domains), (11, "cyber", file_triage),
    (12, "cyber", generated_domains), (13, "cyber", size_sequences), (14, "cyber", behaviour_baseline),
];

/// 8. Intrusion scoring without training: project each flow onto the baseline's principal subspace
/// and back (one 32x32 matrix on the accelerator); flows that do not fit the baseline reconstruct badly.
fn intrusion_pca(c: &mut Client) -> Result<(), String> {
    let (b, nb, d) = load_rows("ids/cap/benign.f32")?;
    let (a, na, _) = load_rows("ids/cap/attack.f32")?;
    let half = nb / 2;
    let (mean, sd) = mean_std(&b[..half * d], half, d);
    let (fit, held, att) = (standardise(&b[..half * d], d, &mean, &sd), standardise(&b[half * d..], d, &mean, &sd), standardise(&a, d, &mean, &sd));
    let p = pca_projector(&covariance(&fit, half, d), d, 8);
    c.linear(p.clone(), d, d, false)?;
    let x: Vec<f32> = [fit.as_slice(), held.as_slice(), att.as_slice()].concat();
    let rows = nb + na;
    let (y, st) = c.run(&x, rows)?;
    let score = row_mse(&x, &y, d);
    let thr = quantile(&score[..half], 0.99);
    let fa = score[half..nb].iter().filter(|&&s| s > thr).count() as f32 / (nb - half) as f32;
    let det = score[nb..].iter().filter(|&&s| s > thr).count() as f32 / na as f32;
    let err = rel_err(&y, &matmul_ref(&x, rows, d, &p, d));
    report(8, "Intrusion scoring (PCA)", &format!("{rows} flow records, 32 features"), &format!("{:.1}% of scan flows flagged, {:.1}% false alarms", 100.0 * det, 100.0 * fa), rows, st, Some(err < 0.05));
    Ok(())
}

/// 9. Log anomaly: kernel log lines become trigram fingerprints; lines unlike the bulk reconstruct badly.
fn log_anomaly(c: &mut Client) -> Result<(), String> {
    let out = std::process::Command::new("journalctl").args(["-k", "-b", "--no-pager", "-o", "cat"]).output().map_err(|e| format!("journalctl: {e}"))?;
    let text = String::from_utf8_lossy(&out.stdout);
    let lines: Vec<&str> = text.lines().filter(|l| l.len() > 20).take(4096).collect();
    if lines.len() < 100 {
        return Err("too few kernel log lines readable".into());
    }
    let (d, n) = (128, lines.len());
    let raw: Vec<f32> = lines.iter().flat_map(|l| trigram_vector(l, d)).collect();
    let (mean, _) = mean_std(&raw, n, d);
    let x: Vec<f32> = raw.iter().enumerate().map(|(i, v)| v - mean[i % d]).collect();
    let p = pca_projector(&covariance(&x, n, d), d, 24);
    c.linear(p.clone(), d, d, false)?;
    let (y, st) = c.run(&x, n)?;
    let score = row_mse(&x, &y, d);
    let thr = quantile(&score, 0.99);
    let worst = (0..n).max_by(|&a, &b| score[a].partial_cmp(&score[b]).unwrap()).unwrap();
    let err = rel_err(&y, &matmul_ref(&x, n, d, &p, d));
    let shown: String = lines[worst].chars().take(34).collect();
    report(9, "Log anomaly", &format!("{n} kernel log lines"), &format!("{} unusual lines; oddest: \"{shown}\"", score.iter().filter(|&&s| s > thr).count()), n, st, Some(err < 0.05));
    Ok(())
}

/// 10. Lookalike domains: each protected name is a row of the weight matrix; a candidate that is very
/// similar to one of them without being identical is a likely typosquat.
fn lookalike_domains(c: &mut Client) -> Result<(), String> {
    let brands = ["paypal.com", "microsoft.com", "google.com", "amazon.com", "apple.com", "github.com", "netflix.com", "facebook.com", "linkedin.com",
        "dropbox.com", "protonmail.com", "wikipedia.org", "cloudflare.com", "fedoraproject.org", "mozilla.org", "kernel.org"];
    let candidates = ["paypa1.com", "rnicrosoft.com", "gooogle.com", "arnazon.com", "github.co", "netfl1x.com", "linkedln.com", "protonmai1.com",
        "paypal.com", "kernel.org", "example.org", "weather.gov", "openstreetmap.org", "rust-lang.org", "python.org", "debian.org"];
    let dim = 256;
    // Fingerprint only the label before the last dot: a shared ".com" or ".org" must not count as resemblance
    // (with whole names, three unrelated .org sites scored above the threshold).
    let label = |n: &str| n.rsplit_once('.').map(|(l, _)| l.to_string()).unwrap_or_else(|| n.to_string());
    let w: Vec<f32> = brands.iter().flat_map(|b| trigram_vector(&label(b), dim)).collect();
    c.linear(w.clone(), brands.len(), dim, false)?;
    let x: Vec<f32> = candidates.iter().flat_map(|q| trigram_vector(&label(q), dim)).collect();
    let (y, st) = c.run(&x, candidates.len())?;
    let best = argmax_rows(&y, brands.len());
    let flagged: Vec<usize> = (0..candidates.len()).filter(|&i| best[i].1 > 0.4 && candidates[i] != brands[best[i].0]).collect();
    let correct = flagged.iter().filter(|&&i| i < 8).count() == 8 && flagged.len() == 8;
    let err = rel_err(&y, &matmul_ref(&x, candidates.len(), dim, &w, brands.len()));
    report(10, "Lookalike domains", &format!("{} names against {} protected", candidates.len(), brands.len()),
        &format!("{} flagged, e.g. {} resembles {}", flagged.len(), candidates[flagged[0]], brands[best[flagged[0]].0]), candidates.len(), st, Some(err < 0.05 && correct));
    Ok(())
}

/// 11. File triage: byte histograms of files compared with per-type centroids (means, not training).
fn file_triage(c: &mut Client) -> Result<(), String> {
    let hist = |path: &std::path::Path| -> Option<(usize, Vec<f32>)> {
        let data = std::fs::read(path).ok()?;
        if data.len() < 2048 {
            return None;
        }
        let head = &data[..data.len().min(65536)];
        let class = if head.starts_with(b"\x7fELF") { 0 } else if head.starts_with(b"\x1f\x8b") || head.starts_with(b"\xfd7zXZ") { 2 } else if head.iter().all(|&b| b == 9 || b == 10 || (32..127).contains(&b)) { 1 } else { return None };
        let mut h = vec![0f32; 256];
        head.iter().for_each(|&b| h[b as usize] += 1.0);
        let norm = h.iter().map(|v| v * v).sum::<f32>().sqrt();
        Some((class, h.iter().map(|v| v / norm).collect()))
    };
    let mut per_class: Vec<Vec<Vec<f32>>> = vec![Vec::new(); 3];
    for dir in ["/usr/bin", "/usr/share/man/man1", "/usr/share/doc"] {
        let mut entries: Vec<_> = std::fs::read_dir(dir).map_err(|e| format!("{dir}: {e}"))?.filter_map(|e| e.ok()).map(|e| e.path()).collect();
        entries.sort();
        for p in entries.iter().take(1500) {
            let files: Vec<std::path::PathBuf> = if p.is_dir() { std::fs::read_dir(p).map(|d| d.filter_map(|e| e.ok()).map(|e| e.path()).take(4).collect()).unwrap_or_default() } else { vec![p.clone()] };
            for f in files {
                if let Some((class, h)) = hist(&f) {
                    if per_class[class].len() < 80 {
                        per_class[class].push(h);
                    }
                }
            }
        }
    }
    if per_class.iter().any(|v| v.len() < 20) {
        return Err(format!("not enough sample files per type: {:?}", per_class.iter().map(|v| v.len()).collect::<Vec<_>>()));
    }
    let mut w = vec![0f32; 3 * 256];
    let (mut x, mut labels) = (Vec::new(), Vec::new());
    for (class, files) in per_class.iter().enumerate() {
        let n_ref = files.len() / 2;
        for h in &files[..n_ref] {
            h.iter().enumerate().for_each(|(k, v)| w[class * 256 + k] += v / n_ref as f32);
        }
        for h in &files[n_ref..] {
            x.extend_from_slice(h);
            labels.push(class);
        }
    }
    c.linear(w.clone(), 3, 256, false)?;
    let (y, st) = c.run(&x, labels.len())?;
    let right = argmax_rows(&y, 3).iter().zip(&labels).filter(|(p, l)| p.0 == **l).count();
    let err = rel_err(&y, &matmul_ref(&x, labels.len(), 256, &w, 3));
    report(11, "File triage", &format!("{} files, byte histograms", labels.len()), &format!("{right} of {} sorted into program, text or compressed", labels.len()), labels.len(), st, Some(err < 0.05));
    Ok(())
}

/// 12. Machine-generated domain names: letter-pair statistics counted from the system word list give
/// one weight per pair; a name's average log-likelihood is a single dot product.
fn generated_domains(c: &mut Client) -> Result<(), String> {
    let words = std::fs::read_to_string("/usr/share/dict/words").map_err(|e| format!("/usr/share/dict/words: {e}"))?;
    let idx = |b: u8| if b.is_ascii_lowercase() { (b - b'a') as usize + 1 } else { 0 };
    let pairs = |s: &str| -> Vec<(usize, usize)> { let v: Vec<usize> = std::iter::once(0).chain(s.bytes().map(idx)).chain(std::iter::once(0)).collect(); v.windows(2).map(|w| (w[0], w[1])).collect() };
    let clean: Vec<&str> = words.lines().filter(|w| w.len() >= 4 && w.bytes().all(|b| b.is_ascii_lowercase())).collect();
    let mut counts = vec![1f64; 27 * 27];
    for w in clean.iter().step_by(2) {
        pairs(w).iter().for_each(|&(a, b)| counts[a * 27 + b] += 1.0);
    }
    let w: Vec<f32> = (0..27 * 27).map(|i| (counts[i] / counts[i / 27 * 27..i / 27 * 27 + 27].iter().sum::<f64>()).ln() as f32).collect();
    let feat = |s: &str| -> Vec<f32> { let p = pairs(s); let mut v = vec![0f32; 27 * 27]; p.iter().for_each(|&(a, b)| v[a * 27 + b] += 1.0 / p.len() as f32); v };
    let mut r = Rng(12);
    let held: Vec<&str> = clean.iter().skip(1).step_by(2).cloned().collect();
    let (n_each, mut x) = (500, Vec::new());
    for _ in 0..n_each {
        let name = format!("{}{}", held[(r.next() % held.len() as u64) as usize], held[(r.next() % held.len() as u64) as usize]);
        x.extend(feat(&name));
    }
    for _ in 0..n_each {
        let name: String = (0..12 + r.next() % 8).map(|_| (b'a' + (r.next() % 26) as u8) as char).collect();
        x.extend(feat(&name));
    }
    c.linear(w.clone(), 1, 27 * 27, false)?;
    let (y, st) = c.run(&x, 2 * n_each)?;
    let thr = (quantile(&y[..n_each], 0.5) + quantile(&y[n_each..], 0.5)) / 2.0;
    let right = y[..n_each].iter().filter(|&&s| s > thr).count() + y[n_each..].iter().filter(|&&s| s <= thr).count();
    let err = rel_err(&y, &matmul_ref(&x, 2 * n_each, 27 * 27, &w, 1));
    report(12, "Generated domain names", "1000 names, 729 letter-pair weights", &format!("{:.1}% of word-like and random names told apart", 100.0 * right as f32 / (2 * n_each) as f32), 2 * n_each, st, Some(err < 0.05));
    Ok(())
}

/// 13. Packet-size sequences: slide a window over a stream of packet sizes and correlate every window
/// with known patterns (a 1-D convolution run as a dense layer over windows).
fn size_sequences(c: &mut Client) -> Result<(), String> {
    let (win, n_pat, stream_len) = (16, 8, 20000);
    let mut r = Rng(13);
    let unit = |v: Vec<f32>| -> Vec<f32> { let m = v.iter().sum::<f32>() / v.len() as f32; let z: Vec<f32> = v.iter().map(|a| a - m).collect(); let n = z.iter().map(|a| a * a).sum::<f32>().sqrt().max(1e-6); z.iter().map(|a| a / n).collect() };
    let patterns: Vec<Vec<f32>> = (0..n_pat).map(|_| (0..win).map(|_| [60.0, 120.0, 517.0, 1460.0][(r.next() % 4) as usize]).collect()).collect();
    let mut stream: Vec<f32> = (0..stream_len).map(|_| 60.0 + (r.next() % 1400) as f32).collect();
    let mut truth = Vec::new();
    let mut pos = 100;
    while pos + win < stream_len {
        let p = (r.next() % n_pat as u64) as usize;
        for k in 0..win {
            stream[pos + k] = patterns[p][k] + r.normal() * 8.0;
        }
        truth.push((pos, p));
        pos += 150 + (r.next() % 200) as usize;
    }
    let w: Vec<f32> = patterns.iter().flat_map(|p| unit(p.clone())).collect();
    let rows = stream_len - win + 1;
    let x: Vec<f32> = (0..rows).flat_map(|i| unit(stream[i..i + win].to_vec())).collect();
    c.linear(w.clone(), n_pat, win, false)?;
    let (y, st) = c.run(&x, rows)?;
    let best = argmax_rows(&y, n_pat);
    let hits: Vec<usize> = (0..rows).filter(|&i| best[i].1 > 0.95).collect();
    let found = truth.iter().filter(|&&(p, k)| hits.contains(&p) && best[p].0 == k).count();
    let extra = hits.iter().filter(|&&i| !truth.iter().any(|&(p, _)| p == i)).count();
    let err = rel_err(&y, &matmul_ref(&x, rows, win, &w, n_pat));
    report(13, "Packet-size patterns", &format!("{rows} windows against {n_pat} patterns"), &format!("{found} of {} embedded patterns found, {extra} false hits", truth.len()), rows, st, Some(err < 0.05));
    Ok(())
}

/// 14. Behaviour baseline: whiten each flow with the baseline's covariance on the accelerator; the
/// squared length of the result is the Mahalanobis distance from normal behaviour.
fn behaviour_baseline(c: &mut Client) -> Result<(), String> {
    let (b, nb, d) = load_rows("ids/cap/benign.f32")?;
    let (a, na, _) = load_rows("ids/cap/attack.f32")?;
    let (mean, sd) = mean_std(&b, nb, d);
    let (xb, xa) = (standardise(&b, d, &mean, &sd), standardise(&a, d, &mean, &sd));
    let (vals, vecs) = jacobi_eigen(&covariance(&xb, nb, d), d);
    let w: Vec<f32> = (0..d).flat_map(|k| { let s = 1.0 / (vals[k].max(0.0) + 0.05).sqrt(); vecs[k * d..(k + 1) * d].iter().map(move |v| (v * s) as f32).collect::<Vec<_>>() }).collect();
    let x: Vec<f32> = [xb.as_slice(), xa.as_slice()].concat();
    c.linear(w.clone(), d, d, false)?;
    let (y, st) = c.run(&x, nb + na)?;
    let dist: Vec<f32> = y.chunks_exact(d).map(|r| r.iter().map(|v| v * v).sum::<f32>().sqrt()).collect();
    let err = rel_err(&y, &matmul_ref(&x, nb + na, d, &w, d));
    report(14, "Behaviour baseline", &format!("{} flow records whitened", nb + na), &format!("median distance {:.1} for baseline, {:.1} for scan traffic", quantile(&dist[..nb], 0.5), quantile(&dist[nb..], 0.5)), nb + na, st, Some(err < 0.05));
    Ok(())
}
