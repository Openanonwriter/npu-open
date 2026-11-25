//! The demonstrations again on real data (numbers are 100 + the original case). Data comes from
//! `scripts/prep_real.py` and public downloads listed in docs/TECHNICAL.md section 21.
use crate::util::*;
use crate::Case;
use npu::proto::Client;
use npu::session::{Layer, Stats};

pub const CASES: [Case; 7] = [
    (103, "real", mnist_centroids), (104, "real", manpage_search), (108, "real", nsl_pca), (110, "real", phishing_lookalikes),
    (112, "real", dga_real), (114, "real", nsl_whitening), (126, "real", nsl_classifier),
];

fn add(a: &mut Stats, b: Stats) {
    a.dsp_us += b.dsp_us;
    a.wall_us += b.wall_us;
    a.pack_us += b.pack_us;
    a.calls += b.calls;
}

/// 103. MNIST digits by nearest centroid: the ten class means are the weight rows (no fitting beyond averaging).
fn mnist_centroids(c: &mut Client) -> Result<(), String> {
    let (x, y, n, d) = load_labelled("data/mnist/test.f32")?;
    let half = n / 2;
    let (mut w, mut cnt) = (vec![0f32; 10 * d], [0f32; 10]);
    for i in 0..half {
        cnt[y[i] as usize] += 1.0;
        (0..d).for_each(|k| w[y[i] as usize * d + k] += x[i * d + k]);
    }
    (0..10).for_each(|j| (0..d).for_each(|k| w[j * d + k] /= cnt[j]));
    let b: Vec<f32> = (0..10).map(|j| -0.5 * w[j * d..(j + 1) * d].iter().map(|v| v * v).sum::<f32>()).collect();
    c.define(&[Layer { n_in: d, n_out: 10, relu: false, w, b }])?;
    let (out, st) = c.run(&x[half * d..], n - half)?;
    let right = argmax_rows(&out, 10).iter().zip(&y[half..]).filter(|(p, l)| p.0 == **l as usize).count();
    report(103, "MNIST nearest centroid", "5000 real digit images, 10 class means", &format!("{:.1}% correct with no trained model", 100.0 * right as f32 / (n - half) as f32), n - half, st, None);
    Ok(())
}

/// 104. Search every man-page summary on this machine. The table is larger than the accelerator's memory,
/// so it is paged through in blocks: load a block of rows as weights, score all queries, keep the best.
fn manpage_search(c: &mut Client) -> Result<(), String> {
    let text = std::fs::read_to_string(data_path("data/live/whatis.txt")).map_err(|e| format!("data/live/whatis.txt: {e} (create with: man -k . | sort -u)"))?;
    let lines: Vec<&str> = text.lines().filter(|l| l.contains(" - ")).collect();
    let (dim, block) = (256, 5120);
    let queries = ["list directory contents", "compress or expand files", "change file mode bits", "report file system disk space usage", "transfer a URL", "search for a pattern in files"];
    let x: Vec<f32> = queries.iter().flat_map(|q| trigram_vector(q, dim)).collect();
    let mut best = vec![(0usize, f32::MIN); queries.len()];
    let mut st = Stats::default();
    for (bi, chunk) in lines.chunks(block).enumerate() {
        let w: Vec<f32> = chunk.iter().flat_map(|l| trigram_vector(l.split_once(" - ").map(|p| p.1).unwrap_or(l), dim)).collect();
        c.linear(w, chunk.len(), dim, false)?;
        let (y, s) = c.run(&x, queries.len())?;
        add(&mut st, s);
        for (qi, (i, v)) in argmax_rows(&y, chunk.len()).into_iter().enumerate() {
            if v > best[qi].1 {
                best[qi] = (bi * block + i, v);
            }
        }
    }
    let names: Vec<String> = best.iter().map(|b| lines[b.0].split_whitespace().next().unwrap_or("?").to_string()).collect();
    report(104, "Man-page search", &format!("{} queries against {} real entries", queries.len(), lines.len()), &format!("top hits: {}", names.join(", ")), queries.len(), st, None);
    Ok(())
}

struct Nsl { train: Vec<f32>, ytrain: Vec<u8>, test: Vec<f32>, ytest: Vec<u8>, d: usize }

/// NSL-KDD, standardised with the statistics of normal training connections.
fn nsl() -> Result<Nsl, String> {
    let (tr, ytr, ntr, d) = load_labelled("data/nslkdd/nsl-train.f32")?;
    let (te, yte, _, _) = load_labelled("data/nslkdd/nsl-test.f32")?;
    let normal = select(&tr, &ytr, d, ntr, |l| l == 0);
    let (mean, sd) = mean_std(&normal, normal.len() / d, d);
    Ok(Nsl { train: standardise(&tr, d, &mean, &sd), ytrain: ytr, test: standardise(&te, d, &mean, &sd), ytest: yte, d })
}

fn nsl_outcome(score_ref: &[f32], score: &[f32], y: &[u8]) -> String {
    let thr = quantile(score_ref, 0.95);
    let rate = |f: &dyn Fn(u8) -> bool| { let idx: Vec<usize> = (0..y.len()).filter(|&i| f(y[i])).collect(); 100.0 * idx.iter().filter(|&&i| score[i] > thr).count() as f32 / idx.len().max(1) as f32 };
    format!("attacks flagged {:.1}% (DoS {:.0}, probe {:.0}, R2L {:.0}, U2R {:.0}), false alarms {:.1}%", rate(&|l| l != 0), rate(&|l| l == 1), rate(&|l| l == 2), rate(&|l| l == 3), rate(&|l| l == 4), rate(&|l| l == 0))
}

/// 108. Unsupervised intrusion scoring on NSL-KDD: fit the principal subspace of normal training
/// connections, score the official test set by reconstruction error. No attack is seen while fitting.
fn nsl_pca(c: &mut Client) -> Result<(), String> {
    let k = nsl()?;
    let d = k.d;
    let fit = select(&k.train, &k.ytrain, d, 20000, |l| l == 0);
    let p = pca_projector(&covariance(&fit, fit.len() / d, d), d, 10);
    c.linear(p, d, d, false)?;
    let (yf, _) = c.run(&fit, fit.len() / d)?;
    let (yt, st) = c.run(&k.test, k.ytest.len())?;
    report(108, "Intrusion scoring, NSL-KDD", &format!("{} real test connections, PCA", k.ytest.len()), &nsl_outcome(&row_mse(&fit, &yf, d), &row_mse(&k.test, &yt, d), &k.ytest), k.ytest.len(), st, None);
    Ok(())
}

/// 114. The same with whitening: distance from normal behaviour in every direction, not only off the main subspace.
fn nsl_whitening(c: &mut Client) -> Result<(), String> {
    let k = nsl()?;
    let d = k.d;
    let fit = select(&k.train, &k.ytrain, d, 20000, |l| l == 0);
    let (vals, vecs) = jacobi_eigen(&covariance(&fit, fit.len() / d, d), d);
    let w: Vec<f32> = (0..d).flat_map(|i| { let s = 1.0 / (vals[i].max(0.0) + 0.1).sqrt(); vecs[i * d..(i + 1) * d].iter().map(move |v| (v * s) as f32).collect::<Vec<_>>() }).collect();
    c.linear(w, d, d, false)?;
    let norm = |y: &[f32]| -> Vec<f32> { y.chunks_exact(d).map(|r| r.iter().map(|v| v * v).sum::<f32>().sqrt()).collect() };
    let (yf, _) = c.run(&fit, fit.len() / d)?;
    let (yt, st) = c.run(&k.test, k.ytest.len())?;
    report(114, "Behaviour baseline, NSL-KDD", &format!("{} real test connections, whitened", k.ytest.len()), &nsl_outcome(&norm(&yf), &norm(&yt), &k.ytest), k.ytest.len(), st, None);
    Ok(())
}

/// 126. Supervised, still without gradient descent: random ReLU features computed on the accelerator,
/// a ridge-regression readout solved in closed form, then both layers run on the accelerator.
fn nsl_classifier(c: &mut Client) -> Result<(), String> {
    let k = nsl()?;
    let (d, feats, n_fit) = (k.d, 512usize, 24000usize);
    let mut r = Rng(126);
    let (w1, b1): (Vec<f32>, Vec<f32>) = (r.normals(feats * d, 0.25), (0..feats).map(|_| r.uniform() - 0.5).collect());
    let step = k.ytrain.len() / n_fit;
    let idx: Vec<usize> = (0..n_fit).map(|i| i * step).collect();
    let xfit: Vec<f32> = idx.iter().flat_map(|&i| k.train[i * d..(i + 1) * d].to_vec()).collect();
    c.define(&[Layer { n_in: d, n_out: feats, relu: true, w: w1.clone(), b: b1.clone() }])?;
    let (phi, _) = c.run(&xfit, n_fit)?;
    let (mut pp, mut py) = (vec![0f64; feats * feats], vec![0f64; feats]);
    for (row, &i) in idx.iter().enumerate() {
        let p = &phi[row * feats..(row + 1) * feats];
        let t = if k.ytrain[i] == 0 { -1.0 } else { 1.0 };
        for a in 0..feats {
            if p[a] != 0.0 {
                py[a] += p[a] as f64 * t;
                let pa = p[a] as f64;
                for b in 0..feats {
                    pp[a * feats + b] += pa * p[b] as f64;
                }
            }
        }
    }
    let w2: Vec<f32> = solve_spd(&pp, feats, &py, 1, 1.0 * n_fit as f64 / 100.0).iter().map(|&v| v as f32).collect();
    c.define(&[Layer { n_in: d, n_out: feats, relu: true, w: w1, b: b1 }, Layer { n_in: feats, n_out: 1, relu: false, w: w2, b: vec![0.0] }])?;
    let (y, st) = c.run(&k.test, k.ytest.len())?;
    let n = k.ytest.len();
    let right = (0..n).filter(|&i| (y[i] > 0.0) == (k.ytest[i] != 0)).count();
    let det = (0..n).filter(|&i| k.ytest[i] != 0 && y[i] > 0.0).count() as f32 / k.ytest.iter().filter(|&&l| l != 0).count() as f32;
    let fa = (0..n).filter(|&i| k.ytest[i] == 0 && y[i] > 0.0).count() as f32 / k.ytest.iter().filter(|&&l| l == 0).count() as f32;
    report(126, "Intrusion classifier, NSL-KDD", &format!("{n} real test connections, 2 layers"), &format!("{:.1}% correct on KDDTest+ ({:.1}% of attacks caught, {:.1}% false alarms)", 100.0 * right as f32 / n as f32, 100.0 * det, 100.0 * fa), n, st, None);
    Ok(())
}

fn letter_pairs(s: &str) -> Vec<usize> {
    let idx = |b: u8| if b.is_ascii_lowercase() { (b - b'a') as usize + 1 } else { 0 };
    let v: Vec<usize> = std::iter::once(0).chain(s.bytes().map(idx)).chain(std::iter::once(0)).collect();
    v.windows(2).map(|w| w[0] * 27 + w[1]).collect()
}

/// 112. Generated domain names on a real labelled set (337k from malware families, 337k legitimate).
/// The readout over 729 letter-pair frequencies (plus length) is fitted by ridge regression in closed form.
fn dga_real(c: &mut Client) -> Result<(), String> {
    let text = std::fs::read_to_string(data_path("data/dga/dga_domains_full.csv")).map_err(|e| format!("data/dga: {e}"))?;
    let d = 27 * 27 + 3;
    let feat = |name: &str| -> Vec<(usize, f32)> {
        let label = name.split('.').next().unwrap_or(name);
        let p = letter_pairs(label);
        let mut v: Vec<(usize, f32)> = p.iter().map(|&i| (i, 1.0 / p.len() as f32)).collect();
        let digits = label.bytes().filter(|b| b.is_ascii_digit()).count() as f32;
        v.extend([(729, label.len() as f32 / 32.0), (730, digits / label.len().max(1) as f32), (731, 1.0)]);
        v
    };
    let rows: Vec<(bool, &str, &str)> = text.lines().filter_map(|l| { let mut p = l.split(','); Some((p.next()? == "dga", p.next()?, p.next()?)) }).collect();
    let (n_fit, n_test) = (60000usize, 40000usize);
    let step = rows.len() / (n_fit + n_test);
    let (mut xtx, mut xty) = (vec![0f64; d * d], vec![0f64; d]);
    for i in 0..n_fit {
        let (dga, _, name) = rows[i * step];
        let f = feat(name);
        for &(a, va) in &f {
            xty[a] += va as f64 * if dga { 1.0 } else { -1.0 };
            for &(b, vb) in &f {
                xtx[a * d + b] += va as f64 * vb as f64;
            }
        }
    }
    let w: Vec<f32> = solve_spd(&xtx, d, &xty, 1, 1.0).iter().map(|&v| v as f32).collect();
    let test: Vec<(bool, &str, &str)> = (0..n_test).map(|i| rows[(n_fit + i) * step]).collect();
    let mut x = vec![0f32; n_test * d];
    for (i, t) in test.iter().enumerate() {
        feat(t.2).iter().for_each(|&(a, v)| x[i * d + a] += v);
    }
    c.linear(w, 1, d, false)?;
    let (y, st) = c.run(&x, n_test)?;
    let right = (0..n_test).filter(|&i| (y[i] > 0.0) == test[i].0).count();
    let mut fam: std::collections::BTreeMap<&str, (usize, usize)> = Default::default();
    for (i, t) in test.iter().enumerate().filter(|(_, t)| t.0) {
        let e = fam.entry(t.1).or_default();
        e.0 += 1;
        e.1 += (y[i] > 0.0) as usize;
    }
    let mut worst: Vec<(&str, f32)> = fam.iter().filter(|(_, v)| v.0 >= 50).map(|(k, v)| (*k, 100.0 * v.1 as f32 / v.0 as f32)).collect();
    worst.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    let hard: Vec<String> = worst.iter().take(3).map(|(k, p)| format!("{k} {p:.0}%")).collect();
    report(112, "Generated domains, real set", &format!("{n_test} real names, 732 features"), &format!("{:.1}% correct; hardest families: {}", 100.0 * right as f32 / n_test as f32, hard.join(", ")), n_test, st, None);
    Ok(())
}

/// 110. Lookalike detection on a live phishing feed: every word of every hostname is compared with the
/// protected names. The same test on the most popular legitimate domains gives the false-alarm rate.
fn phishing_lookalikes(c: &mut Client) -> Result<(), String> {
    let brands = ["paypal", "microsoft", "google", "amazon", "apple", "github", "netflix", "facebook", "linkedin", "dropbox", "instagram", "whatsapp", "telegram",
        "outlook", "office", "icloud", "chase", "wellsfargo", "bankofamerica", "santander", "binance", "coinbase", "metamask", "steam", "roblox", "dhl", "fedex", "usps",
        "ebay", "alibaba", "yahoo", "adobe", "docusign", "spotify", "twitter", "tiktok"];
    let dim = 256;
    let w: Vec<f32> = brands.iter().flat_map(|b| trigram_vector(b, dim)).collect();
    c.linear(w, brands.len(), dim, false)?;
    let words = |host: &str| -> Vec<String> { host.split(|ch: char| !ch.is_ascii_alphabetic()).filter(|t| t.len() >= 4).map(|t| t.to_lowercase()).collect() };
    let host_of = |url: &str| -> String { url.split("://").nth(1).unwrap_or(url).split(['/', ':', '?']).next().unwrap_or("").to_lowercase() };
    let mut st = Stats::default();
    let flag_rate = |c: &mut Client, hosts: &[String], st: &mut Stats| -> Result<(usize, String), String> {
        let (mut x, mut owner) = (Vec::new(), Vec::new());
        for (hi, h) in hosts.iter().enumerate() {
            for t in words(h) {
                x.extend(trigram_vector(&t, dim));
                owner.push((hi, t));
            }
        }
        let (y, s) = c.run(&x, owner.len())?;
        add(st, s);
        let best = argmax_rows(&y, brands.len());
        let mut flagged = vec![false; hosts.len()];
        let mut example = String::new();
        for (i, (hi, t)) in owner.iter().enumerate() {
            // resembles a protected name, and the host is not that name's own site
            let own_site = hosts[*hi].ends_with(&format!("{}.com", brands[best[i].0])) || hosts[*hi].ends_with(&format!("{}.net", brands[best[i].0]));
            if best[i].1 > 0.55 && !own_site {
                flagged[*hi] = true;
                if example.is_empty() {
                    example = format!("\"{t}\" ~ {}", brands[best[i].0]);
                }
            }
        }
        Ok((flagged.iter().filter(|&&f| f).count(), example))
    };
    let feed = std::fs::read_to_string(data_path("data/phish/feed.txt")).map_err(|e| format!("data/phish/feed.txt: {e}"))?;
    let phish: Vec<String> = feed.lines().map(host_of).filter(|h| !h.is_empty()).collect();
    let tranco = std::fs::read_to_string(data_path("data/tranco/top-1m.csv")).map_err(|e| format!("data/tranco: {e}"))?;
    let popular: Vec<String> = tranco.lines().take(20000).filter_map(|l| l.split(',').nth(1).map(|s| s.to_string())).collect();
    let (np, example) = flag_rate(c, &phish, &mut st)?;
    let (nt, _) = flag_rate(c, &popular, &mut st)?;
    report(110, "Lookalikes, live phishing feed", &format!("{} phishing hosts, {} popular sites", phish.len(), popular.len()),
        &format!("{:.1}% of phishing hosts flagged ({example}); {:.2}% of popular sites", 100.0 * np as f32 / phish.len() as f32, 100.0 * nt as f32 / popular.len() as f32), phish.len() + popular.len(), st, None);
    Ok(())
}

/// Random ReLU features computed on the accelerator, then a ridge-regression readout solved in closed
/// form (Cholesky). Returns the two layers. `t` holds `m` target values per fitting row.
fn random_feature_net(c: &mut Client, xfit: &[f32], t: &[f32], m: usize, d: usize, feats: usize, scale: f32, ridge: f64, seed: u64) -> Result<Vec<Layer>, String> {
    let n = xfit.len() / d;
    let mut r = Rng(seed);
    let (w1, b1): (Vec<f32>, Vec<f32>) = (r.normals(feats * d, scale), (0..feats).map(|_| r.uniform() - 0.5).collect());
    let l1 = Layer { n_in: d, n_out: feats, relu: true, w: w1, b: b1 };
    c.define(std::slice::from_ref(&l1))?;
    let (phi, _) = c.run(xfit, n)?;
    let (mut pp, mut pt) = (vec![0f64; feats * feats], vec![0f64; feats * m]);
    for row in 0..n {
        let p = &phi[row * feats..(row + 1) * feats];
        let nz: Vec<usize> = (0..feats).filter(|&a| p[a] != 0.0).collect();
        for &a in &nz {
            let pa = p[a] as f64;
            (0..m).for_each(|j| pt[a * m + j] += pa * t[row * m + j] as f64);
            for &b in &nz {
                if b >= a {
                    pp[a * feats + b] += pa * p[b] as f64;
                }
            }
        }
    }
    for a in 0..feats {
        for b in 0..a {
            pp[a * feats + b] = pp[b * feats + a];
        }
    }
    let sol = solve_spd(&pp, feats, &pt, m, ridge * n as f64);
    let w2: Vec<f32> = (0..m).flat_map(|j| (0..feats).map(move |a| (j, a))).map(|(j, a)| sol[a * m + j] as f32).collect();
    Ok(vec![l1, Layer { n_in: feats, n_out: m, relu: false, w: w2, b: vec![0.0; m] }])
}

pub const CASES2: [Case; 4] = [(203, "real", mnist_features), (212, "real", dga_features), (226, "real", nsl_features), (227, "real", phishing_hosts)];

/// 203. MNIST with random features and a closed-form readout (fit on 30,000 training images).
fn mnist_features(c: &mut Client) -> Result<(), String> {
    let (xt, yt, nt, d) = load_labelled("data/mnist/train.f32")?;
    let (x, y, n, _) = load_labelled("data/mnist/test.f32")?;
    let mut t = vec![-0.1f32; nt * 10];
    (0..nt).for_each(|i| t[i * 10 + yt[i] as usize] = 1.0);
    let feats: usize = std::env::var("NPU_FEATS").ok().and_then(|v| v.parse().ok()).unwrap_or(1024);
    let net = random_feature_net(c, &xt, &t, 10, d, feats, 0.08, 1e-3, 203)?;
    c.define(&net)?;
    let (out, st) = c.run(&x, n)?;
    let right = argmax_rows(&out, 10).iter().zip(&y).filter(|(p, l)| p.0 == **l as usize).count();
    report(203, "MNIST, random features", &format!("10000 real test digits, 784-{feats}-10"), &format!("{:.2}% correct, fitted in closed form (no gradient descent)", 100.0 * right as f32 / n as f32), n, st, None);
    Ok(())
}

/// 226. NSL-KDD with 1024 random features fitted on 60,000 training connections.
fn nsl_features(c: &mut Client) -> Result<(), String> {
    let k = nsl()?;
    let (d, n_fit) = (k.d, 60000usize);
    let step = k.ytrain.len() / n_fit;
    let xfit: Vec<f32> = (0..n_fit).flat_map(|i| k.train[i * step * d..(i * step + 1) * d].to_vec()).collect();
    let t: Vec<f32> = (0..n_fit).map(|i| if k.ytrain[i * step] == 0 { -1.0 } else { 1.0 }).collect();
    let net = random_feature_net(c, &xfit, &t, 1, d, 1024, 0.2, 1e-2, 226)?;
    c.define(&net)?;
    let (y, st) = c.run(&k.test, k.ytest.len())?;
    let n = k.ytest.len();
    let right = (0..n).filter(|&i| (y[i] > 0.0) == (k.ytest[i] != 0)).count();
    let det = (0..n).filter(|&i| k.ytest[i] != 0 && y[i] > 0.0).count() as f32 / k.ytest.iter().filter(|&&l| l != 0).count() as f32;
    let fa = (0..n).filter(|&i| k.ytest[i] == 0 && y[i] > 0.0).count() as f32 / k.ytest.iter().filter(|&&l| l == 0).count() as f32;
    report(226, "NSL-KDD, 1024 random features", &format!("{n} real test connections"), &format!("{:.1}% correct on KDDTest+ ({:.1}% of attacks caught, {:.1}% false alarms)", 100.0 * right as f32 / n as f32, 100.0 * det, 100.0 * fa), n, st, None);
    Ok(())
}

/// 212. Generated domains with a nonlinear layer: letter-pair frequencies through 1024 random features.
fn dga_features(c: &mut Client) -> Result<(), String> {
    let text = std::fs::read_to_string(data_path("data/dga/dga_domains_full.csv")).map_err(|e| format!("data/dga: {e}"))?;
    let d = 27 * 27 + 3;
    let feat = |name: &str, out: &mut [f32]| {
        let label = name.split('.').next().unwrap_or(name);
        let p = letter_pairs(label);
        p.iter().for_each(|&i| out[i] += 4.0 / p.len() as f32);
        out[729] = label.len() as f32 / 16.0;
        out[730] = 4.0 * label.bytes().filter(|b| b.is_ascii_digit()).count() as f32 / label.len().max(1) as f32;
        out[731] = 1.0;
    };
    let rows: Vec<(bool, &str, &str)> = text.lines().filter_map(|l| { let mut p = l.split(','); Some((p.next()? == "dga", p.next()?, p.next()?)) }).collect();
    let (n_fit, n_test) = (40000usize, 40000usize);
    let step = rows.len() / (n_fit + n_test);
    let (mut xfit, mut x) = (vec![0f32; n_fit * d], vec![0f32; n_test * d]);
    (0..n_fit).for_each(|i| feat(rows[i * step].2, &mut xfit[i * d..(i + 1) * d]));
    (0..n_test).for_each(|i| feat(rows[(n_fit + i) * step].2, &mut x[i * d..(i + 1) * d]));
    let t: Vec<f32> = (0..n_fit).map(|i| if rows[i * step].0 { 1.0 } else { -1.0 }).collect();
    let net = random_feature_net(c, &xfit, &t, 1, d, 1024, 0.5, 1e-3, 212)?;
    c.define(&net)?;
    let (y, st) = c.run(&x, n_test)?;
    let right = (0..n_test).filter(|&i| (y[i] > 0.0) == rows[(n_fit + i) * step].0).count();
    let mut fam: std::collections::BTreeMap<&str, (usize, usize)> = Default::default();
    for i in (0..n_test).filter(|&i| rows[(n_fit + i) * step].0) {
        let e = fam.entry(rows[(n_fit + i) * step].1).or_default();
        e.0 += 1;
        e.1 += (y[i] > 0.0) as usize;
    }
    let mut worst: Vec<(&str, f32)> = fam.iter().filter(|(_, v)| v.0 >= 50).map(|(k, v)| (*k, 100.0 * v.1 as f32 / v.0 as f32)).collect();
    worst.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    let hard: Vec<String> = worst.iter().take(3).map(|(k, p)| format!("{k} {p:.0}%")).collect();
    report(212, "Generated domains, 2 layers", &format!("{n_test} real names, 732-1024-1"), &format!("{:.1}% correct; hardest families: {}", 100.0 * right as f32 / n_test as f32, hard.join(", ")), n_test, st, None);
    Ok(())
}

/// 227. Phishing hostnames scored as a whole: hashed character trigrams of the full hostname, ridge readout
/// fitted on half of the live feed and half of the popular sites, tested on the other halves.
fn phishing_hosts(c: &mut Client) -> Result<(), String> {
    let host_of = |url: &str| -> String { url.split("://").nth(1).unwrap_or(url).split(['/', ':', '?']).next().unwrap_or("").to_lowercase() };
    let feed = std::fs::read_to_string(data_path("data/phish/feed.txt")).map_err(|e| format!("data/phish/feed.txt: {e}"))?;
    let mut phish: Vec<String> = feed.lines().map(host_of).filter(|h| !h.is_empty()).collect();
    phish.sort();
    phish.dedup();
    let tranco = std::fs::read_to_string(data_path("data/tranco/top-1m.csv")).map_err(|e| format!("data/tranco: {e}"))?;
    let popular: Vec<String> = tranco.lines().take(20000).filter_map(|l| l.split(',').nth(1).map(|s| s.to_string())).collect();
    let dim = 512;
    let (mut xtx, mut xty) = (vec![0f64; dim * dim], vec![0f64; dim]);
    let mut fit = |h: &str, t: f64, weight: f64| {
        let v = trigram_vector(h, dim);
        let nz: Vec<usize> = (0..dim).filter(|&a| v[a] != 0.0).collect();
        for &a in &nz {
            xty[a] += weight * v[a] as f64 * t;
            for &b in &nz {
                xtx[a * dim + b] += weight * v[a] as f64 * v[b] as f64;
            }
        }
    };
    // even-indexed hosts fit the readout, odd-indexed ones test it; phishing rows are weighted up to balance the classes
    let balance = popular.len() as f64 / phish.len() as f64;
    phish.iter().step_by(2).for_each(|h| fit(h, 1.0, balance));
    popular.iter().step_by(2).for_each(|h| fit(h, -1.0, 1.0));
    let w: Vec<f32> = solve_spd(&xtx, dim, &xty, 1, 5.0).iter().map(|&v| v as f32).collect();
    let (tp, tn): (Vec<&String>, Vec<&String>) = (phish.iter().skip(1).step_by(2).collect(), popular.iter().skip(1).step_by(2).collect());
    let x: Vec<f32> = tp.iter().chain(tn.iter()).flat_map(|h| trigram_vector(h, dim)).collect();
    c.linear(w, 1, dim, false)?;
    let (y, st) = c.run(&x, tp.len() + tn.len())?;
    let thr = quantile(&y[tp.len()..], 0.99);
    let caught = y[..tp.len()].iter().filter(|&&s| s > thr).count();
    report(227, "Phishing host scoring", &format!("{} unseen phishing hosts, {} popular sites", tp.len(), tn.len()), &format!("{:.1}% of phishing hosts caught at 1% false alarms", 100.0 * caught as f32 / tp.len() as f32), tp.len() + tn.len(), st, None);
    Ok(())
}

pub const CASES3: [Case; 2] = [(120, "real", radio_modulation_real), (121, "real", radio_occupancy_real)];

/// Real off-air recordings (rtl_433_tests, unsigned 8-bit I/Q at 250 kS/s) with OOK/FSK labels from `index.txt`.
fn recordings() -> Result<Vec<(bool, Vec<f32>)>, String> {
    let dir = data_path("data/rtl433");
    let index = std::fs::read_to_string(dir.join("index.txt")).map_err(|e| format!("data/rtl433/index.txt: {e}"))?;
    let mut v = Vec::new();
    for l in index.lines() {
        let p: Vec<&str> = l.split_whitespace().collect();
        if p.len() == 3 && p[1] == "250k" {
            if let Ok(b) = std::fs::read(dir.join(p[2])) {
                v.push((p[0] == "FSK", b.iter().take(2 * 128 * 1024).map(|&u| (u as f32 - 127.5) / 127.5).collect()));
            }
        }
    }
    Ok(v)
}

/// 128-point spectra of every frame of one recording, computed on the accelerator. Returns per-frame power (n x 128).
fn spectra(c: &mut Client, iq: &[f32], st: &mut Stats) -> Result<Vec<f32>, String> {
    let n = 128;
    let frames = iq.len() / (2 * n);
    let mut x = vec![0f32; frames * 2 * n];
    for f in 0..frames {
        for t in 0..n {
            x[f * 2 * n + t] = iq[2 * (f * n + t)];
            x[f * 2 * n + n + t] = iq[2 * (f * n + t) + 1];
        }
    }
    let (y, s) = c.run(&x, frames)?;
    add(st, s);
    Ok(y.chunks_exact(2 * n).flat_map(|r| (0..n).map(|k| r[k] * r[k] + r[n + k] * r[n + k]).collect::<Vec<_>>()).collect())
}

/// Frames that hold signal: energy more than six times the median frame (at least the strongest 2%).
fn active_frames(p: &[f32]) -> (Vec<usize>, Vec<f32>) {
    let e: Vec<f32> = p.chunks_exact(128).map(|r| r.iter().skip(1).sum()).collect(); // bin 0 holds the receiver's DC offset
    let thr = (6.0 * quantile(&e, 0.5)).min(quantile(&e, 0.98));
    ((0..e.len()).filter(|&f| e[f] > thr).collect(), e)
}

fn dft128() -> Vec<f32> {
    let n = 128usize;
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

/// 100 numbers describing one recording, all derived from the accelerator's spectra: where the strongest
/// bin sits from frame to frame, the average spectrum around it, and how steady the energy is.
fn radio_features(p: &[f32]) -> (Vec<f32>, f32) {
    let (act, e) = active_frames(p);
    let peak = |f: usize| (1..128).max_by(|&a, &b| p[f * 128 + a].partial_cmp(&p[f * 128 + b]).unwrap()).unwrap();
    let mut votes = [0u32; 128];
    act.iter().for_each(|&f| votes[peak(f)] += 1);
    let modal = (0..128).max_by_key(|&k| votes[k]).unwrap();
    let wrap = |k: usize| ((k + 128 - modal + 64) % 128) as i32 - 64;
    let mut v = vec![0f32; 100];
    let mut away = 0f32;
    for &f in &act {
        let off = wrap(peak(f));
        if off.abs() <= 16 {
            v[(off + 16) as usize] += 1.0 / act.len() as f32;
        }
        away += (off.abs() >= 2) as u32 as f32 / act.len() as f32;
        let tot: f32 = p[f * 128 + 1..(f + 1) * 128].iter().sum();
        for j in 0..64 {
            v[33 + j] += (p[f * 128 + (modal + 128 - 32 + j) % 128] / tot).sqrt() / act.len() as f32;
        }
    }
    let ea: Vec<f32> = act.iter().map(|&f| e[f]).collect();
    let mean = ea.iter().sum::<f32>() / ea.len() as f32;
    v[97] = act.len() as f32 / e.len() as f32;
    v[98] = (ea.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / ea.len() as f32).sqrt() / mean;
    v[99] = away;
    (v, away)
}

/// 120. OOK or FSK, on real recordings. Two answers: a rule with no fitting (does the strongest bin move
/// between frames?), and a closed-form readout over the 100 features, fitted on half the recordings.
fn radio_modulation_real(c: &mut Client) -> Result<(), String> {
    let recs = recordings()?;
    c.linear(dft128(), 256, 256, false)?;
    let mut st = Stats::default();
    let (mut feats, mut away) = (Vec::new(), Vec::new());
    for (_, iq) in &recs {
        let (v, a) = radio_features(&spectra(c, iq, &mut st)?);
        feats.push(v);
        away.push(a);
    }
    let n = recs.len();
    let rule = (0..n).filter(|&i| (away[i] > 0.3) == recs[i].0).count();
    let d = 100;
    let all: Vec<f32> = feats.concat();
    let (mean, sd) = mean_std(&all, n, d);
    let z = standardise(&all, d, &mean, &sd);
    let (fit, test): (Vec<usize>, Vec<usize>) = (0..n).partition(|i| i % 2 == 0);
    let xfit: Vec<f32> = fit.iter().flat_map(|&i| z[i * d..(i + 1) * d].to_vec()).collect();
    let t: Vec<f32> = fit.iter().map(|&i| if recs[i].0 { 1.0 } else { -1.0 }).collect();
    let net = random_feature_net(c, &xfit, &t, 1, d, 256, 0.1, 0.05, 120)?;
    c.define(&net)?;
    let xtest: Vec<f32> = test.iter().flat_map(|&i| z[i * d..(i + 1) * d].to_vec()).collect();
    let (y, _) = c.run(&xtest, test.len())?;
    let (fsk_ok, fsk_n) = test.iter().enumerate().filter(|(_, &i)| recs[i].0).fold((0, 0), |a, (j, _)| (a.0 + (y[j] > 0.0) as usize, a.1 + 1));
    let (ook_ok, ook_n) = test.iter().enumerate().filter(|(_, &i)| !recs[i].0).fold((0, 0), |a, (j, _)| (a.0 + (y[j] <= 0.0) as usize, a.1 + 1));
    let frames: usize = recs.iter().map(|r| r.1.len() / 256).sum();
    report(120, "OOK or FSK, real recordings", &format!("{n} off-air recordings, {frames} spectra"),
        &format!("rule: {rule} of {n} right; fitted readout on unseen half: FSK {fsk_ok}/{fsk_n}, OOK {ook_ok}/{ook_n}"), frames, st, None);
    Ok(())
}

/// 121. Spectrum occupancy on the same recordings: is there a transmission, how many bursts, how wide.
fn radio_occupancy_real(c: &mut Client) -> Result<(), String> {
    let recs = recordings()?;
    c.linear(dft128(), 256, 256, false)?;
    let mut st = Stats::default();
    let (mut bursts, mut widths, mut with_signal) = (Vec::new(), Vec::new(), 0);
    for (_, iq) in &recs {
        let p = spectra(c, iq, &mut st)?;
        let (act, e) = active_frames(&p);
        let strong = act.iter().filter(|&&f| e[f] > 6.0 * quantile(&e, 0.5)).count();
        with_signal += (strong >= 4) as usize;
        bursts.push((0..act.len()).filter(|&j| j == 0 || act[j] > act[j - 1] + 2).count() as f32);
        let avg: Vec<f32> = (1..128).map(|k| act.iter().map(|&f| p[f * 128 + k]).sum::<f32>() / act.len().max(1) as f32).collect();
        let top = avg.iter().cloned().fold(0.0, f32::max);
        widths.push(avg.iter().filter(|&&v| v > 0.1 * top).count() as f32 * 250.0 / 128.0);
    }
    let frames: usize = recs.iter().map(|r| r.1.len() / 256).sum();
    report(121, "Occupancy, real recordings", &format!("{} off-air recordings, {frames} spectra", recs.len()),
        &format!("signal found in {with_signal}; median {} bursts, {:.0} kHz occupied", quantile(&bursts, 0.5), quantile(&widths, 0.5)), frames, st, None);
    Ok(())
}

pub const CASES4: [Case; 3] = [(204, "real", mnist_neighbours), (228, "real", nsl_fusion), (312, "real", dga_rich)];

/// 204. Exact nearest neighbour over 30,000 real images (23.5 million numbers), paged through the
/// accelerator's memory in blocks of 2,000 images: each test digit takes the label of its closest image.
fn mnist_neighbours(c: &mut Client) -> Result<(), String> {
    let (xt, yt, nt, d) = load_labelled("data/mnist/train.f32")?;
    let (x, y, _, _) = load_labelled("data/mnist/test.f32")?;
    let (n, block) = (y.len(), 2000usize);
    let mut best = vec![(0usize, f32::MIN); n];
    let mut st = Stats::default();
    for b0 in (0..nt).step_by(block) {
        let m = block.min(nt - b0);
        let w = xt[b0 * d..(b0 + m) * d].to_vec();
        let bias: Vec<f32> = (0..m).map(|j| -0.5 * w[j * d..(j + 1) * d].iter().map(|v| v * v).sum::<f32>()).collect();
        c.define(&[Layer { n_in: d, n_out: m, relu: false, w, b: bias }])?;
        let (out, s) = c.run_argmax(&x[..n * d], n)?;
        add(&mut st, s);
        for (q, (i, v)) in out.into_iter().enumerate() {
            if v > best[q].1 {
                best[q] = (b0 + i, v);
            }
        }
    }
    let right = (0..n).filter(|&q| yt[best[q].0] == y[q]).count();
    report(204, "Nearest neighbour, MNIST", &format!("{n} digits against {nt} real images"), &format!("{:.1}% correct; {:.0} billion multiply-adds in total", 100.0 * right as f32 / n as f32, (n * nt * d) as f64 / 1e9), n, st, None);
    Ok(())
}

/// 228. NSL-KDD: raise an alarm if either the fitted classifier or the distance from normal behaviour says so.
fn nsl_fusion(c: &mut Client) -> Result<(), String> {
    let k = nsl()?;
    let (d, n, n_fit) = (k.d, k.ytest.len(), 24000usize);
    let step = k.ytrain.len() / n_fit;
    let xfit: Vec<f32> = (0..n_fit).flat_map(|i| k.train[i * step * d..(i * step + 1) * d].to_vec()).collect();
    let t: Vec<f32> = (0..n_fit).map(|i| if k.ytrain[i * step] == 0 { -1.0 } else { 1.0 }).collect();
    let net = random_feature_net(c, &xfit, &t, 1, d, 512, 0.25, 1e-2, 126)?;
    c.define(&net)?;
    let (yc, mut st) = c.run(&k.test, n)?;
    let normal = select(&k.train, &k.ytrain, d, 20000, |l| l == 0);
    let (vals, vecs) = jacobi_eigen(&covariance(&normal, normal.len() / d, d), d);
    let w: Vec<f32> = (0..d).flat_map(|i| { let s = 1.0 / (vals[i].max(0.0) + 0.1).sqrt(); vecs[i * d..(i + 1) * d].iter().map(move |v| (v * s) as f32).collect::<Vec<_>>() }).collect();
    c.linear(w, d, d, false)?;
    let norm = |y: &[f32]| -> Vec<f32> { y.chunks_exact(d).map(|r| r.iter().map(|v| v * v).sum::<f32>().sqrt()).collect() };
    let (yn, _) = c.run(&normal, normal.len() / d)?;
    let (yt, s2) = c.run(&k.test, n)?;
    add(&mut st, s2);
    let thr = quantile(&norm(&yn), 0.97);
    let dist = norm(&yt);
    let alarm: Vec<bool> = (0..n).map(|i| yc[i] > 0.0 || dist[i] > thr).collect();
    // third opinion: label of the nearest of all training connections (as in case 229)
    let (nt, block) = (k.ytrain.len(), 8000usize);
    let mut best = vec![(0usize, f32::MIN); n];
    for b0 in (0..nt).step_by(block) {
        let m = block.min(nt - b0);
        let wb = k.train[b0 * d..(b0 + m) * d].to_vec();
        let bias: Vec<f32> = (0..m).map(|j| -0.5 * wb[j * d..(j + 1) * d].iter().map(|v| v * v).sum::<f32>()).collect();
        c.define(&[Layer { n_in: d, n_out: m, relu: false, w: wb, b: bias }])?;
        let (out, s3) = c.run_argmax(&k.test, n)?;
        add(&mut st, s3);
        for (q, (i, v)) in out.into_iter().enumerate() {
            if v > best[q].1 {
                best[q] = (b0 + i, v);
            }
        }
    }
    let votes: Vec<u8> = (0..n).map(|i| (yc[i] > 0.0) as u8 + (dist[i] > thr) as u8 + (k.ytrain[best[i].0] != 0) as u8).collect();
    let score = |need: u8| -> String {
        let right = (0..n).filter(|&i| (votes[i] >= need) == (k.ytest[i] != 0)).count();
        let fa = (0..n).filter(|&i| k.ytest[i] == 0 && votes[i] >= need).count() as f32 / k.ytest.iter().filter(|&&l| l == 0).count() as f32;
        format!("{:.1}% correct, {:.1}% false alarms", 100.0 * right as f32 / n as f32, 100.0 * fa)
    };
    println!("328 | NSL-KDD, three opinions      | classifier, distance, nearest neighbour | any one: {}; any two: {}; all three: {}", score(1), score(2), score(3));
    let right = (0..n).filter(|&i| alarm[i] == (k.ytest[i] != 0)).count();
    let rate = |f: &dyn Fn(u8) -> bool| { let idx: Vec<usize> = (0..n).filter(|&i| f(k.ytest[i])).collect(); 100.0 * idx.iter().filter(|&&i| alarm[i]).count() as f32 / idx.len().max(1) as f32 };
    report(228, "NSL-KDD, classifier or distance", &format!("{n} real test connections"),
        &format!("{:.1}% correct; attacks {:.1}% (DoS {:.0}, probe {:.0}, R2L {:.0}, U2R {:.0}), false alarms {:.1}%", 100.0 * right as f32 / n as f32, rate(&|l| l != 0), rate(&|l| l == 1), rate(&|l| l == 2), rate(&|l| l == 3), rate(&|l| l == 4), rate(&|l| l == 0)), 2 * n, st, None);
    Ok(())
}

/// 312. Generated domains with richer inputs: letter pairs, hashed letter triples, and shape numbers
/// (length, digits, vowels, longest consonant run, distinct letters). Ridge readout, sparse fit on 120,000 names.
fn dga_rich(c: &mut Client) -> Result<(), String> {
    let text = std::fs::read_to_string(data_path("data/dga/dga_domains_full.csv")).map_err(|e| format!("data/dga: {e}"))?;
    let (d, feat) = rich_dga_inputs();
    let rows: Vec<(bool, &str, &str)> = text.lines().filter_map(|l| { let mut p = l.split(','); Some((p.next()? == "dga", p.next()?, p.next()?)) }).collect();
    let (n_fit, n_test) = (120000usize, 60000usize);
    let step = rows.len() / (n_fit + n_test);
    let (mut xtx, mut xty) = (vec![0f64; d * d], vec![0f64; d]);
    for i in 0..n_fit {
        let (dga, _, name) = rows[i * step];
        let f = feat(name);
        for &(a, va) in &f {
            xty[a] += va as f64 * if dga { 1.0 } else { -1.0 };
            for &(b, vb) in &f {
                xtx[a * d + b] += va as f64 * vb as f64;
            }
        }
    }
    let w: Vec<f32> = solve_spd(&xtx, d, &xty, 1, 0.3).iter().map(|&v| v as f32).collect();
    let mut x = vec![0f32; n_test * d];
    for i in 0..n_test {
        feat(rows[(n_fit + i) * step].2).iter().for_each(|&(a, v)| x[i * d + a] += v);
    }
    c.linear(w, 1, d, false)?;
    let (y, st) = c.run(&x, n_test)?;
    let right = (0..n_test).filter(|&i| (y[i] > 0.0) == rows[(n_fit + i) * step].0).count();
    let mut fam: std::collections::BTreeMap<&str, (usize, usize)> = Default::default();
    for i in (0..n_test).filter(|&i| rows[(n_fit + i) * step].0) {
        let e = fam.entry(rows[(n_fit + i) * step].1).or_default();
        e.0 += 1;
        e.1 += (y[i] > 0.0) as usize;
    }
    let mut worst: Vec<(&str, f32)> = fam.iter().filter(|(_, v)| v.0 >= 50).map(|(k, v)| (*k, 100.0 * v.1 as f32 / v.0 as f32)).collect();
    worst.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    let hard: Vec<String> = worst.iter().take(3).map(|(k, p)| format!("{k} {p:.0}%")).collect();
    let fa = (0..n_test).filter(|&i| !rows[(n_fit + i) * step].0 && y[i] > 0.0).count() as f32 / (0..n_test).filter(|&i| !rows[(n_fit + i) * step].0).count() as f32;
    report(312, "Generated domains, richer inputs", &format!("{n_test} real names, 1764 features"), &format!("{:.1}% correct, {:.1}% false alarms; hardest: {}", 100.0 * right as f32 / n_test as f32, 100.0 * fa, hard.join(", ")), n_test, st, None);
    Ok(())
}

pub const CASES5: [Case; 2] = [(217, "real", wifi_real), (218, "real", ping_real)];

/// Closed-form least squares for one target; returns the weights (features are the columns of `x`).
fn least_squares(x: &[f32], y: &[f32], d: usize, ridge: f64) -> Vec<f32> {
    let n = y.len();
    let (mut xtx, mut xty) = (vec![0f64; d * d], vec![0f64; d]);
    for r in 0..n {
        for i in 0..d {
            xty[i] += x[r * d + i] as f64 * y[r] as f64;
            (0..d).for_each(|j| xtx[i * d + j] += x[r * d + i] as f64 * x[r * d + j] as f64);
        }
    }
    solve_spd(&xtx, d, &xty, 1, ridge).iter().map(|&v| v as f32).collect()
}

fn rms(a: impl Iterator<Item = f32>) -> f32 {
    let (s, n) = a.fold((0f32, 0usize), |(s, n), v| (s + v * v, n + 1));
    (s / n.max(1) as f32).sqrt()
}

/// 218. Round-trip time to the gateway, measured five times a second on this machine's Wi-Fi.
/// A 16-tap linear predictor of the next value (in log milliseconds), fitted on the first 60%.
fn ping_real(c: &mut Client) -> Result<(), String> {
    let (raw, n, _) = load_rows("data/live/ping-gateway.f32")?;
    let s: Vec<f32> = raw.iter().map(|v| (1.0 + v).ln()).collect();
    let (taps, split) = (16usize, n * 6 / 10);
    let rows_of = |a: usize, b: usize| -> (Vec<f32>, Vec<f32>) { ((a..b).flat_map(|t| { let mut r = s[t - taps..t].to_vec(); r.push(1.0); r }).collect(), s[a..b].to_vec()) };
    let (xf, yf) = rows_of(taps, split);
    let w = least_squares(&xf, &yf, taps + 1, 1e-3);
    let (x, truth) = rows_of(split, n);
    c.linear(w, 1, taps + 1, false)?;
    let (y, st) = c.run(&x, truth.len())?;
    let mean = yf.iter().sum::<f32>() / yf.len() as f32;
    let (model, naive, flat) = (rms(y.iter().zip(&truth).map(|(a, b)| a - b)), rms((split..n).map(|t| s[t - 1] - s[t])), rms(truth.iter().map(|b| mean - b)));
    report(218, "Latency forecast, real pings", &format!("{} real round-trip times, 16 taps", truth.len()), &format!("RMS error {model:.3} (log ms); repeat-last {naive:.3}, long-run mean {flat:.3}"), truth.len(), st, None);
    Ok(())
}

/// 217. This machine's Wi-Fi link, sampled twice a second: predict the transmit bitrate from signal
/// strength, receive bitrate, traffic level and the previous sample. Fitted on the first 60%.
fn wifi_real(c: &mut Client) -> Result<(), String> {
    let (raw, n, d0) = load_rows("data/live/wifi.f32")?;
    let col = |r: usize, k: usize| raw[r * d0 + k];
    let feat = |r: usize| -> Vec<f32> { let sig = (col(r, 0) + 45.0) / 5.0; vec![1.0, sig, sig * sig, col(r, 3) / 1000.0, (1.0 + col(r, 5)).ln() / 5.0, col(r - 1, 2) / 1000.0] };
    let split = n * 6 / 10;
    let build = |a: usize, b: usize| -> (Vec<f32>, Vec<f32>) { ((a..b).flat_map(feat).collect(), (a..b).map(|r| col(r, 2) / 1000.0).collect()) };
    let (xf, yf) = build(1, split);
    let w = least_squares(&xf, &yf, 6, 1e-4);
    let (x, truth) = build(split, n);
    c.linear(w, 1, 6, false)?;
    let (y, st) = c.run(&x, truth.len())?;
    let (model, naive) = (1000.0 * rms(y.iter().zip(&truth).map(|(a, b)| a - b)), rms((split..n).map(|r| col(r - 1, 2) - col(r, 2))));
    let changes = (split..n).filter(|&r| col(r, 2) != col(r - 1, 2)).count();
    report(217, "Wi-Fi bitrate, real link", &format!("{} real samples of this link", truth.len()), &format!("RMS error {model:.0} Mbit/s; repeat-last {naive:.0}; the rate changed in {changes} samples"), truth.len(), st, None);
    Ok(())
}

pub const CASES6: [Case; 1] = [(301, "real", throughput)];

/// 301. Throughput check behind the nearest-neighbour figure: one large product, verified against the CPU,
/// timed on the DSP clock and on the wall clock.
fn throughput(c: &mut Client) -> Result<(), String> {
    let (rows, n_in, n_out) = (2048usize, 800usize, 2048usize);
    let mut r = Rng(301);
    let (w, x) = (r.normals(n_out * n_in, 0.05), r.normals(rows * n_in, 1.0));
    c.linear(w.clone(), n_out, n_in, false)?;
    let t = std::time::Instant::now();
    let (y, st) = c.run(&x, rows)?;
    let wall = t.elapsed().as_secs_f64();
    let sample: Vec<usize> = (0..rows).step_by(97).collect();
    let xs: Vec<f32> = sample.iter().flat_map(|&i| x[i * n_in..(i + 1) * n_in].to_vec()).collect();
    let ys: Vec<f32> = sample.iter().flat_map(|&i| y[i * n_out..(i + 1) * n_out].to_vec()).collect();
    let err = rel_err(&ys, &matmul_ref(&xs, sample.len(), n_in, &w, n_out));
    let macs = (rows * n_in * n_out) as f64;
    report(301, "Throughput check", "2048x800 times 800x2048", &format!("{:.0} billion multiply-adds/s on the DSP clock, {:.1} billion/s end to end", macs / st.dsp_us.max(1) as f64 / 1e3, macs / wall / 1e9), rows, st, Some(err < 0.02));
    Ok(())
}

pub const CASES7: [Case; 2] = [(208, "real", own_traffic_scoring), (215, "real", own_traffic_clusters)];

/// 208. Intrusion scoring with a baseline of this machine's real traffic (header-only capture on the
/// default-route interface). Fit on the first 70% of flow records in time order, then measure false
/// alarms on the unseen 30% and detection of the locally generated scans. Whitened distance, no training.
fn own_traffic_scoring(c: &mut Client) -> Result<(), String> {
    let (b, nb, d) = load_rows("ids/cap/real-baseline.f32")?;
    let (a, na, _) = load_rows("ids/cap/attack.f32")?;
    let n_fit = nb * 7 / 10;
    let (mean, sd) = mean_std(&b[..n_fit * d], n_fit, d);
    let (xb, xa) = (standardise(&b, d, &mean, &sd), standardise(&a, d, &mean, &sd));
    let (vals, vecs) = jacobi_eigen(&covariance(&xb[..n_fit * d], n_fit, d), d);
    let w: Vec<f32> = (0..d).flat_map(|k| { let s = 1.0 / (vals[k].max(0.0) + 0.1).sqrt(); vecs[k * d..(k + 1) * d].iter().map(move |v| (v * s) as f32).collect::<Vec<_>>() }).collect();
    c.linear(w, d, d, false)?;
    let x: Vec<f32> = [xb.as_slice(), xa.as_slice()].concat();
    let (y, st) = c.run(&x, nb + na)?;
    let dist: Vec<f32> = y.chunks_exact(d).map(|r| r.iter().map(|v| v * v).sum::<f32>().sqrt()).collect();
    let thr = quantile(&dist[..n_fit], 0.99);
    let fa = dist[n_fit..nb].iter().filter(|&&v| v > thr).count() as f32 / (nb - n_fit) as f32;
    let det = dist[nb..].iter().filter(|&&v| v > thr).count() as f32 / na as f32;
    report(208, "Intrusion scoring, own traffic", &format!("{nb} real flow records + {na} scan records"), &format!("{:.1}% false alarms on unseen real traffic, {:.1}% of scan flows flagged", 100.0 * fa, 100.0 * det), nb + na, st, None);
    Ok(())
}

/// 215. Clustering this machine's real traffic into six groups (nearest-centre step on the accelerator).
fn own_traffic_clusters(c: &mut Client) -> Result<(), String> {
    let (b, n, d) = load_rows("ids/cap/real-baseline.f32")?;
    let (mean, sd) = mean_std(&b, n, d);
    let x = standardise(&b, d, &mean, &sd);
    let k = 6;
    let mut r = Rng(215);
    let mut centres: Vec<f32> = (0..k).flat_map(|_| { let i = (r.next() % n as u64) as usize; x[i * d..(i + 1) * d].to_vec() }).collect();
    let (mut assign, mut st) = (vec![0usize; n], Stats::default());
    for _ in 0..10 {
        let bias: Vec<f32> = (0..k).map(|j| -0.5 * centres[j * d..(j + 1) * d].iter().map(|v| v * v).sum::<f32>()).collect();
        c.define(&[Layer { n_in: d, n_out: k, relu: false, w: centres.clone(), b: bias }])?;
        let (y, s) = c.run(&x, n)?;
        st = s;
        assign = argmax_rows(&y, k).iter().map(|p| p.0).collect();
        for j in 0..k {
            let members: Vec<usize> = (0..n).filter(|&i| assign[i] == j).collect();
            if !members.is_empty() {
                (0..d).for_each(|q| centres[j * d + q] = members.iter().map(|&i| x[i * d + q]).sum::<f32>() / members.len() as f32);
            }
        }
    }
    // describe each cluster by raw features: share of UDP (column 19), of port 443 (26), of port 53 (23), median packets out (0, log scale)
    let mut desc = Vec::new();
    for j in 0..k {
        let m: Vec<usize> = (0..n).filter(|&i| assign[i] == j).collect();
        if m.is_empty() { continue; }
        let share = |col: usize| 100.0 * m.iter().map(|&i| b[i * d + col]).sum::<f32>() / m.len() as f32;
        desc.push(format!("{} ({:.0}% udp, {:.0}% dns, {:.0}% https)", m.len(), share(19), share(23), share(26)));
    }
    report(215, "Clusters, own traffic", &format!("{n} real flow records, 6 centres"), &format!("sizes: {}", desc.join("; ")), n, st, None);
    Ok(())
}

pub const CASES8: [Case; 1] = [(209, "live", live_scoring)];

/// 209. Live scoring: fit the whitened-distance baseline on `ids/cap/real-baseline.f32` and score the
/// flow records of the latest capture (`ids/cap/live.f32`, written by `ids/live2.sh`). Lists the oddest flows.
fn live_scoring(c: &mut Client) -> Result<(), String> {
    let (b, nb, d) = load_rows("ids/cap/real-baseline.f32")?;
    let (l, nl, _) = load_rows("ids/cap/live.f32")?;
    let meta = std::fs::read_to_string(data_path("ids/cap/live.txt")).unwrap_or_default();
    let meta: Vec<&str> = meta.lines().collect();
    let (mean, sd) = mean_std(&b, nb, d);
    let (xb, xl) = (standardise(&b, d, &mean, &sd), standardise(&l, d, &mean, &sd));
    let (vals, vecs) = jacobi_eigen(&covariance(&xb, nb, d), d);
    let w: Vec<f32> = (0..d).flat_map(|k| { let s = 1.0 / (vals[k].max(0.0) + 0.1).sqrt(); vecs[k * d..(k + 1) * d].iter().map(move |v| (v * s) as f32).collect::<Vec<_>>() }).collect();
    c.linear(w, d, d, false)?;
    let norm = |y: &[f32]| -> Vec<f32> { y.chunks_exact(d).map(|r| r.iter().map(|v| v * v).sum::<f32>().sqrt()).collect() };
    let (yb, _) = c.run(&xb, nb)?;
    let (yl, st) = c.run(&xl, nl)?;
    let (thr, dist) = (quantile(&norm(&yb), 0.99), norm(&yl));
    let alarms = dist.iter().filter(|&&v| v > thr).count();
    report(209, "Live flow scoring", &format!("{nl} live flow records, baseline of {nb}"), &format!("{alarms} above the alarm threshold {thr:.1}"), nl, st, None);
    let mut order: Vec<usize> = (0..nl).collect();
    order.sort_by(|&p, &q| dist[q].partial_cmp(&dist[p]).unwrap());
    for &i in order.iter().take(10) {
        // the feature that deviates most from the baseline, in standard deviations
        let worst = (0..d).max_by(|&p, &q| xl[i * d + p].abs().partial_cmp(&xl[i * d + q].abs()).unwrap()).unwrap();
        println!("   {:>7.1} {} {}   ({} is {:+.1} deviations from baseline)", dist[i], if dist[i] > thr { "ALARM" } else { "     " }, meta.get(i).unwrap_or(&"?"), FLOW_FEATURES.get(worst).unwrap_or(&"?"), xl[i * d + worst]);
    }
    Ok(())
}

/// The 32 flow features, in the order `ids/ids.py` writes them.
const FLOW_FEATURES: [&str; 32] = ["packets out", "packets in", "bytes out", "bytes in", "duration", "mean size out", "mean size in", "size spread out", "size spread in",
    "mean gap", "gap spread", "share of bytes out", "share of packets out", "SYN share", "SYN-ACK share", "FIN share", "RST share", "PSH share", "is TCP", "is UDP",
    "well-known port", "registered port", "port 22", "port 53", "port 80", "port 123", "port 443", "port 5353",
    "flows from this source", "ports touched by this source", "hosts touched by this source", "unanswered share for this source"];

pub const CASES9: [Case; 1] = [(223, "real", radio_matched_real)];

/// 223. Matched filtering on real recordings. From each of 24 on-off-keyed recordings take its busiest
/// 16,384 samples; cut a 256-sample template from the strongest part of each; join all 24 excerpts into one
/// stream and correlate every window with all 24 templates on the accelerator (complex, as real pairs).
/// Question: leaving out the template's own position, is the best match another packet in the same excerpt?
fn radio_matched_real(c: &mut Client) -> Result<(), String> {
    let (seg, tl, stride) = (16384usize, 256usize, 16usize);
    let recs: Vec<Vec<f32>> = recordings()?.into_iter().filter(|r| !r.0 && r.1.len() >= 2 * 4 * seg).map(|r| r.1).step_by(4).take(24).collect();
    let n = recs.len();
    let energy = |iq: &[f32], a: usize, len: usize| -> f32 { iq[2 * a..2 * (a + len)].iter().map(|v| v * v).sum() };
    let (mut stream, mut tpl_at) = (Vec::new(), Vec::new());
    for (ri, iq) in recs.iter().enumerate() {
        let total = iq.len() / 2;
        let best_seg = (0..=(total - seg) / 4096).map(|k| k * 4096).max_by(|&a, &b| energy(iq, a, seg).partial_cmp(&energy(iq, b, seg)).unwrap()).unwrap();
        let s = &iq[2 * best_seg..2 * (best_seg + seg)];
        let best_t = (0..(seg - tl) / 64).map(|k| k * 64).max_by(|&a, &b| energy(s, a, tl).partial_cmp(&energy(s, b, tl)).unwrap()).unwrap();
        tpl_at.push(ri * seg + best_t);
        stream.extend_from_slice(s);
    }
    let sample = |i: usize| (stream[2 * i], stream[2 * i + 1]);
    // weights: conjugate of each template, unit energy; rows are [re..., im...]
    let mut w = vec![0f32; 2 * n * 2 * tl];
    for (t, &at) in tpl_at.iter().enumerate() {
        let norm = (0..tl).map(|k| { let (re, im) = sample(at + k); re * re + im * im }).sum::<f32>().sqrt().max(1e-6);
        for k in 0..tl {
            let (re, im) = sample(at + k);
            w[t * 2 * tl + k] = re / norm;
            w[t * 2 * tl + tl + k] = im / norm;
            w[(n + t) * 2 * tl + k] = -im / norm;
            w[(n + t) * 2 * tl + tl + k] = re / norm;
        }
    }
    let rows = (n * seg - tl) / stride;
    let mut x = vec![0f32; rows * 2 * tl];
    let mut wnorm = vec![0f32; rows];
    for r in 0..rows {
        for k in 0..tl {
            let (re, im) = sample(r * stride + k);
            x[r * 2 * tl + k] = re;
            x[r * 2 * tl + tl + k] = im;
            wnorm[r] += re * re + im * im;
        }
    }
    c.linear(w, 2 * n, 2 * tl, false)?;
    let (y, st) = c.run(&x, rows)?;
    let mut same = 0;
    let mut scores = Vec::new();
    for t in 0..n {
        let score = |r: usize| (y[r * 2 * n + t].powi(2) + y[r * 2 * n + n + t].powi(2)).sqrt() / wnorm[r].sqrt().max(1e-3);
        let best = (0..rows).filter(|&r| (r * stride).abs_diff(tpl_at[t]) > 2 * tl).max_by(|&a, &b| score(a).partial_cmp(&score(b)).unwrap()).unwrap();
        same += (best * stride / seg == t) as usize;
        scores.push(score(best));
    }
    report(223, "Matched filter, carrier", &format!("{rows} windows of {n} joined real excerpts"), &format!("{same} of {n}: best other match is in the same device's excerpt (median score {:.2}; mostly matches carrier frequency)", quantile(&scores, 0.5)), rows, st, None);

    // Second form: match the on/off pattern, not the carrier. Envelope (magnitude) averaged over 16 samples,
    // a 256-point template (4,096 samples, about 16 ms) around the same place, zero-mean and unit length.
    let dec = 16usize;
    let env: Vec<f32> = (0..n * seg / dec).map(|i| (0..dec).map(|k| { let (re, im) = sample(i * dec + k); (re * re + im * im).sqrt() }).sum::<f32>() / dec as f32).collect();
    let unit = |v: &[f32]| -> Vec<f32> { let m = v.iter().sum::<f32>() / v.len() as f32; let z: Vec<f32> = v.iter().map(|a| a - m).collect(); let nn = z.iter().map(|a| a * a).sum::<f32>().sqrt().max(1e-6); z.iter().map(|a| a / nn).collect() };
    let starts: Vec<usize> = tpl_at.iter().enumerate().map(|(t, &at)| (at / dec).saturating_sub(tl / 2).clamp(t * seg / dec, (t + 1) * seg / dec - tl)).collect();
    let w2: Vec<f32> = starts.iter().flat_map(|&a| unit(&env[a..a + tl])).collect();
    let rows2 = env.len() - tl;
    let x2: Vec<f32> = (0..rows2).flat_map(|r| unit(&env[r..r + tl])).collect();
    c.linear(w2, n, tl, false)?;
    let (y2, st2) = c.run(&x2, rows2)?;
    let (mut same2, mut scores2) = (0, Vec::new());
    for t in 0..n {
        let best = (0..rows2).filter(|&r| r.abs_diff(starts[t]) > tl).max_by(|&a, &b| y2[a * n + t].partial_cmp(&y2[b * n + t]).unwrap()).unwrap();
        same2 += ((best + tl / 2) * dec / seg == t) as usize;
        scores2.push(y2[best * n + t]);
    }
    report(323, "Matched filter, packet pattern", &format!("{rows2} envelope windows, {n} real devices"), &format!("{same2} of {n}: best other match is in the same device's excerpt (median score {:.2}, chance is 1 in {n})", quantile(&scores2, 0.5)), rows2, st2, None);
    Ok(())
}

pub const CASES10: [Case; 1] = [(216, "real", first_packets_real)];

/// 216. First-packet signatures on this machine's real connections: the first eight packet sizes,
/// directions and gaps of each TCP connection, matched to per-server averages built from earlier connections.
fn first_packets_real(c: &mut Client) -> Result<(), String> {
    let (x, y, n, d) = load_labelled("ids/cap/firstpackets.f32")?;
    if n == 0 {
        println!("216 | First packets, own connections | not run                            | too few new TCP connections per server in the capture (needs 6 or more to one endpoint)");
        return Ok(());
    }
    let classes = *y.iter().max().ok_or("no connections")? as usize + 1;
    let mut seen = vec![0usize; classes];
    let (mut fit, mut test) = (Vec::new(), Vec::new());
    for i in 0..n {
        if seen[y[i] as usize] % 2 == 0 { fit.push(i) } else { test.push(i) }
        seen[y[i] as usize] += 1;
    }
    let (mut w, mut cnt) = (vec![0f32; classes * d], vec![0f32; classes]);
    for &i in &fit {
        cnt[y[i] as usize] += 1.0;
        (0..d).for_each(|k| w[y[i] as usize * d + k] += x[i * d + k]);
    }
    (0..classes).for_each(|j| (0..d).for_each(|k| w[j * d + k] /= cnt[j].max(1.0)));
    let b: Vec<f32> = (0..classes).map(|j| -0.5 * w[j * d..(j + 1) * d].iter().map(|v| v * v).sum::<f32>()).collect();
    c.define(&[Layer { n_in: d, n_out: classes, relu: false, w, b }])?;
    let xt: Vec<f32> = test.iter().flat_map(|&i| x[i * d..(i + 1) * d].to_vec()).collect();
    let (out, st) = c.run(&xt, test.len())?;
    let right = argmax_rows(&out, classes).iter().zip(&test).filter(|(p, &i)| p.0 == y[i] as usize).count();
    report(216, "First packets, own connections", &format!("{} real connections, {classes} servers", test.len()), &format!("{:.1}% matched to the right server (chance {:.1}%)", 100.0 * right as f32 / test.len() as f32, 100.0 / classes as f32), test.len(), st, None);
    Ok(())
}

pub const CASES11: [Case; 1] = [(229, "real", nsl_neighbours)];

/// 229. NSL-KDD with no model at all: each test connection takes the label of its nearest training
/// connection, searching all 125,973 of them in blocks that fit the accelerator's memory.
fn nsl_neighbours(c: &mut Client) -> Result<(), String> {
    let k = nsl()?;
    let (d, n, nt, block) = (k.d, k.ytest.len(), k.ytrain.len(), 8000usize);
    let mut best = vec![(0usize, f32::MIN); n];
    let mut st = Stats::default();
    for b0 in (0..nt).step_by(block) {
        let m = block.min(nt - b0);
        let w = k.train[b0 * d..(b0 + m) * d].to_vec();
        let bias: Vec<f32> = (0..m).map(|j| -0.5 * w[j * d..(j + 1) * d].iter().map(|v| v * v).sum::<f32>()).collect();
        c.define(&[Layer { n_in: d, n_out: m, relu: false, w, b: bias }])?;
        let (out, s) = c.run_argmax(&k.test, n)?;
        add(&mut st, s);
        for (q, (i, v)) in out.into_iter().enumerate() {
            if v > best[q].1 {
                best[q] = (b0 + i, v);
            }
        }
    }
    let alarm: Vec<bool> = best.iter().map(|b| k.ytrain[b.0] != 0).collect();
    let right = (0..n).filter(|&i| alarm[i] == (k.ytest[i] != 0)).count();
    let rate = |f: &dyn Fn(u8) -> bool| { let idx: Vec<usize> = (0..n).filter(|&i| f(k.ytest[i])).collect(); 100.0 * idx.iter().filter(|&&i| alarm[i]).count() as f32 / idx.len().max(1) as f32 };
    report(229, "NSL-KDD nearest neighbour", &format!("{n} test against {nt} training connections"),
        &format!("{:.1}% correct; attacks {:.1}% (DoS {:.0}, probe {:.0}, R2L {:.0}, U2R {:.0}), false alarms {:.1}%", 100.0 * right as f32 / n as f32, rate(&|l| l != 0), rate(&|l| l == 1), rate(&|l| l == 2), rate(&|l| l == 3), rate(&|l| l == 4), rate(&|l| l == 0)), n, st, None);
    Ok(())
}

pub const CASES12: [Case; 1] = [(412, "real", dga_neighbours)];

/// 412. Generated domains by nearest neighbour: each test name takes the label of the most similar of
/// 60,000 labelled names (cosine similarity of hashed letter triples). Word-built families reuse a small
/// vocabulary, so their names resemble each other even though each looks like ordinary words.
fn dga_neighbours(c: &mut Client) -> Result<(), String> {
    let text = std::fs::read_to_string(data_path("data/dga/dga_domains_full.csv")).map_err(|e| format!("data/dga: {e}"))?;
    let rows: Vec<(bool, &str, &str)> = text.lines().filter_map(|l| { let mut p = l.split(','); Some((p.next()? == "dga", p.next()?, p.next()?)) }).collect();
    let (dim, n_ref, n_test, block) = (1024usize, 60000usize, 10000usize, 1000usize);
    let step = rows.len() / (n_ref + n_test);
    let label_of = |name: &str| name.split('.').next().unwrap_or(name).to_string();
    // interleave so that reference and test names come from the whole file, never the same row
    let test: Vec<(bool, &str, &str)> = (0..n_test).map(|i| rows[(i * 7 + 3) * step]).collect();
    let refs: Vec<(bool, &str, &str)> = (0..(n_ref + n_test)).filter(|i| !(i % 7 == 3 && i / 7 < n_test)).take(n_ref).map(|i| rows[i * step]).collect();
    let x: Vec<f32> = test.iter().flat_map(|t| trigram_vector(&label_of(t.2), dim)).collect();
    let mut best = vec![(0usize, f32::MIN); n_test];
    let mut st = Stats::default();
    for b0 in (0..refs.len()).step_by(block) {
        let m = block.min(refs.len() - b0);
        let w: Vec<f32> = refs[b0..b0 + m].iter().flat_map(|r| trigram_vector(&label_of(r.2), dim)).collect();
        c.linear(w, m, dim, false)?;
        let (out, s) = c.run_argmax(&x, n_test)?;
        add(&mut st, s);
        for (q, (i, v)) in out.into_iter().enumerate() {
            if v > best[q].1 {
                best[q] = (b0 + i, v);
            }
        }
    }
    let guess: Vec<bool> = best.iter().map(|b| refs[b.0].0).collect();
    let right = (0..n_test).filter(|&i| guess[i] == test[i].0).count();
    let fa = (0..n_test).filter(|&i| !test[i].0 && guess[i]).count() as f32 / test.iter().filter(|t| !t.0).count() as f32;
    let mut fam: std::collections::BTreeMap<&str, (usize, usize)> = Default::default();
    for i in (0..n_test).filter(|&i| test[i].0) {
        let e = fam.entry(test[i].1).or_default();
        e.0 += 1;
        e.1 += guess[i] as usize;
    }
    let pick = |name: &str| fam.get(name).map(|v| format!("{name} {:.0}%", 100.0 * v.1 as f32 / v.0.max(1) as f32)).unwrap_or_default();
    report(412, "Generated domains, neighbours", &format!("{n_test} real names against {} labelled names", refs.len()),
        &format!("{:.1}% correct, {:.1}% false alarms; {}, {}, {}", 100.0 * right as f32 / n_test as f32, 100.0 * fa, pick("nymaim"), pick("suppobox"), pick("gozi")), n_test, st, None);

    // 512. Combination: the strong readout of case 312 (good on random-looking names) OR a nearest neighbour
    // that is a generated name and close enough (good on word-built names). Three closeness thresholds are
    // reported side by side; none is tuned on these test names.
    let (d, feat) = rich_dga_inputs();
    let (mut xtx, mut xty) = (vec![0f64; d * d], vec![0f64; d]);
    for r in &refs {
        let f = feat(r.2);
        for &(a, va) in &f {
            xty[a] += va as f64 * if r.0 { 1.0 } else { -1.0 };
            for &(b, vb) in &f {
                xtx[a * d + b] += va as f64 * vb as f64;
            }
        }
    }
    let w: Vec<f32> = solve_spd(&xtx, d, &xty, 1, 0.3).iter().map(|&v| v as f32).collect();
    let mut xr = vec![0f32; n_test * d];
    for (i, t) in test.iter().enumerate() {
        feat(t.2).iter().for_each(|&(a, v)| xr[i * d + a] += v);
    }
    c.linear(w, 1, d, false)?;
    let (ys, st2) = c.run(&xr, n_test)?;
    let readout = (0..n_test).filter(|&i| (ys[i] > 0.0) == test[i].0).count();
    let n_legit = test.iter().filter(|t| !t.0).count() as f32;
    let mut parts = Vec::new();
    for tau in [0.5f32, 0.6, 0.7] {
        let both: Vec<bool> = (0..n_test).map(|i| ys[i] > 0.0 || (guess[i] && best[i].1 > tau)).collect();
        let right2 = (0..n_test).filter(|&i| both[i] == test[i].0).count();
        let fa2 = (0..n_test).filter(|&i| !test[i].0 && both[i]).count() as f32 / n_legit;
        let sup = (0..n_test).filter(|&i| test[i].0 && test[i].1 == "suppobox").fold((0, 0), |a, i| (a.0 + both[i] as usize, a.1 + 1));
        parts.push(format!("closeness {tau}: {:.1}% correct, {:.1}% false alarms, suppobox {:.0}%", 100.0 * right2 as f32 / n_test as f32, 100.0 * fa2, 100.0 * sup.0 as f32 / sup.1.max(1) as f32));
    }
    report(512, "Generated domains, combined", &format!("{n_test} real names, readout OR close neighbour"),
        &format!("readout alone {:.1}%; {}", 100.0 * readout as f32 / n_test as f32, parts.join("; ")), n_test, st2, None);
    Ok(())
}

/// The 1,764 inputs of case 312 for one domain name, as (index, value) pairs: letter pairs, hashed letter
/// triples, shape numbers and dictionary-word counts. Returns the input width and the builder.
fn rich_dga_inputs() -> (usize, Box<dyn Fn(&str) -> Vec<(usize, f32)>>) {
    let d = 729 + 1024 + 6 + 5;
    // Word-level inputs: split the name greedily into the longest dictionary words (system word list, 4+ letters).
    let dict: std::collections::HashSet<String> = std::fs::read_to_string("/usr/share/dict/words").unwrap_or_default().lines()
        .filter(|w| w.len() >= 4 && w.bytes().all(|b| b.is_ascii_lowercase())).map(|w| w.to_string()).collect();
    let words = move |label: &str| -> (f32, f32, f32, f32) {
        let b = label.as_bytes();
        let (mut i, mut n, mut covered, mut longest) = (0usize, 0f32, 0usize, 0usize);
        while i < b.len() {
            let hit = (4..=(b.len() - i).min(14)).rev().find(|&l| std::str::from_utf8(&b[i..i + l]).map(|w| dict.contains(w)).unwrap_or(false));
            match hit { Some(l) => { n += 1.0; covered += l; longest = longest.max(l); i += l } None => i += 1 }
        }
        (n, covered as f32 / b.len().max(1) as f32, longest as f32 / 14.0, if covered == b.len() && n == 2.0 { 1.0 } else { 0.0 })
    };
    let feat = move |name: &str| -> Vec<(usize, f32)> {
        let label = name.split('.').next().unwrap_or(name);
        let b = label.as_bytes();
        let p = letter_pairs(label);
        let mut v: Vec<(usize, f32)> = p.iter().map(|&i| (i, 1.0 / p.len() as f32)).collect();
        let padded: Vec<u8> = [b" ", b, b" "].concat();
        let nt = padded.len().saturating_sub(2).max(1) as f32;
        for w in padded.windows(3) {
            let h = (w[0] as u32).wrapping_mul(0x9E37_79B1) ^ (w[1] as u32).wrapping_mul(0x85EB_CA6B) ^ (w[2] as u32).wrapping_mul(0xC2B2_AE35);
            v.push((729 + (h.wrapping_mul(0x27D4_EB2F) >> 16) as usize % 1024, 1.0 / nt));
        }
        let len = b.len().max(1) as f32;
        let vowels = b.iter().filter(|c| b"aeiou".contains(c)).count() as f32;
        let (mut run, mut longest) = (0, 0);
        for ch in b {
            if ch.is_ascii_lowercase() && !b"aeiou".contains(ch) { run += 1; longest = longest.max(run) } else { run = 0 }
        }
        let distinct = (0..=255u8).filter(|c| b.contains(c)).count() as f32;
        v.extend([(1753, len / 32.0), (1754, b.iter().filter(|c| c.is_ascii_digit()).count() as f32 / len), (1755, vowels / len), (1756, longest as f32 / 8.0), (1757, distinct / len), (1758, 1.0)]);
        let (nw, cov, lw, two) = words(label);
        v.extend([(1759, nw / 4.0), (1760, cov), (1761, lw), (1762, two), (1763, cov * len / 32.0)]);
        v
    };
    (d, Box::new(feat))
}
