//! 140. Every kind of operation the matrix accelerator can express through one fully connected layer
//! (y = W x + b, optionally clamped at zero), each checked against float32 on the CPU.
//! The accelerator has one primitive, FP16 multiply-accumulate; everything here is a choice of W and b.
use crate::util::*;
use crate::Case;
use npu::proto::Client;
use npu::session::Layer;

pub const CASES: [Case; 1] = [(140, "ops", sweep)];

fn reference(layers: &[Layer], x: &[f32], rows: usize) -> Vec<f32> {
    let mut cur = x.to_vec();
    for l in layers {
        let mut y = matmul_ref(&cur, rows, l.n_in, &l.w, l.n_out);
        for (i, v) in y.iter_mut().enumerate() {
            *v += l.b[i % l.n_out];
            if l.relu {
                *v = v.max(0.0);
            }
        }
        cur = y;
    }
    cur
}

fn sweep(c: &mut Client) -> Result<(), String> {
    let (d, rows) = (256usize, 4096usize);
    let mut r = Rng(140);
    let x = r.normals(rows * d, 1.0);
    let eye = |f: &dyn Fn(usize, usize) -> f32, n_out: usize| -> Vec<f32> { (0..n_out * d).map(|i| f(i / d, i % d)).collect() };
    let lin = |w: Vec<f32>, n_out: usize, relu: bool, bias: f32| Layer { n_in: d, n_out, relu, w, b: vec![bias; n_out] };
    let dense = r.normals(d * d, 0.06);
    let two = vec![lin(r.normals(d * d, 0.06), d, true, 0.1), Layer { n_in: d, n_out: 16, relu: false, w: r.normals(16 * d, 0.06), b: vec![0.0; 16] }];
    let ops: Vec<(&str, Vec<Layer>)> = vec![
        ("dense matrix multiply", vec![lin(dense.clone(), d, false, 0.0)]),
        ("multiply, add a constant", vec![lin(dense.clone(), d, false, 0.5)]),
        ("multiply, clamp at zero (ReLU)", vec![lin(dense, d, true, 0.0)]),
        ("copy (identity)", vec![lin(eye(&|j, i| (i == j) as u32 as f32, d), d, false, 0.0)]),
        ("scale by a constant", vec![lin(eye(&|j, i| if i == j { 0.25 } else { 0.0 }, d), d, false, 0.0)]),
        ("negate", vec![lin(eye(&|j, i| if i == j { -1.0 } else { 0.0 }, d), d, false, 0.0)]),
        ("absolute value (two clamps, summed)", vec![lin(eye(&|j, i| if i == j / 2 { if j % 2 == 0 { 1.0 } else { -1.0 } } else { 0.0 }, 2 * d), 2 * d, true, 0.0),
            Layer { n_in: 2 * d, n_out: d, relu: false, w: (0..d * 2 * d).map(|k| ((k % (2 * d)) / 2 == k / (2 * d)) as u32 as f32).collect(), b: vec![0.0; d] }]),
        ("reverse / permute", vec![lin(eye(&|j, i| (i == d - 1 - j) as u32 as f32, d), d, false, 0.0)]),
        ("pick columns (gather)", vec![lin(eye(&|j, i| (i == 4 * j) as u32 as f32, 64), 64, false, 0.0)]),
        ("sum of each row", vec![lin(vec![1.0; d], 1, false, 0.0)]),
        ("mean of each row", vec![lin(vec![1.0 / d as f32; d], 1, false, 0.0)]),
        ("difference of neighbours", vec![lin(eye(&|j, i| if i == j { 1.0 } else if i + 1 == j { -1.0 } else { 0.0 }, d), d, false, 0.0)]),
        ("moving average (1-D convolution)", vec![lin(eye(&|j, i| if i + 4 >= j && i <= j + 4 { 1.0 / 9.0 } else { 0.0 }, d), d, false, 0.0)]),
        ("pooling by 4 (average)", vec![lin(eye(&|j, i| if i / 4 == j { 0.25 } else { 0.0 }, 64), 64, false, 0.0)]),
        ("cosine transform", vec![lin(eye(&|j, i| (std::f32::consts::PI * (i as f32 + 0.5) * j as f32 / d as f32).cos() / (d as f32).sqrt(), d), d, false, 0.0)]),
        ("threshold (step via clamp)", vec![lin(eye(&|j, i| if i == j { 4.0 } else { 0.0 }, d), d, true, -2.0)]),
        ("two layers chained on the DSP", two),
    ];
    println!("140 | Operation sweep on the matrix accelerator | {rows} rows of {d} values per operation");
    let mut good = 0;
    for (name, layers) in &ops {
        c.define(layers)?;
        let (y, st) = c.run(&x, rows)?;
        let want = reference(layers, &x, rows);
        let err = rel_err(&y, &want);
        let macs: usize = layers.iter().map(|l| l.n_in * l.n_out).sum::<usize>() * rows;
        good += (err < 0.02) as u32;
        println!("    | {name:<36} | {} | worst error {:.3}% of RMS | {:>6.1} ms on the DSP, {:>6.1} ms end to end | {:>5.0} billion multiply-adds/s on the DSP",
            if err < 0.02 { "correct" } else { "WRONG  " }, 100.0 * err, st.dsp_us as f64 / 1e3, (st.wall_us + st.pack_us) as f64 / 1e3, macs as f64 / st.dsp_us.max(1) as f64 / 1e3);
    }
    // Best match per row: the server's 'A' request.
    c.define(&ops[0].1)?;
    let (best, _) = c.run_argmax(&x, rows)?;
    let want = argmax_rows(&reference(&ops[0].1, &x, rows), d);
    let same = best.iter().zip(&want).filter(|(a, b)| a.0 == b.0).count();
    println!("    | {:<36} | {} | {same} of {rows} rows agree with the CPU", "largest output per row (argmax)", if same * 100 >= rows * 99 { "correct" } else { "WRONG  " });
    println!("    | {} of {} operations correct. Not possible in one layer: multiplying two inputs together, division, square root, sorting, anything data-dependent.", good + (same * 100 >= rows * 99) as u32, ops.len() + 1);
    Ok(())
}
