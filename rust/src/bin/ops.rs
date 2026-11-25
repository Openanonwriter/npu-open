//! Client for the C++ operation sweep (`ops/`): runs every operation on the DSP, repeats it on the CPU and
//! compares checksums. Start through run.sh: `CLIENT=$PWD/rust/target/release/ops ./run.sh 40`.
use npu::fastrpc::{invoke, remotectl_open, scalars, Arg};
use std::time::Instant;

const N: usize = 65536;
const REPEAT: u32 = 16;
const NAMES: [&str; 13] = [
    "8-bit add, saturating", "8-bit absolute difference", "8-bit rounding average", "bitwise xor / and / or", "shifts",
    "8x8 -> 16-bit multiply", "dot product (sum of products)", "sum of absolute differences", "compare and select",
    "population count", "32-bit multiply-add", "float32 multiply-add", "table lookup",
];

fn fnv(b: &[u8]) -> u32 {
    b.iter().fold(2166136261u32, |h, &x| (h ^ x as u32).wrapping_mul(16777619))
}

/// The same operation as `ops/kern.inc`, on the CPU; returns the output bytes.
fn reference(op: usize, a: &[u8], b: &[u8]) -> Vec<u8> {
    let z = a.iter().zip(b).map(|(&x, &y)| (x as u32, y as u32));
    let bytes = |f: &dyn Fn(u32, u32) -> u8| -> Vec<u8> { z.clone().map(|(x, y)| f(x, y)).collect() };
    match op {
        0 => bytes(&|x, y| (x + y).min(255) as u8),
        1 => bytes(&|x, y| x.abs_diff(y) as u8),
        2 => bytes(&|x, y| ((x + y + 1) >> 1) as u8),
        3 => bytes(&|x, y| ((x ^ y) | (x & 0x0f)) as u8),
        4 => bytes(&|x, y| ((x << 3) | (y >> 5)) as u8),
        5 => z.flat_map(|(x, y)| ((x * y) as u16).to_le_bytes()).collect(),
        6 => z.map(|(x, y)| x * y).fold(0u32, u32::wrapping_add).to_le_bytes().to_vec(),
        7 => z.map(|(x, y)| x.abs_diff(y)).sum::<u32>().to_le_bytes().to_vec(),
        8 => bytes(&|x, y| if x > 128 { y as u8 } else { 0 }),
        9 => bytes(&|x, y| (x ^ y).count_ones() as u8),
        10 => z.enumerate().flat_map(|(i, (x, y))| x.wrapping_mul(65537).wrapping_mul(y.wrapping_mul(257)).wrapping_add(i as u32).to_le_bytes()).collect(),
        11 => z.flat_map(|(x, y)| (x as f32 * 0.5 + y as f32 * 1.5).to_le_bytes()).collect(),
        _ => bytes(&|x, y| ((x * 7 + 3) as u8) ^ y as u8),
    }
}

fn main() {
    let fd: i32 = std::env::var("HEXAGONRPC_FD").ok().and_then(|v| v.parse().ok()).expect("start through run.sh (HEXAGONRPC_FD not set)");
    std::thread::sleep(std::time::Duration::from_secs(2)); // the daemon registers its file listener just after starting us
    let h = remotectl_open(fd, "ops").expect("open libops_skel.so on the DSP");
    let seed = 12345u32;
    let call = |method: u32, words: &[u32], out: &mut [u32]| -> u32 { invoke(fd, h, scalars(method, 1, 1), &mut [Arg::input(words), Arg::output(out)]).unwrap_or(0xffff_ffff) };

    // Method 0: the operations.
    let mut out = vec![0u32; 4 * NAMES.len() + 1];
    let ret = call(0, &[N as u32, seed], &mut out);
    println!("ops: sweep ret=0x{ret:x}, HVX lock returned {}", out[4 * NAMES.len()] as i32);
    let (mut a, mut b, mut x) = (vec![0u8; N], vec![0u8; N], seed);
    let mut next = || { x = x.wrapping_mul(1664525).wrapping_add(1013904223); (x >> 24) as u8 };
    for i in 0..N { a[i] = next(); b[i] = next(); }
    let per = |us: u32, bytes: usize| us as f64 * 1000.0 / (REPEAT as f64 * (if bytes == 4 { N } else { N }) as f64);
    let mut good = 0;
    for (op, name) in NAMES.iter().enumerate() {
        let t0 = Instant::now();
        let mut r = Vec::new();
        for _ in 0..REPEAT { r = std::hint::black_box(reference(op, std::hint::black_box(&a), &b)); }
        let cpu_ns = t0.elapsed().as_nanos() as f64 / (REPEAT as f64 * N as f64);
        let want = fnv(&r);
        let (s, v) = (out[4 * op] == want, out[4 * op + 1] == want);
        good += (s && v) as u32;
        println!("ops: {op:>2} {name:<30} scalar {} {:>6.2} ns/element | HVX {} {:>6.2} ns/element ({:>5.1}x) | this CPU, one core {:>5.2} ns/element",
            if s { "correct" } else { "WRONG  " }, per(out[4 * op + 2], r.len()), if v { "correct" } else { "WRONG  " }, per(out[4 * op + 3], r.len()),
            out[4 * op + 2] as f64 / out[4 * op + 3].max(1) as f64, cpu_ns);
    }
    println!("ops: {good} of {} operations correct in both builds", NAMES.len());

    // Method 1: threads.
    let iters = 20_000_000u32;
    let mut want = 1u32;
    for _ in 0..iters { want = want.wrapping_mul(1664525).wrapping_add(1013904223).wrapping_add(want >> 7); }
    for k in [1u32, 2, 3, 4, 6, 8] {
        let mut o = vec![0u32; 18];
        let ret = call(1, &[k, iters], &mut o);
        let ok = (0..k as usize).filter(|&i| o[2 + 2 * i] == want).count();
        let mut ids: Vec<u32> = (0..k as usize).map(|i| o[3 + 2 * i]).collect();
        ids.dedup();
        println!("ops: {k} threads ret=0x{ret:x} err=0x{:x}: {:.3} s for {k} x {iters} steps, {ok} of {k} results correct, {} distinct thread ids", o[1], o[0] as f64 / 1e6, ids.len());
        if ret != 0 || o[1] != 0 { break; }
    }

    // Method 2: C library and compiler support routines supplied by the shell.
    let mut o = vec![0u32; 8];
    let ret = call(2, &[7, seed], &mut o);
    let mut x = seed;
    let mut next = || { x = x.wrapping_mul(1664525).wrapping_add(1013904223); x >> 24 };
    let mut v: Vec<u32> = (0..4096).map(|_| next() << 16 | next() << 8 | next()).collect();
    v.sort_unstable();
    let sorted = fnv(&v.iter().flat_map(|w| w.to_le_bytes()).collect::<Vec<u8>>());
    println!("ops: library ret=0x{ret:x}: qsort of 4096 numbers {} ({} us) | sqrtf(2)={} sinf(1)={} expf(1)={} | 64-bit divide {} | double multiply {} | thread id {}",
        if o[0] == sorted { "correct" } else { "WRONG" }, o[1], f32::from_bits(o[2]), f32::from_bits(o[3]), f32::from_bits(o[4]),
        if o[5] == (0x123456789abcdefu64 / 7) as u32 { "correct" } else { "WRONG" }, if o[6] == 3375 { "correct" } else { "WRONG" }, o[7]);
}
