//! The accelerator's FP16 tile layout, and conversion to and from float32.
//!
//! A tile is 2 KB: 32 rows of 128 bytes. Each row holds two samples interleaved value by value:
//! row j = { s[2j][0], s[2j+1][0], s[2j][1], s[2j+1][1], ... } for 32 values per sample.
//! Activations and outputs use (sample, channel); weights use the same shape with the roles
//! (input channel, output channel). Found from Qualcomm's public HMX manual and confirmed by test.
use std::arch::asm;

pub const TILE_BYTES: usize = 2048;
pub const TILE_CELLS: usize = 1024;

/// Converts 4 + 4 floats to FP16 and interleaves them into 8 halves (NEON, base ARMv8).
#[inline(always)]
unsafe fn cvt_zip4(a: *const f32, b: *const f32, out: *mut u16) {
    asm!(
        "ldr q0, [{a}]", "ldr q1, [{b}]",
        "fcvtn v0.4h, v0.4s", "fcvtn v1.4h, v1.4s",
        "zip1 v0.8h, v0.8h, v1.8h",
        "str q0, [{o}]",
        a = in(reg) a, b = in(reg) b, o = in(reg) out,
        out("v0") _, out("v1") _, options(nostack),
    );
}

/// float32 -> FP16 bits, round to nearest even (uses the same NEON conversion).
pub fn f16_bits(v: f32) -> u16 {
    let a = [v, 0.0, 0.0, 0.0];
    let mut out = [0u16; 8];
    unsafe { cvt_zip4(a.as_ptr(), a.as_ptr(), out.as_mut_ptr()) };
    out[0]
}

/// FP16 bits -> float32 (exact).
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h as u32) & 0x8000) << 16;
    let exp = ((h >> 10) & 0x1f) as u32;
    let frac = (h & 0x3ff) as u32;
    let bits = if exp == 0 {
        if frac == 0 {
            sign
        } else {
            // subnormal: normalise
            let shift = frac.leading_zeros() - 21;
            sign | ((113 - shift) << 23) | ((frac << (shift + 13)) & 0x007f_ffff)
        }
    } else if exp == 31 {
        sign | 0x7f80_0000 | (frac << 13)
    } else {
        sign | ((exp + 112) << 23) | (frac << 13)
    };
    f32::from_bits(bits)
}

pub fn tiles_for(n: usize) -> usize {
    (n + 31) / 32
}

/// Fills the activation tiles for one batch of up to 32 rows of `n_feat` float32 values.
/// `tiles` must hold `tiles_for(n_feat) * 1024` cells. Missing rows and features are zero.
///
/// # Safety
/// `tiles` must be valid for that many u16 writes (it usually points into a shared buffer).
pub unsafe fn pack_batch(tiles: *mut u16, x: &[f32], n_rows: usize, n_feat: usize) {
    let zero = [0f32; 32];
    let nk = tiles_for(n_feat);
    let n_rows = n_rows.min(32);
    for j in 0..16 {
        for k in 0..nk {
            let have = (n_feat - 32 * k).min(32);
            let mut ta = [0f32; 32];
            let mut tb = [0f32; 32];
            let mut a = zero.as_ptr();
            let mut b = zero.as_ptr();
            if 2 * j < n_rows {
                let row = &x[2 * j * n_feat + 32 * k..];
                if have == 32 { a = row.as_ptr() } else { ta[..have].copy_from_slice(&row[..have]); a = ta.as_ptr() }
            }
            if 2 * j + 1 < n_rows {
                let row = &x[(2 * j + 1) * n_feat + 32 * k..];
                if have == 32 { b = row.as_ptr() } else { tb[..have].copy_from_slice(&row[..have]); b = tb.as_ptr() }
            }
            let out = tiles.add(k * TILE_CELLS + 64 * j);
            for c in (0..32).step_by(4) {
                cvt_zip4(a.add(c), b.add(c), out.add(2 * c));
            }
        }
    }
}

/// Weight tiles for output group `g` of a layer: `w` is row-major [n_out][n_in] float32.
/// Cell q of tile k holds input channel 32k + 2*(q/64) + q%2 and output channel 32g + (q%64)/2.
pub fn pack_weights(w: &[f32], n_out: usize, n_in: usize, g: usize) -> Vec<u16> {
    let nk = tiles_for(n_in);
    let mut t = vec![0u16; nk * TILE_CELLS];
    for k in 0..nk {
        for q in 0..TILE_CELLS {
            let i = 32 * k + 2 * (q / 64) + q % 2;
            let o = 32 * g + (q % 64) / 2;
            if o < n_out && i < n_in {
                t[k * TILE_CELLS + q] = f16_bits(w[o * n_in + i]);
            }
        }
    }
    t
}

/// Index of (sample s, channel c) inside one output tile.
#[inline(always)]
pub fn cell(s: usize, c: usize) -> usize {
    64 * (s / 2) + 2 * c + s % 2
}
