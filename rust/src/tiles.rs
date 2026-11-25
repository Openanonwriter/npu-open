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
    // Row r of tile k holds input channels i0 = 32k + 2r and i0 + 1, interleaved over the 32 outputs of the
    // group: { w[o0][i0], w[o0][i0+1], w[o1][i0], w[o1][i0+1], ... }. In row-major [n_out][n_in] storage the two
    // values of each pair are neighbours, so gather the 64 floats of a row and convert them eight at a time.
    let nk = tiles_for(n_in);
    let mut t = vec![0u16; nk * TILE_CELLS];
    let outs = (n_out - 32 * g).min(32);
    for k in 0..nk {
        for r in 0..32 {
            let i0 = 32 * k + 2 * r;
            if i0 >= n_in {
                break;
            }
            let pair = i0 + 1 < n_in;
            let (mut a, mut b) = ([0f32; 32], [0f32; 32]);
            for o in 0..outs {
                let row = (32 * g + o) * n_in + i0;
                a[o] = w[row];
                if pair {
                    b[o] = w[row + 1];
                }
            }
            let out = unsafe { t.as_mut_ptr().add(k * TILE_CELLS + 64 * r) };
            for c in (0..32).step_by(4) {
                unsafe { cvt_zip4(a.as_ptr().add(c), b.as_ptr().add(c), out.add(2 * c)) };
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

/// Converts one 128-byte output row (two samples x 32 channels, interleaved) to float32:
/// `a` gets the 32 channels of the even sample, `b` those of the odd sample.
/// With `neg` (the second convert pass, holding max(-x, 0)), cells that are zero in `pos` take the
/// negated value from `neg`; exactly one of the two is non-zero per cell.
///
/// # Safety
/// `pos` (and `neg` if not null) must point to 64 readable u16 values.
pub unsafe fn unpack_row(pos: *const u16, neg: *const u16, a: &mut [f32; 32], b: &mut [f32; 32]) {
    for c in (0..32).step_by(4) {
        if neg.is_null() {
            asm!(
                "ldr q0, [{p}]",
                "uzp1 v5.8h, v0.8h, v0.8h", "uzp2 v6.8h, v0.8h, v0.8h",
                "fcvtl v5.4s, v5.4h", "fcvtl v6.4s, v6.4h",
                "str q5, [{a}]", "str q6, [{b}]",
                p = in(reg) pos.add(2 * c), a = in(reg) a.as_mut_ptr().add(c), b = in(reg) b.as_mut_ptr().add(c),
                out("v0") _, out("v5") _, out("v6") _, options(nostack),
            );
        } else {
            asm!(
                "ldr q0, [{p}]", "ldr q1, [{n}]",
                "cmeq v2.8h, v0.8h, #0",          // where the first pass is zero
                "cmtst v3.8h, v1.8h, v1.8h",      // where the second pass is non-zero
                "movi v4.8h, #0x80, lsl #8",      // sign bit
                "orr v1.16b, v1.16b, v4.16b", "and v1.16b, v1.16b, v3.16b",
                "bsl v2.16b, v1.16b, v0.16b",     // zero in pass one: take the negated pass two
                "uzp1 v5.8h, v2.8h, v2.8h", "uzp2 v6.8h, v2.8h, v2.8h",
                "fcvtl v5.4s, v5.4h", "fcvtl v6.4s, v6.4h",
                "str q5, [{a}]", "str q6, [{b}]",
                p = in(reg) pos.add(2 * c), n = in(reg) neg.add(2 * c), a = in(reg) a.as_mut_ptr().add(c), b = in(reg) b.as_mut_ptr().add(c),
                out("v0") _, out("v1") _, out("v2") _, out("v3") _, out("v4") _, out("v5") _, out("v6") _, options(nostack),
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_conversions_agree_and_round_trip() {
        for &v in &[0.0f32, 1.0, -1.0, 0.5, 16.0, 65504.0, -65504.0, 1e-4, 3.14159, -2.71828, 6.1e-5, 5.96e-8] {
            let h = f16_bits(v);
            let back = f16_to_f32(h);
            assert_eq!(f16_bits(back), h, "round trip of {v}");
            assert!((back - v).abs() <= v.abs() * 1e-3 + 1e-7, "{v} -> {back}");
        }
        assert_eq!(f16_bits(1.0), 0x3c00);
        assert_eq!(f16_bits(16.0), 0x4c00);
        assert_eq!(f16_bits(-1.0), 0xbc00);
        assert_eq!(f16_to_f32(0x0001), 5.960_464_5e-8); // smallest subnormal
        assert!(f16_to_f32(0x7c00).is_infinite());
    }

    #[test]
    fn pack_matches_the_cell_formula_for_padded_shapes() {
        for &(rows, feat) in &[(32usize, 784usize), (32, 64), (32, 10), (16, 784), (7, 100), (1, 33)] {
            let x: Vec<f32> = (0..rows * feat).map(|i| ((i * 37 % 201) as f32 - 100.0) / 7.0).collect();
            let nk = tiles_for(feat);
            let mut t = vec![0xffffu16; nk * TILE_CELLS];
            unsafe { pack_batch(t.as_mut_ptr(), &x, rows, feat) };
            for k in 0..nk {
                for p in 0..TILE_CELLS {
                    let (s, i) = (2 * (p / 64) + p % 2, 32 * k + (p % 64) / 2);
                    let want = if s < rows && i < feat { f16_bits(x[s * feat + i]) } else { 0 };
                    assert_eq!(t[k * TILE_CELLS + p], want, "rows {rows} feat {feat} tile {k} cell {p}");
                }
            }
        }
    }

    #[test]
    fn unpack_splits_samples_and_merges_sign_passes() {
        let vals: Vec<f32> = (0..64).map(|i| if i % 5 == 0 { 0.0 } else { (i as f32 - 30.0) * 0.75 }).collect();
        let (mut pos, mut neg) = ([0u16; 64], [0u16; 64]);
        for (i, &v) in vals.iter().enumerate() {
            pos[i] = f16_bits(v.max(0.0));
            neg[i] = f16_bits((-v).max(0.0));
        }
        let (mut a, mut b) = ([0f32; 32], [0f32; 32]);
        unsafe { unpack_row(pos.as_ptr(), neg.as_ptr(), &mut a, &mut b) };
        for c in 0..32 {
            assert_eq!(a[c], vals[2 * c], "even sample channel {c}");
            assert_eq!(b[c], vals[2 * c + 1], "odd sample channel {c}");
        }
        unsafe { unpack_row(pos.as_ptr(), std::ptr::null(), &mut a, &mut b) };
        for c in 0..32 {
            assert_eq!(a[c], vals[2 * c].max(0.0));
            assert_eq!(b[c], vals[2 * c + 1].max(0.0));
        }
    }

    #[test]
    fn weights_use_the_transposed_roles() {
        let (n_out, n_in) = (40usize, 70usize);
        let w: Vec<f32> = (0..n_out * n_in).map(|i| i as f32 * 0.01).collect();
        for g in 0..tiles_for(n_out) {
            let t = pack_weights(&w, n_out, n_in, g);
            for k in 0..tiles_for(n_in) {
                for q in 0..TILE_CELLS {
                    let (i, o) = (32 * k + 2 * (q / 64) + q % 2, 32 * g + (q % 64) / 2);
                    let want = if o < n_out && i < n_in { f16_bits(w[o * n_in + i]) } else { 0 };
                    assert_eq!(t[k * TILE_CELLS + q], want);
                }
            }
        }
        assert_eq!(cell(5, 7), 64 * 2 + 14 + 1);
    }
}
