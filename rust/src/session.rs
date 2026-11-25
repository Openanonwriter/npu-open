//! One persistent accelerator session on the DSP library `libnpu_skel.so` (see `npu/npu_skel.c`).
//!
//! DSP methods used: 0 open (acquire + power + describe layers), 1 load (one output group of one
//! layer), 2 run (batches of 32 rows through every layer), 3 close.
use crate::fastrpc::{invoke, remotectl_open, scalars, Arg, SharedBuf};
use crate::tiles::{f16_bits, pack_batch, pack_weights, tiles_for, unpack_row, TILE_BYTES, TILE_CELLS};
use std::time::Instant;

/// One fully connected layer: `y = f(x W^T + b)`, with `w` row-major [n_out][n_in].
/// `relu` selects f = max(x, 0) (one convert pass on the chip); otherwise f is the identity
/// (two convert passes, merged here on the CPU when it is the last layer).
#[derive(Clone)]
pub struct Layer {
    pub n_in: usize,
    pub n_out: usize,
    pub relu: bool,
    pub w: Vec<f32>,
    pub b: Vec<f32>,
}

#[derive(Clone, Copy, Default, Debug)]
pub struct Stats {
    /// Microseconds on the DSP clock inside the run calls (copies + accelerator).
    pub dsp_us: u64,
    /// Microseconds of wall time for the run calls, including the FastRPC round trips.
    pub wall_us: u64,
    /// Microseconds spent converting inputs to FP16 tiles on the CPU.
    pub pack_us: u64,
    /// Microseconds spent converting result tiles back to float32 on the CPU (not sent over the socket).
    pub unpack_us: u64,
    pub calls: u32,
}

struct Model {
    n_in: usize,
    n_out: usize,
    nk0: usize,
    og: usize,
    last_signed: bool,
    batches_per_call: usize,
}

pub struct Session {
    fd: i32,
    handle: u32,
    inb: SharedBuf,
    outb: SharedBuf,
    model: Option<Model>,
    pub vtcm_bytes: u32,
}

/// Bytes before the first output tile; 128 keeps the tiles aligned for the DSP's HVX copies.
const HEADER: usize = 128;
const IN_BYTES: usize = 4 << 20;
const OUT_BYTES: usize = 8 << 20;
const MAX_LAYERS: usize = 8;
/// run flags: bit 0 one "deep" multiply per up-to-32 tiles, bit 1 return both sign passes, bit 2 HVX copies.
const RUN_FLAGS: u32 = 7;
/// MIPS performance request and sleep-latency hint sent with open (see docs/TECHNICAL.md, section 9).
const MIPS: u32 = 2000;
const LATENCY_US: u32 = 100;

impl Session {
    /// Uses the FastRPC fd inherited from `hexagonrpcd -p` (environment variable HEXAGONRPC_FD).
    pub fn connect() -> Result<Session, String> {
        let fd: i32 = std::env::var("HEXAGONRPC_FD").map_err(|_| "HEXAGONRPC_FD not set: start through serve.sh".to_string())?
            .parse().map_err(|_| "bad HEXAGONRPC_FD".to_string())?;
        // hexagonrpcd registers its file-serving listener just after starting us; opening a library needs it.
        std::thread::sleep(std::time::Duration::from_secs(2));
        let handle = remotectl_open(fd, "npu")?;
        let inb = SharedBuf::new(fd, IN_BYTES, true).map_err(|e| format!("shared input buffer: {e}"))?;
        let outb = SharedBuf::new(fd, OUT_BYTES, true).map_err(|e| format!("shared output buffer: {e}"))?;
        Ok(Session { fd, handle, inb, outb, model: None, vtcm_bytes: 0 })
    }

    pub fn close(&mut self) {
        let w = [0u32; 1];
        let _ = invoke(self.fd, self.handle, scalars(3, 1, 0), &mut [Arg::input(&w)]);
        self.model = None;
    }

    /// Replaces the loaded network. Weights go to the accelerator's VTCM once.
    pub fn define(&mut self, layers: &[Layer]) -> Result<(), String> {
        if layers.is_empty() || layers.len() > MAX_LAYERS {
            return Err(format!("1 to {MAX_LAYERS} layers supported"));
        }
        for (i, l) in layers.iter().enumerate() {
            if l.w.len() != l.n_in * l.n_out || l.b.len() != l.n_out || l.n_in == 0 || l.n_out == 0 {
                return Err(format!("layer {i}: weight or bias size does not match {}x{}", l.n_out, l.n_in));
            }
            if i > 0 && tiles_for(l.n_in) != tiles_for(layers[i - 1].n_out) {
                return Err(format!("layer {i}: input width {} does not follow layer {} output width {}", l.n_in, i - 1, layers[i - 1].n_out));
            }
        }
        self.close();
        let mut opened = false;
        // VTCM pages come in powers of four; ask for 4 MB, then 1 MB.
        // 16 MB was tried on 2026-11-25 and refused (the DSP granted 4 MB); asking for it first only adds a failed attempt per model.
        for vt in [4u32 << 20, 1 << 20] {
            let mut words = [0u32; 28];
            words[0] = vt;
            words[1] = layers.len() as u32;
            for (i, l) in layers.iter().enumerate() {
                words[2 + i] = tiles_for(l.n_in) as u32;
                words[10 + i] = tiles_for(l.n_out) as u32;
                words[18 + i] = if l.relu { 0 } else { 1 };
            }
            words[26] = MIPS;
            words[27] = LATENCY_US;
            let mut st = [0u32; 8];
            let ret = invoke(self.fd, self.handle, scalars(0, 1, 1), &mut [Arg::input(&words), Arg::output(&mut st)]).map_err(|e| format!("open: {e}"))?;
            if ret == 0 && st[3] != 0 && st[3] != 0xdead_beef {
                self.vtcm_bytes = st[6];
                opened = true;
                break;
            }
        }
        if !opened {
            return Err("the accelerator session could not be opened: the model may not fit in VTCM, or the DSP is not ready".into());
        }
        for (li, l) in layers.iter().enumerate() {
            for g in 0..tiles_for(l.n_out) {
                let wt = pack_weights(&l.w, l.n_out, l.n_in, g);
                let mut bias = [0u16; 32];
                for o in 0..32 {
                    if 32 * g + o < l.n_out {
                        bias[o] = f16_bits(l.b[32 * g + o]);
                    }
                }
                let words = [li as u32, g as u32];
                let ret = invoke(self.fd, self.handle, scalars(1, 3, 0), &mut [Arg::input(&words), Arg::input(&wt), Arg::input(&bias)]).map_err(|e| format!("load: {e}"))?;
                if ret != 0 {
                    return Err(format!("load layer {li} group {g}: DSP returned 0x{ret:x}"));
                }
            }
        }
        let last = layers.last().unwrap();
        let (nk0, og, last_signed) = (tiles_for(layers[0].n_in), tiles_for(last.n_out), !last.relu);
        let per = if last_signed { 2 * og } else { og };
        let batches_per_call = 32.min(IN_BYTES / (nk0 * TILE_BYTES)).min((OUT_BYTES - HEADER) / (per * TILE_BYTES));
        if batches_per_call == 0 {
            return Err("layer too wide for the shared buffers".into());
        }
        self.model = Some(Model { n_in: layers[0].n_in, n_out: last.n_out, nk0, og, last_signed, batches_per_call });
        Ok(())
    }

    /// Runs `n` rows of `n_in` float32 values through the loaded network; returns `n` rows of `n_out`.
    pub fn run(&mut self, x: &[f32], n: usize) -> Result<(Vec<f32>, Stats), String> {
        let m = self.model.as_ref().ok_or("no model loaded")?;
        if x.len() != n * m.n_in {
            return Err(format!("expected {} values for {} rows of width {}", n * m.n_in, n, m.n_in));
        }
        let per = if m.last_signed { 2 * m.og } else { m.og };
        let (in_b, out_b) = (m.nk0 * TILE_BYTES, per * TILE_BYTES);
        let n_batches = (n + 31) / 32;
        let mut y = vec![0f32; n * m.n_out];
        let mut stats = Stats::default();
        let mut b0 = 0;
        while b0 < n_batches {
            let nb = (n_batches - b0).min(m.batches_per_call);
            let t = Instant::now();
            for b in 0..nb {
                let first = (b0 + b) * 32;
                let rows = (n - first).min(32);
                unsafe { pack_batch((self.inb.ptr as *mut u16).add(b * m.nk0 * TILE_CELLS), &x[first * m.n_in..(first + rows) * m.n_in], rows, m.n_in) };
            }
            stats.pack_us += t.elapsed().as_micros() as u64;
            let words = [nb as u32, RUN_FLAGS];
            let t = Instant::now();
            let ret = invoke(self.fd, self.handle, scalars(2, 2, 1), &mut [Arg::input(&words), Arg::shared(&self.inb, nb * in_b), Arg::shared(&self.outb, HEADER + nb * out_b)])
                .map_err(|e| format!("run: {e}"))?;
            stats.wall_us += t.elapsed().as_micros() as u64;
            stats.calls += 1;
            let st = unsafe { std::slice::from_raw_parts(self.outb.ptr as *const u32, 8) };
            if ret != 0 || st[0] != 0 || st[3] != 0 {
                return Err(format!("run: DSP returned 0x{ret:x}, lock 0x{:x}, unlock 0x{:x}", st[0], st[3]));
            }
            stats.dsp_us += st[1] as u64;
            let tiles = unsafe { std::slice::from_raw_parts(self.outb.ptr.add(HEADER) as *const u16, nb * per * TILE_CELLS) };
            // NEON: one 128-byte row (two samples x 32 channels) at a time; see tiles::unpack_row
            let t = Instant::now();
            let groups = (m.n_out + 31) / 32;
            for b in 0..nb {
                let first = (b0 + b) * 32;
                let rows = (n - first).min(32);
                for j in 0..(rows + 1) / 2 {
                    for g in 0..groups {
                        let pos = unsafe { tiles.as_ptr().add((b * per + g) * TILE_CELLS + 64 * j) };
                        let neg = if m.last_signed { unsafe { tiles.as_ptr().add((b * per + m.og + g) * TILE_CELLS + 64 * j) } } else { std::ptr::null() };
                        let (mut ya, mut yb) = ([0f32; 32], [0f32; 32]);
                        unsafe { unpack_row(pos, neg, &mut ya, &mut yb) };
                        let width = (m.n_out - 32 * g).min(32);
                        let at = (first + 2 * j) * m.n_out + 32 * g;
                        y[at..at + width].copy_from_slice(&ya[..width]);
                        if 2 * j + 1 < rows {
                            y[at + m.n_out..at + m.n_out + width].copy_from_slice(&yb[..width]);
                        }
                    }
                }
            }
            stats.unpack_us += t.elapsed().as_micros() as u64;
            b0 += nb;
        }
        Ok((y, stats))
    }
}
