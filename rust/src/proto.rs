//! Unix-socket protocol between `npu-server` and its clients. All integers little-endian.
//!
//! | Request | Body | Reply after the status word |
//! |---|---|---|
//! | `M` define | u32 layers; per layer u32 n_in, n_out, relu; then per layer f32 w[n_out*n_in], f32 b[n_out] | u32 VTCM kilobytes |
//! | `R` run | u32 rows; f32 x[rows*n_in] | u64 dsp_us, u64 wall_us, u64 pack_us, u32 calls, u32 n_out, f32 y[rows*n_out] |
//! | `Q` quit | none | none |
//!
//! Every reply starts with u32 status: 0 ok, 1 error followed by u32 length and a UTF-8 message.
use crate::session::{Layer, Stats};
use std::io::{self, Read, Write};
use std::os::unix::net::UnixStream;
use std::path::PathBuf;

/// `$NPU_SOCK`, else `$NPU_OPEN/npu.sock`, else `./npu.sock`.
pub fn socket_path() -> PathBuf {
    if let Ok(p) = std::env::var("NPU_SOCK") {
        return p.into();
    }
    PathBuf::from(std::env::var("NPU_OPEN").unwrap_or_else(|_| ".".into())).join("npu.sock")
}

pub fn read_u32(r: &mut impl Read) -> io::Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}
pub fn read_u64(r: &mut impl Read) -> io::Result<u64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}
pub fn read_f32s(r: &mut impl Read, n: usize) -> io::Result<Vec<f32>> {
    let mut v = vec![0f32; n];
    // f32 is plain data and this project is little-endian aarch64 only
    r.read_exact(unsafe { std::slice::from_raw_parts_mut(v.as_mut_ptr() as *mut u8, n * 4) })?;
    Ok(v)
}
pub fn write_f32s(w: &mut impl Write, v: &[f32]) -> io::Result<()> {
    w.write_all(unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) })
}

pub struct Client {
    s: UnixStream,
}

impl Client {
    pub fn connect() -> Result<Client, String> {
        let p = socket_path();
        UnixStream::connect(&p).map(|s| Client { s }).map_err(|e| format!("cannot reach the NPU server at {}: {e} (start it with ./serve.sh start)", p.display()))
    }

    fn status(&mut self) -> Result<(), String> {
        match read_u32(&mut self.s).map_err(|e| e.to_string())? {
            0 => Ok(()),
            _ => {
                let n = read_u32(&mut self.s).map_err(|e| e.to_string())? as usize;
                let mut b = vec![0u8; n];
                self.s.read_exact(&mut b).map_err(|e| e.to_string())?;
                Err(String::from_utf8_lossy(&b).into_owned())
            }
        }
    }

    /// Loads a network; returns the VTCM size in kilobytes.
    pub fn define(&mut self, layers: &[Layer]) -> Result<u32, String> {
        let io = |e: io::Error| e.to_string();
        self.s.write_all(b"M").map_err(io)?;
        self.s.write_all(&(layers.len() as u32).to_le_bytes()).map_err(io)?;
        for l in layers {
            for v in [l.n_in as u32, l.n_out as u32, l.relu as u32] {
                self.s.write_all(&v.to_le_bytes()).map_err(io)?;
            }
        }
        for l in layers {
            write_f32s(&mut self.s, &l.w).map_err(io)?;
            write_f32s(&mut self.s, &l.b).map_err(io)?;
        }
        self.status()?;
        read_u32(&mut self.s).map_err(io)
    }

    /// One linear map `y = x W^T + b` (signed output), or with `relu` `y = max(x W^T + b, 0)`.
    pub fn linear(&mut self, w: Vec<f32>, n_out: usize, n_in: usize, relu: bool) -> Result<u32, String> {
        self.define(&[Layer { n_in, n_out, relu, w, b: vec![0.0; n_out] }])
    }

    pub fn run(&mut self, x: &[f32], rows: usize) -> Result<(Vec<f32>, Stats), String> {
        let io = |e: io::Error| e.to_string();
        self.s.write_all(b"R").map_err(io)?;
        self.s.write_all(&(rows as u32).to_le_bytes()).map_err(io)?;
        write_f32s(&mut self.s, x).map_err(io)?;
        self.status()?;
        let st = Stats { dsp_us: read_u64(&mut self.s).map_err(io)?, wall_us: read_u64(&mut self.s).map_err(io)?, pack_us: read_u64(&mut self.s).map_err(io)?, calls: read_u32(&mut self.s).map_err(io)? };
        let n_out = read_u32(&mut self.s).map_err(io)? as usize;
        Ok((read_f32s(&mut self.s, rows * n_out).map_err(io)?, st))
    }

    pub fn quit(mut self) {
        let _ = self.s.write_all(b"Q");
    }
}
