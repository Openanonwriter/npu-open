//! npu-server: keeps one accelerator session open and serves define/run requests on a Unix socket.
//! Started by `serve.sh` through `hexagonrpcd -p`, which hands it the FastRPC fd.
use npu::proto::{read_f32s, read_u32, socket_path, write_f32s};
use npu::session::{Layer, Session};
use std::io::{Read, Write};
use std::os::unix::net::{UnixListener, UnixStream};

fn fail(s: &mut UnixStream, msg: &str) -> std::io::Result<()> {
    s.write_all(&1u32.to_le_bytes())?;
    s.write_all(&(msg.len() as u32).to_le_bytes())?;
    s.write_all(msg.as_bytes())
}

/// Serves one client until it disconnects. Returns true if it asked the server to quit.
fn serve(sess: &mut Session, s: &mut UnixStream, widths: &mut (usize, usize)) -> std::io::Result<bool> {
    loop {
        let mut op = [0u8; 1];
        if s.read(&mut op)? == 0 {
            return Ok(false);
        }
        match op[0] {
            b'M' => {
                let nl = read_u32(s)? as usize;
                let mut shapes = Vec::new();
                for _ in 0..nl.min(64) {
                    shapes.push((read_u32(s)? as usize, read_u32(s)? as usize, read_u32(s)? != 0));
                }
                let mut layers = Vec::new();
                for &(n_in, n_out, relu) in &shapes {
                    let w = read_f32s(s, n_in * n_out)?;
                    let b = read_f32s(s, n_out)?;
                    layers.push(Layer { n_in, n_out, relu, w, b });
                }
                match sess.define(&layers) {
                    Ok(()) => {
                        *widths = (layers[0].n_in, layers[layers.len() - 1].n_out);
                        s.write_all(&0u32.to_le_bytes())?;
                        s.write_all(&(sess.vtcm_bytes / 1024).to_le_bytes())?;
                    }
                    Err(e) => fail(s, &e)?,
                }
            }
            b'R' => {
                let rows = read_u32(s)? as usize;
                let x = read_f32s(s, rows * widths.0)?;
                match sess.run(&x, rows) {
                    Ok((y, st)) => {
                        s.write_all(&0u32.to_le_bytes())?;
                        for v in [st.dsp_us, st.wall_us, st.pack_us] {
                            s.write_all(&v.to_le_bytes())?;
                        }
                        s.write_all(&st.calls.to_le_bytes())?;
                        s.write_all(&(widths.1 as u32).to_le_bytes())?;
                        write_f32s(s, &y)?;
                    }
                    Err(e) => fail(s, &e)?,
                }
            }
            b'Q' => return Ok(true),
            other => {
                fail(s, &format!("unknown request {other:#x}"))?;
                return Ok(false);
            }
        }
    }
}

fn main() {
    let mut sess = match Session::connect() {
        Ok(s) => s,
        Err(e) => {
            eprintln!("npu-server: {e}");
            std::process::exit(1);
        }
    };
    let path = socket_path();
    let _ = std::fs::remove_file(&path);
    let listener = UnixListener::bind(&path).unwrap_or_else(|e| {
        eprintln!("npu-server: cannot bind {}: {e}", path.display());
        std::process::exit(1);
    });
    eprintln!("npu-server: ready on {}", path.display());
    let mut widths = (0usize, 0usize);
    for conn in listener.incoming() {
        let Ok(mut s) = conn else { continue };
        match serve(&mut sess, &mut s, &mut widths) {
            Ok(true) => break,
            Ok(false) => {}
            Err(e) => eprintln!("npu-server: client error: {e}"),
        }
    }
    sess.close();
    let _ = std::fs::remove_file(&path);
    eprintln!("npu-server: stopped");
}
