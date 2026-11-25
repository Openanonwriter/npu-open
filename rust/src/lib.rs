//! Linux side of the open Hexagon NPU path, standard library only.
//!
//! - `fastrpc`: the kernel's FastRPC ioctls (`/usr/include/misc/fastrpc.h`), called directly.
//! - `tiles`: conversion between float32 rows and the accelerator's FP16 tile layout.
//! - `session`: one persistent accelerator session on the DSP library `libnpu_skel.so`.
//! - `proto`: the Unix-socket protocol between `npu-server` and its clients.
pub mod fastrpc;
pub mod proto;
pub mod session;
pub mod tiles;
