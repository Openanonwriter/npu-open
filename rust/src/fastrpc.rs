//! FastRPC ioctls, straight from the kernel UAPI header `misc/fastrpc.h` (no libhexagonrpc).
//!
//! A call has `n_in` input buffers followed by `n_out` output buffers. The scalars word is
//! `method << 24 | n_in << 16 | n_out << 8`. The ioctl's return value is the DSP's return code
//! (0 on success, DSP errors are 0x80000400 + code); -1 with errno means the kernel refused.
use std::io;
use std::os::raw::{c_int, c_ulong, c_void};

extern "C" {
    fn ioctl(fd: c_int, request: c_ulong, ...) -> c_int;
    fn mmap(addr: *mut c_void, len: usize, prot: c_int, flags: c_int, fd: c_int, off: i64) -> *mut c_void;
}

/// `_IOWR('R', nr, size)` on the generic ioctl encoding (arm64): dir 3 in bits 31:30, size in 29:16.
const fn iowr(nr: u64, size: u64) -> c_ulong {
    ((3u64 << 30) | (size << 16) | ((b'R' as u64) << 8) | nr) as c_ulong
}

/// `struct fastrpc_invoke_args`
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Arg {
    pub ptr: u64,
    pub length: u64,
    /// dma-buf fd for a shared buffer (the kernel maps it instead of copying), or -1.
    pub fd: i32,
    pub attr: u32,
}

impl Arg {
    pub fn input<T>(data: &[T]) -> Arg {
        Arg { ptr: data.as_ptr() as u64, length: std::mem::size_of_val(data) as u64, fd: -1, attr: 0 }
    }
    pub fn output<T>(data: &mut [T]) -> Arg {
        Arg { ptr: data.as_mut_ptr() as u64, length: std::mem::size_of_val(data) as u64, fd: -1, attr: 0 }
    }
    pub fn shared(buf: &SharedBuf, bytes: usize) -> Arg {
        Arg { ptr: buf.ptr as u64, length: bytes as u64, fd: buf.fd, attr: 0 }
    }
}

#[repr(C)]
struct Invoke {
    handle: u32,
    sc: u32,
    args: u64,
}
#[repr(C)]
struct AllocDmaBuf {
    fd: i32,
    flags: u32,
    size: u64,
}
/// `struct fastrpc_mem_map` (64 bytes with padding).
#[repr(C)]
struct MemMap {
    version: i32,
    fd: i32,
    offset: i32,
    flags: u32,
    vaddrin: u64,
    length: u64,
    vaddrout: u64,
    attrs: i32,
    reserved: [i32; 4],
}

const IOCTL_ALLOC_DMA_BUFF: c_ulong = iowr(1, std::mem::size_of::<AllocDmaBuf>() as u64);
const IOCTL_INVOKE: c_ulong = iowr(3, std::mem::size_of::<Invoke>() as u64);
const IOCTL_MEM_MAP: c_ulong = iowr(10, std::mem::size_of::<MemMap>() as u64);

pub fn scalars(method: u32, n_in: u32, n_out: u32) -> u32 {
    ((method & 0x1f) << 24) | ((n_in & 0xff) << 16) | ((n_out & 0xff) << 8)
}

/// Returns the DSP's return code. `args` holds the input buffers first, then the output buffers.
pub fn invoke(fd: i32, handle: u32, sc: u32, args: &mut [Arg]) -> io::Result<u32> {
    let mut inv = Invoke { handle, sc, args: args.as_mut_ptr() as u64 };
    let r = unsafe { ioctl(fd, IOCTL_INVOKE, &mut inv as *mut Invoke) };
    if r == -1 {
        return Err(io::Error::last_os_error());
    }
    Ok(r as u32)
}

/// Opens interface `name`: the DSP loads `lib<name>_skel.so` and returns a handle.
/// This is method 0 of the built-in `remotectl` interface (handle 0). Buffer layout, as
/// libhexagonrpc builds it: in { name_len, err_len }, in name, out { handle, dlerr }, out err text.
pub fn remotectl_open(fd: i32, name: &str) -> Result<u32, String> {
    let mut cname = name.as_bytes().to_vec();
    cname.push(0);
    let mut err = [0u8; 256];
    let lens = [cname.len() as u32, err.len() as u32];
    let mut out = [0u32; 2];
    let mut args = [Arg::input(&lens), Arg::input(&cname), Arg::output(&mut out), Arg::output(&mut err)];
    match invoke(fd, 0, scalars(0, 2, 2), &mut args) {
        Ok(0) if out[1] == 0 => Ok(out[0]),
        Ok(ret) => Err(format!("open {name}: ret=0x{ret:x} dlerr=0x{:x}", out[1])),
        Err(e) => Err(format!("open {name}: {e}")),
    }
}

/// A dma-buf allocated by the FastRPC driver and mapped here. Passed to a call by fd, the kernel
/// maps it for the DSP instead of copying. `persistent` keeps that mapping alive between calls.
pub struct SharedBuf {
    pub ptr: *mut u8,
    pub len: usize,
    pub fd: i32,
}
unsafe impl Send for SharedBuf {}

impl SharedBuf {
    pub fn new(fd: i32, bytes: usize, persistent: bool) -> io::Result<SharedBuf> {
        let mut db = AllocDmaBuf { fd: -1, flags: 0, size: ((bytes + 4095) & !4095) as u64 };
        if unsafe { ioctl(fd, IOCTL_ALLOC_DMA_BUFF, &mut db as *mut AllocDmaBuf) } != 0 {
            return Err(io::Error::last_os_error());
        }
        const PROT_READ_WRITE: c_int = 3;
        const MAP_SHARED: c_int = 1;
        let p = unsafe { mmap(std::ptr::null_mut(), db.size as usize, PROT_READ_WRITE, MAP_SHARED, db.fd, 0) };
        if p as isize == -1 {
            return Err(io::Error::last_os_error());
        }
        if persistent {
            // flags 0 = FASTRPC_MAP_STATIC
            let mut mm = MemMap { version: 0, fd: db.fd, offset: 0, flags: 0, vaddrin: p as u64, length: db.size, vaddrout: 0, attrs: 0, reserved: [0; 4] };
            if unsafe { ioctl(fd, IOCTL_MEM_MAP, &mut mm as *mut MemMap) } != 0 {
                eprintln!("npu-server: persistent map refused ({}), continuing with per-call mapping", io::Error::last_os_error());
            }
        }
        Ok(SharedBuf { ptr: p as *mut u8, len: db.size as usize, fd: db.fd })
    }
}
