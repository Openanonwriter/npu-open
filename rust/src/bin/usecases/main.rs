//! Demonstrations that run on the NPU through `npu-server`. No model here is trained by gradient
//! descent: weights are random, analytic (DFT, templates), or fitted in closed form (PCA, least squares).
//!
//! usage: usecases [math|cyber|net|radio|all|real|live|radioid|<number>...|quit]
//! NPU_DIRECT=1 (started through run.sh, server stopped) runs the in-process radio case 134 instead.
mod cyber;
mod math;
mod net;
mod radio;
mod radioid;
mod real;
mod util;

use npu::proto::Client;

type Case = (u32, &'static str, fn(&mut Client) -> Result<(), String>);

fn cases() -> Vec<Case> {
    let mut v: Vec<Case> = Vec::new();
    v.extend(math::CASES);
    v.extend(cyber::CASES);
    v.extend(net::CASES);
    v.extend(radio::CASES);
    v.extend(real::CASES);
    v.extend(real::CASES2);
    v.extend(real::CASES3);
    v.extend(real::CASES4);
    v.extend(real::CASES5);
    v.extend(real::CASES6);
    v.extend(real::CASES7);
    v.extend(real::CASES8);
    v.extend(real::CASES9);
    v.extend(real::CASES10);
    v.extend(real::CASES11);
    v.extend(real::CASES12);
    v.extend(radioid::CASES);
    v
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if std::env::var("NPU_DIRECT").is_ok() {
        // In-process run without the server: start through run.sh (see radioid::direct).
        if let Err(e) = radioid::direct() {
            println!("134 | FAILED: {e}");
        }
        return;
    }
    let mut c = match Client::connect() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
    };
    if args.first().map(|s| s.as_str()) == Some("quit") {
        c.quit();
        return;
    }
    let want = |id: u32, area: &str| (args.is_empty() && area != "real" && area != "live") || args.iter().any(|a| (a == "all" && area != "real" && area != "live") || a == area || a.parse::<u32>() == Ok(id));
    let mut failed = 0;
    for (id, area, f) in cases() {
        if want(id, area) {
            if let Err(e) = f(&mut c) {
                println!("{id:>2} | FAILED: {e}");
                failed += 1;
            }
        }
    }
    std::process::exit(if failed > 0 { 1 } else { 0 });
}
