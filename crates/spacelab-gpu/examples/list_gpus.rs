//! Enumerate the GPU adapters `wgpu` can reach here — the feasibility probe for the compute backend.
//!
//! `cargo run -p spacelab-gpu --example list_gpus`

fn main() {
    let instance = wgpu::Instance::default();
    let adapters = pollster::block_on(instance.enumerate_adapters(wgpu::Backends::all()));
    println!("{} GPU adapter(s) reachable via wgpu:", adapters.len());
    for a in &adapters {
        let info = a.get_info();
        println!(
            "  [{:?}] {} — backend {:?}, driver {}",
            info.device_type, info.name, info.backend, info.driver
        );
    }
}
