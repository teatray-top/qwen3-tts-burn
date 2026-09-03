//! Does quantised matmul work on this Vulkan device, and is it worth it?
//!
//! The talker is 3.45 GB of the 3.86 GB of weights, so a smaller weight format
//! is the only lever on memory that matters. This measures, per scheme, whether
//! the backend runs it at all, how far the result drifts from f16, and what a
//! matmul of the shape the talker actually issues costs.
//!
//! ```text
//! cargo run --release --example quant_probe
//! ```

use std::time::Instant;

use burn::tensor::{Distribution, Tensor};
use cubecl::quant::scheme::{QuantLevel, QuantMode, QuantScheme, QuantValue};
use qwen3_tts_burn::VulkanBackend as Vk;

// One decode step of the talker: [1, 2048] against a [2048, 6144] projection.
const IN: usize = 2048;
const OUT: usize = 6144;
const ITERS: usize = 50;

fn to_f32(t: Tensor<Vk, 2>) -> Vec<f32> {
    t.into_data()
        .convert::<f32>()
        .to_vec()
        .expect("read tensor")
}

fn main() {
    let dev = Default::default();
    let w: Tensor<Vk, 2> = Tensor::random([IN, OUT], Distribution::Normal(0.0, 0.02), &dev);
    let x: Tensor<Vk, 2> = Tensor::random([1, IN], Distribution::Normal(0.0, 1.0), &dev);

    let base = x.clone().matmul(w.clone());
    let base_v: Vec<f32> = to_f32(base.clone());
    let norm = base_v.iter().map(|v| v * v).sum::<f32>().sqrt();

    let t = Instant::now();
    for _ in 0..ITERS {
        let _ = x.clone().matmul(w.clone()).into_data();
    }
    let f16_ms = t.elapsed().as_secs_f64() * 1000.0 / ITERS as f64;
    println!(
        "f16          {f16_ms:6.2} ms/matmul   {:.1} MB weights",
        (IN * OUT * 2) as f64 / 1e6
    );

    let schemes = [
        ("Q8S per-tensor", QuantValue::Q8S, QuantLevel::Tensor),
        ("Q8S block 128", QuantValue::Q8S, QuantLevel::block([128])),
        ("Q4S block 128", QuantValue::Q4S, QuantLevel::block([128])),
        ("Q4S block 32", QuantValue::Q4S, QuantLevel::block([32])),
        ("Q2S block 32", QuantValue::Q2S, QuantLevel::block([32])),
        ("E4M3 per-tensor", QuantValue::E4M3, QuantLevel::Tensor),
    ];

    for (name, value, level) in schemes {
        let scheme = QuantScheme::default()
            .with_value(value)
            .with_level(level)
            .with_mode(QuantMode::Symmetric);
        let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let qw = w.clone().quantize_dynamic(&scheme);
            let out = x.clone().matmul(qw.clone());
            let v: Vec<f32> = to_f32(out);
            let t = Instant::now();
            for _ in 0..ITERS {
                let _ = x.clone().matmul(qw.clone()).into_data();
            }
            (v, t.elapsed().as_secs_f64() * 1000.0 / ITERS as f64)
        }));
        match res {
            Ok((v, ms)) => {
                let err = base_v
                    .iter()
                    .zip(&v)
                    .map(|(a, b)| (a - b) * (a - b))
                    .sum::<f32>()
                    .sqrt()
                    / norm;
                let bits = value.size_bits();
                let mb = (IN * OUT * bits / 8) as f64 / 1e6;
                println!(
                    "{name:16} {ms:6.2} ms/matmul   {mb:5.1} MB weights   \
                     relative error {:.4}   {:.2}x f16 speed",
                    err,
                    f16_ms / ms
                );
            }
            Err(_) => println!("{name:16} 실행 불가 (backend panic)"),
        }
    }
}
