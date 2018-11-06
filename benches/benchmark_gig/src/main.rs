//! To squeeze the absolute most out of this benchmark, with optimizations specific to the current
//! machine, try this:
//!
//!     RUSTFLAGS="-C target-cpu=native -C target-feature=-avx2" cargo +nightly run --release --bin benchmark_gig
//!
//! Note that we're *disabling* AVX2 with target-feature. I find that when it's enabled, the
//! portable implementation ends up *much* slower. Our AVX2 compress function will compile with
//! AVX2 regardless, because of its local annotations. Also the nightly compiler seems to produce
//! faster code than stable.

extern crate blake2s_simd;

use std::time::{Duration, Instant};

const NS_PER_SEC: u64 = 1_000_000_000;
const INPUT_LEN: usize = 1_000_000_000;
const RUNS: usize = 10;

type HashFn = fn(input: &[u8]);

fn print(d: Duration, message: &str) {
    let nanos: u64 = NS_PER_SEC * d.as_secs() + d.subsec_nanos() as u64;
    let secs: f64 = nanos as f64 / NS_PER_SEC as f64;
    // (bits / ns) = (GB / sec)
    let rate: f64 = INPUT_LEN as f64 / nanos as f64;
    println!("{:.06}s ({:.06} GB/s) {}", secs, rate, message);
}

fn run(input: &[u8], hash_fn: HashFn) {
    let mut fastest = Duration::from_secs(u64::max_value());
    let mut total = Duration::from_secs(0);
    for i in 0..RUNS {
        let before = Instant::now();
        hash_fn(input);
        let after = Instant::now();
        let diff = after - before;
        if i == 0 {
            // Skip the first run, because it pays fixed costs like zeroing memory.
            print(diff, "(ignored)");
        } else {
            print(diff, "");
            total += diff;
            if diff < fastest {
                fastest = diff;
            }
        }
    }
    let average = total / (RUNS - 1) as u32;
    println!("-----");
    print(average, "average");
    print(fastest, "fastest");
    println!("-----");
}

fn hash_portable(input: &[u8]) {
    let mut state = blake2s_simd::State::new();
    blake2s_simd::benchmarks::force_portable(&mut state);
    state.update(input);
    state.finalize();
}

fn hash_update8(input: &[u8]) {
    let mut state0 = blake2s_simd::State::new();
    let mut state1 = blake2s_simd::State::new();
    let mut state2 = blake2s_simd::State::new();
    let mut state3 = blake2s_simd::State::new();
    let mut state4 = blake2s_simd::State::new();
    let mut state5 = blake2s_simd::State::new();
    let mut state6 = blake2s_simd::State::new();
    let mut state7 = blake2s_simd::State::new();
    let eighth = input.len() / 8;
    let input0 = &input[0 * eighth..][..eighth];
    let input1 = &input[1 * eighth..][..eighth];
    let input2 = &input[2 * eighth..][..eighth];
    let input3 = &input[3 * eighth..][..eighth];
    let input4 = &input[4 * eighth..][..eighth];
    let input5 = &input[5 * eighth..][..eighth];
    let input6 = &input[6 * eighth..][..eighth];
    let input7 = &input[7 * eighth..][..eighth];
    blake2s_simd::update8(
        &mut state0,
        &mut state1,
        &mut state2,
        &mut state3,
        &mut state4,
        &mut state5,
        &mut state6,
        &mut state7,
        input0,
        input1,
        input2,
        input3,
        input4,
        input5,
        input6,
        input7,
    );
    blake2s_simd::finalize8(
        &mut state0,
        &mut state1,
        &mut state2,
        &mut state3,
        &mut state4,
        &mut state5,
        &mut state6,
        &mut state7,
    );
}

fn hash_blake2sp(input: &[u8]) {
    blake2s_simd::blake2sp::blake2sp(input);
}

fn main() {
    let input = vec![0; INPUT_LEN];

    // Benchmark the portable implementation.
    println!("run #1, the portable implementation");
    run(&input, hash_portable);

    // Benchmark the 4-way AVX2 implementation.
    println!("run #2, the 4-way AVX2 implementation");
    run(&input, hash_update8);

    // Benchmark blake2sp.
    println!("run #3, blake2sp");
    run(&input, hash_blake2sp);
}
