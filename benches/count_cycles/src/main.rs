#![feature(test)]

extern crate amd64_timer;
extern crate blake2s_simd;
extern crate openssl;
extern crate test;

use blake2s_simd::BLOCKBYTES;

const TOTAL_BYTES_PER_TYPE: usize = 1 << 30; // 1 gigabyte

fn blake2s_compression() -> (u64, usize) {
    const SIZE: usize = blake2s_simd::BLOCKBYTES;
    let iterations = TOTAL_BYTES_PER_TYPE / SIZE;
    let mut total_ticks = 0;
    let input = &[0; blake2s_simd::BLOCKBYTES];
    let mut h = [0; 8];
    for _ in 0..iterations {
        let start = amd64_timer::ticks_modern();
        blake2s_simd::benchmarks::compress_portable(&mut h, input, 0, 0, 0);
        let end = amd64_timer::ticks_modern();
        total_ticks += end - start;
    }
    (total_ticks, iterations * SIZE)
}

fn blake2s_compression_8x() -> (u64, usize) {
    const SIZE: usize = 8 * BLOCKBYTES;
    let iterations = TOTAL_BYTES_PER_TYPE / SIZE;
    let mut total_ticks = 0;
    let msg0 = &[0; BLOCKBYTES];
    let msg1 = &[0; BLOCKBYTES];
    let msg2 = &[0; BLOCKBYTES];
    let msg3 = &[0; BLOCKBYTES];
    let msg4 = &[0; BLOCKBYTES];
    let msg5 = &[0; BLOCKBYTES];
    let msg6 = &[0; BLOCKBYTES];
    let msg7 = &[0; BLOCKBYTES];
    let h0 = &mut [0; 8];
    let h1 = &mut [0; 8];
    let h2 = &mut [0; 8];
    let h3 = &mut [0; 8];
    let h4 = &mut [0; 8];
    let h5 = &mut [0; 8];
    let h6 = &mut [0; 8];
    let h7 = &mut [0; 8];
    for _ in 0..iterations {
        let start = amd64_timer::ticks_modern();
        unsafe {
            blake2s_simd::benchmarks::compress8_avx2(
                h0, h1, h2, h3, h4, h5, h6, h7, msg0, msg1, msg2, msg3, msg4, msg5, msg6, msg7, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            );
        }
        let end = amd64_timer::ticks_modern();
        total_ticks += end - start;
    }
    (total_ticks, iterations * SIZE)
}

fn blake2s_one_mb() -> (u64, usize) {
    const SIZE: usize = 1_000_000;
    let iterations = TOTAL_BYTES_PER_TYPE / SIZE;
    let mut total_ticks = 0;
    for _ in 0..iterations {
        let start = amd64_timer::ticks_modern();
        test::black_box(&blake2s_simd::blake2s(&[0; SIZE]));
        let end = amd64_timer::ticks_modern();
        total_ticks += end - start;
    }
    (total_ticks, iterations * SIZE)
}

fn blake2sp_one_mb() -> (u64, usize) {
    const SIZE: usize = 1_000_000;
    let iterations = TOTAL_BYTES_PER_TYPE / SIZE;
    let mut total_ticks = 0;
    for _ in 0..iterations {
        let start = amd64_timer::ticks_modern();
        test::black_box(&blake2s_simd::blake2sp::blake2sp(&[0; SIZE]));
        let end = amd64_timer::ticks_modern();
        total_ticks += end - start;
    }
    (total_ticks, iterations * SIZE)
}

fn blake2s_update8_one_mb() -> (u64, usize) {
    const SIZE: usize = 8_000_000;
    let iterations = TOTAL_BYTES_PER_TYPE / SIZE;
    let mut total_ticks = 0;
    for _ in 0..iterations {
        let start = amd64_timer::ticks_modern();
        let mut state0 = blake2s_simd::State::new();
        let mut state1 = blake2s_simd::State::new();
        let mut state2 = blake2s_simd::State::new();
        let mut state3 = blake2s_simd::State::new();
        let mut state4 = blake2s_simd::State::new();
        let mut state5 = blake2s_simd::State::new();
        let mut state6 = blake2s_simd::State::new();
        let mut state7 = blake2s_simd::State::new();
        blake2s_simd::update8(
            &mut state0,
            &mut state1,
            &mut state2,
            &mut state3,
            &mut state4,
            &mut state5,
            &mut state6,
            &mut state7,
            &[0; SIZE / 8],
            &[0; SIZE / 8],
            &[0; SIZE / 8],
            &[0; SIZE / 8],
            &[0; SIZE / 8],
            &[0; SIZE / 8],
            &[0; SIZE / 8],
            &[0; SIZE / 8],
        );
        test::black_box(&blake2s_simd::finalize8(
            &mut state0,
            &mut state1,
            &mut state2,
            &mut state3,
            &mut state4,
            &mut state5,
            &mut state6,
            &mut state7,
        ));
        let end = amd64_timer::ticks_modern();
        total_ticks += end - start;
    }
    (total_ticks, iterations * SIZE)
}

fn sha1_openssl_one_mb() -> (u64, usize) {
    const SIZE: usize = 1_000_000;
    let iterations = TOTAL_BYTES_PER_TYPE / SIZE;
    let mut total_ticks = 0;
    for _ in 0..iterations {
        let start = amd64_timer::ticks_modern();
        test::black_box(&openssl::hash::hash(
            openssl::hash::MessageDigest::sha1(),
            &[0; SIZE],
        ));
        let end = amd64_timer::ticks_modern();
        total_ticks += end - start;
    }
    (total_ticks, iterations * SIZE)
}

fn sha512_openssl_one_mb() -> (u64, usize) {
    const SIZE: usize = 1_000_000;
    let iterations = TOTAL_BYTES_PER_TYPE / SIZE;
    let mut total_ticks = 0;
    for _ in 0..iterations {
        let start = amd64_timer::ticks_modern();
        test::black_box(&openssl::hash::hash(
            openssl::hash::MessageDigest::sha512(),
            &[0; SIZE],
        ));
        let end = amd64_timer::ticks_modern();
        total_ticks += end - start;
    }
    (total_ticks, iterations * SIZE)
}

fn main() {
    assert!(is_x86_feature_detected!("avx2"));
    let cases: &[(&str, fn() -> (u64, usize))] = &[
        ("BLAKE2s compression function", blake2s_compression),
        ("BLAKE2s 4-way compression function", blake2s_compression_8x),
        ("BLAKE2s 1 MB", blake2s_one_mb),
        ("BLAKE2sp 1 MB", blake2sp_one_mb),
        ("BLAKE2s update8 1 MB", blake2s_update8_one_mb),
        ("SHA1 OpenSSL 1 MB", sha1_openssl_one_mb),
        ("SHA512 OpenSSL 1 MB", sha512_openssl_one_mb),
    ];

    for &(name, f) in cases.iter() {
        // Warmup loop.
        f();
        // Loop for real.
        let (total_cycles, total_bytes) = f();
        println!(
            "{:34}  {:.3}",
            name,
            total_cycles as f64 / total_bytes as f64
        );
    }
}
