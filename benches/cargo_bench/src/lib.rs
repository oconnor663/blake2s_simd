#![feature(test)]

extern crate blake2s_simd;
extern crate test;

use blake2s_simd::*;
use test::Bencher;

const BLOCK: &[u8; BLOCKBYTES] = &[0; BLOCKBYTES];
const MB: &[u8; 1_000_000] = &[0; 1_000_000];

#[bench]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn bench_blake2s_avx2_compress8(b: &mut Bencher) {
    if !is_x86_feature_detected!("avx2") {
        return;
    }
    b.bytes = BLOCK.len() as u64 * 8;
    let mut h1 = [0; 8];
    let mut h2 = [0; 8];
    let mut h3 = [0; 8];
    let mut h4 = [0; 8];
    let mut h5 = [0; 8];
    let mut h6 = [0; 8];
    let mut h7 = [0; 8];
    let mut h8 = [0; 8];
    b.iter(|| unsafe {
        benchmarks::compress8_avx2(
            &mut h1, &mut h2, &mut h3, &mut h4, &mut h5, &mut h6, &mut h7, &mut h8, BLOCK, BLOCK,
            BLOCK, BLOCK, BLOCK, BLOCK, BLOCK, BLOCK, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0,
        );
    });
}

#[bench]
fn bench_blake2s_portable_compress(b: &mut Bencher) {
    b.bytes = BLOCK.len() as u64;
    let mut h = [0; 8];
    b.iter(|| benchmarks::compress_portable(&mut h, BLOCK, 0, 0, 0));
}

#[bench]
fn bench_blake2s_portable_compress8(b: &mut Bencher) {
    b.bytes = BLOCK.len() as u64 * 8;
    let mut h1 = [0; 8];
    let mut h2 = [0; 8];
    let mut h3 = [0; 8];
    let mut h4 = [0; 8];
    let mut h5 = [0; 8];
    let mut h6 = [0; 8];
    let mut h7 = [0; 8];
    let mut h8 = [0; 8];
    b.iter(|| {
        benchmarks::compress8_portable(
            &mut h1, &mut h2, &mut h3, &mut h4, &mut h5, &mut h6, &mut h7, &mut h8, BLOCK, BLOCK,
            BLOCK, BLOCK, BLOCK, BLOCK, BLOCK, BLOCK, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0,
        );
    });
}

#[bench]
fn bench_blake2s_portable_one_block(b: &mut Bencher) {
    b.bytes = BLOCK.len() as u64;
    b.iter(|| {
        let mut state = State::new();
        benchmarks::force_portable(&mut state);
        state.update(BLOCK);
        state.finalize()
    });
}

#[bench]
fn bench_blake2s_portable_one_mb(b: &mut Bencher) {
    b.bytes = MB.len() as u64;
    b.iter(|| {
        let mut state = State::new();
        benchmarks::force_portable(&mut state);
        state.update(MB);
        state.finalize()
    });
}

#[bench]
fn bench_blake2sp_one_mb(b: &mut Bencher) {
    b.bytes = MB.len() as u64;
    b.iter(|| blake2sp::blake2sp(MB));
}

fn do_update8(input: &[u8]) -> [Hash; 8] {
    let mut state0 = State::new();
    let mut state1 = State::new();
    let mut state2 = State::new();
    let mut state3 = State::new();
    let mut state4 = State::new();
    let mut state5 = State::new();
    let mut state6 = State::new();
    let mut state7 = State::new();
    update8(
        &mut state0,
        &mut state1,
        &mut state2,
        &mut state3,
        &mut state4,
        &mut state5,
        &mut state6,
        &mut state7,
        input,
        input,
        input,
        input,
        input,
        input,
        input,
        input,
    );
    finalize8(
        &mut state0,
        &mut state1,
        &mut state2,
        &mut state3,
        &mut state4,
        &mut state5,
        &mut state6,
        &mut state7,
    )
}

#[bench]
fn bench_blake2s_update8_one_block(b: &mut Bencher) {
    b.bytes = 8 * BLOCK.len() as u64;
    b.iter(|| do_update8(BLOCK));
}

#[bench]
fn bench_blake2s_update8_one_mb(b: &mut Bencher) {
    b.bytes = 8 * MB.len() as u64;
    b.iter(|| do_update8(MB));
}
