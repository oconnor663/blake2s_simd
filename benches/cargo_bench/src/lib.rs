#![feature(test)]

extern crate blake2s_simd;
extern crate test;

use blake2s_simd::*;
use std::mem;
use test::Bencher;

const BLOCK: &[u8; BLOCKBYTES] = &[0; BLOCKBYTES];
const MB: &[u8; 1_000_000] = &[0; 1_000_000];

#[bench]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn bench_load_msg_vecs_naive(b: &mut Bencher) {
    if !is_x86_feature_detected!("avx2") {
        return;
    }
    b.bytes = BLOCK.len() as u64 * 8;
    let block0 = [0; BLOCKBYTES];
    let block1 = [1; BLOCKBYTES];
    let block2 = [2; BLOCKBYTES];
    let block3 = [3; BLOCKBYTES];
    let block4 = [4; BLOCKBYTES];
    let block5 = [5; BLOCKBYTES];
    let block6 = [6; BLOCKBYTES];
    let block7 = [7; BLOCKBYTES];
    #[target_feature(enable = "avx2")]
    unsafe fn inner(
        block0: &[u8; BLOCKBYTES],
        block1: &[u8; BLOCKBYTES],
        block2: &[u8; BLOCKBYTES],
        block3: &[u8; BLOCKBYTES],
        block4: &[u8; BLOCKBYTES],
        block5: &[u8; BLOCKBYTES],
        block6: &[u8; BLOCKBYTES],
        block7: &[u8; BLOCKBYTES],
    ) -> [std::arch::x86_64::__m256i; 16] {
        benchmarks::load_msg_vecs_naive_avx2(
            &block0, &block1, &block2, &block3, &block4, &block5, &block6, &block7,
        )
    }
    b.iter(|| unsafe {
        inner(
            &block0, &block1, &block2, &block3, &block4, &block5, &block6, &block7,
        )
    });
}

#[bench]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn bench_load_msg_vecs_interleave(b: &mut Bencher) {
    if !is_x86_feature_detected!("avx2") {
        return;
    }
    b.bytes = BLOCK.len() as u64 * 8;
    let block0 = [0; BLOCKBYTES];
    let block1 = [1; BLOCKBYTES];
    let block2 = [2; BLOCKBYTES];
    let block3 = [3; BLOCKBYTES];
    let block4 = [4; BLOCKBYTES];
    let block5 = [5; BLOCKBYTES];
    let block6 = [6; BLOCKBYTES];
    let block7 = [7; BLOCKBYTES];
    #[target_feature(enable = "avx2")]
    unsafe fn inner(
        block0: &[u8; BLOCKBYTES],
        block1: &[u8; BLOCKBYTES],
        block2: &[u8; BLOCKBYTES],
        block3: &[u8; BLOCKBYTES],
        block4: &[u8; BLOCKBYTES],
        block5: &[u8; BLOCKBYTES],
        block6: &[u8; BLOCKBYTES],
        block7: &[u8; BLOCKBYTES],
    ) -> [std::arch::x86_64::__m256i; 16] {
        benchmarks::load_msg_vecs_interleave_avx2(
            &block0, &block1, &block2, &block3, &block4, &block5, &block6, &block7,
        )
    }
    b.iter(|| unsafe {
        inner(
            &block0, &block1, &block2, &block3, &block4, &block5, &block6, &block7,
        )
    });
}

#[bench]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn bench_load_msg_vecs_gather(b: &mut Bencher) {
    if !is_x86_feature_detected!("avx2") {
        return;
    }
    b.bytes = BLOCK.len() as u64 * 8;
    let block0 = [0; BLOCKBYTES];
    let block1 = [1; BLOCKBYTES];
    let block2 = [2; BLOCKBYTES];
    let block3 = [3; BLOCKBYTES];
    let block4 = [4; BLOCKBYTES];
    let block5 = [5; BLOCKBYTES];
    let block6 = [6; BLOCKBYTES];
    let block7 = [7; BLOCKBYTES];
    #[target_feature(enable = "avx2")]
    unsafe fn inner(
        block0: &[u8; BLOCKBYTES],
        block1: &[u8; BLOCKBYTES],
        block2: &[u8; BLOCKBYTES],
        block3: &[u8; BLOCKBYTES],
        block4: &[u8; BLOCKBYTES],
        block5: &[u8; BLOCKBYTES],
        block6: &[u8; BLOCKBYTES],
        block7: &[u8; BLOCKBYTES],
    ) -> [std::arch::x86_64::__m256i; 16] {
        benchmarks::load_msg_vecs_gather_avx2(
            &block0, &block1, &block2, &block3, &block4, &block5, &block6, &block7,
        )
    }
    b.iter(|| unsafe {
        inner(
            &block0, &block1, &block2, &block3, &block4, &block5, &block6, &block7,
        )
    });
}

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
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn bench_blake2s_avx2_compress8_inner(b: &mut Bencher) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;
    if !is_x86_feature_detected!("avx2") {
        return;
    }
    b.bytes = BLOCK.len() as u64 * 8;
    unsafe {
        let mut h_vecs: [__m256i; 8] = mem::zeroed();
        b.iter(|| {
            benchmarks::compress8_inner_avx2(
                &mut h_vecs,
                BLOCK,
                BLOCK,
                BLOCK,
                BLOCK,
                BLOCK,
                BLOCK,
                BLOCK,
                BLOCK,
                mem::zeroed(),
                mem::zeroed(),
                mem::zeroed(),
                mem::zeroed(),
            )
        });
    }
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

#[bench]
fn bench_blake2s_8way_one_block(b: &mut Bencher) {
    b.bytes = 8 * BLOCKBYTES as u64;
    let buf = vec![1; BLOCKBYTES];
    b.iter(|| unsafe {
        blake2s_8way(
            &Params::new(),
            &buf,
            &buf,
            &buf,
            &buf,
            &buf,
            &buf,
            &buf,
            &buf,
        )
    });
}

#[bench]
fn bench_blake2s_8way_4096(b: &mut Bencher) {
    b.bytes = 8 * 4096 as u64;
    let buf = vec![1; 4096];
    b.iter(|| unsafe {
        blake2s_8way(
            &Params::new(),
            &buf,
            &buf,
            &buf,
            &buf,
            &buf,
            &buf,
            &buf,
            &buf,
        )
    });
}

#[bench]
fn bench_blake2s_8way_one_mb(b: &mut Bencher) {
    b.bytes = 8 * (1 << 20);
    let buf = vec![1; 1 << 20];
    b.iter(|| unsafe {
        blake2s_8way(
            &Params::new(),
            &buf,
            &buf,
            &buf,
            &buf,
            &buf,
            &buf,
            &buf,
            &buf,
        )
    });
}
