#![feature(test)]

extern crate blake2s_simd;
extern crate test;

use blake2s_simd::*;
use std::mem;
use test::Bencher;

const BLOCK: &[u8; BLOCKBYTES] = &[0; BLOCKBYTES];
const MB: &[u8; 1_000_000] = &[0; 1_000_000];

#[bench]
fn bench_blake2s_sse41_compress(b: &mut Bencher) {
    b.bytes = BLOCK.len() as u64;
    let mut h = [0; 8];
    b.iter(|| unsafe { benchmarks::compress_sse41(&mut h, BLOCK, 0, 0, 0) });
}

#[bench]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn bench_blake2s_avx2_compress8(b: &mut Bencher) {
    if !is_x86_feature_detected!("avx2") {
        return;
    }
    b.bytes = BLOCK.len() as u64 * 8;
    let mut h0 = [0xf0; 8];
    let mut h1 = [0xf1; 8];
    let mut h2 = [0xf2; 8];
    let mut h3 = [0xf3; 8];
    let mut h4 = [0xf4; 8];
    let mut h5 = [0xf5; 8];
    let mut h6 = [0xf6; 8];
    let mut h7 = [0xf7; 8];
    let block0 = [0xf0; BLOCKBYTES];
    let block1 = [0xf1; BLOCKBYTES];
    let block2 = [0xf2; BLOCKBYTES];
    let block3 = [0xf3; BLOCKBYTES];
    let block4 = [0xf4; BLOCKBYTES];
    let block5 = [0xf5; BLOCKBYTES];
    let block6 = [0xf6; BLOCKBYTES];
    let block7 = [0xf7; BLOCKBYTES];
    b.iter(|| unsafe {
        benchmarks::compress8_avx2(
            &mut h0, &mut h1, &mut h2, &mut h3, &mut h4, &mut h5, &mut h6, &mut h7, &block0,
            &block1, &block2, &block3, &block4, &block5, &block6, &block7, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        );
    });
}

#[bench]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn bench_blake2s_sse41_compress4_transposed(b: &mut Bencher) {
    if !is_x86_feature_detected!("sse4.1") {
        return;
    }
    b.bytes = BLOCK.len() as u64 * 4;
    unsafe {
        let mut h_vecs = mem::zeroed();
        let msg1 = [0; BLOCKBYTES];
        let msg2 = [0; BLOCKBYTES];
        let msg3 = [0; BLOCKBYTES];
        let msg4 = [0; BLOCKBYTES];
        let count_low = mem::zeroed();
        let count_high = mem::zeroed();
        let lastblock = mem::zeroed();
        let lastnode = mem::zeroed();
        b.iter(|| {
            benchmarks::compress4_transposed_sse41(
                &mut h_vecs,
                &msg1,
                &msg2,
                &msg3,
                &msg4,
                count_low,
                count_high,
                lastblock,
                lastnode,
            );
            test::black_box(&mut h_vecs);
        });
    }
}

#[bench]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn bench_blake2s_avx2_compress8_vectorized(b: &mut Bencher) {
    if !is_x86_feature_detected!("avx2") {
        return;
    }
    b.bytes = BLOCK.len() as u64 * 8;
    unsafe {
        let mut h_vecs = [AlignedWords8([1; 8]); 8];
        let msg1 = [2; BLOCKBYTES];
        let msg2 = [3; BLOCKBYTES];
        let msg3 = [4; BLOCKBYTES];
        let msg4 = [5; BLOCKBYTES];
        let msg5 = [6; BLOCKBYTES];
        let msg6 = [7; BLOCKBYTES];
        let msg7 = [8; BLOCKBYTES];
        let msg8 = [9; BLOCKBYTES];
        let count_low = mem::zeroed();
        let count_high = mem::zeroed();
        let lastblock = mem::zeroed();
        let lastnode = mem::zeroed();
        b.iter(|| {
            benchmarks::compress8_vectorized_avx2(
                &mut h_vecs,
                &msg1,
                &msg2,
                &msg3,
                &msg4,
                &msg5,
                &msg6,
                &msg7,
                &msg8,
                &count_low,
                &count_high,
                &lastblock,
                &lastnode,
            );
            test::black_box(&mut h_vecs);
        });
    }
}

#[bench]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn bench_blake2s_avx2_compress8_transposed_all(b: &mut Bencher) {
    if !is_x86_feature_detected!("avx2") {
        return;
    }
    b.bytes = BLOCK.len() as u64 * 8;
    unsafe {
        let mut h_vecs = mem::transmute([1u8; 32 * 8]);
        let msg_vecs = mem::transmute([2u8; 32 * 16]);
        let count_low = mem::transmute([3u8; 32]);
        let count_high = mem::transmute([4u8; 32]);
        let lastblock = mem::transmute([5u8; 32]);
        let lastnode = mem::transmute([6u8; 32]);
        b.iter(|| {
            benchmarks::compress8_transposed_all_avx2(
                &mut h_vecs,
                &msg_vecs,
                count_low,
                count_high,
                lastblock,
                lastnode,
            );
            test::black_box(&mut h_vecs);
        });
    }
}

#[bench]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn bench_blake2s_avx2_compress8_transposed_all_with_separate_msg_transpose(b: &mut Bencher) {
    if !is_x86_feature_detected!("avx2") {
        return;
    }
    b.bytes = BLOCK.len() as u64 * 8;
    unsafe {
        let mut h_vecs = mem::transmute([1u8; 32 * 8]);
        let msg_blocks = [[2u8; 64]; 8];
        let count_low = mem::transmute([3u8; 32]);
        let count_high = mem::transmute([4u8; 32]);
        let lastblock = mem::transmute([5u8; 32]);
        let lastnode = mem::transmute([6u8; 32]);
        b.iter(|| {
            let msg_vecs = benchmarks::transpose_msg_vecs_avx2(
                &msg_blocks[0],
                &msg_blocks[1],
                &msg_blocks[2],
                &msg_blocks[3],
                &msg_blocks[4],
                &msg_blocks[5],
                &msg_blocks[6],
                &msg_blocks[7],
            );
            benchmarks::compress8_transposed_all_avx2(
                &mut h_vecs,
                &msg_vecs,
                count_low,
                count_high,
                lastblock,
                lastnode,
            );
            test::black_box(&mut h_vecs);
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
fn bench_blake2s_hash4_one_block(b: &mut Bencher) {
    b.bytes = 4 * BLOCKBYTES as u64;
    let buf = vec![1; BLOCKBYTES];
    b.iter(|| hash4_exact(&Params::new(), &buf, &buf, &buf, &buf));
}

#[bench]
fn bench_blake2s_hash4_4096(b: &mut Bencher) {
    b.bytes = 4 * 4096 as u64;
    let buf = vec![1; 4096];
    b.iter(|| hash4_exact(&Params::new(), &buf, &buf, &buf, &buf));
}

#[bench]
fn bench_blake2s_hash4_one_mb(b: &mut Bencher) {
    b.bytes = 4 * (1 << 20);
    let buf = vec![1; 1 << 20];
    b.iter(|| hash4_exact(&Params::new(), &buf, &buf, &buf, &buf));
}

#[bench]
fn bench_blake2s_hash8_exact_one_block(b: &mut Bencher) {
    b.bytes = 8 * BLOCKBYTES as u64;
    let buf = vec![1; BLOCKBYTES];
    b.iter(|| {
        hash8_exact(
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
fn bench_blake2s_hash8_exact_4096(b: &mut Bencher) {
    b.bytes = 8 * 4096 as u64;
    let buf = vec![1; 4096];
    b.iter(|| {
        hash8_exact(
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
fn bench_blake2s_hash8_exact_one_mb(b: &mut Bencher) {
    b.bytes = 8 * (1 << 20);
    let buf = vec![1; 1 << 20];
    b.iter(|| {
        hash8_exact(
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
fn bench_blake2s_hash8_inexact_one_block(b: &mut Bencher) {
    b.bytes = 8 * BLOCKBYTES as u64;
    let buf = vec![1; BLOCKBYTES];
    b.iter(|| {
        hash8(
            &Params::new(),
            &Params::new(),
            &Params::new(),
            &Params::new(),
            &Params::new(),
            &Params::new(),
            &Params::new(),
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
fn bench_blake2s_hash8_inexact_4096(b: &mut Bencher) {
    b.bytes = 8 * 4096 as u64;
    let buf = vec![1; 4096];
    b.iter(|| {
        hash8(
            &Params::new(),
            &Params::new(),
            &Params::new(),
            &Params::new(),
            &Params::new(),
            &Params::new(),
            &Params::new(),
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
fn bench_blake2s_hash8_inexact_one_mb(b: &mut Bencher) {
    b.bytes = 8 * (1 << 20);
    let buf = vec![1; 1 << 20];
    b.iter(|| {
        hash8(
            &Params::new(),
            &Params::new(),
            &Params::new(),
            &Params::new(),
            &Params::new(),
            &Params::new(),
            &Params::new(),
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
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn bench_blake2s_avx2_compress8_to_bytes_together(b: &mut Bencher) {
    if !is_x86_feature_detected!("avx2") {
        return;
    }
    b.bytes = BLOCK.len() as u64 * 8;
    unsafe {
        let mut h_vecs = [AlignedWords8([1; 8]); 8];
        let msg1 = [2; BLOCKBYTES];
        let msg2 = [3; BLOCKBYTES];
        let msg3 = [4; BLOCKBYTES];
        let msg4 = [5; BLOCKBYTES];
        let msg5 = [6; BLOCKBYTES];
        let msg6 = [7; BLOCKBYTES];
        let msg7 = [8; BLOCKBYTES];
        let msg8 = [9; BLOCKBYTES];
        let count_low = mem::zeroed();
        let count_high = mem::zeroed();
        let lastblock = mem::zeroed();
        let lastnode = mem::zeroed();
        let mut out0 = [0u8; 32];
        let mut out1 = [0u8; 32];
        let mut out2 = [0u8; 32];
        let mut out3 = [0u8; 32];
        let mut out4 = [0u8; 32];
        let mut out5 = [0u8; 32];
        let mut out6 = [0u8; 32];
        let mut out7 = [0u8; 32];
        b.iter(|| {
            benchmarks::compress8_vectorized_to_bytes_avx2(
                &mut h_vecs,
                &msg1,
                &msg2,
                &msg3,
                &msg4,
                &msg5,
                &msg6,
                &msg7,
                &msg8,
                &count_low,
                &count_high,
                &lastblock,
                &lastnode,
                &mut out0,
                &mut out1,
                &mut out2,
                &mut out3,
                &mut out4,
                &mut out5,
                &mut out6,
                &mut out7,
            );
            test::black_box(&h_vecs);
            test::black_box(&out0);
            test::black_box(&out1);
            test::black_box(&out2);
            test::black_box(&out3);
            test::black_box(&out4);
            test::black_box(&out5);
            test::black_box(&out6);
            test::black_box(&out7);
        });
    }
}

#[bench]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn bench_blake2s_avx2_compress8_to_bytes_separate(b: &mut Bencher) {
    if !is_x86_feature_detected!("avx2") {
        return;
    }
    b.bytes = BLOCK.len() as u64 * 8;
    unsafe {
        let mut h_vecs = [AlignedWords8([1; 8]); 8];
        let msg1 = [2; BLOCKBYTES];
        let msg2 = [3; BLOCKBYTES];
        let msg3 = [4; BLOCKBYTES];
        let msg4 = [5; BLOCKBYTES];
        let msg5 = [6; BLOCKBYTES];
        let msg6 = [7; BLOCKBYTES];
        let msg7 = [8; BLOCKBYTES];
        let msg8 = [9; BLOCKBYTES];
        let count_low = mem::zeroed();
        let count_high = mem::zeroed();
        let lastblock = mem::zeroed();
        let lastnode = mem::zeroed();
        let mut out0 = [0u8; 32];
        let mut out1 = [0u8; 32];
        let mut out2 = [0u8; 32];
        let mut out3 = [0u8; 32];
        let mut out4 = [0u8; 32];
        let mut out5 = [0u8; 32];
        let mut out6 = [0u8; 32];
        let mut out7 = [0u8; 32];
        b.iter(|| {
            benchmarks::compress8_vectorized_avx2(
                &mut h_vecs,
                &msg1,
                &msg2,
                &msg3,
                &msg4,
                &msg5,
                &msg6,
                &msg7,
                &msg8,
                &count_low,
                &count_high,
                &lastblock,
                &lastnode,
            );
            benchmarks::export_bytes_avx2(
                &h_vecs, &mut out0, &mut out1, &mut out2, &mut out3, &mut out4, &mut out5,
                &mut out6, &mut out7,
            );
            test::black_box(&h_vecs);
            test::black_box(&out0);
            test::black_box(&out1);
            test::black_box(&out2);
            test::black_box(&out3);
            test::black_box(&out4);
            test::black_box(&out5);
            test::black_box(&out6);
            test::black_box(&out7);
        });
    }
}
