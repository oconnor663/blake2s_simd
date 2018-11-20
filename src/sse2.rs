#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use core::ptr;
use crate::{Block, StateWords, IV, SIGMA};

#[inline(always)]
unsafe fn loadu(p: *const u32) -> __m128i {
    _mm_loadu_si128(p as *const __m128i)
}

#[inline(always)]
unsafe fn storeu(p: *mut u32, x: __m128i) {
    _mm_storeu_si128(p as *mut __m128i, x)
}

#[inline(always)]
unsafe fn setr(a: u32, b: u32, c: u32, d: u32) -> __m128i {
    _mm_setr_epi32(a as i32, b as i32, c as i32, d as i32)
}

#[inline(always)]
unsafe fn xor(a: __m128i, b: __m128i) -> __m128i {
    _mm_xor_si128(a, b)
}

#[inline(always)]
unsafe fn add(a: __m128i, b: __m128i) -> __m128i {
    _mm_add_epi32(a, b)
}

#[inline(always)]
unsafe fn rot7(a: __m128i) -> __m128i {
    xor(_mm_srli_epi32(a, 7), _mm_slli_epi32(a, 32 - 7))
}

#[inline(always)]
unsafe fn rot8(a: __m128i) -> __m128i {
    // NOTE: For SSSE3 or higher this could be implemented with
    // _mm_shuffle_epi8, but on my laptop that's actually slightly slower.
    xor(_mm_srli_epi32(a, 8), _mm_slli_epi32(a, 32 - 8))
}

#[inline(always)]
unsafe fn rot12(a: __m128i) -> __m128i {
    xor(_mm_srli_epi32(a, 12), _mm_slli_epi32(a, 32 - 12))
}

#[inline(always)]
unsafe fn rot16(a: __m128i) -> __m128i {
    // NOTE: For SSSE3 or higher this could be implemented with
    // _mm_shuffle_epi8, but on my laptop that's actually slightly slower.
    xor(_mm_srli_epi32(a, 16), _mm_slli_epi32(a, 32 - 16))
}

#[inline(always)]
unsafe fn g1(
    row1: &mut __m128i,
    row2: &mut __m128i,
    row3: &mut __m128i,
    row4: &mut __m128i,
    m: __m128i,
) {
    *row1 = add(add(*row1, m), *row2);
    *row4 = xor(*row4, *row1);
    *row4 = rot16(*row4);
    *row3 = add(*row3, *row4);
    *row2 = xor(*row2, *row3);
    *row2 = rot12(*row2);
}

#[inline(always)]
unsafe fn g2(
    row1: &mut __m128i,
    row2: &mut __m128i,
    row3: &mut __m128i,
    row4: &mut __m128i,
    m: __m128i,
) {
    *row1 = add(add(*row1, m), *row2);
    *row4 = xor(*row4, *row1);
    *row4 = rot8(*row4);
    *row3 = add(*row3, *row4);
    *row2 = xor(*row2, *row3);
    *row2 = rot7(*row2);
}

// Adapted from https://github.com/rust-lang-nursery/stdsimd/pull/479.
macro_rules! _MM_SHUFFLE {
    ($z:expr, $y:expr, $x:expr, $w:expr) => {
        ($z << 6) | ($y << 4) | ($x << 2) | $w
    };
}

#[inline(always)]
unsafe fn diagonalize(row2: &mut __m128i, row3: &mut __m128i, row4: &mut __m128i) {
    *row4 = _mm_shuffle_epi32(*row4, _MM_SHUFFLE!(2, 1, 0, 3));
    *row3 = _mm_shuffle_epi32(*row3, _MM_SHUFFLE!(1, 0, 3, 2));
    *row2 = _mm_shuffle_epi32(*row2, _MM_SHUFFLE!(0, 3, 2, 1));
}

#[inline(always)]
unsafe fn undiagonalize(row2: &mut __m128i, row3: &mut __m128i, row4: &mut __m128i) {
    *row4 = _mm_shuffle_epi32(*row4, _MM_SHUFFLE!(0, 3, 2, 1));
    *row3 = _mm_shuffle_epi32(*row3, _MM_SHUFFLE!(1, 0, 3, 2));
    *row2 = _mm_shuffle_epi32(*row2, _MM_SHUFFLE!(2, 1, 0, 3));
}

// TODO: Try SSE4.1 shuffle-based loading.
unsafe fn load_msg_words(
    msg: &Block,
    round: usize,
    i1: usize,
    i2: usize,
    i3: usize,
    i4: usize,
) -> __m128i {
    let s = SIGMA[round];
    setr(
        ptr::read_unaligned(msg.as_ptr().add(s[i1] as usize * 4) as *const u8 as *const u32),
        ptr::read_unaligned(msg.as_ptr().add(s[i2] as usize * 4) as *const u8 as *const u32),
        ptr::read_unaligned(msg.as_ptr().add(s[i3] as usize * 4) as *const u8 as *const u32),
        ptr::read_unaligned(msg.as_ptr().add(s[i4] as usize * 4) as *const u8 as *const u32),
    )
}

#[inline(always)]
unsafe fn round(
    row1: &mut __m128i,
    row2: &mut __m128i,
    row3: &mut __m128i,
    row4: &mut __m128i,
    msg: &Block,
    round: usize,
) {
    let m = load_msg_words(msg, round, 0, 2, 4, 6);
    g1(row1, row2, row3, row4, m);
    let m = load_msg_words(msg, round, 1, 3, 5, 7);
    g2(row1, row2, row3, row4, m);
    diagonalize(row2, row3, row4);
    let m = load_msg_words(msg, round, 8, 10, 12, 14);
    g1(row1, row2, row3, row4, m);
    let m = load_msg_words(msg, round, 9, 11, 13, 15);
    g2(row1, row2, row3, row4, m);
    undiagonalize(row2, row3, row4);
}

#[target_feature(enable = "sse2")]
pub unsafe fn compress(h: &mut StateWords, msg: &Block, count: u64, lastblock: u32, lastnode: u32) {
    let mut row1 = loadu(&h[0]);
    let mut row2 = loadu(&h[4]);
    let mut row3 = loadu(&IV[0]);
    let mut row4 = xor(
        loadu(&IV[4]),
        setr(count as u32, (count >> 32) as u32, lastblock, lastnode),
    );

    round(&mut row1, &mut row2, &mut row3, &mut row4, msg, 0);
    round(&mut row1, &mut row2, &mut row3, &mut row4, msg, 1);
    round(&mut row1, &mut row2, &mut row3, &mut row4, msg, 2);
    round(&mut row1, &mut row2, &mut row3, &mut row4, msg, 3);
    round(&mut row1, &mut row2, &mut row3, &mut row4, msg, 4);
    round(&mut row1, &mut row2, &mut row3, &mut row4, msg, 5);
    round(&mut row1, &mut row2, &mut row3, &mut row4, msg, 6);
    round(&mut row1, &mut row2, &mut row3, &mut row4, msg, 7);
    round(&mut row1, &mut row2, &mut row3, &mut row4, msg, 8);
    round(&mut row1, &mut row2, &mut row3, &mut row4, msg, 9);

    storeu(&mut h[0], xor(loadu(&h[0]), xor(row1, row3)));
    storeu(&mut h[4], xor(loadu(&h[4]), xor(row2, row4)));
}
