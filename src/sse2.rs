#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use core::ptr;
use crate::{Block, Hash, Params, StateWords, BLOCKBYTES, IV, OUTBYTES, SIGMA};

#[inline(always)]
unsafe fn loadu(p: *const u32) -> __m128i {
    _mm_loadu_si128(p as *const __m128i)
}

#[inline(always)]
unsafe fn loadu_4(p: *const u8) -> [__m128i; 4] {
    [
        _mm_loadu_si128((p as *const __m128i).add(0)),
        _mm_loadu_si128((p as *const __m128i).add(1)),
        _mm_loadu_si128((p as *const __m128i).add(2)),
        _mm_loadu_si128((p as *const __m128i).add(3)),
    ]
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
unsafe fn set1(a: u32) -> __m128i {
    _mm_set1_epi32(a as i32)
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

#[inline(always)]
unsafe fn transpose_vecs(
    vec_a: __m128i,
    vec_b: __m128i,
    vec_c: __m128i,
    vec_d: __m128i,
) -> [__m128i; 4] {
    // Interleave 32-bit lates. The low unpack is lanes 00/11 and the high is
    // 22/33. Note that this doesn't split the vector into two lanes, as the
    // AVX2 counterparts do.
    let ab_01 = _mm_unpacklo_epi32(vec_a, vec_b);
    let ab_23 = _mm_unpackhi_epi32(vec_a, vec_b);
    let cd_01 = _mm_unpacklo_epi32(vec_c, vec_d);
    let cd_23 = _mm_unpackhi_epi32(vec_c, vec_d);

    // Interleave 64-bit lanes.
    let abcd_0 = _mm_unpacklo_epi64(ab_01, cd_01);
    let abcd_1 = _mm_unpackhi_epi64(ab_01, cd_01);
    let abcd_2 = _mm_unpacklo_epi64(ab_23, cd_23);
    let abcd_3 = _mm_unpackhi_epi64(ab_23, cd_23);

    [abcd_0, abcd_1, abcd_2, abcd_3]
}

#[cfg(test)]
#[test]
fn test_transpose_vecs() {
    unsafe fn cast_out(a: __m128i) -> [u32; 4] {
        core::mem::transmute(a)
    }

    unsafe {
        let vec_a = setr(0x00, 0x01, 0x02, 0x03);
        let vec_b = setr(0x10, 0x11, 0x12, 0x13);
        let vec_c = setr(0x20, 0x21, 0x22, 0x23);
        let vec_d = setr(0x30, 0x31, 0x32, 0x33);

        let expected_a = setr(0x00, 0x10, 0x20, 0x30);
        let expected_b = setr(0x01, 0x11, 0x21, 0x31);
        let expected_c = setr(0x02, 0x12, 0x22, 0x32);
        let expected_d = setr(0x03, 0x13, 0x23, 0x33);

        let [out_a, out_b, out_c, out_d] = transpose_vecs(vec_a, vec_b, vec_c, vec_d);

        assert_eq!(cast_out(expected_a), cast_out(out_a));
        assert_eq!(cast_out(expected_b), cast_out(out_b));
        assert_eq!(cast_out(expected_c), cast_out(out_c));
        assert_eq!(cast_out(expected_d), cast_out(out_d));

        // Check that interleaving again undoes the operation.
        let [out2_a, out2_b, out2_c, out2_d] = transpose_vecs(out_a, out_b, out_c, out_d);
        assert_eq!(cast_out(vec_a), cast_out(out2_a));
        assert_eq!(cast_out(vec_b), cast_out(out2_b));
        assert_eq!(cast_out(vec_c), cast_out(out2_c));
        assert_eq!(cast_out(vec_d), cast_out(out2_d));
    }
}

#[inline(always)]
pub unsafe fn transpose_message_blocks(
    msg_a: &Block,
    msg_b: &Block,
    msg_c: &Block,
    msg_d: &Block,
) -> [__m128i; 16] {
    let [a0, a1, a2, a3] = loadu_4(msg_a.as_ptr());
    let [b0, b1, b2, b3] = loadu_4(msg_b.as_ptr());
    let [c0, c1, c2, c3] = loadu_4(msg_c.as_ptr());
    let [d0, d1, d2, d3] = loadu_4(msg_d.as_ptr());

    let transposed0 = transpose_vecs(a0, b0, c0, d0);
    let transposed1 = transpose_vecs(a1, b1, c1, d1);
    let transposed2 = transpose_vecs(a2, b2, c2, d2);
    let transposed3 = transpose_vecs(a3, b3, c3, d3);

    [
        transposed0[0],
        transposed0[1],
        transposed0[2],
        transposed0[3],
        transposed1[0],
        transposed1[1],
        transposed1[2],
        transposed1[3],
        transposed2[0],
        transposed2[1],
        transposed2[2],
        transposed2[3],
        transposed3[0],
        transposed3[1],
        transposed3[2],
        transposed3[3],
    ]
}

#[inline(always)]
unsafe fn round_4(v: &mut [__m128i; 16], m: &[__m128i; 16], r: usize) {
    v[0] = add(v[0], m[SIGMA[r][0] as usize]);
    v[1] = add(v[1], m[SIGMA[r][2] as usize]);
    v[2] = add(v[2], m[SIGMA[r][4] as usize]);
    v[3] = add(v[3], m[SIGMA[r][6] as usize]);
    v[0] = add(v[0], v[4]);
    v[1] = add(v[1], v[5]);
    v[2] = add(v[2], v[6]);
    v[3] = add(v[3], v[7]);
    v[12] = xor(v[12], v[0]);
    v[13] = xor(v[13], v[1]);
    v[14] = xor(v[14], v[2]);
    v[15] = xor(v[15], v[3]);
    v[12] = rot16(v[12]);
    v[13] = rot16(v[13]);
    v[14] = rot16(v[14]);
    v[15] = rot16(v[15]);
    v[8] = add(v[8], v[12]);
    v[9] = add(v[9], v[13]);
    v[10] = add(v[10], v[14]);
    v[11] = add(v[11], v[15]);
    v[4] = xor(v[4], v[8]);
    v[5] = xor(v[5], v[9]);
    v[6] = xor(v[6], v[10]);
    v[7] = xor(v[7], v[11]);
    v[4] = rot12(v[4]);
    v[5] = rot12(v[5]);
    v[6] = rot12(v[6]);
    v[7] = rot12(v[7]);
    v[0] = add(v[0], m[SIGMA[r][1] as usize]);
    v[1] = add(v[1], m[SIGMA[r][3] as usize]);
    v[2] = add(v[2], m[SIGMA[r][5] as usize]);
    v[3] = add(v[3], m[SIGMA[r][7] as usize]);
    v[0] = add(v[0], v[4]);
    v[1] = add(v[1], v[5]);
    v[2] = add(v[2], v[6]);
    v[3] = add(v[3], v[7]);
    v[12] = xor(v[12], v[0]);
    v[13] = xor(v[13], v[1]);
    v[14] = xor(v[14], v[2]);
    v[15] = xor(v[15], v[3]);
    v[12] = rot8(v[12]);
    v[13] = rot8(v[13]);
    v[14] = rot8(v[14]);
    v[15] = rot8(v[15]);
    v[8] = add(v[8], v[12]);
    v[9] = add(v[9], v[13]);
    v[10] = add(v[10], v[14]);
    v[11] = add(v[11], v[15]);
    v[4] = xor(v[4], v[8]);
    v[5] = xor(v[5], v[9]);
    v[6] = xor(v[6], v[10]);
    v[7] = xor(v[7], v[11]);
    v[4] = rot7(v[4]);
    v[5] = rot7(v[5]);
    v[6] = rot7(v[6]);
    v[7] = rot7(v[7]);

    v[0] = add(v[0], m[SIGMA[r][8] as usize]);
    v[1] = add(v[1], m[SIGMA[r][10] as usize]);
    v[2] = add(v[2], m[SIGMA[r][12] as usize]);
    v[3] = add(v[3], m[SIGMA[r][14] as usize]);
    v[0] = add(v[0], v[5]);
    v[1] = add(v[1], v[6]);
    v[2] = add(v[2], v[7]);
    v[3] = add(v[3], v[4]);
    v[15] = xor(v[15], v[0]);
    v[12] = xor(v[12], v[1]);
    v[13] = xor(v[13], v[2]);
    v[14] = xor(v[14], v[3]);
    v[15] = rot16(v[15]);
    v[12] = rot16(v[12]);
    v[13] = rot16(v[13]);
    v[14] = rot16(v[14]);
    v[10] = add(v[10], v[15]);
    v[11] = add(v[11], v[12]);
    v[8] = add(v[8], v[13]);
    v[9] = add(v[9], v[14]);
    v[5] = xor(v[5], v[10]);
    v[6] = xor(v[6], v[11]);
    v[7] = xor(v[7], v[8]);
    v[4] = xor(v[4], v[9]);
    v[5] = rot12(v[5]);
    v[6] = rot12(v[6]);
    v[7] = rot12(v[7]);
    v[4] = rot12(v[4]);
    v[0] = add(v[0], m[SIGMA[r][9] as usize]);
    v[1] = add(v[1], m[SIGMA[r][11] as usize]);
    v[2] = add(v[2], m[SIGMA[r][13] as usize]);
    v[3] = add(v[3], m[SIGMA[r][15] as usize]);
    v[0] = add(v[0], v[5]);
    v[1] = add(v[1], v[6]);
    v[2] = add(v[2], v[7]);
    v[3] = add(v[3], v[4]);
    v[15] = xor(v[15], v[0]);
    v[12] = xor(v[12], v[1]);
    v[13] = xor(v[13], v[2]);
    v[14] = xor(v[14], v[3]);
    v[15] = rot8(v[15]);
    v[12] = rot8(v[12]);
    v[13] = rot8(v[13]);
    v[14] = rot8(v[14]);
    v[10] = add(v[10], v[15]);
    v[11] = add(v[11], v[12]);
    v[8] = add(v[8], v[13]);
    v[9] = add(v[9], v[14]);
    v[5] = xor(v[5], v[10]);
    v[6] = xor(v[6], v[11]);
    v[7] = xor(v[7], v[8]);
    v[4] = xor(v[4], v[9]);
    v[5] = rot7(v[5]);
    v[6] = rot7(v[6]);
    v[7] = rot7(v[7]);
    v[4] = rot7(v[4]);
}

#[inline(always)]
unsafe fn compress4_transposed_inline(
    h_vecs: &mut [__m128i; 8],
    msg0: &Block,
    msg1: &Block,
    msg2: &Block,
    msg3: &Block,
    count_low: __m128i,
    count_high: __m128i,
    lastblock: __m128i,
    lastnode: __m128i,
) {
    let mut v = [
        h_vecs[0],
        h_vecs[1],
        h_vecs[2],
        h_vecs[3],
        h_vecs[4],
        h_vecs[5],
        h_vecs[6],
        h_vecs[7],
        set1(IV[0]),
        set1(IV[1]),
        set1(IV[2]),
        set1(IV[3]),
        xor(set1(IV[4]), count_low),
        xor(set1(IV[5]), count_high),
        xor(set1(IV[6]), lastblock),
        xor(set1(IV[7]), lastnode),
    ];

    let msg_vecs = transpose_message_blocks(msg0, msg1, msg2, msg3);

    round_4(&mut v, &msg_vecs, 0);
    round_4(&mut v, &msg_vecs, 1);
    round_4(&mut v, &msg_vecs, 2);
    round_4(&mut v, &msg_vecs, 3);
    round_4(&mut v, &msg_vecs, 4);
    round_4(&mut v, &msg_vecs, 5);
    round_4(&mut v, &msg_vecs, 6);
    round_4(&mut v, &msg_vecs, 7);
    round_4(&mut v, &msg_vecs, 8);
    round_4(&mut v, &msg_vecs, 9);

    h_vecs[0] = xor(xor(h_vecs[0], v[0]), v[8]);
    h_vecs[1] = xor(xor(h_vecs[1], v[1]), v[9]);
    h_vecs[2] = xor(xor(h_vecs[2], v[2]), v[10]);
    h_vecs[3] = xor(xor(h_vecs[3], v[3]), v[11]);
    h_vecs[4] = xor(xor(h_vecs[4], v[4]), v[12]);
    h_vecs[5] = xor(xor(h_vecs[5], v[5]), v[13]);
    h_vecs[6] = xor(xor(h_vecs[6], v[6]), v[14]);
    h_vecs[7] = xor(xor(h_vecs[7], v[7]), v[15]);
}

// Currently just for benchmarking.
#[target_feature(enable = "sse2")]
pub unsafe fn compress4_transposed(
    h_vecs: &mut [__m128i; 8],
    msg0: &Block,
    msg1: &Block,
    msg2: &Block,
    msg3: &Block,
    count_low: __m128i,
    count_high: __m128i,
    lastblock: __m128i,
    lastnode: __m128i,
) {
    compress4_transposed_inline(
        h_vecs, msg0, msg1, msg2, msg3, count_low, count_high, lastblock, lastnode,
    );
}

#[inline(always)]
unsafe fn export_hashes(h_vecs: &[__m128i; 8], hash_length: u8) -> [Hash; 4] {
    let mut bytes0 = [0; OUTBYTES];
    let mut bytes1 = [0; OUTBYTES];
    let mut bytes2 = [0; OUTBYTES];
    let mut bytes3 = [0; OUTBYTES];
    // Transpose is its own inverse.
    let deinterleaved_lo = transpose_vecs(h_vecs[0], h_vecs[1], h_vecs[2], h_vecs[3]);
    storeu(&mut bytes0[0] as *mut u8 as *mut _, deinterleaved_lo[0]);
    storeu(&mut bytes1[0] as *mut u8 as *mut _, deinterleaved_lo[1]);
    storeu(&mut bytes2[0] as *mut u8 as *mut _, deinterleaved_lo[2]);
    storeu(&mut bytes3[0] as *mut u8 as *mut _, deinterleaved_lo[3]);
    let deinterleaved_hi = transpose_vecs(h_vecs[4], h_vecs[5], h_vecs[6], h_vecs[7]);
    storeu(&mut bytes0[16] as *mut u8 as *mut _, deinterleaved_hi[0]);
    storeu(&mut bytes1[16] as *mut u8 as *mut _, deinterleaved_hi[1]);
    storeu(&mut bytes2[16] as *mut u8 as *mut _, deinterleaved_hi[2]);
    storeu(&mut bytes3[16] as *mut u8 as *mut _, deinterleaved_hi[3]);
    // BLAKE2 and AVX2 both use little-endian representation, so we can just transmute the word
    // bytes out of each de-interleaved vector.
    [
        Hash {
            len: hash_length,
            bytes: bytes0,
        },
        Hash {
            len: hash_length,
            bytes: bytes1,
        },
        Hash {
            len: hash_length,
            bytes: bytes2,
        },
        Hash {
            len: hash_length,
            bytes: bytes3,
        },
    ]
}

#[target_feature(enable = "sse2")]
pub unsafe fn hash4_exact(
    params: &Params,
    input0: &[u8],
    input1: &[u8],
    input2: &[u8],
    input3: &[u8],
) -> [Hash; 4] {
    // INVARIANTS! The caller must assert:
    //   1. The inputs are the same length.
    //   2. The inputs are a multiple of the block size.
    //   3. The inputs aren't empty.

    let param_words = params.make_words();
    // This creates word vectors in an aready-transposed position.
    let mut h_vecs = [
        set1(param_words[0]),
        set1(param_words[1]),
        set1(param_words[2]),
        set1(param_words[3]),
        set1(param_words[4]),
        set1(param_words[5]),
        set1(param_words[6]),
        set1(param_words[7]),
    ];
    let len = input0.len();
    let mut count = 0;

    loop {
        // Use pointer casts to avoid bounds checks here. The caller has to assert that these exact
        // bounds are valid. Note that if these bounds were wrong, we'd get the wrong hash in any
        // case, because count is an input to the compression function.
        let msg0 = &*(input0.as_ptr().add(count) as *const Block);
        let msg1 = &*(input1.as_ptr().add(count) as *const Block);
        let msg2 = &*(input2.as_ptr().add(count) as *const Block);
        let msg3 = &*(input3.as_ptr().add(count) as *const Block);
        count += BLOCKBYTES;
        let count_low = set1(count as u32);
        let count_high = set1(((count as u64) >> 32) as u32);
        let lastblock = set1(if count == len { !0 } else { 0 });
        let lastnode = set1(if params.last_node && count == len {
            !0
        } else {
            0
        });
        compress4_transposed_inline(
            &mut h_vecs,
            msg0,
            msg1,
            msg2,
            msg3,
            count_low,
            count_high,
            lastblock,
            lastnode,
        );
        if count == len {
            return export_hashes(&h_vecs, params.hash_length);
        }
    }
}
