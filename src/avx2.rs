#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use core::mem;

use crate::AlignedWords8;
use crate::Block;
use crate::Hash;
use crate::Params;
use crate::StateWords;
use crate::BLOCKBYTES;
use crate::IV;
use crate::SIGMA;

#[inline(always)]
unsafe fn loadu(p: *const u32) -> __m256i {
    _mm256_loadu_si256(p as *const __m256i)
}

#[inline(always)]
unsafe fn add(a: __m256i, b: __m256i) -> __m256i {
    _mm256_add_epi32(a, b)
}

#[inline(always)]
unsafe fn xor(a: __m256i, b: __m256i) -> __m256i {
    _mm256_xor_si256(a, b)
}

#[inline(always)]
unsafe fn rot16(x: __m256i) -> __m256i {
    _mm256_shuffle_epi8(
        x,
        _mm256_set_epi8(
            13, 12, 15, 14, 9, 8, 11, 10, 5, 4, 7, 6, 1, 0, 3, 2, 13, 12, 15, 14, 9, 8, 11, 10, 5,
            4, 7, 6, 1, 0, 3, 2,
        ),
    )
}

#[inline(always)]
unsafe fn rot12(x: __m256i) -> __m256i {
    _mm256_or_si256(_mm256_srli_epi32(x, 12), _mm256_slli_epi32(x, 20))
}

#[inline(always)]
unsafe fn rot8(x: __m256i) -> __m256i {
    _mm256_shuffle_epi8(
        x,
        _mm256_set_epi8(
            12, 15, 14, 13, 8, 11, 10, 9, 4, 7, 6, 5, 0, 3, 2, 1, 12, 15, 14, 13, 8, 11, 10, 9, 4,
            7, 6, 5, 0, 3, 2, 1,
        ),
    )
}

#[inline(always)]
unsafe fn rot7(x: __m256i) -> __m256i {
    _mm256_or_si256(_mm256_srli_epi32(x, 7), _mm256_slli_epi32(x, 25))
}

#[inline(always)]
unsafe fn load_256_from_u32(x: u32) -> __m256i {
    _mm256_set1_epi32(x as i32)
}

#[inline(always)]
unsafe fn load_256_from_8xu32(
    x1: u32,
    x2: u32,
    x3: u32,
    x4: u32,
    x5: u32,
    x6: u32,
    x7: u32,
    x8: u32,
) -> __m256i {
    // NOTE: This order of arguments for _mm256_set_epi32 is the reverse of how the ints come out
    // when you transmute them back into an array of u32's.
    _mm256_set_epi32(
        x8 as i32, x7 as i32, x6 as i32, x5 as i32, x4 as i32, x3 as i32, x2 as i32, x1 as i32,
    )
}

// NOTE: Writing out the whole round explicitly in this way gives better
// performance than we get if we factor out the G function. Perhaps the
// compiler doesn't notice that it can group all the adds together like we do
// here, even when G is inlined.
#[inline(always)]
unsafe fn blake2s_round_8x(v: &mut [__m256i; 16], m: &[__m256i; 16], r: usize) {
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

#[target_feature(enable = "avx2")]
pub unsafe fn compress8(
    h0: &mut StateWords,
    h1: &mut StateWords,
    h2: &mut StateWords,
    h3: &mut StateWords,
    h4: &mut StateWords,
    h5: &mut StateWords,
    h6: &mut StateWords,
    h7: &mut StateWords,
    msg0: &Block,
    msg1: &Block,
    msg2: &Block,
    msg3: &Block,
    msg4: &Block,
    msg5: &Block,
    msg6: &Block,
    msg7: &Block,
    count0: u64,
    count1: u64,
    count2: u64,
    count3: u64,
    count4: u64,
    count5: u64,
    count6: u64,
    count7: u64,
    lastblock0: u32,
    lastblock1: u32,
    lastblock2: u32,
    lastblock3: u32,
    lastblock4: u32,
    lastblock5: u32,
    lastblock6: u32,
    lastblock7: u32,
    lastnode0: u32,
    lastnode1: u32,
    lastnode2: u32,
    lastnode3: u32,
    lastnode4: u32,
    lastnode5: u32,
    lastnode6: u32,
    lastnode7: u32,
) {
    let mut h_vecs = [
        loadu(h0.as_ptr()),
        loadu(h1.as_ptr()),
        loadu(h2.as_ptr()),
        loadu(h3.as_ptr()),
        loadu(h4.as_ptr()),
        loadu(h5.as_ptr()),
        loadu(h6.as_ptr()),
        loadu(h7.as_ptr()),
    ];
    transpose_vecs(&mut h_vecs);

    let count_low = load_256_from_8xu32(
        count0 as u32,
        count1 as u32,
        count2 as u32,
        count3 as u32,
        count4 as u32,
        count5 as u32,
        count6 as u32,
        count7 as u32,
    );
    let count_high = load_256_from_8xu32(
        (count0 >> 32) as u32,
        (count1 >> 32) as u32,
        (count2 >> 32) as u32,
        (count3 >> 32) as u32,
        (count4 >> 32) as u32,
        (count5 >> 32) as u32,
        (count6 >> 32) as u32,
        (count7 >> 32) as u32,
    );
    let lastblock = load_256_from_8xu32(
        lastblock0 as u32,
        lastblock1 as u32,
        lastblock2 as u32,
        lastblock3 as u32,
        lastblock4 as u32,
        lastblock5 as u32,
        lastblock6 as u32,
        lastblock7 as u32,
    );
    let lastnode = load_256_from_8xu32(
        lastnode0 as u32,
        lastnode1 as u32,
        lastnode2 as u32,
        lastnode3 as u32,
        lastnode4 as u32,
        lastnode5 as u32,
        lastnode6 as u32,
        lastnode7 as u32,
    );

    let msg_vecs = load_msg_vecs_interleave(msg0, msg1, msg2, msg3, msg4, msg5, msg6, msg7);

    compress8_transposed_inline(
        &mut h_vecs,
        &msg_vecs,
        count_low,
        count_high,
        lastblock,
        lastnode,
    );

    transpose_vecs(&mut h_vecs);

    *h0 = mem::transmute(h_vecs[0]);
    *h1 = mem::transmute(h_vecs[1]);
    *h2 = mem::transmute(h_vecs[2]);
    *h3 = mem::transmute(h_vecs[3]);
    *h4 = mem::transmute(h_vecs[4]);
    *h5 = mem::transmute(h_vecs[5]);
    *h6 = mem::transmute(h_vecs[6]);
    *h7 = mem::transmute(h_vecs[7]);
}

#[inline(always)]
unsafe fn interleave128(a: __m256i, b: __m256i) -> (__m256i, __m256i) {
    (
        _mm256_permute2x128_si256(a, b, 0x20),
        _mm256_permute2x128_si256(a, b, 0x31),
    )
}

#[cfg(test)]
fn cast_out(x: __m256i) -> [u32; 8] {
    unsafe { mem::transmute(x) }
}

#[cfg(test)]
#[test]
fn test_interleave128() {
    #[target_feature(enable = "avx2")]
    unsafe fn inner() {
        let a = load_256_from_8xu32(10, 11, 12, 13, 14, 15, 16, 17);
        let b = load_256_from_8xu32(20, 21, 22, 23, 24, 25, 26, 27);

        let expected_a = load_256_from_8xu32(10, 11, 12, 13, 20, 21, 22, 23);
        let expected_b = load_256_from_8xu32(14, 15, 16, 17, 24, 25, 26, 27);

        let (out_a, out_b) = interleave128(a, b);

        assert_eq!(cast_out(expected_a), cast_out(out_a));
        assert_eq!(cast_out(expected_b), cast_out(out_b));
    }

    #[cfg(feature = "std")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                inner();
            }
        }
    }
}

#[inline(always)]
unsafe fn load_2x256(msg: &[u8; BLOCKBYTES]) -> (__m256i, __m256i) {
    (
        _mm256_loadu_si256(msg.as_ptr() as *const __m256i),
        _mm256_loadu_si256((msg.as_ptr() as *const __m256i).add(1)),
    )
}

#[cfg(test)]
#[test]
fn test_load_2x256() {
    #[target_feature(enable = "avx2")]
    unsafe fn inner() {
        let input: [u64; 8] = [0, 1, 2, 3, 4, 5, 6, 7];
        let input_bytes: [u8; BLOCKBYTES] = mem::transmute(input);
        let (out_a, out_b) = load_2x256(&input_bytes);

        let expected_a = load_256_from_8xu32(0, 0, 1, 0, 2, 0, 3, 0);
        let expected_b = load_256_from_8xu32(4, 0, 5, 0, 6, 0, 7, 0);

        assert_eq!(cast_out(expected_a), cast_out(out_a));
        assert_eq!(cast_out(expected_b), cast_out(out_b));
    }

    #[cfg(feature = "std")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                inner();
            }
        }
    }
}

#[inline(always)]
unsafe fn transpose_vecs(vecs: &mut [__m256i; 8]) {
    // Interleave 32-bit lanes. The low unpack is lanes 00/11/44/55, and the high is 22/33/66/77.
    let ab_0145 = _mm256_unpacklo_epi32(vecs[0], vecs[1]);
    let ab_2367 = _mm256_unpackhi_epi32(vecs[0], vecs[1]);
    let cd_0145 = _mm256_unpacklo_epi32(vecs[2], vecs[3]);
    let cd_2367 = _mm256_unpackhi_epi32(vecs[2], vecs[3]);
    let ef_0145 = _mm256_unpacklo_epi32(vecs[4], vecs[5]);
    let ef_2367 = _mm256_unpackhi_epi32(vecs[4], vecs[5]);
    let gh_0145 = _mm256_unpacklo_epi32(vecs[6], vecs[7]);
    let gh_2367 = _mm256_unpackhi_epi32(vecs[6], vecs[7]);

    // Interleave 64-bit lates. The low unpack is lanes 00/22 and the high is 11/33.
    let abcd_04 = _mm256_unpacklo_epi64(ab_0145, cd_0145);
    let abcd_15 = _mm256_unpackhi_epi64(ab_0145, cd_0145);
    let abcd_26 = _mm256_unpacklo_epi64(ab_2367, cd_2367);
    let abcd_37 = _mm256_unpackhi_epi64(ab_2367, cd_2367);
    let efgh_04 = _mm256_unpacklo_epi64(ef_0145, gh_0145);
    let efgh_15 = _mm256_unpackhi_epi64(ef_0145, gh_0145);
    let efgh_26 = _mm256_unpacklo_epi64(ef_2367, gh_2367);
    let efgh_37 = _mm256_unpackhi_epi64(ef_2367, gh_2367);

    // Interleave 128-bit lanes.
    let (abcdefg_0, abcdefg_4) = interleave128(abcd_04, efgh_04);
    let (abcdefg_1, abcdefg_5) = interleave128(abcd_15, efgh_15);
    let (abcdefg_2, abcdefg_6) = interleave128(abcd_26, efgh_26);
    let (abcdefg_3, abcdefg_7) = interleave128(abcd_37, efgh_37);

    vecs[0] = abcdefg_0;
    vecs[1] = abcdefg_1;
    vecs[2] = abcdefg_2;
    vecs[3] = abcdefg_3;
    vecs[4] = abcdefg_4;
    vecs[5] = abcdefg_5;
    vecs[6] = abcdefg_6;
    vecs[7] = abcdefg_7;
}

#[target_feature(enable = "avx2")]
pub unsafe fn vectorize_words8(words: &mut [AlignedWords8; 8]) {
    let vecs = &mut *(words as *mut _ as *mut [__m256i; 8]);
    transpose_vecs(vecs);
}

#[cfg(test)]
#[test]
fn test_transpose_vecs() {
    #[target_feature(enable = "avx2")]
    unsafe fn inner() {
        let vec_a = load_256_from_8xu32(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07);
        let vec_b = load_256_from_8xu32(0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17);
        let vec_c = load_256_from_8xu32(0x20, 0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27);
        let vec_d = load_256_from_8xu32(0x30, 0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37);
        let vec_e = load_256_from_8xu32(0x40, 0x41, 0x42, 0x43, 0x44, 0x45, 0x46, 0x47);
        let vec_f = load_256_from_8xu32(0x50, 0x51, 0x52, 0x53, 0x54, 0x55, 0x56, 0x57);
        let vec_g = load_256_from_8xu32(0x60, 0x61, 0x62, 0x63, 0x64, 0x65, 0x66, 0x67);
        let vec_h = load_256_from_8xu32(0x70, 0x71, 0x72, 0x73, 0x74, 0x75, 0x76, 0x77);

        let expected_a = load_256_from_8xu32(0x00, 0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70);
        let expected_b = load_256_from_8xu32(0x01, 0x11, 0x21, 0x31, 0x41, 0x51, 0x61, 0x71);
        let expected_c = load_256_from_8xu32(0x02, 0x12, 0x22, 0x32, 0x42, 0x52, 0x62, 0x72);
        let expected_d = load_256_from_8xu32(0x03, 0x13, 0x23, 0x33, 0x43, 0x53, 0x63, 0x73);
        let expected_e = load_256_from_8xu32(0x04, 0x14, 0x24, 0x34, 0x44, 0x54, 0x64, 0x74);
        let expected_f = load_256_from_8xu32(0x05, 0x15, 0x25, 0x35, 0x45, 0x55, 0x65, 0x75);
        let expected_g = load_256_from_8xu32(0x06, 0x16, 0x26, 0x36, 0x46, 0x56, 0x66, 0x76);
        let expected_h = load_256_from_8xu32(0x07, 0x17, 0x27, 0x37, 0x47, 0x57, 0x67, 0x77);

        let mut interleaved = [vec_a, vec_b, vec_c, vec_d, vec_e, vec_f, vec_g, vec_h];
        transpose_vecs(&mut interleaved);

        let [out_a, out_b, out_c, out_d, out_e, out_f, out_g, out_h] = interleaved;
        assert_eq!(cast_out(expected_a), cast_out(out_a));
        assert_eq!(cast_out(expected_b), cast_out(out_b));
        assert_eq!(cast_out(expected_c), cast_out(out_c));
        assert_eq!(cast_out(expected_d), cast_out(out_d));
        assert_eq!(cast_out(expected_e), cast_out(out_e));
        assert_eq!(cast_out(expected_f), cast_out(out_f));
        assert_eq!(cast_out(expected_g), cast_out(out_g));
        assert_eq!(cast_out(expected_h), cast_out(out_h));

        // Check that interleaving again undoes the operation.
        let mut deinterleaved = [out_a, out_b, out_c, out_d, out_e, out_f, out_g, out_h];
        transpose_vecs(&mut deinterleaved);
        let [out2_a, out2_b, out2_c, out2_d, out2_e, out2_f, out2_g, out2_h] = deinterleaved;
        assert_eq!(cast_out(vec_a), cast_out(out2_a));
        assert_eq!(cast_out(vec_b), cast_out(out2_b));
        assert_eq!(cast_out(vec_c), cast_out(out2_c));
        assert_eq!(cast_out(vec_d), cast_out(out2_d));
        assert_eq!(cast_out(vec_e), cast_out(out2_e));
        assert_eq!(cast_out(vec_f), cast_out(out2_f));
        assert_eq!(cast_out(vec_g), cast_out(out2_g));
        assert_eq!(cast_out(vec_h), cast_out(out2_h));
    }

    #[cfg(feature = "std")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                inner();
            }
        }
    }
}

#[inline(always)]
unsafe fn load_msg_vecs_interleave(
    msg_a: &[u8; BLOCKBYTES],
    msg_b: &[u8; BLOCKBYTES],
    msg_c: &[u8; BLOCKBYTES],
    msg_d: &[u8; BLOCKBYTES],
    msg_e: &[u8; BLOCKBYTES],
    msg_f: &[u8; BLOCKBYTES],
    msg_g: &[u8; BLOCKBYTES],
    msg_h: &[u8; BLOCKBYTES],
) -> [__m256i; 16] {
    let (front_a, back_a) = load_2x256(msg_a);
    let (front_b, back_b) = load_2x256(msg_b);
    let (front_c, back_c) = load_2x256(msg_c);
    let (front_d, back_d) = load_2x256(msg_d);
    let (front_e, back_e) = load_2x256(msg_e);
    let (front_f, back_f) = load_2x256(msg_f);
    let (front_g, back_g) = load_2x256(msg_g);
    let (front_h, back_h) = load_2x256(msg_h);

    let mut front_interleaved = [
        front_a, front_b, front_c, front_d, front_e, front_f, front_g, front_h,
    ];
    transpose_vecs(&mut front_interleaved);
    let mut back_interleaved = [
        back_a, back_b, back_c, back_d, back_e, back_f, back_g, back_h,
    ];
    transpose_vecs(&mut back_interleaved);

    [
        front_interleaved[0],
        front_interleaved[1],
        front_interleaved[2],
        front_interleaved[3],
        front_interleaved[4],
        front_interleaved[5],
        front_interleaved[6],
        front_interleaved[7],
        back_interleaved[0],
        back_interleaved[1],
        back_interleaved[2],
        back_interleaved[3],
        back_interleaved[4],
        back_interleaved[5],
        back_interleaved[6],
        back_interleaved[7],
    ]
}

// This function assumes that the state is in transposed form, but not
// necessarily aligned. It accepts input in the usual form of contiguous bytes,
// and it pays the cost of transposing the input.
#[target_feature(enable = "avx2")]
pub unsafe fn compress8_vectorized(
    states: &mut [AlignedWords8; 8],
    msg0: &Block,
    msg1: &Block,
    msg2: &Block,
    msg3: &Block,
    msg4: &Block,
    msg5: &Block,
    msg6: &Block,
    msg7: &Block,
    count_low: &AlignedWords8,
    count_high: &AlignedWords8,
    lastblock: &AlignedWords8,
    lastnode: &AlignedWords8,
) {
    let mut h_vecs = &mut *(states as *mut _ as *mut [__m256i; 8]);

    let msg_vecs = load_msg_vecs_interleave(msg0, msg1, msg2, msg3, msg4, msg5, msg6, msg7);

    compress8_transposed_inline(
        &mut h_vecs,
        &msg_vecs,
        mem::transmute(*count_low),
        mem::transmute(*count_high),
        mem::transmute(*lastblock),
        mem::transmute(*lastnode),
    );
}

#[target_feature(enable = "avx2")]
pub unsafe fn compress8_transposed_all(
    h_vecs: &mut [__m256i; 8],
    msg_vecs: &[__m256i; 16],
    count_low: __m256i,
    count_high: __m256i,
    lastblock: __m256i,
    lastnode: __m256i,
) {
    compress8_transposed_inline(h_vecs, msg_vecs, count_low, count_high, lastblock, lastnode);
}

// This core function assumes that both the state words and the message blocks
// have been transposed across vectors. So the first state vector contains the
// first word of each of the 8 states, and the first message vector contains
// the first word of each of the 8 message blocks. Defining the core this way
// allows us to keep either the state or the message in transposed form in some
// cases, to avoid paying the cost of transposing them.
#[inline(always)]
unsafe fn compress8_transposed_inline(
    h_vecs: &mut [__m256i; 8],
    msg_vecs: &[__m256i; 16],
    count_low: __m256i,
    count_high: __m256i,
    lastblock: __m256i,
    lastnode: __m256i,
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
        load_256_from_u32(IV[0]),
        load_256_from_u32(IV[1]),
        load_256_from_u32(IV[2]),
        load_256_from_u32(IV[3]),
        xor(load_256_from_u32(IV[4]), count_low),
        xor(load_256_from_u32(IV[5]), count_high),
        xor(load_256_from_u32(IV[6]), lastblock),
        xor(load_256_from_u32(IV[7]), lastnode),
    ];

    blake2s_round_8x(&mut v, &msg_vecs, 0);
    blake2s_round_8x(&mut v, &msg_vecs, 1);
    blake2s_round_8x(&mut v, &msg_vecs, 2);
    blake2s_round_8x(&mut v, &msg_vecs, 3);
    blake2s_round_8x(&mut v, &msg_vecs, 4);
    blake2s_round_8x(&mut v, &msg_vecs, 5);
    blake2s_round_8x(&mut v, &msg_vecs, 6);
    blake2s_round_8x(&mut v, &msg_vecs, 7);
    blake2s_round_8x(&mut v, &msg_vecs, 8);
    blake2s_round_8x(&mut v, &msg_vecs, 9);

    h_vecs[0] = xor(h_vecs[0], xor(v[0], v[8]));
    h_vecs[1] = xor(h_vecs[1], xor(v[1], v[9]));
    h_vecs[2] = xor(h_vecs[2], xor(v[2], v[10]));
    h_vecs[3] = xor(h_vecs[3], xor(v[3], v[11]));
    h_vecs[4] = xor(h_vecs[4], xor(v[4], v[12]));
    h_vecs[5] = xor(h_vecs[5], xor(v[5], v[13]));
    h_vecs[6] = xor(h_vecs[6], xor(v[6], v[14]));
    h_vecs[7] = xor(h_vecs[7], xor(v[7], v[15]));
}

#[inline(always)]
unsafe fn export_hashes(h_vecs: &[__m256i; 8], hash_length: u8) -> [Hash; 8] {
    // Interleave is its own inverse.
    let mut deinterleaved = *h_vecs;
    transpose_vecs(&mut deinterleaved);
    // BLAKE2 and x86 both use little-endian representation, so we can just transmute the word
    // bytes out of each de-interleaved vector.
    [
        Hash {
            len: hash_length,
            bytes: mem::transmute(deinterleaved[0]),
        },
        Hash {
            len: hash_length,
            bytes: mem::transmute(deinterleaved[1]),
        },
        Hash {
            len: hash_length,
            bytes: mem::transmute(deinterleaved[2]),
        },
        Hash {
            len: hash_length,
            bytes: mem::transmute(deinterleaved[3]),
        },
        Hash {
            len: hash_length,
            bytes: mem::transmute(deinterleaved[4]),
        },
        Hash {
            len: hash_length,
            bytes: mem::transmute(deinterleaved[5]),
        },
        Hash {
            len: hash_length,
            bytes: mem::transmute(deinterleaved[6]),
        },
        Hash {
            len: hash_length,
            bytes: mem::transmute(deinterleaved[7]),
        },
    ]
}

#[target_feature(enable = "avx2")]
pub unsafe fn hash8_exact(
    // TODO: Separate params for each input.
    params: &Params,
    input0: &[u8],
    input1: &[u8],
    input2: &[u8],
    input3: &[u8],
    input4: &[u8],
    input5: &[u8],
    input6: &[u8],
    input7: &[u8],
) -> [Hash; 8] {
    // INVARIANTS! The caller must assert:
    //   1. The inputs are the same length.
    //   2. The inputs are a multiple of the block size.
    //   3. The inputs aren't empty.

    let param_words = params.make_words();
    // This creates word vectors in an aready-transposed position.
    let mut h_vecs = [
        load_256_from_u32(param_words[0]),
        load_256_from_u32(param_words[1]),
        load_256_from_u32(param_words[2]),
        load_256_from_u32(param_words[3]),
        load_256_from_u32(param_words[4]),
        load_256_from_u32(param_words[5]),
        load_256_from_u32(param_words[6]),
        load_256_from_u32(param_words[7]),
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
        let msg4 = &*(input4.as_ptr().add(count) as *const Block);
        let msg5 = &*(input5.as_ptr().add(count) as *const Block);
        let msg6 = &*(input6.as_ptr().add(count) as *const Block);
        let msg7 = &*(input7.as_ptr().add(count) as *const Block);
        count += BLOCKBYTES;
        let count_low = load_256_from_u32(count as u32);
        let count_high = load_256_from_u32((count as u64 >> 32) as u32);
        let lastblock = load_256_from_u32(if count == len { !0 } else { 0 });
        let lastnode = load_256_from_u32(if params.last_node && count == len {
            !0
        } else {
            0
        });
        let msg_vecs = load_msg_vecs_interleave(msg0, msg1, msg2, msg3, msg4, msg5, msg6, msg7);
        compress8_transposed_inline(
            &mut h_vecs,
            &msg_vecs,
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
