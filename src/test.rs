use super::*;

const EMPTY_HASH: &str = "69217a3079908094e11121d042354a7c1f55b6482ca1a51e1b250dfd1ed0eef9";
const ABC_HASH: &str = "508c5e8c327c14e2e1a72ba34eeb452f37458b209ed63a294d999b4c86675982";
const ONE_BLOCK_HASH: &str = "ae09db7cd54f42b490ef09b6bc541af688e4959bb8c53f359a6f56e38ab454a3";
const THOUSAND_HASH: &str = "37e9dd47498579c5343fd282c13c62ea824cdfc9b0f4f747a41347414640f62c";

const BLOCK_OF_ONES: &str = "dc3c4c7e77f743a2625e771cf71247d0a74821553b38600d0943316d5ff6987f";
const BLOCK_OF_TWOS: &str = "f8b7bb9bd572ec5555592d19b476a5899334a004e9a181fad4b7236990d05329";
const BLOCK_OF_THREES: &str = "43408522fdef2a8c010093bf726dea07954591677cc4bb0c58421dac49942186";
const BLOCK_OF_FOURS: &str = "9d660d85e5a18fab0223c44932695f27639a73726590b563cf7b6aa09a73d594";
const BLOCK_OF_FIVES: &str = "5bca39cc61c6a7beb4767bf2add509bd46cefe44a7755de72fe55707f88fb10d";
const BLOCK_OF_SIXES: &str = "fc4daaafdd1111282445d562226bb98308b1b2682b66df87857a5ea041d01099";
const BLOCK_OF_SEVENS: &str = "e0110d7690b3be39f4d06b37d3ab5352f3a6cbdccf03ae409d0719c88a81c648";
const BLOCK_OF_EIGHTS: &str = "d9f493e092e69be49af15f25fdbc26c62b62e036b361b4431d33c7236fdeb134";

fn compress_one(compress_fn: CompressFn) -> HexString {
    let mut state = State::new();
    // Normally we'd have to be super careful to avoid passing the AVX2 impl here on non-AVX2
    // platforms, but this is test code so no biggie.
    unsafe {
        compress_fn(&mut state.h, &[0; BLOCKBYTES], BLOCKBYTES as u64, !0, 0);
    }
    bytes_to_hex(&state_words_to_bytes(&state.h))
}

fn compress_eight(compress_fn: Compress8Fn) -> [HexString; 8] {
    let mut state1 = State::new();
    let mut state2 = State::new();
    let mut state3 = State::new();
    let mut state4 = State::new();
    let mut state5 = State::new();
    let mut state6 = State::new();
    let mut state7 = State::new();
    let mut state8 = State::new();
    // Normally we'd have to be super careful to avoid passing the AVX2 impl here on non-AVX2
    // platforms, but this is test code so no biggie.
    unsafe {
        compress_fn(
            &mut state1.h,
            &mut state2.h,
            &mut state3.h,
            &mut state4.h,
            &mut state5.h,
            &mut state6.h,
            &mut state7.h,
            &mut state8.h,
            &[1; BLOCKBYTES],
            &[2; BLOCKBYTES],
            &[3; BLOCKBYTES],
            &[4; BLOCKBYTES],
            &[5; BLOCKBYTES],
            &[6; BLOCKBYTES],
            &[7; BLOCKBYTES],
            &[8; BLOCKBYTES],
            BLOCKBYTES as u64,
            BLOCKBYTES as u64,
            BLOCKBYTES as u64,
            BLOCKBYTES as u64,
            BLOCKBYTES as u64,
            BLOCKBYTES as u64,
            BLOCKBYTES as u64,
            BLOCKBYTES as u64,
            !0,
            !0,
            !0,
            !0,
            !0,
            !0,
            !0,
            !0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        );
    }
    [
        bytes_to_hex(&state_words_to_bytes(&state1.h)),
        bytes_to_hex(&state_words_to_bytes(&state2.h)),
        bytes_to_hex(&state_words_to_bytes(&state3.h)),
        bytes_to_hex(&state_words_to_bytes(&state4.h)),
        bytes_to_hex(&state_words_to_bytes(&state5.h)),
        bytes_to_hex(&state_words_to_bytes(&state6.h)),
        bytes_to_hex(&state_words_to_bytes(&state7.h)),
        bytes_to_hex(&state_words_to_bytes(&state8.h)),
    ]
}

#[test]
fn test_all_compression_impls() {
    // Test the portable implementation.
    let expected_1 = HexString::from(ONE_BLOCK_HASH).unwrap();
    assert_eq!(expected_1, compress_one(portable::compress));

    let expected_8 = [
        HexString::from(BLOCK_OF_ONES).unwrap(),
        HexString::from(BLOCK_OF_TWOS).unwrap(),
        HexString::from(BLOCK_OF_THREES).unwrap(),
        HexString::from(BLOCK_OF_FOURS).unwrap(),
        HexString::from(BLOCK_OF_FIVES).unwrap(),
        HexString::from(BLOCK_OF_SIXES).unwrap(),
        HexString::from(BLOCK_OF_SEVENS).unwrap(),
        HexString::from(BLOCK_OF_EIGHTS).unwrap(),
    ];
    assert_eq!(expected_8, compress_eight(portable::compress8));

    // // If we're on an AVX2 platform, test the AVX2 implementation.
    // #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    // #[cfg(feature = "std")]
    // {
    //     if is_x86_feature_detected!("avx2") {
    //         assert_eq!(expected_1, compress_one(avx2::compress));
    //         assert_eq!(expected_4, compress_eight(avx2::compress4));
    //     }
    // }
}

#[test]
fn test_vectors() {
    let io = &[
        (&b""[..], EMPTY_HASH),
        (&b"abc"[..], ABC_HASH),
        (&[0; BLOCKBYTES], ONE_BLOCK_HASH),
        (&[0; 1000], THOUSAND_HASH),
    ];
    // Test each input all at once.
    for &(input, output) in io {
        println!("input {:?}", input);
        let hash = blake2s(input);
        assert_eq!(&hash.to_hex(), output, "hash mismatch");
    }
    // Now in two chunks. This is especially important for the ONE_BLOCK case, because it would be
    // a mistake for update() to call compress, even though the buffer is full.
    for &(input, output) in io {
        let mut state = State::new();
        let split = input.len() / 2;
        state.update(&input[..split]);
        assert_eq!(split as u64, state.count());
        state.update(&input[split..]);
        assert_eq!(input.len() as u64, state.count());
        let hash = state.finalize();
        assert_eq!(&hash.to_hex(), output, "hash mismatch");
    }
    // Now one byte at a time.
    for &(input, output) in io {
        let mut state = State::new();
        let mut count = 0;
        for &b in input {
            state.update(&[b]);
            count += 1;
            assert_eq!(count, state.count());
        }
        let hash = state.finalize();
        assert_eq!(&hash.to_hex(), output, "hash mismatch");
    }
}

#[test]
fn test_multiple_finalizes() {
    let mut state = State::new();
    assert_eq!(&state.finalize().to_hex(), EMPTY_HASH, "hash mismatch");
    assert_eq!(&state.finalize().to_hex(), EMPTY_HASH, "hash mismatch");
    assert_eq!(&state.finalize().to_hex(), EMPTY_HASH, "hash mismatch");
    state.update(b"abc");
    assert_eq!(&state.finalize().to_hex(), ABC_HASH, "hash mismatch");
    assert_eq!(&state.finalize().to_hex(), ABC_HASH, "hash mismatch");
    assert_eq!(&state.finalize().to_hex(), ABC_HASH, "hash mismatch");
}

#[cfg(feature = "std")]
#[test]
fn test_write() {
    use std::io::prelude::*;

    let mut state = State::new();
    state.write_all(&[0; 1000]).unwrap();
    let hash = state.finalize();
    assert_eq!(&hash.to_hex(), THOUSAND_HASH, "hash mismatch");
}

// You can check this case against the equivalent Python:
//
// import hashlib
// hashlib.blake2s(
//     b'foo',
//     digest_size=18,
//     key=b"bar",
//     salt=b"bazbazba",
//     person=b"bing bin",
//     fanout=2,
//     depth=3,
//     leaf_size=0x04050607,
//     node_offset=(2**48 - 1),
//     node_depth=16,
//     inner_size=17,
//     last_node=True,
// ).hexdigest()
#[test]
fn test_all_parameters() {
    let hash = Params::new()
        .hash_length(18)
        // Make sure a shorter key properly overwrites a longer one.
        .key(b"not the real key")
        .key(b"bar")
        .salt(b"bazbazba")
        .personal(b"bing bin")
        .fanout(2)
        .max_depth(3)
        .max_leaf_length(0x04050607)
        .node_offset((1 << 48) - 1)
        .node_depth(16)
        .inner_hash_length(17)
        .to_state()
        .set_last_node(true)
        .update(b"foo")
        .finalize();
    assert_eq!("0d9841e93b8f1d4c0666da56e2bae569c13b", &hash.to_hex());
}

// #[test]
// fn test_all_parameters_blake2bp() {
//     let hash = blake2bp::Params::new()
//         .hash_length(18)
//         // Make sure a shorter key properly overwrites a longer one.
//         .key(b"not the real key")
//         .key(b"bar")
//         .to_state()
//         .update(b"foo")
//         .finalize();
//     assert_eq!("8c54e888a8a01c63da6585c058fe54ea81df", &hash.to_hex());
// }

#[test]
#[should_panic]
fn test_short_hash_length_panics() {
    Params::new().hash_length(0);
}

#[test]
#[should_panic]
fn test_long_hash_length_panics() {
    Params::new().hash_length(OUTBYTES + 1);
}

#[test]
#[should_panic]
fn test_long_key_panics() {
    Params::new().key(&[0; KEYBYTES + 1]);
}

#[test]
#[should_panic]
fn test_long_salt_panics() {
    Params::new().salt(&[0; SALTBYTES + 1]);
}

#[test]
#[should_panic]
fn test_long_personal_panics() {
    Params::new().personal(&[0; PERSONALBYTES + 1]);
}

#[test]
#[should_panic]
fn test_zero_max_depth_panics() {
    Params::new().max_depth(0);
}

#[test]
#[should_panic]
fn test_long_inner_hash_length_panics() {
    Params::new().inner_hash_length(OUTBYTES + 1);
}

// #[test]
// #[should_panic]
// fn test_blake2bp_short_hash_length_panics() {
//     blake2bp::Params::new().hash_length(0);
// }

// #[test]
// #[should_panic]
// fn test_blake2bp_long_hash_length_panics() {
//     blake2bp::Params::new().hash_length(OUTBYTES + 1);
// }

// #[test]
// #[should_panic]
// fn test_blake2bp_long_key_panics() {
//     blake2bp::Params::new().key(&[0; KEYBYTES + 1]);
// }

pub(crate) fn paint_input(buf: &mut [u8]) {
    let mut offset = 0;
    let mut counter: u32 = 1;
    while offset < buf.len() {
        let mut bytes = [0; 4];
        LittleEndian::write_u32(&mut bytes, counter);
        let take = cmp::min(4, buf.len() - offset);
        buf[offset..][..take].copy_from_slice(&bytes[..take]);
        counter += 1;
        offset += take;
    }
}

#[test]
fn test_update8() {
    const INPUT_PREFIX: &[u8] = b"foobarbaz";

    // Define an inner test run function, because we're going to run different permutations of
    // states and inputs.
    fn test_run(
        state0: &mut State,
        state1: &mut State,
        state2: &mut State,
        state3: &mut State,
        state4: &mut State,
        state5: &mut State,
        state6: &mut State,
        state7: &mut State,
        input0: &[u8],
        input1: &[u8],
        input2: &[u8],
        input3: &[u8],
        input4: &[u8],
        input5: &[u8],
        input6: &[u8],
        input7: &[u8],
    ) {
        // Compute the expected hashes the normal way, using cloned copies.
        let expected0 = state0.clone().update(input0).finalize();
        let expected1 = state1.clone().update(input1).finalize();
        let expected2 = state2.clone().update(input2).finalize();
        let expected3 = state3.clone().update(input3).finalize();
        let expected4 = state4.clone().update(input4).finalize();
        let expected5 = state5.clone().update(input5).finalize();
        let expected6 = state6.clone().update(input6).finalize();
        let expected7 = state7.clone().update(input7).finalize();

        // Now do the same thing using the parallel interface.
        update8(
            state0, state1, state2, state3, state4, state5, state6, state7, input0, input1, input2,
            input3, input4, input5, input6, input7,
        );
        let output = finalize8(
            state0, state1, state2, state3, state4, state5, state6, state7,
        );

        assert_eq!(expected0, output[0]);
        assert_eq!(expected1, output[1]);
        assert_eq!(expected2, output[2]);
        assert_eq!(expected3, output[3]);
        assert_eq!(expected4, output[4]);
        assert_eq!(expected5, output[5]);
        assert_eq!(expected6, output[6]);
        assert_eq!(expected7, output[7]);
    }

    // State A is default.
    let mut state_a = State::new();
    // State B sets last node on the state.
    let mut state_b = State::new();
    state_b.set_last_node(true);
    // State C gets a "foobarbaz" prefix.
    let mut state_c = State::new();
    state_c.update(INPUT_PREFIX);
    // State D gets wacky parameters.
    let mut state_d = Params::new()
        .hash_length(18)
        .key(b"bar")
        .salt(b"bazbazba")
        .personal(b"bing bin")
        .fanout(2)
        .max_depth(3)
        .max_leaf_length(0x04050607)
        .node_offset((1 << 48) - 1)
        .node_depth(16)
        .inner_hash_length(17)
        .last_node(true)
        .to_state();
    // States E/F/G/H are also default.
    let mut state_e = State::new();
    let mut state_f = State::new();
    let mut state_g = State::new();
    let mut state_h = State::new();

    let mut input = [0; 75 * BLOCKBYTES];
    paint_input(&mut input);
    let input_i = &input[0_ * BLOCKBYTES..10 * BLOCKBYTES];
    let input_j = &input[10 * BLOCKBYTES..20 * BLOCKBYTES];
    let input_k = &input[20 * BLOCKBYTES..30 * BLOCKBYTES];
    let input_l = &input[30 * BLOCKBYTES..40 * BLOCKBYTES];
    let input_m = &input[40 * BLOCKBYTES..50 * BLOCKBYTES];
    let input_n = &input[50 * BLOCKBYTES..60 * BLOCKBYTES];
    let input_o = &input[60 * BLOCKBYTES..70 * BLOCKBYTES];
    // Input P is short.
    let input_p = &input[70 * BLOCKBYTES..75 * BLOCKBYTES];

    // Loop over eight different permutations of the input.
    for (input0, input1, input2, input3, input4, input5, input6, input7) in &[
        (
            input_i, input_j, input_k, input_l, input_m, input_n, input_o, input_p,
        ),
        (
            input_j, input_k, input_l, input_m, input_n, input_o, input_p, input_i,
        ),
        (
            input_k, input_l, input_m, input_n, input_o, input_p, input_i, input_j,
        ),
        (
            input_l, input_m, input_n, input_o, input_p, input_i, input_j, input_k,
        ),
        (
            input_m, input_n, input_o, input_p, input_i, input_j, input_k, input_l,
        ),
        (
            input_n, input_o, input_p, input_i, input_j, input_k, input_l, input_m,
        ),
        (
            input_o, input_p, input_i, input_j, input_k, input_l, input_m, input_n,
        ),
        (
            input_p, input_i, input_j, input_k, input_l, input_m, input_n, input_o,
        ),
    ] {
        // For each input permutation, run eight permutations of the states.
        test_run(
            &mut state_a,
            &mut state_b,
            &mut state_c,
            &mut state_d,
            &mut state_e,
            &mut state_f,
            &mut state_g,
            &mut state_h,
            input0,
            input1,
            input2,
            input3,
            input4,
            input5,
            input6,
            input7,
        );
        test_run(
            &mut state_b,
            &mut state_c,
            &mut state_d,
            &mut state_e,
            &mut state_f,
            &mut state_g,
            &mut state_h,
            &mut state_a,
            input0,
            input1,
            input2,
            input3,
            input4,
            input5,
            input6,
            input7,
        );
        test_run(
            &mut state_c,
            &mut state_d,
            &mut state_e,
            &mut state_f,
            &mut state_g,
            &mut state_h,
            &mut state_a,
            &mut state_b,
            input0,
            input1,
            input2,
            input3,
            input4,
            input5,
            input6,
            input7,
        );
        test_run(
            &mut state_d,
            &mut state_e,
            &mut state_f,
            &mut state_g,
            &mut state_h,
            &mut state_a,
            &mut state_b,
            &mut state_c,
            input0,
            input1,
            input2,
            input3,
            input4,
            input5,
            input6,
            input7,
        );
        test_run(
            &mut state_e,
            &mut state_f,
            &mut state_g,
            &mut state_h,
            &mut state_a,
            &mut state_b,
            &mut state_c,
            &mut state_d,
            input0,
            input1,
            input2,
            input3,
            input4,
            input5,
            input6,
            input7,
        );
        test_run(
            &mut state_f,
            &mut state_g,
            &mut state_h,
            &mut state_a,
            &mut state_b,
            &mut state_c,
            &mut state_d,
            &mut state_e,
            input0,
            input1,
            input2,
            input3,
            input4,
            input5,
            input6,
            input7,
        );
        test_run(
            &mut state_g,
            &mut state_h,
            &mut state_a,
            &mut state_b,
            &mut state_c,
            &mut state_d,
            &mut state_e,
            &mut state_f,
            input0,
            input1,
            input2,
            input3,
            input4,
            input5,
            input6,
            input7,
        );
        test_run(
            &mut state_h,
            &mut state_a,
            &mut state_b,
            &mut state_c,
            &mut state_d,
            &mut state_e,
            &mut state_f,
            &mut state_g,
            input0,
            input1,
            input2,
            input3,
            input4,
            input5,
            input6,
            input7,
        );
    }
}
