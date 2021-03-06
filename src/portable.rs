use byteorder::{ByteOrder, LittleEndian};

use super::*;

// G is the mixing function, called eight times per round in the compression
// function. V is the 16-word state vector of the compression function, usually
// described as a 4x4 matrix. A, B, C, and D are the mixing indices, set by the
// caller first to the four columns of V, and then to its four diagonals. X and
// Y are words of input, chosen by the caller according to the message
// schedule, SIGMA.
#[inline(always)]
fn g(v: &mut [u32; 16], a: usize, b: usize, c: usize, d: usize, x: u32, y: u32) {
    v[a] = v[a].wrapping_add(v[b]).wrapping_add(x);
    v[d] = (v[d] ^ v[a]).rotate_right(16);
    v[c] = v[c].wrapping_add(v[d]);
    v[b] = (v[b] ^ v[c]).rotate_right(12);
    v[a] = v[a].wrapping_add(v[b]).wrapping_add(y);
    v[d] = (v[d] ^ v[a]).rotate_right(8);
    v[c] = v[c].wrapping_add(v[d]);
    v[b] = (v[b] ^ v[c]).rotate_right(7);
}

#[inline(always)]
fn round(r: usize, m: &[u32; 16], v: &mut [u32; 16]) {
    // Select the message schedule based on the round.
    let s = SIGMA[r];

    // Mix the columns.
    g(v, 0, 4, 8, 12, m[s[0] as usize], m[s[1] as usize]);
    g(v, 1, 5, 9, 13, m[s[2] as usize], m[s[3] as usize]);
    g(v, 2, 6, 10, 14, m[s[4] as usize], m[s[5] as usize]);
    g(v, 3, 7, 11, 15, m[s[6] as usize], m[s[7] as usize]);

    // Mix the rows.
    g(v, 0, 5, 10, 15, m[s[8] as usize], m[s[9] as usize]);
    g(v, 1, 6, 11, 12, m[s[10] as usize], m[s[11] as usize]);
    g(v, 2, 7, 8, 13, m[s[12] as usize], m[s[13] as usize]);
    g(v, 3, 4, 9, 14, m[s[14] as usize], m[s[15] as usize]);
}

// H is the 8-word state vector. `msg` is BLOCKBYTES of input, possibly padded
// with zero bytes in the final block. `count` is the number of bytes fed so
// far, including in this call, though not including padding in the final call.
// `finalize` is set to true only in the final call.
pub fn compress(h: &mut StateWords, msg: &Block, count: u64, lastblock: u32, lastnode: u32) {
    // Initialize the compression state.
    let mut v = [
        h[0],
        h[1],
        h[2],
        h[3],
        h[4],
        h[5],
        h[6],
        h[7],
        IV[0],
        IV[1],
        IV[2],
        IV[3],
        IV[4] ^ count as u32,
        IV[5] ^ (count >> 32) as u32,
        IV[6] ^ lastblock,
        IV[7] ^ lastnode,
    ];

    // Parse the message bytes as ints in little endian order.
    let msg_refs = array_refs!(msg, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4);
    let m = [
        LittleEndian::read_u32(msg_refs.0),
        LittleEndian::read_u32(msg_refs.1),
        LittleEndian::read_u32(msg_refs.2),
        LittleEndian::read_u32(msg_refs.3),
        LittleEndian::read_u32(msg_refs.4),
        LittleEndian::read_u32(msg_refs.5),
        LittleEndian::read_u32(msg_refs.6),
        LittleEndian::read_u32(msg_refs.7),
        LittleEndian::read_u32(msg_refs.8),
        LittleEndian::read_u32(msg_refs.9),
        LittleEndian::read_u32(msg_refs.10),
        LittleEndian::read_u32(msg_refs.11),
        LittleEndian::read_u32(msg_refs.12),
        LittleEndian::read_u32(msg_refs.13),
        LittleEndian::read_u32(msg_refs.14),
        LittleEndian::read_u32(msg_refs.15),
    ];

    round(0, &m, &mut v);
    round(1, &m, &mut v);
    round(2, &m, &mut v);
    round(3, &m, &mut v);
    round(4, &m, &mut v);
    round(5, &m, &mut v);
    round(6, &m, &mut v);
    round(7, &m, &mut v);
    round(8, &m, &mut v);
    round(9, &m, &mut v);

    h[0] ^= v[0] ^ v[8];
    h[1] ^= v[1] ^ v[9];
    h[2] ^= v[2] ^ v[10];
    h[3] ^= v[3] ^ v[11];
    h[4] ^= v[4] ^ v[12];
    h[5] ^= v[5] ^ v[13];
    h[6] ^= v[6] ^ v[14];
    h[7] ^= v[7] ^ v[15];
}

pub fn compress8(
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
    compress(h0, msg0, count0, lastblock0, lastnode0);
    compress(h1, msg1, count1, lastblock1, lastnode1);
    compress(h2, msg2, count2, lastblock2, lastnode2);
    compress(h3, msg3, count3, lastblock3, lastnode3);
    compress(h4, msg4, count4, lastblock4, lastnode4);
    compress(h5, msg5, count5, lastblock5, lastnode5);
    compress(h6, msg6, count6, lastblock6, lastnode6);
    compress(h7, msg7, count7, lastblock7, lastnode7);
}

pub fn hash4_exact(
    params: &Params,
    input0: &[u8],
    input1: &[u8],
    input2: &[u8],
    input3: &[u8],
) -> [Hash; 4] {
    [
        params.to_state().update(input0).finalize(),
        params.to_state().update(input1).finalize(),
        params.to_state().update(input2).finalize(),
        params.to_state().update(input3).finalize(),
    ]
}

pub fn hash8_exact(
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
    [
        params.to_state().update(input0).finalize(),
        params.to_state().update(input1).finalize(),
        params.to_state().update(input2).finalize(),
        params.to_state().update(input3).finalize(),
        params.to_state().update(input4).finalize(),
        params.to_state().update(input5).finalize(),
        params.to_state().update(input6).finalize(),
        params.to_state().update(input7).finalize(),
    ]
}

pub fn vectorize_words8(_words: &mut [AlignedWords8; 8]) {
    // The portable implementation does nothing here. It leaves the state words
    // grouped by state/input, rather than grouped by index.
}

pub fn compress8_vectorized(
    state_words: &mut [AlignedWords8; 8],
    block0: &Block,
    block1: &Block,
    block2: &Block,
    block3: &Block,
    block4: &Block,
    block5: &Block,
    block6: &Block,
    block7: &Block,
    count_low: &AlignedWords8,
    count_high: &AlignedWords8,
    lastblock: &AlignedWords8,
    lastnode: &AlignedWords8,
) {
    // Note that vectorize_words8 above is a no-op. In the portable
    // implementation, the words are still grouped by state/input.
    compress(
        &mut state_words[0],
        block0,
        count_low[0] as u64 + ((count_high[0] as u64) << 32),
        lastblock[0],
        lastnode[0],
    );
    compress(
        &mut state_words[1],
        block1,
        count_low[1] as u64 + ((count_high[1] as u64) << 32),
        lastblock[1],
        lastnode[1],
    );
    compress(
        &mut state_words[2],
        block2,
        count_low[2] as u64 + ((count_high[2] as u64) << 32),
        lastblock[2],
        lastnode[2],
    );
    compress(
        &mut state_words[3],
        block3,
        count_low[3] as u64 + ((count_high[3] as u64) << 32),
        lastblock[3],
        lastnode[3],
    );
    compress(
        &mut state_words[4],
        block4,
        count_low[4] as u64 + ((count_high[4] as u64) << 32),
        lastblock[4],
        lastnode[4],
    );
    compress(
        &mut state_words[5],
        block5,
        count_low[5] as u64 + ((count_high[5] as u64) << 32),
        lastblock[5],
        lastnode[5],
    );
    compress(
        &mut state_words[6],
        block6,
        count_low[6] as u64 + ((count_high[6] as u64) << 32),
        lastblock[6],
        lastnode[6],
    );
    compress(
        &mut state_words[7],
        block7,
        count_low[7] as u64 + ((count_high[7] as u64) << 32),
        lastblock[7],
        lastnode[7],
    );
}
