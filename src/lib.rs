//! EXPERIMENTAL implementation of BLAKE2s

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(feature = "std")]
extern crate core;

#[macro_use]
extern crate arrayref;
extern crate arrayvec;
extern crate byteorder;
extern crate constant_time_eq;

use byteorder::{ByteOrder, LittleEndian};
use core::cmp;
use core::fmt;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod avx2;
mod portable;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod sse41;

pub mod blake2sp;

#[cfg(test)]
mod test;

/// The max hash length.
pub const OUTBYTES: usize = 32;
/// The max key length.
pub const KEYBYTES: usize = 32;
/// The max salt length.
pub const SALTBYTES: usize = 8;
/// The max personalization length.
pub const PERSONALBYTES: usize = 8;
/// The number input bytes passed to each call to the compression function. Small benchmarks need
/// to use an even multiple of `BLOCKBYTES`, or else their apparent throughput will be low.
pub const BLOCKBYTES: usize = 64;

const IV: [u32; 8] = [
    0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19,
];

const SIGMA: [[u8; 16]; 12] = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    [14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3],
    [11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4],
    [7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8],
    [9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13],
    [2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9],
    [12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11],
    [13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10],
    [6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5],
    [10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0],
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    [14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3],
];

// Safety note: The compression interface is unsafe in general, even though the portable
// implementation is safe, because calling the AVX2 implementation on a platform that doesn't
// support AVX2 is undefined behavior.
type CompressFn = unsafe fn(&mut StateWords, &Block, count: u64, lastblock: u32, lastnode: u32);
type Compress8Fn = unsafe fn(
    state0: &mut StateWords,
    state1: &mut StateWords,
    state2: &mut StateWords,
    state3: &mut StateWords,
    state4: &mut StateWords,
    state5: &mut StateWords,
    state6: &mut StateWords,
    state7: &mut StateWords,
    block0: &Block,
    block1: &Block,
    block2: &Block,
    block3: &Block,
    block4: &Block,
    block5: &Block,
    block6: &Block,
    block7: &Block,
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
);
type Hash4ExactFn = unsafe fn(
    params: &Params,
    input0: &[u8],
    input1: &[u8],
    input2: &[u8],
    input3: &[u8],
) -> [Hash; 4];
type Hash8ExactFn = unsafe fn(
    params: &Params,
    input0: &[u8],
    input1: &[u8],
    input2: &[u8],
    input3: &[u8],
    input4: &[u8],
    input5: &[u8],
    input6: &[u8],
    input7: &[u8],
) -> [Hash; 8];
type Vectorize8Fn = unsafe fn(words: &mut [AlignedWords8; 8]);
type Compress8VectorizedFn = unsafe fn(
    vectorized_words: &mut [AlignedWords8; 8],
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
);

type StateWords = [u32; 8];
type Block = [u8; BLOCKBYTES];
type HexString = arrayvec::ArrayString<[u8; 2 * OUTBYTES]>;

#[derive(Copy, Clone)]
#[repr(C, align(32))]
pub struct AlignedWords8(pub [u32; 8]);

impl core::ops::Deref for AlignedWords8 {
    type Target = [u32; 8];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl core::ops::DerefMut for AlignedWords8 {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

pub fn blake2s(input: &[u8]) -> Hash {
    State::new().update(input).finalize()
}

#[derive(Clone)]
pub struct Params {
    hash_length: u8,
    key_length: u8,
    key: [u8; KEYBYTES],
    salt: [u8; SALTBYTES],
    personal: [u8; PERSONALBYTES],
    fanout: u8,
    max_depth: u8,
    max_leaf_length: u32,
    node_offset: u64,
    node_depth: u8,
    inner_hash_length: u8,
    last_node: bool,
}

impl Params {
    pub fn new() -> Self {
        Self::default()
    }

    fn make_words(&self) -> StateWords {
        let (salt_left, salt_right) = array_refs!(&self.salt, 4, 4);
        let (personal_left, personal_right) = array_refs!(&self.personal, 4, 4);
        [
            IV[0]
                ^ self.hash_length as u32
                ^ (self.key_length as u32) << 8
                ^ (self.fanout as u32) << 16
                ^ (self.max_depth as u32) << 24,
            IV[1] ^ (self.max_leaf_length as u32),
            IV[2] ^ (self.node_offset as u32),
            IV[3]
                ^ (self.node_offset >> 32) as u32
                ^ (self.node_depth as u32) << 16
                ^ (self.inner_hash_length as u32) << 24,
            IV[4] ^ LittleEndian::read_u32(salt_left),
            IV[5] ^ LittleEndian::read_u32(salt_right),
            IV[6] ^ LittleEndian::read_u32(personal_left),
            IV[7] ^ LittleEndian::read_u32(personal_right),
        ]
    }

    pub fn to_state(&self) -> State {
        State::with_params(self)
    }

    pub fn hash_length(&mut self, length: usize) -> &mut Self {
        assert!(
            1 <= length && length <= OUTBYTES,
            "Bad hash length: {}",
            length
        );
        self.hash_length = length as u8;
        self
    }

    pub fn key(&mut self, key: &[u8]) -> &mut Self {
        assert!(key.len() <= KEYBYTES, "Bad key length: {}", key.len());
        self.key_length = key.len() as u8;
        self.key = [0; KEYBYTES];
        self.key[..key.len()].copy_from_slice(key);
        self
    }

    pub fn salt(&mut self, salt: &[u8]) -> &mut Self {
        assert!(salt.len() <= SALTBYTES, "Bad salt length: {}", salt.len());
        self.salt = [0; SALTBYTES];
        self.salt[..salt.len()].copy_from_slice(salt);
        self
    }

    pub fn personal(&mut self, personalization: &[u8]) -> &mut Self {
        assert!(
            personalization.len() <= PERSONALBYTES,
            "Bad personalization length: {}",
            personalization.len()
        );
        self.personal = [0; PERSONALBYTES];
        self.personal[..personalization.len()].copy_from_slice(personalization);
        self
    }

    pub fn fanout(&mut self, fanout: u8) -> &mut Self {
        self.fanout = fanout;
        self
    }

    pub fn max_depth(&mut self, depth: u8) -> &mut Self {
        assert!(depth != 0, "Bad max depth: {}", depth);
        self.max_depth = depth;
        self
    }

    pub fn max_leaf_length(&mut self, length: u32) -> &mut Self {
        self.max_leaf_length = length;
        self
    }

    pub fn node_offset(&mut self, offset: u64) -> &mut Self {
        assert!(offset < (1 << 48), "Bad node offset: {}", offset);
        self.node_offset = offset;
        self
    }

    pub fn node_depth(&mut self, depth: u8) -> &mut Self {
        self.node_depth = depth;
        self
    }

    pub fn inner_hash_length(&mut self, length: usize) -> &mut Self {
        assert!(length <= OUTBYTES, "Bad inner hash length: {}", length);
        self.inner_hash_length = length as u8;
        self
    }

    pub fn last_node(&mut self, last_node: bool) -> &mut Self {
        self.last_node = last_node;
        self
    }
}

impl Default for Params {
    fn default() -> Self {
        Self {
            hash_length: OUTBYTES as u8,
            key_length: 0,
            key: [0; KEYBYTES],
            salt: [0; SALTBYTES],
            personal: [0; PERSONALBYTES],
            // NOTE: fanout and max_depth don't default to zero!
            fanout: 1,
            max_depth: 1,
            max_leaf_length: 0,
            node_offset: 0,
            node_depth: 0,
            inner_hash_length: 0,
            last_node: false,
        }
    }
}

#[derive(Clone)]
pub struct State {
    h: StateWords,
    buf: Block,
    buflen: u8,
    count: u64,
    compress_fn: CompressFn,
    last_node: bool,
    hash_length: u8,
}

impl State {
    pub fn new() -> Self {
        Self::with_params(&Params::default())
    }

    fn with_params(params: &Params) -> Self {
        let mut state = Self {
            h: params.make_words(),
            compress_fn: default_compress_impl().0,
            buf: [0; BLOCKBYTES],
            buflen: 0,
            count: 0,
            last_node: params.last_node,
            hash_length: params.hash_length,
        };
        if params.key_length > 0 {
            let mut key_block = [0; BLOCKBYTES];
            key_block[..KEYBYTES].copy_from_slice(&params.key);
            state.update(&key_block);
        }
        state
    }

    fn fill_buf(&mut self, input: &mut &[u8]) {
        let take = cmp::min(BLOCKBYTES - self.buflen as usize, input.len());
        self.buf[self.buflen as usize..self.buflen as usize + take].copy_from_slice(&input[..take]);
        self.buflen += take as u8;
        self.count += take as u64;
        *input = &input[take..];
    }

    // If the state already has some input in its buffer, try to fill the buffer and perform a
    // compression. However, only do the compression if there's more input coming, otherwise it
    // will give the wrong hash it the caller finalizes immediately after.
    fn compress_buffer_if_possible(&mut self, input: &mut &[u8]) {
        if self.buflen > 0 {
            self.fill_buf(input);
            if !input.is_empty() {
                unsafe {
                    (self.compress_fn)(&mut self.h, &self.buf, self.count, 0, 0);
                }
                self.buflen = 0;
            }
        }
    }

    /// Add input to the hash. You can call `update` any number of times.
    pub fn update(&mut self, mut input: &[u8]) -> &mut Self {
        // If we have a partial buffer, try to complete it.
        self.compress_buffer_if_possible(&mut input);
        // While there's more than a block of input left (which also means we cleared the buffer
        // above), compress blocks directly without copying.
        while input.len() > BLOCKBYTES {
            self.count += BLOCKBYTES as u64;
            let block = array_ref!(input, 0, BLOCKBYTES);
            unsafe {
                (self.compress_fn)(&mut self.h, block, self.count, 0, 0);
            }
            input = &input[BLOCKBYTES..];
        }
        // Buffer any remaining input, to be either compressed or finalized in a subsequent call.
        // Note that this represents some copying overhead, which in theory we could avoid in
        // all-at-once setting. A function hardcoded for exactly BLOCKSIZE input bytes is about 10%
        // faster than using this implementation for the same input. But non-multiple sizes still
        // require copying, and the savings disappear into the noise for any larger multiple. Any
        // caller so concerned with performance that they're shaping their hash inputs down to the
        // single byte, should just call the compression function directly.
        self.fill_buf(&mut input);
        self
    }

    /// Finalize the state and return a `Hash`. This method is idempotent, and calling it multiple
    /// times will give the same result. It's also possible to `update` with more input in between.
    pub fn finalize(&mut self) -> Hash {
        for i in self.buflen as usize..BLOCKBYTES {
            self.buf[i] = 0;
        }
        let last_node = if self.last_node { !0 } else { 0 };
        let mut h_copy = self.h;
        unsafe {
            (self.compress_fn)(&mut h_copy, &self.buf, self.count, !0, last_node);
        }
        Hash {
            bytes: state_words_to_bytes(&h_copy),
            len: self.hash_length,
        }
    }

    /// Set a flag indicating that this is the last node of its level in a tree hash. This is
    /// equivalent to [`Params::last_node`], except that it can be set at any time before calling
    /// `finalize`. That allows callers to begin hashing a node without knowing ahead of time
    /// whether it's the last in its level. For more details about the intended use of this flag
    /// [the BLAKE2 spec].
    ///
    /// [`Params::last_node`]: struct.Params.html#method.last_node
    /// [the BLAKE2 spec]: https://blake2.net/blake2.pdf
    pub fn set_last_node(&mut self, last_node: bool) -> &mut Self {
        self.last_node = last_node;
        self
    }

    /// Return the total number of bytes input so far.
    pub fn count(&self) -> u64 {
        self.count
    }
}

fn state_words_to_bytes(state_words: &StateWords) -> [u8; OUTBYTES] {
    let mut bytes = [0; OUTBYTES];
    {
        let refs = mut_array_refs!(&mut bytes, 4, 4, 4, 4, 4, 4, 4, 4);
        LittleEndian::write_u32(refs.0, state_words[0]);
        LittleEndian::write_u32(refs.1, state_words[1]);
        LittleEndian::write_u32(refs.2, state_words[2]);
        LittleEndian::write_u32(refs.3, state_words[3]);
        LittleEndian::write_u32(refs.4, state_words[4]);
        LittleEndian::write_u32(refs.5, state_words[5]);
        LittleEndian::write_u32(refs.6, state_words[6]);
        LittleEndian::write_u32(refs.7, state_words[7]);
    }
    bytes
}

#[cfg(feature = "std")]
impl std::io::Write for State {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.update(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl fmt::Debug for State {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // NB: Don't print the words. Leaking them would allow length extension.
        write!(
            f,
            "State {{ count: {}, hash_length: {}, last_node: {} }}",
            self.count, self.hash_length, self.last_node,
        )
    }
}

impl Default for State {
    fn default() -> Self {
        Self::with_params(&Params::default())
    }
}

/// A finalized BLAKE2 hash, with constant-time equality.
#[derive(Clone, Copy)]
pub struct Hash {
    bytes: [u8; OUTBYTES],
    len: u8,
}

impl Hash {
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes[..self.len as usize]
    }

    pub fn to_hex(&self) -> HexString {
        bytes_to_hex(self.as_bytes())
    }
}

fn bytes_to_hex(bytes: &[u8]) -> HexString {
    let mut s = arrayvec::ArrayString::new();
    let table = b"0123456789abcdef";
    for &b in bytes {
        s.push(table[(b >> 4) as usize] as char);
        s.push(table[(b & 0xf) as usize] as char);
    }
    s
}

/// This implementation is constant time, if the two hashes are the same length.
impl PartialEq for Hash {
    fn eq(&self, other: &Hash) -> bool {
        constant_time_eq::constant_time_eq(&self.as_bytes(), &other.as_bytes())
    }
}

/// This implementation is constant time, if the slice is the same length as the hash.
impl PartialEq<[u8]> for Hash {
    fn eq(&self, other: &[u8]) -> bool {
        constant_time_eq::constant_time_eq(&self.as_bytes(), other)
    }
}

impl Eq for Hash {}

impl AsRef<[u8]> for Hash {
    fn as_ref(&self) -> &[u8] {
        self.as_bytes()
    }
}

impl fmt::Debug for Hash {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Hash(0x{})", self.to_hex())
    }
}

pub fn update8(
    state0: &mut State,
    state1: &mut State,
    state2: &mut State,
    state3: &mut State,
    state4: &mut State,
    state5: &mut State,
    state6: &mut State,
    state7: &mut State,
    mut input0: &[u8],
    mut input1: &[u8],
    mut input2: &[u8],
    mut input3: &[u8],
    mut input4: &[u8],
    mut input5: &[u8],
    mut input6: &[u8],
    mut input7: &[u8],
) {
    // First we need to make sure all the buffers are clear.
    state0.compress_buffer_if_possible(&mut input0);
    state1.compress_buffer_if_possible(&mut input1);
    state2.compress_buffer_if_possible(&mut input2);
    state3.compress_buffer_if_possible(&mut input3);
    state4.compress_buffer_if_possible(&mut input4);
    state5.compress_buffer_if_possible(&mut input5);
    state6.compress_buffer_if_possible(&mut input6);
    state7.compress_buffer_if_possible(&mut input7);
    // Now, as long as all of the states have more than a block of input coming (so that we know we
    // don't need to finalize any of them), compress in parallel directly into their state words.
    let compress8_fn = default_compress_impl().1;
    while input0.len() > BLOCKBYTES
        && input1.len() > BLOCKBYTES
        && input2.len() > BLOCKBYTES
        && input3.len() > BLOCKBYTES
        && input4.len() > BLOCKBYTES
        && input5.len() > BLOCKBYTES
        && input6.len() > BLOCKBYTES
        && input7.len() > BLOCKBYTES
    {
        state0.count += BLOCKBYTES as u64;
        state1.count += BLOCKBYTES as u64;
        state2.count += BLOCKBYTES as u64;
        state3.count += BLOCKBYTES as u64;
        state4.count += BLOCKBYTES as u64;
        state5.count += BLOCKBYTES as u64;
        state6.count += BLOCKBYTES as u64;
        state7.count += BLOCKBYTES as u64;
        unsafe {
            compress8_fn(
                &mut state0.h,
                &mut state1.h,
                &mut state2.h,
                &mut state3.h,
                &mut state4.h,
                &mut state5.h,
                &mut state6.h,
                &mut state7.h,
                array_ref!(input0, 0, BLOCKBYTES),
                array_ref!(input1, 0, BLOCKBYTES),
                array_ref!(input2, 0, BLOCKBYTES),
                array_ref!(input3, 0, BLOCKBYTES),
                array_ref!(input4, 0, BLOCKBYTES),
                array_ref!(input5, 0, BLOCKBYTES),
                array_ref!(input6, 0, BLOCKBYTES),
                array_ref!(input7, 0, BLOCKBYTES),
                state0.count,
                state1.count,
                state2.count,
                state3.count,
                state4.count,
                state5.count,
                state6.count,
                state7.count,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
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
        input0 = &input0[BLOCKBYTES..];
        input1 = &input1[BLOCKBYTES..];
        input2 = &input2[BLOCKBYTES..];
        input3 = &input3[BLOCKBYTES..];
        input4 = &input4[BLOCKBYTES..];
        input5 = &input5[BLOCKBYTES..];
        input6 = &input6[BLOCKBYTES..];
        input7 = &input7[BLOCKBYTES..];
    }
    // Finally, if there's any remaining input, add it into the state the usual way. Note that if
    // one of the inputs is short, this could actually be more work than the loop above. The caller
    // should hopefully arrange for that not to happen.
    state0.update(input0);
    state1.update(input1);
    state2.update(input2);
    state3.update(input3);
    state4.update(input4);
    state5.update(input5);
    state6.update(input6);
    state7.update(input7);
}

pub fn finalize8(
    state0: &mut State,
    state1: &mut State,
    state2: &mut State,
    state3: &mut State,
    state4: &mut State,
    state5: &mut State,
    state6: &mut State,
    state7: &mut State,
) -> [Hash; 8] {
    // Zero out the buffer tails, which might contain bytes from previous blocks.
    for i in state0.buflen as usize..BLOCKBYTES {
        state0.buf[i] = 0;
    }
    for i in state1.buflen as usize..BLOCKBYTES {
        state1.buf[i] = 0;
    }
    for i in state2.buflen as usize..BLOCKBYTES {
        state2.buf[i] = 0;
    }
    for i in state3.buflen as usize..BLOCKBYTES {
        state3.buf[i] = 0;
    }
    for i in state4.buflen as usize..BLOCKBYTES {
        state4.buf[i] = 0;
    }
    for i in state5.buflen as usize..BLOCKBYTES {
        state5.buf[i] = 0;
    }
    for i in state6.buflen as usize..BLOCKBYTES {
        state6.buf[i] = 0;
    }
    for i in state7.buflen as usize..BLOCKBYTES {
        state7.buf[i] = 0;
    }
    // Translate the last node flag of each state into the u64 that BLAKE2 uses.
    let last_node0: u32 = if state0.last_node { !0 } else { 0 };
    let last_node1: u32 = if state1.last_node { !0 } else { 0 };
    let last_node2: u32 = if state2.last_node { !0 } else { 0 };
    let last_node3: u32 = if state3.last_node { !0 } else { 0 };
    let last_node4: u32 = if state4.last_node { !0 } else { 0 };
    let last_node5: u32 = if state5.last_node { !0 } else { 0 };
    let last_node6: u32 = if state6.last_node { !0 } else { 0 };
    let last_node7: u32 = if state7.last_node { !0 } else { 0 };
    // Make copies of all the state words. This step is what makes finalize idempotent.
    let mut h_copy0 = state0.h;
    let mut h_copy1 = state1.h;
    let mut h_copy2 = state2.h;
    let mut h_copy3 = state3.h;
    let mut h_copy4 = state4.h;
    let mut h_copy5 = state5.h;
    let mut h_copy6 = state6.h;
    let mut h_copy7 = state7.h;
    // Do the final parallel compression step.
    let compress8_fn = default_compress_impl().1;
    unsafe {
        compress8_fn(
            &mut h_copy0,
            &mut h_copy1,
            &mut h_copy2,
            &mut h_copy3,
            &mut h_copy4,
            &mut h_copy5,
            &mut h_copy6,
            &mut h_copy7,
            &state0.buf,
            &state1.buf,
            &state2.buf,
            &state3.buf,
            &state4.buf,
            &state5.buf,
            &state6.buf,
            &state7.buf,
            state0.count,
            state1.count,
            state2.count,
            state3.count,
            state4.count,
            state5.count,
            state6.count,
            state7.count,
            !0,
            !0,
            !0,
            !0,
            !0,
            !0,
            !0,
            !0,
            last_node0,
            last_node1,
            last_node2,
            last_node3,
            last_node4,
            last_node5,
            last_node6,
            last_node7,
        );
    }
    // Extract the resulting hashes.
    [
        Hash {
            bytes: state_words_to_bytes(&h_copy0),
            len: state0.hash_length,
        },
        Hash {
            bytes: state_words_to_bytes(&h_copy1),
            len: state1.hash_length,
        },
        Hash {
            bytes: state_words_to_bytes(&h_copy2),
            len: state2.hash_length,
        },
        Hash {
            bytes: state_words_to_bytes(&h_copy3),
            len: state3.hash_length,
        },
        Hash {
            bytes: state_words_to_bytes(&h_copy4),
            len: state4.hash_length,
        },
        Hash {
            bytes: state_words_to_bytes(&h_copy5),
            len: state5.hash_length,
        },
        Hash {
            bytes: state_words_to_bytes(&h_copy6),
            len: state6.hash_length,
        },
        Hash {
            bytes: state_words_to_bytes(&h_copy7),
            len: state7.hash_length,
        },
    ]
}

pub fn hash4_exact(
    // TODO: Separate params for each input.
    params: &Params,
    input0: &[u8],
    input1: &[u8],
    input2: &[u8],
    input3: &[u8],
) -> [Hash; 4] {
    // These asserts are safety invariants for the AVX2 implementation.
    let len = input0.len();
    let same_length = (input1.len() == len) && (input2.len() == len) && (input3.len() == len);
    let even_length = len % BLOCKBYTES == 0;
    let nonempty = len != 0;
    assert!(
        same_length && even_length && nonempty,
        "invalid hash8_exact inputs"
    );

    let hash4_exact_fn = default_compress_impl().3;
    unsafe { hash4_exact_fn(params, input0, input1, input2, input3) }
}

pub fn hash8_exact(
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
    // These asserts are safety invariants for the AVX2 implementation.
    let len = input0.len();
    let same_length = (input1.len() == len)
        && (input2.len() == len)
        && (input3.len() == len)
        && (input4.len() == len)
        && (input5.len() == len)
        && (input6.len() == len)
        && (input7.len() == len);
    let even_length = len % BLOCKBYTES == 0;
    let nonempty = len != 0;
    assert!(
        same_length && even_length && nonempty,
        "invalid hash8_exact inputs"
    );

    let hash8_exact_fn = default_compress_impl().2;
    unsafe {
        hash8_exact_fn(
            params, input0, input1, input2, input3, input4, input5, input6, input7,
        )
    }
}

pub fn hash8(
    params0: &Params,
    params1: &Params,
    params2: &Params,
    params3: &Params,
    params4: &Params,
    params5: &Params,
    params6: &Params,
    params7: &Params,
    input0: &[u8],
    input1: &[u8],
    input2: &[u8],
    input3: &[u8],
    input4: &[u8],
    input5: &[u8],
    input6: &[u8],
    input7: &[u8],
) -> [Hash; 8] {
    // Keying isn't supported yet.
    assert_eq!(0, params0.key_length);
    assert_eq!(0, params1.key_length);
    assert_eq!(0, params2.key_length);
    assert_eq!(0, params3.key_length);
    assert_eq!(0, params4.key_length);
    assert_eq!(0, params5.key_length);
    assert_eq!(0, params6.key_length);
    assert_eq!(0, params7.key_length);

    let fns = default_compress_impl();
    let compress_fn = fns.0;
    let vectorize8_fn = fns.4;
    let compress8_vectorized_fn = fns.5;

    let mut state_words8 = [
        AlignedWords8(params0.make_words()),
        AlignedWords8(params1.make_words()),
        AlignedWords8(params2.make_words()),
        AlignedWords8(params3.make_words()),
        AlignedWords8(params4.make_words()),
        AlignedWords8(params5.make_words()),
        AlignedWords8(params6.make_words()),
        AlignedWords8(params7.make_words()),
    ];
    unsafe {
        vectorize8_fn(&mut state_words8);
    }

    let mut count = 0;
    loop {
        let update_all = input0.len() - count > BLOCKBYTES
            && input1.len() - count > BLOCKBYTES
            && input2.len() - count > BLOCKBYTES
            && input3.len() - count > BLOCKBYTES
            && input4.len() - count > BLOCKBYTES
            && input5.len() - count > BLOCKBYTES
            && input6.len() - count > BLOCKBYTES
            && input7.len() - count > BLOCKBYTES;
        let finalize_all = input0.len() - count == BLOCKBYTES
            && input1.len() - count == BLOCKBYTES
            && input2.len() - count == BLOCKBYTES
            && input3.len() - count == BLOCKBYTES
            && input4.len() - count == BLOCKBYTES
            && input5.len() - count == BLOCKBYTES
            && input6.len() - count == BLOCKBYTES
            && input7.len() - count == BLOCKBYTES;
        if !update_all && !finalize_all {
            // If all the inputs can't be compressed in tandem, break out of
            // the efficient loop and split the states apart.
            break;
        }

        let block0 = array_ref!(input0, count, BLOCKBYTES);
        let block1 = array_ref!(input1, count, BLOCKBYTES);
        let block2 = array_ref!(input2, count, BLOCKBYTES);
        let block3 = array_ref!(input3, count, BLOCKBYTES);
        let block4 = array_ref!(input4, count, BLOCKBYTES);
        let block5 = array_ref!(input5, count, BLOCKBYTES);
        let block6 = array_ref!(input6, count, BLOCKBYTES);
        let block7 = array_ref!(input7, count, BLOCKBYTES);
        count += BLOCKBYTES;

        let count_low = count as u32;
        let count_high = ((count as u64) >> 32) as u32;
        let lastblock;
        let lastnode;
        if finalize_all {
            lastblock = AlignedWords8([!0; 8]);
            lastnode = AlignedWords8([
                if params0.last_node { !0 } else { 0 },
                if params1.last_node { !0 } else { 0 },
                if params2.last_node { !0 } else { 0 },
                if params3.last_node { !0 } else { 0 },
                if params4.last_node { !0 } else { 0 },
                if params5.last_node { !0 } else { 0 },
                if params6.last_node { !0 } else { 0 },
                if params7.last_node { !0 } else { 0 },
            ]);
        } else {
            lastblock = AlignedWords8([0; 8]);
            lastnode = AlignedWords8([0; 8]);
        }
        unsafe {
            compress8_vectorized_fn(
                &mut state_words8,
                block0,
                block1,
                block2,
                block3,
                block4,
                block5,
                block6,
                block7,
                &AlignedWords8([count_low; 8]),
                &AlignedWords8([count_high; 8]),
                &lastblock,
                &lastnode,
            );
        }
        if finalize_all {
            unsafe {
                vectorize8_fn(&mut state_words8);
            }
            return [
                Hash {
                    bytes: state_words_to_bytes(&state_words8[0]),
                    len: params0.hash_length,
                },
                Hash {
                    bytes: state_words_to_bytes(&state_words8[1]),
                    len: params1.hash_length,
                },
                Hash {
                    bytes: state_words_to_bytes(&state_words8[2]),
                    len: params2.hash_length,
                },
                Hash {
                    bytes: state_words_to_bytes(&state_words8[3]),
                    len: params3.hash_length,
                },
                Hash {
                    bytes: state_words_to_bytes(&state_words8[4]),
                    len: params4.hash_length,
                },
                Hash {
                    bytes: state_words_to_bytes(&state_words8[5]),
                    len: params5.hash_length,
                },
                Hash {
                    bytes: state_words_to_bytes(&state_words8[6]),
                    len: params6.hash_length,
                },
                Hash {
                    bytes: state_words_to_bytes(&state_words8[7]),
                    len: params7.hash_length,
                },
            ];
        }
    }

    // Unvectorize the state. Note that vectorize is its own inverse. In the
    // SIMD implementation it's a matrix transposition, and in the portable
    // implementation it's actually a no-op.
    unsafe {
        vectorize8_fn(&mut state_words8);
    }
    // Individually finish each input.
    let finish = |params: &Params, words: &StateWords, input: &[u8]| -> Hash {
        let mut state = State {
            h: *words,
            buf: [0; BLOCKBYTES],
            buflen: 0,
            count: count as u64,
            compress_fn: compress_fn,
            last_node: params.last_node,
            hash_length: params.hash_length,
        };
        state.update(&input[count..]);
        state.finalize()
    };
    [
        finish(params0, &state_words8[0], input0),
        finish(params1, &state_words8[1], input1),
        finish(params2, &state_words8[2], input2),
        finish(params3, &state_words8[3], input3),
        finish(params4, &state_words8[4], input4),
        finish(params5, &state_words8[5], input5),
        finish(params6, &state_words8[6], input6),
        finish(params7, &state_words8[7], input7),
    ]
}

// Safety: The unsafe blocks above rely on this function to never return avx2::compress except on
// platforms where it's safe to call.
#[allow(unreachable_code)]
fn default_compress_impl() -> (
    CompressFn,
    Compress8Fn,
    Hash8ExactFn,
    Hash4ExactFn,
    Vectorize8Fn,
    Compress8VectorizedFn,
) {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        #[cfg(feature = "std")]
        {
            if is_x86_feature_detected!("avx2") {
                return (
                    sse41::compress,
                    avx2::compress8,
                    avx2::hash8_exact,
                    sse41::hash4_exact,
                    avx2::vectorize_words8,
                    avx2::compress8_vectorized,
                );
            }
        }
    }
    // On other platforms (non-x86 or pre-AVX2) use the portable implementation.
    (
        portable::compress,
        portable::compress8,
        portable::hash8_exact,
        portable::hash4_exact,
        portable::vectorize_words8,
        portable::compress8_vectorized,
    )
}

// This module is pub for internal benchmarks only. Please don't use it.
#[doc(hidden)]
pub mod benchmarks {
    pub use crate::portable::compress as compress_portable;
    pub use crate::portable::compress8 as compress8_portable;

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    pub use crate::sse41::compress as compress_sse41;
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    pub use crate::sse41::compress4_transposed as compress4_transposed_sse41;

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    pub use crate::avx2::compress8 as compress8_avx2;
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    pub use crate::avx2::compress8_transposed_all as compress8_transposed_all_avx2;
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    pub use crate::avx2::compress8_vectorized as compress8_vectorized_avx2;
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    pub use crate::avx2::compress8_vectorized_to_bytes as compress8_vectorized_to_bytes_avx2;
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    pub use crate::avx2::export_bytes as export_bytes_avx2;
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    pub use crate::avx2::transpose_msg_vecs as transpose_msg_vecs_avx2;

    // Safety: The portable implementation should be safe to call on any platform.
    pub fn force_portable(state: &mut crate::State) {
        state.compress_fn = compress_portable;
    }
    pub fn force_portable_blake2sp(state: &mut crate::blake2sp::State) {
        crate::blake2sp::force_portable(state);
    }
}
