[![Build Status](https://travis-ci.org/oconnor663/blake2s_simd.svg?branch=master)](https://travis-ci.org/oconnor663/blake2s_simd)

EXPERIMENTAL implementation of BLAKE2s and BLAKE2sp. Much of the code is
copy-pasted from
[`blake2b_simd`](https://github.com/oconnor663/blake2b_simd), which is a
real project, but this is currently only for benchmarking, and it's not
published on crates.io.

Currently this contains a portable implementation of BLAKE2s, and an
AVX2 implementation of the `update8` API, similar to
`blake2b_simd::update4`. Notably, there's no single-instance SIMD
implementation of BLAKE2s here yet.
