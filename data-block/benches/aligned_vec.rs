use std::ops::Add;

use criterion::{criterion_group, criterion_main, Criterion};
use data_block::aligned_vec::AlignedVec;

const LEN: usize = 1024;

#[cfg(target_feature = "avx2")]
const SIMD_INT32_LANES: usize = 8;

#[cfg(target_feature = "avx512f")]
const SIMD_INT32_LANES: usize = 16;

#[cfg(target_feature = "sse2")]
const SIMD_INT32_LANES: usize = 4;

#[cfg(target_feature = "neon")]
const SIMD_INT32_LANES: usize = 4;

#[allow(clippy::uninit_vec)]
pub fn aligned_vec_benchmark(c: &mut Criterion) {
    let mut aligned = AlignedVec::<i32>::with_capacity(LEN);
    aligned
        .clear_and_resize(LEN)
        .iter_mut()
        .for_each(|v| *v = 0);
    let mut rust_vec = Vec::<i32>::with_capacity(1024);
    unsafe { rust_vec.set_len(LEN) };
    rust_vec.iter_mut().for_each(|v| *v = 0);

    init_bench(c, &mut aligned, &mut rust_vec);
    add_bench(c, &mut aligned, &mut rust_vec);
}

fn init_bench(c: &mut Criterion, aligned: &mut AlignedVec<i32>, rust_vec: &mut [i32]) {
    let mut init_group = c.benchmark_group("InitVec");
    init_group.bench_function("AlignedVec", |b| {
        b.iter(|| {
            aligned.as_mut_slice().iter_mut().for_each(|v| *v = -1);
        })
    });

    init_group.bench_function("RustVec", |b| {
        b.iter(|| {
            rust_vec.iter_mut().for_each(|v| *v = -1);
        })
    });
}

#[allow(clippy::uninit_vec)]
fn add_bench(c: &mut Criterion, aligned: &mut AlignedVec<i32>, rust_vec: &mut [i32]) {
    let mut rhs_aligned = AlignedVec::<i32>::with_capacity(LEN);
    rhs_aligned
        .clear_and_resize(LEN)
        .iter_mut()
        .enumerate()
        .for_each(|(i, val)| *val = i as i32);

    let mut dests = AlignedVec::<i32>::with_capacity(LEN);
    let _ = dests.clear_and_resize(LEN);

    let mut add_group = c.benchmark_group("AddVec");
    add_group.bench_function("AlignedVecAdd", |b| {
        b.iter(|| {
            naive_add(
                aligned.as_slice(),
                rhs_aligned.as_slice(),
                dests.as_mut_slice(),
            );
        });
    });

    add_group.bench_function("AlignedVecAddChunk", |b| {
        b.iter(|| {
            chunk_add(
                aligned.as_slice(),
                rhs_aligned.as_slice(),
                dests.as_mut_slice(),
            );
        });
    });

    let mut rhs_rust_vec = Vec::<i32>::with_capacity(1024);
    unsafe { rhs_rust_vec.set_len(LEN) };
    rust_vec.iter_mut().for_each(|v| *v = 0);
    rhs_rust_vec
        .iter_mut()
        .enumerate()
        .for_each(|(i, val)| *val = i as i32);

    let mut dests_rust_vec = Vec::<i32>::with_capacity(LEN);
    unsafe { dests_rust_vec.set_len(LEN) };
    add_group.bench_function("RustVecAdd", |b| {
        b.iter(|| naive_add(rust_vec, &rhs_rust_vec, &mut dests_rust_vec));
    });

    add_group.bench_function("RustVecAddChunk", |b| {
        b.iter(|| chunk_add(rust_vec, &rhs_rust_vec, &mut dests_rust_vec));
    });
}

#[inline]
fn naive_add<T: Add<Output = T> + Copy>(lhs: &[T], rhs: &[T], dest: &mut [T]) {
    lhs.iter()
        .zip(rhs.iter())
        .zip(dest.iter_mut())
        .for_each(|((lhs, rhs), dest)| {
            *dest = *lhs + *rhs;
        });
}

fn chunk_add(lhs: &[i32], rhs: &[i32], dest: &mut [i32]) {
    lhs.chunks_exact(SIMD_INT32_LANES)
        .zip(rhs.chunks_exact(SIMD_INT32_LANES))
        .zip(dest.chunks_exact_mut(SIMD_INT32_LANES))
        .for_each(|((lhs, rhs), dest)| {
            let lhs: [i32; SIMD_INT32_LANES] = lhs.try_into().unwrap();
            let rhs: [i32; SIMD_INT32_LANES] = rhs.try_into().unwrap();
            lhs.iter()
                .zip(rhs.iter())
                .zip(dest)
                .for_each(|((lhs, rhs), dest)| {
                    *dest = *lhs + *rhs;
                })
        });
}

criterion_group!(benches, aligned_vec_benchmark);
criterion_main!(benches);
