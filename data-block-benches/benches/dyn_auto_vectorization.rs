#![cfg_attr(feature = "avx512", feature(avx512_target_feature))]

use criterion::{criterion_group, criterion_main, Criterion};
use data_block::aligned_vec::AlignedVec;
use data_block_benches::dynamic_auto_vectorization_func;

const LEN: usize = 1024;

#[allow(clippy::uninit_vec)]
pub fn dynamic_simd_benchmark(c: &mut Criterion) {
    let mut lhs = AlignedVec::<i32>::with_capacity(LEN);
    lhs.clear_and_resize(LEN).iter_mut().for_each(|v| *v = 0);

    let mut rhs = AlignedVec::<i32>::with_capacity(LEN);
    rhs.clear_and_resize(LEN).iter_mut().for_each(|v| *v = 1);

    let mut dest = AlignedVec::<i32>::with_capacity(LEN);
    let _ = dest.clear_and_resize(LEN);

    c.bench_function("AlignedAddQuickly", |b| {
        b.iter(|| {
            add_dynamic(lhs.as_slice(), rhs.as_slice(), dest.as_mut_slice());
        })
    });

    c.bench_function("AlignedAddFallback", |b| {
        b.iter(|| {
            add_default(lhs.as_slice(), rhs.as_slice(), dest.as_mut_slice());
        })
    });

    let mut lhs = Vec::<i32>::with_capacity(LEN);
    unsafe { lhs.set_len(LEN) };
    lhs.iter_mut().for_each(|v| *v = 0);

    let mut rhs = Vec::<i32>::with_capacity(LEN);
    unsafe { rhs.set_len(LEN) };
    rhs.iter_mut().for_each(|v| *v = 1);

    let mut dest = Vec::<i32>::with_capacity(LEN);
    unsafe { dest.set_len(LEN) };

    c.bench_function("AddQuickly", |b| {
        b.iter(|| {
            add_dynamic(lhs.as_slice(), rhs.as_slice(), dest.as_mut_slice());
        })
    });

    c.bench_function("AddFallback", |b| {
        b.iter(|| {
            add_default(lhs.as_slice(), rhs.as_slice(), dest.as_mut_slice());
        })
    });
}

macro_rules! add {
    ($a:ident, $b:ident, $c:ident) => {
        for ((a, b), c) in $a.iter().zip($b).zip($c) {
            *c = *a + *b;
        }
    };
}

dynamic_auto_vectorization_func!(
    add_avx512,
    add_avx2,
    add_neon,
    add_default,
    add_dynamic,
    add,
    ,
    (a: &[i32], b: &[i32], c: &mut [i32]),
);

criterion_group!(benches, dynamic_simd_benchmark);
criterion_main!(benches);
