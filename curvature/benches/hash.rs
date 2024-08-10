//! Bechmark hash with simd

#![allow(missing_docs)]

use ahash::RandomState;
use criterion::{criterion_group, criterion_main, Criterion};
use rand::Rng;

#[inline(always)]
fn hash_keys<T: std::hash::Hash>(keys: &[T], hashes: &mut [u64]) {
    let random_state = RandomState::new();
    keys.iter().zip(hashes).for_each(|(key, hash)| {
        *hash = random_state.hash_one(key);
    });
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn simd_hash_avx2<T: std::hash::Hash>(keys: &[T], hashes: &mut [u64]) {
    let random_state = RandomState::new();
    keys.iter().zip(hashes).for_each(|(key, hash)| {
        *hash = random_state.hash_one(key);
    });
}

fn hash(c: &mut Criterion) {
    let mut rng = rand::thread_rng();
    let values = (0..1024).map(|_| rng.gen::<u32>()).collect::<Vec<_>>();
    let mut hashes = vec![0; 1024];

    c.bench_function("non-simd-hash", |b| {
        b.iter(|| {
            hash_keys(&values, &mut hashes);
        })
    });

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if std::arch::is_x86_feature_detected!("avx2") {
        c.bench_function("avx2-hash", |b| {
            b.iter(|| unsafe { simd_hash_avx2(&values, &mut hashes) })
        });
    }
}

criterion_group!(benches, hash);
criterion_main!(benches);
