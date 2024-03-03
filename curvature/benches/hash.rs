use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand::{Rng, SeedableRng};

#[derive(Hash)]
struct SerdeKey<T> {
    key: T,
    validity: u64,
}

struct SerdeKeyAndHash<T> {
    key: T,
    hash: u64,
}

fn bench_hash(c: &mut Criterion) {
    let mut rng = rand::rngs::StdRng::seed_from_u64(7);
    let random_state = ahash::RandomState::with_seeds(9, 7, 9, 8);

    // let mut hashes = vec![0; 1024];
    // let vals = (0..1024).map(|_| rng.gen::<u32>()).collect::<Vec<_>>();
    // c.bench_function("hash_u32", |b| {
    //     b.iter(|| {
    //         hashes.iter_mut().zip(&vals).for_each(|(hash, val)| {
    //             *hash = random_state.hash_one(val);
    //         });
    //     })
    // });

    // let vals = (0..1024).map(|_| rng.gen::<u64>()).collect::<Vec<_>>();
    // c.bench_function("hash_u64", |b| {
    //     b.iter(|| {
    //         hashes.iter_mut().zip(&vals).for_each(|(hash, val)| {
    //             *hash = random_state.hash_one(val);
    //         });
    //     })
    // });

    // let vals = (0..1024).map(|_| rng.gen::<[u8; 32]>()).collect::<Vec<_>>();
    // c.bench_function("hash_[u8; 32]", |b| {
    //     b.iter(|| {
    //         hashes.iter_mut().zip(&vals).for_each(|(hash, val)| {
    //             *hash = random_state.hash_one(val);
    //         });
    //     })
    // });

    // let vals = (0..1024)
    //     .map(|_| SerdeKey {
    //         key: rng.gen::<[u8; 32]>(),
    //         validity: rng.gen(),
    //     })
    //     .collect::<Vec<_>>();

    // c.bench_function("hash_serde_key_[u8;32]", |b| {
    //     b.iter(|| {
    //         hashes.iter_mut().zip(&vals).for_each(|(hash, val)| {
    //             *hash = random_state.hash_one(val);
    //         });
    //     })
    // });

    // let mut vals = (0..1024)
    //     .map(|_| SerdeKeyAndHash {
    //         key: rng.gen::<u32>(),
    //         hash: 0,
    //     })
    //     .collect::<Vec<_>>();

    // c.bench_function("hash_u32_and_hash_in_struct", |b| {
    //     b.iter(|| {
    //         vals.iter_mut().for_each(|val| {
    //             val.hash = random_state.hash_one(val.key);
    //         });
    //     })
    // });

    // let mut vals = (0..1024)
    //     .map(|_| SerdeKeyAndHash {
    //         key: rng.gen::<[u8; 32]>(),
    //         hash: 0,
    //     })
    //     .collect::<Vec<_>>();

    // c.bench_function("hash_[u8; 32]_and_hash_in_struct", |b| {
    //     b.iter(|| {
    //         vals.iter_mut().for_each(|val| {
    //             val.hash = random_state.hash_one(val.key);
    //         });
    //     })
    // });

    {
        let iterations = 1000000;
        let mut hashes = vec![0; 1024];
        let vals = (0..1024).map(|_| rng.gen::<[u8; 32]>()).collect::<Vec<_>>();

        let now = std::time::Instant::now();
        (0..iterations).for_each(|_| {
            vals.iter().zip(&mut hashes).for_each(|(val, hash)| {
                *hash = random_state.hash_one(val);
            });
        });
        println!(
            "Hash values into other array takes {} ms",
            now.elapsed().as_millis()
        );

        // Avoid optimization
        black_box(hashes);

        let now = std::time::Instant::now();
        let mut vals = vals
            .into_iter()
            .map(|v| SerdeKeyAndHash { key: v, hash: 0 })
            .collect::<Vec<_>>();

        (0..iterations).for_each(|_| {
            vals.iter_mut()
                .for_each(|val| val.hash = random_state.hash_one(val.key));
        });
        println!(
            "Hash values in same array takes {} ms",
            now.elapsed().as_millis()
        );
    }
}

criterion_group!(benches, bench_hash);
criterion_main!(benches);
