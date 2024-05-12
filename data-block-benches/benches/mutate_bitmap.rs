use criterion::{black_box, criterion_group, criterion_main, Criterion};
use data_block::bitmap::Bitmap;

const NUM_BITS: usize = 2048;
const SLICE_LNE: usize = NUM_BITS / 64;

pub fn mutate_bitmap_benchmark(c: &mut Criterion) {
    let mut bitmap = Bitmap::from_slice_and_len(&[u64::MAX; SLICE_LNE], NUM_BITS);

    c.bench_function("Reset all valid bitmap", |b| {
        b.iter(|| unsafe {
            let mut guard = bitmap.mutate();
            guard.reset(NUM_BITS, (0..NUM_BITS).map(|_i| black_box(true)));
        })
    });

    c.bench_function("Mutate all valid bitmap", |b| {
        b.iter(|| {
            let mut guard = bitmap.mutate();
            guard.mutate_ones(|_index| black_box(true))
        })
    });
}

criterion_group!(benches, mutate_bitmap_benchmark);
criterion_main!(benches);
