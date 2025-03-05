//! Filter

use criterion::{Criterion, criterion_group, criterion_main};
use data_block::{
    array::{Array, ArrayImpl, UInt64Array},
    bitmap::Bitmap,
};

pub fn filter_benchmark(c: &mut Criterion) {
    let source = UInt64Array::from_values_iter(0..1024);
    let mut array = UInt64Array::default();
    let mut selection = Bitmap::new();
    unsafe {
        selection
            .mutate()
            .reset(1024, (0..1024).map(|index| index % 4 != 0));
    }

    let now = std::time::Instant::now();
    (0..9947264).for_each(|_| unsafe { array.filter(&selection, &source) });
    println!("Elapsed: {:?}", now.elapsed());

    // Selection array is much faster!

    let indexes = (0..1024_usize)
        .filter(|&index| index % 4 != 0)
        .collect::<Vec<_>>();

    let now = std::time::Instant::now();
    (0..9947264).for_each(|_| unsafe {
        array.replace_with_trusted_len_values_iterator(
            768,
            indexes
                .iter()
                .map(|&index| source.get_value_unchecked(index)),
        )
    });
    println!("Elapsed: {:?}", now.elapsed());
}

criterion_group!(benches, filter_benchmark);
criterion_main!(benches);
