#![cfg_attr(feature = "avx512", feature(avx512_target_feature))]

use bitvec::slice::BitSlice;
use criterion::{Criterion, criterion_group, criterion_main};
use data_block::bitmap::Bitmap;
use data_block::dynamic_func;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const LEN: usize = 1024;

pub fn selection_benchmark(c: &mut Criterion) {
    let indices = (0..LEN).collect::<Vec<usize>>();
    let mut rng = StdRng::seed_from_u64(100);
    let rhs: i32 = rng.r#gen();
    let lhs: Vec<i32> = (0..LEN).map(|_| rng.r#gen()).collect();
    let mut selected_indices = vec![0; LEN];
    c.bench_function("Selection", |b| {
        b.iter(|| {
            selection(&indices, &lhs, rhs, &mut selected_indices);
        })
    });
    println!("{}", selected_indices.len());

    let indices = Bitmap::from_slice_and_len(&[u64::MAX; 16], LEN);

    c.bench_function("BitmapSelection", |b| {
        b.iter(|| {
            selection_bitmap(&indices, &lhs, rhs, &mut selected_indices);
        })
    });
    println!("{}", selected_indices.len());

    let mut result = vec![false; LEN];

    c.bench_function("FilterDynamic", |b| {
        b.iter(|| gt_dynamic(&lhs, rhs, &mut result))
    });

    c.bench_function("FilterDefault", |b| {
        b.iter(|| gt_default(&lhs, rhs, &mut result))
    });

    println!(
        "True count: {}",
        result.iter().map(|&v| if v { 1 } else { 0 }).sum::<i64>()
    );

    let mut result = bitvec::vec::BitVec::from_iter([false; LEN]);

    c.bench_function("BVFilterDynamic", |b| {
        b.iter(|| gt_bv_dynamic(&lhs, rhs, &mut result))
    });

    c.bench_function("BVFilterDefault", |b| {
        b.iter(|| gt_bv_default(&lhs, rhs, &mut result))
    });
}

/// Selection is pretty expensive(slow). Cache locality? Auto vectorization?
fn selection<T: PartialOrd>(
    indices: &[usize],
    lhs: &[T],
    rhs: T,
    selected_indices: &mut Vec<usize>,
) {
    unsafe {
        let mut current = 0;
        for &indice in indices {
            if lhs.get_unchecked(indice).gt(&rhs) {
                *selected_indices.get_unchecked_mut(current) = indice;
                current += 1;
            }
        }
        selected_indices.set_len(current);
    }
}

/// Selection is pretty expensive(slow). Cache locality? Auto vectorization?
/// Iter ones is slow slow.....
fn selection_bitmap<T: PartialOrd>(
    bitmap: &Bitmap,
    lhs: &[T],
    rhs: T,
    selected_indices: &mut Vec<usize>,
) {
    unsafe {
        let mut current = 0;
        let iter = bitmap.iter_ones();
        iter.for_each(|index| {
            if lhs.get_unchecked(index).gt(&rhs) {
                *selected_indices.get_unchecked_mut(current) = index;
                current += 1;
            }
        });
        selected_indices.set_len(current);
    }
}

macro_rules! gt {
    ($lhs:ident, $rhs:ident, $result:ident) => {
        $lhs.iter()
            .zip($result.iter_mut())
            .for_each(|(lhs, result)| *result = lhs.gt(&$rhs))
    };
}

// Unnecessary computations, but fast !!!!!
dynamic_func!(
    gt,
    <T>,
    (lhs: &[T], rhs: T, result: &mut [bool]),
    where T: PartialOrd
);

macro_rules! gt_bv {
    ($lhs:ident, $rhs:ident, $result:ident) => {
        $lhs.iter().enumerate().for_each(|(i, lhs)| unsafe {
            $result.set_unchecked(i, lhs.gt(&$rhs));
        })
    };
}

dynamic_func!(
    gt_bv,
    <T>,
    (lhs: &[T], rhs: T, result: &mut BitSlice),
    where T: PartialOrd
);

criterion_group!(benches, selection_benchmark);
criterion_main!(benches);
