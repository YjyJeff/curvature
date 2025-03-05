use std::ops::{Add, Div};

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use data_block::array::{Array, PrimitiveArray};
use data_block::bitmap::Bitmap;
use data_block::compute::IntrinsicType;
use data_block::compute::arith::intrinsic::{
    AddExt, DivExt, MulExt, RemExt, SubExt, add_scalar, div_scalar, mul_scalar, sub_scalar,
};
use data_block::element::Element;
use data_block_benches::create_primitive_array_with_seed;
use rand::distributions::{Distribution, Standard};

fn bench_arith_inner<T>(c: &mut Criterion, rhs: T, null_density: f32, seed: u64)
where
    T: IntrinsicType
        + Add<Output = T>
        + Div<Output = T>
        + PartialEq
        + Default
        + DivExt
        + AddExt
        + SubExt
        + MulExt
        + RemExt
        + for<'a> Element<ElementRef<'a> = T>
        + arrow2::compute::arithmetics::basic::NativeArithmetics
        + arrow2::types::NativeType
        + num_traits::NumCast,
    Standard: Distribution<T>,
    PrimitiveArray<T>: Array<Element = T>,
{
    let mut group = c.benchmark_group(format!("Compare{}", std::any::type_name::<T>()));
    (10..=12).step_by(2).for_each(|log2_size| {
        let size = black_box(2usize.pow(log2_size));
        let lhs = create_primitive_array_with_seed::<T>(size, null_density, seed);
        let mut dst = PrimitiveArray::<T>::default();
        let selection = Bitmap::from_slice_and_len(&vec![u64::MAX; size / 64], size);

        unsafe { div_scalar(&selection, &lhs, rhs, &mut dst) };

        let arr_a = arrow2::util::bench_util::create_primitive_array_with_seed::<T>(
            size,
            null_density,
            seed,
        );

        let arrow2_result: Vec<T> = arrow2::compute::arithmetics::basic::div_scalar(&arr_a, &rhs)
            .values_iter()
            .copied()
            .collect::<Vec<_>>();
        assert_eq!(dst.values_iter().collect::<Vec<_>>(), arrow2_result);

        // group.bench_function(BenchmarkId::new("Arrow2: arith scalar", size), |b| {
        //     b.iter(|| arrow2::compute::arithmetics::basic::div_scalar(&arr_a, &rhs));
        // });

        group.bench_function(BenchmarkId::new("DataBlock: add scalar", size), |b| {
            b.iter(|| unsafe {
                add_scalar(&selection, &lhs, rhs, &mut dst);
            });
        });

        group.bench_function(BenchmarkId::new("DataBlock: sub scalar", size), |b| {
            b.iter(|| unsafe {
                sub_scalar(&selection, &lhs, rhs, &mut dst);
            });
        });

        group.bench_function(BenchmarkId::new("DataBlock: mul scalar", size), |b| {
            b.iter(|| unsafe {
                mul_scalar(&selection, &lhs, rhs, &mut dst);
            });
        });

        group.bench_function(BenchmarkId::new("DataBlock: div scalar", size), |b| {
            b.iter(|| unsafe {
                div_scalar(&selection, &lhs, rhs, &mut dst);
            });
        });
    });
}

fn bench_arith(c: &mut Criterion) {
    bench_arith_inner(c, -10_i8, 0.0, 42);
    // bench_arith_inner(c, -10_i16, 0.0, 42);
    // bench_arith_inner(c, -10_i32, 0.0, 42);
    // bench_arith_inner(c, -10_i64, 0.0, 42);
    // bench_arith_inner(c, -10.0_f32, 0.0, 42);
    // bench_arith_inner(c, -10.0_f64, 0.0, 42);

    // TODO: Benchmark portable-simd
}

criterion_group!(benches, bench_arith);
criterion_main!(benches);
