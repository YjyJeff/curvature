use std::ops::{Add, Div};

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use data_block::array::{Array, PrimitiveArray};
use data_block::compute::arith::{
    ArrayAddElement, ArrayDivElement, ArrayMulElement, ArraySubElement,
};
use data_block::compute::{arith, IntrinsicType};
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
        + ArrayDivElement
        + ArrayAddElement
        + ArraySubElement
        + ArrayMulElement
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

        unsafe { arith::div_scalar(&lhs, rhs, &mut dst) };

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

        group.bench_function(BenchmarkId::new("DataBlock: add scalar", size), |b| {
            b.iter(|| unsafe {
                arith::add_scalar(&lhs, rhs, &mut dst);
            });
        });

        group.bench_function(BenchmarkId::new("DataBlock: sub scalar", size), |b| {
            b.iter(|| unsafe {
                arith::sub_scalar(&lhs, rhs, &mut dst);
            });
        });

        group.bench_function(BenchmarkId::new("DataBlock: mul scalar", size), |b| {
            b.iter(|| unsafe {
                arith::mul_scalar(&lhs, rhs, &mut dst);
            });
        });

        group.bench_function(BenchmarkId::new("DataBlock: div scalar", size), |b| {
            b.iter(|| unsafe {
                arith::div_scalar(&lhs, rhs, &mut dst);
            });
        });

        // group.bench_function(BenchmarkId::new("Arrow2: arith scalar", size), |b| {
        //     b.iter(|| arrow2::compute::arithmetics::basic::div_scalar(&arr_a, &rhs));
        // });

        // let mut result = vec![T::default(); size];
        // group.bench_function(BenchmarkId::new("Naive: arith scalar", size), |b| {
        //     b.iter(|| {
        //         lhs.values()
        //             .iter()
        //             .zip(result.iter_mut())
        //             .for_each(|(&lhs, dst)| {
        //                 *dst = lhs / rhs;
        //             });
        //     });
        // });

        let selection =
            data_block::bitmap::Bitmap::from_slice_and_len(&vec![u64::MAX; size / 64], size);

        group.bench_function(
            BenchmarkId::new("DataBlock: selected_add_scalar", size),
            |b| {
                b.iter(|| unsafe {
                    data_block::compute::arith::selected_add_scalar(
                        &selection, &lhs, rhs, &mut dst,
                    );
                })
            },
        );
        group.bench_function(
            BenchmarkId::new("DataBlock: selected_sub_scalar", size),
            |b| {
                b.iter(|| unsafe {
                    data_block::compute::arith::selected_sub_scalar(
                        &selection, &lhs, rhs, &mut dst,
                    );
                })
            },
        );
        group.bench_function(
            BenchmarkId::new("DataBlock: selected_mul_scalar", size),
            |b| {
                b.iter(|| unsafe {
                    data_block::compute::arith::selected_mul_scalar(
                        &selection, &lhs, rhs, &mut dst,
                    );
                })
            },
        );
        group.bench_function(
            BenchmarkId::new("DataBlock: selected_div_scalar", size),
            |b| {
                b.iter(|| unsafe {
                    data_block::compute::arith::selected_div_scalar(
                        &selection, &lhs, rhs, &mut dst,
                    );
                })
            },
        );
    });
}

fn bench_arith(c: &mut Criterion) {
    bench_arith_inner(c, -10_i8, 0.0, 42);
    bench_arith_inner(c, -10_i16, 0.0, 42);
    bench_arith_inner(c, -10_i32, 0.0, 42);
    bench_arith_inner(c, -10_i64, 0.0, 42);
    bench_arith_inner(c, -10.0_f32, 0.0, 42);
    bench_arith_inner(c, -10.0_f64, 0.0, 42);
}

criterion_group!(benches, bench_arith);
criterion_main!(benches);
