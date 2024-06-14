use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use data_block::array::{Array, BooleanArray, PrimitiveArray};
use data_block::bitmap::Bitmap;
use data_block::compute::comparison;
use data_block::compute::comparison::primitive::intrinsic::PartialOrdExt;
#[cfg(feature = "portable_simd")]
use data_block::compute::{
    comparison::primitive::intrinsic::IntrinsicSimdOrd, IntrinsicSimdType, IntrinsicType,
};
use data_block::element::string::StringView;
use pprof::criterion::{Output, PProfProfiler};

use data_block_benches::{
    create_boolean_array, create_primitive_array_with_seed, create_var_string_array,
    create_var_string_array_iter,
};
use rand::distributions::{Distribution, Standard};
use rand::Rng;

#[cfg(not(feature = "portable_simd"))]
fn bench_primitive_array<T, U>(c: &mut Criterion, rhs: T, null_density: f32, seed: u64)
where
    T: PartialOrdExt
        + arrow2::types::NativeType
        + arrow2::compute::comparison::Simd8
        + arrow::datatypes::ArrowNativeTypeOp,
    U: arrow::array::ArrowPrimitiveType<Native = T>,
    Standard: Distribution<T>,
    PrimitiveArray<T>: Array<Element = T>,
    T::Simd: arrow2::compute::comparison::Simd8PartialOrd,
{
    let mut group = c.benchmark_group(format!("Compare{}", std::any::type_name::<T>()));
    (10..=12).step_by(2).for_each(|log2_size| {
        let size = black_box(2usize.pow(log2_size));
        let lhs = create_primitive_array_with_seed::<T>(size, null_density, seed);
        let mut dst =
            BooleanArray::with_capacity(data_block::types::LogicalType::Boolean, size).unwrap();
        let mut selection = Bitmap::new();

        unsafe { comparison::primitive::intrinsic::ge_scalar(&mut selection, &lhs, rhs, &mut dst) };

        let arr_arrow2 = arrow2::util::bench_util::create_primitive_array_with_seed::<T>(
            size,
            null_density,
            seed,
        );

        assert_eq!(
            selection.iter().collect::<Vec<_>>(),
            arrow2::compute::comparison::primitive::gt_eq_scalar(&arr_arrow2, rhs)
                .iter()
                .map(|v| if let Some(v) = v { v } else { false })
                .collect::<Vec<_>>()
        );

        let arr_arrow = arrow::util::bench_util::create_primitive_array_with_seed::<U>(
            size,
            null_density,
            seed,
        );
        let arrow_rhs = arrow::array::PrimitiveArray::<U>::new_scalar(rhs);
        assert_eq!(
            selection.iter().collect::<Vec<_>>(),
            arrow::compute::kernels::cmp::gt_eq(&arr_arrow, &arrow_rhs)
                .unwrap()
                .iter()
                .map(|v| if let Some(v) = v { v } else { false })
                .collect::<Vec<_>>(),
        );

        selection.mutate().clear();
        group.bench_function(BenchmarkId::new("DataBlock: scalar", size), |b| {
            b.iter(|| unsafe {
                comparison::primitive::intrinsic::ge_scalar(&mut selection, &lhs, rhs, &mut dst);
                selection.mutate().clear();
            });
        });

        // group.bench_function(BenchmarkId::new("Arrow2: scalar", size), |b| {
        //     b.iter(|| arrow2::compute::comparison::primitive::gt_eq_scalar(&arr_arrow2, rhs))
        // });

        // group.bench_function(BenchmarkId::new("Arrow: scalar", size), |b| {
        //     b.iter(|| arrow::compute::kernels::cmp::gt_eq(&arr_arrow, &arrow_rhs).unwrap())
        // });

        let mut dst = vec![false; size];

        group.bench_function(BenchmarkId::new("Naive: scalar", size), |b| {
            b.iter(|| data_block_benches::comparison::ge_dynamic(lhs.values(), rhs, &mut dst))
        });
    });
}

#[cfg(feature = "portable_simd")]
fn bench_primitive_array<T, U>(c: &mut Criterion, rhs: T, null_density: f32, seed: u64)
where
    T: PartialOrdExt + arrow2::types::NativeType + arrow2::compute::comparison::Simd8,
    T::SimdType: IntrinsicSimdOrd,
    u64: num_traits::AsPrimitive<<T::SimdType as IntrinsicSimdOrd>::BitMaskType>,
    <T::SimdType as IntrinsicSimdOrd>::BitMaskType: bitvec::store::BitStore,
    Standard: Distribution<T>,
    PrimitiveArray<T>: Array<Element = T>,
    T::Simd: arrow2::compute::comparison::Simd8PartialOrd,
    U: arrow::array::ArrowPrimitiveType<Native = T>,
{
    let mut group = c.benchmark_group(format!("Compare{}", std::any::type_name::<T>()));
    (10..=12).step_by(2).for_each(|log2_size| {
        let size = black_box(2usize.pow(log2_size));
        let lhs = create_primitive_array_with_seed::<T>(size, null_density, seed);
        let mut selection = Bitmap::new();
        let mut dst = BooleanArray::default();

        unsafe {
            comparison::primitive::intrinsic::ge_scalar(&mut selection, &lhs, rhs, &mut dst);
        }

        let arr_arrow2 = arrow2::util::bench_util::create_primitive_array_with_seed::<T>(
            size,
            null_density,
            seed,
        );

        assert_eq!(
            selection.iter().collect::<Vec<_>>(),
            arrow2::compute::comparison::primitive::gt_eq_scalar(&arr_arrow2, rhs)
                .iter()
                .map(|v| if let Some(v) = v { v } else { false })
                .collect::<Vec<_>>()
        );

        selection.mutate().clear();
        group.bench_function(BenchmarkId::new("DataBlock: scalar", size), |b| {
            b.iter(|| unsafe {
                comparison::primitive::intrinsic::ge_scalar(&mut selection, &lhs, rhs, &mut dst);
                selection.mutate().clear();
            });
        });

        // group.bench_function(BenchmarkId::new("Arrow2: scalar", size), |b| {
        //     b.iter(|| arrow2::compute::comparison::primitive::gt_eq_scalar(&arr_arrow2, rhs))
        // });

        // let len = size
        //     / <T::SimdXLanes as data_block_benches::comparison::portable_simd::SimdXLanes>::LANES;
        // let mut dst = Vec::new();
        // group.bench_function(BenchmarkId::new("PortableSimd: scalar", size), |b| {
        //     b.iter(|| unsafe {
        //         dst.clear();
        //         dst.reserve(len);
        //         dst.set_len(len);

        //         data_block_benches::comparison::portable_simd::ge_scalar(
        //             arr_a.values(),
        //             rhs,
        //             &mut dst,
        //         )
        //     })
        // });
    });
}

fn bench_boolean_array(
    c: &mut Criterion,
    rhs: bool,
    null_density: f32,
    true_density: f32,
    seed: u64,
) {
    let mut group = c.benchmark_group("CompareBoolean");

    (10..=12).step_by(2).for_each(|log2_size| {
        let size = black_box(2usize.pow(log2_size));
        let lhs = create_boolean_array(size, null_density, true_density, seed);

        let mut selection = Bitmap::new();

        unsafe { comparison::boolean::ne_scalar(&mut selection, &lhs, rhs) };

        let arr_arrow2 =
            arrow2::util::bench_util::create_boolean_array(size, null_density, true_density);

        assert_eq!(
            selection.iter().collect::<Vec<_>>(),
            arrow2::compute::comparison::boolean::neq_scalar(&arr_arrow2, rhs)
                .iter()
                .map(|v| if let Some(v) = v { v } else { false })
                .collect::<Vec<_>>(),
        );

        let arr_arrow =
            arrow::util::bench_util::create_boolean_array(size, null_density, true_density);
        let arrow_rhs = arrow::array::BooleanArray::new_scalar(rhs);
        assert_eq!(
            selection.iter().collect::<Vec<_>>(),
            arrow::compute::kernels::cmp::neq(&arr_arrow, &arrow_rhs)
                .unwrap()
                .iter()
                .map(|v| if let Some(v) = v { v } else { false })
                .collect::<Vec<_>>()
        );

        selection.mutate().clear();

        group.bench_function(BenchmarkId::new("DataBlock: scalar", size), |b| {
            b.iter(|| unsafe {
                comparison::boolean::ne_scalar(&mut selection, &lhs, rhs);
                selection.mutate().clear();
            });
        });

        group.bench_function(BenchmarkId::new("Arrow2: scalar", size), |b| {
            b.iter(|| arrow2::compute::comparison::boolean::neq_scalar(&arr_arrow2, rhs));
        });

        group.bench_function(BenchmarkId::new("Arrow: scalar", size), |b| {
            b.iter(|| arrow::compute::kernels::cmp::neq(&arr_arrow, &arrow_rhs).unwrap());
        });
    });
}

fn bench_string_array(c: &mut Criterion, rhs: StringView<'_>, null_density: f32, seed: u64) {
    let mut group = c.benchmark_group("CompareString");
    (10..=12).step_by(2).for_each(|log2_size| {
        let size = 2usize.pow(log2_size);

        let lhs = create_var_string_array(size, null_density, seed);
        let mut selection = Bitmap::new();

        let arr_arrow2: arrow2::array::Utf8Array<i32> =
            create_var_string_array_iter(size, null_density, seed).collect();

        unsafe { comparison::string::ge_scalar(&mut selection, &lhs, rhs) }

        selection
            .iter()
            .zip(
                arrow2::compute::comparison::utf8::gt_eq_scalar(&arr_arrow2, rhs.as_str())
                    .iter()
                    .map(|v| if let Some(v) = v { v } else { false }),
            )
            .enumerate()
            .for_each(|(index, (lhs, rhs))| {
                if lhs != rhs {
                    panic!("Different at `{index}`, lhs: `{lhs}`, rhs: `{rhs}`",);
                }
            });

        let arr_arrow: arrow::array::StringArray =
            create_var_string_array_iter(size, null_density, seed).collect();
        let arrow_rhs = arrow::array::StringArray::new_scalar(rhs.as_str());

        assert_eq!(
            selection.iter().collect::<Vec<_>>(),
            arrow::compute::kernels::cmp::gt_eq(&arr_arrow, &arrow_rhs)
                .unwrap()
                .iter()
                .map(|v| if let Some(v) = v { v } else { false })
                .collect::<Vec<_>>()
        );

        selection.mutate().clear();

        group.bench_function(&format!("DataBlock: string 2^{log2_size}"), |b| {
            b.iter(|| unsafe {
                comparison::string::ge_scalar(&mut selection, &lhs, rhs);
                selection.mutate().clear()
            })
        });

        let rhs = rhs.as_str();
        group.bench_function(&format!("Arrow2: string 2^{log2_size}"), |b| {
            b.iter(|| arrow2::compute::comparison::utf8::gt_eq_scalar(&arr_arrow2, rhs))
        });

        group.bench_function(&format!("Arrow: string 2^{log2_size}"), |b| {
            b.iter(|| arrow::compute::kernels::cmp::gt_eq(&arr_arrow, &arrow_rhs).unwrap())
        });
    })
}

fn comparison_benchmark(c: &mut Criterion) {
    bench_primitive_array::<i8, arrow::datatypes::Int8Type>(c, 0, 0.1, 42);
    // bench_primitive_array::<i16, arrow::datatypes::Int16Type>(c, 0, 0.0, 42);
    // bench_primitive_array::<i32, arrow::datatypes::Int32Type>(c, 0, 0.1, 42);
    // bench_primitive_array::<i64, arrow::datatypes::Int64Type>(c, 0, 0.1, 42);
    // bench_primitive_array::<f32, arrow::datatypes::Float32Type>(c, 0.5, 0.0, 42);
    // bench_primitive_array::<f64, arrow::datatypes::Float64Type>(c, 0.5, 0.1, 42);
    // bench_boolean_array(c, true, 0.0, 0.2, 42);
    // bench_string_array(c, StringView::from_static_str("lmnolollmnolol"), 0.2, 42);
}

criterion_group!(name = benches; config = Criterion::default().with_profiler(PProfProfiler::new(100, Output::Flamegraph(None))); targets = comparison_benchmark);
criterion_main!(benches);
