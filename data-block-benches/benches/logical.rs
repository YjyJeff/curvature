use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use data_block::array::{Array, BooleanArray};
use data_block::compute::logical;
use data_block_benches::{create_boolean_array, create_boolean_array_iter};

const NULL_DENSITY: f32 = 0.0;
const TRUE_DENSITY: f32 = 0.2;
const SEED_0: u64 = 99;
const SEED_1: u64 = 66;

fn logical_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("Logical");

    (10..=12).step_by(2).for_each(|log2_size| {
        let size = black_box(2usize.pow(log2_size));
        let lhs = create_boolean_array(size, NULL_DENSITY, TRUE_DENSITY, SEED_0);
        let rhs = create_boolean_array(size, NULL_DENSITY, TRUE_DENSITY, SEED_1);

        let mut dst = BooleanArray::default();

        unsafe { logical::and(&lhs, &rhs, &mut dst) };

        let arr_a = create_boolean_array_iter(size, NULL_DENSITY, TRUE_DENSITY, SEED_0)
            .collect::<arrow2::array::BooleanArray>();
        let arr_b = create_boolean_array_iter(size, NULL_DENSITY, TRUE_DENSITY, SEED_1)
            .collect::<arrow2::array::BooleanArray>();

        assert_eq!(
            dst.iter().collect::<Vec<_>>(),
            arrow2::compute::boolean::and(&arr_a, &arr_b)
                .iter()
                .collect::<Vec<_>>(),
        );

        group.bench_function(BenchmarkId::new("DataBlock: And", size), |b| {
            b.iter(|| unsafe { logical::and(&lhs, &rhs, &mut dst) });
        });

        group.bench_function(BenchmarkId::new("Arrow2: And", size), |b| {
            b.iter(|| {
                arrow2::compute::boolean::and(&arr_a, &arr_b);
            });
        });
    });
}

criterion_group!(benches, logical_benchmark);
criterion_main!(benches);
