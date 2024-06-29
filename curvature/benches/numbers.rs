//! Benchmark the overhead of the morsel-driven parallelism framework

#![allow(missing_docs)]

use std::num::NonZeroU64;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use criterion::{criterion_group, criterion_main, Criterion};
use curvature::common::client_context::{ClientContext, ExecArgs};
use curvature::common::types::ParallelismDegree;
use curvature::common::uuid::QueryId;
use curvature::exec::physical_operator::table_scan::numbers::Numbers;
use curvature::exec::physical_operator::PhysicalOperator;
use curvature::exec::query_executor::QueryExecutor;
use curvature::STANDARD_VECTOR_SIZE;
use data_block::array::ArrayImpl;
use data_block::block::DataBlock;
use data_block::compute::sequence::sequence;
use data_block::types::{Array, LogicalType};
use rayon::prelude::*;

const COUNT: NonZeroU64 = unsafe { NonZeroU64::new_unchecked(100000000) };
const PARALLELISM: ParallelismDegree = unsafe { ParallelismDegree::new_unchecked(10) };

fn bench_read_numbers(c: &mut Criterion) {
    let physical_plan: Arc<dyn PhysicalOperator> = Arc::new(Numbers::new(0, COUNT));

    let gt_sum = (COUNT.get() - 1) * COUNT.get() / 2;

    // WTF? it is faster?
    c.bench_function("QueryExecutorReadNumbers", |b| {
        b.iter(|| {
            let executor = QueryExecutor::try_new(
                &physical_plan,
                Arc::new(ClientContext::new(
                    QueryId::from_u128(4),
                    ExecArgs {
                        parallelism: PARALLELISM,
                    },
                )),
            )
            .unwrap();

            let sum = AtomicU64::new(0);
            executor
                .execute(|block| unsafe {
                    let array = block.get_array(0).unwrap_unchecked();
                    if let ArrayImpl::UInt64(array) = array {
                        sum.fetch_add(array.values_iter().sum::<u64>(), Ordering::Relaxed);
                    } else {
                        unreachable!()
                    }
                })
                .unwrap();

            assert_eq!(sum.load(Ordering::Relaxed), gt_sum);
        })
    });

    c.bench_function("RayonReadNumbers", |b| {
        b.iter(|| {
            let sum = AtomicU64::new(0);
            let count_per_thread = COUNT.get() / PARALLELISM.get() as u64;
            (0..PARALLELISM.get() as usize)
                .into_par_iter()
                .for_each(|i| {
                    let mut output =
                        DataBlock::with_logical_types(vec![LogicalType::UnsignedBigInt]);

                    let mut start = i as u64 * count_per_thread;
                    let end = (i as u64 + 1) * count_per_thread;
                    while start < end {
                        let end = std::cmp::min(start + STANDARD_VECTOR_SIZE as u64, end);
                        let guard = output.mutate_single_array((end - start) as _);
                        guard
                            .mutate(|array| {
                                if let ArrayImpl::UInt64(array) = array {
                                    unsafe { sequence(array, start, end) }
                                    sum.fetch_add(
                                        array.values_iter().sum::<u64>(),
                                        Ordering::Relaxed,
                                    );
                                    Ok::<_, curvature::error::BangError>(())
                                } else {
                                    unreachable!()
                                }
                            })
                            .unwrap();
                        start = end;
                    }
                });

            assert_eq!(sum.load(Ordering::Relaxed), gt_sum);
        })
    });
}

criterion_group!(benches, bench_read_numbers);
criterion_main!(benches);
