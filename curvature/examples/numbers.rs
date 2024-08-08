//! Range example

use std::num::NonZeroU64;
use std::sync::Arc;

use curvature::common::client_context::{ClientContext, ExecArgs};
use curvature::common::expr_operator::arith::ArithOperator;
use curvature::common::expr_operator::comparison::CmpOperator;
use curvature::common::types::ParallelismDegree;
use curvature::common::uuid::QueryId;
use curvature::exec::physical_expr::arith::Arith;
use curvature::exec::physical_expr::comparison::Comparison;
use curvature::exec::physical_expr::constant::Constant;
use curvature::exec::physical_expr::field_ref::FieldRef;
use curvature::exec::physical_expr::function::aggregate::avg::Avg;
use curvature::exec::physical_expr::function::aggregate::count::Count;
use curvature::exec::physical_expr::function::aggregate::min_max::{Max, Min};
use curvature::exec::physical_expr::function::aggregate::sum::Sum;
use curvature::exec::physical_expr::function::aggregate::AggregationFunctionExpr;
use curvature::exec::physical_expr::PhysicalExpr;
use curvature::exec::physical_operator::aggregate::hash_aggregate::serde::{
    FixedSizedSerdeKeySerializer, NonNullableFixedSizedSerdeKeySerializer,
    NonNullableGeneralSerializer,
};
use curvature::exec::physical_operator::aggregate::hash_aggregate::HashAggregate;
use curvature::exec::physical_operator::aggregate::simple_aggregate::SimpleAggregate;
use curvature::exec::physical_operator::filter::Filter;
use curvature::exec::physical_operator::limit::streaming_limit::StreamingLimit;
use curvature::exec::physical_operator::projection::Projection;
use curvature::exec::physical_operator::table_scan::numbers::Numbers;
use curvature::exec::physical_operator::PhysicalOperator;
use curvature::exec::query_executor::QueryExecutor;
use curvature::tree_node::display::IndentDisplayWrapper;
use data_block::array::{ArrayImpl, UInt64Array, UInt8Array};
use data_block::types::LogicalType;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

const HASH_COUNT: NonZeroU64 = unsafe { NonZeroU64::new_unchecked(10000000000) };
const SIMPLE_COUNT: NonZeroU64 = unsafe { NonZeroU64::new_unchecked(100000000000) };
const PARALLELISM: ParallelismDegree = unsafe { ParallelismDegree::new_unchecked(10) };

#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

fn main() {
    tracing_subscriber::registry()
        .with(tracing_subscriber::fmt::layer())
        .with(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let now = std::time::Instant::now();

    // let physical_plan: Arc<dyn PhysicalOperator> = Arc::new(Numbers::new(0, SIMPLE_COUNT));

    // // let physical_plan: Arc<dyn PhysicalOperator> =
    // //     Arc::new(StreamingLimit::try_new(physical_plan, NonZeroU64::new(10).unwrap(), 0).unwrap());

    // let payloads_ref = [FieldRef::new(
    //     0,
    //     LogicalType::UnsignedBigInt,
    //     "number".to_string(),
    // )];

    // // let physical_plan: Arc<dyn PhysicalOperator> = Arc::new(
    // //     SimpleAggregate::try_new(
    // //         physical_plan,
    // //         vec![
    // //             AggregationFunctionExpr::try_new(
    // //                 &payloads_ref,
    // //                 Arc::new(Sum::<UInt64Array>::try_new(LogicalType::UnsignedBigInt).unwrap()),
    // //             )
    // //             .unwrap(),
    // //             AggregationFunctionExpr::try_new(
    // //                 &payloads_ref,
    // //                 Arc::new(Max::<UInt64Array>::try_new(LogicalType::UnsignedBigInt).unwrap()),
    // //             )
    // //             .unwrap(),
    // //             AggregationFunctionExpr::try_new(
    // //                 &payloads_ref,
    // //                 Arc::new(Avg::<UInt64Array>::try_new(LogicalType::UnsignedBigInt).unwrap()),
    // //             )
    // //             .unwrap(),
    // //         ],
    // //     )
    // //     .unwrap(),
    // // );

    // let field_ref: Arc<dyn PhysicalExpr> = Arc::new(FieldRef::new(
    //     0,
    //     LogicalType::UnsignedBigInt,
    //     "number".to_string(),
    // ));

    // let physical_plan: Arc<dyn PhysicalOperator> = Arc::new(
    //     Filter::try_new(
    //         physical_plan,
    //         Arc::new(
    //             Comparison::try_new(
    //                 Arc::new(
    //                     unsafe {
    //                         Constant::try_new(ArrayImpl::UInt8(UInt8Array::from_values_iter([0])))
    //                     }
    //                     .unwrap(),
    //                 ),
    //                 CmpOperator::NotEqual,
    //                 Arc::new(
    //                     Arith::try_new(
    //                         Arc::clone(&field_ref),
    //                         ArithOperator::Rem,
    //                         Arc::new(unsafe {
    //                             Constant::try_new(ArrayImpl::UInt8(UInt8Array::from_values_iter([
    //                                 4,
    //                             ])))
    //                             .unwrap()
    //                         }),
    //                     )
    //                     .unwrap(),
    //                 ),
    //             )
    //             .unwrap(),
    //         ),
    //     )
    //     .unwrap(),
    // );

    // let physical_plan: Arc<dyn PhysicalOperator> = Arc::new(
    //     SimpleAggregate::try_new(
    //         physical_plan,
    //         vec![
    //             AggregationFunctionExpr::try_new(
    //                 &payloads_ref,
    //                 Arc::new(Count::<false>::new(LogicalType::UnsignedBigInt)),
    //             )
    //             .unwrap(),
    //             AggregationFunctionExpr::try_new(
    //                 &payloads_ref,
    //                 Arc::new(Max::<UInt64Array>::try_new(LogicalType::UnsignedBigInt).unwrap()),
    //             )
    //             .unwrap(),
    //             AggregationFunctionExpr::try_new(
    //                 &payloads_ref,
    //                 Arc::new(Min::<UInt64Array>::try_new(LogicalType::UnsignedBigInt).unwrap()),
    //             )
    //             .unwrap(),
    //         ],
    //     )
    //     .unwrap(),
    // );

    // group by number % 3, number %4, number % 5

    let physical_plan: Arc<dyn PhysicalOperator> = Arc::new(Numbers::new(0, HASH_COUNT));

    let field_ref: Arc<dyn PhysicalExpr> = Arc::new(FieldRef::new(
        0,
        LogicalType::UnsignedBigInt,
        "number".to_string(),
    ));

    let physical_plan: Arc<dyn PhysicalOperator> = Arc::new(Projection::new(
        physical_plan,
        vec![
            Arc::clone(&field_ref),
            Arc::new(
                Arith::try_new(
                    Arc::clone(&field_ref),
                    ArithOperator::Rem,
                    Arc::new(unsafe {
                        Constant::try_new(ArrayImpl::UInt8(UInt8Array::from_values_iter([3])))
                            .unwrap()
                    }),
                )
                .unwrap(),
            ),
            Arc::new(
                Arith::try_new(
                    Arc::clone(&field_ref),
                    ArithOperator::Rem,
                    Arc::new(unsafe {
                        Constant::try_new(ArrayImpl::UInt8(UInt8Array::from_values_iter([4])))
                            .unwrap()
                    }),
                )
                .unwrap(),
            ),
            Arc::new(
                Arith::try_new(
                    Arc::clone(&field_ref),
                    ArithOperator::Rem,
                    Arc::new(unsafe {
                        Constant::try_new(ArrayImpl::UInt8(UInt8Array::from_values_iter([5])))
                            .unwrap()
                    }),
                )
                .unwrap(),
            ),
        ],
    ));

    let field_ref = [FieldRef::new(
        0,
        LogicalType::UnsignedBigInt,
        "number".to_string(),
    )];
    let physical_plan: Arc<dyn PhysicalOperator> = Arc::new(
        HashAggregate::<NonNullableFixedSizedSerdeKeySerializer<u64>>::try_new(
            physical_plan,
            &[
                FieldRef::new(1, LogicalType::UnsignedTinyInt, "number % 3".to_string()),
                FieldRef::new(2, LogicalType::UnsignedTinyInt, "number % 4".to_string()),
                FieldRef::new(3, LogicalType::UnsignedTinyInt, "number % 5".to_string()),
            ],
            vec![
                AggregationFunctionExpr::try_new(
                    &field_ref,
                    Arc::new(Max::<UInt64Array>::try_new(LogicalType::UnsignedBigInt).unwrap()),
                )
                .unwrap(),
                AggregationFunctionExpr::try_new(
                    &field_ref,
                    Arc::new(Sum::<UInt64Array>::try_new(LogicalType::UnsignedBigInt).unwrap()),
                )
                .unwrap(),
                AggregationFunctionExpr::try_new(
                    &field_ref,
                    Arc::new(Avg::<UInt64Array>::try_new(LogicalType::UnsignedBigInt).unwrap()),
                )
                .unwrap(),
            ],
        )
        .unwrap(),
    );

    tracing::info!("{}", IndentDisplayWrapper::new(&*physical_plan));

    let client_ctx = Arc::new(ClientContext::new(
        QueryId::from_u128(7),
        ExecArgs {
            parallelism: PARALLELISM,
        },
    ));

    let executor = QueryExecutor::try_new(&physical_plan, Arc::clone(&client_ctx)).unwrap();

    ctrlc::set_handler(move || client_ctx.cancel()).expect("Setting Ctrl-C handler should succeed");

    if let Err(e) = executor.execute(|block| println!("{}", block)) {
        tracing::error!("{}", snafu::Report::from_error(e))
    }
    // executor.execute(|block| {}).unwrap();

    tracing::info!("Elapsed {} sec", now.elapsed().as_millis() as f64 / 1000.0);
}
