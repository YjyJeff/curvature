use std::num::NonZeroU64;
use std::sync::Arc;

use curvature::common::client_context::{ClientContext, ExecArgs};
use curvature::common::types::ParallelismDegree;
use curvature::common::uuid::QueryId;
use curvature::exec::physical_expr::arith::DefaultRemConstantArith;
use curvature::exec::physical_expr::field_ref::FieldRef;
use curvature::exec::physical_expr::function::aggregate::min_max::Max;
use curvature::exec::physical_expr::PhysicalExpr;
use curvature::exec::physical_operator::aggregate::hash_aggregate::serde::NonNullableFixedSizedSerdeKeySerializer;
use curvature::exec::physical_operator::aggregate::hash_aggregate::HashAggregate;
use curvature::exec::physical_operator::numbers::Numbers;
use curvature::exec::physical_operator::projection::Projection;
use curvature::exec::physical_operator::PhysicalOperator;
use curvature::exec::query_executor::QueryExecutor;
use curvature::tree_node::display::IndentDisplayWrapper;
use data_block::array::UInt64Array;
use data_block::types::LogicalType;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

const COUNT: NonZeroU64 = unsafe { NonZeroU64::new_unchecked(10000000000) };
const PARALLELISM: ParallelismDegree = unsafe { ParallelismDegree::new_unchecked(10) };

/// TBD: is u32 much faster than [u8; 32]?
fn main() {
    tracing_subscriber::registry()
        .with(tracing_subscriber::fmt::layer())
        .with(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let now = std::time::Instant::now();
    let field_ref: Arc<dyn PhysicalExpr> = Arc::new(FieldRef::new(
        0,
        LogicalType::UnsignedBigInt,
        "number".to_string(),
    ));

    let physical_plan: Arc<dyn PhysicalOperator> = Arc::new(Numbers::new(0, COUNT));

    let physical_plan: Arc<dyn PhysicalOperator> = Arc::new(Projection::new(
        physical_plan,
        vec![
            Arc::clone(&field_ref),
            Arc::new(DefaultRemConstantArith::new(
                Arc::clone(&field_ref),
                3_u64,
                "number % 3".to_string(),
            )),
            Arc::new(DefaultRemConstantArith::new(
                Arc::clone(&field_ref),
                4_u64,
                "number % 4".to_string(),
            )),
            Arc::new(DefaultRemConstantArith::new(
                Arc::clone(&field_ref),
                5_u64,
                "number % 5".to_string(),
            )),
        ],
    ));

    let physical_plan: Arc<dyn PhysicalOperator> = Arc::new(
        HashAggregate::<NonNullableFixedSizedSerdeKeySerializer<[u8; 32]>>::try_new(
            physical_plan,
            vec![
                Arc::new(FieldRef::new(
                    1,
                    LogicalType::UnsignedBigInt,
                    "number % 3".to_string(),
                )),
                Arc::new(FieldRef::new(
                    2,
                    LogicalType::UnsignedBigInt,
                    "number % 4".to_string(),
                )),
                Arc::new(FieldRef::new(
                    3,
                    LogicalType::UnsignedBigInt,
                    "number % 5".to_string(),
                )),
            ],
            vec![Arc::new(
                Max::<UInt64Array>::try_new(Arc::clone(&field_ref)).unwrap(),
            )],
        )
        .unwrap(),
    );

    tracing::info!("{}", IndentDisplayWrapper::new(&*physical_plan));

    let executor = QueryExecutor::try_new(
        &physical_plan,
        Arc::new(ClientContext {
            query_id: QueryId::from_u128(7),
            exec_args: ExecArgs {
                parallelism: PARALLELISM,
            },
        }),
    )
    .unwrap();

    executor.execute(|block| println!("{}", block)).unwrap();

    tracing::info!("Elapsed {} sec", now.elapsed().as_millis() as f64 / 1000.0);
}
