//! Aggregate based on hash
//!
//! # Concepts
//!
//! - Payload
//! - GroupByKeys

mod hash_table;
pub mod serde;

use std::fmt::Debug;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;
use std::sync::Arc;

use self::hash_table::Element;
use self::hash_table::HashTable;
use self::serde::Serde;
use crate::common::client_context::ClientContext;
use crate::common::types::HashValue;
use crate::error::SendableError;
use crate::exec::physical_expr::function::aggregate::{
    AggregationFunction, AggregationFunctionList,
};
use crate::exec::physical_operator::{
    impl_regular_for_non_regular, use_types_for_impl_regular_for_non_regular, GlobalSinkState,
    GlobalSourceState, LocalSinkState, LocalSourceState, OperatorResult, ParallelismDegree,
    PhysicalOperator, SinkExecStatus, SourceExecStatus, StateStringify, Stringify,
};
use crate::STANDARD_VECTOR_SIZE;
use data_block::block::DataBlock;
use data_block::types::LogicalType;
use hashbrown::raw::RawTable as SwissTable;
use parking_lot::RwLock;
use snafu::ensure;
use snafu::{ResultExt, Snafu};
use strength_reduce::StrengthReducedU64;

use super::ThreadSafeArena;
use_types_for_impl_regular_for_non_regular!();

#[allow(missing_docs)]
#[derive(Debug, Snafu)]
pub enum HashAggregateError {
    #[snafu(display("Aggregation functions passed to HashAggregate is empty. HashAggregate needs at least one aggregation function"))]
    EmptyAggregationFuncs,
}

type Result<T> = std::result::Result<T, HashAggregateError>;

/// A hyper-parameter that determine when to partition the map. Tune me!
const PARTITION_THRESHOLD: usize = 8192;

/// Aggregate based on hash
#[derive(Debug)]
pub struct HashAggregate<S: Serde> {
    output_types: Vec<LogicalType>,
    children: Vec<Arc<dyn PhysicalOperator>>,
    agg_func_list: AggregationFunctionList,
    global_state: Arc<GlobalState<S::SerdeKey>>,
}

#[derive(Debug)]
struct GlobalState<K> {
    /// Swiss tables collected from local states.
    ///
    /// TBD: Use dash map here, such that we can combine the aggregation states
    /// immediately in the `finsh_local_sink` call. Will the synchronization become
    /// bottleneck? Currently, we will combine the swiss tables from different threads
    /// in the `finalize_sink` call with multi-threading
    swiss_tables: RwLock<Vec<SwissTables<Element<K>>>>,
    /// Until now, does any thread has partitioned the swiss table? The `finsh_local_sink`
    /// method should check it. If it is true, the method should partition the swiss
    /// table before insert into `self.swiss_tables`. The `finalize_sink` method should also
    /// check it. If it is true, the method should partition the swiss table that
    /// is not partitioned. All of the memory ordering used to access this variable
    /// is `Relaxed`!
    any_partitioned: AtomicBool,
}

/// Swiss tables collected from local states.
struct SwissTables<K> {
    /// Swiss tables produced by a single thread. It is guaranteed to be non-empty
    tables: Vec<SwissTable<Element<K>>>,
    /// Arena the element in the tables located on
    arena: ThreadSafeArena,
}

impl<K: Debug> Debug for SwissTables<K> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SwissTables {{ [")?;
        for table in self.tables.iter() {
            f.debug_list()
                .entries(unsafe { table.iter().map(|bucket| bucket.as_ref()) })
                .finish()?;
        }
        write!(f, "] }}")
    }
}

impl<S: Serde> HashAggregate<S> {
    /// FIXME: lots of check
    pub fn try_new(
        input: Arc<dyn PhysicalOperator>,
        agg_funcs: Vec<Arc<dyn AggregationFunction>>,
    ) -> Result<Self> {
        ensure!(!agg_funcs.is_empty(), EmptyAggregationFuncsSnafu {});

        let agg_func_list = AggregationFunctionList::new(agg_funcs);
        Ok(Self {
            output_types: agg_func_list.output_types(),
            children: vec![input],
            agg_func_list,
            global_state: Arc::new(GlobalState {
                swiss_tables: RwLock::new(Vec::new()),
                any_partitioned: AtomicBool::new(false),
            }),
        })
    }
}

/// Global source state for [`HashAggregate`]
#[derive(Debug)]
pub struct HashAggregateGlobalSourceState;

impl StateStringify for HashAggregateGlobalSourceState {
    fn name(&self) -> &'static str {
        "HashAggregateGlobalSourceState"
    }

    fn debug(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl GlobalSourceState for HashAggregateGlobalSourceState {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Local source state for [`HashAggregate`]
#[derive(Debug)]
pub struct HashAggregateLocalSourceState;

impl StateStringify for HashAggregateLocalSourceState {
    fn name(&self) -> &'static str {
        "HashAggregateLocalSourceState"
    }

    fn debug(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl LocalSourceState for HashAggregateLocalSourceState {
    fn as_mut_any(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn read_data(&mut self, _output: &mut DataBlock) -> OperatorResult<SourceExecStatus> {
        todo!()
    }
}

/// Global sink state for [`HashAggregate`]
///
/// TBD: Use radix partitioning ?
#[derive(Debug)]
pub struct HashAggregateGlobalSinkState<K> {
    global_state: Arc<GlobalState<K>>,
    partitioning: Option<ModuloPartitioning>,
}

impl<K: Debug> StateStringify for HashAggregateGlobalSinkState<K> {
    fn name(&self) -> &'static str {
        "HashAggregateGlobalSinkState"
    }

    fn debug(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl<K: Send + Sync + 'static + Debug> GlobalSinkState for HashAggregateGlobalSinkState<K> {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Local sink state for [`HashAggregate`]
#[derive(Debug)]
pub struct HashAggregateLocalSinkState<S: Serde> {
    hash_table: HashTable<S>,
}

impl<S: Serde> StateStringify for HashAggregateLocalSinkState<S> {
    fn name(&self) -> &'static str {
        "HashAggregateLocalSinkState"
    }

    fn debug(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl<S: Serde> LocalSinkState for HashAggregateLocalSinkState<S> {
    fn as_mut_any(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

impl<S: Serde> Stringify for HashAggregate<S> {
    fn name(&self) -> &'static str {
        "HashAggregate"
    }

    fn debug(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }

    /// FIXME: display
    fn display(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "HashAggregate")
    }
}

impl<S: Serde> PhysicalOperator for HashAggregate<S> {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn output_types(&self) -> &[LogicalType] {
        &self.output_types
    }

    fn children(&self) -> &[Arc<dyn PhysicalOperator>] {
        &self.children
    }

    impl_regular_for_non_regular!();

    // Source

    fn is_source(&self) -> bool {
        true
    }

    fn source_parallelism_degree(
        &self,
        _global_state: &dyn GlobalSourceState,
    ) -> OperatorResult<ParallelismDegree> {
        todo!()
    }

    fn read_data(
        &self,
        _output: &mut DataBlock,
        _global_state: &dyn GlobalSourceState,
        _local_state: &mut dyn LocalSourceState,
    ) -> OperatorResult<SourceExecStatus> {
        todo!()
    }

    fn global_source_state(
        &self,
        _client_ctx: &ClientContext,
    ) -> OperatorResult<Arc<dyn GlobalSourceState>> {
        Ok(Arc::new(HashAggregateGlobalSourceState))
    }

    fn local_source_state(
        &self,
        _global_state: &dyn GlobalSourceState,
    ) -> OperatorResult<Box<dyn LocalSourceState>> {
        Ok(Box::new(HashAggregateLocalSourceState))
    }

    fn progress(&self, _global_state: &dyn GlobalSourceState) -> OperatorResult<f64> {
        todo!()
    }

    // Sink

    fn is_sink(&self) -> bool {
        true
    }

    fn is_parallel_sink(&self) -> OperatorResult<bool> {
        Ok(true)
    }

    fn write_data(
        &self,
        _input: &DataBlock,
        _global_state: &dyn GlobalSinkState,
        _local_state: &mut dyn LocalSinkState,
    ) -> OperatorResult<SinkExecStatus> {
        todo!()
    }

    fn finish_local_sink(
        &self,
        _global_state: &dyn GlobalSinkState,
        _local_state: &mut dyn LocalSinkState,
    ) -> OperatorResult<()> {
        todo!()
    }

    unsafe fn finalize_sink(&self, _global_state: &dyn GlobalSinkState) -> OperatorResult<()> {
        todo!()
    }

    fn global_sink_state(
        &self,
        client_ctx: &ClientContext,
    ) -> OperatorResult<Arc<dyn GlobalSinkState>> {
        let parallelism = client_ctx.exec_args.parallelism;

        Ok(Arc::new(HashAggregateGlobalSinkState {
            global_state: Arc::clone(&self.global_state),
            partitioning: if parallelism > ParallelismDegree::MIN {
                Some(ModuloPartitioning::new(parallelism))
            } else {
                None
            },
        }))
    }

    fn local_sink_state(
        &self,
        _global_state: &dyn GlobalSinkState,
    ) -> OperatorResult<Box<dyn LocalSinkState>> {
        Ok(Box::new(HashAggregateLocalSinkState::<S> {
            hash_table: HashTable::new(),
        }))
    }
}

/// Trait for partitioning the [`HashValue`] into partition index
///
/// Partitioning is used to partition a single swiss table into multiple swiss tables,
/// such that each thread processes a partitioned table can get correct result.
/// Which means that the parallelism degree is increased
pub trait Partitioning: Send + Sync + 'static {
    /// Create a partitioning
    fn new(partition_count: ParallelismDegree) -> Self;

    /// Get number of partitions
    fn partition_count(&self) -> usize;

    /// Partition the [`HashValue`] into partition index. Note that the returned
    /// partition index should smaller than partition count
    fn partition(&self, hash_value: HashValue) -> usize;
}

/// Partitioning based on modulo operation
#[derive(Debug, Clone)]
pub struct ModuloPartitioning {
    partition_count: usize,
    modulo: StrengthReducedU64,
}

impl Partitioning for ModuloPartitioning {
    fn new(partition_count: ParallelismDegree) -> Self {
        Self {
            partition_count: partition_count.get() as _,
            modulo: StrengthReducedU64::new(partition_count.get() as _),
        }
    }

    #[inline]
    fn partition_count(&self) -> usize {
        self.partition_count
    }

    #[inline]
    fn partition(&self, hash_value: HashValue) -> usize {
        (hash_value % self.modulo) as usize
    }
}

/// Partition the hash table into multiple hash tables
fn partition<S: Serde, P: Partitioning>(
    hash_table: SwissTable<Element<S::SerdeKey>>,
    partitioned: &mut Vec<SwissTable<Element<S::SerdeKey>>>,
    partitioning: &P,
) {
    debug_assert!(partitioned.is_empty());

    partitioned.resize_with(partitioning.partition_count(), || {
        SwissTable::with_capacity(STANDARD_VECTOR_SIZE)
    });

    hash_table.into_iter().for_each(|element| {
        let partition_index = partitioning.partition(element.serde_key_and_hash.hash_value);
        debug_assert!(partition_index < partitioning.partition_count());

        // SAFETY: We already allocate the hash table in the tables and the partition index
        // is guaranteed smaller than the partition count
        let hash_table = unsafe { partitioned.get_unchecked_mut(partition_index) };
        hash_table.insert(element.serde_key_and_hash.hash_value, element, |element| {
            element.serde_key_and_hash.hash_value
        });
    });
}
