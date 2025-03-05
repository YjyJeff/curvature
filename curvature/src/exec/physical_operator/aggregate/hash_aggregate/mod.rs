//! Aggregate based on hash table
//!
//! # Concepts
//!
//! - `Payloads`: For each [`AggregationFunction`], it take expressions as inputs. These
//! inputs are called payloads. Aggregation can compute many [`AggregationFunction`],
//! therefore, their `payloads` may contain redundant computation.
//! For example: `select min(a / 2), sum(a / 2) from table group by a`. To avoid above
//! problem, we enforce the `Payloads` to be [`FieldRef`] and transfer this problem to
//! planner/optimizer! It is the planner/optimizer's responsibility to extract the common
//! expressions, sounds make sense 😄
//!
//! - `GroupByKeys`: expressions after the `group by` keywords. Same with the `Payloads`,
//! it requires that all of them must be [`FieldRef`].
//!
//! [`AggregationFunction`]: crate::exec::physical_expr::function::aggregate::AggregationFunction
//! [`FieldRef`]: crate::exec::physical_expr::field_ref::FieldRef

pub mod hash_table;
pub mod serde;

use curvature_procedural_macro::MetricsSetBuilder;
use data_block::array::ArrayImpl;
use data_block::block::{DataBlock, MutateArrayError};
use data_block::types::{LogicalType, PhysicalSize};
use hashbrown::hash_table::{HashTable as SwissTable, IntoIter};
use parking_lot::Mutex;
use rayon::iter::{IntoParallelIterator, IntoParallelRefMutIterator, ParallelIterator};
use snafu::{ResultExt, Snafu, ensure};
use strength_reduce::StrengthReducedU64;

use std::fmt::Debug;
use std::ops::DerefMut;
use std::sync::Arc;
use std::sync::atomic::Ordering::Relaxed;
use std::sync::atomic::{AtomicBool, AtomicUsize};
use std::time::Duration;

use self::hash_table::{Element, HashTable, HashTableMetrics, probing_swiss_table};
use self::serde::{Serde, SerdeKey};
use crate::common::client_context::ClientContext;
use crate::common::profiler::ScopedTimerGuard;
use crate::common::types::HashValue;
use crate::exec::physical_expr::field_ref::FieldRef;
use crate::exec::physical_expr::function::aggregate::{
    AggregationFunctionExpr, AggregationFunctionList, AggregationStatesPtr,
};
use crate::exec::physical_operator::ext_traits::{SinkOperatorExt, SourceOperatorExt};
use crate::exec::physical_operator::metric::{Count, MetricsSet, Time};
use crate::exec::physical_operator::{
    FinalizeSinkSnafu, GlobalSinkState, GlobalSourceState, LocalSinkState, LocalSourceState,
    OperatorResult, ParallelismDegree, PhysicalOperator, ReadDataSnafu, SinkExecStatus,
    SourceExecStatus, StateStringify, Stringify, WriteDataSnafu, impl_regular_for_non_regular,
    use_types_for_impl_regular_for_non_regular,
};

use crate::STANDARD_VECTOR_SIZE;
use crate::exec::physical_operator::utils::downcast_mut_local_state;

use super::Arena;

use_types_for_impl_regular_for_non_regular!();

#[allow(missing_docs)]
#[derive(Debug, Snafu)]
pub enum HashAggregateError {
    #[snafu(display(
        "Group by keys passed to HashAggregate is empty, use `SimpleAggregate` instead "
    ))]
    EmptyGroupByKeys,
    #[snafu(display(
        "`FieldRef` with index `{ref_index}` is out or range, the input only has `{input_size}` fields"
    ))]
    FieldRefOutOfRange { ref_index: usize, input_size: usize },
    #[snafu(display(
        "TypeMismatch: `FieldRef` with index `{ref_index}` has logical type `{:?}`, however the field in the input has logical type `{:?}`",
        ref_type,
        input_type
    ))]
    FieldRefTypeMismatch {
        ref_index: usize,
        ref_type: LogicalType,
        input_type: LogicalType,
    },
    #[snafu(display(
        "Serde key `{}` is smaller than group by keys `{:?}`",
        serde_key,
        group_by_keys
    ))]
    SerdeKeySmallerThanGroupByKeys {
        serde_key: &'static str,
        group_by_keys: Vec<LogicalType>,
    },
    #[snafu(display(
        "Aggregation functions passed to HashAggregate is empty. HashAggregate needs at least one aggregation function"
    ))]
    EmptyAggregationFuncs,
}

type Result<T> = std::result::Result<T, HashAggregateError>;

/// A hyper-parameter that determine when to partition the map. Tune me!
const PARTITION_THRESHOLD: usize = 8192;

/// FIXME: Group by single key, nullable, non-nullable(do we need serialize?)
///
/// Aggregate based on hash table
///
/// # Generic
///
/// `S`: Serde used to serialize/deserialize `GroupByKeys`
///
/// # Lifetime of the element
///
/// Currently, we use `Arena` to store the elements(`AggregationStates`) of the hash table,
/// because aggregation is a typical phase-oriented allocations: all of the `AggregationStates`
/// will be allocated during the sink phase and deallocated in the source phase. However, it
/// takes a new problem: the `AggregationStates` is allocated in the arena and will not
/// be dropped automatically when we freeze the arena! So we have to drop it by ourselves.
///
/// We need to drop the elements in the following places:
///
/// - [`HashAggregateLocalSinkState`]: When the query is cancelled during the sink phase,
///   each thread will drop this local state. The `AggregationStates` generated by the
///   data-block until now should also be dropped.
///
/// - [`HashAggregateLocalSourceState`]: When the sink phase is completed, now we need to
///   consume the states in the source phase. We may do not consume the all source state
///   because of query is cancelled or `limit` is reached, etc. Therefore, when it goes
///   out of scope, we need to drop the states.
///
/// - [`HashAggregate`]: It contains the shared state used by the sink and source phase.
///   When it goes out of the scope, we need to drop the unconsumed states.
///
/// - Until now, we have dropped all of the states that have not been consumed. How the
///   states that have been consumed works? The states is consumed in the read phase: we
///   will read a batch of states into the array. In this process, we will call the
///   [`take_states`](AggregationFunctionList::take_states) to drop the memory occupied by
///   the states
#[derive(Debug)]
pub struct HashAggregate<S: Serde> {
    output_types: Vec<LogicalType>,
    children: [Arc<dyn PhysicalOperator>; 1],
    group_by_keys_indexes: Vec<usize>,
    agg_func_list: AggregationFunctionList,
    payloads_indexes: Vec<usize>,
    global_state: GlobalState<S::SerdeKey>,
}

#[derive(Debug)]
struct GlobalState<K: SerdeKey> {
    /// Swiss tables collected from different threads. In different phases, this struct
    /// has different meaning. In the `sink` phase, It represents partitioned/un-partitioned
    /// tables collected from different threads. In the `source` phase, it represents
    /// the partitioned/un-partitioned combined tables for reading data, therefore, all of
    /// the `tables` field in the `ThreadLocalTables` should only contain 1 combined table
    ///
    /// TBD: Use dash map here, such that we can combine the aggregation states
    /// immediately in the `finish_local_sink` call. Will the synchronization become
    /// bottleneck? Currently, we will combine the swiss tables from different threads
    /// in the `finalize_sink` call with multi-threading
    collected_swiss_tables: Mutex<Vec<ThreadLocalTables<K>>>,

    /// The total number of elements in the combined tables. It is set by the `finalize_sink`
    /// method and read in the `source` phase. Other methods should never use it! Actually,
    /// it do not need to be atomic, only single thread write it and many threads read it.
    /// The write and read will never happen at the same time. However, the compiler can not
    /// know it.
    total_elements_count: AtomicUsize,

    /// Metics
    metrics: HashAggregateMetics,
}

/// Swiss tables collected from a single thread's local states.
struct ThreadLocalTables<K: SerdeKey> {
    /// Swiss tables produced by a single thread. It is guaranteed to be non-empty
    tables: Vec<SwissTable<Element<K>>>,
    /// Arena the element in the tables located on
    arena: Arena,
}

impl<K: Debug + SerdeKey> Debug for ThreadLocalTables<K> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ThreadLocalTables {{ [")?;
        for table in self.tables.iter() {
            f.debug_list().entries(table.iter()).finish()?;
            write!(f, ", ")?;
        }
        write!(f, "] }}")
    }
}

/// Metrics for the hash aggregate
#[derive(Debug, Default, MetricsSetBuilder)]
pub struct HashAggregateMetics {
    // Sink
    /// The number of rows written to the operator
    num_rows: Count,
    /// Time spent in serialization
    serialize_time: Time,
    /// Time spent in hash the serde key
    hash_time: Time,
    /// Time spent in probing the hash table
    probing_time: Time,
    /// Time spent in update the aggregation states
    update_states_time: Time,

    // Source
    /// Time spent in deserialization
    deserialize_time: Time,
    /// Time spent in take the states into arrays
    take_states_time: Time,
}

impl HashAggregateMetics {
    fn combine_sink_metrics(&self, local_metics: &HashTableMetrics) {
        self.num_rows.add(local_metics.num_rows);
        self.serialize_time
            .add_duration(local_metics.serialize_time);
        self.hash_time.add_duration(local_metics.hash_time);
        self.probing_time.add_duration(local_metics.probing_time);
        self.update_states_time
            .add_duration(local_metics.update_states_time);
    }

    fn combine_source_metrics(&self, local_metrics: &LocalSourceMetrics) {
        self.deserialize_time
            .add_duration(local_metrics.deserialize_time);
        self.take_states_time
            .add_duration(local_metrics.take_states_time);
    }
}

impl<S: Serde> HashAggregate<S> {
    /// Try to create a HashAggregate
    ///
    /// FIXME: Check the type of the group by keys. For example, currently, we do not
    /// support group by list
    pub fn try_new(
        input: Arc<dyn PhysicalOperator>,
        group_by_keys: &[FieldRef],
        agg_funcs: Vec<AggregationFunctionExpr<'_>>,
    ) -> Result<Self> {
        let input_logical_types = input.output_types();
        let input_size = input_logical_types.len();

        let mut output_types = Vec::with_capacity(input_size + agg_funcs.len());
        let mut group_by_keys_indexes = Vec::with_capacity(group_by_keys.len());

        let mut payloads_indexes = Vec::with_capacity(agg_funcs.len());

        ensure!(!group_by_keys.is_empty(), EmptyGroupByKeysSnafu);

        let total_size =
            group_by_keys
                .iter()
                .try_fold(PhysicalSize::Fixed(0), |acc, field_ref| {
                    ensure!(
                        field_ref.field_index < input_size,
                        FieldRefOutOfRangeSnafu {
                            ref_index: field_ref.field_index,
                            input_size
                        }
                    );

                    ensure!(
                        field_ref.output_type == input_logical_types[field_ref.field_index],
                        FieldRefTypeMismatchSnafu {
                            ref_index: field_ref.field_index,
                            ref_type: field_ref.output_type.clone(),
                            input_type: input_logical_types[field_ref.field_index].clone()
                        }
                    );

                    output_types.push(field_ref.output_type.clone());
                    group_by_keys_indexes.push(field_ref.field_index);

                    Ok(acc + field_ref.output_type.physical_type().size())
                })?;

        // Make sure the group by keys can be serde into the serde key
        ensure!(
            total_size <= S::SerdeKey::PHYSICAL_SIZE,
            SerdeKeySmallerThanGroupByKeysSnafu {
                serde_key: std::any::type_name::<S::SerdeKey>(),
                group_by_keys: group_by_keys
                    .iter()
                    .map(|field_ref| field_ref.output_type.clone())
                    .collect::<Vec<_>>()
            }
        );

        ensure!(!agg_funcs.is_empty(), EmptyAggregationFuncsSnafu {});

        agg_funcs.iter().try_for_each(|agg_func| {
            agg_func.payloads.iter().try_for_each(|field_ref| {
                ensure!(
                    field_ref.field_index < input_size,
                    FieldRefOutOfRangeSnafu {
                        ref_index: field_ref.field_index,
                        input_size
                    }
                );

                ensure!(
                    field_ref.output_type == input_logical_types[field_ref.field_index],
                    FieldRefTypeMismatchSnafu {
                        ref_index: field_ref.field_index,
                        ref_type: field_ref.output_type.clone(),
                        input_type: input_logical_types[field_ref.field_index].clone()
                    }
                );

                payloads_indexes.push(field_ref.field_index);
                Ok::<_, HashAggregateError>(())
            })?;

            output_types.push(agg_func.func.return_type());
            Ok(())
        })?;

        let agg_func_list = AggregationFunctionList::new(
            agg_funcs
                .into_iter()
                .map(|agg_func| agg_func.func)
                .collect(),
        );

        Ok(Self {
            output_types,
            children: [input],
            group_by_keys_indexes,
            agg_func_list,
            payloads_indexes,
            global_state: GlobalState {
                collected_swiss_tables: Mutex::new(Vec::new()),
                total_elements_count: AtomicUsize::new(0),
                metrics: HashAggregateMetics::default(),
            },
        })
    }

    /// Combine the thread local swiss tables into the global state
    fn combine_thread_local_tables(&self, thread_local_tables: ThreadLocalTables<S::SerdeKey>) {
        self.global_state
            .collected_swiss_tables
            .lock()
            .push(thread_local_tables)
    }
}

impl<S: Serde> Drop for HashAggregate<S> {
    /// Drop the elements that have not been consumed
    fn drop(&mut self) {
        let mut state_ptrs = Vec::new();
        let mut tables = self.global_state.collected_swiss_tables.lock();
        while let Some(mut thread_local_tables) = tables.pop() {
            // May drop the table in the sink phase, we need to pop all tables
            while let Some(table) = thread_local_tables.tables.pop() {
                state_ptrs.clear();
                state_ptrs.extend(table.into_iter().map(|element| element.agg_states_ptr));
                // SAFETY: the elements are allocated in the `thread_local_tables.arena`
                unsafe { self.agg_func_list.drop_states(&state_ptrs) }
            }
        }
    }
}

/// Global source state for [`HashAggregate`]
#[derive(Debug)]
pub struct HashAggregateGlobalSourceState {
    /// Until now, how many elements have been read
    read_count: AtomicUsize,
}

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
///
/// FIXME: Can we move partial of the swiss table into the local state?
#[allow(missing_debug_implementations)]
pub struct HashAggregateLocalSourceState<S: Serde> {
    table_into_iter: IntoIter<Element<S::SerdeKey>>,
    read_count: *const AtomicUsize,
    /// Arena the table allocated in
    _arena: Arena,

    /// Used for drop. The lifetime of the agg_func_list is the lifetime of the
    /// physical operator that create this local sink state, which guarantees out-lives
    /// this state
    agg_func_list: *const AggregationFunctionList,

    // Auxiliary memory used across the data blocks
    serde_keys: Vec<S::SerdeKey>,
    state_ptrs: Vec<AggregationStatesPtr>,

    metrics: LocalSourceMetrics,
}

#[derive(Debug, Default)]
struct LocalSourceMetrics {
    /// Time spent in deserialization
    deserialize_time: Duration,
    /// Time spent in take the states into arrays
    take_states_time: Duration,
}

/// Drop the elements that have not been consumed. It happens in many cases, for example
/// `limit` may only take the k elements from the table or query is cancelled, etc.
impl<S: Serde> Drop for HashAggregateLocalSourceState<S> {
    fn drop(&mut self) {
        self.state_ptrs.clear();
        self.state_ptrs
            .extend((&mut self.table_into_iter).map(|element| element.agg_states_ptr));

        // SAFETY:
        // - the elements are allocated in the `self._arena`
        // - hash aggregate operator outlive self
        unsafe { (*self.agg_func_list).drop_states(&self.state_ptrs) }
    }
}

impl<S: Serde> StateStringify for HashAggregateLocalSourceState<S> {
    fn name(&self) -> &'static str {
        "HashAggregateLocalSourceState"
    }

    fn debug(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "HashAggregateLocalSourceState<{}>",
            std::any::type_name::<S>()
        )
    }
}

impl<S: Serde + 'static> LocalSourceState for HashAggregateLocalSourceState<S> {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_mut_any(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

/// Global sink state for [`HashAggregate`]
///
/// TBD: Advantage of radix partitioning?
#[derive(Debug)]
pub struct HashAggregateGlobalSinkState {
    partitioning: Option<ModuloPartitioning>,

    /// Until now, does any thread has partitioned the swiss table? The `finish_local_sink`
    /// method should check it. If it is true, the method should partition the swiss
    /// table before insert into `self.collected_swiss_tables`. The `finalize_sink` method should also
    /// check it. If it is true, the method should partition the swiss table that
    /// is not partitioned. All of the memory ordering used to access this variable
    /// is `Relaxed`!
    any_partitioned: AtomicBool,
}

impl StateStringify for HashAggregateGlobalSinkState {
    /// FIXME: Include generic info
    fn name(&self) -> &'static str {
        "HashAggregateGlobalSinkState"
    }

    fn debug(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl GlobalSinkState for HashAggregateGlobalSinkState {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Local sink state for [`HashAggregate`]
#[derive(Debug)]
pub struct HashAggregateLocalSinkState<S: Serde> {
    hash_table: HashTable<S>,

    /// Used for drop. The lifetime of the agg_func_list is the lifetime of the
    /// physical operator that create this local sink state, which guarantees out-lives
    /// this state
    agg_func_list: *const AggregationFunctionList,

    /// Note: The 'static lifetime is fake.....
    /// The group_by_keys is used in the `write_data` method, it reference the input and will be cleared
    /// immediately after the `write_data` is finished. The LocalState requires a 'static
    /// lifetime, in this use case, it will only holds a reference in the execution of `write_data` but it
    /// needs to outlive the `write_data` function. Therefore, I use 'static here to cheat the
    /// compiler.
    group_by_keys: Vec<&'static ArrayImpl>,
    /// Note: The 'static lifetime is fake.....
    payloads: Vec<&'static ArrayImpl>,

    /// Metric used to record the time spent in write data
    write_data_time: Duration,
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

impl<S: Serde> HashAggregateLocalSinkState<S> {
    /// Partition self then combine self into the global state
    #[inline]
    fn partition_then_combine_into<P: Partitioning>(
        &mut self,
        op: &HashAggregate<S>,
        partitioning: &P,
    ) {
        let (swiss_table, arena) = self.hash_table.finalize();
        let mut local_partitioned_tables = Vec::new();
        partition::<S, _>(swiss_table, &mut local_partitioned_tables, partitioning);
        op.combine_thread_local_tables(ThreadLocalTables {
            tables: local_partitioned_tables,
            arena,
        });
    }

    #[inline]
    fn combine_into(&mut self, op: &HashAggregate<S>) {
        let (swiss_table, arena) = self.hash_table.finalize();
        op.combine_thread_local_tables(ThreadLocalTables {
            tables: vec![swiss_table],
            arena,
        });
    }
}

/// Drop the local state. We need to implement it because the query may cancelled in the
/// sink phase.
impl<S: Serde> Drop for HashAggregateLocalSinkState<S> {
    fn drop(&mut self) {
        let (swiss_table, _arena) = self.hash_table.finalize();
        if !swiss_table.is_empty() {
            let state_ptrs = swiss_table
                .into_iter()
                .map(|element| element.agg_states_ptr)
                .collect::<Vec<_>>();

            // SAFETY:
            // - the elements are allocated in the `_arena`
            // - hash aggregate operator outlive self
            unsafe { (*self.agg_func_list).drop_states(&state_ptrs) }
        }
    }
}

impl<S: Serde> Stringify for HashAggregate<S> {
    fn name(&self) -> &'static str {
        "HashAggregate"
    }

    /// FIXME: debug
    fn debug(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }

    fn display(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "HashAggregate: group_by_keys={:?}, aggregation_functions={}",
            self.group_by_keys_indexes, self.agg_func_list
        )
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

    fn metrics(&self) -> MetricsSet {
        self.global_state.metrics.metrics_set()
    }

    impl_regular_for_non_regular!();

    // Source

    fn is_source(&self) -> bool {
        true
    }

    fn source_parallelism_degree(
        &self,
        _global_state: &dyn GlobalSourceState,
    ) -> ParallelismDegree {
        let combined_tables = self
            .global_state
            .collected_swiss_tables
            .try_lock()
            .expect("HashAggregate can not take the lock of the GlobalState in `source_parallelism_degree`, some threads holding the lock");
        unsafe { ParallelismDegree::new_unchecked(combined_tables.len() as _) }
    }

    fn read_data(
        &self,
        output: &mut DataBlock,
        global_state: &dyn GlobalSourceState,
        local_state: &mut dyn LocalSourceState,
    ) -> OperatorResult<SourceExecStatus> {
        self.read_data_in_parallel(output, global_state, local_state)
    }

    fn global_source_state(&self, _client_ctx: &ClientContext) -> Arc<dyn GlobalSourceState> {
        Arc::new(HashAggregateGlobalSourceState {
            read_count: AtomicUsize::new(0),
        })
    }

    fn local_source_state(
        &self,
        global_state: &dyn GlobalSourceState,
    ) -> Box<dyn LocalSourceState> {
        let global_state = self.downcast_ref_global_source_state(global_state);

        let combined_table = self.global_state.collected_swiss_tables.lock().pop();
        let (table_into_iter, arena) = if let Some(mut combined_table) = combined_table {
            (
                combined_table
                    .tables
                    .pop()
                    .expect("The combined table is guaranteed to be non-empty")
                    .into_iter(),
                combined_table.arena,
            )
        } else {
            (SwissTable::new().into_iter(), Arena::new())
        };

        Box::new(HashAggregateLocalSourceState::<S> {
            table_into_iter,
            agg_func_list: &self.agg_func_list as _,
            read_count: &global_state.read_count,
            _arena: arena,
            serde_keys: Vec::new(),
            state_ptrs: Vec::new(),
            metrics: LocalSourceMetrics::default(),
        })
    }

    fn merge_local_source_metrics(&self, local_state: &dyn LocalSourceState) {
        let local_state = self.downcast_ref_local_source_state(local_state);
        tracing::debug!(
            "Deserialize time: `{:?}`, take_states_time: `{:?}`",
            local_state.metrics.deserialize_time,
            local_state.metrics.take_states_time
        );

        self.global_state
            .metrics
            .combine_source_metrics(&local_state.metrics);
    }

    fn progress(&self, global_state: &dyn GlobalSourceState) -> f64 {
        let global_state = self.downcast_ref_global_source_state(global_state);
        let read_count = global_state.read_count.load(Relaxed) as f64;
        let total_count = self.global_state.total_elements_count.load(Relaxed) as f64;
        read_count / total_count
    }

    // Sink

    fn is_sink(&self) -> bool {
        true
    }

    fn write_data(
        &self,
        input: &DataBlock,
        _global_state: &dyn GlobalSinkState,
        local_state: &mut dyn LocalSinkState,
    ) -> OperatorResult<SinkExecStatus> {
        debug_assert_eq!(input.num_arrays(), self.children[0].output_types().len());
        debug_assert!(
            input
                .arrays()
                .iter()
                .map(|array| array.logical_type())
                .zip(self.children[0].output_types())
                .all(|(input, output)| input == output)
        );

        let local_state = downcast_mut_local_state!(
            self,
            local_state,
            <Self as SinkOperatorExt>::LocalSinkState,
            SINK
        );

        let _guard = ScopedTimerGuard::new(&mut local_state.write_data_time);

        self.group_by_keys_indexes.iter().for_each(|&index| unsafe {
            let group_by_key = input
                    .get_array(index)
                    .expect("Index in group_by_keys is guaranteed to be valid, checked by the `try_new` constructor");

            local_state.group_by_keys.push(&*( group_by_key as *const _));
        });

        self.payloads_indexes.iter().for_each(|&index| unsafe{
            let payload = input
                    .get_array(index)
                    .expect("Index in payloads_indexes is guaranteed to be valid, checked by the `try_new` constructor");

            local_state.payloads.push(&*(payload as *const _))
        });

        // SAFETY:
        // - `group_by_keys` and `payloads` are referenced from input, the `DataBlock` guarantees
        // all of the arrays have same length
        //
        // - `group_by_keys` are guaranteed to fit into `S::SerdeKey`, checked by `try_new` constructor
        //
        // - `payloads` are computed from `self.payloads_indexes`, it matches `self.agg_func_list`
        unsafe {
            local_state.hash_table.add_block(
                &local_state.group_by_keys,
                &local_state.payloads,
                input.len(),
                &self.agg_func_list,
            )
        }
        .boxed()
        .context(WriteDataSnafu)?;

        // Important, otherwise use after free happens. Never deleting following code!!!
        local_state.group_by_keys.clear();
        local_state.payloads.clear();

        Ok(SinkExecStatus::NeedMoreInput)
    }

    fn combine_sink(
        &self,
        global_state: &dyn GlobalSinkState,
        local_state: &mut dyn LocalSinkState,
    ) -> OperatorResult<()> {
        let global_state = self.downcast_ref_global_sink_state(global_state);
        let local_state = downcast_mut_local_state!(
            self,
            local_state,
            <Self as SinkOperatorExt>::LocalSinkState,
            SINK
        );

        tracing::debug!(
            "HashAggregator: aggregate {} rows local data into {} entries in  `{:?}`. Serialize time: `{:?}`. Hash time: `{:?}`, Probing time: `{:?}`. Update states time: `{:?}`",
            local_state.hash_table.metrics.num_rows,
            local_state.hash_table.len(),
            local_state.write_data_time,
            local_state.hash_table.metrics.serialize_time,
            local_state.hash_table.metrics.hash_time,
            local_state.hash_table.metrics.probing_time,
            local_state.hash_table.metrics.update_states_time,
        );
        self.global_state
            .metrics
            .combine_sink_metrics(&local_state.hash_table.metrics);

        let now = crate::common::profiler::Instant::now();
        match &global_state.partitioning {
            Some(partitioning) if local_state.hash_table.len() > PARTITION_THRESHOLD => {
                // We need to partition the hash table
                // Info some local threads have many keys, we need to partition to support multi-threads
                global_state.any_partitioned.store(true, Relaxed);
                local_state.partition_then_combine_into(self, partitioning);
            }
            Some(partitioning) if global_state.any_partitioned.load(Relaxed) => {
                local_state.partition_then_combine_into(self, partitioning);
            }
            _ => local_state.combine_into(self),
        }
        tracing::debug!(
            "HashAggregator: combine local sink state into global sink state takes `{:?}`",
            now.elapsed()
        );

        Ok(())
    }

    unsafe fn finalize_sink(&self, global_state: &dyn GlobalSinkState) -> OperatorResult<()> {
        let global_state = self.downcast_ref_global_sink_state(global_state);

        let mut collected_swiss_tables =
            self.global_state.collected_swiss_tables.try_lock().expect(
                "HashAggregate can not take the lock of the GlobalSinkState in `finalize_sink`, 
                    some threads holding the lock. It should never happens, query executor should 
                    guarantees `finalize_sink` is called exactly once and after all of the sink 
                    process have finished",
            );

        match &global_state.partitioning {
            Some(partitioning) => {
                if global_state.any_partitioned.load(Relaxed) {
                    // Partitioning the tables that have not partitioned yet. Unlikely to happen

                    collected_swiss_tables
                        .par_iter_mut()
                        .for_each(|thread_local_tables| {
                            if thread_local_tables.tables.len() == 1 {
                                // SAFETY: the length is 1
                                let swiss_table =
                                    unsafe { thread_local_tables.tables.pop().unwrap_unchecked() };
                                partition::<S, _>(
                                    swiss_table,
                                    &mut thread_local_tables.tables,
                                    partitioning,
                                );
                            }
                        });

                    combine_partitioned_tables(
                        collected_swiss_tables.deref_mut(),
                        &self.agg_func_list,
                        partitioning.partition_count,
                    )?;
                } else {
                    // All of the tables have not partitioned. Combine them into the first table
                    combine_not_partitioned_tables(
                        collected_swiss_tables.deref_mut(),
                        &self.agg_func_list,
                    )?;
                }
            }
            None => {
                // Single thread
                debug_assert_eq!(collected_swiss_tables.len(), 1);
            }
        }

        // Write the total_elements_count
        let total_element_count = collected_swiss_tables
            .iter()
            .map(|thread_local_table| {
                debug_assert_eq!(thread_local_table.tables.len(), 1);
                thread_local_table.tables[0].len()
            })
            .sum::<usize>();

        self.global_state
            .total_elements_count
            .store(total_element_count, Relaxed);
        Ok(())
    }

    fn global_sink_state(&self, client_ctx: &ClientContext) -> Arc<dyn GlobalSinkState> {
        let parallelism = client_ctx.exec_args.parallelism;
        Arc::new(HashAggregateGlobalSinkState {
            any_partitioned: AtomicBool::new(false),
            partitioning: if parallelism > ParallelismDegree::MIN {
                Some(ModuloPartitioning::new(parallelism))
            } else {
                None
            },
        })
    }

    fn local_sink_state(&self, _global_state: &dyn GlobalSinkState) -> Box<dyn LocalSinkState> {
        Box::new(HashAggregateLocalSinkState::<S> {
            hash_table: HashTable::new(),
            agg_func_list: &self.agg_func_list as _,
            group_by_keys: Vec::new(),
            payloads: Vec::new(),
            write_data_time: Duration::default(),
        })
    }
}

impl<S: Serde> SourceOperatorExt for HashAggregate<S> {
    type GlobalSourceState = HashAggregateGlobalSourceState;
    type LocalSourceState = HashAggregateLocalSourceState<S>;

    /// FIXME: The morsel is too large, we view the whole hash table as morsel, will cause
    /// the unbalance problem
    #[inline]
    fn next_morsel(
        &self,
        _global_state: &Self::GlobalSourceState,
        local_state: &mut Self::LocalSourceState,
    ) -> bool {
        if let Some(mut combined_table) = self.global_state.collected_swiss_tables.lock().pop() {
            local_state._arena = combined_table.arena;
            local_state.table_into_iter = combined_table
                .tables
                .pop()
                .expect("The combined table is guaranteed to be non-empty")
                .into_iter();
            true
        } else {
            false
        }
    }

    fn read_local_data(
        &self,
        output: &mut DataBlock,
        local_state: &mut Self::LocalSourceState,
    ) -> OperatorResult<SourceExecStatus> {
        let Some(element) = local_state.table_into_iter.next() else {
            return Ok(SourceExecStatus::Finished);
        };
        local_state.state_ptrs.push(element.agg_states_ptr);
        local_state.serde_keys.push(element.serde_key);

        for _ in 1..STANDARD_VECTOR_SIZE {
            if let Some(element) = local_state.table_into_iter.next() {
                local_state.state_ptrs.push(element.agg_states_ptr);
                local_state.serde_keys.push(element.serde_key);
            } else {
                break;
            }
        }

        // TBD: will synchronize frequently become bottleneck?
        // SAFETY: Global state outlive local state
        unsafe {
            (*local_state.read_count).fetch_add(local_state.state_ptrs.len(), Relaxed);
        }

        let guard = output.mutate_arrays(local_state.state_ptrs.len());
        if let Err(err) = guard.mutate(|arrays| {
            let (group_by_arrays, agg_output_arrays) =
                arrays.split_at_mut(self.group_by_keys_indexes.len());

            // SAFETY: Constructor guarantees the safety
            unsafe {
                let _guard = ScopedTimerGuard::new(&mut local_state.metrics.deserialize_time);
                S::deserialize(group_by_arrays, &local_state.serde_keys);
            }

            // SAFETY: Constructor guarantees the safety
            unsafe {
                let _guard = ScopedTimerGuard::new(&mut local_state.metrics.take_states_time);
                self.agg_func_list
                    .take_states(&local_state.state_ptrs, agg_output_arrays)
                    .boxed()
                    .context(ReadDataSnafu)
            }
        }) {
            match err {
                MutateArrayError::Inner { source } => {
                    return Err(source);
                }
                MutateArrayError::Length { inner } => {
                    panic!(
                        "DataBlock read from HashAggregate's local source state must have same length. Error: `{}`",
                        inner
                    )
                }
            }
        }

        local_state.state_ptrs.clear();
        local_state.serde_keys.clear();

        Ok(SourceExecStatus::HaveMoreOutput)
    }
}

impl<S: Serde> SinkOperatorExt for HashAggregate<S> {
    type GlobalSinkState = HashAggregateGlobalSinkState;
    type LocalSinkState = HashAggregateLocalSinkState<S>;
}

/// FIXME: Reuse the arena and hash table. The following method may fail for multi-threading
///
/// Yeah, we can do it in this way: reuse the arena in the thread_index as combined
/// arena and create new arena if `parallelism > partition`. Combine all of the hash tables
/// into the hash table produced by the first thread! This method sounds queer: elements
/// in one hash table may located in different arena! But we can proof it is correct 😊
fn combine_partitioned_tables<K: SerdeKey>(
    collected_swiss_tables: &mut Vec<ThreadLocalTables<K>>,
    agg_func_list: &AggregationFunctionList,
    partition_count: ParallelismDegree,
) -> OperatorResult<()> {
    let partition_count = partition_count.get() as usize;

    // Check the input is valid, all of them have same partition count
    debug_assert!(collected_swiss_tables.len() >= 2);
    debug_assert!(
        collected_swiss_tables
            .iter()
            .all(|thread_local_table| thread_local_table.tables.len() == partition_count)
    );

    // Transpose the two dimension vector from `parallelism * partition` to `partition * parallelism`
    // Note that the arena still in the collected_swiss_tables
    let transposed = (0..partition_count)
        .map(|partition_index| {
            collected_swiss_tables
                .iter_mut()
                .map(|thread_local_tables| {
                    std::mem::take(&mut thread_local_tables.tables[partition_index])
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    let combined_tables = transposed
        .into_par_iter()
        .map(|partial_tables| {
            let mut combined_table = SwissTable::new();
            let combined_arena = Arena::new();
            let mut combined_state_ptrs = Vec::new();
            let mut partial_state_ptrs = Vec::new();
            // SAFETY: all of the partial tables' arena are located in the collected_swiss_tables,
            // which outlive this function
            partial_tables
                .into_iter()
                .try_for_each(|partial_table| unsafe {
                    combine_swiss_table(
                        partial_table,
                        &mut partial_state_ptrs,
                        &mut combined_table,
                        &combined_arena,
                        &mut combined_state_ptrs,
                        agg_func_list,
                    )
                })
                .map(|_| ThreadLocalTables {
                    tables: vec![combined_table],
                    arena: combined_arena,
                })
        })
        .collect::<OperatorResult<Vec<_>>>()?;

    *collected_swiss_tables = combined_tables;

    Ok(())
}

fn combine_not_partitioned_tables<K: SerdeKey>(
    collected_swiss_tables: &mut Vec<ThreadLocalTables<K>>,
    agg_func_list: &AggregationFunctionList,
) -> OperatorResult<()> {
    debug_assert!(!collected_swiss_tables.is_empty());
    debug_assert!(
        collected_swiss_tables
            .iter()
            .all(|thread_local_tables| thread_local_tables.tables.len() == 1)
    );

    let mut combined_state_ptrs = Vec::new();
    let mut partial_state_ptrs = Vec::new();

    unsafe {
        while collected_swiss_tables.len() >= 2 {
            // SAFETY: length larger than or equal to 2
            let mut partial_table = collected_swiss_tables.pop().unwrap_unchecked();
            // SAFETY: We have already checked all of the ThreadLocalTables have 1 swiss table
            let partial_table = partial_table.tables.pop().unwrap_unchecked();
            // SAFETY: after the first pop, the length is larger than or equal to 1
            let combined_table = collected_swiss_tables.first_mut().unwrap_unchecked();
            // SAFETY: the arena of the partial_table lives until end of the block, has same life time with the above first pop
            combine_swiss_table(
                partial_table,
                &mut partial_state_ptrs,
                combined_table.tables.first_mut().unwrap_unchecked(),
                &combined_table.arena,
                &mut combined_state_ptrs,
                agg_func_list,
            )?;
        }
    }
    Ok(())
}

/// Combine the partial table into combined table
///
/// # Safety
///
/// Arena of the partial table should outlive this function
unsafe fn combine_swiss_table<'a, K: SerdeKey>(
    mut partial_table: SwissTable<Element<K>>,
    partial_state_ptrs: &mut Vec<AggregationStatesPtr>,
    combined_table: &'a mut SwissTable<Element<K>>,
    combined_arena: &'a Arena,
    combined_state_ptrs: &mut Vec<AggregationStatesPtr>,
    agg_func_list: &AggregationFunctionList,
) -> OperatorResult<()> {
    unsafe {
        combined_state_ptrs.resize(partial_table.len(), AggregationStatesPtr::dangling());
        partial_state_ptrs.resize(partial_table.len(), AggregationStatesPtr::dangling());

        partial_table
            .iter_mut()
            .zip(partial_state_ptrs.iter_mut())
            .zip(combined_state_ptrs.iter_mut())
            .for_each(|((element, partial_ptr), combined_ptr)| {
                *partial_ptr = element.agg_states_ptr;

                probing_swiss_table(
                    combined_table,
                    combined_arena,
                    &mut element.serde_key,
                    element.hash_value,
                    combined_ptr,
                    agg_func_list,
                )
            });

        // SAFETY: the partial_state_ptr is guaranteed by the safety of the function
        // The combined_state_ptrs points to the arena that has same lifetime with the combined_table
        // Both of them have length: `partial_table.len()`
        agg_func_list
            .combine_states(partial_state_ptrs, combined_state_ptrs)
            .boxed()
            .context(FinalizeSinkSnafu)
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
    fn partition_count(&self) -> ParallelismDegree;

    /// Partition the [`HashValue`] into partition index. Note that the returned
    /// partition index should smaller than partition count
    fn partition(&self, hash_value: HashValue) -> usize;
}

/// Partitioning based on modulo operation
#[derive(Debug, Clone)]
pub struct ModuloPartitioning {
    partition_count: ParallelismDegree,
    modulo: StrengthReducedU64,
}

impl Partitioning for ModuloPartitioning {
    fn new(partition_count: ParallelismDegree) -> Self {
        Self {
            partition_count,
            modulo: StrengthReducedU64::new(partition_count.get() as _),
        }
    }

    #[inline]
    fn partition_count(&self) -> ParallelismDegree {
        self.partition_count
    }

    #[inline]
    fn partition(&self, hash_value: HashValue) -> usize {
        (hash_value % self.modulo) as usize
    }
}

/// Partition the hash table into multiple hash tables
fn partition<S: Serde, P: Partitioning>(
    swiss_table: SwissTable<Element<S::SerdeKey>>,
    partitioned: &mut Vec<SwissTable<Element<S::SerdeKey>>>,
    partitioning: &P,
) {
    debug_assert!(partitioned.is_empty());

    partitioned.resize_with(partitioning.partition_count().get() as _, || {
        SwissTable::with_capacity(STANDARD_VECTOR_SIZE)
    });

    swiss_table.into_iter().for_each(|element| {
        let hash_value = element.hash_value;
        let partition_index = partitioning.partition(hash_value);
        debug_assert!(partition_index < partitioning.partition_count().get() as _);

        // SAFETY: We already allocate the hash table in the tables and the partition index
        // is guaranteed smaller than the partition count
        let hash_table = unsafe { partitioned.get_unchecked_mut(partition_index) };
        hash_table.insert_unique(hash_value, element, |element| element.hash_value);
    });
}

#[cfg(test)]
mod tests {
    use self::serde::FixedSizedSerdeKeySerializer;

    use super::serde::FixedSizedSerdeKey;

    use super::*;
    use crate::exec::physical_expr::function::aggregate::AggregationFunction;
    use crate::exec::physical_expr::function::aggregate::min_max::Min;
    use crate::exec::physical_expr::function::aggregate::{count::CountStar, sum::Sum};
    use crate::exec::physical_operator::table_scan::empty_table_scan::EmptyTableScan;
    use data_block::array::{Float32Array, Int32Array, Int64Array};

    fn mock_agg_funcs_list() -> AggregationFunctionList {
        let agg_funcs: Vec<Arc<dyn AggregationFunction>> = vec![
            Arc::new(CountStar::new()),
            Arc::new(Min::<Int64Array>::try_new(LogicalType::BigInt).unwrap()),
        ];

        AggregationFunctionList::new(agg_funcs)
    }

    fn mock_swiss_tables(
        agg_func_list: &AggregationFunctionList,
        duplicate_count: usize,
    ) -> Vec<ThreadLocalTables<FixedSizedSerdeKey<u64>>> {
        let mut arenas_array = [Arena::new(), Arena::new()];
        let mut mocked = vec![Vec::new(), Vec::new()];
        (0..duplicate_count).for_each(|_| {
            let khs = (0..6)
                .map(|i| FixedSizedSerdeKey::new(i, 0))
                .collect::<Vec<_>>();

            let ptrs_array = (0..2)
                .map(|i| {
                    (0..4)
                        .map(|_| agg_func_list.alloc_states_in_arena(&arenas_array[i]))
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();

            let indexes = [[0, 1, 4, 5], [1, 2, 3, 4]];

            let elements = indexes
                .iter()
                .enumerate()
                .map(|(i, indexes)| {
                    indexes
                        .iter()
                        .enumerate()
                        .map(|(j, &index)| Element {
                            serde_key: khs[index].clone(),
                            hash_value: crate::common::utils::hash::BUILD_HASHER_DEFAULT
                                .hash_one(&khs[index]),
                            agg_states_ptr: ptrs_array[i][j],
                        })
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            let payloads0 = &[&ArrayImpl::Int64(Int64Array::from_values_iter([
                -1, 4, 6, 9,
            ]))];

            let payloads1 = &[&ArrayImpl::Int64(Int64Array::from_values_iter([
                7, -9, -3, -2,
            ]))];

            unsafe {
                agg_func_list
                    .update_states(payloads0, &ptrs_array[0])
                    .unwrap()
            };
            unsafe {
                agg_func_list
                    .update_states(payloads1, &ptrs_array[1])
                    .unwrap()
            };
            elements
                .into_iter()
                .enumerate()
                .for_each(|(index, elements)| {
                    let mut table = SwissTable::with_capacity(16);
                    elements.into_iter().for_each(|element| {
                        table.insert_unique(element.hash_value, element, |element| {
                            element.hash_value
                        });
                    });
                    mocked[index].push(table)
                });
        });

        mocked
            .into_iter()
            .enumerate()
            .map(|(index, tables)| ThreadLocalTables {
                tables,
                arena: std::mem::take(&mut arenas_array[index]),
            })
            .collect()
    }

    fn assert_combined_table(
        combined: ThreadLocalTables<FixedSizedSerdeKey<u64>>,
        agg_func_list: &AggregationFunctionList,
    ) {
        assert_eq!(combined.tables[0].len(), 6);

        let mut output =
            DataBlock::with_logical_types(vec![LogicalType::UnsignedBigInt, LogicalType::BigInt]);

        let combined_state_ptrs = unsafe {
            combined.tables[0]
                .iter()
                .map(|bucket| bucket.agg_states_ptr)
                .collect::<Vec<_>>()
        };

        let guard = output.mutate_arrays(6);

        let mutate_func = |arrays: &mut [ArrayImpl]| unsafe {
            agg_func_list.take_states(&combined_state_ptrs, arrays)
        };
        guard.mutate(mutate_func).unwrap();

        assert_data_block(
            &output,
            ["1,-1,", "2,4,", "1,-9,", "1,-3,", "2,-2,", "1,9,"],
        );
    }

    pub(super) fn assert_data_block(
        data_block: &DataBlock,
        gt_rows: impl IntoIterator<Item = &'static str>,
    ) {
        let mut formatted_rows = std::collections::HashSet::new();
        (0..data_block.len()).for_each(|index| {
            let row = data_block
                .arrays()
                .iter()
                .map(|array| unsafe {
                    array
                        .get_unchecked(index)
                        .map_or_else(|| "Null,".to_string(), |element| format!("{},", element))
                })
                .collect::<String>();
            formatted_rows.insert(row);
        });

        gt_rows.into_iter().for_each(|row| {
            if !formatted_rows.remove(row) {
                panic!("Data block does not contain row: {}", row)
            }
        });

        if !formatted_rows.is_empty() {
            panic!(
                "Data block has rows the ground truth does not have: {:?}",
                formatted_rows
            )
        }
    }

    #[test]
    fn test_combine_not_partitioned_tables() {
        let agg_func_list = mock_agg_funcs_list();
        let mut swiss_tables = mock_swiss_tables(&agg_func_list, 1);
        combine_not_partitioned_tables(&mut swiss_tables, &agg_func_list).unwrap();

        let combined = swiss_tables.pop().unwrap();
        assert_combined_table(combined, &agg_func_list);
    }

    #[test]
    fn test_combine_partitioned_tables() {
        let agg_func_list = mock_agg_funcs_list();
        let mut swiss_tables = mock_swiss_tables(&agg_func_list, 3);
        combine_partitioned_tables(
            &mut swiss_tables,
            &agg_func_list,
            ParallelismDegree::new(3).unwrap(),
        )
        .unwrap();

        swiss_tables
            .into_iter()
            .for_each(|combined| assert_combined_table(combined, &agg_func_list));
    }

    #[test]
    fn test_hash_aggregate() {
        // -1, -0.0, -1
        // -1,  0.0,  2
        //  3,    N, -9
        //  N, -7.7,  7
        //  7,  3.0, -2
        //  N,    N,  N
        //  3,    N, -7
        //  N,    N, -7
        fn mock_data_block() -> DataBlock {
            DataBlock::try_new(
                vec![
                    ArrayImpl::Int32(Int32Array::from_iter([
                        Some(-1),
                        Some(-1),
                        Some(3),
                        None,
                        Some(7),
                        None,
                        Some(3),
                        None,
                    ])),
                    ArrayImpl::Float32(Float32Array::from_iter([
                        Some(-0.0),
                        Some(0.0),
                        None,
                        Some(-7.7),
                        Some(3.0),
                        None,
                        None,
                        None,
                    ])),
                    ArrayImpl::Int64(Int64Array::from_iter([
                        Some(-1),
                        Some(2),
                        Some(-9),
                        Some(7),
                        Some(-2),
                        None,
                        Some(-7),
                        Some(-7),
                    ])),
                ],
                8,
            )
            .unwrap()
        }

        let client_ctx = crate::common::client_context::tests::mock_client_context();

        let input = Arc::new(EmptyTableScan::new(vec![
            LogicalType::Integer,
            LogicalType::Float,
            LogicalType::BigInt,
        ]));
        let group_by_keys = [
            FieldRef::new(0, LogicalType::Integer, "f0".to_string()),
            FieldRef::new(1, LogicalType::Float, "f1".to_string()),
        ];

        let count_payloads = [];
        let sum_payloads = [FieldRef::new(2, LogicalType::BigInt, "f2".to_string())];
        let agg_funcs = vec![
            AggregationFunctionExpr::try_new(&count_payloads, Arc::new(CountStar::new())).unwrap(),
            AggregationFunctionExpr::try_new(
                &sum_payloads,
                Arc::new(Sum::<Int64Array>::try_new(LogicalType::BigInt).unwrap()),
            )
            .unwrap(),
        ];

        let agg = HashAggregate::<FixedSizedSerdeKeySerializer<u64>>::try_new(
            input,
            &group_by_keys,
            agg_funcs,
        )
        .unwrap();

        // Sink phase
        {
            let global_sink_state = agg.global_sink_state(&client_ctx);
            let func = || {
                let block = mock_data_block();
                let mut local_sink_state = agg.local_sink_state(&*global_sink_state);
                agg.write_data(&block, &*global_sink_state, &mut *local_sink_state)
                    .unwrap();

                agg.combine_sink(&*global_sink_state, &mut *local_sink_state)
                    .unwrap();
            };

            std::thread::scope(|s| {
                s.spawn(func);
                s.spawn(func);
            });

            unsafe {
                agg.finalize_sink(&*global_sink_state).unwrap();
            }
        }

        // Source phase
        {
            let mut output = DataBlock::with_logical_types(agg.output_types().to_owned());

            let global_source_state = agg.global_source_state(&client_ctx);
            let mut local_source_state = agg.local_source_state(&*global_source_state);
            let status = agg
                .read_data(&mut output, &*global_source_state, &mut *local_source_state)
                .unwrap();
            assert!(matches!(status, SourceExecStatus::HaveMoreOutput));

            assert_data_block(
                &output,
                [
                    "7,3.0,2,-4,",
                    "Null,Null,4,-14,",
                    "Null,-7.7,2,14,",
                    "-1,0.0,4,2,",
                    "3,Null,4,-32,",
                ],
            );

            let status = agg
                .read_data(&mut output, &*global_source_state, &mut *local_source_state)
                .unwrap();
            assert!(matches!(status, SourceExecStatus::Finished));
        }
    }
}
