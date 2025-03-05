//! HashTable for hash aggregation

use super::serde::{Serde, SerdeKey};
use crate::common::types::HashValue;
use crate::exec::physical_expr::function::aggregate::{
    AggregationFunctionList, AggregationStatesPtr, Result as AggregationResult,
};

use crate::common::profiler::ScopedTimerGuard;
use crate::exec::physical_operator::aggregate::Arena;
use data_block::array::ArrayImpl;
use hashbrown::raw::RawTable as SwissTable;
use std::time::Duration;

/// HashTable for storing the group by keys and aggregation states. It is not
/// self-contained and require the [`AggregationFunctionList`] to interpret the
/// [`AggregationStatesPtr`]
///
/// # TBD
///
/// Should we partition the hash table if it is too large? The biggest problem
/// is resize. Partitioned hash table can decrease the cost of the resize. However,
/// partition is also not free. In my view, we need it. Almost all of the database
/// implement it. I think it works better in the high cardinality scenario.
///
/// # Generics
///
/// - `S`: Type for serde the `group_by_keys`
pub struct HashTable<S: Serde> {
    swiss_table: SwissTable<Element<S::SerdeKey>>,
    /// Arena used to store the `AggregationStates`. So the arena has same lifetime
    /// with `swiss_tables`, it ensures that as long as the struct exist, we can
    /// access the [`AggregationStatesPtr`] stored in the swiss_table safely.
    ///
    /// TBD: Is it chicken-ribs? Some states are pointer to heap, which means that
    /// the state pointer is in arena however the state is not in arena 😂!
    arena: Arena,

    /// Auxiliary memory allocation reused across data blocks.
    keys: Vec<S::SerdeKey>,
    /// Auxiliary memory allocation reused across data blocks.
    hashes: Vec<HashValue>,
    /// Auxiliary memory allocation reused across data blocks.
    state_ptrs: Vec<AggregationStatesPtr>,

    /// Metrics
    pub metrics: HashTableMetrics,
}

/// Metrics of the hash table
#[derive(Debug, Default)]
pub struct HashTableMetrics {
    /// The number of rows written to the hash table
    pub num_rows: u64,
    /// Time spent in serialization
    pub serialize_time: Duration,
    /// Time spent in hash the serde key
    pub hash_time: Duration,
    /// Time spent in probing the hash table
    pub probing_time: Duration,
    /// Time spent in update the aggregation states
    pub update_states_time: Duration,
}

impl<S: Serde> std::fmt::Debug for HashTable<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "HashTable {{ len: {}, data: ", self.swiss_table.len())?;
        f.debug_list()
            .entries(unsafe { self.swiss_table.iter().map(|bucket| bucket.as_ref()) })
            .finish()?;
        write!(f, " }}")
    }
}

/// Element of the hash table
#[derive(Debug, Clone)]
pub struct Element<K: SerdeKey> {
    /// Serde key
    pub(super) serde_key: K,
    /// Hash value of this element. It is used to avoid rehashing the keys
    /// when we need to resize the hash table. For fixed size keys, it will
    /// be `()` such that Self has small footprint and increasing the cache
    /// locality when probing
    pub(super) hash_value: u64,
    /// Pointer to the combined `AggregationStates`, the pointer is allocated
    /// in the arena
    pub(super) agg_states_ptr: AggregationStatesPtr,
}

unsafe impl<K: Send + SerdeKey> Send for Element<K> {}
unsafe impl<K: Send + Sync + SerdeKey> Sync for Element<K> {}

impl<S: Serde> HashTable<S> {
    /// Create a new hash table
    #[inline]
    pub fn new() -> Self {
        Self {
            swiss_table: SwissTable::new(),
            arena: Arena::new(),
            keys: Vec::new(),
            hashes: Vec::new(),
            state_ptrs: Vec::new(),
            metrics: HashTableMetrics::default(),
        }
    }

    /// Returns the number of elements in the table
    #[inline]
    pub fn len(&self) -> usize {
        self.swiss_table.len()
    }

    /// Returns true if the hash table is empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Add the block to the hash hash table
    ///
    /// # Safety
    ///
    /// - arrays in the `group_by_keys` and `payloads` should have length `len`
    /// - `group_by_keys` should fit into `S::SerdeKey`
    /// - payloads match `agg_func_list`'s signature
    pub(super) unsafe fn add_block(
        &mut self,
        group_by_keys: &[&ArrayImpl],
        payloads: &[&ArrayImpl],
        len: usize,
        agg_func_list: &AggregationFunctionList,
    ) -> AggregationResult<()> {
        unsafe {
            self.metrics.num_rows += len as u64;

            self.keys.resize(len, S::SerdeKey::default());
            self.hashes.resize(len, HashValue::default());
            self.state_ptrs
                .resize(len, AggregationStatesPtr::dangling());

            // 1. Serialize the group by keys into the serde key in elements
            // SAFETY:
            // - `group_by_keys` fit into `S::SerdeKey` is guaranteed by the safety of the function
            // - elements have same length with the keys is guaranteed by above resize
            {
                let _guard = ScopedTimerGuard::new(&mut self.metrics.serialize_time);
                S::serialize(group_by_keys, &mut self.keys);
            }

            // 2. Hash the serde key
            {
                let _guard = ScopedTimerGuard::new(&mut self.metrics.hash_time);
                self.keys
                    .iter()
                    .zip(&mut self.hashes)
                    .for_each(|(key, hash)| {
                        *hash = crate::common::utils::hash::BUILD_HASHER_DEFAULT.hash_one(key);
                    });
            }

            // 3. Probe the hash table, update the ptr in the elements
            {
                let _guard = ScopedTimerGuard::new(&mut self.metrics.probing_time);
                self.keys
                    .iter_mut()
                    .zip(self.hashes.iter())
                    .zip(self.state_ptrs.iter_mut())
                    .for_each(|((key, &hash_value), state_ptr)| {
                        probing_swiss_table(
                            &mut self.swiss_table,
                            &self.arena,
                            key,
                            hash_value,
                            state_ptr,
                            agg_func_list,
                        )
                    });
            }

            // 4. Update the states based on the payloads
            // SAFETY: ptrs are valid is guaranteed by the probing step.
            // Payloads match the agg_func_list's signature is guaranteed by the safety of
            // this function
            let _guard = ScopedTimerGuard::new(&mut self.metrics.update_states_time);
            agg_func_list.update_states(payloads, &self.state_ptrs)
        }
    }

    /// Finalize(Drop) the hash table, take the underling swiss_table and arena.
    #[inline]
    pub(super) fn finalize(&mut self) -> (SwissTable<Element<S::SerdeKey>>, Arena) {
        (
            std::mem::take(&mut self.swiss_table),
            std::mem::take(&mut self.arena),
        )
    }
}

impl<S: Serde> Default for HashTable<S> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

/// Probing the swiss table, set the state_ptr to the ptr in the swiss table
///
/// METAPHYSICS: Until rust 1.79, inline(always) will guarantee it is inlined. Otherwise
/// when `codegen-units=1`, this function will not inlined!
#[inline(always)]
pub(super) fn probing_swiss_table<K: SerdeKey>(
    swiss_table: &mut SwissTable<Element<K>>,
    arena: &Arena,
    key: &mut K,
    hash_value: u64,
    state_ptr: &mut AggregationStatesPtr,
    agg_func_list: &AggregationFunctionList,
) {
    match swiss_table.get(hash_value, |probe_element| probe_element.serde_key.eq(key)) {
        None => {
            // The group by keys do not in the hash table, we need to create a new
            // aggregation states to store its result
            let agg_states_ptr = agg_func_list.alloc_states_in_arena(arena);
            // Insert it into the raw table
            swiss_table.insert(
                hash_value,
                Element {
                    serde_key: std::mem::take(key),
                    hash_value,
                    agg_states_ptr,
                },
                |element| element.hash_value,
            );
            *state_ptr = agg_states_ptr;
        }
        Some(matched_element) => {
            *state_ptr = matched_element.agg_states_ptr;
        }
    }
}

#[cfg(test)]
mod tests {
    use data_block::array::{Int8Array, Int64Array};
    use data_block::block::DataBlock;
    use data_block::types::LogicalType;
    use snafu::Report;
    use std::sync::Arc;

    use super::*;
    use crate::exec::physical_expr::function::aggregate::count::CountStar;
    use crate::exec::physical_expr::function::aggregate::sum::Sum;
    use crate::exec::physical_expr::function::aggregate::{AggregationError, AggregationFunction};
    use crate::exec::physical_operator::aggregate::hash_aggregate::serde::FixedSizedSerdeKeySerializer;
    use crate::exec::physical_operator::aggregate::hash_aggregate::tests::assert_data_block;

    #[test]
    fn test_add_block() -> Report<AggregationError> {
        Report::capture(|| {
            let mut table = HashTable::<FixedSizedSerdeKeySerializer<u16>>::new();
            let agg_funcs: Vec<Arc<dyn AggregationFunction>> = vec![
                Arc::new(CountStar::new()),
                Arc::new(Sum::<Int64Array>::try_new(LogicalType::BigInt)?),
            ];
            let agg_func_list = AggregationFunctionList::new(agg_funcs);

            let array0 = ArrayImpl::Int8(Int8Array::from_iter([
                Some(-1),
                Some(2),
                Some(-1),
                None,
                None,
            ]));
            let array1 = ArrayImpl::Int8(Int8Array::from_iter([
                Some(-1),
                Some(7),
                Some(-1),
                None,
                None,
            ]));
            let array2 = ArrayImpl::Int64(Int64Array::from_iter([
                Some(-4),
                None,
                Some(-3),
                Some(1),
                Some(5),
            ]));

            let group_by_keys = &[&array0, &array1];
            let payloads = &[&array2];
            unsafe { table.add_block(group_by_keys, payloads, 5, &agg_func_list)? }
            unsafe { table.add_block(group_by_keys, payloads, 5, &agg_func_list)? }
            assert_eq!(table.len(), 3);

            // Check states
            table.state_ptrs.clear();
            unsafe {
                table.state_ptrs.extend(
                    table
                        .swiss_table
                        .iter()
                        .map(|bucket| bucket.as_ref().agg_states_ptr),
                );
            }

            let mut output = DataBlock::with_logical_types(vec![
                LogicalType::UnsignedBigInt,
                LogicalType::BigInt,
            ]);
            let guard = output.mutate_arrays(3);
            let mutate_func = |arrays: &mut [ArrayImpl]| unsafe {
                agg_func_list.take_states(&table.state_ptrs, arrays)
            };
            guard.mutate(mutate_func).unwrap();

            assert_data_block(&output, ["4,-14,", "2,Null,", "4,12,"]);

            Ok(())
        })
    }
}
