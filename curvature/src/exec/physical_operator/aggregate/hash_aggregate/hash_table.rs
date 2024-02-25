//! HashTable for hash aggregation

use super::serde::{Serde, SerdeKey};
use crate::common::types::HashValue;
use crate::common::utils::hash::{fixed_build_hasher_default, BuildHasherDefault};
use crate::exec::physical_expr::function::aggregate::{
    AggregationFunctionList, AggregationStatesPtr, Result as AggregationResult,
};

use crate::exec::physical_operator::aggregate::Arena;
use data_block::array::ArrayImpl;
use hashbrown::raw::RawTable as SwissTable;

/// HashTable for storing the group by keys and aggregation states. It is not
/// self-contained and require the [`AggregationFunctionList`] to interpret the
/// [`AggregationStatesPtr`]
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
    /// Build hasher for building hashes
    build_hasher: BuildHasherDefault,

    /// Auxiliary memory allocation reused across data blocks.
    key_and_hash: Vec<SerdeKeyAndHash<S::SerdeKey>>,
    /// Auxiliary memory allocation reused across data blocks.
    state_ptrs: Vec<AggregationStatesPtr>,
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
pub struct Element<K> {
    /// Serde key and its hash
    pub(super) serde_key_and_hash: SerdeKeyAndHash<K>,
    /// Pointer to the combined `AggregationStates`, the pointer is allocated
    /// in the arena
    pub(super) agg_states_ptr: AggregationStatesPtr,
}

/// Serde key and its hash value
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SerdeKeyAndHash<K> {
    /// The serialized `GroupByKeys`, it maybe u64/u32/u128/Vec<u8>
    pub(super) serde_key: K,
    /// Hash value of this element. It is used to avoid rehashing the keys
    /// when we need to resize the hash table
    pub(super) hash_value: HashValue,
}

unsafe impl<K: Send> Send for Element<K> {}
unsafe impl<K: Send + Sync> Sync for Element<K> {}

impl<S: Serde> HashTable<S> {
    /// Create a new hash table
    #[inline]
    pub fn new() -> Self {
        Self {
            swiss_table: SwissTable::new(),
            arena: Arena::new(),
            build_hasher: fixed_build_hasher_default(),
            key_and_hash: Vec::new(),
            state_ptrs: Vec::new(),
        }
    }

    /// Returns the number of elements in the table
    #[inline]
    pub fn len(&self) -> usize {
        self.swiss_table.len()
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
        // Clear is important, such that we make sure all of the key and hash are default
        // and None element is represented by default
        self.key_and_hash.clear();
        self.key_and_hash.resize_with(len, || SerdeKeyAndHash {
            serde_key: S::SerdeKey::default(),
            hash_value: HashValue::default(),
        });

        // We do not need to clear the `state_ptrs` because we will write to all of them
        // immediately
        self.state_ptrs
            .resize(len, AggregationStatesPtr::dangling());

        // 1. Serialize the group by keys into the serde key in elements
        // SAFETY: `group_by_keys` fit into `S::SerdeKey` is guaranteed by the safety of the function
        // elements have same length with the keys is guaranteed by above resize
        S::serialize(group_by_keys, &mut self.key_and_hash, &self.build_hasher);

        // 2. Probe the hash table, update the ptr in the elements
        self.key_and_hash
            .iter_mut()
            .zip(self.state_ptrs.iter_mut())
            .for_each(|(key_and_hash, state_ptr)| {
                probing_swiss_table(
                    &mut self.swiss_table,
                    &self.arena,
                    key_and_hash,
                    state_ptr,
                    agg_func_list,
                )
            });

        // 3. Update the states based on the payloads
        // SAFETY: ptrs are valid is guaranteed by the probing step.
        // Payloads match the agg_func_list's signature is guaranteed by the safety of
        // this function
        agg_func_list.update_states(payloads, &self.state_ptrs)
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
#[inline]
pub(super) fn probing_swiss_table<K: SerdeKey>(
    swiss_table: &mut SwissTable<Element<K>>,
    arena: &Arena,
    key_and_hash: &mut SerdeKeyAndHash<K>,
    state_ptr: &mut AggregationStatesPtr,
    agg_func_list: &AggregationFunctionList,
) {
    match swiss_table.get(key_and_hash.hash_value, |probe_element| {
        probe_element
            .serde_key_and_hash
            .serde_key
            .eq(&key_and_hash.serde_key)
    }) {
        None => {
            // The group by keys do not in the hash table, we need to create a new
            // aggregation states to store its result
            let agg_states_ptr = agg_func_list.alloc_states(arena);
            // Insert it into the raw table
            swiss_table.insert(
                key_and_hash.hash_value,
                Element {
                    serde_key_and_hash: SerdeKeyAndHash {
                        serde_key: std::mem::take(&mut key_and_hash.serde_key),
                        hash_value: key_and_hash.hash_value,
                    },
                    agg_states_ptr,
                },
                |element| element.serde_key_and_hash.hash_value,
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
    use data_block::array::{Int64Array, Int8Array};
    use data_block::block::DataBlock;
    use data_block::types::LogicalType;
    use snafu::Report;
    use std::sync::Arc;

    use super::*;
    use crate::exec::physical_expr::field_ref::FieldRef;
    use crate::exec::physical_expr::function::aggregate::count::CountStart;
    use crate::exec::physical_expr::function::aggregate::sum::Sum;
    use crate::exec::physical_expr::function::aggregate::{
        AggregationError, AggregationFunction, AggregationFunctionList,
    };
    use crate::exec::physical_operator::aggregate::hash_aggregate::serde::FixedSizedSerdeKeySerializer;

    #[test]
    fn test_add_block() -> Report<AggregationError> {
        Report::capture(|| {
            let mut table = HashTable::<FixedSizedSerdeKeySerializer<u16>>::new();
            let agg_funcs: Vec<Arc<dyn AggregationFunction>> = vec![
                Arc::new(CountStart::new()),
                Arc::new(Sum::<Int64Array>::try_new(Arc::new(FieldRef::new(
                    2,
                    LogicalType::BigInt,
                    "Wtf".to_string(),
                )))?),
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

            unsafe { agg_func_list.take_states(&table.state_ptrs, &mut output)? }
            let expect = expect_test::expect![[r#"
                ┌────────────────┬────────┐
                │ UnsignedBigInt │ BigInt │
                ├────────────────┼────────┤
                │ 4              │ -14    │
                ├────────────────┼────────┤
                │ 2              │ Null   │
                ├────────────────┼────────┤
                │ 4              │ 12     │
                └────────────────┴────────┘"#]];
            expect.assert_eq(&output.to_string());
            Ok(())
        })
    }
}
