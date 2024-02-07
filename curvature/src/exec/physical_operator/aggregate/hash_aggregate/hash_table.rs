//! HashTable for hash aggregation

use super::serde::Serde;
use crate::common::types::HashValue;
use crate::common::utils::hash::{fixed_build_hasher_default, BuildHasherDefault};
use crate::exec::physical_expr::function::aggregate::{
    AggregationFunctionList, AggregationStatesPtr, Result as AggregationResult,
};

use crate::exec::physical_operator::aggregate::{Arena, ThreadSafeArena};
use data_block::array::ArrayImpl;
use hashbrown::raw::RawTable as SwissTable;

/// A hash table that will perform partition when too many keys inserted
///
/// # Generics
///
/// - `S`: Type for serde the `group_by_keys`
pub struct HashTable<S: Serde> {
    hash_table: SwissTable<Element<S::SerdeKey>>,
    /// Arena used to store the `AggregationStates`. So the arena has same lifetime
    /// with `swiss_tables`, it ensures that as long as the struct exist, we can
    /// access the [`AggregationStatesPtr`] stored in the swiss_table safely
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
        write!(f, "HashTable")
    }
}

/// Element of the hash table
#[derive(Debug, Clone)]
pub struct Element<K> {
    /// Serde key and its hash
    pub(super) serde_key_and_hash: SerdeKeyAndHash<K>,
    /// Pointer to the combined `AggregationStates`, the pointer is allocated
    /// in the arena
    agg_states_ptr: AggregationStatesPtr,
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
            hash_table: SwissTable::new(),
            arena: Arena::new(),
            build_hasher: fixed_build_hasher_default(),
            key_and_hash: Vec::new(),
            state_ptrs: Vec::new(),
        }
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
        agg_func_list: AggregationFunctionList,
    ) -> AggregationResult<()> {
        self.key_and_hash.resize_with(len, || SerdeKeyAndHash {
            serde_key: S::SerdeKey::default(),
            hash_value: HashValue::default(),
        });

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
            .for_each(|(key_and_hash, ptr)| {
                match self
                    .hash_table
                    .get(key_and_hash.hash_value, |probe_element| {
                        probe_element
                            .serde_key_and_hash
                            .serde_key
                            .eq(&key_and_hash.serde_key)
                    }) {
                    None => {
                        // The group by keys do not in the hash table, we need to create a new
                        // aggregation states to store its result
                        let agg_states_ptr = agg_func_list.alloc_states(&self.arena);
                        // Insert it into the raw table
                        self.hash_table.insert(
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
                        *ptr = agg_states_ptr;
                    }
                    Some(matched_element) => {
                        *ptr = matched_element.agg_states_ptr;
                    }
                }
            });

        // 3. Update the states based on the payloads
        // SAFETY: ptrs are valid is guaranteed by the probing step.
        // Payloads match the agg_func_list's signature is guaranteed by the safety of
        // this function
        agg_func_list.update_states(payloads, &self.state_ptrs)
    }

    /// Finalize(Drop) the hash table, take the underling swiss_table and arena.
    pub(super) fn finalize(&mut self) -> (SwissTable<Element<S::SerdeKey>>, ThreadSafeArena) {
        let arena = std::mem::take(&mut self.arena);
        let hash_table = std::mem::take(&mut self.hash_table);
        (hash_table, ThreadSafeArena::new(arena))
    }
}

impl<S: Serde> Default for HashTable<S> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}
