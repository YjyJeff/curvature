//! A magic hash table that will perform partition when too many keys is inserted

use super::serde::SerdeKey;
use super::BuildHasherDefault;
use crate::common::types::{HashValue, ParallelismDegree};
use crate::exec::physical_expr::function::aggregate::{
    AggregationFunctionList, AggregationStatesPtr,
};
use crate::exec::physical_operator::aggregate::Arena;
use crate::STANDARD_VECTOR_SIZE;
use data_block::array::ArrayImpl;
use hashbrown::raw::RawTable;
use strength_reduce::StrengthReducedU64;

/// Trait for partitioning the [`HashValue`] into partition index
pub trait Partitioning: Send + Sync + 'static {
    /// Create a partitioning
    fn new(partition_count: ParallelismDegree) -> Self;
    /// Partition the [`HashValue`] into partition index
    fn partition(&self, hash_value: HashValue) -> usize;
}

/// Partitioning based on modulo operation
#[derive(Debug)]
#[repr(transparent)]
pub struct ModuloPartitioning(StrengthReducedU64);

impl Partitioning for ModuloPartitioning {
    fn new(partition_count: ParallelismDegree) -> Self {
        Self(StrengthReducedU64::new(partition_count.get() as _))
    }

    #[inline]
    fn partition(&self, hash_value: HashValue) -> usize {
        (hash_value % self.0) as usize
    }
}

/// Element of the hash table
pub struct Element<K> {
    /// `GroupByKeys`, it maybe u64/u32/u128/Vec<u8>
    group_key: K,
    /// Pointer to the combined `AggregationStates`, the pointer is allocated
    /// in the arena
    agg_states_ptr: AggregationStatesPtr,
    /// Hash value of this element. It is used to avoid rehashing the keys
    /// when we need to resize the hash table
    hash_value: HashValue,
}

unsafe impl<K: Send> Send for Element<K> {}
unsafe impl<K: Send + Sync> Sync for Element<K> {}

struct RawHashTable<K> {
    /// We use the swiss table to support probing and insert
    swiss_table: RawTable<Element<K>>,
    /// Memory allocation reused across data blocks.
    elements: Vec<Element<K>>,
}

/// A magic hash table that will perform partition when too many keys is inserted
///
/// # Generics
///
/// - `K`: Type of the serde key
/// - `P`: Partitioning strategy, for example radix/modulo partitioning
pub struct PartitionableHashTable<K, P> {
    /// Swiss tables. If its length is 1, the table is not partitioned. Otherwise,
    /// the table is partitioned
    swiss_tables: Vec<RawHashTable<K>>,
    /// Arena used to store the `AggregationStates`. So the arena has same lifetime
    /// with `swiss_tables`, it ensures that as long as the struct exist, we can
    /// access the [`AggregationStatesPtr`] stored in the swiss_table safely
    arena: Arena,
    /// Build hasher for building hashes
    build_hasher: BuildHasherDefault,
    /// Partitioning strategy
    partitioning: P,

    /// Memory allocation reused across data blocks. It will only be used when the
    /// hash table is partitioned.
    ///
    /// The index of the first dimension is the partition index. And the value in
    /// the second dimension represents the row index in the data block.
    ///
    /// For example: `vec![vec![0,2], vec![1]]` represents the 0,2 row in the data
    /// block belong to the 0 partition and the 1 row belongs to the 1 partition
    partitions_row_indexes: Vec<Vec<usize>>,
}

impl<K: SerdeKey, P: Partitioning> PartitionableHashTable<K, P> {
    #[inline]
    pub fn new(partition_count: ParallelismDegree) -> Self {
        Self {
            swiss_tables: vec![RawHashTable {
                swiss_table: RawTable::new(),
                elements: Vec::with_capacity(STANDARD_VECTOR_SIZE),
            }],
            arena: Arena::new(),
            build_hasher: BuildHasherDefault::default(),
            partitioning: P::new(partition_count),
            partitions_row_indexes: Vec::new(),
        }
    }

    /// Partition the map, it should only be called when self is not partitioned
    fn partition(&mut self) {
        todo!()
    }

    pub fn add_block(
        &mut self,
        group_by_keys: &[&ArrayImpl],
        payloads: &[&ArrayImpl],
        func_list: AggregationFunctionList,
    ) {
        todo!()
    }
}
