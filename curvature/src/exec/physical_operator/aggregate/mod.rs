//! Aggregation operators

use bumpalo::Bump;
/// The `Arena` used in the hash aggregation. Hash aggregation is a typic **phase-oriented**
/// allocation case. We store all of the `GroupByKeys` and the `AggregationStates` of the
/// keys in the hash table, it is insert only and never deleted individually. After
/// iterating all of the `GroupByKeys` in the table, we can delete the hash table as a
/// whole. Using arena here, we can improve the performance of allocating `GroupByKeys` and
/// `AggregationStates`
pub type Arena = Bump;

/// FIXME: verify the memory safety in theory
pub mod hash_aggregate;
pub mod simple_aggregate;
