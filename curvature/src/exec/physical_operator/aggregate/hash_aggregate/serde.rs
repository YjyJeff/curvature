//! Serialize the `GroupByKeys` into memory comparable types

use std::fmt::Debug;
use std::hash::Hash;

/// Trait for the memory comparable serde struct that stores the `GroupByKeys`
pub trait SerdeKey: Eq + Hash + Default + Debug + 'static {}
