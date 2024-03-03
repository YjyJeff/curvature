//! Utils for hash

/// Default hash builder
pub type BuildHasherDefault = ahash::RandomState;

/// Constant build hasher default
pub const BUILD_HASHER_DEFAULT: BuildHasherDefault = BuildHasherDefault::with_seeds(9, 7, 9, 8);
