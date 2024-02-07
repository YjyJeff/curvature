//! Utils for hash

/// Default hash builder. All of the [`BuildHasherDefault`] should be created with the
/// [`fixed_build_hasher_default`] function!
pub type BuildHasherDefault = ahash::RandomState;

/// Create a fixed build hasher default, such that different binaries runs on different
/// machines have identical hash behavior!
#[inline]
pub fn fixed_build_hasher_default() -> BuildHasherDefault {
    BuildHasherDefault::with_seeds(9, 7, 9, 8)
}
