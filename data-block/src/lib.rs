#![cfg_attr(feature = "portable_simd", feature(portable_simd))]

//! # DataBlock
//!
//! `DataBlock` is ours own implementation of [`Arrow`]. This crate uses the
//! implementation described in [`Type Exercise in Rust`] to support the subset
//! of [`Arrow`] format. What's more, `DataBlock` is designed to be used in the
//! query engine of `curvature`, therefore:
//!
//! - it only provides the functions that query engine need to use
//!
//! - optimize the speed in the use case of the query engine
//!
//! [`Arrow`]: https://github.com/apache/arrow-rs
//! [`Type Exercise in Rust`]: https://github.com/skyzh/type-exercise-in-rust
//!
//! # Run
//!
//! `DataBlock` does not support x86 CPUs that do not support `sse2`, run the code
//! in these CPUs will cause undefined behavior. Almost all of the x86 CPUs support
//! `sse2`, if not, the cpu is too old

pub mod aligned_vec;
pub mod array;
pub mod bitmap;
pub mod block;
pub mod compute;
pub mod element;
mod macros;
pub mod types;
pub mod utils;

/// Length of the `DataBlock`, it belongs to the range [0..65535(u16::MAX)]. We choose `u16` here
/// because rule of thumb. Almost all of the query engines' length is less than or equal to
/// 8192, currently 😂
pub type DataBlockLength = u16;

/// Length of the array, it belongs to the range [0..65535(u16::MAX)]. We choose `u16` here
/// because rule of thumb. Almost all of the query engines' length is less than or equal to
/// 8192, currently 😂
#[derive(Debug, Clone, Copy)]
pub enum ArrayLength {
    /// The array has exact this number of elements and at least two distinct values exist
    /// in the array
    Exact(DataBlockLength),
    /// It is a constant array, all of the elements in the array are equal. This array is
    /// special, it can represent array with any length
    Any,
}

impl PartialEq<DataBlockLength> for ArrayLength {
    #[inline]
    fn eq(&self, other: &DataBlockLength) -> bool {
        match self {
            Self::Exact(len) => len == other,
            Self::Any => true,
        }
    }
}

mod private {
    /// Sealed trait protect against downstream implementations
    pub trait Sealed {}
}
