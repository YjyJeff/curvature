#![cfg_attr(feature = "portable_simd", feature(portable_simd))]
#![cfg_attr(feature = "avx512", feature(avx512_target_feature))]

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

pub mod aligned_vec;
pub mod array;
pub mod bitmap;
pub mod block;
pub mod compute;
pub mod element;
mod macros;
pub mod types;
pub mod utils;

mod private {
    /// Sealed trait protect against downstream implementations
    pub trait Sealed {}
}
