#![warn(clippy::todo)]
#![deny(
    rustdoc::broken_intra_doc_links,
    rustdoc::bare_urls,
    rustdoc::private_intra_doc_links,
    rust_2018_idioms,
    missing_docs,
    clippy::needless_borrow,
    clippy::redundant_clone,
    missing_debug_implementations
)]
#![cfg_attr(feature = "portable_simd", feature(portable_simd))]
#![cfg_attr(feature = "intrinsics", feature(core_intrinsics))]

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
mod macros;
pub mod scalar;
pub mod types;
mod utils;

mod private {
    /// Sealed trait protect against downstream implementations
    pub trait Sealed {}
}
