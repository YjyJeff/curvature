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

//! # Curvature
//!
//! `Curvature` is a high performance query engine designed for OLAP.

pub mod common;
pub mod error;
pub mod exec;
mod macros;
pub mod visit;
use self::macros::mutate_data_block_safety;

/// The default vector size used by the curvature
pub const STANDARD_VECTOR_SIZE: usize = 1024;

mod private {
    /// Sealed trait protect against downstream implementations
    pub trait Sealed {}
}
