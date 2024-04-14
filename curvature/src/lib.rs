#![cfg_attr(feature = "likely", feature(core_intrinsics))]
#![cfg_attr(feature = "likely", allow(internal_features))]

//! # Curvature
//!
//! `Curvature` is a high performance query engine designed for OLAP.

pub mod common;
pub mod error;
pub mod exec;
pub mod tree_node;

/// The default vector size used by the curvature
pub const STANDARD_VECTOR_SIZE: usize = 1024;

mod private {
    /// Sealed trait protect against downstream implementations
    pub trait Sealed {}
}
