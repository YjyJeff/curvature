//! Error in curvature

use snafu::Snafu;

/// Sendable error
pub type SendableError = Box<dyn std::error::Error + Send + Sync>;

/// Represent the `!` type
#[derive(Debug, Snafu)]
pub enum BangError {}
