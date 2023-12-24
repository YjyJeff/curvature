//! Error in curvature

/// Sendable error
pub type SendableError = Box<dyn std::error::Error + Send + Sync>;
