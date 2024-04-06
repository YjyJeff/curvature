//! Functions

pub mod aggregate;

use data_block::types::LogicalType;

/// Trait describes the function signature
pub trait Function: Send + Sync + 'static {
    /// Get arguments of the function
    fn arguments(&self) -> &[LogicalType];

    /// Return type of the function
    fn return_type(&self) -> LogicalType;
}
