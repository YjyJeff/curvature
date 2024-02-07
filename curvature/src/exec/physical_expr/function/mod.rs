//! Functions

pub mod aggregate;

use std::sync::Arc;

use data_block::types::LogicalType;

use super::PhysicalExpr;

/// Trait describes the function signature
///
/// TBD: arguments should return Logical type instead of PhysicalExpr!
pub trait Function: Send + Sync + 'static {
    /// Get arguments of the function
    fn arguments(&self) -> &[Arc<dyn PhysicalExpr>];

    /// Return type of the function
    fn return_type(&self) -> LogicalType;
}
