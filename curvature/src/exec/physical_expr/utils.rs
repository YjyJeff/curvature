//! utils

use std::{
    fmt::{Debug, Display},
    sync::Arc,
};

use super::{function::aggregate::AggregationFunction, PhysicalExpr};

/// Wrapper should to display the [`PhysicalExpr`] in compact form
pub struct CompactExprDisplayWrapper<'a>(&'a dyn PhysicalExpr);

impl<'a> CompactExprDisplayWrapper<'a> {
    /// Create a new wrapper
    pub fn new(expr: &'a dyn PhysicalExpr) -> Self {
        Self(expr)
    }
}

impl<'a> Debug for CompactExprDisplayWrapper<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "CompactExprDisplayWrapper")
    }
}

impl<'a> Display for CompactExprDisplayWrapper<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.compact_display(f)
    }
}

/// Compact display array fo expressions
pub fn compact_display_expressions(
    f: &mut std::fmt::Formatter<'_>,
    exprs: &[Arc<dyn PhysicalExpr>],
) -> std::fmt::Result {
    write!(f, "[")?;
    let mut iter = exprs.iter();
    let Some(expr) = iter.next() else {
        return write!(f, "]");
    };
    expr.compact_display(f)?;
    iter.try_for_each(|expr| {
        write!(f, ", ")?;
        expr.compact_display(f)
    })?;
    write!(f, "]")
}

/// Display array of aggregation functions
pub fn display_agg_funcs(
    f: &mut std::fmt::Formatter<'_>,
    agg_funcs: &[Arc<dyn AggregationFunction>],
) -> std::fmt::Result {
    write!(f, "[")?;
    let mut iter = agg_funcs.iter();
    let Some(func) = iter.next() else {
        return write!(f, "]");
    };

    func.display(f)?;

    iter.try_for_each(|func| {
        write!(f, ", ")?;
        func.display(f)
    })?;
    write!(f, "]")
}
