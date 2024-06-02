//! Union operator

use std::sync::Arc;

use crate::common::client_context::ClientContext;
use data_block::block::DataBlock;
use data_block::types::LogicalType;
use snafu::{ensure, Snafu};

use super::{
    impl_regular_for_non_regular, impl_sink_for_non_sink, impl_source_for_non_source,
    use_types_for_impl_regular_for_non_regular, use_types_for_impl_sink_for_non_sink,
    use_types_for_impl_source_for_non_source, OperatorResult, PhysicalOperator, Stringify,
};
use_types_for_impl_regular_for_non_regular!();
use_types_for_impl_source_for_non_source!();
use_types_for_impl_sink_for_non_sink!();

/// Error for unions whose left and right child do not have same output types
#[derive(Debug, Snafu)]
#[snafu(display(
    "Union's left type: `{:?}` does not equals to right type: `{:?}`",
    left,
    right
))]
pub struct UnionInputError {
    left: Vec<LogicalType>,
    right: Vec<LogicalType>,
}

/// Union operator that unions two inputs with same output types
///
/// Note that it is a **fake** physical operator! It is only used to build
/// the pipelines and it will never appeared in the pipeline! Therefore,
/// it will never be executed.
#[derive(Debug)]
pub struct Union {
    output_types: Vec<LogicalType>,
    children: [Arc<dyn PhysicalOperator>; 2],
}

impl Union {
    /// Try to create a new union with left and right child. Returns
    /// error if these two inputs do not have same output types
    ///
    ///
    pub fn try_new(
        left: Arc<dyn PhysicalOperator>,
        right: Arc<dyn PhysicalOperator>,
    ) -> Result<Self, UnionInputError> {
        let left_output_types = left.output_types();
        let right_output_types = right.output_types();
        ensure!(
            left_output_types == right_output_types,
            UnionInputSnafu {
                left: left_output_types.to_owned(),
                right: right_output_types.to_owned(),
            }
        );

        Ok(Self {
            output_types: left_output_types.to_owned(),
            children: [left, right],
        })
    }

    /// Get left child of the union
    pub fn left(&self) -> &Arc<dyn PhysicalOperator> {
        // SAFETY: union have two children, the length of self.children is 2
        unsafe { self.children.get_unchecked(0) }
    }

    /// Get right child of the union
    pub fn right(&self) -> &Arc<dyn PhysicalOperator> {
        // SAFETY: union have two children, the length of self.children is 2
        unsafe { self.children.get_unchecked(1) }
    }
}

impl Stringify for Union {
    fn name(&self) -> &'static str {
        "Union"
    }

    fn debug(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }

    fn display(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Union")
    }
}

impl PhysicalOperator for Union {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn output_types(&self) -> &[LogicalType] {
        &self.output_types
    }

    fn children(&self) -> &[Arc<dyn PhysicalOperator>] {
        &self.children
    }

    fn metrics(&self) -> super::metric::MetricsSet {
        panic!("UnionExec is not executable, it does not have metric")
    }

    impl_regular_for_non_regular!();
    impl_source_for_non_source!();
    impl_sink_for_non_sink!();
}
