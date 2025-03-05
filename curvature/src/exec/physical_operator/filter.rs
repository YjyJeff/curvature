//! Filter operator

use std::sync::Arc;
use std::time::Duration;

use curvature_procedural_macro::MetricsSetBuilder;
use data_block::array::{ArrayImpl, BooleanArray};
use data_block::block::DataBlock;
use data_block::types::LogicalType;
use snafu::{Snafu, ensure};

use crate::common::client_context::ClientContext;
use crate::common::profiler::ScopedTimerGuard;
use crate::exec::physical_expr::constant::Constant;
use crate::exec::physical_expr::executor::Executor as ExprExecutor;
use crate::exec::physical_expr::utils::CompactExprDisplayWrapper;
use crate::exec::physical_expr::{ExprError, PhysicalExpr};
use crate::exec::physical_operator::utils::downcast_mut_local_state;
use crate::tree_node::display::IndentDisplayWrapper;

use super::ext_traits::RegularOperatorExt;
use super::metric::{Count, MetricsSet, Time};
use super::{
    DummyGlobalOperatorState, GlobalOperatorState, LocalOperatorState, OperatorError,
    OperatorExecStatus, OperatorResult, PhysicalOperator, StateStringify, Stringify,
    impl_sink_for_non_sink, impl_source_for_non_source, use_types_for_impl_sink_for_non_sink,
    use_types_for_impl_source_for_non_source,
};

use_types_for_impl_sink_for_non_sink!();
use_types_for_impl_source_for_non_source!();

#[allow(missing_docs)]
#[derive(Debug, Snafu)]
pub enum FilterError {
    #[snafu(display(
        "Predicate `{}` passed to `Filter` is not boolean expression. It has logical type: `{:?}`",
        predicate,
        logical_type
    ))]
    InvalidPredicate {
        predicate: String,
        logical_type: LogicalType,
    },
    #[snafu(display(
        "Filter's predicate can not be constant, it should be optimized by the planner/optimizer, found `{predicate}`"
    ))]
    ConstantPredicate { predicate: String },
    #[snafu(display("Failed to execute the `{}` operator", filter))]
    EvaluatePredicate { filter: String, source: ExprError },
}

/// Filter operator, it filters out the rows that do not match the predicate
#[derive(Debug)]
pub struct Filter {
    /// Input of the filter
    input: [Arc<dyn PhysicalOperator>; 1],
    /// Predicate
    predicate: Arc<dyn PhysicalExpr>,
    output_types: Vec<LogicalType>,

    /// Metrics
    metrics: FilterMetrics,
}

#[derive(Debug, Default, MetricsSetBuilder)]
struct FilterMetrics {
    /// Number of rows passed to filter
    rows_count: Count,
    /// Number of rows that has been filter out
    filter_out_count: Count,
    /// Time spent in evaluating the predicate
    predicate_time: Time,
    /// Time spent in filter the remaining rows to output
    filter_time: Time,
}

impl Filter {
    /// Try to create a new filter with input and predicate
    pub fn try_new(
        input: Arc<dyn PhysicalOperator>,
        predicate: Arc<dyn PhysicalExpr>,
    ) -> Result<Self, FilterError> {
        // Predicate should return bool
        ensure!(
            predicate.output_type() == &LogicalType::Boolean,
            InvalidPredicateSnafu {
                predicate: CompactExprDisplayWrapper::new(&*predicate).to_string(),
                logical_type: predicate.output_type().to_owned(),
            }
        );

        // Predicate can not be constant
        if let Some(predicate) = predicate.as_any().downcast_ref::<Constant>() {
            return ConstantPredicateSnafu {
                predicate: format!("{:?}", predicate),
            }
            .fail();
        }

        let output_types = input.output_types().to_owned();
        Ok(Self {
            input: [input],
            predicate,
            output_types,
            metrics: FilterMetrics::default(),
        })
    }
}

/// Local state of the filter
#[derive(Debug)]
pub struct FilterLocalState {
    executor: ExprExecutor,
    /// Array used to store temp result of the predicate
    temp: ArrayImpl,
    // Metrics
    metrics: LocalStateMetics,
}

#[derive(Debug, Default)]
struct LocalStateMetics {
    rows_count: u64,
    filter_out_count: u64,
    filter_time: Duration,
}

impl StateStringify for FilterLocalState {
    fn name(&self) -> &'static str {
        "FilterLocalState"
    }

    fn debug(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl LocalOperatorState for FilterLocalState {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_mut_any(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

impl Stringify for Filter {
    fn name(&self) -> &'static str {
        "Filter"
    }

    fn debug(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }

    fn display(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Filter: predicate={}",
            CompactExprDisplayWrapper::new(&*self.predicate)
        )
    }
}

impl PhysicalOperator for Filter {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn output_types(&self) -> &[LogicalType] {
        &self.output_types
    }

    fn children(&self) -> &[Arc<dyn PhysicalOperator>] {
        &self.input
    }

    fn metrics(&self) -> MetricsSet {
        self.metrics.metrics_set()
    }

    // Regular Operator

    fn is_regular_operator(&self) -> bool {
        true
    }

    fn execute(
        &self,
        input: &DataBlock,
        output: &mut DataBlock,
        _global_state: &dyn GlobalOperatorState,
        local_state: &mut dyn LocalOperatorState,
    ) -> OperatorResult<OperatorExecStatus> {
        let local_state = downcast_mut_local_state!(self, local_state, FilterLocalState, OPERATOR);

        let selection = local_state
            .executor
            .execute_predicate(input, &*self.predicate, &mut local_state.temp)
            .map_err(|err| OperatorError::Execute {
                source: Box::new(FilterError::EvaluatePredicate {
                    filter: (self as &dyn PhysicalOperator).to_string(),
                    source: err,
                }),
            })?;

        debug_assert!(selection.is_empty() || selection.len() == input.len());

        let new_len = selection.count_ones().unwrap_or(input.len());
        if new_len == 0 {
            // All of the rows have been filter out
            unsafe {
                output.clear();
            }
        } else if output.num_arrays() == 0 {
            // Length only optimization
            unsafe {
                output.set_len(new_len);
            };
        } else {
            // Copy the selected rows into the output
            let _scoped_guard = ScopedTimerGuard::new(&mut local_state.metrics.filter_time);
            unsafe {
                output
                    .filter(selection, input, new_len)
                    .expect("Pipeline executor guarantees the arrays in the output have same types with arrays in the input");
            }
        }

        // Update metrics
        local_state.metrics.rows_count += input.len() as u64;
        local_state.metrics.filter_out_count += (input.len() - new_len) as u64;

        Ok(OperatorExecStatus::NeedMoreInput)
    }

    fn global_operator_state(&self, _client_ctx: &ClientContext) -> Arc<dyn GlobalOperatorState> {
        Arc::new(DummyGlobalOperatorState)
    }

    fn local_operator_state(&self) -> Box<dyn LocalOperatorState> {
        Box::new(FilterLocalState {
            executor: ExprExecutor::new(&*self.predicate),
            temp: ArrayImpl::Boolean(BooleanArray::default()),
            metrics: LocalStateMetics::default(),
        })
    }

    fn merge_local_operator_metrics(&self, local_state: &dyn LocalOperatorState) {
        let local_state = self.downcast_ref_local_operator_state(local_state);
        let execute_predicate_time = local_state.executor.exec_time();
        tracing::debug!(
            "Filter processes `{}` rows, it filters out `{}` rows and remains `{}` rows in `{:?}`. Filter rows takes: `{:?}`. Execute predicate: `{}` takes: `{:?}`. Expr with metric:\n{}",
            local_state.metrics.rows_count,
            local_state.metrics.filter_out_count,
            local_state.metrics.rows_count - local_state.metrics.filter_out_count,
            execute_predicate_time + local_state.metrics.filter_time,
            local_state.metrics.filter_time,
            CompactExprDisplayWrapper::new(&*self.predicate),
            execute_predicate_time,
            IndentDisplayWrapper::new(
                &local_state
                    .executor
                    .displayable_expr_with_metric::<true>(&*self.predicate)
            ),
        );

        self.metrics.rows_count.add(local_state.metrics.rows_count);
        self.metrics
            .filter_out_count
            .add(local_state.metrics.filter_out_count);
        self.metrics
            .filter_time
            .add_duration(local_state.metrics.filter_time);
    }

    impl_source_for_non_source!();

    impl_sink_for_non_sink!();
}

impl RegularOperatorExt for Filter {
    type GlobalOperatorState = DummyGlobalOperatorState;
    type LocalOperatorState = FilterLocalState;
}
