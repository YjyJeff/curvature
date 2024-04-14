//! Utils for the physical operator

/// We have to use macro here. It is pretty annoying, see [`issue`] for details.
/// Fucking borrow checker !!!!!
///
/// [`issue`]: https://github.com/rust-lang/rust/issues/54663
macro_rules! downcast_mut_local_state {
    (SOURCE) => {
        "LocalSourceState"
    };
    (OPERATOR) => {
        "LocalOperatorState"
    };
    (SINK) => {
        "LocalSinkState"
    };
    ($op:ident, $local_state:ident, $ty:ty, $STATE_TY:ident) => {
        if let Some(local_state) = $local_state.as_mut_any().downcast_mut::<$ty>() {
            local_state
        } else {
            panic!(
                "`{}` operator accepts invalid {}: `{}`. PipelineExecutor should guarantee it never happens, it has fatal bug 😭",
                $op.name(),
                downcast_mut_local_state!($STATE_TY),
                $local_state.name()
            )
        }
    };
}

macro_rules! use_types_for_impl_regular_for_non_regular {
    () => {
        use crate::exec::physical_operator::{
            GlobalOperatorState, LocalOperatorState, OperatorExecStatus,
        };
    };
}

macro_rules! impl_regular_for_non_regular {
    () => {
        // Regular Operator

        fn is_regular_operator(&self) -> bool {
            false
        }

        fn execute(
            &self,
            _input: &DataBlock,
            _output: &mut DataBlock,
            _global_state: &dyn GlobalOperatorState,
            _local_state: &mut dyn LocalOperatorState,
        ) -> OperatorResult<OperatorExecStatus> {
            panic!(
                "`{}` is not a regular operator, can not call the `execute` method on it",
                self.name()
            )
        }

        fn global_operator_state(
            &self,
            _client_ctx: &ClientContext,
        ) -> Arc<dyn GlobalOperatorState> {
            panic!(
                "`{}` is not a regular operator, can not call the `global_operator_state` method on it",
                self.name()
            )
        }

        fn local_operator_state(&self) -> Box<dyn LocalOperatorState> {
            panic!(
                "`{}` is not a regular operator, can not call the `local_operator_state` method on it",
                self.name()
            )
        }

        fn merge_local_operator_metrics(&self, _local_state: &dyn LocalOperatorState){
            panic!(
                "`{}` is not a regular operator, can not call the `merge_local_operator_metrics` method on it",
                self.name()
            )
        }
    };
}

macro_rules! use_types_for_impl_source_for_non_source {
    () => {
        use crate::exec::physical_operator::{
            GlobalSourceState, LocalSourceState, ParallelismDegree, SourceExecStatus,
        };
    };
}

macro_rules! impl_source_for_non_source {
    () => {
        // Source

        fn is_source(&self) -> bool {
            false
        }

        fn source_parallelism_degree(
            &self,
            _global_state: &dyn GlobalSourceState,
        ) -> ParallelismDegree {
            panic!(
                "`{}` is not a source operator, can not call the `source_parallelism_degree` method on it",
                self.name()
            )
        }

        fn read_data(
            &self,
            _output: &mut DataBlock,
            _global_state: &dyn GlobalSourceState,
            _local_state: &mut dyn LocalSourceState,
        ) -> OperatorResult<SourceExecStatus> {
            panic!(
                "`{}` is not a source operator, can not call the `read_data` method on it",
                self.name()
            )
        }

        fn global_source_state(&self, _client_ctx: &ClientContext) -> Arc<dyn GlobalSourceState> {
            panic!(
                "`{}` is not a source operator, can not call the `global_source_state` method on it",
                self.name()
            )
        }

        fn local_source_state(
            &self,
            _global_state: &dyn GlobalSourceState,
        ) -> Box<dyn LocalSourceState> {
            panic!(
                "`{}` is not a source operator, can not call the `local_source_state` method on it",
                self.name()
            )
        }

        fn merge_local_source_metrics(&self, _local_state: &dyn LocalSourceState){
            panic!(
                "`{}` is not a source operator, can not call the `merge_local_source_metrics` method on it",
                self.name()
            )
        }

        fn progress(&self, _global_state: &dyn GlobalSourceState) -> f64 {
            panic!(
                "`{}` is not a source operator, can not call the `progress` method on it",
                self.name()
            )
        }
    };
}

macro_rules! use_types_for_impl_sink_for_non_sink {
    () => {
        use crate::exec::physical_operator::{GlobalSinkState, LocalSinkState, SinkExecStatus};
    };
}

macro_rules! impl_sink_for_non_sink {
    () => {
        // Sink

        fn is_sink(&self) -> bool {
            false
        }

        fn write_data(
            &self,
            _input: &DataBlock,
            _global_state: &dyn GlobalSinkState,
            _local_state: &mut dyn LocalSinkState,
        ) -> OperatorResult<SinkExecStatus> {
            panic!(
                "`{}` is not a sink operator, can not call the `write_data` method on it",
                self.name()
            )
        }

        fn combine_sink(
            &self,
            _global_state: &dyn GlobalSinkState,
            _local_state: &mut dyn LocalSinkState,
        ) -> OperatorResult<()> {
            panic!(
                "`{}` is not a sink operator, can not call the `merge_sink` method on it",
                self.name()
            )
        }

        unsafe fn finalize_sink(&self, _global_state: &dyn GlobalSinkState) -> OperatorResult<()> {
            panic!(
                "`{}` is not a sink operator, can not call the `finalize_sink` method on it",
                self.name()
            )
        }

        fn global_sink_state(&self, _client_ctx: &ClientContext) -> Arc<dyn GlobalSinkState> {
            panic!(
                "`{}` is not a sink operator, can not call the `global_sink_state` method on it",
                self.name()
            )
        }

        fn local_sink_state(&self, _global_state: &dyn GlobalSinkState) -> Box<dyn LocalSinkState> {
            panic!(
                "`{}` is not a sink operator, can not call the `global_sink_state` method on it",
                self.name()
            )
        }
    };
}

pub(super) use downcast_mut_local_state;
pub(super) use impl_regular_for_non_regular;
pub(super) use impl_sink_for_non_sink;
pub(super) use impl_source_for_non_source;
pub(super) use use_types_for_impl_regular_for_non_regular;
pub(super) use use_types_for_impl_sink_for_non_sink;
pub(super) use use_types_for_impl_source_for_non_source;
