//! Utils for the physical operator

macro_rules! use_types_for_impl_regular_for_non_regular {
    () => {
        use crate::exec::physical_operator::{
            ExecuteSnafu, GlobalOperatorState, GlobalOperatorStateSnafu, IsParallelOperatorSnafu,
            LocalOperatorState, LocalOperatorStateSnafu, OperatorExecStatus,
        };
    };
}

macro_rules! impl_regular_for_non_regular {
    () => {
        // Regular Operator

        fn is_regular_operator(&self) -> bool {
            false
        }

        fn is_parallel_operator(&self) -> OperatorResult<bool> {
            IsParallelOperatorSnafu { op: self.name() }.fail()
        }

        fn execute(
            &self,
            _input: &DataBlock,
            _output: &mut DataBlock,
            _global_state: &dyn GlobalOperatorState,
            _local_state: &mut dyn LocalOperatorState,
        ) -> OperatorResult<OperatorExecStatus> {
            let error: SendableError = format!(
                "`{}` is not a regular operator. It should never happens because pipeline should be verified before execution",
                self.name()
            )
            .into();
            Err(error).context(ExecuteSnafu { op: self.name() })
        }

        fn global_operator_state(&self) -> OperatorResult<Box<dyn GlobalOperatorState>> {
            GlobalOperatorStateSnafu { op: self.name() }.fail()
        }

        fn local_operator_state(&self) -> OperatorResult<Box<dyn LocalOperatorState>> {
            LocalOperatorStateSnafu { op: self.name() }.fail()
        }
    };
}

macro_rules! use_types_for_impl_source_for_non_source {
    () => {
        use crate::exec::physical_operator::{
            GlobalSourceState, GlobalSourceStateSnafu, LocalSourceState, LocalSourceStateSnafu,
            ParallelismDegree, ProgressSnafu, ReadDataSnafu, SourceParallelismDegreeSnafu,
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
        ) -> OperatorResult<ParallelismDegree> {
            SourceParallelismDegreeSnafu { op: self.name() }.fail()
        }

        fn read_data(
            &self,
            _output: &mut DataBlock,
            _global_state: &dyn GlobalSourceState,
            _local_state: &mut dyn LocalSourceState,
        ) -> OperatorResult<SourceExecStatus> {
            let error: SendableError = format!(
                "`{}` is not a source operator. It should never happens because pipeline should be verified before execution",
                self.name()
            )
            .into();
            Err(error).context(ReadDataSnafu { op: self.name() })
        }

        fn global_source_state(&self) -> OperatorResult<Box<dyn GlobalSourceState>> {
            GlobalSourceStateSnafu { op: self.name() }.fail()
        }

        fn local_source_state(
            &self,
            _global_state: &dyn GlobalSourceState,
        ) -> OperatorResult<Box<dyn LocalSourceState>> {
            LocalSourceStateSnafu { op: self.name() }.fail()
        }

        fn progress(&self, _global_state: &dyn GlobalSourceState) -> OperatorResult<f64> {
            ProgressSnafu { op: self.name() }.fail()
        }
    };
}

macro_rules! use_types_for_impl_sink_for_non_sink {
    () => {
        use crate::exec::physical_operator::{
            FinalizeSinkSnafu, FinishLocalSinkSnafu, GlobalSinkState, GlobalSinkStateSnafu,
            IsParallelSinkSnafu, LocalSinkState, LocalSinkStateSnafu, SinkExecStatus,
            SinkFinalizeStatus, WriteDataSnafu,
        };
    };
}

macro_rules! impl_sink_for_non_sink {
    () => {
        // Sink

        fn is_sink(&self) -> bool {
            false
        }

        fn is_parallel_sink(&self) -> OperatorResult<bool> {
            IsParallelSinkSnafu { op: self.name() }.fail()
        }

        fn write_data(
            &self,
            _input: &DataBlock,
            _global_state: &dyn GlobalSinkState,
            _local_state: &mut dyn LocalSinkState,
        ) -> OperatorResult<SinkExecStatus> {
            let error: SendableError = format!(
                "`{}` is not a sink operator. It should never happens because pipeline should be verified before execution",
                self.name()
            )
            .into();
            Err(error).context(WriteDataSnafu { op: self.name() })
        }

        fn finish_local_sink(
            &self,
            _input: &DataBlock,
            _global_state: &dyn GlobalSinkState,
            _local_state: &mut dyn LocalSinkState,
        ) -> OperatorResult<()> {
            let error: SendableError = format!(
                "`{}` is not a sink operator. It should never happens because pipeline should be verified before execution",
                self.name()
            )
            .into();
            Err(error).context(FinishLocalSinkSnafu { op: self.name() })
        }

        unsafe fn finalize_sink(
            &self,
            _global_state: &dyn GlobalSinkState,
        ) -> OperatorResult<SinkFinalizeStatus>{
            let error: SendableError = format!(
                "`{}` is not a sink operator. It should never happens because pipeline should be verified before execution",
                self.name()
            )
            .into();
            Err(error).context(FinalizeSinkSnafu { op: self.name() })
        }

        fn global_sink_state(&self) -> OperatorResult<Box<dyn GlobalSinkState>> {
            GlobalSinkStateSnafu { op: self.name() }.fail()
        }

        fn local_sink_state(&self) -> OperatorResult<Box<dyn LocalSinkState>> {
            LocalSinkStateSnafu { op: self.name() }.fail()
        }
    };
}

pub(super) use impl_regular_for_non_regular;
pub(super) use impl_sink_for_non_sink;
pub(super) use impl_source_for_non_source;
pub(super) use use_types_for_impl_regular_for_non_regular;
pub(super) use use_types_for_impl_sink_for_non_sink;
pub(super) use use_types_for_impl_source_for_non_source;
