#[inline]
pub fn support_progress(progress: f64) -> bool {
    progress.is_nan()
}

/// Implement source and sink methods for regular operator. Return the default error
macro_rules! impl_source_sink_for_regular_operator {
    () => {
        #[inline]
        fn is_source(&self) -> bool {
            false
        }

        #[inline]
        fn is_parallel_source(&self) -> OperatorResult<bool> {
            IsParallelSourceSnafu { op: self.name() }.fail()
        }

        #[inline]
        fn read_data(
            &self,
            _output: &mut DataBlock,
            _global_state: &dyn GlobalSourceState,
            _local_state: &mut dyn LocalSourceState,
        ) -> OperatorResult<SourceExecStatus> {
            let error: Box<dyn std::error::Error> = format!(
                "`{}` is not a source operator. It should never happens because pipeline should be verified before execution",
                self.name()
            )
            .into();
            Err(error).context(ReadDataSnafu { op: self.name() })
        }

        #[inline]
        fn global_source_state(&self) -> OperatorResult<&dyn GlobalSourceState> {
            GlobalSourceStateSnafu { op: self.name() }.fail()
        }

        #[inline]
        fn local_source_state(&self) -> OperatorResult<Box<dyn LocalSourceState>> {
            LocalSourceStateSnafu { op: self.name() }.fail()
        }

        #[inline]
        fn progress(&self) -> OperatorResult<f64> {
            ProgressSnafu { op: self.name() }.fail()
        }

        // Sink

        #[inline]
        fn is_sink(&self) -> bool {
            false
        }

        #[inline]
        fn is_parallel_sink(&self) -> OperatorResult<bool> {
            IsParallelSinkSnafu { op: self.name() }.fail()
        }

        #[inline]
        fn write_data(
            &self,
            _input: &DataBlock,
            _global_state: &dyn GlobalSinkState,
            _local_state: &mut dyn LocalSinkState,
        ) -> OperatorResult<SinkExecStatus> {
            let error: Box<dyn std::error::Error> = format!(
                "`{}` is not a sink operator. It should never happens because pipeline should be verified before execution",
                self.name()
            )
            .into();
            Err(error).context(WriteDataSnafu { op: self.name() })
        }

        #[inline]
        fn finish_local_sink(
            &self,
            _input: &DataBlock,
            _global_state: &dyn GlobalSinkState,
            _local_state: &mut dyn LocalSinkState,
        ) -> super::Result<()> {
            let error: Box<dyn std::error::Error> = format!(
                "`{}` is not a sink operator. It should never happens because pipeline should be verified before execution",
                self.name()
            )
            .into();
            Err(error).context(FinishLocalSinkSnafu { op: self.name() })
        }

        #[inline]
        fn global_sink_state(&self) -> OperatorResult<&dyn GlobalSinkState> {
            GlobalSinkStateSnafu { op: self.name() }.fail()
        }

        #[inline]
        fn local_sink_state(&self) -> OperatorResult<Box<dyn LocalSinkState>> {
            LocalSinkStateSnafu { op: self.name() }.fail()
        }
    };
}

pub(super) use impl_source_sink_for_regular_operator;
