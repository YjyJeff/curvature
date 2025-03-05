//! Extension traits for operator

use data_block::block::DataBlock;

use super::utils::downcast_mut_local_state;
use super::{
    GlobalOperatorState, GlobalSinkState, GlobalSourceState, LocalOperatorState, LocalSinkState,
    LocalSourceState, OperatorResult, PhysicalOperator, SourceExecStatus,
};

/// Extension of the regular operator
pub trait RegularOperatorExt: PhysicalOperator {
    /// Global operator state of the regular operator
    type GlobalOperatorState: GlobalOperatorState;
    /// Local operator state of the regular operator
    type LocalOperatorState: LocalOperatorState;

    /// Downcast the global state to operator's corresponding [`GlobalOperatorState`]`
    #[inline]
    fn downcast_ref_global_operator_state<'a>(
        &self,
        global_state: &'a dyn GlobalOperatorState,
    ) -> &'a Self::GlobalOperatorState {
        if let Some(global_state) = global_state
            .as_any()
            .downcast_ref::<Self::GlobalOperatorState>()
        {
            global_state
        } else {
            panic!(
                "Regular operator: `{}` accepts invalid GlobalOperatorState: `{}`. PipelineExecutor should guarantee it never happens, it has fatal bug 😭",
                self.name(),
                global_state.name()
            )
        }
    }

    /// Downcast the local state to operator's corresponding [`LocalOperatorState`]`
    #[inline]
    fn downcast_ref_local_operator_state<'a>(
        &self,
        local_state: &'a dyn LocalOperatorState,
    ) -> &'a Self::LocalOperatorState {
        if let Some(local_state) = local_state
            .as_any()
            .downcast_ref::<Self::LocalOperatorState>()
        {
            local_state
        } else {
            panic!(
                "Regular operator: `{}` accepts invalid LocalOperatorState: `{}`. PipelineExecutor should guarantee it never happens, it has fatal bug 😭",
                self.name(),
                local_state.name()
            )
        }
    }
}

/// Extension of the source operator
pub trait SourceOperatorExt: PhysicalOperator {
    /// Global source state of the source operator
    type GlobalSourceState: GlobalSourceState;
    /// Local source state of the source operator
    type LocalSourceState: LocalSourceState;

    /// Downcast the global state to operator's corresponding [`GlobalSourceState`]`
    #[inline]
    fn downcast_ref_global_source_state<'a>(
        &self,
        global_state: &'a dyn GlobalSourceState,
    ) -> &'a Self::GlobalSourceState {
        if let Some(global_state) = global_state
            .as_any()
            .downcast_ref::<Self::GlobalSourceState>()
        {
            global_state
        } else {
            panic!(
                "Source operator: `{}` accepts invalid GlobalSourceState: `{}`. PipelineExecutor should guarantee it never happens, it has fatal bug 😭",
                self.name(),
                global_state.name()
            )
        }
    }

    /// Downcast the local state to operator's corresponding [`LocalSourceState`]`
    #[inline]
    fn downcast_ref_local_source_state<'a>(
        &self,
        local_state: &'a dyn LocalSourceState,
    ) -> &'a Self::LocalSourceState {
        if let Some(local_state) = local_state
            .as_any()
            .downcast_ref::<Self::LocalSourceState>()
        {
            local_state
        } else {
            panic!(
                "Source operator: `{}` accepts invalid LocalSourceState: `{}`. PipelineExecutor should guarantee it never happens, it has fatal bug 😭",
                self.name(),
                local_state.name()
            )
        }
    }

    /// Read data from local source state
    fn read_local_data(
        &self,
        output: &mut DataBlock,
        local_state: &mut Self::LocalSourceState,
    ) -> OperatorResult<SourceExecStatus>;

    /// Fetch next morsel to LocalSourceState. It should only be called when
    /// LocalSourceState is exhausted
    ///
    /// Returns true if the global state has next unassigned morsel. Otherwise,
    /// return false
    fn next_morsel(
        &self,
        global_state: &Self::GlobalSourceState,
        local_state: &mut Self::LocalSourceState,
    ) -> bool;

    /// Read data from the operator that implement [`SourceOperatorExt`]
    #[inline]
    fn read_data_in_parallel(
        &self,
        output: &mut DataBlock,
        global_state: &dyn GlobalSourceState,
        local_state: &mut dyn LocalSourceState,
    ) -> OperatorResult<SourceExecStatus> {
        debug_assert_eq!(self.output_types().len(), output.num_arrays());
        debug_assert!(
            self.output_types()
                .iter()
                .zip(output.arrays())
                .all(|(output_type, output_array)| { output_array.logical_type() == output_type })
        );

        let global_state = self.downcast_ref_global_source_state(global_state);
        let local_state =
            downcast_mut_local_state!(self, local_state, Self::LocalSourceState, SOURCE);

        let status = self.read_local_data(output, local_state)?;
        match status {
            SourceExecStatus::HaveMoreOutput => Ok(SourceExecStatus::HaveMoreOutput),
            SourceExecStatus::Finished => {
                // Morsel in local source is exhausted, fetch next morsel
                if !self.next_morsel(global_state, local_state) {
                    // No more unassigned morsel, stop executing this thread
                    Ok(SourceExecStatus::Finished)
                } else {
                    // Have unassigned morsel, read data from this morsel. It should return
                    // `HaveMoreOutput`
                    self.read_local_data(output, local_state)
                }
            }
        }
    }
}

/// Extension of the sink operator
pub trait SinkOperatorExt: PhysicalOperator {
    /// Global sink state of the sink operator
    type GlobalSinkState: GlobalSinkState;
    /// Local sink state of the sink operator
    type LocalSinkState: LocalSinkState;

    /// Downcast the global state to operator's corresponding [`GlobalSinkState`]`
    #[inline]
    fn downcast_ref_global_sink_state<'a>(
        &self,
        global_state: &'a dyn GlobalSinkState,
    ) -> &'a Self::GlobalSinkState {
        if let Some(global_state) = global_state
            .as_any()
            .downcast_ref::<Self::GlobalSinkState>()
        {
            global_state
        } else {
            panic!(
                "Sink `{}` operator accepts invalid GlobalSinkState: `{}`. PipelineExecutor should guarantee it never happens, it has fatal bug 😭",
                self.name(),
                global_state.name()
            )
        }
    }
}
