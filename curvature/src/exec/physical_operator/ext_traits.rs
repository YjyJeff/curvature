//! Extension traits for operator

use data_block::block::DataBlock;

use super::utils::downcast_mut_local_state;
use super::{
    GlobalSinkState, GlobalSourceState, LocalSinkState, LocalSourceState, OperatorResult,
    PhysicalOperator, SourceExecStatus,
};

/// Extension of the source operator
pub trait SourceOperatorExt: PhysicalOperator {
    /// Global source state of the source operator
    type GlobalSourceState: GlobalSourceState;
    /// Local source state of the source operator
    type LocalSourceState: LocalSourceState;

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
        debug_assert!(self
            .output_types()
            .iter()
            .zip(output.arrays())
            .all(|(output_type, output_array)| { output_array.logical_type() == output_type }));

        let global_state = self.downcast_ref_global_source_state(global_state);
        let local_state =
            downcast_mut_local_state!(self, local_state, Self::LocalSourceState, SOURCE);

        // TBD: Can the morsel fetched from global state be empty?
        // If the morsel can not be empty, we can remove the loop
        loop {
            let status = self.read_local_data(output, local_state)?;
            match status {
                SourceExecStatus::HaveMoreOutput => return Ok(SourceExecStatus::HaveMoreOutput),
                SourceExecStatus::Finished => {
                    // Morsel in local source is exhausted, fetch next morsel
                    if !self.next_morsel(global_state, local_state) {
                        // No more unassigned morsel, stop executing this thread
                        return Ok(SourceExecStatus::Finished);
                    }
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
                "`{}` operator accepts invalid GlobalSinkState: `{}`. PipelineExecutor should guarantee it never happens, it has fatal bug 😭",
                self.name(),
                global_state.name()
            )
        }
    }
}
