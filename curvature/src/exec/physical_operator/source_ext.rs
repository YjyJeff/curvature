//! Source extension

use data_block::block::DataBlock;

use super::{
    GlobalSourceState, InvalidGlobalSourceStateSnafu, InvalidLocalSourceStateSnafu,
    LocalSourceState, OperatorResult, PhysicalOperator, SourceExecStatus,
};

/// We have to use macro here. It is pretty annoying, see [`issue`] for details.
/// Fucking borrow checker !!!!!
///
/// [`issue`]: https://github.com/rust-lang/rust/issues/54663
macro_rules! downcast_mut_local_source_state {
    ($self:ident, $local_state:ident, $ty:ty) => {
        if let Some(local_state) = $local_state.as_mut_any().downcast_mut::<$ty>() {
            local_state
        } else {
            return InvalidLocalSourceStateSnafu {
                op: $self.name(),
                state: $local_state.name(),
            }
            .fail();
        }
    };
}

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
    ) -> OperatorResult<&'a Self::GlobalSourceState> {
        if let Some(global_state) = global_state
            .as_any()
            .downcast_ref::<Self::GlobalSourceState>()
        {
            Ok(global_state)
        } else {
            InvalidGlobalSourceStateSnafu {
                op: self.name(),
                state: global_state.name(),
            }
            .fail()
        }
    }

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
        let global_state = self.downcast_ref_global_source_state(global_state)?;
        let local_state =
            downcast_mut_local_source_state!(self, local_state, Self::LocalSourceState);

        // TBD: Can the morsel fetched from global state be empty?
        // If the morsel can not be empty, we can remove the loop
        loop {
            let status = local_state.read_data(output)?;
            match status {
                SourceExecStatus::HaveMoreOutput => return Ok(status),
                SourceExecStatus::Finished => {
                    // Morsel in local source is exhausted, fetch next morsel
                    if !self.next_morsel(global_state, local_state) {
                        // No more unassigned morsel, stop executing this thread
                        return Ok(status);
                    }
                }
            }
        }
    }
}
