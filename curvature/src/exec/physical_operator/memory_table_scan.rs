//! A table scan operator that contains many in memory [`DataBlock`]s

use data_block::block::{DataBlock, SendableDataBlock};
use data_block::types::LogicalType;
use std::collections::VecDeque;
use std::fmt::Debug;
use std::mem::ManuallyDrop;
use std::sync::atomic::{AtomicIsize, Ordering};
use std::sync::Arc;

use super::{
    impl_regular_for_non_regular, impl_sink_for_non_sink,
    use_types_for_impl_regular_for_non_regular, use_types_for_impl_sink_for_non_sink,
    GlobalSourceState, LocalSourceState, OperatorResult, ParallelismDegree, PhysicalOperator,
    SourceExecStatus, SourceOperatorExt, StateStringify, Stringify, MAX_PARALLELISM_DEGREE,
};
use crate::common::client_context::ClientContext;

use snafu::{ensure, Snafu};

use_types_for_impl_regular_for_non_regular!();
use_types_for_impl_sink_for_non_sink!();

const MORSEL_SIZE: usize = 64;

#[allow(missing_docs)]
#[derive(Debug, Snafu)]
pub enum MemoryTableScanError {
    #[snafu(display("The `SendableDataBlock` passed to `MemoryTableScan` is empty, used `EmptyTableScan` instead"))]
    EmptyDataBlock,
    #[snafu(display(
        "`SendableDataBlock`s have different logical types: {:?}",
        logical_types
    ))]
    InconsistentDataBlocks {
        logical_types: Vec<Vec<LogicalType>>,
    },
}

type Result<T> = std::result::Result<T, MemoryTableScanError>;

/// Task queue, that can only consume
struct StaticTaskQueue {
    inner: ManuallyDrop<Vec<SendableDataBlock>>,
    idx: AtomicIsize,
    // Ptr to start of the inner
    ptr: *const SendableDataBlock,
}

unsafe impl Send for StaticTaskQueue {}
unsafe impl Sync for StaticTaskQueue {}

impl Drop for StaticTaskQueue {
    fn drop(&mut self) {
        let idx = self.idx.load(Ordering::Relaxed);
        // If `idx < 0`, all of the elements in the queue has been consumed.
        // If `idx >= 0`, the queue remains `idx + 1` elements
        let len = std::cmp::max(idx + 1, 0);
        // SAFETY: drop data here, no one can use inner anymore
        unsafe {
            let mut inner = ManuallyDrop::take(&mut self.inner);
            // Set the len, such that we can drop the remaining T in queue. Tasks greater than
            // len will be dropped by consumer
            inner.set_len(len as usize);
        }
    }
}

impl StaticTaskQueue {
    fn new(blocks: Vec<SendableDataBlock>) -> Self {
        let idx = blocks.len() as isize - 1;
        let ptr = blocks.as_ptr();
        Self {
            inner: ManuallyDrop::new(blocks),
            idx: AtomicIsize::new(idx),
            ptr,
        }
    }

    #[inline]
    fn dispatch(&self, dst: &mut VecDeque<DataBlock>) {
        for _ in 0..MORSEL_SIZE {
            let idx = self.idx.fetch_sub(1, Ordering::Relaxed);
            // SAFETY: we have guaranteed idx is valid! It decrease in atomic and is greater
            // than or equal to 0
            if idx >= 0 {
                unsafe {
                    let block = self.ptr.add(idx as usize).read();
                    dst.push_back(block.into_inner())
                }
            } else {
                break;
            }
        }
    }
}

/// Table scan that contains many in memory [`DataBlock`]s
pub struct MemoryTableScan {
    queue: StaticTaskQueue,
    num_blocks: usize,
    output_types: Vec<LogicalType>,
    _children: Vec<Arc<dyn PhysicalOperator>>,
}

/// We can not display the data block in the memory table scan. Some threads may fetching
/// the data block when we perform display. Use after free may happens
impl Debug for MemoryTableScan {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "MemoryTableScan")
    }
}

impl MemoryTableScan {
    /// Try to create a new [`MemoryTableScan`]
    ///
    /// # Note
    ///
    /// The `blocks` argument can not be empty, if you want to pass empty table scan to it,
    /// use [`EmptyTableScan`](crate::exec::physical_operator::empty_table_scan::EmptyTableScan)
    /// instead
    pub fn try_new(blocks: Vec<SendableDataBlock>) -> Result<Self> {
        ensure!(!blocks.is_empty(), EmptyDataBlockSnafu);
        let mut iter = blocks.iter();
        let block = iter
            .next()
            .expect("We have checked the blocks is non-empty");
        let num_arrays = block.as_ref().num_arrays();
        let output_types = block.as_ref().logical_types().cloned().collect::<Vec<_>>();

        let error_ctx = || InconsistentDataBlocksSnafu {
            logical_types: blocks
                .iter()
                .map(|block| block.as_ref().logical_types().cloned().collect::<Vec<_>>())
                .collect::<Vec<_>>(),
        };

        for block in iter {
            ensure!(block.as_ref().num_arrays() == num_arrays, error_ctx());
            ensure!(
                output_types
                    .iter()
                    .zip(block.as_ref().logical_types())
                    .all(|(l, r)| l == r),
                error_ctx()
            );
        }

        let num_blocks = blocks.len();

        Ok(Self {
            queue: StaticTaskQueue::new(blocks),
            num_blocks,
            output_types,
            _children: Vec::new(),
        })
    }
}

/// Global source state for [`MemoryTableScan`]
#[derive(Debug)]
pub struct MemoryTableScanGlobalSourceState;

impl StateStringify for MemoryTableScanGlobalSourceState {
    fn name(&self) -> &'static str {
        "MemoryTableScanGlobalSourceState"
    }

    fn debug(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl GlobalSourceState for MemoryTableScanGlobalSourceState {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Local source state for [`MemoryTableScan`]
#[derive(Debug)]
pub struct MemoryTableScanLocalSourceState(VecDeque<DataBlock>);

impl StateStringify for MemoryTableScanLocalSourceState {
    fn name(&self) -> &'static str {
        "MemoryTableScanLocalSourceState"
    }

    fn debug(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl LocalSourceState for MemoryTableScanLocalSourceState {
    fn as_mut_any(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

impl Stringify for MemoryTableScan {
    fn name(&self) -> &'static str {
        "MemoryTableScan"
    }

    fn debug(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }

    fn display(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "MemoryTableScan: output_types={:?}", self.output_types)
    }
}

impl PhysicalOperator for MemoryTableScan {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn output_types(&self) -> &[data_block::types::LogicalType] {
        &self.output_types
    }

    fn children(&self) -> &[std::sync::Arc<dyn PhysicalOperator>] {
        &self._children
    }

    impl_regular_for_non_regular!();

    // Source
    fn is_source(&self) -> bool {
        true
    }

    fn source_parallelism_degree(
        &self,
        _global_state: &dyn GlobalSourceState,
    ) -> ParallelismDegree {
        let parallelism = (self.num_blocks + (MORSEL_SIZE - 1)) / MORSEL_SIZE;
        if parallelism > MAX_PARALLELISM_DEGREE.get() as usize {
            MAX_PARALLELISM_DEGREE
        } else {
            // SAFETY: MemoryTableScan guarantees the it has at least one data block.
            // The parallelism computation guarantees the parallelism is non zero
            unsafe { ParallelismDegree::new_unchecked(parallelism as _) }
        }
    }

    fn read_data(
        &self,
        output: &mut DataBlock,
        global_state: &dyn GlobalSourceState,
        local_state: &mut dyn LocalSourceState,
    ) -> OperatorResult<SourceExecStatus> {
        self.read_data_in_parallel(output, global_state, local_state)
    }

    fn global_source_state(&self, _client_ctx: &ClientContext) -> Arc<dyn GlobalSourceState> {
        Arc::new(MemoryTableScanGlobalSourceState)
    }

    fn local_source_state(
        &self,
        _global_state: &dyn GlobalSourceState,
    ) -> Box<dyn LocalSourceState> {
        let mut queue = VecDeque::with_capacity(MORSEL_SIZE);
        self.queue.dispatch(&mut queue);
        Box::new(MemoryTableScanLocalSourceState(queue))
    }

    fn progress(&self, _global_state: &dyn GlobalSourceState) -> f64 {
        let idx = self.queue.idx.load(Ordering::Relaxed);
        if idx < 0 {
            1.0
        } else {
            idx as f64 / self.num_blocks as f64
        }
    }

    impl_sink_for_non_sink!();
}

impl SourceOperatorExt for MemoryTableScan {
    type GlobalSourceState = MemoryTableScanGlobalSourceState;
    type LocalSourceState = MemoryTableScanLocalSourceState;

    #[inline]
    fn next_morsel(
        &self,
        _global_state: &Self::GlobalSourceState,
        local_state: &mut Self::LocalSourceState,
    ) -> bool {
        self.queue.dispatch(&mut local_state.0);
        !local_state.0.is_empty()
    }

    #[inline]
    fn read_local_data(
        &self,
        output: &mut DataBlock,
        local_state: &mut Self::LocalSourceState,
    ) -> OperatorResult<SourceExecStatus> {
        match local_state.0.pop_back() {
            Some(block) => {
                *output = block;
                Ok(SourceExecStatus::HaveMoreOutput)
            }
            None => Ok(SourceExecStatus::Finished),
        }
    }
}

#[cfg(test)]
mod tests {
    use data_block::{
        array::{ArrayImpl, Int32Array},
        types::Array,
    };

    use super::*;

    #[test]
    fn test_read_memory_table_scan() {
        fn sum_read_memory_table_scan(table_scan: &MemoryTableScan) -> i32 {
            let mut sum = 0;
            let mut local_state = table_scan.local_source_state(&MemoryTableScanGlobalSourceState);
            let mut output = DataBlock::with_logical_types(vec![LogicalType::Integer]);
            while let SourceExecStatus::HaveMoreOutput = table_scan
                .read_data_in_parallel(
                    &mut output,
                    &MemoryTableScanGlobalSourceState,
                    &mut *local_state,
                )
                .unwrap()
            {
                let ArrayImpl::Int32(array) = output.get_array(0).unwrap() else {
                    panic!("Output array should be `Int32Array`")
                };

                sum += array.values_iter().sum::<i32>();
            }

            sum
        }

        let blocks = (0..2 * MORSEL_SIZE)
            .map(|_| unsafe {
                SendableDataBlock::new(
                    DataBlock::try_new(vec![ArrayImpl::Int32(Int32Array::from_values_iter([1]))])
                        .unwrap(),
                )
            })
            .collect::<Vec<_>>();
        let table_scan = MemoryTableScan::try_new(blocks).unwrap();
        let sum = std::thread::scope(|s| {
            let jh_0 = s.spawn(|| sum_read_memory_table_scan(&table_scan));
            let jh_1 = s.spawn(|| sum_read_memory_table_scan(&table_scan));
            let mut sum = jh_0.join().unwrap();
            sum += jh_1.join().unwrap();
            sum
        });

        assert_eq!(sum, 2 * MORSEL_SIZE as i32);
    }
}
