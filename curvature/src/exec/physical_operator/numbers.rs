//! A table with single field `number` that has logical type `UnsignedBigInt`
//!
//! In this module, the [`Relaxed`] memory ordering is enough, because we not
//! need to synchronize other memory with other threads. [`Relaxed`] guarantees
//! different threads get non-overlapping ranges

use std::cmp::min;
use std::num::NonZeroU64;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering::Relaxed;
use std::sync::Arc;

use super::{
    impl_regular_for_non_regular, impl_sink_for_non_sink,
    use_types_for_impl_regular_for_non_regular, use_types_for_impl_sink_for_non_sink,
    GlobalSourceState, LocalSourceState, OperatorError, OperatorResult, ParallelismDegree,
    PhysicalOperator, SourceExecStatus, SourceOperatorExt, StateStringify, Stringify,
    MAX_PARALLELISM_DEGREE,
};

use_types_for_impl_regular_for_non_regular!();
use_types_for_impl_sink_for_non_sink!();

use crate::error::SendableError;
use crate::STANDARD_VECTOR_SIZE;
use data_block::array::utils::physical_array_name;
use data_block::array::ArrayImpl;
use data_block::block::DataBlock;
use data_block::compute::sequence::sequence;
use data_block::types::LogicalType;
use snafu::{ResultExt, Snafu};

#[derive(Debug, Snafu)]
enum NumbersError {
    #[snafu(display(
        "`Numbers` output 1 field, but the output DataBlock is empty.
          PipelineExecutor should guarantee it never happens, it has fatal bug 😭"
    ))]
    EmptyOutputDataBlock,
    #[snafu(display(
        "Output array should be `UInt64Array`, but we found: {output_array}.
         PipelineExecutor should guarantee it never happens, it has fatal bug 😭"
    ))]
    InvalidOutputArray { output_array: String },
}

#[derive(Debug)]
/// A table with single field `number` that has logical type `UnsignedBigInt`
/// It will generate the numbers: (start..end)
pub struct Numbers {
    /// Start of the range. Inclusive
    start: u64,
    /// End of the range. Exclusive
    end: u64,
    output_types: Vec<LogicalType>,
    _children: Vec<Arc<dyn PhysicalOperator>>,
}

impl Numbers {
    /// Each local source state will fetch MORSEL_SIZE numbers from global
    pub const MORSEL_SIZE: u64 = 128 * STANDARD_VECTOR_SIZE as u64;

    /// Create a new [`Numbers`]. The count has type [`NonZeroU64`] guarantees
    /// the count is zero. If the count is zero, do not create [`Numbers`],
    /// create [`EmptyTableScan`] instead
    ///
    /// [`EmptyTableScan`]: super::empty_table_scan::EmptyTableScan
    pub fn new(start: u64, count: NonZeroU64) -> Self {
        Self {
            start,
            end: start.saturating_add(count.get()),
            output_types: vec![LogicalType::UnsignedBigInt],
            _children: vec![],
        }
    }
}

#[derive(Debug)]
/// Global source state of the [`Numbers`]
pub struct NumbersGlobalSourceState(Arc<AtomicU64>);

impl StateStringify for NumbersGlobalSourceState {
    fn name(&self) -> &'static str {
        "NumbersGlobalSourceState"
    }

    fn debug(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl GlobalSourceState for NumbersGlobalSourceState {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[derive(Debug)]
/// Local source state of the [`Numbers`]
pub struct NumbersLocalSourceState {
    /// Start number of the next array
    current: u64,
    /// End number of this morsel. If current >= end, the morsel is ended
    morsel_end: u64,
}

impl StateStringify for NumbersLocalSourceState {
    fn name(&self) -> &'static str {
        "NumbersLocalSourceState"
    }

    fn debug(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl LocalSourceState for NumbersLocalSourceState {
    fn as_mut_any(&mut self) -> &mut dyn std::any::Any {
        self
    }

    #[inline]
    fn read_data(&mut self, output: &mut DataBlock) -> OperatorResult<SourceExecStatus> {
        // Check stop or not
        if self.current >= self.morsel_end {
            return Ok(SourceExecStatus::Finished);
        }

        // Have more output
        let start = self.current;
        self.current += STANDARD_VECTOR_SIZE as u64;
        let end = min(self.current, self.morsel_end);

        let output = output
            .get_mutable_array(0)
            .ok_or_else(|| OperatorError::ReadData {
                op: self.name(),
                source: Box::new(NumbersError::EmptyOutputDataBlock),
            })?;
        let ArrayImpl::UInt64(array) = output else {
            return Err(OperatorError::ReadData {
                op: self.name(),
                source: Box::new(NumbersError::InvalidOutputArray {
                    output_array: physical_array_name(output),
                }),
            });
        };

        crate::mutate_data_block_safety!();
        unsafe {
            sequence(array, start, end);
        }
        Ok(SourceExecStatus::HaveMoreOutput)
    }
}

impl Stringify for Numbers {
    fn name(&self) -> &'static str {
        "Numbers"
    }

    fn debug(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }

    fn display(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Numbers: ({}..{})", self.start, self.end,)
    }
}

impl PhysicalOperator for Numbers {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn output_types(&self) -> &[LogicalType] {
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
    ) -> OperatorResult<ParallelismDegree> {
        let parallelism = ((self.start - self.end) + Self::MORSEL_SIZE - 1) / Self::MORSEL_SIZE;
        let parallelism = if parallelism > MAX_PARALLELISM_DEGREE.get() as u64 {
            MAX_PARALLELISM_DEGREE
        } else {
            // SAFETY: Numbers produce at least one element. The parallelism computation
            // guarantees the parallelism is non zero
            unsafe { ParallelismDegree::new_unchecked(parallelism as _) }
        };

        Ok(parallelism)
    }

    fn read_data(
        &self,
        output: &mut DataBlock,
        global_state: &dyn GlobalSourceState,
        local_state: &mut dyn LocalSourceState,
    ) -> OperatorResult<SourceExecStatus> {
        self.read_data_in_parallel(output, global_state, local_state)
    }

    fn global_source_state(&self) -> OperatorResult<Arc<dyn GlobalSourceState>> {
        Ok(Arc::new(NumbersGlobalSourceState(Arc::new(
            AtomicU64::new(self.start),
        ))))
    }

    fn local_source_state(
        &self,
        global_state: &dyn GlobalSourceState,
    ) -> OperatorResult<Box<dyn LocalSourceState>> {
        let current = self.downcast_ref_global_source_state(global_state)?;

        let morsel_start = current.0.fetch_add(Self::MORSEL_SIZE, Relaxed);
        if morsel_start < self.end {
            let morsel_end = min(morsel_start + Self::MORSEL_SIZE, self.end);
            Ok(Box::new(NumbersLocalSourceState {
                current: morsel_start,
                morsel_end,
            }))
        } else {
            Ok(Box::new(NumbersLocalSourceState {
                current: self.end,
                morsel_end: self.end,
            }))
        }
    }

    fn progress(&self, global_state: &dyn GlobalSourceState) -> OperatorResult<f64> {
        let current = self.downcast_ref_global_source_state(global_state)?;
        let current = current.0.load(Relaxed);
        if current >= self.end {
            Ok(1.0)
        } else {
            Ok((current - self.start) as f64 / (self.end - self.start) as f64)
        }
    }

    impl_sink_for_non_sink!();
}

impl SourceOperatorExt for Numbers {
    type GlobalSourceState = NumbersGlobalSourceState;
    type LocalSourceState = NumbersLocalSourceState;

    #[inline]
    fn next_morsel(
        &self,
        global_state: &Self::GlobalSourceState,
        local_state: &mut Self::LocalSourceState,
    ) -> bool {
        let morsel_start = global_state.0.fetch_add(Self::MORSEL_SIZE, Relaxed);
        let has_next = morsel_start < self.end;
        if has_next {
            let morsel_end = min(morsel_start + Self::MORSEL_SIZE, self.end);
            *local_state = NumbersLocalSourceState {
                current: morsel_start,
                morsel_end,
            };
        }
        has_next
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use data_block::types::Array;
    use snafu::Report;

    #[test]
    fn test_read_data_from_local_source() -> Report<OperatorError> {
        Report::capture(|| {
            let mut local_state = NumbersLocalSourceState {
                current: 0,
                morsel_end: STANDARD_VECTOR_SIZE as u64,
            };

            let mut output = DataBlock::with_logical_types(vec![LogicalType::UnsignedBigInt]);
            let status = local_state.read_data(&mut output)?;
            assert_eq!(status, SourceExecStatus::HaveMoreOutput);

            let status = local_state.read_data(&mut output)?;
            assert_eq!(status, SourceExecStatus::Finished);

            Ok(())
        })
    }

    #[test]
    fn test_read_numbers_in_parallel() -> Report<OperatorError> {
        fn sum_read_numbers(
            numbers: &Numbers,
            global_state: &dyn GlobalSourceState,
        ) -> OperatorResult<u64> {
            let mut sum = 0;
            let mut local_state = numbers.local_source_state(global_state)?;
            let mut output = DataBlock::with_logical_types(vec![LogicalType::UnsignedBigInt]);
            while let SourceExecStatus::HaveMoreOutput =
                numbers.read_data_in_parallel(&mut output, global_state, &mut *local_state)?
            {
                let ArrayImpl::UInt64(array) = output.get_array(0).unwrap() else {
                    panic!("Output array should be `UInt64Array`")
                };

                sum += array.values_iter().sum::<u64>();
            }

            Ok(sum)
        }

        Report::capture(|| {
            let count = Numbers::MORSEL_SIZE * 3;
            let numbers = Numbers::new(0, NonZeroU64::new(count).unwrap());
            let global_state = numbers.global_source_state()?;
            let sum = std::thread::scope(|s| {
                let jh_0 = s.spawn(|| sum_read_numbers(&numbers, &*global_state));
                let jh_1 = s.spawn(|| sum_read_numbers(&numbers, &*global_state));
                let mut sum = jh_0.join().unwrap()?;
                sum += jh_1.join().unwrap()?;
                Ok(sum)
            })?;

            assert_eq!(sum, (0..count).sum::<u64>());

            Ok(())
        })
    }
}
