//! Count function

use crate::exec::physical_expr::function::Function;
use data_block::array::{Array, ArrayImpl, UInt64Array};
use data_block::types::LogicalType;
use std::alloc::Layout;

use super::{AggregationFunction, AggregationStatesPtr, Result, Stringify};

/// Count(*) function
pub type CountStar = Count<true>;

/// Aggregation state of the count function
#[derive(Debug)]
#[repr(transparent)]
pub struct CountState(u64);

/// Aggregation function that count the number of elements
///
/// If the `STAR` generic is true, the count will take `NULL` into consideration
#[derive(Debug)]
pub struct Count<const STAR: bool> {
    args: Vec<LogicalType>,
}

impl<const STAR: bool> Stringify for Count<STAR> {
    fn name(&self) -> &'static str {
        if STAR { "CountStar" } else { "Count" }
    }

    fn debug(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }

    fn display(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if STAR {
            write!(f, "fn CountStar() -> u64")
        } else {
            write!(f, "fn Count({:?}) -> u64", self.args[0])
        }
    }
}

impl<const STAR: bool> Function for Count<STAR> {
    fn arguments(&self) -> &[LogicalType] {
        &self.args
    }

    fn return_type(&self) -> LogicalType {
        LogicalType::UnsignedBigInt
    }
}

impl<const STAR: bool> AggregationFunction for Count<STAR> {
    fn state_layout(&self) -> std::alloc::Layout {
        Layout::new::<CountState>()
    }

    unsafe fn init_state(&self, ptr: AggregationStatesPtr, state_offset: usize) {
        unsafe { ptr.offset_as_mut::<CountState>(state_offset).0 = 0 };
    }

    unsafe fn update_states(
        &self,
        payloads: &[&ArrayImpl],
        state_ptrs: &[AggregationStatesPtr],
        state_offset: usize,
    ) -> Result<()> {
        if STAR {
            state_ptrs.iter().for_each(|ptr| unsafe {
                let state = ptr.offset_as_mut::<CountState>(state_offset);
                state.0 += 1;
            });
        } else {
            let payload = payloads[0];
            let validity = payload.validity();
            if validity.all_valid() {
                // All of the elements in the array is not null
                state_ptrs.iter().for_each(|ptr| unsafe {
                    let state = ptr.offset_as_mut::<CountState>(state_offset);
                    state.0 += 1;
                });
            } else {
                // Some elements in the array are null, we need to iterate the bitmap and
                // only take the non-null into consideration
                validity.iter_ones().for_each(|index| unsafe {
                    let ptr = *state_ptrs.get_unchecked(index);
                    let state = ptr.offset_as_mut::<CountState>(state_offset);
                    state.0 += 1;
                });
            }
        }

        Ok(())
    }

    unsafe fn batch_update_states(
        &self,
        len: usize,
        payloads: &[&ArrayImpl],
        states_ptr: AggregationStatesPtr,
        state_offset: usize,
    ) -> Result<()> {
        unsafe {
            let len = if STAR {
                len as u64
            } else {
                let validity = payloads[0].validity();
                validity
                    .count_ones()
                    .map_or(len as u64, |count_ones| count_ones as u64)
            };
            let state = states_ptr.offset_as_mut::<CountState>(state_offset);
            state.0 += len;

            Ok(())
        }
    }

    unsafe fn combine_states(
        &self,
        partial_state_ptrs: &[AggregationStatesPtr],
        combined_state_ptrs: &[AggregationStatesPtr],
        state_offset: usize,
    ) -> Result<()> {
        unsafe {
            partial_state_ptrs
                .iter()
                .zip(combined_state_ptrs)
                .for_each(|(partial, combined)| {
                    let combined = combined.offset_as_mut::<CountState>(state_offset);
                    combined.0 += partial.offset_as_mut::<CountState>(state_offset).0;
                });

            Ok(())
        }
    }

    unsafe fn take_states(
        &self,
        state_ptrs: &[AggregationStatesPtr],
        state_offset: usize,
        output: &mut data_block::array::ArrayImpl,
    ) -> Result<()> {
        unsafe {
            let output: &mut UInt64Array = output
                .try_into()
                .expect("Output array of the count should be UInt64Array");

            // TBD: It looks like we do not need to clear it, because it is always empty after
            // it is created!
            output.validity_mut().mutate().clear();

            output.replace_with_trusted_len_values_iterator(
                state_ptrs.len(),
                state_ptrs
                    .iter()
                    .map(|ptr| ptr.offset_as_mut::<CountState>(state_offset).0),
            );

            Ok(())
        }
    }

    /// Do nothing, no memory to drop
    unsafe fn drop_states(&self, _state_ptrs: &[AggregationStatesPtr], _state_offset: usize) {}
}

impl CountStar {
    /// Create a new `Count(*)`
    #[inline]
    pub fn new() -> Self {
        Self { args: vec![] }
    }
}

impl Default for CountStar {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl Count<false> {
    /// Create a new `Count(expr)` function
    #[inline]
    pub fn new(arg: LogicalType) -> Self {
        Self { args: vec![arg] }
    }
}
