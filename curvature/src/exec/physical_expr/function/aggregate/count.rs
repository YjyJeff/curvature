//! Count function

use data_block::array::{ArrayImpl, ScalarArray, UInt64Array};
use data_block::types::{Array, LogicalType};
use snafu::ensure;

use crate::exec::physical_expr::field_ref::FieldRef;
use crate::exec::physical_expr::{function::Function, PhysicalExpr};
use std::{alloc::Layout, sync::Arc};

use super::{AggregationFunction, AggregationStatesPtr, NotFieldRefArgsSnafu, Result, Stringify};

/// Count(*) function
pub type CountStart = Count<true>;

/// Aggregation state of the count function
#[derive(Debug)]
#[repr(transparent)]
pub struct CountState(u64);

/// Aggregation function that count the number of elements
///
/// If the `STAR` generic is true, the count will take `NULL` into consideration
#[derive(Debug)]
pub struct Count<const STAR: bool> {
    args: Vec<Arc<dyn PhysicalExpr>>,
}

impl<const STAR: bool> Stringify for Count<STAR> {
    #[inline]
    fn name(&self) -> &'static str {
        if STAR {
            "CountStar"
        } else {
            "Count"
        }
    }

    fn debug(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }

    fn display(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if STAR {
            write!(f, "CountStar")
        } else {
            write!(f, "Count(")?;
            self.args[0].display(f, true)?;
            write!(f, ")")
        }
    }
}

impl<const STAR: bool> Function for Count<STAR> {
    fn arguments(&self) -> &[Arc<dyn PhysicalExpr>] {
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
            Ok(())
        } else {
            let payload = payloads.get_unchecked(0);
            let validity = payload.validity();
            if validity.is_empty() {
                // All of the elements in the array is not null
                state_ptrs.iter().for_each(|ptr| unsafe {
                    let state = ptr.offset_as_mut::<CountState>(state_offset);
                    state.0 += 1;
                });
            } else {
                // Some elements in the array are null, we need to iterate the bitmap and
                // only take the non-null into consideration
                state_ptrs
                    .iter()
                    .zip(validity.iter())
                    .for_each(|(ptr, not_null)| unsafe {
                        if not_null {
                            let state = ptr.offset_as_mut::<CountState>(state_offset);
                            state.0 += 1;
                        }
                    });
            }
            Ok(())
        }
    }

    unsafe fn combine_states(
        &self,
        partial_state_ptrs: &[AggregationStatesPtr],
        combined_state_ptrs: &[AggregationStatesPtr],
        state_offset: usize,
    ) -> Result<()> {
        partial_state_ptrs
            .iter()
            .zip(combined_state_ptrs)
            .for_each(|(partial, combined)| {
                let combined = combined.offset_as_mut::<CountState>(state_offset);
                combined.0 += partial.offset_as_mut::<CountState>(state_offset).0;
            });

        Ok(())
    }

    unsafe fn take_states(
        &self,
        state_ptrs: &[AggregationStatesPtr],
        state_offset: usize,
        output: &mut data_block::array::ArrayImpl,
    ) -> Result<()> {
        #[cfg(debug_assertions)]
        let output: &mut UInt64Array = output
            .try_into()
            .expect("Output array of the count should be UInt64Array");
        #[cfg(not(debug_assertions))]
        let output: &mut UInt64Array = output.try_into().unwrap_unchecked();

        // TBD: It looks like we do not need to clear it, because it is always empty after
        // it is created!
        output.validity_mut().exactly_once_mut().clear();

        output.replace_with_trusted_len_values_iterator(
            state_ptrs.len(),
            state_ptrs
                .iter()
                .map(|ptr| ptr.offset_as_mut::<CountState>(state_offset).0),
        );

        Ok(())
    }

    /// Do nothing, no memory to drop
    unsafe fn drop_states(&self, _state_ptrs: &[AggregationStatesPtr], _state_offset: usize) {}
}

impl CountStart {
    /// Create a new `Count(*)`
    #[inline]
    pub fn new() -> Self {
        Self { args: vec![] }
    }
}

impl Default for CountStart {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl Count<false> {
    /// Create a new `Count(expr)` function
    #[inline]
    pub fn try_new(arg: Arc<dyn PhysicalExpr>) -> Result<Self> {
        ensure!(
            arg.as_any().downcast_ref::<FieldRef>().is_some(),
            NotFieldRefArgsSnafu {
                func: "Count",
                args: vec![arg]
            }
        );

        Ok(Self { args: vec![arg] })
    }
}
