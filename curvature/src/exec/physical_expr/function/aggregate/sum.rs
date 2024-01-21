//! Sum the numeric array

use num_traits::SaturatingAdd;
use std::alloc::Layout;
use std::fmt::Debug;
use std::marker::PhantomData;
use std::ops::AddAssign;
use std::sync::Arc;

use data_block::array::{Array, ArrayError, ArrayImpl, PrimitiveType};
use data_block::types::{Element, LogicalType};

use crate::exec::physical_expr::function::Function;
use crate::exec::physical_expr::PhysicalExpr;

use super::{
    unary_combine_states, unary_update_states, AggregationFunction, AggregationStatesPtr, Result,
    Stringify, UnaryAggregationState,
};

/// Trait for all of the types that can be result of sum
pub trait SumStateType: PrimitiveType {}

impl SumStateType for u64 {}
impl SumStateType for i64 {}
impl SumStateType for f64 {}

/// Trait for casting payload type to sum type
pub trait PayloadCast: PrimitiveType {
    /// SumType
    type SumType;
    /// Cast
    fn cast(self) -> Self::SumType;
}

macro_rules! impl_payload_cast {
    ($sum_ty:ty, {$($payload_ty:ty),+}) => {
        $(
            impl PayloadCast for $payload_ty {
                type SumType = $sum_ty;

                #[inline]
                fn cast(self) -> Self::SumType {
                    self as _
                }
            }
        )+
    };
}

impl_payload_cast!(u64, {u8, u16, u32, u64});
impl_payload_cast!(i64, {i8, i16, i32, i64});
impl_payload_cast!(f32, {f32, f64});

/// Aggregation state of the sum function
#[derive(Debug)]
pub struct SumState<SumArray: Array> {
    sum: Option<SumArray::Element>,
}

/// Aggregation function that sum the numeric array
///
/// # Generic
///
/// - `PayloadArray`: The type of the numeric array that need to be summed
/// - `SumArray`: The result array type of the sum
pub struct Sum<PayloadArray, SumArray> {
    args: Vec<Arc<dyn PhysicalExpr>>,
    _phantom: PhantomData<(PayloadArray, SumArray)>,
}

/// SAFETY: The sum function does not contain the struct, the generic type is phantom
unsafe impl<PayloadArray, SumArray> Send for Sum<PayloadArray, SumArray> {}
unsafe impl<PayloadArray, SumArray> Sync for Sum<PayloadArray, SumArray> {}

impl<PayloadArray, SumArray> Debug for Sum<PayloadArray, SumArray> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Sum").field("args", &self.args).finish()
    }
}

impl<PayloadArray, SumArray> Stringify for Sum<PayloadArray, SumArray> {
    /// FIXME: contains the numeric type info
    fn name(&self) -> &'static str {
        "Sum"
    }

    fn debug(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }

    fn display(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Sum(")?;
        self.args[0].display(f, true)?;
        write!(f, ")")
    }
}

impl<PayloadArray, SumArray, SumT> Function for Sum<PayloadArray, SumArray>
where
    PayloadArray: Array,
    SumArray: Array<Element = SumT>,
    SumT: SumStateType,
{
    fn arguments(&self) -> &[Arc<dyn PhysicalExpr>] {
        &self.args
    }

    fn return_type(&self) -> LogicalType {
        SumT::LOGICAL_TYPE
    }
}

impl<PayloadArray, SumArray, PayloadT, SumT> AggregationFunction for Sum<PayloadArray, SumArray>
where
    PayloadArray: Array<Element = PayloadT>,
    for<'a> PayloadT: Element<ElementRef<'a> = PayloadT> + PayloadCast<SumType = SumT>,
    SumArray: Array<Element = SumT>,
    SumT: SumStateType + SaturatingAdd + AddAssign,
    for<'a> &'a PayloadArray: TryFrom<&'a ArrayImpl, Error = ArrayError>,
    for<'a> &'a mut SumArray: TryFrom<&'a mut ArrayImpl, Error = ArrayError>,
{
    fn state_layout(&self) -> Layout {
        Layout::new::<SumState<SumArray>>()
    }

    unsafe fn init_state(&self, ptr: AggregationStatesPtr, state_offset: usize) {
        let state = ptr.offset_as_mut::<SumState<SumArray>>(state_offset);
        *state = SumState { sum: None }
    }

    unsafe fn update_states(
        &self,
        payloads: &[&ArrayImpl],
        state_ptrs: &[AggregationStatesPtr],
        state_offset: usize,
    ) -> Result<()> {
        unary_update_states::<PayloadArray, SumState<SumArray>>(payloads, state_ptrs, state_offset);
        Ok(())
    }

    unsafe fn combine_states(
        &self,
        partial_state_ptrs: &[AggregationStatesPtr],
        combined_state_ptrs: &[AggregationStatesPtr],
        state_offset: usize,
    ) {
        unary_combine_states::<PayloadArray, SumState<SumArray>>(
            partial_state_ptrs,
            combined_state_ptrs,
            state_offset,
        )
    }

    unsafe fn finalize(
        &self,
        state_ptrs: &[AggregationStatesPtr],
        state_offset: usize,
        output: &mut ArrayImpl,
    ) -> Result<()> {
        todo!()
    }
}

impl<PayloadArray, SumArray, PayloadT, SumT> UnaryAggregationState<PayloadArray>
    for SumState<SumArray>
where
    PayloadArray: Array<Element = PayloadT>,
    for<'a> PayloadT: Element<ElementRef<'a> = PayloadT> + PayloadCast<SumType = SumT>,
    SumArray: Array<Element = SumT>,
    SumT: SumStateType + SaturatingAdd + AddAssign,
    for<'a> &'a PayloadArray: TryFrom<&'a ArrayImpl, Error = ArrayError>,
    for<'a> &'a mut SumArray: TryFrom<&'a mut ArrayImpl, Error = ArrayError>,
{
    type Func = Sum<PayloadArray, SumArray>;

    type Output = Option<SumT>;

    /// FIXME: accelerate with likely
    #[inline]
    fn update(&mut self, payload_element: PayloadT) {
        let payload_element = payload_element.cast();
        match &mut self.sum {
            Some(sum) => {
                #[cfg(feature = "saturating")]
                {
                    *sum = sum.saturating_add(&payload_element);
                }

                #[cfg(not(feature = "saturating"))]
                {
                    *sum += payload_element;
                }
            }
            None => self.sum = Some(payload_element),
        }
    }

    /// FIXME: accelerate with likely
    #[inline]
    fn combine(&mut self, partial: &Self) {
        if let Some(partial_sum) = partial.sum {
            match &mut self.sum {
                Some(combined_sum) => {
                    #[cfg(feature = "saturating")]
                    {
                        *combined_sum = combined_sum.saturating_add(&partial_sum);
                    }

                    #[cfg(not(feature = "saturating"))]
                    {
                        *combined_sum += partial_sum;
                    }
                }
                None => self.sum = Some(partial_sum),
            }
        }
    }

    #[inline]
    unsafe fn finalize(&mut self) -> Option<SumT> {
        self.sum
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use data_block::array::Int64Array;

    #[test]
    fn test_sum_update() {
        let mut sum_state = SumState::<Int64Array> { sum: None };
        <SumState<Int64Array> as UnaryAggregationState<Int64Array>>::update(&mut sum_state, 10);
        assert_eq!(sum_state.sum, Some(10));
        <SumState<Int64Array> as UnaryAggregationState<Int64Array>>::update(&mut sum_state, 10);
        assert_eq!(sum_state.sum, Some(20));

        #[cfg(feature = "saturating")]
        {
            <SumState<Int64Array> as UnaryAggregationState<Int64Array>>::update(
                &mut sum_state,
                i64::MAX,
            );

            assert_eq!(sum_state.sum, Some(i64::MAX));
        }
    }

    #[test]
    fn test_sum_combine() {
        let mut s0 = SumState::<Int64Array> { sum: None };
        let mut s1 = SumState::<Int64Array> { sum: None };
        <SumState<Int64Array> as UnaryAggregationState<Int64Array>>::combine(&mut s0, &s1);
        assert_eq!(s0.sum, None);

        <SumState<Int64Array> as UnaryAggregationState<Int64Array>>::update(&mut s1, 10);
        <SumState<Int64Array> as UnaryAggregationState<Int64Array>>::combine(&mut s0, &s1);
        assert_eq!(s0.sum, Some(10));

        <SumState<Int64Array> as UnaryAggregationState<Int64Array>>::update(&mut s1, 10);
        <SumState<Int64Array> as UnaryAggregationState<Int64Array>>::combine(&mut s0, &s1);
        assert_eq!(s0.sum, Some(30));
    }
}
