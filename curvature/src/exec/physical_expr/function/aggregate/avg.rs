//! Average aggregation function

use super::sum::{PayloadCast, SumPayloadArray, SumType};

use std::alloc::Layout;
use std::fmt::Debug;
use std::marker::PhantomData;
use std::num::NonZeroUsize;
use std::ops::{Deref, DerefMut};

use data_block::array::{Array, ArrayError, ArrayImpl, PrimitiveArray};
use data_block::element::Element;
use data_block::types::LogicalType;
use snafu::{ensure, Snafu};

use crate::exec::physical_expr::function::aggregate::ArgTypeMismatchSnafu;
use crate::exec::physical_expr::function::Function;

use super::{
    special_optional_unary_batch_update, special_optional_unary_combine_states,
    special_optional_unary_take_states, special_optional_unary_update_states, AggregationFunction,
    AggregationStatesPtr, Result, SpecialOptionalUnaryAggregationState, Stringify,
    UnaryAggregationState,
};

#[cfg(feature = "overflow_checks")]
use super::{CombineStatesSnafu, UpdateStatesSnafu};
#[cfg(feature = "overflow_checks")]
use snafu::ResultExt;

/// Overflow error
#[derive(Debug, Snafu)]
#[snafu(display("Overflow happens when computing the `avg` aggregation function"))]
pub struct OverflowError;

/// Trait bound for all of the sum type that can compute average
pub trait AvgSumType: SumType {
    /// Convert SumType to f64
    fn as_f64(self) -> f64;
}

macro_rules! impl_avg_sum_type {
    ($($ty:ty),+) => {
        $(
            impl AvgSumType for $ty{
                #[inline]
                fn as_f64(self) -> f64 {
                    self as f64
                }
            }
        )+
    };
}

impl_avg_sum_type!(u64, i64, f64);

/// Average's inner aggregation states
#[derive(Debug, PartialEq, Eq)]
pub struct AvgStateInner<S: AvgSumType> {
    /// Sum value
    sum: S,
    /// Count
    count: u64,
}

impl<S: AvgSumType + From<T>, T: PayloadCast<SumType = S>> From<T> for AvgStateInner<S> {
    #[inline]
    fn from(value: T) -> Self {
        Self {
            sum: value.into(),
            count: 1,
        }
    }
}

impl<S: AvgSumType> From<AvgStateInner<S>> for f64 {
    #[inline]
    fn from(value: AvgStateInner<S>) -> Self {
        value.sum.as_f64() / value.count as f64
    }
}

/// Aggregation state for average function
#[derive(Debug, Default)]
#[repr(transparent)]
pub struct AvgState<S: AvgSumType>(Option<AvgStateInner<S>>);

impl<S: AvgSumType> Deref for AvgState<S> {
    type Target = Option<AvgStateInner<S>>;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<S: AvgSumType> DerefMut for AvgState<S> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

/// Aggregation function that calculate the average value of the numeric array
///
/// # Generic
///
/// - `PayloadArray`: The type of the numeric array that need to be summed
pub struct Avg<PayloadArray> {
    args: Vec<LogicalType>,
    _phantom: PhantomData<PayloadArray>,
}

/// SAFETY: The sum function does not contain the struct, the generic type is phantom
unsafe impl<PayloadArray> Send for Avg<PayloadArray> {}
unsafe impl<PayloadArray> Sync for Avg<PayloadArray> {}

impl<PayloadArray> Debug for Avg<PayloadArray> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Avg").field("args", &self.args).finish()
    }
}

impl<PayloadArray> Stringify for Avg<PayloadArray> {
    /// FIXME: contains the numeric type info
    fn name(&self) -> &'static str {
        "Avg"
    }

    fn debug(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }

    fn display(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "fn Avg({:?}) -> f64", self.args[0])
    }
}

impl<PayloadArray> Function for Avg<PayloadArray>
where
    PayloadArray: SumPayloadArray,
    PayloadArray::Element: PayloadCast,
{
    fn arguments(&self) -> &[LogicalType] {
        &self.args
    }

    fn return_type(&self) -> LogicalType {
        LogicalType::Double
    }
}

impl<PayloadArray> AggregationFunction for Avg<PayloadArray>
where
    PayloadArray: SumPayloadArray,
    PayloadArray::Element: PayloadCast,
    <PayloadArray::Element as PayloadCast>::SumType: AvgSumType,
    for<'a> &'a PayloadArray: TryFrom<&'a ArrayImpl, Error = ArrayError>,
    for<'a> &'a mut PayloadArray::SumArray: TryFrom<&'a mut ArrayImpl, Error = ArrayError>,
{
    fn state_layout(&self) -> Layout {
        Layout::new::<AvgState<<PayloadArray::Element as PayloadCast>::SumType>>()
    }

    unsafe fn init_state(&self, ptr: AggregationStatesPtr, state_offset: usize) {
        let state = ptr.offset_as_mut::<AvgState<<PayloadArray::Element as PayloadCast>::SumType>>(
            state_offset,
        );
        *state = AvgState(None);
    }

    unsafe fn update_states(
        &self,
        payloads: &[&ArrayImpl],
        state_ptrs: &[AggregationStatesPtr],
        state_offset: usize,
    ) -> Result<()> {
        special_optional_unary_update_states::<
            PayloadArray,
            AvgState<<PayloadArray::Element as PayloadCast>::SumType>,
        >(payloads, state_ptrs, state_offset)
    }

    unsafe fn batch_update_states(
        &self,
        _len: NonZeroUsize,
        payloads: &[&ArrayImpl],
        states_ptr: AggregationStatesPtr,
        state_offset: usize,
    ) -> Result<()> {
        special_optional_unary_batch_update::<
            PayloadArray,
            AvgState<<PayloadArray::Element as PayloadCast>::SumType>,
        >(payloads, states_ptr, state_offset)
    }

    unsafe fn combine_states(
        &self,
        partial_state_ptrs: &[AggregationStatesPtr],
        combined_state_ptrs: &[AggregationStatesPtr],
        state_offset: usize,
    ) -> Result<()> {
        special_optional_unary_combine_states::<
            PayloadArray,
            AvgState<<PayloadArray::Element as PayloadCast>::SumType>,
        >(partial_state_ptrs, combined_state_ptrs, state_offset)
    }

    unsafe fn take_states(
        &self,
        state_ptrs: &[AggregationStatesPtr],
        state_offset: usize,
        output: &mut ArrayImpl,
    ) -> Result<()> {
        special_optional_unary_take_states::<
            PayloadArray,
            PrimitiveArray<f64>,
            AvgState<<PayloadArray::Element as PayloadCast>::SumType>,
        >(state_ptrs, state_offset, output);
        Ok(())
    }

    /// Do nothing, no memory to drop
    unsafe fn drop_states(&self, _state_ptrs: &[AggregationStatesPtr], _state_offset: usize) {}
}

impl<PayloadArray> UnaryAggregationState<PayloadArray>
    for AvgState<<PayloadArray::Element as PayloadCast>::SumType>
where
    PayloadArray: SumPayloadArray,
    PayloadArray::Element: PayloadCast,
    <PayloadArray::Element as PayloadCast>::SumType: AvgSumType,
    for<'a> &'a PayloadArray: TryFrom<&'a ArrayImpl, Error = ArrayError>,
    for<'a> &'a mut PayloadArray::SumArray: TryFrom<&'a mut ArrayImpl, Error = ArrayError>,
{
    type Func = Avg<PayloadArray>;
}

impl<PayloadArray> SpecialOptionalUnaryAggregationState<PayloadArray>
    for AvgState<<PayloadArray::Element as PayloadCast>::SumType>
where
    PayloadArray: SumPayloadArray,
    PayloadArray::Element: PayloadCast,
    <PayloadArray::Element as PayloadCast>::SumType: AvgSumType,
    for<'a> &'a PayloadArray: TryFrom<&'a ArrayImpl, Error = ArrayError>,
    for<'a> &'a mut PayloadArray::SumArray: TryFrom<&'a mut ArrayImpl, Error = ArrayError>,
{
    type InnerAggStates = AvgStateInner<<PayloadArray::Element as PayloadCast>::SumType>;
    type OutputElement = f64;

    #[inline]
    fn update_inner_agg_states(
        current: &mut Self::InnerAggStates,
        payload_element: <<PayloadArray as Array>::Element as Element>::ElementRef<'_>,
    ) -> Result<()> {
        let payload_element = payload_element.into();
        current.count += 1;

        #[cfg(not(feature = "overflow_checks"))]
        {
            current.sum = <PayloadArray::Element as PayloadCast>::SumType::ADD_FUNC(
                current.sum,
                payload_element,
            );
        }

        #[cfg(feature = "overflow_checks")]
        {
            if let Some(s) = <PayloadArray::Element as PayloadCast>::SumType::ADD_FUNC(
                current.sum,
                payload_element,
            ) {
                current.sum = s;
            } else {
                return Err(Box::new(OverflowError) as _).context(CombineStatesSnafu);
            }
        }

        Ok(())
    }

    #[inline]
    fn combine_inner_agg_states(
        combined: &mut Self::InnerAggStates,
        partial: Self::InnerAggStates,
    ) -> Result<()> {
        combined.count += partial.count;
        #[cfg(not(feature = "overflow_checks"))]
        {
            combined.sum = <PayloadArray::Element as PayloadCast>::SumType::ADD_FUNC(
                combined.sum,
                partial.sum,
            );
        }

        #[cfg(feature = "overflow_checks")]
        {
            if let Some(s) =
                <PayloadArray::Element as PayloadCast>::SumType::ADD_FUNC(combined.sum, partial.sum)
            {
                combined.sum = s;
            } else {
                return Err(Box::new(OverflowError) as _).context(UpdateStatesSnafu);
            }
        }

        Ok(())
    }
}

impl<PayloadArray: SumPayloadArray> Avg<PayloadArray>
where
    PayloadArray::Element: PayloadCast,
    <PayloadArray::Element as PayloadCast>::SumType: AvgSumType,
{
    /// Create a new Sum function
    pub fn try_new(arg: LogicalType) -> Result<Self> {
        ensure!(
            arg.physical_type() == PayloadArray::PHYSICAL_TYPE,
            ArgTypeMismatchSnafu {
                func: "Avg",
                expect_physical_type: PayloadArray::PHYSICAL_TYPE,
                arg
            }
        );

        Ok(Self {
            args: vec![arg],
            _phantom: PhantomData,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use data_block::array::Int32Array;

    #[test]
    fn test_avg_update() {
        let mut sum_state = AvgState::<i64>::default();
        <AvgState<i64> as SpecialOptionalUnaryAggregationState<Int32Array>>::update(
            &mut sum_state,
            10,
        )
        .unwrap();
        assert_eq!(sum_state.0, Some(AvgStateInner { sum: 10, count: 1 }));
        <AvgState<i64> as SpecialOptionalUnaryAggregationState<Int32Array>>::update(
            &mut sum_state,
            10,
        )
        .unwrap();
        assert_eq!(sum_state.0, Some(AvgStateInner { sum: 20, count: 2 }));
    }

    #[test]
    fn test_avg_combine_and_take() {
        let mut s0 = AvgState::<i64>::default();
        let mut s1 = AvgState::<i64>::default();

        unsafe {
            <AvgState<i64> as SpecialOptionalUnaryAggregationState<Int32Array>>::combine(
                &mut s0, &mut s1,
            )
            .unwrap();
            assert_eq!(s0.0, None);

            <AvgState<i64> as SpecialOptionalUnaryAggregationState<Int32Array>>::update(
                &mut s1, 10,
            )
            .unwrap();
            <AvgState<i64> as SpecialOptionalUnaryAggregationState<Int32Array>>::update(
                &mut s1, 20,
            )
            .unwrap();

            <AvgState<i64> as SpecialOptionalUnaryAggregationState<Int32Array>>::update(
                &mut s0, 20,
            )
            .unwrap();
            <AvgState<i64> as SpecialOptionalUnaryAggregationState<Int32Array>>::combine(
                &mut s0, &mut s1,
            )
            .unwrap();
            assert_eq!(s0.0, Some(AvgStateInner { sum: 50, count: 3 }));

            let avg =
                <AvgState<i64> as SpecialOptionalUnaryAggregationState<Int32Array>>::take(&mut s0)
                    .unwrap();
            assert!((avg - 50.0 / 3.0).abs() < f64::EPSILON);
        }
    }

    #[cfg(feature = "overflow_checks")]
    #[test]
    fn test_avg_overflow_checks() {
        let mut avg_state = AvgState(Some(AvgStateInner {
            sum: i64::MIN,
            count: 1,
        }));
        assert!(
            <AvgState<i64> as SpecialOptionalUnaryAggregationState<Int32Array>>::update(
                &mut avg_state,
                -1
            )
            .is_err()
        );

        let mut s0 = AvgState(Some(AvgStateInner {
            sum: i64::MAX,
            count: 100,
        }));
        let mut s1 = AvgState(Some(AvgStateInner { sum: 7, count: 200 }));
        unsafe {
            assert!(
                <AvgState<i64> as SpecialOptionalUnaryAggregationState<Int32Array>>::combine(
                    &mut s0, &mut s1
                )
                .is_err()
            );
        }
    }
}
