//! Average aggregation function

use super::sum::{PayloadCast, SumPayloadArray, SumType};

use std::fmt::Debug;
use std::marker::PhantomData;

use data_block::array::{Array, ArrayError, ArrayImpl, Float64Array};
use data_block::element::Element;
use data_block::types::LogicalType;
use snafu::{ensure, Snafu};

use crate::common::utils::bytemuck::TransparentWrapper;
use crate::exec::physical_expr::function::aggregate::ArgTypeMismatchSnafu;
use crate::exec::physical_expr::function::Function;

use super::{
    AggregationError, SpecialOptionalUAFWrapper, SpecialOptionalUnaryAggregationFunction,
    SpecialOptionalUnaryAggregationState, Stringify,
};

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

/// SAFETY: the `#[repr(transparent)]` is added for AvgState
unsafe impl<S: AvgSumType> TransparentWrapper<Option<AvgStateInner<S>>> for AvgState<S> {}

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
        write!(f, "fn Avg({:?}) -> Double", self.args[0])
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

impl<PayloadArray> SpecialOptionalUnaryAggregationFunction<PayloadArray, Float64Array>
    for Avg<PayloadArray>
where
    PayloadArray: SumPayloadArray,
    PayloadArray::Element: PayloadCast,
    <PayloadArray::Element as PayloadCast>::SumType: AvgSumType,
    for<'a> &'a PayloadArray: TryFrom<&'a ArrayImpl, Error = ArrayError>,
{
    type State = AvgState<<PayloadArray::Element as PayloadCast>::SumType>;
}

impl<PayloadArray> SpecialOptionalUnaryAggregationState<PayloadArray>
    for AvgState<<PayloadArray::Element as PayloadCast>::SumType>
where
    PayloadArray: SumPayloadArray,
    PayloadArray::Element: PayloadCast,
    <PayloadArray::Element as PayloadCast>::SumType: AvgSumType,
    for<'a> &'a PayloadArray: TryFrom<&'a ArrayImpl, Error = ArrayError>,
{
    type InnerAggStates = AvgStateInner<<PayloadArray::Element as PayloadCast>::SumType>;
    type OutputElement = f64;

    #[cfg(not(feature = "overflow_checks"))]
    type Error = crate::error::BangError;

    #[cfg(feature = "overflow_checks")]
    type Error = OverflowError;

    #[inline]
    fn update_inner_agg_states(
        current: &mut Self::InnerAggStates,
        payload_element: <<PayloadArray as Array>::Element as Element>::ElementRef<'_>,
    ) -> Result<(), Self::Error> {
        let payload_element = payload_element.into();
        Self::combine_inner_agg_states(
            current,
            AvgStateInner {
                sum: payload_element,
                count: 1,
            },
        )
    }

    #[inline]
    fn combine_inner_agg_states(
        combined: &mut Self::InnerAggStates,
        partial: Self::InnerAggStates,
    ) -> Result<(), Self::Error> {
        combined.count += partial.count;
        #[cfg(not(feature = "overflow_checks"))]
        {
            combined.sum = <PayloadArray::Element as PayloadCast>::SumType::ADD_FUNC(
                combined.sum,
                partial.sum,
            );

            Ok(())
        }

        #[cfg(feature = "overflow_checks")]
        {
            <PayloadArray::Element as PayloadCast>::SumType::ADD_FUNC(combined.sum, partial.sum)
                .map_or(Err(OverflowError), |sum| {
                    combined.sum = sum;
                    Ok(())
                })
        }
    }
}

impl<PayloadArray> Avg<PayloadArray>
where
    PayloadArray: SumPayloadArray,
    PayloadArray::Element: PayloadCast,
    <PayloadArray::Element as PayloadCast>::SumType: AvgSumType,
    for<'a> &'a PayloadArray: TryFrom<&'a ArrayImpl, Error = ArrayError>,
{
    /// Create a new Sum function
    pub fn try_new(
        arg: LogicalType,
    ) -> Result<SpecialOptionalUAFWrapper<Self, PayloadArray, Float64Array>, AggregationError> {
        ensure!(
            arg.physical_type() == PayloadArray::PHYSICAL_TYPE,
            ArgTypeMismatchSnafu {
                func: "Avg",
                expect_physical_type: PayloadArray::PHYSICAL_TYPE,
                arg
            }
        );

        Ok(SpecialOptionalUAFWrapper::new(Self {
            args: vec![arg],
            _phantom: PhantomData,
        }))
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

        <AvgState<i64> as SpecialOptionalUnaryAggregationState<Int32Array>>::combine(
            &mut s0, &mut s1,
        )
        .unwrap();
        assert_eq!(s0.0, None);

        <AvgState<i64> as SpecialOptionalUnaryAggregationState<Int32Array>>::update(&mut s1, 10)
            .unwrap();
        <AvgState<i64> as SpecialOptionalUnaryAggregationState<Int32Array>>::update(&mut s1, 20)
            .unwrap();

        <AvgState<i64> as SpecialOptionalUnaryAggregationState<Int32Array>>::update(&mut s0, 20)
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
        assert!(
            <AvgState<i64> as SpecialOptionalUnaryAggregationState<Int32Array>>::combine(
                &mut s0, &mut s1
            )
            .is_err()
        );
    }
}
