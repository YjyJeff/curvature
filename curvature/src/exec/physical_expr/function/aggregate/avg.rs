//! Average aggregation function

use super::sum::{PayloadCast, SumPayloadArray, SumType};

use std::alloc::Layout;
use std::fmt::Debug;
use std::marker::PhantomData;

use std::sync::Arc;

use data_block::array::{ArrayError, ArrayImpl, PrimitiveArray};
use data_block::types::LogicalType;
use snafu::{ensure, Snafu};

use crate::exec::physical_expr::field_ref::FieldRef;
use crate::exec::physical_expr::function::aggregate::ArgTypeMismatchSnafu;
use crate::exec::physical_expr::function::Function;
use crate::exec::physical_expr::PhysicalExpr;

use super::{
    unary_combine_states, unary_take_states, unary_update_states, AggregationFunction,
    AggregationStatesPtr, NotFieldRefArgsSnafu, Result, Stringify, UnaryAggregationState,
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

/// Aggregation state for average function
#[derive(Debug)]
pub struct AvgState<S: AvgSumType> {
    sum: Option<S>,
    count: u64,
}

/// Aggregation function that calculate the average value of the numeric array
///
/// # Generic
///
/// - `PayloadArray`: The type of the numeric array that need to be summed
pub struct Avg<PayloadArray> {
    args: Vec<Arc<dyn PhysicalExpr>>,
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
        write!(f, "Avg(")?;
        self.args[0].compact_display(f)?;
        write!(f, ")")
    }
}

impl<PayloadArray> Function for Avg<PayloadArray>
where
    PayloadArray: SumPayloadArray,
    PayloadArray::Element: PayloadCast,
{
    fn arguments(&self) -> &[Arc<dyn PhysicalExpr>] {
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
        *state = AvgState {
            sum: None,
            count: 0,
        }
    }

    unsafe fn update_states(
        &self,
        payloads: &[&ArrayImpl],
        state_ptrs: &[AggregationStatesPtr],
        state_offset: usize,
    ) -> Result<()> {
        unary_update_states::<PayloadArray, AvgState<<PayloadArray::Element as PayloadCast>::SumType>>(
            payloads,
            state_ptrs,
            state_offset,
        )
    }

    unsafe fn combine_states(
        &self,
        partial_state_ptrs: &[AggregationStatesPtr],
        combined_state_ptrs: &[AggregationStatesPtr],
        state_offset: usize,
    ) -> Result<()> {
        unary_combine_states::<
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
        unary_take_states::<
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

    type Output = Option<f64>;

    /// FIXME: accelerate with likely
    #[inline]
    fn update(&mut self, payload_element: PayloadArray::Element) -> Result<()> {
        self.count += 1;
        let payload_element = payload_element.cast();
        match &mut self.sum {
            Some(sum) => {
                #[cfg(not(feature = "overflow_checks"))]
                {
                    *sum = <PayloadArray::Element as PayloadCast>::SumType::ADD_FUNC(
                        *sum,
                        payload_element,
                    );
                }

                #[cfg(feature = "overflow_checks")]
                {
                    if let Some(s) = <PayloadArray::Element as PayloadCast>::SumType::ADD_FUNC(
                        *sum,
                        payload_element,
                    ) {
                        *sum = s;
                    } else {
                        return Err(Box::new(OverflowError) as _).context(CombineStatesSnafu);
                    }
                }
            }
            None => {
                self.sum = Some(payload_element);
            }
        }

        Ok(())
    }

    /// FIXME: accelerate with likely
    #[inline]
    unsafe fn combine(&mut self, partial: &mut Self) -> Result<()> {
        self.count += partial.count;

        if let Some(partial_sum) = Option::take(&mut partial.sum) {
            match &mut self.sum {
                Some(combined_sum) => {
                    #[cfg(not(feature = "overflow_checks"))]
                    {
                        *combined_sum = <PayloadArray::Element as PayloadCast>::SumType::ADD_FUNC(
                            *combined_sum,
                            partial_sum,
                        );
                    }

                    #[cfg(feature = "overflow_checks")]
                    {
                        if let Some(s) = <PayloadArray::Element as PayloadCast>::SumType::ADD_FUNC(
                            *combined_sum,
                            partial_sum,
                        ) {
                            *combined_sum = s;
                        } else {
                            return Err(Box::new(OverflowError) as _).context(UpdateStatesSnafu);
                        }
                    }
                }
                None => {
                    self.sum = Some(partial_sum);
                }
            }
        }

        Ok(())
    }

    #[inline]
    unsafe fn take(&mut self) -> Self::Output {
        self.sum.map(|sum| sum.as_f64() / self.count as f64)
    }
}

impl<PayloadArray: SumPayloadArray> Avg<PayloadArray>
where
    PayloadArray::Element: PayloadCast,
    <PayloadArray::Element as PayloadCast>::SumType: AvgSumType,
{
    /// Create a new Sum function
    pub fn try_new(arg: Arc<dyn PhysicalExpr>) -> Result<Self> {
        ensure!(
            arg.as_any().downcast_ref::<FieldRef>().is_some(),
            NotFieldRefArgsSnafu {
                func: "Avg",
                args: vec![arg]
            }
        );
        ensure!(
            arg.output_type().physical_type() == PayloadArray::PHYSCIAL_TYPE,
            ArgTypeMismatchSnafu {
                func: "Avg",
                expect_physical_type: PayloadArray::PHYSCIAL_TYPE,
                arg_type: arg.output_type().to_owned(),
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
        let mut sum_state = AvgState::<i64> {
            sum: None,
            count: 0,
        };
        <AvgState<i64> as UnaryAggregationState<Int32Array>>::update(&mut sum_state, 10).unwrap();
        assert_eq!(sum_state.sum, Some(10));
        assert_eq!(sum_state.count, 1);
        <AvgState<i64> as UnaryAggregationState<Int32Array>>::update(&mut sum_state, 10).unwrap();
        assert_eq!(sum_state.sum, Some(20));
        assert_eq!(sum_state.count, 2);
    }

    #[test]
    fn test_avg_combine_and_take() {
        let mut s0 = AvgState::<i64> {
            sum: None,
            count: 0,
        };
        let mut s1 = AvgState::<i64> {
            sum: None,
            count: 0,
        };

        unsafe {
            <AvgState<i64> as UnaryAggregationState<Int32Array>>::combine(&mut s0, &mut s1)
                .unwrap();
            assert_eq!(s0.sum, None);
            assert_eq!(s0.count, 0);

            <AvgState<i64> as UnaryAggregationState<Int32Array>>::update(&mut s1, 10).unwrap();
            <AvgState<i64> as UnaryAggregationState<Int32Array>>::update(&mut s1, 20).unwrap();

            <AvgState<i64> as UnaryAggregationState<Int32Array>>::update(&mut s0, 20).unwrap();
            <AvgState<i64> as UnaryAggregationState<Int32Array>>::combine(&mut s0, &mut s1)
                .unwrap();
            assert_eq!(s0.sum, Some(50));
            assert_eq!(s0.count, 3);

            let avg = <AvgState<i64> as UnaryAggregationState<Int32Array>>::take(&mut s0).unwrap();
            assert!((avg - 50.0 / 3.0).abs() < f64::EPSILON);
        }
    }

    #[cfg(feature = "overflow_checks")]
    #[test]
    fn test_avg_overflow_checks() {
        let mut avg_state = AvgState::<i64> {
            sum: Some(i64::MIN),
            count: 1,
        };
        assert!(
            <AvgState<i64> as UnaryAggregationState<Int32Array>>::update(&mut avg_state, -1)
                .is_err()
        );

        let mut s0 = AvgState::<i64> {
            sum: Some(i64::MAX),
            count: 100,
        };
        let mut s1 = AvgState::<i64> {
            sum: Some(7),
            count: 200,
        };
        unsafe {
            assert!(
                <AvgState<i64> as UnaryAggregationState<Int32Array>>::combine(&mut s0, &mut s1)
                    .is_err()
            );
        }
    }
}
