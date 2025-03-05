//! Sum the numeric array
//!
//! There are two arrays here: the `payload` array that need to be summed and the `sum`
//! array that stores the result of the sum. For example, the sum of the `Int32Array`
//! is `Int64Array`. We implement it this way because although all of the elements are
//! in the `i32` range, sum of the elements may overflow.

use std::fmt::Debug;
use std::marker::PhantomData;

use data_block::array::{Array, ArrayError, ArrayImpl, PrimitiveArray, PrimitiveType};
use data_block::element::Element;
use data_block::types::LogicalType;
use snafu::{Snafu, ensure};

use crate::common::utils::bytemuck::TransparentWrapper;
use crate::exec::physical_expr::function::Function;
use crate::exec::physical_expr::function::aggregate::ArgTypeMismatchSnafu;

use super::{
    AggregationError, SpecialOptionalUAFWrapper, SpecialOptionalUnaryAggregationFunction,
    SpecialOptionalUnaryAggregationState, Stringify,
};

/// Overflow error
#[derive(Debug, Snafu)]
#[snafu(display("Overflow happens when computing the `sum` aggregation function"))]
pub struct OverflowError;

/// Trait for constrain the payload array of the sum function, only the arrays implement
/// this trait can be summed
pub trait SumPayloadArray: Array
where
    Self::Element: PayloadCast,
{
    /// The sum array of this payload array
    type SumArray: Array<Element = <Self::Element as PayloadCast>::SumType>;
}

/// Trait for casting payload type to sum type
pub trait PayloadCast: PrimitiveType {
    /// SumType
    type SumType: SumType + From<Self>;
}

/// Trait for all of the types that can be result of sum.
///
/// # Safety
///
/// The type implement [`SumType`] does not have any heap allocation
pub unsafe trait SumType: PrimitiveType {
    #[cfg(not(feature = "overflow_checks"))]
    /// Function for adding two sum type
    const ADD_FUNC: fn(Self, Self) -> Self;

    #[cfg(feature = "overflow_checks")]
    /// Function for adding two sum type
    const ADD_FUNC: fn(Self, Self) -> Option<Self>;
}

#[cfg(feature = "overflow_checks")]
/// Some the f64 and wrapping it in Option
#[inline]
fn float_checked_add(lhs: f64, rhs: f64) -> Option<f64> {
    Some(lhs + rhs)
}

macro_rules! impl_sum_payload_array {
    ($sum_ty:ty, {$($payload_ty:ty),+}, $checked_method:path) => {

        unsafe impl SumType for $sum_ty {
            #[cfg(not(feature = "overflow_checks"))]
            const ADD_FUNC: fn(Self, Self) -> Self = std::ops::Add::add;

            #[cfg(feature = "overflow_checks")]
            /// Function for adding two sum type
            const ADD_FUNC: fn(Self, Self) -> Option<Self> = $checked_method;
        }

        $(
            impl PayloadCast for $payload_ty {
                type SumType = $sum_ty;
            }

            impl SumPayloadArray for PrimitiveArray<$payload_ty> {
                type SumArray = PrimitiveArray<$sum_ty>;
            }
        )+
    };
}

impl_sum_payload_array!(u64,  {u8, u16, u32, u64}, u64::checked_add);
impl_sum_payload_array!(i64,  {i8, i16, i32, i64}, i64::checked_add);
impl_sum_payload_array!(f64,  {f32, f64}, float_checked_add);

/// Aggregation state of the sum function
///
/// TODO: Accelerate batch update with a temp SIMD vec. In this case, we can not
/// implement TransparentWrapper for it
#[derive(Debug)]
#[repr(transparent)]
pub struct SumState<S: SumType>(Option<S>);

/// SAFETY: the `#[repr(transparent)]` is added for SumState
unsafe impl<S: SumType> TransparentWrapper<Option<S>> for SumState<S> {}

/// Aggregation function that sum the numeric array
///
/// # Generic
///
/// - `PayloadArray`: The type of the numeric array that need to be summed
pub struct Sum<PayloadArray> {
    args: Vec<LogicalType>,
    _phantom: PhantomData<PayloadArray>,
}

/// SAFETY: The sum function does not contain the struct, the generic type is phantom
unsafe impl<PayloadArray> Send for Sum<PayloadArray> {}
unsafe impl<PayloadArray> Sync for Sum<PayloadArray> {}

impl<PayloadArray> Debug for Sum<PayloadArray> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Sum").field("args", &self.args).finish()
    }
}

impl<PayloadArray> Stringify for Sum<PayloadArray>
where
    PayloadArray: SumPayloadArray,
    PayloadArray::Element: PayloadCast,
{
    /// FIXME: contains the numeric type info
    fn name(&self) -> &'static str {
        "Sum"
    }

    fn debug(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }

    fn display(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "fn Sum({:?}) -> {:?}", self.args[0], self.return_type())
    }
}

impl<PayloadArray> Function for Sum<PayloadArray>
where
    PayloadArray: SumPayloadArray,
    PayloadArray::Element: PayloadCast,
{
    fn arguments(&self) -> &[LogicalType] {
        &self.args
    }

    /// FIXME: It may return other logical type. For example, sum of the duration
    /// is still duration
    fn return_type(&self) -> LogicalType {
        <PayloadArray::Element as PayloadCast>::SumType::DEFAULT_LOGICAL_TYPE
    }
}

impl<PayloadArray> SpecialOptionalUnaryAggregationFunction<PayloadArray, PayloadArray::SumArray>
    for Sum<PayloadArray>
where
    PayloadArray: SumPayloadArray,
    PayloadArray::Element: PayloadCast,
    for<'a> &'a PayloadArray: TryFrom<&'a ArrayImpl, Error = ArrayError>,
    for<'a> &'a mut PayloadArray::SumArray: TryFrom<&'a mut ArrayImpl, Error = ArrayError>,
{
    type State = SumState<<PayloadArray::Element as PayloadCast>::SumType>;
}

impl<PayloadArray> SpecialOptionalUnaryAggregationState<PayloadArray>
    for SumState<<PayloadArray::Element as PayloadCast>::SumType>
where
    PayloadArray: SumPayloadArray,
    PayloadArray::Element: PayloadCast,
    for<'a> &'a PayloadArray: TryFrom<&'a ArrayImpl, Error = ArrayError>,
    for<'a> &'a mut PayloadArray::SumArray: TryFrom<&'a mut ArrayImpl, Error = ArrayError>,
{
    type InnerAggStates = <PayloadArray::Element as PayloadCast>::SumType;
    type OutputElement = <PayloadArray::Element as PayloadCast>::SumType;

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
        Self::combine_inner_agg_states(current, payload_element)
    }

    #[inline]
    fn combine_inner_agg_states(
        combined: &mut Self::InnerAggStates,
        partial: Self::InnerAggStates,
    ) -> Result<(), Self::Error> {
        #[cfg(not(feature = "overflow_checks"))]
        {
            *combined =
                <PayloadArray::Element as PayloadCast>::SumType::ADD_FUNC(*combined, partial);
            Ok(())
        }

        #[cfg(feature = "overflow_checks")]
        {
            <PayloadArray::Element as PayloadCast>::SumType::ADD_FUNC(*combined, partial).map_or(
                Err(OverflowError),
                |s| {
                    *combined = s;
                    Ok(())
                },
            )
        }
    }
}

impl<PayloadArray> Sum<PayloadArray>
where
    PayloadArray: SumPayloadArray,
    PayloadArray::Element: PayloadCast,
    for<'a> &'a PayloadArray: TryFrom<&'a ArrayImpl, Error = ArrayError>,
    for<'a> &'a mut PayloadArray::SumArray: TryFrom<&'a mut ArrayImpl, Error = ArrayError>,
{
    /// Create a new Sum function
    pub fn try_new(
        arg: LogicalType,
    ) -> Result<
        SpecialOptionalUAFWrapper<Self, PayloadArray, PayloadArray::SumArray>,
        AggregationError,
    > {
        ensure!(
            arg.physical_type() == PayloadArray::PHYSICAL_TYPE,
            ArgTypeMismatchSnafu {
                func: "Sum",
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
    use std::ptr::NonNull;

    use super::*;
    use crate::exec::physical_expr::function::aggregate::{
        AggregationFunction, AggregationStatesPtr,
    };
    use data_block::array::{Float64Array, Int32Array, Int64Array};

    #[test]
    fn test_sum_func_allocate_and_init_state() {
        let sum_func = Sum::<Int32Array>::try_new(LogicalType::Integer).unwrap();

        unsafe {
            let ptr_ = std::alloc::alloc(sum_func.state_layout());
            let ptr = AggregationStatesPtr(NonNull::new_unchecked(ptr_));

            sum_func.init_state(ptr, 0);
            let sum_state = ptr.offset_as_mut::<SumState<i64>>(0);
            assert!(sum_state.0.is_none());

            std::alloc::dealloc(ptr_, sum_func.state_layout())
        }
    }

    #[test]
    fn test_sum_update() {
        let mut sum_state = SumState::<i64>(None);
        <SumState<i64> as SpecialOptionalUnaryAggregationState<Int32Array>>::update(
            &mut sum_state,
            10,
        )
        .unwrap();
        assert_eq!(sum_state.0, Some(10));
        <SumState<i64> as SpecialOptionalUnaryAggregationState<Int32Array>>::update(
            &mut sum_state,
            10,
        )
        .unwrap();
        assert_eq!(sum_state.0, Some(20));
    }

    #[test]
    fn test_sum_combine() {
        let mut s0 = SumState::<i64>(None);
        let mut s1 = SumState::<i64>(None);
        <SumState<i64> as SpecialOptionalUnaryAggregationState<Int64Array>>::combine(
            &mut s0, &mut s1,
        )
        .unwrap();
        assert_eq!(s0.0, None);

        <SumState<i64> as SpecialOptionalUnaryAggregationState<Int64Array>>::update(&mut s1, 10)
            .unwrap();
        <SumState<i64> as SpecialOptionalUnaryAggregationState<Int64Array>>::combine(
            &mut s0, &mut s1,
        )
        .unwrap();
        assert_eq!(s0.0, Some(10));

        <SumState<i64> as SpecialOptionalUnaryAggregationState<Int64Array>>::update(&mut s1, 10)
            .unwrap();
        <SumState<i64> as SpecialOptionalUnaryAggregationState<Int64Array>>::combine(
            &mut s0, &mut s1,
        )
        .unwrap();
        assert_eq!(s0.0, Some(20));
    }

    #[test]
    fn test_sum_finalize() {
        let mut sum_state_0 = SumState(Some(10.0));
        let mut sum_state_1 = SumState::<f64>(None);
        let mut sum_state_2 = SumState::<f64>(Some(-100.99));
        unsafe {
            let ptr_0 = AggregationStatesPtr(NonNull::new_unchecked(
                (&mut sum_state_0) as *mut _ as *mut u8,
            ));
            let ptr_1 = AggregationStatesPtr(NonNull::new_unchecked(
                (&mut sum_state_1) as *mut _ as *mut u8,
            ));
            let ptr_2 = AggregationStatesPtr(NonNull::new_unchecked(
                (&mut sum_state_2) as *mut _ as *mut u8,
            ));

            let ptrs = &[ptr_0, ptr_1, ptr_2];

            let mut output = ArrayImpl::new(LogicalType::Double);

            let sum = Sum::<Float64Array>::try_new(LogicalType::Double).unwrap();

            sum.take_states(ptrs, 0, &mut output).unwrap();

            let expected = expect_test::expect![[r#"
                Float64(
                    Float64Array { logical_type: Double, len: 3, data: [
                        Some(
                            10.0,
                        ),
                        None,
                        Some(
                            -100.99,
                        ),
                    ]},
                )
            "#]];
            expected.assert_debug_eq(&output)
        }
    }

    #[cfg(feature = "overflow_checks")]
    #[test]
    fn test_sum_overflow_checks() {
        let mut sum_state = SumState::<i64>(Some(i64::MAX));
        assert!(
            <SumState<i64> as SpecialOptionalUnaryAggregationState<Int32Array>>::update(
                &mut sum_state,
                1
            )
            .is_err()
        );

        let mut s0 = SumState::<i64>(Some(i64::MIN));
        let mut s1 = SumState::<i64>(Some(-1));
        assert!(
            <SumState<i64> as SpecialOptionalUnaryAggregationState<Int64Array>>::combine(
                &mut s0, &mut s1
            )
            .is_err()
        );
    }
}
