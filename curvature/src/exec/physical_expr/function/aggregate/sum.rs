//! Sum the numeric array
//!
//! There are two arrays here: the `payload` array that need to be summed and the `sum`
//! array that stores the result of the sum. For example, the sum of the `Int32Array`
//! is `Int64Array`. We implement it this way because although all of the elements are
//! in the `i32` range, sum of the elements may overflow.

use std::alloc::Layout;
use std::fmt::Debug;
use std::marker::PhantomData;
use std::ops::AddAssign;
use std::sync::Arc;

use data_block::array::{ArrayError, ArrayImpl, PrimitiveArray, PrimitiveType, ScalarArray};
use data_block::element::interval::DayTime;
use data_block::types::LogicalType;
use snafu::ensure;

use crate::exec::physical_expr::field_ref::FieldRef;
use crate::exec::physical_expr::function::aggregate::ArgTypeMismatchSnafu;
use crate::exec::physical_expr::function::Function;
use crate::exec::physical_expr::PhysicalExpr;

use super::{
    unary_combine_states, unary_take_states, unary_update_states, AggregationFunction,
    AggregationStatesPtr, NotFieldRefArgsSnafu, Result, Stringify, UnaryAggregationState,
};

/// Trait for constrain the payload array of the sum function, only the arrays implement
/// this trait can be summed
pub trait SumPayloadArray: ScalarArray
where
    Self::Element: PayloadCast,
{
    /// The sum array of this payload array
    type SumArray: ScalarArray<Element = <Self::Element as PayloadCast>::SumType>;
}

/// Trait for casting payload type to sum type
pub trait PayloadCast: PrimitiveType {
    /// SumType
    type SumType: SumType;
    /// Cast
    fn cast(self) -> Self::SumType;
}

/// Trait for all of the types that can be result of sum.
///
/// # Safety
///
/// The type implement [`SumType`] does not have any heap allocation
pub unsafe trait SumType: PrimitiveType + AddAssign {}

macro_rules! impl_sum_payload_array {
    ($sum_ty:ty, {$($payload_ty:ty),+}) => {

        unsafe impl SumType for $sum_ty {}

        $(
            impl PayloadCast for $payload_ty {
                type SumType = $sum_ty;

                #[inline]
                fn cast(self) -> Self::SumType {
                    self as _
                }
            }

            impl SumPayloadArray for PrimitiveArray<$payload_ty> {
                type SumArray = PrimitiveArray<$sum_ty>;
            }
        )+
    };
}

impl_sum_payload_array!(u64, {u8, u16, u32, u64});
impl_sum_payload_array!(i64, {i8, i16, i32, i64});
impl_sum_payload_array!(f64, {f32, f64});
impl_sum_payload_array!(DayTime, { DayTime });

/// Aggregation state of the sum function
#[derive(Debug)]
#[repr(transparent)]
pub struct SumState<S: SumType> {
    sum: Option<S>,
}

/// Aggregation function that sum the numeric array
///
/// # Generic
///
/// - `PayloadArray`: The type of the numeric array that need to be summed
pub struct Sum<PayloadArray> {
    args: Vec<Arc<dyn PhysicalExpr>>,
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

impl<PayloadArray> Stringify for Sum<PayloadArray> {
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

impl<PayloadArray> Function for Sum<PayloadArray>
where
    PayloadArray: SumPayloadArray,
    PayloadArray::Element: PayloadCast,
{
    fn arguments(&self) -> &[Arc<dyn PhysicalExpr>] {
        &self.args
    }

    fn return_type(&self) -> LogicalType {
        <PayloadArray::Element as PayloadCast>::SumType::LOGICAL_TYPE
    }
}

impl<PayloadArray> AggregationFunction for Sum<PayloadArray>
where
    PayloadArray: SumPayloadArray,
    PayloadArray::Element: PayloadCast,
    for<'a> &'a PayloadArray: TryFrom<&'a ArrayImpl, Error = ArrayError>,
    for<'a> &'a mut PayloadArray::SumArray: TryFrom<&'a mut ArrayImpl, Error = ArrayError>,
{
    fn state_layout(&self) -> Layout {
        Layout::new::<SumState<<PayloadArray::Element as PayloadCast>::SumType>>()
    }

    unsafe fn init_state(&self, ptr: AggregationStatesPtr, state_offset: usize) {
        let state = ptr.offset_as_mut::<SumState<<PayloadArray::Element as PayloadCast>::SumType>>(
            state_offset,
        );
        *state = SumState { sum: None }
    }

    unsafe fn update_states(
        &self,
        payloads: &[&ArrayImpl],
        state_ptrs: &[AggregationStatesPtr],
        state_offset: usize,
    ) -> Result<()> {
        unary_update_states::<
            PayloadArray,
            SumState<<PayloadArray::Element as PayloadCast>::SumType>,
        >(payloads, state_ptrs, state_offset);
        Ok(())
    }

    unsafe fn combine_states(
        &self,
        partial_state_ptrs: &[AggregationStatesPtr],
        combined_state_ptrs: &[AggregationStatesPtr],
        state_offset: usize,
    ) -> Result<()> {
        unary_combine_states::<
            PayloadArray,
            SumState<<PayloadArray::Element as PayloadCast>::SumType>,
        >(partial_state_ptrs, combined_state_ptrs, state_offset);

        Ok(())
    }

    unsafe fn take_states(
        &self,
        state_ptrs: &[AggregationStatesPtr],
        state_offset: usize,
        output: &mut ArrayImpl,
    ) -> Result<()> {
        unary_take_states::<
            PayloadArray,
            PayloadArray::SumArray,
            SumState<<PayloadArray::Element as PayloadCast>::SumType>,
        >(state_ptrs, state_offset, output);
        Ok(())
    }

    /// Do nothing, no memory to drop
    unsafe fn drop_states(&self, _state_ptrs: &[AggregationStatesPtr], _state_offset: usize) {}
}

impl<PayloadArray> UnaryAggregationState<PayloadArray>
    for SumState<<PayloadArray::Element as PayloadCast>::SumType>
where
    PayloadArray: SumPayloadArray,
    PayloadArray::Element: PayloadCast,
    for<'a> &'a PayloadArray: TryFrom<&'a ArrayImpl, Error = ArrayError>,
    for<'a> &'a mut PayloadArray::SumArray: TryFrom<&'a mut ArrayImpl, Error = ArrayError>,
{
    type Func = Sum<PayloadArray>;

    type Output = Option<<PayloadArray::Element as PayloadCast>::SumType>;

    /// FIXME: accelerate with likely
    #[inline]
    fn update(&mut self, payload_element: PayloadArray::Element) {
        let payload_element = payload_element.cast();
        match &mut self.sum {
            Some(sum) => {
                *sum += payload_element;
            }
            None => self.sum = Some(payload_element),
        }
    }

    /// FIXME: accelerate with likely
    #[inline]
    unsafe fn combine(&mut self, partial: &mut Self) {
        if let Some(partial_sum) = Option::take(&mut partial.sum) {
            match &mut self.sum {
                Some(combined_sum) => {
                    *combined_sum += partial_sum;
                }
                None => self.sum = Some(partial_sum),
            }
        }
    }

    #[inline]
    unsafe fn finalize(&mut self) -> Self::Output {
        self.sum
    }
}

impl<PayloadArray: SumPayloadArray> Sum<PayloadArray>
where
    PayloadArray::Element: PayloadCast,
{
    /// Create a new Sum function
    pub fn try_new(arg: Arc<dyn PhysicalExpr>) -> Result<Self> {
        ensure!(
            arg.as_any().downcast_ref::<FieldRef>().is_some(),
            NotFieldRefArgsSnafu {
                func: "Sum",
                args: vec![arg]
            }
        );
        ensure!(
            arg.output_type().physical_type() == PayloadArray::PHYSCIAL_TYPE,
            ArgTypeMismatchSnafu {
                func: "Sum",
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
    use std::ptr::NonNull;

    use super::*;
    use data_block::array::{Float64Array, Int32Array, Int64Array};

    #[test]
    fn test_sum_func_allocate_and_init_state() {
        let sum_func = Sum::<Int32Array> {
            args: vec![],
            _phantom: PhantomData,
        };

        unsafe {
            let ptr_ = std::alloc::alloc(sum_func.state_layout());
            let ptr = AggregationStatesPtr(NonNull::new_unchecked(ptr_));

            sum_func.init_state(ptr, 0);
            let sum_state = ptr.offset_as_mut::<SumState<i64>>(0);
            assert!(sum_state.sum.is_none());

            std::alloc::dealloc(ptr_, sum_func.state_layout())
        }
    }

    #[test]
    fn test_sum_update() {
        let mut sum_state = SumState::<i64> { sum: None };
        <SumState<i64> as UnaryAggregationState<Int32Array>>::update(&mut sum_state, 10);
        assert_eq!(sum_state.sum, Some(10));
        <SumState<i64> as UnaryAggregationState<Int32Array>>::update(&mut sum_state, 10);
        assert_eq!(sum_state.sum, Some(20));
    }

    #[test]
    fn test_sum_combine() {
        let mut s0 = SumState::<i64> { sum: None };
        let mut s1 = SumState::<i64> { sum: None };
        unsafe {
            <SumState<i64> as UnaryAggregationState<Int64Array>>::combine(&mut s0, &mut s1);
            assert_eq!(s0.sum, None);

            <SumState<i64> as UnaryAggregationState<Int64Array>>::update(&mut s1, 10);
            <SumState<i64> as UnaryAggregationState<Int64Array>>::combine(&mut s0, &mut s1);
            assert_eq!(s0.sum, Some(10));

            <SumState<i64> as UnaryAggregationState<Int64Array>>::update(&mut s1, 10);
            <SumState<i64> as UnaryAggregationState<Int64Array>>::combine(&mut s0, &mut s1);
            assert_eq!(s0.sum, Some(20));
        }
    }

    #[test]
    fn test_sum_finalize() {
        let sum_state_0 = SumState { sum: Some(10.0) };
        let sum_state_1 = SumState::<f64> { sum: None };
        let sum_state_2 = SumState::<f64> { sum: Some(-100.99) };
        unsafe {
            let ptr_0 = AggregationStatesPtr(NonNull::new_unchecked(
                (&sum_state_0) as *const _ as *mut u8,
            ));
            let ptr_1 = AggregationStatesPtr(NonNull::new_unchecked(
                (&sum_state_1) as *const _ as *mut u8,
            ));
            let ptr_2 = AggregationStatesPtr(NonNull::new_unchecked(
                (&sum_state_2) as *const _ as *mut u8,
            ));

            let ptrs = &[ptr_0, ptr_1, ptr_2];

            let mut output = ArrayImpl::new(LogicalType::Double);

            let sum = Sum::<Float64Array> {
                args: vec![],
                _phantom: PhantomData,
            };

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
}
