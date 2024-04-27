//! Min/Max aggregation function

use std::fmt::Debug;
use std::marker::PhantomData;

use data_block::array::{Array, ArrayError, ArrayImpl};
use data_block::types::{Element, LogicalType};
use snafu::ensure;

use super::{
    AggregationError, Function, SpecialOptionalUAFWrapper, SpecialOptionalUnaryAggregationFunction,
    SpecialOptionalUnaryAggregationState, Stringify,
};
use crate::common::utils::bytemuck::TransparentWrapper;
use crate::error::BangError;
use crate::exec::physical_expr::function::aggregate::ArgTypeMismatchSnafu;

/// Trait for constrain the payload array of the min/max function
pub trait MinMaxPayloadArray: Array
where
    for<'a> <Self::Element as Element>::ElementRef<'a>: Ord,
{
}

/// Blanked implementation for all of the ScalarArray whose element can compare
impl<T> MinMaxPayloadArray for T
where
    T: Array,
    for<'a> <T::Element as Element>::ElementRef<'a>: Ord,
{
}

/// Aggregation state of the min/max function
///
/// TODO: Accelerate batch update with a temp SIMD vec. In this case, we can not
/// implement TransparentWrapper for it
#[derive(Debug)]
#[repr(transparent)]
pub struct MinMaxState<const IS_MIN: bool, S>(Option<S>);

/// SAFETY: the `#[repr(transparent)]` is added for MinMaxState
unsafe impl<const IS_MIN: bool, S> TransparentWrapper<Option<S>> for MinMaxState<IS_MIN, S> {}

/// Min/Max aggregation function
///
/// # Generic
///
/// - `PayloadArray`: The type of the payload array that need to calculate min/max
/// - `IS_MIN`: If it is true, it will be min aggregation function
pub struct MinMax<const IS_MIN: bool, PayloadArray> {
    args: Vec<LogicalType>,
    _phantom: PhantomData<PayloadArray>,
}

impl<const IS_MIN: bool, PayloadArray> MinMax<IS_MIN, PayloadArray> {
    fn name_() -> &'static str {
        if IS_MIN {
            "Min"
        } else {
            "Max"
        }
    }
}

/// Min aggregation function
pub type Min<PayloadArray> = MinMax<true, PayloadArray>;
/// Max aggregation function
pub type Max<PayloadArray> = MinMax<false, PayloadArray>;

/// SAFETY: The sum function does not contain the struct, the generic type is phantom
unsafe impl<const IS_MIN: bool, PayloadArray> Send for MinMax<IS_MIN, PayloadArray> {}
unsafe impl<const IS_MIN: bool, PayloadArray> Sync for MinMax<IS_MIN, PayloadArray> {}

impl<const IS_MIN: bool, PayloadArray> Debug for MinMax<IS_MIN, PayloadArray> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct(self.name())
            .field("args", &self.args)
            .finish()
    }
}

impl<const IS_MIN: bool, PayloadArray> Stringify for MinMax<IS_MIN, PayloadArray> {
    fn name(&self) -> &'static str {
        Self::name_()
    }

    fn debug(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }

    fn display(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "fn {}({:?}) -> {:?}",
            self.name(),
            self.args[0],
            self.args[0]
        )
    }
}

impl<const IS_MIN: bool, PayloadArray> Function for MinMax<IS_MIN, PayloadArray>
where
    PayloadArray: Array,
{
    fn arguments(&self) -> &[LogicalType] {
        &self.args
    }

    fn return_type(&self) -> LogicalType {
        self.args[0].clone()
    }
}

impl<const IS_MIN: bool, PayloadArray>
    SpecialOptionalUnaryAggregationFunction<PayloadArray, PayloadArray>
    for MinMax<IS_MIN, PayloadArray>
where
    PayloadArray: MinMaxPayloadArray,
    for<'a> <PayloadArray::Element as Element>::ElementRef<'a>: Ord,
    for<'a> &'a PayloadArray: TryFrom<&'a ArrayImpl, Error = ArrayError>,
    for<'a> &'a mut PayloadArray: TryFrom<&'a mut ArrayImpl, Error = ArrayError>,
{
    type State = MinMaxState<IS_MIN, PayloadArray::Element>;
}

impl<const IS_MIN: bool, PayloadArray> SpecialOptionalUnaryAggregationState<PayloadArray>
    for MinMaxState<IS_MIN, PayloadArray::Element>
where
    PayloadArray: MinMaxPayloadArray,
    for<'a> <PayloadArray::Element as Element>::ElementRef<'a>: Ord,
    for<'a> &'a PayloadArray: TryFrom<&'a ArrayImpl, Error = ArrayError>,
    for<'a> &'a mut PayloadArray: TryFrom<&'a mut ArrayImpl, Error = ArrayError>,
{
    type InnerAggStates = PayloadArray::Element;

    type OutputElement = PayloadArray::Element;

    type Error = BangError;

    #[inline]
    fn update_inner_agg_states(
        current: &mut Self::InnerAggStates,
        payload_element: <<PayloadArray as Array>::Element as Element>::ElementRef<'_>,
    ) -> Result<(), Self::Error> {
        let current_ = <PayloadArray::Element as Element>::upcast_gat(current.as_ref());
        let payload_element_ = <PayloadArray::Element as Element>::upcast_gat(payload_element);
        if IS_MIN {
            if payload_element_ < current_ {
                current.replace_with(payload_element);
            }
        } else if payload_element_ > current_ {
            current.replace_with(payload_element);
        }

        Ok(())
    }

    #[inline]
    fn combine_inner_agg_states(
        combined: &mut Self::InnerAggStates,
        partial: Self::InnerAggStates,
    ) -> Result<(), Self::Error> {
        if IS_MIN {
            if partial.as_ref() < combined.as_ref() {
                *combined = partial;
            }
        } else if partial.as_ref() > combined.as_ref() {
            *combined = partial;
        }
        Ok(())
    }
}

impl<const IS_MIN: bool, PayloadArray> MinMax<IS_MIN, PayloadArray>
where
    PayloadArray: MinMaxPayloadArray,
    for<'a> <PayloadArray::Element as Element>::ElementRef<'a>: Ord,
    for<'a> &'a PayloadArray: TryFrom<&'a ArrayImpl, Error = ArrayError>,
    for<'a> &'a mut PayloadArray: TryFrom<&'a mut ArrayImpl, Error = ArrayError>,
{
    /// Create a new MinMax function
    pub fn try_new(
        arg: LogicalType,
    ) -> Result<SpecialOptionalUAFWrapper<Self, PayloadArray, PayloadArray>, AggregationError> {
        ensure!(
            arg.physical_type() == PayloadArray::PHYSICAL_TYPE,
            ArgTypeMismatchSnafu {
                func: Self::name_(),
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
    use crate::exec::physical_expr::function::aggregate::{
        AggregationFunction, AggregationStatesPtr,
    };
    use data_block::array::StringArray;
    use data_block::element::string::{StringElement, StringView};
    use std::ptr::NonNull;

    #[test]
    fn test_min_max_func_allocate_and_init_state() {
        let min_func = MinMax::<true, StringArray>::try_new(LogicalType::VarChar).unwrap();

        unsafe {
            let ptr_ = std::alloc::alloc(min_func.state_layout());
            let ptr = AggregationStatesPtr(NonNull::new_unchecked(ptr_));

            min_func.init_state(ptr, 0);
            let min_state = ptr.offset_as_mut::<MinMaxState<true, StringElement>>(0);
            assert!(min_state.0.is_none());

            std::alloc::dealloc(ptr_, min_func.state_layout())
        }
    }

    macro_rules! test_min_max_update {
        ($s0:expr, $s1:expr, $s2:expr, $is_min:expr, $gt0:expr, $gt1:expr) => {{
            let mut state = MinMaxState::<$is_min, StringElement>(None);
            <MinMaxState<$is_min, StringElement> as SpecialOptionalUnaryAggregationState<
                StringArray,
            >>::update(&mut state, $s0)
            .unwrap();
            assert_eq!(state.0.as_ref().map(|v| v.view()), Some($s0));

            <MinMaxState<$is_min, StringElement> as SpecialOptionalUnaryAggregationState<
                StringArray,
            >>::update(&mut state, $s1)
            .unwrap();
            assert_eq!(state.0.as_ref().map(|v| v.view()), Some($gt0));

            <MinMaxState<$is_min, StringElement> as SpecialOptionalUnaryAggregationState<
                StringArray,
            >>::update(&mut state, $s2)
            .unwrap();
            assert_eq!(state.0.as_ref().map(|v| v.view()), Some($gt1));
        }};
    }

    #[test]
    fn test_min_update() {
        let s0 = StringView::from_static_str("haha");
        let s1 = StringView::from_static_str("curvature is awesome");
        let s2 = StringView::from_static_str("rust is suitable for database");
        test_min_max_update!(s0, s1, s2, true, s1, s1)
    }

    #[test]
    fn test_max_update() {
        let s0 = StringView::from_static_str("haha");
        let s1 = StringView::from_static_str("curvature is awesome");
        let s2 = StringView::from_static_str("rust is suitable for database");
        test_min_max_update!(s0, s1, s2, false, s0, s2);
    }

    macro_rules! test_min_max_combine {
        ($view_0:expr, $view_1:expr, $view_2:expr, $is_min:expr, $gt0:expr, $gt1:expr) => {{
            let mut s0 = MinMaxState::<$is_min, StringElement>(None);
            let mut s1 = MinMaxState::<$is_min, StringElement>(None);
            <MinMaxState<$is_min, StringElement> as SpecialOptionalUnaryAggregationState<
                StringArray,
            >>::combine(&mut s0, &mut s1)
            .unwrap();
            assert!(s0.0.is_none());

            <MinMaxState<$is_min, StringElement> as SpecialOptionalUnaryAggregationState<
                StringArray,
            >>::update(&mut s0, $view_0)
            .unwrap();
            <MinMaxState<$is_min, StringElement> as SpecialOptionalUnaryAggregationState<
                StringArray,
            >>::combine(&mut s0, &mut s1)
            .unwrap();
            assert_eq!(s0.0.as_ref().map(|v| v.view()), Some($view_0));
            assert!(s1.0.is_none());

            <MinMaxState<$is_min, StringElement> as SpecialOptionalUnaryAggregationState<
                StringArray,
            >>::update(&mut s0, $view_1)
            .unwrap();
            <MinMaxState<$is_min, StringElement> as SpecialOptionalUnaryAggregationState<
                StringArray,
            >>::combine(&mut s0, &mut s1)
            .unwrap();
            assert_eq!(s0.0.as_ref().map(|v| v.view()), Some($gt0));
            assert!(s1.0.is_none());

            <MinMaxState<$is_min, StringElement> as SpecialOptionalUnaryAggregationState<
                StringArray,
            >>::update(&mut s0, $view_2)
            .unwrap();
            <MinMaxState<$is_min, StringElement> as SpecialOptionalUnaryAggregationState<
                StringArray,
            >>::combine(&mut s0, &mut s1)
            .unwrap();
            assert_eq!(s0.0.as_ref().map(|v| v.view()), Some($gt1));
            assert!(s1.0.is_none());
        }};
    }

    #[test]
    fn test_min_combine() {
        let view_0 = StringView::from_static_str("abcdefg");
        let view_1 = StringView::from_static_str("aaa");
        let view_2 = StringView::from_static_str("writing the db in Rust");
        test_min_max_combine!(view_0, view_1, view_2, true, view_1, view_1);
    }

    #[test]
    fn test_max_combine() {
        let view_0 = StringView::from_static_str("abcdefg");
        let view_1 = StringView::from_static_str("aaa");
        let view_2 = StringView::from_static_str("writing the db in Rust");
        test_min_max_combine!(view_0, view_1, view_2, false, view_0, view_2);
    }
}
