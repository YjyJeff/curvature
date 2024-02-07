//! Min/Max aggregation function

use std::alloc::Layout;
use std::cmp::PartialOrd;
use std::fmt::Debug;
use std::marker::PhantomData;
use std::sync::Arc;

use data_block::array::{Array, ArrayError, ArrayImpl, ScalarArray};
use data_block::types::{Element, ElementRef, LogicalType};

use super::{
    unary_combine_states, unary_finalize_states, unary_update_states, AggregationFunction,
    AggregationStatesPtr, Function, Result, Stringify, UnaryAggregationState,
};
use crate::exec::physical_expr::PhysicalExpr;

/// Trait for constrain the payload array of the min/max function
///
/// TBD: Can we constrain the array with ScalarArray?
pub trait MinMaxPayloadArray: ScalarArray
where
    for<'a> <Self::Element as Element>::ElementRef<'a>: PartialOrd,
{
}

/// Blanked implementation for all of the ScalarArray whose element can compare
impl<T> MinMaxPayloadArray for T
where
    T: ScalarArray,
    for<'a> <T::Element as Element>::ElementRef<'a>: PartialOrd,
{
}

/// Aggregation state of the min/max function
#[derive(Debug)]
#[repr(transparent)]
pub struct MinMaxState<const IS_MIN: bool, S> {
    state: Option<S>,
}

/// Min/Max aggregation function
///
/// FIXME: If we compute the min/max for the float array that contains `NaN`, the result
/// will be undetermined
///
/// # Generic
///
/// - `PayloadArray`: The type of the payload array that need to calculate min/max
/// - `IS_MIN`: If it is true, it will be min aggregation function
pub struct MinMax<const IS_MIN: bool, PayloadArray> {
    args: Vec<Arc<dyn PhysicalExpr>>,
    _phantom: PhantomData<PayloadArray>,
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
        if IS_MIN {
            "Min"
        } else {
            "Max"
        }
    }

    fn debug(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }

    fn display(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())?;
        self.args[0].display(f, true)?;
        write!(f, ")")
    }
}

impl<const IS_MIN: bool, PayloadArray> Function for MinMax<IS_MIN, PayloadArray>
where
    PayloadArray: Array,
{
    fn arguments(&self) -> &[Arc<dyn PhysicalExpr>] {
        &self.args
    }

    fn return_type(&self) -> LogicalType {
        self.args[0].output_type().to_owned()
    }
}

impl<const IS_MIN: bool, PayloadArray> AggregationFunction for MinMax<IS_MIN, PayloadArray>
where
    PayloadArray: MinMaxPayloadArray,
    for<'a> <PayloadArray::Element as Element>::ElementRef<'a>: PartialOrd,
    for<'a> &'a PayloadArray: TryFrom<&'a ArrayImpl, Error = ArrayError>,
    for<'a> &'a mut PayloadArray: TryFrom<&'a mut ArrayImpl, Error = ArrayError>,
{
    fn state_layout(&self) -> Layout {
        Layout::new::<MinMaxState<IS_MIN, PayloadArray::Element>>()
    }

    unsafe fn init_state(&self, ptr: AggregationStatesPtr, state_offset: usize) {
        let state = ptr.offset_as_mut::<MinMaxState<IS_MIN, PayloadArray::Element>>(state_offset);
        // Using write here to avoid drop the uninitiated value, especially for `Vec<u8>`  and `String`
        std::ptr::write(state, MinMaxState { state: None });
    }

    unsafe fn update_states(
        &self,
        payloads: &[&data_block::array::ArrayImpl],
        state_ptrs: &[AggregationStatesPtr],
        state_offset: usize,
    ) -> Result<()> {
        unary_update_states::<PayloadArray, MinMaxState<IS_MIN, PayloadArray::Element>>(
            payloads,
            state_ptrs,
            state_offset,
        );
        Ok(())
    }

    unsafe fn combine_states(
        &self,
        partial_state_ptrs: &[AggregationStatesPtr],
        combined_state_ptrs: &[AggregationStatesPtr],
        state_offset: usize,
    ) {
        unary_combine_states::<PayloadArray, MinMaxState<IS_MIN, PayloadArray::Element>>(
            partial_state_ptrs,
            combined_state_ptrs,
            state_offset,
        );
    }

    unsafe fn finalize(
        &self,
        state_ptrs: &[AggregationStatesPtr],
        state_offset: usize,
        output: &mut ArrayImpl,
    ) -> Result<()> {
        unary_finalize_states::<
            PayloadArray,
            PayloadArray,
            MinMaxState<IS_MIN, PayloadArray::Element>,
        >(state_ptrs, state_offset, output);
        Ok(())
    }
}

impl<const IS_MIN: bool, PayloadArray> UnaryAggregationState<PayloadArray>
    for MinMaxState<IS_MIN, PayloadArray::Element>
where
    PayloadArray: MinMaxPayloadArray,
    for<'a> <PayloadArray::Element as Element>::ElementRef<'a>: PartialOrd,
    for<'a> &'a PayloadArray: TryFrom<&'a ArrayImpl, Error = ArrayError>,
    for<'a> &'a mut PayloadArray: TryFrom<&'a mut ArrayImpl, Error = ArrayError>,
{
    type Func = MinMax<IS_MIN, PayloadArray>;

    type Output = Option<PayloadArray::Element>;

    /// FIXME: accelerate with likely
    #[inline]
    fn update(
        &mut self,
        payload_element: <<PayloadArray as Array>::Element as Element>::ElementRef<'_>,
    ) {
        if let Some(current) = &mut self.state {
            let current_ = <PayloadArray::Element as Element>::upcast_gat(current.as_ref());
            let payload_element_ = <PayloadArray::Element as Element>::upcast_gat(payload_element);
            if IS_MIN {
                if payload_element_ < current_ {
                    current.replace_with(payload_element);
                }
            } else if payload_element_ > current_ {
                current.replace_with(payload_element);
            }
        } else {
            self.state = Some(payload_element.to_owned());
        }
    }

    /// FIXME: accelerate with likely
    #[inline]
    unsafe fn combine(&mut self, partial: &mut Self) {
        if let Some(partial) = partial.state.take() {
            match &mut self.state {
                Some(combined) => {
                    if IS_MIN {
                        if partial.as_ref() < combined.as_ref() {
                            self.state = Some(partial);
                        }
                    } else if partial.as_ref() > combined.as_ref() {
                        self.state = Some(partial);
                    }
                }
                None => self.state = Some(partial),
            }
        }
    }

    #[inline]
    unsafe fn finalize(&mut self) -> Self::Output {
        self.state.take()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use data_block::array::StringArray;
    use data_block::element::string::{StringElement, StringView};
    use std::ptr::NonNull;

    #[test]
    fn test_min_max_func_allocate_and_init_state() {
        let min_func = MinMax::<true, StringArray> {
            args: vec![],
            _phantom: PhantomData,
        };

        unsafe {
            let ptr_ = std::alloc::alloc(min_func.state_layout());
            let ptr = AggregationStatesPtr(NonNull::new_unchecked(ptr_));

            min_func.init_state(ptr, 0);
            let min_state = ptr.offset_as_mut::<MinMaxState<true, StringElement>>(0);
            assert!(min_state.state.is_none());

            std::alloc::dealloc(ptr_, min_func.state_layout())
        }
    }

    macro_rules! test_min_max_update {
        ($s0:expr, $s1:expr, $s2:expr, $is_min:expr, $gt0:expr, $gt1:expr) => {{
            let mut state = MinMaxState::<$is_min, StringElement> { state: None };
            <MinMaxState<$is_min, StringElement> as UnaryAggregationState<StringArray>>::update(
                &mut state, $s0,
            );
            assert_eq!(state.state.as_ref().map(|v| v.view()), Some($s0));

            <MinMaxState<$is_min, StringElement> as UnaryAggregationState<StringArray>>::update(
                &mut state, $s1,
            );
            assert_eq!(state.state.as_ref().map(|v| v.view()), Some($gt0));

            <MinMaxState<$is_min, StringElement> as UnaryAggregationState<StringArray>>::update(
                &mut state, $s2,
            );
            assert_eq!(state.state.as_ref().map(|v| v.view()), Some($gt1));
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
            let mut s0 = MinMaxState::<$is_min, StringElement> { state: None };
            let mut s1 = MinMaxState::<$is_min, StringElement> { state: None };
            unsafe {
                <MinMaxState<$is_min, StringElement> as UnaryAggregationState<StringArray>>::combine(
                    &mut s0, &mut s1,
                );
                assert!(s0.state.is_none());

                <MinMaxState<$is_min, StringElement> as UnaryAggregationState<StringArray>>::update(
                    &mut s0, $view_0,
                );
                <MinMaxState<$is_min, StringElement> as UnaryAggregationState<StringArray>>::combine(
                    &mut s0, &mut s1,
                );
                assert_eq!(s0.state.as_ref().map(|v| v.view()), Some($view_0));
                assert!(s1.state.is_none());

                <MinMaxState<$is_min, StringElement> as UnaryAggregationState<StringArray>>::update(
                    &mut s0, $view_1,
                );
                <MinMaxState<$is_min, StringElement> as UnaryAggregationState<StringArray>>::combine(
                    &mut s0, &mut s1,
                );
                assert_eq!(s0.state.as_ref().map(|v| v.view()), Some($gt0));
                assert!(s1.state.is_none());

                <MinMaxState<$is_min, StringElement> as UnaryAggregationState<StringArray>>::update(
                    &mut s0, $view_2,
                );
                <MinMaxState<$is_min, StringElement> as UnaryAggregationState<StringArray>>::combine(
                    &mut s0, &mut s1,
                );
                assert_eq!(s0.state.as_ref().map(|v| v.view()), Some($gt1));
                assert!(s1.state.is_none());
            }
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
