//! Aggregate functions

pub mod count;
pub mod min_max;
pub mod sum;

use std::alloc::{alloc, dealloc, handle_alloc_error, Layout};
use std::fmt::{Debug, Display};
use std::ptr::NonNull;
use std::sync::Arc;

use data_block::array::{Array, ArrayError, ArrayImpl, ScalarArray};
use data_block::types::{Element, LogicalType, PhysicalType};
use snafu::Snafu;

use super::Function;
use crate::common::utils::memory::next_multiple_of_align;
use crate::error::SendableError;
use crate::exec::physical_expr::utils::display_agg_funcs;
use crate::exec::physical_expr::PhysicalExpr;
use crate::exec::physical_operator::aggregate::Arena;

#[allow(missing_docs)]
#[derive(Debug, Snafu)]
pub enum AggregationError {
    #[snafu(display("All of the arguments passed to the `{func}` aggregation function should be `FieldRef`, found arguments `{:?}`", args))]
    NotFieldRefArgs {
        func: &'static str,
        args: Vec<Arc<dyn PhysicalExpr>>,
    },
    #[snafu(display(
        "`{func}` aggregation function expect `{}`, however the arg has logical type `{:?}` with `{}`",
        expect_physical_type,
        arg_type,
        arg_type.physical_type()
    ))]
    ArgTypeMismatch {
        func: &'static str,
        expect_physical_type: PhysicalType,
        arg_type: LogicalType,
    },
    #[snafu(display("Failed to update the aggregation states of the `{func}`"))]
    UpdateStates {
        func: &'static str,
        source: SendableError,
    },
    #[snafu(display("Failed to finalize the aggregation states of the `{func}`"))]
    FinalizeStates {
        func: &'static str,
        source: SendableError,
    },
}

/// Aggregation result
pub type Result<T> = std::result::Result<T, AggregationError>;

/// Pointer that points to the `AggregationStates`
///
/// For each aggregation function, we need an `AggregationState` to store its state across
/// different data blocks. When we need to compute multiple aggregation functions over
/// same input, the naive way is allocating multiple `AggregationState`s individually.
/// However, it is not optimal. Firstly, we do not know how many functions will be used,
/// we need a `Vec` to store these `AggregationState`. Secondly, allocating these states
/// individually is slow, even if we use arena. In most of the cases, these states is
/// pretty small, we need to do lots of small memory allocation. Therefore, we combine
/// multiple `AggregationState`s into a single struct(`AggregationStates`) to solve
/// the above problem. However, we can not define it in the compiler time 😂!!! It is
/// totally dynamic, different combination of aggregation functions can cause different
/// `AggregationStates`. Therefore, we need to generate it dynamically!
///
/// To dynamically generate a struct, we only need to compute the [`Layout`] and its
/// field offsets. We call it [`AggregationStatesLayout`]. Using this layout, we can
/// allocating the struct and access its field with pointer arithmetic. The
/// [`AggregationStatesPtr`] is the pointer that points to this dynamic struct! As
/// we can see, using this pointer is totally **unsafe** !!! Caller should guarantee
/// the layout that used to generate this pointer and the layout that used to access
/// the field should be identical. Otherwise, undefined behavior happens!!
#[derive(Debug, Clone, Copy)]
#[repr(transparent)]
pub struct AggregationStatesPtr(NonNull<u8>);

impl AggregationStatesPtr {
    /// View the ptr.add(offset) as &mut T
    #[inline]
    unsafe fn offset_as_mut<'a, T>(self, offset: usize) -> &'a mut T {
        &mut *(self.0.as_ptr().add(offset) as *mut T)
    }

    /// Create a new dangling pointer
    #[inline]
    pub fn dangling() -> Self {
        AggregationStatesPtr(NonNull::dangling())
    }
}

/// Layout of the `AggregationStates`, see [`AggregationStatesPtr`] for details
#[derive(Debug)]
pub struct AggregationStatesLayout {
    /// Layout of the combined aggregation states
    layout: Layout,
    /// Offsets of each aggregation state in the dynamic struct
    states_offsets: Vec<usize>,
}

impl AggregationStatesLayout {
    fn new(funcs: &[Arc<dyn AggregationFunction>]) -> Self {
        let mut align = 8;
        let mut size = 0;
        let mut states_offsets = Vec::with_capacity(funcs.len());

        // Compute the size and alignment
        funcs.iter().for_each(|func| {
            let state_layout = func.state_layout();
            let state_align = state_layout.align();

            // Start offset of the state is multiple of the alignment
            size = next_multiple_of_align(size, state_align);
            states_offsets.push(size);
            size += state_layout.size();
            align = std::cmp::max(align, state_align);
        });

        size = next_multiple_of_align(size, align);
        // SAFETY: align is power of two
        Self {
            layout: unsafe { Layout::from_size_align_unchecked(size, align) },
            states_offsets,
        }
    }
}

/// A self-contained list of aggregation functions. It knows how to allocate
/// and access the `AggregationStates`
#[derive(Debug)]
pub struct AggregationFunctionList {
    /// List of the aggregation functions
    funcs: Vec<Arc<dyn AggregationFunction>>,
    /// Layout of the `AggregationStates`
    states_layout: AggregationStatesLayout,
    /// Init state of the `AggregationStates`, used to init the allocated memory
    /// in the arena
    ///
    /// Note that we can not use `Vec<u8>` here, because `Vec<u8>`'s alignment is 1.
    /// We have to allocate and free the memory by ourself 😂
    init_states: NonNull<u8>,
}

unsafe impl Send for AggregationFunctionList {}
unsafe impl Sync for AggregationFunctionList {}

impl AggregationFunctionList {
    /// Create a new [`AggregationFunctionList`]
    pub(crate) fn new(funcs: Vec<Arc<dyn AggregationFunction>>) -> Self {
        debug_assert!(!funcs.is_empty());

        let states_layout = AggregationStatesLayout::new(&funcs);

        // Create the init state
        let init_states = unsafe {
            NonNull::new(alloc(states_layout.layout))
                .unwrap_or_else(|| handle_alloc_error(states_layout.layout))
        };
        // SAFETY: pointer to the allocated Vec<u8> is guaranteed to be NonNull
        let ptr = AggregationStatesPtr(init_states);
        funcs
            .iter()
            .zip(states_layout.states_offsets.iter())
            .for_each(|(func, &state_offset)| unsafe {
                func.init_state(ptr, state_offset);
            });

        Self {
            states_layout,
            funcs,
            init_states,
        }
    }

    /// Allocate the initialized `AggregationStates` in the arena, return the pointer to it
    #[inline]
    pub(crate) fn alloc_states(&self, arena: &Arena) -> AggregationStatesPtr {
        let ptr = arena.alloc_layout(self.states_layout.layout);
        // Init the states
        // SAFETY: the init_states and ptr do not overlap
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.init_states.as_ptr(),
                ptr.as_ptr(),
                self.states_layout.layout.size(),
            );
        }
        AggregationStatesPtr(ptr)
    }

    /// Update the states in the arena
    ///
    /// # Safety
    ///
    /// - Payloads should match the signature of `self.funcs`
    /// - `ptr` is a valid pointer
    pub(crate) unsafe fn update_states(
        &self,
        payloads: &[&ArrayImpl],
        state_ptrs: &[AggregationStatesPtr],
    ) -> Result<()> {
        let mut payload_index = 0;
        self.funcs
            .iter()
            .zip(self.states_layout.states_offsets.iter())
            .try_for_each(|(func, &state_offset)| {
                let old = payload_index;
                payload_index += func.arguments().len();
                let func_payloads = &payloads[old..payload_index];

                func.update_states(func_payloads, state_ptrs, state_offset)
            })
    }

    /// Combine the partial states into combined states
    /// # Safety
    ///
    /// - Both `ptrs` should be valid pointer
    /// - Args should have same length
    pub(crate) unsafe fn combine_states(
        &self,
        partial_state_ptrs: &[AggregationStatesPtr],
        combined_state_ptrs: &[AggregationStatesPtr],
    ) -> Result<()> {
        self.funcs
            .iter()
            .zip(&self.states_layout.states_offsets)
            .try_for_each(|(func, &state_offset)| {
                func.combine_states(partial_state_ptrs, combined_state_ptrs, state_offset)
            })
    }

    /// Take the states in the arena,
    ///
    /// # Safety
    ///
    /// - `output` match aggregation functions signature
    /// - `ptr` should be valid pointers
    pub(crate) unsafe fn take_states(
        &self,
        state_ptrs: &[AggregationStatesPtr],
        output: &mut [ArrayImpl],
    ) -> Result<()> {
        debug_assert_eq!(self.funcs.len(), output.len());

        self.funcs
            .iter()
            .zip(&self.states_layout.states_offsets)
            .zip(output)
            .try_for_each(|((func, &state_offset), output)| {
                // SAFETY: the state_offset is guaranteed by the constructor, the output guaranteed by the caller
                func.take_states(state_ptrs, state_offset, output)
            })
    }

    /// Drop the states allocated in the state_ptrs
    pub(crate) unsafe fn drop_states(&self, state_ptrs: &[AggregationStatesPtr]) {
        self.funcs
            .iter()
            .zip(&self.states_layout.states_offsets)
            .for_each(|(func, &state_offset)| func.drop_states(state_ptrs, state_offset));
    }
}

impl Drop for AggregationFunctionList {
    fn drop(&mut self) {
        // Deallocate the init state. The init states is guaranteed to be non-null
        unsafe { dealloc(self.init_states.as_ptr(), self.states_layout.layout) }
    }
}

impl Display for AggregationFunctionList {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        display_agg_funcs(f, &self.funcs)
    }
}

/// Stringify the aggregation function
pub trait Stringify {
    /// Get name of the aggregation function
    fn name(&self) -> &'static str;

    /// Debug message
    fn debug(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result;

    /// Display message
    ///
    /// If `compact` is true, use one line representation for each expression.
    /// Otherwise, prints a tree of expressions one node per line
    fn display(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result;
}

/// Trait for all of the aggregation functions
///
/// # Note
///
/// All of the arguments passed to `AggregationFunction` should be [`FieldRef`]!
/// Through this way, we guarantee different aggregation functions do not need to do
/// redundant expression computation. We enforce the optimizer/planner to identify the
/// redundant expression.
///
/// [`FieldRef`]: crate::exec::physical_expr::field_ref::FieldRef
pub trait AggregationFunction: Function + Stringify {
    /// Layout of the aggregation state, used to compute the dynamic `AggregationStates`
    fn state_layout(&self) -> Layout;

    /// Initialize the allocated state
    ///
    /// # Safety
    ///
    /// - `ptr.add(state_offset)` is valid and points to the state of the function
    ///
    /// - the state is allocated but uninitiated, therefore init it should be careful!
    /// Especially for state that contains memory allocation, you should use
    /// `std::ptr::write` to override the memory and do not drop the old value! Otherwise,
    /// dropping the uninitiated value may cause core dumped!
    unsafe fn init_state(&self, ptr: AggregationStatesPtr, state_offset: usize);

    /// Update the states of this aggregation function based on payloads of the function
    ///
    /// # Arguments
    ///
    /// - `payloads`: payloads this function accept, caller should guarantee it matches
    /// the function's signature
    /// - `state_ptrs`: state pointers that each element in the payload should update
    /// - `state_offset`: offset, in bytes, of this aggregation's state in the `AggregationStates`
    ///
    /// # Safety
    ///
    /// - `payloads` match the function's signature
    /// - `ptr.add(state_offset)` is valid and points to the state of the function
    unsafe fn update_states(
        &self,
        payloads: &[&ArrayImpl],
        state_ptrs: &[AggregationStatesPtr],
        state_offset: usize,
    ) -> Result<()>;

    /// Combine the partial(thread-local) state into combined state
    ///
    /// # Arguments
    ///
    /// - `partial_state_ptrs`: pointers to the partial `AggregationStates`
    /// - `combined_state_ptrs`: pointers to the combined `AggregationStates`
    /// - `state_offset`: offset, in bytes, of this aggregation's state in the `AggregationStates`
    ///
    /// # Safety
    ///
    /// - `ptr.add(state_offset)` is valid and points to the state of the function
    ///
    /// - `partial_state_ptrs` and `combined_state_ptrs` have same length
    ///
    /// - Implementation should free the memory occupied by the `partial-state` that does not
    /// in the arena
    unsafe fn combine_states(
        &self,
        partial_state_ptrs: &[AggregationStatesPtr],
        combined_state_ptrs: &[AggregationStatesPtr],
        state_offset: usize,
    ) -> Result<()>;

    /// Take the aggregation states, write the result of the state into the output array
    ///
    /// # TBD
    ///
    /// This API will cause iterating the hash table two times because it separate the
    /// key and aggregation state. Can we combine them ?
    ///
    /// # Safety
    ///
    /// - `ptr.add(state_offset)` is valid and points to the state of the function
    /// - `output` should match function's signature
    /// - Implementation should free the memory occupied by the state that does not in the
    /// arena
    unsafe fn take_states(
        &self,
        state_ptrs: &[AggregationStatesPtr],
        state_offset: usize,
        output: &mut ArrayImpl,
    ) -> Result<()>;

    /// Drop the states allocated in the `state_ptrs`
    ///
    /// # Safety
    ///
    /// - `ptrs` should be valid
    unsafe fn drop_states(&self, state_ptrs: &[AggregationStatesPtr], state_offset: usize);
}

impl Debug for dyn AggregationFunction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.debug(f)
    }
}

/// Trait for the aggregation state that take unary payload. We do no require all of
/// the unary aggregation state implement this trait. For example, `CountState` does
/// not implement it because it does not care the concrete payload type
///
/// # Generic
///
/// - `PayloadArray`: Array of the payload type
pub trait UnaryAggregationState<PayloadArray: Array> {
    /// Aggregation function
    type Func: AggregationFunction;
    /// Output type of the aggregation state
    type Output;

    /// Update the state based on the element in the payload
    fn update(&mut self, payload_element: <PayloadArray::Element as Element>::ElementRef<'_>);

    /// Combine the unary aggregation state
    ///
    /// # Safety
    ///
    /// Implementation should free the memory occupied by the `partial_state` that does not in arena
    unsafe fn combine(&mut self, partial: &mut Self);

    /// Consume the aggregation state, return the output of the aggregation state
    ///
    /// # Safety
    ///
    /// Implementation should free the memory occupied by the state that does not in arena
    unsafe fn finalize(&mut self) -> Self::Output;
}

/// Update the unary states pointers based on the element in the payload.
///
/// Note that we assume the state does not take `NULL` into consideration
///
/// # Arguments
///
/// - `payload`: payload the function accept. It should have same length with `state_ptrs`
/// - `state_ptrs`: state pointers that each element in the payload should update
/// - `state_offset`: offset, in bytes, of this aggregation's state in the `AggregationStates`
///
/// # Generic
///
/// - `PayloadArray`: Array of the payload type
/// - `S`: Unary aggregation state type
#[inline]
unsafe fn unary_update_states<PayloadArray, S>(
    payloads: &[&ArrayImpl],
    state_ptrs: &[AggregationStatesPtr],
    state_offset: usize,
) where
    PayloadArray: Array,
    S: UnaryAggregationState<PayloadArray>,
    for<'a> &'a PayloadArray: TryFrom<&'a ArrayImpl, Error = ArrayError>,
{
    let payload = payloads[0];

    let payload: &PayloadArray = payload
        .try_into()
        .expect("Unary aggregation's payload array should match its signature");

    let validity = payload.validity();
    if validity.is_empty() {
        // All of the element in the payload is not null
        payload
            .values_iter()
            .zip(state_ptrs)
            .for_each(|(payload_element, ptr)| {
                ptr.offset_as_mut::<S>(state_offset).update(payload_element)
            });
    } else {
        // Some elements are null
        payload
            .values_iter()
            .zip(validity.iter())
            .zip(state_ptrs)
            .for_each(|((payload_element, not_null), ptr)| {
                if not_null {
                    ptr.offset_as_mut::<S>(state_offset).update(payload_element)
                }
            })
    }
}

#[inline]
unsafe fn unary_combine_states<PayloadArray, S>(
    partial_state_ptrs: &[AggregationStatesPtr],
    combined_state_ptrs: &[AggregationStatesPtr],
    state_offset: usize,
) where
    PayloadArray: Array,
    S: UnaryAggregationState<PayloadArray>,
{
    combined_state_ptrs
        .iter()
        .zip(partial_state_ptrs)
        .for_each(|(combined, partial)| {
            let combined = combined.offset_as_mut::<S>(state_offset);
            let partial = partial.offset_as_mut::<S>(state_offset);
            combined.combine(partial)
        });
}

#[inline]
unsafe fn unary_take_states<PayloadArray, OutputArray, S>(
    state_ptrs: &[AggregationStatesPtr],
    state_offset: usize,
    output: &mut ArrayImpl,
) where
    PayloadArray: Array,
    OutputArray: ScalarArray,
    S: UnaryAggregationState<PayloadArray, Output = Option<OutputArray::Element>>,
    for<'a> &'a mut OutputArray: TryFrom<&'a mut ArrayImpl, Error = ArrayError>,
{
    let output: &mut OutputArray = output
        .try_into()
        .expect("Output of the unary aggregation should match its signature");

    let trusted_len_iterator = state_ptrs.iter().map(|ptr| {
        let state = ptr.offset_as_mut::<S>(state_offset);
        state.finalize()
    });

    output.replace_with_trusted_len_iterator(state_ptrs.len(), trusted_len_iterator)
}
