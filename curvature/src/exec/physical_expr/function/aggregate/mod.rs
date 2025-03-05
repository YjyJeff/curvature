//! Aggregate functions

pub mod avg;
pub mod count;
pub mod min_max;
pub mod sum;

use std::alloc::{Layout, alloc, dealloc, handle_alloc_error};
#[cfg(not(feature = "likely"))]
use std::convert::identity as likely;
use std::fmt::{Debug, Display};
#[cfg(feature = "likely")]
use std::intrinsics::likely;

use std::marker::PhantomData;
use std::ptr::NonNull;
use std::sync::Arc;

use data_block::array::iter::ArrayNonNullValuesIter;
use data_block::array::{Array, ArrayError, ArrayImpl};
use data_block::types::{Element, ElementRef, LogicalType, PhysicalType};
use data_block::utils::roundup_to_multiple_of_pow_of_two_base;
use snafu::{ResultExt, Snafu, ensure};

use super::Function;
use crate::common::utils::bytemuck::TransparentWrapper;
use crate::error::SendableError;
use crate::exec::physical_expr::field_ref::FieldRef;
use crate::exec::physical_expr::utils::display_agg_funcs;
use crate::exec::physical_operator::aggregate::Arena;

#[allow(missing_docs)]
#[derive(Debug, Snafu)]
pub enum AggregationError {
    #[snafu(display(
        "`{func}` aggregation function expect `{}` payloads, however, `{}` payloads are accepted",
        expect_payloads_size,
        payloads_size,
    ))]
    PayloadsSizeMismatch {
        func: &'static str,
        expect_payloads_size: usize,
        payloads_size: usize,
    },
    #[snafu(display(
        "`{func}` aggregation function expect the payloads type: `{:?}`, however, `{:?}` payloads are accepted",
        expect_payloads_types,
        payloads_types
    ))]
    PayloadsTypeMismatch {
        func: &'static str,
        expect_payloads_types: Vec<LogicalType>,
        payloads_types: Vec<LogicalType>,
    },
    #[snafu(display(
        "`{func}` aggregation function expect `{}`, however the arg has logical type `{:?}` with `{}`",
        expect_physical_type,
        arg,
        arg.physical_type()
    ))]
    ArgTypeMismatch {
        func: &'static str,
        expect_physical_type: PhysicalType,
        arg: LogicalType,
    },
    #[snafu(display("Failed to update the aggregation states"))]
    UpdateStates { source: SendableError },
    #[snafu(display("Failed to update the aggregation states with batch"))]
    BatchUpdateStates { source: SendableError },
    #[snafu(display("Failed to combine the aggregation states"))]
    CombineStates { source: SendableError },
}

/// Aggregation result
pub type Result<T> = std::result::Result<T, AggregationError>;

/// Expression for aggregation function
///
/// We enforce the payloads to be [`FieldRef`], such that different aggregation functions
/// do not need to do redundant expression computation. We enforce the optimizer/planner
/// to identify the redundant expression.
#[derive(Debug)]
pub struct AggregationFunctionExpr<'a> {
    /// payloads of the aggregation function
    pub(crate) payloads: &'a [FieldRef],
    /// aggregation function
    pub(crate) func: Arc<dyn AggregationFunction>,
    /// Private field used to avoid constructing the struct literally
    _private: (),
}

impl<'a> AggregationFunctionExpr<'a> {
    /// Try to create an aggregation function expression
    pub fn try_new(payloads: &'a [FieldRef], func: Arc<dyn AggregationFunction>) -> Result<Self> {
        let arguments = func.arguments();
        ensure!(
            payloads.len() == arguments.len(),
            PayloadsSizeMismatchSnafu {
                func: func.name(),
                expect_payloads_size: arguments.len(),
                payloads_size: payloads.len()
            }
        );

        ensure!(
            payloads
                .iter()
                .zip(arguments)
                .all(|(payload, arg)| payload.output_type.eq(arg)),
            PayloadsTypeMismatchSnafu {
                func: func.name(),
                expect_payloads_types: arguments.to_vec(),
                payloads_types: payloads
                    .iter()
                    .map(|p| p.output_type.clone())
                    .collect::<Vec<_>>()
            }
        );

        Ok(Self {
            payloads,
            func,
            _private: (),
        })
    }
}

/// Shared pointer that points to the `AggregationStates`
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
///
/// # Drop
///
/// The allocated memory should either be `combined`` into other allocated memory through
/// [`AggregationFunctionList::combine_states()`], `taken` into the output array through
/// [`AggregationFunctionList::take_states()`] or `dropped` through
/// [`AggregationFunctionList::drop_states()`], such that we could free the memory occupied
/// by the state
#[derive(Debug, Clone, Copy)]
#[repr(transparent)]
pub struct AggregationStatesPtr(NonNull<u8>);

impl AggregationStatesPtr {
    /// View the ptr.add(offset) as &mut T
    #[inline]
    unsafe fn offset_as_mut<'a, T>(self, offset: usize) -> &'a mut T {
        unsafe { &mut *(self.0.as_ptr().add(offset).cast::<T>()) }
    }

    /// Create a new dangling pointer
    #[inline]
    pub fn dangling() -> Self {
        AggregationStatesPtr(NonNull::dangling())
    }
}

unsafe impl Send for AggregationStatesPtr {}

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
            size = roundup_to_multiple_of_pow_of_two_base(size, state_align);
            states_offsets.push(size);
            size += state_layout.size();
            align = std::cmp::max(align, state_align);
        });

        size = roundup_to_multiple_of_pow_of_two_base(size, align);
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

    /// Init the allocated pointer
    #[inline]
    unsafe fn init(&self, ptr: NonNull<u8>) -> AggregationStatesPtr {
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.init_states.as_ptr(),
                ptr.as_ptr(),
                self.states_layout.layout.size(),
            );
            AggregationStatesPtr(ptr)
        }
    }

    /// Allocate the initialized `AggregationStates` in the arena, return the pointer to it
    #[inline]
    pub(crate) fn alloc_states_in_arena(&self, arena: &Arena) -> AggregationStatesPtr {
        let ptr = arena.alloc_layout(self.states_layout.layout);
        unsafe { self.init(ptr) }
    }

    /// Allocate the initialized `AggregationStates` in the global memory allocator
    #[inline]
    pub(crate) fn alloc_states(&self) -> AggregationStatesPtr {
        unsafe {
            let ptr = NonNull::new(alloc(self.states_layout.layout))
                .unwrap_or_else(|| handle_alloc_error(self.states_layout.layout));
            self.init(ptr)
        }
    }

    /// Deallocate the given `AggregationStates` that allocated in the global memory allocator
    ///
    /// # Safety
    ///
    /// The `ptr` must be allocated through the [`Self::alloc_states()`] method
    /// and have never been dropped/deallocated before
    #[inline]
    pub(crate) unsafe fn dealloc_states(&self, states_ptr: AggregationStatesPtr) {
        unsafe {
            // Before deallocation, we need to drop the states
            self.drop_states(&[states_ptr]);
            // Deallocate the ptr from the global memory allocator
            dealloc(states_ptr.0.as_ptr(), self.states_layout.layout)
        }
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
        unsafe {
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
    }

    /// Update the states with batch
    ///
    /// # Safety
    ///
    /// - `payloads` match the function's signature and all of them have same length that
    ///   equal to the `len` arg
    /// - `ptr` is a valid pointer
    pub(crate) unsafe fn batch_update_states(
        &self,
        len: usize,
        payloads: &[&ArrayImpl],
        states_ptr: AggregationStatesPtr,
    ) -> Result<()> {
        unsafe {
            let mut payload_index = 0;
            self.funcs
                .iter()
                .zip(self.states_layout.states_offsets.iter())
                .try_for_each(|(func, &state_offset)| {
                    let old = payload_index;
                    payload_index += func.arguments().len();
                    let func_payloads = &payloads[old..payload_index];
                    func.batch_update_states(len, func_payloads, states_ptr, state_offset)
                })
        }
    }

    /// Combine the partial states into combined states
    ///
    /// # Safety
    ///
    /// - Both `ptrs` should be valid pointer
    /// - Args should have same length
    pub unsafe fn combine_states(
        &self,
        partial_state_ptrs: &[AggregationStatesPtr],
        combined_state_ptrs: &[AggregationStatesPtr],
    ) -> Result<()> {
        unsafe {
            self.funcs
                .iter()
                .zip(&self.states_layout.states_offsets)
                .try_for_each(|(func, &state_offset)| {
                    func.combine_states(partial_state_ptrs, combined_state_ptrs, state_offset)
                })
        }
    }

    /// Take the states in the arena,
    ///
    /// # Safety
    ///
    /// - `output` match aggregation functions signature
    /// - `ptr` should be valid pointers
    pub unsafe fn take_states(
        &self,
        state_ptrs: &[AggregationStatesPtr],
        output: &mut [ArrayImpl],
    ) -> Result<()> {
        unsafe {
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
    }

    /// Drop the states allocated in the state_ptrs
    ///
    /// # Safety
    ///
    /// - `ptr` should be valid pointers and have never been dropped before
    pub unsafe fn drop_states(&self, state_ptrs: &[AggregationStatesPtr]) {
        unsafe {
            self.funcs
                .iter()
                .zip(&self.states_layout.states_offsets)
                .for_each(|(func, &state_offset)| func.drop_states(state_ptrs, state_offset));
        }
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
    fn display(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result;
}

/// Trait for all of the aggregation functions
///
/// # TODO
///
/// Put the variable length states into an individual arena?
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
    ///   Especially for state that contains memory allocation, you should use
    ///   `std::ptr::write` to override the memory and do not drop the old value! Otherwise,
    ///   dropping the uninitiated value may cause core dumped!
    unsafe fn init_state(&self, ptr: AggregationStatesPtr, state_offset: usize);

    /// Update the states of this aggregation function based on payloads of the function
    ///
    /// # Arguments
    ///
    /// - `payloads`: payloads this function accept, caller should guarantee it matches
    ///   the function's signature
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

    /// Batch update a single state of the aggregation function based on payload of the function
    ///
    /// # Arguments
    ///
    /// - `payloads`: payloads this function accept, caller should guarantee it matches
    ///   the function's signature
    /// - `states_ptr`: States pointer that need to be updated
    /// - `state_offset`: offset, in bytes, of this aggregation's state in the `AggregationStates`
    ///
    /// # Safety
    ///
    /// - `payloads` match the function's signature and all of them have same length that
    ///   equal to the `len` arg
    /// - `ptr.add(state_offset)` is valid and points to the state of the function
    unsafe fn batch_update_states(
        &self,
        len: usize,
        payloads: &[&ArrayImpl],
        states_ptr: AggregationStatesPtr,
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
    ///   in the arena. And after combine, the `partial-states` should be a valid states for reuse
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
    ///
    /// - `output` should match function's signature
    ///
    /// - Implementation should free the memory occupied by the state that does not in the arena
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
    ///
    /// - Implementation should free the memory occupied by the `partial-state` that does not
    ///   in the arena
    ///
    /// - For each allocated state, it can only be called once
    unsafe fn drop_states(&self, state_ptrs: &[AggregationStatesPtr], state_offset: usize);
}

impl Debug for dyn AggregationFunction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.debug(f)
    }
}

/// A special optional unary aggregation function that has following features:
///
/// - Aggregation state has form `Some(InnerAggState)` and the output has form `Some(Element)`
///
/// - Do not take null into consideration
pub trait SpecialOptionalUnaryAggregationFunction<PayloadArray: Array, OutputArray: Array>:
    Stringify + Function
{
    /// State
    type State: SpecialOptionalUnaryAggregationState<PayloadArray, OutputElement = OutputArray::Element>;
}

/// Trait for the state of the special unary aggregation function
///
/// Lots of unary aggregation state has this form, for example, `avg`/`sum`/`min`/`max`.
/// Using this trait, the state does not need to care about the outer `Option` wrapper,
/// it can focus on implementing how to `update` and `combine` the inner aggregation state
pub trait SpecialOptionalUnaryAggregationState<PayloadArray: Array>:
    TransparentWrapper<Option<Self::InnerAggStates>>
{
    /// The inner aggregation state of the option. The [`From`] trait is required because
    /// we need to turn the first payload_element to the aggregation state
    type InnerAggStates: From<PayloadArray::Element>;

    /// Output element type. The [`From`] trait is required because we need to turn the
    /// aggregation state to the output element
    type OutputElement: Element + From<Self::InnerAggStates>;

    /// Error
    type Error: std::error::Error + Send + Sync + 'static;

    /// Update the state based on a single element in the payload
    #[inline]
    fn update(
        &mut self,
        payload_element: <PayloadArray::Element as Element>::ElementRef<'_>,
    ) -> Result<()> {
        let current = self.peel_mut();
        if likely(current.is_some()) {
            Self::update_inner_agg_states(current.as_mut().unwrap(), payload_element)
                .boxed()
                .context(UpdateStatesSnafu)
        } else {
            *current = Some(payload_element.to_owned().into());
            Ok(())
        }
    }

    /// Batch update the non-null elements in the payload
    ///
    /// allow the single_use_lifetimes here because anonymous lifetimes in `impl Trait` are unstable
    #[allow(single_use_lifetimes)]
    fn batch_update<'p>(
        state: &mut Option<Self::InnerAggStates>,
        mut values_iter: impl Iterator<Item = <PayloadArray::Element as Element>::ElementRef<'p>>,
    ) -> Result<()> {
        match state.take() {
            Some(inner_state) => {
                // In some cases, auto-vectorization will be performed here
                let inner_state = Self::batch_update_dynamic(inner_state, values_iter)?;
                *state = Some(inner_state);
            }
            None => {
                let Some(init) = values_iter.next() else {
                    return Ok(());
                };
                let inner_state = init.to_owned().into();
                // In some cases, auto-vectorization will be performed here
                let inner_state = Self::batch_update_dynamic(inner_state, values_iter)?;
                *state = Some(inner_state);
            }
        }

        Ok(())
    }

    #[allow(single_use_lifetimes)]
    #[doc(hidden)]
    #[cfg(feature = "avx512")]
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "avx512")]
    unsafe fn batch_update_avx512<'p>(
        inner_state: Self::InnerAggStates,
        mut values_iter: impl Iterator<Item = <PayloadArray::Element as Element>::ElementRef<'p>>,
    ) -> Result<Self::InnerAggStates> {
        values_iter
            .try_fold(inner_state, Self::fold_func)
            .boxed()
            .context(BatchUpdateStatesSnafu)
    }

    #[allow(single_use_lifetimes)]
    #[doc(hidden)]
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "avx2")]
    unsafe fn batch_update_avx2<'p>(
        inner_state: Self::InnerAggStates,
        mut values_iter: impl Iterator<Item = <PayloadArray::Element as Element>::ElementRef<'p>>,
    ) -> Result<Self::InnerAggStates> {
        values_iter
            .try_fold(inner_state, Self::fold_func)
            .boxed()
            .context(BatchUpdateStatesSnafu)
    }

    #[allow(single_use_lifetimes)]
    #[doc(hidden)]
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "sse4.2")]
    unsafe fn batch_update_v2<'p>(
        inner_state: Self::InnerAggStates,
        mut values_iter: impl Iterator<Item = <PayloadArray::Element as Element>::ElementRef<'p>>,
    ) -> Result<Self::InnerAggStates> {
        values_iter
            .try_fold(inner_state, Self::fold_func)
            .boxed()
            .context(BatchUpdateStatesSnafu)
    }

    #[allow(single_use_lifetimes)]
    #[doc(hidden)]
    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon")]
    unsafe fn batch_update_neon<'p>(
        inner_state: Self::InnerAggStates,
        mut values_iter: impl Iterator<Item = <PayloadArray::Element as Element>::ElementRef<'p>>,
    ) -> Result<Self::InnerAggStates> {
        values_iter
            .try_fold(inner_state, Self::fold_func)
            .boxed()
            .context(BatchUpdateStatesSnafu)
    }

    #[inline]
    #[allow(single_use_lifetimes)]
    #[cfg_attr(any(target_arch = "x86", target_arch = "x86_64"), allow(unused_mut))]
    #[doc(hidden)]
    fn batch_update_dynamic<'p>(
        inner_state: Self::InnerAggStates,
        mut values_iter: impl Iterator<Item = <PayloadArray::Element as Element>::ElementRef<'p>>,
    ) -> Result<Self::InnerAggStates> {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            // Note that this `unsafe` block is safe because we're testing
            // that the `avx512` feature is indeed available on our CPU.
            #[cfg(feature = "avx512")]
            if std::arch::is_x86_feature_detected!("avx512f") {
                return unsafe { Self::batch_update_avx512(inner_state, values_iter) };
            }

            // Note that this `unsafe` block is safe because we're testing
            // that the `avx2` feature is indeed available on our CPU.
            if std::arch::is_x86_feature_detected!("avx2") {
                return unsafe { Self::batch_update_avx2(inner_state, values_iter) };
            }

            return unsafe { Self::batch_update_v2(inner_state, values_iter) };
        }

        #[cfg(target_arch = "aarch64")]
        {
            // Note that this `unsafe` block is safe because we're testing
            // that the `neon` feature is indeed available on our CPU.
            if std::arch::is_aarch64_feature_detected!("neon") {
                return unsafe { Self::batch_update_neon(inner_state, values_iter) };
            }
        }

        #[cfg_attr(
            any(target_arch = "x86", target_arch = "x86_64"),
            allow(unreachable_code)
        )]
        {
            values_iter
                .try_fold(inner_state, Self::fold_func)
                .boxed()
                .context(BatchUpdateStatesSnafu)
        }
    }

    /// Update the inner aggregation state based on a single payload element
    fn update_inner_agg_states(
        current: &mut Self::InnerAggStates,
        payload_element: <PayloadArray::Element as Element>::ElementRef<'_>,
    ) -> std::result::Result<(), Self::Error>;

    /// Fold function used to update the state. The reason why we use fold function here instead of
    /// update the `&mut Self::InnerAggStates` is that, in some cases(like min/max), updating
    /// the mutable reference can not be auto-vectorized 😂. WTF? The [bug] is reported. What's
    /// more according to the [comment], the generated assembly of this default implementation
    /// is not optimal! In order to generate the best code, we need to write the implementation
    /// manually, however it will cause duplicated code. Currently, we use the default implementation
    /// to keep the code tidy 😂
    ///
    /// [bug]: https://github.com/rust-lang/rust/issues/123837
    /// [comment]: https://github.com/rust-lang/rust/issues/123837#issuecomment-2051017793
    #[inline]
    fn fold_func(
        mut state: Self::InnerAggStates,
        payload_element: <PayloadArray::Element as Element>::ElementRef<'_>,
    ) -> std::result::Result<Self::InnerAggStates, Self::Error> {
        Self::update_inner_agg_states(&mut state, payload_element).map(|_| state)
    }

    /// Combine the partial aggregation state into the combined aggregation state
    #[inline]
    fn combine(&mut self, partial: &mut Self) -> Result<()> {
        // Take is important, such that the partial state will be dropped
        if let Some(partial) = partial.peel_mut().take() {
            let combined = self.peel_mut();
            match combined {
                Some(combined) => Self::combine_inner_agg_states(combined, partial)
                    .boxed()
                    .context(CombineStatesSnafu)?,
                None => *combined = Some(partial),
            }
        }

        Ok(())
    }

    /// Combine partial inner aggregation sates into the combined inner aggregation state
    fn combine_inner_agg_states(
        combined: &mut Self::InnerAggStates,
        partial: Self::InnerAggStates,
    ) -> std::result::Result<(), Self::Error>;

    /// Consume the aggregation state, return the output of the aggregation state
    #[inline]
    fn take(&mut self) -> Option<Self::OutputElement> {
        self.peel_mut().take().map(|state| state.into())
    }
}

/// Wrapper for special optional unary aggregation function. It is used for blanket
/// implementation, all of the special optional unary aggregation function wrapped
/// in this struct will implement [`AggregationFunction`] for it. It is needed because
/// the rust compiler can not compile the following code:
///
/// ```ignore
/// trait GenericTrait<A> {}
/// trait OtherTrait {}
///
/// impl<T, A> OtherTrait for T where T GenericTrait<A> {}
/// ```
/// The official doc from [`E0207`] suggest a wrapper here
///
/// [`E0207`]: https://doc.rust-lang.org/error_codes/E0207.html
#[derive(Debug)]
pub struct SpecialOptionalUAFWrapper<F, PayloadArray, OutputArray>
where
    F: SpecialOptionalUnaryAggregationFunction<PayloadArray, OutputArray>,
    PayloadArray: Array,
    OutputArray: Array,
{
    inner: F,
    _phantom: PhantomData<(PayloadArray, OutputArray)>,
}

impl<F, PayloadArray, OutputArray> SpecialOptionalUAFWrapper<F, PayloadArray, OutputArray>
where
    F: SpecialOptionalUnaryAggregationFunction<PayloadArray, OutputArray>,
    PayloadArray: Array,
    OutputArray: Array,
{
    /// Create a new wrapper with inner
    pub fn new(inner: F) -> Self {
        Self {
            inner,
            _phantom: PhantomData,
        }
    }
}

unsafe impl<F, PayloadArray, OutputArray> Send
    for SpecialOptionalUAFWrapper<F, PayloadArray, OutputArray>
where
    F: SpecialOptionalUnaryAggregationFunction<PayloadArray, OutputArray>,
    PayloadArray: Array,
    OutputArray: Array,
{
}

unsafe impl<F, PayloadArray, OutputArray> Sync
    for SpecialOptionalUAFWrapper<F, PayloadArray, OutputArray>
where
    F: SpecialOptionalUnaryAggregationFunction<PayloadArray, OutputArray>,
    PayloadArray: Array,
    OutputArray: Array,
{
}

impl<F, PayloadArray, OutputArray> Stringify
    for SpecialOptionalUAFWrapper<F, PayloadArray, OutputArray>
where
    F: SpecialOptionalUnaryAggregationFunction<PayloadArray, OutputArray>,
    PayloadArray: Array,
    OutputArray: Array,
{
    fn name(&self) -> &'static str {
        self.inner.name()
    }

    fn debug(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.inner.debug(f)
    }

    fn display(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.inner.display(f)
    }
}

impl<F, PayloadArray, OutputArray> Function
    for SpecialOptionalUAFWrapper<F, PayloadArray, OutputArray>
where
    F: SpecialOptionalUnaryAggregationFunction<PayloadArray, OutputArray>,
    PayloadArray: Array,
    OutputArray: Array,
{
    fn arguments(&self) -> &[LogicalType] {
        self.inner.arguments()
    }

    fn return_type(&self) -> LogicalType {
        self.inner.return_type()
    }
}

/// Blanked implementation
impl<F, PayloadArray, OutputArray> AggregationFunction
    for SpecialOptionalUAFWrapper<F, PayloadArray, OutputArray>
where
    F: SpecialOptionalUnaryAggregationFunction<PayloadArray, OutputArray>,
    PayloadArray: Array,
    for<'a> &'a PayloadArray: TryFrom<&'a ArrayImpl, Error = ArrayError>,
    OutputArray: Array,
    for<'a> &'a mut OutputArray: TryFrom<&'a mut ArrayImpl, Error = ArrayError>,
{
    fn state_layout(&self) -> Layout {
        Layout::new::<F::State>()
    }

    unsafe fn init_state(&self, ptr: AggregationStatesPtr, state_offset: usize) {
        unsafe {
            let ptr = ptr.offset_as_mut::<F::State>(state_offset).peel_mut();

            // Using write here to avoid drop the uninitiated value, especially for `Vec<u8>`  and `String`
            std::ptr::write(ptr, None);
        }
    }

    unsafe fn update_states(
        &self,
        payloads: &[&ArrayImpl],
        state_ptrs: &[AggregationStatesPtr],
        state_offset: usize,
    ) -> Result<()> {
        unsafe {
            let payload = payloads[0];

            let payload: &PayloadArray = payload
                .try_into()
                .expect("Unary aggregation's payload array should match its signature");

            let validity = payload.validity();
            if validity.all_valid() {
                // All of the element in the payload is not null
                payload
                    .values_iter()
                    .zip(state_ptrs)
                    .try_for_each(|(payload_element, ptr)| {
                        ptr.offset_as_mut::<F::State>(state_offset)
                            .update(payload_element)
                    })
            } else {
                // Some elements are null, only update the element that is not null
                validity.iter_ones().try_for_each(|index| {
                    let payload_element = payload.get_value_unchecked(index);
                    let ptr = *state_ptrs.get_unchecked(index);
                    ptr.offset_as_mut::<F::State>(state_offset)
                        .update(payload_element)
                })
            }
        }
    }

    unsafe fn batch_update_states(
        &self,
        _len: usize,
        payloads: &[&ArrayImpl],
        states_ptr: AggregationStatesPtr,
        state_offset: usize,
    ) -> Result<()> {
        unsafe {
            let payload = payloads[0];

            let payload: &PayloadArray = payload
                .try_into()
                .expect("Unary aggregation's payload array should match its signature");

            let validity = payload.validity();
            let state = states_ptr
                .offset_as_mut::<F::State>(state_offset)
                .peel_mut();

            if validity.all_valid() {
                F::State::batch_update(state, payload.values_iter())
            } else {
                F::State::batch_update(state, ArrayNonNullValuesIter::new(payload, validity))
            }
        }
    }

    unsafe fn combine_states(
        &self,
        partial_state_ptrs: &[AggregationStatesPtr],
        combined_state_ptrs: &[AggregationStatesPtr],
        state_offset: usize,
    ) -> Result<()> {
        unsafe {
            combined_state_ptrs
                .iter()
                .zip(partial_state_ptrs)
                .try_for_each(|(combined, partial)| {
                    let combined = combined.offset_as_mut::<F::State>(state_offset);
                    let partial = partial.offset_as_mut::<F::State>(state_offset);
                    combined.combine(partial)
                })
        }
    }

    unsafe fn take_states(
        &self,
        state_ptrs: &[AggregationStatesPtr],
        state_offset: usize,
        output: &mut ArrayImpl,
    ) -> Result<()> {
        unsafe {
            let output: &mut OutputArray = output
                .try_into()
                .expect("Output of the unary aggregation should match its signature");

            let trusted_len_iterator = state_ptrs.iter().map(|ptr| {
                let state = ptr.offset_as_mut::<F::State>(state_offset);
                state.take()
            });

            output.replace_with_trusted_len_iterator(state_ptrs.len(), trusted_len_iterator);

            Ok(())
        }
    }

    unsafe fn drop_states(&self, state_ptrs: &[AggregationStatesPtr], state_offset: usize) {
        unsafe {
            state_ptrs.iter().for_each(|ptr| {
                let state = ptr.offset_as_mut::<F::State>(state_offset);
                drop(state.peel_mut().take());
            });
        }
    }
}
