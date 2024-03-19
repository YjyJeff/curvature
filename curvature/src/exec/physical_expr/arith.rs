//! Expression to perform basic arithmetic like `+`/`-`/`*`/`/`/`%`

use data_block::array::{ArrayError, ArrayImpl, PrimitiveArray};
use data_block::block::DataBlock;
use data_block::compute::arith::{
    ArithFuncTrait, ArrayRemElement, DefaultAddScalar, DefaultDivScalar, DefaultMulScalar,
    DefaultSubScalar, RemCast, RemExt,
};
use data_block::types::{Array, LogicalType, PrimitiveType};
use std::fmt::Debug;
use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};
use std::sync::Arc;

use super::executor::ExprExecCtx;
use super::utils::CompactExprDisplayWrapper;
use super::{ExprResult, PhysicalExpr, Stringify};

/// Add primitive array with constant, the element in the primitive array are interpreted as numbers
pub type DefaultAddConstantArith<T> = ConstantArith<T, DefaultAddScalar>;
/// Sub primitive array with constant, the element in the primitive array are interpreted as numbers
pub type DefaultSubConstantArith<T> = ConstantArith<T, DefaultSubScalar>;
/// Mul primitive array with constant, the element in the primitive array are interpreted as numbers
pub type DefaultMulConstantArith<T> = ConstantArith<T, DefaultMulScalar>;
/// Div primitive array with constant, the element in the primitive array are interpreted as numbers
pub type DefaultDivConstantArith<T> = ConstantArith<T, DefaultDivScalar>;

/// Expression to perform `+`/`-`/`*`/`/`/`%` between Array and constant
#[derive(Debug)]
pub struct ConstantArith<T, F>
where
    PrimitiveArray<T>: Array,
    T: PrimitiveType,
    F: ArithFuncTrait<T>,
{
    children: Vec<Arc<dyn PhysicalExpr>>,
    constant: T,
    name: String,
    output_type: LogicalType,
    _phantom: PhantomData<F>,
}

impl<T, F> ConstantArith<T, F>
where
    PrimitiveArray<T>: Array,
    T: PrimitiveType,
    F: ArithFuncTrait<T>,
{
    /// Create a new constant arithmetic
    pub fn new(input: Arc<dyn PhysicalExpr>, constant: T, name: String) -> Self {
        Self {
            output_type: input.output_type().to_owned(),
            children: vec![input],
            constant,
            name,
            _phantom: PhantomData,
        }
    }
}

impl<T, F> Stringify for ConstantArith<T, F>
where
    PrimitiveArray<T>: Array,
    T: PrimitiveType,
    F: ArithFuncTrait<T>,
    Self: Debug,
{
    /// FIXME: Generic info
    fn name(&self) -> &'static str {
        "ConstantArith"
    }

    fn debug(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }

    fn compact_display(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.children[0].compact_display(f)?;
        write!(f, " {} {}", F::SYMBOL, self.constant)
    }

    fn display(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} {} -> {:?}", F::NAME, self.constant, self.output_type)
    }
}

impl<T, F> PhysicalExpr for ConstantArith<T, F>
where
    PrimitiveArray<T>: Array,
    T: PrimitiveType,
    F: ArithFuncTrait<T>,
    for<'a> &'a PrimitiveArray<T>: TryFrom<&'a ArrayImpl, Error = ArrayError>,
    for<'a> &'a mut PrimitiveArray<T>: TryFrom<&'a mut ArrayImpl, Error = ArrayError>,
    Self: Debug,
{
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn output_type(&self) -> &LogicalType {
        &self.output_type
    }

    fn children(&self) -> &[Arc<dyn PhysicalExpr>] {
        &self.children
    }

    fn execute(
        &self,
        leaf_input: &DataBlock,
        exec_ctx: &mut ExprExecCtx,
        output: &mut ArrayImpl,
    ) -> ExprResult<()> {
        // Execute input
        let mut guard = exec_ctx.intermediate_block.mutate_single_array();

        self.children[0].execute(leaf_input, &mut exec_ctx.children[0], guard.deref_mut())?;

        // Execute self
        let input: &PrimitiveArray<T> = guard.deref().try_into().unwrap_or_else(|_| {
            panic!(
                "`{}` expect the input have `{}`, however the input array `{}Array` has logical type: `{:?}` with `{}`",
                CompactExprDisplayWrapper::new(self),
                T::PHYSICAL_TYPE,
                guard.ident(),
                guard.logical_type(),
                guard.logical_type().physical_type(),
            )
        });

        let output: &mut PrimitiveArray<T> = output.try_into().unwrap_or_else(|_| {
            panic!(
                "`{}` expect the output have `{}`, however the output array `{}Array` has logical type: `{:?}` with `{}`", 
                CompactExprDisplayWrapper::new(self),
                T::PHYSICAL_TYPE,
                guard.ident(),
                guard.logical_type(),
                guard.logical_type().physical_type()
            )
        });

        unsafe {
            F::SCALAR_FUNC(input, self.constant, output);
        }

        Ok(())
    }
}

/// Rem
#[derive(Debug)]
pub struct ConstDiv<T, U> {
    children: Vec<Arc<dyn PhysicalExpr>>,
    constant: U,
    name: String,
    output_type: LogicalType,
    _phantom: PhantomData<T>,
}

impl<T, U> ConstDiv<T, U>
where
    PrimitiveArray<T>: Array,
    PrimitiveArray<U>: Array,
    T: PrimitiveType + RemExt,
    U: PrimitiveType + RemCast<T>,
{
    /// Create a new constant arithmetic
    pub fn new(input: Arc<dyn PhysicalExpr>, constant: U, name: String) -> Self {
        Self {
            output_type: U::LOGICAL_TYPE,
            children: vec![input],
            constant,
            name,
            _phantom: PhantomData,
        }
    }
}

impl<T, U> Stringify for ConstDiv<T, U>
where
    PrimitiveArray<T>: Array,
    PrimitiveArray<U>: Array,
    T: PrimitiveType + ArrayRemElement<U>,
    U: PrimitiveType + RemCast<T>,
    Self: Debug,
{
    /// FIXME: Generic info
    fn name(&self) -> &'static str {
        "ConstantArith"
    }

    fn debug(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }

    fn compact_display(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.children[0].compact_display(f)?;
        write!(f, " % {}", self.constant)
    }

    fn display(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "% {} -> {:?}", self.constant, self.output_type)
    }
}

impl<T, U> PhysicalExpr for ConstDiv<T, U>
where
    PrimitiveArray<T>: Array,
    PrimitiveArray<U>: Array,
    T: PrimitiveType + ArrayRemElement<U>,
    U: PrimitiveType + RemCast<T>,
    for<'a> &'a PrimitiveArray<T>: TryFrom<&'a ArrayImpl, Error = ArrayError>,
    for<'a> &'a mut PrimitiveArray<U>: TryFrom<&'a mut ArrayImpl, Error = ArrayError>,
    Self: Debug,
{
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn output_type(&self) -> &LogicalType {
        &self.output_type
    }

    fn children(&self) -> &[Arc<dyn PhysicalExpr>] {
        &self.children
    }

    fn execute(
        &self,
        leaf_input: &DataBlock,
        exec_ctx: &mut ExprExecCtx,
        output: &mut ArrayImpl,
    ) -> ExprResult<()> {
        // Execute input
        let mut guard = exec_ctx.intermediate_block.mutate_single_array();

        self.children[0].execute(leaf_input, &mut exec_ctx.children[0], guard.deref_mut())?;

        // Execute self
        let input: &PrimitiveArray<T> = guard.deref().try_into().unwrap_or_else(|_| {
            panic!(
                "`{}` expect the input have `{}`, however the input array `{}Array` has logical type: `{:?}` with `{}`",
                CompactExprDisplayWrapper::new(self),
                T::PHYSICAL_TYPE,
                guard.ident(),
                guard.logical_type(),
                guard.logical_type().physical_type(),
            )
        });

        let output: &mut PrimitiveArray<U> = output.try_into().unwrap_or_else(|_| {
            panic!(
                "`{}` expect the output have `{}`, however the output array `{}Array` has logical type: `{:?}` with `{}`", 
                CompactExprDisplayWrapper::new(self),
                T::PHYSICAL_TYPE,
                guard.ident(),
                guard.logical_type(),
                guard.logical_type().physical_type()
            )
        });

        unsafe {
            data_block::compute::arith::rem_scalar(input, self.constant, output);
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::exec::physical_expr::field_ref::FieldRef;

    use super::*;

    #[test]
    fn test_execute_constant_arithmetic() {
        let arith = ConstantArith::<i32, DefaultAddScalar>::new(
            Arc::new(FieldRef::new(0, LogicalType::Integer, "f0".to_string())),
            3,
            "test".to_string(),
        );

        let input = DataBlock::try_new(vec![ArrayImpl::Int32(PrimitiveArray::from_values_iter([
            -1, 0, 1,
        ]))])
        .unwrap();

        let mut output = ArrayImpl::new(LogicalType::Integer);

        arith
            .execute(&input, &mut ExprExecCtx::new(&arith), &mut output)
            .unwrap();

        let expect = expect_test::expect![[r#"
            Int32(
                Int32Array { logical_type: Integer, len: 3, data: [
                    Some(
                        2,
                    ),
                    Some(
                        3,
                    ),
                    Some(
                        4,
                    ),
                ]},
            )
        "#]];
        expect.assert_debug_eq(&output);
    }
}
