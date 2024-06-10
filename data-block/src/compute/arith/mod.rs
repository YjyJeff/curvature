//! Arithmetic on PrimitiveType
//!
//! CPU can not perform division effectively, therefore, except division we can use
//! auto-vectorization to get the best performance. For division, we use
//! strength_reduce to accelerate computation, LLVM will also perform
//! auto-vectorization for it, awesome !!!
//!
//! We can replace the IntrinsicType bound with PrimitiveType bound! However, if
//! we remove it, we can call dynamic function on non intrinsic type! Although
//! it does not matters, we still want to forbid this behavior in the compiler stage
//!
//! TODO: The array is cache line aligned and padded, hint the compiler!

pub mod intrinsic;
use self::intrinsic::{RemCast, RemExt};
use std::ops::{Add, Div, Mul, Rem, Sub};

/// Add two scalars
#[inline(always)]
pub fn scalar_add_scalar<T: Copy + Add<Output = T>>(left: T, right: T) -> T {
    left + right
}

/// Sub two scalars
#[inline(always)]
pub fn scalar_sub_scalar<T: Copy + Sub<Output = T>>(left: T, right: T) -> T {
    left - right
}

/// Multiply two scalars
#[inline(always)]
pub fn scalar_mul_scalar<T: Copy + Mul<Output = T>>(left: T, right: T) -> T {
    left * right
}

/// Divide two scalars
#[inline(always)]
pub fn scalar_div_scalar<T: Copy + Div<Output = T>>(left: T, right: T) -> T {
    left / right
}

/// Rem two scalars
#[inline(always)]
pub fn scalar_rem_scalar<T: RemExt + Rem<Output = T>, U: Copy + RemCast<T>>(
    left: T,
    right: U,
) -> U {
    U::cast_back(left % right.cast())
}
