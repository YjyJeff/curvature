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

use crate::array::{Array, PrimitiveArray};
use crate::private::Sealed;
use crate::types::{IntrinsicType, PrimitiveType};
use std::ops::{Add, Div, Mul, Rem, Sub};
use strength_reduce::{
    StrengthReducedU16, StrengthReducedU32, StrengthReducedU64, StrengthReducedU8,
};

/// Note that this function will be optimized for different target-feature
#[inline(always)]
fn array_scalar_arith<T, U, F>(lhs: &[T], rhs: U, dst: &mut [T], arith: F)
where
    T: PrimitiveType,
    U: Copy,
    F: Fn(T, U) -> T,
{
    lhs.iter()
        .zip(dst)
        .for_each(|(&lhs, dst)| *dst = arith(lhs, rhs));
}

macro_rules! add_scalar {
    ($lhs:ident, $rhs:ident, $dst:ident) => {
        array_scalar_arith($lhs, $rhs, $dst, Add::add)
    };
}

crate::dynamic_func!(
    add_scalar,
    <T>,
    (lhs: &[T], rhs: T, dst: &mut [T]),
    where T: IntrinsicType + Add<Output = T>
);

macro_rules! sub_scalar {
    ($lhs:ident, $rhs:ident, $dst:ident) => {
        array_scalar_arith($lhs, $rhs, $dst, Sub::sub)
    };
}

crate::dynamic_func!(
    sub_scalar,
    <T>,
    (lhs: &[T], rhs: T, dst: &mut [T]),
    where T: IntrinsicType + Sub<Output = T>
);

macro_rules! mul_scalar {
    ($lhs:ident, $rhs:ident, $dst:ident) => {
        array_scalar_arith($lhs, $rhs, $dst, Mul::mul)
    };
}

crate::dynamic_func!(
    mul_scalar,
    <T>,
    (lhs: &[T], rhs: T, dst: &mut [T]),
    where T: IntrinsicType + Mul<Output = T>
);

/// Extent div
pub trait DivExt: PrimitiveType + Div<Self::Divisor, Output = Self> + Sealed {
    /// Divisor that can accelerate div
    type Divisor: Copy;
    /// Create a new divisor from self
    fn new_divisor(self) -> Self::Divisor;
}

/// Extent remainder
pub trait RemExt: PrimitiveType + Rem<Self::Remainder, Output = Self> + Sealed {
    /// Remainder that can accelerate rem
    type Remainder: Copy;
    /// Create a new remainder from self
    fn new_remainder(self) -> Self::Remainder;
}

/// Divisor of signed integer
#[derive(Debug, Clone, Copy)]
pub struct SignedIntegerDivisor<T: Clone + Copy> {
    neg: bool,
    divisor: T,
}

/// Remainder of signed integer
#[derive(Debug, Clone, Copy)]
#[repr(transparent)]
pub struct SignedIntegerRemainder<T: Clone + Copy>(T);

macro_rules! impl_div_ext {
    ($ty:ty) => {
        // Divisor is self
        impl DivExt for $ty {
            type Divisor = $ty;

            #[inline]
            fn new_divisor(self) -> Self::Divisor {
                self
            }
        }

        impl RemExt for $ty {
            type Remainder = $ty;

            #[inline]
            fn new_remainder(self) -> Self::Remainder {
                self
            }
        }
    };
    ($ty:ty, $d_ty:ty) => {
        // Unsigned integer
        impl DivExt for $ty {
            type Divisor = $d_ty;
            #[inline]
            fn new_divisor(self) -> Self::Divisor {
                <$d_ty>::new(self)
            }
        }

        impl RemExt for $ty {
            type Remainder = $d_ty;
            #[inline]
            fn new_remainder(self) -> Self::Remainder {
                <$d_ty>::new(self)
            }
        }
    };
    ($ty:ty, $uty:ty, $d_ty:ty) => {
        // Signed integer
        impl Div<SignedIntegerDivisor<$d_ty>> for $ty {
            type Output = $ty;
            #[inline]
            fn div(self, rhs: SignedIntegerDivisor<$d_ty>) -> Self::Output {
                let (result, neg) = if self > 0 {
                    (self as $uty / rhs.divisor, false)
                } else {
                    ((-self) as $uty / rhs.divisor, true)
                };

                if rhs.neg ^ neg {
                    -(result as $ty)
                } else {
                    result as $ty
                }
            }
        }

        impl Rem<SignedIntegerRemainder<$d_ty>> for $ty {
            type Output = $ty;
            #[inline]
            fn rem(self, rhs: SignedIntegerRemainder<$d_ty>) -> Self::Output {
                if self > 0 {
                    (self as $uty % rhs.0) as $ty
                } else {
                    -(((-self) as $uty % rhs.0) as $ty)
                }
            }
        }

        impl DivExt for $ty {
            type Divisor = SignedIntegerDivisor<$d_ty>;
            #[inline]
            fn new_divisor(self) -> Self::Divisor {
                if self > 0 {
                    SignedIntegerDivisor {
                        neg: false,
                        divisor: <$d_ty>::new(self as $uty),
                    }
                } else {
                    SignedIntegerDivisor {
                        neg: true,
                        divisor: <$d_ty>::new((-self) as $uty),
                    }
                }
            }
        }

        impl RemExt for $ty {
            type Remainder = SignedIntegerRemainder<$d_ty>;

            #[inline]
            fn new_remainder(self) -> Self::Remainder {
                if self > 0 {
                    SignedIntegerRemainder(<$d_ty>::new(self as $uty))
                } else {
                    SignedIntegerRemainder(<$d_ty>::new((-self) as $uty))
                }
            }
        }
    };
}

impl_div_ext!(u8, StrengthReducedU8);
impl_div_ext!(u16, StrengthReducedU16);
impl_div_ext!(u32, StrengthReducedU32);
impl_div_ext!(u64, StrengthReducedU64);
impl_div_ext!(f32);
impl_div_ext!(f64);
impl_div_ext!(i128);

impl_div_ext!(i8, u8, StrengthReducedU8);
impl_div_ext!(i16, u16, StrengthReducedU16);
impl_div_ext!(i32, u32, StrengthReducedU32);
impl_div_ext!(i64, u64, StrengthReducedU64);

macro_rules! div_scalar {
    ($lhs:ident, $rhs:ident, $dst:ident) => {{
        let rhs = T::new_divisor($rhs);
        array_scalar_arith($lhs, rhs, $dst, Div::div)
    }};
}

crate::dynamic_func!(
    div_scalar,
    <T>,
    (lhs: &[T], rhs: T, dst: &mut [T]),
    where T: IntrinsicType + DivExt
);

#[inline]
fn add_scalar_<T>(lhs: &[T], rhs: T, dst: &mut [T])
where
    T: PrimitiveType + Add<Output = T>,
{
    array_scalar_arith(lhs, rhs, dst, Add::add)
}

#[inline]
fn sub_scalar_<T>(lhs: &[T], rhs: T, dst: &mut [T])
where
    T: PrimitiveType + Sub<Output = T>,
{
    array_scalar_arith(lhs, rhs, dst, Sub::sub)
}

#[inline]
fn mul_scalar_<T>(lhs: &[T], rhs: T, dst: &mut [T])
where
    T: PrimitiveType + Mul<Output = T>,
{
    array_scalar_arith(lhs, rhs, dst, Mul::mul)
}

#[inline]
fn div_scalar_<T>(lhs: &[T], rhs: T, dst: &mut [T])
where
    T: PrimitiveType + DivExt,
{
    let rhs = rhs.new_divisor();
    array_scalar_arith(lhs, rhs, dst, Div::div)
}

#[inline]
fn rem_scalar_<T, U>(lhs: &[T], rhs: U, dst: &mut [U])
where
    T: PrimitiveType + RemExt,
    U: RemCast<T>,
{
    let rhs = T::new_remainder(rhs.cast());
    lhs.iter().zip(dst).for_each(|(&lhs, dst)| {
        *dst = U::cast_back(lhs % rhs);
    })
}

// Traits such that intrinsic type and primitive type can call different functions

type ArithFunc<T, U> = fn(&[T], U, &mut [U]);

/// Trait for add array with scalar
pub trait ArrayAddElement: PrimitiveType + Add<Output = Self> {
    /// Add func
    const FUNC: ArithFunc<Self, Self>;
}

/// Trait for sub array with scalar
pub trait ArraySubElement: PrimitiveType + Sub<Output = Self> {
    /// Sub func
    const FUNC: ArithFunc<Self, Self>;
}

/// Trait for multiple array with scalar
pub trait ArrayMulElement: PrimitiveType + Mul<Output = Self> {
    /// Multiple func
    const FUNC: ArithFunc<Self, Self>;
}

/// Trait for div array with scalar
pub trait ArrayDivElement: PrimitiveType + DivExt {
    /// Div func
    const FUNC: ArithFunc<Self, Self>;
}

/// Trait for rem array with scalar
pub trait ArrayRemElement<U>: PrimitiveType + RemExt
where
    U: RemCast<Self>,
{
    /// rem func
    const FUNC: ArithFunc<Self, U>;
}

macro_rules! impl_array_scalar_arith {
    ({$($ty:ty),+}, $add:ident, $sub:ident, $mul:ident, $div:ident, $rem:ident) => {
        // Intrinsic type
        $(
            impl ArrayAddElement for $ty {
                const FUNC: ArithFunc<Self, Self> = $add;
            }

            impl ArraySubElement for $ty {
                const FUNC: ArithFunc<Self, Self> = $sub;
            }

            impl ArrayMulElement for $ty {
                const FUNC: ArithFunc<Self, Self> = $mul;
            }

            impl ArrayDivElement for $ty {
                const FUNC: ArithFunc<Self, Self> = $div;
            }

            impl<U> ArrayRemElement<U> for $ty
            where U: RemCast<Self>
            {
                const FUNC: ArithFunc<Self, U> = $rem;
            }
        )+
    };
}

impl_array_scalar_arith!(
    {i8, u8, i16, u16, i32, u32, i64, u64, f32, f64},
    add_scalar_dynamic,
    sub_scalar_dynamic,
    mul_scalar_dynamic,
    div_scalar_dynamic,
    rem_scalar_dynamic
);

impl_array_scalar_arith!(
    { i128 },
    add_scalar_,
    sub_scalar_,
    mul_scalar_,
    div_scalar_,
    rem_scalar_
);

// TODO: Support arith on Decimal. Take care of the arith on decimal, it may panic if overflow happens. 💩

macro_rules! arith_scalar {
    ($func_name:ident, $op:tt, $trait_bound:ident) => {
        #[doc = concat!("Perform `lhs ", stringify!($op), " rhs` for `PrimitiveArray<T>` with `T`")]
        ///
        /// # Safety
        ///
        /// - `lhs` data and validity should not reference `dst`'s data and validity. In the computation
        /// graph, `lhs` must be the descendant of `dst`
        ///
        /// - No other arrays that reference the `dst`'s data and validity are accessed! In the
        /// computation graph, it will never happens
        pub unsafe fn $func_name<T>(lhs: &PrimitiveArray<T>, rhs: T, dst: &mut PrimitiveArray<T>)
        where
            PrimitiveArray<T>: Array,
            T: $trait_bound,
        {
            dst.validity.reference(&lhs.validity);

            let dst = dst.data.as_mut();
            let uninitialized = dst.clear_and_resize(lhs.len());
            T::FUNC(lhs.data.as_slice(), rhs, uninitialized);
        }
    };
}

arith_scalar!(add_scalar, +, ArrayAddElement);
arith_scalar!(sub_scalar, -, ArraySubElement);
arith_scalar!(mul_scalar, *, ArrayMulElement);
arith_scalar!(div_scalar, /, ArrayDivElement);

/// Trait for arith function
pub trait ArithFuncTrait<T>: Send + Sync + 'static
where
    PrimitiveArray<T>: Array,
    T: PrimitiveType,
{
    /// Name of the arith func
    const NAME: &'static str;
    /// Symbol of the arith func
    const SYMBOL: &'static str;
    /// The arith Function between array and a scalar
    const SCALAR_FUNC: unsafe fn(&PrimitiveArray<T>, T, &mut PrimitiveArray<T>);
}

macro_rules! impl_default_arith_func_trait {
    ($op:tt, $symbol:tt, $scalar_func:ident, $trait_bound:ident) => {
        paste::paste! {
            #[doc = concat!("Default ", stringify!($op), " between array and scalar")]
            #[derive(Debug)]
            pub struct [<Default $op Scalar>];

            impl<T> ArithFuncTrait<T> for [<Default $op Scalar>]
            where
                PrimitiveArray<T>: Array,
                T: PrimitiveType + $trait_bound,
            {
                const NAME: &'static str = stringify!($op);
                const SYMBOL: &'static str  = stringify!($symbol);
                const SCALAR_FUNC: unsafe fn(&PrimitiveArray<T>, T, &mut PrimitiveArray<T>) = $scalar_func;
            }
        }
    };
}

impl_default_arith_func_trait!(Add, +, add_scalar, ArrayAddElement);
impl_default_arith_func_trait!(Sub, -, sub_scalar, ArraySubElement);
impl_default_arith_func_trait!(Mul, *, mul_scalar, ArrayMulElement);
impl_default_arith_func_trait!(Div, /, div_scalar, ArrayDivElement);

/// Trait for casting used in rem
pub trait RemCast<T: PrimitiveType>: PrimitiveType {
    /// Cast self to T
    fn cast(self) -> T;

    /// Cast back T to self
    fn cast_back(val: T) -> Self;
}

macro_rules! impl_rem_cast {
    ($ty:ty, {$($cast_ty:ty),+}) => {
        $(
            impl RemCast<$cast_ty> for $ty {
                #[inline]
                fn cast(self) -> $cast_ty {
                    self as $cast_ty
                }

                #[inline]
                fn cast_back(val: $cast_ty) -> Self{
                    val as $ty
                }
            }
        )+
    };
}

impl_rem_cast!(u8, {u8, u16, u32, u64});
impl_rem_cast!(u16, {u16, u32, u64});
impl_rem_cast!(u32, {u32, u64});
impl_rem_cast!(u64, { u64 });
impl_rem_cast!(i8, {i8, i16, i32, i64, i128});
impl_rem_cast!(i16, {i16, i32, i64, i128});
impl_rem_cast!(i32, {i32, i64, i128});
impl_rem_cast!(i64, { i64, i128 });
impl_rem_cast!(i128, { i128 });

macro_rules! rem_scalar {
    ($lhs:ident, $rhs:ident, $dst:ident) => {{
        let rhs = T::new_remainder($rhs.cast());
        $lhs.iter().zip($dst).for_each(|(&lhs, dst)| {
            *dst = U::cast_back(lhs % rhs);
        })
    }};
}

crate::dynamic_func!(
    rem_scalar,
    <T, U>,
    (lhs: &[T], rhs: U, dst: &mut [U]),
    where T: IntrinsicType + RemExt, U: RemCast<T>
);

/// Perform `lhs % rhs` for `PrimitiveArray<T>` with `U`
///
/// # Safety
///
/// - `lhs` validity should not reference `dst`'s validity. In the computation
/// graph, `lhs` must be the descendant of `dst`
///
/// - No other arrays that reference the `dst`'s data and validity are accessed! In the
/// computation graph, it will never happens
pub unsafe fn rem_scalar<T, U>(lhs: &PrimitiveArray<T>, rhs: U, dst: &mut PrimitiveArray<U>)
where
    PrimitiveArray<T>: Array,
    PrimitiveArray<U>: Array,
    T: PrimitiveType + ArrayRemElement<U>,
    U: RemCast<T>,
{
    dst.validity.reference(&lhs.validity);

    let dst = dst.data.as_mut();
    let uninitialized = dst.clear_and_resize(lhs.len());
    T::FUNC(lhs.data.as_slice(), rhs, uninitialized)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::LogicalType;

    #[test]
    fn test_add_scalar() {
        let lhs = PrimitiveArray::from_values_iter([10, -7, -3, -9, -20, 87, 100, 34, 65]);
        let rhs = 1;
        let mut dst = PrimitiveArray::new(LogicalType::Integer).unwrap();
        unsafe { add_scalar(&lhs, rhs, &mut dst) };
        assert_eq!(dst.values(), &[11, -6, -2, -8, -19, 88, 101, 35, 66]);
    }

    #[test]
    fn test_sub_scalar() {
        let lhs = PrimitiveArray::from_values_iter([10, -7, -3, -9, -20, 87, 100, 34, 65]);
        let rhs = 1;
        let mut dst = PrimitiveArray::new(LogicalType::Integer).unwrap();
        unsafe { sub_scalar(&lhs, rhs, &mut dst) };
        assert_eq!(dst.values(), &[9, -8, -4, -10, -21, 86, 99, 33, 64]);
    }

    #[test]
    fn test_mul_scalar() {
        let lhs = PrimitiveArray::from_values_iter([10, -7, -3, -9, -20, 87, 100, 34, 65]);
        let rhs = 2;
        let mut dst = PrimitiveArray::new(LogicalType::Integer).unwrap();
        unsafe { mul_scalar(&lhs, rhs, &mut dst) };
        assert_eq!(dst.values(), &[20, -14, -6, -18, -40, 174, 200, 68, 130]);
    }

    #[test]
    fn test_div_scalar() {
        let lhs = PrimitiveArray::from_values_iter([10, -7, -3, -9, -20, 87, 100, 34, 65]);
        let rhs = -2;
        let mut dst = PrimitiveArray::new(LogicalType::Integer).unwrap();
        unsafe { div_scalar(&lhs, rhs, &mut dst) };
        assert_eq!(dst.values(), &[-5, 3, 1, 4, 10, -43, -50, -17, -32]);
    }

    #[test]
    fn test_rem_scalar() {
        let lhs = PrimitiveArray::from_values_iter([10, -7, -3, -9, -20, 87, 100, 34, 65]);
        let rhs = -3;
        let mut dst = PrimitiveArray::new(LogicalType::Integer).unwrap();
        unsafe { rem_scalar(&lhs, rhs, &mut dst) };
        assert_eq!(dst.values(), &[1, -1, 0, 0, -2, 0, 1, 1, 2]);
    }
}
