//! Macros used in the data-block

/// Macros for all of the variants in the array
///
/// Tuple: {enum variant name, element type, array type}
#[macro_export]
macro_rules! for_all_variants {
    ($macro:ident) => {
        $macro! {
            {Int8, i8, Int8Array},
            {UInt8, u8, UInt8Array},
            {Int16, i16, Int16Array},
            {UInt16, u16, UInt16Array},
            {Int32, i32, Int32Array},
            {UInt32, u32, UInt32Array},
            {Int64, i64, Int64Array},
            {UInt64, u64, UInt64Array},
            {Float32, f32, Float32Array},
            {Float64, f64, Float64Array},
            {Int128, i128, Int128Array},
            {DayTime, DayTime, DayTimeArray},
            {String, StringElement, StringArray},
            {Binary, Vec<u8>, BinaryArray},
            {Boolean, bool, BooleanArray},
            {List, ListElement, ListArray}
        }
    };
}

pub(crate) use for_all_variants;

/// Call macro for all primitive types.
///
/// Tuple: {enum variant name, element type, array type, logical type variant}
macro_rules! for_all_primitive_types {
    ($macro:ident) => {
        $macro! {
            {Int8, i8, Int8Array, TinyInt},
            {UInt8, u8, UInt8Array, UnsignedTinyInt},
            {Int16, i16, Int16Array, SmallInt},
            {UInt16, u16, UInt16Array, UnsignedSmallInt},
            {Int32, i32, Int32Array, Integer},
            {UInt32, u32, UInt32Array, UnsignedInteger},
            {Int64, i64, Int64Array, BigInt},
            {UInt64, u64, UInt64Array, UnsignedBigInt},
            {Float32, f32, Float32Array, Float},
            {Float64, f64, Float64Array, Double},
            {Int128, i128, Int128Array, HugeInt},
            {DayTime, DayTime, DayTimeArray, IntervalDayTime}
        }
    };
}

pub(crate) use for_all_primitive_types;

/// Call macro for all intrinsic types
///
/// (scalar type, simd type, lanes, bitmask type)
macro_rules! for_all_intrinsic {
    ($macro:ident) => {
        $macro! {
            {i8, i8x64, 64, u64},
            {u8, u8x64, 64, u64},
            {i16, i16x32, 32, u32},
            {u16, u16x32, 32, u32},
            {i32, i32x16, 16, u16},
            {u32, u32x16, 16, u16},
            {i64, i64x8, 8, u8},
            {u64, u64x8, 8, u8},
            {f32, f32x16, 16, u16},
            {f64, f64x8, 8, u8}
        }
    };
}

pub(crate) use for_all_intrinsic;

/// Macro for building the func that optimize for different target features. Note that caller
/// should guarantee the function could be accelerated by target features, for example
/// auto-vectorization could use different SIMD instructions to increase the performance.
/// If you do not make sure your implementation could use these target features, try your
/// code in [`playground`] or [`CompilerExplorer`] to view the LLVM IR and assembly code 😊
///
/// Currently, this impl pattern is designed for auto-vectorization. It only use `avx512`,
/// `avx2` and `neon` features. According to the [`beginner's guard`], `sse` and `sse2` is
/// enabled by default for `x86` and `x86_64` targets.
///
/// Match pattern:
/// (\
///     &emsp; macro that represents the pattern of the function,\
///     &emsp; life time and generic of the function,\
///     &emsp; ( parameters of the function ),\
///     &emsp; complicated trait bounds \
/// )\
///
/// # Example
///
/// A simple example for adding two slices and write the result to a predefined array
///
/// ```
/// use data_block::dynamic_func;
///
/// macro_rules! add {
///     ($lhs:ident, $rhs:ident, $dest:ident) => {
///         for ((lhs, rhs), dest) in $lhs.iter().zip($rhs).zip($dest) {
///             *dest = *lhs + *rhs;
///         }
///     };
/// }
///
/// dynamic_func!(
///     add,
///     <'a, T>,
///     (a: &[T], b: &[T], c: &mut [T]),
///     where T: std::ops::Add<Output = T> + Copy
/// );
///
/// let lhs = [0, 1, 2, 3];
/// let rhs = [0, 1, 2, 3];
/// let mut dest = vec![0; 4];
/// add_dynamic(&lhs, &rhs, &mut dest);
/// assert_eq!(dest, [0, 2, 4, 6]);
///
/// ```
///
/// [`playground`]: https://play.rust-lang.org/
/// [`CompilerExplorer`]: https://godbolt.org/
/// [`beginner's guard`]: https://github.com/rust-lang/portable-simd/blob/master/beginners-guide.md#target-features
#[macro_export]
macro_rules! dynamic_func {
    ($func_impl_macro:ident,
        $(<$($lt:lifetime,)* $($g_ty:ident),+>)?,
        ($($parameter:ident : $parameter_ty:ty),*),
        $(where $($trait_bound:tt)+)?
    ) => {
        paste::paste! {
            #[cfg(feature = "avx512")]
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            #[target_feature(enable = "avx512f")]
            #[inline]
            unsafe fn [<$func_impl_macro _avx512>]$(<$($lt,)* $($g_ty),+>)?($($parameter:$parameter_ty,)*) $(where $($trait_bound)+)?{
                $func_impl_macro!($($parameter),*)
            }

            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            #[target_feature(enable = "avx2")]
            #[inline]
            unsafe fn [<$func_impl_macro _avx2>]$(<$($lt,)* $($g_ty),+>)?($($parameter:$parameter_ty,)*) $(where $($trait_bound)+)?{
                $func_impl_macro!($($parameter),*)
            }

            #[cfg(target_arch = "aarch64")]
            #[target_feature(enable = "neon")]
            #[inline]
            unsafe fn [<$func_impl_macro _neon>]$(<$($lt,)* $($g_ty),+>)?($($parameter:$parameter_ty,)*) $(where $($trait_bound)+)?{
                $func_impl_macro!($($parameter),*)
            }

            #[inline]
            fn [<$func_impl_macro _default>]$(<$($lt,)* $($g_ty),+>)?($($parameter:$parameter_ty,)*) $(where $($trait_bound)+)?{
                $func_impl_macro!($($parameter),*)
            }

            #[inline]
            pub(crate) fn [<$func_impl_macro _dynamic>]$(<$($lt,)* $($g_ty),+>)?($($parameter:$parameter_ty,)*) $(where $($trait_bound)+)? {
                #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                {
                    // Note that this `unsafe` block is safe because we're testing
                    // that the `avx512` feature is indeed available on our CPU.
                    #[cfg(feature = "avx512")]
                    if std::arch::is_x86_feature_detected!("avx512f") {
                        return unsafe { [<$func_impl_macro _avx512>]($($parameter,)*) };
                    }

                    // Note that this `unsafe` block is safe because we're testing
                    // that the `avx2` feature is indeed available on our CPU.
                    if std::arch::is_x86_feature_detected!("avx2") {
                        return unsafe { [<$func_impl_macro _avx2>]($($parameter,)*) };
                    }
                }

                #[cfg(target_arch = "aarch64")]
                {
                    // Note that this `unsafe` block is safe because we're testing
                    // that the `neon` feature is indeed available on our CPU.
                    if std::arch::is_aarch64_feature_detected!("neon"){
                        return unsafe { [<$func_impl_macro _neon>]($($parameter,)*) };
                    }
                }

                [<$func_impl_macro _default>]$(::<$($g_ty),*>)*($($parameter,)*)
            }
        }
    };
}

#[cfg(test)]
mod tests {
    macro_rules! add {
        ($lhs:ident, $rhs:ident, $dest:ident) => {
            for ((lhs, rhs), dest) in $lhs.iter().zip($rhs).zip($dest) {
                *dest = *lhs + *rhs;
            }
        };
    }

    dynamic_func!(
        add,
        <'a, T>,
        (a: &[T], b: &[T], c: &mut [T]),
        where T: std::ops::Add<Output = T> + Copy
    );

    /// An example of using dynamic_auto_vectorization_func
    #[test]
    fn macro_add_dynamic_i32_example() {
        let lhs = [0, 1, 2, 3];
        let rhs = [0, 1, 2, 3];
        let mut dest = vec![0; 4];
        add_dynamic(&lhs, &rhs, &mut dest);
        assert_eq!(dest, [0, 2, 4, 6]);
    }
}
