/// Macro for building the func that optimize for different target features. Note that caller
/// should guarantee the function could be accelerated by target features, for example
/// auto-vectorization could use different SIMD instructions to increase the performance.
/// If you do not make sure your implementation could use these target features, try your
/// code in [`playground`] or [`CompilerExplorer`] to view the LLVM IR and assembly code 😊
///
/// Currently, this impl pattern is designed for auto-vectorization. It only use `avx512`,
/// `avx2` and `neon` features. According to the [`beginner's guard`], `sse` and `sse2` is
/// enabled by default for `x86` and `x86_64` targets
///
/// Match pattern:
/// (\
///     &emsp; function name of avx512,\
///     &emsp; function name of avx2,\
///     &emsp; function name of neon,\
///     &emsp; function name of the default function,\
///     &emsp; function that using above functions to perform dynamic function call based on target feature,\
///     &emsp; macro that represents the pattern of the function,\
///     &emsp; life time and generic of the function,\
///     &emsp; ( parameters of the function ),\
///     &emsp; ( complicated trait bounds)\
/// )\
///
/// [`playground`]: https://play.rust-lang.org/
/// [`CompilerExplorer`]: https://godbolt.org/
/// [`beginner's guard`]: https://github.com/rust-lang/portable-simd/blob/master/beginners-guide.md#target-features
#[macro_export]
macro_rules! dynamic_auto_vectorization_func {
    ($avx512_func:ident,
        $avx2_func:ident,
        $neon_func:ident,
        $default_func:ident,
        $target_feature_func:ident,
        $func_impl_macro:ident,
        $(<$($lt:lifetime,)* $($g_ty:ident),+>)?,
        ($($parameter:ident : $parameter_ty:ty),*),
        $(where $($trait_bound:tt)+)?
    ) => {
        #[cfg(feature = "avx512")]
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        #[target_feature(enable = "avx512f")]
        #[inline]
        unsafe fn $avx512_func$(<$($lt,)* $($g_ty),+>)?($($parameter:$parameter_ty,)*) $(where $($trait_bound)+)?{
            $func_impl_macro!($($parameter),*)
        }

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        #[target_feature(enable = "avx2")]
        #[inline]
        unsafe fn $avx2_func$(<$($lt,)* $($g_ty),+>)?($($parameter:$parameter_ty,)*) $(where $($trait_bound)+)?{
            $func_impl_macro!($($parameter),*)
        }


        #[cfg(target_arch = "aarch64")]
        #[target_feature(enable = "neon")]
        #[inline]
        unsafe fn $neon_func$(<$($lt,)* $($g_ty),+>)?($($parameter:$parameter_ty,)*) $(where $($trait_bound)+)?{
            $func_impl_macro!($($parameter),*)
        }

        #[inline]
        fn $default_func$(<$($lt,)* $($g_ty),+>)?($($parameter:$parameter_ty,)*) $(where $($trait_bound)+)?{
            $func_impl_macro!($($parameter),*)
        }

        #[inline]
        pub fn $target_feature_func$(<$($lt,)* $($g_ty),+>)?($($parameter:$parameter_ty,)*) $(where $($trait_bound)+)? {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            {
                // Note that this `unsafe` block is safe because we're testing
                // that the `avx512` feature is indeed available on our CPU.
                #[cfg(feature = "avx512")]
                if std::arch::is_x86_feature_detected!("avx512f") {
                    return unsafe { $avx512_func($($parameter,)*) };
                }

                // Note that this `unsafe` block is safe because we're testing
                // that the `avx2` feature is indeed available on our CPU.
                if std::arch::is_x86_feature_detected!("avx2") {
                    return unsafe { $avx2_func($($parameter,)*) };
                }
            }

            #[cfg(target_arch = "aarch64")]
            {
                // Note that this `unsafe` block is safe because we're testing
                // that the `neon` feature is indeed available on our CPU.
                if std::arch::is_aarch64_feature_detected!("neon"){
                    return unsafe { $neon_func($($parameter,)*) };
                }
            }

            $default_func$(::<$($g_ty),*>)*($($parameter,)*)
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

    dynamic_auto_vectorization_func!(
        add_avx512,
        add_avx2,
        add_neon,
        add_default,
        add_dynamic,
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
