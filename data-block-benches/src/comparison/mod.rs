// #[cfg(feature = "portable_simd")]
// pub mod portable_simd;

#[inline]
pub fn cmp<T, F>(lhs: &[T], rhs: T, dst: &mut [bool], cmp_func: F)
where
    F: Fn(&T, &T) -> bool,
{
    lhs.iter()
        .zip(dst)
        .for_each(|(lhs, dst)| *dst = cmp_func(lhs, &rhs));
}

macro_rules! eq {
    ($lhs:ident, $rhs:ident, $dst:ident) => {
        cmp($lhs, $rhs, $dst, PartialEq::eq)
    };
}

crate::dynamic_auto_vectorization_func!(
    eq_avx512,
    eq_avx2,
    eq_neon,
    eq_default,
    eq_dynamic,
    eq,
    <T>,
    (lhs: &[T], rhs: T, dst: &mut [bool]),
    where T: PartialOrd + Copy
);

macro_rules! ne {
    ($lhs:ident, $rhs:ident, $dst:ident) => {
        cmp($lhs, $rhs, $dst, PartialEq::ne)
    };
}

crate::dynamic_auto_vectorization_func!(
    ne_avx512,
    ne_avx2,
    ne_neon,
    ne_default,
    ne_dynamic,
    ne,
    <T>,
    (lhs: &[T], rhs: T, dst: &mut [bool]),
    where T: PartialOrd + Copy
);

macro_rules! gt {
    ($lhs:ident, $rhs:ident, $dst:ident) => {
        cmp($lhs, $rhs, $dst, PartialOrd::gt)
    };
}

crate::dynamic_auto_vectorization_func!(
    gt_avx512,
    gt_avx2,
    gt_neon,
    gt_default,
    gt_dynamic,
    gt,
    <T>,
    (lhs: &[T], rhs: T, dst: &mut [bool]),
    where T: PartialOrd + Copy
);

macro_rules! ge {
    ($lhs:ident, $rhs:ident, $dst:ident) => {
        cmp($lhs, $rhs, $dst, PartialOrd::ge)
    };
}

crate::dynamic_auto_vectorization_func!(
    ge_avx512,
    ge_avx2,
    ge_neon,
    ge_default,
    ge_dynamic,
    ge,
    <T>,
    (lhs: &[T], rhs: T, dst: &mut [bool]),
    where T: PartialOrd + Copy
);

macro_rules! lt {
    ($lhs:ident, $rhs:ident, $dst:ident) => {
        cmp($lhs, $rhs, $dst, PartialOrd::lt)
    };
}

crate::dynamic_auto_vectorization_func!(
    lt_avx512,
    lt_avx2,
    lt_neon,
    lt_default,
    lt_dynamic,
    lt,
    <T>,
    (lhs: &[T], rhs: T, dst: &mut [bool]),
    where T: PartialOrd + Copy
);

macro_rules! le {
    ($lhs:ident, $rhs:ident, $dst:ident) => {
        cmp($lhs, $rhs, $dst, PartialOrd::le)
    };
}

crate::dynamic_auto_vectorization_func!(
    le_avx512,
    le_avx2,
    le_neon,
    le_default,
    le_dynamic,
    le,
    <T>,
    (lhs: &[T], rhs: T, dst: &mut [bool]),
    where T: PartialOrd + Copy
);
