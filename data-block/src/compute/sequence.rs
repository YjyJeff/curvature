//! Generate sequence

use crate::array::UInt64Array;
use crate::dynamic_func;

/// Replace the array with sequence (start..end). Caller should guarantee `end >= start`
///
/// # Safety
///
/// No other arrays that reference the `array`'s data and validity are accessed! In the
/// computation graph, it will never happens
pub unsafe fn sequence(array: &mut UInt64Array, start: u64, end: u64) {
    array.validity.as_mut().mutate().clear();

    let array = array.data.as_mut().clear_and_resize((end - start) as usize);

    sequence_assign_dynamic(array, start)
}

macro_rules! sequence_assign {
    ($array:ident, $start:ident) => {
        $array
            .iter_mut()
            .enumerate()
            .for_each(|(i, v)| *v = (i as u64 + $start))
    };
}

dynamic_func!(
    sequence_assign,
    ,
    (array: &mut [u64], start: u64),
);
