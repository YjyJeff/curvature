//! Generate sequence

use crate::array::UInt64Array;
use crate::{dynamic_func, mutate_array_func};

mutate_array_func!(
    /// Replace the array with sequence (start..end)
    #[inline]
    pub unsafe fn sequence(array: &mut UInt64Array, start: u64, end: u64) {
        array.validity.exactly_once_mut().clear();

        let array = array
            .data
            .exactly_once_mut()
            .clear_and_resize((end - start) as usize);

        sequence_assign_dynamic(array, start)
    }
);

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
