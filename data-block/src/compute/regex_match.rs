//! Regex match

use crate::array::{Array, BooleanArray, StringArray};
use crate::mutate_array_func;
use regex::Regex;

mutate_array_func!(
    /// Match regex on a [`StringArray`]
    ///
    /// Note that compile the regex is pretty expensive, caller should compile it and use
    /// the compiled Regex for different [`StringArray`]s.
    pub unsafe fn regex_match_scalar(array: &StringArray, regex: &Regex, dst: &mut BooleanArray) {
        dst.validity.reference(&array.validity);
        dst.data.exactly_once_mut().reset(
            array.len(),
            array.values_iter().map(|v| regex.is_match(v.as_str())),
        )
    }
);
