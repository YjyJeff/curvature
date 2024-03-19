//! Regex match

use crate::array::{Array, BooleanArray, StringArray};
use regex::Regex;

/// Match regex on a [`StringArray`]
///
/// Note that compile the regex is pretty expensive, caller should compile it and use
/// the compiled Regex for different [`StringArray`]s.
///
/// # Safety
///
/// - `array`'s validity should not reference `dst`'s validity. In the computation graph,
/// `array` must be the descendant of `dst`
///
/// - No other arrays that reference the `dst`'s data and validity are accessed! In the
/// computation graph, it will never happens
pub unsafe fn regex_match_scalar(array: &StringArray, regex: &Regex, dst: &mut BooleanArray) {
    dst.validity.reference(&array.validity);
    dst.data.as_mut().reset(
        array.len(),
        array.values_iter().map(|v| regex.is_match(v.as_str())),
    )
}
