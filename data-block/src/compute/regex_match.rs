//! Regex match

use crate::array::{Array, BooleanArray, StringArray};
use crate::bitmap::Bitmap;
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
    dst.data.as_mut().mutate().reset(
        array.len(),
        array.values_iter().map(|v| regex.is_match(v.as_str())),
    )
}

/// Match regex on a [`StringArray`] with given selection array. The selection array will
/// be modified according to the result of the match.
///
/// Note that compile the regex is pretty expensive, caller should compile it and use
/// the compiled Regex for different [`StringArray`]s.
///
/// # Safety
///
/// - `array` and `selection` must have same length. Otherwise, undefined behavior happens
///
/// - `selection` should not be referenced by any array
pub unsafe fn selected_regex_match_scalar(
    selection: &mut Bitmap,
    array: &StringArray,
    regex: &Regex,
) {
    debug_assert_eq!(array.len(), selection.len());

    let mut guard = selection.mutate();
    guard.mutate_ones(|index| {
        // SAFETY: safety contract guarantees `array` and `selection` has same length. The
        // index is always valid
        if let Some(v) = array.get_unchecked(index) {
            regex.is_match(v.as_str())
        } else {
            false
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_selected_regex_match_scalar() {
        let regex = Regex::new("y.y").unwrap();
        let array = StringArray::from_iter([
            Some("yjy"),
            None,
            Some("yyy"),
            Some("abc"),
            Some("data block"),
        ]);

        let mut selection = Bitmap::new();
        selection
            .mutate()
            .clear_and_resize(5)
            .iter_mut()
            .for_each(|v| *v = u64::MAX);

        unsafe {
            selected_regex_match_scalar(&mut selection, &array, &regex);
        }

        assert_eq!(selection.iter_ones().collect::<Vec<_>>(), [0, 2]);
    }
}
