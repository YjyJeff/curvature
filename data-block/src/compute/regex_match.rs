//! Regex match

use crate::array::{Array, StringArray};
use crate::bitmap::Bitmap;
use regex::Regex;

/// Match regex on a [`StringArray`]
///
/// Note that compile the regex is pretty expensive, caller should compile it and use
/// the compiled Regex for different [`StringArray`]s.
///
/// # Safety
///
/// `dst` should not be referenced by any array
pub unsafe fn regex_match_scalar(array: &StringArray, regex: &Regex, dst: &mut Bitmap) {
    let validity = array.validity();
    let mut dst = dst.mutate();
    if validity.all_valid() {
        dst.reset(
            array.len(),
            array.values_iter().map(|v| regex.is_match(v.as_str())),
        )
    } else {
        dst.reset(
            array.len(),
            array
                .values_iter()
                .zip(validity.iter())
                .map(|(v, valid)| valid && regex.is_match(v.as_str())),
        )
    }
}

/// Match regex on a [`StringArray`] with given selection array. The selection array will
/// be modified according to the result of the match.
///
/// Note that compile the regex is pretty expensive, caller should compile it and use
/// the compiled Regex for different [`StringArray`]s.
///
/// # Safety
///
/// - `array` and `selection` should have same length. Otherwise, undefined behavior happens
///
/// - `selection` should not be referenced by any array
pub unsafe fn selected_regex_match_scalar(
    selection: &mut Bitmap,
    array: &StringArray,
    regex: &Regex,
) {
    debug_assert_eq!(array.len(), selection.len());

    let selection_all_valid = selection.all_valid();
    let mut guard = selection.mutate();
    let validity = array.validity();
    if selection_all_valid {
        if validity.all_valid() {
            guard.reset(
                array.len(),
                array
                    .values_iter()
                    .map(|view| regex.is_match(view.as_str())),
            );
        } else {
            guard.reset(
                array.len(),
                array
                    .values_iter()
                    .zip(validity.iter())
                    .map(|(view, valid)| valid && regex.is_match(view.as_str())),
            )
        }
    } else if validity.all_valid() {
        guard.mutate_ones(|index| regex.is_match(array.get_value_unchecked(index).as_str()));
    } else {
        guard.mutate_ones(|index| {
            validity.get_unchecked(index)
                && regex.is_match(array.get_value_unchecked(index).as_str())
        });
    }
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

        unsafe {
            selected_regex_match_scalar(&mut selection, &array, &regex);
        }

        assert_eq!(selection.iter_ones().collect::<Vec<_>>(), [0, 2]);
    }
}
