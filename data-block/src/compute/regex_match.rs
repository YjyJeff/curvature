//! Regex match

use crate::array::{Array, StringArray};
use crate::bitmap::Bitmap;
use crate::compute::logical::and_inplace;
use regex::Regex;

macro_rules! return_with_negated {
    ($negated:ident, $matched:ident) => {
        if $negated { !$matched } else { $matched }
    };
}

/// Match regex on a [`StringArray`] with given selection array. The selection array will
/// be modified according to the result of the match.
///
/// Note that compile the regex is pretty expensive, caller should compile it and use
/// the compiled Regex for different [`StringArray`]s.
///
/// # Safety
///
/// - If the `selection` is not empty, `array` and `selection` should have same length.
///   Otherwise, undefined behavior happens
///
/// - `selection` should not be referenced by any array
pub unsafe fn regex_match_scalar<const NEGATED: bool>(
    selection: &mut Bitmap,
    array: &StringArray,
    regex: &Regex,
) {
    unsafe {
        #[cfg(feature = "verify")]
        assert_selection_is_valid!(selection, array);

        let validity = array.validity();

        if validity.all_valid() && selection.all_valid() {
            selection.mutate().reset(
                array.len(),
                array.values_iter().map(|view| {
                    let matched = regex.is_match(view.as_str());
                    return_with_negated!(NEGATED, matched)
                }),
            );
        } else {
            and_inplace(selection, validity);
            selection.mutate().mutate_ones(|index| {
                let matched = regex.is_match(array.get_value_unchecked(index).as_str());
                return_with_negated!(NEGATED, matched)
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_regex_match_scalar() {
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
            regex_match_scalar::<false>(&mut selection, &array, &regex);
        }

        assert_eq!(selection.iter_ones().collect::<Vec<_>>(), [0, 2]);

        unsafe {
            regex_match_scalar::<false>(&mut selection, &array, &regex);
        }

        assert_eq!(selection.iter_ones().collect::<Vec<_>>(), [0, 2]);
    }
}
