//! Utils

/// Computing the smallest value that is multiple of `base` and greater than
/// or equal to size.
///
/// Note that `base` must be power of two. Otherwise, the returned value is incorrect!
#[inline]
pub fn roundup_to_multiple_of_pow_of_two_base(size: usize, base: usize) -> usize {
    let mask = base - 1;
    (size + mask) & !mask
}

/// Calculate the number of loops
#[inline]
pub(crate) fn roundup_loops(len: usize, batch: usize) -> usize {
    len / batch + (len % batch != 0) as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_roundup_to_multiple_of_pow_of_two_base() {
        assert_eq!(roundup_to_multiple_of_pow_of_two_base(56, 16), 64);
        assert_eq!(roundup_to_multiple_of_pow_of_two_base(7, 16), 16);
        assert_eq!(roundup_to_multiple_of_pow_of_two_base(48, 16), 48);
    }
}
