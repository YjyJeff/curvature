//! Utils for memory computation

/// Computing the smallest value that is multiple of align and greater than
/// or equal to size.
///
/// Note that `align` is the alignment, therefore, it must be power of two.
/// Otherwise, the returned value is incorrect!
#[inline]
pub fn next_multiple_of_align(size: usize, align: usize) -> usize {
    let mask = align - 1;
    if size & mask == 0 {
        // size is the multiple of align
        size
    } else {
        (size + mask) & !mask
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_next_multiple_of_align() {
        assert_eq!(next_multiple_of_align(56, 16), 64);
        assert_eq!(next_multiple_of_align(7, 16), 16);
        assert_eq!(next_multiple_of_align(48, 16), 48);
    }
}
