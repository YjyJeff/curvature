//! Utils

#[inline]
pub fn roundup_to_multiple_of(val: usize, base: usize) -> usize {
    (val + (base - 1)) & !(base - 1)
}

/// Calculate the number of loops
#[inline]
pub fn roundup_loops(len: usize, batch: usize) -> usize {
    len / batch + (len % batch != 0) as usize
}
