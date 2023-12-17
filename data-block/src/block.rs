//! [`DataBlock`] is a collection of [`ArrayImpl`]

use crate::array::ArrayImpl;

/// [`DataBlock`] is a collection of [`ArrayImpl`]
#[derive(Debug)]
pub struct DataBlock {
    arrays: Vec<ArrayImpl>,
    /// Number of element in the data block. All of the arrays should have same length.
    /// If the [`Self::arrays`] is empty and num_rows > 0, it means we only pass the
    /// length to other operator
    length: usize,
}

impl DataBlock {
    /// Get number of elements in the data block
    #[inline]
    pub fn length(&self) -> usize {
        self.length
    }

    /// Get number of arrays in the data block
    #[inline]
    pub fn num_arrays(&self) -> usize {
        self.arrays.len()
    }
}
