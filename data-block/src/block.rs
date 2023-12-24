//! [`DataBlock`] is a collection of [`ArrayImpl`]

use snafu::{ensure, Snafu};

use crate::array::ArrayImpl;
use crate::types::LogicalType;

#[allow(missing_docs)]
#[derive(Debug, Snafu)]
#[snafu(display("Arrays have different length. Arrays: {arrays:?}"))]
pub struct InconsistentLengthError {
    arrays: Vec<ArrayImpl>,
}

type Result<T> = std::result::Result<T, InconsistentLengthError>;

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
    /// Create a new [`DataBlock`] with all of the arrays have same length
    pub fn try_new(arrays: Vec<ArrayImpl>) -> Result<Self> {
        let mut iter = arrays.iter();
        let Some(length) = iter.next().map(|array| array.len()) else {
            return Ok(Self { arrays, length: 0 });
        };

        ensure!(
            iter.all(|array| array.len() == length),
            InconsistentLengthSnafu { arrays }
        );

        Ok(Self { arrays, length })
    }

    /// Create a new [`DataBlock`] without check
    ///
    /// # Safety
    ///
    /// All of the arrays must have the given length
    #[inline]
    pub unsafe fn new_unchecked(arrays: Vec<ArrayImpl>, length: usize) -> Self {
        Self { arrays, length }
    }

    /// Create a new [`DataBlock`] with no arrays, only provide length to it
    #[inline]
    pub fn new_length_only(length: usize) -> Self {
        Self {
            arrays: Vec::new(),
            length,
        }
    }

    /// Create a new [`DataBlock`] with given logical types, all of the arrays
    /// will be empty
    #[inline]
    pub fn with_logical_types(logical_types: Vec<LogicalType>) -> Self {
        // SAFETY: ArrayImpl return empty array
        unsafe { Self::new_unchecked(logical_types.into_iter().map(ArrayImpl::new).collect(), 0) }
    }

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

    /// Get a reference to the array with given index
    #[inline]
    pub fn get_array(&self, index: usize) -> Option<&ArrayImpl> {
        self.arrays.get(index)
    }

    /// Get a mutable reference to the array with given index
    #[inline]
    pub fn get_mutable_array(&mut self, index: usize) -> Option<&mut ArrayImpl> {
        self.arrays.get_mut(index)
    }

    /// Get arrays
    #[inline]
    pub fn arrays(&self) -> &[ArrayImpl] {
        &self.arrays
    }

    /// Get mutable arrays
    #[inline]
    pub fn mutable_arrays(&mut self) -> &mut [ArrayImpl] {
        &mut self.arrays
    }
}
