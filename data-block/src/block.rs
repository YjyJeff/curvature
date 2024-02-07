//! [`DataBlock`] is a collection of [`ArrayImpl`]

use std::fmt::Display;

use snafu::{ensure, Snafu};

use crate::array::ArrayImpl;
use crate::types::LogicalType;
use tabled::builder::Builder as TableBuilder;

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
    pub fn len(&self) -> usize {
        self.length
    }

    /// Returns true if the length is 0
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.length == 0
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

    /// Format the data block with given table builder
    pub fn fmt_table(&self, table_builder: &mut TableBuilder, with_logical_type: bool) {
        if with_logical_type {
            table_builder.push_record(
                self.arrays
                    .iter()
                    .map(|array| format!("{:?}", array.logical_type())),
            );
        }

        (0..self.length).for_each(|index| unsafe {
            table_builder.push_record(self.arrays.iter().map(|array| {
                array
                    .get_unchecked(index)
                    .map_or_else(|| "Null".to_string(), |element| element.to_string())
            }));
        });
    }
}

impl Display for DataBlock {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut table_builder = TableBuilder::default();
        self.fmt_table(&mut table_builder, true);
        write!(
            f,
            "{}",
            table_builder
                .build()
                .with(tabled::settings::style::Style::modern())
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_display_data_block() {
        let block = DataBlock::try_new(vec![
            ArrayImpl::Int32([Some(10), None, Some(-1)].into_iter().collect()),
            ArrayImpl::Float32([None, Some(-1.0), Some(9.9)].into_iter().collect()),
        ])
        .unwrap();

        let expect = expect_test::expect![[r#"
            ┌─────────┬───────┐
            │ Integer │ Float │
            ├─────────┼───────┤
            │ 10      │ Null  │
            ├─────────┼───────┤
            │ Null    │ -1.0  │
            ├─────────┼───────┤
            │ -1      │ 9.9   │
            └─────────┴───────┘"#]];
        expect.assert_eq(&block.to_string());
    }
}
