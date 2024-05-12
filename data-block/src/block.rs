//! [`DataBlock`] is a collection of [`ArrayImpl`]

use std::fmt::Display;
use std::ops::{Deref, DerefMut};

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

    /// Get a reference to the array with given index without check
    ///
    /// # Safety
    ///
    /// index should be valid
    #[inline]
    pub unsafe fn get_array_unchecked(&self, index: usize) -> &ArrayImpl {
        self.arrays.get_unchecked(index)
    }

    /// Get a mutable reference to the array. Caller should guarantee the data block
    /// only has single array
    #[inline]
    pub fn mutate_single_array(&mut self) -> MutateArrayGuard<'_> {
        debug_assert_eq!(self.num_arrays(), 1);

        MutateArrayGuard {
            length: &mut self.length,
            array: &mut self.arrays[0],
        }
    }

    /// Get arrays
    #[inline]
    pub fn arrays(&self) -> &[ArrayImpl] {
        &self.arrays
    }

    /// Get the logical types of the data block
    #[inline]
    pub fn logical_types(&self) -> impl Iterator<Item = &LogicalType> {
        self.arrays.iter().map(|array| array.logical_type())
    }

    /// Get mutable arrays
    #[inline]
    pub fn mutate_arrays(&mut self) -> MutateArraysGuard<'_> {
        MutateArraysGuard {
            length: &mut self.length,
            arrays: &mut self.arrays,
        }
    }

    /// Get the guard to mutate the data block
    ///
    /// # Safety
    ///
    /// Caller should guarantee after the mutation, all of the arrays have same length
    /// and the length field is calibrated
    #[inline]
    pub unsafe fn mutate(&mut self) -> MutateDataBlockGuard<'_> {
        MutateDataBlockGuard {
            arrays: &mut self.arrays,
            length: &mut self.length,
        }
    }

    /// Format the data block with given table builder
    /// FIXME: contains logical type info
    fn fmt_table(&self, table_builder: &mut TableBuilder, with_logical_type: bool) {
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

/// Struct for mutate the data block with single array
#[derive(Debug)]
pub struct MutateArrayGuard<'a> {
    length: &'a mut usize,
    array: &'a mut ArrayImpl,
}

impl Deref for MutateArrayGuard<'_> {
    type Target = ArrayImpl;

    #[inline]
    fn deref(&self) -> &Self::Target {
        self.array
    }
}

impl DerefMut for MutateArrayGuard<'_> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.array
    }
}

/// Set DataBlock's length when array has mutated
impl Drop for MutateArrayGuard<'_> {
    #[inline]
    fn drop(&mut self) {
        *self.length = self.array.len();
    }
}

/// Struct for mutate the data block
#[derive(Debug)]
pub struct MutateArraysGuard<'a> {
    length: &'a mut usize,
    arrays: &'a mut [ArrayImpl],
}

impl MutateArraysGuard<'_> {
    /// Mutate the arrays
    ///
    /// # Safety
    ///
    /// The function passed to mutate the arrays must guarantee all of the arrays after
    /// mutation should have same length
    #[inline]
    pub unsafe fn mutate<F, E>(self, mutate_func: F) -> std::result::Result<(), E>
    where
        F: FnOnce(&mut [ArrayImpl]) -> std::result::Result<(), E>,
    {
        mutate_func(self.arrays)?;
        if let Some(array) = self.arrays.first() {
            *self.length = array.len();
        }
        #[cfg(debug_assertions)]
        {
            self.arrays.iter().enumerate().for_each(|(i, array)| {
                if array.len() != *self.length {
                    panic!(
                        "Mutate arrays in the data block should produce same length. First len: {}, {}th len: {}", 
                        *self.length,
                        i,
                        array.len()
                    )
                }
            });
        }
        Ok(())
    }
}

/// Struct for mutate the data block, it will expose the inner arrays. This struct is
/// totally unsafe, more danger than [`MutateArraysGuard`]. User should calibrate the
/// length manually. Only use it when you want the maximum freedom
#[derive(Debug)]
pub struct MutateDataBlockGuard<'a> {
    /// Inner arrays
    pub arrays: &'a mut [ArrayImpl],
    /// Used to calibrate the length
    pub length: &'a mut usize,
}

/// DataBlock that can be sent between threads
#[derive(Debug)]
#[repr(transparent)]
pub struct SendableDataBlock(DataBlock);

impl AsRef<DataBlock> for SendableDataBlock {
    #[inline]
    fn as_ref(&self) -> &DataBlock {
        &self.0
    }
}

impl SendableDataBlock {
    /// Create a new [`SendableDataBlock`]
    ///
    /// # Safety
    ///
    /// All of the arrays in the `block` do not referenced by other arrays
    #[inline]
    pub unsafe fn new(block: DataBlock) -> Self {
        Self(block)
    }

    /// Get [`DataBlock`]
    #[inline]
    pub fn into_inner(self) -> DataBlock {
        self.0
    }
}

unsafe impl Send for SendableDataBlock {}

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
