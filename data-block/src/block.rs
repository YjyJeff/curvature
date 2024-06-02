//! [`DataBlock`] is a collection of [`ArrayImpl`]

use std::fmt::Display;

use snafu::{ensure, AsErrorSource, ResultExt, Snafu};

use crate::array::{ArrayError, ArrayImpl};
use crate::bitmap::Bitmap;
use crate::types::LogicalType;
use tabled::builder::Builder as TableBuilder;

/// The length of array does not equal to the length of the data block
#[derive(Debug, Snafu)]
#[snafu(display("The `{index}`th FlatArray has length: `{array_len}`, it does not equal to the specified length: `{length}`"))]
pub struct InconsistentLengthError {
    /// Index of the array in data block
    pub index: usize,
    /// length of the array
    pub array_len: usize,
    /// length of the data block
    pub length: usize,
}

/// Error happened when mutating the array
#[derive(Debug, Snafu)]
pub enum MutateArrayError<E: AsErrorSource> {
    /// Inner error when mutating the array
    Inner {
        /// Source
        source: E,
    },
    /// The array's length after mutation is invalid
    Length {
        /// inner
        inner: InconsistentLengthError,
    },
}

/// [`DataBlock`] is a collection of [`ArrayImpl`]
///
/// FIXME: All of the mutation should know its length in advance!
#[derive(Debug)]
pub struct DataBlock {
    /// Arrays in the data block. There are two kinds of arrays here:
    ///
    /// - `FlatArray`: array with length > 1
    ///
    /// - `ConstantArray`: array with length = 1. It is physically represented as a
    /// `FlatArray` with length = 1. It represents all of the elements in the
    /// array has same value. It can represent array with any length
    ///
    /// All of the `FlatArray`s must have length equal to `self.length`
    arrays: Vec<ArrayImpl>,
    /// Number of element in the data block.
    ///
    /// If the [`Self::arrays`] is empty and num_rows > 0, it means we only pass the
    /// length to other operator
    length: usize,
}

impl DataBlock {
    /// Create a new [`DataBlock`] with all of the arrays have same length
    pub fn try_new(arrays: Vec<ArrayImpl>, length: usize) -> Result<Self, InconsistentLengthError> {
        arrays.iter().enumerate().try_for_each(|(index, array)| {
            let array_len = array.len();
            ensure!(
                array_len == length || array_len == 1,
                InconsistentLengthSnafu {
                    index,
                    array_len,
                    length,
                }
            );
            Ok::<_, InconsistentLengthError>(())
        })?;

        Ok(Self { arrays, length })
    }

    /// Create a new [`DataBlock`] without check
    ///
    /// # Safety
    ///
    /// All of the `FlatArray`s must have the given length
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

    /// Get a mutable reference to the array
    ///
    /// # Notes
    ///
    /// - The data block should only has single array
    ///
    /// - Before mutation, the length of the array after mutation is know. Caller should
    /// guarantee if the array after mutation is a `FlatArray`, it should have same length
    /// with the specified `length` argument
    #[inline]
    pub fn mutate_single_array(&mut self, length: usize) -> MutateArrayGuard<'_> {
        debug_assert_eq!(self.num_arrays(), 1);

        self.length = length;

        MutateArrayGuard {
            length,
            array: &mut self.arrays[0],
        }
    }

    /// Get mutable arrays
    ///
    /// # Notes
    ///
    /// Before mutation, the length of the array after mutation is know. Caller should
    /// guarantee if the array after mutation is a `FlatArray`, it should have same length
    /// with the specified `length` argument
    #[inline]
    pub fn mutate_arrays(&mut self, length: usize) -> MutateArraysGuard<'_> {
        self.length = length;

        MutateArraysGuard {
            length,
            arrays: &mut self.arrays,
        }
    }

    /// Get the mutable array without check. It is totally unsafe! It will not preserve
    /// the length invariance of the data block
    ///
    /// # Safety
    ///
    /// Only use it when you know the length invariance can be broken
    pub unsafe fn arrays_mut_unchecked(&mut self) -> &mut [ArrayImpl] {
        &mut self.arrays
    }

    /// Set the length to new_len
    ///
    /// # Safety
    ///
    /// The arrays in the data block should be empty. Otherwise, the invariance
    /// may break
    pub unsafe fn set_len(&mut self, new_len: usize) {
        debug_assert_eq!(self.num_arrays(), 0);
        self.length = new_len;
    }

    /// # Safety
    ///
    /// - If the `selection` is not empty, `source` and `selection` should have same length.
    /// Otherwise, undefined behavior happens
    ///
    /// - If the `selection` is not empty, its count_ones must equal to length
    ///
    /// - `selection` should not be referenced by any array
    pub unsafe fn filter(
        &mut self,
        selection: &Bitmap,
        source: &Self,
        length: usize,
    ) -> Result<(), ArrayError> {
        debug_assert!(selection.is_empty() || selection.count_ones() == Some(length));
        self.length = length;
        self.arrays
            .iter_mut()
            .zip(&source.arrays)
            .try_for_each(|(lhs, rhs)| lhs.filter(selection, rhs))
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

        (0..self.length).for_each(|mut index| {
            table_builder.push_record(self.arrays.iter().map(|array| {
                if array.len() == 1 {
                    // Constant Array
                    index = 0;
                }
                array
                    .get(index)
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
    length: usize,
    array: &'a mut ArrayImpl,
}

impl MutateArrayGuard<'_> {
    /// Mutate the array
    ///
    /// The function passed to mutate the array must guarantee the arrays after
    /// mutation is a `ConstantArray` or `FlatArray` that has same length with the length
    /// passed to create the guard
    #[inline]
    pub fn mutate<F, E>(self, mutate_func: F) -> Result<(), MutateArrayError<E>>
    where
        E: AsErrorSource + snafu::Error,
        F: FnOnce(&mut ArrayImpl) -> Result<(), E>,
    {
        mutate_func(self.array).context(InnerSnafu)?;

        let array_len = self.array.len();
        ensure!(
            array_len == self.length || array_len == 1,
            LengthSnafu {
                inner: InconsistentLengthError {
                    index: 0_usize,
                    array_len,
                    length: self.length
                }
            }
        );

        Ok(())
    }
}

/// Struct for mutate the data block
#[derive(Debug)]
pub struct MutateArraysGuard<'a> {
    length: usize,
    arrays: &'a mut [ArrayImpl],
}

impl MutateArraysGuard<'_> {
    /// Mutate the arrays
    ///
    /// The function passed to mutate the arrays must guarantee all of the arrays after
    /// mutation is a `ConstantArray` or `FlatArray` that has same length with the length
    /// passed to create the guard
    #[inline]
    pub fn mutate<F, E>(self, mutate_func: F) -> std::result::Result<(), MutateArrayError<E>>
    where
        E: AsErrorSource + snafu::Error,
        F: FnOnce(&mut [ArrayImpl]) -> std::result::Result<(), E>,
    {
        mutate_func(self.arrays).context(InnerSnafu)?;

        self.arrays
            .iter()
            .enumerate()
            .try_for_each(|(index, array)| {
                let array_len = array.len();
                ensure!(
                    array_len == self.length || array_len == 1,
                    LengthSnafu {
                        inner: InconsistentLengthError {
                            index,
                            array_len,
                            length: self.length
                        }
                    }
                );
                Ok::<_, MutateArrayError<E>>(())
            })
    }
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
        let block = DataBlock::try_new(
            vec![
                ArrayImpl::Int32([Some(10), None, Some(-1)].into_iter().collect()),
                ArrayImpl::Float32([None, Some(-1.0), Some(9.9)].into_iter().collect()),
            ],
            3,
        )
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
