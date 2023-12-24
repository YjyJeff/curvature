//! utils

use super::ArrayImpl;

/// Get the physical array name, used for error handling
pub fn physical_array_name(array: &ArrayImpl) -> String {
    format!("{:?}Array", array.logical_type().physical_type())
}
