//! [`FieldRef`] expression

use data_block::array::ArrayImpl;
use data_block::block::DataBlock;
use data_block::types::LogicalType;
use snafu::Snafu;
use std::sync::Arc;

use super::{Error as ExprError, ExprResult, PhysicalExpr, Stringify};

#[allow(missing_docs)]
#[derive(Debug, Snafu)]
pub enum Error {
    #[snafu(display(
        "FieldIndex: `{}` is out of range. The input DataChunk only has {} arrays. 
         Planner should guarantee it never happens, it has fatal bug 😭",
        field_index,
        num_fields
    ))]
    IndexOutOfRange {
        field_index: usize,
        num_fields: usize,
    },
    #[snafu(display(
        "Invalid output array, pipeline executor should guarantee 
         it never happens, it has fatal bug 😭"
    ))]
    InvalidOutputArray {
        source: data_block::array::ArrayError,
    },
}

/// Represents index ino the `DataBlock` that pass through the executor
#[derive(Debug)]
pub struct FieldRef {
    /// Index of the field
    field_index: usize,
    /// Output type of the field
    output_type: LogicalType,
    /// Name of the field, only used for debug and display
    field: String,
    /// It should always be empty! It can not have children
    children: Vec<Arc<dyn PhysicalExpr>>,
}

impl FieldRef {
    /// Create a new [`FieldRef`]
    #[inline]
    pub fn new(field_index: usize, output_type: LogicalType, field: String) -> Self {
        Self {
            field_index,
            output_type,
            field,
            children: Vec::new(),
        }
    }
}

impl Stringify for FieldRef {
    fn name(&self) -> &'static str {
        "FieldRef"
    }

    fn debug(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }

    fn display(&self, f: &mut std::fmt::Formatter<'_>, compact: bool) -> std::fmt::Result {
        if compact {
            write!(f, "{}(#{})", self.field, self.field_index)
        } else {
            write!(
                f,
                "{}(#{}) -> {:?}",
                self.field, self.field_index, self.output_type
            )
        }
    }
}

impl PhysicalExpr for FieldRef {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn output_type(&self) -> &LogicalType {
        &self.output_type
    }

    fn children(&self) -> &[Arc<dyn PhysicalExpr>] {
        &self.children
    }

    fn execute(&self, input: &DataBlock, output: &mut ArrayImpl) -> ExprResult<()> {
        let input = input
            .get_array(self.field_index)
            .ok_or_else(|| ExprError::Execute {
                expr: self.name(),
                source: Box::new(Error::IndexOutOfRange {
                    field_index: self.field_index,
                    num_fields: input.num_arrays(),
                }),
            })?;

        output.reference(input).map_err(|e| ExprError::Execute {
            expr: self.name(),
            source: Box::new(Error::InvalidOutputArray { source: e }),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use data_block::array::{Int32Array, StringArray};
    use expect_test::expect;

    #[test]
    fn test_execute_field_ref() {
        let field_ref = FieldRef::new(1, LogicalType::Integer, "caprice".to_string());

        let input = DataBlock::try_new(vec![
            ArrayImpl::Int32(Int32Array::from_iter([Some(10), Some(-1), None, Some(2)])),
            ArrayImpl::String(StringArray::from_iter([
                None,
                Some("Curvature"),
                Some("Curvature is awesome"),
                Some("DuckDB and morsel driven is awesome"),
            ])),
        ])
        .unwrap();

        let output = &mut ArrayImpl::String(StringArray::new(LogicalType::VarChar).unwrap());

        field_ref.execute(&input, output).unwrap();

        let expect = expect![[r#"
            String(
                StringArray { len: 4, data: [
                    None,
                    Some(
                        Curvature,
                    ),
                    Some(
                        Curvature is awesome,
                    ),
                    Some(
                        DuckDB and morsel driven is awesome,
                    ),
                ]}
                ,
            )
        "#]];
        expect.assert_debug_eq(output);
    }
}
