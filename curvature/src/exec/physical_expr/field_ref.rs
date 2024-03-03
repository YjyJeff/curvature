//! [`FieldRef`] expression

use data_block::array::ArrayImpl;
use data_block::block::DataBlock;
use data_block::types::LogicalType;
use snafu::Snafu;
use std::sync::Arc;

use super::executor::ExprExecCtx;
use super::{ExprResult, PhysicalExpr, Stringify};

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
}

/// Represents a reference of the field in the `DataBlock` that passed through the executor
#[derive(Debug)]
pub struct FieldRef {
    /// Index of the field
    pub(crate) field_index: usize,
    /// Output type of the field
    pub(crate) output_type: LogicalType,
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

    fn display(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}(#{}) -> {:?}",
            self.field, self.field_index, self.output_type
        )
    }

    fn compact_display(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}(#{})", self.field, self.field_index)
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

    fn execute(
        &self,
        leaf_input: &DataBlock,
        _exec_ctx: &mut ExprExecCtx,
        output: &mut ArrayImpl,
    ) -> ExprResult<()> {
        let input = leaf_input.get_array(self.field_index).unwrap_or_else(|| {
            panic!(
                "`FieldRef` with index {}, however the leaf_input only have `{}` arrays",
                self.field_index,
                leaf_input.num_arrays(),
            )
        });

        output
            .reference(input)
            .unwrap_or_else(|e| panic!("`FieldRef`'s output is invalid, {}", e));

        Ok(())
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

        field_ref
            .execute(&input, &mut ExprExecCtx::new(&field_ref), output)
            .unwrap();

        let expect = expect![[r#"
            String(
                StringArray { logical_type: VarChar, len: 4, data: [
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
