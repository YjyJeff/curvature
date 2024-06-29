//! [`FieldRef`] expression

use data_block::array::ArrayImpl;
use data_block::bitmap::Bitmap;
use data_block::block::DataBlock;
use data_block::compute::logical::and_inplace;
use data_block::types::{Array, LogicalType};
use std::sync::Arc;

use super::executor::ExprExecCtx;
use super::{ExprResult, PhysicalExpr, Stringify};

/// Represents a reference of the field in the `DataBlock` that passed through the executor
///
/// # Note
///
/// It is the only one expression that do not store the result in the `selection` if the
/// field has `Boolean` type
#[derive(Debug)]
pub struct FieldRef {
    /// Index of the field
    pub(crate) field_index: usize,
    /// Output type of the field
    pub(crate) output_type: LogicalType,
    /// Name of the field, only used for debug and display
    field: String,
    /// It should always be empty! It can not have children
    children: [Arc<dyn PhysicalExpr>; 0],
    /// Alias of the field
    alias: String,
}

impl FieldRef {
    /// Create a new [`FieldRef`]
    pub fn new(field_index: usize, output_type: LogicalType, field: String) -> Self {
        Self::new_with_alias(field_index, output_type, field, String::new())
    }

    /// Create a new [`FieldRef`] expression with explicit alias
    pub fn new_with_alias(
        field_index: usize,
        output_type: LogicalType,
        field: String,
        alias: String,
    ) -> Self {
        Self {
            field_index,
            output_type,
            field,
            children: [],
            alias,
        }
    }

    /// Copy the filed into the selection array. Only the [`Conjunction`] expression will call it
    ///
    /// [`Conjunction`]: crate::exec::physical_expr::conjunction::Conjunction
    ///
    /// # Safety
    ///
    /// `selection` should be empty or has the same length with `input.len()`
    pub(crate) unsafe fn copy_into_selection(&self, input: &DataBlock, selection: &mut Bitmap) {
        debug_assert!(selection.is_empty() || selection.len() == input.len());

        let array = input.get_array(self.field_index).unwrap_or_else(|| {
            panic!(
                "`FieldRef` with index {}, however the leaf_input only have `{}` arrays",
                self.field_index,
                input.num_arrays(),
            )
        });

        let ArrayImpl::Boolean(array) = array else {
            panic!("`copy_into_selection` requires the input array is an `BooleanArray`, found `{}`Array", array.ident())
        };

        if array.len() == 1 {
            // Flatten it !
            let valid = array.validity().all_valid();
            if !valid || !array.data().all_valid() {
                selection.mutate().set_all_invalid(input.len());
            }
        } else {
            and_inplace(selection, array.data());
            and_inplace(selection, array.validity());
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
        write!(f, "{}", self.field)
    }

    fn alias(&self) -> &str {
        &self.alias
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
        _selection: &mut Bitmap,
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

        debug_assert!(output.len() == 1 || output.len() == leaf_input.len());

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

        let input = DataBlock::try_new(
            vec![
                ArrayImpl::Int32(Int32Array::from_iter([Some(10), Some(-1), None, Some(2)])),
                ArrayImpl::String(StringArray::from_iter([
                    None,
                    Some("Curvature"),
                    Some("Curvature is awesome"),
                    Some("DuckDB and morsel driven is awesome"),
                ])),
            ],
            4,
        )
        .unwrap();

        let output = &mut ArrayImpl::String(StringArray::new(LogicalType::VarChar).unwrap());

        field_ref
            .execute(
                &input,
                &mut Bitmap::new(),
                &mut ExprExecCtx::new(&field_ref),
                output,
            )
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
