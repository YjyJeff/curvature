//! Perform regex match on a `StringArray`

use std::sync::Arc;

use data_block::bitmap::Bitmap;
use regex::{Error as RegexError, Regex, RegexBuilder};
use snafu::{ensure, Snafu};

use data_block::array::{Array, ArrayImpl, StringArray};
use data_block::block::DataBlock;
use data_block::compute::regex_match::regex_match_scalar;
use data_block::types::{LogicalType, PhysicalType};

use super::executor::ExprExecCtx;
use super::utils::CompactExprDisplayWrapper;
use super::{execute_single_child, ExprResult, PhysicalExpr, Stringify};

/// Error returned by creating the [`RegexMatch`]
#[derive(Debug, Snafu)]
pub enum RegexMatchError {
    /// Input is invalid
    #[snafu(display("Input of the `RegexMatch` should produce `StringArray`, however the input: `{}` has logical type: `{:?}`", input, logical_type))]
    InvalidInput {
        /// Input passed to create the RegexMatch
        input: String,
        /// Logical type of the input
        logical_type: LogicalType,
    },
    /// Failed to construct regex
    #[snafu(display("Failed to construct a new Regex with pattern: {pattern}"))]
    ConstructRegex {
        /// Pattern used to construct the regex
        pattern: String,
        /// Source
        source: RegexError,
    },
}

/// Perform regex match on a [`StringArray`]
#[derive(Debug)]
pub struct RegexMatch<const NEGATED: bool, const CASE_INSENSITIVE: bool> {
    regex: Regex,
    pattern: String,
    output_type: LogicalType,
    children: [Arc<dyn PhysicalExpr>; 1],
}

impl<const NEGATED: bool, const CASE_INSENSITIVE: bool> RegexMatch<NEGATED, CASE_INSENSITIVE> {
    /// Try to create a [`RegexMatch`] expression
    pub fn try_new(input: Arc<dyn PhysicalExpr>, pattern: String) -> Result<Self, RegexMatchError> {
        ensure!(
            input.output_type().physical_type() == PhysicalType::String,
            InvalidInputSnafu {
                input: CompactExprDisplayWrapper::new(&*input).to_string(),
                logical_type: input.output_type().to_owned()
            }
        );

        // TODO: fast path? like start with, end with, will regex check the fast path?
        match RegexBuilder::new(&pattern)
            .case_insensitive(CASE_INSENSITIVE)
            .build()
        {
            Ok(regex) => Ok(Self {
                regex,
                pattern,
                output_type: LogicalType::Boolean,
                children: [input],
            }),

            Err(error) => Err(RegexMatchError::ConstructRegex {
                pattern,
                source: error,
            }),
        }
    }
}

impl<const NEGATED: bool, const CASE_INSENSITIVE: bool> Stringify
    for RegexMatch<NEGATED, CASE_INSENSITIVE>
{
    fn name(&self) -> &'static str {
        if !NEGATED && !CASE_INSENSITIVE {
            "~"
        } else if !NEGATED && CASE_INSENSITIVE {
            "~*"
        } else if NEGATED && !CASE_INSENSITIVE {
            "!~"
        } else {
            "!~*"
        }
    }

    fn debug(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }

    fn display(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} {}", self.name(), self.pattern)
    }

    fn compact_display(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "(")?;
        self.children[0].compact_display(f)?;
        write!(f, ") {} {}", self.name(), self.pattern)
    }
}

impl<const NEGATED: bool, const CASE_INSENSITIVE: bool> PhysicalExpr
    for RegexMatch<NEGATED, CASE_INSENSITIVE>
{
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
        selection: &mut Bitmap,
        exec_ctx: &mut ExprExecCtx,
        _output: &mut ArrayImpl,
    ) -> ExprResult<()> {
        // Execute input
        execute_single_child(self, leaf_input, selection, exec_ctx)?;

        // Execute self
        let array = unsafe { exec_ctx.intermediate_block.get_array_unchecked(0) };
        let array: &StringArray = array.try_into().unwrap_or_else(|_| {
                panic!(
                    "`{}` expect the input have `StringArray`, however the input array `{}Array` has logical type: `{:?}` with `{}`",
                    CompactExprDisplayWrapper::new(self),
                    array.ident(),
                    array.logical_type(),
                    array.logical_type().physical_type(),
                );
            });

        if array.len() == 1 {
            let element = unsafe { array.get_value_unchecked(0) };
            let mut matched = self.regex.is_match(element.as_str());
            if NEGATED {
                matched = !matched;
            }
            if !matched {
                // All of the elements in the array is not matched
                selection.mutate().set_all_invalid(leaf_input.len());
            }
        } else {
            // Execute regex match on the FlatArray
            // SAFETY: Expression executor guarantees the selection is either empty or has same
            // length with input array(execute expression guarantees the output has same length
            // with input)
            unsafe {
                regex_match_scalar::<NEGATED>(selection, array, &self.regex);
            }

            // Boolean expression, check the selection
            debug_assert!(selection.is_empty() || selection.len() == leaf_input.len());
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exec::physical_expr::field_ref::FieldRef;
    use data_block::array::BooleanArray;

    #[test]
    fn test_regex_match() {
        let input: Arc<dyn PhysicalExpr> =
            Arc::new(FieldRef::new(0, LogicalType::VarChar, "col0".to_string()));
        let regex_match =
            RegexMatch::<false, false>::try_new(Arc::clone(&input), ".a".to_string()).unwrap();
        let mut exec_ctx = ExprExecCtx::new(&regex_match);
        let mut output = ArrayImpl::Boolean(BooleanArray::try_new(LogicalType::Boolean).unwrap());

        let leaf_input = DataBlock::try_new(
            vec![ArrayImpl::String(StringArray::from_iter([
                None,
                Some("Regex expression"),
                Some("HaHa"),
                Some("HAHA"),
            ]))],
            4,
        )
        .unwrap();

        {
            let mut selection = Bitmap::new();
            regex_match
                .execute(&leaf_input, &mut selection, &mut exec_ctx, &mut output)
                .unwrap();

            assert_eq!(
                selection.iter().collect::<Vec<_>>(),
                [false, false, true, false]
            )
        }

        {
            let mut selection = Bitmap::new();
            let regex_match =
                RegexMatch::<true, false>::try_new(Arc::clone(&input), ".a".to_string()).unwrap();
            regex_match
                .execute(&leaf_input, &mut selection, &mut exec_ctx, &mut output)
                .unwrap();

            assert_eq!(
                selection.iter().collect::<Vec<_>>(),
                [false, true, false, true]
            )
        }

        {
            let mut selection = Bitmap::new();
            let regex_match =
                RegexMatch::<false, true>::try_new(Arc::clone(&input), ".a".to_string()).unwrap();
            regex_match
                .execute(&leaf_input, &mut selection, &mut exec_ctx, &mut output)
                .unwrap();

            assert_eq!(
                selection.iter().collect::<Vec<_>>(),
                [false, false, true, true]
            )
        }

        {
            let mut selection = Bitmap::new();
            let regex_match =
                RegexMatch::<true, true>::try_new(Arc::clone(&input), ".a".to_string()).unwrap();
            regex_match
                .execute(&leaf_input, &mut selection, &mut exec_ctx, &mut output)
                .unwrap();

            assert_eq!(
                selection.iter().collect::<Vec<_>>(),
                [false, true, false, false]
            )
        }
    }
}
