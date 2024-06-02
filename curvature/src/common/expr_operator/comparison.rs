//! Compare expressions

use std::fmt::Display;

use data_block::types::LogicalType;

/// Comparison operators
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CmpOperator {
    /// ==
    Equal,
    /// !=
    NotEqual,
    /// >
    GreaterThan,
    /// >=
    GreaterThanOrEqualTo,
    /// <
    LessThan,
    /// <=
    LessThanOrEqualTo,
}

impl CmpOperator {
    /// Get the symbol
    pub fn symbol_ident(&self) -> &'static str {
        match self {
            Self::Equal => "==",
            Self::NotEqual => "!=",
            Self::GreaterThan => ">",
            Self::GreaterThanOrEqualTo => ">=",
            Self::LessThan => "<",
            Self::LessThanOrEqualTo => "<=",
        }
    }

    /// Get the ident of the op
    pub fn ident(&self) -> &'static str {
        match self {
            Self::Equal => "Equal",
            Self::NotEqual => "NotEqual",
            Self::GreaterThan => "GreaterThan",
            Self::GreaterThanOrEqualTo => "GreaterThanOrEqualTo",
            Self::LessThan => "LessThan",
            Self::LessThanOrEqualTo => "LessThanOrEqualTo",
        }
    }
}

impl Display for CmpOperator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.symbol_ident())
    }
}

/// Returns true if these two logical types can perform comparison
///
/// FIXME: Support comparison between different types?
///
/// # Notes
///
/// If comparison between different types is supported, you should also support it in the
/// [`Comparison`](crate::exec::physical_expr::comparison::Comparison) expression
pub fn can_compare(left: &LogicalType, right: &LogicalType, op: CmpOperator) -> bool {
    use LogicalType::*;
    match (left, right) {
        (Boolean, Boolean)
        | (TinyInt, TinyInt)
        | (SmallInt, SmallInt)
        | (Integer, Integer)
        | (BigInt, BigInt)
        | (HugeInt, HugeInt)
        | (UnsignedTinyInt, UnsignedTinyInt)
        | (UnsignedSmallInt, UnsignedSmallInt)
        | (UnsignedInteger, UnsignedInteger)
        | (UnsignedBigInt, UnsignedBigInt)
        | (Float, Float)
        | (Double, Double)
        | (VarChar, VarChar)
        | (VarBinary, VarBinary)
        | (Timestamp(_), Timestamp(_))
        | (Timestamptz { .. }, Timestamptz { .. })
        | (Time, Time)
        | (Timetz { .. }, Timetz { .. })
        | (IntervalDayTime, IntervalDayTime)
        | (IntervalYearMonth, IntervalYearMonth)
        | (Date, Date) => true,
        (
            Decimal {
                precision: left_p,
                scale: left_s,
            },
            Decimal {
                precision: right_p,
                scale: right_s,
            },
        ) => {
            // Currently, only same precision is supported
            (left_p == right_p) && (left_s == right_s)
        }
        (Uuid, Uuid) | (IPv4, IPv4) | (IPv6, IPv6)
            if matches!(op, CmpOperator::Equal | CmpOperator::NotEqual) =>
        {
            true
        }

        (
            List {
                element_type: left,
                is_nullable: _,
            },
            List {
                element_type: right,
                is_nullable: _,
            },
        ) if matches!(op, CmpOperator::Equal | CmpOperator::NotEqual) => {
            can_compare(left, right, op)
        }

        _ => false,
    }
}
