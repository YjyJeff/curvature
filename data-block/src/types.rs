//! Re-export types and traits
//!
//! Relationships:
//!
//! - [`AllocType`] is the superset of the [`PrimitiveType`], actually they are the same.
//! [`IntrinsicType`] is the subset of the [`PrimitiveType`], types implement the
//! [`IntrinsicType`] can use SIDM to perform acceleration

use std::fmt::Display;
use std::num::NonZeroU8;
use std::sync::Arc;

pub use crate::aligned_vec::AllocType;
pub use crate::array::primitive::PrimitiveType;
pub use crate::compute::IntrinsicType;

#[cfg(feature = "portable_simd")]
pub use crate::compute::IntrinsicSimdType;

pub use crate::array::Array;
pub use crate::scalar::{Scalar, ScalarRef};

/// Physical type has a one-to-one mapping to each struct that implements [`Scalar`]
///
/// [`PhysicalType`] and `encoding` could determine the memory representation of the
/// [`Array`].
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PhysicalType {
    /// Boolean
    Boolean,
    /// Signed 8-bit integer, also known as `TINYINT`
    Int8,
    /// Unsigned 8-bit integer
    UInt8,
    /// Signed 16-bit integer, also known as `SMALLINT`
    Int16,
    /// Unsigned 16-bit integer
    UInt16,
    /// Signed 32-bit integer, also known as `INTEGER`
    Int32,
    /// Unsigned 32-bit integer
    UInt32,
    /// Signed 64-bit integer, also known as `BIGINT`
    Int64,
    /// Unsigned 64-bit integer
    UInt64,
    /// Signed 128-bit integer, also known as `HUGEINT`
    Int128,
    /// 32-bit float number, also known as `FLOAT/REAL`
    Float32,
    /// 64-bit float number, also known as `DOUBLE`
    Float64,
    /// Variable length Utf-8 String, also known as `VARCHAR`
    String,
    /// Variable length binary array, also known as `VARBINARY`
    Binary,
    /// DayTime in interval. 32 bit days and 32 bit milliseconds
    DayTime,
    // Complex types
    /// List of a physical type
    List = 64,
}

impl Display for PhysicalType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "PhysicalType::{:?}", self)
    }
}

/// All of the supported logical types. Different logical types may have same [`PhysicalType`].
///
/// It add some semantic above the physical type. Operations between [`Array`]s
/// should have different behavior based on the associated [`LogicalType`]s
#[derive(Debug, Clone)]
pub enum LogicalType {
    /// Boolean value represent `true` or `false`
    Boolean,
    /// Signed 8-bit integer
    TinyInt,
    /// Signed 16-bit integer
    SmallInt,
    /// Signed 32-bit integer
    Integer,
    /// Signed 64-bit integer
    BigInt,
    /// Signed 128-bit integer
    HugeInt,
    /// Unsigned 8-bit integer
    UnsignedTinyInt,
    /// Unsigned 16-bit integer
    UnsignedSmallInt,
    /// Unsigned 32-bit integer
    UnsignedInteger,
    /// Unsigned 64-bit integer
    UnsignedBigInt,
    /// 32-bit float number
    Float,
    /// 64-bit float number
    Double,
    /// Fixed-point numbers with precision and scale
    ///
    /// For example number `3.14` has a precision of `3` and a scale of `2`.
    /// DECIMAL types are backed by `BIGINT` and `HUGEINT` physical types, which store
    /// the unscaled value. For example, the unscaled value of decimal 123.45 is 12345.
    /// `BIGINT` is used up to 18 precision, the range is
    /// [-10<sup>18</sup> + 1, 10<sup>18</sup> -1]. `HUGEINT` is used up to 38 precision,
    /// the range is [-10<sup>38</sup> + 1, 10<sup>38</sup> -1].
    ///
    /// Performance is slow if the width is greater than 18. Most of the CPUs do not
    /// support i128. It is recommended to use precision less than or equal to 18
    Decimal {
        /// 1 <= precision <= 38
        precision: NonZeroU8,
        /// 0 <= scale <= 38
        scale: u8,
    },
    /// Variable length Utf-8 String
    VarChar,
    /// Variable length binary array
    VarBinary,
    /// Timestamp with given unit. No matter what [`TimeUnit`] is used here, the
    /// Timestamp is stored with i64. int64 can store around 292 years
    /// in nanos ~ till 2262-04-12.
    ///
    /// Timestamp does not contain the timezone info. Display the Timestamp could
    /// based on the timezone of the session/database
    ///
    /// SQL type name:
    /// - TIMESTAMP_S:  Seconds
    /// - TIMESTAMP_MS: Milliseconds
    /// - TIMESTAMP:    Microseconds
    /// - TIMESTAMP_NS: Nanoseconds
    Timestamp(TimeUnit),
    /// Timestamp with time zone information, the TimeUnit is micros. Also known
    /// as `TIMESTAMP WITH TIME ZONE`
    ///
    /// Note that we do not store the time zone internally, it has the same physical
    /// type with `Timestamp`. The time zone is only ued to adjust the time parsed
    /// from the string. Display the timestamptz has the same behavior with timestamp:
    /// the timezone is ignored and based on the timezone of the session/database
    /// This implementation has the same behavior with the [`postgres`]
    ///
    /// [`postgres`]: https://www.postgresql.org/docs/current/datatype-datetime.html#DATATYPE-DATETIME-INPUT-TIME-STAMPS
    Timestamptz {
        /// offset of the timezone in seconds in range (-86400, 86400)
        tz_offset: i32,
    },
    /// A 64-bit time representing the elapsed time since midnight in the unit of microsecond
    ///
    /// Instance can be created with the data that satisfy the ISO 8601
    /// format(`hh:mm:ss[.zzzzzz][+-TT[:tt]]`). If the timezone is specified, we will
    /// ignore it
    ///
    /// Note that it is used in rare cases, user should prefer Timestamp
    Time,
    /// A 64-bit time representing the elapsed time since midnight in the unit of microsecond
    ///
    /// Instance can be created with the data that satisfy the ISO 8601
    /// format(`hh:mm:ss[.zzzzzz][+-TT[:tt]]`). If the timezone is not specified, we
    /// will use the timezone of the session/database
    ///
    /// Note that it is used in rare cases, user should prefer Timestamptz
    Timetz {
        /// offset of the timezone in seconds in range (-86400, 86400)
        tz_offset: i32,
    },
    /// DAY_TIME interval in SQL style
    ///
    /// Indicates the number of elapsed days and milliseconds, stored as 2 contiguous 32-bit integers (days, milliseconds) (8-bytes in total).
    IntervalDayTime,
    /// YEAR_MONTH interval in SQL style
    ///
    /// Indicates the number of elapsed whole months, stored as 4-byte integers.
    IntervalYearMonth,
    /// Date is represented as the number of days since epoch start using i32.
    ///
    /// Date can be created with the data that satisfy the ISO 8601
    /// format(`YYYY-MM-DD`)
    Date,
    /// Universally unique identifier
    Uuid,

    // Complex types
    /// List of LogicalType. The child type can be scalar type or complex type
    List {
        /// Type of the element in list
        ///
        /// We use Arc here to avoid repeated memory allocation. During the plan
        /// phase, Logical type is cloned frequently.
        element_type: Arc<LogicalType>,
        /// Is the element in list nullable?
        is_nullable: bool,
    },
}

/// TimeUnit
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TimeUnit {
    /// Time in seconds
    Seconds,
    /// Time in milliseconds(ms)
    Millisecond,
    /// Time in microseconds(us)
    Microsecond,
    /// Time in nanoseconds(ns)
    Nanosecond,
}

impl LogicalType {
    /// Get the underling physical type of the logical type
    pub fn physical_type(&self) -> PhysicalType {
        match self {
            Self::Boolean => PhysicalType::Boolean,
            Self::TinyInt => PhysicalType::Int8,
            Self::SmallInt => PhysicalType::Int16,
            Self::Integer => PhysicalType::Int32,
            Self::BigInt => PhysicalType::Int64,
            Self::HugeInt => PhysicalType::Int128,
            Self::UnsignedTinyInt => PhysicalType::UInt8,
            Self::UnsignedSmallInt => PhysicalType::UInt16,
            Self::UnsignedInteger => PhysicalType::UInt32,
            Self::UnsignedBigInt => PhysicalType::UInt64,
            Self::Float => PhysicalType::Float32,
            Self::Double => PhysicalType::Float64,
            Self::Decimal { precision, .. } => {
                if *precision <= unsafe { NonZeroU8::new_unchecked(18) } {
                    PhysicalType::Int64
                } else {
                    PhysicalType::Int128
                }
            }
            Self::VarChar => PhysicalType::String,
            Self::VarBinary => PhysicalType::Binary,
            Self::Timestamp(_) | Self::Timestamptz { .. } => PhysicalType::Int64,
            Self::Time | Self::Timetz { .. } => PhysicalType::Int64,
            Self::IntervalDayTime => PhysicalType::DayTime,
            Self::IntervalYearMonth => PhysicalType::Int32,
            Self::Date => PhysicalType::Int32,
            Self::Uuid => PhysicalType::Int128,

            // Complex types
            Self::List { .. } => PhysicalType::List,
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_logical_type_size() {
        println!("{}", std::mem::size_of::<LogicalType>());
        println!("{:?}", PhysicalType::Int128);
    }
}
