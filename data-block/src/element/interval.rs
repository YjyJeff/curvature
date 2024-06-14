//! Interval adapted from [duckdb](https://duckdb.org/docs/sql/data_types/interval)

use std::fmt::Display;
use std::hash::Hash;
use std::ops::{Add, Sub};

/// Intervals represent a period of time. This period can be measured in a specific
/// unit or combination of units, for example years, days, or seconds. Intervals are
/// generally used to modify timestamps or dates by either adding or subtracting them.
///
/// # Notes
///
/// See [Notes on Comparison and Ordering](https://docs.rs/arrow/latest/arrow/datatypes/struct.IntervalMonthDayNanoType.html#note-on-comparing-and-ordering-for-calendar-types)
/// for details.
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub struct Interval {
    /// Months
    months: i32,
    /// Days
    days: i32,
    /// Micros
    micros: i64,
}

impl Interval {
    /// Create a new [`Interval`]
    ///
    /// `days` and `micros` breaks the invariance are also supported
    pub fn new(months: i32, days: i32, micros: i64) -> Self {
        Self {
            months,
            days,
            micros,
        }
    }

    /// Checked add
    #[inline]
    pub fn checked_add(self, other: Self) -> Option<Self> {
        todo!()
    }
}

impl Display for Interval {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.months != 0 {
            let years = self.months / 12;
            if years != 0 {
                write!(f, "{} years ", years)?;
            }
            let months = self.months - years * 12;
            if months != 0 {
                write!(f, " {} months ", months)?;
            }
        }
        todo!()
    }
}

impl Add<Interval> for Interval {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Interval) -> Self::Output {
        Interval {
            months: self.months + rhs.months,
            days: self.days + rhs.days,
            micros: self.micros + rhs.micros,
        }
    }
}

impl Sub<Interval> for Interval {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Interval) -> Self::Output {
        Interval {
            months: self.months - rhs.months,
            days: self.days - rhs.days,
            micros: self.micros - rhs.micros,
        }
    }
}
