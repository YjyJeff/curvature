//! Interval of DayTime

use std::fmt::Display;

/// DayTime of the interval
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DayTime {
    pub(crate) day: i32,
    pub(crate) mills: i32,
}

impl Display for DayTime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}
