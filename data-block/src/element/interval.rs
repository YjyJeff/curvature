//! Interval of DayTime

use std::fmt::Display;
use std::ops::{Add, AddAssign, Sub, SubAssign};

const MILLS_OF_DAY: i32 = 24 * 60 * 60 * 1000;

/// DayTime of the interval
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DayTime {
    day: i32,
    mills: i32,
}

impl DayTime {
    /// Checked add day time
    #[inline]
    pub fn checked_add(self, rhs: Self) -> Option<Self> {
        let mut day = self.day.checked_add(rhs.day)?;
        let mut mills = self.mills + rhs.mills;
        if mills >= MILLS_OF_DAY {
            day = day.checked_add(1)?;
            mills -= MILLS_OF_DAY;
        }
        Some(DayTime { day, mills })
    }
}

impl Display for DayTime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl Add for DayTime {
    type Output = DayTime;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        let mut day = self.day + rhs.day;
        let mut mills = self.mills + rhs.mills;
        if mills >= MILLS_OF_DAY {
            day += 1;
            mills -= MILLS_OF_DAY;
        }
        DayTime { day, mills }
    }
}

impl AddAssign for DayTime {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.day += rhs.day;
        self.mills += rhs.mills;
        if self.mills >= MILLS_OF_DAY {
            self.day += 1;
            self.mills -= MILLS_OF_DAY
        }
    }
}

impl Sub for DayTime {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        let mut day = self.day - rhs.day;
        let mut mills = self.mills - rhs.mills;
        if mills < -MILLS_OF_DAY {
            day -= 1;
            mills += MILLS_OF_DAY;
        }
        DayTime { day, mills }
    }
}

impl SubAssign for DayTime {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.day -= rhs.day;
        self.mills -= rhs.mills;
        if self.mills < -MILLS_OF_DAY {
            self.day -= 1;
            self.mills += MILLS_OF_DAY;
        }
    }
}

impl Default for DayTime {
    #[inline]
    fn default() -> Self {
        Self { day: 0, mills: 0 }
    }
}
