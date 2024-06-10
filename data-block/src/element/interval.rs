//! Interval of DayTime. Do we really need it?

use std::fmt::Display;
use std::hash::{Hash, Hasher};
use std::ops::{Add, AddAssign, Sub, SubAssign};

const MILLS_OF_DAY: i32 = 24 * 60 * 60 * 1000;

/// DayTime of the interval
#[repr(C, align(8))]
#[derive(Debug, Clone, Copy, PartialOrd, Ord)]
pub struct DayTime {
    day: i32,
    mills: i32,
}

impl PartialEq for DayTime {
    /// Accelerate equality comparison
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.as_i64() == other.as_i64()
    }
}

impl Eq for DayTime {}

impl Hash for DayTime {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.day.hash(state);
        self.mills.hash(state);
    }
}

impl DayTime {
    /// View day time as i64
    #[inline]
    pub fn as_i64(&self) -> i64 {
        unsafe { *((self as *const _) as *const i64) }
    }

    /// Checked add day time
    #[inline]
    pub fn checked_add(self, rhs: Self) -> Option<Self> {
        let mut day = self.day.checked_add(rhs.day)?;
        let mut mills = self.mills + rhs.mills;
        if mills >= MILLS_OF_DAY {
            day = day.checked_add(1)?;
            mills -= MILLS_OF_DAY;
        } else if mills <= -MILLS_OF_DAY {
            day = day.checked_sub(1)?;
            mills += MILLS_OF_DAY;
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
        } else if mills <= -MILLS_OF_DAY {
            day -= 1;
            mills += MILLS_OF_DAY;
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
        } else if self.mills <= MILLS_OF_DAY {
            self.day -= 1;
            self.mills += MILLS_OF_DAY;
        }
    }
}

impl Sub for DayTime {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        let mut day = self.day - rhs.day;
        let mut mills = self.mills - rhs.mills;
        if mills >= MILLS_OF_DAY {
            day += 1;
            mills -= MILLS_OF_DAY;
        } else if mills <= -MILLS_OF_DAY {
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
        if self.mills >= MILLS_OF_DAY {
            self.day += 1;
            self.mills -= MILLS_OF_DAY
        } else if self.mills <= MILLS_OF_DAY {
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
