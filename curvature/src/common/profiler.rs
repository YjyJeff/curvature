//! CPU profiler

use std::time::Duration;

#[cfg(miri)]
pub use std::time::Instant;

#[cfg(not(miri))]
pub use quanta::Instant;

/// Guard for profiling
#[derive(Debug)]
pub struct ScopedTimerGuard<'a> {
    now: Instant,
    accumulation: &'a mut Duration,
}

impl<'a> ScopedTimerGuard<'a> {
    /// Create a new guard
    #[inline]
    pub fn new(accumulation: &'a mut Duration) -> Self {
        Self {
            now: Instant::now(),
            accumulation,
        }
    }
}

impl<'a> Drop for ScopedTimerGuard<'a> {
    #[inline]
    fn drop(&mut self) {
        *self.accumulation += self.now.elapsed();
    }
}
