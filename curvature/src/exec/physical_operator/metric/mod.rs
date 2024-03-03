//! Metrics for physical operator

use quanta::Instant;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering::Relaxed;
use std::time::Duration;

/// Measure a potentially non contiguous duration of time, in nanoseconds
#[derive(Debug)]
pub struct Time(AtomicU64);

impl Time {
    /// Create a new [`Time`]
    #[inline]
    pub fn new() -> Self {
        Self(AtomicU64::new(0))
    }

    /// Add duration to the time metric
    #[inline]
    pub fn add_duration(&self, duration: Duration) {
        self.0.fetch_add(duration.as_nanos() as _, Relaxed);
    }

    /// Get the number of nanoseconds
    #[inline]
    pub fn nanoseconds(&self) -> u64 {
        self.0.load(Relaxed)
    }
}

impl Default for Time {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

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
