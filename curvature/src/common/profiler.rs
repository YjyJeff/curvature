//! CPU profiler

use std::time::Duration;

#[cfg(miri)]
pub use std::time::Instant;

#[cfg(not(miri))]
pub use quanta::Instant;

/// Efficient profiler for profiling
#[derive(Debug)]
pub struct Profiler {
    now: Instant,
    /// Number of tuples that has been processed
    tuples_count: u64,
    duration: Duration,
}

impl Profiler {
    /// Create a new profiler
    #[inline]
    pub fn new() -> Self {
        let now = Instant::now();
        Self {
            now,
            tuples_count: 0,
            duration: Duration::default(),
        }
    }

    /// Start profile
    #[inline]
    pub fn start_profile(&mut self, process_tuple_count: u64) -> ProfilerGuard<'_> {
        ProfilerGuard {
            profiler: self,
            process_tuple_count,
        }
    }

    /// Get the elapsed duration
    #[inline]
    pub fn elapsed(&self) -> Duration {
        self.duration
    }
}

impl Default for Profiler {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

/// Guard of the profiler, when it is dropped, it will record the
/// profiling result
#[derive(Debug)]
pub struct ProfilerGuard<'a> {
    profiler: &'a mut Profiler,
    process_tuple_count: u64,
}

impl<'a> Drop for ProfilerGuard<'a> {
    #[inline]
    fn drop(&mut self) {
        self.profiler.tuples_count += self.process_tuple_count;
        let now = Instant::now();
        self.profiler.duration += now.duration_since(self.profiler.now);
        self.profiler.now = now;
    }
}
