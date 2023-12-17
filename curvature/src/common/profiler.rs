//! CPU profiler

use std::time::Duration;

use quanta::Clock;

/// Efficient profiler for profiling
#[derive(Debug)]
pub struct Profiler {
    clock: Clock,
    start: u64,
    current: u64,
    /// Number of tuples that has been processed
    tuples_count: u64,
}

impl Profiler {
    /// Create a new profiler
    #[inline]
    pub fn new() -> Self {
        let clock = Clock::new();
        let start = clock.raw();
        Self {
            clock,
            start,
            current: start,
            tuples_count: 0,
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
        self.clock.delta(self.start, self.current)
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
        self.profiler.current = self.profiler.clock.raw();
    }
}
