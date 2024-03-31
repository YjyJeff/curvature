//! Metrics for physical operator
//!
//! Metrics in Curvature will be gathered after the pipeline executor has finished
//! execution. Which means that when the pipeline executor is executing, we can not
//! see its metric via the operator 😂. The advantage of this design is that: we
//! do not need to synchronize the metrics frequently, as we all know, synchronization
//! is expensive 🐶(cache line will [`ping-pong`] between cores caches). Caller will
//! not inspect the metrics frequently, they usually inspect it after the query is
//! finished to explore the performance issue. Therefore, this design is acceptable 😄
//!
//! [`ping-pong`]: https://assets.bitbashing.io/papers/concurrency-primer.pdf

use std::borrow::Cow;
use std::collections::HashMap;
use std::fmt::{Debug, Display};
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering::Relaxed;
use std::time::Duration;

/// Measure a potentially non contiguous duration of time, in nanoseconds
#[derive(Debug)]
#[repr(transparent)]
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

    /// Get the value
    #[inline]
    pub fn value(&self) -> u64 {
        self.nanoseconds()
    }

    /// Get duration
    #[inline]
    pub fn duration(&self) -> Duration {
        Duration::from_nanos(self.nanoseconds())
    }
}

impl Default for Time {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

/// A counter to record things such as number of input or output rows
#[derive(Debug)]
#[repr(transparent)]
pub struct Count(AtomicU64);

impl Default for Count {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl Count {
    /// create a new counter
    #[inline]
    pub fn new() -> Self {
        Self(AtomicU64::new(0))
    }

    /// Add `n` to the metric's value
    #[inline]
    pub fn add(&self, n: u64) {
        self.0.fetch_add(n, Relaxed);
    }

    /// Get the current value
    #[inline]
    pub fn value(&self) -> u64 {
        self.0.load(Relaxed)
    }
}

/// All of the supported metric values
#[derive(Debug)]
pub enum MetricValue {
    /// Time
    Time(u64),
    /// Count
    Count(u64),
}

/// As the name is expected to mostly be constant strings,
/// use a [`Cow`] to avoid copying / allocations in this common case.
pub type MetricName = Cow<'static, str>;

/// The set of metrics
#[derive(Debug)]
pub struct MetricsSet {
    /// Name of the metric set
    pub name: &'static str,
    /// Metrics
    pub metrics: HashMap<MetricName, MetricValue>,
}

impl MetricsSet {
    /// Merge other MetricsSet into self
    ///
    /// self and the other metrics should be produced by the same physical operator,
    /// otherwise the `merge` makes no sense. Which means that they should have same
    /// keys and the value type should be identical. This function will panic if the
    /// caller violate the above contract
    pub fn merge(&mut self, other: Self) {
        assert_eq!(&self.name, &other.name);

        self.metrics.iter_mut().for_each(|(key, value)| {
            let other_value = other
                .metrics
                .get(key)
                .unwrap_or_else(|| panic!("Key `{}` should exist in other", key));

            match (value, other_value) {
                (MetricValue::Count(value), MetricValue::Count(other_value))
                | (MetricValue::Time(value), MetricValue::Time(other_value)) => {
                    *value += *other_value;
                }
                (lhs, rhs) => {
                    panic!(
                        "Merge `MetricsSet` with different value type: lhs `{:?}`, rhs: `{:?}`",
                        lhs, rhs
                    )
                }
            }
        });
    }
}

impl Display for MetricsSet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} {{", self.name)?;
        self.metrics
            .iter()
            .try_for_each(|(name, value)| match value {
                MetricValue::Count(cnt) => write!(f, " {name} = {}, ", *cnt),
                MetricValue::Time(time) => write!(f, " {name} = {:?}", Duration::from_nanos(*time)),
            })?;

        write!(f, " }}")
    }
}
