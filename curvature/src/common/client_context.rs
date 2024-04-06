//! [`ClientContext`] holds the information relevant to the current client session during
//! the query

use super::types::ParallelismDegree;
use super::uuid::QueryId;
use std::sync::atomic::{AtomicBool, Ordering};

/// Holds the information relevant to the current client session
#[derive(Debug)]
pub struct ClientContext {
    /// Query id
    pub query_id: QueryId,
    /// Whether or not this query is cancelled
    is_cancelled: AtomicBool,
    /// Execution args
    pub exec_args: ExecArgs,
}

impl ClientContext {
    /// Create a new [`ClientContext`]
    #[inline]
    pub fn new(query_id: QueryId, exec_args: ExecArgs) -> Self {
        Self {
            query_id,
            is_cancelled: AtomicBool::new(false),
            exec_args,
        }
    }

    /// Whether or not this query is cancelled
    #[inline]
    pub fn is_cancelled(&self) -> bool {
        self.is_cancelled.load(Ordering::Relaxed)
    }

    /// Cancel the query
    #[inline]
    pub fn cancel(&self) {
        self.is_cancelled.store(true, Ordering::Relaxed);
    }
}

/// Arguments for execution
#[derive(Debug)]
pub struct ExecArgs {
    /// Execution parallelism
    pub parallelism: ParallelismDegree,
}

#[cfg(test)]
pub mod tests {
    use super::*;

    pub(crate) fn mock_client_context() -> ClientContext {
        ClientContext {
            query_id: QueryId::from_u128(44),
            is_cancelled: AtomicBool::new(false),
            exec_args: ExecArgs {
                parallelism: ParallelismDegree::new(2).unwrap(),
            },
        }
    }
}
