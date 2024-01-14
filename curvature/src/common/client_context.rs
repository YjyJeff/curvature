//! [`ClientContext`] holds the information relevant to the current client session during
//! the query

use super::types::ParallelismDegree;
use super::uuid::QueryId;

/// Holds the information relevant to the current client session
#[derive(Debug)]
pub struct ClientContext {
    /// Query id
    pub query_id: QueryId,
    /// Execution args
    pub exec_args: ExecArgs,
}

/// Arguments for execution
#[derive(Debug)]
pub struct ExecArgs {
    /// Execution parallelism
    pub parallelism: ParallelismDegree,
}
