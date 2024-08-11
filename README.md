# Curvature

`Curvature` is a high performance query engine that integrates lots of advanced techniques, like [Vectorization](https://www.cidrdb.org/cidr2005/papers/P19.pdf), [Morsel-Driven Parallelism](https://15721.courses.cs.cmu.edu/spring2016/papers/p743-leis.pdf), etc. `Curvature`'s execution model are adapted from [Duckdb](https://duckdb.org/) and its `Array`/`Vector` model are adapted from [Arrow](https://arrow.apache.org/) and [Velox](https://vldb.org/pvldb/vol15/p3372-pedreira.pdf), which is implemented with the style described in [type-exercise-in-urst](https://github.com/skyzh/type-exercise-in-rust).

I have been using [datafusion](https://github.com/apache/arrow-datafusion) for a period of time. It is a high performance query engine based on [Arrow](https://arrow.apache.org/). It is great, but I am not satisfied with it. Because:
- [Tokio](https://tokio.rs/). Tokio is an awesome runtime! But, in my view, it is not suited for OLAP. Firstly, as we all know, linux does not support async io for filesystem(You may say `AIO`, Linux Torvalds said it is a [horrible ad-hoc design](https://lwn.net/Articles/671657/)). Read the file system asynchronously is achieved by put it in a thread pool, it is pretty inefficient. [`Io-Uring`](https://kernel.dk/io_uring.pdf) can solve the problem perfectly. We will use the [`monoio`](https://github.com/bytedance/monoio) to implement `Curvature` and schedule the `Task` manually.

- Pull based model: Datafusion use the pull based model to execute the query. According to the [issue](https://github.com/ClickHouse/ClickHouse/issues/34045), push based model is more flexible. 

- Datafusion will create a lot of tasks and schedule them on the Tokio, these tasks have dependencies  but tokio can not be aware of it. Therefore, the tokio will pull them independently and the task, whose children is not ready, will be pending. Other threads whose task queue is empty can steal these tasks. Which means that the execution is not NUMA aware, which may cause lots of cache miss. Using the thread-per-core model and assign the tasks manually could solve this problem. It is the key point of the [Morsel-Driven Parallelism](https://15721.courses.cs.cmu.edu/spring2016/papers/p743-leis.pdf)

- [Arrow](data-block/README.md)

# Concepts
- `PhysicalPlan`: A tree, each node is a `PhysicalOperator`
- `PhysicalOpertor`: Operator like `Filter`, `Projection`, etc. Note that it is **NOT EXECUTABLE**. To execute `PhysicalOperator`, `PhysicalOperatorState` is needed. It contains the states that each operator should be aware of. For example, the global state for `TableScan`, such that each parallelism unit(`Task`) of `TableScan` can be synchronized.
- `Expression`: expression used in `where` condition, `select` arguments, etc
- `ExpressionExecutor`: A `Expression` is not executable, `ExpressionExecutor` is used to execute it. It will take expression and DataChunk, produce the result to output DataChunk. It consists of `ExpressionState` that used to keep track of the intermediate states of the execution process.
- `Pipeline`: A fragment of `PhysicalPlan`. The `PhysicalPlan` can be break into multiple fragments call `Pipeline` via `Pipeline Breaker`(like `HashAggregator`, `Join`, `Sort`, etc). `Pipeline Breaker` can not flow the individual DataChunk, it has to collect all of the DataChunks, then It produce DataChunks to other `PhysicalOperator`. A `Pipeline` has three roles of `PhysicalOperator`. A `Sink` in `Child Pipeline` is also the `Source` in `Parent Pipeline`, which means that a `PhysicalOperator` can belong to different `Pipelines`.
    - `Source`: First operator in the `Pipeline`, it emit data to result
    - `Normal Operator`: Internal operator in the `Pipeline`, consume DataChunk produced by other operator, and produce result. 
    - `Sink`: Last operator in the `Pipeline`, it only consumes the data, does not produce any result.
- `Event`: Split the execution into `Events`, such that Sink's finalize could be only called once. Like `union`, the sink is shared by different pipeline. If we execute the pipeline with finalize method, the finalize will be called multiple times.

# Minimum Support Rust Version(MSRV)
This crate is guaranteed to compile on the latest stable Rust

# Contributing
- Code should follow the [style.md](./docs/style.md)
- Zero `cargo check` warning
- Zero `cargo clippy` warning
- Zero `FAILED` in `cargo test`
- Zero `FAILED` in `cargo +nightly miri test` especially when you have unsafe functions!

# Hardware requirement
`Curvature` requires the x86-64 CPUs must new than [`x86-64-v2`](https://en.wikipedia.org/wiki/X86-64#Microarchitecture_levels) microarchitecture.
Because we will use `sse4.2` by default. And x86_64 CPU should support [`AES`](https://en.wikipedia.org/wiki/AES_instruction_set), we
will also use it by default