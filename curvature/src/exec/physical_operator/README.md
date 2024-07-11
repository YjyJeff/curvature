# Physical Operators

## TableScan
- `EmptyTableScan`: A table scan operator that do not produce any data
- `MemoryTableScan`: Produce the DataBlock registered in memory
- `Numbers`: A table with single field `number`, it will generate the sequence `start..end`
- [ ] `ParquetTableScan`: Table scan that query parquet files

## Regular 
- `Projection`: Select the expressions from input
- `Filter`: Filter out rows that do not satisfy the condition
- `Union`: Union two inputs that have same schema. It is a fake physical operator and never appears in the pipeline execution
- `StreamingLimit`: Limit

## Sink
- Aggregation
    - `SimpleAggregate`: Aggregation without group by
    - `HashAggregate`: Aggregation with group by using hash table
    - [ ] Group by i8/u8, using `[u8; 256]` as hash table
    - [ ] Group by single key, could we save the serialization time?
    - [ ] [String hash table](https://www.mdpi.com/2076-3417/10/6/1915)
- Sort
    - [ ] TopK: Returns the top k elements. It should support offset as well
    - [ ] Sort: Sort all of the rows and returns it in order

## Reference
### Aggregation
- [Hash tables in ClickHouse](https://clickhouse.com/blog/hash-tables-in-clickhouse-and-zero-cost-abstractions)
- [Parallel Grouped Aggregation in DuckDB](https://duckdb.org/2022/03/07/aggregate-hashtable.html)
