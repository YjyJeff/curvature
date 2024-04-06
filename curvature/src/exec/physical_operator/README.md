# Physical Operators

## TableScan
- `EmptyTableScan`: A table scan operator that do not produce any data
- `MemoryTableScan`: Produce the DataBlock registered in memory
- `Numbers`: A table with single field `number`, it will generate the sequence `start..end`

## Regular 
- `Projection`: Select the expressions from input
- Aggregation
    - `SimpleAggregate`: Aggregation without group by
    - `HashAggregate`: Aggregation with group by using hash table
- `Union`: Union two inputs that have same schema