# Benchmark
Note that Curvature only has the execution layer, for the following queries, we directly construct the physical plan and execute it. Therefore, it lacks the parse, plan and optimization phase. It may sounds unfair. However, time spent in these phase is negligible

| SQL      | env      | Curvature | ClickHouse |
|----------|----------|-----------|-------------|
| SELECT count(number) FROM numbers_mt(100000000000) where number % 4 != 0 |  dev cloud 16-threads | 8.7s  | 10.7s |
| SELECT count(*) FROM numbers_mt(100000000000) where number % 4 != 0 |  dev cloud 16-threads | 8.7s  | 4.1s |
| SELECT count(number),max(number),min(number) FROM numbers_mt(100000000000) where number % 4 != 0 |  dev cloud 16-threads | 9.9s  | 15.4s |
| SELECT avg(number), max(number), sum(number) FROM numbers_mt(100000000000) |  dev cloud 16-threads | 2.4s | 6.8s |
| SELECT max(number), sum(number), avg(number) FROM numbers_mt(10000000000) GROUP BY number % 3, number % 4, number % 5|  dev cloud  16-threads |  6.3s | 6.3s |