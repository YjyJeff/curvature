# Benchmark
Note that Curvature only has the execution layer, for the following queries, we directly construct the physical plan and execute it. Therefore, it lacks the parse, plan and optimization phase. It may sounds unfair. However, time spent in these phase is negligible

| SQL      | env      | Curvature | ClickHouse 24.12.1.1487 |
|----------|----------|-----------|-------------|
| SELECT count(number) FROM numbers_mt(100000000000) where number % 4 != 0 |  dev cloud 32-threads | 7.4s  | 8.0s |
| SELECT count(*) FROM numbers_mt(100000000000) where number % 4 != 0 |  dev cloud 32-threads | 7.4s  | 4.0s |
| SELECT count(number),max(number),min(number) FROM numbers_mt(100000000000) where number % 4 != 0 |  dev cloud 32-threads | 8.5s  | 8.9s |
| SELECT avg(number), max(number), sum(number) FROM numbers_mt(100000000000) |  dev cloud 32-threads | 2.5s | 3.9s |
| SELECT max(number), sum(number), avg(number) FROM numbers_mt(10000000000) GROUP BY number % 3, number % 4, number % 5|  dev cloud  32-threads |  5.27s | 5.19s |
| SELECT count(number) FROM numbers_mt(100000000000) where number % 4 != 0 | mac m1  10-threads | 11.2s  | 20.1s |
| SELECT count(*) FROM numbers_mt(100000000000) where number % 4 != 0 | mac m1  10-threads | 11.2s  | 10.3s |
| SELECT count(number),max(number),min(number) FROM numbers_mt(100000000000) where number % 4 != 0 |  mac m1 10-threads | 13.8s  | 22.4s |
| SELECT avg(number), max(number), sum(number) FROM numbers_mt(100000000000) | mac m1  10-threads | 5.7s | 9.1s |
| SELECT max(number), sum(number), avg(number) FROM numbers_mt(10000000000) GROUP BY number % 3, number % 4, number % 5|  mac m1 10-threads |  7.8s | 7.8s |