# Curvature

# Tittle-tattle

## Performance
To achieve the high performance, you need to:

- **Enumerate all of the cases and overfit the query with specific code**. It is the most important hack! To achieve it, **generic and specilization is all you need 😂** 

- Reuse memory. 

## Storage
From the file system's view, we should avoid lots of small files, otherwise, we will have poor performance. However, from the query engines view, morsel based on small files benefit the load-balance and is easy to implement. We could also use the morsel based on the row-group in the PAX file. Multiple threads will open and read the same file, if the file content is not in the page cache and HDD is used, it will make the reader head constantly jump back and forth, therefore, we have very bad throughput