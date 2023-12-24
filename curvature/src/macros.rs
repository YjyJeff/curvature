//! Macros

macro_rules! mutate_data_block_safety {
    () => {
        // SAFETY: We are in the context of query execution, the pipeline executor
        // will guarantee all of the DataBlock will be visited and written once then
        // read multiple times
    };
}

pub(super) use mutate_data_block_safety;
