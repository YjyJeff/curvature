//! Aggregation operators

use bumpalo::Bump;
/// The `Arena` used in the hash aggregation. Hash aggregation is a typic **phase-oriented**
/// allocation case. We store all of the `GroupByKeys` and the `AggregationStates` of the
/// keys in the hash table, it is insert only and never deleted individually. After
/// iterating all of the `GroupByKeys` in the table, we can delete the hash table as a
/// whole. Using arena here, we can improve the performance of allocating `GroupByKeys` and
/// `AggregationStates`
pub type Arena = Bump;

mod private {
    use super::Arena;

    /// Arena that is thread safe, the thread safety is achieved by we can not access the arena
    /// anymore! We simply keep the arena to make sure the memory allocated in the arena are
    /// valid
    #[derive(Debug)]
    pub struct ThreadSafeArena(Arena);

    impl ThreadSafeArena {
        /// Create a new thread safe arena
        pub fn new(arena: Arena) -> ThreadSafeArena {
            Self(arena)
        }
    }

    unsafe impl Sync for ThreadSafeArena {}
}

pub use self::private::ThreadSafeArena;
pub mod hash_aggregate;
