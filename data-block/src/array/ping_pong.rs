//! PingPongPtr designed for query execution
//!
//! You may think we can use `get_mut_unchecked` to accelerate the PingPongPtr.
//! However, compared to computation we can see that the `get_mut` is cold.
//!
//! TODO:
//!
//! - Can PingPongPtr reduces the number of times inner data is cloned? We need
//!   at least one clone to send the data between threads
//!
//! - Should we allocate the memory in memory pool, and give the burden to
//!   memory pool?
use crate::private::Sealed;
use either::Either;
use std::fmt::Debug;
use std::marker::PhantomData;
use std::ops::Deref;
use std::rc::Rc;
use std::sync::Arc;

/// PingPongPtr use ping or pong?
type IsPing = bool;

/// Trait for Reference counted pointer
pub trait RefCountPtr<T>: Deref<Target = T> + Clone + Sealed {
    /// Create a new pointer based on the inner data
    fn new(inner: T) -> Self;

    /// Returns a mutable reference into the given RefCountPtr Pointer, if there are
    /// no other pointers point to the same allocation
    fn get_mut(this: &mut Self) -> Option<&mut T>;
}

impl<T> Sealed for Rc<T> {}
impl<T> Sealed for Arc<T> {}

impl<T> RefCountPtr<T> for Rc<T> {
    #[inline]
    fn new(inner: T) -> Self {
        Rc::new(inner)
    }

    #[inline]
    fn get_mut(this: &mut Self) -> Option<&mut T> {
        Rc::get_mut(this)
    }
}

impl<T> RefCountPtr<T> for Arc<T> {
    #[inline]
    fn new(inner: T) -> Self {
        Arc::new(inner)
    }

    #[inline]
    fn get_mut(this: &mut Self) -> Option<&mut T> {
        Arc::get_mut(this)
    }
}

/// PingPong Pointer designed for query execution. It provides three methods:
/// - deref: read the data inside the pointer
/// - reference: mutate self, let self reference to other ping pong pointer
/// - exactly_once_mut: get the mutable reference to the data
///
///
/// # Why
///
/// Executing/Interpreting the expression is heavily used in the database. Each
/// expression's output should be written to the result. The result could be created
/// every time the expression is executed or a pre-allocated memory region that reused
/// across different DataBlocks.
///
/// In the first case, memory allocator has heavy pressure, lots of dynamic allocation
/// and de-allocation happens in the execution. The official implementation of the
/// `Arrow` use this method. Therefore, the second case is better. In the real word,
/// outputs may need to refer to the input, for example `ColumnExpr` read the input and
/// copy the input to the output. If we do not use `RefCountPtr`, we still need to copy lots
/// of memory between pre-allocated memory. To avoid this case, traditional
/// implementation use `Copy-On-Write(COW)` to avoid this problem. However, `COW` will
/// cause dynamic memory allocation. If we write different results, produced by
/// different DataBlock, to `COW` memory region and the region is always referred by
/// other memory region, we need to create a new memory region and copy the old memory
/// region every time. Which means that it fallbacks to the first case.
///
/// Could we use two memory region to solve the above problem? In expression execution,
/// we only want to write to a memory location and does not care about the old value
/// in the memory location. Therefore, we may call it `Create-On-Write`.
/// Instead of creating a new memory, we write the result to another memory region that
/// do not referred by others. Then the subsequent expression will release the reference
/// to the old one and refer to the new one. After the end of the expression, the old
/// one will not be referred by anyone else. Which means that, we can write the result
/// to the old one now! Let's call it [`PingPongPtr`] 😊
///
/// # Proof: Correctness of the [`PingPongPtr`] in the expression execution
///
/// Proposition: After the expression execution, one of the ping and pong is the unique
/// owner of the data
///
/// As we all know, an expression is represented as a **Tree** in the memory. Each node in
/// the tree is a concrete atomic expression, the expression can not be splitted into
/// smaller expression anymore. To execute the expression, DFS algorithm is used. Start
/// from the leaf node, each node takes inputs and produce outputs. Its outputs will be
/// the inputs of its parent. Therefore, we can view the edge in the expression tree as
/// data. Let's draw the [`line graph`] of the expression tree! We get a new tree, node
/// of the tree is the data produced by the expression and edge is the expression. Let's
/// execute this tree with DFS now. We can see that each node(data) will be written once
/// and then may read multiple times.
///
/// In the beginning state, each node(data) does not reference to other node(data). All
/// of the node(data)'s ping and pong are unique owner of the data
///
/// Assume after the k-th execution, the above proposition still holds. Let's proof after
/// the (k+1)th execution, the above proposition still holds.
///
/// For any node(`N``) in the tree, it will be write first via either `reference` or
/// `exactly_once_mut` methods.
///
/// - If the node is write via `reference`, then all of the subsequent node can not
/// create a reference to this node's ping and pong anymore. Then all of the subsequent
/// node will be visited and write once, which means that any node(`V`) that reference
/// to `N`, the reference will be dropped. Therefore, `N` is the unique
/// owner of both ping and pong
///
/// - If the node is write via `exactly_once_mut` methods, assume ping is being written
/// and can be read in the subsequent execution. It implies that pong may be referenced by
/// other nodes. Then all of the subsequent nodes will be visited and wite once,
/// which means that any node(`V`) reference to `N`, the reference will be dropped.
/// Therefore, `N` is the unique owner of the pong. Next execution can write to pong
///
/// [`line graph`]: https://en.wikipedia.org/wiki/Line_graph
pub struct PingPongPtr<T, P: RefCountPtr<T> = Rc<T>> {
    ping: P,
    pong: P,
    // Reference to other data or the data is stored in ping
    reference: Either<P, IsPing>,
    _phantom: PhantomData<T>,
}

impl<T: Debug, P: RefCountPtr<T>> Debug for PingPongPtr<T, P> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "PingPongPtr{{ ")?;
        match &self.reference {
            Either::Left(reference) => {
                write!(f, "Reference: {:?} }}", reference.deref())
            }
            Either::Right(is_ping) => {
                if *is_ping {
                    write!(f, "Ping: {:?} }}", self.ping.deref())
                } else {
                    write!(f, "Pong: {:?} }}", &self.pong.deref())
                }
            }
        }
    }
}

macro_rules! read_inner {
    ($self:ident) => {
        match &$self.reference {
            Either::Left(reference) => reference,
            Either::Right(is_ping) => {
                if *is_ping {
                    &$self.ping
                } else {
                    &$self.pong
                }
            }
        }
    };
}

impl<T, P: RefCountPtr<T>> Deref for PingPongPtr<T, P> {
    type Target = T;
    /// Get the data inside the pointer
    #[inline]
    fn deref(&self) -> &Self::Target {
        read_inner!(self)
    }
}

impl<T: Default, P: RefCountPtr<T>> Default for PingPongPtr<T, P> {
    #[inline]
    fn default() -> Self {
        Self {
            ping: P::new(T::default()),
            pong: P::new(T::default()),
            reference: Either::Right(true),
            _phantom: PhantomData,
        }
    }
}

impl<T: Default, P: RefCountPtr<T>> PingPongPtr<T, P> {
    /// Create a new PingPongPtr
    #[inline]
    pub fn new(inner: T) -> Self {
        Self {
            ping: P::new(inner),
            pong: P::new(T::default()),
            reference: Either::Right(true),
            _phantom: PhantomData,
        }
    }
}

impl<T, P: RefCountPtr<T>> PingPongPtr<T, P> {
    /// Reference self to other PingPongPtr. If self reference to other pointer,
    /// the reference will be dropped such that the data that referenced to will
    /// become a unique owner and can be mutated again.
    ///
    /// After this function, reads from self will read the data that self reference to.
    /// Therefore, subsequent execution can not reference to self.ping and self.pong
    #[inline]
    pub fn reference(&mut self, other: &Self) {
        let other = read_inner!(other);
        self.reference = Either::Left(P::clone(other));
    }

    /// Get the mutable reference to the underling data. The mutable reference will
    /// point to either self.ping or self.pong. If self reference to other pointer,
    /// the reference will be dropped such that the data that referenced to will
    /// become a unique owner and can be mutated again.
    ///
    /// After this function, reads from self will read the data in ping or pong.
    ///
    /// # Safety
    ///
    /// For each pointer, caller should guarantee this method should only be called
    /// exactly once, in each expression execution iteration
    #[inline]
    pub unsafe fn exactly_once_mut(&mut self) -> &mut T {
        match &mut self.reference {
            Either::Left(_) => {
                // self is reference to others now, ping and pong are both mutable now,
                // we will clear the reference and write to ping
                self.reference = Either::Right(true);
                P::get_mut(&mut self.ping)
                    .expect("Self reference to others, ping and pong must be unique")
            }
            Either::Right(is_ping) => {
                // self is the owner of the data. Now, we have two choice:
                // 1. try to write to the data that has been written in the previous
                //    iteration. If it is referenced by others, we write to another data.
                //    This can save space if no one will reference this data in the whole
                //    execution. But we need to do some extra work
                // 2. write to the data that has not written in the previous iteration.
                //    This will always use the extra space, maybe wast some space

                // We choose the second choice.
                *is_ping = !*is_ping;
                if *is_ping {
                    P::get_mut(&mut self.ping)
                        .expect("Pong is exposed in the previous step, Ping must not be referenced")
                } else {
                    P::get_mut(&mut self.pong)
                        .expect("Ping is exposed in the previous step, Pong must not be referenced")
                }
            }
        }
    }
}

///////////////////// Send array across threads ///////////////////////////////////

// impl<T: Clone + Default> PingPongPtr<T> {
//     /// Clone the inner T and return a SendablePingPongPtr that can be sent between threads
//     ///
//     /// If we are the only owner of the Rc, it can be sent between threads 😊
//     pub fn clone_send(&self) -> SendablePingPongPtr<T> {
//         let new = match &self.reference {
//             Either::Left(reference) => reference.as_ref().clone(),
//             Either::Right(is_ping) => {
//                 if *is_ping {
//                     self.ping.as_ref().clone()
//                 } else {
//                     self.pong.as_ref().clone()
//                 }
//             }
//         };

//         SendablePingPongPtr(PingPongPtr {
//             ping: Rc::new(new),
//             pong: Rc::new(T::default()),
//             reference: Either::Right(true),
//         })
//     }
// }

// /// A PingPongPtr that is sendable. This has the same memory representation with
// /// PingPongPtr, therefore, caller can use `transmute` to convert it directly to
// /// PingPongPtr
// #[repr(transparent)]
// pub struct SendablePingPongPtr<T>(PingPongPtr<T>);

// impl<T> SendablePingPongPtr<T> {
//     /// Get the inner PingPongPtr from the self
//     #[inline]
//     pub fn into_ping_pong(self) -> PingPongPtr<T> {
//         self.0
//     }
// }

// unsafe impl<T> Send for SendablePingPongPtr<T> {}

// impl<T: Debug> Debug for SendablePingPongPtr<T> {
//     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
//         write!(f, "SendablePingPongPtr{{ ")?;
//         match &self.0.reference {
//             Either::Left(reference) => {
//                 write!(f, "Reference: {:?} }}", reference)
//             }
//             Either::Right(is_ping) => {
//                 if *is_ping {
//                     write!(f, "Ping: {:?} }}", &self.0.ping)
//                 } else {
//                     write!(f, "Pong: {:?} }}", &self.0.pong)
//                 }
//             }
//         }
//     }
// }
