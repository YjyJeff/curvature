//! uuids in the crate.

use uuid::Uuid;
macro_rules! make_id {
    ($name:ident, $comment:expr) => {
        #[doc = $comment]
        #[repr(transparent)]
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        pub struct $name(Uuid);

        impl $name {
            /// Construct Self from u128
            #[inline]
            pub const fn from_u128(val: u128) -> Self {
                Self(Uuid::from_u128(val))
            }
        }

        impl std::fmt::Display for $name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "{:?}", self.0)
            }
        }
    };
}

make_id!(QueryId, "[`QueryId`] is the unique identifier of the query");
