//! Serialize the `GroupByKeys` into memory comparable types

use std::convert::identity;
use std::fmt::Debug;
use std::hash::Hash;
use std::marker::PhantomData;
use std::mem::size_of;

use crate::common::utils::hash::BuildHasherDefault;

use super::hash_table::SerdeKeyAndHash;
use data_block::array::ArrayImpl;
use data_block::bitmap::BitStore;
use data_block::element::interval::DayTime;
use data_block::types::{Array, Element};

/// Trait for the memory equality comparable serde struct that stores the `GroupByKeys`
pub trait SerdeKey: Eq + Hash + Default + Debug + Send + Sync + 'static {}

/// Trait for serialize the `GroupByKeys` into `SerdeKey` and compute hash
pub trait Serde: Debug + 'static {
    /// The serialized `GroupByKeys`
    type SerdeKey: SerdeKey;

    /// Serialize the `GroupByKeys` into the prepared keys and compute the hash value
    /// of the serde key
    ///
    /// # Safety
    ///
    /// - Arrays should fit into Self::SerdeKey, otherwise, panic in debug mode and
    /// undefined behavior happens in release mode
    ///
    /// - keys should have same length with the array in the arrays
    unsafe fn serialize(
        arrays: &[&ArrayImpl],
        keys: &mut [SerdeKeyAndHash<Self::SerdeKey>],
        build_hasher: &BuildHasherDefault,
    );
}

/// Extension for float
trait FloatExt: num_traits::Float {
    // Normalize the float, make `NaN`/`-Nan` and `-0.0`/`0.0` consistent
    #[inline]
    fn normalize(self) -> Self {
        if self.is_nan() {
            Self::nan()
        } else if self.is_zero() {
            Self::zero()
        } else {
            self
        }
    }
}

impl FloatExt for f32 {}
impl FloatExt for f64 {}

/// Serde key that contains the keys that all of them are fixed size.
///
/// # Notes
///
/// - The keys stored in the serde key is not aligned! Deserializer should take care of
/// it! We design it this way because: We hope the key is as small as possible such that
/// we can serialize more cases into the it!
///
/// - It can hold up to 32 keys, therefore, we use [`BitStore`] as its validity.
///
/// TBD: Should we separate the validity with value? We can also serialize the
/// validity into values such that compare is much faster! However, it also means
/// we need larger key to store `GroupByKeys`
#[derive(Debug, Default, PartialEq, Eq, Hash, Clone)]
pub struct FixedSizedSerdeKey<K: Eq + Hash + Default + Clone + Debug + 'static> {
    key: K,
    validity: BitStore,
}

impl SerdeKey for FixedSizedSerdeKey<u16> {}
impl SerdeKey for FixedSizedSerdeKey<u32> {}
impl SerdeKey for FixedSizedSerdeKey<u64> {}
impl SerdeKey for FixedSizedSerdeKey<u128> {}
impl SerdeKey for FixedSizedSerdeKey<[u8; 32]> {}

/// Serializer
#[derive(Debug)]
pub struct FixedSizedSerdeKeySerializer<K> {
    _phantom: PhantomData<K>,
}

macro_rules! impl_fixed_sized_serde_key_serializer {
    ($serde_key_ty:ty,
        $({$int_variant:ident, $int_primitive_ty:ty}),+,
        $([$float_variant:ident, $float_primitive_ty:ty]),*
    ) => {
        impl Serde for FixedSizedSerdeKeySerializer<$serde_key_ty> {
            type SerdeKey = FixedSizedSerdeKey<$serde_key_ty>;

            unsafe fn serialize(
                arrays: &[&ArrayImpl],
                keys: &mut [SerdeKeyAndHash<Self::SerdeKey>],
                build_hasher: &BuildHasherDefault,
            ){
                // Clear the key
                keys.iter_mut().for_each(|key| key.serde_key = Self::SerdeKey::default());

                let mut offset_in_byte = 0;
                for (index, array) in arrays.iter().enumerate() {
                    match array {
                        ArrayImpl::Boolean(array) => {
                            serialize_scalar_fixed_sized_array(array, keys, offset_in_byte, index, identity);
                            offset_in_byte += size_of::<bool>();
                        }
                        $(
                            ArrayImpl::$int_variant(array) => {
                                serialize_scalar_fixed_sized_array(array, keys, offset_in_byte, index, identity);
                                offset_in_byte += size_of::<$int_primitive_ty>();
                            }
                        )+
                        $(
                            ArrayImpl::$float_variant(array) => {
                                serialize_scalar_fixed_sized_array(array, keys, offset_in_byte, index, <$float_primitive_ty as FloatExt>::normalize);
                                offset_in_byte += size_of::<$float_primitive_ty>();
                            }
                        )*
                        _ => {
                            #[cfg(debug_assertions)]
                            {
                                unreachable!(
                                    "FixedSizedSerdeKeySerializer<{}> can not serialize {} array. Caller breaks the safety contract",
                                    stringify!($serde_key_ty),
                                    array.ident()
                                )
                            }

                            #[cfg(not(debug_assertions))]
                            {
                                std::hint::unreachable_unchecked()
                            }
                        }
                    }
                }

                // Compute the hash value of the serde key
                keys.iter_mut().for_each(|key_and_hash| {
                    key_and_hash.hash_value = build_hasher.hash_one(&key_and_hash.serde_key)
                });
            }
        }
    };
}

impl_fixed_sized_serde_key_serializer!(u16,
    {Int8, i8}, {UInt8, u8},
);
impl_fixed_sized_serde_key_serializer!(u32,
    {Int8, i8}, {UInt8, u8}, {Int16, i16}, {UInt16, u16},
);

impl_fixed_sized_serde_key_serializer!(u64,
    {Int8, i8}, {UInt8, u8}, {Int16, i16}, {UInt16, u16}, {Int32, i32}, {UInt32, u32},
    [Float32, f32]
);
impl_fixed_sized_serde_key_serializer!(u128,
    {Int8, i8}, {UInt8, u8}, {Int16, i16}, {UInt16, u16}, {Int32, i32}, {UInt32, u32},
    {Int64, i64}, {UInt64, u64}, {DayTime, DayTime}, [Float32, f32], [Float64, f64]
);
impl_fixed_sized_serde_key_serializer!([u8; 32],
    {Int8, i8}, {UInt8, u8}, {Int16, i16}, {UInt16, u16}, {Int32, i32}, {UInt32, u32},
    {Int64, i64}, {UInt64, u64}, {DayTime, DayTime}, {Int128, i128},
    [Float32, f32], [Float64, f64]
);

#[inline]
unsafe fn serialize_scalar_fixed_sized_array<A, T, K, F>(
    buffer: &A,
    keys: &mut [SerdeKeyAndHash<FixedSizedSerdeKey<K>>],
    offset_in_byte: usize,
    index: usize,
    func: F,
) where
    A: Array<Element = T>,
    for<'a> T: Element<ElementRef<'a> = T>,
    K: Eq + Hash + Default + Clone + Debug + 'static,
    F: Fn(T) -> T,
{
    // Assert we have enough space
    debug_assert!(offset_in_byte + size_of::<T>() <= size_of::<K>());

    let mask = 1 << index;

    buffer
        .values_iter()
        .zip(buffer.validity().iter())
        .zip(keys)
        .for_each(|((val, is_validity), serde_key_and_hash)| {
            if is_validity {
                let val = func(val);
                let src = (&val) as *const _ as *const u8;
                let dst = (&mut serde_key_and_hash.serde_key.key) as *mut _ as *mut u8;
                // size_of::<T> is important, it is a constant, give compiler lots of
                // optimization info
                std::ptr::copy_nonoverlapping(src, dst.add(offset_in_byte), size_of::<T>());

                serde_key_and_hash.serde_key.validity |= mask;
            } else {
                // We do not need to clear the bit, caller should guarantee the bit in
                // the index is 0 by default. And we do not need to set the value, caller
                // should guarantee the value is 0
            }
        });
}

#[cfg(test)]
mod tests {
    use data_block::array::PrimitiveArray;

    use crate::common::utils::hash::fixed_build_hasher_default;

    use super::*;

    #[test]
    fn test_serialize_into_fixed_sized_key() {
        let array0 = ArrayImpl::Int8(PrimitiveArray::<i8>::from_iter([Some(-1), None, Some(8)]));
        let array1 = ArrayImpl::Int16(PrimitiveArray::<i16>::from_iter([
            None,
            Some(1024),
            Some(-1),
        ]));
        let keys = &mut vec![
            SerdeKeyAndHash {
                serde_key: FixedSizedSerdeKey::default(),
                hash_value: 0,
            };
            3
        ];

        let build_hasher = fixed_build_hasher_default();

        unsafe {
            FixedSizedSerdeKeySerializer::<u32>::serialize(&[&array0, &array1], keys, &build_hasher)
        };

        let gt_keys = &[
            SerdeKeyAndHash {
                serde_key: FixedSizedSerdeKey {
                    key: 0xff,
                    validity: 0x1,
                },
                hash_value: 151715128400565071,
            },
            SerdeKeyAndHash {
                serde_key: FixedSizedSerdeKey {
                    key: 0x040000,
                    validity: 0x2,
                },
                hash_value: 14657423813582509500,
            },
            SerdeKeyAndHash {
                serde_key: FixedSizedSerdeKey {
                    key: 0xffff08,
                    validity: 0x3,
                },
                hash_value: 11657053571334308329,
            },
        ];

        assert!(keys.iter().zip(gt_keys).all(|(key, gt)| key == gt));
    }

    #[test]
    fn test_serialize_float() {
        let array0 = ArrayImpl::Int8(PrimitiveArray::<i8>::from_iter([Some(-1), None, Some(8)]));
        let array1 = ArrayImpl::Float32(PrimitiveArray::<f32>::from_iter([
            None,
            Some(-f32::NAN),
            Some(-0.0),
        ]));
        let array2 = ArrayImpl::Float32(PrimitiveArray::<f32>::from_iter([
            None,
            Some(f32::NAN),
            Some(0.0),
        ]));

        let keys0 = &mut vec![
            SerdeKeyAndHash {
                serde_key: FixedSizedSerdeKey::default(),
                hash_value: 0,
            };
            3
        ];
        let keys1 = &mut vec![
            SerdeKeyAndHash {
                serde_key: FixedSizedSerdeKey::default(),
                hash_value: 0,
            };
            3
        ];

        let build_hasher = fixed_build_hasher_default();

        unsafe {
            FixedSizedSerdeKeySerializer::<u64>::serialize(
                &[&array0, &array1],
                keys0,
                &build_hasher,
            )
        };
        unsafe {
            FixedSizedSerdeKeySerializer::<u64>::serialize(
                &[&array0, &array2],
                keys1,
                &build_hasher,
            )
        };

        assert!(keys0.iter().zip(keys1).all(|(k0, k1)| k0 == k1));
    }
}
