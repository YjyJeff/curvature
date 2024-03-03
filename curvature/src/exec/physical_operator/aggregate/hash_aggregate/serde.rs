//! Serialize the `GroupByKeys` into memory comparable types

use std::convert::identity;
use std::fmt::Debug;
use std::hash::Hash;
use std::marker::PhantomData;
use std::mem::size_of;

use data_block::array::{Array, ArrayImpl, ScalarArray};
use data_block::bitmap::BitStore;
use data_block::element::interval::DayTime;
use data_block::types::{Element, PhysicalSize};

/// Trait for the memory equality comparable serde struct that stores the `GroupByKeys`
pub trait SerdeKey: Eq + Hash + Default + Debug + Clone + Send + Sync + 'static {
    /// Physical size of the serde key
    const PHYSICAL_SIZE: PhysicalSize;
}

/// Trait for serialize the `GroupByKeys` into `SerdeKey` and compute hash
pub trait Serde: Debug + 'static {
    /// The serialized `GroupByKeys`
    type SerdeKey: SerdeKey;

    /// Serialize the `GroupByKeys` into the prepared keys
    ///
    /// # Safety
    ///
    /// - Arrays should fit into Self::SerdeKey, otherwise, panic in debug mode and
    /// undefined behavior happens in release mode
    ///
    /// - keys should have same length with the array in the arrays
    unsafe fn serialize(arrays: &[&ArrayImpl], keys: &mut [Self::SerdeKey]);

    /// Deserialize the serde `GroupByKeys` to arrays
    ///
    /// # Safety
    ///
    /// - Arrays should fit into Self::SerdeKey, otherwise, panic in debug mode and
    /// undefined behavior happens in release mode
    ///
    /// - The output arrays should have same logical types with the serialized `GroupByKeys`
    unsafe fn deserialize(arrays: &mut [ArrayImpl], keys: &[Self::SerdeKey]);
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
/// - The keys stored in the serde key is **not aligned**! Deserializer should take care of
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

macro_rules! for_all_fixed_sized_serde_key {
    ($macro:ident) => {
        $macro! {
            <u16, {Int8, i8}, {UInt8, u8},>,
            <u32, {Int8, i8}, {UInt8, u8}, {Int16, i16}, {UInt16, u16},>,
            <u64, {Int8, i8}, {UInt8, u8}, {Int16, i16}, {UInt16, u16}, {Int32, i32}, {UInt32, u32}, [Float32, f32]>,
            <u128, {Int8, i8}, {UInt8, u8}, {Int16, i16}, {UInt16, u16}, {Int32, i32}, {UInt32, u32}, {Int64, i64}, {UInt64, u64}, {DayTime, DayTime}, [Float32, f32], [Float64, f64]>,
            <[u8; 32], {Int8, i8}, {UInt8, u8}, {Int16, i16}, {UInt16, u16}, {Int32, i32}, {UInt32, u32}, {Int64, i64}, {UInt64, u64}, {DayTime, DayTime}, {Int128, i128}, [Float32, f32], [Float64, f64]>
        }
    };
}

macro_rules! impl_fixed_sized_serde_key {
    ($(<$ty:ty, $({$_:ident, $__:ty}),+, $([$___:ident, $____:ty]),* >),+) => {
        $(
            impl SerdeKey for FixedSizedSerdeKey<$ty> {
                const PHYSICAL_SIZE: PhysicalSize = PhysicalSize::Fixed(size_of::<$ty>());
            }

            impl SerdeKey for $ty {
                const PHYSICAL_SIZE: PhysicalSize = PhysicalSize::Fixed(size_of::<$ty>());
            }
        )+
    };
}

for_all_fixed_sized_serde_key!(impl_fixed_sized_serde_key);

macro_rules! impl_fixed_sized_serde_key_serializer {
    ($(<$serde_key_ty:ty,
        $({$int_variant:ident, $int_primitive_ty:ty}),+,
        $([$float_variant:ident, $float_primitive_ty:ty]),*
    >),+) => {
        $(
            impl Serde for FixedSizedSerdeKeySerializer<$serde_key_ty> {
                type SerdeKey = FixedSizedSerdeKey<$serde_key_ty>;

                unsafe fn serialize(arrays: &[&ArrayImpl], keys: &mut [Self::SerdeKey]) {
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
                }

                unsafe fn deserialize(arrays: &mut [ArrayImpl], keys: &[Self::SerdeKey]){
                    let mut offset_in_byte = 0;
                    for (index, array) in arrays.iter_mut().enumerate() {
                        match array{
                            ArrayImpl::Boolean(array) => {
                                deserialize_scalar_fixed_sized_array(array, keys, offset_in_byte, index);
                                offset_in_byte += size_of::<bool>();
                            }
                            $(
                                ArrayImpl::$int_variant(array) => {
                                    deserialize_scalar_fixed_sized_array(array, keys, offset_in_byte, index);
                                    offset_in_byte += size_of::<$int_primitive_ty>();
                                }
                            )+
                            $(
                                ArrayImpl::$float_variant(array) => {
                                    deserialize_scalar_fixed_sized_array(array, keys, offset_in_byte, index);
                                    offset_in_byte += size_of::<$float_primitive_ty>();
                                }
                            )*
                            _ => {
                                #[cfg(debug_assertions)]
                                {
                                    unreachable!(
                                        "FixedSizedSerdeKeySerializer<{}> can not deserialize {} array. Caller breaks the safety contract",
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
                }
            }

            impl Serde for NonNullableFixedSizedSerdeKeySerializer<$serde_key_ty> {
                type SerdeKey = $serde_key_ty;

                unsafe fn serialize(arrays: &[&ArrayImpl], keys: &mut [Self::SerdeKey]){
                    let mut offset_in_byte = 0;
                    for array in arrays {
                        match array {
                            ArrayImpl::Boolean(array) => {
                                serialize_non_nullable_scalar_fixed_sized_array(array, keys, offset_in_byte, identity);
                                offset_in_byte += size_of::<bool>();
                            }
                            $(
                                ArrayImpl::$int_variant(array) => {
                                    serialize_non_nullable_scalar_fixed_sized_array(array, keys, offset_in_byte, identity);
                                    offset_in_byte += size_of::<$int_primitive_ty>();
                                }
                            )+
                            $(
                                ArrayImpl::$float_variant(array) => {
                                    serialize_non_nullable_scalar_fixed_sized_array(array, keys, offset_in_byte, <$float_primitive_ty as FloatExt>::normalize);
                                    offset_in_byte += size_of::<$float_primitive_ty>();
                                }
                            )*
                            _ => {
                                #[cfg(debug_assertions)]
                                {
                                    unreachable!(
                                        "NonNullableFixedSizedSerdeKeySerializer<{}> can not serialize {} array. Caller breaks the safety contract",
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
                }

                unsafe fn deserialize(arrays: &mut [ArrayImpl], keys: &[Self::SerdeKey]){
                    let mut offset_in_byte = 0;
                    for array in arrays.iter_mut() {
                        match array{
                            ArrayImpl::Boolean(array) => {
                                deserialize_non_nullable_scalar_fixed_sized_array(array, keys, offset_in_byte);
                                offset_in_byte += size_of::<bool>();
                            }
                            $(
                                ArrayImpl::$int_variant(array) => {
                                    deserialize_non_nullable_scalar_fixed_sized_array(array, keys, offset_in_byte);
                                    offset_in_byte += size_of::<$int_primitive_ty>();
                                }
                            )+
                            $(
                                ArrayImpl::$float_variant(array) => {
                                    deserialize_non_nullable_scalar_fixed_sized_array(array, keys, offset_in_byte);
                                    offset_in_byte += size_of::<$float_primitive_ty>();
                                }
                            )*
                            _ => {
                                #[cfg(debug_assertions)]
                                {
                                    unreachable!(
                                        "NonNullableFixedSizedSerdeKeySerializer<{}> can not deserialize {} array. Caller breaks the safety contract",
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
                }
            }
        )+
    };
}

for_all_fixed_sized_serde_key!(impl_fixed_sized_serde_key_serializer);

/// Serializer for serializing the arrays into the the key that has fixed size
#[derive(Debug)]
pub struct FixedSizedSerdeKeySerializer<K> {
    _phantom: PhantomData<K>,
}

/// Serializer for serializing the arrays into the the key that has fixed size. All of
/// the arrays should be non-nullable
#[derive(Debug)]
pub struct NonNullableFixedSizedSerdeKeySerializer<K> {
    _phantom: PhantomData<K>,
}

#[inline]
unsafe fn serialize_scalar_fixed_sized_array<A, T, K, F>(
    array: &A,
    keys: &mut [FixedSizedSerdeKey<K>],
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

    let set_mask = 1 << index;
    let clear_mask = !set_mask;

    let validity = array.validity();
    if validity.is_empty() {
        array.values_iter().zip(keys).for_each(|(val, serde_key)| {
            let val = func(val);
            let src = (&val) as *const _ as *const u8;
            let dst = (&mut serde_key.key) as *mut _ as *mut u8;
            // size_of::<T> is important, it is a constant, give compiler lots of
            // optimization info
            std::ptr::copy_nonoverlapping(src, dst.add(offset_in_byte), size_of::<T>());

            serde_key.validity |= set_mask;
        });
    } else {
        array
            .values_iter()
            .zip(array.validity().iter())
            .zip(keys)
            .for_each(|((val, is_valid), serde_key)| {
                if is_valid {
                    let val = func(val);
                    let src = (&val) as *const _ as *const u8;
                    let dst = (&mut serde_key.key) as *mut _ as *mut u8;
                    // size_of::<T> is important, it is a constant, give compiler lots of
                    // optimization info
                    std::ptr::copy_nonoverlapping(src, dst.add(offset_in_byte), size_of::<T>());

                    serde_key.validity |= set_mask;
                } else {
                    serde_key.validity &= clear_mask;
                }
            });
    }
}

#[inline]
unsafe fn deserialize_scalar_fixed_sized_array<A, T, K>(
    array: &mut A,
    keys: &[FixedSizedSerdeKey<K>],
    offset_in_byte: usize,
    index: usize,
) where
    A: ScalarArray<Element = T>,
    for<'a> T: Element<ElementRef<'a> = T>,
    K: Eq + Hash + Default + Clone + Debug + 'static,
{
    let mask = 1 << index;

    let trusted_len_iterator = keys.iter().map(|serde_key| {
        if serde_key.validity & mask != 0 {
            let ptr = (&serde_key.key) as *const _ as *const u8;
            Some(std::ptr::read_unaligned(ptr.add(offset_in_byte) as *const T))
        } else {
            None
        }
    });

    array.replace_with_trusted_len_iterator(keys.len(), trusted_len_iterator)
}

#[inline]
unsafe fn serialize_non_nullable_scalar_fixed_sized_array<A, T, K, F>(
    array: &A,
    keys: &mut [K],
    offset_in_byte: usize,
    func: F,
) where
    A: Array<Element = T>,
    for<'a> T: Element<ElementRef<'a> = T>,
    K: Eq + Hash + Default + Clone + Debug + 'static,
    F: Fn(T) -> T,
{
    // Assert we have enough space
    debug_assert!(offset_in_byte + size_of::<T>() <= size_of::<K>());

    array.values_iter().zip(keys).for_each(|(val, serde_key)| {
        let val = func(val);
        let src = (&val) as *const _ as *const u8;
        let dst = serde_key as *mut _ as *mut u8;
        // size_of::<T> is important, it is a constant, give compiler lots of
        // optimization info
        std::ptr::copy_nonoverlapping(src, dst.add(offset_in_byte), size_of::<T>());
    });
}

#[inline]
unsafe fn deserialize_non_nullable_scalar_fixed_sized_array<A, T, K>(
    array: &mut A,
    keys: &[K],
    offset_in_byte: usize,
) where
    A: ScalarArray<Element = T>,
    for<'a> T: Element<ElementRef<'a> = T>,
    K: Eq + Hash + Default + Clone + Debug + 'static,
{
    let trusted_len_iterator = keys.iter().map(|serde_key| {
        let ptr = serde_key as *const _ as *const u8;
        std::ptr::read_unaligned(ptr.add(offset_in_byte) as *const T)
    });

    array.replace_with_trusted_len_values_iterator(keys.len(), trusted_len_iterator)
}

#[cfg(test)]
mod tests {
    use data_block::{array::PrimitiveArray, block::DataBlock, types::LogicalType};

    use super::*;

    impl<K: Eq + Hash + Default + Clone + Debug + 'static> FixedSizedSerdeKey<K> {
        pub fn new(key: K, validity: BitStore) -> Self {
            Self { key, validity }
        }
    }

    #[test]
    fn test_serialize_into_fixed_sized_key() {
        let array0 = ArrayImpl::Int8(PrimitiveArray::<i8>::from_iter([Some(-1), None, Some(8)]));
        let array1 = ArrayImpl::Int16(PrimitiveArray::<i16>::from_iter([
            None,
            Some(1024),
            Some(-1),
        ]));
        let keys = &mut vec![FixedSizedSerdeKey::default(); 3];

        unsafe { FixedSizedSerdeKeySerializer::<u32>::serialize(&[&array0, &array1], keys) };

        let gt_keys = [
            FixedSizedSerdeKey {
                key: 0xff,
                validity: 0x1,
            },
            FixedSizedSerdeKey {
                key: 0x040000,
                validity: 0x2,
            },
            FixedSizedSerdeKey {
                key: 0xffff08,
                validity: 0x3,
            },
        ];

        keys.iter()
            .zip(&gt_keys)
            .for_each(|(key, gt)| assert_eq!(key, gt));
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

        let keys0 = &mut vec![FixedSizedSerdeKey::default(); 3];
        let keys1 = &mut vec![FixedSizedSerdeKey::default(); 3];

        unsafe { FixedSizedSerdeKeySerializer::<u64>::serialize(&[&array0, &array1], keys0) };
        unsafe { FixedSizedSerdeKeySerializer::<u64>::serialize(&[&array0, &array2], keys1) };

        assert!(keys0.iter().zip(keys1).all(|(k0, k1)| k0 == k1));
    }

    #[test]
    fn test_roundtrip_fixed_sized_key() {
        let array0 = ArrayImpl::Int8(PrimitiveArray::<i8>::from_iter([
            Some(-3),
            Some(7),
            None,
            Some(-100),
            Some(10),
        ]));
        let array1 = ArrayImpl::Int16(PrimitiveArray::<i16>::from_iter([
            None,
            None,
            Some(100),
            Some(-2),
            Some(-4),
        ]));

        let mut keys = vec![FixedSizedSerdeKey::default(); 5];

        unsafe { FixedSizedSerdeKeySerializer::<u32>::serialize(&[&array0, &array1], &mut keys) }

        let mut block =
            DataBlock::with_logical_types(vec![LogicalType::TinyInt, LogicalType::SmallInt]);

        let guard = block.mutate_arrays();
        unsafe {
            let mutate_func = |arrays: &mut [ArrayImpl]| {
                FixedSizedSerdeKeySerializer::<u32>::deserialize(arrays, &keys);
                Ok::<_, ()>(())
            };

            guard.mutate(mutate_func).unwrap();
        }

        match (array0, block.arrays().first().unwrap()) {
            (ArrayImpl::Int8(l), ArrayImpl::Int8(r)) => {
                assert!(l.iter().zip(r.iter()).all(|(l, r)| l == r))
            }
            _ => panic!(),
        }

        match (array1, block.arrays().last().unwrap()) {
            (ArrayImpl::Int16(l), ArrayImpl::Int16(r)) => {
                assert!(l.iter().zip(r.iter()).all(|(l, r)| l == r))
            }
            _ => panic!(),
        }
    }
}
