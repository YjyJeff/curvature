#![cfg_attr(feature = "portable_simd", feature(portable_simd))]
#![cfg_attr(feature = "avx512", feature(avx512_target_feature))]

pub mod bitmap;
pub mod comparison;
mod macros;

use rand::distributions::{Alphanumeric, DistString, Distribution, Standard};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use data_block::array::{BooleanArray, PrimitiveArray, PrimitiveType, StringArray};

/// Creates a new [`PrimitiveArray`] from random values with a pre-set seed.
pub fn create_primitive_array_with_seed<T>(
    size: usize,
    null_density: f32,
    seed: u64,
) -> PrimitiveArray<T>
where
    T: PrimitiveType,
    Standard: Distribution<T>,
{
    let mut rng = StdRng::seed_from_u64(seed);

    (0..size)
        .map(|_| {
            if rng.gen::<f32>() < null_density {
                None
            } else {
                Some(rng.gen())
            }
        })
        .collect::<PrimitiveArray<T>>()
}

pub fn create_boolean_array_iter(
    size: usize,
    null_density: f32,
    true_density: f32,
    seed: u64,
) -> impl Iterator<Item = Option<bool>>
where
    Standard: Distribution<bool>,
{
    let mut rng = StdRng::seed_from_u64(seed);
    (0..size).map(move |_| {
        if rng.gen::<f32>() < null_density {
            None
        } else {
            let value = rng.gen::<f32>() < true_density;
            Some(value)
        }
    })
}

/// Creates an random (but fixed-seeded) array of a given size and null density
pub fn create_boolean_array(
    size: usize,
    null_density: f32,
    true_density: f32,
    seed: u64,
) -> BooleanArray
where
    Standard: Distribution<bool>,
{
    create_boolean_array_iter(size, null_density, true_density, seed).collect()
}

/// Creates an random (but fixed-seeded) [`StringArray`] of a given length, number of characters and null density.
pub fn create_string_array(
    length: usize,
    size: usize,
    null_density: f32,
    seed: u64,
) -> StringArray {
    let mut rng = StdRng::seed_from_u64(seed);

    (0..length)
        .map(|_| {
            if rng.gen::<f32>() < null_density {
                None
            } else {
                Some(Alphanumeric.sample_string(&mut rng, size))
            }
        })
        .collect()
}

pub fn create_var_string_array_iter(
    length: usize,
    null_density: f32,
    seed: u64,
) -> impl Iterator<Item = Option<String>> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..length).map(move |_| {
        if rng.gen::<f32>() < null_density {
            None
        } else {
            let size = rng.gen_range(0..=128);
            Some(Alphanumeric.sample_string(&mut rng, size))
        }
    })
}

/// Creates an random (but fixed-seeded) [`StringArray`] of a given length, number of characters and null density.
pub fn create_var_string_array(length: usize, null_density: f32, seed: u64) -> StringArray {
    create_var_string_array_iter(length, null_density, seed).collect()
}
