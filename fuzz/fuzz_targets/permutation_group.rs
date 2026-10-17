//! Fuzzing target for [`PermutationGroup`].
//!
//! Decodes a degree, generators and sets from arbitrary bytes and checks the
//! order, membership, orbits and smallest images against brute-force closure.

#![no_main]

use arbitrary::Unstructured;
use geometric_traits::test_utils::check_permutation_group_invariants;
use libfuzzer_sys::fuzz_target;

type FuzzInput = (u8, Vec<Vec<u8>>, Vec<Vec<u8>>);

/// Keeps the brute-force closure at most `7!` elements.
const MAX_DEGREE: usize = 7;

/// Decodes a permutation from a Lehmer code, or for one byte in eight takes
/// the raw bytes so malformed generators reach the validation path.
fn decode_generator(degree: usize, bytes: &[u8]) -> Vec<usize> {
    let Some((&mode, code)) = bytes.split_first() else {
        return (0..degree).collect();
    };
    if mode % 8 == 0 {
        return code.iter().map(|&byte| usize::from(byte) % (degree + 1)).collect();
    }
    let mut remaining: Vec<usize> = (0..degree).collect();
    let mut generator = Vec::with_capacity(degree);
    for index in 0..degree {
        let choice = code.get(index).map_or(0, |&byte| usize::from(byte));
        generator.push(remaining.remove(choice % remaining.len()));
    }
    generator
}

// `arbitrary`, not `arbitrary_take_rest`, so crash files replay in tests
fuzz_target!(|bytes: &[u8]| {
    let Ok((degree_byte, raw_generators, raw_sets)) =
        Unstructured::new(bytes).arbitrary::<FuzzInput>()
    else {
        return;
    };
    let degree = usize::from(degree_byte) % (MAX_DEGREE + 1);
    let generators: Vec<Vec<usize>> =
        raw_generators.iter().take(4).map(|raw| decode_generator(degree, raw)).collect();
    let sets: Vec<Vec<usize>> = raw_sets
        .iter()
        .take(4)
        .map(|raw| raw.iter().map(|&byte| usize::from(byte) % (degree + 1)).collect())
        .collect();
    check_permutation_group_invariants(degree, &generators, &sets);
});
