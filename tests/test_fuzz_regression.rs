//! Regression tests that exercise the same code paths as fuzz targets.
//!
//! These tests construct instances via the `Arbitrary` trait from fixed byte
//! patterns and from the fuzz corpus, then run the same invariant checks that
//! the corresponding honggfuzz targets use (via shared functions in
//! `test_utils`).
#![cfg(feature = "arbitrary")]

use std::path::Path;

use arbitrary::Arbitrary;
use geometric_traits::{
    impls::{CSR2D, SquareCSR2D, ValuedCSR2D},
    naive_structs::GenericGraph,
    prelude::*,
    test_utils::{
        self, check_kahn_ordering, check_lap_sparse_wrapper_invariants,
        check_lap_square_invariants, check_louvain_invariants, check_padded_diagonal_invariants,
        check_padded_matrix2d_invariants, check_sparse_matrix_invariants,
        check_valued_matrix_invariants, from_bytes, replay_dir,
    },
    traits::MonopartiteGraph,
};

// ============================================================================
// Byte patterns: varied sizes and content to exercise Arbitrary impls
// ============================================================================

/// Generate a set of byte slices that exercise different code paths in the
/// Arbitrary impls (empty, small, medium, large, edge-case values).
fn test_byte_patterns() -> Vec<Vec<u8>> {
    let mut patterns: Vec<Vec<u8>> = vec![
        // Minimal
        vec![0],
        vec![1],
        vec![255],
        // Small: enough for header but few/no edges
        vec![0, 0],
        vec![1, 1],
        vec![3, 3, 0],
        vec![2, 2, 0, 0],
        // Medium: enough for a few edges
        vec![3, 4, 2, 0, 1, 1, 2],
        vec![5, 5, 3, 0, 0, 1, 1, 2, 2],
        vec![3, 3, 4, 0, 0, 0, 1, 1, 0, 2, 1],
        // Large: many edges, potential MaxedOut paths
        vec![10, 10, 8, 0, 1, 0, 3, 1, 2, 2, 0, 3, 4, 4, 5, 5, 6, 7, 8],
        // All same value: triggers dedup heavily
        vec![5, 5, 6, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        // Descending: tests sort path
        vec![4, 4, 4, 3, 3, 2, 2, 1, 1, 0, 0],
        // Max u8 values: tests MaxedOut error paths
        vec![255, 255, 3, 254, 254, 253, 253, 252, 252],
        // Realistic graph-like pattern
        vec![6, 6, 10, 0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 4, 3, 2, 3, 5, 4, 3, 5, 4],
    ];

    // Add some longer random-ish patterns
    for seed in 0u8..20 {
        let mut pattern = Vec::with_capacity(50);
        for i in 0..50u8 {
            pattern.push(seed.wrapping_mul(7).wrapping_add(i.wrapping_mul(13)));
        }
        patterns.push(pattern);
    }

    patterns
}

// ============================================================================
// Helper: run a check on all patterns, skipping those that don't produce T
// ============================================================================

fn for_each_instance<T, F>(check: F)
where
    T: for<'a> Arbitrary<'a>,
    F: Fn(&T),
{
    let patterns = test_byte_patterns();
    let mut constructed = 0;
    for pattern in &patterns {
        if let Some(instance) = from_bytes::<T>(pattern) {
            check(&instance);
            constructed += 1;
        }
    }
    assert!(
        constructed > 0,
        "No instances of {} could be constructed from any byte pattern",
        core::any::type_name::<T>()
    );
}

// ============================================================================
// CSR2D (mirrors fuzz/fuzz_targets/csr2d.rs)
// ============================================================================

type TestCSR = CSR2D<u16, u8, u8>;

#[test]
fn test_arbitrary_csr2d_invariants() {
    for_each_instance::<TestCSR, _>(check_sparse_matrix_invariants);
}

#[test]
fn test_replay_csr2d_corpus() {
    let corpus_dir = Path::new("fuzz/hfuzz_workspace/csr2d/input");
    for instance in replay_dir::<TestCSR>(corpus_dir) {
        check_sparse_matrix_invariants(&instance);
    }
}

// ============================================================================
// ValuedCSR2D (mirrors fuzz/fuzz_targets/valued_csr2d.rs)
// ============================================================================

type TestValuedCSR = ValuedCSR2D<u16, u8, u8, f64>;

#[test]
fn test_arbitrary_valued_csr2d_invariants() {
    for_each_instance::<TestValuedCSR, _>(check_valued_matrix_invariants);
}

#[test]
fn test_replay_valued_csr2d_corpus() {
    let corpus_dir = Path::new("fuzz/hfuzz_workspace/valued_csr2d/input");
    for instance in replay_dir::<TestValuedCSR>(corpus_dir) {
        check_valued_matrix_invariants(&instance);
    }
}

// ============================================================================
// SquareCSR2D (mirrors fuzz/fuzz_targets/tarjan.rs, johnson_cycle.rs, kahn.rs)
// ============================================================================

type TestSquareCSR = SquareCSR2D<CSR2D<u16, u8, u8>>;

#[test]
fn test_arbitrary_square_csr2d_invariants() {
    for_each_instance::<TestSquareCSR, _>(|sq| {
        assert_eq!(sq.number_of_rows(), sq.number_of_columns());
        check_sparse_matrix_invariants(sq.as_ref());
    });
}

#[test]
fn test_arbitrary_square_csr2d_tarjan() {
    for_each_instance::<TestSquareCSR, _>(|sq| {
        let _sccs: Vec<_> = sq.tarjan().collect();
    });
}

#[test]
fn test_arbitrary_square_csr2d_johnson() {
    for_each_instance::<TestSquareCSR, _>(|sq| {
        let _cycles: Vec<_> = sq.johnson().collect();
    });
}

#[test]
fn test_arbitrary_square_csr2d_kahn() {
    for_each_instance::<TestSquareCSR, _>(|sq| {
        check_kahn_ordering(sq, 255);
    });
}

#[test]
fn test_replay_tarjan_corpus() {
    let corpus_dir = Path::new("fuzz/hfuzz_workspace/tarjan/input");
    for instance in replay_dir::<TestSquareCSR>(corpus_dir) {
        let _sccs: Vec<_> = instance.tarjan().collect();
    }
}

#[test]
fn test_replay_johnson_corpus() {
    let corpus_dir = Path::new("fuzz/hfuzz_workspace/johnson_cycle/input");
    for instance in replay_dir::<TestSquareCSR>(corpus_dir) {
        let _cycles: Vec<_> = instance.johnson().collect();
    }
}

#[test]
fn test_replay_kahn_corpus() {
    let corpus_dir = Path::new("fuzz/hfuzz_workspace/kahn/input");
    for instance in replay_dir::<TestSquareCSR>(corpus_dir) {
        check_kahn_ordering(&instance, 255);
    }
}

// ============================================================================
// GenericGraph (mirrors fuzz/fuzz_targets/root_nodes.rs, sink_nodes.rs, etc.)
// ============================================================================

type TestGraph = GenericGraph<u8, SquareCSR2D<CSR2D<u16, u8, u8>>>;

#[test]
fn test_arbitrary_generic_graph_construction() {
    for_each_instance::<TestGraph, _>(|_graph| {});
}

#[test]
fn test_replay_root_nodes_corpus() {
    let _: Vec<TestGraph> = replay_dir(Path::new("fuzz/hfuzz_workspace/root_nodes/input"));
}

#[test]
fn test_replay_sink_nodes_corpus() {
    let _: Vec<TestGraph> = replay_dir(Path::new("fuzz/hfuzz_workspace/sink_nodes/input"));
}

// ============================================================================
// GenericMatrix2DWithPaddedDiagonal
// (mirrors fuzz/fuzz_targets/generic_matrix2d_with_padded_diagonal.rs)
// ============================================================================

type TestPaddedDiag = test_utils::FuzzPaddedDiag;

#[test]
fn test_arbitrary_padded_diagonal_invariants() {
    for_each_instance::<TestPaddedDiag, _>(check_padded_diagonal_invariants);
}

#[test]
fn test_replay_padded_diagonal_corpus() {
    let corpus_dir = Path::new("fuzz/hfuzz_workspace/generic_matrix2d_with_padded_diagonal/input");
    for instance in replay_dir::<TestPaddedDiag>(corpus_dir) {
        check_padded_diagonal_invariants(&instance);
    }
}

// ============================================================================
// Hopcroft-Karp (mirrors fuzz/fuzz_targets/hopcroft_karp.rs)
// ============================================================================

#[test]
fn test_arbitrary_hopcroft_karp() {
    for_each_instance::<TestCSR, _>(|csr| {
        let _ = csr.hopcroft_karp();
    });
}

#[test]
fn test_replay_hopcroft_karp_corpus() {
    let corpus_dir = Path::new("fuzz/hfuzz_workspace/hopcroft_karp/input");
    for instance in replay_dir::<TestCSR>(corpus_dir) {
        let _ = instance.hopcroft_karp();
    }
}

// ============================================================================
// LAP (mirrors fuzz/fuzz_targets/lap.rs)
// ============================================================================

#[test]
fn test_arbitrary_lap_invariants() {
    for_each_instance::<TestValuedCSR, _>(|csr| {
        check_lap_sparse_wrapper_invariants(csr);
        check_lap_square_invariants(csr);
    });
}

#[test]
fn test_replay_lap_corpus() {
    let corpus_dir = Path::new("fuzz/hfuzz_workspace/lap/input");
    for instance in replay_dir::<TestValuedCSR>(corpus_dir) {
        check_lap_sparse_wrapper_invariants(&instance);
        check_lap_square_invariants(&instance);
    }
}

// ============================================================================
// PaddedMatrix2D (mirrors fuzz/fuzz_targets/padded_matrix2d.rs)
// ============================================================================

type TestPaddedCSR = ValuedCSR2D<u16, u8, u8, u8>;

#[test]
fn test_arbitrary_padded_matrix2d() {
    for_each_instance::<TestPaddedCSR, _>(|csr| {
        check_padded_matrix2d_invariants(csr);
    });
}

// ============================================================================
// Louvain (mirrors fuzz/fuzz_targets/louvain.rs)
// ============================================================================

#[test]
fn test_arbitrary_louvain() {
    for_each_instance::<TestValuedCSR, _>(check_louvain_invariants);
}

#[test]
fn test_replay_louvain_corpus() {
    let corpus_dir = Path::new("fuzz/hfuzz_workspace/louvain/input");
    for instance in replay_dir::<TestValuedCSR>(corpus_dir) {
        check_louvain_invariants(&instance);
    }
}

// ============================================================================
// Wu-Palmer (mirrors fuzz/fuzz_targets/wu_palmer.rs)
// ============================================================================

#[test]
fn test_arbitrary_wu_palmer() {
    for_each_instance::<TestGraph, _>(|csr| {
        if let Ok(wu_palmer) = csr.wu_palmer() {
            let node_ids: Vec<u8> = csr.node_ids().collect();
            test_utils::check_similarity_invariants(&wu_palmer, &node_ids, 10);
        }
    });
}

#[test]
fn test_arbitrary_wu_palmer_targeted_dense_patterns() {
    let dense_like_seeds: [u8; 6] = [11, 37, 73, 109, 149, 193];
    let mut constructed = 0usize;

    for &seed in &dense_like_seeds {
        let mut pattern = Vec::with_capacity(128);
        for offset in 0..128u8 {
            pattern.push(seed.wrapping_mul(17).wrapping_add(offset.wrapping_mul(29)));
        }

        if let Some(csr) = from_bytes::<TestGraph>(&pattern) {
            constructed += 1;
            if let Ok(wu_palmer) = csr.wu_palmer() {
                let node_ids: Vec<u8> = csr.node_ids().collect();
                test_utils::check_similarity_invariants(&wu_palmer, &node_ids, 16);
            }
        }
    }

    assert!(
        constructed > 0,
        "No TestGraph instances could be constructed from dense-like patterns"
    );
}

// ============================================================================
// Lin (mirrors fuzz/fuzz_targets/lin.rs)
// ============================================================================

type LinInput = (Vec<usize>, GenericGraph<u8, SquareCSR2D<CSR2D<u16, u8, u8>>>);

#[test]
fn test_arbitrary_lin() {
    for_each_instance::<LinInput, _>(|(occurrences, csr)| {
        if let Ok(lin) = csr.lin(occurrences.as_ref()) {
            let node_ids: Vec<u8> = csr.node_ids().collect();
            test_utils::check_similarity_invariants(&lin, &node_ids, 10);
        }
    });
}

// ============================================================================
// Direct Arbitrary construction with specific byte patterns
// ============================================================================

#[test]
fn test_arbitrary_csr2d_empty_bytes() {
    if let Some(csr) = from_bytes::<TestCSR>(&[]) {
        check_sparse_matrix_invariants(&csr);
    }
}

#[test]
fn test_arbitrary_csr2d_single_byte() {
    let _ = from_bytes::<TestCSR>(&[0]);
    let _ = from_bytes::<TestCSR>(&[1]);
    let _ = from_bytes::<TestCSR>(&[255]);
}

#[test]
fn test_arbitrary_csr2d_maxed_out_paths() {
    let mut pattern = vec![255u8, 255];
    for i in 0..=255u8 {
        for j in (0..=255u8).step_by(64) {
            pattern.push(i);
            pattern.push(j);
        }
    }
    let _ = from_bytes::<TestCSR>(&pattern);
}

#[test]
fn test_arbitrary_csr2d_dedup_path() {
    let mut pattern = vec![2u8, 2];
    for _ in 0..50 {
        pattern.push(0);
        pattern.push(0);
    }
    let _ = from_bytes::<TestCSR>(&pattern);
}
