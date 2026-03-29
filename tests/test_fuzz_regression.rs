//! Regression tests that exercise the same code paths as fuzz targets.
//!
//! These tests construct instances via the `Arbitrary` trait from fixed byte
//! patterns and from a stable checked-in raw fixture, then run the same
//! invariant checks that the corresponding honggfuzz targets use (via shared
//! functions in `test_utils`).
#![cfg(feature = "arbitrary")]

use std::path::Path;

use arbitrary::Arbitrary;
use geometric_traits::{
    impls::{CSR2D, SquareCSR2D, SymmetricCSR2D, ValuedCSR2D, VecMatrix2D},
    naive_structs::GenericGraph,
    prelude::*,
    test_utils::{
        self, FuzzBlossomVCase, FuzzStructuredBlossomVCase, check_blossom_v_invariants,
        check_floyd_warshall_invariants, check_gabow_1976_invariants, check_gth_invariants,
        check_kahn_ordering, check_karp_sipser_invariants, check_lap_sparse_wrapper_invariants,
        check_lap_square_invariants, check_leiden_invariants, check_louvain_invariants,
        check_padded_diagonal_invariants, check_padded_matrix2d_invariants,
        check_pairwise_bfs_matches_unit_floyd_warshall,
        check_pairwise_dijkstra_matches_floyd_warshall, check_sparse_matrix_invariants,
        check_structured_blossom_v_invariants, check_valued_matrix_invariants, from_bytes,
        replay_dir,
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

fn replay_shared_fixture<T>() -> Vec<T>
where
    T: for<'a> Arbitrary<'a>,
{
    let path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/fuzz/sample.bin");
    let bytes = std::fs::read(&path)
        .unwrap_or_else(|_| panic!("failed to read fixture {}", path.display()));
    from_bytes::<T>(&bytes).into_iter().collect()
}

fn decode_hex_fixture(hex: &str) -> Vec<u8> {
    let filtered: String = hex.chars().filter(|c| !c.is_whitespace()).collect();
    assert_eq!(filtered.len() % 2, 0, "hex fixture should contain an even number of nybbles");
    filtered
        .as_bytes()
        .chunks_exact(2)
        .map(|pair| {
            let s = std::str::from_utf8(pair).expect("hex fixture should be valid ASCII");
            u8::from_str_radix(s, 16).expect("hex fixture should only contain valid hex digits")
        })
        .collect()
}

fn assert_blossom_v_honggfuzz_replay_ok(label: &str, pattern: &[u8]) {
    let instance = from_bytes::<FuzzBlossomVCase>(pattern)
        .expect("saved honggfuzz crash bytes should decode as FuzzBlossomVCase");
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        check_blossom_v_invariants(&instance);
    }));
    if let Err(payload) = result {
        let msg = if let Some(s) = payload.downcast_ref::<&str>() {
            (*s).to_string()
        } else if let Some(s) = payload.downcast_ref::<String>() {
            s.clone()
        } else {
            "<non-string panic payload>".to_string()
        };
        panic!("Blossom V {label} failed for bytes {:?} decoded as {:?}: {msg}", pattern, instance);
    }
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
    for instance in replay_shared_fixture::<TestCSR>() {
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
    for instance in replay_shared_fixture::<TestValuedCSR>() {
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
fn test_arbitrary_pairwise_bfs() {
    for_each_instance::<TestSquareCSR, _>(check_pairwise_bfs_matches_unit_floyd_warshall);
}

#[test]
fn test_replay_tarjan_corpus() {
    for instance in replay_shared_fixture::<TestSquareCSR>() {
        let _sccs: Vec<_> = instance.tarjan().collect();
    }
}

#[test]
fn test_replay_johnson_corpus() {
    for instance in replay_shared_fixture::<TestSquareCSR>() {
        let _cycles: Vec<_> = instance.johnson().collect();
    }
}

#[test]
fn test_replay_kahn_corpus() {
    for instance in replay_shared_fixture::<TestSquareCSR>() {
        check_kahn_ordering(&instance, 255);
    }
}

#[test]
fn test_replay_pairwise_bfs_corpus() {
    for instance in replay_shared_fixture::<TestSquareCSR>() {
        check_pairwise_bfs_matches_unit_floyd_warshall(&instance);
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
    let _: Vec<TestGraph> = replay_shared_fixture();
}

// ============================================================================
// Exact Karp-Sipser preprocessing
// ============================================================================

type TestSymmetricCSR = SymmetricCSR2D<CSR2D<u16, u8, u8>>;

#[test]
fn test_arbitrary_karp_sipser_invariants() {
    for_each_instance::<TestSymmetricCSR, _>(|graph| {
        if graph.order() as usize <= 64 {
            check_karp_sipser_invariants(graph);
        }
    });
}

#[test]
fn test_replay_karp_sipser_corpus() {
    for instance in replay_shared_fixture::<TestSymmetricCSR>() {
        if instance.order() as usize <= 64 {
            check_karp_sipser_invariants(&instance);
        }
    }
}

#[test]
fn test_replay_sink_nodes_corpus() {
    let _: Vec<TestGraph> = replay_shared_fixture();
}

// ============================================================================
// Gabow 1976 (mirrors fuzz/fuzz_targets/gabow_1976.rs)
// ============================================================================

#[test]
fn test_arbitrary_gabow_1976() {
    for_each_instance::<TestSymmetricCSR, _>(|graph| {
        if graph.order() as usize <= 128 {
            check_gabow_1976_invariants(graph);
        }
    });
}

#[test]
fn test_replay_gabow_1976_corpus() {
    for instance in replay_shared_fixture::<TestSymmetricCSR>() {
        if instance.order() as usize <= 128 {
            check_gabow_1976_invariants(&instance);
        }
    }
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
    for instance in replay_shared_fixture::<TestPaddedDiag>() {
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

// ============================================================================
// Blossom V (mirrors fuzz/fuzz_targets/blossom_v.rs)
// ============================================================================

#[test]
fn test_arbitrary_blossom_v_invariants() {
    let patterns = test_byte_patterns();
    let mut constructed = 0usize;

    for pattern in &patterns {
        if let Some(instance) = from_bytes::<FuzzBlossomVCase>(pattern) {
            constructed += 1;
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                check_blossom_v_invariants(&instance);
            }));
            if let Err(payload) = result {
                let msg = if let Some(s) = payload.downcast_ref::<&str>() {
                    (*s).to_string()
                } else if let Some(s) = payload.downcast_ref::<String>() {
                    s.clone()
                } else {
                    "<non-string panic payload>".to_string()
                };
                panic!(
                    "Blossom V arbitrary invariant failure for pattern {:?} decoded as {:?}: {msg}",
                    pattern, instance
                );
            }
        }
    }

    assert!(constructed > 0, "No FuzzBlossomVCase instances could be constructed");
}

#[test]
fn test_replay_blossom_v_corpus() {
    let corpus_dir = Path::new("fuzz/hfuzz_workspace/blossom_v/input");
    for instance in replay_dir::<FuzzBlossomVCase>(corpus_dir) {
        check_blossom_v_invariants(&instance);
    }
}

#[test]
fn test_arbitrary_blossom_v_structured_invariants() {
    let patterns = test_byte_patterns();
    let mut constructed = 0usize;

    for pattern in &patterns {
        if let Some(instance) = from_bytes::<FuzzStructuredBlossomVCase>(pattern) {
            constructed += 1;
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                check_structured_blossom_v_invariants(&instance);
            }));
            if let Err(payload) = result {
                let msg = if let Some(s) = payload.downcast_ref::<&str>() {
                    (*s).to_string()
                } else if let Some(s) = payload.downcast_ref::<String>() {
                    s.clone()
                } else {
                    "<non-string panic payload>".to_string()
                };
                panic!(
                    "Structured Blossom V arbitrary invariant failure for pattern {:?} decoded as {:?}: {msg}",
                    pattern, instance
                );
            }
        }
    }

    assert!(constructed > 0, "No FuzzStructuredBlossomVCase instances could be constructed");
}

#[test]
fn test_replay_blossom_v_structured_corpus() {
    let corpus_dir = Path::new("fuzz/hfuzz_workspace/blossom_v_structured/input");
    for instance in replay_dir::<FuzzStructuredBlossomVCase>(corpus_dir) {
        check_structured_blossom_v_invariants(&instance);
    }
}

#[test]
fn test_replay_blossom_v_honggfuzz_sigabrt_case() {
    let pattern: &[u8] = &[
        0x02, 0x13, 0x00, 0x01, 0xc7, 0x1c, 0x71, 0x9a, 0x99, 0x99, 0x99, 0x99, 0x99, 0x99, 0x01,
        0xc7, 0x1c, 0x71, 0xc8, 0x05, 0x45, 0x87, 0x28, 0x07, 0xcd, 0x14, 0x19, 0x03, 0x5d, 0x6e,
        0x7b, 0x00, 0x00, 0x00, 0x00, 0x00, 0xab, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0x00, 0x04,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    ];
    let instance = from_bytes::<FuzzBlossomVCase>(pattern)
        .expect("saved honggfuzz crash bytes should decode as FuzzBlossomVCase");
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        check_blossom_v_invariants(&instance);
    }));
    if let Err(payload) = result {
        let msg = if let Some(s) = payload.downcast_ref::<&str>() {
            (*s).to_string()
        } else if let Some(s) = payload.downcast_ref::<String>() {
            s.clone()
        } else {
            "<non-string panic payload>".to_string()
        };
        panic!(
            "Blossom V honggfuzz replay failed for bytes {:?} decoded as {:?}: {msg}",
            pattern, instance
        );
    }
}

#[test]
fn test_replay_blossom_v_honggfuzz_sigabrt_case_2() {
    let pattern: &[u8] = &[
        0xe1, 0x17, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x18, 0x00, 0xab, 0xaa, 0x47, 0x41, 0xc5, 0x4a, 0xd5, 0xaf, 0xef, 0x13, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0xe5, 0x32, 0x3f, 0x1f, 0x84, 0xd2, 0x39, 0xcd, 0x6e, 0x9a, 0x21,
        0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x99, 0x00, 0x0c, 0x99, 0x99, 0x9a, 0xaf, 0x14,
        0x05, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x28, 0x1e, 0xa5, 0x91, 0x4f, 0x45, 0x99,
        0x5f, 0x95, 0x28, 0x64, 0x12, 0x45, 0x45, 0x45, 0xc3, 0x99, 0x5f, 0x95, 0x00, 0x00, 0x00,
        0xe5, 0x32, 0x3f, 0x1f, 0x84, 0xcd, 0x6e, 0x9a, 0x99, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x13, 0x32, 0x99, 0x99, 0x99, 0x99, 0x99, 0x01, 0xc1, 0x89, 0x32, 0xe3, 0x68, 0x12,
        0x7d, 0x93, 0xaf, 0x14, 0x0a, 0x0a, 0x3b, 0x0c,
    ];
    let instance = from_bytes::<FuzzBlossomVCase>(pattern)
        .expect("saved honggfuzz crash bytes should decode as FuzzBlossomVCase");
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        check_blossom_v_invariants(&instance);
    }));
    if let Err(payload) = result {
        let msg = if let Some(s) = payload.downcast_ref::<&str>() {
            (*s).to_string()
        } else if let Some(s) = payload.downcast_ref::<String>() {
            s.clone()
        } else {
            "<non-string panic payload>".to_string()
        };
        panic!(
            "Blossom V honggfuzz replay 2 failed for bytes {:?} decoded as {:?}: {msg}",
            pattern, instance
        );
    }
}

#[test]
fn test_replay_blossom_v_honggfuzz_sigabrt_case_3() {
    let pattern: &[u8] = &[
        0xe0, 0x6e, 0x8e, 0xfc, 0x13, 0x00, 0x00, 0x00, 0x00, 0xab, 0xaa, 0x47, 0x41, 0xc5, 0x4a,
        0xd5, 0xaf, 0xef, 0x0c, 0x00, 0x98, 0xf2, 0xda, 0xb6, 0xf1, 0x7a, 0x04, 0xa9, 0xb7, 0xfb,
        0x13, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xe5, 0x31, 0x3f, 0x1f, 0x84, 0x00, 0x13,
        0x32, 0x99, 0x99, 0x99, 0x99, 0x99, 0x0b, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00,
    ];
    let instance = from_bytes::<FuzzBlossomVCase>(pattern)
        .expect("saved honggfuzz crash bytes should decode as FuzzBlossomVCase");
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        check_blossom_v_invariants(&instance);
    }));
    if let Err(payload) = result {
        let msg = if let Some(s) = payload.downcast_ref::<&str>() {
            (*s).to_string()
        } else if let Some(s) = payload.downcast_ref::<String>() {
            s.clone()
        } else {
            "<non-string panic payload>".to_string()
        };
        panic!(
            "Blossom V honggfuzz replay 3 failed for bytes {:?} decoded as {:?}: {msg}",
            pattern, instance
        );
    }
}

#[test]
fn test_replay_blossom_v_honggfuzz_sigabrt_case_4() {
    let pattern: &[u8] = &[
        0x51, 0x6a, 0xc6, 0xec, 0x82, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x13, 0x37, 0x14,
        0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0xef, 0x0b, 0x2c, 0x90, 0xa9, 0x36, 0x40, 0xc9, 0xa7, 0x6a, 0xcb,
        0x45, 0x46, 0x10, 0x19, 0x8d, 0x1a, 0xba, 0xa7, 0xa4, 0x8b, 0x11, 0xc5, 0x78, 0x3b, 0x03,
        0xd8, 0xb6, 0x32, 0x2b, 0xde, 0x78, 0xfd, 0x21, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x7f, 0xb6, 0x30, 0x09, 0x26, 0xb5,
        0xce, 0xd8, 0x6d, 0xe6, 0x5e, 0x6b, 0xd4, 0x6d, 0x81, 0xb4, 0x44, 0xa9, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x40, 0xc5, 0xdc, 0x94, 0xeb, 0xf8, 0x9a, 0x2a, 0x40, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0xd0, 0xfc, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0xff, 0xff, 0xff, 0xff, 0xff, 0x7f, 0xaa, 0x7d, 0x13, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x54, 0xcc, 0x4c, 0x0a, 0x12, 0x31, 0x71, 0x03, 0xa1, 0x0c, 0x2d, 0x6a, 0x77,
        0xc1, 0x7d, 0x90, 0xeb, 0x12, 0x2c, 0xd2, 0x19, 0x7e, 0x15, 0x5b, 0xfe, 0x85, 0xaa, 0x1f,
        0x57, 0xce, 0x6c, 0xec, 0x46, 0x96, 0x96, 0x53, 0xf1, 0x45, 0x2d, 0x73, 0x02, 0x6b, 0x2f,
        0x20, 0x64, 0xa5, 0xc4, 0x9c, 0x21, 0x36, 0xe4, 0xc5, 0xbc, 0x09, 0x51, 0x6d, 0x63, 0x18,
        0x7c, 0x00, 0xc8, 0x71, 0x1c, 0xc7, 0x71, 0x1c, 0xc7, 0x01, 0x3c, 0x91, 0xc2, 0xfe, 0xb5,
        0x42, 0x8a, 0x20, 0xda, 0xe0, 0x11, 0x0e, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x57, 0xb3, 0x6a, 0x0c, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x6e, 0x0f, 0xda, 0xcf,
        0x8a, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x18, 0x42, 0x78, 0x10, 0x18, 0x42,
        0x78, 0x10, 0x12, 0x9f, 0x16, 0xaf, 0xc1, 0xce, 0x3a, 0xe4, 0x7b, 0x88, 0x5a, 0x1e, 0xb0,
        0xe6, 0xf7, 0x4e, 0x4c, 0x0e, 0x07, 0xcb, 0xc3, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
        0x1f, 0xaa, 0xaa, 0xaa, 0x0a, 0xda, 0x0e, 0x30, 0x6c, 0xc1, 0xda, 0x7b, 0x53, 0xfe, 0xcc,
        0x54, 0xdc, 0x94, 0xeb, 0xf8, 0x9a, 0x2a, 0xd0, 0xfc, 0x00, 0x00, 0x0b, 0xff, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x13, 0x00, 0x10, 0xde, 0x6e, 0x84, 0xe6, 0xda, 0x0f, 0xda, 0x11,
        0xb6, 0x82, 0x69, 0xd7, 0x8c, 0xfd, 0x0b, 0x93, 0xb4, 0xc4, 0x71, 0x55, 0xbb, 0x4b, 0xfa,
        0x32, 0x29, 0x91, 0x10, 0x00, 0x00, 0x00, 0x00, 0x32, 0xbb, 0x31, 0xc8, 0x25, 0x25, 0x25,
        0x25, 0x25, 0x25, 0x25, 0x25, 0x25, 0x25, 0x25, 0x25, 0x25, 0x25, 0x25, 0x25, 0x25, 0x25,
        0x25, 0x25, 0x25, 0x25, 0x25, 0x25, 0x25, 0x25, 0x25, 0x25, 0x25, 0x25, 0x25, 0x25, 0x25,
        0x25, 0x25, 0x25, 0x25, 0x25, 0x25, 0x25, 0x25, 0x25, 0x25, 0x25, 0x25, 0x25, 0x25, 0x25,
        0x25, 0x25, 0x25, 0x25, 0x25, 0x25, 0x25, 0x25, 0x25, 0x25, 0x25, 0x25, 0x25, 0x25, 0x25,
        0x25, 0x25, 0x25, 0x25, 0x25, 0x25, 0x25, 0x25, 0x0b, 0x25, 0x37, 0xa1, 0x79, 0xa5, 0x91,
        0xb3, 0x69, 0x6e, 0x0f, 0xda, 0xcf, 0x8a, 0x76, 0x16, 0x6e, 0x17, 0x18, 0x42, 0x78, 0x10,
        0x12, 0x9f, 0x16, 0xaf, 0xc1, 0xce, 0x3a, 0xe4, 0x7b, 0x88, 0x5a, 0x1e, 0xb0, 0xe6, 0xf7,
        0x4e, 0x4c, 0x0e, 0x07, 0xcb, 0xc3, 0xbf, 0x46, 0xa3, 0xf2, 0x59, 0x80, 0x03, 0xc1, 0x0a,
        0x91, 0x5b, 0xb7, 0xda, 0x0e, 0x30, 0x6c, 0xc1, 0xda, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x02, 0x7b, 0x4a, 0xd5, 0x00, 0x0b, 0x00, 0x00, 0x55, 0x7b, 0xc4, 0xe5, 0xe1, 0x9b,
        0x77, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x07, 0x61, 0x4a, 0x83, 0xa9, 0x6c, 0x31,
        0x31, 0x4c, 0x16, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0x36, 0x00, 0x00, 0x0c,
        0xaf, 0xef, 0x13, 0x00, 0x00, 0x00, 0x57, 0x05, 0x08, 0x7f, 0xff, 0xff, 0xff, 0xff, 0x00,
        0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xab, 0xff, 0xff, 0xff, 0x00, 0x00, 0x00, 0x13, 0x99,
        0xe3, 0x68, 0x12, 0x7d, 0x93, 0xaf, 0x14, 0x28, 0x64, 0x12, 0x45, 0x45, 0x45, 0x99, 0x5f,
        0x95, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0c, 0x28, 0x64, 0x12, 0x45, 0x45, 0x45,
        0xc3, 0x99, 0x5f, 0x95, 0x00, 0x00, 0x00, 0xe5, 0x31, 0x32, 0x3f, 0x1f, 0x84, 0xcd, 0x6e,
        0x9a, 0x99, 0x99, 0x01, 0xc7, 0x1c, 0x01, 0x01, 0x01, 0x01, 0x71, 0xc7, 0x1c, 0x71, 0xc8,
        0x99, 0x99, 0x99, 0x99, 0x99, 0x01, 0xc1, 0x89, 0x32, 0xe3, 0x68, 0x12, 0x7d, 0x93, 0xaf,
        0x14, 0x0a, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x18, 0xff, 0xff, 0xff, 0xff, 0x7f,
        0x0b, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    ];
    let instance = from_bytes::<FuzzBlossomVCase>(pattern)
        .expect("saved honggfuzz crash bytes should decode as FuzzBlossomVCase");
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        check_blossom_v_invariants(&instance);
    }));
    if let Err(payload) = result {
        let msg = if let Some(s) = payload.downcast_ref::<&str>() {
            (*s).to_string()
        } else if let Some(s) = payload.downcast_ref::<String>() {
            s.clone()
        } else {
            "<non-string panic payload>".to_string()
        };
        panic!(
            "Blossom V honggfuzz replay 4 failed for bytes {:?} decoded as {:?}: {msg}",
            pattern, instance
        );
    }
}

#[test]
fn test_replay_blossom_v_honggfuzz_sigabrt_case_5() {
    let pattern: &[u8] = &[
        0x51, 0x6a, 0xc6, 0xec, 0x82, 0x36, 0x14, 0x8f, 0x57, 0xd4, 0x13, 0xd9, 0xb9, 0x62, 0x1e,
        0xeb, 0x73, 0xbc, 0x20, 0x9d, 0x01, 0x00, 0x00, 0x00, 0xb8, 0x90, 0xa9, 0x00, 0x01, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x36, 0x40, 0xc9, 0xa7, 0x6a, 0xcb, 0x45, 0x46, 0x10, 0x19,
        0x8d, 0x2d, 0x34, 0x32, 0x33, 0x35, 0x38, 0x38, 0x78, 0x3b, 0x03, 0xd8, 0xb6, 0x32, 0x2b,
        0xde, 0x0a, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xab, 0x78, 0xfd, 0x68, 0xd3, 0x5e, 0x80,
        0x05, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x5e, 0xf5, 0x01, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x15, 0x9e, 0xe1, 0x7f, 0x0c, 0x00, 0x0b, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0xb6, 0x31, 0x09, 0x26, 0xb5, 0xce, 0xd8, 0x6d, 0xe6,
        0x5e, 0x6b, 0xd4, 0x6d, 0x81, 0xbf, 0x44, 0xa9, 0xce, 0x3f, 0x3d, 0x8e, 0xab, 0xaa, 0x47,
        0x41, 0xc5, 0x19, 0xc3, 0xe6, 0x0b, 0xd8, 0x74, 0xff, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x18, 0x00, 0xff, 0xff, 0xff, 0xff, 0xff, 0x7f, 0xdc, 0x94, 0xeb, 0xf8, 0x9a, 0x2a,
        0xd0, 0xfc, 0x00, 0x00, 0x0b, 0xff, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x13, 0x00, 0x10,
        0x00, 0x0a, 0x84, 0xe6, 0xda, 0x0f, 0xda, 0x11, 0xb6, 0x82, 0x69, 0xd7, 0x8c, 0xfd, 0x56,
        0x55, 0x55, 0x54, 0x55, 0x00, 0x55, 0x05, 0x0b, 0x05, 0x00, 0xc4, 0x79, 0x55, 0xbb, 0x4b,
        0xfa, 0x33, 0x29, 0x91, 0x10, 0x00, 0x00, 0x00, 0x00, 0x32, 0xbb, 0x20, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x01, 0x91, 0xb3, 0x69, 0x6e, 0x00, 0x00, 0x00, 0x00, 0x76, 0x16, 0x6e,
        0x00, 0x00, 0x0c, 0x00, 0x55, 0x56, 0xbc, 0x09, 0x51, 0x6d, 0x63, 0x18, 0x7c, 0x00, 0x3c,
        0x91, 0xc2, 0xfe, 0xc8, 0x71, 0x1c, 0xc7, 0x71, 0x1c, 0xc7, 0x01, 0x57, 0x85, 0xfe, 0xcc,
        0x54, 0xaa, 0x7d, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x54, 0xcc, 0x4c,
        0x0a, 0x12, 0x31, 0x71, 0x03, 0x5e, 0x0c, 0x2d, 0x6a, 0x77, 0xc1, 0x7d, 0x91, 0xeb, 0x12,
        0x2c, 0xc2, 0x19, 0x7e, 0x15, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x55, 0x6c, 0x01, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xec, 0x4e, 0x96, 0x96, 0x53, 0xf1, 0x45, 0x2d, 0x73,
        0x02, 0x6b, 0x2f, 0x20, 0x64, 0xa5, 0xc4, 0x9c, 0x21, 0x36, 0xe4, 0xc5, 0x05, 0x55, 0x55,
        0x55, 0x55, 0x55, 0x18, 0x01, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x12, 0x9f, 0x16,
        0xaf, 0xc1, 0xce, 0x3a, 0xe4, 0x7b, 0x88, 0x5a, 0x1e, 0xb0, 0xe6, 0xf7, 0x4e, 0x4c, 0x0e,
        0x07, 0xcb, 0xc3, 0xbf, 0x46, 0x2d, 0x35, 0x35, 0x35, 0x34, 0x35, 0x34, 0x36, 0x36, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x20, 0x09, 0xf2, 0x59, 0x80, 0x03, 0xc1, 0x0a, 0xdb,
        0xdb, 0xdb, 0xdb, 0xdb, 0xdb, 0xdb, 0xdb, 0xdb, 0xdb, 0x4a, 0xd5, 0x00, 0x0b, 0x00, 0x00,
        0x55, 0x7b, 0xc4, 0xe5, 0xe1, 0x9b, 0x77, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00,
        0x00, 0x4a, 0x83, 0xa9, 0x6c, 0x31, 0x31, 0xfc, 0xa0, 0x75, 0xaf, 0x4c, 0x16, 0x00, 0x00,
        0x00, 0x00, 0xba, 0xfc, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xaf, 0xef, 0x13, 0x00, 0x00,
        0x00, 0x57, 0x05, 0x08, 0x7f, 0x07, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xff, 0xaa,
        0xaa, 0xab, 0xff, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0xff, 0xff, 0x00, 0x00,
        0x00, 0x13, 0x99, 0xe3, 0x68, 0x12, 0x7d, 0x93, 0xaf, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x54, 0xcc, 0x4c, 0x54, 0xcc, 0x4c, 0x0a, 0x02, 0x00,
    ];
    let instance = from_bytes::<FuzzBlossomVCase>(pattern)
        .expect("saved honggfuzz crash bytes should decode as FuzzBlossomVCase");
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        check_blossom_v_invariants(&instance);
    }));
    if let Err(payload) = result {
        let msg = if let Some(s) = payload.downcast_ref::<&str>() {
            (*s).to_string()
        } else if let Some(s) = payload.downcast_ref::<String>() {
            s.clone()
        } else {
            "<non-string panic payload>".to_string()
        };
        panic!(
            "Blossom V honggfuzz replay 5 failed for bytes {:?} decoded as {:?}: {msg}",
            pattern, instance
        );
    }
}

#[test]
fn test_replay_blossom_v_honggfuzz_sigabrt_case_6() {
    let pattern =
        decode_hex_fixture(include_str!("fixtures/blossom_v_honggfuzz_sigabrt_case_6.hex"));
    assert_blossom_v_honggfuzz_replay_ok("honggfuzz replay 6", &pattern);
}

#[test]
fn test_replay_blossom_v_honggfuzz_sigabrt_case_7() {
    let pattern =
        decode_hex_fixture(include_str!("fixtures/blossom_v_honggfuzz_sigabrt_case_7.hex"));
    assert_blossom_v_honggfuzz_replay_ok("honggfuzz replay 7", &pattern);
}

#[test]
fn test_replay_blossom_v_honggfuzz_sigabrt_case_8() {
    let pattern =
        decode_hex_fixture(include_str!("fixtures/blossom_v_honggfuzz_sigabrt_case_8.hex"));
    assert_blossom_v_honggfuzz_replay_ok("honggfuzz replay 8", &pattern);
}

#[test]
fn test_replay_hopcroft_karp_corpus() {
    for instance in replay_shared_fixture::<TestCSR>() {
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
    for instance in replay_shared_fixture::<TestValuedCSR>() {
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
    for instance in replay_shared_fixture::<TestValuedCSR>() {
        check_louvain_invariants(&instance);
    }
}

// ============================================================================
// Floyd-Warshall (mirrors fuzz/fuzz_targets/floyd_warshall.rs)
// ============================================================================

#[test]
fn test_arbitrary_floyd_warshall() {
    for_each_instance::<TestValuedCSR, _>(check_floyd_warshall_invariants);
}

#[test]
fn test_replay_floyd_warshall_corpus() {
    for instance in replay_shared_fixture::<TestValuedCSR>() {
        check_floyd_warshall_invariants(&instance);
    }
}

#[test]
fn test_arbitrary_pairwise_dijkstra() {
    for_each_instance::<TestValuedCSR, _>(check_pairwise_dijkstra_matches_floyd_warshall);
}

#[test]
fn test_replay_pairwise_dijkstra_corpus() {
    for instance in replay_shared_fixture::<TestValuedCSR>() {
        check_pairwise_dijkstra_matches_floyd_warshall(&instance);
    }
}

// ============================================================================
// Dense GTH (mirrors fuzz/fuzz_targets/gth.rs)
// ============================================================================

type TestDenseMatrix = VecMatrix2D<f64>;

#[test]
fn test_arbitrary_gth() {
    for_each_instance::<TestDenseMatrix, _>(check_gth_invariants);
}

#[test]
fn test_replay_gth_corpus() {
    for instance in replay_shared_fixture::<TestDenseMatrix>() {
        check_gth_invariants(&instance);
    }
}

// ============================================================================
// Leiden (mirrors fuzz/fuzz_targets/leiden.rs)
// ============================================================================

#[test]
fn test_arbitrary_leiden() {
    for_each_instance::<TestValuedCSR, _>(check_leiden_invariants);
}

#[test]
fn test_replay_leiden_corpus() {
    for instance in replay_shared_fixture::<TestValuedCSR>() {
        check_leiden_invariants(&instance);
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
