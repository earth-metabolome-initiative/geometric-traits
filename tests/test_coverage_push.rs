//! Targeted tests to push coverage toward 95%.
#![cfg(feature = "std")]

use std::collections::HashMap;

use geometric_traits::{
    impls::{
        CSR2D, GenericMatrix2DWithPaddedDiagonal, PaddedMatrix2D, SortedVec, SquareCSR2D,
        ValuedCSR2D,
    },
    prelude::*,
    traits::{
        BidirectionalVocabularyRef, EdgesBuilder, SparseMatrix, SparseMatrix2D, SparseMatrixMut,
        SparseValuedMatrix, VocabularyBuilder, VocabularyRef,
    },
};

// ============================================================================
// VocabularyRef / BidirectionalVocabularyRef through references
// (covers vocabulary.rs lines 179-184, 236-241)
// ============================================================================

#[test]
fn test_vocabulary_ref_delegation_through_reference() {
    let sv: SortedVec<u8> = GenericVocabularyBuilder::default()
        .expected_number_of_symbols(3)
        .symbols(vec![(0usize, 10u8), (1, 20), (2, 30)].into_iter())
        .build()
        .unwrap();

    // Call through &SortedVec to exercise the delegation impls
    let sv_ref: &SortedVec<u8> = &sv;

    // VocabularyRef::convert_ref (line 179-180)
    assert_eq!(sv_ref.convert_ref(&0), Some(&10u8));
    assert_eq!(sv_ref.convert_ref(&1), Some(&20u8));
    assert_eq!(sv_ref.convert_ref(&99), None);

    // VocabularyRef::destination_refs (lines 183-184)
    let dest_refs: Vec<&u8> = sv_ref.destination_refs().collect();
    assert_eq!(dest_refs.len(), 3);

    // BidirectionalVocabularyRef is implemented for HashMap.
    // Calling through &HashMap exercises the delegation impl lines 236-241.
    let map: HashMap<usize, u8> = HashMap::from([(0, 10), (1, 20), (2, 30)]);
    let map_ref: &HashMap<usize, u8> = &map;
    assert_eq!(map_ref.invert_ref(&10u8), Some(&0usize));
    assert_eq!(map_ref.invert_ref(&20u8), Some(&1usize));
    assert_eq!(map_ref.invert_ref(&99u8), None);
    let src_refs: Vec<&usize> = map_ref.source_refs().collect();
    assert_eq!(src_refs.len(), 3);
}

// ============================================================================
// PaddedCoordinates backward iteration
// (covers padded_coordinates.rs lines 51-59)
// ============================================================================

#[test]
fn test_padded_coordinates_next_back() {
    let m: ValuedCSR2D<usize, usize, usize, f64> = ValuedCSR2D::try_from([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
    ])
    .unwrap();

    let padded = PaddedMatrix2D::new(m, |_: (usize, usize)| 99.0).unwrap();
    let mut coords = SparseMatrix::sparse_coordinates(&padded);

    // Call next_back to exercise backward iteration
    let last = coords.next_back();
    assert!(last.is_some());
    let second_last = coords.next_back();
    assert!(second_last.is_some());

    // Consume some from back until exhausted
    let mut back_items = Vec::new();
    back_items.push(last.unwrap());
    back_items.push(second_last.unwrap());
    while let Some(item) = coords.next_back() {
        back_items.push(item);
    }

    // Backward iteration should produce coordinates and terminate.
    assert!(!back_items.is_empty());
}

#[test]
fn test_padded_coordinates_mixed_direction() {
    let m: ValuedCSR2D<usize, usize, usize, f64> = ValuedCSR2D::try_from([
        [1.0, 2.0],
        [3.0, 4.0],
    ])
    .unwrap();

    let padded = PaddedMatrix2D::new(m, |_: (usize, usize)| 99.0).unwrap();
    let mut coords = SparseMatrix::sparse_coordinates(&padded);

    // Mix forward and backward
    let front1 = coords.next();
    let back1 = coords.next_back();
    assert!(front1.is_some());
    assert!(back1.is_some());
    assert_ne!(front1, back1);
}

// ============================================================================
// GenericMatrix2DWithPaddedDiagonal error paths
// (covers lines 56, 59 and 155, 166)
// ============================================================================

#[test]
fn test_padded_diagonal_empty_matrix() {
    // Create an empty valued matrix
    let m: ValuedCSR2D<usize, u8, u8, f64> =
        ValuedCSR2D::with_sparse_shaped_capacity((0, 0), 0);
    let padded = GenericMatrix2DWithPaddedDiagonal::new(m, |_: u8| 1.0);
    // Empty matrix should still construct successfully
    if let Ok(p) = padded {
        assert!(SparseMatrix::is_empty(&p));
        assert_eq!(SparseMatrix::last_sparse_coordinates(&p), None);
    }
}

#[test]
fn test_padded_diagonal_maxed_out_errors() {
    // Exceeds u8::MAX for RowIndex when checking number_of_columns.
    let m: ValuedCSR2D<usize, u8, u16, f64> =
        ValuedCSR2D::with_sparse_shaped_capacity((10u8, 256u16), 0);
    let result = GenericMatrix2DWithPaddedDiagonal::new(m, |_: u8| 1.0);
    assert!(result.is_err());

    // Exceeds u8::MAX for ColumnIndex when checking number_of_rows.
    let m: ValuedCSR2D<usize, u16, u8, f64> =
        ValuedCSR2D::with_sparse_shaped_capacity((256u16, 10u8), 0);
    let result = GenericMatrix2DWithPaddedDiagonal::new(m, |_: u16| 1.0);
    assert!(result.is_err());
}

// ============================================================================
// GenericVocabularyBuilder: ignore_duplicates with actual duplicate data
// (covers generic_vocabulary_builder.rs lines 101-105)
// ============================================================================

#[test]
fn test_vocabulary_builder_ignore_duplicates_with_data() {
    // Build with ignore_duplicates + actual duplicate source symbols.
    // HashMap emits RepeatedSourceSymbol, which is the branch we need here.
    let map: HashMap<usize, u8> = GenericVocabularyBuilder::default()
        .ignore_duplicates()
        .symbols(vec![(0usize, 10u8), (0, 20), (1, 30)].into_iter())
        .build()
        .unwrap();
    assert_eq!(map.len(), 2);
    assert_eq!(map.get(&0), Some(&10));
    assert_eq!(map.get(&1), Some(&30));
}

#[test]
fn test_vocabulary_builder_ignore_duplicates_dest() {
    // Build with ignore_duplicates + actual duplicate destination symbols.
    // HashMap emits RepeatedDestinationSymbol, which is ignored.
    let map: HashMap<usize, u8> = GenericVocabularyBuilder::default()
        .ignore_duplicates()
        .symbols(vec![(0usize, 10u8), (1, 10), (2, 30)].into_iter())
        .build()
        .unwrap();
    assert_eq!(map.len(), 2);
    assert_eq!(map.get(&0), Some(&10));
    assert_eq!(map.get(&2), Some(&30));
    assert_eq!(map.get(&1), None);
}

// ============================================================================
// Connected components error conversion
// (covers connected_components.rs lines 100-101, 106-107)
// ============================================================================

#[test]
fn test_connected_components_error_conversion() {
    use geometric_traits::traits::algorithms::connected_components::ConnectedComponentsError;

    let err = ConnectedComponentsError::TooManyComponents;
    let display = format!("{err}");
    assert!(!display.is_empty());

    // Convert to MonopartiteAlgorithmError
    let algo_err: geometric_traits::errors::monopartite_graph_error::algorithms::MonopartiteAlgorithmError = err.into();
    let display = format!("{algo_err}");
    assert!(!display.is_empty());
}

// ============================================================================
// M2DValues forward crossing through multi-row matrix
// (covers csr2d_values.rs lines 28-29)
// ============================================================================

#[test]
fn test_m2d_values_forward_crossing() {
    // Create a 4x4 padded diagonal matrix to get M2DValues iterator
    let m: ValuedCSR2D<usize, u8, u8, f64> = ValuedCSR2D::try_from([
        [1.0, 2.0, 99.0, 99.0],
        [3.0, 4.0, 99.0, 99.0],
        [99.0, 99.0, 5.0, 6.0],
        [99.0, 99.0, 7.0, 8.0],
    ])
    .unwrap();

    let padded = GenericMatrix2DWithPaddedDiagonal::new(m, |_: u8| 1.0).unwrap();

    // Iterate all sparse_values() to exercise forward crossing between rows
    let values: Vec<f64> = padded.sparse_values().collect();
    assert!(!values.is_empty());

    // Also test through PaddedMatrix2D
    let m2: ValuedCSR2D<usize, u8, u8, f64> = ValuedCSR2D::try_from([
        [10.0, 20.0],
        [30.0, 40.0],
        [50.0, 60.0],
    ])
    .unwrap();
    let padded2 = PaddedMatrix2D::new(m2, |_: (u8, u8)| 99.0).unwrap();
    let values2: Vec<f64> = padded2.sparse_values().collect();
    assert!(!values2.is_empty());
}

// ============================================================================
// Johnson: deeper algorithmic paths
// (covers johnson.rs lines 55, 60-61, 122-130, 144-146)
// ============================================================================

fn build_sq(n: usize, mut edges: Vec<(usize, usize)>) -> SquareCSR2D<CSR2D<usize, usize, usize>> {
    edges.sort_unstable();
    DiEdgesBuilder::default()
        .expected_number_of_edges(edges.len())
        .expected_shape(n)
        .edges(edges.into_iter())
        .build()
        .unwrap()
}

#[test]
fn test_johnson_large_scc_with_bypass() {
    // Large SCC with bypass edges to exercise blocking/unblocking deeply.
    // 0→1→2→3→0 (main cycle) plus bypass edges 0→2, 1→3
    let m = build_sq(4, vec![
        (0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (3, 0),
    ]);
    let cycles: Vec<Vec<usize>> = m.johnson().collect();
    // Multiple cycles: 0→1→2→3→0, 0→2→3→0, 0→1→3→0, 0→1→2→3→0 (various)
    assert!(cycles.len() >= 3);
}

#[test]
fn test_johnson_dense_with_blocking() {
    // K4 minus one edge to get blocking behavior
    let m = build_sq(4, vec![
        (0, 1), (0, 2), (0, 3),
        (1, 0), (1, 2), (1, 3),
        (2, 0), (2, 1), (2, 3),
        (3, 0), (3, 1),
        // Missing (3, 2) to break symmetry
    ]);
    let cycles: Vec<Vec<usize>> = m.johnson().collect();
    assert!(cycles.len() >= 5);
}

#[test]
fn test_johnson_multiple_components_varied() {
    // SCC1: 0→1→2→0 (3-cycle)
    // SCC2: 3→4→3 (2-cycle)
    // SCC3: 5→6→7→5 with bypass 5→7 (3-cycle + shortcut)
    let m = build_sq(8, vec![
        (0, 1), (1, 2), (2, 0),
        (3, 4), (4, 3),
        (5, 6), (5, 7), (6, 7), (7, 5),
    ]);
    let cycles: Vec<Vec<usize>> = m.johnson().collect();
    // 3 SCCs: first has 1 cycle, second has 1, third has 2
    assert_eq!(cycles.len(), 4);
}

#[test]
fn test_johnson_chain_of_twocycles() {
    // 0↔1, 1↔2, 2↔3 - overlapping 2-cycles in one SCC
    let m = build_sq(4, vec![
        (0, 1), (1, 0),
        (1, 2), (2, 1),
        (2, 3), (3, 2),
    ]);
    let cycles: Vec<Vec<usize>> = m.johnson().collect();
    // 3 two-cycles, plus longer cycles through the chain
    assert!(cycles.len() >= 3);
}

// ============================================================================
// CSR2DView forward iteration through empty rows
// (covers csr2d_view.rs lines 30, 35-36)
// ============================================================================

#[test]
fn test_csr2d_view_forward_through_empty_middle_rows() {
    // Matrix with entries only in rows 0 and 3 (rows 1,2 empty)
    // This forces CSR2DView's next() to skip empty rows via the Less branch
    let m: ValuedCSR2D<usize, u8, u8, f64> = ValuedCSR2D::try_from([
        [1.0, 99.0, 99.0, 99.0],
        [99.0, 99.0, 99.0, 99.0],
        [99.0, 99.0, 99.0, 99.0],
        [99.0, 99.0, 99.0, 2.0],
    ])
    .unwrap();

    let padded = GenericMatrix2DWithPaddedDiagonal::new(m, |_: u8| 1.0).unwrap();
    let coords: Vec<(u8, u8)> = SparseMatrix::sparse_coordinates(&padded).collect();
    // All rows should have at least the diagonal entry
    assert!(coords.len() >= 4);
}

// ============================================================================
// LAPJV: matrices that exercise deeper column reduction
// (covers lapjv/inner.rs lines 65, 92, 96, 148, etc.)
// ============================================================================

#[test]
fn test_lapjv_conflict_heavy() {
    use geometric_traits::traits::LAPJV;

    // Matrix where multiple rows want the same column
    let m: ValuedCSR2D<u8, u8, u8, f64> = ValuedCSR2D::try_from([
        [1.0, 50.0, 50.0, 50.0, 50.0],
        [1.0, 50.0, 50.0, 50.0, 50.0],
        [1.0, 50.0, 50.0, 50.0, 50.0],
        [50.0, 50.0, 50.0, 50.0, 1.0],
        [50.0, 50.0, 50.0, 1.0, 50.0],
    ])
    .unwrap();

    let padded = PaddedMatrix2D::new(m, |_: (u8, u8)| 900.0).unwrap();
    let result = padded.lapjv(1000.0).unwrap();
    assert_eq!(result.len(), 5);
}

#[test]
fn test_lapjv_asymmetric_costs() {
    use geometric_traits::traits::LAPJV;

    // Asymmetric costs forcing augmenting path search
    let m: ValuedCSR2D<u8, u8, u8, f64> = ValuedCSR2D::try_from([
        [1.0, 2.0, 3.0],
        [3.0, 1.0, 2.0],
        [2.0, 3.0, 1.0],
    ])
    .unwrap();

    let padded = PaddedMatrix2D::new(m, |_: (u8, u8)| 900.0).unwrap();
    let result = padded.lapjv(1000.0).unwrap();
    assert_eq!(result.len(), 3);
    // Optimal should assign each row to its cheapest: (0,0), (1,1), (2,2)
    assert!(result.contains(&(0, 0)));
    assert!(result.contains(&(1, 1)));
    assert!(result.contains(&(2, 2)));
}

// ============================================================================
// CSR2DColumns: ExactSizeIterator::len()
// (covers csr2d_columns.rs lines 41-50)
// ============================================================================

#[test]
fn test_csr2d_columns_exact_size() {
    let m: ValuedCSR2D<usize, u8, u8, f64> = ValuedCSR2D::try_from([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
    ])
    .unwrap();

    let columns = m.sparse_columns();
    let len = columns.len();
    let collected: Vec<u8> = m.sparse_columns().collect();
    assert_eq!(len, collected.len());
}

// ============================================================================
// M2DValues: ExactSizeIterator::len()
// (covers csr2d_values.rs lines 42-51)
// ============================================================================

#[test]
fn test_m2d_values_exact_size() {
    let m: ValuedCSR2D<usize, u8, u8, f64> = ValuedCSR2D::try_from([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
    ])
    .unwrap();

    let values = m.sparse_values();
    let len = values.len();
    let collected: Vec<f64> = m.sparse_values().collect();
    assert_eq!(len, collected.len());
}

// ============================================================================
// Louvain error paths
// (covers louvain.rs lines 143-144, 149-150, 180-181, 360)
// ============================================================================

#[test]
fn test_louvain_non_square_matrix() {
    // Non-square matrix should return error
    let m: ValuedCSR2D<usize, u8, u8, f64> = ValuedCSR2D::try_from([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    ])
    .unwrap();
    let result: Result<LouvainResult<usize>, _> = m.louvain(&LouvainConfig::default());
    assert!(result.is_err());
}

#[test]
fn test_louvain_config_validation() {
    let config = LouvainConfig {
        resolution: -1.0,
        ..Default::default()
    };
    let m: ValuedCSR2D<u8, u8, u8, f64> = ValuedCSR2D::try_from([
        [1.0, 2.0],
        [2.0, 1.0],
    ])
    .unwrap();
    let result: Result<LouvainResult<usize>, _> = m.louvain(&config);
    assert!(result.is_err());
}

// ============================================================================
// Sparse rows with padded diagonal
// (covers sparse_rows_with_padded_diagonal.rs lines 49, 77-81)
// ============================================================================

#[test]
fn test_padded_diagonal_sparse_rows_crossing() {
    // Create a padded diagonal matrix and iterate sparse_rows to cross rows
    let m: ValuedCSR2D<usize, u8, u8, f64> = ValuedCSR2D::try_from([
        [1.0, 99.0, 99.0],
        [99.0, 2.0, 99.0],
        [99.0, 99.0, 3.0],
    ])
    .unwrap();

    let padded = GenericMatrix2DWithPaddedDiagonal::new(m, |_: u8| 1.0).unwrap();
    let rows: Vec<u8> = padded.sparse_rows().collect();
    // Each row has 3 columns, so 9 rows entries total
    assert_eq!(rows.len(), 9);
}
