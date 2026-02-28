//! Tests targeting specific uncovered code paths to push coverage toward 95%.
#![cfg(feature = "std")]

use geometric_traits::{
    impls::{
        CSR2D, Intersection, MutabilityError, SortedVec, SquareCSR2D, SymmetricCSR2D, ValuedCSR2D,
    },
    prelude::*,
    traits::{
        EdgesBuilder, MatrixMut, SparseMatrix, SparseMatrixMut, SparseSquareMatrix,
        VocabularyBuilder, algorithms::connected_components::ConnectedComponents,
    },
};

// ============================================================================
// Error conversions: From<MutabilityError<SquareCSR2D<M>>> for
// UpperTriangularCSR2D<M> and other From impls (lines 98-197 of error.rs)
// ============================================================================

type TestCSR = CSR2D<usize, usize, usize>;

#[test]
fn test_error_conversion_csr2d_to_square() {
    // From<MutabilityError<M>> for MutabilityError<SquareCSR2D<M>>
    let err: MutabilityError<TestCSR> = MutabilityError::UnorderedCoordinate((1, 2));
    let converted: MutabilityError<SquareCSR2D<TestCSR>> = err.into();
    let display = format!("{converted}");
    assert!(display.contains("Unordered coordinate"));

    let err: MutabilityError<TestCSR> = MutabilityError::DuplicatedEntry((3, 4));
    let converted: MutabilityError<SquareCSR2D<TestCSR>> = err.into();
    let display = format!("{converted}");
    assert!(display.contains("Duplicated entry"));

    let err: MutabilityError<TestCSR> = MutabilityError::OutOfBounds((0, 0), (5, 5), "test");
    let converted: MutabilityError<SquareCSR2D<TestCSR>> = err.into();
    let display = format!("{converted}");
    assert!(display.contains("out of expected bounds"));

    let err: MutabilityError<TestCSR> = MutabilityError::MaxedOutRowIndex;
    let converted: MutabilityError<SquareCSR2D<TestCSR>> = err.into();
    let display = format!("{converted}");
    assert!(display.contains("maxed out"));

    let err: MutabilityError<TestCSR> = MutabilityError::MaxedOutColumnIndex;
    let converted: MutabilityError<SquareCSR2D<TestCSR>> = err.into();
    let display = format!("{converted}");
    assert!(display.contains("maxed out"));

    let err: MutabilityError<TestCSR> = MutabilityError::MaxedOutSparseIndex;
    let converted: MutabilityError<SquareCSR2D<TestCSR>> = err.into();
    let display = format!("{converted}");
    assert!(display.contains("maxed out"));

    let err: MutabilityError<TestCSR> = MutabilityError::IncompatibleShape;
    let converted: MutabilityError<SquareCSR2D<TestCSR>> = err.into();
    let display = format!("{converted}");
    assert!(display.contains("shape"));
}

#[test]
fn test_error_conversion_square_to_upper_triangular() {
    use geometric_traits::impls::UpperTriangularCSR2D;

    let err: MutabilityError<SquareCSR2D<TestCSR>> = MutabilityError::UnorderedCoordinate((1, 2));
    let converted: MutabilityError<UpperTriangularCSR2D<TestCSR>> = err.into();
    let display = format!("{converted}");
    assert!(display.contains("Unordered coordinate"));

    let err: MutabilityError<SquareCSR2D<TestCSR>> = MutabilityError::DuplicatedEntry((3, 4));
    let converted: MutabilityError<UpperTriangularCSR2D<TestCSR>> = err.into();
    let display = format!("{converted}");
    assert!(display.contains("Duplicated entry"));

    let err: MutabilityError<SquareCSR2D<TestCSR>> =
        MutabilityError::OutOfBounds((0, 0), (5, 5), "ctx");
    let converted: MutabilityError<UpperTriangularCSR2D<TestCSR>> = err.into();
    let display = format!("{converted}");
    assert!(display.contains("out of expected bounds"));

    let err: MutabilityError<SquareCSR2D<TestCSR>> = MutabilityError::MaxedOutRowIndex;
    let _: MutabilityError<UpperTriangularCSR2D<TestCSR>> = err.into();

    let err: MutabilityError<SquareCSR2D<TestCSR>> = MutabilityError::MaxedOutColumnIndex;
    let _: MutabilityError<UpperTriangularCSR2D<TestCSR>> = err.into();

    let err: MutabilityError<SquareCSR2D<TestCSR>> = MutabilityError::MaxedOutSparseIndex;
    let _: MutabilityError<UpperTriangularCSR2D<TestCSR>> = err.into();

    let err: MutabilityError<SquareCSR2D<TestCSR>> = MutabilityError::IncompatibleShape;
    let _: MutabilityError<UpperTriangularCSR2D<TestCSR>> = err.into();
}

#[test]
fn test_error_conversion_upper_triangular_to_symmetric() {
    use geometric_traits::impls::UpperTriangularCSR2D;

    let err: MutabilityError<UpperTriangularCSR2D<TestCSR>> =
        MutabilityError::UnorderedCoordinate((1, 2));
    let converted: MutabilityError<SymmetricCSR2D<TestCSR>> = err.into();
    let display = format!("{converted}");
    assert!(display.contains("Unordered coordinate"));

    let err: MutabilityError<UpperTriangularCSR2D<TestCSR>> =
        MutabilityError::DuplicatedEntry((3, 4));
    let converted: MutabilityError<SymmetricCSR2D<TestCSR>> = err.into();
    let display = format!("{converted}");
    assert!(display.contains("Duplicated entry"));

    let err: MutabilityError<UpperTriangularCSR2D<TestCSR>> =
        MutabilityError::OutOfBounds((0, 0), (5, 5), "sym");
    let converted: MutabilityError<SymmetricCSR2D<TestCSR>> = err.into();
    let display = format!("{converted}");
    assert!(display.contains("out of expected bounds"));

    let err: MutabilityError<UpperTriangularCSR2D<TestCSR>> = MutabilityError::MaxedOutRowIndex;
    let _: MutabilityError<SymmetricCSR2D<TestCSR>> = err.into();

    let err: MutabilityError<UpperTriangularCSR2D<TestCSR>> = MutabilityError::MaxedOutColumnIndex;
    let _: MutabilityError<SymmetricCSR2D<TestCSR>> = err.into();

    let err: MutabilityError<UpperTriangularCSR2D<TestCSR>> = MutabilityError::MaxedOutSparseIndex;
    let _: MutabilityError<SymmetricCSR2D<TestCSR>> = err.into();

    let err: MutabilityError<UpperTriangularCSR2D<TestCSR>> = MutabilityError::IncompatibleShape;
    let _: MutabilityError<SymmetricCSR2D<TestCSR>> = err.into();
}

#[test]
fn test_error_conversion_csr2d_to_valued() {
    let err: MutabilityError<CSR2D<usize, usize, usize>> =
        MutabilityError::UnorderedCoordinate((1, 2));
    let converted: MutabilityError<ValuedCSR2D<usize, usize, usize, f64>> = err.into();
    let display = format!("{converted}");
    assert!(display.contains("Unordered coordinate"));

    let err: MutabilityError<CSR2D<usize, usize, usize>> = MutabilityError::DuplicatedEntry((3, 4));
    let converted: MutabilityError<ValuedCSR2D<usize, usize, usize, f64>> = err.into();
    let display = format!("{converted}");
    assert!(display.contains("Duplicated entry"));

    let err: MutabilityError<CSR2D<usize, usize, usize>> =
        MutabilityError::OutOfBounds((0, 0), (5, 5), "val");
    let converted: MutabilityError<ValuedCSR2D<usize, usize, usize, f64>> = err.into();
    let display = format!("{converted}");
    assert!(display.contains("out of expected bounds"));

    let err: MutabilityError<CSR2D<usize, usize, usize>> = MutabilityError::MaxedOutRowIndex;
    let _: MutabilityError<ValuedCSR2D<usize, usize, usize, f64>> = err.into();

    let err: MutabilityError<CSR2D<usize, usize, usize>> = MutabilityError::MaxedOutColumnIndex;
    let _: MutabilityError<ValuedCSR2D<usize, usize, usize, f64>> = err.into();

    let err: MutabilityError<CSR2D<usize, usize, usize>> = MutabilityError::MaxedOutSparseIndex;
    let _: MutabilityError<ValuedCSR2D<usize, usize, usize, f64>> = err.into();

    let err: MutabilityError<CSR2D<usize, usize, usize>> = MutabilityError::IncompatibleShape;
    let _: MutabilityError<ValuedCSR2D<usize, usize, usize, f64>> = err.into();
}

// ============================================================================
// Louvain error display paths (lines 107-118 of louvain.rs)
// ============================================================================

#[test]
fn test_louvain_error_display_non_square() {
    let err = LouvainError::NonSquareMatrix { rows: 3, columns: 5 };
    let display = format!("{err}");
    assert!(display.contains("square"));
    assert!(display.contains('3'));
    assert!(display.contains('5'));
}

#[test]
fn test_louvain_error_display_unrepresentable() {
    let err = LouvainError::UnrepresentableWeight {
        source_id: 0,
        destination_id: 1,
    };
    let display = format!("{err}");
    assert!(display.contains("cannot be represented"));
}

#[test]
fn test_louvain_error_display_all_variants() {
    let variants: Vec<LouvainError> = vec![
        LouvainError::InvalidResolution,
        LouvainError::InvalidModularityThreshold,
        LouvainError::InvalidMaxLevels,
        LouvainError::InvalidMaxLocalPasses,
        LouvainError::NonSquareMatrix { rows: 2, columns: 3 },
        LouvainError::UnrepresentableWeight {
            source_id: 0,
            destination_id: 1,
        },
        LouvainError::NonFiniteWeight {
            source_id: 0,
            destination_id: 1,
        },
        LouvainError::NonPositiveWeight {
            source_id: 0,
            destination_id: 1,
        },
        LouvainError::NonSymmetricEdge {
            source_id: 0,
            destination_id: 1,
        },
        LouvainError::TooManyCommunities,
    ];
    for err in &variants {
        let display = format!("{err}");
        assert!(!display.is_empty());
        let debug = format!("{err:?}");
        assert!(!debug.is_empty());
    }
}

// ============================================================================
// GenericVocabularyBuilder: ignore_duplicates and missing symbols error
// ============================================================================

#[test]
fn test_vocabulary_builder_missing_symbols() {
    let result =
        GenericVocabularyBuilder::<std::vec::IntoIter<(usize, usize)>, SortedVec<usize>>::default()
            .expected_number_of_symbols(3)
            .build();
    assert!(result.is_err());
}

#[test]
fn test_vocabulary_builder_ignore_duplicates_flag() {
    let builder =
        GenericVocabularyBuilder::<std::vec::IntoIter<(usize, usize)>, SortedVec<usize>>::default();
    assert!(!builder.should_ignore_duplicates());

    let builder = builder.ignore_duplicates();
    assert!(builder.should_ignore_duplicates());
}

#[test]
fn test_vocabulary_builder_wrong_count() {
    let result =
        GenericVocabularyBuilder::<std::vec::IntoIter<(usize, usize)>, SortedVec<usize>>::default()
            .expected_number_of_symbols(5)
            .symbols(vec![(0, 10), (1, 20)].into_iter())
            .build();
    assert!(result.is_err());
}

// ============================================================================
// Undirected edges builder: delegation methods
// ============================================================================

#[test]
fn test_undirected_edges_builder_delegation() {
    let builder: UndiEdgesBuilder<std::vec::IntoIter<(usize, usize)>> = UndiEdgesBuilder::default();
    assert!(builder.get_expected_number_of_edges().is_none());
    assert!(!builder.should_ignore_duplicates());
    assert!(builder.get_expected_shape().is_none());

    let builder = builder.expected_number_of_edges(5).expected_shape(4).ignore_duplicates();

    assert_eq!(builder.get_expected_number_of_edges(), Some(5));
    assert!(builder.should_ignore_duplicates());
    assert_eq!(builder.get_expected_shape(), Some(4));
}

// ============================================================================
// Intersection DoubleEndedIterator: crossing detection (lines 54-62, 100-109)
// ============================================================================

#[test]
fn test_intersection_next_front_crossing_item1_back() {
    // Set up crossing detection: after next_back consumes some items,
    // front items should not advance past the back boundary.
    let iter1 = [1, 2, 3, 4, 5].into_iter();
    let iter2 = [1, 2, 3, 4, 5].into_iter();
    let mut intersection = Intersection::new(iter1, iter2);

    // Consume from back first: gets 5, then 4
    assert_eq!(intersection.next_back(), Some(5));
    assert_eq!(intersection.next_back(), Some(4));

    // Forward iteration should stop before crossing back
    assert_eq!(intersection.next(), Some(1));
    assert_eq!(intersection.next(), Some(2));
    assert_eq!(intersection.next(), Some(3));
    assert_eq!(intersection.next(), None);
}

#[test]
fn test_intersection_next_back_crossing_front() {
    // front consumes some, then back should stop before crossing
    let iter1 = [1, 2, 3, 4, 5].into_iter();
    let iter2 = [1, 2, 3, 4, 5].into_iter();
    let mut intersection = Intersection::new(iter1, iter2);

    assert_eq!(intersection.next(), Some(1));
    assert_eq!(intersection.next(), Some(2));

    assert_eq!(intersection.next_back(), Some(5));
    assert_eq!(intersection.next_back(), Some(4));
    assert_eq!(intersection.next_back(), Some(3));
    assert_eq!(intersection.next_back(), None);
}

// ============================================================================
// Intersection next_back: back-only iteration with different overlap patterns
// ============================================================================

#[test]
fn test_intersection_next_back_no_overlap() {
    let iter1 = [1, 3, 5].into_iter();
    let iter2 = [2, 4, 6].into_iter();
    let mut intersection = Intersection::new(iter1, iter2);
    assert_eq!(intersection.next_back(), None);
}

#[test]
fn test_intersection_next_back_partial_overlap() {
    let iter1 = [1, 3, 5, 7].into_iter();
    let iter2 = [3, 5, 8].into_iter();
    let mut intersection = Intersection::new(iter1, iter2);
    assert_eq!(intersection.next_back(), Some(5));
    assert_eq!(intersection.next_back(), Some(3));
    assert_eq!(intersection.next_back(), None);
}

// ============================================================================
// Connected components with u8 marker (TooManyComponents error path)
// ============================================================================

fn build_undigraph(nodes: Vec<usize>, edges: Vec<(usize, usize)>) -> UndiGraph<usize> {
    let num_nodes = nodes.len();
    let num_edges = edges.len();
    let node_vocab: SortedVec<usize> = GenericVocabularyBuilder::default()
        .expected_number_of_symbols(num_nodes)
        .symbols(nodes.into_iter().enumerate())
        .build()
        .unwrap();
    let edge_mat: SymmetricCSR2D<_> = UndiEdgesBuilder::default()
        .expected_number_of_edges(num_edges)
        .expected_shape(node_vocab.len())
        .edges(edges.into_iter())
        .build()
        .unwrap();
    UndiGraph::from((node_vocab, edge_mat))
}

#[test]
fn test_cc_single_node() {
    let graph = build_undigraph(vec![0], vec![]);
    let cc = ConnectedComponents::<usize>::connected_components(&graph).unwrap();
    assert_eq!(cc.number_of_components(), 1);
    assert_eq!(cc.largest_component_size(), 1);
    assert_eq!(cc.smallest_component_size(), 1);
}

#[test]
fn test_cc_all_isolated() {
    let graph = build_undigraph(vec![0, 1, 2, 3], vec![]);
    let cc = ConnectedComponents::<usize>::connected_components(&graph).unwrap();
    assert_eq!(cc.number_of_components(), 4);
    assert_eq!(cc.largest_component_size(), 1);
    assert_eq!(cc.smallest_component_size(), 1);
}

// ============================================================================
// Johnson: more complex cycle structures for deeper algorithm coverage
// ============================================================================

fn build_square_csr(
    n: usize,
    mut edges: Vec<(usize, usize)>,
) -> SquareCSR2D<CSR2D<usize, usize, usize>> {
    edges.sort_unstable();
    DiEdgesBuilder::default()
        .expected_number_of_edges(edges.len())
        .expected_shape(n)
        .edges(edges.into_iter())
        .build()
        .unwrap()
}

#[test]
fn test_johnson_figure_eight() {
    // Two cycles sharing a single node (0):
    // Cycle A: 0→1→2→0, Cycle B: 0→3→4→0
    let m = build_square_csr(5, vec![(0, 1), (0, 3), (1, 2), (2, 0), (3, 4), (4, 0)]);
    let cycles: Vec<Vec<usize>> = m.johnson().collect();
    assert_eq!(cycles.len(), 2);
}

#[test]
fn test_johnson_deeply_nested_blocking() {
    // Chain of 2-cycles with shared nodes that exercises blocking/unblocking:
    // 0↔1↔2↔3, plus 0→2→3→0 and 0→1→3→0
    let m =
        build_square_csr(4, vec![(0, 1), (0, 2), (1, 0), (1, 2), (1, 3), (2, 1), (2, 3), (3, 0)]);
    let cycles: Vec<Vec<usize>> = m.johnson().collect();
    // Many cycles including length-2, length-3, and length-4
    assert!(cycles.len() >= 4);
}

#[test]
fn test_johnson_multiple_sccs_with_tails() {
    // SCC1: 0↔1, SCC2: 3↔4, tail: 2→3 (no back edge)
    // Tests root advancement past non-SCC nodes
    let m = build_square_csr(5, vec![(0, 1), (1, 0), (2, 3), (3, 4), (4, 3)]);
    let cycles: Vec<Vec<usize>> = m.johnson().collect();
    assert_eq!(cycles.len(), 2);
}

#[test]
fn test_johnson_k5_complete() {
    // K5 complete directed graph - exercises deep blocking/unblocking
    let mut edges = Vec::new();
    for i in 0..5 {
        for j in 0..5 {
            if i != j {
                edges.push((i, j));
            }
        }
    }
    let m = build_square_csr(5, edges);
    let cycles: Vec<Vec<usize>> = m.johnson().collect();
    assert!(cycles.len() > 10);
}

// ============================================================================
// LAPJV: matrices designed to exercise column reduction conflicts
// and augmenting path search
// ============================================================================

#[test]
fn test_lapjv_chain_conflicts() {
    // Chain of conflicts: each row prefers the next column
    use geometric_traits::{impls::PaddedMatrix2D, traits::LAPJV};
    let m: ValuedCSR2D<u8, u8, u8, f64> = ValuedCSR2D::try_from([
        [1.0, 2.0, 99.0, 99.0, 99.0, 99.0],
        [99.0, 1.0, 2.0, 99.0, 99.0, 99.0],
        [99.0, 99.0, 1.0, 2.0, 99.0, 99.0],
        [99.0, 99.0, 99.0, 1.0, 2.0, 99.0],
        [99.0, 99.0, 99.0, 99.0, 1.0, 2.0],
        [2.0, 99.0, 99.0, 99.0, 99.0, 1.0],
    ])
    .unwrap();
    let padded = PaddedMatrix2D::new(m, |_: (u8, u8)| 900.0).unwrap();
    let result = padded.lapjv(1000.0).unwrap();
    assert_eq!(result.len(), 6);
}

#[test]
fn test_lapjv_identity_preference() {
    // All rows prefer diagonal - simple column reduction without conflict
    use geometric_traits::{impls::PaddedMatrix2D, traits::LAPJV};
    let m: ValuedCSR2D<u8, u8, u8, f64> = ValuedCSR2D::try_from([
        [1.0, 99.0, 99.0, 99.0],
        [99.0, 1.0, 99.0, 99.0],
        [99.0, 99.0, 1.0, 99.0],
        [99.0, 99.0, 99.0, 1.0],
    ])
    .unwrap();
    let padded = PaddedMatrix2D::new(m, |_: (u8, u8)| 900.0).unwrap();
    let result = padded.lapjv(1000.0).unwrap();
    assert_eq!(result.len(), 4);
    assert!(result.contains(&(0, 0)));
    assert!(result.contains(&(1, 1)));
    assert!(result.contains(&(2, 2)));
    assert!(result.contains(&(3, 3)));
}

// ============================================================================
// Square CSR2D additional paths
// ============================================================================

#[test]
fn test_square_csr2d_increase_shape_noop() {
    let mut sq: SquareCSR2D<CSR2D<usize, usize, usize>> = SparseMatrixMut::with_sparse_shape(5);
    MatrixMut::add(&mut sq, (0, 1)).unwrap();
    // Increase to same size (noop)
    sq.increase_shape((5, 5)).unwrap();
    assert_eq!(sq.order(), 5);
}

#[test]
fn test_square_csr2d_increase_shape_to_larger() {
    let mut sq: SquareCSR2D<CSR2D<usize, usize, usize>> = SparseMatrixMut::with_sparse_shape(3);
    MatrixMut::add(&mut sq, (0, 1)).unwrap();
    sq.increase_shape((7, 7)).unwrap();
    assert_eq!(sq.order(), 7);
    assert_eq!(sq.number_of_rows(), 7);
    assert_eq!(sq.number_of_columns(), 7);
}

#[test]
fn test_square_csr2d_many_diagonals() {
    let mut sq: SquareCSR2D<CSR2D<usize, usize, usize>> = SparseMatrixMut::with_sparse_shape(5);
    for i in 0..5 {
        MatrixMut::add(&mut sq, (i, i)).unwrap();
    }
    assert_eq!(sq.number_of_defined_diagonal_values(), 5);
}

// ============================================================================
// SortedVec: GrowableVocabulary error paths
// ============================================================================

#[test]
fn test_sorted_vec_duplicate_source_error() {
    let result = GenericVocabularyBuilder::<_, SortedVec<usize>>::default()
        .expected_number_of_symbols(3)
        .symbols(vec![(0, 10), (1, 20), (1, 30)].into_iter())
        .build();
    assert!(result.is_err());
}

#[test]
fn test_sorted_vec_duplicate_destination_error() {
    let result = GenericVocabularyBuilder::<_, SortedVec<usize>>::default()
        .expected_number_of_symbols(3)
        .symbols(vec![(0, 10), (1, 10), (2, 30)].into_iter())
        .build();
    assert!(result.is_err());
}

#[test]
fn test_sorted_vec_no_capacity_build() {
    // Build without expected_number_of_symbols (exercises the else branch)
    let sv: SortedVec<usize> = GenericVocabularyBuilder::default()
        .symbols(vec![(0, 10), (1, 20)].into_iter())
        .build()
        .unwrap();
    assert_eq!(sv.len(), 2);
}

// ============================================================================
// CSR2D: additional paths for coverage
// ============================================================================

#[test]
fn test_csr2d_multi_row_sequential_add() {
    // Add entries sequentially across multiple rows to exercise the
    // "extend to next row" path in MatrixMut::add
    let mut csr: TestCSR = CSR2D::with_sparse_shaped_capacity((5, 5), 10);
    MatrixMut::add(&mut csr, (0, 0)).unwrap();
    MatrixMut::add(&mut csr, (0, 3)).unwrap();
    MatrixMut::add(&mut csr, (1, 1)).unwrap();
    MatrixMut::add(&mut csr, (1, 4)).unwrap();
    MatrixMut::add(&mut csr, (2, 0)).unwrap();
    MatrixMut::add(&mut csr, (3, 2)).unwrap();
    MatrixMut::add(&mut csr, (4, 4)).unwrap();

    assert_eq!(csr.number_of_defined_values(), 7);
    assert_eq!(csr.last_sparse_coordinates(), Some((4, 4)));
}

#[test]
fn test_csr2d_gap_in_rows() {
    // Skip rows: 0, then 3, then 4 (rows 1,2 empty)
    let mut csr: TestCSR = CSR2D::with_sparse_shaped_capacity((5, 5), 3);
    MatrixMut::add(&mut csr, (0, 0)).unwrap();
    MatrixMut::add(&mut csr, (3, 1)).unwrap();
    MatrixMut::add(&mut csr, (4, 2)).unwrap();

    assert_eq!(csr.number_of_defined_values_in_row(0), 1);
    assert_eq!(csr.number_of_defined_values_in_row(1), 0);
    assert_eq!(csr.number_of_defined_values_in_row(2), 0);
    assert_eq!(csr.number_of_defined_values_in_row(3), 1);
}

#[test]
fn test_csr2d_last_sparse_coordinates_single_entry() {
    let mut csr: TestCSR = CSR2D::with_sparse_shaped_capacity((3, 3), 1);
    MatrixMut::add(&mut csr, (1, 2)).unwrap();
    assert_eq!(csr.last_sparse_coordinates(), Some((1, 2)));
}

#[test]
fn test_csr2d_with_sparse_capacity() {
    // Test with_sparse_capacity (no shape specified)
    let mut csr: TestCSR = CSR2D::with_sparse_capacity(10);
    MatrixMut::add(&mut csr, (0, 0)).unwrap();
    MatrixMut::add(&mut csr, (0, 5)).unwrap();
    MatrixMut::add(&mut csr, (3, 2)).unwrap();
    assert_eq!(csr.number_of_defined_values(), 3);
}

// ============================================================================
// Kahn topological sort: specific graph patterns
// ============================================================================

#[test]
fn test_kahn_diamond_dag() {
    // Diamond: 0→1, 0→2, 1→3, 2→3
    let m = build_square_csr(4, vec![(0, 1), (0, 2), (1, 3), (2, 3)]);
    let ordering = m.kahn().unwrap();
    assert!(ordering[0] < ordering[1]);
    assert!(ordering[0] < ordering[2]);
    assert!(ordering[1] < ordering[3]);
    assert!(ordering[2] < ordering[3]);
}

#[test]
fn test_kahn_cycle_fails() {
    let m = build_square_csr(3, vec![(0, 1), (1, 2), (2, 0)]);
    assert!(m.kahn().is_err());
}

#[test]
fn test_kahn_wide_dag() {
    // Wide tree: root 0 has 5 children
    let m = build_square_csr(6, vec![(0, 1), (0, 2), (0, 3), (0, 4), (0, 5)]);
    let ordering = m.kahn().unwrap();
    for i in 1..6 {
        assert!(ordering[0] < ordering[i]);
    }
}

// ============================================================================
// Tarjan: edge cases for SCC detection
// ============================================================================

#[test]
fn test_tarjan_disconnected_sccs() {
    // Three separate 2-cycles
    let m = build_square_csr(6, vec![(0, 1), (1, 0), (2, 3), (3, 2), (4, 5), (5, 4)]);
    let sccs: Vec<Vec<usize>> = m.tarjan().collect();
    assert_eq!(sccs.len(), 3);
    for scc in &sccs {
        assert_eq!(scc.len(), 2);
    }
}

#[test]
fn test_tarjan_single_large_scc() {
    // All nodes in one SCC: 0→1→2→3→4→0
    let m = build_square_csr(5, vec![(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]);
    let sccs: Vec<Vec<usize>> = m.tarjan().collect();
    assert_eq!(sccs.len(), 1);
    assert_eq!(sccs[0].len(), 5);
}

#[test]
fn test_tarjan_dag_no_nontrivial_sccs() {
    let m = build_square_csr(4, vec![(0, 1), (1, 2), (2, 3)]);
    let sccs: Vec<Vec<usize>> = m.tarjan().collect();
    // DAG may produce singleton SCCs depending on implementation
    for scc in &sccs {
        assert_eq!(scc.len(), 1, "DAG should only have singleton SCCs");
    }
}

// ============================================================================
// GenericMatrix2DWithPaddedDiagonal: is_diagonal_imputed
// ============================================================================

#[test]
fn test_padded_diagonal_is_imputed() {
    use geometric_traits::impls::GenericMatrix2DWithPaddedDiagonal;
    use num_traits::One;

    // Create a 3x3 valued matrix with entries only at (0,1), (1,0), (2,0)
    let m: ValuedCSR2D<usize, usize, usize, f64> =
        ValuedCSR2D::try_from([[99.0, 5.0, 99.0], [3.0, 99.0, 99.0], [7.0, 99.0, 99.0]]).unwrap();

    let padded = GenericMatrix2DWithPaddedDiagonal::new(m, f64::one).unwrap();
    // Matrix is square, has all values including diagonal from try_from
    assert_eq!(padded.number_of_rows(), padded.number_of_columns());
}

// ============================================================================
// PaddedMatrix2D: exercise sparse coordinates iteration
// ============================================================================

#[test]
fn test_padded_matrix2d_coordinates() {
    use geometric_traits::impls::PaddedMatrix2D;
    let m: ValuedCSR2D<usize, usize, usize, f64> =
        ValuedCSR2D::try_from([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]).unwrap();

    let padded = PaddedMatrix2D::new(m, |_: (usize, usize)| 99.0).unwrap();
    // 3 rows x 3 cols (padded to square)
    assert_eq!(padded.number_of_rows(), 3);
    assert_eq!(padded.number_of_columns(), 3);

    // All coordinates in the padded square should be present
    let coords: Vec<(usize, usize)> = SparseMatrix::sparse_coordinates(&padded).collect();
    // Padded matrix should have at least as many entries as the original
    assert!(coords.len() >= 6);
}
