//! Tests for the first search-based canonizer.
#![cfg(feature = "std")]
#![allow(clippy::pedantic)]

use geometric_traits::{
    impls::{SortedVec, SymmetricCSR2D, ValuedCSR2D},
    naive_structs::GenericGraph,
    prelude::*,
    traits::{
        CanonSplittingHeuristic, CanonicalLabelingOptions, Edges, MonoplexGraph,
        SparseValuedMatrix2D, VocabularyBuilder, canonical_label_labeled_simple_graph,
        canonical_label_labeled_simple_graph_with_options,
    },
};

type LabeledUndirectedEdges = SymmetricCSR2D<ValuedCSR2D<usize, usize, usize, u8>>;
type LabeledUndirectedGraph = GenericGraph<SortedVec<usize>, LabeledUndirectedEdges>;

fn build_bidirectional_labeled_graph(
    number_of_nodes: usize,
    edges: &[(usize, usize, u8)],
) -> LabeledUndirectedGraph {
    let nodes: SortedVec<usize> = GenericVocabularyBuilder::default()
        .expected_number_of_symbols(number_of_nodes)
        .symbols((0..number_of_nodes).enumerate())
        .build()
        .unwrap();
    let mut upper_edges: Vec<(usize, usize, u8)> = edges
        .iter()
        .map(|&(source, destination, label)| {
            if source <= destination {
                (source, destination, label)
            } else {
                (destination, source, label)
            }
        })
        .collect();
    upper_edges.sort_unstable_by(|left, right| left.0.cmp(&right.0).then(left.1.cmp(&right.1)));
    upper_edges.dedup();
    let edges: LabeledUndirectedEdges =
        SymmetricCSR2D::from_sorted_upper_triangular_entries(number_of_nodes, upper_edges).unwrap();

    GenericGraph::from((nodes, edges))
}

#[test]
fn test_canonizer_is_relabeling_invariant_for_labeled_cycle() {
    let first = build_bidirectional_labeled_graph(4, &[(0, 1, 1), (1, 2, 2), (2, 3, 1), (3, 0, 2)]);
    let first_matrix = Edges::matrix(first.edges());
    let first_result = canonical_label_labeled_simple_graph(
        &first,
        |node| [10_u8, 10, 20, 20][node],
        |left, right| first_matrix.sparse_value_at(left, right).unwrap(),
    );

    let second =
        build_bidirectional_labeled_graph(4, &[(0, 2, 1), (2, 1, 2), (1, 3, 1), (3, 0, 2)]);
    let second_matrix = Edges::matrix(second.edges());
    let second_result = canonical_label_labeled_simple_graph(
        &second,
        |node| [10_u8, 20, 10, 20][node],
        |left, right| second_matrix.sparse_value_at(left, right).unwrap(),
    );

    assert_eq!(first_result.certificate, second_result.certificate);
}

#[test]
fn test_canonizer_distinguishes_non_isomorphic_label_patterns() {
    let path = build_bidirectional_labeled_graph(4, &[(0, 1, 1), (1, 2, 1), (2, 3, 2)]);
    let path_matrix = Edges::matrix(path.edges());
    let path_result = canonical_label_labeled_simple_graph(
        &path,
        |node| [1_u8, 1, 1, 1][node],
        |left, right| path_matrix.sparse_value_at(left, right).unwrap(),
    );

    let star = build_bidirectional_labeled_graph(4, &[(0, 1, 1), (0, 2, 1), (0, 3, 2)]);
    let star_matrix = Edges::matrix(star.edges());
    let star_result = canonical_label_labeled_simple_graph(
        &star,
        |node| [1_u8, 1, 1, 1][node],
        |left, right| star_matrix.sparse_value_at(left, right).unwrap(),
    );

    assert_ne!(path_result.certificate, star_result.certificate);
}

#[test]
fn test_canonizer_returns_dense_vertex_order() {
    let graph = build_bidirectional_labeled_graph(5, &[(0, 1, 3), (1, 2, 3), (2, 3, 4), (3, 4, 3)]);
    let matrix = Edges::matrix(graph.edges());
    let result = canonical_label_labeled_simple_graph(
        &graph,
        |node| [2_u8, 1, 1, 1, 2][node],
        |left, right| matrix.sparse_value_at(left, right).unwrap(),
    );

    let mut sorted = result.order.clone();
    sorted.sort_unstable();
    assert_eq!(sorted, vec![0, 1, 2, 3, 4]);
    assert_eq!(result.certificate.vertex_labels.len(), 5);
    assert_eq!(result.certificate.upper_triangle_edge_labels.len(), 10);
    assert!(result.stats.search_nodes >= result.stats.leaf_nodes);
    assert!(result.stats.leaf_nodes >= 1);
    assert!(result.stats.pruned_sibling_orbits >= result.stats.pruned_root_orbits);
}

#[test]
fn test_canonizer_reports_automorphisms_on_symmetric_graph() {
    let graph = build_bidirectional_labeled_graph(
        8,
        &[
            (0, 4, 1),
            (0, 5, 1),
            (0, 6, 1),
            (0, 7, 1),
            (1, 4, 1),
            (1, 5, 1),
            (1, 6, 1),
            (1, 7, 1),
            (2, 4, 1),
            (2, 5, 1),
            (2, 6, 1),
            (2, 7, 1),
            (3, 4, 1),
            (3, 5, 1),
            (3, 6, 1),
            (3, 7, 1),
        ],
    );
    let matrix = Edges::matrix(graph.edges());
    let result = canonical_label_labeled_simple_graph(
        &graph,
        |node| if node < 4 { 0_u8 } else { 1_u8 },
        |left, right| matrix.sparse_value_at(left, right).unwrap(),
    );

    assert!(result.stats.search_nodes >= result.stats.leaf_nodes);
    assert!(result.stats.leaf_nodes >= 2);
    assert!(result.stats.pruned_sibling_orbits >= result.stats.pruned_root_orbits);
}

#[test]
fn test_default_canonizer_matches_explicit_bliss_fsm_heuristic() {
    let graph = build_bidirectional_labeled_graph(
        8,
        &[(0, 1, 1), (1, 2, 2), (2, 3, 1), (3, 0, 2), (0, 4, 3), (1, 5, 3), (2, 6, 3), (3, 7, 3)],
    );
    let matrix = Edges::matrix(graph.edges());
    let default_result = canonical_label_labeled_simple_graph(
        &graph,
        |node| [5_u8, 5, 6, 6, 1, 1, 1, 1][node],
        |left, right| matrix.sparse_value_at(left, right).unwrap(),
    );
    let explicit_fsm = canonical_label_labeled_simple_graph_with_options(
        &graph,
        |node| [5_u8, 5, 6, 6, 1, 1, 1, 1][node],
        |left, right| matrix.sparse_value_at(left, right).unwrap(),
        CanonicalLabelingOptions {
            splitting_heuristic: CanonSplittingHeuristic::FirstSmallestMaxNeighbours,
        },
    );

    assert_eq!(default_result.certificate, explicit_fsm.certificate);
}

#[test]
fn test_each_basic_bliss_heuristic_is_relabeling_invariant() {
    let first = build_bidirectional_labeled_graph(4, &[(0, 1, 1), (1, 2, 2), (2, 3, 1), (3, 0, 2)]);
    let first_matrix = Edges::matrix(first.edges());
    let second =
        build_bidirectional_labeled_graph(4, &[(0, 2, 1), (2, 1, 2), (1, 3, 1), (3, 0, 2)]);
    let second_matrix = Edges::matrix(second.edges());

    let heuristics = [
        CanonSplittingHeuristic::First,
        CanonSplittingHeuristic::FirstSmallest,
        CanonSplittingHeuristic::FirstLargest,
        CanonSplittingHeuristic::FirstMaxNeighbours,
        CanonSplittingHeuristic::FirstSmallestMaxNeighbours,
        CanonSplittingHeuristic::FirstLargestMaxNeighbours,
    ];

    for heuristic in heuristics {
        let first_result = canonical_label_labeled_simple_graph_with_options(
            &first,
            |node| [10_u8, 10, 20, 20][node],
            |left, right| first_matrix.sparse_value_at(left, right).unwrap(),
            CanonicalLabelingOptions { splitting_heuristic: heuristic },
        );
        let second_result = canonical_label_labeled_simple_graph_with_options(
            &second,
            |node| [10_u8, 20, 10, 20][node],
            |left, right| second_matrix.sparse_value_at(left, right).unwrap(),
            CanonicalLabelingOptions { splitting_heuristic: heuristic },
        );

        assert_eq!(
            first_result.certificate, second_result.certificate,
            "heuristic {:?} lost relabeling invariance",
            heuristic
        );
    }
}

#[test]
fn test_max_neighbour_heuristics_match_pinned_dense_uniform_n9_stats() {
    let graph = build_bidirectional_labeled_graph(
        9,
        &[
            (0, 1, 0),
            (0, 2, 0),
            (0, 3, 0),
            (0, 4, 0),
            (0, 5, 0),
            (0, 6, 0),
            (0, 7, 0),
            (0, 8, 0),
            (1, 2, 0),
            (1, 3, 0),
            (1, 4, 0),
            (1, 5, 0),
            (1, 6, 0),
            (1, 8, 0),
            (2, 3, 0),
            (2, 6, 0),
            (2, 7, 0),
            (2, 8, 0),
            (3, 6, 0),
            (3, 7, 0),
            (3, 8, 0),
            (4, 6, 0),
            (4, 7, 0),
            (5, 6, 0),
            (5, 7, 0),
            (6, 8, 0),
            (7, 8, 0),
        ],
    );
    let matrix = Edges::matrix(graph.edges());
    let heuristics = [
        CanonSplittingHeuristic::First,
        CanonSplittingHeuristic::FirstSmallest,
        CanonSplittingHeuristic::FirstMaxNeighbours,
        CanonSplittingHeuristic::FirstSmallestMaxNeighbours,
        CanonSplittingHeuristic::FirstLargestMaxNeighbours,
    ];

    for heuristic in heuristics {
        let result = canonical_label_labeled_simple_graph_with_options(
            &graph,
            |_| 0_u8,
            |left, right| matrix.sparse_value_at(left, right).unwrap(),
            CanonicalLabelingOptions { splitting_heuristic: heuristic },
        );
        assert_eq!(
            (result.stats.search_nodes, result.stats.leaf_nodes),
            (15, 6),
            "heuristic {:?} regressed on dense_uniform_n9_m27",
            heuristic
        );
    }

    let first_largest = canonical_label_labeled_simple_graph_with_options(
        &graph,
        |_| 0_u8,
        |left, right| matrix.sparse_value_at(left, right).unwrap(),
        CanonicalLabelingOptions { splitting_heuristic: CanonSplittingHeuristic::FirstLargest },
    );
    assert_eq!((first_largest.stats.search_nodes, first_largest.stats.leaf_nodes), (12, 5));
}
