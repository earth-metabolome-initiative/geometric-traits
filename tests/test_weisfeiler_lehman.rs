//! Tests for Weisfeiler-Lehman refinement with optional seeds and edge labels.
#![cfg(feature = "std")]

use geometric_traits::{
    impls::{CSR2D, SortedVec, SymmetricCSR2D, ValuedCSR2D},
    naive_structs::GenericGraph,
    prelude::*,
    traits::{
        Edges, EdgesBuilder, MonoplexGraph, SparseValuedMatrix2D, VocabularyBuilder,
        WeisfeilerLehmanColoring,
    },
};

type LabeledUndirectedEdges = SymmetricCSR2D<ValuedCSR2D<usize, usize, usize, u8>>;
type LabeledUndirectedGraph = GenericGraph<SortedVec<usize>, LabeledUndirectedEdges>;

fn build_undirected_graph(number_of_nodes: usize, edges: &[(usize, usize)]) -> UndiGraph<usize> {
    let nodes: SortedVec<usize> = GenericVocabularyBuilder::default()
        .expected_number_of_symbols(number_of_nodes)
        .symbols((0..number_of_nodes).enumerate())
        .build()
        .unwrap();
    let mut canonical_edges: Vec<(usize, usize)> = edges
        .iter()
        .map(
            |&(source, destination)| {
                if source <= destination { (source, destination) } else { (destination, source) }
            },
        )
        .collect();
    canonical_edges.sort_unstable();
    canonical_edges.dedup();
    let edges: SymmetricCSR2D<CSR2D<usize, usize, usize>> = UndiEdgesBuilder::default()
        .expected_number_of_edges(canonical_edges.len())
        .expected_shape(number_of_nodes)
        .edges(canonical_edges.into_iter())
        .build()
        .unwrap();

    UndiGraph::from((nodes, edges))
}

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

fn assert_dense_classes(colors: &[usize]) {
    let mut unique = colors.to_vec();
    unique.sort_unstable();
    unique.dedup();
    assert_eq!(unique, (0..unique.len()).collect::<Vec<_>>());
}

fn assert_all_distinct(colors: &[usize]) {
    let mut unique = colors.to_vec();
    unique.sort_unstable();
    unique.dedup();
    assert_eq!(unique.len(), colors.len());
}

#[test]
fn test_wl_coloring_handles_empty_graph_across_entry_points() {
    let graph = build_undirected_graph(0, &[]);

    assert_eq!(graph.wl_coloring(), Vec::<usize>::new());
    assert_eq!(graph.wl_coloring_with_seed::<u8>(&[]), Vec::<usize>::new());
    assert_eq!(
        graph.wl_coloring_with_edge_colors::<u8, _>(|_, _| unreachable!()),
        Vec::<usize>::new()
    );
    assert_eq!(
        graph.wl_coloring_with_seed_and_edge_colors::<u8, u8, _>(&[], |_, _| unreachable!()),
        Vec::<usize>::new()
    );
}

#[test]
fn test_wl_coloring_with_seed_distinguishes_isolated_chemistry_like_invariants() {
    let graph = build_undirected_graph(3, &[]);
    let seed_colors = ["element:C", "element:N", "element:O"];

    let colors = graph.wl_coloring_with_seed(&seed_colors);

    assert_all_distinct(&colors);
    assert_dense_classes(&colors);

    let pair_cases = [
        ("aliphatic carbon vs aromatic carbon", ["syntax:organic_subset:C", "aromatic:element:C"]),
        ("isotopically labeled vs unlabeled carbon", ["isotope:13:C", "isotope:none:C"]),
        ("charged vs neutral nitrogen", ["charge:+1:N", "charge:0:N"]),
        ("wildcard vs concrete element", ["symbol:*", "symbol:C"]),
        (
            "organic-subset carbon vs bracket carbon",
            ["syntax:organic_subset:C", "syntax:bracket:C"],
        ),
        ("atom class 1 vs atom class 2", ["class:1:C", "class:2:C"]),
    ];

    for (label, pair_seed_colors) in pair_cases {
        let graph = build_undirected_graph(2, &[]);
        let colors = graph.wl_coloring_with_seed(&pair_seed_colors);

        assert_ne!(colors[0], colors[1], "{label}");
        assert_dense_classes(&colors);
    }
}

#[test]
fn test_wl_coloring_keeps_isolated_equal_atoms_tied() {
    let graph = build_undirected_graph(2, &[]);

    let colors = graph.wl_coloring();

    assert_eq!(colors, [0, 0]);
    assert_dense_classes(&colors);
}

#[test]
fn test_wl_coloring_preserves_terminal_symmetry_on_path() {
    let graph = build_undirected_graph(3, &[(0, 1), (1, 2)]);

    let colors = graph.wl_coloring();

    assert_eq!(colors[0], colors[2]);
    assert_ne!(colors[0], colors[1]);
    assert_dense_classes(&colors);
}

#[test]
fn test_wl_coloring_with_seed_separates_chemically_distinct_path_positions() {
    let graph = build_undirected_graph(4, &[(0, 1), (1, 2), (2, 3)]);
    let seed_colors = [0u8, 0, 0, 1];

    let colors = graph.wl_coloring_with_seed(&seed_colors);

    assert_ne!(colors[1], colors[2]);
    assert_ne!(colors[0], colors[3]);
    assert_dense_classes(&colors);
}

#[test]
fn test_wl_coloring_keeps_benzene_like_cycle_symmetric() {
    let graph = build_undirected_graph(6, &[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)]);
    let seed_colors = [7u8; 6];

    let colors = graph.wl_coloring_with_seed(&seed_colors);

    assert!(colors.iter().all(|&color| color == colors[0]));
    assert_dense_classes(&colors);
}

#[test]
fn test_wl_coloring_with_edge_colors_distinguishes_bond_kinds() {
    let graph = build_bidirectional_labeled_graph(3, &[(0, 1, 1), (1, 2, 2)]);
    let matrix = Edges::matrix(graph.edges());

    let topology_only = graph.wl_coloring();
    let edge_labeled = graph.wl_coloring_with_edge_colors(|source, destination| {
        matrix.sparse_value_at(source, destination).unwrap()
    });

    assert_eq!(topology_only[0], topology_only[2]);
    assert_ne!(edge_labeled[0], edge_labeled[2]);
    assert_dense_classes(&edge_labeled);
}

#[test]
fn test_wl_coloring_with_edge_colors_separates_single_and_double_bond_components() {
    let graph = build_bidirectional_labeled_graph(4, &[(0, 1, 1), (2, 3, 2)]);
    let matrix = Edges::matrix(graph.edges());

    let topology_only = graph.wl_coloring();
    let edge_labeled = graph.wl_coloring_with_edge_colors(|source, destination| {
        matrix.sparse_value_at(source, destination).unwrap()
    });

    assert_eq!(topology_only[0], topology_only[2]);
    assert_eq!(edge_labeled[0], edge_labeled[1]);
    assert_eq!(edge_labeled[2], edge_labeled[3]);
    assert_ne!(edge_labeled[0], edge_labeled[2]);
    assert_dense_classes(&edge_labeled);
}

#[test]
fn test_wl_coloring_with_edge_colors_distinguishes_aromatic_and_single_bond_rings() {
    let graph = build_bidirectional_labeled_graph(
        12,
        &[
            (0, 1, 1),
            (1, 2, 1),
            (2, 3, 1),
            (3, 4, 1),
            (4, 5, 1),
            (5, 0, 1),
            (6, 7, 2),
            (7, 8, 2),
            (8, 9, 2),
            (9, 10, 2),
            (10, 11, 2),
            (11, 6, 2),
        ],
    );
    let matrix = Edges::matrix(graph.edges());

    let topology_only = graph.wl_coloring();
    let edge_labeled = graph.wl_coloring_with_edge_colors(|source, destination| {
        matrix.sparse_value_at(source, destination).unwrap()
    });

    assert!(topology_only.iter().all(|&color| color == topology_only[0]));
    assert!(edge_labeled[..6].iter().all(|&color| color == edge_labeled[0]));
    assert!(edge_labeled[6..].iter().all(|&color| color == edge_labeled[6]));
    assert_ne!(edge_labeled[0], edge_labeled[6]);
    assert_dense_classes(&edge_labeled);
}

#[test]
fn test_wl_coloring_with_seed_and_edge_colors_combines_both_sources_of_information() {
    let graph = build_bidirectional_labeled_graph(3, &[(0, 1, 2), (1, 2, 1)]);
    let matrix = Edges::matrix(graph.edges());
    let seed_colors = [0u8, 0, 1];

    let colors = graph
        .wl_coloring_with_seed_and_edge_colors(&seed_colors, |source, destination| {
            matrix.sparse_value_at(source, destination).unwrap()
        });

    assert_ne!(colors[0], colors[1]);
    assert_ne!(colors[1], colors[2]);
    assert_ne!(colors[0], colors[2]);
    assert_dense_classes(&colors);
}

#[test]
fn test_wl_coloring_is_deterministic_and_dense() {
    let graph = build_bidirectional_labeled_graph(
        6,
        &[(0, 1, 2), (1, 2, 1), (2, 3, 1), (3, 4, 2), (4, 5, 1)],
    );
    let matrix = Edges::matrix(graph.edges());
    let seed_colors = ["seed:b", "seed:a", "seed:b", "seed:a", "seed:b", "seed:a"];

    let first = graph.wl_coloring_with_seed_and_edge_colors(&seed_colors, |source, destination| {
        matrix.sparse_value_at(source, destination).unwrap()
    });
    let second = graph
        .wl_coloring_with_seed_and_edge_colors(&seed_colors, |source, destination| {
            matrix.sparse_value_at(source, destination).unwrap()
        });

    assert_eq!(first, second);
    assert_dense_classes(&first);
}

#[test]
fn test_wl_coloring_with_seed_gives_stable_dense_classes_for_non_contiguous_equal_keys() {
    let graph = build_undirected_graph(5, &[]);
    let seed_colors = ["beta", "alpha", "beta", "alpha", "beta"];

    let colors = graph.wl_coloring_with_seed(&seed_colors);

    assert_eq!(colors, [1, 0, 1, 0, 1]);
    assert_dense_classes(&colors);
}
