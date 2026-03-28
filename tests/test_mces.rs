//! Integration tests for the MCES builder.
#![cfg(feature = "std")]

use geometric_traits::{
    impls::{CSR2D, EdgeContexts, SortedVec, SquareCSR2D, SymmetricCSR2D, ValuedCSR2D},
    prelude::*,
    traits::{
        MatrixMut, SparseMatrixMut, SquareMatrix, TypedNode, VocabularyBuilder,
        algorithms::randomized_graphs::*,
    },
};

type UndirectedGraph = SymmetricCSR2D<CSR2D<usize, usize, usize>>;

fn wrap_undi(g: UndirectedGraph) -> UndiGraph<usize> {
    let n = g.order();
    let nodes: SortedVec<usize> = GenericVocabularyBuilder::default()
        .expected_number_of_symbols(n)
        .symbols((0..n).enumerate())
        .build()
        .unwrap();
    UndiGraph::from((nodes, g))
}

// ===========================================================================
// Basic MCES
// ===========================================================================

#[test]
fn test_mces_identical_triangles() {
    let g1 = wrap_undi(complete_graph(3));
    let g2 = wrap_undi(complete_graph(3));

    let result = McesBuilder::new(&g1, &g2).compute_unlabeled();

    // Identical K3: MCES should match all 3 edges.
    assert_eq!(result.matched_edges().len(), 3);
    assert_eq!(result.vertex_matches().len(), 3);
    assert_eq!(result.fragment_count(), 1);

    // Johnson similarity for identical graphs should be 1.0.
    let j = result.johnson_similarity();
    assert!((j - 1.0).abs() < 1e-10, "Johnson similarity for identical K3 should be 1.0, got {j}");
}

#[test]
fn test_mces_identical_k4() {
    // K4 vs K4: previously panicked due to symmetric vertex mapping conflicts.
    let g1 = wrap_undi(complete_graph(4));
    let g2 = wrap_undi(complete_graph(4));

    let result = McesBuilder::new(&g1, &g2).compute_unlabeled();

    // K4 has 6 edges. MCES should match all 6.
    assert_eq!(result.matched_edges().len(), 6);

    // Johnson similarity for identical graphs should be 1.0.
    let j = result.johnson_similarity();
    assert!((j - 1.0).abs() < 1e-10, "Johnson similarity for identical K4 should be 1.0, got {j}");
}

#[test]
fn test_mces_k4_vs_k3() {
    // K4 vs K3: MCES = 3 edges (the K3 subgraph).
    let g1 = wrap_undi(complete_graph(4));
    let g2 = wrap_undi(complete_graph(3));

    let result = McesBuilder::new(&g1, &g2).compute_unlabeled();

    assert_eq!(result.matched_edges().len(), 3);
    assert!(result.johnson_similarity() > 0.0);
    assert!(result.johnson_similarity() < 1.0);
}

#[test]
fn test_mces_path_vs_path() {
    let g1 = wrap_undi(path_graph(4)); // 0-1-2-3, 3 edges
    let g2 = wrap_undi(path_graph(3)); // 0-1-2, 2 edges

    let result = McesBuilder::new(&g1, &g2).compute_unlabeled();

    // P4 vs P3: MCES should match 2 edges (the shared path).
    assert_eq!(result.matched_edges().len(), 2);
    assert!(result.vertex_matches().len() >= 2);
    assert_eq!(result.fragment_count(), 1);
}

#[test]
fn test_mces_disjoint_graphs() {
    // K3 vs P4 with no structural overlap? Actually they do share edges.
    // Use a star vs a cycle to get a small MCES.
    let g1 = wrap_undi(complete_graph(3)); // triangle
    let g2 = wrap_undi(path_graph(2)); // single edge

    let result = McesBuilder::new(&g1, &g2).compute_unlabeled();

    // P2 has 1 edge. K3 has 3 edges. MCES = 1 edge.
    assert_eq!(result.matched_edges().len(), 1);
    assert!(result.johnson_similarity() > 0.0);
    assert!(result.johnson_similarity() < 1.0);
}

#[test]
fn test_mces_empty_graph() {
    let g1 = wrap_undi(path_graph(1)); // single vertex, no edges
    let g2 = wrap_undi(path_graph(1));

    let result = McesBuilder::new(&g1, &g2).compute_unlabeled();

    // No edges → no MCES.
    assert_eq!(result.matched_edges().len(), 0);
}

// ===========================================================================
// Partition
// ===========================================================================

#[test]
fn test_mces_with_partition() {
    // Use asymmetric graphs to avoid vertex mapping conflicts.
    let g1 = wrap_undi(path_graph(4)); // 3 edges
    let g2 = wrap_undi(path_graph(4)); // 3 edges

    let result = McesBuilder::new(&g1, &g2).with_partition(true).compute_unlabeled();

    // P4 vs P4: MCES should match all 3 edges.
    assert_eq!(result.matched_edges().len(), 3);
}

#[test]
fn test_mces_without_partition() {
    let g1 = wrap_undi(complete_graph(3));
    let g2 = wrap_undi(complete_graph(3));

    let with = McesBuilder::new(&g1, &g2).with_partition(true).compute_unlabeled();
    let without = McesBuilder::new(&g1, &g2).with_partition(false).compute_unlabeled();

    // Same result regardless of partition for identical graphs.
    assert_eq!(with.matched_edges().len(), without.matched_edges().len());
}

// ===========================================================================
// Custom pair filter
// ===========================================================================

#[test]
fn test_mces_custom_pair_filter() {
    let g1 = wrap_undi(complete_graph(3)); // 3 edges
    let g2 = wrap_undi(complete_graph(3)); // 3 edges

    // Only allow pairs where both indices are 0 → at most 1 vertex pair.
    let result =
        McesBuilder::new(&g1, &g2).with_pair_filter(|i, j| i == 0 && j == 0).compute_unlabeled();

    // Only 1 vertex pair → max clique size 1 → 1 matched edge.
    assert_eq!(result.matched_edges().len(), 1);
}

// ===========================================================================
// GraphSimilarities
// ===========================================================================

#[test]
fn test_mces_result_similarities() {
    // Use asymmetric graphs to avoid vertex mapping conflicts from symmetry.
    let g1 = wrap_undi(path_graph(5)); // 0-1-2-3-4, 4 edges
    let g2 = wrap_undi(path_graph(4)); // 0-1-2-3, 3 edges

    let result = McesBuilder::new(&g1, &g2).compute_unlabeled();

    // Verify all similarity methods are callable and return sensible values.
    let j = result.johnson_similarity();
    let t = result.edge_jaccard_similarity();
    let d = result.edge_dice_similarity();

    assert!(j >= 0.0 && j <= 1.0, "Johnson out of range: {j}");
    assert!(t >= 0.0 && t <= 1.0, "Tanimoto out of range: {t}");
    assert!(d >= 0.0 && d <= 1.0, "Dice out of range: {d}");
    assert!(j > 0.0, "Should have positive similarity");
}

// ===========================================================================
// Ranking
// ===========================================================================

#[test]
fn test_mces_all_cliques_ranked() {
    let g1 = wrap_undi(complete_graph(4));
    let g2 = wrap_undi(path_graph(4));

    let result = McesBuilder::new(&g1, &g2).compute_unlabeled();

    // Verify cliques are ranked (best first).
    let cliques = result.all_cliques();
    if cliques.len() >= 2 {
        // First clique should have ≤ fragment count of second.
        assert!(cliques[0].fragment_count() <= cliques[1].fragment_count());
    }
}

#[test]
fn test_mces_search_mode_controls_enumeration() {
    let g1 = build_colored_graph(&[Color::Red, Color::Red, Color::Red], vec![(0, 1, 1), (1, 2, 1)]);
    let g2 = build_colored_graph(
        &[Color::Red, Color::Red, Color::Red, Color::Red],
        vec![(0, 1, 1), (2, 3, 1)],
    );

    let default_result = McesBuilder::new(&g1, &g2).compute_labeled();
    let partial_enumeration = McesBuilder::new(&g1, &g2)
        .with_search_mode(McesSearchMode::PartialEnumeration)
        .compute_labeled();
    let all_best =
        McesBuilder::new(&g1, &g2).with_search_mode(McesSearchMode::AllBest).compute_labeled();

    assert_eq!(default_result.matched_edges().len(), 1);
    assert_eq!(partial_enumeration.matched_edges().len(), 1);
    assert_eq!(all_best.matched_edges().len(), 1);
    assert_eq!(default_result.all_cliques().len(), partial_enumeration.all_cliques().len());
    assert!(
        all_best.all_cliques().len() > partial_enumeration.all_cliques().len(),
        "AllBest should retain more tied maxima than PartialEnumeration on this fixture",
    );
}

#[test]
fn test_mces_builder_largest_fragment_metric_matches_explicit_ranker() {
    let g1 = wrap_undi(cycle_graph(4));
    let g2 = wrap_undi(path_graph(4));

    let via_builder = McesBuilder::new(&g1, &g2)
        .with_search_mode(McesSearchMode::AllBest)
        .with_largest_fragment_metric(LargestFragmentMetric::Atoms)
        .compute_unlabeled();
    let via_ranker = McesBuilder::new(&g1, &g2)
        .with_search_mode(McesSearchMode::AllBest)
        .with_ranker(
            FragmentCountRanker
                .then(LargestFragmentMetricRanker::new(LargestFragmentMetric::Atoms)),
        )
        .compute_unlabeled();

    assert_eq!(via_builder.matched_edges(), via_ranker.matched_edges());
    assert_eq!(via_builder.vertex_matches(), via_ranker.vertex_matches());
    assert_eq!(via_builder.fragment_count(), via_ranker.fragment_count());
    assert_eq!(via_builder.largest_fragment_size(), via_ranker.largest_fragment_size());
    assert_eq!(via_builder.all_cliques().len(), via_ranker.all_cliques().len());
    assert_eq!(
        via_builder.all_cliques()[0].largest_fragment_atom_count(),
        via_ranker.all_cliques()[0].largest_fragment_atom_count(),
    );
}

#[test]
fn test_mces_builder_product_vertex_ordering_identity_matches_default() {
    let g1 = wrap_undi(cycle_graph(4));
    let g2 = wrap_undi(path_graph(4));

    let default =
        McesBuilder::new(&g1, &g2).with_search_mode(McesSearchMode::AllBest).compute_unlabeled();
    let identity = McesBuilder::new(&g1, &g2)
        .with_search_mode(McesSearchMode::AllBest)
        .with_product_vertex_ordering(|left_lg, right_lg, _first_edge, _second_edge| {
            (left_lg, right_lg)
        })
        .compute_unlabeled();

    assert_eq!(default.matched_edges(), identity.matched_edges());
    assert_eq!(default.vertex_matches(), identity.vertex_matches());
    assert_eq!(default.fragment_count(), identity.fragment_count());
    assert_eq!(default.largest_fragment_size(), identity.largest_fragment_size());
    assert_eq!(default.all_cliques().len(), identity.all_cliques().len());
}

// ===========================================================================
// Labeled MCES
// ===========================================================================

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
enum Color {
    Red,
    Green,
    Blue,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct ColoredNode {
    id: usize,
    color: Color,
}

impl TypedNode for ColoredNode {
    type NodeType = Color;
    fn node_type(&self) -> Self::NodeType {
        self.color
    }
}

type ColoredGraph = geometric_traits::naive_structs::GenericGraph<
    SortedVec<ColoredNode>,
    SymmetricCSR2D<ValuedCSR2D<usize, usize, usize, u8>>,
>;

fn build_colored_graph(node_colors: &[Color], edges: Vec<(usize, usize, u8)>) -> ColoredGraph {
    let n = node_colors.len();
    let nodes_vec: Vec<ColoredNode> =
        node_colors.iter().enumerate().map(|(i, &c)| ColoredNode { id: i, color: c }).collect();
    let nodes: SortedVec<ColoredNode> = GenericVocabularyBuilder::default()
        .expected_number_of_symbols(n)
        .symbols(nodes_vec.into_iter().enumerate())
        .build()
        .unwrap();

    let undi = build_valued_undi_edges(n, edges);
    geometric_traits::naive_structs::GenericGraph::from((nodes, undi))
}

fn build_valued_undi_edges(
    n: usize,
    mut edges: Vec<(usize, usize, u8)>,
) -> SymmetricCSR2D<ValuedCSR2D<usize, usize, usize, u8>> {
    edges.sort_unstable_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
    edges.dedup();

    let mut all_entries = Vec::with_capacity(edges.len() * 2);
    for (src, dst, bond_type) in edges {
        all_entries.push((src, dst, bond_type));
        all_entries.push((dst, src, bond_type));
    }
    all_entries.sort_unstable_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));

    let mut valued: ValuedCSR2D<usize, usize, usize, u8> =
        SparseMatrixMut::with_sparse_shaped_capacity((n, n), all_entries.len());
    for (src, dst, bond_type) in all_entries {
        MatrixMut::add(&mut valued, (src, dst, bond_type)).unwrap();
    }

    SymmetricCSR2D::from_parts(SquareCSR2D::from_parts(valued, 0))
}

fn count_labeled_product_vertices(
    g1: &ColoredGraph,
    g2: &ColoredGraph,
    first_contexts: Option<&EdgeContexts<u8>>,
    second_contexts: Option<&EdgeContexts<u8>>,
) -> usize {
    let lg1 = g1.labeled_line_graph();
    let lg2 = g2.labeled_line_graph();

    let node_types_first: Vec<Color> = g1.nodes().map(|symbol| symbol.node_type()).collect();
    let node_types_second: Vec<Color> = g2.nodes().map(|symbol| symbol.node_type()).collect();

    let first_bond_labels: Vec<(Color, u8, Color)> = lg1
        .edge_map()
        .iter()
        .map(|&(src, dst)| {
            let left = node_types_first[src];
            let right = node_types_first[dst];
            let bond_type = geometric_traits::traits::Edges::matrix(g1.edges())
                .sparse_value_at(src, dst)
                .unwrap();
            if left <= right { (left, bond_type, right) } else { (right, bond_type, left) }
        })
        .collect();
    let second_bond_labels: Vec<(Color, u8, Color)> = lg2
        .edge_map()
        .iter()
        .map(|&(src, dst)| {
            let left = node_types_second[src];
            let right = node_types_second[dst];
            let bond_type = geometric_traits::traits::Edges::matrix(g2.edges())
                .sparse_value_at(src, dst)
                .unwrap();
            if left <= right { (left, bond_type, right) } else { (right, bond_type, left) }
        })
        .collect();

    let product = lg1.graph().labeled_modular_product_filtered(
        lg2.graph(),
        |i, j| {
            let contexts_match = match (first_contexts, second_contexts) {
                (Some(first_contexts), Some(second_contexts)) => {
                    first_contexts.compatible_with(i, second_contexts, j)
                }
                (None, None) => true,
                _ => false,
            };
            first_bond_labels[i] == second_bond_labels[j] && contexts_match
        },
        |left, right| left == right,
    );

    product.vertex_pairs().len()
}

#[test]
fn test_labeled_mces_same_colors() {
    // Two identical colored paths: R-G-B
    let g1 =
        build_colored_graph(&[Color::Red, Color::Green, Color::Blue], vec![(0, 1, 1), (1, 2, 1)]);
    let g2 =
        build_colored_graph(&[Color::Red, Color::Green, Color::Blue], vec![(0, 1, 1), (1, 2, 1)]);

    let result = McesBuilder::new(&g1, &g2).compute_labeled();

    // Identical colored paths: MCES should match both edges.
    assert_eq!(result.matched_edges().len(), 2);
    let j = result.johnson_similarity();
    assert!(
        (j - 1.0).abs() < 1e-10,
        "Johnson similarity for identical colored paths should be 1.0, got {j}"
    );
}

#[test]
fn test_labeled_mces_different_colors_reduce_match() {
    // G1: R-G-B (path). G2: R-B-G (path with swapped middle/end colors).
    // Only the B-G bond label is shared, so labeled MCES can match exactly
    // one edge while unlabeled MCES still matches the full path.
    let g1 =
        build_colored_graph(&[Color::Red, Color::Green, Color::Blue], vec![(0, 1, 1), (1, 2, 1)]);
    let g2 =
        build_colored_graph(&[Color::Red, Color::Blue, Color::Green], vec![(0, 1, 1), (1, 2, 1)]);

    let labeled_result = McesBuilder::new(&g1, &g2).compute_labeled();
    let unlabeled_result = McesBuilder::new(&g1, &g2).compute_unlabeled();

    assert_eq!(labeled_result.matched_edges().len(), 1);
    assert_eq!(unlabeled_result.matched_edges().len(), 2);
}

#[test]
fn test_labeled_mces_custom_pair_filter_is_intersected_with_bond_labels() {
    let g1 =
        build_colored_graph(&[Color::Red, Color::Green, Color::Blue], vec![(0, 1, 1), (1, 2, 1)]);
    let g2 =
        build_colored_graph(&[Color::Red, Color::Blue, Color::Green], vec![(0, 1, 1), (1, 2, 1)]);

    let result = McesBuilder::new(&g1, &g2)
        // (0,0) is caller-allowed but bond-label-incompatible: R-G != R-B.
        .with_pair_filter(|i, j| i == 0 && j == 0)
        .compute_labeled();

    assert_eq!(result.matched_edges().len(), 0);
}

#[test]
fn test_labeled_mces_different_bond_types_block_match() {
    let g1 = build_colored_graph(&[Color::Red, Color::Green], vec![(0, 1, 1)]);
    let g2 = build_colored_graph(&[Color::Red, Color::Green], vec![(0, 1, 2)]);

    let labeled_result = McesBuilder::new(&g1, &g2).compute_labeled();
    let unlabeled_result = McesBuilder::new(&g1, &g2).compute_unlabeled();

    assert_eq!(labeled_result.matched_edges().len(), 0);
    assert_eq!(unlabeled_result.matched_edges().len(), 1);
}

#[test]
fn test_labeled_mces_can_ignore_edge_values() {
    let g1 = build_colored_graph(&[Color::Red, Color::Green], vec![(0, 1, 1)]);
    let g2 = build_colored_graph(&[Color::Red, Color::Green], vec![(0, 1, 2)]);

    let default_result = McesBuilder::new(&g1, &g2).compute_labeled();
    let ignored_result = McesBuilder::new(&g1, &g2).with_ignore_edge_values(true).compute_labeled();

    assert_eq!(default_result.matched_edges().len(), 0);
    assert_eq!(ignored_result.matched_edges().len(), 1);
}

#[test]
fn test_labeled_mces_canonicalizes_endpoint_order() {
    let g1 = build_colored_graph(&[Color::Red, Color::Green], vec![(0, 1, 7)]);
    let g2 = build_colored_graph(&[Color::Green, Color::Red], vec![(0, 1, 7)]);

    let result = McesBuilder::new(&g1, &g2).compute_labeled();

    assert_eq!(result.matched_edges().len(), 1);
}

#[test]
fn test_labeled_mces_edge_contexts_allow_matching_signatures() {
    let g1 = build_colored_graph(&[Color::Red, Color::Green], vec![(0, 1, 1)]);
    let g2 = build_colored_graph(&[Color::Red, Color::Green], vec![(0, 1, 1)]);
    let first_contexts = EdgeContexts::<u8>::from_rows(vec![vec![7]]);
    let second_contexts = EdgeContexts::<u8>::from_rows(vec![vec![7]]);

    let result = McesBuilder::new(&g1, &g2)
        .with_edge_contexts(&first_contexts, &second_contexts)
        .compute_labeled();

    assert_eq!(result.matched_edges().len(), 1);
}

#[test]
fn test_labeled_mces_edge_contexts_change_product_vertex_admission() {
    let g1 = build_colored_graph(&[Color::Red, Color::Red, Color::Red], vec![(0, 1, 1), (1, 2, 1)]);
    let g2 = build_colored_graph(&[Color::Red, Color::Red, Color::Red], vec![(0, 1, 1), (1, 2, 1)]);
    let first_contexts = EdgeContexts::<u8>::from_rows(vec![vec![1], vec![2]]);
    let second_contexts = EdgeContexts::<u8>::from_rows(vec![vec![1], vec![3]]);

    let without_contexts = count_labeled_product_vertices(&g1, &g2, None, None);
    let with_contexts =
        count_labeled_product_vertices(&g1, &g2, Some(&first_contexts), Some(&second_contexts));

    assert_eq!(without_contexts, 4);
    assert_eq!(with_contexts, 1);
}

#[test]
fn test_labeled_mces_edge_contexts_reject_mixed_empty_and_non_empty_rows() {
    let g1 = build_colored_graph(&[Color::Red, Color::Green], vec![(0, 1, 1)]);
    let g2 = build_colored_graph(&[Color::Red, Color::Green], vec![(0, 1, 1)]);
    let first_contexts = EdgeContexts::<u8>::from_rows(vec![vec![7]]);
    let second_contexts = EdgeContexts::<u8>::from_rows(vec![Vec::new()]);

    let result = McesBuilder::new(&g1, &g2)
        .with_edge_contexts(&first_contexts, &second_contexts)
        .compute_labeled();

    assert_eq!(result.matched_edges().len(), 0);
}

#[test]
fn test_labeled_mces_edge_contexts_reject_disjoint_signatures() {
    let g1 = build_colored_graph(&[Color::Red, Color::Green], vec![(0, 1, 1)]);
    let g2 = build_colored_graph(&[Color::Red, Color::Green], vec![(0, 1, 1)]);
    let first_contexts = EdgeContexts::<u8>::from_rows(vec![vec![7]]);
    let second_contexts = EdgeContexts::<u8>::from_rows(vec![vec![9]]);

    let result = McesBuilder::new(&g1, &g2)
        .with_edge_contexts(&first_contexts, &second_contexts)
        .compute_labeled();

    assert_eq!(result.matched_edges().len(), 0);
}

#[test]
fn test_labeled_mces_edge_contexts_allow_shared_signatures_among_multiple_values() {
    let g1 = build_colored_graph(&[Color::Red, Color::Green], vec![(0, 1, 1)]);
    let g2 = build_colored_graph(&[Color::Red, Color::Green], vec![(0, 1, 1)]);
    let first_contexts = EdgeContexts::<u8>::from_rows(vec![vec![7, 9]]);
    let second_contexts = EdgeContexts::<u8>::from_rows(vec![vec![9, 11]]);

    let result = McesBuilder::new(&g1, &g2)
        .with_edge_contexts(&first_contexts, &second_contexts)
        .compute_labeled();

    assert_eq!(result.matched_edges().len(), 1);
}

#[test]
fn test_labeled_mces_edge_contexts_compose_with_pair_filter() {
    let g1 = build_colored_graph(&[Color::Red, Color::Green], vec![(0, 1, 1)]);
    let g2 = build_colored_graph(&[Color::Red, Color::Green], vec![(0, 1, 1)]);
    let first_contexts = EdgeContexts::<u8>::from_rows(vec![vec![7]]);
    let second_contexts = EdgeContexts::<u8>::from_rows(vec![vec![7]]);

    let result = McesBuilder::new(&g1, &g2)
        .with_edge_contexts(&first_contexts, &second_contexts)
        .with_pair_filter(|_, _| false)
        .compute_labeled();

    assert_eq!(result.matched_edges().len(), 0);
}

#[test]
fn test_labeled_mces_edge_contexts_empty_on_both_sides_preserve_existing_matches() {
    let g1 = build_colored_graph(&[Color::Red, Color::Green], vec![(0, 1, 1)]);
    let g2 = build_colored_graph(&[Color::Red, Color::Green], vec![(0, 1, 1)]);
    let first_contexts = EdgeContexts::<u8>::from_rows(vec![Vec::new()]);
    let second_contexts = EdgeContexts::<u8>::from_rows(vec![Vec::new()]);

    let result = McesBuilder::new(&g1, &g2)
        .with_edge_contexts(&first_contexts, &second_contexts)
        .compute_labeled();

    assert_eq!(result.matched_edges().len(), 1);
}

#[test]
#[should_panic(expected = "edge contexts for the first graph must have one row per original edge")]
fn test_labeled_mces_edge_contexts_validate_row_counts() {
    let g1 = build_colored_graph(&[Color::Red, Color::Green], vec![(0, 1, 1)]);
    let g2 = build_colored_graph(&[Color::Red, Color::Green], vec![(0, 1, 1)]);
    let first_contexts = EdgeContexts::<u8>::from_rows(vec![vec![7], vec![8]]);
    let second_contexts = EdgeContexts::<u8>::from_rows(vec![vec![7]]);

    let _ = McesBuilder::new(&g1, &g2)
        .with_edge_contexts(&first_contexts, &second_contexts)
        .compute_labeled();
}

// ===========================================================================
// Pre-screening
// ===========================================================================

#[test]
fn test_mces_similarity_threshold_passes() {
    // Identical graphs → similarity bound = 1.0 → any threshold < 1 passes.
    let g1 = wrap_undi(path_graph(4));
    let g2 = wrap_undi(path_graph(4));

    let result = McesBuilder::new(&g1, &g2).with_similarity_threshold(0.5).compute_unlabeled();

    assert_eq!(result.matched_edges().len(), 3);
}

#[test]
fn test_mces_similarity_threshold_rejects() {
    // Very different graphs → low similarity bound → high threshold rejects.
    let g1 = wrap_undi(complete_graph(6)); // 15 edges
    let g2 = wrap_undi(path_graph(2)); // 1 edge

    let result = McesBuilder::new(&g1, &g2).with_similarity_threshold(0.99).compute_unlabeled();

    // Should be rejected by screening → empty result.
    assert_eq!(result.matched_edges().len(), 0);
}

#[test]
fn test_mces_distance_threshold_passes() {
    // Identical graphs → distance bound = 0 → any threshold > 0 passes.
    let g1 = wrap_undi(path_graph(4));
    let g2 = wrap_undi(path_graph(4));

    let result = McesBuilder::new(&g1, &g2).with_distance_threshold(1.0).compute_unlabeled();

    assert_eq!(result.matched_edges().len(), 3);
}

#[test]
fn test_mces_distance_threshold_rejects() {
    // Very different → high distance → low threshold rejects.
    let g1 = wrap_undi(complete_graph(6)); // 15 edges
    let g2 = wrap_undi(path_graph(2)); // 1 edge

    let result = McesBuilder::new(&g1, &g2).with_distance_threshold(0.0).compute_unlabeled();

    // distance bound > 0 for different graphs → rejected.
    assert_eq!(result.matched_edges().len(), 0);
}

#[test]
fn test_mces_screening_zero_threshold_passes_all() {
    let g1 = wrap_undi(complete_graph(6));
    let g2 = wrap_undi(path_graph(2));

    // Similarity threshold 0 → everything passes.
    let result = McesBuilder::new(&g1, &g2).with_similarity_threshold(0.0).compute_unlabeled();

    assert!(result.matched_edges().len() >= 1);
}

#[test]
fn test_mces_screening_preserves_correctness() {
    // Same result with and without screening when not rejected.
    let g1 = wrap_undi(path_graph(5));
    let g2 = wrap_undi(path_graph(4));

    let without = McesBuilder::new(&g1, &g2).compute_unlabeled();
    let with = McesBuilder::new(&g1, &g2).with_similarity_threshold(0.1).compute_unlabeled();

    assert_eq!(without.matched_edges().len(), with.matched_edges().len());
}

// ===========================================================================
// Delta-Y filtering
// ===========================================================================

#[test]
fn test_mces_delta_y_k3_vs_k1_3() {
    // K3 (triangle) vs K1,3 (claw/star): Whitney's exception.
    // Their line graphs are isomorphic, so unlabeled MCES finds a 3-edge match.
    // But the degree sequences differ: K3=[2,2,2], K1,3=[1,1,1,3].
    // With Delta-Y enabled, the spurious match is discarded.
    let g1 = wrap_undi(complete_graph(3));
    let g2 = wrap_undi(star_graph(4));

    let with_dy = McesBuilder::new(&g1, &g2).with_delta_y(true).compute_unlabeled();

    let without_dy = McesBuilder::new(&g1, &g2).with_delta_y(false).compute_unlabeled();

    // Without Delta-Y: should find 3 matched edges (spurious).
    assert_eq!(without_dy.matched_edges().len(), 3);

    // With Delta-Y: the 3-edge match is discarded. There might be smaller
    // valid matches, or none.
    assert!(with_dy.matched_edges().len() < 3, "Delta-Y should discard the spurious K3/K1,3 match");
}

#[test]
fn test_mces_delta_y_partial_enumeration_matches_all_best() {
    let g1 = wrap_undi(complete_graph(3));
    let g2 = wrap_undi(star_graph(4));

    let partial_enumeration = McesBuilder::new(&g1, &g2)
        .with_search_mode(McesSearchMode::PartialEnumeration)
        .with_delta_y(true)
        .compute_unlabeled();
    let all_best = McesBuilder::new(&g1, &g2)
        .with_search_mode(McesSearchMode::AllBest)
        .with_delta_y(true)
        .compute_unlabeled();

    assert_eq!(partial_enumeration.matched_edges().len(), all_best.matched_edges().len());
}

#[test]
fn test_mces_delta_y_preserves_valid_matches() {
    // K3 vs K3: same structure, no Delta-Y. Matches should survive.
    let g1 = wrap_undi(complete_graph(3));
    let g2 = wrap_undi(complete_graph(3));

    let result = McesBuilder::new(&g1, &g2).with_delta_y(true).compute_unlabeled();

    assert_eq!(result.matched_edges().len(), 3);
}

#[test]
fn test_mces_delta_y_paths_no_false_positive() {
    // P4 vs P4: identical paths. Delta-Y should not interfere.
    let g1 = wrap_undi(path_graph(4));
    let g2 = wrap_undi(path_graph(4));

    let result = McesBuilder::new(&g1, &g2).with_delta_y(true).compute_unlabeled();

    assert_eq!(result.matched_edges().len(), 3);
}
