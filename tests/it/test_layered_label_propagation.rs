//! Tests for the Layered Label Propagation node-ordering algorithm.
#![cfg(feature = "std")]

use std::collections::BTreeSet;

use geometric_traits::{
    impls::{BitSquareMatrix, CSR2D, SymmetricCSR2D},
    prelude::*,
    traits::{
        EdgesBuilder, LayeredLabelPropagationError, UndirectedMonopartiteMonoplexGraph,
        algorithms::ModularProduct,
    },
};
use num_traits::AsPrimitive;

type UndirectedGraph = SymmetricCSR2D<CSR2D<usize, usize, usize>>;
type UndirectedVecGraph = GenericGraph<Vec<usize>, UndirectedGraph>;

fn build_undirected_graph(num_nodes: usize, edges: &[(usize, usize)]) -> UndirectedVecGraph {
    let mut edges: Vec<_> = edges
        .iter()
        .copied()
        .map(|(left, right)| if left <= right { (left, right) } else { (right, left) })
        .collect();
    edges.sort_unstable();
    let matrix: UndirectedGraph = UndiEdgesBuilder::default()
        .expected_number_of_edges(edges.len())
        .expected_shape(num_nodes)
        .edges(edges.into_iter())
        .build()
        .unwrap();
    GenericGraph::from(((0..num_nodes).collect::<Vec<_>>(), matrix))
}

fn build_two_triangles_with_bridge() -> UndirectedVecGraph {
    build_undirected_graph(6, &[(0, 1), (0, 2), (1, 2), (2, 3), (3, 4), (3, 5), (4, 5)])
}

fn build_undirected_graph_with_self_loop() -> UndirectedVecGraph {
    build_undirected_graph(3, &[(0, 0), (0, 1)])
}

fn build_branching_tree() -> UndirectedVecGraph {
    build_undirected_graph(6, &[(0, 1), (0, 2), (1, 3), (1, 4), (2, 5)])
}

fn build_degree_biased_branching() -> UndirectedVecGraph {
    build_undirected_graph(6, &[(0, 1), (0, 2), (1, 3), (2, 4), (2, 5)])
}

fn build_star_graph_5() -> UndirectedVecGraph {
    build_undirected_graph(5, &[(0, 1), (0, 2), (0, 3), (0, 4)])
}

fn build_cycle_graph_5() -> UndirectedVecGraph {
    build_undirected_graph(5, &[(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)])
}

fn assert_is_permutation(order: &[usize], n: usize) {
    assert_eq!(order.len(), n, "ordering must contain exactly one entry per node");
    let mut seen = vec![false; n];
    for &node in order {
        assert!(node < n, "ordering contains out-of-range node {node}");
        assert!(!seen[node], "ordering contains duplicate node {node}");
        seen[node] = true;
    }
}

fn positions(order: &[usize]) -> Vec<usize> {
    let mut positions = vec![0usize; order.len()];
    for (index, &node) in order.iter().enumerate() {
        positions[node] = index;
    }
    positions
}

fn assert_block_is_contiguous(order: &[usize], block: &[usize]) {
    let positions = positions(order);
    let mut block_positions: Vec<usize> = block.iter().map(|&node| positions[node]).collect();
    block_positions.sort_unstable();
    for window in block_positions.windows(2) {
        assert_eq!(window[0] + 1, window[1], "block {block:?} is not contiguous in {order:?}");
    }
}

fn collect_symbolized_edges<G>(graph: &G) -> BTreeSet<((usize, usize), (usize, usize))>
where
    G: UndirectedMonopartiteMonoplexGraph<NodeSymbol = (usize, usize)>,
{
    let symbols: Vec<_> = graph.nodes().collect();
    let mut edges = BTreeSet::new();
    for node in graph.node_ids() {
        for neighbor in graph.neighbors(node) {
            if node < neighbor {
                let left = symbols[node.as_()];
                let right = symbols[neighbor.as_()];
                let edge = if left <= right { (left, right) } else { (right, left) };
                edges.insert(edge);
            }
        }
    }
    edges
}

#[test]
fn test_llp_returns_a_permutation_and_keeps_communities_contiguous() {
    let graph = build_two_triangles_with_bridge();
    let sorter = LayeredLabelPropagationSorter::new(
        LAYERED_LABEL_PROPAGATION_DEFAULT_GAMMAS.to_vec(),
        100,
        7,
        None,
        false,
    )
    .unwrap();

    let order = sorter.sort_nodes(&graph);

    assert_is_permutation(&order, graph.number_of_nodes());
    assert_block_is_contiguous(&order, &[0, 1, 2]);
    assert_block_is_contiguous(&order, &[3, 4, 5]);
}

#[test]
fn test_llp_is_deterministic_for_a_fixed_seed() {
    let graph = build_two_triangles_with_bridge();
    let sorter = LayeredLabelPropagationSorter::new(
        LAYERED_LABEL_PROPAGATION_DEFAULT_GAMMAS.to_vec(),
        100,
        19,
        None,
        false,
    )
    .unwrap();

    let first = sorter.sort_nodes(&graph);
    let second = sorter.sort_nodes(&graph);

    assert_eq!(first, second);
}

#[test]
fn test_llp_matches_law_oracles_on_small_graphs() {
    // These expected orders were generated with LAW 2.7.1
    // `it.unimi.dsi.law.graph.LayeredLabelPropagation` using:
    // - `DEFAULT_GAMMAS`
    // - `MAX_UPDATES`
    // - `seed = 7`
    // - `numberOfThreads = 1`
    //
    // LAW returns an inverse permutation (old node -> new position), whereas
    // this crate's `NodeSorter` returns the order form (new position -> old
    // node), so the LAW result was inverted before being recorded here.
    let cases = [
        ("branching_tree", build_branching_tree(), vec![1, 3, 4, 0, 2, 5]),
        ("degree_biased_branching", build_degree_biased_branching(), vec![1, 3, 0, 2, 4, 5]),
        ("star_5", build_star_graph_5(), vec![0, 3, 1, 2, 4]),
        ("cycle_5", build_cycle_graph_5(), vec![4, 3, 0, 1, 2]),
    ];

    let sorter = LayeredLabelPropagationSorter::new(
        LAYERED_LABEL_PROPAGATION_DEFAULT_GAMMAS.to_vec(),
        100,
        7,
        None,
        false,
    )
    .unwrap();

    for (name, graph, expected) in cases {
        assert_eq!(sorter.sort_nodes(&graph), expected, "LLP order for `{name}` diverged from LAW");
    }
}

#[test]
fn test_llp_rejects_negative_gamma() {
    assert_eq!(
        LayeredLabelPropagationSorter::new(vec![1.0, -0.5], 100, 7, None, false),
        Err(LayeredLabelPropagationError::InvalidGamma)
    );
}

#[test]
fn test_llp_rejects_nan_gamma() {
    assert_eq!(
        LayeredLabelPropagationSorter::new(vec![f64::NAN], 100, 7, None, false),
        Err(LayeredLabelPropagationError::InvalidGamma)
    );
}

#[test]
fn test_llp_rejects_zero_max_updates() {
    assert_eq!(
        LayeredLabelPropagationSorter::new(vec![1.0], 0, 7, None, false),
        Err(LayeredLabelPropagationError::InvalidMaxUpdates)
    );
}

#[test]
fn test_llp_exact_start_order_matches_running_on_the_reordered_graph() {
    let graph = build_two_triangles_with_bridge();
    let start_order = vec![3, 4, 5, 0, 1, 2];

    let exact_sorter = LayeredLabelPropagationSorter::new(
        LAYERED_LABEL_PROPAGATION_DEFAULT_GAMMAS.to_vec(),
        100,
        11,
        Some(start_order.clone()),
        true,
    )
    .unwrap();
    let direct = exact_sorter.sort_nodes(&graph);

    let reordered = apply_node_order_to_graph(&graph, &start_order);
    let plain = LayeredLabelPropagationSorter::new(
        LAYERED_LABEL_PROPAGATION_DEFAULT_GAMMAS.to_vec(),
        100,
        11,
        None,
        false,
    )
    .unwrap()
    .sort_nodes(&reordered);
    let mapped_back: Vec<usize> = plain.into_iter().map(|node| start_order[node]).collect();

    assert_eq!(direct, mapped_back);
}

#[test]
fn test_llp_order_can_be_applied_to_a_modular_product_graph_without_changing_structure() {
    let left = BitSquareMatrix::from_symmetric_edges(3, [(0, 1), (1, 2)]);
    let right = BitSquareMatrix::from_symmetric_edges(3, [(0, 1), (1, 2)]);
    let graph = left.modular_product_filtered(&right, |_, _| true).into_graph();
    let sorter = LayeredLabelPropagationSorter::new(
        LAYERED_LABEL_PROPAGATION_DEFAULT_GAMMAS.to_vec(),
        100,
        23,
        None,
        false,
    )
    .unwrap();

    let order = sorter.sort_nodes(&graph);
    let reordered = apply_node_order_to_graph(&graph, &order);

    assert_is_permutation(&order, graph.number_of_nodes());
    assert_eq!(graph.nodes().collect::<BTreeSet<_>>(), reordered.nodes().collect::<BTreeSet<_>>());
    assert_eq!(collect_symbolized_edges(&graph), collect_symbolized_edges(&reordered));
}

#[test]
#[should_panic(expected = "LayeredLabelPropagationSorter requires a loopless graph")]
fn test_llp_rejects_graphs_with_self_loops() {
    let graph = build_undirected_graph_with_self_loop();
    let _ = LayeredLabelPropagationSorter::default().sort_nodes(&graph);
}
