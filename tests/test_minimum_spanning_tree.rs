//! Tests for the minimum-spanning-tree traits, checked against the NetworkX
//! corpus and each other. Exact-edge comparison is used only for
//! distinct-weight cases (an MST is not unique under ties); weight, edge count,
//! acyclicity, and spanning are checked everywhere.
#![cfg(feature = "std")]

#[path = "support/minimum_spanning_tree_fixture.rs"]
mod fixture;

use fixture::{
    MinimumSpanningTreeFixtureCase, WeightedMatrix, build_weighted_matrix, load_fixture_suite,
};
use geometric_traits::{
    naive_structs::GenericEdgesBuilder,
    prelude::*,
    traits::{
        EdgesBuilder,
        algorithms::minimum_spanning_tree::{Boruvka, Kruskal, MinimumSpanningTreeError, Prim},
    },
};

const FIXTURE_NAME: &str = "minimum_spanning_tree_networkx.json.gz";

/// Builds a symmetric weighted matrix from an inline undirected edge list.
fn build(node_count: usize, edges: &[(usize, usize, f64)]) -> WeightedMatrix {
    let mut directed = Vec::with_capacity(edges.len() * 2);
    for &(source, destination, weight) in edges {
        directed.push((source, destination, weight));
        if source != destination {
            directed.push((destination, source, weight));
        }
    }
    directed.sort_unstable_by(|(ls, ld, _), (rs, rd, _)| (ls, ld).cmp(&(rs, rd)));
    GenericEdgesBuilder::<_, WeightedMatrix>::default()
        .expected_number_of_edges(directed.len())
        .expected_shape((node_count, node_count))
        .edges(directed.into_iter())
        .build()
        .unwrap()
}

/// Runs all three algorithms on a matrix, returning labeled results.
fn run_all(matrix: &WeightedMatrix) -> Vec<(&'static str, WeightedForest<usize, f64>)> {
    vec![
        ("kruskal", matrix.minimum_spanning_tree_kruskal().unwrap()),
        ("prim", matrix.minimum_spanning_tree_prim().unwrap()),
        ("boruvka", matrix.minimum_spanning_tree_boruvka().unwrap()),
    ]
}

fn approx_eq(left: f64, right: f64) -> bool {
    (left - right).abs() <= 1e-6 + 1e-9 * left.abs().max(right.abs())
}

/// Minimal union-find used to validate that a result is an acyclic forest.
struct DisjointSet {
    parent: Vec<usize>,
}

impl DisjointSet {
    fn new(size: usize) -> Self {
        Self { parent: (0..size).collect() }
    }

    fn find(&mut self, node: usize) -> usize {
        let mut root = node;
        while self.parent[root] != root {
            root = self.parent[root];
        }
        let mut current = node;
        while self.parent[current] != root {
            let next = self.parent[current];
            self.parent[current] = root;
            current = next;
        }
        root
    }

    /// Returns `true` when the two nodes were in different sets (a real merge).
    fn union(&mut self, left: usize, right: usize) -> bool {
        let (left_root, right_root) = (self.find(left), self.find(right));
        if left_root == right_root {
            return false;
        }
        self.parent[left_root] = right_root;
        true
    }
}

/// Canonical undirected edge key, endpoints sorted.
fn edge_key(source: usize, destination: usize) -> (usize, usize) {
    if source < destination { (source, destination) } else { (destination, source) }
}

/// Asserts the structural validity invariants of a minimum spanning forest.
fn assert_valid_forest(
    label: &str,
    case_name: &str,
    node_count: usize,
    graph_edges: &[(usize, usize, f64)],
    expected_components: usize,
    forest: &WeightedForest<usize, f64>,
) {
    assert_eq!(forest.number_of_nodes(), node_count, "{case_name}/{label}: node count");
    assert_eq!(
        forest.number_of_components(),
        expected_components,
        "{case_name}/{label}: component count",
    );
    assert_eq!(
        forest.len(),
        node_count - expected_components,
        "{case_name}/{label}: edge count must be n - components",
    );

    let available: std::collections::HashSet<(usize, usize)> =
        graph_edges.iter().map(|&(s, d, _)| edge_key(s, d)).collect();

    let mut disjoint = DisjointSet::new(node_count);
    let mut summed = 0.0;
    for (source, destination, weight) in forest.edge_iter() {
        assert!(source < destination, "{case_name}/{label}: edges must be canonical");
        assert!(
            available.contains(&edge_key(source, destination)),
            "{case_name}/{label}: edge ({source}, {destination}) is not in the graph",
        );
        assert!(
            disjoint.union(source, destination),
            "{case_name}/{label}: forest contains a cycle at ({source}, {destination})",
        );
        summed += weight;
    }
    assert!(
        approx_eq(summed, forest.total_weight()),
        "{case_name}/{label}: reported total weight {} disagrees with edge sum {summed}",
        forest.total_weight(),
    );
}

// ----------------------------------------------------------------------------
// Hand-checkable cases
// ----------------------------------------------------------------------------

#[test]
fn triangle_drops_heaviest_edge() {
    let matrix = build(3, &[(0, 1, 1.0), (1, 2, 2.0), (0, 2, 3.0)]);
    for (label, forest) in run_all(&matrix) {
        assert!(approx_eq(forest.total_weight(), 3.0), "{label}: weight");
        assert_eq!(forest.number_of_components(), 1, "{label}: connected");
        assert_eq!(forest.len(), 2, "{label}: two edges");
        let keys: std::collections::HashSet<(usize, usize)> =
            forest.edge_iter().map(|(s, d, _)| edge_key(s, d)).collect();
        assert_eq!(keys, std::collections::HashSet::from([(0, 1), (1, 2)]), "{label}: edge set");
    }
}

#[test]
fn two_components_forest() {
    // Two disjoint paths: 0-1-2 and 3-4-5.
    let matrix = build(6, &[(0, 1, 1.0), (1, 2, 2.0), (3, 4, 1.0), (4, 5, 5.0)]);
    for (label, forest) in run_all(&matrix) {
        assert_eq!(forest.number_of_components(), 2, "{label}: two components");
        assert_eq!(forest.len(), 4, "{label}: four edges");
        assert!(approx_eq(forest.total_weight(), 9.0), "{label}: weight");
    }
}

#[test]
fn isolated_nodes_are_their_own_components() {
    // One edge plus two isolated nodes.
    let matrix = build(4, &[(0, 1, 3.0)]);
    for (label, forest) in run_all(&matrix) {
        assert_eq!(forest.number_of_components(), 3, "{label}: three components");
        assert_eq!(forest.len(), 1, "{label}: one edge");
        assert!(approx_eq(forest.total_weight(), 3.0), "{label}: weight");
    }
}

#[test]
fn all_ties_square_picks_any_three_edges() {
    let matrix = build(4, &[(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0), (3, 0, 1.0)]);
    let edges = [(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0), (3, 0, 1.0)];
    for (label, forest) in run_all(&matrix) {
        assert!(approx_eq(forest.total_weight(), 3.0), "{label}: weight");
        assert_valid_forest(label, "square_all_ties", 4, &edges, 1, &forest);
    }
}

#[test]
fn integer_and_negative_weights() {
    use geometric_traits::impls::ValuedCSR2D;

    // The generic rework must work for non-f64 weights and negative weights.
    // A triangle with weights -3, -1, 2: the MST keeps the two most negative
    // edges for a total of -4.
    type IntMatrix = ValuedCSR2D<usize, usize, usize, i64>;
    let mut directed = Vec::new();
    for &(source, destination, weight) in &[(0usize, 1usize, -3i64), (1, 2, -1), (0, 2, 2)] {
        directed.push((source, destination, weight));
        directed.push((destination, source, weight));
    }
    directed.sort_unstable_by(|(ls, ld, _), (rs, rd, _)| (ls, ld).cmp(&(rs, rd)));
    let matrix: IntMatrix = GenericEdgesBuilder::<_, IntMatrix>::default()
        .expected_number_of_edges(directed.len())
        .expected_shape((3, 3))
        .edges(directed.into_iter())
        .build()
        .unwrap();

    let kruskal = matrix.minimum_spanning_tree_kruskal().unwrap();
    assert_eq!(kruskal.total_weight(), -4);
    assert_eq!(kruskal.number_of_components(), 1);
    assert_eq!(kruskal.parent_array(), matrix.minimum_spanning_tree_prim().unwrap().parent_array());
    assert_eq!(
        kruskal.parent_array(),
        matrix.minimum_spanning_tree_boruvka().unwrap().parent_array()
    );
}

// ----------------------------------------------------------------------------
// NetworkX ground-truth corpus
// ----------------------------------------------------------------------------

#[test]
fn fixture_header_is_well_formed() {
    let suite = load_fixture_suite(FIXTURE_NAME);
    assert_eq!(suite.schema_version, 1);
    assert_eq!(suite.algorithm, "minimum_spanning_tree_networkx_reference");
    assert_eq!(suite.graph_kind, "simple_undirected_weighted");
    assert!(!suite.cases.is_empty());
}

#[test]
fn matches_networkx_for_every_algorithm() {
    let suite = load_fixture_suite(FIXTURE_NAME);
    for case in &suite.cases {
        let matrix = build_weighted_matrix(case);
        for (label, forest) in run_all(&matrix) {
            assert!(
                approx_eq(forest.total_weight(), case.mst_total_weight),
                "{}/{label}: total weight {} vs NetworkX {}",
                case.name,
                forest.total_weight(),
                case.mst_total_weight,
            );
            assert_eq!(forest.len(), case.mst_edge_count, "{}/{label}: edge count", case.name);
            assert_valid_forest(
                label,
                &case.name,
                case.node_count,
                &case.edges,
                case.number_of_components,
                &forest,
            );
            if case.weights_distinct {
                assert_exact_edges(label, case, &forest);
            }
        }
    }
}

fn assert_exact_edges(
    label: &str,
    case: &MinimumSpanningTreeFixtureCase,
    forest: &WeightedForest<usize, f64>,
) {
    let actual: std::collections::HashSet<(usize, usize)> =
        forest.edge_iter().map(|(s, d, _)| edge_key(s, d)).collect();
    let expected: std::collections::HashSet<(usize, usize)> =
        case.mst_edges.iter().map(|&(s, d, _)| edge_key(s, d)).collect();
    assert_eq!(actual, expected, "{}/{label}: exact edge set under distinct weights", case.name);
}

#[test]
fn algorithms_agree_on_total_weight() {
    let suite = load_fixture_suite(FIXTURE_NAME);
    for case in &suite.cases {
        let matrix = build_weighted_matrix(case);
        let kruskal = matrix.minimum_spanning_tree_kruskal().unwrap();
        let prim = matrix.minimum_spanning_tree_prim().unwrap();
        let boruvka = matrix.minimum_spanning_tree_boruvka().unwrap();
        assert!(
            approx_eq(kruskal.total_weight(), prim.total_weight()),
            "{}: kruskal vs prim weight",
            case.name,
        );
        assert!(
            approx_eq(kruskal.total_weight(), boruvka.total_weight()),
            "{}: kruskal vs boruvka weight",
            case.name,
        );
    }
}

#[test]
fn algorithms_agree_on_parent_array_for_unique_mst() {
    // Distinct weights make the MST unique, so the rooted parent array (rooted
    // at each component's lowest index) must be identical across all three.
    let matrix = build(
        6,
        &[
            (0, 1, 1.0),
            (1, 2, 2.0),
            (2, 3, 3.0),
            (3, 4, 4.0),
            (4, 5, 5.0),
            (0, 5, 9.0),
            (1, 4, 8.0),
            (2, 5, 7.0),
        ],
    );
    let kruskal = matrix.minimum_spanning_tree_kruskal().unwrap();
    let prim = matrix.minimum_spanning_tree_prim().unwrap();
    let boruvka = matrix.minimum_spanning_tree_boruvka().unwrap();
    assert_eq!(kruskal.parent_array(), prim.parent_array(), "kruskal vs prim parent array");
    assert_eq!(kruskal.parent_array(), boruvka.parent_array(), "kruskal vs boruvka parent array");
}

#[test]
fn forest_predicates() {
    // Connected triangle: a single spanning tree, non-empty.
    let connected =
        build(3, &[(0, 1, 1.0), (1, 2, 2.0), (0, 2, 3.0)]).minimum_spanning_tree_kruskal().unwrap();
    assert!(!connected.is_empty());
    assert!(connected.is_spanning_tree());

    // Four isolated nodes: empty forest, four components, not a spanning tree.
    let isolated = build(4, &[]).minimum_spanning_tree_prim().unwrap();
    assert!(isolated.is_empty());
    assert!(!isolated.is_spanning_tree());

    // Single node: empty forest but a degenerate spanning tree (one component).
    let single = build(1, &[]).minimum_spanning_tree_boruvka().unwrap();
    assert!(single.is_empty());
    assert!(single.is_spanning_tree());
}

// ----------------------------------------------------------------------------
// Error paths
// ----------------------------------------------------------------------------

#[test]
fn non_square_matrix_is_rejected() {
    // A 2-by-3 valued matrix is not a graph adjacency.
    let matrix: WeightedMatrix = GenericEdgesBuilder::<_, WeightedMatrix>::default()
        .expected_number_of_edges(1)
        .expected_shape((2, 3))
        .edges([(0usize, 1usize, 1.0)].into_iter())
        .build()
        .unwrap();
    assert!(matches!(
        matrix.minimum_spanning_tree_kruskal(),
        Err(MinimumSpanningTreeError::NonSquareMatrix { rows: 2, columns: 3 })
    ));
    assert!(matches!(
        matrix.minimum_spanning_tree_prim(),
        Err(MinimumSpanningTreeError::NonSquareMatrix { rows: 2, columns: 3 })
    ));
    assert!(matches!(
        matrix.minimum_spanning_tree_boruvka(),
        Err(MinimumSpanningTreeError::NonSquareMatrix { rows: 2, columns: 3 })
    ));
}

#[test]
fn non_finite_weight_is_rejected() {
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let matrix = build(2, &[(0, 1, bad)]);
        assert!(
            matches!(
                matrix.minimum_spanning_tree_kruskal(),
                Err(MinimumSpanningTreeError::NonFiniteWeight { source_id: 0, destination_id: 1 })
            ),
            "weight {bad} should be rejected",
        );
    }
}
