//! Fixture-based integration tests for graph-level node ordering and scoring.
#![cfg(feature = "std")]

#[path = "support/node_ordering_fixture.rs"]
mod node_ordering_fixture;

use geometric_traits::{
    impls::{BitSquareMatrix, CSR2D, SortedVec, SquareCSR2D, SymmetricCSR2D, ValuedCSR2D},
    naive_structs::GenericGraph,
    prelude::*,
    traits::{
        Edges, SparseValuedMatrix2D, SquareMatrix, VocabularyBuilder,
        algorithms::{ModularProduct, randomized_graphs::*},
    },
};
use node_ordering_fixture::{build_undigraph, load_fixture_suite};

type UndirectedGraph = SymmetricCSR2D<CSR2D<usize, usize, usize>>;
type DirectedGraph = SquareCSR2D<CSR2D<usize, usize, usize>>;
type WeightedUndirectedGraph = SymmetricCSR2D<ValuedCSR2D<usize, usize, usize, i32>>;

fn wrap_undi(g: UndirectedGraph) -> UndiGraph<usize> {
    let n = g.order();
    let nodes: SortedVec<usize> = GenericVocabularyBuilder::default()
        .expected_number_of_symbols(n)
        .symbols((0..n).enumerate())
        .build()
        .unwrap();
    UndiGraph::from((nodes, g))
}

fn wrap_di(g: DirectedGraph) -> DiGraph<usize> {
    let n = g.order();
    let nodes: SortedVec<usize> = GenericVocabularyBuilder::default()
        .expected_number_of_symbols(n)
        .symbols((0..n).enumerate())
        .build()
        .unwrap();
    DiGraph::from((nodes, g))
}

fn wrap_undi_vec(g: UndirectedGraph) -> GenericGraph<Vec<usize>, UndirectedGraph> {
    let n = g.order();
    GenericGraph::from(((0..n).collect::<Vec<_>>(), g))
}

#[derive(Clone, Copy, Debug)]
enum GraphFixture {
    Path(usize),
    Cycle(usize),
    Complete(usize),
    Star(usize),
    BranchingTree,
    DegreeBiasedBranching,
    TriangleWithTail,
    PathWithIsolatedNode,
}

impl GraphFixture {
    fn build(self) -> UndiGraph<usize> {
        match self {
            Self::Path(n) => wrap_undi(path_graph(n)),
            Self::Cycle(n) => wrap_undi(cycle_graph(n)),
            Self::Complete(n) => wrap_undi(complete_graph(n)),
            Self::Star(n) => wrap_undi(star_graph(n)),
            Self::BranchingTree => {
                let nodes: SortedVec<usize> = GenericVocabularyBuilder::default()
                    .expected_number_of_symbols(6)
                    .symbols((0..6).enumerate())
                    .build()
                    .unwrap();
                let matrix: SymmetricCSR2D<CSR2D<usize, usize, usize>> =
                    UndiEdgesBuilder::default()
                        .expected_number_of_edges(5)
                        .expected_shape(6)
                        .edges([(0, 1), (0, 2), (1, 3), (1, 4), (2, 5)].into_iter())
                        .build()
                        .unwrap();
                UndiGraph::from((nodes, matrix))
            }
            Self::DegreeBiasedBranching => {
                let nodes: SortedVec<usize> = GenericVocabularyBuilder::default()
                    .expected_number_of_symbols(6)
                    .symbols((0..6).enumerate())
                    .build()
                    .unwrap();
                let matrix: SymmetricCSR2D<CSR2D<usize, usize, usize>> =
                    UndiEdgesBuilder::default()
                        .expected_number_of_edges(5)
                        .expected_shape(6)
                        .edges([(0, 1), (0, 2), (1, 3), (2, 4), (2, 5)].into_iter())
                        .build()
                        .unwrap();
                UndiGraph::from((nodes, matrix))
            }
            Self::TriangleWithTail => {
                let nodes: SortedVec<usize> = GenericVocabularyBuilder::default()
                    .expected_number_of_symbols(5)
                    .symbols((0..5).enumerate())
                    .build()
                    .unwrap();
                let matrix: SymmetricCSR2D<CSR2D<usize, usize, usize>> =
                    UndiEdgesBuilder::default()
                        .expected_number_of_edges(5)
                        .expected_shape(5)
                        .edges([(0, 1), (0, 2), (1, 2), (2, 3), (3, 4)].into_iter())
                        .build()
                        .unwrap();
                UndiGraph::from((nodes, matrix))
            }
            Self::PathWithIsolatedNode => {
                let nodes: SortedVec<usize> = GenericVocabularyBuilder::default()
                    .expected_number_of_symbols(4)
                    .symbols((0..4).enumerate())
                    .build()
                    .unwrap();
                let matrix: SymmetricCSR2D<CSR2D<usize, usize, usize>> =
                    UndiEdgesBuilder::default()
                        .expected_number_of_edges(2)
                        .expected_shape(4)
                        .edges([(0, 1), (1, 2)].into_iter())
                        .build()
                        .unwrap();
                UndiGraph::from((nodes, matrix))
            }
        }
    }
}

struct OrderingFixture {
    name: &'static str,
    graph: GraphFixture,
}

struct ExactOrderingFixture {
    name: &'static str,
    graph: GraphFixture,
    expected_order: &'static [usize],
}

struct ScoringFixture {
    name: &'static str,
    graph: GraphFixture,
    expected_scores: &'static [usize],
    expected_descending: &'static [usize],
    expected_ascending: &'static [usize],
}

struct CoreLexicographicFixture {
    name: &'static str,
    graph: GraphFixture,
    expected_core_numbers: &'static [usize],
    expected_order: &'static [usize],
}

struct FloatingScoringFixture {
    name: &'static str,
    graph: GraphFixture,
    expected_scores: &'static [f64],
    expected_descending: &'static [usize],
}

struct ConfigurableFloatingScoringFixture {
    name: &'static str,
    graph: GraphFixture,
    alpha: f64,
    beta: f64,
    max_iter: usize,
    tolerance: f64,
    normalized: bool,
    expected_scores: &'static [f64],
    expected_descending: &'static [usize],
}

struct BetweennessScoringFixture {
    name: &'static str,
    graph: GraphFixture,
    normalized: bool,
    endpoints: bool,
    expected_scores: &'static [f64],
    expected_descending: &'static [usize],
}

struct ClosenessScoringFixture {
    name: &'static str,
    graph: GraphFixture,
    wf_improved: bool,
    expected_scores: &'static [f64],
    expected_descending: &'static [usize],
}

#[derive(Clone, Copy, Debug, Default)]
struct WrongLengthScorer;

impl NodeScorer<UndiGraph<usize>> for WrongLengthScorer {
    type Score = usize;

    fn score_nodes(&self, _graph: &UndiGraph<usize>) -> Vec<Self::Score> {
        vec![1]
    }
}

fn assert_is_permutation(order: &[usize], n: usize, context: &str) {
    assert_eq!(order.len(), n, "ordering `{context}` does not contain exactly one entry per node");
    let mut seen = vec![false; n];
    for &node in order {
        assert!(node < n, "ordering `{context}` contains out-of-range node {node}");
        assert!(!seen[node], "ordering `{context}` contains duplicate node {node}");
        seen[node] = true;
    }
}

fn assert_is_smallest_last_order(graph: &UndiGraph<usize>, order: &[usize], context: &str) {
    let n = graph.number_of_nodes();
    assert_is_permutation(order, n, context);

    let mut active = vec![true; n];
    let mut degrees: Vec<usize> = (0..n).map(|node| graph.degree(node)).collect();

    for &node in order.iter().rev() {
        let min_degree =
            (0..n).filter(|&candidate| active[candidate]).map(|candidate| degrees[candidate]).min();
        assert_eq!(
            Some(degrees[node]),
            min_degree,
            "ordering `{context}` is not a valid smallest-last sequence at removed node {node}"
        );

        active[node] = false;
        for neighbor in graph.neighbors(node) {
            if active[neighbor] {
                degrees[neighbor] = degrees[neighbor].saturating_sub(1);
            }
        }
    }
}

fn assert_scores_close(actual: &[f64], expected: &[f64], tolerance: f64, context: &str) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "score vector `{context}` does not contain exactly one entry per node"
    );
    for (index, (actual_score, expected_score)) in actual.iter().zip(expected.iter()).enumerate() {
        let delta = (actual_score - expected_score).abs();
        assert!(
            delta <= tolerance,
            "score vector `{context}` differs at node {index}: actual={actual_score}, expected={expected_score}, delta={delta}, tolerance={tolerance}"
        );
    }
}

const KATZ_TOLERANCE: f64 = 1.0e-11;
const BETWEENNESS_TOLERANCE: f64 = 2.0e-12;
const CLOSENESS_TOLERANCE: f64 = 1.0e-12;
const LOCAL_CLUSTERING_TOLERANCE: f64 = 1.0e-12;

const DEGENERACY_FIXTURES: &[OrderingFixture] = &[
    OrderingFixture { name: "path_4", graph: GraphFixture::Path(4) },
    OrderingFixture { name: "star_5", graph: GraphFixture::Star(5) },
    OrderingFixture { name: "complete_4", graph: GraphFixture::Complete(4) },
    OrderingFixture { name: "cycle_4", graph: GraphFixture::Cycle(4) },
];

const WELSH_POWELL_FIXTURES: &[OrderingFixture] = &[
    OrderingFixture { name: "path_4", graph: GraphFixture::Path(4) },
    OrderingFixture { name: "star_5", graph: GraphFixture::Star(5) },
    OrderingFixture { name: "triangle_with_tail", graph: GraphFixture::TriangleWithTail },
];

const DSATUR_FIXTURES: &[ExactOrderingFixture] = &[
    ExactOrderingFixture {
        name: "path_4",
        graph: GraphFixture::Path(4),
        expected_order: &[1, 2, 0, 3],
    },
    ExactOrderingFixture {
        name: "star_5",
        graph: GraphFixture::Star(5),
        expected_order: &[0, 1, 2, 3, 4],
    },
    ExactOrderingFixture {
        name: "triangle_with_tail",
        graph: GraphFixture::TriangleWithTail,
        expected_order: &[2, 0, 1, 3, 4],
    },
    ExactOrderingFixture {
        name: "path_with_isolated_node",
        graph: GraphFixture::PathWithIsolatedNode,
        expected_order: &[1, 0, 2, 3],
    },
];

const BFS_FROM_MAX_DEGREE_FIXTURES: &[ExactOrderingFixture] = &[
    ExactOrderingFixture {
        name: "path_4",
        graph: GraphFixture::Path(4),
        expected_order: &[1, 0, 2, 3],
    },
    ExactOrderingFixture {
        name: "branching_tree",
        graph: GraphFixture::BranchingTree,
        expected_order: &[1, 0, 3, 4, 2, 5],
    },
    ExactOrderingFixture {
        name: "path_with_isolated_node",
        graph: GraphFixture::PathWithIsolatedNode,
        expected_order: &[1, 0, 2, 3],
    },
];

const DFS_FROM_MAX_DEGREE_FIXTURES: &[ExactOrderingFixture] = &[
    ExactOrderingFixture {
        name: "path_4",
        graph: GraphFixture::Path(4),
        expected_order: &[1, 0, 2, 3],
    },
    ExactOrderingFixture {
        name: "branching_tree",
        graph: GraphFixture::BranchingTree,
        expected_order: &[1, 0, 2, 5, 3, 4],
    },
    ExactOrderingFixture {
        name: "path_with_isolated_node",
        graph: GraphFixture::PathWithIsolatedNode,
        expected_order: &[1, 0, 2, 3],
    },
];

const SECOND_ORDER_DEGREE_FIXTURES: &[ScoringFixture] = &[
    ScoringFixture {
        name: "path_4",
        graph: GraphFixture::Path(4),
        expected_scores: &[2, 3, 3, 2],
        expected_descending: &[1, 2, 0, 3],
        expected_ascending: &[0, 3, 1, 2],
    },
    ScoringFixture {
        name: "complete_4",
        graph: GraphFixture::Complete(4),
        expected_scores: &[9, 9, 9, 9],
        expected_descending: &[0, 1, 2, 3],
        expected_ascending: &[0, 1, 2, 3],
    },
];

const DEGREE_FIXTURES: &[ScoringFixture] = &[
    ScoringFixture {
        name: "path_4",
        graph: GraphFixture::Path(4),
        expected_scores: &[1, 2, 2, 1],
        expected_descending: &[1, 2, 0, 3],
        expected_ascending: &[0, 3, 1, 2],
    },
    ScoringFixture {
        name: "star_5",
        graph: GraphFixture::Star(5),
        expected_scores: &[4, 1, 1, 1, 1],
        expected_descending: &[0, 1, 2, 3, 4],
        expected_ascending: &[1, 2, 3, 4, 0],
    },
];

const TRIANGLE_COUNT_FIXTURES: &[ScoringFixture] = &[
    ScoringFixture {
        name: "path_4",
        graph: GraphFixture::Path(4),
        expected_scores: &[0, 0, 0, 0],
        expected_descending: &[0, 1, 2, 3],
        expected_ascending: &[0, 1, 2, 3],
    },
    ScoringFixture {
        name: "complete_4",
        graph: GraphFixture::Complete(4),
        expected_scores: &[3, 3, 3, 3],
        expected_descending: &[0, 1, 2, 3],
        expected_ascending: &[0, 1, 2, 3],
    },
    ScoringFixture {
        name: "triangle_with_tail",
        graph: GraphFixture::TriangleWithTail,
        expected_scores: &[1, 1, 1, 0, 0],
        expected_descending: &[0, 1, 2, 3, 4],
        expected_ascending: &[3, 4, 0, 1, 2],
    },
];

const DEGENERACY_DEGREE_FIXTURES: &[CoreLexicographicFixture] = &[
    CoreLexicographicFixture {
        name: "path_4",
        graph: GraphFixture::Path(4),
        expected_core_numbers: &[1, 1, 1, 1],
        expected_order: &[1, 2, 0, 3],
    },
    CoreLexicographicFixture {
        name: "star_5",
        graph: GraphFixture::Star(5),
        expected_core_numbers: &[1, 1, 1, 1, 1],
        expected_order: &[0, 1, 2, 3, 4],
    },
    CoreLexicographicFixture {
        name: "triangle_with_tail",
        graph: GraphFixture::TriangleWithTail,
        expected_core_numbers: &[2, 2, 2, 1, 1],
        expected_order: &[2, 0, 1, 3, 4],
    },
];

const PAGERANK_FIXTURES: &[FloatingScoringFixture] = &[
    FloatingScoringFixture {
        name: "path_4",
        graph: GraphFixture::Path(4),
        expected_scores: &[
            0.17543839772251532,
            0.32456160227748454,
            0.32456160227748454,
            0.17543839772251532,
        ],
        expected_descending: &[1, 2, 0, 3],
    },
    FloatingScoringFixture {
        name: "complete_4",
        graph: GraphFixture::Complete(4),
        expected_scores: &[0.25, 0.25, 0.25, 0.25],
        expected_descending: &[0, 1, 2, 3],
    },
    FloatingScoringFixture {
        name: "star_5",
        graph: GraphFixture::Star(5),
        expected_scores: &[
            0.47567668878363595,
            0.13108082780409108,
            0.13108082780409108,
            0.13108082780409108,
            0.13108082780409108,
        ],
        expected_descending: &[0, 1, 2, 3, 4],
    },
    FloatingScoringFixture {
        name: "triangle_with_tail",
        graph: GraphFixture::TriangleWithTail,
        expected_scores: &[
            0.19182193316290375,
            0.19182193316290375,
            0.28340244242002904,
            0.21259959667728376,
            0.12035409457687965,
        ],
        expected_descending: &[2, 3, 0, 1, 4],
    },
];

const KATZ_FIXTURES: &[ConfigurableFloatingScoringFixture] = &[
    ConfigurableFloatingScoringFixture {
        name: "path_4_norm",
        graph: GraphFixture::Path(4),
        alpha: 0.25,
        beta: 1.0,
        max_iter: 1000,
        tolerance: 1.0e-12,
        normalized: true,
        expected_scores: &[0.441726104299, 0.552157630374, 0.552157630374, 0.441726104299],
        expected_descending: &[1, 2, 0, 3],
    },
    ConfigurableFloatingScoringFixture {
        name: "triangle_tail_raw",
        graph: GraphFixture::TriangleWithTail,
        alpha: 0.2,
        beta: 1.5,
        max_iter: 1000,
        tolerance: 1.0e-12,
        normalized: false,
        expected_scores: &[
            2.639563106796,
            2.639563106796,
            3.058252427184,
            2.51213592233,
            2.002427184466,
        ],
        expected_descending: &[2, 0, 1, 3, 4],
    },
];

const BETWEENNESS_FIXTURES: &[BetweennessScoringFixture] = &[
    BetweennessScoringFixture {
        name: "path_4_default",
        graph: GraphFixture::Path(4),
        normalized: true,
        endpoints: false,
        expected_scores: &[0.0, 0.666666666667, 0.666666666667, 0.0],
        expected_descending: &[1, 2, 0, 3],
    },
    BetweennessScoringFixture {
        name: "complete_4_default",
        graph: GraphFixture::Complete(4),
        normalized: true,
        endpoints: false,
        expected_scores: &[0.0, 0.0, 0.0, 0.0],
        expected_descending: &[0, 1, 2, 3],
    },
    BetweennessScoringFixture {
        name: "star_5_default",
        graph: GraphFixture::Star(5),
        normalized: true,
        endpoints: false,
        expected_scores: &[1.0, 0.0, 0.0, 0.0, 0.0],
        expected_descending: &[0, 1, 2, 3, 4],
    },
    BetweennessScoringFixture {
        name: "triangle_with_tail_default",
        graph: GraphFixture::TriangleWithTail,
        normalized: true,
        endpoints: false,
        expected_scores: &[0.0, 0.0, 0.666666666667, 0.5, 0.0],
        expected_descending: &[2, 3, 0, 1, 4],
    },
];

const CLOSENESS_FIXTURES: &[ClosenessScoringFixture] = &[
    ClosenessScoringFixture {
        name: "path_4_default",
        graph: GraphFixture::Path(4),
        wf_improved: true,
        expected_scores: &[0.5, 0.75, 0.75, 0.5],
        expected_descending: &[1, 2, 0, 3],
    },
    ClosenessScoringFixture {
        name: "complete_4_default",
        graph: GraphFixture::Complete(4),
        wf_improved: true,
        expected_scores: &[1.0, 1.0, 1.0, 1.0],
        expected_descending: &[0, 1, 2, 3],
    },
    ClosenessScoringFixture {
        name: "star_5_default",
        graph: GraphFixture::Star(5),
        wf_improved: true,
        expected_scores: &[1.0, 0.571428571429, 0.571428571429, 0.571428571429, 0.571428571429],
        expected_descending: &[0, 1, 2, 3, 4],
    },
    ClosenessScoringFixture {
        name: "triangle_with_tail_default",
        graph: GraphFixture::TriangleWithTail,
        wf_improved: true,
        expected_scores: &[0.571428571429, 0.571428571429, 0.8, 0.666666666667, 0.444444444444],
        expected_descending: &[2, 3, 0, 1, 4],
    },
    ClosenessScoringFixture {
        name: "path_with_isolated_node_default",
        graph: GraphFixture::PathWithIsolatedNode,
        wf_improved: true,
        expected_scores: &[0.444444444444, 0.666666666667, 0.444444444444, 0.0],
        expected_descending: &[1, 0, 2, 3],
    },
];

const LOCAL_CLUSTERING_FIXTURES: &[FloatingScoringFixture] = &[
    FloatingScoringFixture {
        name: "path_4",
        graph: GraphFixture::Path(4),
        expected_scores: &[0.0, 0.0, 0.0, 0.0],
        expected_descending: &[0, 1, 2, 3],
    },
    FloatingScoringFixture {
        name: "complete_4",
        graph: GraphFixture::Complete(4),
        expected_scores: &[1.0, 1.0, 1.0, 1.0],
        expected_descending: &[0, 1, 2, 3],
    },
    FloatingScoringFixture {
        name: "triangle_with_tail",
        graph: GraphFixture::TriangleWithTail,
        expected_scores: &[1.0, 1.0, 0.333333333333, 0.0, 0.0],
        expected_descending: &[0, 1, 2, 3, 4],
    },
];

#[test]
fn test_degeneracy_sorter_fixtures() {
    for fixture in DEGENERACY_FIXTURES {
        let graph = fixture.graph.build();
        let order = DegeneracySorter.sort_nodes(&graph);
        assert_is_smallest_last_order(&graph, &order, fixture.name);
    }
}

#[test]
fn test_welsh_powell_sorter_fixtures() {
    let sorter = DescendingScoreSorter::new(DegreeScorer);

    for fixture in WELSH_POWELL_FIXTURES {
        let graph = fixture.graph.build();
        let expected = DescendingScoreSorter::new(DegreeScorer).sort_nodes(&graph);
        assert_eq!(
            sorter.sort_nodes(&graph),
            expected,
            "welsh-powell ordering fixture failed for {}",
            fixture.name
        );
    }
}

#[test]
fn test_dsatur_sorter_fixtures() {
    for fixture in DSATUR_FIXTURES {
        let graph = fixture.graph.build();
        assert_eq!(
            DsaturSorter.sort_nodes(&graph),
            fixture.expected_order,
            "dsatur ordering fixture failed for {}",
            fixture.name
        );
    }
}

#[test]
fn test_bfs_from_max_degree_sorter_fixtures() {
    let sorter = BfsTraversalSorter::new(
        TraversalSeedStrategy::MaxOutDegree,
        TraversalNeighborOrder::NodeIdAscending,
    );

    for fixture in BFS_FROM_MAX_DEGREE_FIXTURES {
        let graph = fixture.graph.build();
        assert_eq!(
            sorter.sort_nodes(&graph),
            fixture.expected_order,
            "bfs-from-max-degree ordering fixture failed for {}",
            fixture.name
        );
    }
}

#[test]
fn test_dfs_from_max_degree_sorter_fixtures() {
    let sorter = DfsTraversalSorter::new(
        TraversalSeedStrategy::MaxOutDegree,
        TraversalNeighborOrder::NodeIdAscending,
    );

    for fixture in DFS_FROM_MAX_DEGREE_FIXTURES {
        let graph = fixture.graph.build();
        assert_eq!(
            sorter.sort_nodes(&graph),
            fixture.expected_order,
            "dfs-from-max-degree ordering fixture failed for {}",
            fixture.name
        );
    }
}

#[test]
fn test_bfs_traversal_sorter_defaults_use_node_id_ascending() {
    let graph = GraphFixture::BranchingTree.build();
    let sorter = BfsTraversalSorter::default();
    assert_eq!(sorter.sort_nodes(&graph), &[0, 1, 2, 3, 4, 5]);
}

#[test]
fn test_dfs_traversal_sorter_defaults_use_node_id_ascending() {
    let graph = GraphFixture::BranchingTree.build();
    let sorter = DfsTraversalSorter::default();
    assert_eq!(sorter.sort_nodes(&graph), &[0, 1, 3, 4, 2, 5]);
}

#[test]
fn test_bfs_traversal_sorter_supports_descending_neighbor_order() {
    let graph = GraphFixture::DegreeBiasedBranching.build();
    let sorter = BfsTraversalSorter::new(
        TraversalSeedStrategy::NodeIdAscending,
        TraversalNeighborOrder::NodeIdDescending,
    );
    assert_eq!(sorter.sort_nodes(&graph), &[0, 2, 1, 5, 4, 3]);
}

#[test]
fn test_dfs_traversal_sorter_supports_out_degree_descending_neighbors() {
    let graph = GraphFixture::DegreeBiasedBranching.build();
    let sorter = DfsTraversalSorter::new(
        TraversalSeedStrategy::NodeIdAscending,
        TraversalNeighborOrder::OutDegreeDescending,
    );
    assert_eq!(sorter.sort_nodes(&graph), &[0, 2, 4, 5, 1, 3]);
}

#[test]
fn test_bfs_traversal_sorter_supports_out_degree_descending_neighbors() {
    let graph = GraphFixture::DegreeBiasedBranching.build();
    let sorter = BfsTraversalSorter::new(
        TraversalSeedStrategy::NodeIdAscending,
        TraversalNeighborOrder::OutDegreeDescending,
    );
    assert_eq!(sorter.sort_nodes(&graph), &[0, 2, 1, 4, 5, 3]);
}

#[test]
fn test_dfs_traversal_sorter_supports_descending_neighbor_order() {
    let graph = GraphFixture::DegreeBiasedBranching.build();
    let sorter = DfsTraversalSorter::new(
        TraversalSeedStrategy::NodeIdAscending,
        TraversalNeighborOrder::NodeIdDescending,
    );
    assert_eq!(sorter.sort_nodes(&graph), &[0, 2, 5, 4, 1, 3]);
}

#[test]
fn test_traversal_sorters_handle_empty_graph() {
    let directed_edges: DirectedGraph = DiEdgesBuilder::default()
        .expected_number_of_edges(0)
        .expected_shape(0)
        .edges(core::iter::empty::<(usize, usize)>())
        .build()
        .unwrap();
    let graph = wrap_di(directed_edges);

    assert!(BfsTraversalSorter::default().sort_nodes(&graph).is_empty());
    assert!(DfsTraversalSorter::default().sort_nodes(&graph).is_empty());
}

#[test]
fn test_traversal_sorters_use_directed_out_degree_seed_policy() {
    let edges: DirectedGraph = DiEdgesBuilder::default()
        .expected_number_of_edges(4)
        .expected_shape(6)
        .edges([(0, 1), (0, 2), (2, 3), (4, 5)].into_iter())
        .build()
        .unwrap();
    let graph = wrap_di(edges);

    let bfs = BfsTraversalSorter::new(
        TraversalSeedStrategy::MaxOutDegree,
        TraversalNeighborOrder::NodeIdAscending,
    );
    let dfs = DfsTraversalSorter::new(
        TraversalSeedStrategy::MaxOutDegree,
        TraversalNeighborOrder::NodeIdAscending,
    );

    assert_eq!(bfs.sort_nodes(&graph), &[0, 1, 2, 3, 4, 5]);
    assert_eq!(dfs.sort_nodes(&graph), &[0, 1, 2, 3, 4, 5]);
}

#[test]
#[should_panic(expected = "node order must contain exactly one entry per node")]
fn test_apply_node_order_to_graph_panics_on_wrong_length() {
    let graph = wrap_undi_vec(path_graph(4));
    let _ = apply_node_order_to_graph(&graph, &[0, 1, 2]);
}

#[test]
#[should_panic(expected = "node order contains duplicate node 1")]
fn test_apply_node_order_to_graph_panics_on_duplicate_node() {
    let graph = wrap_undi_vec(path_graph(4));
    let _ = apply_node_order_to_graph(&graph, &[0, 1, 1, 2]);
}

#[test]
#[should_panic(expected = "node order contains out-of-range node 4")]
fn test_apply_node_order_to_graph_panics_on_out_of_range_node() {
    let graph = wrap_undi_vec(path_graph(4));
    let _ = apply_node_order_to_graph(&graph, &[0, 1, 2, 4]);
}

#[test]
fn test_apply_node_order_to_graph_reorders_undirected_symbols_and_edges() {
    let graph = wrap_undi_vec(path_graph(4));
    let reordered = apply_node_order_to_graph(&graph, &[1, 0, 2, 3]);

    let symbols: Vec<_> = reordered.nodes().collect();
    assert_eq!(symbols, vec![1, 0, 2, 3]);
    assert_eq!(reordered.neighbors(0).collect::<Vec<_>>(), vec![1, 2]);
    assert_eq!(reordered.neighbors(1).collect::<Vec<_>>(), vec![0]);
    assert_eq!(reordered.neighbors(2).collect::<Vec<_>>(), vec![0, 3]);
    assert_eq!(reordered.neighbors(3).collect::<Vec<_>>(), vec![2]);
}

#[test]
fn test_apply_node_order_to_graph_reorders_directed_symbols_and_edges() {
    let nodes = (0..4).collect::<Vec<_>>();
    let edges: DirectedGraph = DiEdgesBuilder::default()
        .expected_number_of_edges(3)
        .expected_shape(4)
        .edges([(0, 2), (2, 1), (3, 3)].into_iter())
        .build()
        .unwrap();
    let graph = GenericGraph::from((nodes, edges));

    let reordered = apply_node_order_to_graph(&graph, &[2, 0, 1, 3]);

    let symbols: Vec<_> = reordered.nodes().collect();
    assert_eq!(symbols, vec![2, 0, 1, 3]);
    assert_eq!(reordered.successors(0).collect::<Vec<_>>(), vec![2]);
    assert_eq!(reordered.successors(1).collect::<Vec<_>>(), vec![0]);
    assert!(reordered.successors(2).collect::<Vec<_>>().is_empty());
    assert_eq!(reordered.successors(3).collect::<Vec<_>>(), vec![3]);
}

#[test]
fn test_apply_node_order_to_graph_preserves_weighted_edge_labels() {
    let nodes = vec!["a", "b", "c"];
    let edges: ValuedCSR2D<usize, usize, usize, f64> =
        GenericEdgesBuilder::<_, ValuedCSR2D<usize, usize, usize, f64>>::default()
            .expected_number_of_edges(2)
            .expected_shape((3, 3))
            .edges([(0, 1, 1.5), (2, 0, 2.5)].into_iter())
            .build()
            .unwrap();
    let graph: GenericGraph<Vec<&str>, ValuedCSR2D<usize, usize, usize, f64>> =
        GenericGraph::from((nodes, edges));

    let reordered = apply_node_order_to_graph(&graph, &[2, 0, 1]);

    let symbols: Vec<_> = reordered.nodes().collect();
    assert_eq!(symbols, vec!["c", "a", "b"]);
    assert_eq!(reordered.successors(0).collect::<Vec<_>>(), vec![1]);
    assert_eq!(reordered.successor_weights(0).collect::<Vec<_>>(), vec![2.5]);
    assert_eq!(reordered.successors(1).collect::<Vec<_>>(), vec![2]);
    assert_eq!(reordered.successor_weights(1).collect::<Vec<_>>(), vec![1.5]);
    assert!(reordered.successors(2).collect::<Vec<_>>().is_empty());
}

#[test]
fn test_apply_node_order_to_graph_preserves_weighted_undirected_edge_labels() {
    let nodes = vec!["a", "b", "c", "d"];
    let edges = WeightedUndirectedGraph::from_sorted_upper_triangular_entries(
        4,
        [(0, 1, 7), (0, 2, 11), (2, 3, 13)],
    )
    .unwrap();
    let graph: GenericGraph<Vec<&str>, WeightedUndirectedGraph> =
        GenericGraph::from((nodes, edges));

    let reordered = apply_node_order_to_graph(&graph, &[2, 0, 3, 1]);

    let reordered_matrix = Edges::matrix(reordered.edges());
    let symbols: Vec<_> = reordered.nodes().collect();
    assert_eq!(symbols, vec!["c", "a", "d", "b"]);
    assert_eq!(reordered.neighbors(0).collect::<Vec<_>>(), vec![1, 2]);
    assert_eq!(reordered.neighbors(1).collect::<Vec<_>>(), vec![0, 3]);
    assert_eq!(reordered.neighbors(2).collect::<Vec<_>>(), vec![0]);
    assert_eq!(reordered.neighbors(3).collect::<Vec<_>>(), vec![1]);
    assert_eq!(reordered_matrix.sparse_value_at(0, 1), Some(11));
    assert_eq!(reordered_matrix.sparse_value_at(0, 2), Some(13));
    assert_eq!(reordered_matrix.sparse_value_at(1, 3), Some(7));
}

#[test]
fn test_apply_node_order_to_graph_supports_modular_product_graphs() {
    let left = BitSquareMatrix::from_symmetric_edges(2, [(0, 1)]);
    let right = BitSquareMatrix::from_symmetric_edges(2, [(0, 1)]);
    let graph = left.modular_product_filtered(&right, |_, _| true).into_graph();

    let reordered = apply_node_order_to_graph(&graph, &[3, 1, 2, 0]);

    assert_eq!(reordered.node_ids().collect::<Vec<_>>(), vec![0, 1, 2, 3]);
    assert_eq!(reordered.nodes().collect::<Vec<_>>(), vec![(1, 1), (0, 1), (1, 0), (0, 0)]);
    assert_eq!(reordered.neighbors(0).collect::<Vec<_>>(), vec![3]);
    assert_eq!(reordered.neighbors(1).collect::<Vec<_>>(), vec![2]);
    assert_eq!(reordered.neighbors(2).collect::<Vec<_>>(), vec![1]);
    assert_eq!(reordered.neighbors(3).collect::<Vec<_>>(), vec![0]);
}

#[test]
fn test_second_order_degree_scorer_fixtures() {
    for fixture in SECOND_ORDER_DEGREE_FIXTURES {
        let graph = fixture.graph.build();
        assert_eq!(
            SecondOrderDegreeScorer.score_nodes(&graph),
            fixture.expected_scores,
            "second-order degree scorer fixture failed for {}",
            fixture.name
        );
    }
}

#[test]
fn test_descending_second_order_degree_sorter_fixtures() {
    let sorter = DescendingScoreSorter::new(SecondOrderDegreeScorer);

    for fixture in SECOND_ORDER_DEGREE_FIXTURES {
        let graph = fixture.graph.build();
        assert_eq!(
            sorter.sort_nodes(&graph),
            fixture.expected_descending,
            "descending second-order degree ordering fixture failed for {}",
            fixture.name
        );
    }
}

#[test]
fn test_ascending_second_order_degree_sorter_fixtures() {
    let sorter = AscendingScoreSorter::new(SecondOrderDegreeScorer);

    for fixture in SECOND_ORDER_DEGREE_FIXTURES {
        let graph = fixture.graph.build();
        assert_eq!(
            sorter.sort_nodes(&graph),
            fixture.expected_ascending,
            "ascending second-order degree ordering fixture failed for {}",
            fixture.name
        );
    }
}

#[test]
fn test_degree_scorer_fixtures() {
    for fixture in DEGREE_FIXTURES {
        let graph = fixture.graph.build();
        assert_eq!(
            DegreeScorer.score_nodes(&graph),
            fixture.expected_scores,
            "degree scorer fixture failed for {}",
            fixture.name
        );
    }
}

#[test]
fn test_core_number_scorer_fixtures() {
    for fixture in DEGENERACY_DEGREE_FIXTURES {
        let graph = fixture.graph.build();
        assert_eq!(
            CoreNumberScorer.score_nodes(&graph),
            fixture.expected_core_numbers,
            "core number scorer fixture failed for {}",
            fixture.name
        );
    }
}

#[test]
fn test_descending_lexicographic_core_degree_sorter_fixtures() {
    let sorter = DescendingLexicographicScoreSorter::new(CoreNumberScorer, DegreeScorer);

    for fixture in DEGENERACY_DEGREE_FIXTURES {
        let graph = fixture.graph.build();
        assert_eq!(
            sorter.sort_nodes(&graph),
            fixture.expected_order,
            "degeneracy + degree ordering fixture failed for {}",
            fixture.name
        );
    }
}

#[test]
fn test_descending_degree_sorter_fixtures() {
    let sorter = DescendingScoreSorter::new(DegreeScorer);

    for fixture in DEGREE_FIXTURES {
        let graph = fixture.graph.build();
        assert_eq!(
            sorter.sort_nodes(&graph),
            fixture.expected_descending,
            "descending degree ordering fixture failed for {}",
            fixture.name
        );
    }
}

#[test]
fn test_ascending_degree_sorter_fixtures() {
    let sorter = AscendingScoreSorter::new(DegreeScorer);

    for fixture in DEGREE_FIXTURES {
        let graph = fixture.graph.build();
        assert_eq!(
            sorter.sort_nodes(&graph),
            fixture.expected_ascending,
            "ascending degree ordering fixture failed for {}",
            fixture.name
        );
    }
}

#[test]
fn test_triangle_count_scorer_fixtures() {
    for fixture in TRIANGLE_COUNT_FIXTURES {
        let graph = fixture.graph.build();
        assert_eq!(
            TriangleCountScorer.score_nodes(&graph),
            fixture.expected_scores,
            "triangle count scorer fixture failed for {}",
            fixture.name
        );
    }
}

#[test]
fn test_descending_triangle_count_sorter_fixtures() {
    let sorter = DescendingScoreSorter::new(TriangleCountScorer);

    for fixture in TRIANGLE_COUNT_FIXTURES {
        let graph = fixture.graph.build();
        assert_eq!(
            sorter.sort_nodes(&graph),
            fixture.expected_descending,
            "descending triangle-count ordering fixture failed for {}",
            fixture.name
        );
    }
}

#[test]
fn test_ascending_triangle_count_sorter_fixtures() {
    let sorter = AscendingScoreSorter::new(TriangleCountScorer);

    for fixture in TRIANGLE_COUNT_FIXTURES {
        let graph = fixture.graph.build();
        assert_eq!(
            sorter.sort_nodes(&graph),
            fixture.expected_ascending,
            "ascending triangle-count ordering fixture failed for {}",
            fixture.name
        );
    }
}

#[test]
#[should_panic(expected = "node scorer must return one score per node")]
fn test_score_sorter_panics_on_wrong_length() {
    let graph = GraphFixture::Path(4).build();
    let _ = DescendingScoreSorter::new(WrongLengthScorer).sort_nodes(&graph);
}

#[test]
#[should_panic(expected = "primary node scorer must return one score per node")]
fn test_lexicographic_sorter_panics_on_wrong_primary_length() {
    let graph = GraphFixture::Path(4).build();
    let _ =
        DescendingLexicographicScoreSorter::new(WrongLengthScorer, DegreeScorer).sort_nodes(&graph);
}

#[test]
#[should_panic(expected = "secondary node scorer must return one score per node")]
fn test_lexicographic_sorter_panics_on_wrong_secondary_length() {
    let graph = GraphFixture::Path(4).build();
    let _ =
        DescendingLexicographicScoreSorter::new(DegreeScorer, WrongLengthScorer).sort_nodes(&graph);
}

#[test]
fn test_node_ordering_ground_truth_metadata() {
    let fixture = load_fixture_suite("node_ordering_ground_truth.json.gz");
    assert_eq!(fixture.schema_version, 10);
    assert_eq!(fixture.generator, "networkx");
    assert_eq!(fixture.networkx_version, "3.3");
    assert!(!fixture.python_version.is_empty());
    assert_eq!(fixture.pagerank_rounding_decimals, 12);
    assert_eq!(fixture.katz_rounding_decimals, 12);
    assert_eq!(fixture.betweenness_rounding_decimals, 12);
    assert_eq!(fixture.closeness_rounding_decimals, 12);
    assert_eq!(fixture.local_clustering_rounding_decimals, 12);
    assert_eq!(fixture.cases.len(), 10_000);
    assert!(fixture.cases.iter().all(|case| {
        case.networkx_smallest_last.len() == case.n
            && case.canonical_smallest_last.len() == case.n
            && case.core_numbers.len() == case.n
            && case.degeneracy_degree_descending.len() == case.n
            && case.welsh_powell_descending.len() == case.n
            && case.dsatur_order.len() == case.n
            && case.bfs_from_max_degree.len() == case.n
            && case.dfs_from_max_degree.len() == case.n
            && case.pagerank_scores.len() == case.n
            && case.pagerank_descending.len() == case.n
            && case.katz_scores.len() == case.n
            && case.katz_descending.len() == case.n
            && case.betweenness_scores.len() == case.n
            && case.betweenness_descending.len() == case.n
            && case.closeness_scores.len() == case.n
            && case.closeness_descending.len() == case.n
            && case.triangle_counts.len() == case.n
            && case.triangle_descending.len() == case.n
            && case.local_clustering_scores.len() == case.n
            && case.local_clustering_descending.len() == case.n
            && case.pagerank_alpha > 0.0
            && case.pagerank_alpha < 1.0
            && case.pagerank_max_iter > 0
            && case.pagerank_tol > 0.0
            && case.katz_alpha > 0.0
            && case.katz_beta > 0.0
            && case.katz_max_iter > 0
            && case.katz_tol > 0.0
    }));
    let pagerank_parameter_sets: std::collections::BTreeSet<(u64, usize, u64)> = fixture
        .cases
        .iter()
        .map(|case| {
            (case.pagerank_alpha.to_bits(), case.pagerank_max_iter, case.pagerank_tol.to_bits())
        })
        .collect();
    assert!(
        pagerank_parameter_sets.len() >= 4,
        "pagerank oracle should contain multiple distinct parameter sets"
    );
    let katz_parameter_sets: std::collections::BTreeSet<(u64, u64, usize, u64, bool)> = fixture
        .cases
        .iter()
        .map(|case| {
            (
                case.katz_alpha.to_bits(),
                case.katz_beta.to_bits(),
                case.katz_max_iter,
                case.katz_tol.to_bits(),
                case.katz_normalized,
            )
        })
        .collect();
    assert!(
        katz_parameter_sets.len() >= 4,
        "katz oracle should contain multiple distinct parameter sets"
    );
    let betweenness_parameter_sets: std::collections::BTreeSet<(bool, bool)> = fixture
        .cases
        .iter()
        .map(|case| (case.betweenness_normalized, case.betweenness_endpoints))
        .collect();
    assert_eq!(
        betweenness_parameter_sets.len(),
        4,
        "betweenness oracle should contain all four normalized/endpoints parameter combinations"
    );
    let closeness_parameter_sets: std::collections::BTreeSet<bool> =
        fixture.cases.iter().map(|case| case.closeness_wf_improved).collect();
    assert_eq!(
        closeness_parameter_sets.len(),
        2,
        "closeness oracle should contain both wf_improved parameter values"
    );
}

#[test]
fn test_triangle_count_scorer_ground_truth() {
    let fixture = load_fixture_suite("node_ordering_ground_truth.json.gz");

    for case in fixture.cases {
        let graph = build_undigraph(&case);
        let context = format!("triangle count ground truth {} ({})", case.name, case.family);
        assert_eq!(
            TriangleCountScorer.score_nodes(&graph),
            case.triangle_counts,
            "triangle count scorer ground truth failed for {context}"
        );
    }
}

#[test]
fn test_descending_triangle_count_sorter_ground_truth() {
    let fixture = load_fixture_suite("node_ordering_ground_truth.json.gz");
    let sorter = DescendingScoreSorter::new(TriangleCountScorer);

    for case in fixture.cases {
        let graph = build_undigraph(&case);
        let context = format!("triangle order {} ({})", case.name, case.family);
        assert_eq!(
            sorter.sort_nodes(&graph),
            case.triangle_descending,
            "triangle descending order ground truth failed for {context}"
        );
    }
}

#[test]
fn test_degeneracy_sorter_ground_truth_invariants() {
    let fixture = load_fixture_suite("node_ordering_ground_truth.json.gz");

    for case in fixture.cases {
        let graph = build_undigraph(&case);
        let context = format!("{} ({})", case.name, case.family);
        let order = DegeneracySorter.sort_nodes(&graph);
        assert_is_smallest_last_order(&graph, &order, &context);
    }
}

#[test]
fn test_core_number_scorer_ground_truth() {
    let fixture = load_fixture_suite("node_ordering_ground_truth.json.gz");

    for case in fixture.cases {
        let graph = build_undigraph(&case);
        let context = format!("{} ({})", case.name, case.family);
        assert_eq!(
            CoreNumberScorer.score_nodes(&graph),
            case.core_numbers,
            "core number scorer ground truth failed for {context}"
        );
    }
}

#[test]
fn test_descending_lexicographic_core_degree_sorter_ground_truth() {
    let fixture = load_fixture_suite("node_ordering_ground_truth.json.gz");
    let sorter = DescendingLexicographicScoreSorter::new(CoreNumberScorer, DegreeScorer);

    for case in fixture.cases {
        let graph = build_undigraph(&case);
        let context = format!("{} ({})", case.name, case.family);
        assert_eq!(
            sorter.sort_nodes(&graph),
            case.degeneracy_degree_descending,
            "degeneracy + degree ground truth failed for {context}"
        );
    }
}

#[test]
fn test_welsh_powell_sorter_ground_truth() {
    let fixture = load_fixture_suite("node_ordering_ground_truth.json.gz");
    let sorter = DescendingScoreSorter::new(DegreeScorer);

    for case in fixture.cases {
        let graph = build_undigraph(&case);
        let context = format!("welsh-powell order {} ({})", case.name, case.family);
        assert_eq!(
            sorter.sort_nodes(&graph),
            case.welsh_powell_descending,
            "welsh-powell ground truth failed for {context}"
        );
    }
}

#[test]
fn test_dsatur_sorter_ground_truth() {
    let fixture = load_fixture_suite("node_ordering_ground_truth.json.gz");

    for case in fixture.cases {
        let graph = build_undigraph(&case);
        let context = format!("dsatur order {} ({})", case.name, case.family);
        assert_eq!(
            DsaturSorter.sort_nodes(&graph),
            case.dsatur_order,
            "dsatur ground truth failed for {context}"
        );
    }
}

#[test]
fn test_bfs_from_max_degree_sorter_ground_truth() {
    let fixture = load_fixture_suite("node_ordering_ground_truth.json.gz");
    let sorter = BfsTraversalSorter::new(
        TraversalSeedStrategy::MaxOutDegree,
        TraversalNeighborOrder::NodeIdAscending,
    );

    for case in fixture.cases {
        let graph = build_undigraph(&case);
        let context = format!("bfs-from-max-degree order {} ({})", case.name, case.family);
        assert_eq!(
            sorter.sort_nodes(&graph),
            case.bfs_from_max_degree,
            "bfs-from-max-degree ground truth failed for {context}"
        );
    }
}

#[test]
fn test_dfs_from_max_degree_sorter_ground_truth() {
    let fixture = load_fixture_suite("node_ordering_ground_truth.json.gz");
    let sorter = DfsTraversalSorter::new(
        TraversalSeedStrategy::MaxOutDegree,
        TraversalNeighborOrder::NodeIdAscending,
    );

    for case in fixture.cases {
        let graph = build_undigraph(&case);
        let context = format!("dfs-from-max-degree order {} ({})", case.name, case.family);
        assert_eq!(
            sorter.sort_nodes(&graph),
            case.dfs_from_max_degree,
            "dfs-from-max-degree ground truth failed for {context}"
        );
    }
}

#[test]
fn test_pagerank_scorer_fixtures() {
    for fixture in PAGERANK_FIXTURES {
        let graph = fixture.graph.build();
        let context = format!("pagerank fixture {}", fixture.name);
        assert_scores_close(
            &PageRankScorer::default().score_nodes(&graph),
            fixture.expected_scores,
            1.0e-12,
            &context,
        );
    }
}

#[test]
fn test_descending_pagerank_sorter_fixtures() {
    let sorter = DescendingScoreSorter::new(PageRankScorer::default());

    for fixture in PAGERANK_FIXTURES {
        let graph = fixture.graph.build();
        assert_eq!(
            sorter.sort_nodes(&graph),
            fixture.expected_descending,
            "descending pagerank ordering fixture failed for {}",
            fixture.name
        );
    }
}

#[test]
fn test_pagerank_scorer_ground_truth() {
    let fixture = load_fixture_suite("node_ordering_ground_truth.json.gz");

    for case in fixture.cases {
        let graph = build_undigraph(&case);
        let context = format!("pagerank ground truth {} ({})", case.name, case.family);
        let scorer = PageRankScorerBuilder::default()
            .alpha(case.pagerank_alpha)
            .max_iter(case.pagerank_max_iter)
            .tolerance(case.pagerank_tol)
            .build();
        assert_scores_close(&scorer.score_nodes(&graph), &case.pagerank_scores, 1.0e-12, &context);
    }
}

#[test]
fn test_descending_pagerank_sorter_ground_truth() {
    let fixture = load_fixture_suite("node_ordering_ground_truth.json.gz");

    for case in fixture.cases {
        let graph = build_undigraph(&case);
        let context = format!("pagerank order {} ({})", case.name, case.family);
        let scorer = PageRankScorerBuilder::default()
            .alpha(case.pagerank_alpha)
            .max_iter(case.pagerank_max_iter)
            .tolerance(case.pagerank_tol)
            .build();
        let sorter = DescendingScoreSorter::new(scorer);
        assert_eq!(
            sorter.sort_nodes(&graph),
            case.pagerank_descending,
            "pagerank descending order ground truth failed for {context}"
        );
    }
}

#[test]
fn test_katz_centrality_scorer_fixtures() {
    for fixture in KATZ_FIXTURES {
        let graph = fixture.graph.build();
        let context = format!("katz fixture {}", fixture.name);
        let scorer = KatzCentralityScorerBuilder::default()
            .alpha(fixture.alpha)
            .beta(fixture.beta)
            .max_iter(fixture.max_iter)
            .tolerance(fixture.tolerance)
            .normalized(fixture.normalized)
            .build();
        assert_scores_close(
            &scorer.score_nodes(&graph),
            fixture.expected_scores,
            KATZ_TOLERANCE,
            &context,
        );
    }
}

#[test]
fn test_descending_katz_centrality_sorter_fixtures() {
    for fixture in KATZ_FIXTURES {
        let graph = fixture.graph.build();
        let scorer = KatzCentralityScorerBuilder::default()
            .alpha(fixture.alpha)
            .beta(fixture.beta)
            .max_iter(fixture.max_iter)
            .tolerance(fixture.tolerance)
            .normalized(fixture.normalized)
            .build();
        let sorter = DescendingScoreSorter::new(scorer);
        assert_eq!(
            sorter.sort_nodes(&graph),
            fixture.expected_descending,
            "descending katz centrality ordering fixture failed for {}",
            fixture.name
        );
    }
}

#[test]
fn test_katz_centrality_scorer_ground_truth() {
    let fixture = load_fixture_suite("node_ordering_ground_truth.json.gz");

    for case in fixture.cases {
        let graph = build_undigraph(&case);
        let context = format!("katz ground truth {} ({})", case.name, case.family);
        let scorer = KatzCentralityScorerBuilder::default()
            .alpha(case.katz_alpha)
            .beta(case.katz_beta)
            .max_iter(case.katz_max_iter)
            .tolerance(case.katz_tol)
            .normalized(case.katz_normalized)
            .build();
        assert_scores_close(
            &scorer.score_nodes(&graph),
            &case.katz_scores,
            KATZ_TOLERANCE,
            &context,
        );
    }
}

#[test]
fn test_descending_katz_centrality_sorter_ground_truth() {
    let fixture = load_fixture_suite("node_ordering_ground_truth.json.gz");

    for case in fixture.cases {
        let graph = build_undigraph(&case);
        let context = format!("katz order {} ({})", case.name, case.family);
        let scorer = KatzCentralityScorerBuilder::default()
            .alpha(case.katz_alpha)
            .beta(case.katz_beta)
            .max_iter(case.katz_max_iter)
            .tolerance(case.katz_tol)
            .normalized(case.katz_normalized)
            .build();
        let sorter = DescendingScoreSorter::new(scorer);
        assert_eq!(
            sorter.sort_nodes(&graph),
            case.katz_descending,
            "katz descending order ground truth failed for {context}"
        );
    }
}

#[test]
fn test_betweenness_centrality_scorer_fixtures() {
    for fixture in BETWEENNESS_FIXTURES {
        let graph = fixture.graph.build();
        let context = format!("betweenness fixture {}", fixture.name);
        let scorer = BetweennessCentralityScorerBuilder::default()
            .normalized(fixture.normalized)
            .endpoints(fixture.endpoints)
            .build();
        assert_scores_close(
            &scorer.score_nodes(&graph),
            fixture.expected_scores,
            BETWEENNESS_TOLERANCE,
            &context,
        );
    }
}

#[test]
fn test_descending_betweenness_centrality_sorter_fixtures() {
    for fixture in BETWEENNESS_FIXTURES {
        let graph = fixture.graph.build();
        let scorer = BetweennessCentralityScorerBuilder::default()
            .normalized(fixture.normalized)
            .endpoints(fixture.endpoints)
            .build();
        let sorter = DescendingScoreSorter::new(scorer);
        assert_eq!(
            sorter.sort_nodes(&graph),
            fixture.expected_descending,
            "descending betweenness centrality ordering fixture failed for {}",
            fixture.name
        );
    }
}

#[test]
fn test_betweenness_centrality_scorer_ground_truth() {
    let fixture = load_fixture_suite("node_ordering_ground_truth.json.gz");

    for case in fixture.cases {
        let graph = build_undigraph(&case);
        let context = format!("betweenness ground truth {} ({})", case.name, case.family);
        let scorer = BetweennessCentralityScorerBuilder::default()
            .normalized(case.betweenness_normalized)
            .endpoints(case.betweenness_endpoints)
            .build();
        assert_scores_close(
            &scorer.score_nodes(&graph),
            &case.betweenness_scores,
            BETWEENNESS_TOLERANCE,
            &context,
        );
    }
}

#[test]
fn test_descending_betweenness_centrality_sorter_ground_truth() {
    let fixture = load_fixture_suite("node_ordering_ground_truth.json.gz");

    for case in fixture.cases {
        let graph = build_undigraph(&case);
        let context = format!("betweenness order {} ({})", case.name, case.family);
        let scorer = BetweennessCentralityScorerBuilder::default()
            .normalized(case.betweenness_normalized)
            .endpoints(case.betweenness_endpoints)
            .build();
        let sorter = DescendingScoreSorter::new(scorer);
        assert_eq!(
            sorter.sort_nodes(&graph),
            case.betweenness_descending,
            "betweenness descending order ground truth failed for {context}"
        );
    }
}

#[test]
fn test_closeness_centrality_scorer_fixtures() {
    for fixture in CLOSENESS_FIXTURES {
        let graph = fixture.graph.build();
        let context = format!("closeness fixture {}", fixture.name);
        let scorer =
            ClosenessCentralityScorerBuilder::default().wf_improved(fixture.wf_improved).build();
        assert_scores_close(
            &scorer.score_nodes(&graph),
            fixture.expected_scores,
            CLOSENESS_TOLERANCE,
            &context,
        );
    }
}

#[test]
fn test_descending_closeness_centrality_sorter_fixtures() {
    for fixture in CLOSENESS_FIXTURES {
        let graph = fixture.graph.build();
        let scorer =
            ClosenessCentralityScorerBuilder::default().wf_improved(fixture.wf_improved).build();
        let sorter = DescendingScoreSorter::new(scorer);
        assert_eq!(
            sorter.sort_nodes(&graph),
            fixture.expected_descending,
            "descending closeness centrality ordering fixture failed for {}",
            fixture.name
        );
    }
}

#[test]
fn test_closeness_centrality_scorer_ground_truth() {
    let fixture = load_fixture_suite("node_ordering_ground_truth.json.gz");

    for case in fixture.cases {
        let graph = build_undigraph(&case);
        let context = format!("closeness ground truth {} ({})", case.name, case.family);
        let scorer = ClosenessCentralityScorerBuilder::default()
            .wf_improved(case.closeness_wf_improved)
            .build();
        assert_scores_close(
            &scorer.score_nodes(&graph),
            &case.closeness_scores,
            CLOSENESS_TOLERANCE,
            &context,
        );
    }
}

#[test]
fn test_descending_closeness_centrality_sorter_ground_truth() {
    let fixture = load_fixture_suite("node_ordering_ground_truth.json.gz");

    for case in fixture.cases {
        let graph = build_undigraph(&case);
        let context = format!("closeness order {} ({})", case.name, case.family);
        let scorer = ClosenessCentralityScorerBuilder::default()
            .wf_improved(case.closeness_wf_improved)
            .build();
        let sorter = DescendingScoreSorter::new(scorer);
        assert_eq!(
            sorter.sort_nodes(&graph),
            case.closeness_descending,
            "closeness descending order ground truth failed for {context}"
        );
    }
}

#[test]
fn test_local_clustering_coefficient_scorer_ground_truth() {
    let fixture = load_fixture_suite("node_ordering_ground_truth.json.gz");

    for case in fixture.cases {
        let graph = build_undigraph(&case);
        let context = format!("local clustering ground truth {} ({})", case.name, case.family);
        assert_scores_close(
            &LocalClusteringCoefficientScorer.score_nodes(&graph),
            &case.local_clustering_scores,
            LOCAL_CLUSTERING_TOLERANCE,
            &context,
        );
    }
}

#[test]
fn test_descending_local_clustering_coefficient_sorter_ground_truth() {
    let fixture = load_fixture_suite("node_ordering_ground_truth.json.gz");
    let sorter = DescendingScoreSorter::new(LocalClusteringCoefficientScorer);

    for case in fixture.cases {
        let graph = build_undigraph(&case);
        let context = format!("local clustering order {} ({})", case.name, case.family);
        assert_eq!(
            sorter.sort_nodes(&graph),
            case.local_clustering_descending,
            "local clustering descending order ground truth failed for {context}"
        );
    }
}

#[test]
fn test_local_clustering_coefficient_scorer_fixtures() {
    for fixture in LOCAL_CLUSTERING_FIXTURES {
        let graph = fixture.graph.build();
        let context = format!("local clustering fixture {}", fixture.name);
        assert_scores_close(
            &LocalClusteringCoefficientScorer.score_nodes(&graph),
            fixture.expected_scores,
            LOCAL_CLUSTERING_TOLERANCE,
            &context,
        );
    }
}

#[test]
fn test_descending_local_clustering_coefficient_sorter_fixtures() {
    let sorter = DescendingScoreSorter::new(LocalClusteringCoefficientScorer);

    for fixture in LOCAL_CLUSTERING_FIXTURES {
        let graph = fixture.graph.build();
        assert_eq!(
            sorter.sort_nodes(&graph),
            fixture.expected_descending,
            "descending local clustering ordering fixture failed for {}",
            fixture.name
        );
    }
}

#[test]
fn test_pagerank_builder_defaults_match_default() {
    let scorer = PageRankScorer::builder().build();
    assert_eq!(scorer, PageRankScorer::default());
}

#[test]
fn test_pagerank_builder_custom_parameters_fixture() {
    let graph = GraphFixture::TriangleWithTail.build();
    let scorer =
        PageRankScorerBuilder::default().alpha(0.9).max_iter(200).tolerance(1.0e-10).build();

    assert_scores_close(
        &scorer.score_nodes(&graph),
        &[0.193827828898, 0.193827828898, 0.288684353061, 0.209420682119, 0.114239307024],
        1.0e-12,
        "pagerank builder custom fixture",
    );

    let sorter = DescendingScoreSorter::new(scorer);
    assert_eq!(
        sorter.sort_nodes(&graph),
        [2, 3, 0, 1, 4],
        "pagerank builder custom ordering fixture failed"
    );
}

#[test]
fn test_katz_builder_defaults_match_default() {
    let scorer = KatzCentralityScorer::builder().build();
    assert_eq!(scorer, KatzCentralityScorer::default());
}

#[test]
fn test_katz_safe_alpha_from_max_degree_helper() {
    assert!((KatzCentralityScorer::safe_alpha_from_max_degree(0) - 0.1).abs() < 1.0e-15);
    assert!((KatzCentralityScorer::safe_alpha_from_max_degree(2) - 0.45).abs() < 1.0e-15);
    assert!((KatzCentralityScorer::safe_alpha_from_max_degree(10) - 0.09).abs() < 1.0e-15);
}

#[test]
#[should_panic(expected = "KatzCentralityScorer failed to converge")]
fn test_katz_default_alpha_can_fail_on_dense_graph() {
    let graph = GraphFixture::Complete(20).build();
    let _ = KatzCentralityScorerBuilder::default().max_iter(32).build().score_nodes(&graph);
}

#[test]
fn test_katz_builder_safe_alpha_from_graph_converges_on_dense_graph() {
    let graph = GraphFixture::Complete(20).build();
    let scorer =
        KatzCentralityScorerBuilder::default().safe_alpha_from_graph(&graph).max_iter(1000).build();

    assert_eq!(scorer, KatzCentralityScorerBuilder::default().alpha(0.9 / 19.0).build());
    let safe_scores = scorer.score_nodes(&graph);
    assert_eq!(safe_scores.len(), 20);
    assert!(safe_scores.iter().all(|score| (*score).is_finite()));
}

#[test]
fn test_betweenness_builder_defaults_match_default() {
    let scorer = BetweennessCentralityScorer::builder().build();
    assert_eq!(scorer, BetweennessCentralityScorer::default());
}

#[test]
fn test_closeness_builder_defaults_match_default() {
    let scorer = ClosenessCentralityScorer::builder().build();
    assert_eq!(scorer, ClosenessCentralityScorer::default());
}

#[test]
#[should_panic(expected = "PageRankScorer failed to converge")]
fn test_pagerank_scorer_panics_on_non_convergence() {
    let graph = GraphFixture::Path(4).build();
    let _ = PageRankScorerBuilder::default().max_iter(0).build().score_nodes(&graph);
}

#[test]
fn test_katz_builder_custom_parameters_fixture() {
    let graph = GraphFixture::TriangleWithTail.build();
    let scorer = KatzCentralityScorerBuilder::default()
        .alpha(0.2)
        .beta(1.5)
        .max_iter(1000)
        .tolerance(1.0e-12)
        .normalized(false)
        .build();

    assert_scores_close(
        &scorer.score_nodes(&graph),
        &[2.639563106796, 2.639563106796, 3.058252427184, 2.51213592233, 2.002427184466],
        KATZ_TOLERANCE,
        "katz builder custom fixture",
    );

    let sorter = DescendingScoreSorter::new(scorer);
    assert_eq!(
        sorter.sort_nodes(&graph),
        [2, 0, 1, 3, 4],
        "katz builder custom ordering fixture failed"
    );
}

#[test]
fn test_betweenness_builder_custom_parameters_fixture() {
    let graph = GraphFixture::TriangleWithTail.build();
    let scorer =
        BetweennessCentralityScorerBuilder::default().normalized(false).endpoints(true).build();

    assert_scores_close(
        &scorer.score_nodes(&graph),
        &[4.0, 4.0, 8.0, 7.0, 4.0],
        BETWEENNESS_TOLERANCE,
        "betweenness builder custom fixture",
    );

    let sorter = DescendingScoreSorter::new(scorer);
    assert_eq!(
        sorter.sort_nodes(&graph),
        [2, 3, 0, 1, 4],
        "betweenness builder custom ordering fixture failed"
    );
}

#[test]
fn test_closeness_builder_custom_parameters_fixture() {
    let graph = GraphFixture::PathWithIsolatedNode.build();
    let scorer = ClosenessCentralityScorerBuilder::default().wf_improved(false).build();

    assert_scores_close(
        &scorer.score_nodes(&graph),
        &[0.666666666667, 1.0, 0.666666666667, 0.0],
        CLOSENESS_TOLERANCE,
        "closeness builder custom fixture",
    );

    let sorter = DescendingScoreSorter::new(scorer);
    assert_eq!(
        sorter.sort_nodes(&graph),
        [1, 0, 2, 3],
        "closeness builder custom ordering fixture failed"
    );
}

#[test]
#[should_panic(expected = "KatzCentralityScorer failed to converge")]
fn test_katz_scorer_panics_on_non_convergence() {
    let graph = GraphFixture::Path(4).build();
    let _ = KatzCentralityScorerBuilder::default().max_iter(0).build().score_nodes(&graph);
}
