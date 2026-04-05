//! Fixture-based integration tests for graph-level node ordering and scoring.
#![cfg(feature = "std")]

#[path = "support/node_ordering_fixture.rs"]
mod node_ordering_fixture;

use geometric_traits::{
    impls::{CSR2D, SortedVec, SymmetricCSR2D},
    prelude::*,
    traits::{SquareMatrix, VocabularyBuilder, algorithms::randomized_graphs::*},
};
use node_ordering_fixture::{build_undigraph, load_fixture_suite};

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

#[derive(Clone, Copy, Debug)]
enum GraphFixture {
    Path(usize),
    Cycle(usize),
    Complete(usize),
    Star(usize),
    TriangleWithTail,
}

impl GraphFixture {
    fn build(self) -> UndiGraph<usize> {
        match self {
            Self::Path(n) => wrap_undi(path_graph(n)),
            Self::Cycle(n) => wrap_undi(cycle_graph(n)),
            Self::Complete(n) => wrap_undi(complete_graph(n)),
            Self::Star(n) => wrap_undi(star_graph(n)),
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
        }
    }
}

struct OrderingFixture {
    name: &'static str,
    graph: GraphFixture,
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

const DEGENERACY_FIXTURES: &[OrderingFixture] = &[
    OrderingFixture { name: "path_4", graph: GraphFixture::Path(4) },
    OrderingFixture { name: "star_5", graph: GraphFixture::Star(5) },
    OrderingFixture { name: "complete_4", graph: GraphFixture::Complete(4) },
    OrderingFixture { name: "cycle_4", graph: GraphFixture::Cycle(4) },
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

#[test]
fn test_degeneracy_sorter_fixtures() {
    for fixture in DEGENERACY_FIXTURES {
        let graph = fixture.graph.build();
        let order = DegeneracySorter.sort_nodes(&graph);
        assert_is_smallest_last_order(&graph, &order, fixture.name);
    }
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
fn test_node_ordering_ground_truth_metadata() {
    let fixture = load_fixture_suite("node_ordering_ground_truth.json.gz");
    assert_eq!(fixture.schema_version, 4);
    assert_eq!(fixture.generator, "networkx");
    assert_eq!(fixture.networkx_version, "3.3");
    assert!(!fixture.python_version.is_empty());
    assert_eq!(fixture.pagerank_rounding_decimals, 12);
    assert_eq!(fixture.cases.len(), 10_000);
    assert!(fixture.cases.iter().all(|case| {
        case.networkx_smallest_last.len() == case.n
            && case.canonical_smallest_last.len() == case.n
            && case.core_numbers.len() == case.n
            && case.degeneracy_degree_descending.len() == case.n
            && case.pagerank_scores.len() == case.n
            && case.pagerank_descending.len() == case.n
            && case.pagerank_alpha > 0.0
            && case.pagerank_alpha < 1.0
            && case.pagerank_max_iter > 0
            && case.pagerank_tol > 0.0
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
fn test_pagerank_builder_defaults_match_default() {
    let scorer = PageRankScorerBuilder::default().build();
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
