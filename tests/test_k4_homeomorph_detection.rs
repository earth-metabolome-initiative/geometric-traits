//! Integration tests for exact `K_4` homeomorph detection on undirected
//! graphs.
#![cfg(feature = "std")]

#[path = "support/planarity_fixture.rs"]
mod planarity_fixture;

use geometric_traits::{
    errors::{MonopartiteError, monopartite_graph_error::algorithms::MonopartiteAlgorithmError},
    impls::{CSR2D, SortedVec, SymmetricCSR2D},
    prelude::*,
    traits::{EdgesBuilder, K4HomeomorphDetection, K4HomeomorphError, VocabularyBuilder},
};
use planarity_fixture::{PlanarityFixtureCase, build_undigraph};

fn k4_case(
    name: &str,
    family: &str,
    node_count: usize,
    edges: &[[usize; 2]],
    expected: bool,
) -> (PlanarityFixtureCase, bool) {
    (
        PlanarityFixtureCase {
            name: name.to_string(),
            family: family.to_string(),
            node_count,
            edges: edges.to_vec(),
            is_planar: true,
            is_outerplanar: false,
            planarity_obstruction_family: None,
            outerplanarity_obstruction_family: None,
            notes: "K4 homeomorph detection regression/semantic case.".to_string(),
        },
        expected,
    )
}

#[test]
fn test_k4_homeomorph_rejects_self_loops() {
    let nodes: SortedVec<usize> = GenericVocabularyBuilder::default()
        .expected_number_of_symbols(3)
        .symbols((0..3).enumerate())
        .build()
        .unwrap();
    let edges: SymmetricCSR2D<CSR2D<usize, usize, usize>> = UndiEdgesBuilder::default()
        .expected_number_of_edges(2)
        .expected_shape(3)
        .edges([(0usize, 0usize), (0usize, 1usize)].into_iter())
        .build()
        .unwrap();
    let graph: UndiGraph<usize> = UndiGraph::from((nodes, edges));

    assert!(matches!(
        graph.has_k4_homeomorph(),
        Err(MonopartiteError::AlgorithmError(MonopartiteAlgorithmError::K4HomeomorphError(
            K4HomeomorphError::SelfLoopsUnsupported
        )))
    ));
}

#[test]
#[allow(clippy::too_many_lines)]
fn test_k4_homeomorph_semantic_cases() {
    let cases = [
        k4_case("path_four", "tree", 4, &[[0, 1], [1, 2], [2, 3]], false),
        k4_case(
            "theta_three_length_two_paths",
            "theta",
            5,
            &[[0, 2], [2, 1], [0, 3], [3, 1], [0, 4], [4, 1]],
            false,
        ),
        k4_case(
            "diamond_k4_minus_edge",
            "outerplanar_with_chord",
            4,
            &[[0, 1], [0, 2], [1, 2], [1, 3], [2, 3]],
            false,
        ),
        k4_case(
            "k23_complete_bipartite",
            "outerplanarity_obstruction",
            5,
            &[[0, 2], [0, 3], [0, 4], [1, 2], [1, 3], [1, 4]],
            false,
        ),
        k4_case(
            "k4_complete",
            "outerplanarity_obstruction",
            4,
            &[[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]],
            true,
        ),
        k4_case(
            "k5_complete",
            "planarity_obstruction",
            5,
            &[[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]],
            true,
        ),
        k4_case(
            "k4_subdivision_simple",
            "k4_subdivision",
            5,
            &[[0, 4], [4, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]],
            true,
        ),
        k4_case(
            "erdos_renyi_0001210",
            "erdos_renyi",
            7,
            &[[0, 1], [0, 4], [0, 6], [1, 2], [1, 3], [1, 5], [2, 6], [3, 4], [4, 6], [5, 6]],
            true,
        ),
        k4_case(
            "erdos_renyi_0025860",
            "erdos_renyi",
            15,
            &[
                [0, 4],
                [0, 6],
                [0, 11],
                [1, 8],
                [1, 10],
                [2, 7],
                [3, 4],
                [3, 11],
                [4, 8],
                [4, 11],
                [6, 7],
                [6, 8],
                [6, 12],
                [7, 10],
                [8, 11],
                [8, 12],
                [9, 12],
                [9, 13],
                [9, 14],
                [11, 13],
                [11, 14],
            ],
            true,
        ),
        k4_case(
            "erdos_renyi_0048200",
            "erdos_renyi",
            16,
            &[
                [0, 5],
                [0, 7],
                [0, 10],
                [1, 7],
                [1, 14],
                [2, 4],
                [2, 14],
                [5, 10],
                [5, 13],
                [5, 15],
                [6, 9],
                [7, 10],
                [7, 11],
                [7, 12],
                [7, 13],
                [8, 9],
                [8, 10],
                [8, 14],
                [9, 11],
                [10, 13],
                [12, 14],
            ],
            true,
        ),
        k4_case(
            "erdos_renyi_0078570",
            "erdos_renyi",
            16,
            &[
                [0, 7],
                [0, 12],
                [0, 14],
                [1, 9],
                [1, 15],
                [2, 4],
                [2, 11],
                [3, 10],
                [3, 12],
                [3, 15],
                [4, 5],
                [4, 7],
                [4, 13],
                [5, 13],
                [6, 9],
                [6, 10],
                [6, 14],
                [6, 15],
                [7, 11],
                [7, 12],
                [8, 10],
                [8, 12],
                [9, 15],
                [11, 15],
                [13, 14],
            ],
            true,
        ),
        k4_case(
            "erdos_renyi_0094300",
            "erdos_renyi",
            10,
            &[
                [0, 2],
                [0, 4],
                [0, 6],
                [0, 8],
                [1, 7],
                [1, 9],
                [2, 6],
                [3, 5],
                [3, 7],
                [3, 9],
                [4, 6],
                [4, 7],
                [5, 8],
                [5, 9],
                [6, 8],
                [6, 9],
                [8, 9],
            ],
            true,
        ),
        k4_case(
            "erdos_renyi_0185170",
            "erdos_renyi",
            16,
            &[
                [0, 10],
                [0, 11],
                [1, 2],
                [1, 10],
                [2, 8],
                [2, 12],
                [2, 15],
                [3, 5],
                [3, 12],
                [4, 10],
                [5, 9],
                [5, 10],
                [5, 13],
                [6, 7],
                [6, 11],
                [7, 11],
                [7, 12],
                [7, 13],
                [9, 12],
                [10, 14],
            ],
            true,
        ),
        k4_case(
            "erdos_renyi_0397370",
            "erdos_renyi",
            14,
            &[
                [0, 4],
                [0, 5],
                [1, 5],
                [1, 6],
                [1, 9],
                [2, 4],
                [3, 4],
                [3, 9],
                [3, 10],
                [3, 13],
                [4, 10],
                [6, 7],
                [7, 8],
                [7, 10],
                [7, 11],
                [7, 12],
                [7, 13],
                [8, 11],
                [8, 13],
                [9, 13],
                [12, 13],
            ],
            true,
        ),
        k4_case(
            "erdos_renyi_0517930",
            "erdos_renyi",
            8,
            &[
                [0, 2],
                [0, 3],
                [0, 5],
                [1, 3],
                [1, 4],
                [1, 7],
                [2, 3],
                [2, 5],
                [3, 4],
                [3, 6],
                [3, 7],
                [5, 7],
                [6, 7],
            ],
            true,
        ),
        k4_case(
            "erdos_renyi_0562680",
            "erdos_renyi",
            13,
            &[
                [0, 9],
                [0, 10],
                [0, 12],
                [1, 8],
                [1, 9],
                [2, 3],
                [2, 4],
                [3, 6],
                [3, 8],
                [3, 9],
                [3, 11],
                [3, 12],
                [4, 5],
                [5, 9],
                [6, 10],
                [6, 11],
                [7, 10],
                [8, 9],
            ],
            false,
        ),
    ];

    for (case, expected) in cases {
        let graph = build_undigraph(&case);
        assert_eq!(
            graph.has_k4_homeomorph().unwrap(),
            expected,
            "K4 homeomorph mismatch on {} ({})",
            case.name,
            case.family
        );
    }
}

fn assert_k4_case(case: &PlanarityFixtureCase, expected: bool) {
    let graph = build_undigraph(case);
    assert_eq!(
        graph.has_k4_homeomorph().unwrap(),
        expected,
        "K4 homeomorph mismatch on {} ({})",
        case.name,
        case.family
    );
}

#[test]
fn test_k4_homeomorph_accepts_erdos_renyi_0397370_regression() {
    let (case, expected) = k4_case(
        "erdos_renyi_0397370",
        "erdos_renyi",
        14,
        &[
            [0, 4],
            [0, 5],
            [1, 5],
            [1, 6],
            [1, 9],
            [2, 4],
            [3, 4],
            [3, 9],
            [3, 10],
            [3, 13],
            [4, 10],
            [6, 7],
            [7, 8],
            [7, 10],
            [7, 11],
            [7, 12],
            [7, 13],
            [8, 11],
            [8, 13],
            [9, 13],
            [12, 13],
        ],
        true,
    );
    assert_k4_case(&case, expected);
}

#[test]
fn test_k4_homeomorph_accepts_erdos_renyi_0517930_regression() {
    let (case, expected) = k4_case(
        "erdos_renyi_0517930",
        "erdos_renyi",
        8,
        &[
            [0, 2],
            [0, 3],
            [0, 5],
            [1, 3],
            [1, 4],
            [1, 7],
            [2, 3],
            [2, 5],
            [3, 4],
            [3, 6],
            [3, 7],
            [5, 7],
            [6, 7],
        ],
        true,
    );
    assert_k4_case(&case, expected);
}

#[test]
fn test_k4_homeomorph_rejects_erdos_renyi_0562680_regression() {
    let (case, expected) = k4_case(
        "erdos_renyi_0562680",
        "erdos_renyi",
        13,
        &[
            [0, 9],
            [0, 10],
            [0, 12],
            [1, 8],
            [1, 9],
            [2, 3],
            [2, 4],
            [3, 6],
            [3, 8],
            [3, 9],
            [3, 11],
            [3, 12],
            [4, 5],
            [5, 9],
            [6, 10],
            [6, 11],
            [7, 10],
            [8, 9],
        ],
        false,
    );
    assert_k4_case(&case, expected);
}
