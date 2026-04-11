//! Integration tests for exact `K_{3,3}` homeomorph detection on undirected
//! graphs.
#![cfg(feature = "std")]

#[path = "support/planarity_fixture.rs"]
mod planarity_fixture;

use geometric_traits::{
    errors::{MonopartiteError, monopartite_graph_error::algorithms::MonopartiteAlgorithmError},
    impls::{CSR2D, SortedVec, SymmetricCSR2D},
    prelude::*,
    traits::{EdgesBuilder, K33HomeomorphDetection, K33HomeomorphError, VocabularyBuilder},
};
use planarity_fixture::{PlanarityFixtureCase, build_undigraph};

fn k33_case(
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
            notes: "K33 homeomorph detection regression/semantic case.".to_string(),
        },
        expected,
    )
}

#[test]
fn test_k33_homeomorph_rejects_self_loops() {
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
        graph.has_k33_homeomorph(),
        Err(MonopartiteError::AlgorithmError(MonopartiteAlgorithmError::K33HomeomorphError(
            K33HomeomorphError::SelfLoopsUnsupported
        )))
    ));
}

#[test]
#[allow(clippy::too_many_lines)]
fn test_k33_homeomorph_semantic_cases() {
    let cases = [
        k33_case("path_four", "tree", 4, &[[0, 1], [1, 2], [2, 3]], false),
        k33_case(
            "k23_complete_bipartite",
            "k23",
            5,
            &[[0, 2], [0, 3], [0, 4], [1, 2], [1, 3], [1, 4]],
            false,
        ),
        k33_case(
            "erdos_renyi_000000",
            "erdos_renyi",
            13,
            &[
                [0, 1],
                [0, 9],
                [0, 10],
                [0, 11],
                [0, 12],
                [1, 7],
                [1, 8],
                [1, 9],
                [2, 6],
                [2, 10],
                [3, 4],
                [4, 6],
                [4, 8],
                [4, 10],
                [4, 12],
                [5, 12],
                [6, 9],
                [7, 8],
                [7, 12],
                [8, 10],
                [9, 10],
                [9, 12],
                [10, 11],
                [10, 12],
                [11, 12],
            ],
            true,
        ),
        k33_case(
            "erdos_renyi_001692",
            "erdos_renyi",
            14,
            &[
                [0, 2],
                [0, 5],
                [0, 6],
                [0, 7],
                [0, 10],
                [0, 13],
                [1, 6],
                [1, 10],
                [1, 12],
                [2, 3],
                [2, 6],
                [2, 7],
                [2, 10],
                [3, 11],
                [3, 13],
                [5, 7],
                [5, 9],
                [5, 12],
                [5, 13],
                [6, 8],
                [7, 10],
                [8, 12],
                [11, 12],
            ],
            true,
        ),
        k33_case(
            "erdos_renyi_007290",
            "erdos_renyi",
            9,
            &[
                [0, 1],
                [0, 2],
                [0, 4],
                [0, 5],
                [0, 7],
                [0, 8],
                [1, 2],
                [1, 4],
                [1, 7],
                [2, 5],
                [2, 6],
                [3, 4],
                [3, 5],
                [3, 8],
                [4, 5],
                [5, 7],
                [5, 8],
                [7, 8],
            ],
            true,
        ),
        k33_case(
            "erdos_renyi_017523",
            "erdos_renyi",
            7,
            &[
                [0, 1],
                [0, 2],
                [0, 6],
                [1, 2],
                [1, 3],
                [1, 4],
                [1, 6],
                [2, 3],
                [2, 4],
                [2, 5],
                [3, 6],
                [4, 6],
                [5, 6],
            ],
            true,
        ),
        k33_case(
            "erdos_renyi_0237330",
            "erdos_renyi",
            7,
            &[
                [0, 1],
                [0, 3],
                [0, 4],
                [0, 5],
                [1, 2],
                [1, 3],
                [1, 4],
                [1, 5],
                [2, 5],
                [3, 4],
                [3, 5],
                [3, 6],
                [4, 5],
                [4, 6],
                [5, 6],
            ],
            true,
        ),
        k33_case(
            "erdos_renyi_0199830",
            "erdos_renyi",
            10,
            &[
                [0, 1],
                [0, 2],
                [0, 5],
                [0, 7],
                [1, 2],
                [1, 5],
                [1, 7],
                [2, 4],
                [2, 6],
                [2, 7],
                [4, 7],
                [4, 9],
                [5, 7],
                [5, 9],
                [6, 9],
                [7, 9],
                [8, 9],
            ],
            true,
        ),
        k33_case(
            "k33_complete_bipartite",
            "k33",
            6,
            &[[0, 3], [0, 4], [0, 5], [1, 3], [1, 4], [1, 5], [2, 3], [2, 4], [2, 5]],
            true,
        ),
        k33_case(
            "k33_subdivision",
            "k33_subdivision",
            15,
            &[
                [0, 6],
                [6, 3],
                [0, 7],
                [7, 4],
                [0, 8],
                [8, 5],
                [1, 9],
                [9, 3],
                [1, 10],
                [10, 4],
                [1, 11],
                [11, 5],
                [2, 12],
                [12, 3],
                [2, 13],
                [13, 4],
                [2, 14],
                [14, 5],
            ],
            true,
        ),
        k33_case(
            "k5_complete",
            "k5",
            5,
            &[[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]],
            false,
        ),
        k33_case(
            "k6_complete",
            "clique",
            6,
            &[
                [0, 1],
                [0, 2],
                [0, 3],
                [0, 4],
                [0, 5],
                [1, 2],
                [1, 3],
                [1, 4],
                [1, 5],
                [2, 3],
                [2, 4],
                [2, 5],
                [3, 4],
                [3, 5],
                [4, 5],
            ],
            true,
        ),
        k33_case(
            "wheel_seven",
            "planar",
            7,
            &[
                [0, 1],
                [0, 2],
                [0, 3],
                [0, 4],
                [0, 5],
                [0, 6],
                [1, 2],
                [2, 3],
                [3, 4],
                [4, 5],
                [5, 6],
                [1, 6],
            ],
            false,
        ),
    ];

    for (case, expected) in cases {
        let graph = build_undigraph(&case);
        assert_eq!(
            graph.has_k33_homeomorph().unwrap(),
            expected,
            "K33 homeomorph mismatch on {} ({})",
            case.name,
            case.family
        );
    }
}
