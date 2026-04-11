//! Integration tests for exact `K_{2,3}` homeomorph detection on undirected
//! graphs.
#![cfg(feature = "std")]

#[path = "support/planarity_fixture.rs"]
mod planarity_fixture;

use geometric_traits::{
    errors::{MonopartiteError, monopartite_graph_error::algorithms::MonopartiteAlgorithmError},
    impls::{CSR2D, SortedVec, SymmetricCSR2D},
    prelude::*,
    traits::{
        EdgesBuilder, K23HomeomorphDetection, K23HomeomorphError, K33HomeomorphDetection,
        OuterplanarityDetection, PlanarityDetection, VocabularyBuilder,
    },
};
use planarity_fixture::{PlanarityFixtureCase, build_undigraph};

fn k23_case(
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
            notes: "K23 homeomorph detection regression/semantic case.".to_string(),
        },
        expected,
    )
}

#[test]
fn test_k23_homeomorph_rejects_self_loops() {
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
        graph.has_k23_homeomorph(),
        Err(MonopartiteError::AlgorithmError(MonopartiteAlgorithmError::K23HomeomorphError(
            K23HomeomorphError::SelfLoopsUnsupported
        )))
    ));
}

#[test]
fn test_k23_homeomorph_semantic_cases() {
    let cases = [
        k23_case("path_four", "tree", 4, &[[0, 1], [1, 2], [2, 3]], false),
        k23_case(
            "diamond_k4_minus_edge",
            "outerplanar_with_chord",
            4,
            &[[0, 1], [0, 2], [1, 2], [1, 3], [2, 3]],
            false,
        ),
        k23_case(
            "k4_complete",
            "outerplanarity_obstruction",
            4,
            &[[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]],
            false,
        ),
        k23_case(
            "k23_complete_bipartite",
            "outerplanarity_obstruction",
            5,
            &[[0, 2], [0, 3], [0, 4], [1, 2], [1, 3], [1, 4]],
            true,
        ),
        k23_case(
            "k33_complete_bipartite",
            "planarity_obstruction",
            6,
            &[[0, 3], [0, 4], [0, 5], [1, 3], [1, 4], [1, 5], [2, 3], [2, 4], [2, 5]],
            true,
        ),
        k23_case(
            "k5_complete",
            "planarity_obstruction",
            5,
            &[[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]],
            true,
        ),
        k23_case(
            "subdivided_k4_edge",
            "k4_subdivision",
            5,
            &[[0, 2], [2, 1], [0, 3], [3, 1], [0, 4], [4, 1], [3, 4]],
            true,
        ),
        k23_case(
            "k5_subdivision_000296",
            "k5_subdivision",
            16,
            &[
                [0, 1],
                [0, 5],
                [0, 6],
                [0, 7],
                [1, 3],
                [1, 4],
                [1, 9],
                [2, 4],
                [2, 5],
                [2, 10],
                [2, 11],
                [3, 4],
                [3, 6],
                [3, 15],
                [4, 8],
                [7, 8],
                [9, 10],
                [11, 12],
                [12, 13],
                [13, 14],
                [14, 15],
            ],
            true,
        ),
    ];

    for (case, expected) in cases {
        let graph = build_undigraph(&case);
        assert_eq!(
            graph.has_k23_homeomorph().unwrap(),
            expected,
            "K23 homeomorph mismatch on {} ({})",
            case.name,
            case.family
        );
    }
}

#[test]
fn test_k23_homeomorph_accepts_theta_regression() {
    let (case, expected) = k23_case(
        "theta_three_length_two_paths",
        "theta",
        5,
        &[[0, 2], [2, 1], [0, 3], [3, 1], [0, 4], [4, 1]],
        true,
    );
    let graph = build_undigraph(&case);

    assert!(
        graph.has_k23_homeomorph().unwrap() == expected,
        "K23 homeomorph should accept {} ({})",
        case.name,
        case.family
    );
}

#[allow(clippy::fn_params_excessive_bools)]
fn assert_combined_topology_surface_case(
    case: &PlanarityFixtureCase,
    expected_planar: bool,
    expected_outerplanar: bool,
    expected_k23: bool,
    expected_k33: bool,
) {
    let graph = build_undigraph(case);
    let is_planar = graph.is_planar().unwrap();
    let is_outerplanar = graph.is_outerplanar().unwrap();
    let has_k23_homeomorph = graph.has_k23_homeomorph().unwrap();
    let has_k33_homeomorph = graph.has_k33_homeomorph().unwrap();

    assert_eq!(is_planar, expected_planar, "planarity mismatch on {}", case.name);
    assert_eq!(is_outerplanar, expected_outerplanar, "outerplanarity mismatch on {}", case.name);
    assert_eq!(has_k23_homeomorph, expected_k23, "K23 mismatch on {}", case.name);
    assert_eq!(has_k33_homeomorph, expected_k33, "K33 mismatch on {}", case.name);
}

#[test]
fn test_k23_homeomorph_accepts_erdos_renyi_0089600_combined_reference_regression() {
    let (case, _) = k23_case(
        "erdos_renyi_0089600",
        "erdos_renyi",
        9,
        &[
            [0, 2],
            [0, 3],
            [0, 4],
            [0, 6],
            [0, 7],
            [1, 2],
            [1, 6],
            [1, 8],
            [2, 6],
            [3, 7],
            [3, 8],
            [4, 5],
            [4, 7],
            [5, 6],
            [5, 7],
            [6, 7],
            [7, 8],
        ],
        true,
    );
    assert_combined_topology_surface_case(&case, true, false, true, false);
}

#[test]
fn test_k23_homeomorph_accepts_erdos_renyi_0152020_combined_reference_regression() {
    let (case, _) = k23_case(
        "erdos_renyi_0152020",
        "erdos_renyi",
        13,
        &[
            [0, 2],
            [0, 4],
            [0, 8],
            [0, 9],
            [0, 11],
            [1, 11],
            [2, 6],
            [2, 8],
            [2, 10],
            [3, 11],
            [4, 9],
            [4, 10],
            [8, 11],
            [8, 12],
            [9, 11],
        ],
        true,
    );
    assert_combined_topology_surface_case(&case, true, false, true, false);
}

#[test]
fn test_k23_homeomorph_accepts_erdos_renyi_0154460_combined_reference_regression() {
    let (case, _) = k23_case(
        "erdos_renyi_0154460",
        "erdos_renyi",
        8,
        &[
            [0, 1],
            [0, 3],
            [0, 4],
            [0, 6],
            [0, 7],
            [1, 3],
            [1, 5],
            [1, 7],
            [2, 7],
            [3, 4],
            [4, 7],
            [5, 6],
            [5, 7],
            [6, 7],
        ],
        true,
    );
    assert_combined_topology_surface_case(&case, true, false, true, false);
}

#[test]
fn test_k23_homeomorph_rejects_erdos_renyi_0268250_combined_reference_regression() {
    let (case, _) = k23_case(
        "erdos_renyi_0268250",
        "erdos_renyi",
        9,
        &[[0, 1], [0, 3], [0, 4], [0, 7], [0, 8], [1, 3], [4, 5], [4, 7], [4, 8], [7, 8]],
        false,
    );
    assert_combined_topology_surface_case(&case, true, false, false, false);
}

#[test]
fn test_k23_homeomorph_rejects_erdos_renyi_0900090_combined_reference_regression() {
    let (case, _) = k23_case(
        "erdos_renyi_0900090",
        "erdos_renyi",
        9,
        &[[0, 4], [1, 2], [2, 3], [3, 7], [4, 5], [4, 7], [4, 8], [5, 7], [5, 8], [7, 8]],
        false,
    );
    assert_combined_topology_surface_case(&case, true, false, false, false);
}

#[test]
fn test_k23_homeomorph_accepts_erdos_renyi_0293070_combined_reference_regression() {
    let (case, _) = k23_case(
        "erdos_renyi_0293070",
        "erdos_renyi",
        12,
        &[
            [0, 4],
            [0, 7],
            [0, 9],
            [0, 11],
            [1, 2],
            [1, 4],
            [1, 7],
            [2, 7],
            [3, 6],
            [3, 7],
            [3, 11],
            [4, 10],
            [5, 6],
            [5, 7],
            [5, 8],
            [5, 11],
            [6, 7],
            [7, 8],
            [7, 9],
            [7, 11],
            [9, 10],
            [9, 11],
            [10, 11],
        ],
        true,
    );
    assert_combined_topology_surface_case(&case, false, false, true, true);
}

#[test]
fn test_k23_homeomorph_accepts_erdos_renyi_0307890_combined_reference_regression() {
    let (case, _) = k23_case(
        "erdos_renyi_0307890",
        "erdos_renyi",
        12,
        &[
            [0, 3],
            [0, 4],
            [0, 5],
            [0, 6],
            [0, 7],
            [1, 4],
            [1, 6],
            [1, 10],
            [2, 3],
            [2, 7],
            [2, 11],
            [3, 4],
            [3, 6],
            [3, 11],
            [4, 5],
            [5, 7],
            [6, 8],
            [6, 10],
            [8, 9],
            [8, 10],
        ],
        true,
    );
    assert_combined_topology_surface_case(&case, true, false, true, false);
}
