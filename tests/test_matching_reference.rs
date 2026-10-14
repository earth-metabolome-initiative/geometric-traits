//! Shared reference-corpus tests for maximum matching algorithms.
#![cfg(feature = "std")]

#[path = "support/max_matching_oracle.rs"]
mod max_matching_oracle;

use geometric_traits::{
    impls::{CSR2D, SymmetricCSR2D},
    prelude::*,
};

type TestGraph = SymmetricCSR2D<CSR2D<usize, usize, usize>>;

#[derive(Debug)]
struct ReferenceCase {
    name: &'static str,
    n: usize,
    edges: Vec<(usize, usize)>,
}

fn case(name: &'static str, n: usize, edges: &[(usize, usize)]) -> ReferenceCase {
    ReferenceCase { name, n, edges: edges.to_vec() }
}

fn build_graph(n: usize, edges: &[(usize, usize)]) -> TestGraph {
    let mut sorted_edges = edges.to_vec();
    sorted_edges.sort_unstable();
    UndiEdgesBuilder::default()
        .expected_number_of_edges(sorted_edges.len())
        .expected_shape(n)
        .edges(sorted_edges.into_iter())
        .build()
        .unwrap()
}

fn validate_matching(
    algorithm: &str,
    case_name: &str,
    matrix: &impl SparseSquareMatrix<Index = usize>,
    matching: &[(usize, usize)],
    expected_size: usize,
) {
    assert_eq!(
        matching.len(),
        expected_size,
        "{algorithm} size mismatch on reference case {case_name}"
    );
    let mut used = vec![false; matrix.order()];
    for &(u, v) in matching {
        assert!(u < v, "{algorithm} returned unordered edge ({u}, {v}) on {case_name}");
        assert!(
            matrix.has_entry(u, v),
            "{algorithm} returned missing edge ({u}, {v}) on {case_name}"
        );
        assert!(!used[u], "{algorithm} reused vertex {u} on {case_name}");
        assert!(!used[v], "{algorithm} reused vertex {v} on {case_name}");
        used[u] = true;
        used[v] = true;
    }
}

fn core_reference_cases() -> Vec<ReferenceCase> {
    vec![
        case("triangle", 3, &[(0, 1), (0, 2), (1, 2)]),
        case("path_p5", 5, &[(0, 1), (1, 2), (2, 3), (3, 4)]),
        case("k4", 4, &[(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]),
        case(
            "petersen",
            10,
            &[
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 4),
                (0, 4),
                (0, 5),
                (1, 6),
                (2, 7),
                (3, 8),
                (4, 9),
                (5, 7),
                (5, 8),
                (6, 8),
                (6, 9),
                (7, 9),
            ],
        ),
        case("blossom_required", 6, &[(0, 1), (0, 2), (1, 2), (2, 3), (3, 4), (4, 5)]),
        case("two_blossoms", 6, &[(0, 1), (0, 2), (1, 2), (2, 3), (3, 4), (3, 5), (4, 5)]),
        case("hexagon", 6, &[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 5)]),
    ]
}

fn blossom_reference_cases() -> Vec<ReferenceCase> {
    vec![
        case(
            "complete_k5",
            5,
            &[(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)],
        ),
        case(
            "nested_blossom",
            7,
            &[(0, 1), (0, 2), (1, 2), (2, 3), (2, 4), (3, 4), (4, 5), (5, 6)],
        ),
        case(
            "nested_blossom_deep",
            10,
            &[
                (0, 1),
                (0, 2),
                (1, 2),
                (2, 3),
                (3, 4),
                (3, 5),
                (4, 5),
                (5, 6),
                (6, 7),
                (6, 8),
                (7, 8),
                (8, 9),
            ],
        ),
        case(
            "wheel_w5",
            6,
            &[(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (1, 2), (2, 3), (3, 4), (4, 5), (1, 5)],
        ),
        case("barbell", 7, &[(0, 1), (0, 2), (1, 2), (2, 3), (3, 4), (4, 5), (4, 6), (5, 6)]),
        case(
            "cube_q3",
            8,
            &[
                (0, 1),
                (0, 2),
                (0, 4),
                (1, 3),
                (1, 5),
                (2, 3),
                (2, 6),
                (3, 7),
                (4, 5),
                (4, 6),
                (5, 7),
                (6, 7),
            ],
        ),
        case(
            "friendship",
            7,
            &[(0, 1), (0, 2), (1, 2), (0, 3), (0, 4), (3, 4), (0, 5), (0, 6), (5, 6)],
        ),
    ]
}

fn generated_reference_cases() -> Vec<ReferenceCase> {
    let mut complete_k7 = Vec::new();
    for i in 0..7 {
        for j in (i + 1)..7 {
            complete_k7.push((i, j));
        }
    }

    let mut complete_bipartite_k33 = Vec::new();
    for i in 0..3 {
        for j in 3..6 {
            complete_bipartite_k33.push((i, j));
        }
    }

    vec![
        ReferenceCase { name: "complete_k7", n: 7, edges: complete_k7 },
        ReferenceCase { name: "k33", n: 6, edges: complete_bipartite_k33 },
    ]
}

fn reference_cases() -> Vec<ReferenceCase> {
    let mut cases = core_reference_cases();
    cases.extend(blossom_reference_cases());
    cases.extend(generated_reference_cases());
    cases
}

fn assert_reference_corpus_matches(
    algorithm: &str,
    solve: impl Fn(&TestGraph) -> Vec<(usize, usize)>,
) {
    for case in reference_cases() {
        let graph = build_graph(case.n, &case.edges);
        let matching = solve(&graph);
        let oracle_size = max_matching_oracle::maximum_matching_size(case.n, &case.edges);
        validate_matching(algorithm, case.name, &graph, &matching, oracle_size);
    }
}

#[test]
fn test_blossom_reference_corpus() {
    assert_reference_corpus_matches("blossom", Blossom::blossom);
}

#[test]
fn test_micali_vazirani_reference_corpus() {
    assert_reference_corpus_matches("micali_vazirani", |graph| graph.micali_vazirani().unwrap());
}

#[test]
fn test_gabow_1976_reference_corpus() {
    assert_reference_corpus_matches("gabow_1976", Gabow1976::gabow_1976);
}
