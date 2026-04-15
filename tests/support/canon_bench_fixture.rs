#![cfg(feature = "std")]
#![allow(clippy::pedantic)]

use std::collections::{BTreeMap, BTreeSet};

use geometric_traits::{
    impls::{SortedVec, SymmetricCSR2D, ValuedCSR2D},
    naive_structs::GenericGraph,
    prelude::GenericVocabularyBuilder,
    traits::VocabularyBuilder,
};
use rand::{Rng, SeedableRng, rngs::SmallRng};

type LabeledUndirectedEdges = SymmetricCSR2D<ValuedCSR2D<usize, usize, usize, u8>>;
pub(crate) type LabeledUndirectedGraph = GenericGraph<SortedVec<usize>, LabeledUndirectedEdges>;

#[derive(Clone)]
pub(crate) struct CanonCase {
    pub(crate) name: String,
    pub(crate) graph: LabeledUndirectedGraph,
    pub(crate) vertex_labels: Vec<u8>,
    pub(crate) edges: Vec<(usize, usize, u8)>,
}

impl CanonCase {
    pub(crate) fn number_of_nodes(&self) -> usize {
        self.vertex_labels.len()
    }

    pub(crate) fn number_of_edges(&self) -> usize {
        self.edges.len()
    }
}

pub(crate) fn benchmark_cases() -> Vec<CanonCase> {
    vec![
        path_case("path_n24", 24),
        cycle_case("cycle_n20_alternating", 20),
        ladder_case("ladder_r10", 10),
        complete_bipartite_case("complete_bipartite_4x4", 4, 4),
        random_connected_case("random_sparse_n24_m36", 0xCA10_0001, 24, 36),
        random_connected_case("random_dense_n20_m80", 0xCA10_0002, 20, 80),
    ]
}

pub(crate) fn scaling_cases() -> Vec<CanonCase> {
    vec![
        cycle_case("cycle_n08_alternating", 8),
        cycle_case("cycle_n12_alternating", 12),
        cycle_case("cycle_n16_alternating", 16),
        cycle_case("cycle_n20_alternating", 20),
        ladder_case("ladder_r04", 4),
        ladder_case("ladder_r06", 6),
        ladder_case("ladder_r08", 8),
        ladder_case("ladder_r10", 10),
        random_connected_case("random_sparse_n12_m18", 0xCA10_1001, 12, 18),
        random_connected_case("random_sparse_n18_m30", 0xCA10_1002, 18, 30),
        random_connected_case("random_sparse_n24_m42", 0xCA10_1003, 24, 42),
        random_connected_case("random_sparse_n30_m54", 0xCA10_1004, 30, 54),
    ]
}

pub(crate) fn timeout_cases() -> Vec<CanonCase> {
    vec![
        fuzz_timeout_dense_complete_n31_single_override_16_22_2(),
        fuzz_timeout_dense_complete_n31_profile_55_11(),
        fuzz_timeout_dense_complete_n31_profile_210_21(),
    ]
}

fn build_labeled_graph(
    number_of_nodes: usize,
    vertex_labels: Vec<u8>,
    edges: Vec<(usize, usize, u8)>,
) -> CanonCase {
    assert_eq!(vertex_labels.len(), number_of_nodes);

    let nodes: SortedVec<usize> = GenericVocabularyBuilder::default()
        .expected_number_of_symbols(number_of_nodes)
        .symbols((0..number_of_nodes).enumerate())
        .build()
        .unwrap();

    let mut upper_edges = edges
        .iter()
        .map(|&(source, destination, label)| {
            let (left, right) =
                if source < destination { (source, destination) } else { (destination, source) };
            (left, right, label)
        })
        .collect::<Vec<_>>();
    upper_edges.sort_unstable_by(|left, right| {
        left.0.cmp(&right.0).then(left.1.cmp(&right.1)).then(left.2.cmp(&right.2))
    });
    upper_edges.dedup();

    let graph = GenericGraph::from((
        nodes,
        SymmetricCSR2D::from_sorted_upper_triangular_entries(number_of_nodes, upper_edges.clone())
            .unwrap(),
    ));

    CanonCase { name: String::new(), graph, vertex_labels, edges: upper_edges }
}

fn path_case(name: &str, number_of_nodes: usize) -> CanonCase {
    let vertex_labels =
        (0..number_of_nodes).map(|index| [0_u8, 1, 2, 1][index % 4]).collect::<Vec<_>>();
    let edges = (0..(number_of_nodes - 1))
        .map(|index| (index, index + 1, [1_u8, 2, 1][index % 3]))
        .collect::<Vec<_>>();
    with_name(build_labeled_graph(number_of_nodes, vertex_labels, edges), name)
}

fn cycle_case(name: &str, number_of_nodes: usize) -> CanonCase {
    assert!(number_of_nodes >= 4);
    let vertex_labels =
        (0..number_of_nodes).map(|index| [0_u8, 0, 1, 1][index % 4]).collect::<Vec<_>>();
    let mut edges = (0..(number_of_nodes - 1))
        .map(|index| (index, index + 1, [1_u8, 2][index % 2]))
        .collect::<Vec<_>>();
    edges.push((number_of_nodes - 1, 0, [1_u8, 2][(number_of_nodes - 1) % 2]));
    with_name(build_labeled_graph(number_of_nodes, vertex_labels, edges), name)
}

fn ladder_case(name: &str, rungs: usize) -> CanonCase {
    assert!(rungs >= 2);
    let number_of_nodes = rungs * 2;
    let mut vertex_labels = Vec::with_capacity(number_of_nodes);
    for row in 0..2 {
        for column in 0..rungs {
            vertex_labels.push(if row == 0 {
                [0_u8, 1][column % 2]
            } else {
                [1_u8, 0][column % 2]
            });
        }
    }

    let mut edges = Vec::new();
    for column in 0..rungs {
        edges.push((column, rungs + column, 3_u8));
    }
    for column in 0..(rungs - 1) {
        edges.push((column, column + 1, 1_u8));
        edges.push((rungs + column, rungs + column + 1, 2_u8));
    }

    with_name(build_labeled_graph(number_of_nodes, vertex_labels, edges), name)
}

fn complete_bipartite_case(name: &str, left: usize, right: usize) -> CanonCase {
    let number_of_nodes = left + right;
    let vertex_labels =
        (0..number_of_nodes).map(|index| if index < left { 0_u8 } else { 1_u8 }).collect();
    let mut edges = Vec::with_capacity(left * right);
    for src in 0..left {
        for dst in left..number_of_nodes {
            edges.push((src, dst, 1_u8));
        }
    }
    with_name(build_labeled_graph(number_of_nodes, vertex_labels, edges), name)
}

fn random_connected_case(
    name: &str,
    seed: u64,
    number_of_nodes: usize,
    number_of_edges: usize,
) -> CanonCase {
    assert!(number_of_nodes >= 2);
    assert!(number_of_edges >= number_of_nodes - 1);
    assert!(number_of_edges <= number_of_nodes * (number_of_nodes - 1) / 2);

    let mut rng = SmallRng::seed_from_u64(seed);
    let vertex_labels = (0..number_of_nodes).map(|_| rng.gen_range(0_u8..=3)).collect::<Vec<_>>();
    let mut seen = BTreeSet::new();
    let mut edges = Vec::with_capacity(number_of_edges);

    for source in 0..(number_of_nodes - 1) {
        let label = [1_u8, 2, 3][source % 3];
        seen.insert((source, source + 1));
        edges.push((source, source + 1, label));
    }

    while edges.len() < number_of_edges {
        let mut source = rng.gen_range(0..number_of_nodes);
        let mut destination = rng.gen_range(0..number_of_nodes);
        if source == destination {
            continue;
        }
        if source > destination {
            std::mem::swap(&mut source, &mut destination);
        }
        if !seen.insert((source, destination)) {
            continue;
        }
        edges.push((source, destination, rng.gen_range(1_u8..=3)));
    }

    with_name(build_labeled_graph(number_of_nodes, vertex_labels, edges), name)
}

fn complete_graph_case_with_overrides(
    name: &str,
    vertex_labels: Vec<u8>,
    overrides: &[(usize, usize, u8)],
) -> CanonCase {
    let number_of_nodes = vertex_labels.len();
    let mut override_map = BTreeMap::new();
    for &(source, destination, label) in overrides {
        let (left, right) =
            if source < destination { (source, destination) } else { (destination, source) };
        override_map.insert((left, right), label);
    }

    let mut edges = Vec::with_capacity(number_of_nodes * (number_of_nodes - 1) / 2);
    for source in 0..number_of_nodes {
        for destination in (source + 1)..number_of_nodes {
            let label = override_map.get(&(source, destination)).copied().unwrap_or(1_u8);
            edges.push((source, destination, label));
        }
    }

    with_name(build_labeled_graph(number_of_nodes, vertex_labels, edges), name)
}

fn repeating_triplet_vertex_labels(number_of_nodes: usize) -> Vec<u8> {
    (0..number_of_nodes).map(|index| [3_u8, 3, 1][index % 3]).collect()
}

fn fuzz_timeout_dense_complete_n31_profile_55_11() -> CanonCase {
    complete_graph_case_with_overrides(
        "fuzz_timeout_dense_complete_n31_profile_55_11",
        repeating_triplet_vertex_labels(31),
        &[
            (0, 3, 2),
            (0, 4, 2),
            (2, 5, 2),
            (2, 6, 3),
            (2, 9, 2),
            (2, 10, 0),
            (2, 11, 2),
            (4, 15, 3),
            (4, 16, 2),
            (4, 17, 0),
            (4, 20, 2),
            (4, 21, 2),
            (6, 30, 2),
            (7, 8, 3),
            (7, 11, 2),
            (7, 12, 0),
            (7, 13, 2),
            (9, 27, 3),
            (9, 28, 2),
            (9, 29, 0),
            (10, 12, 2),
            (10, 13, 2),
            (13, 17, 2),
            (13, 18, 3),
            (13, 21, 2),
            (13, 22, 0),
            (13, 23, 2),
            (17, 22, 3),
            (17, 23, 2),
            (17, 24, 0),
            (17, 27, 2),
            (17, 28, 2),
            (23, 29, 2),
            (23, 30, 3),
            (24, 27, 2),
            (24, 28, 0),
            (24, 29, 2),
        ],
    )
}

fn fuzz_timeout_dense_complete_n31_single_override_16_22_2() -> CanonCase {
    complete_graph_case_with_overrides(
        "fuzz_timeout_dense_complete_n31_single_override_16_22_2",
        repeating_triplet_vertex_labels(31),
        &[(16, 22, 2)],
    )
}

fn fuzz_timeout_dense_complete_n31_profile_210_21() -> CanonCase {
    complete_graph_case_with_overrides(
        "fuzz_timeout_dense_complete_n31_profile_210_21",
        repeating_triplet_vertex_labels(31),
        &[
            (0, 3, 2),
            (0, 4, 2),
            (4, 24, 0),
            (6, 25, 2),
            (6, 26, 0),
            (6, 27, 2),
            (16, 19, 2),
            (16, 22, 2),
            (16, 23, 2),
        ],
    )
}

fn with_name(mut case: CanonCase, name: &str) -> CanonCase {
    case.name = name.to_owned();
    case
}
