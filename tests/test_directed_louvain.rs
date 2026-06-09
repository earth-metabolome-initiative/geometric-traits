//! Tests for the directed Louvain community detection trait.
#![cfg(feature = "std")]

use geometric_traits::{
    impls::ValuedCSR2D,
    prelude::*,
    traits::{LouvainConfig, ModularityError},
};

type WeightedMatrix = ValuedCSR2D<usize, usize, usize, f64>;

/// Builds a directed (possibly asymmetric) weighted adjacency matrix.
fn build_directed_graph(node_count: usize, edges: Vec<(usize, usize, f64)>) -> WeightedMatrix {
    let mut edges = edges;
    edges.sort_unstable_by(|(ls, ld, _), (rs, rd, _)| (ls, ld).cmp(&(rs, rd)));
    GenericEdgesBuilder::<_, WeightedMatrix>::default()
        .expected_number_of_edges(edges.len())
        .expected_shape((node_count, node_count))
        .edges(edges.into_iter())
        .build()
        .unwrap()
}

#[test]
fn test_two_disjoint_directed_cycles_recover_optimum() {
    let graph = build_directed_graph(4, vec![(0, 1, 1.0), (1, 0, 1.0), (2, 3, 1.0), (3, 2, 1.0)]);
    let result =
        DirectedLouvain::<usize>::directed_louvain(&graph, &LouvainConfig::default()).unwrap();
    let partition = result.final_partition();

    assert_eq!(partition[0], partition[1]);
    assert_eq!(partition[2], partition[3]);
    assert_ne!(partition[0], partition[2]);
    assert!((result.final_modularity() - 0.5).abs() < 1.0e-9);
}

#[test]
fn test_final_modularity_matches_directed_modularity_of_partition() {
    let graph = build_directed_graph(
        6,
        vec![
            (0, 1, 3.0),
            (1, 2, 3.0),
            (2, 0, 3.0),
            (3, 4, 3.0),
            (4, 5, 3.0),
            (5, 3, 3.0),
            (2, 3, 1.0),
        ],
    );
    let config = LouvainConfig::default();
    let result = DirectedLouvain::<usize>::directed_louvain(&graph, &config).unwrap();

    let recomputed = DirectedModularity::<usize>::directed_modularity(
        &graph,
        result.final_partition(),
        config.resolution,
    )
    .unwrap();
    assert!((result.final_modularity() - recomputed).abs() < 1.0e-9);
}

#[test]
fn test_modularity_is_non_decreasing_across_levels() {
    let graph = build_directed_graph(
        8,
        vec![
            (0, 1, 4.0),
            (1, 2, 4.0),
            (2, 0, 4.0),
            (3, 4, 4.0),
            (4, 5, 4.0),
            (5, 3, 4.0),
            (6, 7, 4.0),
            (7, 6, 4.0),
            (2, 3, 0.5),
            (5, 6, 0.5),
        ],
    );
    let result =
        DirectedLouvain::<usize>::directed_louvain(&graph, &LouvainConfig::default()).unwrap();

    let levels = result.levels();
    assert!(!levels.is_empty());
    for window in levels.windows(2) {
        assert!(
            window[1].modularity() >= window[0].modularity() - 1.0e-9,
            "modularity decreased: {} -> {}",
            window[0].modularity(),
            window[1].modularity(),
        );
    }
}

#[test]
fn test_directed_louvain_is_deterministic_for_a_fixed_seed() {
    let graph = build_directed_graph(
        6,
        vec![
            (0, 1, 2.0),
            (1, 0, 2.0),
            (1, 2, 2.0),
            (2, 1, 2.0),
            (3, 4, 2.0),
            (4, 5, 2.0),
            (5, 3, 2.0),
            (2, 3, 0.3),
        ],
    );
    let config = LouvainConfig { seed: 7, ..LouvainConfig::default() };
    let first = DirectedLouvain::<usize>::directed_louvain(&graph, &config).unwrap();
    let second = DirectedLouvain::<usize>::directed_louvain(&graph, &config).unwrap();

    assert_eq!(first.final_partition(), second.final_partition());
    assert!((first.final_modularity() - second.final_modularity()).abs() <= 1.0e-12);
}

#[test]
fn test_strongly_asymmetric_coupling_separates() {
    // A dense one-way bundle from cluster {0,1,2} to cluster {3,4,5} should not
    // merge the two clusters, because directed modularity is unimpressed by a
    // purely one-directional cut.
    let mut edges =
        vec![(0, 1, 5.0), (1, 2, 5.0), (2, 0, 5.0), (3, 4, 5.0), (4, 5, 5.0), (5, 3, 5.0)];
    for source in 0..3 {
        for destination in 3..6 {
            edges.push((source, destination, 1.0));
        }
    }
    let graph = build_directed_graph(6, edges);
    let result =
        DirectedLouvain::<usize>::directed_louvain(&graph, &LouvainConfig::default()).unwrap();
    let partition = result.final_partition();

    assert_eq!(partition[0], partition[1]);
    assert_eq!(partition[1], partition[2]);
    assert_eq!(partition[3], partition[4]);
    assert_eq!(partition[4], partition[5]);
    assert_ne!(partition[0], partition[3]);
}

#[test]
fn test_sources_sinks_and_self_loops_are_handled() {
    // Node 0 is a pure source, node 4 a pure sink, node 2 carries a self-loop.
    let graph = build_directed_graph(
        5,
        vec![(0, 1, 1.0), (1, 2, 1.0), (2, 2, 2.0), (2, 3, 1.0), (3, 4, 1.0)],
    );
    let result =
        DirectedLouvain::<usize>::directed_louvain(&graph, &LouvainConfig::default()).unwrap();
    assert_eq!(result.final_partition().len(), 5);
    assert!(result.final_modularity().is_finite());
}

#[test]
fn test_directed_louvain_rejects_invalid_resolution() {
    let graph = build_directed_graph(2, vec![(0, 1, 1.0), (1, 0, 1.0)]);
    let config = LouvainConfig { resolution: 0.0, ..LouvainConfig::default() };
    let error = DirectedLouvain::<usize>::directed_louvain(&graph, &config).unwrap_err();
    assert!(matches!(error, ModularityError::InvalidResolution));
}

#[test]
fn test_directed_louvain_rejects_non_positive_weight() {
    let graph = build_directed_graph(2, vec![(0, 1, 0.0)]);
    let error =
        DirectedLouvain::<usize>::directed_louvain(&graph, &LouvainConfig::default()).unwrap_err();
    assert!(matches!(error, ModularityError::NonPositiveWeight { .. }));
}

#[test]
fn test_directed_louvain_marker_overflow_returns_error() {
    let graph = build_directed_graph(300, Vec::new());
    let error =
        DirectedLouvain::<u8>::directed_louvain(&graph, &LouvainConfig::default()).unwrap_err();
    assert!(matches!(error, ModularityError::TooManyCommunities));
}
