//! Regression tests against external Louvain/Leiden reference implementations.
#![cfg(feature = "std")]

use std::{collections::HashMap, sync::OnceLock};

use geometric_traits::{
    impls::ValuedCSR2D,
    prelude::*,
    traits::{LeidenConfig, LouvainConfig},
};
use serde::Deserialize;

type WeightedMatrix = ValuedCSR2D<usize, usize, usize, f64>;

const GROUND_TRUTH_JSON: &str = include_str!("fixtures/modularity_ground_truth.json");

#[derive(Debug, Deserialize)]
struct GroundTruthFixture {
    schema_version: u32,
    parameters: GroundTruthParameters,
    cases: Vec<GroundTruthCase>,
}

#[derive(Debug, Deserialize)]
struct GroundTruthParameters {
    resolution: f64,
    seed: u64,
    leiden_iterations: i64,
}

#[derive(Debug, Deserialize)]
struct GroundTruthCase {
    id: String,
    node_count: usize,
    undirected_edges: Vec<GroundTruthEdge>,
    louvain: GroundTruthResult,
    leiden: GroundTruthResult,
}

#[derive(Debug, Deserialize)]
struct GroundTruthEdge {
    source: usize,
    target: usize,
    weight: f64,
}

#[derive(Debug, Deserialize)]
struct GroundTruthResult {
    partition: Vec<usize>,
    modularity: f64,
}

fn fixture() -> &'static GroundTruthFixture {
    static FIXTURE: OnceLock<GroundTruthFixture> = OnceLock::new();
    FIXTURE.get_or_init(|| {
        serde_json::from_str(GROUND_TRUTH_JSON)
            .expect("`tests/fixtures/modularity_ground_truth.json` must be valid JSON")
    })
}

fn build_weighted_graph(
    node_count: usize,
    directed_edges: Vec<(usize, usize, f64)>,
) -> WeightedMatrix {
    let mut edges = directed_edges;
    edges.sort_unstable_by(
        |(left_source, left_destination, _), (right_source, right_destination, _)| {
            (left_source, left_destination).cmp(&(right_source, right_destination))
        },
    );

    GenericEdgesBuilder::<_, WeightedMatrix>::default()
        .expected_number_of_edges(edges.len())
        .expected_shape((node_count, node_count))
        .edges(edges.into_iter())
        .build()
        .unwrap()
}

fn build_undirected_weighted_graph(case: &GroundTruthCase) -> WeightedMatrix {
    let mut directed_edges = Vec::with_capacity(case.undirected_edges.len() * 2);
    for edge in &case.undirected_edges {
        directed_edges.push((edge.source, edge.target, edge.weight));
        if edge.source != edge.target {
            directed_edges.push((edge.target, edge.source, edge.weight));
        }
    }
    build_weighted_graph(case.node_count, directed_edges)
}

fn partitions_are_equivalent(expected: &[usize], actual: &[usize]) -> bool {
    if expected.len() != actual.len() {
        return false;
    }

    let mut expected_to_actual: HashMap<usize, usize> = HashMap::new();
    let mut actual_to_expected: HashMap<usize, usize> = HashMap::new();

    for (expected_label, actual_label) in expected.iter().copied().zip(actual.iter().copied()) {
        if let Some(mapped_actual) = expected_to_actual.get(&expected_label) {
            if *mapped_actual != actual_label {
                return false;
            }
            continue;
        }

        if let Some(mapped_expected) = actual_to_expected.get(&actual_label) {
            if *mapped_expected != expected_label {
                return false;
            }
            continue;
        }

        expected_to_actual.insert(expected_label, actual_label);
        actual_to_expected.insert(actual_label, expected_label);
    }

    true
}

fn approx_equal(left: f64, right: f64, relative_tolerance: f64) -> bool {
    let scale = left.abs().max(right.abs()).max(1.0);
    (left - right).abs() <= scale * relative_tolerance
}

fn partition_communities_are_connected(graph: &WeightedMatrix, partition: &[usize]) -> bool {
    let node_count = partition.len();
    if node_count == 0 {
        return true;
    }

    let number_of_communities =
        partition.iter().copied().max().map_or(0usize, |max| max.saturating_add(1));
    let mut nodes_per_community: Vec<Vec<usize>> = vec![Vec::new(); number_of_communities];
    for (node, community) in partition.iter().copied().enumerate() {
        nodes_per_community[community].push(node);
    }

    let mut is_member = vec![false; node_count];
    let mut visited = vec![false; node_count];
    let mut stack = Vec::new();

    for nodes in nodes_per_community {
        if nodes.len() <= 1 {
            continue;
        }

        for node in &nodes {
            is_member[*node] = true;
        }

        stack.clear();
        let start = nodes[0];
        stack.push(start);
        visited[start] = true;

        let mut visited_count = 0usize;
        while let Some(node) = stack.pop() {
            visited_count += 1;
            for destination in graph.sparse_row(node) {
                if destination < node_count && is_member[destination] && !visited[destination] {
                    visited[destination] = true;
                    stack.push(destination);
                }
            }
        }

        if visited_count != nodes.len() {
            return false;
        }

        for node in &nodes {
            is_member[*node] = false;
            visited[*node] = false;
        }
    }

    true
}

#[test]
fn test_ground_truth_fixture_metadata() {
    let fixture = fixture();
    assert_eq!(fixture.schema_version, 1);
    assert!(approx_equal(fixture.parameters.resolution, 1.0, 1.0e-12));
    assert_eq!(fixture.parameters.seed, 42);
    assert_eq!(fixture.parameters.leiden_iterations, -1);
    assert!(!fixture.cases.is_empty());
}

#[test]
fn test_louvain_matches_ground_truth_fixture() {
    let fixture = fixture();
    let config = LouvainConfig {
        resolution: fixture.parameters.resolution,
        seed: fixture.parameters.seed,
        ..LouvainConfig::default()
    };

    for case in &fixture.cases {
        let graph = build_undirected_weighted_graph(case);
        let result = Louvain::<usize>::louvain(&graph, &config).unwrap();
        let final_partition = result.final_partition();

        assert_eq!(final_partition.len(), case.node_count, "case `{}` partition length", case.id);
        assert_eq!(
            case.louvain.partition.len(),
            case.node_count,
            "case `{}` fixture length",
            case.id
        );
        assert!(
            partitions_are_equivalent(&case.louvain.partition, final_partition),
            "case `{}` partition mismatch: expected {:?}, got {:?}",
            case.id,
            case.louvain.partition,
            final_partition,
        );
        assert!(
            approx_equal(case.louvain.modularity, result.final_modularity(), 1.0e-8),
            "case `{}` modularity mismatch: expected {:.12}, got {:.12}",
            case.id,
            case.louvain.modularity,
            result.final_modularity(),
        );
    }
}

#[test]
fn test_leiden_matches_ground_truth_fixture() {
    let fixture = fixture();
    let config = LeidenConfig {
        resolution: fixture.parameters.resolution,
        seed: fixture.parameters.seed,
        ..LeidenConfig::default()
    };

    for case in &fixture.cases {
        let graph = build_undirected_weighted_graph(case);
        let result = Leiden::<usize>::leiden(&graph, &config).unwrap();
        let final_partition = result.final_partition();

        assert_eq!(final_partition.len(), case.node_count, "case `{}` partition length", case.id);
        assert_eq!(
            case.leiden.partition.len(),
            case.node_count,
            "case `{}` fixture length",
            case.id
        );
        assert!(
            partitions_are_equivalent(&case.leiden.partition, final_partition),
            "case `{}` partition mismatch: expected {:?}, got {:?}",
            case.id,
            case.leiden.partition,
            final_partition,
        );
        assert!(
            approx_equal(case.leiden.modularity, result.final_modularity(), 1.0e-8),
            "case `{}` modularity mismatch: expected {:.12}, got {:.12}",
            case.id,
            case.leiden.modularity,
            result.final_modularity(),
        );
        assert!(
            partition_communities_are_connected(&graph, final_partition),
            "case `{}` produced a disconnected Leiden community",
            case.id,
        );
    }
}
