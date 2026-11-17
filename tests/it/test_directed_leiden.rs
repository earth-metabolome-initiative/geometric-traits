//! Tests for the directed Leiden community detection trait.
#![cfg(feature = "std")]

use geometric_traits::{
    impls::ValuedCSR2D,
    prelude::*,
    traits::{LeidenConfig, ModularityError},
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

/// Returns whether every community induces a weakly connected subgraph (the
/// Leiden well-connectedness guarantee, with arc direction ignored).
fn communities_are_weakly_connected(
    graph: &WeightedMatrix,
    node_count: usize,
    partition: &[usize],
) -> bool {
    if node_count == 0 {
        return true;
    }

    // Build an undirected adjacency from the directed arcs.
    let mut undirected: Vec<Vec<usize>> = vec![Vec::new(); node_count];
    for source in 0..node_count {
        for destination in graph.sparse_row(source) {
            if destination < node_count && destination != source {
                undirected[source].push(destination);
                undirected[destination].push(source);
            }
        }
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
        for &node in &nodes {
            is_member[node] = true;
        }

        stack.clear();
        stack.push(nodes[0]);
        visited[nodes[0]] = true;
        let mut visited_count = 0usize;
        while let Some(node) = stack.pop() {
            visited_count += 1;
            for &neighbor in &undirected[node] {
                if is_member[neighbor] && !visited[neighbor] {
                    visited[neighbor] = true;
                    stack.push(neighbor);
                }
            }
        }

        if visited_count != nodes.len() {
            return false;
        }
        for &node in &nodes {
            is_member[node] = false;
            visited[node] = false;
        }
    }

    true
}

#[test]
fn test_two_disjoint_directed_cycles_recover_optimum() {
    let graph = build_directed_graph(4, vec![(0, 1, 1.0), (1, 0, 1.0), (2, 3, 1.0), (3, 2, 1.0)]);
    let result =
        DirectedLeiden::<usize>::directed_leiden(&graph, &LeidenConfig::default()).unwrap();
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
    let config = LeidenConfig::default();
    let result = DirectedLeiden::<usize>::directed_leiden(&graph, &config).unwrap();

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
        DirectedLeiden::<usize>::directed_leiden(&graph, &LeidenConfig::default()).unwrap();

    let levels = result.levels();
    assert_ne!(levels, []);
    for window in levels.windows(2) {
        assert!(window[1].modularity() >= window[0].modularity() - 1.0e-9);
    }
}

#[test]
fn test_directed_leiden_communities_are_weakly_connected() {
    let graph = build_directed_graph(
        9,
        vec![
            (0, 1, 5.0),
            (1, 2, 5.0),
            (2, 0, 5.0),
            (3, 4, 5.0),
            (4, 5, 5.0),
            (5, 3, 5.0),
            (6, 7, 5.0),
            (7, 8, 5.0),
            (8, 6, 5.0),
            (2, 3, 0.4),
            (5, 6, 0.4),
            (8, 0, 0.4),
        ],
    );
    let result =
        DirectedLeiden::<usize>::directed_leiden(&graph, &LeidenConfig::default()).unwrap();
    assert!(communities_are_weakly_connected(&graph, 9, result.final_partition()));
}

#[test]
fn test_directed_leiden_is_deterministic_for_a_fixed_seed() {
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
    let config = LeidenConfig { seed: 7, ..LeidenConfig::default() };
    let first = DirectedLeiden::<usize>::directed_leiden(&graph, &config).unwrap();
    let second = DirectedLeiden::<usize>::directed_leiden(&graph, &config).unwrap();

    assert_eq!(first.final_partition(), second.final_partition());
    assert!((first.final_modularity() - second.final_modularity()).abs() <= 1.0e-12);
}

#[test]
fn test_sources_sinks_and_self_loops_are_handled() {
    let graph = build_directed_graph(
        5,
        vec![(0, 1, 1.0), (1, 2, 1.0), (2, 2, 2.0), (2, 3, 1.0), (3, 4, 1.0)],
    );
    let result =
        DirectedLeiden::<usize>::directed_leiden(&graph, &LeidenConfig::default()).unwrap();
    assert_eq!(result.final_partition().len(), 5);
    assert!(result.final_modularity().is_finite());
    assert!(communities_are_weakly_connected(&graph, 5, result.final_partition()));
}

#[test]
fn test_directed_leiden_rejects_invalid_theta() {
    let graph = build_directed_graph(2, vec![(0, 1, 1.0), (1, 0, 1.0)]);
    let config = LeidenConfig { theta: 0.0, ..LeidenConfig::default() };
    let error = DirectedLeiden::<usize>::directed_leiden(&graph, &config).unwrap_err();
    assert!(matches!(error, ModularityError::InvalidTheta));
}

#[test]
fn test_directed_leiden_rejects_invalid_resolution() {
    let graph = build_directed_graph(2, vec![(0, 1, 1.0), (1, 0, 1.0)]);
    let config = LeidenConfig { resolution: -1.0, ..LeidenConfig::default() };
    let error = DirectedLeiden::<usize>::directed_leiden(&graph, &config).unwrap_err();
    assert!(matches!(error, ModularityError::InvalidResolution));
}

#[test]
fn test_directed_leiden_marker_overflow_returns_error() {
    let graph = build_directed_graph(300, Vec::new());
    let error =
        DirectedLeiden::<u8>::directed_leiden(&graph, &LeidenConfig::default()).unwrap_err();
    assert!(matches!(error, ModularityError::TooManyCommunities));
}

/// Returns whether contracting each community of `partition` to one node, and
/// keeping every arc between distinct communities, gives an acyclic graph.
fn contraction_is_acyclic(graph: &WeightedMatrix, partition: &[usize]) -> bool {
    let number_of_communities =
        partition.iter().copied().max().map_or(0usize, |max| max.saturating_add(1));
    let mut successors: Vec<Vec<usize>> = vec![Vec::new(); number_of_communities];
    let mut in_degrees = vec![0usize; number_of_communities];
    for (source, &from) in partition.iter().enumerate() {
        for destination in graph.sparse_row(source) {
            let to = partition[destination];
            if from != to {
                successors[from].push(to);
                in_degrees[to] += 1;
            }
        }
    }

    let mut ready: Vec<usize> =
        (0..number_of_communities).filter(|&community| in_degrees[community] == 0).collect();
    let mut sorted = 0usize;
    while let Some(community) = ready.pop() {
        sorted += 1;
        for &successor in &successors[community] {
            in_degrees[successor] -= 1;
            if in_degrees[successor] == 0 {
                ready.push(successor);
            }
        }
    }
    sorted == number_of_communities
}

/// Returns the nodes reachable from `start` along arcs, `start` included.
fn reachable_from(graph: &WeightedMatrix, node_count: usize, start: usize) -> Vec<bool> {
    let mut reached = vec![false; node_count];
    let mut stack = vec![start];
    reached[start] = true;
    while let Some(node) = stack.pop() {
        for destination in graph.sparse_row(node) {
            if !reached[destination] {
                reached[destination] = true;
                stack.push(destination);
            }
        }
    }
    reached
}

/// Asserts every guarantee of `acyclic_directed_leiden` on `graph`: at every
/// level the communities contract to an acyclic graph, are weakly connected
/// and keep each strongly connected component whole, and the same
/// configuration reproduces the same result.
fn assert_acyclic_directed_leiden_guarantees(
    graph: &WeightedMatrix,
    node_count: usize,
    config: &LeidenConfig,
) {
    let result = DirectedLeiden::<usize>::acyclic_directed_leiden(graph, config).unwrap();
    let reachability: Vec<Vec<bool>> =
        (0..node_count).map(|node| reachable_from(graph, node_count, node)).collect();

    assert_ne!(result.levels(), []);
    for level in result.levels() {
        let partition = level.partition();
        assert_eq!(partition.len(), node_count);
        assert!(
            contraction_is_acyclic(graph, partition),
            "partition {partition:?} contracts to a cycle"
        );
        assert!(
            communities_are_weakly_connected(graph, node_count, partition),
            "partition {partition:?} has a weakly disconnected community"
        );
        for (first, reached_from_first) in reachability.iter().enumerate() {
            for (second, &forward) in reached_from_first.iter().enumerate() {
                if forward && reachability[second][first] {
                    assert_eq!(
                        partition[first], partition[second],
                        "strongly connected nodes {first} and {second} are split in {partition:?}"
                    );
                }
            }
        }
    }

    let repeated = DirectedLeiden::<usize>::acyclic_directed_leiden(graph, config).unwrap();
    assert_eq!(result, repeated);
}

/// Builds a pseudo-random directed graph from `seed`. With `acyclic` every arc
/// goes from a larger to a smaller node index, so the graph is a DAG.
fn pseudo_random_graph(seed: u64, node_count: usize, acyclic: bool) -> WeightedMatrix {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let mut edges = Vec::new();
    for source in 0..node_count {
        for destination in 0..node_count {
            if source == destination || (acyclic && source < destination) {
                continue;
            }
            if next() % 100 < 30 {
                let weight = f64::from(u32::try_from(next() % 3).unwrap() + 1);
                edges.push((source, destination, weight));
            }
        }
    }
    build_directed_graph(node_count, edges)
}

#[test]
fn test_acyclic_directed_leiden_breaks_the_cyclic_contraction_of_directed_leiden() {
    let graph = build_directed_graph(
        5,
        vec![(1, 0, 1.0), (2, 1, 1.0), (3, 0, 1.0), (3, 1, 1.0), (4, 1, 1.0)],
    );
    let config = LeidenConfig::default();
    let unconstrained = DirectedLeiden::<usize>::directed_leiden(&graph, &config).unwrap();
    assert!(!contraction_is_acyclic(&graph, unconstrained.final_partition()));

    let result = DirectedLeiden::<usize>::acyclic_directed_leiden(&graph, &config).unwrap();
    assert_ne!(result.final_partition(), [0, 1, 1, 0, 1]);
    assert_acyclic_directed_leiden_guarantees(&graph, 5, &config);
}

#[test]
fn test_acyclic_directed_leiden_on_random_dags() {
    for seed in 0..400 {
        let node_count = 4 + usize::try_from(seed % 12).unwrap();
        let graph = pseudo_random_graph(seed, node_count, true);
        let config = LeidenConfig { seed, ..LeidenConfig::default() };
        assert_acyclic_directed_leiden_guarantees(&graph, node_count, &config);
    }
    for seed in 0..10 {
        let graph = pseudo_random_graph(seed, 60, true);
        assert_acyclic_directed_leiden_guarantees(&graph, 60, &LeidenConfig::default());
    }
}

#[test]
fn test_acyclic_directed_leiden_on_random_cyclic_graphs() {
    for seed in 0..200 {
        let node_count = 4 + usize::try_from(seed % 12).unwrap();
        let graph = pseudo_random_graph(seed, node_count, false);
        let config = LeidenConfig { seed, ..LeidenConfig::default() };
        assert_acyclic_directed_leiden_guarantees(&graph, node_count, &config);
    }
}

#[test]
fn test_acyclic_directed_leiden_merges_each_cycle() {
    // Two directed triangles joined by one arc condense to two nodes.
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
    let result =
        DirectedLeiden::<usize>::acyclic_directed_leiden(&graph, &LeidenConfig::default()).unwrap();
    assert_eq!(result.final_partition(), [0, 0, 0, 1, 1, 1]);
}

#[test]
fn test_acyclic_directed_leiden_rejects_invalid_theta() {
    let graph = build_directed_graph(2, vec![(0, 1, 1.0)]);
    let config = LeidenConfig { theta: 0.0, ..LeidenConfig::default() };
    let error = DirectedLeiden::<usize>::acyclic_directed_leiden(&graph, &config).unwrap_err();
    assert!(matches!(error, ModularityError::InvalidTheta));
}
