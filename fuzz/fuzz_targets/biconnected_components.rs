//! Fuzz harness for the Hopcroft-Tarjan biconnected-components decomposition.
//!
//! The harness uses a small exact oracle on simple graphs and also exercises
//! the explicit self-loop rejection path.

use std::collections::BTreeSet;

use arbitrary::Arbitrary;
use geometric_traits::{
    errors::{MonopartiteError, monopartite_graph_error::algorithms::MonopartiteAlgorithmError},
    impls::{CSR2D, SymmetricCSR2D, UpperTriangularCSR2D},
    naive_structs::{GenericGraph, GenericUndirectedMonopartiteEdgesBuilder},
    traits::{
        BiconnectedComponents, BiconnectedComponentsError, EdgesBuilder, MonopartiteGraph,
        MonoplexGraph, MonoplexMonopartiteGraph,
    },
};
use honggfuzz::fuzz;

type UndirectedGraph = GenericGraph<Vec<u8>, SymmetricCSR2D<CSR2D<usize, usize, usize>>>;
type Edge = [usize; 2];

#[derive(Arbitrary, Debug)]
struct FuzzBiconnectedCase {
    order: u8,
    edges: Vec<(u8, u8, u8)>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct OracleBlock {
    edges: Vec<Edge>,
    vertices: Vec<usize>,
}

fn main() {
    loop {
        fuzz!(|case: FuzzBiconnectedCase| {
            let graph = build_graph(&case);

            if graph.has_self_loops() {
                assert!(matches!(
                    graph.biconnected_components(),
                    Err(MonopartiteError::AlgorithmError(
                        MonopartiteAlgorithmError::BiconnectedComponentsError(
                            BiconnectedComponentsError::SelfLoopsUnsupported
                        )
                    ))
                ));
                assert!(matches!(
                    graph.is_biconnected(),
                    Err(MonopartiteError::AlgorithmError(
                        MonopartiteAlgorithmError::BiconnectedComponentsError(
                            BiconnectedComponentsError::SelfLoopsUnsupported
                        )
                    ))
                ));
                return;
            }

            let decomposition = graph.biconnected_components().unwrap();
            let order = graph.number_of_nodes();
            let expected_edges = logical_simple_edges(&graph);
            let expected_blocks = maximal_biconnected_blocks(order, &expected_edges);
            let expected_edge_components: Vec<Vec<Edge>> =
                expected_blocks.iter().map(|block| block.edges.clone()).collect();
            let expected_vertex_components: Vec<Vec<usize>> =
                expected_blocks.iter().map(|block| block.vertices.clone()).collect();
            let expected_articulation_points = articulation_points(order, &expected_edges);
            let expected_bridges = bridges(order, &expected_edges);
            let expected_connected_components =
                connected_components(&(0..order).collect::<Vec<_>>(), &expected_edges);
            let expected_omitted_vertices = omitted_vertices(order, &expected_vertex_components);
            let expected_cyclic_component_ids = cyclic_component_ids(
                &expected_edge_components,
                &expected_vertex_components,
            );
            let expected_is_biconnected =
                order >= 2 && expected_connected_components.len() == 1 && expected_articulation_points.is_empty();

            assert_eq!(
                decomposition.edge_biconnected_components().cloned().collect::<Vec<_>>(),
                expected_edge_components
            );
            assert_eq!(
                decomposition.vertex_biconnected_components().cloned().collect::<Vec<_>>(),
                expected_vertex_components
            );
            assert_eq!(
                decomposition.articulation_points().collect::<Vec<_>>(),
                expected_articulation_points
            );
            assert_eq!(decomposition.bridges().collect::<Vec<_>>(), expected_bridges);
            assert_eq!(
                decomposition.vertices_without_biconnected_component().collect::<Vec<_>>(),
                expected_omitted_vertices
            );
            assert_eq!(
                decomposition.cyclic_biconnected_component_ids().collect::<Vec<_>>(),
                expected_cyclic_component_ids
            );
            assert_eq!(
                decomposition.number_of_biconnected_components(),
                expected_edge_components.len()
            );
            assert_eq!(
                decomposition.number_of_connected_components(),
                expected_connected_components.len()
            );
            assert_eq!(decomposition.is_biconnected(), expected_is_biconnected);
            assert_eq!(graph.is_biconnected().unwrap(), expected_is_biconnected);
        });
    }
}

fn build_graph(case: &FuzzBiconnectedCase) -> UndirectedGraph {
    // Keep the oracle cheap enough for fuzzing while still exploring
    // disconnected, sparse, and cyclic cases.
    let order = usize::from(case.order % 9);
    let nodes: Vec<u8> = (0..order)
        .map(|index| u8::try_from(index).expect("fuzz node index should fit into u8"))
        .collect();
    let edges = normalized_edges(order, &case.edges);
    let adjacency = GenericUndirectedMonopartiteEdgesBuilder::<
        _,
        UpperTriangularCSR2D<CSR2D<usize, usize, usize>>,
        SymmetricCSR2D<CSR2D<usize, usize, usize>>,
    >::default()
    .expected_number_of_edges(edges.len())
    .expected_shape(order)
    .edges(edges.into_iter())
    .build()
    .unwrap();
    GenericGraph::from((nodes, adjacency))
}

fn normalized_edges(order: usize, raw_edges: &[(u8, u8, u8)]) -> Vec<(usize, usize)> {
    let mut edges = BTreeSet::new();
    if order == 0 {
        return Vec::new();
    }
    for &(left, right, keep) in raw_edges {
        if keep & 1 == 0 {
            continue;
        }
        let left = usize::from(left) % order;
        let right = usize::from(right) % order;
        if left <= right {
            edges.insert((left, right));
        } else {
            edges.insert((right, left));
        }
    }
    edges.into_iter().collect()
}

fn logical_simple_edges(graph: &UndirectedGraph) -> BTreeSet<Edge> {
    graph
        .sparse_coordinates()
        .filter_map(|(source, destination)| {
            if source == destination {
                None
            } else if source <= destination {
                Some([source, destination])
            } else {
                Some([destination, source])
            }
        })
        .collect()
}

fn connected_components(vertices: &[usize], edges: &BTreeSet<Edge>) -> Vec<Vec<usize>> {
    if vertices.is_empty() {
        return Vec::new();
    }

    let active = vertices.iter().copied().collect::<BTreeSet<_>>();
    let mut unseen = active.clone();
    let mut components = Vec::new();

    while let Some(&start) = unseen.iter().next() {
        unseen.remove(&start);
        let mut stack = vec![start];
        let mut component = vec![start];

        while let Some(vertex) = stack.pop() {
            for neighbor in neighbors_in_subset(vertex, &active, edges) {
                if unseen.remove(&neighbor) {
                    stack.push(neighbor);
                    component.push(neighbor);
                }
            }
        }

        component.sort_unstable();
        components.push(component);
    }

    components.sort();
    components
}

fn neighbors_in_subset(vertex: usize, active: &BTreeSet<usize>, edges: &BTreeSet<Edge>) -> Vec<usize> {
    let mut neighbors = Vec::new();
    for &[left, right] in edges {
        if left == vertex && active.contains(&right) {
            neighbors.push(right);
        } else if right == vertex && active.contains(&left) {
            neighbors.push(left);
        }
    }
    neighbors.sort_unstable();
    neighbors.dedup();
    neighbors
}

fn induced_edges(vertices: &[usize], edges: &BTreeSet<Edge>) -> BTreeSet<Edge> {
    let active = vertices.iter().copied().collect::<BTreeSet<_>>();
    edges
        .iter()
        .copied()
        .filter(|[left, right]| active.contains(left) && active.contains(right))
        .collect()
}

fn is_connected(vertices: &[usize], edges: &BTreeSet<Edge>) -> bool {
    connected_components(vertices, edges).len() <= 1
}

fn articulation_points(order: usize, edges: &BTreeSet<Edge>) -> Vec<usize> {
    let baseline = connected_components(&(0..order).collect::<Vec<_>>(), edges).len();
    let mut points = Vec::new();
    for removed in 0..order {
        let remainder = (0..order).filter(|&vertex| vertex != removed).collect::<Vec<_>>();
        let reduced_edges = edges
            .iter()
            .copied()
            .filter(|[left, right]| *left != removed && *right != removed)
            .collect::<BTreeSet<_>>();
        if connected_components(&remainder, &reduced_edges).len() > baseline {
            points.push(removed);
        }
    }
    points
}

fn bridges(order: usize, edges: &BTreeSet<Edge>) -> Vec<Edge> {
    let baseline = connected_components(&(0..order).collect::<Vec<_>>(), edges).len();
    let mut cut_edges = Vec::new();
    for &edge in edges {
        let mut reduced_edges = edges.clone();
        reduced_edges.remove(&edge);
        if connected_components(&(0..order).collect::<Vec<_>>(), &reduced_edges).len() > baseline {
            cut_edges.push(edge);
        }
    }
    cut_edges
}

fn is_biconnected_block(vertices: &[usize], graph_edges: &BTreeSet<Edge>) -> bool {
    if vertices.len() < 2 {
        return false;
    }

    let block_edges = induced_edges(vertices, graph_edges);
    if block_edges.is_empty() {
        return false;
    }

    if vertices.len() == 2 {
        return block_edges.len() == 1;
    }

    if !is_connected(vertices, &block_edges) {
        return false;
    }

    for &removed in vertices {
        let remainder = vertices
            .iter()
            .copied()
            .filter(|&vertex| vertex != removed)
            .collect::<Vec<_>>();
        if !is_connected(&remainder, &induced_edges(&remainder, &block_edges)) {
            return false;
        }
    }

    true
}

fn maximal_biconnected_blocks(order: usize, edges: &BTreeSet<Edge>) -> Vec<OracleBlock> {
    let mut valid_blocks = Vec::new();

    for mask in 0usize..(1usize << order) {
        if mask.count_ones() < 2 {
            continue;
        }
        let vertices = vertices_from_mask(order, mask);
        if is_biconnected_block(&vertices, edges) {
            valid_blocks.push(vertices);
        }
    }

    let mut maximal_blocks = valid_blocks
        .iter()
        .filter(|vertices| !valid_blocks.iter().any(|other| is_strict_subset(vertices, other)))
        .map(|vertices| OracleBlock {
            edges: induced_edges(vertices, edges).into_iter().collect(),
            vertices: vertices.clone(),
        })
        .collect::<Vec<_>>();
    maximal_blocks.sort();
    maximal_blocks
}

fn vertices_from_mask(order: usize, mask: usize) -> Vec<usize> {
    (0..order).filter(|bit| mask & (1usize << bit) != 0).collect()
}

fn is_strict_subset(left: &[usize], right: &[usize]) -> bool {
    left.len() < right.len() && left.iter().all(|vertex| right.contains(vertex))
}

fn omitted_vertices(order: usize, vertex_components: &[Vec<usize>]) -> Vec<usize> {
    let mut memberships = vec![0usize; order];
    for component in vertex_components {
        for &vertex in component {
            memberships[vertex] += 1;
        }
    }
    memberships
        .into_iter()
        .enumerate()
        .filter_map(|(vertex, membership_count)| (membership_count == 0).then_some(vertex))
        .collect()
}

fn cyclic_component_ids(
    edge_components: &[Vec<Edge>],
    vertex_components: &[Vec<usize>],
) -> Vec<usize> {
    edge_components
        .iter()
        .zip(vertex_components.iter())
        .enumerate()
        .filter_map(|(component_id, (edges, vertices))| {
            (!edges.is_empty() && edges.len() >= vertices.len()).then_some(component_id)
        })
        .collect()
}
