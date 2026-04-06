//! Fuzz harness for `apply_node_order_to_graph`.
//!
//! Covers valid node orders on small directed, undirected,
//! weighted, and modular-product graphs. For each permutation it checks:
//! - node symbols follow the requested order
//! - adjacency / weights are preserved under renaming
//! - applying the inverse permutation round-trips to the original graph

use std::collections::{BTreeMap, BTreeSet};

use arbitrary::Arbitrary;
use geometric_traits::{
    impls::{BitSquareMatrix, CSR2D, SquareCSR2D, SymmetricCSR2D, ValuedCSR2D},
    naive_structs::{GenericEdgesBuilder, GenericGraph, GenericUndirectedMonopartiteEdgesBuilder},
    prelude::*,
    traits::{
        Edges, MonopartiteGraph, MonoplexGraph, SparseMatrix2D,
        algorithms::apply_node_order_to_graph,
    },
};
use honggfuzz::fuzz;

type DirectedGraph = SquareCSR2D<CSR2D<usize, usize, usize>>;
type UndirectedGraph = SymmetricCSR2D<CSR2D<usize, usize, usize>>;
type DirectedWeightedGraph = ValuedCSR2D<usize, usize, usize, u8>;
type UndirectedWeightedGraph = SymmetricCSR2D<ValuedCSR2D<usize, usize, usize, u8>>;

#[derive(Arbitrary, Debug)]
struct FuzzNodeOrderCase {
    kind: u8,
    left_order: u8,
    right_order: u8,
    left_edges: Vec<(u8, u8, u8)>,
    right_edges: Vec<(u8, u8, u8)>,
    permutation_data: Vec<u8>,
}

fn main() {
    loop {
        fuzz!(|case: FuzzNodeOrderCase| {
            match case.kind % 5 {
                0 => fuzz_directed_unweighted(&case),
                1 => fuzz_undirected_unweighted(&case),
                2 => fuzz_directed_weighted(&case),
                3 => fuzz_undirected_weighted(&case),
                _ => fuzz_modular_product(&case),
            }
        });
    }
}

fn fuzz_directed_unweighted(case: &FuzzNodeOrderCase) {
    let order = usize::from(case.left_order % 8);
    let nodes: Vec<u8> = (0..order).map(|index| index as u8).collect();
    let edges = build_directed_edges(order, &case.left_edges);
    let graph: GenericGraph<Vec<u8>, DirectedGraph> = GenericGraph::from((nodes, edges));
    let order = valid_order(order, &case.permutation_data);
    let reordered = apply_node_order_to_graph(&graph, &order);
    assert_directed_unweighted_reorder(&graph, &reordered, &order);
}

fn fuzz_undirected_unweighted(case: &FuzzNodeOrderCase) {
    let order = usize::from(case.left_order % 8);
    let nodes: Vec<u8> = (0..order).map(|index| index as u8).collect();
    let edges = build_undirected_edges(order, &case.left_edges);
    let graph: GenericGraph<Vec<u8>, UndirectedGraph> = GenericGraph::from((nodes, edges));
    let order = valid_order(order, &case.permutation_data);
    let reordered = apply_node_order_to_graph(&graph, &order);
    assert_undirected_unweighted_reorder(&graph, &reordered, &order);
}

fn fuzz_directed_weighted(case: &FuzzNodeOrderCase) {
    let order = usize::from(case.left_order % 8);
    let nodes: Vec<u8> = (0..order).map(|index| index as u8).collect();
    let edges = build_directed_weighted_edges(order, &case.left_edges);
    let graph: GenericGraph<Vec<u8>, DirectedWeightedGraph> = GenericGraph::from((nodes, edges));
    let order = valid_order(order, &case.permutation_data);
    let reordered = apply_node_order_to_graph(&graph, &order);
    assert_directed_weighted_reorder(&graph, &reordered, &order);
}

fn fuzz_undirected_weighted(case: &FuzzNodeOrderCase) {
    let order = usize::from(case.left_order % 8);
    let nodes: Vec<u8> = (0..order).map(|index| index as u8).collect();
    let edges = build_undirected_weighted_edges(order, &case.left_edges);
    let graph: GenericGraph<Vec<u8>, UndirectedWeightedGraph> = GenericGraph::from((nodes, edges));
    let order = valid_order(order, &case.permutation_data);
    let reordered = apply_node_order_to_graph(&graph, &order);
    assert_undirected_weighted_reorder(&graph, &reordered, &order);
}

fn fuzz_modular_product(case: &FuzzNodeOrderCase) {
    let left_order = usize::from(case.left_order % 6);
    let right_order = usize::from(case.right_order % 6);
    let left = BitSquareMatrix::from_symmetric_edges(
        left_order,
        undirected_unweighted_edge_vec(left_order, &case.left_edges),
    );
    let right = BitSquareMatrix::from_symmetric_edges(
        right_order,
        undirected_unweighted_edge_vec(right_order, &case.right_edges),
    );
    let graph = left.modular_product_filtered(&right, |_, _| true).into_graph();
    let order = valid_order(graph.number_of_nodes(), &case.permutation_data);
    let reordered = apply_node_order_to_graph(&graph, &order);
    assert_modular_product_reorder(&graph, &reordered, &order);
}

fn build_directed_edges(order: usize, raw_edges: &[(u8, u8, u8)]) -> DirectedGraph {
    let mut edges = BTreeSet::new();
    for &(source, destination, keep) in raw_edges {
        if order == 0 || keep & 1 == 0 {
            continue;
        }
        edges.insert((usize::from(source) % order, usize::from(destination) % order));
    }

    GenericEdgesBuilder::<_, DirectedGraph>::default()
        .expected_number_of_edges(edges.len())
        .expected_shape(order)
        .edges(edges.into_iter())
        .build()
        .unwrap()
}

fn undirected_unweighted_edge_vec(order: usize, raw_edges: &[(u8, u8, u8)]) -> Vec<(usize, usize)> {
    let mut edges = BTreeSet::new();
    for &(left, right, keep) in raw_edges {
        if order == 0 || keep & 1 == 0 {
            continue;
        }
        let left = usize::from(left) % order;
        let right = usize::from(right) % order;
        let edge = if left <= right { (left, right) } else { (right, left) };
        edges.insert(edge);
    }
    edges.into_iter().collect()
}

fn build_undirected_edges(order: usize, raw_edges: &[(u8, u8, u8)]) -> UndirectedGraph {
    GenericUndirectedMonopartiteEdgesBuilder::<
        _,
        UpperTriangularCSR2D<CSR2D<usize, usize, usize>>,
        UndirectedGraph,
    >::default()
    .expected_number_of_edges(undirected_unweighted_edge_vec(order, raw_edges).len())
    .expected_shape(order)
    .edges(undirected_unweighted_edge_vec(order, raw_edges).into_iter())
    .build()
    .unwrap()
}

fn build_directed_weighted_edges(order: usize, raw_edges: &[(u8, u8, u8)]) -> DirectedWeightedGraph {
    let mut edges = BTreeMap::new();
    for &(source, destination, weight) in raw_edges {
        if order == 0 {
            continue;
        }
        edges.insert((usize::from(source) % order, usize::from(destination) % order), weight);
    }

    GenericEdgesBuilder::<_, DirectedWeightedGraph>::default()
        .expected_number_of_edges(edges.len())
        .expected_shape((order, order))
        .edges(edges.into_iter().map(|((source, destination), weight)| (source, destination, weight)))
        .build()
        .unwrap()
}

fn build_undirected_weighted_edges(
    order: usize,
    raw_edges: &[(u8, u8, u8)],
) -> UndirectedWeightedGraph {
    let mut edges = BTreeMap::new();
    for &(left, right, weight) in raw_edges {
        if order == 0 {
            continue;
        }
        let left = usize::from(left) % order;
        let right = usize::from(right) % order;
        let edge = if left <= right { (left, right) } else { (right, left) };
        edges.insert(edge, weight);
    }

    UndirectedWeightedGraph::from_sorted_upper_triangular_entries(
        order,
        edges.into_iter().map(|((left, right), weight)| (left, right, weight)),
    )
    .unwrap()
}

fn valid_order(number_of_nodes: usize, data: &[u8]) -> Vec<usize> {
    let mut permutation: Vec<usize> = (0..number_of_nodes).collect();
    if number_of_nodes == 0 {
        return permutation;
    }
    for (index, &byte) in data.iter().enumerate() {
        permutation.swap(index % number_of_nodes, usize::from(byte) % number_of_nodes);
    }
    permutation
}

fn inverse(order: &[usize]) -> Vec<usize> {
    let mut inverse = vec![0; order.len()];
    for (new_index, &old_index) in order.iter().enumerate() {
        inverse[old_index] = new_index;
    }
    inverse
}

fn assert_directed_unweighted_reorder(
    original: &GenericGraph<Vec<u8>, DirectedGraph>,
    reordered: &GenericGraph<Vec<u8>, DirectedGraph>,
    order: &[usize],
) {
    let expected_nodes: Vec<_> = order.iter().map(|&index| original.nodes_vocabulary()[index]).collect();
    assert_eq!(reordered.nodes().collect::<Vec<_>>(), expected_nodes);

    let inverse = inverse(order);
    let original_matrix = Edges::matrix(original.edges());
    let reordered_matrix = Edges::matrix(reordered.edges());
    for source in 0..order.len() {
        for destination in 0..order.len() {
            assert_eq!(
                original_matrix.has_entry(source, destination),
                reordered_matrix.has_entry(inverse[source], inverse[destination])
            );
        }
    }

    let restored = apply_node_order_to_graph(reordered, &inverse);
    assert_eq!(restored.nodes().collect::<Vec<_>>(), original.nodes().collect::<Vec<_>>());
    let restored_matrix = Edges::matrix(restored.edges());
    for source in 0..order.len() {
        for destination in 0..order.len() {
            assert_eq!(
                original_matrix.has_entry(source, destination),
                restored_matrix.has_entry(source, destination)
            );
        }
    }
}

fn assert_undirected_unweighted_reorder(
    original: &GenericGraph<Vec<u8>, UndirectedGraph>,
    reordered: &GenericGraph<Vec<u8>, UndirectedGraph>,
    order: &[usize],
) {
    let expected_nodes: Vec<_> = order.iter().map(|&index| original.nodes_vocabulary()[index]).collect();
    assert_eq!(reordered.nodes().collect::<Vec<_>>(), expected_nodes);

    let inverse = inverse(order);
    let original_matrix = Edges::matrix(original.edges());
    let reordered_matrix = Edges::matrix(reordered.edges());
    for source in 0..order.len() {
        for destination in 0..order.len() {
            assert_eq!(
                original_matrix.has_entry(source, destination),
                reordered_matrix.has_entry(inverse[source], inverse[destination])
            );
        }
    }

    let restored = apply_node_order_to_graph(reordered, &inverse);
    assert_eq!(restored.nodes().collect::<Vec<_>>(), original.nodes().collect::<Vec<_>>());
}

fn assert_directed_weighted_reorder(
    original: &GenericGraph<Vec<u8>, DirectedWeightedGraph>,
    reordered: &GenericGraph<Vec<u8>, DirectedWeightedGraph>,
    order: &[usize],
) {
    let expected_nodes: Vec<_> = order.iter().map(|&index| original.nodes_vocabulary()[index]).collect();
    assert_eq!(reordered.nodes().collect::<Vec<_>>(), expected_nodes);

    let inverse = inverse(order);
    let original_matrix = Edges::matrix(original.edges());
    let reordered_matrix = Edges::matrix(reordered.edges());
    for source in 0..order.len() {
        for destination in 0..order.len() {
            assert_eq!(
                original_matrix.sparse_value_at(source, destination),
                reordered_matrix.sparse_value_at(inverse[source], inverse[destination])
            );
        }
    }

    let restored = apply_node_order_to_graph(reordered, &inverse);
    assert_eq!(restored.nodes().collect::<Vec<_>>(), original.nodes().collect::<Vec<_>>());
}

fn assert_undirected_weighted_reorder(
    original: &GenericGraph<Vec<u8>, UndirectedWeightedGraph>,
    reordered: &GenericGraph<Vec<u8>, UndirectedWeightedGraph>,
    order: &[usize],
) {
    let expected_nodes: Vec<_> = order.iter().map(|&index| original.nodes_vocabulary()[index]).collect();
    assert_eq!(reordered.nodes().collect::<Vec<_>>(), expected_nodes);

    let inverse = inverse(order);
    let original_matrix = Edges::matrix(original.edges());
    let reordered_matrix = Edges::matrix(reordered.edges());
    for source in 0..order.len() {
        for destination in 0..order.len() {
            assert_eq!(
                original_matrix.sparse_value_at(source, destination),
                reordered_matrix.sparse_value_at(inverse[source], inverse[destination])
            );
        }
    }

    let restored = apply_node_order_to_graph(reordered, &inverse);
    assert_eq!(restored.nodes().collect::<Vec<_>>(), original.nodes().collect::<Vec<_>>());
}

fn assert_modular_product_reorder<I1, I2>(
    original: &ModularProductGraph<I1, I2>,
    reordered: &ModularProductGraph<I1, I2>,
    order: &[usize],
) where
    I1: geometric_traits::traits::Symbol,
    I2: geometric_traits::traits::Symbol,
{
    let expected_nodes: Vec<_> = order.iter().map(|&index| original.nodes_vocabulary()[index].clone()).collect();
    assert_eq!(reordered.nodes().collect::<Vec<_>>(), expected_nodes);

    let inverse = inverse(order);
    let original_matrix = original.matrix();
    let reordered_matrix = reordered.matrix();
    for source in 0..order.len() {
        for destination in 0..order.len() {
            assert_eq!(
                original_matrix.has_entry(source, destination),
                reordered_matrix.has_entry(inverse[source], inverse[destination])
            );
        }
    }

    let restored = apply_node_order_to_graph(reordered, &inverse);
    assert_eq!(restored.nodes().collect::<Vec<_>>(), original.nodes().collect::<Vec<_>>());
}
