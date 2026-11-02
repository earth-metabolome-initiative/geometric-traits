//! End-to-end check that the TMAP back-end composes: a weighted graph yields a
//! minimum spanning forest, the forest symmetrizes into an undirected graph,
//! and `ForceAtlas2` lays it out into finite coordinates.
#![cfg(feature = "std")]

use geometric_traits::{
    impls::ValuedCSR2D,
    naive_structs::GenericEdgesBuilder,
    prelude::*,
    traits::{
        EdgesBuilder,
        algorithms::minimum_spanning_tree::{Boruvka, Kruskal, Prim},
    },
};

type WeightedMatrix = ValuedCSR2D<usize, usize, usize, f64>;

/// Builds a symmetric weighted matrix from an inline undirected edge list.
fn build(node_count: usize, edges: &[(usize, usize, f64)]) -> WeightedMatrix {
    let mut directed = Vec::with_capacity(edges.len() * 2);
    for &(source, destination, weight) in edges {
        directed.push((source, destination, weight));
        directed.push((destination, source, weight));
    }
    directed.sort_unstable_by(|(ls, ld, _), (rs, rd, _)| (ls, ld).cmp(&(rs, rd)));
    GenericEdgesBuilder::<_, WeightedMatrix>::default()
        .expected_number_of_edges(directed.len())
        .expected_shape((node_count, node_count))
        .edges(directed.into_iter())
        .build()
        .unwrap()
}

#[test]
fn mst_symmetrize_layout_pipeline() {
    // A connected graph with distinct weights, so the MST is unique.
    let node_count = 6;
    let matrix = build(
        node_count,
        &[
            (0, 1, 1.0),
            (1, 2, 2.0),
            (2, 3, 3.0),
            (3, 4, 4.0),
            (4, 5, 5.0),
            (0, 5, 9.0),
            (1, 4, 8.0),
            (2, 5, 7.0),
        ],
    );

    // The three algorithms agree on the rooted forest for a unique MST.
    let forest = matrix.minimum_spanning_tree_prim().unwrap();
    assert_eq!(
        forest.parent_array(),
        matrix.minimum_spanning_tree_kruskal().unwrap().parent_array()
    );
    assert_eq!(
        forest.parent_array(),
        matrix.minimum_spanning_tree_boruvka().unwrap().parent_array()
    );

    // The forest spans the connected graph and is rooted at node 0.
    assert_eq!(forest.number_of_components(), 1);
    assert_eq!(forest.number_of_nodes(), node_count);
    assert_eq!(forest.len(), node_count - 1);
    assert!(forest.is_root(0));

    // Symmetrize and check the undirected adjacency.
    let symmetric = forest.symmetrize();
    assert!(symmetric.as_matrix().is_symmetric());
    assert_eq!(symmetric.as_matrix().number_of_defined_values(), 2 * (node_count - 1));

    let forest_edges: std::collections::HashSet<(usize, usize)> =
        forest.edge_iter().map(|(source, destination, _)| (source, destination)).collect();
    let mut symmetric_edges: std::collections::HashSet<(usize, usize)> =
        std::collections::HashSet::new();
    for row in 0..symmetric.as_matrix().order() {
        for column in symmetric.as_matrix().sparse_row(row) {
            if row < column {
                symmetric_edges.insert((row, column));
            }
        }
    }
    assert_eq!(symmetric_edges, forest_edges);

    // The symmetric forest is one connected component as a graph.
    let graph = symmetric.as_graph();
    let components =
        ConnectedComponents::<usize>::connected_components(&graph).expect("components computed");
    assert_eq!(components.number_of_components(), 1);

    // Lay it out. ForceAtlas2 resolves through the deref to the inner matrix.
    let layout = symmetric.force_atlas2(&ForceAtlas2Config::default()).unwrap();
    assert_eq!(layout.num_points(), node_count);
    assert!(layout.coordinates_flat().iter().copied().all(f64::is_finite));
}
