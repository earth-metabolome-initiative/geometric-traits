//! Tests for labeled line graph construction.
#![cfg(feature = "alloc")]

use geometric_traits::{
    impls::{SortedVec, SymmetricCSR2D},
    prelude::*,
    traits::{EdgesBuilder, SquareMatrix, TypedNode, VocabularyBuilder},
};

// ============================================================================
// Typed node infrastructure
// ============================================================================

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
enum Color {
    Red,
    Green,
    Blue,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct ColoredNode {
    id: usize,
    color: Color,
}

impl TypedNode for ColoredNode {
    type NodeType = Color;

    fn node_type(&self) -> Self::NodeType {
        self.color
    }
}

type ColoredGraph = geometric_traits::naive_structs::GenericGraph<
    SortedVec<ColoredNode>,
    SymmetricCSR2D<geometric_traits::impls::CSR2D<usize, usize, usize>>,
>;

/// Build an undirected graph with colored nodes.
fn build_colored_graph(node_colors: &[Color], edges: Vec<(usize, usize)>) -> ColoredGraph {
    let n = node_colors.len();
    let nodes_vec: Vec<ColoredNode> =
        node_colors.iter().enumerate().map(|(i, &c)| ColoredNode { id: i, color: c }).collect();
    let nodes: SortedVec<ColoredNode> = GenericVocabularyBuilder::default()
        .expected_number_of_symbols(n)
        .symbols(nodes_vec.into_iter().enumerate())
        .build()
        .unwrap();

    let undi: SymmetricCSR2D<_> = build_undi_edges(n, edges);
    geometric_traits::naive_structs::GenericGraph::from((nodes, undi))
}

type UndiBuilder<I> = geometric_traits::naive_structs::GenericUndirectedMonopartiteEdgesBuilder<
    I,
    geometric_traits::impls::UpperTriangularCSR2D<
        geometric_traits::impls::CSR2D<usize, usize, usize>,
    >,
    SymmetricCSR2D<geometric_traits::impls::CSR2D<usize, usize, usize>>,
>;

fn build_undi_edges(
    n: usize,
    mut edges: Vec<(usize, usize)>,
) -> SymmetricCSR2D<geometric_traits::impls::CSR2D<usize, usize, usize>> {
    edges.sort_unstable();
    edges.dedup();
    UndiBuilder::default()
        .expected_number_of_edges(edges.len())
        .expected_shape(n)
        .edges(edges.into_iter())
        .build()
        .unwrap()
}

// ============================================================================
// Tests
// ============================================================================

#[test]
fn test_triangle_k3_all_different_types() {
    // Triangle with R, G, B nodes.
    // Edges: (0,1), (0,2), (1,2).
    // L(G) = K3 with 3 vertices.
    // Edge (e01, e02) shares node 0 (Red) -> label Red
    // Edge (e01, e12) shares node 1 (Green) -> label Green
    // Edge (e02, e12) shares node 2 (Blue) -> label Blue
    let g =
        build_colored_graph(&[Color::Red, Color::Green, Color::Blue], vec![(0, 1), (0, 2), (1, 2)]);

    let lg = g.labeled_line_graph();
    assert_eq!(lg.number_of_vertices(), 3);
    assert_eq!(lg.graph().order(), 3);

    // Check that all edges exist (K3 line graph).
    assert!(lg.graph().has_entry(0, 1));
    assert!(lg.graph().has_entry(0, 2));
    assert!(lg.graph().has_entry(1, 2));

    // Check labels.
    // edge_map: [(0,1), (0,2), (1,2)] (sorted src<dst)
    // LG vertex 0 = edge (0,1), vertex 1 = edge (0,2), vertex 2 = edge (1,2)
    // LG edge (0,1): edges (0,1) and (0,2) share node 0 -> Red
    // LG edge (0,2): edges (0,1) and (1,2) share node 1 -> Green
    // LG edge (1,2): edges (0,2) and (1,2) share node 2 -> Blue
    assert_eq!(lg.graph().sparse_value_at(0, 1), Some(Color::Red));
    assert_eq!(lg.graph().sparse_value_at(0, 2), Some(Color::Green));
    assert_eq!(lg.graph().sparse_value_at(1, 2), Some(Color::Blue));

    // Symmetric values.
    assert_eq!(lg.graph().sparse_value_at(1, 0), Some(Color::Red));
    assert_eq!(lg.graph().sparse_value_at(2, 0), Some(Color::Green));
    assert_eq!(lg.graph().sparse_value_at(2, 1), Some(Color::Blue));
}

#[test]
fn test_path_with_typed_nodes() {
    // Path: 0-1-2, colors: R, G, B
    // Edges: (0,1), (1,2)
    // L(G) has 2 vertices, 1 edge.
    // The edge connects e01 and e12 at shared node 1 -> Green
    let g = build_colored_graph(&[Color::Red, Color::Green, Color::Blue], vec![(0, 1), (1, 2)]);

    let lg = g.labeled_line_graph();
    assert_eq!(lg.number_of_vertices(), 2);
    assert!(lg.graph().has_entry(0, 1));
    assert_eq!(lg.graph().sparse_value_at(0, 1), Some(Color::Green));
}

#[test]
fn test_star_graph_all_labels_center() {
    // Star: center=0 (Red), leaves 1,2,3 (Green)
    // Edges: (0,1), (0,2), (0,3)
    // L(G) = K3 (all edges share center 0)
    // All LG edge labels = Red (the center's type)
    let g = build_colored_graph(
        &[Color::Red, Color::Green, Color::Green, Color::Green],
        vec![(0, 1), (0, 2), (0, 3)],
    );

    let lg = g.labeled_line_graph();
    assert_eq!(lg.number_of_vertices(), 3);

    // All 3 edges should have label Red.
    for i in 0..3 {
        for j in (i + 1)..3 {
            assert!(lg.graph().has_entry(i, j));
            assert_eq!(
                lg.graph().sparse_value_at(i, j),
                Some(Color::Red),
                "edge ({i},{j}) should be Red"
            );
        }
    }
}

#[test]
fn test_empty_graph() {
    let g = build_colored_graph(&[Color::Red, Color::Green], Vec::new());
    let lg = g.labeled_line_graph();
    assert_eq!(lg.number_of_vertices(), 0);
    assert_eq!(lg.graph().order(), 0);
}

#[test]
fn test_single_edge() {
    let g = build_colored_graph(&[Color::Red, Color::Green], vec![(0, 1)]);
    let lg = g.labeled_line_graph();
    assert_eq!(lg.number_of_vertices(), 1);
    // No LG edges (single vertex).
    assert!(!lg.graph().has_entry(0, 0));
}

#[test]
fn test_result_satisfies_valued_traits() {
    let g =
        build_colored_graph(&[Color::Red, Color::Green, Color::Blue], vec![(0, 1), (0, 2), (1, 2)]);

    let lg = g.labeled_line_graph();
    let graph = lg.graph();

    // SparseSquareMatrix
    let _ = graph.order();
    let _ = graph.number_of_defined_diagonal_values();

    // SparseValuedMatrix2D
    let _: Option<Color> = graph.sparse_value_at(0, 1);

    // sparse_row_values
    let vals: Vec<Color> = graph.sparse_row_values(0).collect();
    assert_eq!(vals.len(), 2); // vertex 0 is adjacent to vertices 1 and 2
}

#[test]
fn test_edge_map_matches_unlabeled_line_graph() {
    // The edge_map and structural adjacency of the labeled line graph should
    // match the unlabeled line graph exactly.
    let colors = &[Color::Red, Color::Green, Color::Blue, Color::Red];
    let edges = vec![(0, 1), (1, 2), (2, 3)];
    let g = build_colored_graph(colors, edges);

    let labeled_lg = g.labeled_line_graph();
    let unlabeled_lg = g.line_graph();

    assert_eq!(labeled_lg.number_of_vertices(), unlabeled_lg.number_of_vertices());
    assert_eq!(labeled_lg.edge_map(), unlabeled_lg.edge_map());

    let n = labeled_lg.number_of_vertices();
    for i in 0..n {
        for j in 0..n {
            assert_eq!(
                labeled_lg.graph().has_entry(i, j),
                unlabeled_lg.graph().has_entry(i, j),
                "structural mismatch at ({i}, {j})"
            );
        }
    }
}

#[test]
fn test_end_to_end_labeled_lg_to_labeled_modular_product_to_max_clique() {
    // Build two small typed graphs and run the full pipeline:
    // labeled_line_graph -> labeled_modular_product -> maximum_clique
    // G1: triangle R-G-B
    let g1 =
        build_colored_graph(&[Color::Red, Color::Green, Color::Blue], vec![(0, 1), (0, 2), (1, 2)]);
    // G2: path R-G-B
    let g2 = build_colored_graph(&[Color::Red, Color::Green, Color::Blue], vec![(0, 1), (1, 2)]);

    let lg1 = g1.labeled_line_graph();
    let lg2 = g2.labeled_line_graph();

    // All pairs of line graph vertices.
    let n1 = lg1.number_of_vertices();
    let n2 = lg2.number_of_vertices();
    let pairs: Vec<(usize, usize)> = (0..n1).flat_map(|i| (0..n2).map(move |j| (i, j))).collect();

    let mp = lg1.graph().labeled_modular_product(lg2.graph(), &pairs, |a, b| a == b);

    // The result should be a valid BitSquareMatrix.
    assert_eq!(mp.order(), pairs.len());
    for i in 0..mp.order() {
        assert!(!mp.has_entry(i, i));
    }

    // Run max clique on it.
    let clique = mp.maximum_clique();
    // The max clique size should be >= 1 (at least one compatible edge pair).
    assert!(!clique.is_empty());
}
