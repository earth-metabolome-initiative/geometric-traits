//! Tests for line graph construction (undirected and directed).
#![cfg(feature = "std")]

use geometric_traits::{
    impls::{CSR2D, SortedVec, SquareCSR2D, SymmetricCSR2D},
    prelude::*,
    traits::{
        Edges, EdgesBuilder, SquareMatrix, VocabularyBuilder, algorithms::randomized_graphs::*,
    },
};

type UndirectedGraph = SymmetricCSR2D<CSR2D<usize, usize, usize>>;

fn undi_edge_count(g: &UndirectedGraph) -> usize {
    Edges::number_of_edges(g) / 2
}

fn di_edge_count(g: &SquareCSR2D<CSR2D<usize, usize, usize>>) -> usize {
    Edges::number_of_edges(g)
}

/// Wrap a SymmetricCSR2D from a graph generator into a full UndiGraph.
fn wrap_undi(g: UndirectedGraph) -> UndiGraph<usize> {
    let n = g.order();
    let nodes: SortedVec<usize> = GenericVocabularyBuilder::default()
        .expected_number_of_symbols(n)
        .symbols((0..n).enumerate())
        .build()
        .unwrap();
    UndiGraph::from((nodes, g))
}

/// Build a DiGraph from a list of edges.
fn build_digraph(n: usize, mut edges: Vec<(usize, usize)>) -> DiGraph<usize> {
    edges.sort_unstable();
    let nodes: SortedVec<usize> = GenericVocabularyBuilder::default()
        .expected_number_of_symbols(n)
        .symbols((0..n).enumerate())
        .build()
        .unwrap();
    let edge_mat: SquareCSR2D<_> = DiEdgesBuilder::default()
        .expected_number_of_edges(edges.len())
        .expected_shape(n)
        .edges(edges.into_iter())
        .build()
        .unwrap();
    DiGraph::from((nodes, edge_mat))
}

// ============================================================================
// Undirected line graph tests
// ============================================================================

#[test]
fn test_line_graph_empty() {
    let g = wrap_undi(path_graph(1));
    let lg = g.line_graph();
    assert_eq!(lg.number_of_vertices(), 0);
    assert_eq!(lg.graph().order(), 0);
}

#[test]
fn test_line_graph_single_edge() {
    let g = wrap_undi(complete_graph(2));
    let lg = g.line_graph();
    assert_eq!(lg.number_of_vertices(), 1);
    assert_eq!(undi_edge_count(lg.graph()), 0);
}

#[test]
fn test_line_graph_path_p3() {
    let g = wrap_undi(path_graph(3));
    let lg = g.line_graph();
    assert_eq!(lg.number_of_vertices(), 2);
    assert_eq!(undi_edge_count(lg.graph()), 1);
}

#[test]
fn test_line_graph_triangle_k3() {
    let g = wrap_undi(complete_graph(3));
    let lg = g.line_graph();
    assert_eq!(lg.number_of_vertices(), 3);
    assert_eq!(undi_edge_count(lg.graph()), 3);
}

#[test]
fn test_line_graph_star_s4() {
    let g = wrap_undi(star_graph(4));
    let lg = g.line_graph();
    assert_eq!(lg.number_of_vertices(), 3);
    assert_eq!(undi_edge_count(lg.graph()), 3);
}

#[test]
fn test_line_graph_k4() {
    let g = wrap_undi(complete_graph(4));
    let lg = g.line_graph();
    assert_eq!(lg.number_of_vertices(), 6);
    assert_eq!(undi_edge_count(lg.graph()), 12);
}

#[test]
fn test_line_graph_cycle_c4() {
    let g = wrap_undi(cycle_graph(4));
    let lg = g.line_graph();
    assert_eq!(lg.number_of_vertices(), 4);
    assert_eq!(undi_edge_count(lg.graph()), 4);
}

#[test]
fn test_line_graph_cycle_c5() {
    let g = wrap_undi(cycle_graph(5));
    let lg = g.line_graph();
    assert_eq!(lg.number_of_vertices(), 5);
    assert_eq!(undi_edge_count(lg.graph()), 5);
}

#[test]
fn test_line_graph_complete_graphs() {
    for n in 3..=7 {
        let g = wrap_undi(complete_graph(n));
        let lg = g.line_graph();
        let expected_v = n * (n - 1) / 2;
        let expected_e = n * (n - 1) * (n - 2) / 2;
        assert_eq!(lg.number_of_vertices(), expected_v, "L(K{n}): expected {expected_v} vertices");
        assert_eq!(undi_edge_count(lg.graph()), expected_e, "L(K{n}): expected {expected_e} edges");
    }
}

#[test]
fn test_line_graph_edge_formula() {
    for n in [5, 8, 10] {
        let raw = erdos_renyi_gnp(42, n, 0.4);
        let g = wrap_undi(raw.clone());
        let lg = g.line_graph();

        let expected_lg_edges: usize = (0..n)
            .map(|v| {
                let d: usize = Edges::out_degree(&raw, v);
                d.saturating_sub(1) * d / 2
            })
            .sum();

        assert_eq!(
            undi_edge_count(lg.graph()),
            expected_lg_edges,
            "Edge count formula failed for n={n}"
        );
    }
}

#[test]
fn test_line_graph_edge_map_consistency() {
    let g = wrap_undi(complete_graph(5));
    let lg = g.line_graph();
    let em = lg.edge_map();

    for (src, dst) in Edges::sparse_coordinates(lg.graph()) {
        if src < dst {
            let (a1, a2) = em[src];
            let (b1, b2) = em[dst];
            let shared = a1 == b1 || a1 == b2 || a2 == b1 || a2 == b2;
            assert!(
                shared,
                "L(G) edge ({src},{dst}): original edges ({a1:?},{a2:?}) and ({b1:?},{b2:?}) share no endpoint"
            );
        }
    }
}

// ============================================================================
// Directed line graph tests
// ============================================================================

#[test]
fn test_directed_line_graph_chain() {
    let g = build_digraph(3, vec![(0, 1), (1, 2)]);
    let lg = g.directed_line_graph();
    assert_eq!(lg.number_of_vertices(), 2);
    assert_eq!(di_edge_count(lg.graph()), 1);
}

#[test]
fn test_directed_line_graph_diamond() {
    // 0->1, 0->2, 1->3, 2->3
    // |E(L)| = sum_v in_deg(v)*out_deg(v) = 0*2 + 1*1 + 1*1 + 2*0 = 2
    let g = build_digraph(4, vec![(0, 1), (0, 2), (1, 3), (2, 3)]);
    let lg = g.directed_line_graph();
    assert_eq!(lg.number_of_vertices(), 4);
    assert_eq!(di_edge_count(lg.graph()), 2);
}

#[test]
fn test_directed_line_graph_self_loop() {
    // Edges: (0,1)=idx0, (1,1)=idx1, (1,2)=idx2
    // outgoing[0]=[0], outgoing[1]=[1,2]
    // incoming[1]=[0,1], incoming[2]=[2]
    // Vertex v=1: incoming[1] x outgoing[1] = {0,1} x {1,2} =
    // (0,1),(0,2),(1,1),(1,2) So the self-loop edge (1,1) at index 1 has a
    // self-loop in L(G).
    let g = build_digraph(3, vec![(0, 1), (1, 1), (1, 2)]);
    let lg = g.directed_line_graph();
    assert_eq!(lg.number_of_vertices(), 3);

    // Edge (1,1) is index 1 in sorted order.
    let self_loop_vertex = 1;
    assert_eq!(lg.original_edge(self_loop_vertex), (1, 1));
    let succs: Vec<usize> = Edges::successors(lg.graph(), self_loop_vertex).collect();
    assert!(succs.contains(&self_loop_vertex), "Self-loop vertex should have a self-loop in L(G)");
}

#[test]
fn test_directed_line_graph_complete_k3() {
    // 6 arcs, in_deg = out_deg = 2 for each vertex.
    // |E(L)| = 3 * 2 * 2 = 12.
    let g = build_digraph(3, vec![(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]);
    let lg = g.directed_line_graph();
    assert_eq!(lg.number_of_vertices(), 6);
    assert_eq!(di_edge_count(lg.graph()), 12);
}

#[test]
fn test_directed_line_graph_edge_count_formula() {
    let edges = vec![(0, 1), (0, 2), (1, 2), (2, 0), (2, 3), (3, 1)];
    let n = 4;
    let g = build_digraph(n, edges.clone());

    let mut in_deg = vec![0usize; n];
    let mut out_deg = vec![0usize; n];
    for &(s, d) in &edges {
        out_deg[s] += 1;
        in_deg[d] += 1;
    }
    let expected: usize = (0..n).map(|v| in_deg[v] * out_deg[v]).sum();

    let lg = g.directed_line_graph();
    assert_eq!(di_edge_count(lg.graph()), expected);
}

#[test]
fn test_directed_line_graph_empty() {
    let g = build_digraph(2, Vec::new());
    let lg = g.directed_line_graph();
    assert_eq!(lg.number_of_vertices(), 0);
    assert_eq!(di_edge_count(lg.graph()), 0);
}

#[test]
fn test_line_graph_into_graph() {
    let g = wrap_undi(path_graph(3));
    let lg = g.line_graph();
    let graph = lg.into_graph();
    assert_eq!(undi_edge_count(&graph), 1);
}
