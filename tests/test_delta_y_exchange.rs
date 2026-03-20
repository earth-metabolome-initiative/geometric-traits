//! Integration tests for DeltaYExchange trait.
#![cfg(feature = "std")]

use geometric_traits::{
    impls::{BitSquareMatrix, CSR2D, SortedVec, SymmetricCSR2D},
    prelude::*,
    traits::{SquareMatrix, VocabularyBuilder, algorithms::randomized_graphs::*},
};

type UndirectedGraph = SymmetricCSR2D<CSR2D<usize, usize, usize>>;

/// Wrap a SymmetricCSR2D into a full UndiGraph (for line_graph).
fn wrap_undi(g: UndirectedGraph) -> UndiGraph<usize> {
    let n = g.order();
    let nodes: SortedVec<usize> = GenericVocabularyBuilder::default()
        .expected_number_of_symbols(n)
        .symbols((0..n).enumerate())
        .build()
        .unwrap();
    UndiGraph::from((nodes, g))
}

// ============================================================================
// Tests using SymmetricCSR2D directly to verify blanket impl
// ============================================================================

#[test]
fn test_delta_y_on_symmetric_csr2d_k3_vs_k1_3() {
    let k3 = complete_graph(3);
    let k1_3 = star_graph(4); // K_{1,3}

    let k3_edges: Vec<(usize, usize)> = vec![(0, 1), (0, 2), (1, 2)];
    let k1_3_edges: Vec<(usize, usize)> = vec![(0, 1), (0, 2), (0, 3)];

    assert_eq!(k3.edge_subgraph_degree_sequence(&k3_edges), vec![2, 2, 2]);
    assert_eq!(k1_3.edge_subgraph_degree_sequence(&k1_3_edges), vec![1, 1, 1, 3]);
    assert!(k3.has_delta_y_exchange(&k3_edges, &k1_3, &k1_3_edges));
}

#[test]
fn test_delta_y_on_symmetric_csr2d_same_structure() {
    let g1 = path_graph(4);
    let g2 = path_graph(4);
    let edges: Vec<(usize, usize)> = vec![(0, 1), (1, 2), (2, 3)];

    assert!(!g1.has_delta_y_exchange(&edges, &g2, &edges));
}

#[test]
fn test_delta_y_cross_type_bitsquarematrix_vs_symmetric_csr2d() {
    let bsm = BitSquareMatrix::from_symmetric_edges(3, vec![(0, 1), (0, 2), (1, 2)]);
    let csr = star_graph(4);

    let bsm_edges: Vec<(usize, usize)> = vec![(0, 1), (0, 2), (1, 2)];
    let csr_edges: Vec<(usize, usize)> = vec![(0, 1), (0, 2), (0, 3)];

    assert!(bsm.has_delta_y_exchange(&bsm_edges, &csr, &csr_edges));
}

#[test]
fn test_delta_y_symmetry() {
    let g1 = complete_graph(3);
    let g2 = star_graph(4);

    let e1: Vec<(usize, usize)> = vec![(0, 1), (0, 2), (1, 2)];
    let e2: Vec<(usize, usize)> = vec![(0, 1), (0, 2), (0, 3)];

    let fwd = g1.has_delta_y_exchange(&e1, &g2, &e2);
    let rev = g2.has_delta_y_exchange(&e2, &g1, &e1);
    assert_eq!(fwd, rev);
}

// ============================================================================
// End-to-end: line graph → modular product → max clique → Delta-Y check
// ============================================================================

#[test]
fn test_end_to_end_line_graph_delta_y() {
    // G1 = K3 (triangle), G2 = K1,3 (claw).
    // Both have isomorphic line graphs (K3), so the modular product of
    // L(G1) and L(G2) will yield a maximum clique of size 3.
    // Delta-Y detection should flag this as invalid.
    let g1_raw = complete_graph(3);
    let g2_raw = star_graph(4);
    let g1 = wrap_undi(g1_raw.clone());
    let g2 = wrap_undi(g2_raw.clone());

    let lg1 = g1.line_graph();
    let lg2 = g2.line_graph();

    assert_eq!(lg1.number_of_vertices(), 3);
    assert_eq!(lg2.number_of_vertices(), 3);

    let n1 = lg1.number_of_vertices();
    let n2 = lg2.number_of_vertices();
    let pairs: Vec<(usize, usize)> = (0..n1).flat_map(|i| (0..n2).map(move |j| (i, j))).collect();

    let mp = lg1.graph().modular_product(lg2.graph(), &pairs);
    let cliques = mp.all_maximum_cliques();

    assert!(!cliques.is_empty());
    assert_eq!(cliques[0].len(), 3);

    // For each clique, map back to original edges and check Delta-Y.
    // Use the raw SymmetricCSR2D for DeltaYExchange (it implements
    // SparseSquareMatrix).
    for clique in &cliques {
        let self_edges: Vec<(usize, usize)> = clique
            .iter()
            .map(|&v| {
                let (i, _) = (v / n2, v % n2);
                lg1.original_edge(i)
            })
            .collect();
        let other_edges: Vec<(usize, usize)> = clique
            .iter()
            .map(|&v| {
                let (_, j) = (v / n2, v % n2);
                lg2.original_edge(j)
            })
            .collect();

        assert!(
            g1_raw.has_delta_y_exchange(&self_edges, &g2_raw, &other_edges),
            "Delta-Y exchange should be detected for K3 vs K1,3 clique"
        );
    }
}

#[test]
fn test_end_to_end_no_delta_y_for_matching_graphs() {
    // Two copies of the same graph: K4. No Delta-Y should be detected.
    let g1_raw = complete_graph(4);
    let g2_raw = complete_graph(4);
    let g1 = wrap_undi(g1_raw.clone());
    let g2 = wrap_undi(g2_raw.clone());

    let lg1 = g1.line_graph();
    let lg2 = g2.line_graph();

    let n1 = lg1.number_of_vertices();
    let n2 = lg2.number_of_vertices();
    let pairs: Vec<(usize, usize)> = (0..n1).flat_map(|i| (0..n2).map(move |j| (i, j))).collect();

    let mp = lg1.graph().modular_product(lg2.graph(), &pairs);
    let clique = mp.maximum_clique();

    let self_edges: Vec<(usize, usize)> = clique
        .iter()
        .map(|&v| {
            let (i, _) = (v / n2, v % n2);
            lg1.original_edge(i)
        })
        .collect();
    let other_edges: Vec<(usize, usize)> = clique
        .iter()
        .map(|&v| {
            let (_, j) = (v / n2, v % n2);
            lg2.original_edge(j)
        })
        .collect();

    assert!(
        !g1_raw.has_delta_y_exchange(&self_edges, &g2_raw, &other_edges),
        "No Delta-Y exchange expected for identical K4 graphs"
    );
}
