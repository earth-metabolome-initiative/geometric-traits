//! Tests for undirected graph generator families.
#![cfg(feature = "std")]

use geometric_traits::{
    impls::{CSR2D, SymmetricCSR2D},
    prelude::*,
    traits::algorithms::randomized_graphs::*,
};

type UndirectedGraph = SymmetricCSR2D<CSR2D<usize, usize, usize>>;

fn edge_count(g: &UndirectedGraph) -> usize {
    // SymmetricCSR2D stores both directions; upper-triangular count = total / 2
    geometric_traits::traits::Edges::number_of_edges(g) / 2
}

fn same_graph(left: &UndirectedGraph, right: &UndirectedGraph) -> bool {
    left.order() == right.order()
        && geometric_traits::traits::Edges::number_of_edges(left)
            == geometric_traits::traits::Edges::number_of_edges(right)
        && (0..left.order()).all(|row| left.sparse_row(row).eq(right.sparse_row(row)))
}

fn windmill_matching_size(num_cliques: usize, clique_size: usize) -> usize {
    num_cliques * ((clique_size - 1) / 2) + usize::from(clique_size % 2 == 0)
}

// ============================================================================
// Deterministic graph families
// ============================================================================

#[test]
fn test_complete_graph_k1() {
    let g = complete_graph(1);
    assert_eq!(g.order(), 1);
    assert_eq!(edge_count(&g), 0);
}

#[test]
fn test_complete_graph_k4() {
    let g = complete_graph(4);
    assert_eq!(g.order(), 4);
    assert_eq!(edge_count(&g), 6);
}

#[test]
fn test_complete_graph_k7() {
    let g = complete_graph(7);
    assert_eq!(g.order(), 7);
    assert_eq!(edge_count(&g), 21);
}

#[test]
fn test_cycle_graph_c5() {
    let g = cycle_graph(5);
    assert_eq!(g.order(), 5);
    assert_eq!(edge_count(&g), 5);
}

#[test]
fn test_cycle_graph_c3() {
    let g = cycle_graph(3);
    assert_eq!(g.order(), 3);
    assert_eq!(edge_count(&g), 3);
}

#[test]
fn test_cycle_graph_c2() {
    // Degenerate: fewer than 3 vertices
    let g = cycle_graph(2);
    assert_eq!(g.order(), 2);
}

#[test]
fn test_path_graph_p5() {
    let g = path_graph(5);
    assert_eq!(g.order(), 5);
    assert_eq!(edge_count(&g), 4);
}

#[test]
fn test_path_graph_p1() {
    let g = path_graph(1);
    assert_eq!(g.order(), 1);
    assert_eq!(edge_count(&g), 0);
}

#[test]
fn test_star_graph_s5() {
    let g = star_graph(5);
    assert_eq!(g.order(), 5);
    assert_eq!(edge_count(&g), 4);
}

#[test]
fn test_star_graph_matching() {
    // Star: max matching = 1
    let g = star_graph(6);
    let matching = g.blossom();
    assert_eq!(matching.len(), 1);
}

#[test]
fn test_grid_graph_2x3() {
    let g = grid_graph(2, 3);
    // 6 vertices, 2*3*2 - 2 - 3 = 7 edges
    assert_eq!(g.order(), 6);
    assert_eq!(edge_count(&g), 7);
}

#[test]
fn test_grid_graph_3x3() {
    let g = grid_graph(3, 3);
    // 9 vertices, 12 edges
    assert_eq!(g.order(), 9);
    assert_eq!(edge_count(&g), 12);
}

#[test]
fn test_hexagonal_lattice_graph_1x1() {
    let g = hexagonal_lattice_graph(1, 1);
    // Single hexagon = C6
    assert_eq!(g.order(), 6);
    assert_eq!(edge_count(&g), 6);
    assert_eq!(g.blossom().len(), 3);
}

#[test]
fn test_hexagonal_lattice_graph_zero_dimension() {
    let empty_rows = hexagonal_lattice_graph(0, 3);
    assert_eq!(empty_rows.order(), 0);
    assert_eq!(edge_count(&empty_rows), 0);

    let empty_cols = hexagonal_lattice_graph(3, 0);
    assert_eq!(empty_cols.order(), 0);
    assert_eq!(edge_count(&empty_cols), 0);
}

#[test]
fn test_hexagonal_lattice_graph_1x2() {
    let g = hexagonal_lattice_graph(1, 2);
    // Two fused hexagons
    assert_eq!(g.order(), 10);
    assert_eq!(edge_count(&g), 11);
    assert_eq!(g.blossom().len(), 5);
}

#[test]
fn test_hexagonal_lattice_graph_2x2() {
    let g = hexagonal_lattice_graph(2, 2);
    // 4-cell parallelogram benzenoid patch
    assert_eq!(g.order(), 16);
    assert_eq!(edge_count(&g), 19);
    assert_eq!(g.blossom().len(), 8);
}

#[test]
fn test_triangular_lattice_graph_zero_dimension() {
    let empty = triangular_lattice_graph(0, 0);
    assert_eq!(empty.order(), 0);
    assert_eq!(edge_count(&empty), 0);

    let empty_rows = triangular_lattice_graph(0, 3);
    assert_eq!(empty_rows.order(), 0);
    assert_eq!(edge_count(&empty_rows), 0);

    let empty_cols = triangular_lattice_graph(3, 0);
    assert_eq!(empty_cols.order(), 0);
    assert_eq!(edge_count(&empty_cols), 0);
}

#[test]
fn test_triangular_lattice_graph_1x1() {
    let g = triangular_lattice_graph(1, 1);
    assert_eq!(g.order(), 1);
    assert_eq!(edge_count(&g), 0);
}

#[test]
fn test_triangular_lattice_graph_1x4_matches_path() {
    let g = triangular_lattice_graph(1, 4);
    assert_eq!(g.order(), 4);
    assert_eq!(edge_count(&g), 3);
    assert!(same_graph(&g, &path_graph(4)));
}

#[test]
fn test_triangular_lattice_graph_4x1_matches_path() {
    let g = triangular_lattice_graph(4, 1);
    assert_eq!(g.order(), 4);
    assert_eq!(edge_count(&g), 3);
    assert!(same_graph(&g, &path_graph(4)));
}

#[test]
fn test_triangular_lattice_graph_2x2() {
    let g = triangular_lattice_graph(2, 2);
    // 4 vertices with one diagonal added to each unit cell
    assert_eq!(g.order(), 4);
    assert_eq!(edge_count(&g), 5);
    assert_eq!(g.sparse_row(0).collect::<Vec<_>>(), vec![1, 2, 3]);
    assert_eq!(g.sparse_row(1).collect::<Vec<_>>(), vec![0, 3]);
    assert_eq!(g.sparse_row(2).collect::<Vec<_>>(), vec![0, 3]);
    assert_eq!(g.sparse_row(3).collect::<Vec<_>>(), vec![0, 1, 2]);
}

#[test]
fn test_triangular_lattice_graph_2x3_orientation() {
    let g = triangular_lattice_graph(2, 3);
    assert_eq!(g.order(), 6);
    assert_eq!(edge_count(&g), 9);
    assert_eq!(g.sparse_row(0).collect::<Vec<_>>(), vec![1, 3, 4]);
    assert_eq!(g.sparse_row(1).collect::<Vec<_>>(), vec![0, 2, 4, 5]);
    assert_eq!(g.sparse_row(2).collect::<Vec<_>>(), vec![1, 5]);
    assert_eq!(g.sparse_row(3).collect::<Vec<_>>(), vec![0, 4]);
    assert_eq!(g.sparse_row(4).collect::<Vec<_>>(), vec![0, 1, 3, 5]);
    assert_eq!(g.sparse_row(5).collect::<Vec<_>>(), vec![1, 2, 4]);
}

#[test]
fn test_triangular_lattice_graph_3x3() {
    let g = triangular_lattice_graph(3, 3);
    assert_eq!(g.order(), 9);
    assert_eq!(edge_count(&g), 16);
    assert_eq!(g.sparse_row(4).count(), 6);
}

#[test]
fn test_torus_graph_3x3() {
    let g = torus_graph(3, 3);
    // 9 vertices, 4-regular → 9*4/2 = 18 edges
    assert_eq!(g.order(), 9);
    assert_eq!(edge_count(&g), 18);
}

#[test]
fn test_torus_graph_4x4() {
    let g = torus_graph(4, 4);
    // 16 vertices, 4-regular → 32 edges
    assert_eq!(g.order(), 16);
    assert_eq!(edge_count(&g), 32);
}

#[test]
fn test_hypercube_q3() {
    let g = hypercube_graph(3);
    // 8 vertices, 12 edges, 3-regular
    assert_eq!(g.order(), 8);
    assert_eq!(edge_count(&g), 12);
}

#[test]
fn test_hypercube_q4() {
    let g = hypercube_graph(4);
    // 16 vertices, 32 edges, 4-regular
    assert_eq!(g.order(), 16);
    assert_eq!(edge_count(&g), 32);
}

#[test]
fn test_hypercube_q1() {
    let g = hypercube_graph(1);
    // 2 vertices, 1 edge
    assert_eq!(g.order(), 2);
    assert_eq!(edge_count(&g), 1);
}

#[test]
fn test_barbell_graph() {
    // Two K_3 connected by path of 2 internal vertices
    let g = barbell_graph(3, 2);
    // 2*3 + 2 = 8 vertices
    // K_3 edges: 3 each = 6
    // Bridge path: 3 edges (k-1 → k, k → k+1, k+1 → k+2 where k+2 is start of
    // second clique)
    assert_eq!(g.order(), 8);
}

#[test]
fn test_barbell_graph_no_path() {
    // Two K_3 directly connected
    let g = barbell_graph(3, 0);
    // 6 vertices, 3 + 3 + 1 = 7 edges
    assert_eq!(g.order(), 6);
    assert_eq!(edge_count(&g), 7);
}

#[test]
fn test_crown_graph_3() {
    let g = crown_graph(3);
    // 6 vertices, 3*(3-1) = 6 edges, 2-regular
    assert_eq!(g.order(), 6);
    assert_eq!(edge_count(&g), 6);
}

#[test]
fn test_crown_graph_4() {
    let g = crown_graph(4);
    // 8 vertices, 4*3 = 12 edges, 3-regular
    assert_eq!(g.order(), 8);
    assert_eq!(edge_count(&g), 12);
}

#[test]
fn test_wheel_graph_5() {
    let g = wheel_graph(5);
    // 6 vertices, 10 edges
    assert_eq!(g.order(), 6);
    assert_eq!(edge_count(&g), 10);
}

#[test]
fn test_complete_bipartite_k33() {
    let g = complete_bipartite_graph(3, 3);
    // 6 vertices, 9 edges
    assert_eq!(g.order(), 6);
    assert_eq!(edge_count(&g), 9);
}

#[test]
fn test_complete_bipartite_k27() {
    let g = complete_bipartite_graph(2, 7);
    // 9 vertices, 14 edges
    assert_eq!(g.order(), 9);
    assert_eq!(edge_count(&g), 14);
}

#[test]
fn test_petersen_graph() {
    let g = petersen_graph();
    assert_eq!(g.order(), 10);
    assert_eq!(edge_count(&g), 15);
    // 3-regular
    let matching = g.blossom();
    assert_eq!(matching.len(), 5);
}

#[test]
fn test_turan_graph_6_3() {
    let g = turan_graph(6, 3);
    // T(6,3): 3 parts of 2, edges = 6*5/2 - 3*1 = 12
    assert_eq!(g.order(), 6);
    assert_eq!(edge_count(&g), 12);
}

#[test]
fn test_turan_graph_7_2() {
    let g = turan_graph(7, 2);
    // T(7,2): bipartite (3,4), edges = 3*4 = 12
    assert_eq!(g.order(), 7);
    assert_eq!(edge_count(&g), 12);
}

#[test]
fn test_friendship_graph_3() {
    let g = friendship_graph(3);
    // 7 vertices, 9 edges
    assert_eq!(g.order(), 7);
    assert_eq!(edge_count(&g), 9);
    // Max matching = 3 (one per triangle, hub exposed)
    let matching = g.blossom();
    assert_eq!(matching.len(), 3);
}

#[test]
fn test_friendship_graph_0() {
    let g = friendship_graph(0);
    assert_eq!(g.order(), 1);
    assert_eq!(edge_count(&g), 0);
}

#[test]
fn test_windmill_graph_1_4() {
    let g = windmill_graph(1, 4);
    assert_eq!(g.order(), 4);
    assert_eq!(edge_count(&g), 6);
    assert_eq!(g.blossom().len(), windmill_matching_size(1, 4));
}

#[test]
fn test_windmill_graph_3_4() {
    let g = windmill_graph(3, 4);
    assert_eq!(g.order(), 10);
    assert_eq!(edge_count(&g), 18);
    assert_eq!(g.blossom().len(), windmill_matching_size(3, 4));
}

#[test]
fn test_windmill_graph_5_2_matches_star() {
    let g = windmill_graph(5, 2);
    let star = star_graph(6);
    assert!(same_graph(&g, &star));
}

#[test]
fn test_windmill_graph_4_3_matches_friendship() {
    let windmill = windmill_graph(4, 3);
    let friendship = friendship_graph(4);
    assert!(same_graph(&windmill, &friendship));
}

#[test]
#[should_panic(expected = "windmill_graph requires num_cliques >= 1")]
fn test_windmill_graph_zero_cliques_panics() {
    let _ = windmill_graph(0, 3);
}

#[test]
#[should_panic(expected = "windmill_graph requires clique_size >= 2")]
fn test_windmill_graph_clique_size_one_panics() {
    let _ = windmill_graph(2, 1);
}

// ============================================================================
// Random graph models
// ============================================================================

#[test]
fn test_erdos_renyi_gnm_deterministic() {
    let g1 = erdos_renyi_gnm(42, 20, 30);
    let g2 = erdos_renyi_gnm(42, 20, 30);
    assert_eq!(g1.order(), g2.order());
    assert_eq!(
        geometric_traits::traits::Edges::number_of_edges(&g1),
        geometric_traits::traits::Edges::number_of_edges(&g2)
    );
}

#[test]
fn test_erdos_renyi_gnm_edge_count() {
    let g = erdos_renyi_gnm(123, 10, 15);
    assert_eq!(g.order(), 10);
    assert_eq!(edge_count(&g), 15);
}

#[test]
fn test_erdos_renyi_gnm_zero_edges() {
    let g = erdos_renyi_gnm(1, 10, 0);
    assert_eq!(g.order(), 10);
    assert_eq!(edge_count(&g), 0);
}

#[test]
fn test_erdos_renyi_gnm_max_edges() {
    // Request more edges than possible → capped
    let g = erdos_renyi_gnm(1, 5, 100);
    assert_eq!(g.order(), 5);
    assert_eq!(edge_count(&g), 10); // 5*4/2 = 10
}

#[test]
fn test_erdos_renyi_gnp_empty() {
    let g = erdos_renyi_gnp(42, 10, 0.0);
    assert_eq!(g.order(), 10);
    assert_eq!(edge_count(&g), 0);
}

#[test]
fn test_erdos_renyi_gnp_complete() {
    let g = erdos_renyi_gnp(42, 5, 1.0);
    assert_eq!(g.order(), 5);
    assert_eq!(edge_count(&g), 10);
}

#[test]
fn test_erdos_renyi_gnp_deterministic() {
    let g1 = erdos_renyi_gnp(99, 20, 0.3);
    let g2 = erdos_renyi_gnp(99, 20, 0.3);
    assert_eq!(
        geometric_traits::traits::Edges::number_of_edges(&g1),
        geometric_traits::traits::Edges::number_of_edges(&g2)
    );
}

#[test]
fn test_barabasi_albert_basic() {
    let g = barabasi_albert(42, 20, 2);
    assert_eq!(g.order(), 20);
    // At least m*(n-m-1) edges from attachment + initial clique
    assert!(edge_count(&g) >= 3);
}

#[test]
fn test_barabasi_albert_deterministic() {
    let g1 = barabasi_albert(42, 30, 3);
    let g2 = barabasi_albert(42, 30, 3);
    assert_eq!(
        geometric_traits::traits::Edges::number_of_edges(&g1),
        geometric_traits::traits::Edges::number_of_edges(&g2)
    );
}

#[test]
fn test_barabasi_albert_small() {
    // n <= m+1 → just the initial clique
    let g = barabasi_albert(42, 3, 3);
    assert_eq!(g.order(), 3);
}

#[test]
fn test_watts_strogatz_no_rewiring() {
    // beta = 0 → pure ring lattice
    let g = watts_strogatz(42, 10, 4, 0.0);
    assert_eq!(g.order(), 10);
    assert_eq!(edge_count(&g), 20); // n*k/2 = 10*4/2 = 20
}

#[test]
fn test_watts_strogatz_deterministic() {
    let g1 = watts_strogatz(42, 20, 4, 0.3);
    let g2 = watts_strogatz(42, 20, 4, 0.3);
    assert_eq!(
        geometric_traits::traits::Edges::number_of_edges(&g1),
        geometric_traits::traits::Edges::number_of_edges(&g2)
    );
}

#[test]
fn test_random_regular_graph_basic() {
    let g = random_regular_graph(42, 10, 4).expect("10-vertex 4-regular graph should exist");
    assert_eq!(g.order(), 10);
    assert_eq!(edge_count(&g), 20); // n*k/2
}

#[test]
fn test_random_regular_graph_deterministic() {
    let g1 = random_regular_graph(42, 12, 4).expect("12-vertex 4-regular graph should exist");
    let g2 = random_regular_graph(42, 12, 4).expect("12-vertex 4-regular graph should exist");
    assert_eq!(
        geometric_traits::traits::Edges::number_of_edges(&g1),
        geometric_traits::traits::Edges::number_of_edges(&g2)
    );
}

#[test]
fn test_stochastic_block_model_deterministic() {
    let g1 = stochastic_block_model(42, &[5, 5], 0.8, 0.1);
    let g2 = stochastic_block_model(42, &[5, 5], 0.8, 0.1);
    assert_eq!(
        geometric_traits::traits::Edges::number_of_edges(&g1),
        geometric_traits::traits::Edges::number_of_edges(&g2)
    );
}

#[test]
fn test_stochastic_block_model_no_edges() {
    let g = stochastic_block_model(42, &[5, 5], 0.0, 0.0);
    assert_eq!(g.order(), 10);
    assert_eq!(edge_count(&g), 0);
}

#[test]
fn test_configuration_model_basic() {
    // Regular degree-2 sequence (like a cycle)
    let g = configuration_model(42, &[2, 2, 2, 2, 2]);
    assert_eq!(g.order(), 5);
    // May lose some edges due to self-loops/multi-edges
    assert!(edge_count(&g) <= 5);
}

#[test]
fn test_chung_lu_deterministic() {
    let g1 = chung_lu(42, &[3.0, 3.0, 3.0, 3.0, 3.0]);
    let g2 = chung_lu(42, &[3.0, 3.0, 3.0, 3.0, 3.0]);
    assert_eq!(
        geometric_traits::traits::Edges::number_of_edges(&g1),
        geometric_traits::traits::Edges::number_of_edges(&g2)
    );
}

#[test]
fn test_random_geometric_graph_deterministic() {
    let g1 = random_geometric_graph(42, 20, 0.3);
    let g2 = random_geometric_graph(42, 20, 0.3);
    assert_eq!(
        geometric_traits::traits::Edges::number_of_edges(&g1),
        geometric_traits::traits::Edges::number_of_edges(&g2)
    );
}

#[test]
fn test_random_geometric_graph_zero_radius() {
    let g = random_geometric_graph(42, 10, 0.0);
    assert_eq!(g.order(), 10);
    assert_eq!(edge_count(&g), 0);
}

// ============================================================================
// Cross-checks: matching algorithm agrees across generators
// ============================================================================

#[test]
fn test_complete_matching() {
    // K_n perfect matching = floor(n/2)
    for n in 2..=8 {
        let g = complete_graph(n);
        let matching = g.blossom();
        assert_eq!(matching.len(), n / 2, "K_{n} matching size");
    }
}

#[test]
fn test_cycle_matching() {
    // C_n matching = floor(n/2)
    for n in 3..=10 {
        let g = cycle_graph(n);
        let matching = g.blossom();
        assert_eq!(matching.len(), n / 2, "C_{n} matching size");
    }
}

#[test]
fn test_path_matching() {
    // P_n matching = floor(n/2)
    for n in 2..=10 {
        let g = path_graph(n);
        let matching = g.blossom();
        assert_eq!(matching.len(), n / 2, "P_{n} matching size");
    }
}

#[test]
fn test_crown_perfect_matching() {
    // Crown graph has a perfect matching of size n
    for n in 2..=6 {
        let g = crown_graph(n);
        let matching = g.blossom();
        assert_eq!(matching.len(), n, "Cr_{n} perfect matching");
    }
}

#[test]
fn test_hypercube_perfect_matching() {
    // Q_d has a perfect matching for d >= 1
    for d in 1..=4 {
        let g = hypercube_graph(d);
        let matching = g.blossom();
        assert_eq!(matching.len(), 1 << (d - 1), "Q_{d} perfect matching");
    }
}

#[test]
fn test_complete_bipartite_matching() {
    // K_{m,n} matching = min(m, n)
    let g = complete_bipartite_graph(3, 5);
    let matching = g.blossom();
    assert_eq!(matching.len(), 3);
}

#[test]
fn test_grid_matching() {
    // 2x3 grid: perfect matching = 3
    let g = grid_graph(2, 3);
    let matching = g.blossom();
    assert_eq!(matching.len(), 3);
}

// ============================================================================
// Blossom vs Micali-Vazirani cross-check on random graphs
// ============================================================================

#[test]
fn test_erdos_renyi_blossom_vs_mv() {
    for seed in 1..=5 {
        let g = erdos_renyi_gnm(seed, 30, 50);
        let blossom = g.blossom();
        let mv = g.micali_vazirani();
        assert_eq!(blossom.len(), mv.len(), "seed={seed} matching size mismatch");
    }
}

#[test]
fn test_barabasi_albert_blossom_vs_mv() {
    for seed in 1..=5 {
        let g = barabasi_albert(seed, 30, 3);
        let blossom = g.blossom();
        let mv = g.micali_vazirani();
        assert_eq!(blossom.len(), mv.len(), "seed={seed} matching size mismatch");
    }
}

// ============================================================================
// Edge-case coverage for generator branches
// ============================================================================

#[test]
fn test_star_graph_n0() {
    let g = star_graph(0);
    assert_eq!(g.order(), 0);
}

#[test]
fn test_star_graph_n1() {
    let g = star_graph(1);
    assert_eq!(g.order(), 1);
    assert_eq!(edge_count(&g), 0);
}

#[test]
fn test_barabasi_albert_n0() {
    let g = barabasi_albert(42, 0, 1);
    assert_eq!(g.order(), 0);
}

#[test]
fn test_barabasi_albert_n1() {
    let g = barabasi_albert(42, 1, 1);
    assert_eq!(g.order(), 1);
    assert_eq!(edge_count(&g), 0);
}

#[test]
fn test_erdos_renyi_gnm_n0() {
    let g = erdos_renyi_gnm(42, 0, 0);
    assert_eq!(g.order(), 0);
}

#[test]
fn test_erdos_renyi_gnm_n1() {
    let g = erdos_renyi_gnm(42, 1, 5);
    assert_eq!(g.order(), 1);
    assert_eq!(edge_count(&g), 0);
}

#[test]
fn test_stochastic_block_model_n0() {
    let g = stochastic_block_model(42, &[], 0.5, 0.1);
    assert_eq!(g.order(), 0);
}

#[test]
fn test_stochastic_block_model_n1() {
    let g = stochastic_block_model(42, &[1], 0.5, 0.1);
    assert_eq!(g.order(), 1);
    assert_eq!(edge_count(&g), 0);
}

#[test]
fn test_chung_lu_n0() {
    let g = chung_lu(42, &[]);
    assert_eq!(g.order(), 0);
}

#[test]
fn test_chung_lu_n1() {
    let g = chung_lu(42, &[5.0]);
    assert_eq!(g.order(), 1);
    assert_eq!(edge_count(&g), 0);
}

#[test]
fn test_chung_lu_zero_weights() {
    let g = chung_lu(42, &[0.0, 0.0, 0.0]);
    assert_eq!(g.order(), 3);
    assert_eq!(edge_count(&g), 0);
}

#[test]
fn test_chung_lu_mixed_zero_weights() {
    // Some zero weights → p <= 0.0 for those pairs
    let g = chung_lu(42, &[0.0, 5.0, 5.0, 0.0]);
    assert_eq!(g.order(), 4);
    // Only pair (1,2) can have an edge
    assert!(edge_count(&g) <= 1);
}

#[test]
fn test_configuration_model_empty() {
    let g = configuration_model(42, &[0, 0, 0]);
    assert_eq!(g.order(), 3);
    assert_eq!(edge_count(&g), 0);
}

#[test]
fn test_configuration_model_n0() {
    let g = configuration_model(42, &[]);
    assert_eq!(g.order(), 0);
}

#[test]
fn test_random_regular_k0() {
    let g = random_regular_graph(42, 5, 0).expect("zero-degree regular graph should exist");
    assert_eq!(g.order(), 5);
    assert_eq!(edge_count(&g), 0);
}

#[test]
fn test_random_regular_n0() {
    let g = random_regular_graph(42, 0, 0).expect("empty regular graph should exist");
    assert_eq!(g.order(), 0);
}

#[test]
fn test_random_regular_3reg() {
    // 3-regular on 8 vertices
    let g = random_regular_graph(42, 8, 3).expect("8-vertex 3-regular graph should exist");
    assert_eq!(g.order(), 8);
    assert_eq!(edge_count(&g), 12); // n*k/2
}

#[test]
fn test_random_regular_k2() {
    // k = 2 → random 2-regular graph (union of cycles)
    let g = random_regular_graph(42, 6, 2).expect("6-vertex 2-regular graph should exist");
    assert_eq!(g.order(), 6);
    assert_eq!(edge_count(&g), 6); // n*k/2
}

#[test]
fn test_random_regular_odd_stub_count_error() {
    let err = random_regular_graph(42, 5, 3).unwrap_err();
    assert_eq!(err, RandomRegularGraphError::OddStubCount { n: 5, k: 3 });
}

#[test]
fn test_random_regular_degree_too_large_error() {
    let err = random_regular_graph(42, 5, 5).unwrap_err();
    assert_eq!(err, RandomRegularGraphError::DegreeTooLarge { n: 5, k: 5 });
}

#[test]
fn test_watts_strogatz_full_rewiring() {
    // beta = 1.0 → all edges rewired, high chance of hitting bail-out
    let g = watts_strogatz(42, 10, 4, 1.0);
    assert_eq!(g.order(), 10);
}

#[test]
fn test_watts_strogatz_nearly_complete_rewiring() {
    // Dense ring + high rewiring to trigger the attempt-limit bail-out.
    // n=6, k=4 → each vertex connected to 4 of 5 others → almost K_6.
    // With beta=1.0, rewiring tries to find a new target but all slots
    // are already taken, triggering the bail-out path.
    for seed in 1..=50 {
        let g = watts_strogatz(seed, 6, 4, 1.0);
        assert_eq!(g.order(), 6);
    }
    // n=4, k=2 with full rewiring on a very small graph.
    for seed in 1..=20 {
        let g = watts_strogatz(seed, 4, 2, 1.0);
        assert_eq!(g.order(), 4);
    }
}

// ============================================================================
// Micali-Vazirani deep coverage: complex blossom structures
// ============================================================================

fn build_graph(n: usize, edges: &[(usize, usize)]) -> UndirectedGraph {
    let mut sorted_edges: Vec<(usize, usize)> = edges.to_vec();
    sorted_edges.sort_unstable();
    UndiEdgesBuilder::default()
        .expected_number_of_edges(sorted_edges.len())
        .expected_shape(n)
        .edges(sorted_edges.into_iter())
        .build()
        .unwrap()
}

#[test]
fn test_mv_deeply_nested_blossoms() {
    // Chain of 4 triangles with tails: forces deep blossom nesting and
    // path compression in DSU.
    // tri0: 0-1-2, tri1: 2-3-4, tri2: 4-5-6, tri3: 6-7-8, tail: 8-9
    let g = build_graph(
        10,
        &[
            (0, 1),
            (0, 2),
            (1, 2),
            (2, 3),
            (2, 4),
            (3, 4),
            (4, 5),
            (4, 6),
            (5, 6),
            (6, 7),
            (6, 8),
            (7, 8),
            (8, 9),
        ],
    );
    let blossom = g.blossom();
    let mv = g.micali_vazirani();
    assert_eq!(blossom.len(), mv.len());
}

#[test]
fn test_mv_interlocking_blossoms() {
    // Two pentagons sharing an edge, with additional paths.
    // Pentagon 1: 0-1-2-3-4-0
    // Pentagon 2: 3-4-5-6-7-3
    // Plus tail: 7-8, 8-9
    let g = build_graph(
        10,
        &[
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (0, 4),
            (3, 5),
            (4, 5),
            (5, 6),
            (6, 7),
            (3, 7),
            (7, 8),
            (8, 9),
        ],
    );
    let blossom = g.blossom();
    let mv = g.micali_vazirani();
    assert_eq!(blossom.len(), mv.len());
}

#[test]
fn test_mv_large_random_graphs() {
    // Larger random graphs exercise more MV internal paths.
    for seed in 10..=25 {
        let g = erdos_renyi_gnm(seed, 50, 80);
        let blossom = g.blossom();
        let mv = g.micali_vazirani();
        assert_eq!(blossom.len(), mv.len(), "seed={seed}");
    }
}

#[test]
fn test_mv_dense_random_graphs() {
    // Dense graphs trigger bridge re-discovery and bud merging.
    for seed in 1..=10 {
        let g = erdos_renyi_gnm(seed, 30, 120);
        let blossom = g.blossom();
        let mv = g.micali_vazirani();
        assert_eq!(blossom.len(), mv.len(), "seed={seed}");
    }
}

#[test]
fn test_mv_on_complete_graphs() {
    // Complete graphs force many blossom contractions.
    for n in 3..=12 {
        let g = complete_graph(n);
        let blossom = g.blossom();
        let mv = g.micali_vazirani();
        assert_eq!(blossom.len(), mv.len(), "K_{n}");
    }
}

#[test]
fn test_mv_on_crown_graphs() {
    // Crown graphs: bipartite (n-1)-regular, forces augmenting paths.
    for n in 2..=8 {
        let g = crown_graph(n);
        let blossom = g.blossom();
        let mv = g.micali_vazirani();
        assert_eq!(blossom.len(), mv.len(), "Cr_{n}");
    }
}

#[test]
fn test_mv_on_friendship_graphs() {
    // Multiple triangles sharing a hub.
    for n in 1..=8 {
        let g = friendship_graph(n);
        let blossom = g.blossom();
        let mv = g.micali_vazirani();
        assert_eq!(blossom.len(), mv.len(), "F_{n}");
    }
}

#[test]
fn test_mv_on_windmill_graphs() {
    for num_cliques in 1..=6 {
        for clique_size in 2..=5 {
            let g = windmill_graph(num_cliques, clique_size);
            let blossom = g.blossom();
            let mv = g.micali_vazirani();
            assert_eq!(blossom.len(), mv.len(), "windmill({num_cliques},{clique_size})");
        }
    }
}

#[test]
fn test_mv_on_wheel_graphs() {
    // Wheels have mixed even/odd cycles.
    for n in 3..=12 {
        let g = wheel_graph(n);
        let blossom = g.blossom();
        let mv = g.micali_vazirani();
        assert_eq!(blossom.len(), mv.len(), "W_{n}");
    }
}

#[test]
fn test_mv_blossom_with_tail_variations() {
    // Triangle with varying tail lengths to exercise augment_even vs
    // augment_through_bridge at different levels.
    for tail_len in 0..=8 {
        let n = 3 + tail_len;
        let mut edges: Vec<(usize, usize)> = vec![(0, 1), (0, 2), (1, 2)];
        for i in 0..tail_len {
            edges.push((2 + i, 3 + i));
        }
        let g = build_graph(n, &edges);
        let blossom = g.blossom();
        let mv = g.micali_vazirani();
        assert_eq!(blossom.len(), mv.len(), "tri+tail({tail_len})");
    }
}

#[test]
fn test_mv_multiple_disjoint_odd_cycles() {
    // Multiple disjoint odd cycles of varying sizes, each needing blossoms.
    // C3 + C5 + C7 with tails
    let g = build_graph(
        19,
        &[
            // C3: 0-1-2
            (0, 1),
            (1, 2),
            (0, 2),
            // tail: 2-3
            (2, 3),
            // C5: 4-5-6-7-8
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 8),
            (4, 8),
            // tail: 8-9
            (8, 9),
            // C7: 10-11-12-13-14-15-16
            (10, 11),
            (11, 12),
            (12, 13),
            (13, 14),
            (14, 15),
            (15, 16),
            (10, 16),
            // tails: 16-17, 17-18
            (16, 17),
            (17, 18),
        ],
    );
    let blossom = g.blossom();
    let mv = g.micali_vazirani();
    assert_eq!(blossom.len(), mv.len());
}

#[test]
fn test_mv_barabasi_albert_various() {
    // Scale-free graphs with hubs trigger varied blossom structures.
    for seed in 1..=15 {
        let g = barabasi_albert(seed, 40, 3);
        let blossom = g.blossom();
        let mv = g.micali_vazirani();
        assert_eq!(blossom.len(), mv.len(), "BA seed={seed}");
    }
}

#[test]
fn test_mv_watts_strogatz_various() {
    // Small-world graphs with clusters.
    for seed in 1..=10 {
        let g = watts_strogatz(seed, 30, 4, 0.3);
        let blossom = g.blossom();
        let mv = g.micali_vazirani();
        assert_eq!(blossom.len(), mv.len(), "WS seed={seed}");
    }
}

#[test]
fn test_mv_random_regular_various() {
    // Regular graphs: uniform degree.
    for seed in 1..=5 {
        let g = random_regular_graph(seed, 20, 4).expect("20-vertex 4-regular graph should exist");
        let blossom = g.blossom();
        let mv = g.micali_vazirani();
        assert_eq!(blossom.len(), mv.len(), "RRG seed={seed}");
    }
}

#[test]
fn test_mv_on_petersen() {
    let g = petersen_graph();
    let blossom = g.blossom();
    let mv = g.micali_vazirani();
    assert_eq!(blossom.len(), mv.len());
    assert_eq!(mv.len(), 5);
}

#[test]
fn test_mv_on_hypercubes() {
    for d in 1..=5 {
        let g = hypercube_graph(d);
        let blossom = g.blossom();
        let mv = g.micali_vazirani();
        assert_eq!(blossom.len(), mv.len(), "Q_{d}");
    }
}

#[test]
fn test_mv_on_barbell() {
    // Barbell with various path lengths.
    for p in 0..=5 {
        let g = barbell_graph(4, p);
        let blossom = g.blossom();
        let mv = g.micali_vazirani();
        assert_eq!(blossom.len(), mv.len(), "barbell(4,{p})");
    }
}

#[test]
fn test_mv_on_grid_graphs() {
    for r in 2..=6 {
        for c in 2..=6 {
            let g = grid_graph(r, c);
            let blossom = g.blossom();
            let mv = g.micali_vazirani();
            assert_eq!(blossom.len(), mv.len(), "grid({r},{c})");
        }
    }
}

#[test]
fn test_mv_on_hexagonal_lattice_graphs() {
    for rows in 1..=4 {
        for cols in 1..=4 {
            let g = hexagonal_lattice_graph(rows, cols);
            let blossom = g.blossom();
            let mv = g.micali_vazirani();
            assert_eq!(blossom.len(), mv.len(), "hexagonal({rows},{cols})");
        }
    }
}

#[test]
fn test_mv_on_torus_graphs() {
    for r in 3..=6 {
        for c in 3..=6 {
            let g = torus_graph(r, c);
            let blossom = g.blossom();
            let mv = g.micali_vazirani();
            assert_eq!(blossom.len(), mv.len(), "torus({r},{c})");
        }
    }
}

#[test]
fn test_mv_on_turan_graphs() {
    // Dense multi-partite graphs with many blossoms.
    for n in 4..=12 {
        for r in 2..=n.min(5) {
            let g = turan_graph(n, r);
            let blossom = g.blossom();
            let mv = g.micali_vazirani();
            assert_eq!(blossom.len(), mv.len(), "turan({n},{r})");
        }
    }
}

#[test]
fn test_mv_many_random_seeds() {
    // Brute-force many random graphs to maximize internal path coverage.
    for seed in 1..=50 {
        let g = erdos_renyi_gnm(seed, 25, 45);
        let blossom = g.blossom();
        let mv = g.micali_vazirani();
        assert_eq!(blossom.len(), mv.len(), "gnm seed={seed}");
    }
}

#[test]
fn test_mv_sparse_random() {
    // Sparse random graphs exercise different augmentation paths.
    for seed in 1..=30 {
        let g = erdos_renyi_gnm(seed, 40, 30);
        let blossom = g.blossom();
        let mv = g.micali_vazirani();
        assert_eq!(blossom.len(), mv.len(), "sparse seed={seed}");
    }
}

#[test]
fn test_mv_gnp_medium_density() {
    // G(n,p) at moderate density for variety.
    for seed in 1..=20 {
        let g = erdos_renyi_gnp(seed, 30, 0.2);
        let blossom = g.blossom();
        let mv = g.micali_vazirani();
        assert_eq!(blossom.len(), mv.len(), "gnp seed={seed}");
    }
}
