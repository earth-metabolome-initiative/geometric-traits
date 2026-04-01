//! Tests for the clique ranking system.
#![cfg(feature = "alloc")]

use core::cmp::Ordering;

use geometric_traits::prelude::*;

// ===========================================================================
// EagerCliqueInfo construction
// ===========================================================================

#[test]
fn test_eager_clique_info_path_match() {
    // Path 0-1-2 matched to 10-11-12.
    // LG edges: 0=(0,1), 1=(1,2) and 0=(10,11), 1=(11,12)
    // Clique: [0, 1] in product pairs [(0,0), (1,1)]
    let clique = vec![0, 1];
    let vertex_pairs = [(0_usize, 0_usize), (1, 1)];
    let edge_map1 = [(0_u32, 1_u32), (1, 2)];
    let edge_map2 = [(10_u32, 11_u32), (11, 12)];

    let info = EagerCliqueInfo::new(
        clique.clone(),
        &vertex_pairs,
        &edge_map1,
        &edge_map2,
        |_, _, _, _| true,
    );

    assert_eq!(info.clique(), &clique[..]);
    assert_eq!(info.matched_edges().len(), 2);
    assert_eq!(info.vertex_matches().len(), 3); // 0↔10, 1↔11, 2↔12
    assert_eq!(info.fragment_count(), 1); // single connected component
    assert_eq!(info.largest_fragment_size(), 2); // 2 edges in the component
    assert_eq!(info.largest_fragment_edge_count(), 2);
    assert_eq!(info.largest_fragment_atom_count(), 3);
}

#[test]
fn test_eager_clique_info_two_fragments() {
    // Two disjoint edges: (0,1)↔(10,11) and (5,6)↔(15,16).
    // No shared endpoints → 2 fragments, each with 1 edge.
    let clique = vec![0, 1];
    let vertex_pairs = [(0_usize, 0_usize), (1, 1)];
    let edge_map1 = [(0_u32, 1_u32), (5, 6)];
    let edge_map2 = [(10_u32, 11_u32), (15, 16)];

    let info =
        EagerCliqueInfo::new(clique, &vertex_pairs, &edge_map1, &edge_map2, |_, _, _, _| true);

    assert_eq!(info.fragment_count(), 2);
    assert_eq!(info.largest_fragment_size(), 1);
    assert_eq!(info.largest_fragment_edge_count(), 1);
    assert_eq!(info.largest_fragment_atom_count(), 2);
}

#[test]
fn test_eager_clique_info_single_edge() {
    let clique = vec![0];
    let vertex_pairs = [(0_usize, 0_usize)];
    let edge_map1 = [(0_u32, 1_u32)];
    let edge_map2 = [(10_u32, 11_u32)];

    let info =
        EagerCliqueInfo::new(clique, &vertex_pairs, &edge_map1, &edge_map2, |_, _, _, _| true);

    assert_eq!(info.fragment_count(), 1);
    assert_eq!(info.largest_fragment_size(), 1);
    assert_eq!(info.largest_fragment_edge_count(), 1);
    assert_eq!(info.largest_fragment_atom_count(), 2);
    assert_eq!(info.vertex_matches().len(), 2);
}

#[test]
fn test_eager_clique_info_empty() {
    let info = EagerCliqueInfo::<u32>::new(vec![], &[], &[], &[], |_, _, _, _| true);

    assert_eq!(info.clique().len(), 0);
    assert_eq!(info.matched_edges().len(), 0);
    assert_eq!(info.vertex_matches().len(), 0);
    assert_eq!(info.fragment_count(), 0);
    assert_eq!(info.largest_fragment_size(), 0);
    assert_eq!(info.largest_fragment_edge_count(), 0);
    assert_eq!(info.largest_fragment_atom_count(), 0);
}

#[test]
fn test_eager_clique_info_distinguishes_edge_and_atom_fragment_sizes() {
    let path = EagerCliqueInfo::new(
        vec![0, 1, 2],
        &[(0_usize, 0_usize), (1, 1), (2, 2)],
        &[(0_u32, 1_u32), (1, 2), (2, 3)],
        &[(10_u32, 11_u32), (11, 12), (12, 13)],
        |_, _, _, _| true,
    );
    let triangle = EagerCliqueInfo::new(
        vec![0, 1, 2],
        &[(0_usize, 0_usize), (1, 1), (2, 2)],
        &[(0_u32, 1_u32), (1, 2), (0, 2)],
        &[(10_u32, 11_u32), (11, 12), (10, 12)],
        |_, _, _, _| true,
    );

    assert_eq!(path.largest_fragment_edge_count(), 3);
    assert_eq!(triangle.largest_fragment_edge_count(), 3);
    assert_eq!(path.largest_fragment_atom_count(), 4);
    assert_eq!(triangle.largest_fragment_atom_count(), 3);
}

// ===========================================================================
// FragmentCountRanker
// ===========================================================================

#[test]
fn test_fragment_count_ranker_fewer_is_better() {
    // Clique A: 1 fragment. Clique B: 2 fragments.
    let a = EagerCliqueInfo::new(
        vec![0, 1],
        &[(0_usize, 0_usize), (1, 1)],
        &[(0_u32, 1_u32), (1, 2)],
        &[(10_u32, 11_u32), (11, 12)],
        |_, _, _, _| true,
    );
    let b = EagerCliqueInfo::new(
        vec![0, 1],
        &[(0_usize, 0_usize), (1, 1)],
        &[(0_u32, 1_u32), (5, 6)],
        &[(10_u32, 11_u32), (15, 16)],
        |_, _, _, _| true,
    );

    let ranker = FragmentCountRanker;
    // a (1 fragment) < b (2 fragments) → a is better → Less
    assert_eq!(ranker.compare(&a, &b), Ordering::Less);
    assert_eq!(ranker.compare(&b, &a), Ordering::Greater);
    assert_eq!(ranker.compare(&a, &a), Ordering::Equal);
}

// ===========================================================================
// ChainedRanker
// ===========================================================================

/// Custom ranker: prefer more vertex matches (more = better → reverse).
struct MoreVerticesRanker;

impl<I: CliqueInfo> CliqueRanker<I> for MoreVerticesRanker {
    fn compare(&self, a: &I, b: &I) -> Ordering {
        b.vertex_matches().len().cmp(&a.vertex_matches().len())
    }
}

#[test]
fn test_chained_ranker_tiebreak() {
    // Both have 1 fragment, but different vertex match counts.
    let a = EagerCliqueInfo::new(
        vec![0, 1],
        &[(0_usize, 0_usize), (1, 1)],
        &[(0_u32, 1_u32), (1, 2)], // path → 3 vertex matches
        &[(10_u32, 11_u32), (11, 12)],
        |_, _, _, _| true,
    );
    let b = EagerCliqueInfo::new(
        vec![0, 1],
        &[(0_usize, 0_usize), (1, 1)],
        &[(0_u32, 1_u32), (1, 2)], // also path → 3 vertex matches
        &[(10_u32, 11_u32), (11, 12)],
        |_, _, _, _| true,
    );

    let ranker = FragmentCountRanker.then(MoreVerticesRanker);
    // Same fragment count, same vertex matches → Equal
    assert_eq!(CliqueRanker::compare(&ranker, &a, &b), Ordering::Equal);
}

#[test]
fn test_chained_ranker_first_decides() {
    // a has 1 fragment, b has 2 → FragmentCountRanker decides, MoreVerticesRanker
    // never consulted.
    let a = EagerCliqueInfo::new(
        vec![0, 1],
        &[(0_usize, 0_usize), (1, 1)],
        &[(0_u32, 1_u32), (1, 2)],
        &[(10_u32, 11_u32), (11, 12)],
        |_, _, _, _| true,
    );
    let b = EagerCliqueInfo::new(
        vec![0, 1],
        &[(0_usize, 0_usize), (1, 1)],
        &[(0_u32, 1_u32), (5, 6)],
        &[(10_u32, 11_u32), (15, 16)],
        |_, _, _, _| true,
    );

    let ranker = FragmentCountRanker.then(MoreVerticesRanker);
    assert_eq!(CliqueRanker::compare(&ranker, &a, &b), Ordering::Less); // a is better (fewer frags)
}

#[test]
fn test_triple_chain() {
    // Verify three-level chaining compiles and works.
    struct AlwaysEqual;
    impl<I: CliqueInfo> CliqueRanker<I> for AlwaysEqual {
        fn compare(&self, _a: &I, _b: &I) -> Ordering {
            Ordering::Equal
        }
    }

    let a = EagerCliqueInfo::new(
        vec![0],
        &[(0_usize, 0_usize)],
        &[(0_u32, 1_u32)],
        &[(10_u32, 11_u32)],
        |_, _, _, _| true,
    );

    let ranker = FragmentCountRanker.then(AlwaysEqual).then(MoreVerticesRanker);
    // Same clique → all rankers return Equal.
    assert_eq!(CliqueRanker::compare(&ranker, &a, &a), Ordering::Equal);
}

// ===========================================================================
// Integration: modular product → clique → EagerCliqueInfo → ranking
// ===========================================================================

#[test]
fn test_integration_modular_product_to_ranking() {
    // Directly construct line-graph-like data and modular product.
    // Simulate: G1 triangle (edges: (0,1),(0,2),(1,2)), G2 path (edges:
    // (0,1),(1,2)). LG1 has 3 vertices (one per G1 edge), LG2 has 2 vertices
    // (one per G2 edge).
    use geometric_traits::impls::BitSquareMatrix;

    // LG1: K3 line graph (all 3 edges share endpoints → complete)
    let lg1_adj = BitSquareMatrix::from_symmetric_edges(3, vec![(0, 1), (0, 2), (1, 2)]);
    // LG2: single edge (both G2 edges share vertex 1)
    let lg2_adj = BitSquareMatrix::from_symmetric_edges(2, vec![(0, 1)]);

    let edge_map1: Vec<(u32, u32)> = vec![(0, 1), (0, 2), (1, 2)];
    let edge_map2: Vec<(u32, u32)> = vec![(0, 1), (1, 2)];

    // Modular product of line graphs.
    let mp = lg1_adj.modular_product_filtered(&lg2_adj, |_, _| true);

    // All maximum cliques.
    let cliques = mp.matrix().all_maximum_cliques();
    assert!(!cliques.is_empty());

    // Build EagerCliqueInfo for each clique.
    let infos: Vec<_> = cliques
        .iter()
        .map(|c| {
            EagerCliqueInfo::new(
                c.clone(),
                mp.vertex_pairs(),
                &edge_map1,
                &edge_map2,
                |_, _, _, _| true,
            )
        })
        .collect();

    // Verify all infos have valid data.
    for info in &infos {
        assert!(!info.clique().is_empty());
        assert!(!info.matched_edges().is_empty());
        assert!(info.fragment_count() >= 1);
        assert!(info.largest_fragment_size() >= 1);
    }

    // Rank by fragment count.
    let ranker = FragmentCountRanker;
    let mut sorted_infos = infos;
    sorted_infos.sort_by(|a, b| ranker.compare(a, b));

    // Best (first) should have fewest fragments.
    let best_frags = sorted_infos[0].fragment_count();
    for info in &sorted_infos {
        assert!(info.fragment_count() >= best_frags);
    }
}

// ===========================================================================
// LargestFragmentRanker
// ===========================================================================

#[test]
fn test_largest_fragment_ranker_bigger_wins() {
    // a: path of 3 edges (largest fragment = 3)
    let a = EagerCliqueInfo::new(
        vec![0, 1, 2],
        &[(0_usize, 0_usize), (1, 1), (2, 2)],
        &[(0_u32, 1_u32), (1, 2), (2, 3)],
        &[(10_u32, 11_u32), (11, 12), (12, 13)],
        |_, _, _, _| true,
    );
    // b: single edge (largest fragment = 1)
    let b = EagerCliqueInfo::new(
        vec![0],
        &[(0_usize, 0_usize)],
        &[(0_u32, 1_u32)],
        &[(10_u32, 11_u32)],
        |_, _, _, _| true,
    );

    let ranker = LargestFragmentRanker;
    // a (3 edges) is better than b (1 edge) → Less
    assert_eq!(ranker.compare(&a, &b), Ordering::Less);
    assert_eq!(ranker.compare(&b, &a), Ordering::Greater);
}

#[test]
fn test_largest_fragment_ranker_equal() {
    let a = EagerCliqueInfo::new(
        vec![0],
        &[(0_usize, 0_usize)],
        &[(0_u32, 1_u32)],
        &[(10_u32, 11_u32)],
        |_, _, _, _| true,
    );
    let b = EagerCliqueInfo::new(
        vec![0],
        &[(0_usize, 0_usize)],
        &[(5_u32, 6_u32)],
        &[(15_u32, 16_u32)],
        |_, _, _, _| true,
    );

    assert_eq!(LargestFragmentRanker.compare(&a, &b), Ordering::Equal);
}

#[test]
fn test_largest_fragment_metric_ranker_atoms_prefers_more_vertices() {
    let path = EagerCliqueInfo::new(
        vec![0, 1, 2],
        &[(0_usize, 0_usize), (1, 1), (2, 2)],
        &[(0_u32, 1_u32), (1, 2), (2, 3)],
        &[(10_u32, 11_u32), (11, 12), (12, 13)],
        |_, _, _, _| true,
    );
    let triangle = EagerCliqueInfo::new(
        vec![0, 1, 2],
        &[(0_usize, 0_usize), (1, 1), (2, 2)],
        &[(0_u32, 1_u32), (1, 2), (0, 2)],
        &[(10_u32, 11_u32), (11, 12), (10, 12)],
        |_, _, _, _| true,
    );

    let edge_ranker = LargestFragmentMetricRanker::new(LargestFragmentMetric::Edges);
    let atom_ranker = LargestFragmentMetricRanker::new(LargestFragmentMetric::Atoms);

    assert_eq!(edge_ranker.compare(&path, &triangle), Ordering::Equal);
    assert_eq!(atom_ranker.compare(&path, &triangle), Ordering::Less);
    assert_eq!(atom_ranker.compare(&triangle, &path), Ordering::Greater);
}

// ===========================================================================
// FnRanker
// ===========================================================================

#[test]
fn test_fn_ranker_custom_logic() {
    // Custom: prefer more vertex matches.
    let ranker = FnRanker::new(|a: &EagerCliqueInfo<u32>, b: &EagerCliqueInfo<u32>| {
        b.vertex_matches().len().cmp(&a.vertex_matches().len())
    });

    // a: path → 3 vertex matches
    let a = EagerCliqueInfo::new(
        vec![0, 1],
        &[(0_usize, 0_usize), (1, 1)],
        &[(0_u32, 1_u32), (1, 2)],
        &[(10_u32, 11_u32), (11, 12)],
        |_, _, _, _| true,
    );
    // b: single edge → 2 vertex matches
    let b = EagerCliqueInfo::new(
        vec![0],
        &[(0_usize, 0_usize)],
        &[(0_u32, 1_u32)],
        &[(10_u32, 11_u32)],
        |_, _, _, _| true,
    );

    // a has more vertex matches → Less (a is better)
    assert_eq!(ranker.compare(&a, &b), Ordering::Less);
}

#[test]
fn test_fn_ranker_in_chain() {
    // Chain: fragment count → largest fragment → custom (vertex count)
    let a = EagerCliqueInfo::new(
        vec![0, 1],
        &[(0_usize, 0_usize), (1, 1)],
        &[(0_u32, 1_u32), (1, 2)],
        &[(10_u32, 11_u32), (11, 12)],
        |_, _, _, _| true,
    );
    let b = EagerCliqueInfo::new(
        vec![0, 1],
        &[(0_usize, 0_usize), (1, 1)],
        &[(0_u32, 1_u32), (1, 2)],
        &[(10_u32, 11_u32), (11, 12)],
        |_, _, _, _| true,
    );

    let ranker = FragmentCountRanker.then(LargestFragmentRanker).then(FnRanker::new(
        |a: &EagerCliqueInfo<u32>, b: &EagerCliqueInfo<u32>| {
            b.vertex_matches().len().cmp(&a.vertex_matches().len())
        },
    ));

    // Identical cliques → all Equal
    assert_eq!(CliqueRanker::compare(&ranker, &a, &b), Ordering::Equal);
}

// ===========================================================================
// Default MCES ranking chain
// ===========================================================================

#[test]
fn test_default_mces_ranking_chain() {
    // Three cliques:
    // c1: 1 fragment, 3 edges (path)
    // c2: 2 fragments, largest=1 (two disjoint edges)
    // c3: 1 fragment, 1 edge
    let c1 = EagerCliqueInfo::new(
        vec![0, 1, 2],
        &[(0_usize, 0_usize), (1, 1), (2, 2)],
        &[(0_u32, 1_u32), (1, 2), (2, 3)],
        &[(10_u32, 11_u32), (11, 12), (12, 13)],
        |_, _, _, _| true,
    );
    let c2 = EagerCliqueInfo::new(
        vec![0, 1],
        &[(0_usize, 0_usize), (1, 1)],
        &[(0_u32, 1_u32), (5, 6)],
        &[(10_u32, 11_u32), (15, 16)],
        |_, _, _, _| true,
    );
    let c3 = EagerCliqueInfo::new(
        vec![0],
        &[(0_usize, 0_usize)],
        &[(0_u32, 1_u32)],
        &[(10_u32, 11_u32)],
        |_, _, _, _| true,
    );

    assert_eq!(c1.fragment_count(), 1);
    assert_eq!(c2.fragment_count(), 2);
    assert_eq!(c3.fragment_count(), 1);
    assert_eq!(c1.largest_fragment_size(), 3);
    assert_eq!(c3.largest_fragment_size(), 1);

    let ranker = FragmentCountRanker.then(LargestFragmentRanker);

    // c1 vs c2: c1 has fewer fragments → c1 wins
    assert_eq!(CliqueRanker::compare(&ranker, &c1, &c2), Ordering::Less);

    // c1 vs c3: same fragment count (1), c1 has larger fragment → c1 wins
    assert_eq!(CliqueRanker::compare(&ranker, &c1, &c3), Ordering::Less);

    // c3 vs c2: c3 (1 frag) wins over c2 (2 frags)
    assert_eq!(CliqueRanker::compare(&ranker, &c3, &c2), Ordering::Less);

    // Sort: c1, c3, c2 (best to worst)
    let mut cliques = [&c2, &c3, &c1];
    cliques.sort_by(|a, b| CliqueRanker::compare(&ranker, *a, *b));
    assert_eq!(cliques[0].largest_fragment_size(), 3); // c1
    assert_eq!(cliques[1].largest_fragment_size(), 1); // c3
    assert_eq!(cliques[2].fragment_count(), 2); // c2
}
