//! TDD scaffolding for the forthcoming VF2 implementation.
#![cfg(feature = "std")]

use std::cell::Cell;

use geometric_traits::{
    impls::{CSR2D, SortedVec, SquareCSR2D, SymmetricCSR2D},
    prelude::*,
    traits::{EdgesBuilder, VocabularyBuilder},
};

fn build_undigraph(node_count: usize, mut edges: Vec<(usize, usize)>) -> UndiGraph<usize> {
    let nodes: SortedVec<usize> = GenericVocabularyBuilder::default()
        .expected_number_of_symbols(node_count)
        .symbols((0..node_count).enumerate())
        .build()
        .unwrap();
    edges.sort_unstable();
    let edges: SymmetricCSR2D<CSR2D<usize, usize, usize>> = UndiEdgesBuilder::default()
        .expected_number_of_edges(edges.len())
        .expected_shape(node_count)
        .edges(edges.into_iter())
        .build()
        .unwrap();
    UndiGraph::from((nodes, edges))
}

fn build_digraph(node_count: usize, mut edges: Vec<(usize, usize)>) -> DiGraph<usize> {
    let nodes: SortedVec<usize> = GenericVocabularyBuilder::default()
        .expected_number_of_symbols(node_count)
        .symbols((0..node_count).enumerate())
        .build()
        .unwrap();
    edges.sort_unstable();
    let edges: SquareCSR2D<_> = DiEdgesBuilder::default()
        .expected_number_of_edges(edges.len())
        .expected_shape(node_count)
        .edges(edges.into_iter())
        .build()
        .unwrap();
    DiGraph::from((nodes, edges))
}

fn mapping_matches(pairs: &[(usize, usize)], expected: &[(usize, usize)]) -> bool {
    pairs.len() == expected.len()
        && expected
            .iter()
            .all(|expected_pair| pairs.iter().any(|actual_pair| actual_pair == expected_pair))
}

#[test]
fn test_vf2_empty_graphs_are_isomorphic() {
    let query = build_undigraph(0, Vec::new());
    let target = build_undigraph(0, Vec::new());

    assert!(query.vf2(&target).with_mode(Vf2Mode::Isomorphism).has_match());
}

#[test]
fn test_vf2_isomorphism_accepts_triangle_vs_triangle() {
    let query = build_undigraph(3, vec![(0, 1), (0, 2), (1, 2)]);
    let target = build_undigraph(3, vec![(0, 1), (0, 2), (1, 2)]);

    assert!(query.vf2(&target).with_mode(Vf2Mode::Isomorphism).has_match());
}

#[test]
fn test_vf2_isomorphism_rejects_path_vs_triangle() {
    let query = build_undigraph(3, vec![(0, 1), (1, 2)]);
    let target = build_undigraph(3, vec![(0, 1), (0, 2), (1, 2)]);

    assert!(!query.vf2(&target).with_mode(Vf2Mode::Isomorphism).has_match());
}

#[test]
fn test_vf2_induced_subgraph_rejects_path_inside_triangle() {
    let query = build_undigraph(3, vec![(0, 1), (1, 2)]);
    let target = build_undigraph(3, vec![(0, 1), (0, 2), (1, 2)]);

    assert!(!query.vf2(&target).with_mode(Vf2Mode::InducedSubgraphIsomorphism).has_match());
}

#[test]
fn test_vf2_subgraph_accepts_path_inside_triangle() {
    let query = build_undigraph(3, vec![(0, 1), (1, 2)]);
    let target = build_undigraph(3, vec![(0, 1), (0, 2), (1, 2)]);

    assert!(query.vf2(&target).with_mode(Vf2Mode::SubgraphIsomorphism).has_match());
}

#[test]
fn test_vf2_monomorphism_accepts_path_inside_triangle() {
    let query = build_undigraph(3, vec![(0, 1), (1, 2)]);
    let target = build_undigraph(3, vec![(0, 1), (0, 2), (1, 2)]);

    assert!(query.vf2(&target).with_mode(Vf2Mode::Monomorphism).has_match());
}

#[test]
fn test_vf2_subgraph_accepts_sparse_graph_inside_clique() {
    let query = build_undigraph(4, vec![(0, 1), (1, 3), (2, 3)]);
    let target = build_undigraph(4, vec![(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]);

    assert!(query.vf2(&target).with_mode(Vf2Mode::SubgraphIsomorphism).has_match());
}

#[test]
fn test_vf2_induced_subgraph_allows_extra_edges_to_unmatched_target_nodes() {
    let query = build_undigraph(2, vec![(0, 1)]);
    let target = build_undigraph(3, vec![(0, 1), (1, 2)]);

    assert!(query.vf2(&target).with_mode(Vf2Mode::InducedSubgraphIsomorphism).has_match());
}

#[test]
fn test_vf2_induced_subgraph_accepts_single_isolated_node_inside_nonisolated_target() {
    let query = build_undigraph(1, Vec::new());
    let target = build_undigraph(3, vec![(0, 1), (1, 2)]);
    let mut visited = 0usize;

    let exhausted = query
        .vf2(&target)
        .with_mode(Vf2Mode::InducedSubgraphIsomorphism)
        .for_each_match(|mapping| {
            visited += 1;
            assert_eq!(mapping.len(), 1);
            true
        });

    assert!(exhausted);
    assert_eq!(visited, 3);
}

#[test]
fn test_vf2_induced_subgraph_accepts_edge_plus_isolated_after_target_component_switch() {
    let query = build_undigraph(3, vec![(0, 1)]);
    let target = build_undigraph(5, vec![(0, 1), (0, 4), (1, 2), (1, 4), (2, 3)]);

    assert!(query.vf2(&target).with_mode(Vf2Mode::InducedSubgraphIsomorphism).has_match());
}

#[test]
fn test_vf2_subgraph_counts_two_isolated_nodes_inside_triangle() {
    let query = build_undigraph(2, Vec::new());
    let target = build_undigraph(3, vec![(0, 1), (0, 2), (1, 2)]);
    let mut visited = 0usize;

    assert!(!query.vf2(&target).with_mode(Vf2Mode::InducedSubgraphIsomorphism).has_match());
    assert!(query.vf2(&target).with_mode(Vf2Mode::SubgraphIsomorphism).has_match());

    let exhausted =
        query.vf2(&target).with_mode(Vf2Mode::SubgraphIsomorphism).for_each_match(|_| {
            visited += 1;
            true
        });

    assert!(exhausted);
    assert_eq!(visited, 6);
}

#[test]
fn test_vf2_monomorphism_counts_two_isolated_nodes_inside_triangle() {
    let query = build_undigraph(2, Vec::new());
    let target = build_undigraph(3, vec![(0, 1), (0, 2), (1, 2)]);
    let mut visited = 0usize;

    let exhausted = query.vf2(&target).with_mode(Vf2Mode::Monomorphism).for_each_match(|_| {
        visited += 1;
        true
    });

    assert!(exhausted);
    assert_eq!(visited, 6);
}

#[test]
fn test_vf2_node_match_can_filter_candidates() {
    let query = build_undigraph(2, vec![(0, 1)]);
    let target = build_undigraph(2, vec![(0, 1)]);

    assert!(
        !query
            .vf2(&target)
            .with_mode(Vf2Mode::Isomorphism)
            .with_node_match(|query_node, target_node| query_node == target_node && query_node == 0)
            .has_match()
    );
}

#[test]
fn test_vf2_node_match_is_not_reevaluated_during_structural_feasibility() {
    let query = build_undigraph(1, Vec::new());
    let target = build_undigraph(3, Vec::new());
    let calls = Cell::new(0usize);

    assert!(
        query
            .vf2(&target)
            .with_mode(Vf2Mode::SubgraphIsomorphism)
            .with_node_match(|_, _| {
                calls.set(calls.get() + 1);
                true
            })
            .has_match()
    );

    assert_eq!(calls.get(), 3);
}

#[test]
fn test_vf2_edge_match_can_filter_candidates() {
    let query = build_undigraph(2, vec![(0, 1)]);
    let target = build_undigraph(2, vec![(0, 1)]);

    assert!(
        !query
            .vf2(&target)
            .with_mode(Vf2Mode::Isomorphism)
            .with_edge_match(|_, _, _, _| false)
            .has_match()
    );
}

#[test]
fn test_vf2_final_match_can_reject_complete_mappings() {
    let query = build_undigraph(2, vec![(0, 1)]);
    let target = build_undigraph(2, vec![(0, 1)]);

    assert!(
        !query.vf2(&target).with_mode(Vf2Mode::Isomorphism).with_final_match(|_| false).has_match()
    );
}

#[test]
fn test_vf2_final_match_can_skip_earlier_mappings_and_find_a_later_one() {
    let query = build_undigraph(1, Vec::new());
    let target = build_undigraph(3, Vec::new());

    let mapping = query
        .vf2(&target)
        .with_mode(Vf2Mode::SubgraphIsomorphism)
        .with_final_match(|pairs| matches!(pairs, [(0, 2)]))
        .first_match()
        .expect("the final matcher should allow the third embedding");

    assert_eq!(mapping.pairs(), &[(0, 2)]);
}

#[test]
fn test_prepared_vf2_can_reuse_the_same_preprocessed_pair() {
    let query = build_undigraph(2, vec![(0, 1)]);
    let target = build_undigraph(3, vec![(0, 1), (1, 2)]);
    let prepared_query = query.prepare_vf2();
    let prepared_target = target.prepare_vf2();
    let matcher =
        prepared_query.vf2(&prepared_target).with_mode(Vf2Mode::InducedSubgraphIsomorphism);
    let mut visited = 0usize;

    assert!(matcher.has_match());
    assert!(matcher.first_match().is_some());
    assert!(matcher.for_each_match(|_| {
        visited += 1;
        true
    }));
    assert_eq!(visited, 4);
}

#[test]
fn test_prepared_vf2_supports_semantic_hooks() {
    let query = build_undigraph(1, Vec::new());
    let target = build_undigraph(3, Vec::new());
    let prepared_query = query.prepare_vf2();
    let prepared_target = target.prepare_vf2();

    let mapping = prepared_query
        .vf2(&prepared_target)
        .with_mode(Vf2Mode::SubgraphIsomorphism)
        .with_node_match(|query_node, target_node| query_node == 0 && target_node == 2)
        .first_match()
        .expect("prepared VF2 should still honor semantic hooks");

    assert_eq!(mapping.pairs(), &[(0, 2)]);
}

#[test]
fn test_vf2_for_each_mapping_streams_borrowed_pairs() {
    let query = build_undigraph(1, Vec::new());
    let target = build_undigraph(3, Vec::new());
    let mut visited = Vec::new();

    let exhausted =
        query.vf2(&target).with_mode(Vf2Mode::SubgraphIsomorphism).for_each_mapping(|mapping| {
            visited.push(mapping[0]);
            true
        });

    assert!(exhausted);
    assert_eq!(visited, vec![(0, 0), (0, 1), (0, 2)]);
}

#[test]
fn test_prepared_vf2_for_each_mapping_can_stop_early() {
    let query = build_undigraph(1, Vec::new());
    let target = build_undigraph(3, Vec::new());
    let prepared_query = query.prepare_vf2();
    let prepared_target = target.prepare_vf2();
    let mut visited = 0usize;

    let exhausted = prepared_query
        .vf2(&prepared_target)
        .with_mode(Vf2Mode::SubgraphIsomorphism)
        .for_each_mapping(|mapping| {
            visited += 1;
            assert_eq!(mapping.len(), 1);
            false
        });

    assert!(!exhausted);
    assert_eq!(visited, 1);
}

#[test]
fn test_vf2_for_each_match_only_visits_mappings_accepted_by_final_match() {
    let query = build_undigraph(1, Vec::new());
    let target = build_undigraph(3, Vec::new());
    let mut visited = 0usize;

    let exhausted = query
        .vf2(&target)
        .with_mode(Vf2Mode::SubgraphIsomorphism)
        .with_final_match(|pairs| matches!(pairs, [(0, 1 | 2)]))
        .for_each_match(|mapping| {
            visited += 1;
            assert!(matches!(mapping.pairs(), [(0, 1 | 2)]));
            true
        });

    assert!(exhausted);
    assert_eq!(visited, 2);
}

#[test]
fn test_vf2_first_match_returns_a_full_mapping() {
    let query = build_undigraph(2, vec![(0, 1)]);
    let target = build_undigraph(2, vec![(0, 1)]);

    let mapping = query
        .vf2(&target)
        .with_mode(Vf2Mode::Isomorphism)
        .first_match()
        .expect("single edge vs single edge should match");

    assert_eq!(mapping.len(), 2);
}

#[test]
fn test_vf2_for_each_match_can_visit_multiple_embeddings() {
    let query = build_undigraph(1, Vec::new());
    let target = build_undigraph(2, Vec::new());
    let mut visited = 0usize;

    let exhausted =
        query.vf2(&target).with_mode(Vf2Mode::SubgraphIsomorphism).for_each_match(|mapping| {
            visited += 1;
            !mapping.is_empty()
        });

    assert!(exhausted);
    assert_eq!(visited, 2);
}

#[test]
fn test_vf2_empty_query_has_one_empty_embedding_in_subgraph_modes() {
    let query = build_undigraph(0, Vec::new());
    let target = build_undigraph(3, vec![(0, 1), (1, 2)]);

    for mode in [Vf2Mode::InducedSubgraphIsomorphism, Vf2Mode::SubgraphIsomorphism] {
        let mut visited = 0usize;
        let exhausted = query.vf2(&target).with_mode(mode).for_each_match(|mapping| {
            visited += 1;
            assert!(mapping.is_empty());
            true
        });
        assert!(exhausted);
        assert_eq!(visited, 1);
    }
}

#[test]
fn test_vf2_triangle_isomorphism_has_six_embeddings() {
    let query = build_undigraph(3, vec![(0, 1), (0, 2), (1, 2)]);
    let target = build_undigraph(3, vec![(0, 1), (0, 2), (1, 2)]);
    let mut visited = 0usize;

    let exhausted = query.vf2(&target).with_mode(Vf2Mode::Isomorphism).for_each_match(|_| {
        visited += 1;
        true
    });

    assert!(exhausted);
    assert_eq!(visited, 6);
}

#[test]
fn test_vf2_for_each_match_stops_early_without_corrupting_state() {
    let query = build_undigraph(2, vec![(0, 1)]);
    let target = build_undigraph(3, vec![(0, 1), (0, 2), (1, 2)]);
    let mut visited = 0usize;

    let exhausted =
        query.vf2(&target).with_mode(Vf2Mode::SubgraphIsomorphism).for_each_match(|_| {
            visited += 1;
            visited < 3
        });

    assert!(!exhausted);
    assert_eq!(visited, 3);
}

#[test]
fn test_vf2_isomorphism_respects_self_loops() {
    let query = build_digraph(2, vec![(0, 0), (0, 1)]);
    let target = build_digraph(2, vec![(0, 0), (0, 1)]);

    assert!(query.vf2(&target).with_mode(Vf2Mode::Isomorphism).has_match());
}

#[test]
fn test_vf2_undirected_isomorphism_respects_self_loops() {
    let query = build_undigraph(2, vec![(0, 0), (0, 1)]);
    let target = build_undigraph(2, vec![(0, 0), (0, 1)]);

    assert!(query.vf2(&target).with_mode(Vf2Mode::Isomorphism).has_match());
}

#[test]
fn test_vf2_subgraph_allows_target_self_loop_when_query_has_none_but_induced_rejects() {
    let query = build_digraph(1, Vec::new());
    let target = build_digraph(1, vec![(0, 0)]);

    assert!(query.vf2(&target).with_mode(Vf2Mode::SubgraphIsomorphism).has_match());
    assert!(query.vf2(&target).with_mode(Vf2Mode::Monomorphism).has_match());
    assert!(!query.vf2(&target).with_mode(Vf2Mode::InducedSubgraphIsomorphism).has_match());
}

#[test]
fn test_vf2_directed_induced_subgraph_allows_extra_edges_to_unmatched_target_nodes() {
    let query = build_digraph(2, vec![(0, 1)]);
    let target = build_digraph(3, vec![(0, 1), (1, 2), (2, 1)]);

    assert!(query.vf2(&target).with_mode(Vf2Mode::InducedSubgraphIsomorphism).has_match());
}

#[test]
fn test_vf2_directed_subgraph_respects_edge_direction() {
    let query = build_digraph(3, vec![(0, 1), (0, 2)]);
    let target = build_digraph(3, vec![(1, 0), (2, 0)]);

    assert!(!query.vf2(&target).with_mode(Vf2Mode::SubgraphIsomorphism).has_match());
}

#[test]
fn test_vf2_directed_induced_rejects_bidirected_pair_but_subgraph_accepts() {
    let query = build_digraph(2, vec![(0, 1)]);
    let target = build_digraph(2, vec![(0, 1), (1, 0)]);
    let mut visited = 0usize;

    assert!(!query.vf2(&target).with_mode(Vf2Mode::InducedSubgraphIsomorphism).has_match());

    let exhausted =
        query.vf2(&target).with_mode(Vf2Mode::SubgraphIsomorphism).for_each_match(|_| {
            visited += 1;
            true
        });

    assert!(exhausted);
    assert_eq!(visited, 2);
}

#[test]
fn test_vf2_directed_induced_rejects_extra_edge_between_frontiers_but_subgraph_accepts() {
    let query = build_digraph(3, vec![(2, 0), (0, 1)]);
    let target = build_digraph(3, vec![(2, 0), (0, 1), (2, 1)]);

    assert!(!query.vf2(&target).with_mode(Vf2Mode::Isomorphism).has_match());
    assert!(!query.vf2(&target).with_mode(Vf2Mode::InducedSubgraphIsomorphism).has_match());
    assert!(query.vf2(&target).with_mode(Vf2Mode::SubgraphIsomorphism).has_match());
}

#[test]
fn test_vf2_directed_isomorphism_rejects_in_star_vs_out_star() {
    let query = build_digraph(3, vec![(0, 2), (1, 2)]);
    let target = build_digraph(3, vec![(0, 1), (0, 2)]);

    assert!(!query.vf2(&target).with_mode(Vf2Mode::Isomorphism).has_match());
}

#[test]
fn test_vf2_directed_cycle_isomorphism_has_three_embeddings() {
    let query = build_digraph(3, vec![(0, 1), (1, 2), (2, 0)]);
    let target = build_digraph(3, vec![(0, 1), (1, 2), (2, 0)]);
    let mut visited = 0usize;

    let exhausted = query.vf2(&target).with_mode(Vf2Mode::Isomorphism).for_each_match(|_| {
        visited += 1;
        true
    });

    assert!(exhausted);
    assert_eq!(visited, 3);
}

#[test]
fn test_vf2_directed_diamond_isomorphism_has_two_embeddings() {
    let query = build_digraph(4, vec![(0, 2), (0, 3), (2, 1), (3, 1)]);
    let target = build_digraph(4, vec![(0, 2), (0, 3), (2, 1), (3, 1)]);
    let mut visited = 0usize;

    let exhausted = query.vf2(&target).with_mode(Vf2Mode::Isomorphism).for_each_match(|_| {
        visited += 1;
        true
    });

    assert!(exhausted);
    assert_eq!(visited, 2);
}

#[test]
fn test_vf2_directed_induced_edge_plus_isolated_uses_unmatched_fallback_correctly() {
    let query = build_digraph(3, vec![(0, 1)]);
    let target = build_digraph(4, vec![(0, 1), (1, 3)]);

    assert!(
        query
            .vf2(&target)
            .with_mode(Vf2Mode::InducedSubgraphIsomorphism)
            .with_final_match(|pairs| mapping_matches(pairs, &[(0, 0), (1, 1), (2, 2)]))
            .has_match()
    );
    assert!(
        !query
            .vf2(&target)
            .with_mode(Vf2Mode::InducedSubgraphIsomorphism)
            .with_final_match(|pairs| mapping_matches(pairs, &[(0, 0), (1, 1), (2, 3)]))
            .has_match()
    );
}

#[test]
fn test_vf2_directed_induced_edge_plus_isolated_rejects_extra_incoming_edge_to_isolated_target() {
    let query = build_digraph(3, vec![(0, 1)]);
    let target = build_digraph(4, vec![(0, 1), (3, 1)]);

    assert!(
        query
            .vf2(&target)
            .with_mode(Vf2Mode::InducedSubgraphIsomorphism)
            .with_final_match(|pairs| mapping_matches(pairs, &[(0, 0), (1, 1), (2, 2)]))
            .has_match()
    );
    assert!(
        !query
            .vf2(&target)
            .with_mode(Vf2Mode::InducedSubgraphIsomorphism)
            .with_final_match(|pairs| mapping_matches(pairs, &[(0, 0), (1, 1), (2, 3)]))
            .has_match()
    );
}

#[test]
fn test_vf2_directed_isolated_pair_counts_differ_between_induced_and_subgraph() {
    let query = build_digraph(2, Vec::new());
    let target = build_digraph(3, vec![(0, 1)]);
    let mut induced = 0usize;
    let mut subgraph = 0usize;

    let induced_exhausted =
        query.vf2(&target).with_mode(Vf2Mode::InducedSubgraphIsomorphism).for_each_match(|_| {
            induced += 1;
            true
        });
    let subgraph_exhausted =
        query.vf2(&target).with_mode(Vf2Mode::SubgraphIsomorphism).for_each_match(|_| {
            subgraph += 1;
            true
        });

    assert!(induced_exhausted);
    assert!(subgraph_exhausted);
    assert_eq!(induced, 4);
    assert_eq!(subgraph, 6);
}
