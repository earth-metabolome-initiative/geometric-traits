//! Semantic fixture contract tests for future planarity and outerplanarity
//! detection traits.
#![cfg(feature = "std")]

#[path = "support/planarity_fixture.rs"]
mod planarity_fixture;

use std::collections::BTreeSet;

use geometric_traits::traits::TreeDetection;
use planarity_fixture::{build_undigraph, normalize_edge, semantic_cases};

#[test]
fn test_planarity_semantic_cases_are_simple_and_canonical() {
    let cases = semantic_cases();
    let mut seen_names = BTreeSet::new();

    for case in &cases {
        assert!(
            seen_names.insert(case.name.as_str()),
            "duplicate semantic case name {}",
            case.name
        );
        assert!(!case.notes.is_empty(), "fixture case {} must explain its purpose", case.name);

        let normalized_edges: Vec<[usize; 2]> =
            case.edges.iter().copied().map(normalize_edge).collect();
        assert_eq!(
            normalized_edges, case.edges,
            "fixture case {} must store canonical graph edges",
            case.name
        );

        let unique_edges: BTreeSet<[usize; 2]> = case.edges.iter().copied().collect();
        assert_eq!(
            unique_edges.len(),
            case.edges.len(),
            "fixture case {} must not duplicate edges",
            case.name
        );

        for edge in &case.edges {
            assert_ne!(
                edge[0], edge[1],
                "fixture case {} should stay in the simple-graph regime",
                case.name
            );
            assert!(
                edge[0] < case.node_count && edge[1] < case.node_count,
                "fixture case {} references out-of-range edge {:?}",
                case.name,
                edge
            );
        }
    }
}

#[test]
fn test_planarity_semantic_cases_respect_basic_graph_theorems() {
    for case in semantic_cases() {
        assert!(
            !case.is_outerplanar || case.is_planar,
            "outerplanar fixture case {} must also be planar",
            case.name
        );

        let edge_count = case.edges.len();
        if case.is_planar && case.node_count >= 3 {
            assert!(
                edge_count <= 3 * case.node_count - 6,
                "planar fixture case {} violates the simple planar edge bound",
                case.name
            );
        }
        if case.is_outerplanar && case.node_count >= 2 {
            assert!(
                edge_count <= 2 * case.node_count - 3,
                "outerplanar fixture case {} violates the simple outerplanar edge bound",
                case.name
            );
        }
    }
}

#[test]
fn test_planarity_semantic_tree_cases_match_existing_tree_detection() {
    for case in semantic_cases()
        .into_iter()
        .filter(|case| matches!(case.family.as_str(), "edge_case" | "tree"))
    {
        let graph = build_undigraph(&case);
        assert!(graph.is_forest(), "fixture case {} should remain acyclic", case.name);
        assert!(case.is_planar, "forest case {} must be planar", case.name);
        assert!(case.is_outerplanar, "forest case {} must be outerplanar", case.name);
    }
}

#[test]
fn test_planarity_semantic_obstruction_metadata_is_present() {
    for case in semantic_cases() {
        if !case.is_planar {
            assert!(
                case.planarity_obstruction_family.is_some(),
                "nonplanar case {} must identify a planarity obstruction family",
                case.name
            );
        }
        if case.is_planar && !case.is_outerplanar {
            assert!(
                case.outerplanarity_obstruction_family.is_some(),
                "planar-but-not-outerplanar case {} must identify an outerplanarity obstruction family",
                case.name
            );
        }
    }
}
