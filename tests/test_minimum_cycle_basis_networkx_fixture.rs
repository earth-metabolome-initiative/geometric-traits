//! NetworkX-backed reference checks for the graph-level minimum-cycle-basis
//! trait.
#![cfg(feature = "std")]

#[path = "support/minimum_cycle_basis_fixture.rs"]
mod minimum_cycle_basis_fixture;

use std::collections::BTreeSet;

use geometric_traits::prelude::*;
use minimum_cycle_basis_fixture::{build_undigraph, load_fixture_suite};

const FIXTURE_NAME: &str = "minimum_cycle_basis_networkx_1000.json.gz";

#[test]
fn test_minimum_cycle_basis_networkx_fixture_header() {
    let suite = load_fixture_suite(FIXTURE_NAME);
    assert_eq!(suite.schema_version, 1);
    assert_eq!(suite.algorithm, "minimum_cycle_basis_networkx_reference");
    assert_eq!(suite.graph_kind, "simple_undirected");
}

#[test]
fn test_minimum_cycle_basis_matches_networkx_fixture() {
    let suite = load_fixture_suite(FIXTURE_NAME);

    for case in suite.cases {
        let graph = build_undigraph(&case);
        let result = graph.minimum_cycle_basis().unwrap();
        let actual_cycles =
            normalize_cycles(result.minimum_cycle_basis().cloned().collect::<Vec<_>>());
        let expected_cycles = normalize_cycles(case.minimum_cycle_basis.clone());
        let actual_weight = actual_cycles.iter().map(Vec::len).sum::<usize>();

        assert_eq!(result.cycle_rank(), case.cycle_rank, "{}: cycle-rank mismatch", case.name);
        assert_eq!(result.len(), case.basis_size, "{}: basis-size mismatch", case.name);
        assert_eq!(actual_weight, case.total_weight, "{}: total-weight mismatch", case.name);
        assert_eq!(actual_cycles, expected_cycles, "{}: exact basis mismatch", case.name);
        assert_eq!(
            case.minimum_cycle_basis.iter().map(Vec::len).sum::<usize>(),
            case.total_weight,
            "{}: invalid fixture weight",
            case.name
        );
        assert!(
            actual_cycles.iter().all(|cycle| is_simple_cycle_in_graph(cycle, &case.edges)),
            "{}: implementation returned a non-cycle",
            case.name
        );
    }
}

fn normalize_cycles(mut cycles: Vec<Vec<usize>>) -> Vec<Vec<usize>> {
    for cycle in &mut cycles {
        *cycle = normalize_cycle(cycle.clone());
    }
    cycles
        .sort_unstable_by(|left, right| left.len().cmp(&right.len()).then_with(|| left.cmp(right)));
    cycles
}

fn normalize_cycle(mut cycle: Vec<usize>) -> Vec<usize> {
    if cycle.is_empty() {
        return cycle;
    }
    let start =
        cycle.iter().enumerate().min_by_key(|(_, node)| **node).map_or(0, |(index, _)| index);
    cycle.rotate_left(start);
    if cycle.len() > 2 && cycle[cycle.len() - 1] < cycle[1] {
        cycle[1..].reverse();
    }
    cycle
}

fn is_simple_cycle_in_graph(cycle: &[usize], edges: &[[usize; 2]]) -> bool {
    if cycle.len() < 3 {
        return false;
    }
    let mut unique_nodes = cycle.to_vec();
    unique_nodes.sort_unstable();
    unique_nodes.dedup();
    if unique_nodes.len() != cycle.len() {
        return false;
    }

    let edge_set = edges
        .iter()
        .copied()
        .map(|[left, right]| (left.min(right), left.max(right)))
        .collect::<BTreeSet<_>>();
    for pair in cycle.windows(2) {
        if !edge_set.contains(&(pair[0].min(pair[1]), pair[0].max(pair[1]))) {
            return false;
        }
    }
    let last = cycle[cycle.len() - 1];
    let first = cycle[0];
    edge_set.contains(&(last.min(first), last.max(first)))
}
