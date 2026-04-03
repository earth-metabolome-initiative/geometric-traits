//! NetworkX-backed VF2 oracle suite over one merged fixture corpus.
#![cfg(feature = "std")]

#[path = "support/vf2_fixture_suite.rs"]
mod vf2_fixture_suite;

use std::time::Duration;

use geometric_traits::traits::Vf2;
use vf2_fixture_suite::{
    BuiltDiGraph, BuiltUndiGraph, FixtureCase, OracleKind, SemanticMatch, build_digraph,
    build_undigraph, canonicalize_mapping_pairs, collect_matches, load_fixture_suite,
    parse_match_mode,
};

const FIXTURE_NAME: &str = "vf2_networkx_fixture_suite.json.gz";

fn fixture() -> vf2_fixture_suite::FixtureSuite {
    load_fixture_suite(FIXTURE_NAME)
}

fn count_cases(
    cases: &[FixtureCase],
    oracle_kind: OracleKind,
    semantic_match: SemanticMatch,
    directed: bool,
    self_loops: bool,
) -> usize {
    cases
        .iter()
        .filter(|case| {
            case.oracle_kind == oracle_kind
                && case.semantic_match == semantic_match
                && case.directed == directed
                && case.self_loops == self_loops
        })
        .count()
}

fn count_cases_with_mode(
    cases: &[FixtureCase],
    oracle_kind: OracleKind,
    semantic_match: SemanticMatch,
    directed: bool,
    self_loops: bool,
    match_mode: &str,
) -> usize {
    cases
        .iter()
        .filter(|case| {
            case.oracle_kind == oracle_kind
                && case.semantic_match == semantic_match
                && case.directed == directed
                && case.self_loops == self_loops
                && case.match_mode == match_mode
        })
        .count()
}

fn assert_boolean_case(case: &FixtureCase) {
    match (case.directed, case.semantic_match) {
        (false, SemanticMatch::None) => {
            let query = build_undigraph(&case.query);
            let target = build_undigraph(&case.target);
            let actual = query
                .graph
                .vf2(&target.graph)
                .with_mode(parse_match_mode(&case.match_mode))
                .has_match();
            assert_eq!(
                actual, case.expected_has_match,
                "case {} from {} did not match the stored NetworkX boolean oracle",
                case.name, case.source_fixture
            );
        }
        (true, SemanticMatch::None) => {
            let query = build_digraph(&case.query);
            let target = build_digraph(&case.target);
            let actual = query
                .graph
                .vf2(&target.graph)
                .with_mode(parse_match_mode(&case.match_mode))
                .has_match();
            assert_eq!(
                actual, case.expected_has_match,
                "case {} from {} did not match the stored NetworkX boolean oracle",
                case.name, case.source_fixture
            );
        }
        (false, SemanticMatch::LabelEquality) => {
            let query = build_undigraph(&case.query);
            let target = build_undigraph(&case.target);
            assert_boolean_labeled_undirected(case, &query, &target);
        }
        (true, SemanticMatch::LabelEquality) => {
            let query = build_digraph(&case.query);
            let target = build_digraph(&case.target);
            assert_boolean_labeled_directed(case, &query, &target);
        }
    }
}

fn assert_boolean_labeled_undirected(
    case: &FixtureCase,
    query: &BuiltUndiGraph,
    target: &BuiltUndiGraph,
) {
    let actual = query
        .graph
        .vf2(&target.graph)
        .with_mode(parse_match_mode(&case.match_mode))
        .with_node_match(|query_node, target_node| {
            query.node_labels[query_node] == target.node_labels[target_node]
        })
        .with_edge_match(|query_src, query_dst, target_src, target_dst| {
            query.edge_label(query_src, query_dst) == target.edge_label(target_src, target_dst)
        })
        .has_match();
    assert_eq!(
        actual, case.expected_has_match,
        "labeled case {} from {} did not match the stored NetworkX boolean oracle",
        case.name, case.source_fixture
    );
}

fn assert_boolean_labeled_directed(
    case: &FixtureCase,
    query: &BuiltDiGraph,
    target: &BuiltDiGraph,
) {
    let actual = query
        .graph
        .vf2(&target.graph)
        .with_mode(parse_match_mode(&case.match_mode))
        .with_node_match(|query_node, target_node| {
            query.node_labels[query_node] == target.node_labels[target_node]
        })
        .with_edge_match(|query_src, query_dst, target_src, target_dst| {
            query.edge_label(query_src, query_dst) == target.edge_label(target_src, target_dst)
        })
        .has_match();
    assert_eq!(
        actual, case.expected_has_match,
        "labeled case {} from {} did not match the stored NetworkX boolean oracle",
        case.name, case.source_fixture
    );
}

fn assert_embedding_case(case: &FixtureCase) {
    match (case.directed, case.semantic_match) {
        (false, SemanticMatch::None) => {
            let query = build_undigraph(&case.query);
            let target = build_undigraph(&case.target);
            let actual_matches =
                collect_matches(&query.graph, &target.graph, parse_match_mode(&case.match_mode));
            assert_embedding_results(case, &actual_matches);
        }
        (true, SemanticMatch::None) => {
            let query = build_digraph(&case.query);
            let target = build_digraph(&case.target);
            let actual_matches =
                collect_matches(&query.graph, &target.graph, parse_match_mode(&case.match_mode));
            assert_embedding_results(case, &actual_matches);
        }
        (false, SemanticMatch::LabelEquality) => {
            let query = build_undigraph(&case.query);
            let target = build_undigraph(&case.target);
            let mut actual_matches = Vec::new();
            let exhausted = query
                .graph
                .vf2(&target.graph)
                .with_mode(parse_match_mode(&case.match_mode))
                .with_node_match(|query_node, target_node| {
                    query.node_labels[query_node] == target.node_labels[target_node]
                })
                .with_edge_match(|query_src, query_dst, target_src, target_dst| {
                    query.edge_label(query_src, query_dst)
                        == target.edge_label(target_src, target_dst)
                })
                .for_each_match(|mapping| {
                    actual_matches.push(canonicalize_mapping_pairs(mapping.pairs().to_vec()));
                    true
                });
            assert!(exhausted);
            actual_matches.sort_unstable();
            assert_embedding_results(case, &actual_matches);
        }
        (true, SemanticMatch::LabelEquality) => {
            let query = build_digraph(&case.query);
            let target = build_digraph(&case.target);
            let mut actual_matches = Vec::new();
            let exhausted = query
                .graph
                .vf2(&target.graph)
                .with_mode(parse_match_mode(&case.match_mode))
                .with_node_match(|query_node, target_node| {
                    query.node_labels[query_node] == target.node_labels[target_node]
                })
                .with_edge_match(|query_src, query_dst, target_src, target_dst| {
                    query.edge_label(query_src, query_dst)
                        == target.edge_label(target_src, target_dst)
                })
                .for_each_match(|mapping| {
                    actual_matches.push(canonicalize_mapping_pairs(mapping.pairs().to_vec()));
                    true
                });
            assert!(exhausted);
            actual_matches.sort_unstable();
            assert_embedding_results(case, &actual_matches);
        }
    }
}

fn assert_embedding_results(case: &FixtureCase, actual_matches: &[Vec<[usize; 2]>]) {
    assert_eq!(
        !actual_matches.is_empty(),
        case.expected_has_match,
        "case {} from {} had the wrong boolean result",
        case.name,
        case.source_fixture
    );
    assert_eq!(
        actual_matches.len(),
        case.expected_match_count,
        "case {} from {} had the wrong match count",
        case.name,
        case.source_fixture
    );
    assert_eq!(
        actual_matches,
        case.expected_matches.as_slice(),
        "case {} from {} had the wrong embeddings",
        case.name,
        case.source_fixture
    );
}

#[test]
fn test_vf2_networkx_fixture_suite_metadata_header() {
    let suite = fixture();
    assert_eq!(suite.schema_version, 1);
    assert_eq!(suite.generator, "networkx");
    assert_eq!(suite.networkx_timing_unit, "ns");
    assert_eq!(suite.cases.len(), 40_792);
    assert!(
        suite.cases.iter().all(|case| case.networkx_ns > 0),
        "every merged NetworkX oracle case should have a non-zero timing"
    );
    assert!(
        Duration::from_nanos(suite.cases.iter().map(|case| case.networkx_ns).sum::<u64>())
            > Duration::ZERO
    );
}

#[test]
fn test_vf2_networkx_fixture_suite_boolean_family_counts() {
    let suite = fixture();
    assert_eq!(
        count_cases(&suite.cases, OracleKind::Boolean, SemanticMatch::None, false, false),
        13_333
    );
    assert_eq!(
        count_cases(&suite.cases, OracleKind::Boolean, SemanticMatch::None, true, false),
        13_333
    );
    assert_eq!(
        count_cases(&suite.cases, OracleKind::Boolean, SemanticMatch::None, false, true),
        3_333
    );
    assert_eq!(
        count_cases(&suite.cases, OracleKind::Boolean, SemanticMatch::None, true, true),
        3_333
    );
    assert_eq!(
        count_cases(&suite.cases, OracleKind::Boolean, SemanticMatch::LabelEquality, false, false,),
        1_333
    );
    assert_eq!(
        count_cases(&suite.cases, OracleKind::Boolean, SemanticMatch::LabelEquality, false, true,),
        1_333
    );
    assert_eq!(
        count_cases(&suite.cases, OracleKind::Boolean, SemanticMatch::LabelEquality, true, false,),
        1_333
    );
    assert_eq!(
        count_cases(&suite.cases, OracleKind::Boolean, SemanticMatch::LabelEquality, true, true,),
        1_333
    );
}

#[test]
fn test_vf2_networkx_fixture_suite_embedding_family_counts() {
    let suite = fixture();
    assert_eq!(
        count_cases(&suite.cases, OracleKind::Embeddings, SemanticMatch::None, false, false),
        266
    );
    assert_eq!(
        count_cases(&suite.cases, OracleKind::Embeddings, SemanticMatch::None, true, false),
        266
    );
    assert_eq!(
        count_cases(&suite.cases, OracleKind::Embeddings, SemanticMatch::None, false, true),
        266
    );
    assert_eq!(
        count_cases(&suite.cases, OracleKind::Embeddings, SemanticMatch::None, true, true),
        266
    );
    assert_eq!(
        count_cases(
            &suite.cases,
            OracleKind::Embeddings,
            SemanticMatch::LabelEquality,
            false,
            false,
        ),
        266
    );
    assert_eq!(
        count_cases(
            &suite.cases,
            OracleKind::Embeddings,
            SemanticMatch::LabelEquality,
            false,
            true,
        ),
        266
    );
    assert_eq!(
        count_cases(
            &suite.cases,
            OracleKind::Embeddings,
            SemanticMatch::LabelEquality,
            true,
            false,
        ),
        266
    );
    assert_eq!(
        count_cases(&suite.cases, OracleKind::Embeddings, SemanticMatch::LabelEquality, true, true,),
        266
    );
}

#[test]
fn test_vf2_networkx_fixture_suite_monomorphism_counts() {
    let suite = fixture();
    assert_eq!(
        count_cases_with_mode(
            &suite.cases,
            OracleKind::Boolean,
            SemanticMatch::None,
            false,
            false,
            "monomorphism",
        ),
        3_333
    );
    assert_eq!(
        count_cases_with_mode(
            &suite.cases,
            OracleKind::Embeddings,
            SemanticMatch::LabelEquality,
            true,
            true,
            "monomorphism",
        ),
        66
    );
}

#[test]
fn test_vf2_networkx_boolean_oracle_cases() {
    let suite = fixture();
    for case in suite.cases.iter().filter(|case| case.oracle_kind == OracleKind::Boolean) {
        assert_boolean_case(case);
    }
}

#[test]
fn test_vf2_networkx_embedding_oracle_cases() {
    let suite = fixture();
    for case in suite.cases.iter().filter(|case| case.oracle_kind == OracleKind::Embeddings) {
        assert_embedding_case(case);
    }
}
