//! Criterion benchmark for VF2 against the merged committed NetworkX oracle.

#[path = "../tests/support/vf2_fixture_suite.rs"]
mod vf2_fixture_suite;

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use geometric_traits::prelude::*;
use vf2_fixture_suite::{
    BuiltDiGraph, BuiltUndiGraph, FixtureCase, OracleKind, SemanticMatch, build_digraph,
    build_undigraph, load_fixture_suite, parse_match_mode,
};

const FIXTURE_NAME: &str = "vf2_networkx_fixture_suite.json.gz";

#[derive(Clone)]
struct UndirectedCase {
    query: BuiltUndiGraph,
    target: BuiltUndiGraph,
    mode: Vf2Mode,
    expected_has_match: bool,
    semantic_match: SemanticMatch,
}

#[derive(Clone)]
struct DirectedCase {
    query: BuiltDiGraph,
    target: BuiltDiGraph,
    mode: Vf2Mode,
    expected_has_match: bool,
    semantic_match: SemanticMatch,
}

fn benchmark_label(group_name: &str, networkx_total_nanos: u64) -> String {
    format!("{group_name}_networkx_total={networkx_total_nanos}ns")
}

fn assert_undirected_cases_match_oracle(cases: &[UndirectedCase], group_name: &str) {
    for (index, case) in cases.iter().enumerate() {
        let actual = match case.semantic_match {
            SemanticMatch::None => {
                case.query.graph.vf2(&case.target.graph).with_mode(case.mode).has_match()
            }
            SemanticMatch::LabelEquality => {
                case.query
                    .graph
                    .vf2(&case.target.graph)
                    .with_mode(case.mode)
                    .with_node_match(|query_node, target_node| {
                        case.query.node_labels[query_node] == case.target.node_labels[target_node]
                    })
                    .with_edge_match(|query_src, query_dst, target_src, target_dst| {
                        case.query.edge_label(query_src, query_dst)
                            == case.target.edge_label(target_src, target_dst)
                    })
                    .has_match()
            }
        };
        assert_eq!(
            actual, case.expected_has_match,
            "undirected benchmark case {index} from `{group_name}` did not match the stored NetworkX oracle"
        );
    }
}

fn assert_directed_cases_match_oracle(cases: &[DirectedCase], group_name: &str) {
    for (index, case) in cases.iter().enumerate() {
        let actual = match case.semantic_match {
            SemanticMatch::None => {
                case.query.graph.vf2(&case.target.graph).with_mode(case.mode).has_match()
            }
            SemanticMatch::LabelEquality => {
                case.query
                    .graph
                    .vf2(&case.target.graph)
                    .with_mode(case.mode)
                    .with_node_match(|query_node, target_node| {
                        case.query.node_labels[query_node] == case.target.node_labels[target_node]
                    })
                    .with_edge_match(|query_src, query_dst, target_src, target_dst| {
                        case.query.edge_label(query_src, query_dst)
                            == case.target.edge_label(target_src, target_dst)
                    })
                    .has_match()
            }
        };
        assert_eq!(
            actual, case.expected_has_match,
            "directed benchmark case {index} from `{group_name}` did not match the stored NetworkX oracle"
        );
    }
}

fn load_undirected_cases(
    cases: &[FixtureCase],
    semantic_match: SemanticMatch,
    self_loops: bool,
) -> (Vec<UndirectedCase>, u64) {
    let selected: Vec<&FixtureCase> = cases
        .iter()
        .filter(|case| {
            case.oracle_kind == OracleKind::Boolean
                && !case.directed
                && case.semantic_match == semantic_match
                && case.self_loops == self_loops
        })
        .collect();
    let networkx_total_nanos = selected.iter().map(|case| case.networkx_ns).sum();
    let built = selected
        .into_iter()
        .map(|case| {
            UndirectedCase {
                query: build_undigraph(&case.query),
                target: build_undigraph(&case.target),
                mode: parse_match_mode(&case.match_mode),
                expected_has_match: case.expected_has_match,
                semantic_match,
            }
        })
        .collect();
    (built, networkx_total_nanos)
}

fn load_directed_cases(
    cases: &[FixtureCase],
    semantic_match: SemanticMatch,
    self_loops: bool,
) -> (Vec<DirectedCase>, u64) {
    let selected: Vec<&FixtureCase> = cases
        .iter()
        .filter(|case| {
            case.oracle_kind == OracleKind::Boolean
                && case.directed
                && case.semantic_match == semantic_match
                && case.self_loops == self_loops
        })
        .collect();
    let networkx_total_nanos = selected.iter().map(|case| case.networkx_ns).sum();
    let built = selected
        .into_iter()
        .map(|case| {
            DirectedCase {
                query: build_digraph(&case.query),
                target: build_digraph(&case.target),
                mode: parse_match_mode(&case.match_mode),
                expected_has_match: case.expected_has_match,
                semantic_match,
            }
        })
        .collect();
    (built, networkx_total_nanos)
}

fn bench_undirected_group(
    c: &mut Criterion,
    group_name: &str,
    cases: &[UndirectedCase],
    networkx_total_nanos: u64,
) {
    assert_undirected_cases_match_oracle(cases, group_name);
    let mut group = c.benchmark_group(group_name);
    group.throughput(Throughput::Elements(
        u64::try_from(cases.len()).expect("fixture size should fit into u64"),
    ));
    group.bench_function(
        BenchmarkId::new("rust_has_match_total", benchmark_label(group_name, networkx_total_nanos)),
        |b| {
            b.iter(|| {
                let matches = cases
                    .iter()
                    .filter(|case| {
                        match case.semantic_match {
                            SemanticMatch::None => {
                                case.query
                                    .graph
                                    .vf2(&case.target.graph)
                                    .with_mode(case.mode)
                                    .has_match()
                            }
                            SemanticMatch::LabelEquality => {
                                case.query
                                    .graph
                                    .vf2(&case.target.graph)
                                    .with_mode(case.mode)
                                    .with_node_match(|query_node, target_node| {
                                        case.query.node_labels[query_node]
                                            == case.target.node_labels[target_node]
                                    })
                                    .with_edge_match(
                                        |query_src, query_dst, target_src, target_dst| {
                                            case.query.edge_label(query_src, query_dst)
                                                == case.target.edge_label(target_src, target_dst)
                                        },
                                    )
                                    .has_match()
                            }
                        }
                    })
                    .count();
                black_box(matches);
            });
        },
    );
    group.finish();
}

fn bench_directed_group(
    c: &mut Criterion,
    group_name: &str,
    cases: &[DirectedCase],
    networkx_total_nanos: u64,
) {
    assert_directed_cases_match_oracle(cases, group_name);
    let mut group = c.benchmark_group(group_name);
    group.throughput(Throughput::Elements(
        u64::try_from(cases.len()).expect("fixture size should fit into u64"),
    ));
    group.bench_function(
        BenchmarkId::new("rust_has_match_total", benchmark_label(group_name, networkx_total_nanos)),
        |b| {
            b.iter(|| {
                let matches = cases
                    .iter()
                    .filter(|case| {
                        match case.semantic_match {
                            SemanticMatch::None => {
                                case.query
                                    .graph
                                    .vf2(&case.target.graph)
                                    .with_mode(case.mode)
                                    .has_match()
                            }
                            SemanticMatch::LabelEquality => {
                                case.query
                                    .graph
                                    .vf2(&case.target.graph)
                                    .with_mode(case.mode)
                                    .with_node_match(|query_node, target_node| {
                                        case.query.node_labels[query_node]
                                            == case.target.node_labels[target_node]
                                    })
                                    .with_edge_match(
                                        |query_src, query_dst, target_src, target_dst| {
                                            case.query.edge_label(query_src, query_dst)
                                                == case.target.edge_label(target_src, target_dst)
                                        },
                                    )
                                    .has_match()
                            }
                        }
                    })
                    .count();
                black_box(matches);
            });
        },
    );
    group.finish();
}

fn bench_vf2_fixture_suite(c: &mut Criterion) {
    let suite = load_fixture_suite(FIXTURE_NAME);

    let (undirected, undirected_ns) =
        load_undirected_cases(&suite.cases, SemanticMatch::None, false);
    bench_undirected_group(c, "vf2_fixture_undirected", &undirected, undirected_ns);

    let (directed, directed_ns) = load_directed_cases(&suite.cases, SemanticMatch::None, false);
    bench_directed_group(c, "vf2_fixture_directed", &directed, directed_ns);

    let (undirected_loops, undirected_loops_ns) =
        load_undirected_cases(&suite.cases, SemanticMatch::None, true);
    bench_undirected_group(
        c,
        "vf2_fixture_undirected_self_loop",
        &undirected_loops,
        undirected_loops_ns,
    );

    let (directed_loops, directed_loops_ns) =
        load_directed_cases(&suite.cases, SemanticMatch::None, true);
    bench_directed_group(c, "vf2_fixture_directed_self_loop", &directed_loops, directed_loops_ns);

    let (labeled_undirected, labeled_undirected_ns) =
        load_undirected_cases(&suite.cases, SemanticMatch::LabelEquality, false);
    bench_undirected_group(
        c,
        "vf2_fixture_labeled_undirected",
        &labeled_undirected,
        labeled_undirected_ns,
    );

    let (labeled_undirected_loops, labeled_undirected_loops_ns) =
        load_undirected_cases(&suite.cases, SemanticMatch::LabelEquality, true);
    bench_undirected_group(
        c,
        "vf2_fixture_labeled_undirected_self_loop",
        &labeled_undirected_loops,
        labeled_undirected_loops_ns,
    );

    let (labeled_directed, labeled_directed_ns) =
        load_directed_cases(&suite.cases, SemanticMatch::LabelEquality, false);
    bench_directed_group(c, "vf2_fixture_labeled_directed", &labeled_directed, labeled_directed_ns);

    let (labeled_directed_loops, labeled_directed_loops_ns) =
        load_directed_cases(&suite.cases, SemanticMatch::LabelEquality, true);
    bench_directed_group(
        c,
        "vf2_fixture_labeled_directed_self_loop",
        &labeled_directed_loops,
        labeled_directed_loops_ns,
    );
}

criterion_group!(benches, bench_vf2_fixture_suite);
criterion_main!(benches);
