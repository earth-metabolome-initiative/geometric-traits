//! Criterion benchmarks for graph-level node ordering algorithms.

#[path = "../tests/support/node_ordering_fixture.rs"]
mod node_ordering_fixture;

use std::{collections::BTreeMap, hint::black_box, time::Duration};

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use geometric_traits::prelude::*;
use node_ordering_fixture::{PreparedNodeOrderingCase, prepare_cases};

const FIXTURE_NAME: &str = "node_ordering_ground_truth.json.gz";

fn assert_is_permutation(order: &[usize], n: usize, context: &str) {
    assert_eq!(
        order.len(),
        n,
        "benchmark ordering `{context}` does not contain exactly one entry per node"
    );
    let mut seen = vec![false; n];
    for &node in order {
        assert!(node < n, "benchmark ordering `{context}` contains out-of-range node {node}");
        assert!(!seen[node], "benchmark ordering `{context}` contains duplicate node {node}");
        seen[node] = true;
    }
}

fn assert_is_smallest_last_order(graph: &UndiGraph<usize>, order: &[usize], context: &str) {
    let n = graph.number_of_nodes();
    assert_is_permutation(order, n, context);

    let mut active = vec![true; n];
    let mut degrees: Vec<usize> = (0..n).map(|node| graph.degree(node)).collect();

    for &node in order.iter().rev() {
        let min_degree =
            (0..n).filter(|&candidate| active[candidate]).map(|candidate| degrees[candidate]).min();
        assert_eq!(
            Some(degrees[node]),
            min_degree,
            "benchmark ordering `{context}` is not a valid smallest-last sequence at removed node {node}"
        );

        active[node] = false;
        for neighbor in graph.neighbors(node) {
            if active[neighbor] {
                degrees[neighbor] = degrees[neighbor].saturating_sub(1);
            }
        }
    }
}

fn assert_cases_are_smallest_last<S>(
    cases: &[PreparedNodeOrderingCase],
    group_name: &str,
    sorter: S,
) where
    S: NodeSorter<UndiGraph<usize>> + Copy,
{
    for case in cases {
        let order = sorter.sort_nodes(&case.graph);
        let context = format!("{group_name}::{} ({})", case.name, case.family);
        assert_is_smallest_last_order(&case.graph, &order, &context);
    }
}

fn assert_cases_match_exact_order<S, F>(
    cases: &[PreparedNodeOrderingCase],
    group_name: &str,
    sorter: S,
    expected_order: F,
) where
    S: NodeSorter<UndiGraph<usize>> + Copy,
    F: Fn(&PreparedNodeOrderingCase) -> &[usize],
{
    for case in cases {
        let order = sorter.sort_nodes(&case.graph);
        let context = format!("{group_name}::{} ({})", case.name, case.family);
        assert_eq!(order, expected_order(case), "benchmark ordering `{context}` mismatched oracle");
    }
}

fn total_nodes(cases: &[&PreparedNodeOrderingCase]) -> usize {
    cases.iter().map(|case| case.canonical_smallest_last.len()).sum()
}

fn bench_sorter<S>(
    c: &mut Criterion,
    group_name: &str,
    cases: &[PreparedNodeOrderingCase],
    sorter: S,
) where
    S: NodeSorter<UndiGraph<usize>> + Copy,
{
    let case_refs: Vec<&PreparedNodeOrderingCase> = cases.iter().collect();
    let mut total_group = c.benchmark_group(group_name);
    total_group.sample_size(10);
    total_group.warm_up_time(Duration::from_millis(500));
    total_group.measurement_time(Duration::from_secs(3));
    total_group.throughput(Throughput::Elements(
        u64::try_from(case_refs.len()).expect("fixture size should fit into u64"),
    ));
    total_group.bench_function(
        BenchmarkId::new(
            "total_cases",
            format!("cases={}_nodes={}", case_refs.len(), total_nodes(&case_refs)),
        ),
        |b| {
            b.iter(|| {
                let total = case_refs
                    .iter()
                    .map(|case| sorter.sort_nodes(&case.graph).len())
                    .sum::<usize>();
                black_box(total);
            });
        },
    );
    total_group.finish();

    let mut families: BTreeMap<String, Vec<&PreparedNodeOrderingCase>> = BTreeMap::new();
    for case in cases {
        families.entry(case.family.clone()).or_default().push(case);
    }

    let mut family_group = c.benchmark_group(format!("{group_name}_by_family"));
    family_group.sample_size(10);
    family_group.warm_up_time(Duration::from_millis(500));
    family_group.measurement_time(Duration::from_secs(2));

    for (family, family_cases) in families {
        family_group.throughput(Throughput::Elements(
            u64::try_from(family_cases.len()).expect("family case count should fit into u64"),
        ));
        family_group.bench_function(
            BenchmarkId::new(
                "family_cases",
                format!(
                    "{family}_cases={}_nodes={}",
                    family_cases.len(),
                    total_nodes(&family_cases)
                ),
            ),
            |b| {
                b.iter(|| {
                    let total = family_cases
                        .iter()
                        .map(|case| sorter.sort_nodes(&case.graph).len())
                        .sum::<usize>();
                    black_box(total);
                });
            },
        );
    }
    family_group.finish();
}

fn bench_degeneracy(c: &mut Criterion) {
    let cases = prepare_cases(FIXTURE_NAME);
    assert_cases_are_smallest_last(&cases, "node_ordering_degeneracy", DegeneracySorter);
    bench_sorter(c, "node_ordering_degeneracy", &cases, DegeneracySorter);
}

fn bench_degeneracy_degree(c: &mut Criterion) {
    let cases = prepare_cases(FIXTURE_NAME);
    let sorter = DescendingLexicographicScoreSorter::new(CoreNumberScorer, DegreeScorer);
    assert_cases_match_exact_order(&cases, "node_ordering_degeneracy_degree", sorter, |case| {
        &case.degeneracy_degree_descending
    });
    bench_sorter(c, "node_ordering_degeneracy_degree", &cases, sorter);
}

criterion_group!(benches, bench_degeneracy, bench_degeneracy_degree);
criterion_main!(benches);
