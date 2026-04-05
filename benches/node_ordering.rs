//! Criterion benchmarks for graph-level node ordering algorithms.

#[path = "../tests/support/node_ordering_fixture.rs"]
mod node_ordering_fixture;

use std::{collections::BTreeMap, hint::black_box, time::Duration};

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use geometric_traits::{
    impls::{CSR2D, SortedVec, SymmetricCSR2D},
    prelude::*,
    traits::{SquareMatrix, VocabularyBuilder, algorithms::randomized_graphs::erdos_renyi_gnp},
};
use node_ordering_fixture::{PreparedNodeOrderingCase, prepare_cases};

const FIXTURE_NAME: &str = "node_ordering_ground_truth.json.gz";
const PAGERANK_TOLERANCE: f64 = 1.0e-12;

type UndirectedGraph = SymmetricCSR2D<CSR2D<usize, usize, usize>>;

fn wrap_undi(g: UndirectedGraph) -> UndiGraph<usize> {
    let n = g.order();
    let nodes: SortedVec<usize> = GenericVocabularyBuilder::default()
        .expected_number_of_symbols(n)
        .symbols((0..n).enumerate())
        .build()
        .unwrap();
    UndiGraph::from((nodes, g))
}

#[derive(Clone)]
struct ScalingCase {
    name: String,
    graph: UndiGraph<usize>,
}

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

fn assert_scores_close(actual: &[f64], expected: &[f64], tolerance: f64, context: &str) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "benchmark scores `{context}` do not contain exactly one entry per node"
    );
    for (index, (actual_score, expected_score)) in actual.iter().zip(expected.iter()).enumerate() {
        let delta = (actual_score - expected_score).abs();
        assert!(
            delta <= tolerance,
            "benchmark scores `{context}` differ at node {index}: actual={actual_score}, expected={expected_score}, delta={delta}, tolerance={tolerance}"
        );
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

fn assert_cases_match_pagerank_scores(cases: &[PreparedNodeOrderingCase], group_name: &str) {
    for case in cases {
        let scorer = PageRankScorerBuilder::default()
            .alpha(case.pagerank_alpha)
            .max_iter(case.pagerank_max_iter)
            .tolerance(case.pagerank_tol)
            .build();
        let pagerank_scores = scorer.score_nodes(&case.graph);
        let context = format!("{group_name}::{} ({})", case.name, case.family);
        assert_scores_close(&pagerank_scores, &case.pagerank_scores, PAGERANK_TOLERANCE, &context);
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

fn total_scaling_nodes(cases: &[&ScalingCase]) -> usize {
    cases.iter().map(|case| case.graph.number_of_nodes()).sum()
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

fn bench_pagerank_scorer_cases(
    c: &mut Criterion,
    group_name: &str,
    cases: &[PreparedNodeOrderingCase],
) {
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
                let checksum = case_refs
                    .iter()
                    .map(|case| {
                        PageRankScorerBuilder::default()
                            .alpha(case.pagerank_alpha)
                            .max_iter(case.pagerank_max_iter)
                            .tolerance(case.pagerank_tol)
                            .build()
                            .score_nodes(&case.graph)
                            .into_iter()
                            .map(f64::to_bits)
                            .fold(0u64, u64::wrapping_add)
                    })
                    .fold(0u64, |accumulator, case_checksum| {
                        accumulator.wrapping_add(case_checksum)
                    });
                black_box(checksum);
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
                    let checksum = family_cases
                        .iter()
                        .map(|case| {
                            PageRankScorerBuilder::default()
                                .alpha(case.pagerank_alpha)
                                .max_iter(case.pagerank_max_iter)
                                .tolerance(case.pagerank_tol)
                                .build()
                                .score_nodes(&case.graph)
                                .into_iter()
                                .map(f64::to_bits)
                                .fold(0u64, u64::wrapping_add)
                        })
                        .fold(0u64, |accumulator, case_checksum| {
                            accumulator.wrapping_add(case_checksum)
                        });
                    black_box(checksum);
                });
            },
        );
    }
    family_group.finish();
}

fn bench_pagerank_scorer_scaling(c: &mut Criterion, group_name: &str, cases: &[ScalingCase]) {
    let case_refs: Vec<&ScalingCase> = cases.iter().collect();
    let mut total_group = c.benchmark_group(group_name);
    total_group.sample_size(10);
    total_group.warm_up_time(Duration::from_millis(500));
    total_group.measurement_time(Duration::from_secs(3));
    total_group.throughput(Throughput::Elements(
        u64::try_from(case_refs.len()).expect("scaling case count should fit into u64"),
    ));
    total_group.bench_function(
        BenchmarkId::new(
            "total_cases",
            format!("cases={}_nodes={}", case_refs.len(), total_scaling_nodes(&case_refs)),
        ),
        |b| {
            b.iter(|| {
                let checksum = case_refs
                    .iter()
                    .map(|case| {
                        PageRankScorer::default()
                            .score_nodes(&case.graph)
                            .into_iter()
                            .map(f64::to_bits)
                            .fold(0u64, u64::wrapping_add)
                    })
                    .fold(0u64, |accumulator, case_checksum| {
                        accumulator.wrapping_add(case_checksum)
                    });
                black_box(checksum);
            });
        },
    );
    total_group.finish();

    let mut size_group = c.benchmark_group(format!("{group_name}_by_case"));
    size_group.sample_size(10);
    size_group.warm_up_time(Duration::from_millis(500));
    size_group.measurement_time(Duration::from_secs(2));

    for case in cases {
        size_group.throughput(Throughput::Elements(
            u64::try_from(case.graph.number_of_nodes()).expect("node count should fit into u64"),
        ));
        size_group.bench_function(BenchmarkId::new("case", &case.name), |b| {
            b.iter(|| {
                let checksum = PageRankScorer::default()
                    .score_nodes(&case.graph)
                    .into_iter()
                    .map(f64::to_bits)
                    .fold(0u64, u64::wrapping_add);
                black_box(checksum);
            });
        });
    }
    size_group.finish();
}

fn pagerank_scaling_cases() -> Vec<ScalingCase> {
    let sparse = [(64usize, 0.05f64), (128, 0.05), (256, 0.05), (512, 0.05)];
    let dense = [(64usize, 0.20f64), (128, 0.20), (256, 0.20), (512, 0.20)];

    sparse
        .into_iter()
        .enumerate()
        .map(|(index, (n, p))| {
            ScalingCase {
                name: format!("sparse_gnp_{n}_p_{p:.2}_seed_{index}"),
                graph: wrap_undi(erdos_renyi_gnp(index as u64 + 1, n, p)),
            }
        })
        .chain(dense.into_iter().enumerate().map(|(index, (n, p))| {
            ScalingCase {
                name: format!("dense_gnp_{n}_p_{p:.2}_seed_{index}"),
                graph: wrap_undi(erdos_renyi_gnp(index as u64 + 101, n, p)),
            }
        }))
        .collect()
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

fn bench_pagerank_scorer(c: &mut Criterion) {
    let cases = prepare_cases(FIXTURE_NAME);
    assert_cases_match_pagerank_scores(&cases, "node_ordering_pagerank_scorer");
    bench_pagerank_scorer_cases(c, "node_ordering_pagerank_scorer", &cases);
}

fn bench_pagerank_sorter(c: &mut Criterion) {
    let cases = prepare_cases(FIXTURE_NAME);
    for case in &cases {
        let sorter = DescendingScoreSorter::new(
            PageRankScorerBuilder::default()
                .alpha(case.pagerank_alpha)
                .max_iter(case.pagerank_max_iter)
                .tolerance(case.pagerank_tol)
                .build(),
        );
        let order = sorter.sort_nodes(&case.graph);
        let context = format!("node_ordering_pagerank_sorter::{} ({})", case.name, case.family);
        assert_eq!(
            order, case.pagerank_descending,
            "benchmark ordering `{context}` mismatched oracle"
        );
    }

    let case_refs: Vec<&PreparedNodeOrderingCase> = cases.iter().collect();
    let mut total_group = c.benchmark_group("node_ordering_pagerank_sorter");
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
                    .map(|case| {
                        DescendingScoreSorter::new(
                            PageRankScorerBuilder::default()
                                .alpha(case.pagerank_alpha)
                                .max_iter(case.pagerank_max_iter)
                                .tolerance(case.pagerank_tol)
                                .build(),
                        )
                        .sort_nodes(&case.graph)
                        .len()
                    })
                    .sum::<usize>();
                black_box(total);
            });
        },
    );
    total_group.finish();

    let mut families: BTreeMap<String, Vec<&PreparedNodeOrderingCase>> = BTreeMap::new();
    for case in &cases {
        families.entry(case.family.clone()).or_default().push(case);
    }

    let mut family_group = c.benchmark_group("node_ordering_pagerank_sorter_by_family");
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
                        .map(|case| {
                            DescendingScoreSorter::new(
                                PageRankScorerBuilder::default()
                                    .alpha(case.pagerank_alpha)
                                    .max_iter(case.pagerank_max_iter)
                                    .tolerance(case.pagerank_tol)
                                    .build(),
                            )
                            .sort_nodes(&case.graph)
                            .len()
                        })
                        .sum::<usize>();
                    black_box(total);
                });
            },
        );
    }
    family_group.finish();
}

fn bench_pagerank_scaling(c: &mut Criterion) {
    let cases = pagerank_scaling_cases();
    bench_pagerank_scorer_scaling(c, "node_ordering_pagerank_scaling", &cases);
}

criterion_group!(
    benches,
    bench_degeneracy,
    bench_degeneracy_degree,
    bench_pagerank_scorer,
    bench_pagerank_sorter,
    bench_pagerank_scaling
);
criterion_main!(benches);
