//! Criterion benchmarks for graph-level node ordering algorithms.

#[path = "../tests/support/node_ordering_fixture.rs"]
mod node_ordering_fixture;

use std::{collections::BTreeMap, hint::black_box, time::Duration};

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use geometric_traits::{
    impls::{CSR2D, SortedVec, SymmetricCSR2D},
    prelude::*,
    traits::{
        SquareMatrix, VocabularyBuilder,
        algorithms::randomized_graphs::{cycle_graph, erdos_renyi_gnp, path_graph, star_graph},
    },
};
use node_ordering_fixture::{PreparedNodeOrderingCase, prepare_cases};
use num_traits::cast;

const FIXTURE_NAME: &str = "node_ordering_ground_truth.json.gz";
const PAGERANK_TOLERANCE: f64 = 1.0e-12;
const KATZ_TOLERANCE: f64 = 1.0e-11;
const BETWEENNESS_TOLERANCE: f64 = 2.0e-12;
const CLOSENESS_TOLERANCE: f64 = 1.0e-12;
const LOCAL_CLUSTERING_TOLERANCE: f64 = 1.0e-12;
const POWER_ITERATION_EIGENVECTOR_TOLERANCE: f64 = 2.0e-12;

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

fn complete_bipartite(left: usize, right: usize) -> UndiGraph<usize> {
    let order = left + right;
    let nodes: SortedVec<usize> = GenericVocabularyBuilder::default()
        .expected_number_of_symbols(order)
        .symbols((0..order).enumerate())
        .build()
        .unwrap();
    let edges = (0..left).flat_map(|u| (left..order).map(move |v| (u, v)));
    let matrix: UndirectedGraph = UndiEdgesBuilder::default()
        .expected_number_of_edges(left * right)
        .expected_shape(order)
        .edges(edges)
        .build()
        .unwrap();
    UndiGraph::from((nodes, matrix))
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
    sorter: &S,
) where
    S: NodeSorter<UndiGraph<usize>>,
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

fn assert_cases_match_power_iteration_eigenvector_scores(
    cases: &[PreparedNodeOrderingCase],
    group_name: &str,
) {
    for case in cases {
        let scorer = PowerIterationEigenvectorCentralityScorerBuilder::default()
            .max_iter(case.power_iteration_eigenvector_max_iter)
            .tolerance(case.power_iteration_eigenvector_tol)
            .build();
        let eigenvector_scores = scorer.score_nodes(&case.graph);
        let context = format!("{group_name}::{} ({})", case.name, case.family);
        assert_scores_close(
            &eigenvector_scores,
            &case.power_iteration_eigenvector_scores,
            POWER_ITERATION_EIGENVECTOR_TOLERANCE,
            &context,
        );
    }
}

fn assert_cases_match_katz_scores(cases: &[PreparedNodeOrderingCase], group_name: &str) {
    for case in cases {
        let scorer = KatzCentralityScorerBuilder::default()
            .alpha(case.katz_alpha)
            .beta(case.katz_beta)
            .max_iter(case.katz_max_iter)
            .tolerance(case.katz_tol)
            .normalized(case.katz_normalized)
            .build();
        let katz_scores = scorer.score_nodes(&case.graph);
        let context = format!("{group_name}::{} ({})", case.name, case.family);
        assert_scores_close(&katz_scores, &case.katz_scores, KATZ_TOLERANCE, &context);
    }
}

fn assert_cases_match_betweenness_scores(cases: &[PreparedNodeOrderingCase], group_name: &str) {
    for case in cases {
        let scorer = BetweennessCentralityScorerBuilder::default()
            .normalized(case.betweenness_normalized)
            .endpoints(case.betweenness_endpoints)
            .build();
        let betweenness_scores = scorer.score_nodes(&case.graph);
        let context = format!("{group_name}::{} ({})", case.name, case.family);
        assert_scores_close(
            &betweenness_scores,
            &case.betweenness_scores,
            BETWEENNESS_TOLERANCE,
            &context,
        );
    }
}

fn assert_cases_match_closeness_scores(cases: &[PreparedNodeOrderingCase], group_name: &str) {
    for case in cases {
        let scorer = ClosenessCentralityScorerBuilder::default()
            .wf_improved(case.closeness_wf_improved)
            .build();
        let closeness_scores = scorer.score_nodes(&case.graph);
        let context = format!("{group_name}::{} ({})", case.name, case.family);
        assert_scores_close(
            &closeness_scores,
            &case.closeness_scores,
            CLOSENESS_TOLERANCE,
            &context,
        );
    }
}

fn assert_cases_match_triangle_counts(cases: &[PreparedNodeOrderingCase], group_name: &str) {
    let scorer = TriangleCountScorer::new(MotifCountOrdering::IncreasingDegree);
    for case in cases {
        let triangle_counts = scorer.score_nodes(&case.graph);
        let context = format!("{group_name}::{} ({})", case.name, case.family);
        assert_eq!(
            triangle_counts, case.triangle_counts,
            "benchmark scores `{context}` mismatched oracle"
        );
    }
}

fn assert_cases_match_local_clustering_scores(
    cases: &[PreparedNodeOrderingCase],
    group_name: &str,
) {
    for case in cases {
        let local_clustering_scores = LocalClusteringCoefficientScorer.score_nodes(&case.graph);
        let context = format!("{group_name}::{} ({})", case.name, case.family);
        assert_scores_close(
            &local_clustering_scores,
            &case.local_clustering_scores,
            LOCAL_CLUSTERING_TOLERANCE,
            &context,
        );
    }
}

fn assert_cases_match_exact_order<S, F>(
    cases: &[PreparedNodeOrderingCase],
    group_name: &str,
    sorter: &S,
    expected_order: F,
) where
    S: NodeSorter<UndiGraph<usize>>,
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

fn assert_sorter_returns_permutations<S>(
    cases: &[PreparedNodeOrderingCase],
    group_name: &str,
    sorter: &S,
) where
    S: NodeSorter<UndiGraph<usize>>,
{
    for case in cases {
        let order = sorter.sort_nodes(&case.graph);
        let context = format!("{group_name}::{} ({})", case.name, case.family);
        assert_is_permutation(&order, case.graph.number_of_nodes(), &context);
    }
}

fn bench_sorter_scaling<S>(c: &mut Criterion, group_name: &str, cases: &[ScalingCase], sorter: &S)
where
    S: NodeSorter<UndiGraph<usize>> + Clone,
{
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
        {
            let case_refs = case_refs.clone();
            let sorter = sorter.clone();
            move |b| {
                b.iter(|| {
                    let total = case_refs
                        .iter()
                        .map(|case| sorter.sort_nodes(&case.graph).len())
                        .sum::<usize>();
                    black_box(total);
                });
            }
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
        size_group.bench_function(BenchmarkId::new("case", &case.name), {
            let graph = case.graph.clone();
            let sorter = sorter.clone();
            move |b| {
                b.iter(|| {
                    let total = sorter.sort_nodes(&graph).len();
                    black_box(total);
                });
            }
        });
    }
    size_group.finish();
}

fn bench_sorter<S>(
    c: &mut Criterion,
    group_name: &str,
    cases: &[PreparedNodeOrderingCase],
    sorter: &S,
) where
    S: NodeSorter<UndiGraph<usize>> + Clone,
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
        {
            let case_refs = case_refs.clone();
            let sorter = sorter.clone();
            move |b| {
                b.iter(|| {
                    let total = case_refs
                        .iter()
                        .map(|case| sorter.sort_nodes(&case.graph).len())
                        .sum::<usize>();
                    black_box(total);
                });
            }
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
            {
                let family_cases = family_cases.clone();
                let sorter = sorter.clone();
                move |b| {
                    b.iter(|| {
                        let total = family_cases
                            .iter()
                            .map(|case| sorter.sort_nodes(&case.graph).len())
                            .sum::<usize>();
                        black_box(total);
                    });
                }
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

fn bench_katz_scorer_cases(
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
                        KatzCentralityScorerBuilder::default()
                            .alpha(case.katz_alpha)
                            .beta(case.katz_beta)
                            .max_iter(case.katz_max_iter)
                            .tolerance(case.katz_tol)
                            .normalized(case.katz_normalized)
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
                            KatzCentralityScorerBuilder::default()
                                .alpha(case.katz_alpha)
                                .beta(case.katz_beta)
                                .max_iter(case.katz_max_iter)
                                .tolerance(case.katz_tol)
                                .normalized(case.katz_normalized)
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

fn bench_betweenness_scorer_cases(
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
                        BetweennessCentralityScorerBuilder::default()
                            .normalized(case.betweenness_normalized)
                            .endpoints(case.betweenness_endpoints)
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
                            BetweennessCentralityScorerBuilder::default()
                                .normalized(case.betweenness_normalized)
                                .endpoints(case.betweenness_endpoints)
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

fn bench_closeness_scorer_cases(
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
                        ClosenessCentralityScorerBuilder::default()
                            .wf_improved(case.closeness_wf_improved)
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
                            ClosenessCentralityScorerBuilder::default()
                                .wf_improved(case.closeness_wf_improved)
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

fn bench_katz_scorer_scaling(c: &mut Criterion, group_name: &str, cases: &[ScalingCase]) {
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
                        katz_scaling_scorer(&case.graph)
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
                let checksum = katz_scaling_scorer(&case.graph)
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

fn centrality_scaling_cases() -> Vec<ScalingCase> {
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

fn betweenness_scaling_cases() -> Vec<ScalingCase> {
    let sparse = [(32usize, 0.05f64), (64, 0.05), (96, 0.05), (128, 0.05)];
    let dense = [(32usize, 0.20f64), (64, 0.20), (96, 0.20), (128, 0.20)];

    sparse
        .into_iter()
        .enumerate()
        .map(|(index, (n, p))| {
            ScalingCase {
                name: format!("betweenness_sparse_gnp_{n}_p_{p:.2}_seed_{index}"),
                graph: wrap_undi(erdos_renyi_gnp(index as u64 + 1_001, n, p)),
            }
        })
        .chain(dense.into_iter().enumerate().map(|(index, (n, p))| {
            ScalingCase {
                name: format!("betweenness_dense_gnp_{n}_p_{p:.2}_seed_{index}"),
                graph: wrap_undi(erdos_renyi_gnp(index as u64 + 1_101, n, p)),
            }
        }))
        .collect()
}

fn closeness_scaling_cases() -> Vec<ScalingCase> {
    let sparse = [(32usize, 0.05f64), (64, 0.05), (96, 0.05), (128, 0.05)];
    let dense = [(32usize, 0.20f64), (64, 0.20), (96, 0.20), (128, 0.20)];

    sparse
        .into_iter()
        .enumerate()
        .map(|(index, (n, p))| {
            ScalingCase {
                name: format!("closeness_sparse_gnp_{n}_p_{p:.2}_seed_{index}"),
                graph: wrap_undi(erdos_renyi_gnp(index as u64 + 1_201, n, p)),
            }
        })
        .chain(dense.into_iter().enumerate().map(|(index, (n, p))| {
            ScalingCase {
                name: format!("closeness_dense_gnp_{n}_p_{p:.2}_seed_{index}"),
                graph: wrap_undi(erdos_renyi_gnp(index as u64 + 1_301, n, p)),
            }
        }))
        .collect()
}

fn dsatur_scaling_cases() -> Vec<ScalingCase> {
    let sparse = [(64usize, 0.05f64), (128, 0.05), (256, 0.05), (512, 0.05)];
    let dense = [(64usize, 0.20f64), (128, 0.20), (256, 0.20), (512, 0.20)];

    sparse
        .into_iter()
        .enumerate()
        .map(|(index, (n, p))| {
            ScalingCase {
                name: format!("dsatur_sparse_gnp_{n}_p_{p:.2}_seed_{index}"),
                graph: wrap_undi(erdos_renyi_gnp(index as u64 + 1_601, n, p)),
            }
        })
        .chain(dense.into_iter().enumerate().map(|(index, (n, p))| {
            ScalingCase {
                name: format!("dsatur_dense_gnp_{n}_p_{p:.2}_seed_{index}"),
                graph: wrap_undi(erdos_renyi_gnp(index as u64 + 1_701, n, p)),
            }
        }))
        .collect()
}

fn triangle_scaling_cases() -> Vec<ScalingCase> {
    let sparse = [(64usize, 0.05f64), (128, 0.05), (256, 0.05), (512, 0.05)];
    let dense = [(64usize, 0.20f64), (96, 0.20), (128, 0.20), (160, 0.20)];

    sparse
        .into_iter()
        .enumerate()
        .map(|(index, (n, p))| {
            ScalingCase {
                name: format!("triangle_sparse_gnp_{n}_p_{p:.2}_seed_{index}"),
                graph: wrap_undi(erdos_renyi_gnp(index as u64 + 1_401, n, p)),
            }
        })
        .chain(dense.into_iter().enumerate().map(|(index, (n, p))| {
            ScalingCase {
                name: format!("triangle_dense_gnp_{n}_p_{p:.2}_seed_{index}"),
                graph: wrap_undi(erdos_renyi_gnp(index as u64 + 1_501, n, p)),
            }
        }))
        .collect()
}

fn traversal_scaling_cases() -> Vec<ScalingCase> {
    let sparse = [(64usize, 0.05f64), (128, 0.05), (256, 0.05), (512, 0.05)];
    let dense = [(64usize, 0.20f64), (128, 0.20), (256, 0.20), (512, 0.20)];

    sparse
        .into_iter()
        .enumerate()
        .map(|(index, (n, p))| {
            ScalingCase {
                name: format!("traversal_sparse_gnp_{n}_p_{p:.2}_seed_{index}"),
                graph: wrap_undi(erdos_renyi_gnp(index as u64 + 1_801, n, p)),
            }
        })
        .chain(dense.into_iter().enumerate().map(|(index, (n, p))| {
            ScalingCase {
                name: format!("traversal_dense_gnp_{n}_p_{p:.2}_seed_{index}"),
                graph: wrap_undi(erdos_renyi_gnp(index as u64 + 1_901, n, p)),
            }
        }))
        .collect()
}

fn llp_scaling_cases() -> Vec<ScalingCase> {
    let sparse = [(64usize, 0.05f64), (128, 0.05), (256, 0.05), (384, 0.05)];
    let dense = [(64usize, 0.20f64), (96, 0.20), (128, 0.20), (160, 0.20)];

    sparse
        .into_iter()
        .enumerate()
        .map(|(index, (n, p))| {
            ScalingCase {
                name: format!("llp_sparse_gnp_{n}_p_{p:.2}_seed_{index}"),
                graph: wrap_undi(erdos_renyi_gnp(index as u64 + 2_001, n, p)),
            }
        })
        .chain(dense.into_iter().enumerate().map(|(index, (n, p))| {
            ScalingCase {
                name: format!("llp_dense_gnp_{n}_p_{p:.2}_seed_{index}"),
                graph: wrap_undi(erdos_renyi_gnp(index as u64 + 2_101, n, p)),
            }
        }))
        .collect()
}

fn representative_llp_fixture_cases() -> Vec<PreparedNodeOrderingCase> {
    let mut families: BTreeMap<String, Vec<PreparedNodeOrderingCase>> = BTreeMap::new();
    for case in prepare_cases(FIXTURE_NAME) {
        families.entry(case.family.clone()).or_default().push(case);
    }

    let mut selected = Vec::new();
    for mut family_cases in families.into_values() {
        family_cases.sort_unstable_by(|left, right| {
            left.graph
                .number_of_nodes()
                .cmp(&right.graph.number_of_nodes())
                .then_with(|| left.name.cmp(&right.name))
        });

        let mut pick_indices = vec![0usize, family_cases.len() / 2, family_cases.len() - 1];
        pick_indices.sort_unstable();
        pick_indices.dedup();
        for index in pick_indices {
            selected.push(family_cases[index].clone());
        }
    }

    selected
}

fn katz_scaling_scorer(graph: &UndiGraph<usize>) -> KatzCentralityScorer {
    let max_degree = (0..graph.number_of_nodes()).map(|node| graph.degree(node)).max().unwrap_or(0);
    let safe_denominator = cast::<usize, f64>(if max_degree == 0 { 1 } else { max_degree + 1 })
        .expect("graph sizes and degrees must fit into f64 for Katz centrality scaling benchmarks");
    KatzCentralityScorerBuilder::default()
        .alpha(0.8 / safe_denominator)
        .beta(1.5)
        .max_iter(1200)
        .tolerance(1.0e-12)
        .normalized(true)
        .build()
}

fn betweenness_scaling_scorer() -> BetweennessCentralityScorer {
    BetweennessCentralityScorer::default()
}

fn closeness_scaling_scorer() -> ClosenessCentralityScorer {
    ClosenessCentralityScorer::default()
}

fn power_iteration_eigenvector_scaling_scorer() -> PowerIterationEigenvectorCentralityScorer {
    PowerIterationEigenvectorCentralityScorerBuilder::default()
        .max_iter(8_192)
        .tolerance(1.0e-6)
        .build()
}

fn power_iteration_eigenvector_scaling_cases() -> Vec<ScalingCase> {
    let structural = [
        ScalingCase { name: "path_128".to_string(), graph: wrap_undi(path_graph(128)) },
        ScalingCase { name: "path_256".to_string(), graph: wrap_undi(path_graph(256)) },
        ScalingCase { name: "path_512".to_string(), graph: wrap_undi(path_graph(512)) },
        ScalingCase { name: "cycle_128".to_string(), graph: wrap_undi(cycle_graph(128)) },
        ScalingCase { name: "cycle_256".to_string(), graph: wrap_undi(cycle_graph(256)) },
        ScalingCase { name: "cycle_512".to_string(), graph: wrap_undi(cycle_graph(512)) },
        ScalingCase { name: "star_128".to_string(), graph: wrap_undi(star_graph(128)) },
        ScalingCase { name: "star_256".to_string(), graph: wrap_undi(star_graph(256)) },
        ScalingCase { name: "star_512".to_string(), graph: wrap_undi(star_graph(512)) },
        ScalingCase {
            name: "complete_bipartite_64_64".to_string(),
            graph: complete_bipartite(64, 64),
        },
        ScalingCase {
            name: "complete_bipartite_128_128".to_string(),
            graph: complete_bipartite(128, 128),
        },
        ScalingCase {
            name: "complete_bipartite_256_256".to_string(),
            graph: complete_bipartite(256, 256),
        },
    ];
    let random =
        [(128usize, 0.05f64), (256, 0.05), (512, 0.05), (128, 0.20), (256, 0.20), (512, 0.20)];

    structural
        .into_iter()
        .chain(random.into_iter().enumerate().map(|(index, (n, p))| {
            ScalingCase {
                name: format!("gnp_{n}_p_{p:.2}_seed_{index}"),
                graph: wrap_undi(erdos_renyi_gnp(index as u64 + 3_001, n, p)),
            }
        }))
        .collect()
}

fn bench_triangle_scorer(c: &mut Criterion) {
    let cases = prepare_cases(FIXTURE_NAME);
    assert_cases_match_triangle_counts(&cases, "node_ordering_triangle_scorer");
    let case_refs: Vec<&PreparedNodeOrderingCase> = cases.iter().collect();
    let scorer = TriangleCountScorer::new(MotifCountOrdering::IncreasingDegree);

    let mut total_group = c.benchmark_group("node_ordering_triangle_scorer");
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
                        scorer
                            .score_nodes(&case.graph)
                            .into_iter()
                            .map(|value| {
                                u64::try_from(value).expect("triangle count should fit into u64")
                            })
                            .fold(0u64, u64::wrapping_add)
                    })
                    .fold(0u64, u64::wrapping_add);
                black_box(checksum);
            });
        },
    );
    total_group.finish();

    let mut families: BTreeMap<String, Vec<&PreparedNodeOrderingCase>> = BTreeMap::new();
    for case in &cases {
        families.entry(case.family.clone()).or_default().push(case);
    }

    let mut family_group = c.benchmark_group("node_ordering_triangle_scorer_by_family");
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
                            scorer
                                .score_nodes(&case.graph)
                                .into_iter()
                                .map(|value| {
                                    u64::try_from(value)
                                        .expect("triangle count should fit into u64")
                                })
                                .fold(0u64, u64::wrapping_add)
                        })
                        .fold(0u64, u64::wrapping_add);
                    black_box(checksum);
                });
            },
        );
    }
    family_group.finish();
}

fn bench_triangle_sorter(c: &mut Criterion) {
    let cases = prepare_cases(FIXTURE_NAME);
    let sorter =
        DescendingScoreSorter::new(TriangleCountScorer::new(MotifCountOrdering::IncreasingDegree));
    assert_cases_match_exact_order(&cases, "node_ordering_triangle_sorter", &sorter, |case| {
        &case.triangle_descending
    });
    bench_sorter(c, "node_ordering_triangle_sorter", &cases, &sorter);
}

fn bench_local_clustering_scorer(c: &mut Criterion) {
    let cases = prepare_cases(FIXTURE_NAME);
    assert_cases_match_local_clustering_scores(&cases, "node_ordering_local_clustering_scorer");
    let case_refs: Vec<&PreparedNodeOrderingCase> = cases.iter().collect();

    let mut total_group = c.benchmark_group("node_ordering_local_clustering_scorer");
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
                        LocalClusteringCoefficientScorer
                            .score_nodes(&case.graph)
                            .into_iter()
                            .map(f64::to_bits)
                            .fold(0u64, u64::wrapping_add)
                    })
                    .fold(0u64, u64::wrapping_add);
                black_box(checksum);
            });
        },
    );
    total_group.finish();

    let mut families: BTreeMap<String, Vec<&PreparedNodeOrderingCase>> = BTreeMap::new();
    for case in &cases {
        families.entry(case.family.clone()).or_default().push(case);
    }

    let mut family_group = c.benchmark_group("node_ordering_local_clustering_scorer_by_family");
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
                            LocalClusteringCoefficientScorer
                                .score_nodes(&case.graph)
                                .into_iter()
                                .map(f64::to_bits)
                                .fold(0u64, u64::wrapping_add)
                        })
                        .fold(0u64, u64::wrapping_add);
                    black_box(checksum);
                });
            },
        );
    }
    family_group.finish();
}

fn bench_local_clustering_sorter(c: &mut Criterion) {
    let cases = prepare_cases(FIXTURE_NAME);
    let sorter = DescendingScoreSorter::new(LocalClusteringCoefficientScorer);
    assert_cases_match_exact_order(
        &cases,
        "node_ordering_local_clustering_sorter",
        &sorter,
        |case| &case.local_clustering_descending,
    );
    bench_sorter(c, "node_ordering_local_clustering_sorter", &cases, &sorter);
}

fn bench_degeneracy(c: &mut Criterion) {
    let cases = prepare_cases(FIXTURE_NAME);
    assert_cases_are_smallest_last(&cases, "node_ordering_degeneracy", &DegeneracySorter);
    bench_sorter(c, "node_ordering_degeneracy", &cases, &DegeneracySorter);
}

fn bench_degeneracy_degree(c: &mut Criterion) {
    let cases = prepare_cases(FIXTURE_NAME);
    let sorter = DescendingLexicographicScoreSorter::new(CoreNumberScorer, DegreeScorer);
    assert_cases_match_exact_order(&cases, "node_ordering_degeneracy_degree", &sorter, |case| {
        &case.degeneracy_degree_descending
    });
    bench_sorter(c, "node_ordering_degeneracy_degree", &cases, &sorter);
}

fn bench_welsh_powell(c: &mut Criterion) {
    let cases = prepare_cases(FIXTURE_NAME);
    let sorter = DescendingScoreSorter::new(DegreeScorer);
    assert_cases_match_exact_order(&cases, "node_ordering_welsh_powell", &sorter, |case| {
        &case.welsh_powell_descending
    });
    bench_sorter(c, "node_ordering_welsh_powell", &cases, &sorter);
}

fn bench_dsatur(c: &mut Criterion) {
    let cases = prepare_cases(FIXTURE_NAME);
    assert_cases_match_exact_order(&cases, "node_ordering_dsatur", &DsaturSorter, |case| {
        &case.dsatur_order
    });
    bench_sorter(c, "node_ordering_dsatur", &cases, &DsaturSorter);
}

fn bench_bfs_from_max_degree(c: &mut Criterion) {
    let cases = prepare_cases(FIXTURE_NAME);
    let sorter = BfsTraversalSorter::new(
        TraversalSeedStrategy::MaxOutDegree,
        TraversalNeighborOrder::NodeIdAscending,
    );
    assert_cases_match_exact_order(&cases, "node_ordering_bfs_from_max_degree", &sorter, |case| {
        &case.bfs_from_max_degree
    });
    bench_sorter(c, "node_ordering_bfs_from_max_degree", &cases, &sorter);
}

fn bench_dfs_from_max_degree(c: &mut Criterion) {
    let cases = prepare_cases(FIXTURE_NAME);
    let sorter = DfsTraversalSorter::new(
        TraversalSeedStrategy::MaxOutDegree,
        TraversalNeighborOrder::NodeIdAscending,
    );
    assert_cases_match_exact_order(&cases, "node_ordering_dfs_from_max_degree", &sorter, |case| {
        &case.dfs_from_max_degree
    });
    bench_sorter(c, "node_ordering_dfs_from_max_degree", &cases, &sorter);
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

fn bench_power_iteration_eigenvector_scorer(c: &mut Criterion) {
    let cases = prepare_cases(FIXTURE_NAME);
    assert_cases_match_power_iteration_eigenvector_scores(
        &cases,
        "node_ordering_power_iteration_eigenvector_scorer",
    );
    let case_refs: Vec<&PreparedNodeOrderingCase> = cases.iter().collect();

    let mut total_group = c.benchmark_group("node_ordering_power_iteration_eigenvector_scorer");
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
                        PowerIterationEigenvectorCentralityScorerBuilder::default()
                            .max_iter(case.power_iteration_eigenvector_max_iter)
                            .tolerance(case.power_iteration_eigenvector_tol)
                            .build()
                            .score_nodes(&case.graph)
                            .into_iter()
                            .map(f64::to_bits)
                            .fold(0u64, u64::wrapping_add)
                    })
                    .fold(0u64, u64::wrapping_add);
                black_box(checksum);
            });
        },
    );
    total_group.finish();

    let mut families: BTreeMap<String, Vec<&PreparedNodeOrderingCase>> = BTreeMap::new();
    for case in &cases {
        families.entry(case.family.clone()).or_default().push(case);
    }

    let mut family_group =
        c.benchmark_group("node_ordering_power_iteration_eigenvector_scorer_by_family");
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
                            PowerIterationEigenvectorCentralityScorerBuilder::default()
                                .max_iter(case.power_iteration_eigenvector_max_iter)
                                .tolerance(case.power_iteration_eigenvector_tol)
                                .build()
                                .score_nodes(&case.graph)
                                .into_iter()
                                .map(f64::to_bits)
                                .fold(0u64, u64::wrapping_add)
                        })
                        .fold(0u64, u64::wrapping_add);
                    black_box(checksum);
                });
            },
        );
    }
    family_group.finish();
}

fn bench_power_iteration_eigenvector_sorter(c: &mut Criterion) {
    let cases = prepare_cases(FIXTURE_NAME);
    for case in &cases {
        let sorter = DescendingScoreSorter::new(
            PowerIterationEigenvectorCentralityScorerBuilder::default()
                .max_iter(case.power_iteration_eigenvector_max_iter)
                .tolerance(case.power_iteration_eigenvector_tol)
                .build(),
        );
        let order = sorter.sort_nodes(&case.graph);
        let context = format!(
            "node_ordering_power_iteration_eigenvector_sorter::{} ({})",
            case.name, case.family
        );
        assert_eq!(
            order, case.power_iteration_eigenvector_descending,
            "benchmark ordering `{context}` mismatched oracle"
        );
    }

    let case_refs: Vec<&PreparedNodeOrderingCase> = cases.iter().collect();
    let mut total_group = c.benchmark_group("node_ordering_power_iteration_eigenvector_sorter");
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
                            PowerIterationEigenvectorCentralityScorerBuilder::default()
                                .max_iter(case.power_iteration_eigenvector_max_iter)
                                .tolerance(case.power_iteration_eigenvector_tol)
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

    let mut family_group =
        c.benchmark_group("node_ordering_power_iteration_eigenvector_sorter_by_family");
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
                                PowerIterationEigenvectorCentralityScorerBuilder::default()
                                    .max_iter(case.power_iteration_eigenvector_max_iter)
                                    .tolerance(case.power_iteration_eigenvector_tol)
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

fn bench_katz_scorer(c: &mut Criterion) {
    let cases = prepare_cases(FIXTURE_NAME);
    assert_cases_match_katz_scores(&cases, "node_ordering_katz_scorer");
    bench_katz_scorer_cases(c, "node_ordering_katz_scorer", &cases);
}

fn bench_katz_sorter(c: &mut Criterion) {
    let cases = prepare_cases(FIXTURE_NAME);
    for case in &cases {
        let sorter = DescendingScoreSorter::new(
            KatzCentralityScorerBuilder::default()
                .alpha(case.katz_alpha)
                .beta(case.katz_beta)
                .max_iter(case.katz_max_iter)
                .tolerance(case.katz_tol)
                .normalized(case.katz_normalized)
                .build(),
        );
        let order = sorter.sort_nodes(&case.graph);
        let context = format!("node_ordering_katz_sorter::{} ({})", case.name, case.family);
        assert_eq!(order, case.katz_descending, "benchmark ordering `{context}` mismatched oracle");
    }

    let case_refs: Vec<&PreparedNodeOrderingCase> = cases.iter().collect();
    let mut total_group = c.benchmark_group("node_ordering_katz_sorter");
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
                            KatzCentralityScorerBuilder::default()
                                .alpha(case.katz_alpha)
                                .beta(case.katz_beta)
                                .max_iter(case.katz_max_iter)
                                .tolerance(case.katz_tol)
                                .normalized(case.katz_normalized)
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

    let mut family_group = c.benchmark_group("node_ordering_katz_sorter_by_family");
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
                                KatzCentralityScorerBuilder::default()
                                    .alpha(case.katz_alpha)
                                    .beta(case.katz_beta)
                                    .max_iter(case.katz_max_iter)
                                    .tolerance(case.katz_tol)
                                    .normalized(case.katz_normalized)
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

fn bench_betweenness_scorer(c: &mut Criterion) {
    let cases = prepare_cases(FIXTURE_NAME);
    assert_cases_match_betweenness_scores(&cases, "node_ordering_betweenness_scorer");
    bench_betweenness_scorer_cases(c, "node_ordering_betweenness_scorer", &cases);
}

fn bench_betweenness_sorter(c: &mut Criterion) {
    let cases = prepare_cases(FIXTURE_NAME);
    for case in &cases {
        let sorter = DescendingScoreSorter::new(
            BetweennessCentralityScorerBuilder::default()
                .normalized(case.betweenness_normalized)
                .endpoints(case.betweenness_endpoints)
                .build(),
        );
        let order = sorter.sort_nodes(&case.graph);
        let context = format!("node_ordering_betweenness_sorter::{} ({})", case.name, case.family);
        assert_eq!(
            order, case.betweenness_descending,
            "benchmark ordering `{context}` mismatched oracle"
        );
    }

    let case_refs: Vec<&PreparedNodeOrderingCase> = cases.iter().collect();
    let mut total_group = c.benchmark_group("node_ordering_betweenness_sorter");
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
                            BetweennessCentralityScorerBuilder::default()
                                .normalized(case.betweenness_normalized)
                                .endpoints(case.betweenness_endpoints)
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

    let mut family_group = c.benchmark_group("node_ordering_betweenness_sorter_by_family");
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
                                BetweennessCentralityScorerBuilder::default()
                                    .normalized(case.betweenness_normalized)
                                    .endpoints(case.betweenness_endpoints)
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

fn bench_closeness_scorer(c: &mut Criterion) {
    let cases = prepare_cases(FIXTURE_NAME);
    assert_cases_match_closeness_scores(&cases, "node_ordering_closeness_scorer");
    bench_closeness_scorer_cases(c, "node_ordering_closeness_scorer", &cases);
}

fn bench_closeness_sorter(c: &mut Criterion) {
    let cases = prepare_cases(FIXTURE_NAME);
    for case in &cases {
        let sorter = DescendingScoreSorter::new(
            ClosenessCentralityScorerBuilder::default()
                .wf_improved(case.closeness_wf_improved)
                .build(),
        );
        let order = sorter.sort_nodes(&case.graph);
        let context = format!("node_ordering_closeness_sorter::{} ({})", case.name, case.family);
        assert_eq!(
            order, case.closeness_descending,
            "benchmark ordering `{context}` mismatched oracle"
        );
    }

    let case_refs: Vec<&PreparedNodeOrderingCase> = cases.iter().collect();
    let mut total_group = c.benchmark_group("node_ordering_closeness_sorter");
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
                            ClosenessCentralityScorerBuilder::default()
                                .wf_improved(case.closeness_wf_improved)
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

    let mut family_group = c.benchmark_group("node_ordering_closeness_sorter_by_family");
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
                                ClosenessCentralityScorerBuilder::default()
                                    .wf_improved(case.closeness_wf_improved)
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
    let cases = centrality_scaling_cases();
    bench_pagerank_scorer_scaling(c, "node_ordering_pagerank_scaling", &cases);
}

fn bench_katz_scaling(c: &mut Criterion) {
    let cases = centrality_scaling_cases();
    bench_katz_scorer_scaling(c, "node_ordering_katz_scaling", &cases);
}

fn bench_betweenness_scaling(c: &mut Criterion) {
    let cases = betweenness_scaling_cases();
    let case_refs: Vec<&ScalingCase> = cases.iter().collect();
    let mut total_group = c.benchmark_group("node_ordering_betweenness_scaling");
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
                        betweenness_scaling_scorer()
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

    let mut size_group = c.benchmark_group("node_ordering_betweenness_scaling_by_case");
    size_group.sample_size(10);
    size_group.warm_up_time(Duration::from_millis(500));
    size_group.measurement_time(Duration::from_secs(2));

    for case in cases {
        size_group.throughput(Throughput::Elements(
            u64::try_from(case.graph.number_of_nodes()).expect("node count should fit into u64"),
        ));
        size_group.bench_function(BenchmarkId::new("case", &case.name), |b| {
            b.iter(|| {
                let checksum = betweenness_scaling_scorer()
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

fn bench_closeness_scaling(c: &mut Criterion) {
    let cases = closeness_scaling_cases();
    let case_refs: Vec<&ScalingCase> = cases.iter().collect();
    let mut total_group = c.benchmark_group("node_ordering_closeness_scaling");
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
                        closeness_scaling_scorer()
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

    let mut size_group = c.benchmark_group("node_ordering_closeness_scaling_by_case");
    size_group.sample_size(10);
    size_group.warm_up_time(Duration::from_millis(500));
    size_group.measurement_time(Duration::from_secs(2));

    for case in cases {
        size_group.throughput(Throughput::Elements(
            u64::try_from(case.graph.number_of_nodes()).expect("node count should fit into u64"),
        ));
        size_group.bench_function(BenchmarkId::new("case", &case.name), |b| {
            b.iter(|| {
                let checksum = closeness_scaling_scorer()
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

fn bench_power_iteration_eigenvector_scaling(c: &mut Criterion) {
    let cases = power_iteration_eigenvector_scaling_cases();
    let case_refs: Vec<&ScalingCase> = cases.iter().collect();
    let mut total_group = c.benchmark_group("node_ordering_power_iteration_eigenvector_scaling");
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
                        power_iteration_eigenvector_scaling_scorer()
                            .score_nodes(&case.graph)
                            .into_iter()
                            .map(f64::to_bits)
                            .fold(0u64, u64::wrapping_add)
                    })
                    .fold(0u64, u64::wrapping_add);
                black_box(checksum);
            });
        },
    );
    total_group.finish();

    let mut size_group =
        c.benchmark_group("node_ordering_power_iteration_eigenvector_scaling_by_case");
    size_group.sample_size(10);
    size_group.warm_up_time(Duration::from_millis(500));
    size_group.measurement_time(Duration::from_secs(2));

    for case in cases {
        size_group.throughput(Throughput::Elements(
            u64::try_from(case.graph.number_of_nodes()).expect("node count should fit into u64"),
        ));
        size_group.bench_function(BenchmarkId::new("case", &case.name), |b| {
            b.iter(|| {
                let checksum = power_iteration_eigenvector_scaling_scorer()
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

fn bench_triangle_scaling(c: &mut Criterion) {
    let cases = triangle_scaling_cases();
    let case_refs: Vec<&ScalingCase> = cases.iter().collect();
    let mut total_group = c.benchmark_group("node_ordering_triangle_scaling");
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
                        TriangleCountScorer::new(MotifCountOrdering::IncreasingDegree)
                            .score_nodes(&case.graph)
                            .into_iter()
                            .map(|value| {
                                u64::try_from(value).expect("triangle count should fit into u64")
                            })
                            .fold(0u64, u64::wrapping_add)
                    })
                    .fold(0u64, u64::wrapping_add);
                black_box(checksum);
            });
        },
    );
    total_group.finish();

    let mut size_group = c.benchmark_group("node_ordering_triangle_scaling_by_case");
    size_group.sample_size(10);
    size_group.warm_up_time(Duration::from_millis(500));
    size_group.measurement_time(Duration::from_secs(2));

    for case in cases {
        size_group.throughput(Throughput::Elements(
            u64::try_from(case.graph.number_of_nodes()).expect("node count should fit into u64"),
        ));
        size_group.bench_function(BenchmarkId::new("case", &case.name), |b| {
            b.iter(|| {
                let checksum = TriangleCountScorer::new(MotifCountOrdering::IncreasingDegree)
                    .score_nodes(&case.graph)
                    .into_iter()
                    .map(|value| u64::try_from(value).expect("triangle count should fit into u64"))
                    .fold(0u64, u64::wrapping_add);
                black_box(checksum);
            });
        });
    }
    size_group.finish();
}

fn bench_local_clustering_scaling(c: &mut Criterion) {
    let cases = triangle_scaling_cases();
    let case_refs: Vec<&ScalingCase> = cases.iter().collect();
    let mut total_group = c.benchmark_group("node_ordering_local_clustering_scaling");
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
                        LocalClusteringCoefficientScorer
                            .score_nodes(&case.graph)
                            .into_iter()
                            .map(f64::to_bits)
                            .fold(0u64, u64::wrapping_add)
                    })
                    .fold(0u64, u64::wrapping_add);
                black_box(checksum);
            });
        },
    );
    total_group.finish();

    let mut size_group = c.benchmark_group("node_ordering_local_clustering_scaling_by_case");
    size_group.sample_size(10);
    size_group.warm_up_time(Duration::from_millis(500));
    size_group.measurement_time(Duration::from_secs(2));

    for case in cases {
        size_group.throughput(Throughput::Elements(
            u64::try_from(case.graph.number_of_nodes()).expect("node count should fit into u64"),
        ));
        size_group.bench_function(BenchmarkId::new("case", &case.name), |b| {
            b.iter(|| {
                let checksum = LocalClusteringCoefficientScorer
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

fn bench_welsh_powell_scaling(c: &mut Criterion) {
    let cases = centrality_scaling_cases();
    bench_sorter_scaling(
        c,
        "node_ordering_welsh_powell_scaling",
        &cases,
        &DescendingScoreSorter::new(DegreeScorer),
    );
}

fn bench_dsatur_scaling(c: &mut Criterion) {
    let cases = dsatur_scaling_cases();
    bench_sorter_scaling(c, "node_ordering_dsatur_scaling", &cases, &DsaturSorter);
}

fn bench_bfs_from_max_degree_scaling(c: &mut Criterion) {
    let cases = traversal_scaling_cases();
    let sorter = BfsTraversalSorter::new(
        TraversalSeedStrategy::MaxOutDegree,
        TraversalNeighborOrder::NodeIdAscending,
    );
    bench_sorter_scaling(c, "node_ordering_bfs_from_max_degree_scaling", &cases, &sorter);
}

fn bench_dfs_from_max_degree_scaling(c: &mut Criterion) {
    let cases = traversal_scaling_cases();
    let sorter = DfsTraversalSorter::new(
        TraversalSeedStrategy::MaxOutDegree,
        TraversalNeighborOrder::NodeIdAscending,
    );
    bench_sorter_scaling(c, "node_ordering_dfs_from_max_degree_scaling", &cases, &sorter);
}

fn bench_layered_label_propagation(c: &mut Criterion) {
    let cases = representative_llp_fixture_cases();
    let sorter = LayeredLabelPropagationSorter::default();
    assert_sorter_returns_permutations(&cases, "node_ordering_layered_label_propagation", &sorter);
    bench_sorter(c, "node_ordering_layered_label_propagation", &cases, &sorter);
}

fn bench_layered_label_propagation_scaling(c: &mut Criterion) {
    let cases = llp_scaling_cases();
    bench_sorter_scaling(
        c,
        "node_ordering_layered_label_propagation_scaling",
        &cases,
        &LayeredLabelPropagationSorter::default(),
    );
}

criterion_group!(
    benches,
    bench_degeneracy,
    bench_degeneracy_degree,
    bench_welsh_powell,
    bench_welsh_powell_scaling,
    bench_dsatur,
    bench_dsatur_scaling,
    bench_bfs_from_max_degree,
    bench_bfs_from_max_degree_scaling,
    bench_dfs_from_max_degree,
    bench_dfs_from_max_degree_scaling,
    bench_layered_label_propagation,
    bench_layered_label_propagation_scaling,
    bench_triangle_scorer,
    bench_triangle_sorter,
    bench_triangle_scaling,
    bench_local_clustering_scorer,
    bench_local_clustering_sorter,
    bench_local_clustering_scaling,
    bench_power_iteration_eigenvector_scorer,
    bench_power_iteration_eigenvector_sorter,
    bench_power_iteration_eigenvector_scaling,
    bench_pagerank_scorer,
    bench_pagerank_sorter,
    bench_pagerank_scaling,
    bench_katz_scorer,
    bench_katz_sorter,
    bench_katz_scaling,
    bench_betweenness_scorer,
    bench_betweenness_sorter,
    bench_betweenness_scaling,
    bench_closeness_scorer,
    bench_closeness_sorter,
    bench_closeness_scaling
);
criterion_main!(benches);
