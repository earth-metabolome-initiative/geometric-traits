//! Domain-neutral corpus tests for the labeled simple-graph canonizer.
#![cfg(feature = "std")]
#![allow(clippy::pedantic)]

#[path = "support/canon_bench_fixture.rs"]
#[allow(dead_code)]
mod canon_bench_fixture;

use std::{
    fmt::Write,
    panic::{AssertUnwindSafe, catch_unwind},
};

use canon_bench_fixture::{CanonCase, benchmark_cases, scaling_cases};
use geometric_traits::{
    impls::{SortedVec, SymmetricCSR2D, ValuedCSR2D},
    naive_structs::GenericGraph,
    prelude::*,
    traits::{
        CanonSplittingHeuristic, CanonicalLabelingOptions, Edges, MonoplexGraph,
        SparseValuedMatrix2D, VocabularyBuilder, canonical_label_labeled_simple_graph,
        canonical_label_labeled_simple_graph_with_options,
    },
};
use rand::{Rng, SeedableRng, rngs::SmallRng, seq::SliceRandom};
use rayon::prelude::*;

type LabeledUndirectedEdges = SymmetricCSR2D<ValuedCSR2D<usize, usize, usize, u8>>;
type LabeledUndirectedGraph = GenericGraph<SortedVec<usize>, LabeledUndirectedEdges>;

fn build_bidirectional_labeled_graph(
    number_of_nodes: usize,
    edges: &[(usize, usize, u8)],
) -> LabeledUndirectedGraph {
    let nodes: SortedVec<usize> = GenericVocabularyBuilder::default()
        .expected_number_of_symbols(number_of_nodes)
        .symbols((0..number_of_nodes).enumerate())
        .build()
        .unwrap();
    let mut upper_edges: Vec<(usize, usize, u8)> = edges
        .iter()
        .map(|&(source, destination, label)| {
            if source <= destination {
                (source, destination, label)
            } else {
                (destination, source, label)
            }
        })
        .collect();
    upper_edges.sort_unstable_by(|left, right| {
        left.0.cmp(&right.0).then(left.1.cmp(&right.1)).then(left.2.cmp(&right.2))
    });
    upper_edges.dedup();
    let edges: LabeledUndirectedEdges =
        SymmetricCSR2D::from_sorted_upper_triangular_entries(number_of_nodes, upper_edges).unwrap();

    GenericGraph::from((nodes, edges))
}

fn permuted_case(case: &CanonCase) -> (LabeledUndirectedGraph, Vec<u8>) {
    let order = case.number_of_nodes();
    let permutation = (0..order).map(|index| (order + 2 - index) % order).collect::<Vec<_>>();
    let mut vertex_labels = vec![0_u8; order];
    for (old, &new) in permutation.iter().enumerate() {
        vertex_labels[new] = case.vertex_labels[old];
    }
    let edges = case
        .edges
        .iter()
        .map(|&(source, destination, label)| (permutation[source], permutation[destination], label))
        .collect::<Vec<_>>();
    (build_bidirectional_labeled_graph(order, &edges), vertex_labels)
}

fn canonical_certificate(
    case: &CanonCase,
) -> geometric_traits::traits::LabeledSimpleGraphCertificate<u8, u8> {
    let matrix = Edges::matrix(case.graph.edges());
    canonical_label_labeled_simple_graph(
        &case.graph,
        |node| case.vertex_labels[node],
        |left, right| matrix.sparse_value_at(left, right).unwrap(),
    )
    .certificate
}

fn random_case_count() -> usize {
    std::env::var("GEOMETRIC_TRAITS_CANON_RANDOM_CASES")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(128)
}

fn random_case_base_seed() -> u64 {
    std::env::var("GEOMETRIC_TRAITS_CANON_RANDOM_BASE_SEED")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(0xCA90_0000_0000_0001)
}

fn random_case_parallelism() -> usize {
    std::env::var("GEOMETRIC_TRAITS_CANON_RANDOM_THREADS")
        .ok()
        .and_then(|value| value.parse().ok())
        .filter(|&threads| threads > 0)
        .unwrap_or_else(|| {
            std::thread::available_parallelism().map(std::num::NonZeroUsize::get).unwrap_or(1)
        })
}

fn random_labeled_simple_case(
    seed: u64,
) -> (LabeledUndirectedGraph, Vec<u8>, Vec<(usize, usize, u8)>) {
    let mut rng = SmallRng::seed_from_u64(seed);
    let number_of_nodes = rng.gen_range(2..=14);
    let vertex_palette = rng.gen_range(1_u8..=5);
    let edge_palette = rng.gen_range(1_u8..=5);
    let vertex_labels =
        (0..number_of_nodes).map(|_| rng.gen_range(0_u8..vertex_palette)).collect::<Vec<_>>();
    let connected = rng.gen_bool(0.6);
    let max_edges = number_of_nodes * (number_of_nodes - 1) / 2;
    let min_edges = if connected { number_of_nodes - 1 } else { 0 };
    let target_edges = rng.gen_range(min_edges..=max_edges);

    let mut seen = std::collections::BTreeSet::new();
    let mut edges = Vec::with_capacity(target_edges);
    if connected {
        for node in 1..number_of_nodes {
            let parent = rng.gen_range(0..node);
            let (left, right) = if parent < node { (parent, node) } else { (node, parent) };
            seen.insert((left, right));
            edges.push((left, right, rng.gen_range(0_u8..edge_palette)));
        }
    }

    let mut remaining_pairs = Vec::with_capacity(max_edges.saturating_sub(edges.len()));
    for left in 0..number_of_nodes {
        for right in (left + 1)..number_of_nodes {
            if seen.contains(&(left, right)) {
                continue;
            }
            remaining_pairs.push((left, right));
        }
    }
    remaining_pairs.shuffle(&mut rng);

    for (left, right) in remaining_pairs.into_iter().take(target_edges.saturating_sub(edges.len()))
    {
        edges.push((left, right, rng.gen_range(0_u8..edge_palette)));
    }

    edges.sort_unstable_by(|left, right| {
        left.0.cmp(&right.0).then(left.1.cmp(&right.1)).then(left.2.cmp(&right.2))
    });
    let graph = build_bidirectional_labeled_graph(number_of_nodes, &edges);
    (graph, vertex_labels, edges)
}

fn permuted_random_case(
    graph: &LabeledUndirectedGraph,
    vertex_labels: &[u8],
    edges: &[(usize, usize, u8)],
    seed: u64,
) -> (LabeledUndirectedGraph, Vec<u8>) {
    let mut permutation = (0..graph.number_of_nodes()).collect::<Vec<_>>();
    let mut rng = SmallRng::seed_from_u64(seed);
    permutation.shuffle(&mut rng);

    let mut permuted_labels = vec![0_u8; vertex_labels.len()];
    for (old, &new) in permutation.iter().enumerate() {
        permuted_labels[new] = vertex_labels[old];
    }
    let permuted_edges = edges
        .iter()
        .map(|&(left, right, label)| (permutation[left], permutation[right], label))
        .collect::<Vec<_>>();

    (build_bidirectional_labeled_graph(graph.number_of_nodes(), &permuted_edges), permuted_labels)
}

fn exercise_canonizer_case_without_panic(graph: &LabeledUndirectedGraph, vertex_labels: &[u8]) {
    let matrix = Edges::matrix(graph.edges());
    let heuristics = [
        CanonSplittingHeuristic::First,
        CanonSplittingHeuristic::FirstSmallest,
        CanonSplittingHeuristic::FirstLargest,
        CanonSplittingHeuristic::FirstMaxNeighbours,
        CanonSplittingHeuristic::FirstSmallestMaxNeighbours,
        CanonSplittingHeuristic::FirstLargestMaxNeighbours,
    ];

    let _default = canonical_label_labeled_simple_graph(
        graph,
        |node| vertex_labels[node],
        |left, right| matrix.sparse_value_at(left, right).unwrap(),
    );

    for heuristic in heuristics {
        let _ = canonical_label_labeled_simple_graph_with_options(
            graph,
            |node| vertex_labels[node],
            |left, right| matrix.sparse_value_at(left, right).unwrap(),
            CanonicalLabelingOptions { splitting_heuristic: heuristic },
        );
    }
}

fn describe_random_case(
    seed: u64,
    vertex_labels: &[u8],
    edges: &[(usize, usize, u8)],
    reason: &str,
) -> String {
    let mut description = String::new();
    let _ = writeln!(&mut description, "seed={seed}");
    let _ = writeln!(&mut description, "vertex_labels={vertex_labels:?}");
    let _ = writeln!(&mut description, "edges={edges:?}");
    let _ = write!(&mut description, "reason={reason}");
    description
}

fn collect_random_case_panics(
    base_seed: u64,
    case_count: usize,
    parallelism: usize,
) -> Vec<String> {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(parallelism)
        .build()
        .expect("rayon pool for random canonizer crash testing should build");

    let mut failures = pool.install(|| {
        (0..case_count)
            .into_par_iter()
            .filter_map(|offset| {
                let seed = base_seed.wrapping_add(offset as u64);
                let (graph, vertex_labels, edges) = random_labeled_simple_case(seed);
                let result = catch_unwind(AssertUnwindSafe(|| {
                    exercise_canonizer_case_without_panic(&graph, &vertex_labels);
                    let (permuted_graph, permuted_labels) = permuted_random_case(
                        &graph,
                        &vertex_labels,
                        &edges,
                        seed ^ 0x9E37_79B9_7F4A_7C15,
                    );
                    exercise_canonizer_case_without_panic(&permuted_graph, &permuted_labels);
                }));

                result.err().map(|_| {
                    describe_random_case(seed, &vertex_labels, &edges, "canonizer panicked")
                })
            })
            .collect::<Vec<_>>()
    });
    failures.sort();
    failures.truncate(8);
    failures
}

#[test]
fn test_canonizer_is_relabeling_invariant_across_generic_benchmark_corpus() {
    for case in benchmark_cases() {
        let original_certificate = canonical_certificate(&case);
        let (permuted_graph, permuted_labels) = permuted_case(&case);
        let permuted_matrix = Edges::matrix(permuted_graph.edges());
        let permuted_certificate = canonical_label_labeled_simple_graph(
            &permuted_graph,
            |node| permuted_labels[node],
            |left, right| permuted_matrix.sparse_value_at(left, right).unwrap(),
        )
        .certificate;

        assert_eq!(
            original_certificate, permuted_certificate,
            "benchmark case {} lost relabeling invariance",
            case.name
        );
    }
}

#[test]
fn test_canonizer_is_relabeling_invariant_across_generic_scaling_corpus() {
    for case in scaling_cases() {
        let original_certificate = canonical_certificate(&case);
        let (permuted_graph, permuted_labels) = permuted_case(&case);
        let permuted_matrix = Edges::matrix(permuted_graph.edges());
        let permuted_certificate = canonical_label_labeled_simple_graph(
            &permuted_graph,
            |node| permuted_labels[node],
            |left, right| permuted_matrix.sparse_value_at(left, right).unwrap(),
        )
        .certificate;

        assert_eq!(
            original_certificate, permuted_certificate,
            "scaling case {} lost relabeling invariance",
            case.name
        );
    }
}

#[test]
fn test_default_heuristic_matches_explicit_bliss_fsm_on_generic_benchmark_corpus() {
    for case in benchmark_cases() {
        let matrix = Edges::matrix(case.graph.edges());
        let reference = canonical_label_labeled_simple_graph(
            &case.graph,
            |node| case.vertex_labels[node],
            |left, right| matrix.sparse_value_at(left, right).unwrap(),
        )
        .certificate;
        let explicit_fsm = canonical_label_labeled_simple_graph_with_options(
            &case.graph,
            |node| case.vertex_labels[node],
            |left, right| matrix.sparse_value_at(left, right).unwrap(),
            CanonicalLabelingOptions {
                splitting_heuristic: CanonSplittingHeuristic::FirstSmallestMaxNeighbours,
            },
        )
        .certificate;
        assert_eq!(
            reference, explicit_fsm,
            "default heuristic drifted from explicit fsm on generic benchmark case {}",
            case.name
        );
    }
}

#[test]
#[ignore = "random stress test for canonizer panics across many labeled generic graphs"]
fn test_canonizer_does_not_panic_on_random_generic_graphs() {
    let case_count = random_case_count();
    let base_seed = random_case_base_seed();
    let parallelism = random_case_parallelism();
    let failures = collect_random_case_panics(base_seed, case_count, parallelism);

    assert!(
        failures.is_empty(),
        "found {} canonizer crash cases out of {} random graphs (threads={}, base_seed={}):\n\n{}",
        failures.len(),
        case_count,
        parallelism,
        base_seed,
        failures.join("\n\n")
    );
}

#[test]
#[ignore = "parallel crash soak across multiple deterministic random seed windows"]
fn test_canonizer_does_not_panic_on_random_seed_windows() {
    let parallelism = 4;
    let windows = [(0xCA90_0000_0000_0001_u64, 64_usize), (0xCA90_0000_0000_1001_u64, 64_usize)];
    let window_count = 2_usize;

    let failures = windows
        .into_iter()
        .flat_map(|(base_seed, case_count)| {
            collect_random_case_panics(base_seed, case_count, parallelism).into_iter().map(
                move |failure| {
                    format!("window base_seed={base_seed} case_count={case_count}\n{failure}")
                },
            )
        })
        .collect::<Vec<_>>();

    assert!(
        failures.is_empty(),
        "found {} canonizer crash cases across {} deterministic windows (threads={}):\n\n{}",
        failures.len(),
        window_count,
        parallelism,
        failures.join("\n\n")
    );
}
