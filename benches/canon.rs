//! Criterion benchmarks for the current simple-undirected labeled canonizer.

#[path = "../tests/support/canon_bench_fixture.rs"]
mod canon_bench_fixture;

use std::{hint::black_box, time::Duration};

use canon_bench_fixture::{CanonCase, benchmark_cases, scaling_cases, timeout_cases};
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use geometric_traits::{
    prelude::*,
    traits::{
        CanonSplittingHeuristic, CanonicalLabelingOptions, Edges, SparseValuedMatrix2D,
        canonical_label_labeled_simple_graph, canonical_label_labeled_simple_graph_with_options,
    },
};

fn run_rust_case(case: &CanonCase) {
    run_rust_case_with_heuristic(case, CanonSplittingHeuristic::FirstSmallest);
}

fn run_rust_case_with_heuristic(case: &CanonCase, splitting_heuristic: CanonSplittingHeuristic) {
    let matrix = Edges::matrix(case.graph.edges());
    let result = if splitting_heuristic == CanonSplittingHeuristic::FirstSmallest {
        canonical_label_labeled_simple_graph(
            &case.graph,
            |node| case.vertex_labels[node],
            |left, right| matrix.sparse_value_at(left, right).unwrap(),
        )
    } else {
        canonical_label_labeled_simple_graph_with_options(
            &case.graph,
            |node| case.vertex_labels[node],
            |left, right| matrix.sparse_value_at(left, right).unwrap(),
            CanonicalLabelingOptions { splitting_heuristic },
        )
    };
    assert_eq!(result.order.len(), case.number_of_nodes());
    black_box(result);
}

fn bench_rust_group(c: &mut Criterion, group_name: &str, cases: &[CanonCase]) {
    bench_rust_group_with_config(c, group_name, cases, 10, Duration::from_secs(2));
}

fn bench_rust_group_with_config(
    c: &mut Criterion,
    group_name: &str,
    cases: &[CanonCase],
    sample_size: usize,
    measurement_time: Duration,
) {
    for case in cases {
        run_rust_case(case);
    }

    let mut group = c.benchmark_group(group_name);
    group.sample_size(sample_size);
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(measurement_time);

    for case in cases {
        group.throughput(Throughput::Elements(
            u64::try_from(case.number_of_nodes()).expect("node count should fit into u64"),
        ));
        let parameter =
            format!("{}__n{}_m{}", case.name, case.number_of_nodes(), case.number_of_edges());
        group.bench_with_input(BenchmarkId::new("rust_ir", parameter), case, |b, case| {
            b.iter(|| run_rust_case(case));
        });
    }

    group.finish();
}

fn bench_rust_heuristics_group(c: &mut Criterion, group_name: &str, cases: &[CanonCase]) {
    bench_rust_heuristics_group_with_config(c, group_name, cases, 10, Duration::from_secs(2));
}

fn bench_rust_heuristics_group_with_config(
    c: &mut Criterion,
    group_name: &str,
    cases: &[CanonCase],
    sample_size: usize,
    measurement_time: Duration,
) {
    let heuristics = [
        ("f", CanonSplittingHeuristic::First),
        ("fs", CanonSplittingHeuristic::FirstSmallest),
        ("fl", CanonSplittingHeuristic::FirstLargest),
        ("fm", CanonSplittingHeuristic::FirstMaxNeighbours),
        ("fsm", CanonSplittingHeuristic::FirstSmallestMaxNeighbours),
        ("flm", CanonSplittingHeuristic::FirstLargestMaxNeighbours),
    ];

    for &(_, heuristic) in &heuristics {
        for case in cases {
            run_rust_case_with_heuristic(case, heuristic);
        }
    }

    let mut group = c.benchmark_group(group_name);
    group.sample_size(sample_size);
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(measurement_time);

    for &(heuristic_name, heuristic) in &heuristics {
        for case in cases {
            group.throughput(Throughput::Elements(
                u64::try_from(case.number_of_nodes()).expect("node count should fit into u64"),
            ));
            let parameter = format!(
                "{}__{}__n{}_m{}",
                heuristic_name,
                case.name,
                case.number_of_nodes(),
                case.number_of_edges()
            );
            group.bench_with_input(
                BenchmarkId::new("rust_ir_heuristic", parameter),
                case,
                |b, case| {
                    b.iter(|| run_rust_case_with_heuristic(case, heuristic));
                },
            );
        }
    }

    group.finish();
}

fn bench_canon_cases(c: &mut Criterion) {
    let cases = benchmark_cases();
    bench_rust_group(c, "canon_cases_rust", &cases);
}

fn bench_canon_scaling(c: &mut Criterion) {
    let cases = scaling_cases();
    bench_rust_group(c, "canon_scaling_rust", &cases);
}

fn bench_canon_heuristics(c: &mut Criterion) {
    let cases = scaling_cases();
    bench_rust_heuristics_group(c, "canon_scaling_rust_heuristics", &cases);
}

fn bench_canon_timeout_cases(c: &mut Criterion) {
    let cases = timeout_cases();
    bench_rust_group_with_config(c, "canon_timeout_cases_rust", &cases, 10, Duration::from_secs(8));
}

fn bench_canon_timeout_heuristics(c: &mut Criterion) {
    let cases = timeout_cases();
    bench_rust_heuristics_group_with_config(
        c,
        "canon_timeout_cases_rust_heuristics",
        &cases,
        10,
        Duration::from_secs(8),
    );
}

criterion_group!(
    benches,
    bench_canon_cases,
    bench_canon_scaling,
    bench_canon_heuristics,
    bench_canon_timeout_cases,
    bench_canon_timeout_heuristics
);
criterion_main!(benches);
