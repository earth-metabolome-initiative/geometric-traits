//! Criterion benchmarks for the current simple-undirected labeled canonizer,
//! with an optional local `bliss` CLI comparison path.

#[path = "../tests/support/bliss_oracle.rs"]
#[allow(dead_code)]
mod bliss_oracle;
#[path = "../tests/support/canon_bench_fixture.rs"]
mod canon_bench_fixture;

use std::{
    fs,
    hint::black_box,
    path::{Path, PathBuf},
    process::{Command, Stdio},
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use canon_bench_fixture::{CanonCase, benchmark_cases, scaling_cases, timeout_cases};
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use geometric_traits::{
    prelude::*,
    traits::{
        CanonSplittingHeuristic, CanonicalLabelingOptions, Edges, SparseValuedMatrix2D,
        canonical_label_labeled_simple_graph, canonical_label_labeled_simple_graph_with_options,
    },
};

#[derive(Clone)]
struct PreparedBlissCliCase {
    name: String,
    input_path: PathBuf,
    number_of_nodes: usize,
    number_of_edges: usize,
}

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

fn run_bliss_cli(bliss: &Path, input_path: &Path) {
    let status = Command::new(bliss)
        .arg("-can")
        .arg("-v=1")
        .arg("-sh=fsm")
        .arg("-fr=y")
        .arg("-cr=y")
        .arg(input_path)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .expect("bliss benchmark invocation should start");
    assert!(status.success(), "bliss benchmark invocation should succeed");
    black_box(status);
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

fn bench_bliss_group(c: &mut Criterion, group_name: &str, cases: &[CanonCase]) {
    let Some(bliss) = bliss_oracle::locate_bliss_binary() else {
        return;
    };

    let (temp_root, prepared_cases) =
        prepare_bliss_cli_cases(cases).expect("bliss cases should prepare");
    for case in &prepared_cases {
        run_bliss_cli(&bliss, &case.input_path);
    }

    let mut group = c.benchmark_group(group_name);
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(2));

    for case in &prepared_cases {
        group.throughput(Throughput::Elements(
            u64::try_from(case.number_of_nodes).expect("node count should fit into u64"),
        ));
        let parameter =
            format!("{}__n{}_m{}", case.name, case.number_of_nodes, case.number_of_edges);
        group.bench_with_input(BenchmarkId::new("bliss_cli", parameter), case, |b, case| {
            b.iter(|| run_bliss_cli(&bliss, &case.input_path));
        });
    }

    group.finish();
    let _ = fs::remove_dir_all(temp_root);
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

fn prepare_bliss_cli_cases(
    cases: &[CanonCase],
) -> Result<(PathBuf, Vec<PreparedBlissCliCase>), String> {
    let temp_root = make_temp_dir()?;
    let mut prepared = Vec::with_capacity(cases.len());

    for case in cases {
        let encoded =
            bliss_oracle::encode_labeled_simple_graph_as_dimacs(&case.vertex_labels, &case.edges)?;
        let file_stem = sanitize_case_name(&case.name);
        let input_path = temp_root.join(format!("{file_stem}.dimacs"));
        fs::write(&input_path, encoded.dimacs).map_err(|error| error.to_string())?;
        prepared.push(PreparedBlissCliCase {
            name: case.name.clone(),
            input_path,
            number_of_nodes: case.number_of_nodes(),
            number_of_edges: case.number_of_edges(),
        });
    }

    Ok((temp_root, prepared))
}

fn make_temp_dir() -> Result<PathBuf, String> {
    let nanos =
        SystemTime::now().duration_since(UNIX_EPOCH).map_err(|error| error.to_string())?.as_nanos();
    let path = std::env::temp_dir()
        .join(format!("geometric-traits-canon-bench-{nanos}-{}", std::process::id()));
    fs::create_dir_all(&path).map_err(|error| error.to_string())?;
    Ok(path)
}

fn sanitize_case_name(name: &str) -> String {
    name.chars().map(|ch| if ch.is_ascii_alphanumeric() { ch } else { '_' }).collect()
}

fn bench_canon_cases(c: &mut Criterion) {
    let cases = benchmark_cases();
    bench_rust_group(c, "canon_cases_rust", &cases);
    bench_bliss_group(c, "canon_cases_bliss_cli", &cases);
}

fn bench_canon_scaling(c: &mut Criterion) {
    let cases = scaling_cases();
    bench_rust_group(c, "canon_scaling_rust", &cases);
    bench_bliss_group(c, "canon_scaling_bliss_cli", &cases);
}

fn bench_canon_heuristics(c: &mut Criterion) {
    let cases = scaling_cases();
    bench_rust_heuristics_group(c, "canon_scaling_rust_heuristics", &cases);
}

fn bench_canon_timeout_cases(c: &mut Criterion) {
    let cases = timeout_cases();
    bench_rust_group_with_config(c, "canon_timeout_cases_rust", &cases, 10, Duration::from_secs(8));
    bench_bliss_group(c, "canon_timeout_cases_bliss_cli", &cases);
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
