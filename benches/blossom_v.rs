#![allow(clippy::pedantic)]

//! Criterion benchmark comparing the Rust Blossom V implementation against the
//! local C++ Blossom V reference on the same deterministic weighted instances.

use std::{
    collections::BTreeSet,
    hint::black_box,
    io::{BufRead, BufReader, BufWriter, Write},
    path::PathBuf,
    process::{Child, ChildStdin, ChildStdout, Command, Stdio},
    time::{Duration, Instant},
};

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use geometric_traits::{
    impls::ValuedCSR2D, prelude::*, traits::algorithms::blossom_v_unchecked_support_feasible,
};
use rand::{Rng, SeedableRng, rngs::SmallRng, seq::SliceRandom};

type Vcsr = ValuedCSR2D<usize, usize, usize, i32>;

#[derive(Clone)]
struct WeightedCase {
    name: String,
    edges: Vec<(usize, usize, i32)>,
    graph: Vcsr,
    json_line: String,
}

struct CppBatchSolver {
    child: Child,
    stdin: BufWriter<ChildStdin>,
    stdout: BufReader<ChildStdout>,
}

impl CppBatchSolver {
    fn new(binary: &PathBuf) -> Self {
        let mut child = Command::new(binary)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .unwrap_or_else(|e| {
                panic!("failed to start C++ Blossom V benchmark helper at {binary:?}: {e}")
            });

        let stdin = child.stdin.take().expect("child stdin missing");
        let stdout = child.stdout.take().expect("child stdout missing");

        Self { child, stdin: BufWriter::new(stdin), stdout: BufReader::new(stdout) }
    }

    fn solve_cost(&mut self, json_line: &str) -> i32 {
        writeln!(self.stdin, "{json_line}").expect("failed to write case to C++ solver");
        self.stdin.flush().expect("failed to flush C++ solver stdin");

        let mut line = String::new();
        self.stdout.read_line(&mut line).expect("failed to read C++ solver output");
        assert!(
            !line.trim().is_empty(),
            "C++ solver returned an empty output line for benchmark input"
        );

        parse_cost(&line).unwrap_or_else(|| panic!("failed to parse cost from C++ output: {line}"))
    }
}

impl Drop for CppBatchSolver {
    fn drop(&mut self) {
        let _ = self.stdin.flush();
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn parse_cost(line: &str) -> Option<i32> {
    let start = line.find("\"cost\":")?;
    let rest = &line[start + "\"cost\":".len()..];
    let end = rest.find([',', '}']).unwrap_or(rest.len());
    rest[..end].trim().parse().ok()
}

fn cpp_batch_solver_bin() -> Option<PathBuf> {
    if let Some(path) = std::env::var_os("BLOSSOM_V_CPP_BIN") {
        return Some(PathBuf::from(path));
    }

    for candidate in ["/tmp/blossom5-v2.05.src/batch_solve2", "/tmp/blossom5-v2.05.src/batch_solve"]
    {
        let path = PathBuf::from(candidate);
        if path.is_file() {
            return Some(path);
        }
    }

    None
}

fn build_valued_graph(n: usize, edges: &[(usize, usize, i32)]) -> Vcsr {
    let mut sorted_edges: Vec<(usize, usize, i32)> = Vec::new();
    for &(i, j, w) in edges {
        if i == j {
            continue;
        }
        let (lo, hi) = if i < j { (i, j) } else { (j, i) };
        sorted_edges.push((lo, hi, w));
        sorted_edges.push((hi, lo, w));
    }
    sorted_edges.sort_unstable();
    sorted_edges.dedup_by(|a, b| a.0 == b.0 && a.1 == b.1);

    let mut vcsr: Vcsr = SparseMatrixMut::with_sparse_shaped_capacity((n, n), sorted_edges.len());
    for (r, c, v) in sorted_edges {
        MatrixMut::add(&mut vcsr, (r, c, v)).unwrap();
    }
    vcsr
}

fn matching_cost(edges: &[(usize, usize, i32)], matching: &[(usize, usize)]) -> i32 {
    matching
        .iter()
        .map(|&(u, v)| {
            edges
                .iter()
                .find_map(|&(a, b, w)| ((a == u && b == v) || (a == v && b == u)).then_some(w))
                .unwrap_or_else(|| panic!("missing matching edge ({u},{v}) in benchmark graph"))
        })
        .sum()
}

fn graph_json_line(n: usize, edges: &[(usize, usize, i32)]) -> String {
    let mut line = String::new();
    line.push_str("{\"n\":");
    line.push_str(&n.to_string());
    line.push_str(",\"m\":");
    line.push_str(&edges.len().to_string());
    line.push_str(",\"edges\":[");
    for (idx, &(u, v, w)) in edges.iter().enumerate() {
        if idx > 0 {
            line.push(',');
        }
        line.push('[');
        line.push_str(&u.to_string());
        line.push(',');
        line.push_str(&v.to_string());
        line.push(',');
        line.push_str(&w.to_string());
        line.push(']');
    }
    line.push_str("]}");
    line
}

fn make_case(name: &str, seed: u64, n: usize, extra_edges: usize) -> WeightedCase {
    assert!(n >= 2 && n % 2 == 0);

    let mut rng = SmallRng::seed_from_u64(seed);
    let mut perm: Vec<usize> = (0..n).collect();
    perm.shuffle(&mut rng);

    let mut edges_set: BTreeSet<(usize, usize)> = BTreeSet::new();
    let mut weighted_edges = Vec::new();

    for pair in perm.chunks_exact(2) {
        let u = pair[0].min(pair[1]);
        let v = pair[0].max(pair[1]);
        edges_set.insert((u, v));
        let w = rng.gen_range(-250..=-50);
        weighted_edges.push((u, v, w));
    }

    while weighted_edges.len() < n / 2 + extra_edges {
        let mut u = rng.gen_range(0..n);
        let mut v = rng.gen_range(0..n);
        if u == v {
            continue;
        }
        if u > v {
            std::mem::swap(&mut u, &mut v);
        }
        if !edges_set.insert((u, v)) {
            continue;
        }
        let w = rng.gen_range(-100..=100);
        weighted_edges.push((u, v, w));
    }

    let graph = build_valued_graph(n, &weighted_edges);
    let json_line = graph_json_line(n, &weighted_edges);

    WeightedCase { name: name.to_string(), edges: weighted_edges, graph, json_line }
}

fn benchmark_cases() -> Vec<WeightedCase> {
    vec![
        make_case("n64_sparse", 0x00B1_0550_0001, 64, 96),
        make_case("n128_sparse", 0x00B1_0550_0002, 128, 320),
        make_case("n128_dense", 0x00B1_0550_0003, 128, 1400),
    ]
}

fn verify_case_alignment(case: &WeightedCase, cpp_bin: &PathBuf) {
    let rust_matching = case
        .graph
        .blossom_v()
        .unwrap_or_else(|e| panic!("Rust Blossom V failed on benchmark case {}: {e:?}", case.name));
    let rust_cost = matching_cost(&case.edges, &rust_matching);
    let rust_unchecked_matching =
        blossom_v_unchecked_support_feasible(&case.graph).unwrap_or_else(|e| {
            panic!("Rust unchecked Blossom V failed on benchmark case {}: {e:?}", case.name)
        });
    let rust_unchecked_cost = matching_cost(&case.edges, &rust_unchecked_matching);

    let mut cpp = CppBatchSolver::new(cpp_bin);
    let cpp_cost = cpp.solve_cost(&case.json_line);

    assert_eq!(
        rust_cost, cpp_cost,
        "Rust/C++ Blossom V cost mismatch on benchmark case {}",
        case.name
    );
    assert_eq!(
        rust_cost, rust_unchecked_cost,
        "Rust checked/unchecked Blossom V cost mismatch on benchmark case {}",
        case.name
    );
}

fn bench_blossom_v(c: &mut Criterion) {
    let cases = benchmark_cases();
    let cpp_bin = cpp_batch_solver_bin();

    if let Some(ref bin) = cpp_bin {
        for case in &cases {
            verify_case_alignment(case, bin);
        }
    } else {
        eprintln!(
            "Skipping C++ Blossom V benchmark: set BLOSSOM_V_CPP_BIN or provide /tmp/blossom5-v2.05.src/batch_solve2"
        );
    }

    let mut group = c.benchmark_group("blossom_v");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(4));

    for case in &cases {
        group.bench_with_input(BenchmarkId::new("Rust", &case.name), case, |b, case| {
            b.iter(|| {
                let matching = case.graph.blossom_v().expect("benchmark case should solve");
                black_box(matching);
            });
        });

        group.bench_with_input(BenchmarkId::new("RustNoPrecheck", &case.name), case, |b, case| {
            b.iter(|| {
                let matching = blossom_v_unchecked_support_feasible(&case.graph)
                    .expect("benchmark case should solve");
                black_box(matching);
            });
        });

        if let Some(ref bin) = cpp_bin {
            group.bench_with_input(BenchmarkId::new("CppBatch", &case.name), case, |b, case| {
                let bin = bin.clone();
                b.iter_custom(|iters| {
                    let mut solver = CppBatchSolver::new(&bin);
                    let start = Instant::now();
                    for _ in 0..iters {
                        let cost = solver.solve_cost(&case.json_line);
                        black_box(cost);
                    }
                    start.elapsed()
                });
            });
        }
    }

    group.finish();
}

criterion_group!(benches, bench_blossom_v);
criterion_main!(benches);
