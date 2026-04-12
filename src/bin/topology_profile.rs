//! Fixed-scenario topology profiling harness for callgrind-style analysis.

use std::{env, hint::black_box, process::ExitCode};

use geometric_traits::{
    impls::{CSR2D, SortedVec, SymmetricCSR2D},
    prelude::*,
    traits::{
        K4HomeomorphDetection, K33HomeomorphDetection, OuterplanarityDetection, PlanarityDetection,
        SquareMatrix, VocabularyBuilder,
        algorithms::randomized_graphs::{complete_bipartite_graph, path_graph},
    },
};

type UndirectedAdjacency = SymmetricCSR2D<CSR2D<usize, usize, usize>>;

#[derive(Clone, Copy)]
enum Scenario {
    PlanarityPath4096,
    OuterplanarityPath4096,
    K33Path4096,
    K33CompleteBipartiteK3_2048,
    K33FuzzerRegression20260412,
    K4FuzzerRegression20260411,
}

fn usage() -> &'static str {
    "Usage: cargo run --release --bin topology_profile -- <scenario> [iterations]\n\
     Scenarios:\n\
       planarity_path_4096\n\
       outerplanarity_path_4096\n\
       k33_path_4096\n\
       k33_complete_bipartite_k3_2048\n\
       k33_fuzzer_regression_20260412\n\
       k4_fuzzer_regression_20260411"
}

fn parse_scenario(name: &str) -> Option<Scenario> {
    match name {
        "planarity_path_4096" => Some(Scenario::PlanarityPath4096),
        "outerplanarity_path_4096" => Some(Scenario::OuterplanarityPath4096),
        "k33_path_4096" => Some(Scenario::K33Path4096),
        "k33_complete_bipartite_k3_2048" => Some(Scenario::K33CompleteBipartiteK3_2048),
        "k33_fuzzer_regression_20260412" => Some(Scenario::K33FuzzerRegression20260412),
        "k4_fuzzer_regression_20260411" => Some(Scenario::K4FuzzerRegression20260411),
        _ => None,
    }
}

fn wrap_undi(graph: UndirectedAdjacency) -> UndiGraph<usize> {
    let order = graph.order();
    let nodes: SortedVec<usize> = GenericVocabularyBuilder::default()
        .expected_number_of_symbols(order)
        .symbols((0..order).enumerate())
        .build()
        .unwrap();
    UndiGraph::from((nodes, graph))
}

fn normalize_edge([left, right]: [usize; 2]) -> [usize; 2] {
    if left <= right { [left, right] } else { [right, left] }
}

fn build_undigraph(node_count: usize, edges: &[[usize; 2]]) -> UndiGraph<usize> {
    let nodes: SortedVec<usize> = GenericVocabularyBuilder::default()
        .expected_number_of_symbols(node_count)
        .symbols((0..node_count).enumerate())
        .build()
        .unwrap();
    let mut normalized_edges: Vec<(usize, usize)> = edges
        .iter()
        .copied()
        .map(normalize_edge)
        .map(|[source, destination]| (source, destination))
        .collect();
    normalized_edges.sort_unstable();
    let matrix: UndirectedAdjacency = UndiEdgesBuilder::default()
        .expected_number_of_edges(normalized_edges.len())
        .expected_shape(node_count)
        .edges(normalized_edges.into_iter())
        .build()
        .unwrap();
    UndiGraph::from((nodes, matrix))
}

fn run_scenario(scenario: Scenario, iterations: usize) -> u64 {
    match scenario {
        Scenario::PlanarityPath4096 => {
            let graph = wrap_undi(path_graph(4_096));
            let mut checksum = 0u64;
            for _ in 0..iterations {
                checksum = checksum.wrapping_add(u64::from(graph.is_planar().unwrap()));
            }
            checksum
        }
        Scenario::OuterplanarityPath4096 => {
            let graph = wrap_undi(path_graph(4_096));
            let mut checksum = 0u64;
            for _ in 0..iterations {
                checksum = checksum.wrapping_add(u64::from(graph.is_outerplanar().unwrap()));
            }
            checksum
        }
        Scenario::K33Path4096 => {
            let graph = wrap_undi(path_graph(4_096));
            let mut checksum = 0u64;
            for _ in 0..iterations {
                checksum = checksum.wrapping_add(u64::from(graph.has_k33_homeomorph().unwrap()));
            }
            checksum
        }
        Scenario::K33CompleteBipartiteK3_2048 => {
            let graph = wrap_undi(complete_bipartite_graph(3, 2_048));
            let mut checksum = 0u64;
            for _ in 0..iterations {
                checksum = checksum.wrapping_add(u64::from(graph.has_k33_homeomorph().unwrap()));
            }
            checksum
        }
        Scenario::K33FuzzerRegression20260412 => {
            let graph = build_undigraph(
                16,
                &[
                    [0, 1],
                    [0, 7],
                    [0, 11],
                    [0, 15],
                    [1, 5],
                    [1, 11],
                    [2, 4],
                    [3, 5],
                    [3, 10],
                    [3, 15],
                    [4, 5],
                    [4, 12],
                    [4, 13],
                    [5, 6],
                    [5, 7],
                    [5, 10],
                    [5, 11],
                    [5, 12],
                    [5, 15],
                    [7, 8],
                    [7, 11],
                    [7, 15],
                    [8, 9],
                    [9, 15],
                    [11, 12],
                    [11, 15],
                    [14, 15],
                ],
            );
            let mut checksum = 0u64;
            for _ in 0..iterations {
                checksum = checksum.wrapping_add(u64::from(graph.has_k33_homeomorph().unwrap()));
            }
            checksum
        }
        Scenario::K4FuzzerRegression20260411 => {
            let graph = build_undigraph(
                15,
                &[
                    [0, 3],
                    [0, 10],
                    [0, 11],
                    [3, 10],
                    [3, 11],
                    [3, 12],
                    [3, 13],
                    [4, 12],
                    [4, 14],
                    [7, 12],
                    [7, 14],
                    [9, 10],
                    [9, 13],
                    [9, 14],
                    [10, 11],
                ],
            );
            let mut checksum = 0u64;
            for _ in 0..iterations {
                checksum = checksum.wrapping_add(u64::from(graph.has_k4_homeomorph().unwrap()));
            }
            checksum
        }
    }
}

fn main() -> ExitCode {
    let mut args = env::args().skip(1);
    let Some(scenario_name) = args.next() else {
        eprintln!("{}", usage());
        return ExitCode::from(2);
    };
    let Some(scenario) = parse_scenario(&scenario_name) else {
        eprintln!("Unknown scenario: {scenario_name}\n\n{}", usage());
        return ExitCode::from(2);
    };
    let iterations = args
        .next()
        .map(|value| value.parse::<usize>())
        .transpose()
        .unwrap_or_else(|_| {
            eprintln!("Iterations must be a positive integer.");
            std::process::exit(2);
        })
        .unwrap_or(1);

    let checksum = run_scenario(scenario, iterations);
    println!("scenario={scenario_name} iterations={iterations} checksum={}", black_box(checksum));
    ExitCode::SUCCESS
}
