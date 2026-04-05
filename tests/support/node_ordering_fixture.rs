#![cfg(feature = "std")]
#![allow(dead_code)]

use std::{fs, io::Read as _, path::Path};

use flate2::read::GzDecoder;
use geometric_traits::{
    impls::{CSR2D, SortedVec, SymmetricCSR2D},
    prelude::*,
    traits::{EdgesBuilder, VocabularyBuilder},
};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct NodeOrderingGroundTruthFixture {
    pub schema_version: u32,
    pub generator: String,
    pub networkx_version: String,
    pub python_version: String,
    pub reference_ordering: String,
    pub tie_break: String,
    pub pagerank_rounding_decimals: u32,
    pub katz_rounding_decimals: u32,
    pub betweenness_rounding_decimals: u32,
    pub closeness_rounding_decimals: u32,
    pub cases: Vec<NodeOrderingGroundTruthCase>,
}

#[derive(Debug, Deserialize)]
pub struct NodeOrderingGroundTruthCase {
    pub name: String,
    pub family: String,
    pub n: usize,
    pub edges: Vec<[usize; 2]>,
    pub networkx_smallest_last: Vec<usize>,
    pub canonical_smallest_last: Vec<usize>,
    pub core_numbers: Vec<usize>,
    pub degeneracy_degree_descending: Vec<usize>,
    pub pagerank_alpha: f64,
    pub pagerank_max_iter: usize,
    pub pagerank_tol: f64,
    pub pagerank_scores: Vec<f64>,
    pub pagerank_descending: Vec<usize>,
    pub katz_alpha: f64,
    pub katz_beta: f64,
    pub katz_max_iter: usize,
    pub katz_tol: f64,
    pub katz_normalized: bool,
    pub katz_scores: Vec<f64>,
    pub katz_descending: Vec<usize>,
    pub betweenness_normalized: bool,
    pub betweenness_endpoints: bool,
    pub betweenness_scores: Vec<f64>,
    pub betweenness_descending: Vec<usize>,
    pub closeness_wf_improved: bool,
    pub closeness_scores: Vec<f64>,
    pub closeness_descending: Vec<usize>,
}

#[derive(Clone)]
pub struct PreparedNodeOrderingCase {
    pub name: String,
    pub family: String,
    pub graph: UndiGraph<usize>,
    pub canonical_smallest_last: Vec<usize>,
    pub core_numbers: Vec<usize>,
    pub degeneracy_degree_descending: Vec<usize>,
    pub pagerank_alpha: f64,
    pub pagerank_max_iter: usize,
    pub pagerank_tol: f64,
    pub pagerank_scores: Vec<f64>,
    pub pagerank_descending: Vec<usize>,
    pub katz_alpha: f64,
    pub katz_beta: f64,
    pub katz_max_iter: usize,
    pub katz_tol: f64,
    pub katz_normalized: bool,
    pub katz_scores: Vec<f64>,
    pub katz_descending: Vec<usize>,
    pub betweenness_normalized: bool,
    pub betweenness_endpoints: bool,
    pub betweenness_scores: Vec<f64>,
    pub betweenness_descending: Vec<usize>,
    pub closeness_wf_improved: bool,
    pub closeness_scores: Vec<f64>,
    pub closeness_descending: Vec<usize>,
}

pub fn fixture_path(relative_path: &str) -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(relative_path)
}

pub fn load_fixture_suite(relative_path: &str) -> NodeOrderingGroundTruthFixture {
    let fixture_gz = fs::read(fixture_path(relative_path))
        .unwrap_or_else(|_| panic!("failed to read tests/fixtures/{relative_path}"));
    let mut json = String::new();
    GzDecoder::new(fixture_gz.as_slice())
        .read_to_string(&mut json)
        .expect("gzip decompression failed");
    serde_json::from_str(&json)
        .unwrap_or_else(|_| panic!("`tests/fixtures/{relative_path}` must contain valid JSON"))
}

pub fn build_undigraph(case: &NodeOrderingGroundTruthCase) -> UndiGraph<usize> {
    let nodes: SortedVec<usize> = GenericVocabularyBuilder::default()
        .expected_number_of_symbols(case.n)
        .symbols((0..case.n).enumerate())
        .build()
        .unwrap();
    let mut edges: Vec<(usize, usize)> =
        case.edges.iter().map(|&[source, destination]| (source, destination)).collect();
    edges.sort_unstable();
    let matrix: SymmetricCSR2D<CSR2D<usize, usize, usize>> = UndiEdgesBuilder::default()
        .expected_number_of_edges(edges.len())
        .expected_shape(case.n)
        .edges(edges.into_iter())
        .build()
        .unwrap();
    UndiGraph::from((nodes, matrix))
}

pub fn prepare_cases(relative_path: &str) -> Vec<PreparedNodeOrderingCase> {
    load_fixture_suite(relative_path)
        .cases
        .into_iter()
        .map(|case| {
            let graph = build_undigraph(&case);
            PreparedNodeOrderingCase {
                name: case.name,
                family: case.family,
                graph,
                canonical_smallest_last: case.canonical_smallest_last,
                core_numbers: case.core_numbers,
                degeneracy_degree_descending: case.degeneracy_degree_descending,
                pagerank_alpha: case.pagerank_alpha,
                pagerank_max_iter: case.pagerank_max_iter,
                pagerank_tol: case.pagerank_tol,
                pagerank_scores: case.pagerank_scores,
                pagerank_descending: case.pagerank_descending,
                katz_alpha: case.katz_alpha,
                katz_beta: case.katz_beta,
                katz_max_iter: case.katz_max_iter,
                katz_tol: case.katz_tol,
                katz_normalized: case.katz_normalized,
                katz_scores: case.katz_scores,
                katz_descending: case.katz_descending,
                betweenness_normalized: case.betweenness_normalized,
                betweenness_endpoints: case.betweenness_endpoints,
                betweenness_scores: case.betweenness_scores,
                betweenness_descending: case.betweenness_descending,
                closeness_wf_improved: case.closeness_wf_improved,
                closeness_scores: case.closeness_scores,
                closeness_descending: case.closeness_descending,
            }
        })
        .collect()
}
