#![cfg(feature = "std")]
#![allow(dead_code)]

#[path = "fixture_io.rs"]
mod fixture_io;
#[path = "undigraph_fixture.rs"]
mod undigraph_fixture;

use geometric_traits::prelude::*;
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
    pub local_clustering_rounding_decimals: u32,
    pub power_iteration_eigenvector_rounding_decimals: u32,
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
    pub welsh_powell_descending: Vec<usize>,
    pub dsatur_order: Vec<usize>,
    pub bfs_from_max_degree: Vec<usize>,
    pub dfs_from_max_degree: Vec<usize>,
    pub power_iteration_eigenvector_max_iter: usize,
    pub power_iteration_eigenvector_tol: f64,
    pub power_iteration_eigenvector_scores: Vec<f64>,
    pub power_iteration_eigenvector_descending: Vec<usize>,
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
    pub triangle_counts: Vec<usize>,
    pub triangle_descending: Vec<usize>,
    pub local_clustering_scores: Vec<f64>,
    pub local_clustering_descending: Vec<usize>,
}

#[derive(Clone)]
pub struct PreparedNodeOrderingCase {
    pub name: String,
    pub family: String,
    pub graph: UndiGraph<usize>,
    pub canonical_smallest_last: Vec<usize>,
    pub core_numbers: Vec<usize>,
    pub degeneracy_degree_descending: Vec<usize>,
    pub welsh_powell_descending: Vec<usize>,
    pub dsatur_order: Vec<usize>,
    pub bfs_from_max_degree: Vec<usize>,
    pub dfs_from_max_degree: Vec<usize>,
    pub power_iteration_eigenvector_max_iter: usize,
    pub power_iteration_eigenvector_tol: f64,
    pub power_iteration_eigenvector_scores: Vec<f64>,
    pub power_iteration_eigenvector_descending: Vec<usize>,
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
    pub triangle_counts: Vec<usize>,
    pub triangle_descending: Vec<usize>,
    pub local_clustering_scores: Vec<f64>,
    pub local_clustering_descending: Vec<usize>,
}

pub fn load_fixture_suite(relative_path: &str) -> NodeOrderingGroundTruthFixture {
    fixture_io::load_fixture_json(relative_path)
}

pub fn build_undigraph(case: &NodeOrderingGroundTruthCase) -> UndiGraph<usize> {
    undigraph_fixture::build_undigraph(case.n, &case.edges)
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
                welsh_powell_descending: case.welsh_powell_descending,
                dsatur_order: case.dsatur_order,
                bfs_from_max_degree: case.bfs_from_max_degree,
                dfs_from_max_degree: case.dfs_from_max_degree,
                power_iteration_eigenvector_max_iter: case.power_iteration_eigenvector_max_iter,
                power_iteration_eigenvector_tol: case.power_iteration_eigenvector_tol,
                power_iteration_eigenvector_scores: case.power_iteration_eigenvector_scores,
                power_iteration_eigenvector_descending: case.power_iteration_eigenvector_descending,
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
                triangle_counts: case.triangle_counts,
                triangle_descending: case.triangle_descending,
                local_clustering_scores: case.local_clustering_scores,
                local_clustering_descending: case.local_clustering_descending,
            }
        })
        .collect()
}
