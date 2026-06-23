//! Loader for the NetworkX minimum-spanning-tree ground-truth corpus.
#![cfg(feature = "std")]
#![allow(dead_code)]

#[path = "fixture_io.rs"]
mod fixture_io;

use geometric_traits::{
    impls::ValuedCSR2D, naive_structs::GenericEdgesBuilder, traits::EdgesBuilder,
};
use serde::Deserialize;

/// Weighted adjacency matrix used to feed the minimum-spanning-tree traits.
pub type WeightedMatrix = ValuedCSR2D<usize, usize, usize, f64>;

#[derive(Debug, Deserialize)]
pub struct MinimumSpanningTreeFixtureSuite {
    pub schema_version: u32,
    pub algorithm: String,
    pub graph_kind: String,
    pub generator: String,
    pub primary_oracle: String,
    pub networkx_version: String,
    pub cases: Vec<MinimumSpanningTreeFixtureCase>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct MinimumSpanningTreeFixtureCase {
    pub name: String,
    pub family: String,
    pub node_count: usize,
    pub edges: Vec<(usize, usize, f64)>,
    pub mst_total_weight: f64,
    pub mst_edge_count: usize,
    pub number_of_components: usize,
    pub weights_distinct: bool,
    pub mst_edges: Vec<(usize, usize, f64)>,
    pub notes: String,
}

pub fn load_fixture_suite(relative_path: &str) -> MinimumSpanningTreeFixtureSuite {
    fixture_io::load_fixture_json(relative_path)
}

/// Builds a symmetric weighted matrix from a fixture case, materializing both
/// directions of every undirected edge.
pub fn build_weighted_matrix(case: &MinimumSpanningTreeFixtureCase) -> WeightedMatrix {
    let mut directed = Vec::with_capacity(case.edges.len() * 2);
    for &(source, destination, weight) in &case.edges {
        directed.push((source, destination, weight));
        if source != destination {
            directed.push((destination, source, weight));
        }
    }
    directed.sort_unstable_by(|(ls, ld, _), (rs, rd, _)| (ls, ld).cmp(&(rs, rd)));

    GenericEdgesBuilder::<_, WeightedMatrix>::default()
        .expected_number_of_edges(directed.len())
        .expected_shape((case.node_count, case.node_count))
        .edges(directed.into_iter())
        .build()
        .unwrap()
}
