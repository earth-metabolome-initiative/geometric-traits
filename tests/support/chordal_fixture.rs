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
pub struct ChordalFixtureSuite {
    pub schema_version: u32,
    pub algorithm: String,
    pub graph_kind: String,
    pub generator: String,
    pub networkx_version: String,
    pub python_version: String,
    pub seed: u64,
    pub cases: Vec<ChordalFixtureCase>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct ChordalFixtureCase {
    pub name: String,
    pub family: String,
    pub node_count: usize,
    pub edges: Vec<[usize; 2]>,
    pub is_chordal: bool,
}

pub fn fixture_path(relative_path: &str) -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(relative_path)
}

pub fn load_fixture_suite(relative_path: &str) -> ChordalFixtureSuite {
    let fixture_gz = fs::read(fixture_path(relative_path))
        .unwrap_or_else(|_| panic!("failed to read tests/fixtures/{relative_path}"));
    let mut json = String::new();
    GzDecoder::new(fixture_gz.as_slice())
        .read_to_string(&mut json)
        .expect("gzip decompression failed");
    serde_json::from_str(&json)
        .unwrap_or_else(|_| panic!("`tests/fixtures/{relative_path}` must contain valid JSON"))
}

pub fn normalize_edge([left, right]: [usize; 2]) -> [usize; 2] {
    if left <= right { [left, right] } else { [right, left] }
}

pub fn build_undigraph(case: &ChordalFixtureCase) -> UndiGraph<usize> {
    let nodes: SortedVec<usize> = GenericVocabularyBuilder::default()
        .expected_number_of_symbols(case.node_count)
        .symbols((0..case.node_count).enumerate())
        .build()
        .unwrap();
    let mut edges: Vec<(usize, usize)> = case
        .edges
        .iter()
        .copied()
        .map(normalize_edge)
        .map(|[source, destination]| (source, destination))
        .collect();
    edges.sort_unstable();
    let matrix: SymmetricCSR2D<CSR2D<usize, usize, usize>> = UndiEdgesBuilder::default()
        .expected_number_of_edges(edges.len())
        .expected_shape(case.node_count)
        .edges(edges.into_iter())
        .build()
        .unwrap();
    UndiGraph::from((nodes, matrix))
}
