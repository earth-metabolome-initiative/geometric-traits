#![cfg(feature = "std")]
#![allow(dead_code)]

#[path = "fixture_io.rs"]
mod fixture_io;
#[path = "undigraph_fixture.rs"]
mod undigraph_fixture;

use geometric_traits::prelude::*;
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

pub fn load_fixture_suite(relative_path: &str) -> ChordalFixtureSuite {
    fixture_io::load_fixture_json(relative_path)
}

pub fn normalize_edge([left, right]: [usize; 2]) -> [usize; 2] {
    undigraph_fixture::normalize_edge([left, right])
}

pub fn build_undigraph(case: &ChordalFixtureCase) -> UndiGraph<usize> {
    undigraph_fixture::build_undigraph(case.node_count, &case.edges)
}
