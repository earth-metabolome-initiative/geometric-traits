#![cfg(feature = "std")]
#![allow(dead_code)]

#[path = "fixture_io.rs"]
mod fixture_io;
#[path = "undigraph_fixture.rs"]
mod undigraph_fixture;

use geometric_traits::prelude::*;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct TopologicalValidityFixtureSuite {
    pub schema_version: u32,
    pub algorithm: String,
    pub graph_kind: String,
    pub generator: String,
    pub primary_oracle: String,
    pub family_sequence: Vec<String>,
    pub count: usize,
    pub cases: Vec<TopologicalValidityFixtureCase>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct TopologicalValidityFixtureCase {
    pub name: String,
    pub family: String,
    pub node_count: usize,
    pub edges: Vec<[usize; 2]>,
    pub is_planar: bool,
    pub is_outerplanar: bool,
    pub has_k23_homeomorph: bool,
    pub has_k33_homeomorph: bool,
    pub has_k4_homeomorph: bool,
    pub notes: String,
}

pub fn load_fixture_suite(relative_path: &str) -> TopologicalValidityFixtureSuite {
    fixture_io::load_fixture_json(relative_path)
}

pub fn build_undigraph(case: &TopologicalValidityFixtureCase) -> UndiGraph<usize> {
    undigraph_fixture::build_undigraph(case.node_count, &case.edges)
}
