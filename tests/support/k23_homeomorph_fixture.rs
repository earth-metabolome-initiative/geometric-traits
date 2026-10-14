#![cfg(feature = "std")]
#![allow(dead_code)]

#[path = "fixture_io.rs"]
mod fixture_io;
#[path = "undigraph_fixture.rs"]
mod undigraph_fixture;

use geometric_traits::prelude::*;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct K23HomeomorphFixtureSuite {
    pub schema_version: u32,
    pub algorithm: String,
    pub graph_kind: String,
    pub generator: String,
    pub primary_oracle: String,
    pub family_sequence: Vec<String>,
    pub cases: Vec<K23HomeomorphFixtureCase>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct K23HomeomorphFixtureCase {
    pub name: String,
    pub family: String,
    pub node_count: usize,
    pub edges: Vec<[usize; 2]>,
    pub has_k23_homeomorph: bool,
    pub notes: String,
}

pub fn load_fixture_suite(relative_path: &str) -> K23HomeomorphFixtureSuite {
    fixture_io::load_fixture_json(relative_path)
}

pub fn build_undigraph(case: &K23HomeomorphFixtureCase) -> UndiGraph<usize> {
    undigraph_fixture::build_undigraph(case.node_count, &case.edges)
}
