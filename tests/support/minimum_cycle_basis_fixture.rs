#![cfg(feature = "std")]
#![allow(dead_code)]

#[path = "fixture_io.rs"]
mod fixture_io;
#[path = "undigraph_fixture.rs"]
mod undigraph_fixture;

use geometric_traits::prelude::*;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct MinimumCycleBasisFixtureSuite {
    pub schema_version: u32,
    pub algorithm: String,
    pub graph_kind: String,
    pub generator: String,
    pub primary_oracle: String,
    pub cases: Vec<MinimumCycleBasisFixtureCase>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct MinimumCycleBasisFixtureCase {
    pub name: String,
    pub family: String,
    pub node_count: usize,
    pub edges: Vec<[usize; 2]>,
    pub cycle_rank: usize,
    pub basis_size: usize,
    pub total_weight: usize,
    pub minimum_cycle_basis: Vec<Vec<usize>>,
    pub notes: String,
}

pub fn load_fixture_suite(relative_path: &str) -> MinimumCycleBasisFixtureSuite {
    fixture_io::load_fixture_json(relative_path)
}

pub fn build_undigraph(case: &MinimumCycleBasisFixtureCase) -> UndiGraph<usize> {
    undigraph_fixture::build_undigraph(case.node_count, &case.edges)
}
