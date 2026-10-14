#![cfg(feature = "std")]
#![allow(dead_code)]

#[path = "fixture_io.rs"]
mod fixture_io;
#[path = "undigraph_fixture.rs"]
mod undigraph_fixture;

use geometric_traits::prelude::*;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct PlanarityFixtureSuite {
    pub schema_version: u32,
    pub algorithm: String,
    pub graph_kind: String,
    pub generator: String,
    pub primary_oracle: String,
    pub cases: Vec<PlanarityFixtureCase>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct PlanarityFixtureCase {
    pub name: String,
    pub family: String,
    pub node_count: usize,
    pub edges: Vec<[usize; 2]>,
    pub is_planar: bool,
    pub is_outerplanar: bool,
    #[serde(default)]
    pub planarity_obstruction_family: Option<String>,
    #[serde(default)]
    pub outerplanarity_obstruction_family: Option<String>,
    pub notes: String,
}

pub fn load_fixture_suite(relative_path: &str) -> PlanarityFixtureSuite {
    fixture_io::load_fixture_json(relative_path)
}

pub fn normalize_edge(edge: [usize; 2]) -> [usize; 2] {
    undigraph_fixture::normalize_edge(edge)
}

pub fn build_undigraph(case: &PlanarityFixtureCase) -> UndiGraph<usize> {
    undigraph_fixture::build_undigraph(case.node_count, &case.edges)
}

#[allow(clippy::too_many_lines)]
pub fn semantic_cases() -> Vec<PlanarityFixtureCase> {
    vec![
        PlanarityFixtureCase {
            name: "isolated_singleton".to_string(),
            family: "edge_case".to_string(),
            node_count: 1,
            edges: vec![],
            is_planar: true,
            is_outerplanar: true,
            planarity_obstruction_family: None,
            outerplanarity_obstruction_family: None,
            notes: "A lone vertex is vacuously planar and outerplanar.".to_string(),
        },
        PlanarityFixtureCase {
            name: "single_edge_dyad".to_string(),
            family: "edge_case".to_string(),
            node_count: 2,
            edges: vec![[0, 1]],
            is_planar: true,
            is_outerplanar: true,
            planarity_obstruction_family: None,
            outerplanarity_obstruction_family: None,
            notes: "The smallest connected simple graph stays planar and outerplanar.".to_string(),
        },
        PlanarityFixtureCase {
            name: "path_four".to_string(),
            family: "tree".to_string(),
            node_count: 4,
            edges: vec![[0, 1], [1, 2], [2, 3]],
            is_planar: true,
            is_outerplanar: true,
            planarity_obstruction_family: None,
            outerplanarity_obstruction_family: None,
            notes: "Every tree is outerplanar.".to_string(),
        },
        PlanarityFixtureCase {
            name: "triangle_cycle".to_string(),
            family: "cycle".to_string(),
            node_count: 3,
            edges: vec![[0, 1], [0, 2], [1, 2]],
            is_planar: true,
            is_outerplanar: true,
            planarity_obstruction_family: None,
            outerplanarity_obstruction_family: None,
            notes: "A simple cycle is outerplanar.".to_string(),
        },
        PlanarityFixtureCase {
            name: "diamond_k4_minus_edge".to_string(),
            family: "outerplanar_with_chord".to_string(),
            node_count: 4,
            edges: vec![[0, 1], [0, 2], [1, 2], [1, 3], [2, 3]],
            is_planar: true,
            is_outerplanar: true,
            planarity_obstruction_family: None,
            outerplanarity_obstruction_family: None,
            notes: "A cycle with one noncrossing chord remains outerplanar.".to_string(),
        },
        PlanarityFixtureCase {
            name: "k4_complete".to_string(),
            family: "outerplanarity_obstruction".to_string(),
            node_count: 4,
            edges: vec![[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]],
            is_planar: true,
            is_outerplanar: false,
            planarity_obstruction_family: None,
            outerplanarity_obstruction_family: Some("K4".to_string()),
            notes: "K4 is planar but not outerplanar.".to_string(),
        },
        PlanarityFixtureCase {
            name: "k23_complete_bipartite".to_string(),
            family: "outerplanarity_obstruction".to_string(),
            node_count: 5,
            edges: vec![[0, 2], [0, 3], [0, 4], [1, 2], [1, 3], [1, 4]],
            is_planar: true,
            is_outerplanar: false,
            planarity_obstruction_family: None,
            outerplanarity_obstruction_family: Some("K2,3".to_string()),
            notes: "K2,3 is planar but forbidden for outerplanarity.".to_string(),
        },
        PlanarityFixtureCase {
            name: "k33_complete_bipartite".to_string(),
            family: "planarity_obstruction".to_string(),
            node_count: 6,
            edges: vec![[0, 3], [0, 4], [0, 5], [1, 3], [1, 4], [1, 5], [2, 3], [2, 4], [2, 5]],
            is_planar: false,
            is_outerplanar: false,
            planarity_obstruction_family: Some("K3,3".to_string()),
            outerplanarity_obstruction_family: Some("nonplanar".to_string()),
            notes: "K3,3 is one of Kuratowski's forbidden nonplanar graphs.".to_string(),
        },
        PlanarityFixtureCase {
            name: "k5_complete".to_string(),
            family: "planarity_obstruction".to_string(),
            node_count: 5,
            edges: vec![
                [0, 1],
                [0, 2],
                [0, 3],
                [0, 4],
                [1, 2],
                [1, 3],
                [1, 4],
                [2, 3],
                [2, 4],
                [3, 4],
            ],
            is_planar: false,
            is_outerplanar: false,
            planarity_obstruction_family: Some("K5".to_string()),
            outerplanarity_obstruction_family: Some("nonplanar".to_string()),
            notes: "K5 is the other Kuratowski obstruction family.".to_string(),
        },
    ]
}
