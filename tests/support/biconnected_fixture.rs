#![cfg(feature = "std")]
#![allow(dead_code)]

#[path = "fixture_io.rs"]
mod fixture_io;
#[path = "undigraph_fixture.rs"]
mod undigraph_fixture;

use geometric_traits::prelude::*;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct BiconnectedFixtureSuite {
    pub schema_version: u32,
    pub algorithm: String,
    pub graph_kind: String,
    pub generator: String,
    pub primary_oracle: String,
    pub dyad_is_biconnected_component: bool,
    pub isolated_vertices_form_biconnected_components: bool,
    pub component_ordering: String,
    pub cases: Vec<BiconnectedFixtureCase>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct BiconnectedFixtureCase {
    pub name: String,
    pub family: String,
    pub node_count: usize,
    pub edges: Vec<[usize; 2]>,
    pub connected_components: Vec<Vec<usize>>,
    #[serde(default)]
    pub vertices_without_biconnected_component: Vec<usize>,
    pub edge_biconnected_components: Vec<Vec<[usize; 2]>>,
    pub vertex_biconnected_components: Vec<Vec<usize>>,
    pub articulation_points: Vec<usize>,
    pub bridges: Vec<[usize; 2]>,
    #[serde(default)]
    pub cyclic_biconnected_component_indices: Vec<usize>,
    pub is_biconnected: bool,
    pub notes: String,
}

pub fn load_fixture_suite(relative_path: &str) -> BiconnectedFixtureSuite {
    fixture_io::load_fixture_json(relative_path)
}

pub fn normalize_edge([left, right]: [usize; 2]) -> [usize; 2] {
    undigraph_fixture::normalize_edge([left, right])
}

pub fn build_undigraph(case: &BiconnectedFixtureCase) -> UndiGraph<usize> {
    undigraph_fixture::build_undigraph(case.node_count, &case.edges)
}

fn edge_case_semantic_cases() -> Vec<BiconnectedFixtureCase> {
    vec![
        BiconnectedFixtureCase {
            name: "isolated_singleton".to_string(),
            family: "edge_case".to_string(),
            node_count: 1,
            edges: vec![],
            connected_components: vec![vec![0]],
            vertices_without_biconnected_component: vec![0],
            edge_biconnected_components: vec![],
            vertex_biconnected_components: vec![],
            articulation_points: vec![],
            bridges: vec![],
            cyclic_biconnected_component_indices: vec![],
            is_biconnected: false,
            notes: "Pins the isolate convention: a lone vertex is connected to itself but does not produce any edge-based bicomponent.".to_string(),
        },
        BiconnectedFixtureCase {
            name: "single_edge_dyad".to_string(),
            family: "edge_case".to_string(),
            node_count: 2,
            edges: vec![[0, 1]],
            connected_components: vec![vec![0, 1]],
            vertices_without_biconnected_component: vec![],
            edge_biconnected_components: vec![vec![[0, 1]]],
            vertex_biconnected_components: vec![vec![0, 1]],
            articulation_points: vec![],
            bridges: vec![[0, 1]],
            cyclic_biconnected_component_indices: vec![],
            is_biconnected: true,
            notes: "Pins the dyad convention used by NetworkX, JGraphT, Boost practice, and igraph.".to_string(),
        },
    ]
}

fn tree_and_basic_cycle_semantic_cases() -> Vec<BiconnectedFixtureCase> {
    vec![
        BiconnectedFixtureCase {
            name: "path_four".to_string(),
            family: "tree".to_string(),
            node_count: 4,
            edges: vec![[0, 1], [1, 2], [2, 3]],
            connected_components: vec![vec![0, 1, 2, 3]],
            vertices_without_biconnected_component: vec![],
            edge_biconnected_components: vec![vec![[0, 1]], vec![[1, 2]], vec![[2, 3]]],
            vertex_biconnected_components: vec![vec![0, 1], vec![1, 2], vec![2, 3]],
            articulation_points: vec![1, 2],
            bridges: vec![[0, 1], [1, 2], [2, 3]],
            cyclic_biconnected_component_indices: vec![],
            is_biconnected: false,
            notes: "Every edge is its own block in a tree; internal vertices become articulation points.".to_string(),
        },
        BiconnectedFixtureCase {
            name: "triangle_cycle".to_string(),
            family: "cyclic".to_string(),
            node_count: 3,
            edges: vec![[0, 1], [0, 2], [1, 2]],
            connected_components: vec![vec![0, 1, 2]],
            vertices_without_biconnected_component: vec![],
            edge_biconnected_components: vec![vec![[0, 1], [0, 2], [1, 2]]],
            vertex_biconnected_components: vec![vec![0, 1, 2]],
            articulation_points: vec![],
            bridges: vec![],
            cyclic_biconnected_component_indices: vec![0],
            is_biconnected: true,
            notes: "Minimal cyclic ring block.".to_string(),
        },
    ]
}

fn ring_system_semantic_cases() -> Vec<BiconnectedFixtureCase> {
    vec![
        BiconnectedFixtureCase {
            name: "fused_bicyclic_diamond".to_string(),
            family: "ring_system".to_string(),
            node_count: 4,
            edges: vec![[0, 1], [0, 2], [1, 2], [1, 3], [2, 3]],
            connected_components: vec![vec![0, 1, 2, 3]],
            vertices_without_biconnected_component: vec![],
            edge_biconnected_components: vec![vec![[0, 1], [0, 2], [1, 2], [1, 3], [2, 3]]],
            vertex_biconnected_components: vec![vec![0, 1, 2, 3]],
            articulation_points: vec![],
            bridges: vec![],
            cyclic_biconnected_component_indices: vec![0],
            is_biconnected: true,
            notes: "Fused rings share an edge and must remain one ring block.".to_string(),
        },
        BiconnectedFixtureCase {
            name: "spiro_bicyclic".to_string(),
            family: "ring_system".to_string(),
            node_count: 5,
            edges: vec![[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [3, 4]],
            connected_components: vec![vec![0, 1, 2, 3, 4]],
            vertices_without_biconnected_component: vec![],
            edge_biconnected_components: vec![
                vec![[0, 1], [0, 2], [1, 2]],
                vec![[0, 3], [0, 4], [3, 4]],
            ],
            vertex_biconnected_components: vec![vec![0, 1, 2], vec![0, 3, 4]],
            articulation_points: vec![0],
            bridges: vec![],
            cyclic_biconnected_component_indices: vec![0, 1],
            is_biconnected: false,
            notes: "Spiro rings share one articulation vertex and must split into two ring blocks."
                .to_string(),
        },
        BiconnectedFixtureCase {
            name: "theta_graph".to_string(),
            family: "ring_system".to_string(),
            node_count: 4,
            edges: vec![[0, 1], [0, 2], [0, 3], [1, 3], [2, 3]],
            connected_components: vec![vec![0, 1, 2, 3]],
            vertices_without_biconnected_component: vec![],
            edge_biconnected_components: vec![vec![[0, 1], [0, 2], [0, 3], [1, 3], [2, 3]]],
            vertex_biconnected_components: vec![vec![0, 1, 2, 3]],
            articulation_points: vec![],
            bridges: vec![],
            cyclic_biconnected_component_indices: vec![0],
            is_biconnected: true,
            notes: "A bridged bicyclic core stays one block despite having multiple cycles."
                .to_string(),
        },
    ]
}

fn mixed_semantic_cases() -> Vec<BiconnectedFixtureCase> {
    vec![
        BiconnectedFixtureCase {
            name: "cycle_with_tail".to_string(),
            family: "mixed".to_string(),
            node_count: 5,
            edges: vec![[0, 1], [0, 2], [1, 2], [2, 3], [3, 4]],
            connected_components: vec![vec![0, 1, 2, 3, 4]],
            vertices_without_biconnected_component: vec![],
            edge_biconnected_components: vec![
                vec![[0, 1], [0, 2], [1, 2]],
                vec![[2, 3]],
                vec![[3, 4]],
            ],
            vertex_biconnected_components: vec![vec![0, 1, 2], vec![2, 3], vec![3, 4]],
            articulation_points: vec![2, 3],
            bridges: vec![[2, 3], [3, 4]],
            cyclic_biconnected_component_indices: vec![0],
            is_biconnected: false,
            notes: "Models a ring with chain substitution; the tail must peel off as bridge dyads.".to_string(),
        },
        BiconnectedFixtureCase {
            name: "disconnected_cycle_path_isolate".to_string(),
            family: "mixed".to_string(),
            node_count: 7,
            edges: vec![[0, 1], [0, 2], [1, 2], [3, 4], [4, 5]],
            connected_components: vec![vec![0, 1, 2], vec![3, 4, 5], vec![6]],
            vertices_without_biconnected_component: vec![6],
            edge_biconnected_components: vec![
                vec![[0, 1], [0, 2], [1, 2]],
                vec![[3, 4]],
                vec![[4, 5]],
            ],
            vertex_biconnected_components: vec![vec![0, 1, 2], vec![3, 4], vec![4, 5]],
            articulation_points: vec![4],
            bridges: vec![[3, 4], [4, 5]],
            cyclic_biconnected_component_indices: vec![0],
            is_biconnected: false,
            notes: "Pins disconnected handling and makes isolate omission explicit.".to_string(),
        },
        BiconnectedFixtureCase {
            name: "triangle_barbell_bridge".to_string(),
            family: "mixed".to_string(),
            node_count: 6,
            edges: vec![[0, 1], [0, 2], [1, 2], [2, 3], [3, 4], [3, 5], [4, 5]],
            connected_components: vec![vec![0, 1, 2, 3, 4, 5]],
            vertices_without_biconnected_component: vec![],
            edge_biconnected_components: vec![
                vec![[0, 1], [0, 2], [1, 2]],
                vec![[2, 3]],
                vec![[3, 4], [3, 5], [4, 5]],
            ],
            vertex_biconnected_components: vec![vec![0, 1, 2], vec![2, 3], vec![3, 4, 5]],
            articulation_points: vec![2, 3],
            bridges: vec![[2, 3]],
            cyclic_biconnected_component_indices: vec![0, 2],
            is_biconnected: false,
            notes: "Two ring systems connected by a single bridge edge must split cleanly into cycle, bridge, cycle.".to_string(),
        },
    ]
}

pub fn semantic_cases() -> Vec<BiconnectedFixtureCase> {
    let mut cases = Vec::new();
    cases.extend(edge_case_semantic_cases());
    cases.extend(tree_and_basic_cycle_semantic_cases());
    cases.extend(ring_system_semantic_cases());
    cases.extend(mixed_semantic_cases());
    cases
}
