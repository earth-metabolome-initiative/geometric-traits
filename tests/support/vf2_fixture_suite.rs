#![cfg(feature = "std")]
#![allow(dead_code)]

use std::{collections::HashMap, fs, io::Read as _, path::Path};

use flate2::read::GzDecoder;
use geometric_traits::{
    impls::{CSR2D, SortedVec, SquareCSR2D, SymmetricCSR2D},
    prelude::*,
    traits::{EdgesBuilder, MonoplexMonopartiteGraph, VocabularyBuilder},
};
use serde::Deserialize;

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum OracleKind {
    Boolean,
    Embeddings,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SemanticMatch {
    #[default]
    None,
    LabelEquality,
}

#[derive(Debug, Deserialize)]
pub struct FixtureSuite {
    pub schema_version: u32,
    pub generator: String,
    pub networkx_timing_unit: String,
    pub cases: Vec<FixtureCase>,
}

#[derive(Debug, Deserialize)]
pub struct FixtureCase {
    pub source_fixture: String,
    pub name: String,
    pub oracle_kind: OracleKind,
    #[serde(default)]
    pub semantic_match: SemanticMatch,
    pub directed: bool,
    #[serde(default)]
    pub self_loops: bool,
    pub match_mode: String,
    pub query: FixtureGraph,
    pub target: FixtureGraph,
    pub expected_has_match: bool,
    #[serde(default)]
    pub expected_match_count: usize,
    #[serde(default)]
    pub expected_matches: Vec<Vec<[usize; 2]>>,
    pub networkx_ns: u64,
}

#[derive(Debug, Deserialize)]
pub struct FixtureGraph {
    pub node_count: usize,
    #[serde(default)]
    pub node_labels: Vec<u8>,
    pub edges: Vec<FixtureEdge>,
}

#[derive(Debug, Deserialize)]
pub struct FixtureEdge {
    pub src: usize,
    pub dst: usize,
    #[serde(default)]
    pub label: Option<u8>,
}

#[derive(Clone)]
pub struct BuiltUndiGraph {
    pub graph: UndiGraph<usize>,
    pub node_labels: Vec<u8>,
    edge_labels: HashMap<(usize, usize), u8>,
}

impl BuiltUndiGraph {
    pub fn edge_label(&self, src: usize, dst: usize) -> Option<u8> {
        let key = if src <= dst { (src, dst) } else { (dst, src) };
        self.edge_labels.get(&key).copied()
    }
}

#[derive(Clone)]
pub struct BuiltDiGraph {
    pub graph: DiGraph<usize>,
    pub node_labels: Vec<u8>,
    edge_labels: HashMap<(usize, usize), u8>,
}

impl BuiltDiGraph {
    pub fn edge_label(&self, src: usize, dst: usize) -> Option<u8> {
        self.edge_labels.get(&(src, dst)).copied()
    }
}

pub fn fixture_path(relative_path: &str) -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(relative_path)
}

pub fn load_fixture_suite(relative_path: &str) -> FixtureSuite {
    let fixture_gz = fs::read(fixture_path(relative_path))
        .unwrap_or_else(|_| panic!("failed to read tests/fixtures/{relative_path}"));
    let mut json = String::new();
    GzDecoder::new(fixture_gz.as_slice())
        .read_to_string(&mut json)
        .expect("gzip decompression failed");
    serde_json::from_str(&json)
        .unwrap_or_else(|_| panic!("`tests/fixtures/{relative_path}` must contain valid JSON"))
}

pub fn build_undigraph(graph: &FixtureGraph) -> BuiltUndiGraph {
    let nodes: SortedVec<usize> = GenericVocabularyBuilder::default()
        .expected_number_of_symbols(graph.node_count)
        .symbols((0..graph.node_count).enumerate())
        .build()
        .unwrap();
    let mut edges: Vec<(usize, usize)> =
        graph.edges.iter().map(|edge| (edge.src.min(edge.dst), edge.src.max(edge.dst))).collect();
    edges.sort_unstable();
    let edge_labels = graph
        .edges
        .iter()
        .filter_map(|edge| {
            edge.label.map(|label| ((edge.src.min(edge.dst), edge.src.max(edge.dst)), label))
        })
        .collect();
    let edges: SymmetricCSR2D<CSR2D<usize, usize, usize>> = UndiEdgesBuilder::default()
        .expected_number_of_edges(edges.len())
        .expected_shape(graph.node_count)
        .edges(edges.into_iter())
        .build()
        .unwrap();
    BuiltUndiGraph {
        graph: UndiGraph::from((nodes, edges)),
        node_labels: graph.node_labels.clone(),
        edge_labels,
    }
}

pub fn build_digraph(graph: &FixtureGraph) -> BuiltDiGraph {
    let nodes: SortedVec<usize> = GenericVocabularyBuilder::default()
        .expected_number_of_symbols(graph.node_count)
        .symbols((0..graph.node_count).enumerate())
        .build()
        .unwrap();
    let mut edges: Vec<(usize, usize)> =
        graph.edges.iter().map(|edge| (edge.src, edge.dst)).collect();
    edges.sort_unstable();
    let edge_labels = graph
        .edges
        .iter()
        .filter_map(|edge| edge.label.map(|label| ((edge.src, edge.dst), label)))
        .collect();
    let edges: SquareCSR2D<_> = DiEdgesBuilder::default()
        .expected_number_of_edges(edges.len())
        .expected_shape(graph.node_count)
        .edges(edges.into_iter())
        .build()
        .unwrap();
    BuiltDiGraph {
        graph: DiGraph::from((nodes, edges)),
        node_labels: graph.node_labels.clone(),
        edge_labels,
    }
}

pub fn parse_match_mode(mode: &str) -> Vf2Mode {
    match mode {
        "graph_isomorphism" => Vf2Mode::Isomorphism,
        "induced_subgraph_isomorphism" => Vf2Mode::InducedSubgraphIsomorphism,
        "subgraph_isomorphism" => Vf2Mode::SubgraphIsomorphism,
        "monomorphism" => Vf2Mode::Monomorphism,
        _ => panic!("unknown VF2 match mode `{mode}` in fixture"),
    }
}

pub fn canonicalize_mapping_pairs(mut pairs: Vec<(usize, usize)>) -> Vec<[usize; 2]> {
    pairs.sort_unstable();
    pairs.into_iter().map(|(query_node, target_node)| [query_node, target_node]).collect()
}

pub fn collect_matches<G>(query: &G, target: &G, mode: Vf2Mode) -> Vec<Vec<[usize; 2]>>
where
    G: MonoplexMonopartiteGraph<NodeId = usize>,
{
    let mut matches = Vec::new();
    let exhausted = query.vf2(target).with_mode(mode).for_each_match(|mapping| {
        matches.push(canonicalize_mapping_pairs(mapping.pairs().to_vec()));
        true
    });
    assert!(exhausted, "the embedding oracle should exhaust every match");
    matches.sort_unstable();
    matches
}
