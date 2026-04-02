//! Submodule providing crates that define algorithms for graphs.

#[cfg(feature = "alloc")]
pub mod connected_components;
#[cfg(feature = "alloc")]
pub use connected_components::ConnectedComponents;
#[cfg(feature = "alloc")]
pub mod cycle_detection;
#[cfg(feature = "alloc")]
pub use cycle_detection::CycleDetection;
#[cfg(feature = "alloc")]
mod node_classification;
#[cfg(feature = "alloc")]
pub mod root_nodes;
#[cfg(feature = "alloc")]
pub use root_nodes::RootNodes;
#[cfg(feature = "alloc")]
pub mod sink_nodes;
#[cfg(feature = "alloc")]
pub use sink_nodes::SinkNodes;
#[cfg(feature = "alloc")]
pub mod simple_path;
#[cfg(feature = "alloc")]
pub use simple_path::SimplePath;
#[cfg(feature = "alloc")]
pub mod resnik;
#[cfg(feature = "alloc")]
pub use resnik::{Resnik, ResnikResult};
#[cfg(feature = "alloc")]
pub mod information_content;
#[cfg(feature = "alloc")]
pub use information_content::{
    InformationContent, InformationContentError, InformationContentResult,
};
#[cfg(feature = "alloc")]
pub mod lin;
#[cfg(feature = "alloc")]
pub use lin::{Lin, LinResult};
#[cfg(feature = "alloc")]
pub mod singleton_nodes;
#[cfg(feature = "alloc")]
pub use singleton_nodes::SingletonNodes;
#[cfg(feature = "alloc")]
pub mod wu_palmer;
#[cfg(feature = "alloc")]
pub use wu_palmer::{WuPalmer, WuPalmerResult};
pub mod randomized_graphs;
#[cfg(all(feature = "alloc", any(feature = "std", feature = "hashbrown")))]
pub use randomized_graphs::RandomizedDAG;
mod assignment;
pub use assignment::*;
#[cfg(feature = "alloc")]
mod weighted_assignment;
#[cfg(feature = "alloc")]
pub use weighted_assignment::*;
#[cfg(feature = "alloc")]
mod kahn;
#[cfg(feature = "alloc")]
pub use kahn::*;
#[cfg(feature = "alloc")]
mod johnson;
#[cfg(feature = "alloc")]
pub use johnson::*;
#[cfg(feature = "alloc")]
mod floyd_warshall;
#[cfg(feature = "alloc")]
pub use floyd_warshall::*;
#[cfg(feature = "alloc")]
mod pairwise_bfs;
#[cfg(feature = "alloc")]
pub use pairwise_bfs::*;
#[cfg(feature = "alloc")]
mod pairwise_dijkstra;
#[cfg(feature = "alloc")]
pub use pairwise_dijkstra::*;
#[cfg(feature = "alloc")]
mod tarjan;
#[cfg(feature = "alloc")]
pub use tarjan::*;
#[cfg(feature = "alloc")]
mod modularity;
#[cfg(feature = "alloc")]
pub use modularity::ModularityError;
#[cfg(feature = "alloc")]
mod louvain;
#[cfg(feature = "alloc")]
pub use louvain::*;
#[cfg(feature = "alloc")]
mod leiden;
#[cfg(feature = "alloc")]
pub use leiden::*;
#[cfg(feature = "alloc")]
mod jacobi;
#[cfg(feature = "alloc")]
pub use jacobi::*;
#[cfg(feature = "alloc")]
mod gth;
#[cfg(feature = "alloc")]
pub use gth::*;
#[cfg(feature = "alloc")]
mod mds;
#[cfg(feature = "alloc")]
pub use mds::*;
#[cfg(feature = "alloc")]
mod blossom;
#[cfg(feature = "alloc")]
mod matching_utils;
#[cfg(feature = "alloc")]
pub use blossom::*;
#[cfg(feature = "alloc")]
mod gabow_1976;
#[cfg(feature = "alloc")]
pub use gabow_1976::*;
#[cfg(feature = "alloc")]
mod micali_vazirani;
#[cfg(feature = "alloc")]
pub use micali_vazirani::*;
#[cfg(feature = "alloc")]
mod kocay;
#[cfg(feature = "alloc")]
pub use kocay::*;
#[cfg(feature = "alloc")]
mod blossom_v;
#[cfg(feature = "alloc")]
pub use blossom_v::*;
#[cfg(feature = "alloc")]
pub mod line_graph;
#[cfg(feature = "alloc")]
pub use line_graph::{LineGraph, LineGraphResult};
#[cfg(feature = "alloc")]
pub mod modular_product;
#[cfg(feature = "alloc")]
pub use modular_product::{ModularProduct, ModularProductResult};
#[cfg(feature = "alloc")]
pub mod maximum_clique;
#[cfg(feature = "alloc")]
pub use maximum_clique::{
    MaximumClique, OwnedPartitionLabels, PartitionInfo, PartitionedMaximumClique,
};
#[cfg(feature = "alloc")]
pub mod delta_y_exchange;
#[cfg(feature = "alloc")]
pub use delta_y_exchange::DeltaYExchange;
#[cfg(feature = "alloc")]
mod labeled_line_graph;
#[cfg(feature = "alloc")]
pub use labeled_line_graph::LabeledLineGraph;
#[cfg(feature = "alloc")]
mod vertex_match_inference;
#[cfg(feature = "alloc")]
pub use vertex_match_inference::{infer_vertex_matches, shared_endpoint};
pub mod graph_similarities;
pub use graph_similarities::{
    GraphSimilarities, braun_blanquet_similarity, cosine_similarity, dice_similarity,
    johnson_similarity, kulczynski_similarity, mcconnaughey_similarity, overlap_similarity,
    sokal_sneath_similarity, tanimoto_similarity, tversky_similarity,
};
#[cfg(feature = "alloc")]
pub mod clique_ranking;
#[cfg(feature = "alloc")]
pub mod mces;
#[cfg(feature = "alloc")]
pub use clique_ranking::{
    ChainedRanker, CliqueInfo, CliqueRanker, CliqueRankerExt, EagerCliqueInfo, FnRanker,
    FragmentCountRanker, LargestFragmentMetric, LargestFragmentMetricRanker, LargestFragmentRanker,
    MatchedEdgePair,
};
#[cfg(feature = "alloc")]
pub use mces::{McesBuilder, McesResult, McesSearchMode};
