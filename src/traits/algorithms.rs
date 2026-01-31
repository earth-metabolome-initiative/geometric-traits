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
mod tarjan;
#[cfg(feature = "alloc")]
pub use tarjan::*;
