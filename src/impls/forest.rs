//! Submodule providing forest data structures.
//!
//! A forest stores a collection of rooted trees as a parent array indexed by
//! node. [`Forest`] is the unweighted directed child-to-parent adjacency,
//! [`WeightedForest`] attaches an edge weight to every tree edge, and
//! [`SymmetricForest`] is the undirected symmetric view obtained from a
//! weighted forest.

mod rooted_forest;
mod symmetric_forest;
mod weighted_forest;

pub use rooted_forest::{Forest, ForestNonRootEntries};
pub use symmetric_forest::SymmetricForest;
pub use weighted_forest::{WeightedForest, WeightedForestNonRootEntries};
