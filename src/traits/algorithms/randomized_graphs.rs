//! Submodule providing traits to generate randomized and structured graphs.

#[cfg(feature = "alloc")]
mod xorshift64;
#[cfg(feature = "alloc")]
pub use xorshift64::XorShift64;

#[cfg(feature = "alloc")]
pub(crate) mod builder_utils;

#[cfg(all(feature = "alloc", any(feature = "std", feature = "hashbrown")))]
mod randomized_dag;
#[cfg(all(feature = "alloc", any(feature = "std", feature = "hashbrown")))]
pub use randomized_dag::RandomizedDAG;

// ── Deterministic graph families ────────────────────────────────────────────

#[cfg(feature = "alloc")]
mod complete;
#[cfg(feature = "alloc")]
pub use complete::complete_graph;

#[cfg(feature = "alloc")]
mod cycle;
#[cfg(feature = "alloc")]
pub use cycle::cycle_graph;

#[cfg(feature = "alloc")]
mod path;
#[cfg(feature = "alloc")]
pub use path::path_graph;

#[cfg(feature = "alloc")]
mod star;
#[cfg(feature = "alloc")]
pub use star::star_graph;

#[cfg(feature = "alloc")]
mod grid;
#[cfg(feature = "alloc")]
pub use grid::grid_graph;

#[cfg(feature = "alloc")]
mod hexagonal_lattice;
#[cfg(feature = "alloc")]
pub use hexagonal_lattice::hexagonal_lattice_graph;

#[cfg(feature = "alloc")]
mod triangular_lattice;
#[cfg(feature = "alloc")]
pub use triangular_lattice::triangular_lattice_graph;

#[cfg(feature = "alloc")]
mod torus;
#[cfg(feature = "alloc")]
pub use torus::torus_graph;

#[cfg(feature = "alloc")]
mod hypercube;
#[cfg(feature = "alloc")]
pub use hypercube::hypercube_graph;

#[cfg(feature = "alloc")]
mod barbell;
#[cfg(feature = "alloc")]
pub use barbell::barbell_graph;

#[cfg(feature = "alloc")]
mod crown;
#[cfg(feature = "alloc")]
pub use crown::crown_graph;

#[cfg(feature = "alloc")]
mod wheel;
#[cfg(feature = "alloc")]
pub use wheel::wheel_graph;

#[cfg(feature = "alloc")]
mod complete_bipartite;
#[cfg(feature = "alloc")]
pub use complete_bipartite::complete_bipartite_graph;

#[cfg(feature = "alloc")]
mod petersen;
#[cfg(feature = "alloc")]
pub use petersen::petersen_graph;

#[cfg(feature = "alloc")]
mod turan;
#[cfg(feature = "alloc")]
pub use turan::turan_graph;

#[cfg(feature = "alloc")]
mod windmill;
#[cfg(feature = "alloc")]
pub use windmill::windmill_graph;

#[cfg(feature = "alloc")]
mod friendship;
#[cfg(feature = "alloc")]
pub use friendship::friendship_graph;

// ── Random graph models ─────────────────────────────────────────────────────

#[cfg(all(feature = "alloc", any(feature = "std", feature = "hashbrown")))]
mod erdos_renyi_gnm;
#[cfg(all(feature = "alloc", any(feature = "std", feature = "hashbrown")))]
pub use erdos_renyi_gnm::erdos_renyi_gnm;

#[cfg(feature = "alloc")]
mod erdos_renyi_gnp;
#[cfg(feature = "alloc")]
pub use erdos_renyi_gnp::erdos_renyi_gnp;

#[cfg(feature = "alloc")]
mod barabasi_albert;
#[cfg(feature = "alloc")]
pub use barabasi_albert::barabasi_albert;

#[cfg(feature = "alloc")]
mod watts_strogatz;
#[cfg(feature = "alloc")]
pub use watts_strogatz::watts_strogatz;

#[cfg(all(feature = "alloc", any(feature = "std", feature = "hashbrown")))]
mod random_regular;
#[cfg(all(feature = "alloc", any(feature = "std", feature = "hashbrown")))]
pub use random_regular::{RandomRegularGraphError, random_regular_graph};

#[cfg(feature = "alloc")]
mod stochastic_block_model;
#[cfg(feature = "alloc")]
pub use stochastic_block_model::stochastic_block_model;

#[cfg(all(feature = "alloc", any(feature = "std", feature = "hashbrown")))]
mod configuration_model;
#[cfg(all(feature = "alloc", any(feature = "std", feature = "hashbrown")))]
pub use configuration_model::configuration_model;

#[cfg(feature = "alloc")]
mod chung_lu;
#[cfg(feature = "alloc")]
pub use chung_lu::chung_lu;

#[cfg(feature = "alloc")]
mod random_geometric;
#[cfg(feature = "alloc")]
pub use random_geometric::random_geometric_graph;
