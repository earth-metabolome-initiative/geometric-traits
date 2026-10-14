//! Submodule declaring `K_4` homeomorph detection for simple undirected
//! graphs.
//!
//! The current implementation is a boolean detector layered on the crate's
//! internal Boyer-style edge-addition embedding engine.

use num_traits::AsPrimitive;

use super::topology_wrapper_macros::define_planarity_derived_error;
use crate::traits::{
    UndirectedMonopartiteMonoplexGraph,
    algorithms::{PlanarityError, planarity_detection},
};

define_planarity_derived_error!(
    /// Error type for `K_4` homeomorph detection.
    pub enum K4HomeomorphError => K4HomeomorphError,
    mapper = map_planarity_error_to_k4_homeomorph_error,
    self_loop = "The K4 homeomorph algorithm currently supports only simple undirected graphs and does not accept self-loops.",
    parallel = "The K4 homeomorph algorithm currently supports only simple undirected graphs and does not accept parallel edges."
);

/// Trait providing `K_4` homeomorph detection for simple undirected graphs.
pub trait K4HomeomorphDetection: UndirectedMonopartiteMonoplexGraph {
    /// Returns whether the graph contains a subgraph homeomorphic to `K_4`.
    ///
    /// The current implementation is a boolean detector driven by the crate's
    /// internal Boyer-style edge-addition embedding engine.
    ///
    /// # Errors
    ///
    /// Returns an error when the graph violates the simple-undirected contract,
    /// such as by containing self-loops, parallel edges, or malformed edge
    /// endpoints from a custom graph implementation.
    #[inline]
    fn has_k4_homeomorph(&self) -> Result<bool, crate::errors::MonopartiteError<Self>>
    where
        Self: Sized,
        Self::NodeId: AsPrimitive<usize>,
    {
        has_k4_homeomorph_simple_undirected_graph(self)
            .map_err(map_planarity_error_to_k4_homeomorph_error::<Self>)
    }
}

impl<G: ?Sized + UndirectedMonopartiteMonoplexGraph> K4HomeomorphDetection for G {}

pub(crate) fn has_k4_homeomorph_simple_undirected_graph<G>(
    graph: &G,
) -> Result<bool, PlanarityError>
where
    G: UndirectedMonopartiteMonoplexGraph,
    G::NodeId: AsPrimitive<usize>,
{
    planarity_detection::has_k4_homeomorph_simple_undirected_graph(graph)
}

#[cfg(test)]
mod tests {
    use super::{K4HomeomorphError, map_planarity_error_to_k4_homeomorph_error};
    use crate::{
        errors::{
            MonopartiteError, monopartite_graph_error::algorithms::MonopartiteAlgorithmError,
        },
        prelude::UndiGraph,
        traits::algorithms::planarity_detection::{
            preprocessing::LocalSimpleGraph, run_k4_homeomorph_engine,
        },
    };

    #[test]
    fn test_map_planarity_parallel_edge_error_to_k4_error() {
        let error: MonopartiteError<UndiGraph<usize>> = map_planarity_error_to_k4_homeomorph_error(
            crate::traits::algorithms::PlanarityError::ParallelEdgesUnsupported,
        );

        assert!(matches!(
            error,
            MonopartiteError::AlgorithmError(MonopartiteAlgorithmError::K4HomeomorphError(
                K4HomeomorphError::ParallelEdgesUnsupported
            ))
        ));
    }

    #[test]
    fn test_map_planarity_invalid_endpoint_error_to_k4_error() {
        let error: MonopartiteError<UndiGraph<usize>> = map_planarity_error_to_k4_homeomorph_error(
            crate::traits::algorithms::PlanarityError::InvalidEdgeEndpoint {
                endpoint: 9,
                node_count: 4,
            },
        );

        assert!(matches!(
            error,
            MonopartiteError::AlgorithmError(MonopartiteAlgorithmError::K4HomeomorphError(
                K4HomeomorphError::InvalidEdgeEndpoint { endpoint: 9, node_count: 4 }
            ))
        ));
    }

    #[test]
    fn test_run_k4_engine_detects_k4() {
        let graph =
            LocalSimpleGraph::from_edges(4, &[[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]])
                .unwrap();

        assert!(run_k4_homeomorph_engine(&graph.preprocess()));
    }

    #[test]
    fn test_run_k4_engine_rejects_theta() {
        let graph =
            LocalSimpleGraph::from_edges(5, &[[0, 2], [2, 1], [0, 3], [3, 1], [0, 4], [4, 1]])
                .unwrap();

        assert!(!run_k4_homeomorph_engine(&graph.preprocess()));
    }
}
