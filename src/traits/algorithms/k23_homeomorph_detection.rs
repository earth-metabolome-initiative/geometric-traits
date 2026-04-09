//! Submodule declaring `K_{2,3}` homeomorph detection for simple undirected
//! graphs.
//!
//! The current implementation follows Boyer's edge-addition outerplanarity
//! search structure for `K_{2,3}` detection. It treats `K_{2,3}` search as a
//! blocked-bicomp handler layered on the same DFS/pertinence/ext-face state
//! used by the outerplanarity engine.

use num_traits::AsPrimitive;

use crate::traits::{
    MonopartiteGraph, UndirectedMonopartiteMonoplexGraph,
    algorithms::{PlanarityError, planarity_detection},
};

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
/// Error type for `K_{2,3}` homeomorph detection.
pub enum K23HomeomorphError {
    /// The graph contains self-loops, which are unsupported by the intended
    /// simple undirected implementation.
    #[error(
        "The K23 homeomorph algorithm currently supports only simple undirected graphs and does not accept self-loops."
    )]
    SelfLoopsUnsupported,
    /// Parallel edges are unsupported by the intended public contract.
    #[error(
        "The K23 homeomorph algorithm currently supports only simple undirected graphs and does not accept parallel edges."
    )]
    ParallelEdgesUnsupported,
    /// The graph implementation exposed an endpoint outside the node range.
    #[error(
        "The graph exposed edge endpoint {endpoint}, which is out of range for node_count={node_count}."
    )]
    InvalidEdgeEndpoint {
        /// The offending endpoint value exposed by the graph.
        endpoint: usize,
        /// The graph node count used to validate endpoints.
        node_count: usize,
    },
}

impl From<K23HomeomorphError>
    for crate::errors::monopartite_graph_error::algorithms::MonopartiteAlgorithmError
{
    #[inline]
    fn from(error: K23HomeomorphError) -> Self {
        Self::K23HomeomorphError(error)
    }
}

impl<G: MonopartiteGraph> From<K23HomeomorphError> for crate::errors::MonopartiteError<G> {
    #[inline]
    fn from(error: K23HomeomorphError) -> Self {
        Self::AlgorithmError(error.into())
    }
}

#[allow(clippy::needless_pass_by_value)]
#[inline]
fn map_planarity_error_to_k23_homeomorph_error<G: MonopartiteGraph>(
    error: PlanarityError,
) -> crate::errors::MonopartiteError<G> {
    match error {
        PlanarityError::SelfLoopsUnsupported => K23HomeomorphError::SelfLoopsUnsupported.into(),
        PlanarityError::ParallelEdgesUnsupported => {
            K23HomeomorphError::ParallelEdgesUnsupported.into()
        }
        PlanarityError::InvalidEdgeEndpoint { endpoint, node_count } => {
            K23HomeomorphError::InvalidEdgeEndpoint { endpoint, node_count }.into()
        }
    }
}

/// Trait providing `K_{2,3}` homeomorph detection for simple undirected graphs.
pub trait K23HomeomorphDetection: UndirectedMonopartiteMonoplexGraph {
    /// Returns whether the graph contains a subgraph homeomorphic to `K_{2,3}`.
    ///
    /// The implementation follows Boyer's outerplanarity-obstruction search:
    /// it runs the outerplanarity walkdown, intercepts blocked bicomps, and
    /// distinguishes `K_{2,3}` from separable `K_4` obstructions using the
    /// same blocked-bicomp state.
    ///
    /// # Errors
    ///
    /// Returns an error when the graph violates the simple-undirected contract,
    /// such as by containing self-loops, parallel edges, or malformed edge
    /// endpoints from a custom graph implementation.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use geometric_traits::{
    ///     impls::{CSR2D, SortedVec, SymmetricCSR2D},
    ///     prelude::*,
    ///     traits::{EdgesBuilder, K23HomeomorphDetection, VocabularyBuilder},
    /// };
    ///
    /// fn build_undigraph(node_count: usize, edges: &[(usize, usize)]) -> UndiGraph<usize> {
    ///     let nodes: SortedVec<usize> = GenericVocabularyBuilder::default()
    ///         .expected_number_of_symbols(node_count)
    ///         .symbols((0..node_count).enumerate())
    ///         .build()
    ///         .unwrap();
    ///
    ///     let mut normalized_edges = edges.to_vec();
    ///     for (left, right) in &mut normalized_edges {
    ///         if *left > *right {
    ///             core::mem::swap(left, right);
    ///         }
    ///     }
    ///     normalized_edges.sort_unstable();
    ///
    ///     let matrix: SymmetricCSR2D<CSR2D<usize, usize, usize>> = UndiEdgesBuilder::default()
    ///         .expected_number_of_edges(normalized_edges.len())
    ///         .expected_shape(node_count)
    ///         .edges(normalized_edges.into_iter())
    ///         .build()
    ///         .unwrap();
    ///
    ///     UndiGraph::from((nodes, matrix))
    /// }
    ///
    /// let k23 = build_undigraph(5, &[(0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4)]);
    /// let k4 = build_undigraph(4, &[(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]);
    ///
    /// assert!(k23.has_k23_homeomorph()?);
    /// assert!(!k4.has_k23_homeomorph()?);
    /// # Ok::<(), geometric_traits::errors::MonopartiteError<UndiGraph<usize>>>(())
    /// ```
    #[inline]
    fn has_k23_homeomorph(&self) -> Result<bool, crate::errors::MonopartiteError<Self>>
    where
        Self: Sized,
        Self::NodeId: AsPrimitive<usize>,
    {
        has_k23_homeomorph_simple_undirected_graph(self)
            .map_err(map_planarity_error_to_k23_homeomorph_error::<Self>)
    }
}

impl<G: ?Sized + UndirectedMonopartiteMonoplexGraph> K23HomeomorphDetection for G {}

pub(crate) fn has_k23_homeomorph_simple_undirected_graph<G>(
    graph: &G,
) -> Result<bool, PlanarityError>
where
    G: UndirectedMonopartiteMonoplexGraph,
    G::NodeId: AsPrimitive<usize>,
{
    planarity_detection::has_k23_homeomorph_simple_undirected_graph(graph)
}

#[cfg(test)]
mod tests {
    use super::{K23HomeomorphError, map_planarity_error_to_k23_homeomorph_error};
    use crate::{
        errors::{
            MonopartiteError, monopartite_graph_error::algorithms::MonopartiteAlgorithmError,
        },
        prelude::UndiGraph,
        traits::algorithms::planarity_detection::{
            preprocessing::LocalSimpleGraph, run_k23_homeomorph_engine,
        },
    };

    #[test]
    fn test_map_planarity_parallel_edge_error_to_k23_error() {
        let error: MonopartiteError<UndiGraph<usize>> = map_planarity_error_to_k23_homeomorph_error(
            crate::traits::algorithms::PlanarityError::ParallelEdgesUnsupported,
        );

        assert!(matches!(
            error,
            MonopartiteError::AlgorithmError(MonopartiteAlgorithmError::K23HomeomorphError(
                K23HomeomorphError::ParallelEdgesUnsupported
            ))
        ));
    }

    #[test]
    fn test_map_planarity_invalid_endpoint_error_to_k23_error() {
        let error: MonopartiteError<UndiGraph<usize>> = map_planarity_error_to_k23_homeomorph_error(
            crate::traits::algorithms::PlanarityError::InvalidEdgeEndpoint {
                endpoint: 9,
                node_count: 4,
            },
        );

        assert!(matches!(
            error,
            MonopartiteError::AlgorithmError(MonopartiteAlgorithmError::K23HomeomorphError(
                K23HomeomorphError::InvalidEdgeEndpoint { endpoint: 9, node_count: 4 }
            ))
        ));
    }

    #[test]
    fn test_run_k23_engine_detects_three_path_theta() {
        let graph =
            LocalSimpleGraph::from_edges(5, &[[0, 2], [2, 1], [0, 3], [3, 1], [0, 4], [4, 1]])
                .unwrap();

        assert!(run_k23_homeomorph_engine(&graph.preprocess()));
    }

    #[test]
    fn test_run_k23_engine_rejects_k4_pair() {
        let graph =
            LocalSimpleGraph::from_edges(4, &[[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]])
                .unwrap();

        assert!(!run_k23_homeomorph_engine(&graph.preprocess()));
    }
}
