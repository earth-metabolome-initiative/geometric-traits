//! Submodule declaring `K_{3,3}` homeomorph detection for simple undirected
//! graphs.
//!
//! The current implementation is a boolean detector layered on the crate's
//! internal Boyer-style edge-addition embedding engine.

use num_traits::AsPrimitive;

use super::topology_wrapper_macros::define_planarity_derived_error;
use crate::traits::{
    UndirectedMonopartiteMonoplexGraph,
    algorithms::{
        PlanarityError,
        planarity_detection::{self, preprocessing::LocalSimpleGraph},
    },
};

define_planarity_derived_error!(
    /// Error type for `K_{3,3}` homeomorph detection.
    pub enum K33HomeomorphError => K33HomeomorphError,
    mapper = map_planarity_error_to_k33_homeomorph_error,
    self_loop = "The K33 homeomorph algorithm currently supports only simple undirected graphs and does not accept self-loops.",
    parallel = "The K33 homeomorph algorithm currently supports only simple undirected graphs and does not accept parallel edges."
);

/// Trait providing `K_{3,3}` homeomorph detection for simple undirected graphs.
pub trait K33HomeomorphDetection: UndirectedMonopartiteMonoplexGraph {
    /// Returns whether the graph contains a subgraph homeomorphic to `K_{3,3}`.
    ///
    /// The current implementation is a boolean detector driven by the crate's
    /// internal Boyer-style edge-addition embedding engine.
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
    ///     traits::{EdgesBuilder, K33HomeomorphDetection, VocabularyBuilder},
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
    /// let k33 = build_undigraph(
    ///     6,
    ///     &[(0, 3), (0, 4), (0, 5), (1, 3), (1, 4), (1, 5), (2, 3), (2, 4), (2, 5)],
    /// );
    /// let k5 = build_undigraph(
    ///     5,
    ///     &[(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)],
    /// );
    ///
    /// assert!(k33.has_k33_homeomorph()?);
    /// assert!(!k5.has_k33_homeomorph()?);
    /// # Ok::<(), geometric_traits::errors::MonopartiteError<UndiGraph<usize>>>(())
    /// ```
    #[inline]
    fn has_k33_homeomorph(&self) -> Result<bool, crate::errors::MonopartiteError<Self>>
    where
        Self: Sized,
        Self::NodeId: AsPrimitive<usize>,
    {
        has_k33_homeomorph_simple_undirected_graph(self)
            .map_err(map_planarity_error_to_k33_homeomorph_error::<Self>)
    }
}

impl<G: ?Sized + UndirectedMonopartiteMonoplexGraph> K33HomeomorphDetection for G {}

pub(crate) fn has_k33_homeomorph_simple_undirected_graph<G>(
    graph: &G,
) -> Result<bool, PlanarityError>
where
    G: UndirectedMonopartiteMonoplexGraph,
    G::NodeId: AsPrimitive<usize>,
{
    let simple_graph = LocalSimpleGraph::try_from_undirected_graph(graph)?;
    let preprocessing = simple_graph.preprocess();
    Ok(planarity_detection::run_k33_homeomorph_engine(&preprocessing))
}

#[cfg(test)]
mod tests {
    use super::{K33HomeomorphError, map_planarity_error_to_k33_homeomorph_error};
    use crate::{
        errors::{
            MonopartiteError, monopartite_graph_error::algorithms::MonopartiteAlgorithmError,
        },
        prelude::UndiGraph,
    };

    #[test]
    fn test_map_planarity_parallel_edge_error_to_k33_error() {
        let error: MonopartiteError<UndiGraph<usize>> = map_planarity_error_to_k33_homeomorph_error(
            crate::traits::algorithms::PlanarityError::ParallelEdgesUnsupported,
        );

        assert!(matches!(
            error,
            MonopartiteError::AlgorithmError(MonopartiteAlgorithmError::K33HomeomorphError(
                K33HomeomorphError::ParallelEdgesUnsupported
            ))
        ));
    }

    #[test]
    fn test_map_planarity_invalid_endpoint_error_to_k33_error() {
        let error: MonopartiteError<UndiGraph<usize>> = map_planarity_error_to_k33_homeomorph_error(
            crate::traits::algorithms::PlanarityError::InvalidEdgeEndpoint {
                endpoint: 9,
                node_count: 4,
            },
        );

        assert!(matches!(
            error,
            MonopartiteError::AlgorithmError(MonopartiteAlgorithmError::K33HomeomorphError(
                K33HomeomorphError::InvalidEdgeEndpoint { endpoint: 9, node_count: 4 }
            ))
        ));
    }
}
