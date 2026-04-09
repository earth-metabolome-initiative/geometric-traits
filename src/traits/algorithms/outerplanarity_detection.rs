//! Submodule declaring outerplanarity detection for simple undirected graphs.
//!
//! Outerplanarity is implemented as a mode of the same simple-undirected
//! edge-addition embedding engine used for planarity, plus the final
//! external-face coverage check required by the outerplanar contract.

use num_traits::AsPrimitive;

use crate::traits::{MonopartiteGraph, PlanarityDetection, UndirectedMonopartiteMonoplexGraph};

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
/// Error type for outerplanarity detection.
pub enum OuterplanarityError {
    /// The graph contains self-loops, which are unsupported by the intended
    /// simple undirected implementation.
    #[error(
        "The outerplanarity algorithm currently supports only simple undirected graphs and does not accept self-loops."
    )]
    SelfLoopsUnsupported,
    /// Parallel edges are unsupported by the intended public contract.
    #[error(
        "The outerplanarity algorithm currently supports only simple undirected graphs and does not accept parallel edges."
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

impl From<OuterplanarityError>
    for crate::errors::monopartite_graph_error::algorithms::MonopartiteAlgorithmError
{
    #[inline]
    fn from(error: OuterplanarityError) -> Self {
        Self::OuterplanarityError(error)
    }
}

impl<G: MonopartiteGraph> From<OuterplanarityError> for crate::errors::MonopartiteError<G> {
    #[inline]
    fn from(error: OuterplanarityError) -> Self {
        Self::AlgorithmError(error.into())
    }
}

#[allow(clippy::needless_pass_by_value)]
#[inline]
fn map_planarity_error_to_outerplanarity_error<G: MonopartiteGraph>(
    error: crate::traits::algorithms::PlanarityError,
) -> crate::errors::MonopartiteError<G> {
    match error {
        crate::traits::algorithms::PlanarityError::SelfLoopsUnsupported => {
            OuterplanarityError::SelfLoopsUnsupported.into()
        }
        crate::traits::algorithms::PlanarityError::ParallelEdgesUnsupported => {
            OuterplanarityError::ParallelEdgesUnsupported.into()
        }
        crate::traits::algorithms::PlanarityError::InvalidEdgeEndpoint { endpoint, node_count } => {
            OuterplanarityError::InvalidEdgeEndpoint { endpoint, node_count }.into()
        }
    }
}

/// Trait providing outerplanarity detection for simple undirected graphs.
pub trait OuterplanarityDetection: UndirectedMonopartiteMonoplexGraph + PlanarityDetection {
    /// Returns whether the graph is outerplanar.
    ///
    /// The implementation reuses the crate's internal simple-undirected
    /// edge-addition embedding engine in outerplanar mode and then checks that
    /// every primary vertex lies on the external face of the resulting
    /// embedding.
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
    ///     traits::{EdgesBuilder, OuterplanarityDetection, VocabularyBuilder},
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
    /// let outerplanar = build_undigraph(4, &[(0, 1), (1, 2), (2, 3), (0, 2), (1, 3)]);
    /// let planar_but_not_outerplanar =
    ///     build_undigraph(4, &[(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]);
    ///
    /// assert!(outerplanar.is_outerplanar()?);
    /// assert!(!planar_but_not_outerplanar.is_outerplanar()?);
    /// # Ok::<(), geometric_traits::errors::MonopartiteError<UndiGraph<usize>>>(())
    /// ```
    #[inline]
    fn is_outerplanar(&self) -> Result<bool, crate::errors::MonopartiteError<Self>>
    where
        Self: Sized,
        Self::NodeId: AsPrimitive<usize>,
    {
        crate::traits::algorithms::planarity_detection::is_outerplanar_simple_undirected_graph(self)
            .map_err(map_planarity_error_to_outerplanarity_error::<Self>)
    }
}

impl<G> OuterplanarityDetection for G where
    G: ?Sized + UndirectedMonopartiteMonoplexGraph + PlanarityDetection
{
}

#[cfg(test)]
mod tests {
    use super::{OuterplanarityError, map_planarity_error_to_outerplanarity_error};
    use crate::{
        errors::{
            MonopartiteError, monopartite_graph_error::algorithms::MonopartiteAlgorithmError,
        },
        prelude::UndiGraph,
        traits::algorithms::PlanarityError,
    };

    #[test]
    fn test_map_planarity_self_loop_error_to_outerplanarity_error() {
        let error: MonopartiteError<UndiGraph<usize>> =
            map_planarity_error_to_outerplanarity_error(PlanarityError::SelfLoopsUnsupported);

        assert!(matches!(
            error,
            MonopartiteError::AlgorithmError(MonopartiteAlgorithmError::OuterplanarityError(
                OuterplanarityError::SelfLoopsUnsupported
            ))
        ));
    }

    #[test]
    fn test_map_planarity_parallel_edge_error_to_outerplanarity_error() {
        let error: MonopartiteError<UndiGraph<usize>> =
            map_planarity_error_to_outerplanarity_error(PlanarityError::ParallelEdgesUnsupported);

        assert!(matches!(
            error,
            MonopartiteError::AlgorithmError(MonopartiteAlgorithmError::OuterplanarityError(
                OuterplanarityError::ParallelEdgesUnsupported
            ))
        ));
    }

    #[test]
    fn test_map_planarity_invalid_endpoint_error_to_outerplanarity_error() {
        let error: MonopartiteError<UndiGraph<usize>> =
            map_planarity_error_to_outerplanarity_error(PlanarityError::InvalidEdgeEndpoint {
                endpoint: 9,
                node_count: 4,
            });

        assert!(matches!(
            error,
            MonopartiteError::AlgorithmError(MonopartiteAlgorithmError::OuterplanarityError(
                OuterplanarityError::InvalidEdgeEndpoint { endpoint: 9, node_count: 4 }
            ))
        ));
    }
}
