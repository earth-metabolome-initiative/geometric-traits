//! Submodule providing the `DirectedLeiden` trait and its blanket
//! implementation for weighted directed graphs.
//!
//! Directed Leiden runs the same greedy multi-level skeleton as the undirected
//! [`Leiden`](super::Leiden), including the refinement phase and the
//! well-connectedness guarantee (it reuses [`LeidenConfig`] and
//! [`LeidenResult`] and the shared `leiden_levels` driver), but optimizes the
//! directed (Leicht-Newman / Dugue-Perez) modularity scored by
//! [`DirectedModularity`](super::DirectedModularity). It therefore accepts an
//! asymmetric matrix, and each detected community is weakly connected.
//! [`acyclic_directed_leiden`](DirectedLeiden::acyclic_directed_leiden) further
//! keeps the graph contracting each community to a node acyclic.

use num_traits::{AsPrimitive, ToPrimitive};

use super::{
    directed_community::{AcyclicDirectedWorkingGraph, DirectedWorkingGraph},
    leiden::{LeidenConfig, LeidenResult, leiden_levels},
    modularity::{ModularityError, singleton_members},
};
use crate::traits::{Finite, Number, PositiveInteger, SparseValuedMatrix2D};

/// Trait providing the directed Leiden community detection algorithm.
///
/// The graph is a weighted, square (directed) matrix: entry `(i, j)` is the
/// weight of the arc from `i` to `j`. Weights must be finite and strictly
/// positive. Unlike [`Leiden`](super::Leiden), the matrix need not be symmetric
/// and self-loops are allowed. Each detected community is weakly connected.
pub trait DirectedLeiden<Marker: AsPrimitive<usize> + PositiveInteger = usize>:
    SparseValuedMatrix2D + Sized
where
    Self::RowIndex: AsPrimitive<usize>,
    Self::ColumnIndex: AsPrimitive<usize>,
    Self::Value: Number + ToPrimitive + Finite,
{
    /// Executes directed Leiden with the provided configuration.
    ///
    /// # Errors
    ///
    /// Returns an error when:
    /// - the configuration is invalid;
    /// - the matrix is not square;
    /// - at least one weight is non-finite or non-positive;
    /// - the resulting number of communities cannot fit into `Marker`.
    ///
    /// # Examples
    ///
    /// ```
    /// use geometric_traits::{impls::ValuedCSR2D, prelude::*, traits::LeidenConfig};
    ///
    /// // Two disjoint directed 2-cycles: {0,1} and {2,3}.
    /// let edges: ValuedCSR2D<usize, usize, usize, f64> =
    ///     GenericEdgesBuilder::<_, ValuedCSR2D<usize, usize, usize, f64>>::default()
    ///         .expected_number_of_edges(4)
    ///         .expected_shape((4, 4))
    ///         .edges(vec![(0, 1, 1.0), (1, 0, 1.0), (2, 3, 1.0), (3, 2, 1.0)].into_iter())
    ///         .build()
    ///         .unwrap();
    ///
    /// let result =
    ///     DirectedLeiden::<usize>::directed_leiden(&edges, &LeidenConfig::default()).unwrap();
    /// let partition = result.final_partition();
    /// assert_eq!(partition[0], partition[1]);
    /// assert_eq!(partition[2], partition[3]);
    /// assert_ne!(partition[0], partition[2]);
    /// ```
    #[inline]
    fn directed_leiden(
        &self,
        config: &LeidenConfig,
    ) -> Result<LeidenResult<Marker>, ModularityError> {
        config.validate()?;
        let graph = DirectedWorkingGraph::from_matrix(self)?;
        leiden_levels(graph, config, singleton_members(self.number_of_rows().as_()), &mut |_| {})
    }

    /// Executes directed Leiden restricted to partitions whose contraction is
    /// acyclic.
    ///
    /// Contracting each community to one node, and keeping every arc between
    /// distinct communities, gives an acyclic graph at every level of the
    /// result, so the communities can be ordered by their dependencies. Nodes
    /// on a common directed cycle, that is in the same strongly connected
    /// component, always share a community, and each community is weakly
    /// connected.
    ///
    /// Local moving and refinement start from the condensation of the graph
    /// and skip every move that would close a cycle in the contraction.
    ///
    /// # Errors
    ///
    /// As [`directed_leiden`](Self::directed_leiden).
    ///
    /// # Examples
    ///
    /// ```
    /// use geometric_traits::{impls::ValuedCSR2D, prelude::*, traits::LeidenConfig};
    ///
    /// // A DAG on which `directed_leiden` returns the communities {0, 3} and
    /// // {1, 2, 4}, which the arcs 1 -> 0 and 3 -> 1 join into a cycle.
    /// let edges: ValuedCSR2D<usize, usize, usize, f64> = GenericEdgesBuilder::<
    ///     _,
    ///     ValuedCSR2D<usize, usize, usize, f64>,
    /// >::default()
    /// .expected_number_of_edges(5)
    /// .expected_shape((5, 5))
    /// .edges(vec![(1, 0, 1.0), (2, 1, 1.0), (3, 0, 1.0), (3, 1, 1.0), (4, 1, 1.0)].into_iter())
    /// .build()
    /// .unwrap();
    ///
    /// let result =
    ///     DirectedLeiden::<usize>::acyclic_directed_leiden(&edges, &LeidenConfig::default()).unwrap();
    /// assert_eq!(result.final_partition(), [0, 0, 1, 2, 3]);
    /// ```
    #[inline]
    fn acyclic_directed_leiden(
        &self,
        config: &LeidenConfig,
    ) -> Result<LeidenResult<Marker>, ModularityError> {
        config.validate()?;
        let graph = DirectedWorkingGraph::from_matrix(self)?;
        let (condensation, members) = AcyclicDirectedWorkingGraph::condense(&graph)?;
        leiden_levels(condensation, config, members, &mut |_| {})
    }
}

impl<G, Marker> DirectedLeiden<Marker> for G
where
    G: SparseValuedMatrix2D + Sized,
    Marker: AsPrimitive<usize> + PositiveInteger,
    G::RowIndex: AsPrimitive<usize>,
    G::ColumnIndex: AsPrimitive<usize>,
    G::Value: Number + ToPrimitive + Finite,
{
}
