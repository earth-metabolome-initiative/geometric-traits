//! Submodule providing the `DirectedLouvain` trait and its blanket
//! implementation for weighted directed graphs.
//!
//! Directed Louvain runs the same greedy multi-level skeleton as the undirected
//! [`Louvain`](super::Louvain) (it reuses [`LouvainConfig`] and
//! [`LouvainResult`] and the shared `louvain_levels` driver) but optimizes the
//! directed (Leicht-Newman / Dugue-Perez) modularity scored by
//! [`DirectedModularity`](super::DirectedModularity). It therefore accepts an
//! asymmetric matrix: arc direction informs the detected communities.

use num_traits::{AsPrimitive, ToPrimitive};

use super::{
    directed_community::DirectedWorkingGraph,
    louvain::{LouvainConfig, LouvainResult, louvain_levels},
    modularity::{ModularityError, validate_common_config},
};
use crate::traits::{Finite, Number, PositiveInteger, SparseValuedMatrix2D};

/// Trait providing the directed Louvain community detection algorithm.
///
/// The graph is a weighted, square (directed) matrix: entry `(i, j)` is the
/// weight of the arc from `i` to `j`. Weights must be finite and strictly
/// positive. Unlike [`Louvain`](super::Louvain), the matrix need not be
/// symmetric and self-loops are allowed.
pub trait DirectedLouvain<Marker: AsPrimitive<usize> + PositiveInteger = usize>:
    SparseValuedMatrix2D + Sized
where
    Self::RowIndex: AsPrimitive<usize>,
    Self::ColumnIndex: AsPrimitive<usize>,
    Self::Value: Number + ToPrimitive + Finite,
{
    /// Executes directed Louvain with the provided configuration.
    ///
    /// # Errors
    ///
    /// Returns an error when:
    /// - the configuration is invalid;
    /// - the matrix is not square;
    /// - at least one weight is non-finite or non-positive;
    /// - the resulting number of communities cannot fit into `Marker`.
    ///
    /// # Complexity
    ///
    /// O(L * P * (V + E)) time and O(V + E) space, where L is the number of
    /// coarsening levels, P the number of local-moving passes per level, V the
    /// number of nodes and E the number of arcs.
    ///
    /// # Examples
    ///
    /// ```
    /// use geometric_traits::{impls::ValuedCSR2D, prelude::*, traits::LouvainConfig};
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
    ///     DirectedLouvain::<usize>::directed_louvain(&edges, &LouvainConfig::default()).unwrap();
    /// let partition = result.final_partition();
    /// assert_eq!(partition[0], partition[1]);
    /// assert_eq!(partition[2], partition[3]);
    /// assert_ne!(partition[0], partition[2]);
    /// ```
    #[inline]
    fn directed_louvain(
        &self,
        config: &LouvainConfig,
    ) -> Result<LouvainResult<Marker>, ModularityError> {
        validate_common_config(
            config.resolution,
            config.modularity_threshold,
            config.max_levels,
            config.max_local_passes,
        )?;

        let graph = DirectedWorkingGraph::from_matrix(self)?;
        let original_number_of_nodes = self.number_of_rows().as_();
        louvain_levels(graph, config, original_number_of_nodes, &mut |_| {})
    }
}

impl<G, Marker> DirectedLouvain<Marker> for G
where
    G: SparseValuedMatrix2D + Sized,
    Marker: AsPrimitive<usize> + PositiveInteger,
    G::RowIndex: AsPrimitive<usize>,
    G::ColumnIndex: AsPrimitive<usize>,
    G::Value: Number + ToPrimitive + Finite,
{
}
