//! Submodule providing the [`SymmetricForest`] type, the undirected symmetric
//! view of a weighted forest.

#[cfg(feature = "mem_dbg")]
use alloc::{string::String, vec::Vec};
use core::{fmt::Debug, ops::Deref};

use multi_ranged::Step;
use num_traits::AsPrimitive;

use crate::{
    impls::{SymmetricCSR2D, ValuedCSR2D},
    naive_structs::GenericGraph,
    traits::{BidirectionalVocabulary, PositiveInteger, TryFromUsize},
};

/// Type alias for the inner symmetric valued CSR matrix of a forest.
///
/// The index type `T` must satisfy the matrix index bounds required by the
/// underlying [`SymmetricCSR2D`] and [`ValuedCSR2D`] types.
type ForestMatrix<T, F> = SymmetricCSR2D<ValuedCSR2D<T, T, T, F>>;

#[cfg_attr(feature = "mem_size", derive(mem_dbg::MemSize))]
#[cfg_attr(feature = "mem_size", mem_size(rec))]
#[cfg_attr(feature = "mem_dbg", derive(mem_dbg::MemDbg))]
#[derive(Clone, PartialEq, Eq, Hash)]
/// The undirected symmetric view of a weighted forest.
///
/// This wraps a [`SymmetricCSR2D`] storing one undirected weighted edge per
/// tree edge. It dereferences to the inner symmetric matrix, so layout
/// algorithms such as `force_atlas2` (blanket implemented for any
/// `SparseValuedMatrix2D`) resolve directly on the wrapper. It also exposes
/// [`as_graph`](Self::as_graph) to obtain a full undirected graph view for
/// component and distance queries.
pub struct SymmetricForest<T: Step + PositiveInteger + AsPrimitive<usize> + TryFromUsize, F> {
    /// The underlying symmetric valued adjacency matrix.
    matrix: ForestMatrix<T, F>,
}

impl<T: Step + PositiveInteger + AsPrimitive<usize> + TryFromUsize + Debug, F: Debug> Debug
    for SymmetricForest<T, F>
{
    #[inline]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("SymmetricForest").field("matrix", &self.matrix).finish()
    }
}

impl<T: Step + PositiveInteger + AsPrimitive<usize> + TryFromUsize, F> SymmetricForest<T, F> {
    #[inline]
    #[must_use]
    /// Builds a symmetric forest from its underlying symmetric matrix.
    pub fn from_parts(matrix: ForestMatrix<T, F>) -> Self {
        Self { matrix }
    }

    #[inline]
    #[must_use]
    /// Returns a reference to the underlying symmetric matrix.
    pub fn as_matrix(&self) -> &ForestMatrix<T, F> {
        &self.matrix
    }

    #[inline]
    #[must_use]
    /// Consumes the wrapper and returns the underlying symmetric matrix.
    pub fn into_matrix(self) -> ForestMatrix<T, F> {
        self.matrix
    }
}

impl<T: Step + PositiveInteger + AsPrimitive<usize> + TryFromUsize, F> Deref
    for SymmetricForest<T, F>
{
    type Target = ForestMatrix<T, F>;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.matrix
    }
}

impl<T, F> SymmetricForest<T, F>
where
    T: Step
        + PositiveInteger
        + AsPrimitive<usize>
        + TryFromUsize
        + BidirectionalVocabulary<SourceSymbol = T, DestinationSymbol = T>,
    F: Clone,
{
    #[inline]
    #[must_use]
    /// Returns an undirected graph view of the forest.
    ///
    /// The node identifiers form the identity vocabulary `0..order`, and the
    /// edges are the inner symmetric matrix. The returned [`GenericGraph`]
    /// implements the full undirected monopartite graph stack, so callers can
    /// run `connected_components`, `diameter`, and similar algorithms on it.
    pub fn as_graph(&self) -> GenericGraph<T, ForestMatrix<T, F>>
    where
        ForestMatrix<T, F>: Clone,
    {
        let order = crate::traits::SquareMatrix::order(&self.matrix);
        GenericGraph::from((order, self.matrix.clone()))
    }
}

#[cfg(all(test, feature = "alloc"))]
mod tests {
    use alloc::vec::Vec;

    use crate::{
        impls::WeightedForest,
        traits::{
            ConnectedComponents, ForceAtlas2, ForceAtlas2Config, SizedSparseMatrix, SparseMatrix2D,
            SparseSquareMatrix, SparseValuedMatrix2D, SquareMatrix,
        },
    };

    type TestWeightedForest = WeightedForest<usize, f64>;

    fn two_component_forest() -> TestWeightedForest {
        // Roots are nodes 0 and 3. Tree A: 0 <- {1 (w 5) <- 2 (w 7)}.
        // Tree B: 3 <- 4 (w 9).
        WeightedForest::from_parent_array(vec![(0, 0.0), (0, 5.0), (1, 7.0), (3, 0.0), (3, 9.0)])
    }

    #[test]
    fn test_symmetrize_is_symmetric() {
        let forest = two_component_forest();
        let symmetric = forest.symmetrize();
        assert!(symmetric.as_matrix().is_symmetric());
        // Two directed entries per undirected edge: 2 * (n - roots) = 2 * 3.
        assert_eq!(symmetric.as_matrix().number_of_defined_values(), 6);
    }

    #[test]
    fn test_symmetrize_edge_set_matches_forest() {
        let forest = two_component_forest();
        let symmetric = forest.symmetrize();

        let forest_edges: Vec<(usize, usize, f64)> = forest.edge_iter().collect();

        // Collect the upper-triangular entries of the symmetric matrix.
        let mut symmetric_edges: Vec<(usize, usize, f64)> = Vec::new();
        for row in 0..symmetric.as_matrix().order() {
            for (column, weight) in symmetric
                .as_matrix()
                .sparse_row(row)
                .zip(symmetric.as_matrix().sparse_row_values(row))
            {
                if row <= column {
                    symmetric_edges.push((row, column, weight));
                }
            }
        }

        assert_eq!(symmetric_edges, forest_edges);
    }

    #[test]
    fn test_force_atlas2_runs() {
        let forest = two_component_forest();
        let symmetric = forest.symmetrize();

        // Resolves through Deref to the inner symmetric matrix.
        let result = symmetric.force_atlas2(&ForceAtlas2Config::default()).unwrap();
        assert_eq!(result.num_points(), forest.number_of_nodes());
        assert!(result.coordinates_flat().iter().all(|value| value.is_finite()));
    }

    #[test]
    fn test_connected_components_match_forest() {
        let forest = two_component_forest();
        let symmetric = forest.symmetrize();
        let graph = symmetric.as_graph();
        let components = ConnectedComponents::<usize>::connected_components(&graph)
            .expect("components computed");
        assert_eq!(components.number_of_components(), forest.number_of_components());
    }

    #[test]
    fn test_into_matrix_preserves_adjacency() {
        let forest = two_component_forest();
        let symmetric = forest.symmetrize();
        let defined = symmetric.as_matrix().number_of_defined_values();
        let matrix = symmetric.into_matrix();
        assert!(matrix.is_symmetric());
        assert_eq!(matrix.number_of_defined_values(), defined);
    }

    #[test]
    fn test_debug_contains_type_name() {
        let forest = two_component_forest();
        let symmetric = forest.symmetrize();
        let debug = alloc::format!("{symmetric:?}");
        assert!(debug.contains("SymmetricForest"));
    }
}
