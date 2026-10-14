//! Undirected Newman-Girvan modularity (Newman and Girvan, 2004, "Finding and
//! evaluating community structure in networks", Phys. Rev. E 69, 026113) and
//! the [`UndirectedModularity`] trait that evaluates it on a weighted,
//! symmetric square matrix and a partition.
//!
//! ```text
//! Q = (1 / 2m) * sum_ij [ A_ij - gamma * k_i * k_j / (2m) ] * delta(c_i, c_j)
//! ```
//!
//! Here `m` is the total edge weight, `A_ij` the symmetric weight, and `k_i`
//! the weighted degree of node `i`. This is the undirected twin of
//! [`DirectedModularity`](super::DirectedModularity) and matches the modularity
//! that [`Louvain`](super::Louvain) and [`Leiden`](super::Leiden) report for
//! the partitions they discover.

use alloc::vec::Vec;

use num_traits::{AsPrimitive, ToPrimitive};

use super::modularity::{ModularityError, UndirectedView, build_working_graph};
use crate::traits::{Finite, Number, SparseValuedMatrix2D};

/// Evaluates the undirected (Newman-Girvan) modularity of a partition on a
/// weighted, symmetric square (undirected) graph.
///
/// Entry `(i, j)` of the matrix is the weight of the undirected edge between
/// `i` and `j`. Weights must be finite, strictly positive, and symmetric, like
/// the input to [`Louvain`](super::Louvain).
pub trait UndirectedModularity<Marker: AsPrimitive<usize> = usize>:
    SparseValuedMatrix2D + Sized
where
    Self::RowIndex: AsPrimitive<usize>,
    Self::ColumnIndex: AsPrimitive<usize>,
    Self::Value: Number + ToPrimitive + Finite,
{
    /// Returns the undirected (Newman-Girvan) modularity of `partition`, where
    /// `partition[i]` is the community id of node `i`.
    ///
    /// The value matches the modularity that [`Louvain`](super::Louvain) and
    /// [`Leiden`](super::Leiden) report for a partition they discover.
    ///
    /// # Errors
    ///
    /// Returns an error when the resolution is not finite and strictly
    /// positive, the matrix is not square or not symmetric, a weight is
    /// non-finite, unrepresentable as `f64`, or non-positive, or
    /// `partition.len()` does not equal the number of nodes.
    ///
    /// # Examples
    ///
    /// ```
    /// use geometric_traits::{impls::ValuedCSR2D, prelude::*};
    ///
    /// // Two disjoint edges: {0,1} and {2,3}.
    /// let edges: ValuedCSR2D<usize, usize, usize, f64> =
    ///     GenericEdgesBuilder::<_, ValuedCSR2D<usize, usize, usize, f64>>::default()
    ///         .expected_number_of_edges(4)
    ///         .expected_shape((4, 4))
    ///         .edges(vec![(0, 1, 1.0), (1, 0, 1.0), (2, 3, 1.0), (3, 2, 1.0)].into_iter())
    ///         .build()
    ///         .unwrap();
    ///
    /// let q =
    ///     UndirectedModularity::<usize>::undirected_modularity(&edges, &[0, 0, 1, 1], 1.0).unwrap();
    /// assert!((q - 0.5).abs() < 1e-12);
    /// ```
    #[inline]
    fn undirected_modularity(
        &self,
        partition: &[Marker],
        resolution: f64,
    ) -> Result<f64, ModularityError> {
        if !resolution.is_finite() || resolution <= 0.0 {
            return Err(ModularityError::InvalidResolution);
        }

        let graph = build_working_graph(self)?;
        let view = UndirectedView::from_working_graph(&graph);
        let number_of_nodes = view.number_of_nodes();
        if partition.len() != number_of_nodes {
            return Err(ModularityError::PartitionLengthMismatch {
                expected: number_of_nodes,
                found: partition.len(),
            });
        }

        let usize_partition: Vec<usize> =
            partition.iter().map(|community| community.as_()).collect();
        Ok(view.modularity(&usize_partition, resolution))
    }
}

impl<G, Marker> UndirectedModularity<Marker> for G
where
    G: SparseValuedMatrix2D + Sized,
    Marker: AsPrimitive<usize>,
    G::RowIndex: AsPrimitive<usize>,
    G::ColumnIndex: AsPrimitive<usize>,
    G::Value: Number + ToPrimitive + Finite,
{
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use super::{
        super::modularity::{UndirectedView, build_working_graph},
        ModularityError, UndirectedModularity,
    };
    use crate::{impls::ValuedCSR2D, naive_structs::GenericEdgesBuilder, traits::EdgesBuilder};

    type WeightedMatrix = ValuedCSR2D<usize, usize, usize, f64>;

    const TOLERANCE: f64 = 1e-12;

    /// Builds a square matrix from the given directed entries verbatim, so
    /// asymmetric inputs can be exercised.
    fn build(node_count: usize, edges: Vec<(usize, usize, f64)>) -> WeightedMatrix {
        let mut edges = edges;
        edges.sort_unstable_by(|(ls, ld, _), (rs, rd, _)| (ls, ld).cmp(&(rs, rd)));
        GenericEdgesBuilder::<_, WeightedMatrix>::default()
            .expected_number_of_edges(edges.len())
            .expected_shape((node_count, node_count))
            .edges(edges.into_iter())
            .build()
            .unwrap()
    }

    /// Mirrors every undirected edge into both directions, yielding a symmetric
    /// matrix.
    fn build_undirected(node_count: usize, edges: &[(usize, usize, f64)]) -> WeightedMatrix {
        let mut directed = Vec::with_capacity(edges.len() * 2);
        for &(source, destination, weight) in edges {
            directed.push((source, destination, weight));
            if source != destination {
                directed.push((destination, source, weight));
            }
        }
        build(node_count, directed)
    }

    /// Modularity computed straight from the private view, the value the trait
    /// must reproduce.
    fn view_modularity(matrix: &WeightedMatrix, partition: &[usize], resolution: f64) -> f64 {
        let graph = build_working_graph(matrix).unwrap();
        UndirectedView::from_working_graph(&graph).modularity(partition, resolution)
    }

    #[test]
    fn test_two_disjoint_edges() {
        let matrix = build_undirected(4, &[(0, 1, 1.0), (2, 3, 1.0)]);
        // Correct partition: 0.5; single community: 0.0; singletons: -0.25.
        assert!((view_modularity(&matrix, &[0, 0, 1, 1], 1.0) - 0.5).abs() < TOLERANCE);
        assert!(view_modularity(&matrix, &[0, 0, 0, 0], 1.0).abs() < TOLERANCE);
        assert!((view_modularity(&matrix, &[0, 1, 2, 3], 1.0) - (-0.25)).abs() < TOLERANCE);
    }

    #[test]
    fn test_single_edge() {
        let matrix = build_undirected(2, &[(0, 1, 1.0)]);
        assert!(view_modularity(&matrix, &[0, 0], 1.0).abs() < TOLERANCE);
        // Singletons: 2 * (0 - (1/2)^2) = -0.5.
        assert!((view_modularity(&matrix, &[0, 1], 1.0) - (-0.5)).abs() < TOLERANCE);
    }

    #[test]
    fn test_resolution_scales_expected_term() {
        let matrix = build_undirected(4, &[(0, 1, 1.0), (2, 3, 1.0)]);
        // With gamma=2 each community contributes 0.25 - 2*0.0625 ... = 0 ->
        // total 0.
        assert!(view_modularity(&matrix, &[0, 0, 1, 1], 2.0).abs() < TOLERANCE);
    }

    #[test]
    fn test_trait_matches_view() {
        let matrix = build_undirected(4, &[(0, 1, 1.0), (1, 2, 2.0), (2, 3, 1.0), (3, 0, 2.0)]);
        for partition in [vec![0, 0, 1, 1], vec![0, 1, 0, 1], vec![0, 0, 0, 0]] {
            let trait_value =
                UndirectedModularity::<usize>::undirected_modularity(&matrix, &partition, 1.0)
                    .unwrap();
            let view_value = view_modularity(&matrix, &partition, 1.0);
            assert!((trait_value - view_value).abs() < TOLERANCE);
        }
    }

    #[test]
    fn test_trait_validates_partition_length() {
        let matrix = build_undirected(4, &[(0, 1, 1.0), (2, 3, 1.0)]);
        let error = UndirectedModularity::<usize>::undirected_modularity(&matrix, &[0, 0, 1], 1.0)
            .unwrap_err();
        assert!(matches!(
            error,
            ModularityError::PartitionLengthMismatch { expected: 4, found: 3 }
        ));
    }

    #[test]
    fn test_trait_rejects_invalid_resolution() {
        let matrix = build_undirected(2, &[(0, 1, 1.0)]);
        for resolution in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            let error =
                UndirectedModularity::<usize>::undirected_modularity(&matrix, &[0, 0], resolution)
                    .unwrap_err();
            assert!(matches!(error, ModularityError::InvalidResolution));
        }
    }

    #[test]
    fn test_trait_rejects_non_symmetric_matrix() {
        // Only the forward edge exists, so the matrix is not symmetric.
        let matrix = build(2, vec![(0, 1, 1.0)]);
        let error = UndirectedModularity::<usize>::undirected_modularity(&matrix, &[0, 0], 1.0)
            .unwrap_err();
        assert!(matches!(error, ModularityError::NonSymmetricEdge { .. }));
    }

    #[test]
    fn test_trait_rejects_non_square_matrix() {
        let matrix: WeightedMatrix = GenericEdgesBuilder::<_, WeightedMatrix>::default()
            .expected_number_of_edges(1)
            .expected_shape((2, 3))
            .edges(vec![(0, 1, 1.0)].into_iter())
            .build()
            .unwrap();
        let error = UndirectedModularity::<usize>::undirected_modularity(&matrix, &[0, 0], 1.0)
            .unwrap_err();
        assert!(matches!(error, ModularityError::NonSquareMatrix { rows: 2, columns: 3 }));
    }
}
