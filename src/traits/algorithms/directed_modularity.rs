//! Directed generalization of modularity (Leicht and Newman, 2008, "Community
//! Structure in Directed Networks", arXiv:0709.4500) and the
//! [`DirectedModularity`] trait that evaluates it on a weighted square matrix
//! and a partition.
//!
//! The directed formula keeps the out-strength and in-strength of each node
//! separate, so edge direction carries community information:
//!
//! ```text
//! Q = (1 / m) * sum_ij [ A_ij - gamma * k_i^out * k_j^in / m ] * delta(c_i, c_j)
//! ```
//!
//! Here `m` is the total edge weight, `k_i^out` the row sum of node `i`, and
//! `k_j^in` the column sum of node `j`. This matches the directed branch of
//! NetworkX `community.modularity`.

use alloc::vec::Vec;

use num_traits::{AsPrimitive, ToPrimitive};

use super::modularity::ModularityError;
use crate::traits::{Finite, Number, SparseValuedMatrix2D};

/// Read-only view over a weighted directed graph, shared by the directed
/// modularity metric and the Leicht-Newman detector.
///
/// Edges are always read from the underlying matrix through the
/// [`SparseValuedMatrix2D`] trait. Only the per-node strengths and the total
/// weight are cached, because the spectral method reuses them in every
/// iteration. Self-loops are allowed and no symmetry is required.
pub(crate) struct DirectedView<'matrix, M: SparseValuedMatrix2D> {
    matrix: &'matrix M,
    /// Row handle of each node, indexed by node id.
    row_handles: Vec<M::RowIndex>,
    /// Out-strength (sum of outgoing edge weights) of each node.
    pub(crate) out_degree: Vec<f64>,
    /// In-strength (sum of incoming edge weights) of each node.
    pub(crate) in_degree: Vec<f64>,
    /// Total edge weight `m = sum_ij A_ij`.
    pub(crate) total_weight: f64,
}

impl<'matrix, M> DirectedView<'matrix, M>
where
    M: SparseValuedMatrix2D,
    M::RowIndex: AsPrimitive<usize>,
    M::ColumnIndex: AsPrimitive<usize>,
    M::Value: ToPrimitive + Finite,
{
    /// Builds a view over `matrix`, validating that it is square and that every
    /// weight is finite, representable as `f64`, and strictly positive.
    pub(crate) fn new(matrix: &'matrix M) -> Result<Self, ModularityError> {
        let rows = matrix.number_of_rows().as_();
        let columns = matrix.number_of_columns().as_();
        if rows != columns {
            return Err(ModularityError::NonSquareMatrix { rows, columns });
        }

        let mut row_handles = Vec::with_capacity(rows);
        let mut out_degree = vec![0.0; rows];
        let mut in_degree = vec![0.0; rows];

        for row_id in matrix.row_indices() {
            let source = row_id.as_();
            // The algorithm indexes per-node state by node id, so it relies on
            // `row_indices()` yielding the contiguous range `0..rows` in order,
            // making `row_handles[node]` the row whose id is `node`.
            debug_assert_eq!(source, row_handles.len(), "node ids must be contiguous and ordered");
            row_handles.push(row_id);
            for (column_id, weight) in
                matrix.sparse_row(row_id).zip(matrix.sparse_row_values(row_id))
            {
                let destination = column_id.as_();
                if !weight.is_finite() {
                    return Err(ModularityError::NonFiniteWeight {
                        source_id: source,
                        destination_id: destination,
                    });
                }
                let weight = weight.to_f64().ok_or(ModularityError::UnrepresentableWeight {
                    source_id: source,
                    destination_id: destination,
                })?;
                if !weight.is_finite() {
                    return Err(ModularityError::NonFiniteWeight {
                        source_id: source,
                        destination_id: destination,
                    });
                }
                if weight <= 0.0 {
                    return Err(ModularityError::NonPositiveWeight {
                        source_id: source,
                        destination_id: destination,
                    });
                }
                out_degree[source] += weight;
                in_degree[destination] += weight;
            }
        }

        let total_weight = out_degree.iter().sum();
        Ok(Self { matrix, row_handles, out_degree, in_degree, total_weight })
    }

    /// Number of nodes in the view.
    pub(crate) fn number_of_nodes(&self) -> usize {
        self.row_handles.len()
    }

    /// Visits every outgoing edge of `node` as a `(destination, weight)` pair,
    /// reading the row straight from the matrix.
    pub(crate) fn for_each_out_edge<F: FnMut(usize, f64)>(&self, node: usize, mut visit: F) {
        let handle = self.row_handles[node];
        for (column, value) in
            self.matrix.sparse_row(handle).zip(self.matrix.sparse_row_values(handle))
        {
            if let Some(weight) = value.to_f64() {
                visit(column.as_(), weight);
            }
        }
    }

    /// Directed modularity of `partition`, whose entries must be community ids
    /// in `0..number_of_communities`.
    pub(crate) fn modularity(&self, partition: &[usize], resolution: f64) -> f64 {
        if self.total_weight <= 0.0 || !self.total_weight.is_normal() {
            return 0.0;
        }

        let number_of_communities = partition.iter().copied().max().map_or(0, |max| max + 1);
        let mut internal_weight = vec![0.0; number_of_communities];
        let mut out_strength = vec![0.0; number_of_communities];
        let mut in_strength = vec![0.0; number_of_communities];

        for (node, community) in partition.iter().copied().enumerate() {
            out_strength[community] += self.out_degree[node];
            in_strength[community] += self.in_degree[node];
            self.for_each_out_edge(node, |destination, weight| {
                if partition[destination] == community {
                    internal_weight[community] += weight;
                }
            });
        }

        let inverse_total_weight = 1.0 / self.total_weight;
        (0..number_of_communities).fold(0.0, |modularity, community| {
            let internal_fraction = internal_weight[community] * inverse_total_weight;
            let expected_fraction = resolution
                * (out_strength[community] * inverse_total_weight)
                * (in_strength[community] * inverse_total_weight);
            modularity + internal_fraction - expected_fraction
        })
    }
}

/// Evaluates the directed (Leicht-Newman) modularity of a partition on a
/// weighted, square (directed) graph.
///
/// Entry `(i, j)` of the matrix is the weight of the directed edge from `i` to
/// `j`. Weights must be finite and strictly positive. Unlike
/// [`Louvain`](super::Louvain), the matrix need not be symmetric.
pub trait DirectedModularity<Marker: AsPrimitive<usize> = usize>:
    SparseValuedMatrix2D + Sized
where
    Self::RowIndex: AsPrimitive<usize>,
    Self::ColumnIndex: AsPrimitive<usize>,
    Self::Value: Number + ToPrimitive + Finite,
{
    /// Returns the directed modularity of `partition`, where `partition[i]` is
    /// the community id of node `i`.
    ///
    /// The value matches the directed branch of NetworkX
    /// `community.modularity`.
    ///
    /// # Errors
    ///
    /// Returns an error when the resolution is not finite and strictly
    /// positive, the matrix is not square, a weight is non-finite,
    /// unrepresentable as `f64`, or non-positive, or `partition.len()` does not
    /// equal the number of nodes.
    ///
    /// # Examples
    ///
    /// ```
    /// use geometric_traits::{impls::ValuedCSR2D, prelude::*};
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
    /// let q = DirectedModularity::<usize>::directed_modularity(&edges, &[0, 0, 1, 1], 1.0).unwrap();
    /// assert!((q - 0.5).abs() < 1e-12);
    /// ```
    #[inline]
    fn directed_modularity(
        &self,
        partition: &[Marker],
        resolution: f64,
    ) -> Result<f64, ModularityError> {
        if !resolution.is_finite() || resolution <= 0.0 {
            return Err(ModularityError::InvalidResolution);
        }

        let view = DirectedView::new(self)?;
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

impl<G, Marker> DirectedModularity<Marker> for G
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

    use super::{DirectedModularity, DirectedView, ModularityError};
    use crate::{impls::ValuedCSR2D, naive_structs::GenericEdgesBuilder, traits::EdgesBuilder};

    type WeightedMatrix = ValuedCSR2D<usize, usize, usize, f64>;

    const TOLERANCE: f64 = 1e-12;

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

    fn modularity(matrix: &WeightedMatrix, partition: &[usize], resolution: f64) -> f64 {
        DirectedView::new(matrix).unwrap().modularity(partition, resolution)
    }

    #[test]
    fn test_single_directed_edge_is_zero_for_both_partitions() {
        let matrix = build(2, vec![(0, 1, 1.0)]);
        assert!(modularity(&matrix, &[0, 0], 1.0).abs() < TOLERANCE);
        assert!(modularity(&matrix, &[0, 1], 1.0).abs() < TOLERANCE);
    }

    #[test]
    fn test_bidirectional_pair_single_community_is_zero() {
        let matrix = build(2, vec![(0, 1, 1.0), (1, 0, 1.0)]);
        assert!(modularity(&matrix, &[0, 0], 1.0).abs() < TOLERANCE);
        // Singletons: -[ (1*1)/4 + (1*1)/4 ] = -0.5.
        assert!((modularity(&matrix, &[0, 1], 1.0) - (-0.5)).abs() < TOLERANCE);
    }

    #[test]
    fn test_two_disjoint_directed_cycles() {
        let matrix = build(4, vec![(0, 1, 1.0), (1, 0, 1.0), (2, 3, 1.0), (3, 2, 1.0)]);
        // Correct partition: 0.5; single community: 0.0.
        assert!((modularity(&matrix, &[0, 0, 1, 1], 1.0) - 0.5).abs() < TOLERANCE);
        assert!(modularity(&matrix, &[0, 0, 0, 0], 1.0).abs() < TOLERANCE);
    }

    #[test]
    fn test_self_loop_is_handled() {
        // 0->0 (w=2), 0->1 (w=1), 1->0 (w=1). m=4.
        let matrix = build(2, vec![(0, 0, 2.0), (0, 1, 1.0), (1, 0, 1.0)]);
        assert!(modularity(&matrix, &[0, 0], 1.0).abs() < TOLERANCE);
        // [0,1]: [2/4 - (3*3)/16] + [0 - (1*1)/16] = -0.125.
        assert!((modularity(&matrix, &[0, 1], 1.0) - (-0.125)).abs() < TOLERANCE);
    }

    #[test]
    fn test_resolution_scales_expected_term() {
        let matrix = build(4, vec![(0, 1, 1.0), (1, 0, 1.0), (2, 3, 1.0), (3, 2, 1.0)]);
        // With gamma=2: each community 0.5 - 2*0.25 = 0.0 -> total 0.0.
        assert!(modularity(&matrix, &[0, 0, 1, 1], 2.0).abs() < TOLERANCE);
    }

    #[test]
    fn test_trait_matches_view_and_validates_length() {
        let matrix = build(4, vec![(0, 1, 1.0), (1, 0, 1.0), (2, 3, 1.0), (3, 2, 1.0)]);
        let q =
            DirectedModularity::<usize>::directed_modularity(&matrix, &[0, 0, 1, 1], 1.0).unwrap();
        assert!((q - 0.5).abs() < TOLERANCE);

        let error =
            DirectedModularity::<usize>::directed_modularity(&matrix, &[0, 0, 1], 1.0).unwrap_err();
        assert!(matches!(
            error,
            ModularityError::PartitionLengthMismatch { expected: 4, found: 3 }
        ));
    }

    #[test]
    fn test_trait_rejects_invalid_resolution() {
        let matrix = build(2, vec![(0, 1, 1.0), (1, 0, 1.0)]);
        let error =
            DirectedModularity::<usize>::directed_modularity(&matrix, &[0, 0], 0.0).unwrap_err();
        assert!(matches!(error, ModularityError::InvalidResolution));
    }

    #[test]
    fn test_view_rejects_non_square() {
        let matrix: WeightedMatrix = GenericEdgesBuilder::<_, WeightedMatrix>::default()
            .expected_number_of_edges(1)
            .expected_shape((2, 3))
            .edges(vec![(0, 1, 1.0)].into_iter())
            .build()
            .unwrap();
        assert!(matches!(
            DirectedView::new(&matrix),
            Err(ModularityError::NonSquareMatrix { rows: 2, columns: 3 })
        ));
    }

    #[test]
    fn test_view_rejects_non_positive_weight() {
        let matrix = build(2, vec![(0, 1, -1.0)]);
        assert!(matches!(
            DirectedView::new(&matrix),
            Err(ModularityError::NonPositiveWeight { .. })
        ));
    }

    #[test]
    fn test_view_keeps_directed_degrees_separate() {
        // 0->1 only: out_degree=[3,0], in_degree=[0,3].
        let matrix = build(2, vec![(0, 1, 3.0)]);
        let view = DirectedView::new(&matrix).unwrap();
        assert_eq!(view.out_degree, vec![3.0, 0.0]);
        assert_eq!(view.in_degree, vec![0.0, 3.0]);
        assert!((view.total_weight - 3.0).abs() < TOLERANCE);
    }
}
