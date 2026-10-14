//! Submodule providing the `PairwiseBFS` trait and its blanket implementation
//! for sparse square matrices.
//!
//! The algorithm runs one breadth-first search from each source node and
//! returns the resulting all-pairs shortest-path distances for the unweighted
//! case.
use alloc::{collections::VecDeque, vec::Vec};

use num_traits::AsPrimitive;

use crate::{
    impls::VecMatrix2D,
    traits::{SparseMatrix2D, SquareMatrix},
};

/// Trait providing all-pairs shortest-path distances for unweighted graphs via
/// repeated breadth-first search.
///
/// Missing entries in the sparse matrix are interpreted as absent edges.
/// Every present edge has unit cost. The result is returned as a dense matrix
/// whose entries are `Option<usize>`:
/// - `Some(distance)` when a path exists;
/// - `None` when the destination is unreachable.
///
/// # Complexity
///
/// O(V * (V + E)) time and O(V²) space.
///
/// # Examples
///
/// ```
/// use geometric_traits::{
///     impls::{CSR2D, SquareCSR2D},
///     prelude::*,
///     traits::EdgesBuilder,
/// };
///
/// let edges: SquareCSR2D<CSR2D<usize, usize, usize>> = DiEdgesBuilder::default()
///     .expected_number_of_edges(3)
///     .expected_shape(4)
///     .edges(vec![(0, 1), (1, 2), (2, 3)].into_iter())
///     .build()
///     .unwrap();
///
/// let distances = edges.pairwise_bfs();
/// assert_eq!(distances.value((0, 3)), Some(3));
/// assert_eq!(distances.value((3, 0)), None);
/// ```
pub trait PairwiseBFS: SquareMatrix + SparseMatrix2D + Sized
where
    Self::Index: AsPrimitive<usize>,
    Self::RowIndex: AsPrimitive<usize>,
    Self::ColumnIndex: AsPrimitive<usize>,
{
    /// Computes all-pairs shortest-path distances in the unweighted case.
    #[inline]
    fn pairwise_bfs(&self) -> VecMatrix2D<Option<usize>> {
        let order = self.order().as_();
        if order == 0 {
            return VecMatrix2D::new(0, 0, Vec::new());
        }

        let mut all_distances = vec![None; order * order];
        let mut queue = VecDeque::with_capacity(order);

        for source_id in self.row_indices() {
            let source = source_id.as_();
            let source_offset = source * order;

            let distances = &mut all_distances[source_offset..source_offset + order];
            distances.fill(None);
            distances[source] = Some(0);

            queue.clear();
            queue.push_back(source_id);

            while let Some(node_id) = queue.pop_front() {
                let node = node_id.as_();
                let distance =
                    distances[node].expect("BFS queue only contains already-reached nodes");

                for destination_id in self.sparse_row(node_id) {
                    let destination = destination_id.as_();
                    if distances[destination].is_none() {
                        distances[destination] = Some(distance + 1);
                        queue.push_back(destination_id);
                    }
                }
            }
        }

        VecMatrix2D::new(order, order, all_distances)
    }
}

impl<M> PairwiseBFS for M
where
    M: SquareMatrix + SparseMatrix2D + Sized,
    M::Index: AsPrimitive<usize>,
    M::RowIndex: AsPrimitive<usize>,
    M::ColumnIndex: AsPrimitive<usize>,
{
}
