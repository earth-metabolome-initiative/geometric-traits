//! Minimum-cost maximum balanced flow in general graphs.
//!
//! The public entry point in this module solves the following lexicographic
//! optimization problem on an undirected capacitated graph:
//! 1. maximize the total integral flow assigned to edges, subject to the
//!    per-vertex budgets;
//! 2. among those maximum-flow assignments, minimize the total edge cost.
//!
//! The current implementation is hybrid and exact:
//! - it first uses [`crate::traits::Kocay`] to determine the maximum feasible
//!   total flow value;
//! - tree components are solved by dynamic programming;
//! - bipartite components are solved as ordinary minimum-cost maximum flow;
//! - remaining non-bipartite components are reduced to weighted perfect
//!   matching and solved with [`crate::traits::BlossomV`].
//!
//! The returned edge flows are integral, maximize total flow, and among those
//! attain minimum total cost.

mod bipartite;
mod shared;
mod solver;
mod tree;

use alloc::vec::Vec;

use num_traits::AsPrimitive;
use solver::minimum_cost_balanced_flow_impl;

use crate::traits::{Number, PositiveInteger, SparseValuedMatrix2D};

/// Minimum-cost maximum balanced flow in an undirected capacitated graph.
///
/// The capacity matrix encodes an undirected graph: each positive entry
/// `(i, j)` means that the undirected edge `{i, j}` has that capacity. Vertex
/// budgets cap the sum of incident flow at each vertex. The `edge_costs`
/// matrix supplies the per-unit cost of sending flow along each edge.
///
/// This trait solves the exact lexicographic problem:
/// 1. maximize total flow;
/// 2. among all such maximum-flow solutions, minimize total cost.
///
/// The returned triples `(row, column, flow)` always satisfy:
/// - `row < column`,
/// - `0 < flow <= capacity(row, column)`,
/// - each vertex budget,
/// - maximum total flow, and
/// - minimum total cost among all maximum-flow assignments.
///
/// Internally, the implementation decomposes connected components and then
/// chooses an exact specialized solver per component shape:
/// - tree components use dynamic programming;
/// - bipartite components use minimum-cost maximum flow;
/// - general non-bipartite components fall back to a weighted matching
///   reduction solved with Blossom V.
///
/// # Examples
///
/// ```
/// use geometric_traits::{
///     impls::ValuedCSR2D,
///     traits::{MatrixMut, MinimumCostBalancedFlow, SparseMatrixMut},
/// };
///
/// type CapacityGraph = ValuedCSR2D<usize, usize, usize, usize>;
/// type CostGraph = ValuedCSR2D<usize, usize, usize, i64>;
///
/// let mut capacities: CapacityGraph = SparseMatrixMut::with_sparse_shaped_capacity((3, 3), 6);
/// for (row, column, capacity) in
///     [(0usize, 1usize, 1usize), (0, 2, 1), (1, 0, 1), (1, 2, 1), (2, 0, 1), (2, 1, 1)]
/// {
///     MatrixMut::add(&mut capacities, (row, column, capacity)).unwrap();
/// }
///
/// let mut costs: CostGraph = SparseMatrixMut::with_sparse_shaped_capacity((3, 3), 6);
/// for (row, column, cost) in
///     [(0usize, 1usize, 9i64), (0, 2, 1), (1, 0, 9), (1, 2, 4), (2, 0, 1), (2, 1, 4)]
/// {
///     MatrixMut::add(&mut costs, (row, column, cost)).unwrap();
/// }
///
/// let flow = capacities.minimum_cost_balanced_flow(&[1, 1, 1], &costs);
///
/// // Only one unit of flow can be placed, so the solver picks the cheapest edge.
/// assert_eq!(flow, vec![(0, 2, 1)]);
/// ```
pub trait MinimumCostBalancedFlow: SparseValuedMatrix2D + Sized
where
    Self::Value: PositiveInteger,
    Self::RowIndex: PositiveInteger,
    Self::ColumnIndex: PositiveInteger,
{
    /// Computes a minimum-cost maximum balanced flow.
    ///
    /// # Arguments
    ///
    /// * `vertex_budgets` - Budget (maximum total flow) per vertex.
    /// * `edge_costs` - Cost matrix with the same support as the capacity
    ///   matrix on all positive-capacity edges. If both `(i, j)` and `(j, i)`
    ///   are present, their costs must agree.
    ///
    /// # Panics
    ///
    /// Panics if:
    /// - either matrix is not square,
    /// - the two matrix orders differ,
    /// - `vertex_budgets.len()` does not equal the matrix order,
    /// - a positive-capacity edge is missing a cost,
    /// - or the cost matrix disagrees across the two directions of an
    ///   undirected edge.
    fn minimum_cost_balanced_flow<C>(
        &self,
        vertex_budgets: &[Self::Value],
        edge_costs: &C,
    ) -> Vec<(Self::RowIndex, Self::ColumnIndex, Self::Value)>
    where
        C: SparseValuedMatrix2D<RowIndex = Self::RowIndex, ColumnIndex = Self::ColumnIndex>,
        C::Value: Number + AsPrimitive<i64>,
    {
        minimum_cost_balanced_flow_impl(self, vertex_budgets, edge_costs)
    }
}

impl<M> MinimumCostBalancedFlow for M
where
    M: SparseValuedMatrix2D,
    M::Value: PositiveInteger,
    M::RowIndex: PositiveInteger,
    M::ColumnIndex: PositiveInteger,
{
}

#[cfg(test)]
mod tests {
    use super::{
        bipartite::bipartite_minimum_cost_balanced_flow, shared::OriginalEdge,
        tree::tree_minimum_cost_balanced_flow,
    };

    #[test]
    fn test_private_tree_helper_handles_empty_input() {
        assert!(tree_minimum_cost_balanced_flow(0, &[], &[]).is_empty());
    }

    #[test]
    fn test_private_tree_helper_panics_on_disconnected_input() {
        let edges = [OriginalEdge { u: 0, v: 1, capacity: 1, cost: 3 }];
        let result =
            std::panic::catch_unwind(|| tree_minimum_cost_balanced_flow(3, &edges, &[1, 1, 1]));
        assert!(result.is_err());
    }

    #[test]
    fn test_private_bipartite_helper_panics_on_invalid_coloring() {
        let edges = [OriginalEdge { u: 0, v: 1, capacity: 1, cost: 2 }];
        let result = std::panic::catch_unwind(|| {
            bipartite_minimum_cost_balanced_flow(2, &edges, &[1, 1], &[0, 0])
        });
        assert!(result.is_err());
    }

    #[test]
    fn test_private_bipartite_helper_solves_simple_case() {
        let edges = [OriginalEdge { u: 0, v: 1, capacity: 2, cost: 5 }];
        let flow = bipartite_minimum_cost_balanced_flow(2, &edges, &[2, 2], &[0, 1]);
        assert_eq!(flow, vec![(0, 1, 2)]);
    }
}
