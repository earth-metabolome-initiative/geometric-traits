//! Submodule providing the `Kahn` trait and its blanket implementation for
//! sparse matrices, which provides the Kahn's algorithm for topological
//! sorting.
use alloc::vec::Vec;

use num_traits::{ConstOne, ConstZero};

use num_traits::AsPrimitive;

use crate::traits::{SparseMatrix2D, SquareMatrix};

#[derive(Debug, Clone, PartialEq)]
/// Error enumeration for Kahn's algorithm.
pub enum KahnError {
    /// Error when the graph contains a cycle.
    Cycle,
}

/// Kahn's algorithm for topological sorting.
pub trait Kahn: SquareMatrix + SparseMatrix2D {
    /// Returns the indices to rearrange the rows of the matrix in a topological
    /// order.
    ///
    /// # Errors
    ///
    /// * If the graph contains a cycle, an error is returned.
    ///
    /// # Complexity
    ///
    /// O(V + E) time and O(V) space.
    ///
    /// # Examples
    ///
    /// ```
    /// use geometric_traits::{
    ///     impls::{SortedVec, SquareCSR2D},
    ///     prelude::*,
    ///     traits::{EdgesBuilder, VocabularyBuilder},
    /// };
    ///
    /// let nodes: Vec<usize> = vec![0, 1, 2];
    /// let edges: Vec<(usize, usize)> = vec![(0, 1), (1, 2)];
    /// let nodes: SortedVec<usize> = GenericVocabularyBuilder::default()
    ///     .expected_number_of_symbols(nodes.len())
    ///     .symbols(nodes.into_iter().enumerate())
    ///     .build()
    ///     .unwrap();
    /// let edges: SquareCSR2D<_> = DiEdgesBuilder::default()
    ///     .expected_number_of_edges(edges.len())
    ///     .expected_shape(nodes.len())
    ///     .edges(edges.into_iter())
    ///     .build()
    ///     .unwrap();
    ///
    /// let topological_order = edges.kahn().unwrap();
    /// assert_eq!(topological_order.len(), 3);
    /// ```
    fn kahn(&self) -> Result<Vec<Self::Index>, KahnError> {
        let mut in_degree = vec![Self::Index::ZERO; self.order().as_()];
        let mut frontier = Vec::new();
        let mut temporary_frontier = Vec::new();
        let mut number_of_visited_nodes = Self::Index::ZERO;
        let mut topological_order = vec![Self::Index::ZERO; self.order().as_()];

        for row_id in self.row_indices() {
            for successor_id in self.sparse_row(row_id) {
                in_degree[successor_id.as_()] += Self::Index::ONE;
            }
        }

        frontier.extend(
            self.row_indices().filter(|row_id| in_degree[row_id.as_()] == Self::Index::ZERO),
        );

        while !frontier.is_empty() {
            temporary_frontier.clear();

            for row_id in frontier.drain(..) {
                topological_order[row_id.as_()] = number_of_visited_nodes;
                number_of_visited_nodes += Self::Index::ONE;
                temporary_frontier.extend(self.sparse_row(row_id).filter(|successor_id| {
                    in_degree[successor_id.as_()] -= Self::Index::ONE;
                    in_degree[successor_id.as_()] == Self::Index::ZERO
                }));
            }

            core::mem::swap(&mut frontier, &mut temporary_frontier);
        }

        if number_of_visited_nodes != self.order() {
            return Err(KahnError::Cycle);
        }

        Ok(topological_order)
    }
}

impl<G: SquareMatrix + SparseMatrix2D> Kahn for G {}
