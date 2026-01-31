//! Submodule providing the Hopcroft-Karp algorithm for the assignment problem.
#![cfg(feature = "alloc")]
use alloc::vec::Vec;

mod partial_assignment;

use partial_assignment::PartialAssignment;

use crate::traits::SparseMatrix2D;

#[derive(Debug, Clone, PartialEq, Eq)]
/// Enumeration of the errors that might occur during the Hopcroft-Karp
/// algorithm.
pub enum HopcroftKarpError {
    /// The provided distance type is not large enough to be used in the
    /// algorithm for the provided graph.
    InsufficientDistanceType,
}

/// Trait providing the Hopcroft-Karp algorithm for the assignment problem.
pub trait HopcroftKarp: SparseMatrix2D {
    #[allow(clippy::type_complexity)]
    /// Return the assignment as assigned by the Hopcroft-Karp algorithm.
    ///
    /// # Errors
    ///
    /// Returns `HopcroftKarpError::InsufficientDistanceType` if the distance
    /// type used in the algorithm is not large enough to represent all the
    /// distances in the bipartite graph. This can occur when the graph is
    /// too large or complex for the selected distance type.
    ///
    /// # Examples
    ///
    /// ```
    /// use geometric_traits::{
    ///     impls::{CSR2D, SortedVec},
    ///     prelude::*,
    ///     traits::{EdgesBuilder, VocabularyBuilder},
    /// };
    ///
    /// let nodes: Vec<usize> = vec![0, 1, 2];
    /// let edges: Vec<(usize, usize)> = vec![(0, 1), (1, 2), (2, 0)];
    /// let nodes: SortedVec<usize> = GenericVocabularyBuilder::default()
    ///     .expected_number_of_symbols(nodes.len())
    ///     .symbols(nodes.into_iter().enumerate())
    ///     .build()
    ///     .unwrap();
    /// let edges: CSR2D<_, _, _> = GenericEdgesBuilder::default()
    ///     .expected_number_of_edges(edges.len())
    ///     .expected_shape((nodes.len(), nodes.len()))
    ///     .edges(edges.into_iter())
    ///     .build()
    ///     .unwrap();
    ///
    /// let assignment = edges.hopcroft_karp().unwrap();
    /// assert_eq!(assignment.len(), 3);
    /// ```
    fn hopcroft_karp(&self) -> Result<Vec<(Self::RowIndex, Self::ColumnIndex)>, HopcroftKarpError> {
        let mut partial_assignment: PartialAssignment<'_, Self, u32> =
            PartialAssignment::from(self);
        while partial_assignment.bfs()? {
            for left_node_id in self.row_indices() {
                if !partial_assignment.has_successor(left_node_id) {
                    partial_assignment.dfs(Some(left_node_id));
                }
            }
        }

        Ok(partial_assignment.into())
    }
}

impl<M> HopcroftKarp for M where M: SparseMatrix2D {}
