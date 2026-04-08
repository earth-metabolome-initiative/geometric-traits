//! Exact relevant-cycle computation for undirected simple graphs.
//!
//! Relevant cycles are the union of all minimum cycle bases. Equivalently, a
//! cycle is relevant if it cannot be represented as the xor-sum of strictly
//! shorter cycles.

use alloc::vec::Vec;

use super::initial_cycle_families::{XorBasis, cyclic_component_cycle_candidates};
use crate::traits::{
    BiconnectedComponentsError, MonopartiteGraph, PositiveInteger,
    UndirectedMonopartiteMonoplexGraph,
};

/// Result returned by [`RelevantCycles::relevant_cycles`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RelevantCyclesResult<NodeId: PositiveInteger> {
    relevant_cycles: Vec<Vec<NodeId>>,
}

impl<NodeId: PositiveInteger> RelevantCyclesResult<NodeId> {
    /// Returns the relevant cycles.
    #[inline]
    pub fn relevant_cycles(&self) -> core::slice::Iter<'_, Vec<NodeId>> {
        self.relevant_cycles.iter()
    }

    /// Returns the number of relevant cycles.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.relevant_cycles.len()
    }

    /// Returns whether the result is empty.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.relevant_cycles.is_empty()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
/// Error type for relevant-cycle computation.
pub enum RelevantCyclesError {
    /// Error raised while decomposing the graph into biconnected components.
    #[error("{0}")]
    BiconnectedComponentsError(BiconnectedComponentsError),
}

impl From<BiconnectedComponentsError> for RelevantCyclesError {
    #[inline]
    fn from(error: BiconnectedComponentsError) -> Self {
        Self::BiconnectedComponentsError(error)
    }
}

impl From<RelevantCyclesError>
    for crate::errors::monopartite_graph_error::algorithms::MonopartiteAlgorithmError
{
    #[inline]
    fn from(error: RelevantCyclesError) -> Self {
        Self::RelevantCyclesError(error)
    }
}

impl<G: MonopartiteGraph> From<RelevantCyclesError> for crate::errors::MonopartiteError<G> {
    #[inline]
    fn from(error: RelevantCyclesError) -> Self {
        Self::AlgorithmError(error.into())
    }
}

/// Trait providing the exact relevant cycles for undirected simple graphs.
pub trait RelevantCycles: UndirectedMonopartiteMonoplexGraph {
    /// Computes the relevant cycles of the graph.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[cfg(feature = "alloc")]
    /// # {
    /// use geometric_traits::{
    ///     impls::{CSR2D, SortedVec, SymmetricCSR2D, UpperTriangularCSR2D},
    ///     naive_structs::{GenericGraph, GenericUndirectedMonopartiteEdgesBuilder},
    ///     prelude::*,
    ///     traits::{EdgesBuilder, VocabularyBuilder},
    /// };
    ///
    /// let nodes: SortedVec<usize> = GenericVocabularyBuilder::default()
    ///     .expected_number_of_symbols(4)
    ///     .symbols((0..4).enumerate())
    ///     .build()
    ///     .unwrap();
    /// let edges: SymmetricCSR2D<CSR2D<usize, usize, usize>> =
    ///     GenericUndirectedMonopartiteEdgesBuilder::<
    ///         _,
    ///         UpperTriangularCSR2D<CSR2D<usize, usize, usize>>,
    ///         SymmetricCSR2D<CSR2D<usize, usize, usize>>,
    ///     >::default()
    ///     .expected_number_of_edges(5)
    ///     .expected_shape(4)
    ///     .edges([(0, 1), (0, 2), (0, 3), (1, 2), (2, 3)].into_iter())
    ///     .build()
    ///     .unwrap();
    /// let graph: GenericGraph<SortedVec<usize>, SymmetricCSR2D<CSR2D<usize, usize, usize>>> =
    ///     GenericGraph::from((nodes, edges));
    ///
    /// let result = graph.relevant_cycles().unwrap();
    /// assert_eq!(
    ///     result.relevant_cycles().cloned().collect::<Vec<_>>(),
    ///     vec![vec![0, 1, 2], vec![0, 2, 3]]
    /// );
    /// # }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if the graph cannot be decomposed into biconnected
    /// components under the current simple-undirected graph contract.
    fn relevant_cycles(&self) -> Result<RelevantCyclesResult<Self::NodeId>, RelevantCyclesError>
    where
        Self: Sized,
    {
        let component_candidates = cyclic_component_cycle_candidates(self)?;
        let mut relevant_cycles = Vec::new();

        for candidates in component_candidates {
            if candidates.is_empty() {
                continue;
            }

            let mut span = XorBasis::new(candidates[0].edge_bits.len() * 64);
            let mut group_start = 0usize;
            while group_start < candidates.len() {
                let group_length = candidates[group_start].length;
                let mut group_end = group_start + 1;
                while group_end < candidates.len() && candidates[group_end].length == group_length {
                    group_end += 1;
                }

                let mut rows_to_insert = Vec::new();
                for candidate in &candidates[group_start..group_end] {
                    if span.is_independent(&candidate.edge_bits) {
                        relevant_cycles.push(candidate.nodes.clone());
                        rows_to_insert.push(candidate.edge_bits.clone());
                    }
                }
                for row in rows_to_insert {
                    span.insert(row);
                }

                group_start = group_end;
            }
        }

        relevant_cycles.sort_unstable_by(|left, right| {
            left.len().cmp(&right.len()).then_with(|| left.cmp(right))
        });
        Ok(RelevantCyclesResult { relevant_cycles })
    }
}

impl<G: ?Sized + UndirectedMonopartiteMonoplexGraph> RelevantCycles for G {}
