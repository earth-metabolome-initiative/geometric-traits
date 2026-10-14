//! Exact essential-cycle computation for undirected simple graphs.
//!
//! Essential cycles are the intersection of all minimum cycle bases.

use alloc::vec::Vec;

use super::initial_cycle_families::{XorBasis, cyclic_component_cycle_candidates};
use crate::traits::{
    BiconnectedComponentsError, MonopartiteGraph, PositiveInteger,
    UndirectedMonopartiteMonoplexGraph,
};

/// Result returned by [`EssentialCycles::essential_cycles`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EssentialCyclesResult<NodeId: PositiveInteger> {
    essential_cycles: Vec<Vec<NodeId>>,
}

impl<NodeId: PositiveInteger> EssentialCyclesResult<NodeId> {
    /// Returns the essential cycles.
    #[inline]
    pub fn essential_cycles(&self) -> core::slice::Iter<'_, Vec<NodeId>> {
        self.essential_cycles.iter()
    }

    /// Returns the number of essential cycles.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.essential_cycles.len()
    }

    /// Returns whether the result is empty.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.essential_cycles.is_empty()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
/// Error type for essential-cycle computation.
pub enum EssentialCyclesError {
    /// Error raised while decomposing the graph into biconnected components.
    #[error("{0}")]
    BiconnectedComponentsError(BiconnectedComponentsError),
}

impl From<BiconnectedComponentsError> for EssentialCyclesError {
    #[inline]
    fn from(error: BiconnectedComponentsError) -> Self {
        Self::BiconnectedComponentsError(error)
    }
}

impl From<EssentialCyclesError>
    for crate::errors::monopartite_graph_error::algorithms::MonopartiteAlgorithmError
{
    #[inline]
    fn from(error: EssentialCyclesError) -> Self {
        Self::EssentialCyclesError(error)
    }
}

impl<G: MonopartiteGraph> From<EssentialCyclesError> for crate::errors::MonopartiteError<G> {
    #[inline]
    fn from(error: EssentialCyclesError) -> Self {
        Self::AlgorithmError(error.into())
    }
}

/// Trait providing the exact essential cycles for undirected simple graphs.
pub trait EssentialCycles: UndirectedMonopartiteMonoplexGraph {
    /// Computes the essential cycles of the graph.
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
    ///     .expected_number_of_edges(6)
    ///     .expected_shape(4)
    ///     .edges([(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)].into_iter())
    ///     .build()
    ///     .unwrap();
    /// let graph: GenericGraph<SortedVec<usize>, SymmetricCSR2D<CSR2D<usize, usize, usize>>> =
    ///     GenericGraph::from((nodes, edges));
    ///
    /// let result = graph.essential_cycles().unwrap();
    /// assert!(result.is_empty());
    /// # }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if the graph cannot be decomposed into biconnected
    /// components under the current simple-undirected graph contract.
    fn essential_cycles(&self) -> Result<EssentialCyclesResult<Self::NodeId>, EssentialCyclesError>
    where
        Self: Sized,
    {
        let component_candidates = cyclic_component_cycle_candidates(self)?;
        let mut essential_cycles = Vec::new();

        for candidates in component_candidates {
            if candidates.is_empty() {
                continue;
            }

            let mut shorter_basis = XorBasis::new(candidates[0].edge_bits.len() * 64);
            let mut group_start = 0usize;

            while group_start < candidates.len() {
                let group_length = candidates[group_start].length;
                let mut group_end = group_start + 1;
                while group_end < candidates.len() && candidates[group_end].length == group_length {
                    group_end += 1;
                }

                let mut group_basis = shorter_basis.clone();
                let mut members_of_basis = Vec::new();
                for (candidate_index, candidate) in
                    candidates.iter().enumerate().take(group_end).skip(group_start)
                {
                    if group_basis.insert(candidate.edge_bits.clone()) {
                        members_of_basis.push(candidate_index);
                    }
                }

                let basis_size_after_group = group_basis.rank();
                for &candidate_index in &members_of_basis {
                    let mut alternate_basis = shorter_basis.clone();
                    for (group_index, candidate) in
                        candidates[group_start..group_end].iter().enumerate()
                    {
                        let absolute_index = group_start + group_index;
                        if absolute_index != candidate_index {
                            alternate_basis.insert(candidate.edge_bits.clone());
                        }
                    }
                    if alternate_basis.rank() < basis_size_after_group {
                        essential_cycles.push(candidates[candidate_index].nodes.clone());
                    }
                }

                shorter_basis = group_basis;
                group_start = group_end;
            }
        }

        essential_cycles.sort_unstable_by(|left, right| {
            left.len().cmp(&right.len()).then_with(|| left.cmp(right))
        });
        Ok(EssentialCyclesResult { essential_cycles })
    }
}

impl<G: ?Sized + UndirectedMonopartiteMonoplexGraph> EssentialCycles for G {}
