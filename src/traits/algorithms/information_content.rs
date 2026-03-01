//! Submodule providing `Information Content` structs
//! and methods for use with algorithms such as `Resnik`
use alloc::vec::Vec;

mod error;
use core::ops::Index;

pub use error::InformationContentError;
use num_traits::AsPrimitive;

use crate::traits::{
    Kahn, MonoplexMonopartiteGraph, RootNodes, SingletonNodes, SinkNodes, edges::Edges,
};

/// Result of information content computation.
#[derive(Debug, PartialEq)]
pub struct InformationContentResult<'graph, G: ?Sized + MonoplexMonopartiteGraph> {
    /// The graph to be analyzed
    graph: &'graph G,
    /// Information content per node
    information_contents: Vec<f64>,
    /// Maximum information content across all nodes.
    max_information_content: f64,
    /// Root nodes of the graph
    root_nodes: Vec<G::NodeId>,
}

impl<G: ?Sized + MonoplexMonopartiteGraph> InformationContentResult<'_, G> {
    /// Returns a reference to underlying graph
    pub(crate) fn graph(&self) -> &G {
        self.graph
    }
    /// Returns a reference to the root nodes of the graph
    pub(crate) fn root_nodes(&self) -> &[G::NodeId] {
        &self.root_nodes
    }
}

impl<G: ?Sized + MonoplexMonopartiteGraph> Index<G::NodeId> for InformationContentResult<'_, G> {
    type Output = f64;
    #[inline]
    fn index(&self, index: G::NodeId) -> &Self::Output {
        &self.information_contents[index.as_()]
    }
}

/// Trait for computing information content from propagated occurrences.
///
/// Notes:
/// - This module assumes byte/float-safe sums. It does not perform DAG
///   validation
///
/// # Complexity
///
/// `information_content` runs in O(V + E) time and O(V) space, where V is the
/// number of nodes and E the number of edges.
pub trait InformationContent: MonoplexMonopartiteGraph {
    /// Computes per-node information content
    /// # Errors
    /// - `NotDag` when the graph contains a cycle
    /// - `UnequalOccurrenceSize` when occurrence lengths do not match the
    ///   expected size
    /// - `SinkNodeZeroOccurrence` when a sink (or singleton) node has zero
    ///   occurrences
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
    /// let graph: DiGraph<usize> = DiGraph::from((nodes, edges));
    ///
    /// let ic = graph.information_content(&[1, 1, 1]).unwrap();
    /// assert!(ic[0] >= 0.0);
    /// ```
    #[inline]
    fn information_content(
        &self,
        occurrences: &[usize],
    ) -> Result<InformationContentResult<'_, Self>, InformationContentError> {
        // Check whether the graph is a DAG (characterize by having no cycles)
        let topological_ordering = self.edges().matrix().kahn()?;
        // Validate occurrences length.
        let expected = self.number_of_nodes().as_();
        if occurrences.len() != expected {
            return Err(InformationContentError::UnequalOccurrenceSize {
                expected,
                found: occurrences.len(),
            });
        }

        for sink_node in self.sink_nodes().into_iter().chain(self.singleton_nodes()) {
            if occurrences[sink_node.as_()] == 0 {
                // raise error that sink_node has a zero occurrence
                return Err(InformationContentError::SinkNodeZeroOccurrence);
            }
        }

        // Propagate occurrences iteratively in reverse topological order
        // (sinks first, roots last) so that each node's successors are always
        // processed before the node itself.
        let mut sorted_nodes: Vec<Self::NodeId> = self.node_ids().collect();
        sorted_nodes.sort_unstable_by(|&a, &b| {
            topological_ordering[b.as_()].as_().cmp(&topological_ordering[a.as_()].as_())
        });

        let mut propagated_occurrences = vec![0usize; expected];
        for &node_id in &sorted_nodes {
            let mut acc = occurrences[node_id.as_()];
            for successor in self.successors(node_id) {
                acc = acc.saturating_add(propagated_occurrences[successor.as_()]);
            }
            propagated_occurrences[node_id.as_()] = acc;
        }

        // total information content, initialized to 0
        let mut total_occurrences: usize = 0;
        for occurrence in occurrences {
            total_occurrences = total_occurrences.saturating_add(*occurrence);
        }
        let mut max_information_content = 0.0;

        let information_contents = propagated_occurrences
            .into_iter()
            .map(|propagated_occurrence| {
                #[allow(clippy::cast_precision_loss)]
                let information_content =
                    -(propagated_occurrence as f64 / total_occurrences as f64).ln();
                if information_content > max_information_content {
                    max_information_content = information_content;
                }
                information_content
            })
            .collect::<Vec<f64>>();

        let root_nodes = self.root_nodes();

        Ok(InformationContentResult {
            information_contents,
            max_information_content,
            root_nodes,
            graph: self,
        })
    }
}

impl<T> InformationContent for T where T: MonoplexMonopartiteGraph + ?Sized {}
