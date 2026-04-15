//! Public canonization API plus lower-level building blocks for
//! individualization-refinement over dense vertex identifiers.

use num_traits::AsPrimitive;

use crate::traits::MonoplexMonopartiteGraph;

mod partition;
mod refine;
mod search;

pub use partition::{
    BacktrackableOrderedPartition, OrderedPartitionCells, PartitionBacktrackPoint, PartitionCellId,
    PartitionCellView,
};
pub(crate) use refine::RefinementTrace;
pub use refine::refine_partition_to_labeled_equitable;
pub use search::{
    CanonSplittingHeuristic, CanonicalLabelingOptions, CanonicalLabelingResult,
    CanonicalSearchStats, LabeledSimpleGraphCertificate, canonical_label_labeled_simple_graph,
    canonical_label_labeled_simple_graph_with_options,
};

/// Trait exposing canonical labeling for simple undirected graphs with
/// total-order vertex and edge labels.
///
/// The trait-style surface matches the rest of the crate's algorithm API. The
/// free functions remain available as compatibility wrappers around the same
/// implementation.
pub trait CanonicalLabeling: MonoplexMonopartiteGraph {
    /// Computes a canonical labeling using the default canonizer options.
    #[inline]
    fn canonical_labeling<VertexLabel, EdgeLabel, VF, EF>(
        &self,
        vertex_label: VF,
        edge_label: EF,
    ) -> CanonicalLabelingResult<VertexLabel, EdgeLabel>
    where
        Self: Sized,
        Self::NodeId: AsPrimitive<usize>,
        VertexLabel: Ord + Clone,
        EdgeLabel: Ord + Clone,
        VF: FnMut(Self::NodeId) -> VertexLabel,
        EF: FnMut(Self::NodeId, Self::NodeId) -> EdgeLabel,
    {
        search::canonical_label_labeled_simple_graph(self, vertex_label, edge_label)
    }

    /// Computes a canonical labeling using explicit canonizer options.
    #[inline]
    fn canonical_labeling_with_options<VertexLabel, EdgeLabel, VF, EF>(
        &self,
        vertex_label: VF,
        edge_label: EF,
        options: CanonicalLabelingOptions,
    ) -> CanonicalLabelingResult<VertexLabel, EdgeLabel>
    where
        Self: Sized,
        Self::NodeId: AsPrimitive<usize>,
        VertexLabel: Ord + Clone,
        EdgeLabel: Ord + Clone,
        VF: FnMut(Self::NodeId) -> VertexLabel,
        EF: FnMut(Self::NodeId, Self::NodeId) -> EdgeLabel,
    {
        search::canonical_label_labeled_simple_graph_with_options(
            self,
            vertex_label,
            edge_label,
            options,
        )
    }
}

impl<G: ?Sized + MonoplexMonopartiteGraph> CanonicalLabeling for G {}
