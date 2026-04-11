//! Building blocks for graph canonization algorithms.
//!
//! The current module exposes the first low-level primitive needed by an
//! individualization-refinement canonizer: a backtrackable ordered partition
//! over dense vertex identifiers.

mod partition;
mod refine;
mod search;

pub use partition::{
    BacktrackableOrderedPartition, OrderedPartitionCells, PartitionBacktrackPoint, PartitionCellId,
    PartitionCellView,
};
pub use refine::refine_partition_to_labeled_equitable;
pub(crate) use refine::{
    RefinementTrace, refine_partition_to_labeled_equitable_with_trace,
    refine_partition_to_labeled_equitable_with_trace_from_splitters,
};
pub use search::{
    CanonSplittingHeuristic, CanonicalLabelingOptions, CanonicalLabelingResult,
    CanonicalSearchStats, LabeledSimpleGraphCertificate, canonical_label_labeled_simple_graph,
    canonical_label_labeled_simple_graph_with_options,
};
