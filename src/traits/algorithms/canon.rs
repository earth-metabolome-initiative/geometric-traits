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
pub(crate) use refine::RefinementTrace;
pub use refine::refine_partition_to_labeled_equitable;
pub use search::{
    CanonSplittingHeuristic, CanonicalLabelingOptions, CanonicalLabelingResult,
    CanonicalSearchStats, LabeledSimpleGraphCertificate, canonical_label_labeled_simple_graph,
    canonical_label_labeled_simple_graph_with_options,
};
