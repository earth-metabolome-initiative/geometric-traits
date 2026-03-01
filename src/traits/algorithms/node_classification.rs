//! Shared helpers for node classification algorithms.
#![cfg(feature = "alloc")]

use alloc::vec::Vec;

use num_traits::AsPrimitive;

use crate::traits::MonoplexMonopartiteGraph;

/// Returns `(has_predecessor, has_successor)` flags for every node id.
pub(super) fn predecessor_successor_flags<G: ?Sized + MonoplexMonopartiteGraph>(
    graph: &G,
) -> (Vec<bool>, Vec<bool>) {
    let number_of_nodes = graph.number_of_nodes().as_();
    let mut has_predecessor = vec![false; number_of_nodes];
    let mut has_successor = vec![false; number_of_nodes];

    for node in graph.node_ids() {
        for successor_node_id in graph.successors(node) {
            has_predecessor[successor_node_id.as_()] = true;
            has_successor[node.as_()] = true;
        }
    }

    (has_predecessor, has_successor)
}
