use alloc::vec::Vec;

use num_traits::AsPrimitive;

use super::NodeSorter;
use crate::traits::UndirectedMonopartiteMonoplexGraph;

const UNCOLORED: usize = usize::MAX;

/// DSATUR ordering.
///
/// This is the classic Brélaz saturation-degree ordering used by greedy
/// coloring. On each step it selects the uncolored vertex with maximum
/// `(saturation_degree, degree)`, breaking remaining ties by smaller node id,
/// then assigns the smallest available color before updating neighbor
/// saturations.
///
/// The implementation is intentionally deterministic and matches the supported
/// `NetworkX` reference contract on relabeled undirected simple graphs, where
/// graph iteration order is ascending node id.
///
/// References:
/// - Brélaz, D. (1979). New methods to color the vertices of a graph.
///   *Communications of the ACM*, 22(4), 251-256. DOI: `10.1145/359094.359101`
/// - `NetworkX` `strategy_saturation_largest_first`
#[derive(Clone, Copy, Debug, Default)]
pub struct DsaturSorter;

impl<G> NodeSorter<G> for DsaturSorter
where
    G: UndirectedMonopartiteMonoplexGraph,
{
    fn sort_nodes(&self, graph: &G) -> Vec<G::NodeId> {
        let n = graph.number_of_nodes().as_();
        if n == 0 {
            return Vec::new();
        }

        let nodes: Vec<G::NodeId> = graph.node_ids().collect();
        debug_assert_eq!(nodes.len(), n);
        debug_assert!(nodes.iter().enumerate().all(|(i, node)| (*node).as_() == i));

        let degrees: Vec<usize> = nodes.iter().map(|&node| graph.degree(node).as_()).collect();
        let mut colors = vec![UNCOLORED; n];
        let mut saturation = vec![0usize; n];
        let mut neighbor_distinct_colors = vec![vec![false; n]; n];
        let mut used_neighbor_colors = vec![false; n];
        let mut order = Vec::with_capacity(n);

        for _ in 0..n {
            let mut best_index = None;
            let mut best_saturation = 0usize;
            let mut best_degree = 0usize;

            for node_index in 0..n {
                if colors[node_index] != UNCOLORED {
                    continue;
                }

                let node_saturation = saturation[node_index];
                let node_degree = degrees[node_index];
                let is_better = best_index.is_none()
                    || node_saturation > best_saturation
                    || (node_saturation == best_saturation
                        && (node_degree > best_degree
                            || (node_degree == best_degree
                                && node_index < best_index.expect("best index must exist"))));

                if is_better {
                    best_index = Some(node_index);
                    best_saturation = node_saturation;
                    best_degree = node_degree;
                }
            }

            let node_index = best_index.expect("there must be an uncolored node until completion");
            let node = nodes[node_index];
            order.push(node);

            used_neighbor_colors.fill(false);
            for neighbor in graph.neighbors(node) {
                let neighbor_index = neighbor.as_();
                let neighbor_color = colors[neighbor_index];
                if neighbor_color != UNCOLORED {
                    used_neighbor_colors[neighbor_color] = true;
                }
            }

            let color = used_neighbor_colors
                .iter()
                .position(|used| !*used)
                .expect("at least one color below n must remain available");
            colors[node_index] = color;

            for neighbor in graph.neighbors(node) {
                let neighbor_index = neighbor.as_();
                if colors[neighbor_index] != UNCOLORED {
                    continue;
                }
                if !neighbor_distinct_colors[neighbor_index][color] {
                    neighbor_distinct_colors[neighbor_index][color] = true;
                    saturation[neighbor_index] += 1;
                }
            }
        }

        order
    }
}
