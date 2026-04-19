use alloc::{vec, vec::Vec};

use num_traits::AsPrimitive;

use crate::traits::UndirectedMonopartiteMonoplexGraph;

pub(super) struct UndirectedMotifContext<NodeId> {
    pub(super) nodes: Vec<NodeId>,
    pub(super) rank: Vec<usize>,
    pub(super) in_cover: Vec<bool>,
}

/// Vertex-cover ordering choices for motif scorers.
///
/// These match the schemas described in Sect. 3 of Cappelletti et al. (ICCS
/// 2023). Preferred choices from the paper:
/// - triangles: [`MotifCountOrdering::IncreasingDegree`]
/// - squares: no clear overall winner; [`MotifCountOrdering::Natural`] is the
///   neutral baseline
///
/// # Examples
/// ```
/// use geometric_traits::traits::algorithms::{
///     MotifCountOrdering, SquareCountScorer, TriangleCountScorer,
/// };
///
/// assert_eq!(TriangleCountScorer::default().ordering(), MotifCountOrdering::IncreasingDegree);
/// assert_eq!(SquareCountScorer::default().ordering(), MotifCountOrdering::Natural);
/// ```
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MotifCountOrdering {
    /// Natural node order. The resulting cover contains every non-isolated
    /// node.
    Natural,
    /// Degree-descending order, adding only the source endpoint of each edge to
    /// the cover.
    DecreasingDegree,
    /// Degree-ascending order, adding only the source endpoint of each edge to
    /// the cover.
    IncreasingDegree,
}

pub(super) fn build_undirected_motif_context<G>(
    graph: &G,
    ordering: MotifCountOrdering,
) -> UndirectedMotifContext<G::NodeId>
where
    G: UndirectedMonopartiteMonoplexGraph,
{
    let n = graph.number_of_nodes().as_();
    let nodes: Vec<G::NodeId> = graph.node_ids().collect();
    debug_assert_eq!(nodes.len(), n);
    debug_assert!(nodes.iter().enumerate().all(|(i, node)| (*node).as_() == i));

    let degrees: Vec<usize> = nodes.iter().map(|&node| graph.degree(node).as_()).collect();
    let rank = build_rank(&degrees, ordering);
    let mut in_cover = vec![false; n];

    match ordering {
        MotifCountOrdering::Natural => {
            for (node_index, &degree) in degrees.iter().enumerate() {
                if degree > 0 {
                    in_cover[node_index] = true;
                }
            }
        }
        MotifCountOrdering::DecreasingDegree | MotifCountOrdering::IncreasingDegree => {
            for &node in &nodes {
                let node_index = node.as_();
                for neighbor in graph.neighbors(node) {
                    let neighbor_index = neighbor.as_();
                    if node_index > neighbor_index {
                        continue;
                    }

                    let source_index = if rank[node_index] <= rank[neighbor_index] {
                        node_index
                    } else {
                        neighbor_index
                    };
                    in_cover[source_index] = true;
                }
            }
        }
    }

    UndirectedMotifContext { nodes, rank, in_cover }
}

fn build_rank(degrees: &[usize], ordering: MotifCountOrdering) -> Vec<usize> {
    let n = degrees.len();
    let mut order: Vec<usize> = (0..n).collect();
    match ordering {
        MotifCountOrdering::Natural => {}
        MotifCountOrdering::DecreasingDegree => {
            order.sort_unstable_by_key(|&node_index| {
                (core::cmp::Reverse(degrees[node_index]), node_index)
            });
        }
        MotifCountOrdering::IncreasingDegree => {
            order.sort_unstable_by_key(|&node_index| (degrees[node_index], node_index));
        }
    }

    let mut rank = vec![0usize; n];
    for (position, node_index) in order.into_iter().enumerate() {
        rank[node_index] = position;
    }
    rank
}
