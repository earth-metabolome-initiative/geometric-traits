use alloc::vec::Vec;

use num_traits::AsPrimitive;

use crate::traits::MonoplexMonopartiteGraph;

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct RefinementKey<EdgeColor> {
    current_color: usize,
    neighborhood: Vec<(EdgeColor, usize)>,
}

#[inline]
fn dense_rank<T>(keys: &[T]) -> Vec<usize>
where
    T: Ord + Clone,
{
    let mut keyed_indices: Vec<(T, usize)> =
        keys.iter().cloned().enumerate().map(|(index, key)| (key, index)).collect();
    keyed_indices.sort_unstable_by(|(left_key, left_index), (right_key, right_index)| {
        left_key.cmp(right_key).then_with(|| left_index.cmp(right_index))
    });

    let mut colors = vec![0usize; keys.len()];
    let mut current_color = 0usize;

    for (offset, (_, index)) in keyed_indices.iter().enumerate() {
        if offset > 0 && keyed_indices[offset - 1].0 != keyed_indices[offset].0 {
            current_color += 1;
        }
        colors[*index] = current_color;
    }

    colors
}

#[inline]
fn uniform_seed_colors<G>(graph: &G) -> Vec<()>
where
    G: MonoplexMonopartiteGraph + ?Sized,
{
    vec![(); graph.number_of_nodes().as_()]
}

#[inline]
fn unit_edge_color<NodeId>(_: NodeId, _: NodeId) {}

/// Weisfeiler-Lehman partition refinement for dense node ids.
///
/// This implementation is intended as a ranking primitive. It returns dense,
/// deterministic color classes rather than a full canonical numbering.
///
/// The refinement supports:
/// - uniform or caller-provided initial node colors
/// - unlabeled or caller-provided edge labels
/// - repeated refinement until convergence
///
/// The result is useful as a stable partition for later ordering or
/// traversal-planning steps.
pub trait WeisfeilerLehmanColoring: MonoplexMonopartiteGraph {
    /// Topology-only Weisfeiler-Lehman coloring with a uniform initial color
    /// and unlabeled edges.
    #[must_use]
    #[inline]
    fn wl_coloring(&self) -> Vec<usize> {
        self.wl_coloring_with_seed(&uniform_seed_colors(self))
    }

    /// Weisfeiler-Lehman coloring with caller-provided initial node colors and
    /// unlabeled edges.
    #[must_use]
    #[inline]
    fn wl_coloring_with_seed<SeedColor>(&self, seed_colors: &[SeedColor]) -> Vec<usize>
    where
        SeedColor: Ord + Clone,
    {
        self.wl_coloring_with_seed_and_edge_colors(seed_colors, unit_edge_color::<Self::NodeId>)
    }

    /// Weisfeiler-Lehman coloring with uniform initial node colors and
    /// caller-provided edge labels.
    #[must_use]
    #[inline]
    fn wl_coloring_with_edge_colors<EdgeColor, F>(&self, edge_colors: F) -> Vec<usize>
    where
        EdgeColor: Ord + Clone,
        F: FnMut(Self::NodeId, Self::NodeId) -> EdgeColor,
    {
        self.wl_coloring_with_seed_and_edge_colors(&uniform_seed_colors(self), edge_colors)
    }

    /// Weisfeiler-Lehman coloring with caller-provided initial node colors and
    /// caller-provided edge labels.
    ///
    /// `seed_colors` must contain exactly one entry per dense node id.
    ///
    /// The refinement key for each node is:
    /// - its current color
    /// - the sorted multiset of `(edge_color, neighbor_color)` pairs over its
    ///   outgoing neighbors
    ///
    /// The output classes are dense and deterministic.
    #[must_use]
    fn wl_coloring_with_seed_and_edge_colors<SeedColor, EdgeColor, F>(
        &self,
        seed_colors: &[SeedColor],
        mut edge_colors: F,
    ) -> Vec<usize>
    where
        SeedColor: Ord + Clone,
        EdgeColor: Ord + Clone,
        F: FnMut(Self::NodeId, Self::NodeId) -> EdgeColor,
    {
        const WRONG_LENGTH_MESSAGE: &str =
            "seed colors must contain exactly one entry per dense node id";

        let node_count = self.number_of_nodes().as_();
        assert!(seed_colors.len() == node_count, "{WRONG_LENGTH_MESSAGE}");

        let nodes: Vec<Self::NodeId> = self.node_ids().collect();
        debug_assert_eq!(nodes.len(), node_count);
        debug_assert!(nodes.iter().enumerate().all(|(index, node)| (*node).as_() == index));

        let mut colors = dense_rank(seed_colors);

        loop {
            let mut keys = Vec::with_capacity(node_count);

            for &node in &nodes {
                let mut neighborhood: Vec<(EdgeColor, usize)> = self
                    .successors(node)
                    .map(|neighbor| (edge_colors(node, neighbor), colors[neighbor.as_()]))
                    .collect();
                neighborhood.sort_unstable();

                keys.push(RefinementKey { current_color: colors[node.as_()], neighborhood });
            }

            let next_colors = dense_rank(&keys);
            if next_colors == colors {
                return colors;
            }
            colors = next_colors;
        }
    }
}

impl<G> WeisfeilerLehmanColoring for G where G: MonoplexMonopartiteGraph {}
