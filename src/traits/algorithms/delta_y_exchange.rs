//! Delta-Y exchange detection for MCES post-filtering.
//!
//! Whitney (1932) proved that connected graphs are uniquely determined by
//! their line graphs, with the sole exception of K₃ (triangle) and K₁,₃
//! (claw/star), which have isomorphic line graphs. In the MCES pipeline
//! this means a maximum clique in the modular product of two line graphs
//! may spuriously match a triangle in one graph to a claw in the other.
//!
//! The detection mechanism compares the sorted degree sequences of the
//! edge-induced subgraphs in both original graphs. If they differ, the
//! corresponding edges cannot be structurally equivalent and the match
//! is invalid.
//!
//! **Important**: this comparison is a necessary condition for a Delta-Y
//! exchange, not a sufficient one in general. It is designed to be used as
//! a post-filter on edge correspondences obtained from a maximum clique in
//! the modular product of two line graphs (the RASCAL pipeline). Outside
//! that context, a degree-sequence mismatch indicates structural
//! non-equivalence but is not necessarily a Delta-Y exchange in the
//! Whitney sense.
//!
//! # Complexity
//! `edge_subgraph_degree_sequence`: O(|edges| + V) time, O(V) space.
//!
//! # References
//! - Whitney (1932). "Congruent Graphs and the Connectivity of Graphs."
//!   *American Journal of Mathematics* 54(1):150-168.
//! - Beineke (1968). "Derived Graphs of Finite Graphs." In *Beiträge zur
//!   Graphentheorie*, Teubner.
//! - Raymond, Gardiner, Willett (2002). "RASCAL: Calculation of Graph
//!   Similarity using Maximum Common Edge Subgraphs." *The Computer Journal*
//!   45(6):631-644.

use alloc::vec::Vec;

use num_traits::AsPrimitive;

use crate::traits::SparseSquareMatrix;

/// Trait for detecting Delta-Y exchanges between edge-induced subgraphs.
///
/// Used as a post-filter on MCES cliques to discard matches arising from
/// the Whitney K₃/K₁,₃ exception. The methods compare sorted degree
/// sequences of edge-induced subgraphs, which is only meaningful when
/// the edge sets come from a valid line-graph clique correspondence
/// (see the [module-level docs](self) for details).
pub trait DeltaYExchange: SparseSquareMatrix {
    /// Computes the sorted degree sequence of the subgraph induced by `edges`.
    ///
    /// For each edge `(u, v)` in `edges`, increments the degree of both `u`
    /// and `v`. Returns the sorted vector of non-zero degrees.
    ///
    /// # Preconditions
    /// - All indices in `edges` must be less than `self.order()`.
    /// - Edges should be distinct and loop-free (`u != v`).
    /// - Violations of index bounds will panic; duplicates or self-loops will
    ///   silently produce incorrect degree counts.
    ///
    /// # Complexity
    /// O(|edges| + V) time, O(V) space.
    #[must_use]
    fn edge_subgraph_degree_sequence(&self, edges: &[(Self::Index, Self::Index)]) -> Vec<usize>;

    /// Returns `true` if the edge-induced subgraphs have different sorted
    /// degree sequences, indicating structural non-equivalence.
    ///
    /// In the MCES pipeline, a `true` result means the clique match
    /// corresponds to a Delta-Y exchange and should be discarded.
    #[must_use]
    fn has_delta_y_exchange<M: SparseSquareMatrix>(
        &self,
        self_edges: &[(Self::Index, Self::Index)],
        other: &M,
        other_edges: &[(M::Index, M::Index)],
    ) -> bool {
        self.edge_subgraph_degree_sequence(self_edges)
            != other.edge_subgraph_degree_sequence(other_edges)
    }
}

impl<G: SparseSquareMatrix> DeltaYExchange for G {
    fn edge_subgraph_degree_sequence(&self, edges: &[(Self::Index, Self::Index)]) -> Vec<usize> {
        let n: usize = self.order().as_();
        let mut counts = vec![0usize; n];
        for &(u, v) in edges {
            counts[u.as_()] += 1;
            counts[v.as_()] += 1;
        }
        let mut seq: Vec<usize> = counts.into_iter().filter(|&d| d > 0).collect();
        seq.sort_unstable();
        seq
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::impls::BitSquareMatrix;

    /// K3 (triangle) vs K1,3 (claw): classic Delta-Y case.
    /// K3 degree sequence: [2, 2, 2], K1,3: [1, 1, 1, 3].
    #[test]
    fn test_k3_vs_k1_3() {
        let k3 = BitSquareMatrix::from_symmetric_edges(3, vec![(0, 1), (0, 2), (1, 2)]);
        let k3_edges: Vec<(usize, usize)> = vec![(0, 1), (0, 2), (1, 2)];

        let k1_3 = BitSquareMatrix::from_symmetric_edges(4, vec![(0, 1), (0, 2), (0, 3)]);
        let k1_3_edges: Vec<(usize, usize)> = vec![(0, 1), (0, 2), (0, 3)];

        assert_eq!(k3.edge_subgraph_degree_sequence(&k3_edges), vec![2, 2, 2]);
        assert_eq!(k1_3.edge_subgraph_degree_sequence(&k1_3_edges), vec![1, 1, 1, 3]);
        assert!(k3.has_delta_y_exchange(&k3_edges, &k1_3, &k1_3_edges));
    }

    /// K3 vs K3: same structure, no exchange.
    #[test]
    fn test_k3_vs_k3() {
        let g1 = BitSquareMatrix::from_symmetric_edges(3, vec![(0, 1), (0, 2), (1, 2)]);
        let g2 = BitSquareMatrix::from_symmetric_edges(3, vec![(0, 1), (0, 2), (1, 2)]);
        let edges: Vec<(usize, usize)> = vec![(0, 1), (0, 2), (1, 2)];

        assert!(!g1.has_delta_y_exchange(&edges, &g2, &edges));
    }

    /// Single matched edge in both graphs: sequences [1, 1].
    #[test]
    fn test_single_edge() {
        let g1 = BitSquareMatrix::from_symmetric_edges(3, vec![(0, 1), (1, 2)]);
        let g2 = BitSquareMatrix::from_symmetric_edges(4, vec![(2, 3)]);
        let e1: Vec<(usize, usize)> = vec![(0, 1)];
        let e2: Vec<(usize, usize)> = vec![(2, 3)];

        assert_eq!(g1.edge_subgraph_degree_sequence(&e1), vec![1, 1]);
        assert_eq!(g2.edge_subgraph_degree_sequence(&e2), vec![1, 1]);
        assert!(!g1.has_delta_y_exchange(&e1, &g2, &e2));
    }

    /// Empty edge sets: both sequences empty, no exchange.
    #[test]
    fn test_empty_edges() {
        let g1 = BitSquareMatrix::from_symmetric_edges(3, vec![(0, 1)]);
        let g2 = BitSquareMatrix::from_symmetric_edges(3, vec![(1, 2)]);
        let empty: Vec<(usize, usize)> = vec![];

        assert_eq!(g1.edge_subgraph_degree_sequence(&empty), Vec::<usize>::new());
        assert!(!g1.has_delta_y_exchange(&empty, &g2, &empty));
    }

    /// P3 (path of 2 edges) in both: sorted degree sequences [1, 1, 2].
    #[test]
    fn test_path_vs_path() {
        let g1 = BitSquareMatrix::from_symmetric_edges(3, vec![(0, 1), (1, 2)]);
        let g2 = BitSquareMatrix::from_symmetric_edges(3, vec![(0, 1), (1, 2)]);
        let edges: Vec<(usize, usize)> = vec![(0, 1), (1, 2)];

        assert_eq!(g1.edge_subgraph_degree_sequence(&edges), vec![1, 1, 2]);
        assert!(!g1.has_delta_y_exchange(&edges, &g2, &edges));
    }

    /// Degree-sequence mismatch: 3 edges forming a path in K4 vs 3 spoke
    /// edges from K1,4. Not a true Delta-Y case (Whitney exception), but
    /// demonstrates that the filter catches structural non-equivalence.
    #[test]
    fn test_k4_path_vs_star() {
        // K4: take edges forming path 0-1-2-3 → [1, 1, 2, 2]
        let k4 = BitSquareMatrix::from_symmetric_edges(
            4,
            vec![(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)],
        );
        let path_edges: Vec<(usize, usize)> = vec![(0, 1), (1, 2), (2, 3)];

        // K1,4: take 3 spoke edges → [1, 1, 1, 3]
        let star = BitSquareMatrix::from_symmetric_edges(5, vec![(0, 1), (0, 2), (0, 3), (0, 4)]);
        let star_edges: Vec<(usize, usize)> = vec![(0, 1), (0, 2), (0, 3)];

        assert_eq!(k4.edge_subgraph_degree_sequence(&path_edges), vec![1, 1, 2, 2]);
        assert_eq!(star.edge_subgraph_degree_sequence(&star_edges), vec![1, 1, 1, 3]);
        assert!(k4.has_delta_y_exchange(&path_edges, &star, &star_edges));
    }
}
