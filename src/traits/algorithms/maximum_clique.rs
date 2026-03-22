//! Branch-and-bound maximum clique enumeration (MCSa variant).
//!
//! Finds all cliques of maximum size in an undirected graph stored as a
//! [`BitSquareMatrix`]. The root-level vertex ordering uses degeneracy
//! ordering (smallest-last / minimum-width), which processes periphery
//! vertices first and dense-core vertices last. Combined with greedy
//! sequential coloring as an upper bound and bitset-parallel candidate-set
//! operations, this corresponds to the MCSa variant from Prosser (2012),
//! which consistently outperforms simple degree ordering (MCQ) on standard
//! benchmarks.
//!
//! Two modes are supported:
//! - **Single**: find one maximum clique (prune branches that can only tie).
//! - **Enumerate**: find all maximum cliques (allow branches that tie).
//!
//! # Complexity
//! O(3^(n/3)) worst case; much better in practice with coloring pruning.
//!
//! # References
//! - Matula, Beck (1983). "Smallest-last ordering and clustering and graph
//!   coloring algorithms." *JACM* 30(3):417-427.
//! - Tomita, Seki (2003). "An efficient branch-and-bound algorithm for finding
//!   a maximum clique." *LNCS* 2731:278-289.
//! - Tomita, Sutani, Higashi, Takahashi, Wakatsuki (2010). "A simple and faster
//!   branch-and-bound algorithm for finding a maximum clique." *LNCS*
//!   6213:191-203.
//! - San Segundo, Rodriguez-Losada, Jimenez (2011). "An exact bit-parallel
//!   algorithm for the maximum clique problem." *Computers & OR* 38(2).
//! - Prosser (2012). "Exact Algorithms for Maximum Clique: A Computational
//!   Study." *Algorithms* 5(4):545-587.
//!
//! # Example
//! ```
//! use geometric_traits::{impls::BitSquareMatrix, prelude::*};
//!
//! // K4 (complete graph on 4 vertices)
//! let k4 = BitSquareMatrix::from_symmetric_edges(
//!     4,
//!     vec![(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)],
//! );
//! let cliques = k4.all_maximum_cliques();
//! assert_eq!(cliques.len(), 1);
//! assert_eq!(cliques[0].len(), 4);
//! ```

use alloc::vec::Vec;

use bitvec::{order::Lsb0, vec::BitVec};

use crate::{
    impls::BitSquareMatrix,
    traits::{SizedRowsSparseMatrix2D, SparseMatrix2D, SquareMatrix},
};

/// Trait for finding maximum cliques in an undirected graph.
pub trait MaximumClique {
    /// Returns one maximum clique (a clique of the largest possible size).
    ///
    /// The returned `Vec<usize>` contains the vertex indices of the clique,
    /// sorted in ascending order.
    #[must_use]
    fn maximum_clique(&self) -> Vec<usize>;

    /// Returns all maximum cliques (all cliques whose size equals ω(G)).
    ///
    /// Each clique is a `Vec<usize>` of vertex indices sorted in ascending
    /// order. The outer vector contains no duplicates.
    #[must_use]
    fn all_maximum_cliques(&self) -> Vec<Vec<usize>>;

    /// Returns one maximum clique subject to a partition constraint.
    ///
    /// `partition[v]` is the group index of vertex `v`. At most one vertex
    /// per group may appear in the clique. The partition also provides a
    /// tighter upper bound (number of non-empty groups in candidate set).
    ///
    /// `partition.len()` must equal `self.order()`.
    #[must_use]
    fn maximum_clique_with_partition(&self, partition: &[usize]) -> Vec<usize>;

    /// Returns all maximum cliques subject to a partition constraint.
    ///
    /// `partition[v]` is the group index of vertex `v`. At most one vertex
    /// per group may appear in any returned clique.
    ///
    /// `partition.len()` must equal `self.order()`.
    #[must_use]
    fn all_maximum_cliques_with_partition(&self, partition: &[usize]) -> Vec<Vec<usize>>;
}

impl MaximumClique for BitSquareMatrix {
    fn maximum_clique(&self) -> Vec<usize> {
        let results = bb_clique_search(self, false, None);
        results.into_iter().next().unwrap_or_default()
    }

    fn all_maximum_cliques(&self) -> Vec<Vec<usize>> {
        bb_clique_search(self, true, None)
    }

    fn maximum_clique_with_partition(&self, partition: &[usize]) -> Vec<usize> {
        debug_assert_eq!(partition.len(), self.order(), "partition length must equal graph order");
        let results = bb_clique_search(self, false, Some(partition));
        results.into_iter().next().unwrap_or_default()
    }

    fn all_maximum_cliques_with_partition(&self, partition: &[usize]) -> Vec<Vec<usize>> {
        debug_assert_eq!(partition.len(), self.order(), "partition length must equal graph order");
        bb_clique_search(self, true, Some(partition))
    }
}

// ============================================================================
// Internal algorithm
// ============================================================================

/// Stack frame for the iterative branch-and-bound search.
struct Frame {
    /// Candidate vertices, ordered by color (ascending color number).
    /// We process from the end (highest color) toward the front.
    candidates: Vec<usize>,
    /// Color number for each candidate (parallel to `candidates`).
    colors: Vec<usize>,
    /// Candidate bitmask (vertices still available for branching).
    p_mask: BitVec,
    /// Index into `candidates` for the next vertex to try (counting from end).
    next_idx: usize,
}

/// Counts the number of distinct partition groups with at least one vertex
/// set in `mask`.
fn partition_upper_bound(partition: &[usize], mask: &BitVec, seen: &mut [bool]) -> usize {
    seen.fill(false);
    let mut count = 0;
    for v in mask.iter_ones() {
        let g = partition[v];
        if !seen[g] {
            seen[g] = true;
            count += 1;
        }
    }
    count
}

/// Clears bits in `mask` for vertices that share the same partition group
/// as `selected_vertex`.
fn clear_same_group(partition: &[usize], mask: &mut BitVec, selected_vertex: usize) {
    let group = partition[selected_vertex];
    let to_clear: Vec<usize> = mask.iter_ones().filter(|&u| partition[u] == group).collect();
    for u in to_clear {
        mask.set(u, false);
    }
}

/// Performs the branch-and-bound maximum clique search.
///
/// When `enumerate` is true, finds ALL maximum cliques.
/// When false, finds one maximum clique (prunes ties).
///
/// When `partition` is `Some`, enforces that at most one vertex per
/// partition group appears in any clique, and uses the partition group
/// count as an additional (tighter) upper bound.
fn bb_clique_search(
    adj: &BitSquareMatrix,
    enumerate: bool,
    partition: Option<&[usize]>,
) -> Vec<Vec<usize>> {
    let n = adj.order();
    if n == 0 {
        return vec![vec![]];
    }

    // Precompute partition metadata.
    let num_groups = partition.map_or(0, |p| p.iter().copied().max().unwrap_or(0) + 1);
    let mut seen_buf = vec![false; num_groups];

    // Initial vertex ordering by degeneracy (smallest-last).
    let order = degeneracy_ordering(adj);

    // Initial candidate set: all vertices.
    let p_mask: BitVec<usize, Lsb0> = BitVec::repeat(true, n);

    // Color the initial candidate set.
    let (candidates, colors) = greedy_color_candidates(adj, &order, &p_mask);

    let mut best_size: usize = 0;
    let mut best_cliques: Vec<Vec<usize>> = Vec::new();
    let mut clique: Vec<usize> = Vec::new();

    // Stack of frames for iterative DFS.
    let mut stack: Vec<Frame> = Vec::new();
    stack.push(Frame { candidates, colors, p_mask, next_idx: 0 });

    while let Some(frame) = stack.last_mut() {
        let depth = clique.len();
        let cand_len = frame.candidates.len();

        if frame.next_idx >= cand_len {
            // Exhausted all candidates at this level.
            if !clique.is_empty() {
                clique.pop();
            }
            stack.pop();
            continue;
        }

        // Pick the next candidate from the end (highest color first).
        let pos = cand_len - 1 - frame.next_idx;
        let v = frame.candidates[pos];
        let color = frame.colors[pos];

        // Prune: check if this branch can improve on (or match) the best.
        let mut upper = depth + color; // color is 1-indexed, so depth + color = max achievable
        if let Some(part) = partition {
            upper = upper.min(depth + partition_upper_bound(part, &frame.p_mask, &mut seen_buf));
        }
        if enumerate {
            if upper < best_size {
                // Cannot reach best_size; backtrack.
                if !clique.is_empty() {
                    clique.pop();
                }
                stack.pop();
                continue;
            }
        } else if upper <= best_size {
            // Cannot beat best_size; backtrack.
            if !clique.is_empty() {
                clique.pop();
            }
            stack.pop();
            continue;
        }

        frame.next_idx += 1;

        // Remove v from the candidate mask for sibling branches.
        frame.p_mask.set(v, false);

        // Compute new candidate set: P ∩ N(v).
        let mut new_p = adj.row_and(v, &frame.p_mask);

        // Partition constraint: remove same-group vertices from candidates.
        if let Some(part) = partition {
            clear_same_group(part, &mut new_p, v);
        }

        let new_p_count = new_p.count_ones();

        // Add v to the current clique.
        clique.push(v);
        let new_depth = clique.len();

        if new_p_count == 0 {
            // Leaf: no more candidates.
            if new_depth > best_size {
                best_size = new_depth;
                best_cliques.clear();
                let mut c = clique.clone();
                c.sort_unstable();
                best_cliques.push(c);
            } else if enumerate && new_depth == best_size {
                let mut c = clique.clone();
                c.sort_unstable();
                best_cliques.push(c);
            }
            clique.pop();
        } else {
            // Collect vertices in new_p and color them.
            let new_verts: Vec<usize> = new_p.iter_ones().collect();
            let (cands, cols) = greedy_color_candidates(adj, &new_verts, &new_p);
            let new_ub = cols.last().copied().unwrap_or(0);

            // Combine coloring bound with partition bound.
            let mut achievable = new_depth + new_ub;
            if let Some(part) = partition {
                achievable =
                    achievable.min(new_depth + partition_upper_bound(part, &new_p, &mut seen_buf));
            }

            let dominated =
                if enumerate { achievable < best_size } else { achievable <= best_size };

            if dominated {
                clique.pop();
            } else {
                stack.push(Frame { candidates: cands, colors: cols, p_mask: new_p, next_idx: 0 });
                // clique keeps v; it will be popped when this frame is
                // exhausted.
            }
        }
    }

    best_cliques
}

/// Computes the degeneracy ordering (smallest-last) of the graph.
///
/// Uses the Matula & Beck (1983) bucket-queue algorithm: repeatedly extract
/// the minimum-degree vertex, append to the output, and decrement its
/// neighbors' degrees. The returned ordering lists periphery vertices first
/// and dense-core vertices last.
///
/// O(V+E) time, O(V) space.
fn degeneracy_ordering(adj: &BitSquareMatrix) -> Vec<usize> {
    let n = adj.order();
    debug_assert!(n > 0, "degeneracy_ordering called on empty graph");

    // Compute initial degrees.
    let mut degree: Vec<usize> = (0..n).map(|v| adj.number_of_defined_values_in_row(v)).collect();
    let max_deg = degree.iter().copied().max().unwrap_or(0);

    // Bucket array with O(1) swap-remove. bucket[d] holds vertices with degree d.
    let mut buckets: Vec<Vec<usize>> = vec![Vec::new(); max_deg + 1];
    let mut bucket_pos: Vec<usize> = vec![0; n];

    for v in 0..n {
        bucket_pos[v] = buckets[degree[v]].len();
        buckets[degree[v]].push(v);
    }

    let mut removed = vec![false; n];
    let mut order = Vec::with_capacity(n);
    let mut min_bucket: usize = 0;

    for _ in 0..n {
        // Find the lowest non-empty bucket.
        while buckets[min_bucket].is_empty() {
            min_bucket += 1;
        }

        // Extract last vertex from this bucket (O(1) pop).
        let v = buckets[min_bucket].pop().unwrap();
        removed[v] = true;
        order.push(v);

        // Decrement degree of non-removed neighbors and move them in buckets.
        for u in adj.sparse_row(v) {
            if !removed[u] {
                let old_d = degree[u];
                debug_assert!(old_d > 0, "non-removed neighbor has degree 0");
                let new_d = old_d - 1;
                degree[u] = new_d;

                // Remove u from bucket[old_d] by swapping with last element.
                let pos = bucket_pos[u];
                let last = buckets[old_d].len() - 1;
                if pos != last {
                    let other = buckets[old_d][last];
                    buckets[old_d][pos] = other;
                    bucket_pos[other] = pos;
                }
                buckets[old_d].pop();

                // Insert u into bucket[new_d].
                bucket_pos[u] = buckets[new_d].len();
                buckets[new_d].push(u);

                // min_bucket can decrease.
                if new_d < min_bucket {
                    min_bucket = new_d;
                }
            }
        }
    }

    order
}

/// Greedy sequential coloring of candidate vertices.
///
/// Given a set of candidate vertices (ordered by `vertex_order`) restricted
/// to those set in `p_mask`, assigns colors 1, 2, ... using a greedy
/// sequential strategy: each vertex gets the smallest color not used by
/// any of its already-colored neighbors in the candidate set.
///
/// Returns `(candidates, colors)` where both vectors are sorted by color
/// (ascending). The number of colors used equals `colors.last()`.
fn greedy_color_candidates(
    adj: &BitSquareMatrix,
    vertex_order: &[usize],
    p_mask: &BitVec,
) -> (Vec<usize>, Vec<usize>) {
    let n = p_mask.len();
    let mut color_of: Vec<usize> = vec![0; n]; // 0 = uncolored
    let mut max_color: usize = 0;
    let mut used: BitVec<usize, Lsb0> = BitVec::repeat(false, 2);
    let mut color_buckets: Vec<Vec<usize>> = Vec::new();

    // Color each vertex in the given order.
    for &v in vertex_order {
        debug_assert!(p_mask[v], "vertex_order contains vertex not in p_mask");

        // Find colors used by neighbors of v that are in p_mask and already colored.
        used.fill(false);
        if max_color + 2 > used.len() {
            used.resize(max_color + 2, false);
        }
        for neighbor in adj.sparse_row(v) {
            if p_mask[neighbor] && color_of[neighbor] > 0 {
                used.set(color_of[neighbor], true);
            }
        }

        // Find smallest unused color (starting from 1).
        let mut c = 1;
        while c < used.len() && used[c] {
            c += 1;
        }
        color_of[v] = c;
        if c > max_color {
            max_color = c;
            color_buckets.resize_with(c + 1, Vec::new);
        }
        color_buckets[c].push(v);
    }

    // Flatten buckets into candidates and colors (ascending by color).
    let total: usize = color_buckets.iter().map(Vec::len).sum();
    let mut candidates = Vec::with_capacity(total);
    let mut colors = Vec::with_capacity(total);
    for (c, bucket) in color_buckets.iter().enumerate().skip(1) {
        for &v in bucket {
            candidates.push(v);
            colors.push(c);
        }
    }

    (candidates, colors)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_degeneracy_ordering_single_vertex() {
        let g = BitSquareMatrix::from_symmetric_edges(1, vec![]);
        assert_eq!(degeneracy_ordering(&g), vec![0]);
    }

    #[test]
    fn test_degeneracy_ordering_k4() {
        let g = BitSquareMatrix::from_symmetric_edges(
            4,
            vec![(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)],
        );
        let order = degeneracy_ordering(&g);
        assert_eq!(order.len(), 4);
        let mut sorted = order.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_degeneracy_ordering_star() {
        // Star S4: center 0, leaves 1-4.
        let g = BitSquareMatrix::from_symmetric_edges(5, vec![(0, 1), (0, 2), (0, 3), (0, 4)]);
        let order = degeneracy_ordering(&g);
        assert_eq!(order.len(), 5);
        // All vertices present.
        let mut sorted = order.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, vec![0, 1, 2, 3, 4]);
        // First 3 entries must be leaves (degree 1 < center's degree).
        // After removing 3 leaves the center and last leaf both have degree 1,
        // so their relative order is implementation-defined.
        for &v in &order[..3] {
            assert_ne!(v, 0, "center should not be among the first 3 removed");
        }
    }

    #[test]
    fn test_degeneracy_ordering_path() {
        // Path: 0-1-2-3
        let g = BitSquareMatrix::from_symmetric_edges(4, vec![(0, 1), (1, 2), (2, 3)]);
        let order = degeneracy_ordering(&g);
        assert_eq!(order.len(), 4);
        let mut sorted = order.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, vec![0, 1, 2, 3]);
        // Endpoints (degree 1) are removed first.
        let first_two: Vec<usize> = order[..2].to_vec();
        assert!(first_two.contains(&0) || first_two.contains(&3));
    }

    #[test]
    fn test_degeneracy_ordering_isolated() {
        // 4 isolated vertices — all degree 0.
        let g = BitSquareMatrix::from_symmetric_edges(4, vec![]);
        let order = degeneracy_ordering(&g);
        assert_eq!(order.len(), 4);
        let mut sorted = order.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, vec![0, 1, 2, 3]);
    }
}
