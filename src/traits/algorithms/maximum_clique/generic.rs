//! Generic bit-parallel maximum clique search (MCSa variant).

use alloc::vec::Vec;

use bitvec::{order::Lsb0, vec::BitVec};

use crate::{
    impls::BitSquareMatrix,
    traits::{SizedRowsSparseMatrix2D, SparseMatrix2D, SquareMatrix},
};

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

/// Performs the generic branch-and-bound maximum clique search.
pub(crate) fn search<F>(
    adj: &BitSquareMatrix,
    enumerate: bool,
    mut accept_clique: F,
) -> Vec<Vec<usize>>
where
    F: FnMut(&[usize]) -> bool,
{
    let n = adj.order();
    if n == 0 {
        let empty = Vec::new();
        return if accept_clique(&empty) { vec![empty] } else { Vec::new() };
    }

    let order = degeneracy_ordering(adj);
    let p_mask: BitVec<usize, Lsb0> = BitVec::repeat(true, n);
    let (candidates, colors) = greedy_color_candidates(adj, &order, &p_mask);

    let mut best_size: usize = 0;
    let mut best_cliques: Vec<Vec<usize>> = Vec::new();
    let mut clique: Vec<usize> = Vec::new();
    let mut stack: Vec<Frame> = Vec::new();
    stack.push(Frame { candidates, colors, p_mask, next_idx: 0 });

    while let Some(frame) = stack.last_mut() {
        let depth = clique.len();
        let cand_len = frame.candidates.len();

        if frame.next_idx >= cand_len {
            if !clique.is_empty() {
                clique.pop();
            }
            stack.pop();
            continue;
        }

        let pos = cand_len - 1 - frame.next_idx;
        let v = frame.candidates[pos];
        let upper = depth + frame.colors[pos];

        if enumerate {
            if upper < best_size {
                if !clique.is_empty() {
                    clique.pop();
                }
                stack.pop();
                continue;
            }
        } else if upper <= best_size {
            if !clique.is_empty() {
                clique.pop();
            }
            stack.pop();
            continue;
        }

        frame.next_idx += 1;
        frame.p_mask.set(v, false);

        let new_p = adj.row_and(v, &frame.p_mask);
        let new_p_count = new_p.count_ones();

        clique.push(v);
        let new_depth = clique.len();

        if new_p_count == 0 {
            let is_candidate = new_depth > best_size || (enumerate && new_depth == best_size);
            if is_candidate {
                let mut candidate = clique.clone();
                candidate.sort_unstable();
                if accept_clique(&candidate) {
                    if new_depth > best_size {
                        best_size = new_depth;
                        best_cliques.clear();
                    }
                    best_cliques.push(candidate);
                }
            }
            clique.pop();
        } else {
            let new_verts: Vec<usize> = new_p.iter_ones().collect();
            let (cands, cols) = greedy_color_candidates(adj, &new_verts, &new_p);
            let new_ub = cols.last().copied().unwrap_or(0);
            let achievable = new_depth + new_ub;
            let dominated =
                if enumerate { achievable < best_size } else { achievable <= best_size };

            if dominated {
                clique.pop();
            } else {
                stack.push(Frame { candidates: cands, colors: cols, p_mask: new_p, next_idx: 0 });
            }
        }
    }

    best_cliques
}

/// Computes the degeneracy ordering (smallest-last) of the graph.
fn degeneracy_ordering(adj: &BitSquareMatrix) -> Vec<usize> {
    let n = adj.order();
    debug_assert!(n > 0, "degeneracy_ordering called on empty graph");

    let mut degree: Vec<usize> = (0..n).map(|v| adj.number_of_defined_values_in_row(v)).collect();
    let max_deg = degree.iter().copied().max().unwrap_or(0);

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
        while buckets[min_bucket].is_empty() {
            min_bucket += 1;
        }

        let v = buckets[min_bucket].pop().unwrap();
        removed[v] = true;
        order.push(v);

        for u in adj.sparse_row(v) {
            if !removed[u] {
                let old_d = degree[u];
                debug_assert!(old_d > 0, "non-removed neighbor has degree 0");
                let new_d = old_d - 1;
                degree[u] = new_d;

                let pos = bucket_pos[u];
                let last = buckets[old_d].len() - 1;
                if pos != last {
                    let other = buckets[old_d][last];
                    buckets[old_d][pos] = other;
                    bucket_pos[other] = pos;
                }
                buckets[old_d].pop();

                bucket_pos[u] = buckets[new_d].len();
                buckets[new_d].push(u);

                if new_d < min_bucket {
                    min_bucket = new_d;
                }
            }
        }
    }

    order
}

/// Greedy sequential coloring of candidate vertices.
fn greedy_color_candidates(
    adj: &BitSquareMatrix,
    vertex_order: &[usize],
    p_mask: &BitVec,
) -> (Vec<usize>, Vec<usize>) {
    let n = p_mask.len();
    let mut color_of: Vec<usize> = vec![0; n];
    let mut max_color: usize = 0;
    let mut used: BitVec<usize, Lsb0> = BitVec::repeat(false, 2);
    let mut color_buckets: Vec<Vec<usize>> = Vec::new();

    for &v in vertex_order {
        debug_assert!(p_mask[v], "vertex_order contains vertex not in p_mask");

        used.fill(false);
        if max_color + 2 > used.len() {
            used.resize(max_color + 2, false);
        }
        for neighbor in adj.sparse_row(v) {
            if p_mask[neighbor] && color_of[neighbor] > 0 {
                used.set(color_of[neighbor], true);
            }
        }

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
        let g = BitSquareMatrix::from_symmetric_edges(5, vec![(0, 1), (0, 2), (0, 3), (0, 4)]);
        let order = degeneracy_ordering(&g);
        assert_eq!(order.len(), 5);
        let mut sorted = order.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, vec![0, 1, 2, 3, 4]);
        for &v in &order[..3] {
            assert_ne!(v, 0, "center should not be among the first 3 removed");
        }
    }

    #[test]
    fn test_degeneracy_ordering_path() {
        let g = BitSquareMatrix::from_symmetric_edges(4, vec![(0, 1), (1, 2), (2, 3)]);
        let order = degeneracy_ordering(&g);
        assert_eq!(order.len(), 4);
        let mut sorted = order.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, vec![0, 1, 2, 3]);
        let first_two: Vec<usize> = order[..2].to_vec();
        assert!(first_two.contains(&0) || first_two.contains(&3));
    }

    #[test]
    fn test_degeneracy_ordering_isolated() {
        let g = BitSquareMatrix::from_symmetric_edges(4, vec![]);
        let order = degeneracy_ordering(&g);
        assert_eq!(order.len(), 4);
        let mut sorted = order.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, vec![0, 1, 2, 3]);
    }
}
