//! Partition-driven exact maximum clique search for MCES-style workloads.

use alloc::vec::Vec;

use crate::{
    impls::BitSquareMatrix,
    traits::{SparseMatrix2D, SquareMatrix},
};

/// Which side of the pair list should define the search partitions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PartitionSide {
    /// Group product vertices by the first graph's bond ids.
    First,
    /// Group product vertices by the second graph's bond ids.
    Second,
}

/// Partition information for MCES-aware maximum clique search.
///
/// Contains the double-sided partition `(g1_bond_idx, g2_bond_idx)` and
/// per-bond label indices for the label-aware upper bound.
pub struct PartitionInfo<'a> {
    /// For each modular product vertex: `(g1_bond_idx, g2_bond_idx)`.
    pub pairs: &'a [(usize, usize)],
    /// For each G1 bond: its label index.
    pub g1_labels: &'a [usize],
    /// For each G2 bond: its label index.
    pub g2_labels: &'a [usize],
    /// Number of distinct label types.
    pub num_labels: usize,
    /// Which side should define the partition buckets.
    pub partition_side: PartitionSide,
}

/// Owned partition labels for convenience.
pub struct OwnedPartitionLabels {
    /// G1 labels (all zeros for unlabeled).
    pub g1_labels: Vec<usize>,
    /// G2 labels (all zeros for unlabeled).
    pub g2_labels: Vec<usize>,
    /// Number of label types.
    pub num_labels: usize,
}

impl OwnedPartitionLabels {
    /// Creates unlabeled partition labels (all bonds get label 0).
    #[must_use]
    pub fn unlabeled(pairs: &[(usize, usize)]) -> Self {
        let num_g1 = pairs.iter().map(|&(g1, _)| g1).max().map_or(0, |m| m + 1);
        let num_g2 = pairs.iter().map(|&(_, g2)| g2).max().map_or(0, |m| m + 1);
        Self { g1_labels: vec![0; num_g1], g2_labels: vec![0; num_g2], num_labels: 1 }
    }

    /// Creates a [`PartitionInfo`] borrowing from this data and the given
    /// pairs.
    #[must_use]
    pub fn as_info<'a>(&'a self, pairs: &'a [(usize, usize)]) -> PartitionInfo<'a> {
        PartitionInfo {
            pairs,
            g1_labels: &self.g1_labels,
            g2_labels: &self.g2_labels,
            num_labels: self.num_labels,
            partition_side: PartitionSide::First,
        }
    }
}

#[derive(Clone)]
struct PartitionSearchState<'a> {
    adj: &'a BitSquareMatrix,
    info: &'a PartitionInfo<'a>,
    parts: Vec<Vec<usize>>,
    g1_counts: Vec<usize>,
    g2_counts: Vec<usize>,
    g1_type_counts: Vec<usize>,
    g2_type_counts: Vec<usize>,
}

fn vertices_compatible(
    adj: &BitSquareMatrix,
    info: &PartitionInfo<'_>,
    left: usize,
    right: usize,
) -> bool {
    let (left_g1, left_g2) = info.pairs[left];
    let (right_g1, right_g2) = info.pairs[right];
    left_g1 != right_g1 && left_g2 != right_g2 && adj.has_entry(left, right)
}

impl<'a> PartitionSearchState<'a> {
    fn new(adj: &'a BitSquareMatrix, info: &'a PartitionInfo<'a>, lower_bound: usize) -> Self {
        let mut parts_by_side = vec![
            Vec::new();
            match info.partition_side {
                PartitionSide::First => info.g1_labels.len(),
                PartitionSide::Second => info.g2_labels.len(),
            }
        ];
        let mut g1_counts = vec![0; info.g1_labels.len()];
        let mut g2_counts = vec![0; info.g2_labels.len()];

        for (vertex, &(g1, g2)) in info.pairs.iter().enumerate() {
            let partition_index = match info.partition_side {
                PartitionSide::First => g1,
                PartitionSide::Second => g2,
            };
            parts_by_side[partition_index].push(vertex);
            g1_counts[g1] += 1;
            g2_counts[g2] += 1;
        }

        let mut state = Self {
            adj,
            info,
            parts: parts_by_side.into_iter().filter(|part| !part.is_empty()).collect(),
            g1_counts,
            g2_counts,
            g1_type_counts: vec![0; info.num_labels.max(1)],
            g2_type_counts: vec![0; info.num_labels.max(1)],
        };
        state.sort_partitions();
        state.reassign_vertices(lower_bound);
        state.parts.retain(|part| !part.is_empty());
        state.sort_partitions();
        state.recalculate_type_counts();
        state
    }

    fn is_empty(&self) -> bool {
        self.parts.is_empty()
    }

    fn num_parts(&self) -> usize {
        self.parts.len()
    }

    fn upper_bound(&self) -> usize {
        (0..self.g1_type_counts.len())
            .map(|label| self.g1_type_counts[label].min(self.g2_type_counts[label]))
            .sum()
    }

    fn pop_last_vertex(&mut self) -> usize {
        let part = self.parts.last_mut().expect("partition set is empty");
        let vertex = part.pop().expect("last partition is empty");
        if part.is_empty() {
            self.parts.pop();
        }
        self.decrement_vertex_counts(vertex);
        vertex
    }

    fn prune_vertices(&mut self, selected_vertex: usize) {
        let adj = self.adj;
        let info = self.info;
        let mut removed_vertices = Vec::new();
        for part in &mut self.parts {
            part.retain(|&vertex| {
                let keep = vertices_compatible(adj, info, vertex, selected_vertex);
                if !keep {
                    removed_vertices.push(vertex);
                }
                keep
            });
        }
        for vertex in removed_vertices {
            self.decrement_vertex_counts(vertex);
        }
        self.parts.retain(|part| !part.is_empty());
        self.sort_partitions();
    }

    fn sort_partitions(&mut self) {
        for part in &mut self.parts {
            part.sort_unstable();
        }
        self.parts.sort_by(|left, right| {
            right.len().cmp(&left.len()).then_with(|| left.first().cmp(&right.first()))
        });
    }

    fn reassign_vertices(&mut self, lower_bound: usize) {
        if self.parts.len() <= 1 {
            return;
        }
        let pivot = lower_bound.min(self.parts.len().saturating_sub(1));
        for index in (pivot + 1..self.parts.len()).rev() {
            let mut moved = Vec::new();
            let source_vertices = self.parts[index].clone();
            for vertex in source_vertices {
                for target in 0..=pivot {
                    if self.parts[target].iter().all(|&existing| {
                        !vertices_compatible(self.adj, self.info, vertex, existing)
                    }) {
                        self.parts[target].push(vertex);
                        moved.push(vertex);
                        break;
                    }
                }
            }
            if !moved.is_empty() {
                self.parts[index].retain(|vertex| !moved.contains(vertex));
            }
        }
    }

    fn recalculate_type_counts(&mut self) {
        self.g1_type_counts.fill(0);
        self.g2_type_counts.fill(0);

        for (g1, &count) in self.g1_counts.iter().enumerate() {
            if count > 0 {
                self.g1_type_counts[self.info.g1_labels[g1]] += 1;
            }
        }
        for (g2, &count) in self.g2_counts.iter().enumerate() {
            if count > 0 {
                self.g2_type_counts[self.info.g2_labels[g2]] += 1;
            }
        }
    }

    fn decrement_vertex_counts(&mut self, vertex: usize) {
        let (g1, g2) = self.info.pairs[vertex];
        self.g1_counts[g1] -= 1;
        if self.g1_counts[g1] == 0 {
            self.g1_type_counts[self.info.g1_labels[g1]] -= 1;
        }
        self.g2_counts[g2] -= 1;
        if self.g2_counts[g2] == 0 {
            self.g2_type_counts[self.info.g2_labels[g2]] -= 1;
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct PartitionProfile {
    sum_sq_widths: usize,
    max_width: usize,
    non_empty_partitions: usize,
}

fn partition_profile(
    pairs: &[(usize, usize)],
    side_len: usize,
    side: PartitionSide,
) -> PartitionProfile {
    let mut widths = vec![0usize; side_len];
    for &(g1, g2) in pairs {
        let index = match side {
            PartitionSide::First => g1,
            PartitionSide::Second => g2,
        };
        widths[index] += 1;
    }
    PartitionProfile {
        sum_sq_widths: widths.iter().map(|width| width * width).sum(),
        max_width: widths.iter().copied().max().unwrap_or(0),
        non_empty_partitions: widths.into_iter().filter(|&width| width > 0).count(),
    }
}

/// Chooses the partition side with the flatter initial bucket profile.
#[must_use]
pub fn choose_partition_side(
    pairs: &[(usize, usize)],
    g1_len: usize,
    g2_len: usize,
) -> PartitionSide {
    let first = partition_profile(pairs, g1_len, PartitionSide::First);
    let second = partition_profile(pairs, g2_len, PartitionSide::Second);
    if second < first { PartitionSide::Second } else { PartitionSide::First }
}

/// Performs the partition-driven maximum clique search.
pub(crate) fn search<F>(
    adj: &BitSquareMatrix,
    partition: &PartitionInfo<'_>,
    enumerate: bool,
    initial_lower_bound: usize,
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

    let state = PartitionSearchState::new(adj, partition, initial_lower_bound);
    let mut clique = Vec::new();
    let mut best_size = initial_lower_bound;
    let mut best_cliques = Vec::new();

    dfs(state, &mut clique, enumerate, &mut best_size, &mut best_cliques, &mut accept_clique);

    best_cliques
}

fn dfs<F>(
    state: PartitionSearchState<'_>,
    clique: &mut Vec<usize>,
    enumerate: bool,
    best_size: &mut usize,
    best_cliques: &mut Vec<Vec<usize>>,
    accept_clique: &mut F,
) where
    F: FnMut(&[usize]) -> bool,
{
    if state.is_empty() {
        return;
    }

    let parts_bound = clique.len() + state.num_parts();
    if enumerate {
        if parts_bound < *best_size {
            return;
        }
    } else if parts_bound <= *best_size {
        return;
    }

    let label_bound = clique.len() + state.upper_bound();
    if enumerate {
        if label_bound < *best_size {
            return;
        }
    } else if label_bound <= *best_size {
        return;
    }

    let mut without_vertex = state;
    let selected = without_vertex.pop_last_vertex();
    let mut with_vertex = without_vertex.clone();

    clique.push(selected);
    maybe_update_best(clique, enumerate, best_size, best_cliques, accept_clique);
    with_vertex.prune_vertices(selected);
    dfs(with_vertex, clique, enumerate, best_size, best_cliques, accept_clique);
    clique.pop();

    if !without_vertex.is_empty() {
        dfs(without_vertex, clique, enumerate, best_size, best_cliques, accept_clique);
    }
}

fn maybe_update_best<F>(
    clique: &[usize],
    enumerate: bool,
    best_size: &mut usize,
    best_cliques: &mut Vec<Vec<usize>>,
    accept_clique: &mut F,
) where
    F: FnMut(&[usize]) -> bool,
{
    let size = clique.len();
    if size < *best_size {
        return;
    }
    if size == *best_size && !enumerate && !best_cliques.is_empty() {
        return;
    }

    let mut candidate = clique.to_vec();
    candidate.sort_unstable();
    if !accept_clique(&candidate) {
        return;
    }

    if size > *best_size {
        *best_size = size;
        best_cliques.clear();
    }

    if (best_cliques.is_empty() || enumerate || size > best_cliques[0].len())
        && !best_cliques.contains(&candidate)
    {
        best_cliques.push(candidate);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_partition_state_handles_unsorted_pairs() {
        let adj = BitSquareMatrix::from_symmetric_edges(4, vec![(0, 1), (0, 2), (1, 2), (1, 3)]);
        let pairs = vec![(2, 0), (0, 0), (2, 1), (1, 1)];
        let labels = OwnedPartitionLabels::unlabeled(&pairs);
        let info = labels.as_info(&pairs);
        let state = PartitionSearchState::new(&adj, &info, 0);

        assert_eq!(state.num_parts(), 3);
        assert!(state.parts.iter().all(|part| !part.is_empty()));
    }

    #[test]
    fn test_partition_state_reassigns_vertices_into_lower_partitions() {
        let adj = BitSquareMatrix::from_edges(4, vec![(1, 2), (2, 1), (2, 3), (3, 2)]);
        let pairs = vec![(0, 0), (1, 1), (2, 2), (3, 3)];
        let labels = OwnedPartitionLabels::unlabeled(&pairs);
        let info = labels.as_info(&pairs);
        let state = PartitionSearchState::new(&adj, &info, 0);

        assert_eq!(state.num_parts(), 3);
        assert!(state.parts.iter().any(|part| part.len() == 2));
    }

    #[test]
    fn test_partition_state_upper_bound_tracks_label_projection() {
        let adj = BitSquareMatrix::from_edges(3, vec![(0, 1), (1, 0)]);
        let pairs = vec![(0, 0), (1, 1), (2, 1)];
        let labels =
            OwnedPartitionLabels { g1_labels: vec![0, 1, 1], g2_labels: vec![0, 1], num_labels: 2 };
        let info = labels.as_info(&pairs);
        let mut state = PartitionSearchState::new(&adj, &info, 0);
        assert_eq!(state.upper_bound(), 2);

        let removed = state.pop_last_vertex();
        state.prune_vertices(removed);
        assert!(state.upper_bound() <= 2);
    }

    #[test]
    fn test_partition_profile_prefers_flatter_buckets() {
        let pairs = vec![(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1)];

        assert_eq!(choose_partition_side(&pairs, 3, 3), PartitionSide::First);
    }

    #[test]
    fn test_partition_state_can_group_by_second_side() {
        let adj = BitSquareMatrix::from_symmetric_edges(4, vec![(0, 1), (0, 2), (1, 2), (1, 3)]);
        let pairs = vec![(2, 0), (0, 0), (2, 1), (1, 1)];
        let labels = OwnedPartitionLabels::unlabeled(&pairs);
        let mut info = labels.as_info(&pairs);
        info.partition_side = PartitionSide::Second;
        let state = PartitionSearchState::new(&adj, &info, 0);

        assert_eq!(state.num_parts(), 2);
        assert!(state.parts.iter().all(|part| !part.is_empty()));
    }
}
