//! Partition-driven exact maximum clique search for MCES-style workloads.

use alloc::{format, string::String, vec::Vec};

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
    non_empty_parts: usize,
    active_parts: Vec<usize>,
    active_positions: Vec<usize>,
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
            non_empty_parts: 0,
            active_parts: Vec::new(),
            active_positions: Vec::new(),
            g1_counts,
            g2_counts,
            g1_type_counts: vec![0; info.num_labels.max(1)],
            g2_type_counts: vec![0; info.num_labels.max(1)],
        };
        state.sort_partitions();
        state.reassign_vertices(lower_bound);
        state.parts.retain(|part| !part.is_empty());
        state.sort_partitions();
        state.non_empty_parts = state.parts.len();
        state.active_parts = (0..state.parts.len()).collect();
        state.active_positions = (0..state.parts.len()).collect();
        state.recalculate_type_counts();
        state
    }

    fn is_empty(&self) -> bool {
        self.non_empty_parts == 0
    }

    fn num_parts(&self) -> usize {
        self.non_empty_parts
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
            self.non_empty_parts -= 1;
        }
        self.decrement_vertex_counts(vertex);
        vertex
    }

    fn selected_part_index(&self) -> usize {
        let mut selected_index = None;
        let mut selected_len = usize::MAX;
        let mut selected_first = 0usize;

        for &index in &self.active_parts {
            let part = &self.parts[index];
            let part_len = part.len();
            let part_first = *part.first().expect("non-empty part must have a first vertex");
            if selected_index.is_none()
                || part_len < selected_len
                || (part_len == selected_len && part_first > selected_first)
            {
                selected_index = Some(index);
                selected_len = part_len;
                selected_first = part_first;
            }
        }

        selected_index.expect("non-empty partition set must have a selectable part")
    }

    fn pop_selected_vertex_in_place(&mut self, part_index: usize) -> usize {
        let (vertex, became_empty) = {
            let part = &mut self.parts[part_index];
            let vertex = part.pop().expect("selected partition is empty");
            (vertex, part.is_empty())
        };
        if became_empty {
            self.non_empty_parts -= 1;
            self.deactivate_part(part_index);
        }
        self.decrement_vertex_counts(vertex);
        vertex
    }

    fn restore_selected_vertex_in_place(&mut self, part_index: usize, vertex: usize) {
        let was_empty = self.parts[part_index].is_empty();
        if was_empty {
            self.non_empty_parts += 1;
            self.activate_part(part_index);
        }
        self.parts[part_index].push(vertex);
        increment_vertex_counts_raw(
            self.info,
            &mut self.g1_counts,
            &mut self.g2_counts,
            &mut self.g1_type_counts,
            &mut self.g2_type_counts,
            vertex,
        );
    }

    fn prune_vertices(&mut self, selected_vertex: usize) -> usize {
        let info = self.info;
        let adj = self.adj;
        let selected_neighbors = adj.row_bitslice(selected_vertex);
        let (selected_g1, selected_g2) = info.pairs[selected_vertex];
        let mut removed_count = 0usize;
        for part in &mut self.parts {
            let mut index = 0usize;
            while index < part.len() {
                let candidate = part[index];
                let (candidate_g1, candidate_g2) = info.pairs[candidate];
                if candidate_g1 == selected_g1
                    || candidate_g2 == selected_g2
                    || !selected_neighbors[candidate]
                {
                    part[index] = *part.last().expect("non-empty partition during prune");
                    part.pop();
                    decrement_vertex_counts_raw(
                        info,
                        &mut self.g1_counts,
                        &mut self.g2_counts,
                        &mut self.g1_type_counts,
                        &mut self.g2_type_counts,
                        candidate,
                    );
                    removed_count += 1;
                } else {
                    index += 1;
                }
            }
        }
        self.parts.retain(|part| !part.is_empty());
        self.non_empty_parts = self.parts.len();
        self.sort_partitions();
        removed_count
    }

    fn prune_vertices_in_place(
        &mut self,
        selected_vertex: usize,
        trail: &mut Vec<PruneUndo>,
    ) -> usize {
        let info = self.info;
        let adj = self.adj;
        let selected_neighbors = adj.row_bitslice(selected_vertex);
        let (selected_g1, selected_g2) = info.pairs[selected_vertex];
        let mut removed_count = 0usize;

        for part_index in 0..self.parts.len() {
            let mut part_emptied = false;
            {
                let part = &mut self.parts[part_index];
                let mut index = 0usize;
                while index < part.len() {
                    let candidate = part[index];
                    let (candidate_g1, candidate_g2) = info.pairs[candidate];
                    if candidate_g1 == selected_g1
                        || candidate_g2 == selected_g2
                        || !selected_neighbors[candidate]
                    {
                        let last_index = part.len() - 1;
                        let swapped_vertex = (index != last_index).then(|| part[last_index]);
                        part[index] = part[last_index];
                        part.pop();
                        part_emptied = part.is_empty();
                        decrement_vertex_counts_raw(
                            info,
                            &mut self.g1_counts,
                            &mut self.g2_counts,
                            &mut self.g1_type_counts,
                            &mut self.g2_type_counts,
                            candidate,
                        );
                        trail.push(PruneUndo {
                            part_index,
                            index,
                            removed_vertex: candidate,
                            swapped_vertex,
                        });
                        removed_count += 1;
                    } else {
                        index += 1;
                    }
                }
            }
            if part_emptied {
                self.non_empty_parts -= 1;
                self.deactivate_part(part_index);
            }
        }

        removed_count
    }

    fn restore_pruned_vertices_in_place(&mut self, trail: &mut Vec<PruneUndo>, checkpoint: usize) {
        while trail.len() > checkpoint {
            let undo = trail.pop().expect("trail checkpoint must be valid");
            let was_empty = self.parts[undo.part_index].is_empty();
            if was_empty {
                self.non_empty_parts += 1;
                self.activate_part(undo.part_index);
            }
            let part = &mut self.parts[undo.part_index];
            match undo.swapped_vertex {
                Some(swapped_vertex) => {
                    part.push(swapped_vertex);
                    part[undo.index] = undo.removed_vertex;
                }
                None => part.push(undo.removed_vertex),
            }
            increment_vertex_counts_raw(
                self.info,
                &mut self.g1_counts,
                &mut self.g2_counts,
                &mut self.g1_type_counts,
                &mut self.g2_type_counts,
                undo.removed_vertex,
            );
        }
    }

    fn sort_partitions(&mut self) {
        self.parts.sort_by(|left, right| {
            right.len().cmp(&left.len()).then_with(|| left.first().cmp(&right.first()))
        });
    }

    fn deactivate_part(&mut self, part_index: usize) {
        let position = self.active_positions[part_index];
        debug_assert_ne!(position, usize::MAX);
        self.active_parts.swap_remove(position);
        if position < self.active_parts.len() {
            let swapped = self.active_parts[position];
            self.active_positions[swapped] = position;
        }
        self.active_positions[part_index] = usize::MAX;
    }

    fn activate_part(&mut self, part_index: usize) {
        if self.active_positions[part_index] != usize::MAX {
            return;
        }
        self.active_positions[part_index] = self.active_parts.len();
        self.active_parts.push(part_index);
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
        decrement_vertex_counts_raw(
            self.info,
            &mut self.g1_counts,
            &mut self.g2_counts,
            &mut self.g1_type_counts,
            &mut self.g2_type_counts,
            vertex,
        );
    }
}

fn decrement_vertex_counts_raw(
    info: &PartitionInfo<'_>,
    g1_counts: &mut [usize],
    g2_counts: &mut [usize],
    g1_type_counts: &mut [usize],
    g2_type_counts: &mut [usize],
    vertex: usize,
) {
    let (g1, g2) = info.pairs[vertex];
    g1_counts[g1] -= 1;
    if g1_counts[g1] == 0 {
        g1_type_counts[info.g1_labels[g1]] -= 1;
    }
    g2_counts[g2] -= 1;
    if g2_counts[g2] == 0 {
        g2_type_counts[info.g2_labels[g2]] -= 1;
    }
}

fn increment_vertex_counts_raw(
    info: &PartitionInfo<'_>,
    g1_counts: &mut [usize],
    g2_counts: &mut [usize],
    g1_type_counts: &mut [usize],
    g2_type_counts: &mut [usize],
    vertex: usize,
) {
    let (g1, g2) = info.pairs[vertex];
    if g1_counts[g1] == 0 {
        g1_type_counts[info.g1_labels[g1]] += 1;
    }
    g1_counts[g1] += 1;
    if g2_counts[g2] == 0 {
        g2_type_counts[info.g2_labels[g2]] += 1;
    }
    g2_counts[g2] += 1;
}

#[derive(Clone, Copy, Debug)]
struct PruneUndo {
    part_index: usize,
    index: usize,
    removed_vertex: usize,
    swapped_vertex: Option<usize>,
}

/// Chooses the partition side using an RDKit-style smaller-first policy.
///
/// When the two line graphs have different numbers of bond vertices, the
/// smaller side is preferred. If they are equal in size, the first side is
/// kept to avoid introducing an extra bucket-profile heuristic that can drift
/// away from RDKit's default traversal.
#[must_use]
pub fn choose_partition_side(
    _pairs: &[(usize, usize)],
    g1_len: usize,
    g2_len: usize,
) -> PartitionSide {
    if g1_len < g2_len {
        return PartitionSide::First;
    }
    if g2_len < g1_len {
        return PartitionSide::Second;
    }
    PartitionSide::First
}

const DEFAULT_PARTIAL_ENUMERATION_CAP: usize = 10_000;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PartitionSearchPolicy {
    SingleAccepted,
    PartialEnumeration { cap: usize },
    AllBest,
}

impl PartitionSearchPolicy {
    fn prunes_ties(self) -> bool {
        !matches!(self, Self::AllBest)
    }
}

/// Performs the partition-driven maximum clique search.
pub(crate) fn search<F>(
    adj: &BitSquareMatrix,
    partition: &PartitionInfo<'_>,
    enumerate: bool,
    initial_lower_bound: usize,
    accept_clique: F,
) -> Vec<Vec<usize>>
where
    F: FnMut(&[usize]) -> bool,
{
    if enumerate {
        search_with_policy(
            adj,
            partition,
            PartitionSearchPolicy::AllBest,
            initial_lower_bound,
            accept_clique,
        )
    } else {
        partial_search(adj, partition, initial_lower_bound, accept_clique)
    }
}

#[doc(hidden)]
pub fn partial_search<F>(
    adj: &BitSquareMatrix,
    partition: &PartitionInfo<'_>,
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

    let mut state = PartitionSearchState::new(adj, partition, initial_lower_bound);
    let mut clique = Vec::new();
    let mut best_size = initial_lower_bound;
    let mut best_cliques = Vec::new();
    let mut trail = Vec::new();

    dfs_partial_in_place(
        &mut state,
        &mut clique,
        DEFAULT_PARTIAL_ENUMERATION_CAP,
        &mut best_size,
        &mut best_cliques,
        &mut accept_clique,
        &mut trail,
    );

    best_cliques
}

#[doc(hidden)]
pub fn all_best_search<F>(
    adj: &BitSquareMatrix,
    partition: &PartitionInfo<'_>,
    initial_lower_bound: usize,
    accept_clique: F,
) -> Vec<Vec<usize>>
where
    F: FnMut(&[usize]) -> bool,
{
    search_with_policy(
        adj,
        partition,
        PartitionSearchPolicy::AllBest,
        initial_lower_bound,
        accept_clique,
    )
}

#[doc(hidden)]
pub fn greedy_lower_bound<F>(
    adj: &BitSquareMatrix,
    partition: &PartitionInfo<'_>,
    initial_lower_bound: usize,
    mut accept_clique: F,
) -> usize
where
    F: FnMut(&[usize]) -> bool,
{
    if adj.order() == 0 {
        return initial_lower_bound;
    }

    let mut state = PartitionSearchState::new(adj, partition, initial_lower_bound);
    let mut clique = Vec::new();
    let mut best_size = initial_lower_bound;

    while !state.is_empty() {
        let parts_bound = clique.len() + state.num_parts();
        if parts_bound <= best_size {
            break;
        }

        let label_bound = clique.len() + state.upper_bound();
        if label_bound <= best_size {
            break;
        }

        let selected = state.pop_last_vertex();
        clique.push(selected);

        let mut candidate = clique.clone();
        candidate.sort_unstable();
        if candidate.len() > best_size && accept_clique(&candidate) {
            best_size = candidate.len();
        }

        state.prune_vertices(selected);
    }

    best_size
}

/// Diagnostic trace for a target-guided partial search.
#[doc(hidden)]
pub struct PartitionSearchTrace {
    /// Accepted best cliques returned by the search.
    pub best_cliques: Vec<Vec<usize>>,
    /// Human-readable trace lines for target-reachable branches.
    pub events: Vec<String>,
}

/// Aggregate counters collected during partitioned search.
#[doc(hidden)]
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PartitionSearchStats {
    /// Number of DFS calls entered.
    pub dfs_calls: usize,
    /// Number of DFS calls that returned because no partitions remained.
    pub empty_state_returns: usize,
    /// Maximum clique depth reached during search.
    pub max_depth: usize,
    /// Number of branches pruned by the partition-count upper bound.
    pub parts_bound_prunes: usize,
    /// Number of DFS states whose partition-count upper bound exactly matched
    /// the current best size.
    pub parts_bound_equal_best: usize,
    /// Number of branches pruned by the label-aware upper bound.
    pub label_bound_prunes: usize,
    /// Number of DFS states whose label-aware upper bound exactly matched the
    /// current best size.
    pub label_bound_equal_best: usize,
    /// Number of vertices selected via `pop_last_vertex`.
    pub vertices_popped: usize,
    /// Total number of vertices removed from candidate partitions during
    /// pruning.
    pub vertices_pruned: usize,
    /// Number of recursive "take vertex" branches explored.
    pub take_branches: usize,
    /// Number of recursive "skip vertex" branches explored.
    pub skip_branches: usize,
    /// Number of candidate cliques offered to `maybe_update_best`.
    pub maybe_update_calls: usize,
    /// Number of strict best-size improvements accepted.
    pub best_size_improvements: usize,
    /// DFS call index at which the final best size was first reached.
    pub last_best_improvement_dfs_call: usize,
    /// Number of accepted best cliques retained.
    pub retained_best_cliques: usize,
    /// Number of full `PartitionSearchState` clones performed.
    pub state_clones: usize,
    /// Total number of partitions copied across all state clones.
    pub cloned_partitions: usize,
    /// Total number of candidate vertices copied across all state clones.
    pub cloned_vertices: usize,
    /// Total number of label buckets scanned by the label-aware upper bound.
    pub upper_bound_labels_scanned: usize,
    /// Total number of partitions scanned while selecting the next part in the
    /// in-place partial path.
    pub selected_part_scans: usize,
    /// Total number of prune undo entries restored in the in-place partial
    /// path.
    pub restored_vertices: usize,
    /// Total number of candidate vertices inspected by in-place pruning in the
    /// partial path profile.
    pub prune_candidate_checks: usize,
}

/// Accepted best cliques plus aggregate search counters.
#[doc(hidden)]
pub struct PartitionSearchProfile {
    /// Accepted best cliques returned by the search.
    pub best_cliques: Vec<Vec<usize>>,
    /// Aggregate counters collected during search.
    pub stats: PartitionSearchStats,
}

/// Runs partial search while tracing only branches that could still reach the
/// given target clique.
#[doc(hidden)]
#[must_use]
pub fn trace_partial_search_to_target<F>(
    adj: &BitSquareMatrix,
    partition: &PartitionInfo<'_>,
    initial_lower_bound: usize,
    target_clique: &[usize],
    accept_clique: F,
) -> PartitionSearchTrace
where
    F: FnMut(&[usize]) -> bool,
{
    let mut target = target_clique.to_vec();
    target.sort_unstable();
    search_with_policy_trace(
        adj,
        partition,
        PartitionSearchPolicy::PartialEnumeration { cap: DEFAULT_PARTIAL_ENUMERATION_CAP },
        initial_lower_bound,
        target,
        accept_clique,
    )
}

/// Runs partitioned search while collecting aggregate counters.
#[doc(hidden)]
#[must_use]
pub fn profile_search<F>(
    adj: &BitSquareMatrix,
    partition: &PartitionInfo<'_>,
    enumerate: bool,
    initial_lower_bound: usize,
    accept_clique: F,
) -> PartitionSearchProfile
where
    F: FnMut(&[usize]) -> bool,
{
    if enumerate {
        search_with_policy_profile(
            adj,
            partition,
            PartitionSearchPolicy::AllBest,
            initial_lower_bound,
            initial_lower_bound,
            accept_clique,
        )
    } else {
        profile_partial_search(adj, partition, initial_lower_bound, accept_clique)
    }
}

#[doc(hidden)]
#[must_use]
pub fn profile_search_with_bounds<F>(
    adj: &BitSquareMatrix,
    partition: &PartitionInfo<'_>,
    enumerate: bool,
    state_lower_bound: usize,
    best_size_seed: usize,
    accept_clique: F,
) -> PartitionSearchProfile
where
    F: FnMut(&[usize]) -> bool,
{
    if enumerate {
        search_with_policy_profile(
            adj,
            partition,
            PartitionSearchPolicy::AllBest,
            state_lower_bound,
            best_size_seed,
            accept_clique,
        )
    } else {
        profile_partial_search_with_bounds(
            adj,
            partition,
            state_lower_bound,
            best_size_seed,
            accept_clique,
        )
    }
}

fn search_with_policy<F>(
    adj: &BitSquareMatrix,
    partition: &PartitionInfo<'_>,
    policy: PartitionSearchPolicy,
    initial_lower_bound: usize,
    accept_clique: F,
) -> Vec<Vec<usize>>
where
    F: FnMut(&[usize]) -> bool,
{
    search_with_policy_root_pruning(
        adj,
        partition,
        policy,
        initial_lower_bound,
        accept_clique,
        |_| false,
    )
}

fn search_with_policy_profile<F>(
    adj: &BitSquareMatrix,
    partition: &PartitionInfo<'_>,
    policy: PartitionSearchPolicy,
    state_lower_bound: usize,
    best_size_seed: usize,
    mut accept_clique: F,
) -> PartitionSearchProfile
where
    F: FnMut(&[usize]) -> bool,
{
    let n = adj.order();
    if n == 0 {
        let empty = Vec::new();
        let best_cliques = if accept_clique(&empty) { vec![empty] } else { Vec::new() };
        return PartitionSearchProfile { best_cliques, stats: PartitionSearchStats::default() };
    }

    let state = PartitionSearchState::new(adj, partition, state_lower_bound);
    let mut clique = Vec::new();
    let mut best_size = best_size_seed;
    let mut best_cliques = Vec::new();
    let mut stats = PartitionSearchStats::default();

    dfs_profile(
        state,
        &mut clique,
        policy,
        &mut best_size,
        &mut best_cliques,
        &mut accept_clique,
        &mut stats,
    );

    PartitionSearchProfile { best_cliques, stats }
}

fn profile_partial_search<F>(
    adj: &BitSquareMatrix,
    partition: &PartitionInfo<'_>,
    initial_lower_bound: usize,
    mut accept_clique: F,
) -> PartitionSearchProfile
where
    F: FnMut(&[usize]) -> bool,
{
    let n = adj.order();
    if n == 0 {
        let empty = Vec::new();
        let best_cliques = if accept_clique(&empty) { vec![empty] } else { Vec::new() };
        return PartitionSearchProfile { best_cliques, stats: PartitionSearchStats::default() };
    }

    let mut state = PartitionSearchState::new(adj, partition, initial_lower_bound);
    let mut clique = Vec::new();
    let mut best_size = initial_lower_bound;
    let mut best_cliques = Vec::new();
    let mut stats = PartitionSearchStats::default();
    let mut trail = Vec::new();

    dfs_partial_in_place_profile(
        &mut state,
        &mut clique,
        DEFAULT_PARTIAL_ENUMERATION_CAP,
        &mut best_size,
        &mut best_cliques,
        &mut accept_clique,
        &mut trail,
        &mut stats,
    );

    PartitionSearchProfile { best_cliques, stats }
}

fn profile_partial_search_with_bounds<F>(
    adj: &BitSquareMatrix,
    partition: &PartitionInfo<'_>,
    state_lower_bound: usize,
    best_size_seed: usize,
    mut accept_clique: F,
) -> PartitionSearchProfile
where
    F: FnMut(&[usize]) -> bool,
{
    let n = adj.order();
    if n == 0 {
        let empty = Vec::new();
        let best_cliques = if accept_clique(&empty) { vec![empty] } else { Vec::new() };
        return PartitionSearchProfile { best_cliques, stats: PartitionSearchStats::default() };
    }

    let mut state = PartitionSearchState::new(adj, partition, state_lower_bound);
    let mut clique = Vec::new();
    let mut best_size = best_size_seed;
    let mut best_cliques = Vec::new();
    let mut stats = PartitionSearchStats::default();
    let mut trail = Vec::new();

    dfs_partial_in_place_profile(
        &mut state,
        &mut clique,
        DEFAULT_PARTIAL_ENUMERATION_CAP,
        &mut best_size,
        &mut best_cliques,
        &mut accept_clique,
        &mut trail,
        &mut stats,
    );

    PartitionSearchProfile { best_cliques, stats }
}

fn prune_vertices_in_place_profile(
    state: &mut PartitionSearchState<'_>,
    selected_vertex: usize,
    trail: &mut Vec<PruneUndo>,
    stats: &mut PartitionSearchStats,
) -> usize {
    let info = state.info;
    let adj = state.adj;
    let selected_neighbors = adj.row_bitslice(selected_vertex);
    let (selected_g1, selected_g2) = info.pairs[selected_vertex];
    let mut removed_count = 0usize;

    for part_index in 0..state.parts.len() {
        let mut part_emptied = false;
        {
            let part = &mut state.parts[part_index];
            let mut index = 0usize;
            while index < part.len() {
                stats.prune_candidate_checks += 1;
                let candidate = part[index];
                let (candidate_g1, candidate_g2) = info.pairs[candidate];
                if candidate_g1 == selected_g1
                    || candidate_g2 == selected_g2
                    || !selected_neighbors[candidate]
                {
                    let last_index = part.len() - 1;
                    let swapped_vertex = (index != last_index).then(|| part[last_index]);
                    part[index] = part[last_index];
                    part.pop();
                    part_emptied = part.is_empty();
                    decrement_vertex_counts_raw(
                        info,
                        &mut state.g1_counts,
                        &mut state.g2_counts,
                        &mut state.g1_type_counts,
                        &mut state.g2_type_counts,
                        candidate,
                    );
                    trail.push(PruneUndo {
                        part_index,
                        index,
                        removed_vertex: candidate,
                        swapped_vertex,
                    });
                    removed_count += 1;
                } else {
                    index += 1;
                }
            }
        }
        if part_emptied {
            state.non_empty_parts -= 1;
            state.deactivate_part(part_index);
        }
    }

    removed_count
}

#[doc(hidden)]
pub fn partial_search_with_root_pruning<F, G>(
    adj: &BitSquareMatrix,
    partition: &PartitionInfo<'_>,
    initial_lower_bound: usize,
    accept_clique: F,
    skip_equivalent_root: G,
) -> Vec<Vec<usize>>
where
    F: FnMut(&[usize]) -> bool,
    G: FnMut(usize) -> bool,
{
    search_with_policy_root_pruning(
        adj,
        partition,
        PartitionSearchPolicy::PartialEnumeration { cap: DEFAULT_PARTIAL_ENUMERATION_CAP },
        initial_lower_bound,
        accept_clique,
        skip_equivalent_root,
    )
}

fn search_with_policy_root_pruning<F, G>(
    adj: &BitSquareMatrix,
    partition: &PartitionInfo<'_>,
    policy: PartitionSearchPolicy,
    initial_lower_bound: usize,
    mut accept_clique: F,
    mut skip_equivalent_root: G,
) -> Vec<Vec<usize>>
where
    F: FnMut(&[usize]) -> bool,
    G: FnMut(usize) -> bool,
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

    dfs(
        state,
        &mut clique,
        policy,
        &mut best_size,
        &mut best_cliques,
        &mut accept_clique,
        &mut skip_equivalent_root,
    );

    best_cliques
}

fn search_with_policy_trace<F>(
    adj: &BitSquareMatrix,
    partition: &PartitionInfo<'_>,
    policy: PartitionSearchPolicy,
    initial_lower_bound: usize,
    target_clique: Vec<usize>,
    mut accept_clique: F,
) -> PartitionSearchTrace
where
    F: FnMut(&[usize]) -> bool,
{
    let n = adj.order();
    if n == 0 {
        let empty = Vec::new();
        let best_cliques = if accept_clique(&empty) { vec![empty] } else { Vec::new() };
        return PartitionSearchTrace { best_cliques, events: Vec::new() };
    }

    let state = PartitionSearchState::new(adj, partition, initial_lower_bound);
    let mut clique = Vec::new();
    let mut best_size = initial_lower_bound;
    let mut best_cliques = Vec::new();
    let mut events = Vec::new();

    dfs_trace(
        state,
        &mut clique,
        policy,
        &mut best_size,
        &mut best_cliques,
        &mut accept_clique,
        &target_clique,
        &mut events,
    );

    PartitionSearchTrace { best_cliques, events }
}

fn trace_target_reachable(
    state: &PartitionSearchState<'_>,
    clique: &[usize],
    target_clique: &[usize],
) -> bool {
    clique.iter().all(|vertex| target_clique.binary_search(vertex).is_ok())
        && target_clique.iter().all(|target| {
            clique.contains(target)
                || state.parts.iter().any(|part| part.binary_search(target).is_ok())
        })
}

fn trace_target_suffix(clique: &[usize], target_clique: &[usize]) -> Vec<usize> {
    target_clique.iter().copied().filter(|target| !clique.contains(target)).collect()
}

fn dfs<F, G>(
    state: PartitionSearchState<'_>,
    clique: &mut Vec<usize>,
    policy: PartitionSearchPolicy,
    best_size: &mut usize,
    best_cliques: &mut Vec<Vec<usize>>,
    accept_clique: &mut F,
    skip_equivalent_root: &mut G,
) where
    F: FnMut(&[usize]) -> bool,
    G: FnMut(usize) -> bool,
{
    if state.is_empty() {
        return;
    }

    let parts_bound = clique.len() + state.num_parts();
    if policy.prunes_ties() {
        if parts_bound <= *best_size {
            return;
        }
    } else if parts_bound < *best_size {
        return;
    }

    let label_bound = clique.len() + state.upper_bound();
    if policy.prunes_ties() {
        if label_bound <= *best_size {
            return;
        }
    } else if label_bound < *best_size {
        return;
    }

    let mut without_vertex = state;
    let selected = without_vertex.pop_last_vertex();
    clique.push(selected);
    if clique.len() == 1 && skip_equivalent_root(selected) {
        clique.pop();
        if !without_vertex.is_empty() {
            dfs(
                without_vertex,
                clique,
                policy,
                best_size,
                best_cliques,
                accept_clique,
                skip_equivalent_root,
            );
        }
        return;
    }

    let mut with_vertex = without_vertex.clone();
    maybe_update_best(clique, policy, best_size, best_cliques, accept_clique);
    with_vertex.prune_vertices(selected);
    dfs(with_vertex, clique, policy, best_size, best_cliques, accept_clique, skip_equivalent_root);
    clique.pop();

    if !without_vertex.is_empty() {
        dfs(
            without_vertex,
            clique,
            policy,
            best_size,
            best_cliques,
            accept_clique,
            skip_equivalent_root,
        );
    }
}

fn dfs_partial_in_place<F>(
    state: &mut PartitionSearchState<'_>,
    clique: &mut Vec<usize>,
    cap: usize,
    best_size: &mut usize,
    best_cliques: &mut Vec<Vec<usize>>,
    accept_clique: &mut F,
    trail: &mut Vec<PruneUndo>,
) where
    F: FnMut(&[usize]) -> bool,
{
    if state.is_empty() {
        return;
    }

    let parts_bound = clique.len() + state.num_parts();
    if parts_bound <= *best_size {
        return;
    }

    let label_bound = clique.len() + state.upper_bound();
    if label_bound <= *best_size {
        return;
    }

    let selected_part = state.selected_part_index();
    let selected = state.pop_selected_vertex_in_place(selected_part);
    clique.push(selected);

    maybe_update_best(
        clique,
        PartitionSearchPolicy::PartialEnumeration { cap },
        best_size,
        best_cliques,
        accept_clique,
    );

    let checkpoint = trail.len();
    state.prune_vertices_in_place(selected, trail);
    dfs_partial_in_place(state, clique, cap, best_size, best_cliques, accept_clique, trail);
    state.restore_pruned_vertices_in_place(trail, checkpoint);

    clique.pop();
    if !state.is_empty() {
        dfs_partial_in_place(state, clique, cap, best_size, best_cliques, accept_clique, trail);
    }
    state.restore_selected_vertex_in_place(selected_part, selected);
}

fn dfs_partial_in_place_profile<F>(
    state: &mut PartitionSearchState<'_>,
    clique: &mut Vec<usize>,
    cap: usize,
    best_size: &mut usize,
    best_cliques: &mut Vec<Vec<usize>>,
    accept_clique: &mut F,
    trail: &mut Vec<PruneUndo>,
    stats: &mut PartitionSearchStats,
) where
    F: FnMut(&[usize]) -> bool,
{
    stats.dfs_calls += 1;
    stats.max_depth = stats.max_depth.max(clique.len());

    if state.is_empty() {
        stats.empty_state_returns += 1;
        return;
    }

    let parts_bound = clique.len() + state.num_parts();
    if parts_bound == *best_size {
        stats.parts_bound_equal_best += 1;
    }
    if parts_bound <= *best_size {
        stats.parts_bound_prunes += 1;
        return;
    }

    stats.upper_bound_labels_scanned += state.g1_type_counts.len();
    let label_bound = clique.len() + state.upper_bound();
    if label_bound == *best_size {
        stats.label_bound_equal_best += 1;
    }
    if label_bound <= *best_size {
        stats.label_bound_prunes += 1;
        return;
    }

    stats.selected_part_scans += state.parts.len();
    let selected_part = state.selected_part_index();
    let selected = state.pop_selected_vertex_in_place(selected_part);
    stats.vertices_popped += 1;
    clique.push(selected);

    stats.maybe_update_calls += 1;
    let before_best_size = *best_size;
    let before_len = best_cliques.len();
    maybe_update_best(
        clique,
        PartitionSearchPolicy::PartialEnumeration { cap },
        best_size,
        best_cliques,
        accept_clique,
    );
    if *best_size > before_best_size {
        stats.best_size_improvements += 1;
        stats.last_best_improvement_dfs_call = stats.dfs_calls;
        stats.retained_best_cliques += best_cliques.len();
    } else if best_cliques.len() > before_len {
        stats.retained_best_cliques += best_cliques.len() - before_len;
    }

    let checkpoint = trail.len();
    let removed = prune_vertices_in_place_profile(state, selected, trail, stats);
    stats.vertices_pruned += removed;
    stats.take_branches += 1;
    dfs_partial_in_place_profile(
        state,
        clique,
        cap,
        best_size,
        best_cliques,
        accept_clique,
        trail,
        stats,
    );
    stats.restored_vertices += trail.len() - checkpoint;
    state.restore_pruned_vertices_in_place(trail, checkpoint);

    clique.pop();
    if !state.is_empty() {
        stats.skip_branches += 1;
        dfs_partial_in_place_profile(
            state,
            clique,
            cap,
            best_size,
            best_cliques,
            accept_clique,
            trail,
            stats,
        );
    }
    state.restore_selected_vertex_in_place(selected_part, selected);
}

fn dfs_trace<F>(
    state: PartitionSearchState<'_>,
    clique: &mut Vec<usize>,
    policy: PartitionSearchPolicy,
    best_size: &mut usize,
    best_cliques: &mut Vec<Vec<usize>>,
    accept_clique: &mut F,
    target_clique: &[usize],
    events: &mut Vec<String>,
) where
    F: FnMut(&[usize]) -> bool,
{
    if state.is_empty() {
        return;
    }

    let reachable = trace_target_reachable(&state, clique, target_clique);
    let parts_bound = clique.len() + state.num_parts();
    let label_bound = clique.len() + state.upper_bound();

    if reachable {
        events.push(format!(
            "enter depth={} clique={:?} remaining_target={:?} best_size={} parts_bound={} label_bound={} parts={:?}",
            clique.len(),
            clique,
            trace_target_suffix(clique, target_clique),
            *best_size,
            parts_bound,
            label_bound,
            state.parts,
        ));
    }

    if policy.prunes_ties() {
        if parts_bound <= *best_size {
            if reachable {
                events.push(format!(
                    "prune depth={} reason=parts_bound clique={:?} remaining_target={:?} best_size={} parts_bound={}",
                    clique.len(),
                    clique,
                    trace_target_suffix(clique, target_clique),
                    *best_size,
                    parts_bound,
                ));
            }
            return;
        }
    } else if parts_bound < *best_size {
        if reachable {
            events.push(format!(
                "prune depth={} reason=parts_bound clique={:?} remaining_target={:?} best_size={} parts_bound={}",
                clique.len(),
                clique,
                trace_target_suffix(clique, target_clique),
                *best_size,
                parts_bound,
            ));
        }
        return;
    }

    if policy.prunes_ties() {
        if label_bound <= *best_size {
            if reachable {
                events.push(format!(
                    "prune depth={} reason=label_bound clique={:?} remaining_target={:?} best_size={} label_bound={}",
                    clique.len(),
                    clique,
                    trace_target_suffix(clique, target_clique),
                    *best_size,
                    label_bound,
                ));
            }
            return;
        }
    } else if label_bound < *best_size {
        if reachable {
            events.push(format!(
                "prune depth={} reason=label_bound clique={:?} remaining_target={:?} best_size={} label_bound={}",
                clique.len(),
                clique,
                trace_target_suffix(clique, target_clique),
                *best_size,
                label_bound,
            ));
        }
        return;
    }

    let mut without_vertex = state;
    let selected = without_vertex.pop_last_vertex();
    let mut with_vertex = without_vertex.clone();

    clique.push(selected);
    let before_best_size = *best_size;
    let before_len = best_cliques.len();
    maybe_update_best(clique, policy, best_size, best_cliques, accept_clique);
    let mut sorted_clique = clique.clone();
    sorted_clique.sort_unstable();
    if *best_size != before_best_size || best_cliques.len() != before_len {
        events.push(format!(
            "accept depth={} clique={:?} sorted={:?} best_size={} retained={}",
            clique.len(),
            clique,
            sorted_clique,
            *best_size,
            best_cliques.len(),
        ));
    }
    if sorted_clique == target_clique {
        events.push(format!(
            "target_found depth={} clique={:?} best_size={} retained={}",
            clique.len(),
            sorted_clique,
            *best_size,
            best_cliques.len(),
        ));
    }
    with_vertex.prune_vertices(selected);
    if trace_target_reachable(&with_vertex, clique, target_clique) {
        events.push(format!(
            "branch depth={} take={} clique={:?} remaining_target={:?}",
            clique.len(),
            selected,
            clique,
            trace_target_suffix(clique, target_clique),
        ));
    }
    dfs_trace(
        with_vertex,
        clique,
        policy,
        best_size,
        best_cliques,
        accept_clique,
        target_clique,
        events,
    );
    clique.pop();

    if !without_vertex.is_empty() {
        if trace_target_reachable(&without_vertex, clique, target_clique) {
            events.push(format!(
                "branch depth={} skip={} clique={:?} remaining_target={:?}",
                clique.len(),
                selected,
                clique,
                trace_target_suffix(clique, target_clique),
            ));
        }
        dfs_trace(
            without_vertex,
            clique,
            policy,
            best_size,
            best_cliques,
            accept_clique,
            target_clique,
            events,
        );
    }
}

fn dfs_profile<F>(
    state: PartitionSearchState<'_>,
    clique: &mut Vec<usize>,
    policy: PartitionSearchPolicy,
    best_size: &mut usize,
    best_cliques: &mut Vec<Vec<usize>>,
    accept_clique: &mut F,
    stats: &mut PartitionSearchStats,
) where
    F: FnMut(&[usize]) -> bool,
{
    stats.dfs_calls += 1;
    stats.max_depth = stats.max_depth.max(clique.len());

    if state.is_empty() {
        stats.empty_state_returns += 1;
        return;
    }

    let parts_bound = clique.len() + state.num_parts();
    if parts_bound == *best_size {
        stats.parts_bound_equal_best += 1;
    }
    if policy.prunes_ties() {
        if parts_bound <= *best_size {
            stats.parts_bound_prunes += 1;
            return;
        }
    } else if parts_bound < *best_size {
        stats.parts_bound_prunes += 1;
        return;
    }

    let label_bound = clique.len() + state.upper_bound();
    if label_bound == *best_size {
        stats.label_bound_equal_best += 1;
    }
    if policy.prunes_ties() {
        if label_bound <= *best_size {
            stats.label_bound_prunes += 1;
            return;
        }
    } else if label_bound < *best_size {
        stats.label_bound_prunes += 1;
        return;
    }

    let mut without_vertex = state;
    let selected = without_vertex.pop_last_vertex();
    stats.vertices_popped += 1;
    clique.push(selected);

    stats.state_clones += 1;
    stats.cloned_partitions += without_vertex.parts.len();
    stats.cloned_vertices += without_vertex.parts.iter().map(Vec::len).sum::<usize>();
    let mut with_vertex = without_vertex.clone();
    stats.maybe_update_calls += 1;
    let before_best_size = *best_size;
    let before_len = best_cliques.len();
    maybe_update_best(clique, policy, best_size, best_cliques, accept_clique);
    if *best_size > before_best_size {
        stats.best_size_improvements += 1;
        stats.last_best_improvement_dfs_call = stats.dfs_calls;
        stats.retained_best_cliques += best_cliques.len();
    } else if best_cliques.len() > before_len {
        stats.retained_best_cliques += best_cliques.len() - before_len;
    }

    let removed = with_vertex.prune_vertices(selected);
    stats.vertices_pruned += removed;
    stats.take_branches += 1;
    dfs_profile(with_vertex, clique, policy, best_size, best_cliques, accept_clique, stats);
    clique.pop();

    if !without_vertex.is_empty() {
        stats.skip_branches += 1;
        dfs_profile(without_vertex, clique, policy, best_size, best_cliques, accept_clique, stats);
    }
}

fn maybe_update_best<F>(
    clique: &[usize],
    policy: PartitionSearchPolicy,
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
    if size == *best_size
        && matches!(policy, PartitionSearchPolicy::SingleAccepted)
        && !best_cliques.is_empty()
    {
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

    if best_cliques.contains(&candidate) {
        return;
    }

    match policy {
        PartitionSearchPolicy::SingleAccepted => {
            if best_cliques.is_empty() || size > best_cliques[0].len() {
                best_cliques.push(candidate);
            }
        }
        PartitionSearchPolicy::PartialEnumeration { cap } => {
            if best_cliques.len() < cap {
                best_cliques.push(candidate);
            }
        }
        PartitionSearchPolicy::AllBest => {
            best_cliques.push(candidate);
        }
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
    fn test_partition_side_keeps_first_side_when_graph_sizes_tie() {
        let pairs = vec![(0, 0), (0, 1), (0, 2), (1, 0), (2, 1)];

        assert_eq!(choose_partition_side(&pairs, 3, 3), PartitionSide::First);
    }

    #[test]
    fn test_partition_side_prefers_smaller_graph_over_flatter_profile() {
        let pairs = vec![(0, 0), (1, 0), (2, 1), (3, 1), (4, 2), (5, 2), (6, 3), (7, 3)];

        assert_eq!(choose_partition_side(&pairs, 8, 4), PartitionSide::Second);
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

    #[test]
    fn test_partial_enumeration_retains_equal_size_accepted_cliques() {
        let mut best_size = 2;
        let mut best_cliques = vec![vec![0, 1]];

        maybe_update_best(
            &[2, 3],
            PartitionSearchPolicy::PartialEnumeration { cap: 10_000 },
            &mut best_size,
            &mut best_cliques,
            &mut |_| true,
        );

        assert_eq!(best_size, 2);
        assert_eq!(best_cliques, vec![vec![0, 1], vec![2, 3]]);
    }

    #[test]
    fn test_partial_enumeration_clears_smaller_retained_cliques_on_improvement() {
        let mut best_size = 2;
        let mut best_cliques = vec![vec![0, 1], vec![2, 3]];

        maybe_update_best(
            &[0, 1, 4],
            PartitionSearchPolicy::PartialEnumeration { cap: 10_000 },
            &mut best_size,
            &mut best_cliques,
            &mut |_| true,
        );

        assert_eq!(best_size, 3);
        assert_eq!(best_cliques, vec![vec![0, 1, 4]]);
    }

    #[test]
    fn test_partial_enumeration_deduplicates_and_honors_cap() {
        let mut best_size = 2;
        let mut best_cliques = vec![vec![0, 1]];

        maybe_update_best(
            &[1, 0],
            PartitionSearchPolicy::PartialEnumeration { cap: 2 },
            &mut best_size,
            &mut best_cliques,
            &mut |_| true,
        );
        maybe_update_best(
            &[2, 3],
            PartitionSearchPolicy::PartialEnumeration { cap: 2 },
            &mut best_size,
            &mut best_cliques,
            &mut |_| true,
        );
        maybe_update_best(
            &[4, 5],
            PartitionSearchPolicy::PartialEnumeration { cap: 2 },
            &mut best_size,
            &mut best_cliques,
            &mut |_| true,
        );

        assert_eq!(best_size, 2);
        assert_eq!(best_cliques, vec![vec![0, 1], vec![2, 3]]);
    }
}
