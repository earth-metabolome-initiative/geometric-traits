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
    non_empty_parts: usize,
    active_parts: Vec<usize>,
    active_positions: Vec<usize>,
    g1_counts: Vec<usize>,
    g2_counts: Vec<usize>,
    g1_type_counts: Vec<usize>,
    g2_type_counts: Vec<usize>,
}

#[inline]
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

    #[inline]
    fn is_empty(&self) -> bool {
        self.non_empty_parts == 0
    }

    #[inline]
    fn num_parts(&self) -> usize {
        self.non_empty_parts
    }

    #[inline]
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

    #[inline]
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

    #[inline]
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

#[derive(Clone)]
struct U32PartitionSearchState<'a> {
    adj: &'a BitSquareMatrix,
    info: &'a PartitionInfo<'a>,
    parts: Vec<Vec<u32>>,
    non_empty_parts: usize,
    active_parts: Vec<u32>,
    active_positions: Vec<u32>,
    g1_counts: Vec<usize>,
    g2_counts: Vec<usize>,
    g1_type_counts: Vec<usize>,
    g2_type_counts: Vec<usize>,
}

#[derive(Clone, Copy, Debug)]
struct U32PruneUndo {
    part_index: u32,
    index: u32,
    removed_vertex: u32,
    swapped_vertex: Option<u32>,
}

impl<'a> U32PartitionSearchState<'a> {
    fn new(adj: &'a BitSquareMatrix, info: &'a PartitionInfo<'a>, lower_bound: usize) -> Self {
        assert!(
            adj.order() <= u32::MAX as usize,
            "u32 partition state requires solver order <= u32::MAX"
        );
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
            parts_by_side[partition_index].push(to_u32_index(vertex));
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
        state.active_parts = (0..state.parts.len()).map(to_u32_index).collect();
        state.active_positions = (0..state.parts.len()).map(to_u32_index).collect();
        state.recalculate_type_counts();
        state
    }

    #[inline]
    fn is_empty(&self) -> bool {
        self.non_empty_parts == 0
    }

    #[inline]
    fn num_parts(&self) -> usize {
        self.non_empty_parts
    }

    #[inline]
    fn upper_bound(&self) -> usize {
        (0..self.g1_type_counts.len())
            .map(|label| self.g1_type_counts[label].min(self.g2_type_counts[label]))
            .sum()
    }

    fn selected_part_index(&self) -> usize {
        let mut selected_index = None;
        let mut selected_len = usize::MAX;
        let mut selected_first = 0u32;

        for &index_u32 in &self.active_parts {
            let index = to_usize_index(index_u32);
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
        let vertex = to_usize_index(vertex);
        decrement_vertex_counts_raw(
            self.info,
            &mut self.g1_counts,
            &mut self.g2_counts,
            &mut self.g1_type_counts,
            &mut self.g2_type_counts,
            vertex,
        );
        vertex
    }

    fn restore_selected_vertex_in_place(&mut self, part_index: usize, vertex: usize) {
        let was_empty = self.parts[part_index].is_empty();
        if was_empty {
            self.non_empty_parts += 1;
            self.activate_part(part_index);
        }
        self.parts[part_index].push(to_u32_index(vertex));
        increment_vertex_counts_raw(
            self.info,
            &mut self.g1_counts,
            &mut self.g2_counts,
            &mut self.g1_type_counts,
            &mut self.g2_type_counts,
            vertex,
        );
    }

    fn prune_vertices_in_place(
        &mut self,
        selected_vertex: usize,
        trail: &mut Vec<U32PruneUndo>,
    ) -> usize {
        let info = self.info;
        let selected_neighbors = self.adj.row_bitslice(selected_vertex);
        let (selected_g1, selected_g2) = info.pairs[selected_vertex];
        let mut removed_count = 0usize;

        for part_index in 0..self.parts.len() {
            let mut part_emptied = false;
            {
                let part = &mut self.parts[part_index];
                let mut index = 0usize;
                while index < part.len() {
                    let candidate_u32 = part[index];
                    let candidate = to_usize_index(candidate_u32);
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
                        trail.push(U32PruneUndo {
                            part_index: to_u32_index(part_index),
                            index: to_u32_index(index),
                            removed_vertex: candidate_u32,
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

    fn restore_pruned_vertices_in_place(
        &mut self,
        trail: &mut Vec<U32PruneUndo>,
        checkpoint: usize,
    ) {
        while trail.len() > checkpoint {
            let undo = trail.pop().expect("trail checkpoint must be valid");
            let part_index = to_usize_index(undo.part_index);
            let was_empty = self.parts[part_index].is_empty();
            if was_empty {
                self.non_empty_parts += 1;
                self.activate_part(part_index);
            }
            let part = &mut self.parts[part_index];
            match undo.swapped_vertex {
                Some(swapped_vertex) => {
                    part.push(swapped_vertex);
                    part[to_usize_index(undo.index)] = undo.removed_vertex;
                }
                None => part.push(undo.removed_vertex),
            }
            increment_vertex_counts_raw(
                self.info,
                &mut self.g1_counts,
                &mut self.g2_counts,
                &mut self.g1_type_counts,
                &mut self.g2_type_counts,
                to_usize_index(undo.removed_vertex),
            );
        }
    }

    #[inline]
    fn deactivate_part(&mut self, part_index: usize) {
        let position = self.active_positions[part_index];
        debug_assert_ne!(position, u32::MAX);
        let position = to_usize_index(position);
        self.active_parts.swap_remove(position);
        if position < self.active_parts.len() {
            let swapped = self.active_parts[position];
            self.active_positions[to_usize_index(swapped)] = to_u32_index(position);
        }
        self.active_positions[part_index] = u32::MAX;
    }

    #[inline]
    fn activate_part(&mut self, part_index: usize) {
        if self.active_positions[part_index] != u32::MAX {
            return;
        }
        self.active_positions[part_index] = to_u32_index(self.active_parts.len());
        self.active_parts.push(to_u32_index(part_index));
    }

    fn sort_partitions(&mut self) {
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
            for vertex_u32 in source_vertices {
                let vertex = to_usize_index(vertex_u32);
                for target in 0..=pivot {
                    if self.parts[target].iter().all(|&existing| {
                        !vertices_compatible(self.adj, self.info, vertex, to_usize_index(existing))
                    }) {
                        self.parts[target].push(vertex_u32);
                        moved.push(vertex_u32);
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
}

#[inline]
fn to_u32_index(index: usize) -> u32 {
    u32::try_from(index).expect("index must fit in u32")
}

#[inline]
fn to_usize_index(index: u32) -> usize {
    index as usize
}

#[inline]
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

#[inline]
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

/// Chooses the partition side using RDKit's initial molecule-swap rule.
///
/// RASCAL swaps the two molecules only when the first has strictly more
/// atoms than the second. The partition-driving side should mirror that same
/// choice: smaller atom count first, tie => keep the first input.
#[must_use]
pub fn choose_partition_side_by_atom_counts(
    first_vertices: usize,
    second_vertices: usize,
) -> PartitionSide {
    if first_vertices <= second_vertices { PartitionSide::First } else { PartitionSide::Second }
}

const DEFAULT_PARTIAL_ENUMERATION_CAP: usize = 10_000;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PartitionSearchPolicy {
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
        partial_search_u32(adj, partition, initial_lower_bound, accept_clique)
    }
}

#[doc(hidden)]
pub fn partial_search<F>(
    adj: &BitSquareMatrix,
    partition: &PartitionInfo<'_>,
    initial_lower_bound: usize,
    accept_clique: F,
) -> Vec<Vec<usize>>
where
    F: FnMut(&[usize]) -> bool,
{
    partial_search_with_bounds(
        adj,
        partition,
        initial_lower_bound,
        initial_lower_bound,
        accept_clique,
    )
}

#[doc(hidden)]
pub fn partial_search_with_bounds<F>(
    adj: &BitSquareMatrix,
    partition: &PartitionInfo<'_>,
    state_lower_bound: usize,
    best_size_seed: usize,
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

    let mut state = PartitionSearchState::new(adj, partition, state_lower_bound);
    let mut clique = Vec::new();
    let mut best_size = best_size_seed;
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
pub fn partial_search_u32<F>(
    adj: &BitSquareMatrix,
    partition: &PartitionInfo<'_>,
    initial_lower_bound: usize,
    accept_clique: F,
) -> Vec<Vec<usize>>
where
    F: FnMut(&[usize]) -> bool,
{
    partial_search_u32_with_bounds(
        adj,
        partition,
        initial_lower_bound,
        initial_lower_bound,
        accept_clique,
    )
}

#[doc(hidden)]
pub fn partial_search_u32_with_bounds<F>(
    adj: &BitSquareMatrix,
    partition: &PartitionInfo<'_>,
    state_lower_bound: usize,
    best_size_seed: usize,
    mut accept_clique: F,
) -> Vec<Vec<usize>>
where
    F: FnMut(&[usize]) -> bool,
{
    if adj.order() > u32::MAX as usize {
        return partial_search_with_bounds(
            adj,
            partition,
            state_lower_bound,
            best_size_seed,
            accept_clique,
        );
    }

    let n = adj.order();
    if n == 0 {
        let empty = Vec::new();
        return if accept_clique(&empty) { vec![empty] } else { Vec::new() };
    }

    let mut state = U32PartitionSearchState::new(adj, partition, state_lower_bound);
    let mut clique = Vec::new();
    let mut best_size = best_size_seed;
    let mut best_cliques = Vec::new();
    let mut trail = Vec::new();

    dfs_partial_u32_in_place(
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

fn search_with_policy<F>(
    adj: &BitSquareMatrix,
    partition: &PartitionInfo<'_>,
    policy: PartitionSearchPolicy,
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

    dfs(state, &mut clique, policy, &mut best_size, &mut best_cliques, &mut accept_clique);

    best_cliques
}

#[must_use]
pub(crate) fn partial_u32_best_size_with_budget<F>(
    adj: &BitSquareMatrix,
    partition: &PartitionInfo<'_>,
    state_lower_bound: usize,
    best_size_seed: usize,
    max_dfs_calls: usize,
    mut accept_clique: F,
) -> usize
where
    F: FnMut(&[usize]) -> bool,
{
    if adj.order() > u32::MAX as usize {
        let best_cliques = partial_search_with_bounds(
            adj,
            partition,
            state_lower_bound,
            best_size_seed,
            accept_clique,
        );
        return best_cliques
            .first()
            .map_or(best_size_seed, |clique| clique.len().max(best_size_seed));
    }

    let n = adj.order();
    if n == 0 {
        let empty = Vec::new();
        let best_size = if accept_clique(&empty) { 0 } else { best_size_seed };
        return best_size;
    }

    let mut state = U32PartitionSearchState::new(adj, partition, state_lower_bound);
    let mut clique = Vec::new();
    let mut best_size = best_size_seed;
    let mut best_cliques = Vec::new();
    let mut trail = Vec::new();
    let mut dfs_calls = 0usize;

    dfs_partial_u32_in_place_budgeted(
        &mut state,
        &mut clique,
        DEFAULT_PARTIAL_ENUMERATION_CAP,
        &mut best_size,
        &mut best_cliques,
        &mut accept_clique,
        &mut trail,
        &mut dfs_calls,
        max_dfs_calls,
    );

    best_size
}

fn dfs<F>(
    state: PartitionSearchState<'_>,
    clique: &mut Vec<usize>,
    policy: PartitionSearchPolicy,
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
    let mut with_vertex = without_vertex.clone();
    maybe_update_best(clique, policy, best_size, best_cliques, accept_clique);
    with_vertex.prune_vertices(selected);
    dfs(with_vertex, clique, policy, best_size, best_cliques, accept_clique);
    clique.pop();

    if !without_vertex.is_empty() {
        dfs(without_vertex, clique, policy, best_size, best_cliques, accept_clique);
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

fn dfs_partial_u32_in_place<F>(
    state: &mut U32PartitionSearchState<'_>,
    clique: &mut Vec<usize>,
    cap: usize,
    best_size: &mut usize,
    best_cliques: &mut Vec<Vec<usize>>,
    accept_clique: &mut F,
    trail: &mut Vec<U32PruneUndo>,
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
    dfs_partial_u32_in_place(state, clique, cap, best_size, best_cliques, accept_clique, trail);
    state.restore_pruned_vertices_in_place(trail, checkpoint);

    clique.pop();
    if !state.is_empty() {
        dfs_partial_u32_in_place(state, clique, cap, best_size, best_cliques, accept_clique, trail);
    }
    state.restore_selected_vertex_in_place(selected_part, selected);
}

fn dfs_partial_u32_in_place_budgeted<F>(
    state: &mut U32PartitionSearchState<'_>,
    clique: &mut Vec<usize>,
    cap: usize,
    best_size: &mut usize,
    best_cliques: &mut Vec<Vec<usize>>,
    accept_clique: &mut F,
    trail: &mut Vec<U32PruneUndo>,
    dfs_calls: &mut usize,
    max_dfs_calls: usize,
) where
    F: FnMut(&[usize]) -> bool,
{
    if *dfs_calls >= max_dfs_calls {
        return;
    }
    *dfs_calls += 1;

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
    dfs_partial_u32_in_place_budgeted(
        state,
        clique,
        cap,
        best_size,
        best_cliques,
        accept_clique,
        trail,
        dfs_calls,
        max_dfs_calls,
    );
    state.restore_pruned_vertices_in_place(trail, checkpoint);

    clique.pop();
    if !state.is_empty() {
        dfs_partial_u32_in_place_budgeted(
            state,
            clique,
            cap,
            best_size,
            best_cliques,
            accept_clique,
            trail,
            dfs_calls,
            max_dfs_calls,
        );
    }
    state.restore_selected_vertex_in_place(selected_part, selected);
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
    fn test_partition_side_keeps_first_side_when_atom_counts_tie() {
        assert_eq!(choose_partition_side_by_atom_counts(3, 3), PartitionSide::First);
    }

    #[test]
    fn test_partition_side_prefers_smaller_atom_count() {
        assert_eq!(choose_partition_side_by_atom_counts(8, 4), PartitionSide::Second);
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
