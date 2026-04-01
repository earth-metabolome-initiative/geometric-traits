use alloc::vec::Vec;

use num_traits::AsPrimitive;

#[cfg(test)]
use super::GenericPairQueues;
use super::{BlossomVState, EdgeQueueKeyStore, GenericQueueState, NONE, SchedulerCurrent};
use crate::traits::{Number, PositiveInteger, SparseValuedMatrix2D};

impl<M: SparseValuedMatrix2D + ?Sized> BlossomVState<M>
where
    M::Value: Number + AsPrimitive<i64>,
    M::RowIndex: PositiveInteger,
    M::ColumnIndex: PositiveInteger,
{
    pub(super) fn add_generic_tree_edge(&mut self, current_root: u32, other_root: u32) -> usize {
        let idx = self.scheduler_tree_edges.len();
        #[cfg(test)]
        {
            self.ensure_generic_tree_slot(current_root);
            self.ensure_generic_tree_slot(other_root);
            if self.test_state.generic_pairs.len() <= idx {
                self.test_state.generic_pairs.resize_with(idx + 1, GenericPairQueues::default);
            }
            self.test_state.generic_pairs[idx] = GenericPairQueues::new(current_root, other_root);
        }
        self.ensure_scheduler_tree_slot(current_root);
        self.ensure_scheduler_tree_slot(other_root);
        self.ensure_scheduler_tree_edge_slot(idx);
        self.scheduler_tree_edges[idx].head = [other_root, current_root];
        self.scheduler_tree_edges[idx].next[0] =
            self.scheduler_trees[current_root as usize].first[0];
        self.scheduler_tree_edges[idx].next[1] = self.scheduler_trees[other_root as usize].first[1];
        self.scheduler_trees[current_root as usize].first[0] = Some(idx);
        self.scheduler_trees[other_root as usize].first[1] = Some(idx);
        self.scheduler_trees[other_root as usize].current =
            SchedulerCurrent::Pair { pair_idx: idx, dir: 0 };
        #[cfg(test)]
        {
            self.sync_generic_root_topology_from_scheduler(current_root);
            self.sync_generic_root_topology_from_scheduler(other_root);
        }
        idx
    }

    #[inline]
    pub(super) fn add_generic_tree_edge_with_other_current_dir(
        &mut self,
        current_root: u32,
        other_root: u32,
    ) -> (usize, usize) {
        let pair_idx = self.add_generic_tree_edge(current_root, other_root);
        let dir = match self.scheduler_trees[other_root as usize].current {
            SchedulerCurrent::Pair { pair_idx: current_pair_idx, dir } => {
                debug_assert_eq!(
                    current_pair_idx, pair_idx,
                    "add_generic_tree_edge set unexpected pair current for root {other_root}",
                );
                dir
            }
            SchedulerCurrent::Root | SchedulerCurrent::None => {
                debug_assert!(
                    false,
                    "add_generic_tree_edge did not seed pair current for root {other_root}",
                );
                0
            }
        };
        (pair_idx, dir)
    }

    pub(super) fn ensure_generic_tree_edge(&mut self, current_root: u32, other_root: u32) -> usize {
        if let Some(pair_idx) = self.scheduler_tree_edge_index(current_root, other_root) {
            pair_idx
        } else {
            self.add_generic_tree_edge(current_root, other_root)
        }
    }

    pub(super) fn replace_generic_tree_root(&mut self, old_root: u32, new_root: u32) {
        if old_root == NONE || new_root == NONE || old_root == new_root {
            return;
        }
        #[cfg(test)]
        self.ensure_generic_tree_slot(new_root);
        self.ensure_scheduler_tree_slot(new_root);
        if (old_root as usize) < self.scheduler_trees.len() {
            for dir in 0..2usize {
                let old_first = self.scheduler_trees[old_root as usize].first[dir];
                if let Some(start) = old_first {
                    let new_first = self.scheduler_trees[new_root as usize].first[dir];
                    let mut cursor = Some(start);
                    let mut tail = start;
                    while let Some(pair_idx) = cursor {
                        if pair_idx < self.scheduler_tree_edges.len()
                            && self.scheduler_tree_edge_dir(pair_idx, old_root) == Some(dir)
                        {
                            self.scheduler_tree_edges[pair_idx].head[1 - dir] = new_root;
                            #[cfg(test)]
                            self.sync_generic_pair_head_from_scheduler(pair_idx);
                        }
                        tail = pair_idx;
                        cursor = self.scheduler_tree_edges[pair_idx].next[dir];
                    }
                    if tail < self.scheduler_tree_edges.len() {
                        self.scheduler_tree_edges[tail].next[dir] = new_first;
                    }
                    self.scheduler_trees[new_root as usize].first[dir] = Some(start);
                    self.scheduler_trees[old_root as usize].first[dir] = None;
                }
            }
            self.scheduler_trees[old_root as usize].current = SchedulerCurrent::None;
        }
        #[cfg(test)]
        {
            self.sync_generic_root_topology_from_scheduler(old_root);
            self.sync_generic_root_topology_from_scheduler(new_root);
        }
    }

    pub(super) fn vec_push_edge(edges: &mut Vec<u32>, slots: &mut [usize], e_idx: u32) {
        let e_usize = e_idx as usize;
        debug_assert_eq!(slots[e_usize], usize::MAX);
        slots[e_usize] = edges.len();
        edges.push(e_idx);
    }

    pub(super) fn vec_remove_edge(edges: &mut Vec<u32>, slots: &mut [usize], e_idx: u32) {
        let e_usize = e_idx as usize;
        let mut pos = slots[e_usize];
        if pos >= edges.len() || edges[pos] != e_idx {
            let Some(found) = edges.iter().position(|&e| e == e_idx) else {
                slots[e_usize] = usize::MAX;
                return;
            };
            pos = found;
        }

        let removed = edges.swap_remove(pos);
        debug_assert_eq!(removed, e_idx);
        if pos < edges.len() {
            let moved = edges[pos];
            slots[moved as usize] = pos;
        }
        slots[e_usize] = usize::MAX;
    }

    #[inline]
    pub(super) fn clear_generic_queue_state(&mut self, e_idx: u32) {
        self.remove_edge_from_generic_queue(e_idx);
    }

    #[inline]
    pub(super) fn set_generic_pq0_root_slot(
        &mut self,
        e_idx: u32,
        root: u32,
        preserve_stamp: bool,
    ) {
        if (e_idx as usize) >= self.edge_num || root == NONE {
            return;
        }
        self.ensure_scheduler_tree_slot(root);
        let old_stamp = self.edge_queue_stamp(e_idx);
        self.remove_edge_from_generic_queue(e_idx);
        let new_stamp = if preserve_stamp {
            old_stamp
        } else {
            self.generic_queue_epoch = self.generic_queue_epoch.wrapping_add(1);
            self.generic_queue_epoch
        };
        self.set_edge_queue_stamp(e_idx, new_stamp);
        Self::vec_push_edge(
            &mut self.scheduler_trees[root as usize].pq0,
            self.edge_queue_slot.as_mut_slice(),
            e_idx,
        );
        {
            let mut keys =
                EdgeQueueKeyStore::new(self.edges.as_mut_slice(), self.edge_queue_stamp.as_slice());
            self.scheduler_trees[root as usize].pq0_heap.add(
                e_idx,
                &mut keys,
                self.pq_nodes.as_mut_slice(),
            );
        }
        self.set_edge_queue_owner(e_idx, GenericQueueState::Pq0 { root });
        #[cfg(test)]
        self.sync_generic_root_queues_from_scheduler(root);
    }

    #[inline]
    pub(super) fn set_generic_pq0(&mut self, e_idx: u32, root: u32) {
        self.set_generic_pq0_root_slot(e_idx, root, false);
    }

    #[inline]
    pub(super) fn set_generic_pq_blossoms_root_slot(
        &mut self,
        e_idx: u32,
        root: u32,
        preserve_stamp: bool,
    ) {
        if (e_idx as usize) >= self.edge_num || root == NONE {
            return;
        }
        self.ensure_scheduler_tree_slot(root);
        let old_stamp = self.edge_queue_stamp(e_idx);
        self.remove_edge_from_generic_queue(e_idx);
        let new_stamp = if preserve_stamp {
            old_stamp
        } else {
            self.generic_queue_epoch = self.generic_queue_epoch.wrapping_add(1);
            self.generic_queue_epoch
        };
        self.set_edge_queue_stamp(e_idx, new_stamp);
        Self::vec_push_edge(
            &mut self.scheduler_trees[root as usize].pq_blossoms,
            self.edge_queue_slot.as_mut_slice(),
            e_idx,
        );
        {
            let mut keys =
                EdgeQueueKeyStore::new(self.edges.as_mut_slice(), self.edge_queue_stamp.as_slice());
            self.scheduler_trees[root as usize].pq_blossoms_heap.add(
                e_idx,
                &mut keys,
                self.pq_nodes.as_mut_slice(),
            );
        }
        self.set_edge_queue_owner(e_idx, GenericQueueState::PqBlossoms { root });
        #[cfg(test)]
        self.sync_generic_root_queues_from_scheduler(root);
    }

    #[inline]
    pub(super) fn set_generic_pq00_local_slot(
        &mut self,
        e_idx: u32,
        root: u32,
        preserve_stamp: bool,
    ) {
        if (e_idx as usize) >= self.edge_num || root == NONE {
            return;
        }
        self.ensure_scheduler_tree_slot(root);
        let old_stamp = self.edge_queue_stamp(e_idx);
        self.remove_edge_from_generic_queue(e_idx);
        let new_stamp = if preserve_stamp {
            old_stamp
        } else {
            self.generic_queue_epoch = self.generic_queue_epoch.wrapping_add(1);
            self.generic_queue_epoch
        };
        self.set_edge_queue_stamp(e_idx, new_stamp);
        Self::vec_push_edge(
            &mut self.scheduler_trees[root as usize].pq00_local,
            self.edge_queue_slot.as_mut_slice(),
            e_idx,
        );
        {
            let mut keys =
                EdgeQueueKeyStore::new(self.edges.as_mut_slice(), self.edge_queue_stamp.as_slice());
            self.scheduler_trees[root as usize].pq00_local_heap.add(
                e_idx,
                &mut keys,
                self.pq_nodes.as_mut_slice(),
            );
        }
        self.set_edge_queue_owner(e_idx, GenericQueueState::Pq00Local { root });
        #[cfg(test)]
        self.sync_generic_root_queues_from_scheduler(root);
    }

    #[inline]
    pub(super) fn set_generic_pq00(&mut self, e_idx: u32, left_root: u32, right_root: u32) {
        if (e_idx as usize) >= self.edge_num || left_root == NONE || right_root == NONE {
            return;
        }
        self.ensure_scheduler_tree_slot(left_root);
        self.ensure_scheduler_tree_slot(right_root);
        if left_root == right_root {
            self.set_generic_pq00_local_slot(e_idx, left_root, false);
        } else {
            self.remove_edge_from_generic_queue(e_idx);
            let pair_idx = self.ensure_generic_tree_edge(left_root, right_root);
            self.ensure_scheduler_tree_edge_slot(pair_idx);
            self.generic_queue_epoch = self.generic_queue_epoch.wrapping_add(1);
            self.set_edge_queue_stamp(e_idx, self.generic_queue_epoch);
            Self::vec_push_edge(
                &mut self.scheduler_tree_edges[pair_idx].pq00,
                self.edge_queue_slot.as_mut_slice(),
                e_idx,
            );
            {
                let mut keys = EdgeQueueKeyStore::new(
                    self.edges.as_mut_slice(),
                    self.edge_queue_stamp.as_slice(),
                );
                self.scheduler_tree_edges[pair_idx].pq00_heap.add(
                    e_idx,
                    &mut keys,
                    self.pq_nodes.as_mut_slice(),
                );
            }
            self.set_edge_queue_owner(e_idx, GenericQueueState::Pq00Pair { pair_idx });
            #[cfg(test)]
            self.sync_generic_pair_queues_from_scheduler(pair_idx);
        }
    }

    #[inline]
    pub(super) fn set_generic_pq01(&mut self, e_idx: u32, current_root: u32, other_root: u32) {
        if (e_idx as usize) >= self.edge_num
            || current_root == NONE
            || other_root == NONE
            || current_root == other_root
        {
            return;
        }
        self.ensure_scheduler_tree_slot(current_root);
        self.ensure_scheduler_tree_slot(other_root);
        self.remove_edge_from_generic_queue(e_idx);
        let pair_idx = self.ensure_generic_tree_edge(current_root, other_root);
        let dir = self.scheduler_tree_edge_dir(pair_idx, current_root).unwrap_or(0);
        self.set_generic_pq01_pair_slot(e_idx, pair_idx, dir, false);
    }

    #[inline]
    pub(super) fn set_generic_pq01_other_side(
        &mut self,
        e_idx: u32,
        current_root: u32,
        other_root: u32,
    ) {
        if (e_idx as usize) >= self.edge_num
            || current_root == NONE
            || other_root == NONE
            || current_root == other_root
        {
            return;
        }
        self.ensure_scheduler_tree_slot(current_root);
        self.ensure_scheduler_tree_slot(other_root);
        self.remove_edge_from_generic_queue(e_idx);
        let (pair_idx, dir) = self
            .scheduler_pair_dir_from_active_root(current_root, other_root)
            .or_else(|| self.scheduler_current_pair_dir(other_root))
            .unwrap_or_else(|| {
                self.add_generic_tree_edge_with_other_current_dir(current_root, other_root)
            });
        self.set_generic_pq01_pair_slot(e_idx, pair_idx, dir, false);
    }

    #[inline]
    pub(super) fn set_generic_pq01_pair_slot(
        &mut self,
        e_idx: u32,
        pair_idx: usize,
        dir: usize,
        preserve_stamp: bool,
    ) {
        if (e_idx as usize) >= self.edge_num
            || pair_idx >= self.scheduler_tree_edges.len()
            || dir > 1
        {
            return;
        }
        self.ensure_scheduler_tree_edge_slot(pair_idx);
        let old_stamp = self.edge_queue_stamp(e_idx);
        self.remove_edge_from_generic_queue(e_idx);
        let new_stamp = if preserve_stamp {
            old_stamp
        } else {
            self.generic_queue_epoch = self.generic_queue_epoch.wrapping_add(1);
            self.generic_queue_epoch
        };
        self.set_edge_queue_stamp(e_idx, new_stamp);
        Self::vec_push_edge(
            &mut self.scheduler_tree_edges[pair_idx].pq01[dir],
            self.edge_queue_slot.as_mut_slice(),
            e_idx,
        );
        {
            let mut keys =
                EdgeQueueKeyStore::new(self.edges.as_mut_slice(), self.edge_queue_stamp.as_slice());
            self.scheduler_tree_edges[pair_idx].pq01_heap[dir].add(
                e_idx,
                &mut keys,
                self.pq_nodes.as_mut_slice(),
            );
        }
        self.set_edge_queue_owner(e_idx, GenericQueueState::Pq01Pair { pair_idx, dir });
        #[cfg(test)]
        self.sync_generic_pair_queues_from_scheduler(pair_idx);
    }

    #[allow(clippy::too_many_lines)]
    pub(super) fn remove_edge_from_generic_queue(&mut self, e_idx: u32) {
        if (e_idx as usize) >= self.edge_num {
            return;
        }
        let state = self.edge_queue_owner(e_idx);
        match state {
            GenericQueueState::None => {}
            GenericQueueState::Pq0 { root } => {
                if (root as usize) < self.scheduler_trees.len() {
                    if self.pq_nodes[e_idx as usize].is_in_heap() {
                        let mut keys = EdgeQueueKeyStore::new(
                            self.edges.as_mut_slice(),
                            self.edge_queue_stamp.as_slice(),
                        );
                        self.scheduler_trees[root as usize].pq0_heap.remove(
                            e_idx,
                            &mut keys,
                            self.pq_nodes.as_mut_slice(),
                        );
                    }
                    Self::vec_remove_edge(
                        &mut self.scheduler_trees[root as usize].pq0,
                        self.edge_queue_slot.as_mut_slice(),
                        e_idx,
                    );
                    #[cfg(test)]
                    self.sync_generic_root_queues_from_scheduler(root);
                }
            }
            GenericQueueState::Pq00Local { root } => {
                if (root as usize) < self.scheduler_trees.len() {
                    if self.pq_nodes[e_idx as usize].is_in_heap() {
                        let mut keys = EdgeQueueKeyStore::new(
                            self.edges.as_mut_slice(),
                            self.edge_queue_stamp.as_slice(),
                        );
                        self.scheduler_trees[root as usize].pq00_local_heap.remove(
                            e_idx,
                            &mut keys,
                            self.pq_nodes.as_mut_slice(),
                        );
                    }
                    Self::vec_remove_edge(
                        &mut self.scheduler_trees[root as usize].pq00_local,
                        self.edge_queue_slot.as_mut_slice(),
                        e_idx,
                    );
                    #[cfg(test)]
                    self.sync_generic_root_queues_from_scheduler(root);
                }
            }
            GenericQueueState::Pq00Pair { pair_idx } => {
                if pair_idx < self.scheduler_tree_edges.len() {
                    if self.pq_nodes[e_idx as usize].is_in_heap() {
                        let mut keys = EdgeQueueKeyStore::new(
                            self.edges.as_mut_slice(),
                            self.edge_queue_stamp.as_slice(),
                        );
                        self.scheduler_tree_edges[pair_idx].pq00_heap.remove(
                            e_idx,
                            &mut keys,
                            self.pq_nodes.as_mut_slice(),
                        );
                    }
                    Self::vec_remove_edge(
                        &mut self.scheduler_tree_edges[pair_idx].pq00,
                        self.edge_queue_slot.as_mut_slice(),
                        e_idx,
                    );
                    #[cfg(test)]
                    self.sync_generic_pair_queues_from_scheduler(pair_idx);
                }
            }
            GenericQueueState::Pq01Pair { pair_idx, dir } => {
                if pair_idx < self.scheduler_tree_edges.len() {
                    if self.pq_nodes[e_idx as usize].is_in_heap() {
                        let mut keys = EdgeQueueKeyStore::new(
                            self.edges.as_mut_slice(),
                            self.edge_queue_stamp.as_slice(),
                        );
                        self.scheduler_tree_edges[pair_idx].pq01_heap[dir].remove(
                            e_idx,
                            &mut keys,
                            self.pq_nodes.as_mut_slice(),
                        );
                    }
                    Self::vec_remove_edge(
                        &mut self.scheduler_tree_edges[pair_idx].pq01[dir],
                        self.edge_queue_slot.as_mut_slice(),
                        e_idx,
                    );
                    #[cfg(test)]
                    self.sync_generic_pair_queues_from_scheduler(pair_idx);
                }
            }
            GenericQueueState::PqBlossoms { root } => {
                if (root as usize) < self.scheduler_trees.len() {
                    if self.pq_nodes[e_idx as usize].is_in_heap() {
                        let mut keys = EdgeQueueKeyStore::new(
                            self.edges.as_mut_slice(),
                            self.edge_queue_stamp.as_slice(),
                        );
                        self.scheduler_trees[root as usize].pq_blossoms_heap.remove(
                            e_idx,
                            &mut keys,
                            self.pq_nodes.as_mut_slice(),
                        );
                    }
                    Self::vec_remove_edge(
                        &mut self.scheduler_trees[root as usize].pq_blossoms,
                        self.edge_queue_slot.as_mut_slice(),
                        e_idx,
                    );
                    #[cfg(test)]
                    self.sync_generic_root_queues_from_scheduler(root);
                }
            }
        }
        self.set_edge_queue_owner(e_idx, GenericQueueState::None);
        self.set_edge_queue_slot(e_idx, usize::MAX);
        self.set_edge_queue_stamp(e_idx, 0);
    }

    pub(super) fn replace_generic_queue_root(&mut self, old_root: u32, new_root: u32) {
        if old_root == NONE || new_root == NONE || old_root == new_root {
            return;
        }
        self.replace_generic_tree_root(old_root, new_root);
        if (old_root as usize) >= self.scheduler_trees.len() {
            return;
        }

        let affected_pq0 = core::mem::take(&mut self.scheduler_trees[old_root as usize].pq0);
        for e_idx in affected_pq0 {
            self.set_generic_pq0_root_slot(e_idx, new_root, true);
        }

        let affected_pq00_local =
            core::mem::take(&mut self.scheduler_trees[old_root as usize].pq00_local);
        for e_idx in affected_pq00_local {
            self.set_generic_pq00_local_slot(e_idx, new_root, true);
        }

        let affected_pq_blossoms =
            core::mem::take(&mut self.scheduler_trees[old_root as usize].pq_blossoms);
        for e_idx in affected_pq_blossoms {
            self.set_generic_pq_blossoms_root_slot(e_idx, new_root, true);
        }
    }

    pub(super) fn detach_generic_root_after_augment(&mut self, root: u32) {
        if root == NONE || (root as usize) >= self.scheduler_trees.len() {
            return;
        }

        let remaining_pq0 = core::mem::take(&mut self.scheduler_trees[root as usize].pq0);
        for e_idx in remaining_pq0 {
            self.remove_edge_from_generic_queue(e_idx);
        }

        let remaining_pq00_local =
            core::mem::take(&mut self.scheduler_trees[root as usize].pq00_local);
        for e_idx in remaining_pq00_local {
            self.remove_edge_from_generic_queue(e_idx);
        }

        let remaining_pq_blossoms =
            core::mem::take(&mut self.scheduler_trees[root as usize].pq_blossoms);
        for e_idx in remaining_pq_blossoms {
            self.remove_edge_from_generic_queue(e_idx);
        }

        self.detach_scheduler_root_topology(root);
        #[cfg(test)]
        self.sync_generic_root_topology_from_scheduler(root);
    }
}
