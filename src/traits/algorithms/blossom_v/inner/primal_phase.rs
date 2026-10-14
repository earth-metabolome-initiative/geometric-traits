use alloc::vec::Vec;

use num_traits::AsPrimitive;

use super::{
    BlossomVState, FREE, GenericQueueState, MINUS, NONE, PLUS, SchedulerCurrent, arc_dir, arc_edge,
    arc_rev, make_arc,
};
#[cfg(test)]
use super::{GenericPrimalEvent, GenericPrimalStepTrace, normalized_edge_pair};
use crate::traits::{Number, PositiveInteger, SparseValuedMatrix2D};

impl<M: SparseValuedMatrix2D + ?Sized> BlossomVState<M>
where
    M::Value: Number + AsPrimitive<i64>,
    M::RowIndex: PositiveInteger,
    M::ColumnIndex: PositiveInteger,
{
    #[cfg(test)]
    pub(super) fn apply_generic_grow(&mut self, e_idx: u32, plus: u32, free: u32) {
        let event = GenericPrimalEvent::Grow {
            edge: normalized_edge_pair(self.edges[e_idx as usize].head0),
            plus,
            free,
        };
        let before = self.test_strict_parity_snapshot();
        self.grow(e_idx, plus, free);
        if let Some((augment_edge, left, right)) = self.grow_tree_after_absorb(plus, free) {
            self.augment(augment_edge, left, right);
        }
        let after = self.test_strict_parity_snapshot();
        self.test_state.generic_primal_steps.push(GenericPrimalStepTrace { event, before, after });
    }

    #[cfg(not(test))]
    pub(super) fn apply_generic_grow(&mut self, e_idx: u32, plus: u32, free: u32) {
        self.grow(e_idx, plus, free);
        if let Some((augment_edge, left, right)) = self.grow_tree_after_absorb(plus, free) {
            self.augment(augment_edge, left, right);
        } else {
            self.maybe_write_debug_trace_snapshot("GROW_AFTER");
        }
    }

    #[cfg(test)]
    pub(super) fn apply_generic_shrink(&mut self, e_idx: u32, left: u32, right: u32) {
        let event = GenericPrimalEvent::Shrink {
            edge: normalized_edge_pair(self.edges[e_idx as usize].head0),
            left,
            right,
        };
        let before = self.test_strict_parity_snapshot();
        self.shrink(e_idx, left, right);
        let after = self.test_strict_parity_snapshot();
        self.test_state.generic_primal_steps.push(GenericPrimalStepTrace { event, before, after });
    }

    #[cfg(not(test))]
    pub(super) fn apply_generic_shrink(&mut self, e_idx: u32, left: u32, right: u32) {
        self.shrink(e_idx, left, right);
        self.maybe_write_debug_trace_snapshot("SHRINK_AFTER");
    }

    #[cfg(test)]
    pub(super) fn apply_generic_augment(&mut self, e_idx: u32, left: u32, right: u32) {
        let event = GenericPrimalEvent::Augment {
            edge: normalized_edge_pair(self.edges[e_idx as usize].head0),
            left,
            right,
        };
        let before = self.test_strict_parity_snapshot();
        self.augment(e_idx, left, right);
        let after = self.test_strict_parity_snapshot();
        self.test_state.generic_primal_steps.push(GenericPrimalStepTrace { event, before, after });
    }

    #[cfg(not(test))]
    pub(super) fn apply_generic_augment(&mut self, e_idx: u32, left: u32, right: u32) {
        self.augment(e_idx, left, right);
        self.maybe_write_debug_trace_snapshot("AUGMENT_AFTER");
        self.maybe_write_debug_queue_summary("after AUGMENT_AFTER");
    }

    pub(super) fn perform_generic_expand(&mut self, b: u32) {
        let match_arc = self.nodes[b as usize].match_arc;
        if match_arc != NONE && (arc_edge(match_arc) as usize) < self.edge_num {
            let match_edge = arc_edge(match_arc) as usize;
            core::mem::swap(&mut self.edges[match_edge].slack, &mut self.nodes[b as usize].y);
        }
        self.expand(b);
    }

    #[cfg(test)]
    pub(super) fn apply_generic_expand(&mut self, b: u32) {
        let event = GenericPrimalEvent::Expand { blossom: b };
        let before = self.test_strict_parity_snapshot();
        self.perform_generic_expand(b);
        let after = self.test_strict_parity_snapshot();
        self.test_state.generic_primal_steps.push(GenericPrimalStepTrace { event, before, after });
    }

    #[cfg(not(test))]
    pub(super) fn apply_generic_expand(&mut self, b: u32) {
        self.perform_generic_expand(b);
        self.maybe_write_debug_trace_snapshot("EXPAND_AFTER");
    }

    #[allow(clippy::manual_swap, clippy::too_many_lines)]
    pub(super) fn grow_node(&mut self, plus_node: u32) -> Option<(u32, u32, u32)> {
        let root = self.find_tree_root(plus_node);
        if root == NONE {
            return None;
        }
        let eps = self.tree_eps(root);
        let mut augment_edge = None;
        let mut incident = self.take_incident_scratch();
        self.collect_incident_edges_into(plus_node, &mut incident);

        for &(e_idx, dir) in &incident {
            let other = self.edge_head_outer(e_idx, dir);
            if other == NONE || other == plus_node || !self.nodes[other as usize].is_outer {
                continue;
            }

            if self.nodes[other as usize].flag == FREE {
                self.edges[e_idx as usize].slack += eps;
                if self.edges[e_idx as usize].slack <= 0 {
                    self.clear_generic_queue_state(e_idx);
                    self.grow(e_idx, plus_node, other);
                    self.nodes[other as usize].y += eps;
                    let new_plus = self.arc_head_raw(self.nodes[other as usize].match_arc);
                    if new_plus != NONE {
                        self.nodes[new_plus as usize].y -= eps;
                    }
                } else {
                    self.set_generic_pq0(e_idx, root);
                }
                continue;
            }

            let other_root = self.find_tree_root(other);
            if self.nodes[other as usize].flag == PLUS
                && self.nodes[other as usize].is_processed
                && other_root != NONE
                && other_root != root
                && self.edges[e_idx as usize].slack <= self.tree_eps(other_root)
            {
                augment_edge = Some((e_idx, plus_node, other));
            }

            self.edges[e_idx as usize].slack += eps;
            if self.nodes[other as usize].flag == PLUS
                && self.nodes[other as usize].is_processed
                && other_root != NONE
            {
                self.set_generic_pq00(e_idx, root, other_root);
            } else if self.nodes[other as usize].flag == MINUS
                && other_root != NONE
                && other_root != root
            {
                let (pair_idx, dir) = self
                    .scheduler_pair_dir_from_active_root(root, other_root)
                    .or_else(|| self.scheduler_current_pair_dir(other_root))
                    .unwrap_or_else(|| {
                        self.add_generic_tree_edge_with_other_current_dir(root, other_root)
                    });
                self.set_generic_pq01_pair_slot(e_idx, pair_idx, dir, false);
            }
        }

        self.nodes[plus_node as usize].is_processed = true;

        if !self.nodes[plus_node as usize].is_tree_root {
            let minus = self.arc_head_raw(self.nodes[plus_node as usize].match_arc);
            if minus != NONE {
                self.nodes[minus as usize].is_processed = true;
                if self.nodes[minus as usize].is_blossom {
                    let match_edge = arc_edge(self.nodes[plus_node as usize].match_arc);
                    let tmp = self.edges[match_edge as usize].slack;
                    self.edges[match_edge as usize].slack = self.nodes[minus as usize].y;
                    self.nodes[minus as usize].y = tmp;
                    self.queue_processed_plus_blossom_match_edge(plus_node);
                }
            }
        }

        self.restore_incident_scratch(incident);
        augment_edge
    }

    pub(super) fn queue_processed_plus_blossom_match_edge(&mut self, plus_node: u32) {
        if plus_node == NONE
            || (plus_node as usize) >= self.nodes.len()
            || !self.nodes[plus_node as usize].is_outer
            || self.nodes[plus_node as usize].flag != PLUS
            || !self.nodes[plus_node as usize].is_processed
        {
            return;
        }

        let root = self.find_tree_root(plus_node);
        if root == NONE {
            return;
        }

        let minus = self.arc_head_outer(self.nodes[plus_node as usize].match_arc);
        if minus == NONE
            || (minus as usize) >= self.nodes.len()
            || !self.nodes[minus as usize].is_blossom
            || !self.nodes[minus as usize].is_processed
        {
            return;
        }

        let match_edge = arc_edge(self.nodes[plus_node as usize].match_arc);
        if (match_edge as usize) < self.edge_num {
            self.set_generic_pq_blossoms_root_slot(match_edge, root, false);
        }
    }

    pub(super) fn requeue_edges_after_expand(&mut self, node: u32) {
        if node == NONE
            || (node as usize) >= self.nodes.len()
            || !self.nodes[node as usize].is_outer
        {
            return;
        }

        let root = self.find_tree_root(node);
        if root == NONE {
            return;
        }

        let is_processed_plus =
            self.nodes[node as usize].flag == PLUS && self.nodes[node as usize].is_processed;
        let mut incident = self.take_incident_scratch();
        self.collect_incident_edges_into(node, &mut incident);
        for &(e_idx, dir) in &incident {
            if !matches!(self.edge_queue_owner(e_idx), GenericQueueState::None) {
                continue;
            }

            let other = self.edge_head_outer(e_idx, dir);
            if other == NONE || other == node || !self.nodes[other as usize].is_outer {
                continue;
            }

            let other_root = self.find_tree_root(other);
            if is_processed_plus {
                match self.nodes[other as usize].flag {
                    FREE => {
                        self.set_generic_pq0(e_idx, root);
                        continue;
                    }
                    PLUS if self.nodes[other as usize].is_processed && other_root != NONE => {
                        self.set_generic_pq00(e_idx, root, other_root);
                        continue;
                    }
                    MINUS if other_root != NONE && other_root != root => {
                        self.set_generic_pq01(e_idx, root, other_root);
                        continue;
                    }
                    _ => {}
                }
            }

            if other_root == NONE || other_root == root {
                continue;
            }
            if self.nodes[other as usize].flag != PLUS || !self.nodes[other as usize].is_processed {
                continue;
            }

            match self.nodes[node as usize].flag {
                FREE => self.set_generic_pq0(e_idx, other_root),
                PLUS if self.nodes[node as usize].is_processed => {
                    self.set_generic_pq00(e_idx, other_root, root);
                }
                MINUS => {
                    self.set_generic_pq01(e_idx, other_root, root);
                }
                _ => {}
            }
        }
        self.restore_incident_scratch(incident);

        if is_processed_plus {
            let minus = self.arc_head_outer(self.nodes[node as usize].match_arc);
            if minus != NONE
                && (minus as usize) < self.nodes.len()
                && self.nodes[minus as usize].is_blossom
                && self.nodes[minus as usize].is_processed
            {
                let match_edge = arc_edge(self.nodes[node as usize].match_arc);
                if (match_edge as usize) < self.edge_num
                    && matches!(self.edge_queue_owner(match_edge), GenericQueueState::None)
                {
                    self.set_generic_pq_blossoms_root_slot(match_edge, root, false);
                }
            }
        }
    }

    /// Augment along the path from `u` (in tree 1) to `v` (in tree 2)
    /// through edge `edge_idx`, flipping the matching along both paths
    /// to the roots, then removing both trees.
    pub(super) fn augment(&mut self, edge_idx: u32, _u: u32, _v: u32) {
        self.normalize_edge_outer_heads(edge_idx);

        let u = self.edges[edge_idx as usize].head[0];
        let v = self.edges[edge_idx as usize].head[1];
        let root_u = self.find_tree_root(u);
        let root_v = self.find_tree_root(v);

        let mut members_u = self.take_tree_members_u_scratch();
        let mut members_v = self.take_tree_members_v_scratch();
        self.collect_tree_members_with_scratch(root_u, &mut members_u);
        self.collect_tree_members_with_scratch(root_v, &mut members_v);

        self.prepare_tree_for_augment(root_u, &members_u);
        self.prepare_tree_for_augment(root_v, &members_v);

        self.augment_path_to_root(u);
        self.augment_path_to_root(v);

        self.nodes[u as usize].match_arc = make_arc(edge_idx, 1);
        self.nodes[v as usize].match_arc = make_arc(edge_idx, 0);

        self.root_list_remove(root_u);
        self.root_list_remove(root_v);
        self.detach_generic_root_after_augment(root_u);
        self.detach_generic_root_after_augment(root_v);

        self.free_tree_members(&members_u);
        self.free_tree_members(&members_v);
        self.restore_tree_members_u_scratch(members_u);
        self.restore_tree_members_v_scratch(members_v);

        self.tree_num -= 2;
    }

    #[allow(clippy::manual_swap, clippy::too_many_lines)]
    pub(super) fn prepare_tree_for_augment(&mut self, root: u32, members: &[u32]) {
        if root == NONE || !self.nodes[root as usize].is_outer {
            return;
        }
        let eps = self.tree_eps(root);

        let mut incident_pairs = self.take_incident_pairs_scratch();
        let mut pair_marks = self.take_pair_marks_scratch();
        if pair_marks.len() != self.scheduler_tree_edges.len() {
            pair_marks.resize(self.scheduler_tree_edges.len(), 0);
        }
        if (root as usize) < self.scheduler_trees.len() {
            for dir in 0..2usize {
                let mut pair_cursor = self.scheduler_trees[root as usize].first[dir];
                while let Some(pair_idx) = pair_cursor {
                    if pair_idx >= self.scheduler_tree_edges.len() {
                        break;
                    }
                    let seen_mask = 1u8 << dir;
                    if pair_marks[pair_idx] & seen_mask == 0 {
                        pair_marks[pair_idx] |= seen_mask;
                        incident_pairs.push((pair_idx, dir));
                    }
                    pair_cursor = self.scheduler_tree_edges[pair_idx].next[dir];
                }
            }
        }

        for (pair_idx, current_side) in &incident_pairs {
            if *pair_idx >= self.scheduler_tree_edges.len() {
                continue;
            }
            if self.scheduler_tree_edge_dir(*pair_idx, root) != Some(*current_side) {
                continue;
            }
            let Some(other_root) = self.scheduler_tree_edge_other(*pair_idx, root) else {
                continue;
            };
            if (other_root as usize) < self.scheduler_trees.len() {
                self.scheduler_trees[other_root as usize].current =
                    SchedulerCurrent::Pair { pair_idx: *pair_idx, dir: *current_side };
            }
        }

        let mut members_mask = self.take_members_mask_scratch();
        if members_mask.len() != self.nodes.len() {
            members_mask.resize(self.nodes.len(), false);
        }
        for &v in members {
            members_mask[v as usize] = true;
        }

        let mut current = self.nodes[root as usize].first_tree_child;
        let mut incident = self.take_incident_scratch();
        let mut queue_edges = self.take_queue_edges_scratch();
        while current != NONE {
            let plus = current;
            let minus = self.arc_head_raw(self.nodes[plus as usize].match_arc);
            if minus != NONE
                && (minus as usize) < self.nodes.len()
                && self.nodes[minus as usize].is_processed
            {
                if self.nodes[minus as usize].is_blossom {
                    let match_arc = self.nodes[minus as usize].match_arc;
                    if match_arc != NONE {
                        let match_edge = arc_edge(match_arc) as usize;
                        if match_edge < self.edge_num {
                            let tmp = self.edges[match_edge].slack;
                            self.edges[match_edge].slack = self.nodes[minus as usize].y;
                            self.nodes[minus as usize].y = tmp;
                            self.clear_generic_queue_state(match_edge as u32);
                        }
                    }
                }

                self.collect_incident_edges_into(minus, &mut incident);
                for &(e_idx, dir) in &incident {
                    self.normalize_edge_outer_heads(e_idx);
                    let other = self.edges[e_idx as usize].head[dir];
                    if other == NONE {
                        continue;
                    }
                    if self.nodes[other as usize].flag == PLUS
                        && self.nodes[other as usize].is_processed
                    {
                        let other_root = self.find_tree_root(other);
                        if other_root != root {
                            self.edges[e_idx as usize].slack += eps;
                            if other_root != NONE
                                && matches!(self.edge_queue_owner(e_idx), GenericQueueState::None)
                            {
                                self.set_generic_pq0(e_idx, other_root);
                            }
                        }
                    } else {
                        self.edges[e_idx as usize].slack += eps;
                    }
                }
            }

            current = self.next_tree_plus(plus, root).unwrap_or(NONE);
        }
        self.restore_incident_scratch(incident);

        for &(pair_idx, current_side) in &incident_pairs {
            if pair_idx >= self.scheduler_tree_edges.len() {
                continue;
            }
            if self.scheduler_tree_edge_dir(pair_idx, root) != Some(current_side) {
                continue;
            }
            let other_side = 1 - current_side;
            let Some(other_root) = self.scheduler_tree_edge_other(pair_idx, root) else {
                continue;
            };

            if (other_root as usize) < self.scheduler_trees.len() {
                queue_edges.clear();
                core::mem::swap(
                    &mut queue_edges,
                    &mut self.scheduler_tree_edges[pair_idx].pq01[other_side],
                );
                for &e_idx in &queue_edges {
                    self.set_generic_pq0(e_idx, other_root);
                }
            }

            queue_edges.clear();
            core::mem::swap(&mut queue_edges, &mut self.scheduler_tree_edges[pair_idx].pq00);
            for &e_idx in &queue_edges {
                if (e_idx as usize) < self.edge_num {
                    self.edges[e_idx as usize].slack -= eps;
                    self.normalize_edge_outer_heads(e_idx);
                }
                if other_root != NONE {
                    self.set_generic_pq0(e_idx, other_root);
                } else {
                    self.clear_generic_queue_state(e_idx);
                }
            }

            queue_edges.clear();
            core::mem::swap(
                &mut queue_edges,
                &mut self.scheduler_tree_edges[pair_idx].pq01[current_side],
            );
            for &e_idx in &queue_edges {
                if (e_idx as usize) < self.edge_num {
                    self.edges[e_idx as usize].slack -= eps;
                    self.normalize_edge_outer_heads(e_idx);
                }
                if matches!(
                    self.edge_queue_owner(e_idx),
                    GenericQueueState::Pq01Pair { pair_idx: q_pair_idx, dir }
                        if q_pair_idx == pair_idx && dir == current_side
                ) {
                    self.clear_generic_queue_state(e_idx);
                }
            }

            if (other_root as usize) < self.scheduler_trees.len() {
                self.scheduler_trees[other_root as usize].current = SchedulerCurrent::None;
            }
        }

        if (root as usize) < self.scheduler_trees.len() {
            queue_edges.clear();
            core::mem::swap(&mut queue_edges, &mut self.scheduler_trees[root as usize].pq0);
            for &e_idx in &queue_edges {
                if (e_idx as usize) < self.edge_num {
                    self.edges[e_idx as usize].slack -= eps;
                    self.normalize_edge_outer_heads(e_idx);
                }
                self.clear_generic_queue_state(e_idx);
            }

            queue_edges.clear();
            core::mem::swap(&mut queue_edges, &mut self.scheduler_trees[root as usize].pq00_local);
            for &e_idx in &queue_edges {
                if matches!(
                    self.edge_queue_owner(e_idx),
                    GenericQueueState::Pq00Local { root: q_root } if q_root == root
                ) {
                    self.clear_generic_queue_state(e_idx);
                }
                let _ = self.process_edge00(e_idx, true);
            }
        }

        self.nodes[root as usize].flag = FREE;
        self.nodes[root as usize].is_processed = false;
        self.nodes[root as usize].y += eps;

        let mut current = self.nodes[root as usize].first_tree_child;
        while current != NONE {
            let plus = current;
            let minus = self.arc_head_raw(self.nodes[plus as usize].match_arc);
            if minus != NONE && members_mask[minus as usize] {
                self.nodes[minus as usize].flag = FREE;
                self.nodes[minus as usize].is_processed = false;
                self.nodes[minus as usize].y -= eps;
            }
            self.nodes[plus as usize].flag = FREE;
            self.nodes[plus as usize].is_processed = false;
            self.nodes[plus as usize].y += eps;

            current = self.next_tree_plus(plus, root).unwrap_or(NONE);
        }
        for &v in members {
            members_mask[v as usize] = false;
        }
        for &(pair_idx, _) in &incident_pairs {
            pair_marks[pair_idx] = 0;
        }
        self.restore_queue_edges_scratch(queue_edges);
        self.restore_members_mask_scratch(members_mask);
        self.restore_incident_pairs_scratch(incident_pairs);
        self.restore_pair_marks_scratch(pair_marks);
    }

    /// Flip matching along the alternating path from node `v` to its tree root.
    pub(super) fn augment_path_to_root(&mut self, i0: u32) {
        if self.nodes[i0 as usize].is_tree_root {
            return;
        }

        let mut i = i0;
        let mut j = self.arc_head_outer(self.nodes[i as usize].match_arc);
        let mut aa = self.nodes[j as usize].tree_parent_arc;
        i = self.arc_head_outer(aa);
        self.nodes[j as usize].match_arc = aa;

        while !self.nodes[i as usize].is_tree_root {
            j = self.arc_head_outer(self.nodes[i as usize].match_arc);
            self.nodes[i as usize].match_arc = arc_rev(aa);
            aa = self.nodes[j as usize].tree_parent_arc;
            let next_i = self.arc_head_outer(aa);
            self.nodes[j as usize].match_arc = aa;
            i = next_i;
        }

        self.nodes[i as usize].match_arc = arc_rev(aa);
    }

    pub(super) fn free_tree_members(&mut self, members: &[u32]) {
        for &v in members {
            self.nodes[v as usize].flag = FREE;
            self.nodes[v as usize].is_tree_root = false;
            self.nodes[v as usize].tree_eps = 0;
            self.nodes[v as usize].is_processed = false;
            self.nodes[v as usize].first_tree_child = NONE;
            self.nodes[v as usize].tree_sibling_next = NONE;
            self.nodes[v as usize].tree_sibling_prev = NONE;
            self.nodes[v as usize].tree_parent_arc = NONE;
            self.nodes[v as usize].tree_root = NONE;
        }
    }

    pub(super) fn collect_tree_members_with_scratch(&mut self, root: u32, members: &mut Vec<u32>) {
        let max_members = self.nodes.len();
        let mut stack = self.take_queue_edges_scratch();
        stack.push(root);
        while let Some(plus) = stack.pop() {
            if members.len() >= max_members {
                break;
            }
            members.push(plus);
            if !self.nodes[plus as usize].is_tree_root {
                let marc = self.nodes[plus as usize].match_arc;
                if marc != NONE {
                    let me = arc_edge(marc) as usize;
                    let md = arc_dir(marc);
                    let minus = self.edge_head_outer(me as u32, md);
                    members.push(minus);
                }
            }
            let mut child = self.nodes[plus as usize].first_tree_child;
            while child != NONE {
                stack.push(child);
                child = self.nodes[child as usize].tree_sibling_next;
            }
        }
        self.restore_queue_edges_scratch(stack);
    }
}
