use alloc::{vec, vec::Vec};

use num_traits::AsPrimitive;

use super::{BlossomVState, FREE, NONE, PLUS, arc_dir, arc_edge, make_arc};
#[cfg(test)]
use super::{Edge, InitGlobalEvent, InitGlobalStepTrace, normalized_edge_pair};
use crate::traits::{Number, PositiveInteger, SparseValuedMatrix2D};

impl<M: SparseValuedMatrix2D + ?Sized> BlossomVState<M>
where
    M::Value: Number + AsPrimitive<i64>,
    M::RowIndex: PositiveInteger,
    M::ColumnIndex: PositiveInteger,
{
    pub(super) fn init_global_ported(&mut self) {
        if self.tree_num == 0 {
            return;
        }

        let mut best_edge = vec![NONE; self.nodes.len()];
        let mut root = 0usize;
        while root < self.nodes.len() {
            if self.nodes[root].is_outer && self.nodes[root].is_tree_root {
                self.init_global_process_root(root as u32, &mut best_edge);
                if best_edge.len() < self.nodes.len() {
                    best_edge.resize(self.nodes.len(), NONE);
                }
            }
            if self.tree_num == 0 {
                break;
            }
            root += 1;
        }

        if self.tree_num == 0 {
            self.expand_solved_outer_blossoms();
        } else {
            self.init_global_finalize();
        }
    }

    /// Full startup search for a single greedy tree root.
    #[allow(clippy::bool_to_int_with_if, clippy::too_many_lines)]
    pub(super) fn init_global_process_root(&mut self, root: u32, best_edge: &mut Vec<u32>) -> bool {
        #[derive(Clone, Copy, PartialEq, Eq)]
        enum CriticalKind {
            None,
            Augment,
            Shrink,
        }

        if !self.nodes[root as usize].is_tree_root {
            return false;
        }

        self.init_clear_best_edges(best_edge);
        for node in &mut self.nodes {
            node.is_processed = false;
        }

        let mut eps = 0i64;
        let mut critical_arc = NONE;
        let mut critical_eps = i64::MAX;
        let mut critical_kind = CriticalKind::None;
        let mut branch_root = root;
        let mut current = root;
        let mut changed = false;

        loop {
            self.nodes[current as usize].is_processed = true;
            self.nodes[current as usize].y -= eps;
            if !self.nodes[current as usize].is_tree_root {
                let minus = self.arc_head_raw(self.nodes[current as usize].match_arc);
                self.nodes[minus as usize].y += eps;
            }

            let incident = self.incident_edges(current);
            let mut stop_search = false;
            let mut stop_after = incident.len();
            for (incident_idx, (e_idx, dir)) in incident.iter().copied().enumerate() {
                self.edges[e_idx as usize].slack += eps;
                let other = self.edges[e_idx as usize].head[dir];
                if other == current {
                    continue;
                }

                if self.find_tree_root(other) == root && self.nodes[other as usize].flag == PLUS {
                    let mut slack = self.edges[e_idx as usize].slack;
                    if !self.nodes[other as usize].is_processed {
                        slack += eps;
                    }
                    if critical_arc == NONE || 2 * critical_eps > slack {
                        critical_kind = CriticalKind::Shrink;
                        critical_eps = slack / 2;
                        critical_arc = make_arc(e_idx, dir);
                        if critical_eps <= eps {
                            stop_search = true;
                            stop_after = incident_idx + 1;
                            break;
                        }
                    }
                } else if self.nodes[other as usize].flag == PLUS {
                    if critical_arc == NONE || critical_eps >= self.edges[e_idx as usize].slack {
                        critical_kind = CriticalKind::Augment;
                        critical_eps = self.edges[e_idx as usize].slack;
                        critical_arc = make_arc(e_idx, dir);
                        if critical_eps <= eps {
                            stop_search = true;
                            stop_after = incident_idx + 1;
                            break;
                        }
                    }
                } else if self.nodes[other as usize].flag == FREE
                    && self.nodes[other as usize].is_outer
                {
                    let slack = self.edges[e_idx as usize].slack;
                    if slack > eps {
                        if slack < critical_eps {
                            let prev = best_edge[other as usize];
                            if prev == NONE || slack < self.edges[prev as usize].slack {
                                best_edge[other as usize] = e_idx;
                            }
                        }
                    } else {
                        best_edge[other as usize] = NONE;
                        let next_plus = self.arc_head_outer(self.nodes[other as usize].match_arc);
                        if next_plus as usize >= best_edge.len() {
                            best_edge.resize(self.nodes.len(), NONE);
                        }
                        best_edge[next_plus as usize] = NONE;
                        self.apply_init_global_grow(e_idx, current, other);
                        changed = true;
                    }
                }
            }

            if stop_search {
                for (e_idx, _dir) in incident[stop_after..].iter().copied() {
                    self.edges[e_idx as usize].slack += eps;
                }
                break;
            }

            if let Some(next) = self.next_tree_plus(current, branch_root) {
                current = next;
                continue;
            }

            if let Some(e_idx) = self.init_pick_best_edge(best_edge, critical_eps) {
                let slack = self.edges[e_idx as usize].slack;
                if slack < critical_eps {
                    eps = slack;
                    let dir = if self.nodes[self.edges[e_idx as usize].head[0] as usize].flag
                        == FREE
                        && self.nodes[self.edges[e_idx as usize].head[0] as usize].is_outer
                    {
                        0
                    } else {
                        1
                    };
                    let free = self.edges[e_idx as usize].head[dir];
                    if self.nodes[free as usize].flag != FREE || !self.nodes[free as usize].is_outer
                    {
                        best_edge[free as usize] = NONE;
                        continue;
                    }
                    let plus = self.edges[e_idx as usize].head[1 - dir];
                    best_edge[free as usize] = NONE;
                    let next_plus = self.arc_head_raw(self.nodes[free as usize].match_arc);
                    if next_plus as usize >= best_edge.len() {
                        best_edge.resize(self.nodes.len(), NONE);
                    }
                    best_edge[next_plus as usize] = NONE;
                    self.apply_init_global_grow(e_idx, plus, free);
                    changed = true;
                    branch_root = next_plus;
                    current = next_plus;
                    continue;
                }
            }

            break;
        }

        let final_eps = if critical_eps == i64::MAX { eps } else { critical_eps };
        if final_eps > 0 {
            changed = true;
        }
        self.init_global_cleanup(root, final_eps, best_edge);

        if critical_arc == NONE {
            return changed;
        }

        let e_idx = arc_edge(critical_arc);
        let dir = arc_dir(critical_arc);
        let left = self.edges[e_idx as usize].head[1 - dir];
        let right = self.edges[e_idx as usize].head[dir];
        match critical_kind {
            CriticalKind::Augment => self.apply_init_global_augment(e_idx, left, right, root),
            CriticalKind::Shrink => self.apply_init_global_shrink(e_idx, left, right, root),
            CriticalKind::None => {}
        }
        true
    }

    #[cfg(test)]
    pub(super) fn init_global_event_grow(
        e_idx: u32,
        plus: u32,
        free: u32,
        edges: &[Edge],
    ) -> InitGlobalEvent {
        let edge = normalized_edge_pair(edges[e_idx as usize].head0);
        InitGlobalEvent::Grow { edge, plus, free }
    }

    #[cfg(test)]
    pub(super) fn apply_init_global_grow(&mut self, e_idx: u32, plus: u32, free: u32) {
        let event = Self::init_global_event_grow(e_idx, plus, free, &self.edges);
        let before = self.test_strict_parity_snapshot();
        self.init_grow(e_idx, plus, free);
        let after = self.test_strict_parity_snapshot();
        self.test_state.init_global_trace.push(event.clone());
        self.test_state.init_global_steps.push(InitGlobalStepTrace { event, before, after });
    }

    #[cfg(not(test))]
    pub(super) fn apply_init_global_grow(&mut self, e_idx: u32, plus: u32, free: u32) {
        self.init_grow(e_idx, plus, free);
        self.maybe_write_debug_trace_snapshot("GROW_AFTER");
    }

    #[cfg(test)]
    pub(super) fn init_global_event_augment(
        e_idx: u32,
        left: u32,
        right: u32,
        edges: &[Edge],
    ) -> InitGlobalEvent {
        let edge = normalized_edge_pair(edges[e_idx as usize].head0);
        InitGlobalEvent::Augment { edge, left, right }
    }

    #[cfg(test)]
    pub(super) fn init_global_event_shrink(
        e_idx: u32,
        left: u32,
        right: u32,
        edges: &[Edge],
    ) -> InitGlobalEvent {
        let edge = normalized_edge_pair(edges[e_idx as usize].head0);
        InitGlobalEvent::Shrink { edge, left, right }
    }

    #[cfg(test)]
    pub(super) fn apply_init_global_augment(
        &mut self,
        e_idx: u32,
        left: u32,
        right: u32,
        root: u32,
    ) {
        let event = Self::init_global_event_augment(e_idx, left, right, &self.edges);
        let before = self.test_strict_parity_snapshot();
        self.init_augment_branch(left, root);
        if self.nodes[right as usize].is_outer {
            self.init_augment_branch(right, right);
        } else {
            self.init_expand(right);
            self.tree_num -= 1;
        }
        let u_dir = if self.edges[e_idx as usize].head[0] != right { 1 } else { 0 };
        self.nodes[left as usize].match_arc = make_arc(e_idx, u_dir);
        self.nodes[right as usize].match_arc = make_arc(e_idx, 1 - u_dir);
        let after = self.test_strict_parity_snapshot();
        self.test_state.init_global_trace.push(event.clone());
        self.test_state.init_global_steps.push(InitGlobalStepTrace { event, before, after });
    }

    #[cfg(not(test))]
    pub(super) fn apply_init_global_augment(
        &mut self,
        e_idx: u32,
        left: u32,
        right: u32,
        root: u32,
    ) {
        self.init_augment_branch(left, root);
        if self.nodes[right as usize].is_outer {
            self.init_augment_branch(right, right);
        } else {
            self.init_expand(right);
            self.tree_num -= 1;
        }
        let u_dir = if self.edges[e_idx as usize].head[0] != right { 1 } else { 0 };
        self.nodes[left as usize].match_arc = make_arc(e_idx, u_dir);
        self.nodes[right as usize].match_arc = make_arc(e_idx, 1 - u_dir);
        self.maybe_write_debug_trace_snapshot("AUGMENT_AFTER");
    }

    #[cfg(test)]
    pub(super) fn apply_init_global_shrink(
        &mut self,
        e_idx: u32,
        left: u32,
        right: u32,
        root: u32,
    ) {
        let event = Self::init_global_event_shrink(e_idx, left, right, &self.edges);
        let before = self.test_strict_parity_snapshot();
        self.init_shrink(e_idx, root);
        let after = self.test_strict_parity_snapshot();
        self.test_state.init_global_trace.push(event.clone());
        self.test_state.init_global_steps.push(InitGlobalStepTrace { event, before, after });
    }

    #[cfg(not(test))]
    pub(super) fn apply_init_global_shrink(
        &mut self,
        e_idx: u32,
        _left: u32,
        _right: u32,
        root: u32,
    ) {
        self.init_shrink(e_idx, root);
        self.maybe_write_debug_trace_snapshot("SHRINK_AFTER");
    }
}
