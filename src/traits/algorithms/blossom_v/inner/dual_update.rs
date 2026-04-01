use alloc::vec::Vec;

use num_traits::AsPrimitive;

use super::{BlossomVState, NONE};
use crate::traits::{Number, PositiveInteger, SparseValuedMatrix2D};

impl<M: SparseValuedMatrix2D + ?Sized> BlossomVState<M>
where
    M::Value: Number + AsPrimitive<i64>,
    M::RowIndex: PositiveInteger,
    M::ColumnIndex: PositiveInteger,
{
    pub(super) fn ensure_dual_pair_tree_caps(
        &mut self,
        tree_edge_idx: usize,
        root_to_var: &[usize],
        pair_tree_eps00: &mut [i64],
        pair_tree_eps01_dir0: &mut [i64],
        pair_tree_eps01_dir1: &mut [i64],
        pair_tree_ready: &mut [bool],
    ) {
        if pair_tree_ready[tree_edge_idx] {
            return;
        }

        let root_left = self.scheduler_tree_edges[tree_edge_idx].head[0];
        let root_right = self.scheduler_tree_edges[tree_edge_idx].head[1];
        if root_left != NONE
            && root_right != NONE
            && (root_left as usize) < root_to_var.len()
            && (root_right as usize) < root_to_var.len()
            && root_to_var[root_left as usize] != usize::MAX
            && root_to_var[root_right as usize] != usize::MAX
        {
            let eps_left = self.tree_eps(root_left);
            let eps_right = self.tree_eps(root_right);

            if let Some(e_idx) = self.scheduler_tree_edge_min_pq00_edge_for_duals(
                tree_edge_idx,
                root_left,
                root_right,
            ) {
                pair_tree_eps00[tree_edge_idx] =
                    self.edges[e_idx as usize].slack - eps_left - eps_right;
            }
            if let Some(e_idx) = self.scheduler_tree_edge_min_pq01_edge_for_duals(
                tree_edge_idx,
                0,
                root_right,
                root_left,
            ) {
                pair_tree_eps01_dir0[tree_edge_idx] =
                    self.edges[e_idx as usize].slack - eps_right + eps_left;
            }
            if let Some(e_idx) = self.scheduler_tree_edge_min_pq01_edge_for_duals(
                tree_edge_idx,
                1,
                root_left,
                root_right,
            ) {
                pair_tree_eps01_dir1[tree_edge_idx] =
                    self.edges[e_idx as usize].slack - eps_left + eps_right;
            }
        }

        pair_tree_ready[tree_edge_idx] = true;
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn compute_dual_component_eps(
        &mut self,
        start: usize,
        roots: &[u32],
        root_to_var: &[usize],
        local_caps: &[i64],
        fixed_tree: usize,
        inf_cap: i64,
        deltas: &[i64],
        pair_tree_eps00: &mut [i64],
        pair_tree_eps01_dir0: &mut [i64],
        pair_tree_eps01_dir1: &mut [i64],
        pair_tree_ready: &mut [bool],
        marks: &mut [usize],
        queue: &mut Vec<usize>,
        component: &mut Vec<usize>,
    ) -> Option<i64> {
        queue.clear();
        component.clear();
        queue.push(start);
        component.push(start);
        marks[start] = start;

        let mut queue_head = 0usize;
        let mut eps = local_caps[start];

        while queue_head < queue.len() {
            let t = queue[queue_head];
            queue_head += 1;
            let t_root = roots[t];
            for dir in 0..2usize {
                let mut pair_cursor = if (t_root as usize) < self.scheduler_trees.len() {
                    self.scheduler_trees[t_root as usize].first[dir]
                } else {
                    None
                };
                while let Some(tree_edge_idx) = pair_cursor {
                    let next_pair = self.scheduler_tree_edges[tree_edge_idx].next[dir];
                    let Some(other_root) = self.scheduler_tree_edge_other(tree_edge_idx, t_root)
                    else {
                        pair_cursor = next_pair;
                        continue;
                    };
                    if (other_root as usize) >= root_to_var.len() {
                        pair_cursor = next_pair;
                        continue;
                    }
                    let t2 = root_to_var[other_root as usize];
                    if t2 == usize::MAX || t == t2 {
                        pair_cursor = next_pair;
                        continue;
                    }

                    self.ensure_dual_pair_tree_caps(
                        tree_edge_idx,
                        root_to_var,
                        pair_tree_eps00,
                        pair_tree_eps01_dir0,
                        pair_tree_eps01_dir1,
                        pair_tree_ready,
                    );

                    let eps00 = pair_tree_eps00[tree_edge_idx];
                    if marks[t2] == start {
                        if eps00 < inf_cap {
                            eps = eps.min(eps00 / 2);
                        }
                        pair_cursor = next_pair;
                        continue;
                    }

                    let eps01_forward = if dir == 0 {
                        pair_tree_eps01_dir0[tree_edge_idx]
                    } else {
                        pair_tree_eps01_dir1[tree_edge_idx]
                    };
                    let reverse_dir = 1 - dir;
                    let eps01_reverse = if reverse_dir == 0 {
                        pair_tree_eps01_dir0[tree_edge_idx]
                    } else {
                        pair_tree_eps01_dir1[tree_edge_idx]
                    };

                    let eps2 = if marks[t2] == fixed_tree {
                        deltas[t2]
                    } else if eps01_forward > 0 && eps01_reverse > 0 {
                        0
                    } else {
                        marks[t2] = start;
                        queue.push(t2);
                        component.push(t2);
                        eps = eps.min(local_caps[t2]);
                        if eps00 < inf_cap {
                            eps = eps.min(eps00);
                        }
                        pair_cursor = next_pair;
                        continue;
                    };

                    if eps00 < inf_cap {
                        eps = eps.min(eps00 - eps2);
                    }
                    if eps01_forward < inf_cap {
                        eps = eps.min(eps2 + eps01_forward);
                    }

                    pair_cursor = next_pair;
                }
            }
        }

        (eps < inf_cap).then_some(eps)
    }

    /// Update dual variables uniformly. Returns false if no progress possible.
    #[allow(clippy::too_many_lines)]
    pub(super) fn update_duals(&mut self) -> bool {
        let mut roots = core::mem::take(&mut self.scratch.dual_roots);
        let mut seen = core::mem::take(&mut self.scratch.dual_seen);
        let mut root_to_var = core::mem::take(&mut self.scratch.dual_root_to_var);
        let mut local_caps = core::mem::take(&mut self.scratch.dual_local_caps);
        let mut pair_tree_eps00 = core::mem::take(&mut self.scratch.dual_pair_tree_eps00);
        let mut pair_tree_eps01_dir0 = core::mem::take(&mut self.scratch.dual_pair_tree_eps01_dir0);
        let mut pair_tree_eps01_dir1 = core::mem::take(&mut self.scratch.dual_pair_tree_eps01_dir1);
        let mut pair_tree_ready = core::mem::take(&mut self.scratch.dual_pair_tree_ready);
        let mut deltas = core::mem::take(&mut self.scratch.dual_deltas);
        let mut marks = core::mem::take(&mut self.scratch.dual_marks);
        let mut queue = core::mem::take(&mut self.scratch.dual_queue);
        let mut component = core::mem::take(&mut self.scratch.dual_component);

        self.fill_current_root_list(&mut roots, &mut seen);
        let result = if roots.is_empty() {
            false
        } else {
            let inf_cap = i64::MAX / 4;
            self.build_dual_root_index(&roots, &mut root_to_var);
            self.seed_dual_local_caps(&roots, inf_cap, &mut local_caps);
            self.reset_dual_pair_tree_caps(
                inf_cap,
                &mut pair_tree_eps00,
                &mut pair_tree_eps01_dir0,
                &mut pair_tree_eps01_dir1,
                &mut pair_tree_ready,
            );
            deltas.clear();
            deltas.resize(roots.len(), 0);
            let all_roots_processed =
                roots.iter().all(|&root| self.nodes[root as usize].is_processed);
            let fixed_tree = roots.len();
            marks.clear();
            marks.resize(roots.len(), usize::MAX);
            queue.clear();
            component.clear();
            let mut unbounded_component = false;

            for start in 0..roots.len() {
                if marks[start] != usize::MAX {
                    continue;
                }

                let Some(eps) = self.compute_dual_component_eps(
                    start,
                    &roots,
                    &root_to_var,
                    &local_caps,
                    fixed_tree,
                    inf_cap,
                    &deltas,
                    &mut pair_tree_eps00,
                    &mut pair_tree_eps01_dir0,
                    &mut pair_tree_eps01_dir1,
                    &mut pair_tree_ready,
                    &mut marks,
                    &mut queue,
                    &mut component,
                ) else {
                    unbounded_component = true;
                    break;
                };

                for &t in &component {
                    deltas[t] = eps;
                    marks[t] = fixed_tree;
                }
            }

            if (unbounded_component && all_roots_processed)
                || deltas.iter().all(|&delta| delta <= 0)
            {
                false
            } else {
                for (var, &root) in roots.iter().enumerate() {
                    let delta = deltas[var].max(0);
                    if delta > 0 {
                        self.nodes[root as usize].tree_eps += delta;
                    }
                }

                true
            }
        };

        self.scratch.dual_roots = roots;
        self.scratch.dual_seen = seen;
        self.scratch.dual_root_to_var = root_to_var;
        self.scratch.dual_local_caps = local_caps;
        self.scratch.dual_pair_tree_eps00 = pair_tree_eps00;
        self.scratch.dual_pair_tree_eps01_dir0 = pair_tree_eps01_dir0;
        self.scratch.dual_pair_tree_eps01_dir1 = pair_tree_eps01_dir1;
        self.scratch.dual_pair_tree_ready = pair_tree_ready;
        self.scratch.dual_deltas = deltas;
        self.scratch.dual_marks = marks;
        self.scratch.dual_queue = queue;
        self.scratch.dual_component = component;

        result
    }
}
