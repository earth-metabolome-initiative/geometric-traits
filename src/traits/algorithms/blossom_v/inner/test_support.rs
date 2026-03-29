#![allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]

use alloc::vec::Vec;

use num_traits::AsPrimitive;

use super::*;

#[derive(Clone, Default)]
pub(super) struct GenericTreeQueues {
    pub(super) pq0: Vec<u32>,
    pub(super) pq00_local: Vec<u32>,
    pub(super) pq_blossoms: Vec<u32>,
    pub(super) tree_edges: [Vec<usize>; 2],
}

#[derive(Clone, Default)]
pub(super) struct GenericPairQueues {
    pub(super) head: [u32; 2],
    pub(super) pq00: Vec<u32>,
    pub(super) pq01: [Vec<u32>; 2],
}

impl GenericPairQueues {
    pub(super) fn new(current_root: u32, other_root: u32) -> Self {
        Self { head: [other_root, current_root], pq00: Vec::new(), pq01: [Vec::new(), Vec::new()] }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum InitGlobalEvent {
    Grow { edge: (u32, u32), plus: u32, free: u32 },
    Augment { edge: (u32, u32), left: u32, right: u32 },
    Shrink { edge: (u32, u32), left: u32, right: u32 },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct NodeParitySnapshot {
    pub(super) flag: u8,
    pub(super) is_outer: bool,
    pub(super) is_tree_root: bool,
    pub(super) is_processed: bool,
    pub(super) tree_root: Option<u32>,
    pub(super) match_partner: Option<u32>,
    pub(super) match_edge: Option<(u32, u32)>,
    pub(super) tree_parent_edge: Option<(u32, u32)>,
    pub(super) first_tree_child: Option<u32>,
    pub(super) tree_sibling_prev: Option<u32>,
    pub(super) tree_sibling_next: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct StrictParitySnapshot {
    pub(super) y: Vec<i64>,
    pub(super) edge_slacks: Vec<((u32, u32), i64)>,
    pub(super) nodes: Vec<NodeParitySnapshot>,
    pub(super) tree_num: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct InitGlobalStepTrace {
    pub(super) event: InitGlobalEvent,
    pub(super) before: StrictParitySnapshot,
    pub(super) after: StrictParitySnapshot,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum GenericPrimalEvent {
    Grow { edge: (u32, u32), plus: u32, free: u32 },
    Augment { edge: (u32, u32), left: u32, right: u32 },
    Shrink { edge: (u32, u32), left: u32, right: u32 },
    Expand { blossom: u32 },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct GenericPrimalStepTrace {
    pub(super) event: GenericPrimalEvent,
    pub(super) before: StrictParitySnapshot,
    pub(super) after: StrictParitySnapshot,
}

#[derive(Debug, PartialEq, Eq)]
pub(super) struct GreedyInitSnapshot {
    pub(super) y: Vec<i64>,
    pub(super) slacks: Vec<i64>,
    pub(super) matching: Vec<i32>,
    pub(super) flags: Vec<u8>,
    pub(super) is_outer: Vec<bool>,
    pub(super) tree_eps_by_node: Vec<i64>,
    pub(super) tree_num: usize,
}

pub(super) fn mark_tree_roots_processed<M>(state: &mut BlossomVState<M>)
where
    M: SparseValuedMatrix2D + ?Sized,
    M::Value: Number + AsPrimitive<i64>,
    M::RowIndex: PositiveInteger,
    M::ColumnIndex: PositiveInteger,
{
    let roots = state.current_root_list();
    for root in roots {
        state.seed_tree_root_frontier(root);
    }
}

pub(super) fn rebuild_scheduler_tree_mirror<M>(state: &mut BlossomVState<M>)
where
    M: SparseValuedMatrix2D + ?Sized,
    M::Value: Number + AsPrimitive<i64>,
    M::RowIndex: PositiveInteger,
    M::ColumnIndex: PositiveInteger,
{
    for node in &mut state.pq_nodes {
        *node = PQNode::RESET;
    }
    state.scheduler_trees = vec![PersistentTreeState::default(); state.nodes.len()];
    state.scheduler_tree_edges =
        vec![PersistentTreeEdgeState::default(); state.generic_pairs.len()];

    for (pair_idx, pair) in state.generic_pairs.iter().enumerate() {
        state.scheduler_tree_edges[pair_idx].head = pair.head;
        state.scheduler_tree_edges[pair_idx].pq00 = pair.pq00.clone();
        state.scheduler_tree_edges[pair_idx].pq01 = pair.pq01.clone();
        for &e_idx in &pair.pq00 {
            if (e_idx as usize) >= state.edge_num {
                continue;
            }
            state.scheduler_tree_edges[pair_idx].pq00_heap.add(
                e_idx,
                state.edges.as_mut_slice(),
                state.pq_nodes.as_mut_slice(),
            );
        }
        for dir in 0..2usize {
            for &e_idx in &pair.pq01[dir] {
                if (e_idx as usize) >= state.edge_num {
                    continue;
                }
                state.scheduler_tree_edges[pair_idx].pq01_heap[dir].add(
                    e_idx,
                    state.edges.as_mut_slice(),
                    state.pq_nodes.as_mut_slice(),
                );
            }
        }
    }

    for (root_idx, tree) in state.generic_trees.iter().enumerate() {
        if root_idx >= state.scheduler_trees.len() {
            break;
        }
        let root = root_idx as u32;
        state.scheduler_trees[root_idx].root = root;
        state.scheduler_trees[root_idx].eps = state.tree_eps(root);
        state.scheduler_trees[root_idx].pq0 = tree.pq0.clone();
        state.scheduler_trees[root_idx].pq00_local = tree.pq00_local.clone();
        state.scheduler_trees[root_idx].pq_blossoms = tree.pq_blossoms.clone();
        for &e_idx in &tree.pq0 {
            if (e_idx as usize) >= state.edge_num {
                continue;
            }
            state.scheduler_trees[root_idx].pq0_heap.add(
                e_idx,
                state.edges.as_mut_slice(),
                state.pq_nodes.as_mut_slice(),
            );
        }
        for &e_idx in &tree.pq00_local {
            if (e_idx as usize) >= state.edge_num {
                continue;
            }
            state.scheduler_trees[root_idx].pq00_local_heap.add(
                e_idx,
                state.edges.as_mut_slice(),
                state.pq_nodes.as_mut_slice(),
            );
        }
        for &e_idx in &tree.pq_blossoms {
            if (e_idx as usize) >= state.edge_num {
                continue;
            }
            state.scheduler_trees[root_idx].pq_blossoms_heap.add(
                e_idx,
                state.edges.as_mut_slice(),
                state.pq_nodes.as_mut_slice(),
            );
        }

        for dir in 0..2usize {
            state.scheduler_trees[root_idx].first[dir] = tree.tree_edges[dir].first().copied();
            for (pos, &pair_idx) in tree.tree_edges[dir].iter().enumerate() {
                if pair_idx >= state.scheduler_tree_edges.len() {
                    continue;
                }
                state.scheduler_tree_edges[pair_idx].next[dir] =
                    tree.tree_edges[dir].get(pos + 1).copied();
            }
        }
    }
}

impl<M> BlossomVState<M>
where
    M: SparseValuedMatrix2D + ?Sized,
    M::Value: Number + AsPrimitive<i64>,
    M::RowIndex: PositiveInteger,
    M::ColumnIndex: PositiveInteger,
{
    pub(super) fn ensure_generic_tree_slot(&mut self, root: u32) {
        let needed = root as usize + 1;
        if self.generic_trees.len() < needed {
            self.generic_trees.resize_with(needed, GenericTreeQueues::default);
        }
    }

    pub(super) fn sync_generic_root_topology_from_scheduler(&mut self, root: u32) {
        if root == NONE {
            return;
        }
        self.ensure_generic_tree_slot(root);
        if (root as usize) >= self.scheduler_trees.len() {
            self.generic_trees[root as usize].tree_edges = [Vec::new(), Vec::new()];
            return;
        }

        for dir in 0..2usize {
            let mut edges = Vec::new();
            let mut cursor = self.scheduler_trees[root as usize].first[dir];
            let mut safety = self.scheduler_tree_edges.len() + 1;
            while let Some(pair_idx) = cursor {
                if pair_idx >= self.scheduler_tree_edges.len() || safety == 0 {
                    break;
                }
                if self.scheduler_tree_edge_dir(pair_idx, root) == Some(dir) {
                    edges.push(pair_idx);
                }
                cursor = self.scheduler_tree_edges[pair_idx].next[dir];
                safety -= 1;
            }
            self.generic_trees[root as usize].tree_edges[dir] = edges;
        }
    }

    pub(super) fn sync_generic_root_queues_from_scheduler(&mut self, root: u32) {
        if root == NONE {
            return;
        }
        self.ensure_generic_tree_slot(root);
        if (root as usize) >= self.scheduler_trees.len() {
            self.generic_trees[root as usize].pq0.clear();
            self.generic_trees[root as usize].pq00_local.clear();
            self.generic_trees[root as usize].pq_blossoms.clear();
            return;
        }
        self.generic_trees[root as usize].pq0 = self.scheduler_trees[root as usize].pq0.clone();
        self.generic_trees[root as usize].pq00_local =
            self.scheduler_trees[root as usize].pq00_local.clone();
        self.generic_trees[root as usize].pq_blossoms =
            self.scheduler_trees[root as usize].pq_blossoms.clone();
    }

    pub(super) fn sync_generic_pair_queues_from_scheduler(&mut self, pair_idx: usize) {
        if pair_idx >= self.generic_pairs.len() {
            return;
        }
        if pair_idx >= self.scheduler_tree_edges.len() {
            self.generic_pairs[pair_idx].pq00.clear();
            self.generic_pairs[pair_idx].pq01 = [Vec::new(), Vec::new()];
            return;
        }
        self.generic_pairs[pair_idx].pq00 = self.scheduler_tree_edges[pair_idx].pq00.clone();
        self.generic_pairs[pair_idx].pq01 = self.scheduler_tree_edges[pair_idx].pq01.clone();
    }

    pub(super) fn sync_generic_pair_head_from_scheduler(&mut self, pair_idx: usize) {
        if pair_idx >= self.generic_pairs.len() {
            return;
        }
        if pair_idx >= self.scheduler_tree_edges.len() {
            self.generic_pairs[pair_idx].head = [NONE, NONE];
            return;
        }
        self.generic_pairs[pair_idx].head = self.scheduler_tree_edges[pair_idx].head;
    }

    pub(super) fn queue_pair_touches_root(&self, pair_idx: usize, root: u32) -> bool {
        if pair_idx >= self.scheduler_tree_edges.len() {
            return false;
        }
        let head = self.scheduler_tree_edges[pair_idx].head;
        head[0] == root || head[1] == root
    }

    pub(super) fn set_generic_pq00_pair_slot(
        &mut self,
        e_idx: u32,
        pair_idx: usize,
        preserve_stamp: bool,
    ) {
        if (e_idx as usize) >= self.edge_num || pair_idx >= self.scheduler_tree_edges.len() {
            return;
        }
        self.ensure_scheduler_tree_edge_slot(pair_idx);
        let old_stamp = self.edge_queue_stamp(e_idx);
        self.remove_edge_from_generic_queue(e_idx);
        if !self.scheduler_tree_edges[pair_idx].pq00.contains(&e_idx) {
            self.scheduler_tree_edges[pair_idx].pq00.push(e_idx);
        }
        self.scheduler_tree_edges[pair_idx].pq00_heap.add(
            e_idx,
            self.edges.as_mut_slice(),
            self.pq_nodes.as_mut_slice(),
        );
        self.set_edge_queue_owner(e_idx, GenericQueueState::Pq00Pair { pair_idx });
        if preserve_stamp {
            self.set_edge_queue_stamp(e_idx, old_stamp);
        } else {
            self.generic_queue_epoch = self.generic_queue_epoch.wrapping_add(1);
            self.set_edge_queue_stamp(e_idx, self.generic_queue_epoch);
        }
        self.sync_generic_pair_queues_from_scheduler(pair_idx);
    }

    pub(super) fn clear_generic_queues_for_root(&mut self, root: u32) {
        if root == NONE {
            return;
        }
        let affected = (0..self.edge_num as u32)
            .filter(|&e_idx| {
                match self.edge_queue_owner(e_idx) {
                    GenericQueueState::None => false,
                    GenericQueueState::Pq0 { root: owner }
                    | GenericQueueState::Pq00Local { root: owner }
                    | GenericQueueState::PqBlossoms { root: owner } => owner == root,
                    GenericQueueState::Pq00Pair { pair_idx }
                    | GenericQueueState::Pq01Pair { pair_idx, .. } => {
                        self.queue_pair_touches_root(pair_idx, root)
                    }
                }
            })
            .collect::<Vec<_>>();
        for e_idx in affected {
            self.remove_edge_from_generic_queue(e_idx);
        }
        self.detach_scheduler_root_topology(root);
        self.sync_generic_root_topology_from_scheduler(root);
    }

    pub(super) fn test_state_snapshot(&self) -> GreedyInitSnapshot {
        let y = (0..self.node_num).map(|v| self.nodes[v].y).collect();
        let slacks = (0..self.edge_num).map(|e| self.edges[e].slack).collect();
        let matching = (0..self.node_num)
            .map(|v| {
                let arc = self.nodes[v].match_arc;
                match (arc != NONE).then(|| {
                    let e = arc_edge(arc) as usize;
                    let dir = arc_dir(arc);
                    self.edges[e].head[dir]
                }) {
                    Some(partner) => partner as i32,
                    None => -1,
                }
            })
            .collect();
        let flags = (0..self.node_num).map(|v| self.nodes[v].flag).collect();
        let is_outer = (0..self.node_num).map(|v| self.nodes[v].is_outer).collect();
        let tree_eps_by_node = (0..self.node_num)
            .map(|v| {
                let node = &self.nodes[v];
                if node.is_outer && node.flag != FREE && node.tree_root != NONE {
                    self.tree_eps(node.tree_root)
                } else {
                    0
                }
            })
            .collect();

        GreedyInitSnapshot {
            y,
            slacks,
            matching,
            flags,
            is_outer,
            tree_eps_by_node,
            tree_num: self.tree_num,
        }
    }

    pub(super) fn test_strict_parity_snapshot(&self) -> StrictParitySnapshot {
        let nodes = (0..self.node_num)
            .map(|v| {
                let node = &self.nodes[v];
                let match_partner = if node.match_arc == NONE {
                    None
                } else {
                    let e = arc_edge(node.match_arc) as usize;
                    let dir = arc_dir(node.match_arc);
                    Some(self.edges[e].head[dir])
                };
                let match_edge = (node.match_arc != NONE).then(|| {
                    normalized_edge_pair(self.edges[arc_edge(node.match_arc) as usize].head0)
                });
                let tree_parent_edge = (node.tree_parent_arc != NONE).then(|| {
                    normalized_edge_pair(self.edges[arc_edge(node.tree_parent_arc) as usize].head0)
                });
                NodeParitySnapshot {
                    flag: node.flag,
                    is_outer: node.is_outer,
                    is_tree_root: node.is_tree_root,
                    is_processed: node.is_processed,
                    tree_root: (node.tree_root != NONE).then_some(node.tree_root),
                    match_partner,
                    match_edge,
                    tree_parent_edge,
                    first_tree_child: (node.first_tree_child != NONE)
                        .then_some(node.first_tree_child),
                    tree_sibling_prev: (node.tree_sibling_prev != NONE)
                        .then_some(node.tree_sibling_prev),
                    tree_sibling_next: (node.tree_sibling_next != NONE)
                        .then_some(node.tree_sibling_next),
                }
            })
            .collect();

        let mut edge_slacks = (0..self.edge_num)
            .map(|e| (normalized_edge_pair(self.edges[e].head0), self.edges[e].slack))
            .collect::<Vec<_>>();
        edge_slacks.sort_unstable();

        StrictParitySnapshot {
            y: (0..self.node_num).map(|v| self.nodes[v].y).collect(),
            edge_slacks,
            nodes,
            tree_num: self.tree_num,
        }
    }

    pub(super) fn test_init_global_trace(&self) -> &[InitGlobalEvent] {
        &self.init_global_trace
    }

    pub(super) fn test_init_global_steps(&self) -> &[InitGlobalStepTrace] {
        &self.init_global_steps
    }

    pub(super) fn test_generic_primal_steps(&self) -> &[GenericPrimalStepTrace] {
        &self.generic_primal_steps
    }
}

pub(super) trait SchedulerMirrorTestExt {
    fn mark_tree_roots_processed(&mut self);
    fn rebuild_scheduler_tree_mirror(&mut self);
}

impl<M> SchedulerMirrorTestExt for BlossomVState<M>
where
    M: SparseValuedMatrix2D + ?Sized,
    M::Value: Number + AsPrimitive<i64>,
    M::RowIndex: PositiveInteger,
    M::ColumnIndex: PositiveInteger,
{
    fn mark_tree_roots_processed(&mut self) {
        mark_tree_roots_processed(self);
    }

    fn rebuild_scheduler_tree_mirror(&mut self) {
        rebuild_scheduler_tree_mirror(self);
    }
}

pub(super) trait TestAccessorExt {
    fn test_edge_count(&self) -> usize;
    fn test_edge_slack(&self, e: usize) -> i64;
    fn test_edge_endpoints(&self, e: usize) -> (u32, u32);
    fn test_is_matched(&self, v: usize) -> bool;
    fn test_match_partner(&self, v: usize) -> Option<u32>;
    fn test_tree_num(&self) -> usize;
    fn test_is_tree_root(&self, v: usize) -> bool;
    fn test_flag(&self, v: usize) -> u8;
    fn test_degree(&self, v: usize) -> usize;
    fn test_init_snapshot(&self) -> GreedyInitSnapshot;
}

impl<M> TestAccessorExt for BlossomVState<M>
where
    M: SparseValuedMatrix2D + ?Sized,
    M::Value: Number + AsPrimitive<i64>,
    M::RowIndex: PositiveInteger,
    M::ColumnIndex: PositiveInteger,
{
    fn test_edge_count(&self) -> usize {
        self.edge_num
    }

    fn test_edge_slack(&self, e: usize) -> i64 {
        self.edges[e].slack
    }

    fn test_edge_endpoints(&self, e: usize) -> (u32, u32) {
        (self.edges[e].head0[0], self.edges[e].head0[1])
    }

    fn test_is_matched(&self, v: usize) -> bool {
        self.nodes[v].match_arc != NONE
    }

    fn test_match_partner(&self, v: usize) -> Option<u32> {
        let arc = self.nodes[v].match_arc;
        if arc == NONE {
            return None;
        }
        let e = arc_edge(arc) as usize;
        let dir = arc_dir(arc);
        Some(self.edges[e].head[dir])
    }

    fn test_tree_num(&self) -> usize {
        self.tree_num
    }

    fn test_is_tree_root(&self, v: usize) -> bool {
        self.nodes[v].is_tree_root
    }

    fn test_flag(&self, v: usize) -> u8 {
        self.nodes[v].flag
    }

    fn test_degree(&self, v: usize) -> usize {
        let mut count = 0;
        self.for_each_edge(v as u32, |_, _, _| count += 1);
        count
    }

    fn test_init_snapshot(&self) -> GreedyInitSnapshot {
        self.test_state_snapshot()
    }
}
