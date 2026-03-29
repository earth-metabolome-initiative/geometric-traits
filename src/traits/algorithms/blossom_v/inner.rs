//! Internal implementation of the Blossom V algorithm.
#![allow(
    clippy::bool_to_int_with_if,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::if_not_else,
    clippy::needless_range_loop,
    clippy::question_mark
)]

use alloc::{vec, vec::Vec};

use num_traits::AsPrimitive;

use super::{
    BlossomVError, MatchingResult,
    pairing_heap::{PQKeyStore, PQNode, PairingHeap},
};
use crate::traits::{Number, PositiveInteger, SparseValuedMatrix2D};

#[cfg(test)]
mod test_support;

#[cfg(test)]
use self::test_support::{
    GenericPairQueues, GenericPrimalEvent, GenericPrimalStepTrace, GenericTreeQueues,
    GreedyInitSnapshot, InitGlobalEvent, InitGlobalStepTrace, NodeParitySnapshot,
    StrictParitySnapshot,
};

// ===== Constants =====

/// All costs are doubled internally for half-integrality of dual variables.
const COST_FACTOR: i64 = 2;

/// Sentinel for "no node/edge/tree".
const NONE: u32 = u32::MAX;

/// Node label: "+" (even, outer in tree).
const PLUS: u8 = 0;
/// Node label: "−" (odd, inner in tree).
const MINUS: u8 = 1;
/// Node label: free (not in any tree).
const FREE: u8 = 2;

// ===== Internal data structures =====

/// A node in the graph (either an original vertex or a blossom pseudo-node).
#[derive(Clone)]
struct Node {
    // --- Flags ---
    /// Whether this is an exterior (surface-level) node.
    is_outer: bool,
    /// Label: PLUS, MINUS, or FREE.
    flag: u8,
    /// Whether this node is the root of an alternating tree.
    is_tree_root: bool,
    /// Whether this node's edges have been scanned during tree growth.
    is_processed: bool,
    /// Whether this node is a contracted blossom (pseudo-node).
    is_blossom: bool,

    // --- Core fields ---
    /// Head of edge list for direction 0 and 1. NONE if empty.
    first: [u32; 2],
    /// Matching arc (edge_idx * 2 + dir). NONE if unmatched.
    match_arc: u32,
    /// Dual variable (doubled).
    y: i64,

    // --- Tree fields (when is_outer && flag == PLUS) ---
    /// First child in the alternating tree.
    first_tree_child: u32,
    /// Previous sibling (circular).
    tree_sibling_prev: u32,
    /// Next sibling (NONE-terminated).
    tree_sibling_next: u32,

    // --- Tree field (when is_outer && flag == MINUS) ---
    /// Parent arc in the alternating tree.
    tree_parent_arc: u32,

    // --- Tree membership (when is_outer) ---
    /// Tree root node index. NONE if free.
    tree_root: u32,
    /// Lazy tree epsilon stored on the current tree root.
    tree_eps: i64,

    // --- Blossom fields (when !is_outer) ---
    /// Next node in the blossom cycle (arc = edge_idx * 2 + dir).
    blossom_sibling_arc: u32,
    /// Parent blossom node index.
    blossom_parent: u32,
    /// Snapshot of tree eps when this node was shrunk into a blossom.
    blossom_eps: i64,
    /// Head of the self-loop edge list for this blossom.
    blossom_selfloops: u32,
    /// Path-compression pointer for finding the penultimate blossom ancestor.
    blossom_grandparent: u32,
}

impl Node {
    fn new_vertex() -> Self {
        Self {
            is_outer: true,
            flag: FREE,
            is_tree_root: false,
            is_processed: false,
            is_blossom: false,
            first: [NONE; 2],
            match_arc: NONE,
            y: 0,
            first_tree_child: NONE,
            tree_sibling_prev: NONE,
            tree_sibling_next: NONE,
            tree_parent_arc: NONE,
            tree_root: NONE,
            tree_eps: 0,
            blossom_sibling_arc: NONE,
            blossom_parent: NONE,
            blossom_eps: 0,
            blossom_selfloops: NONE,
            blossom_grandparent: NONE,
        }
    }
}

/// An edge in the graph.
#[derive(Clone)]
struct Edge {
    /// Current endpoints (may change due to blossom contraction).
    head: [u32; 2],
    /// Original endpoints (never change).
    head0: [u32; 2],
    /// Next edge in the circular list for each direction.
    next: [u32; 2],
    /// Previous edge in the circular list for each direction.
    prev: [u32; 2],
    /// Reduced cost (slack), lazily maintained. Stored as `cost*2 - y[i] -
    /// y[j]`.
    slack: i64,
}

impl Edge {
    fn new(u: u32, v: u32, cost: i64) -> Self {
        Self {
            head: [u, v],
            head0: [u, v],
            next: [NONE; 2],
            prev: [NONE; 2],
            slack: cost * COST_FACTOR,
        }
    }
}

impl PQKeyStore for [Edge] {
    #[inline]
    fn get_key(&self, idx: u32) -> i64 {
        self[idx as usize].slack
    }

    #[inline]
    fn set_key(&mut self, idx: u32, key: i64) {
        self[idx as usize].slack = key;
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GenericQueueState {
    None,
    Pq0 { root: u32 },
    Pq00Local { root: u32 },
    Pq00Pair { pair_idx: usize },
    Pq01Pair { pair_idx: usize, dir: usize },
    PqBlossoms { root: u32 },
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
enum SchedulerCurrent {
    #[default]
    None,
    Root,
    Pair {
        pair_idx: usize,
        dir: usize,
    },
}

#[derive(Clone, Debug, Default)]
struct PersistentTreeState {
    root: u32,
    eps: i64,
    first: [Option<usize>; 2],
    current: SchedulerCurrent,
    pq0: Vec<u32>,
    pq0_heap: PairingHeap,
    pq00_local: Vec<u32>,
    pq00_local_heap: PairingHeap,
    pq_blossoms: Vec<u32>,
    pq_blossoms_heap: PairingHeap,
}

#[derive(Clone, Debug, Default)]
struct PersistentTreeEdgeState {
    head: [u32; 2],
    next: [Option<usize>; 2],
    pq00: Vec<u32>,
    pq00_heap: PairingHeap,
    pq01: [Vec<u32>; 2],
    pq01_heap: [PairingHeap; 2],
}

#[derive(Default)]
struct BlossomVScratch {
    incident_edges: Vec<(u32, usize)>,
    queue_edges: Vec<u32>,
    incident_pairs: Vec<(usize, usize)>,
    tree_members_u: Vec<u32>,
    tree_members_v: Vec<u32>,
    members_mask: Vec<bool>,
    node_work_a: Vec<u32>,
    node_work_b: Vec<u32>,
    edge_moves: Vec<(u32, usize, u32)>,
    dual_roots: Vec<u32>,
    dual_seen: Vec<bool>,
    dual_root_to_var: Vec<usize>,
    dual_local_caps: Vec<i64>,
    dual_pair_eps00: Vec<i64>,
    dual_pair_eps01: Vec<i64>,
    dual_deltas: Vec<i64>,
    dual_marks: Vec<usize>,
    dual_queue: Vec<usize>,
    dual_component: Vec<usize>,
}

// ===== Arc encoding =====
// An "arc" is an edge with a direction, encoded as `edge_idx * 2 + dir`.

#[inline]
fn arc_edge(arc: u32) -> u32 {
    arc >> 1
}

#[inline]
fn arc_dir(arc: u32) -> usize {
    (arc & 1) as usize
}

#[inline]
fn make_arc(edge_idx: u32, dir: usize) -> u32 {
    edge_idx * 2 + dir as u32
}

#[inline]
fn arc_rev(arc: u32) -> u32 {
    arc ^ 1
}

#[cfg(test)]
#[inline]
fn normalized_edge_pair(head: [u32; 2]) -> (u32, u32) {
    if head[0] < head[1] { (head[0], head[1]) } else { (head[1], head[0]) }
}

// ===== Edge list operations =====

/// Add edge `e` to node `node`'s adjacency list for direction `dir`.
fn edge_list_add(nodes: &mut [Node], edges: &mut [Edge], node: u32, e: u32, dir: usize) {
    let nu = node as usize;
    let eu = e as usize;
    if nodes[nu].first[dir] != NONE {
        let first = nodes[nu].first[dir];
        let fu = first as usize;
        let last = edges[fu].prev[dir];
        edges[eu].prev[dir] = last;
        edges[eu].next[dir] = first;
        edges[last as usize].next[dir] = e;
        edges[fu].prev[dir] = e;
    } else {
        nodes[nu].first[dir] = e;
        edges[eu].prev[dir] = e;
        edges[eu].next[dir] = e;
    }
    edges[eu].head[1 - dir] = node;
}

/// Remove edge `e` from node `node`'s adjacency list for direction `dir`.
fn edge_list_remove(nodes: &mut [Node], edges: &mut [Edge], node: u32, e: u32, dir: usize) {
    let nu = node as usize;
    let eu = e as usize;
    if edges[eu].prev[dir] == e {
        // Only edge in the list
        nodes[nu].first[dir] = NONE;
    } else {
        let p = edges[eu].prev[dir];
        let n = edges[eu].next[dir];
        assert_ne!(
            p, NONE,
            "edge_list_remove: prev is NONE for node={node} edge={e} dir={dir} head={:?} prev={:?} next={:?}",
            edges[eu].head, edges[eu].prev, edges[eu].next
        );
        assert_ne!(
            n, NONE,
            "edge_list_remove: next is NONE for node={node} edge={e} dir={dir} head={:?} prev={:?} next={:?}",
            edges[eu].head, edges[eu].prev, edges[eu].next
        );
        edges[p as usize].next[dir] = n;
        edges[n as usize].prev[dir] = p;
        // Match Blossom V's REMOVE_EDGE macro: advance the owner head to the
        // removed edge's successor even when `e` was not the recorded `first`.
        // Any element in the circular list is a valid head, and this keeps the
        // list walk stable across repeated removals/reinsertions.
        nodes[nu].first[dir] = n;
    }
}

// ===== Main state =====

/// Internal state for the Blossom V algorithm.
pub(super) struct BlossomVState<M: SparseValuedMatrix2D + ?Sized> {
    _marker: core::marker::PhantomData<fn() -> M>,
    node_num: usize,
    edge_num: usize,
    nodes: Vec<Node>,
    edges: Vec<Edge>,
    edge_queue_owner: Vec<GenericQueueState>,
    edge_queue_stamp: Vec<u64>,
    pq_nodes: Vec<PQNode>,
    root_list_head: u32,
    #[cfg(test)]
    generic_trees: Vec<GenericTreeQueues>,
    #[cfg(test)]
    generic_pairs: Vec<GenericPairQueues>,
    scheduler_trees: Vec<PersistentTreeState>,
    scheduler_tree_edges: Vec<PersistentTreeEdgeState>,
    scratch: BlossomVScratch,
    generic_queue_epoch: u64,
    tree_num: usize,
    blossom_count: usize,
    #[cfg(test)]
    init_global_trace: Vec<InitGlobalEvent>,
    #[cfg(test)]
    init_global_steps: Vec<InitGlobalStepTrace>,
    #[cfg(test)]
    generic_primal_steps: Vec<GenericPrimalStepTrace>,
}

impl<M: SparseValuedMatrix2D + ?Sized> BlossomVState<M>
where
    M::Value: Number + AsPrimitive<i64>,
    M::RowIndex: PositiveInteger,
    M::ColumnIndex: PositiveInteger,
{
    #[cfg(feature = "std")]
    #[cold]
    #[inline(never)]
    fn maybe_write_debug_trace_snapshot(&self, op: &str) {
        use std::io::Write as _;

        let Some(path) = std::env::var_os("BLOSSOM_V_DEBUG_TRACE_FILE") else {
            return;
        };
        let Ok(mut file) = std::fs::OpenOptions::new().create(true).append(true).open(path) else {
            return;
        };

        let y = (0..self.node_num).map(|v| self.nodes[v].y).collect::<Vec<_>>();
        let slacks = (0..self.edge_num).map(|e| self.edges[e].slack).collect::<Vec<_>>();
        let matching = (0..self.node_num)
            .map(|v| {
                let arc = self.nodes[v].match_arc;
                if arc == NONE {
                    -1
                } else {
                    self.edges[arc_edge(arc) as usize].head[arc_dir(arc)] as i32
                }
            })
            .collect::<Vec<_>>();
        let flags = (0..self.node_num).map(|v| self.nodes[v].flag).collect::<Vec<_>>();
        let is_outer =
            (0..self.node_num).map(|v| u8::from(self.nodes[v].is_outer)).collect::<Vec<_>>();

        let _ = writeln!(
            file,
            "{{\"op\":\"{}\",\"y\":{:?},\"slacks\":{:?},\"match\":{:?},\"flags\":{:?},\"is_outer\":{:?},\"tree_num\":{}}}",
            op, y, slacks, matching, flags, is_outer, self.tree_num
        );
    }

    #[cfg(feature = "std")]
    #[cold]
    #[inline(never)]
    fn maybe_write_debug_queue_summary(&self, label: &str) {
        use std::io::Write as _;

        let Some(path) = std::env::var_os("BLOSSOM_V_DEBUG_TRACE_FILE") else {
            return;
        };
        let Ok(mut file) = std::fs::OpenOptions::new().create(true).append(true).open(path) else {
            return;
        };

        let roots = self.current_root_list();
        let _ = writeln!(file, "# {label} roots={roots:?}");
        for &root in &roots {
            if (root as usize) >= self.scheduler_trees.len() {
                continue;
            }
            let tree = &self.scheduler_trees[root as usize];
            let collect_tree_edges = |dir| {
                let mut pair_edges = Vec::new();
                let mut cursor = tree.first[dir];
                let mut safety = self.scheduler_tree_edges.len() + 1;
                while let Some(pair_idx) = cursor {
                    if pair_idx >= self.scheduler_tree_edges.len() || safety == 0 {
                        break;
                    }
                    if self.scheduler_tree_edge_dir(pair_idx, root) == Some(dir) {
                        pair_edges.push(pair_idx);
                    }
                    cursor = self.scheduler_tree_edges[pair_idx].next[dir];
                    safety -= 1;
                }
                pair_edges
            };
            let _ = writeln!(
                file,
                "# root {} pq0={:?} pq00_local={:?} pq_blossoms={:?} tree_edges0={:?} tree_edges1={:?}",
                root,
                tree.pq0,
                tree.pq00_local,
                tree.pq_blossoms,
                collect_tree_edges(0),
                collect_tree_edges(1)
            );
        }
        for (pair_idx, pair) in self.scheduler_tree_edges.iter().enumerate() {
            if pair.head[0] == NONE && pair.head[1] == NONE {
                continue;
            }
            if pair.pq00.is_empty() && pair.pq01[0].is_empty() && pair.pq01[1].is_empty() {
                continue;
            }
            let _ = writeln!(
                file,
                "# pair {} head={:?} pq00={:?} pq01_0={:?} pq01_1={:?}",
                pair_idx, pair.head, pair.pq00, pair.pq01[0], pair.pq01[1]
            );
        }
    }

    #[cfg(not(feature = "std"))]
    #[cold]
    #[inline(never)]
    fn maybe_write_debug_queue_summary(&self, _label: &str) {}

    #[cfg(not(feature = "std"))]
    #[cold]
    #[inline(never)]
    fn maybe_write_debug_trace_snapshot(&self, _op: &str) {}

    pub(super) fn new(matrix: &M) -> Self {
        let n: usize = matrix.number_of_rows().as_();

        // Count edges (upper triangle only for symmetric matrix)
        let mut edge_count = 0usize;
        for i in matrix.row_indices() {
            let iu: usize = i.as_();
            for j in matrix.sparse_row(i) {
                let ju: usize = j.as_();
                if ju > iu {
                    edge_count += 1;
                }
            }
        }

        let mut nodes: Vec<Node> = (0..n).map(|_| Node::new_vertex()).collect();
        let mut edges: Vec<Edge> = Vec::with_capacity(edge_count);
        let pq_nodes: Vec<PQNode> = vec![PQNode::RESET; edge_count];

        // Build edges and adjacency lists
        for i in matrix.row_indices() {
            let iu: usize = i.as_();
            for (j, val) in matrix.sparse_row(i).zip(matrix.sparse_row_values(i)) {
                let ju: usize = j.as_();
                if ju <= iu {
                    continue;
                }
                let cost: i64 = val.as_();
                let e_idx = edges.len() as u32;
                edges.push(Edge::new(iu as u32, ju as u32, cost));
                // direction 0: edge seen from node iu (tail=iu, head=ju)
                edge_list_add(&mut nodes, &mut edges, iu as u32, e_idx, 0);
                // direction 1: edge seen from node ju (tail=ju, head=iu)
                edge_list_add(&mut nodes, &mut edges, ju as u32, e_idx, 1);
            }
        }

        let edge_num = edges.len();
        let mut state = Self {
            _marker: core::marker::PhantomData,
            node_num: n,
            edge_num,
            nodes,
            edges,
            edge_queue_owner: vec![GenericQueueState::None; edge_num],
            edge_queue_stamp: vec![0; edge_num],
            pq_nodes,
            root_list_head: NONE,
            #[cfg(test)]
            generic_trees: vec![GenericTreeQueues::default(); n],
            #[cfg(test)]
            generic_pairs: Vec::new(),
            scheduler_trees: Vec::new(),
            scheduler_tree_edges: Vec::new(),
            scratch: BlossomVScratch::default(),
            generic_queue_epoch: 0,
            tree_num: 0,
            blossom_count: 0,
            #[cfg(test)]
            init_global_trace: Vec::new(),
            #[cfg(test)]
            init_global_steps: Vec::new(),
            #[cfg(test)]
            generic_primal_steps: Vec::new(),
        };

        if n > 0 {
            state.init_greedy();
        }

        state
    }

    /// Greedy initialization: set initial dual variables and matching.
    ///
    /// For each node v, set y[v] = min incident edge cost (doubled).
    /// Then greedily match on zero-slack edges.
    fn init_greedy(&mut self) {
        let n = self.node_num;

        // Start with all nodes unmatched. During greedy init, matched nodes
        // are marked FREE while unmatched nodes remain PLUS and later become
        // tree roots, mirroring Blossom V's InitGreedy().
        for v in 0..n {
            self.nodes[v].flag = PLUS;
            self.nodes[v].is_tree_root = false;
            self.nodes[v].tree_eps = 0;
            self.nodes[v].is_processed = false;
            self.nodes[v].match_arc = NONE;
            self.nodes[v].tree_root = NONE;
            self.nodes[v].tree_parent_arc = NONE;
            self.nodes[v].first_tree_child = NONE;
            self.nodes[v].tree_sibling_prev = NONE;
            self.nodes[v].tree_sibling_next = NONE;
            self.nodes[v].y = i64::MAX;
        }

        // Step 1: Set y[v] = min(cost * COST_FACTOR) / 2 over incident edges.
        for v in 0..n {
            let mut min_cost = i64::MAX;
            self.for_each_edge(v as u32, |_e_idx, _dir, edge| {
                if edge.slack < min_cost {
                    min_cost = edge.slack;
                }
            });
            if min_cost == i64::MAX {
                self.nodes[v].y = 0;
            } else {
                // With doubled costs, halving the minimum incident slack gives
                // the initial feasible dual for the node.
                self.nodes[v].y = min_cost / 2;
            }
        }

        // Step 2: Update slacks: slack(e) = cost*2 - y[u] - y[v].
        // Since slack was initialized as cost*2, subtract both endpoint duals.
        for e in 0..self.edge_num {
            let u = self.edges[e].head0[0] as usize;
            let v = self.edges[e].head0[1] as usize;
            self.edges[e].slack -= self.nodes[u].y + self.nodes[v].y;
        }

        // Step 3: Process nodes sequentially. Raise y[v] by the minimum
        // incident slack, match any edge whose slack is at most that minimum,
        // then subtract the minimum from all incident edge slacks.
        for v in 0..n {
            if self.nodes[v].flag == FREE {
                continue;
            }

            let mut incident: Vec<(u32, usize, u32)> = Vec::new();
            let mut min_slack = i64::MAX;
            self.for_each_edge(v as u32, |e_idx, dir, edge| {
                if edge.slack < min_slack {
                    min_slack = edge.slack;
                }
                incident.push((e_idx, dir, edge.head[dir]));
            });
            if min_slack == i64::MAX {
                continue;
            }

            self.nodes[v].y += min_slack;

            for &(e_idx, dir, other) in &incident {
                if self.edges[e_idx as usize].slack <= min_slack
                    && self.nodes[v].flag == PLUS
                    && self.nodes[other as usize].flag == PLUS
                {
                    self.nodes[v].flag = FREE;
                    self.nodes[other as usize].flag = FREE;
                    self.nodes[v].match_arc = make_arc(e_idx, dir);
                    self.nodes[other as usize].match_arc = make_arc(e_idx, 1 - dir);
                }
            }

            for &(e_idx, _dir, _) in &incident {
                self.edges[e_idx as usize].slack -= min_slack;
            }
        }

        // Step 4: Create trees for unmatched nodes.
        self.tree_num = 0;
        self.root_list_head = NONE;
        for v in 0..n {
            if self.nodes[v].flag != FREE {
                self.nodes[v].is_tree_root = true;
                self.nodes[v].tree_eps = 0;
                self.nodes[v].flag = PLUS;
                self.nodes[v].tree_root = v as u32;
                self.root_list_append(v as u32);
                self.tree_num += 1;
            }
        }
    }

    /// Startup phase ported from Blossom V's `InitGlobal()`.
    fn init_global(&mut self) {
        #[cfg(test)]
        {
            self.init_global_trace.clear();
            self.init_global_steps.clear();
        }

        if self.tree_num == 0 {
            return;
        }

        self.init_global_ported();
    }

    fn init_global_ported(&mut self) {
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
    fn init_global_process_root(&mut self, root: u32, best_edge: &mut Vec<u32>) -> bool {
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

    fn generic_primal_pass_once(&mut self) -> bool {
        let mut root = self.root_list_head;
        while root != NONE {
            let root_usize = root as usize;
            let next_root = self.nodes[root_usize].tree_sibling_next;
            let next_next_root = if next_root != NONE {
                self.nodes[next_root as usize].tree_sibling_next
            } else {
                NONE
            };

            if self.nodes[root_usize].is_outer
                && self.nodes[root_usize].is_tree_root
                && self.process_tree_primal(root)
            {
                return true;
            }

            root = next_root;
            if root != NONE && !self.nodes[root as usize].is_tree_root {
                root = next_next_root;
            }
        }

        if let Some(blossom) = self.find_global_expand_fallback_blossom() {
            self.apply_generic_expand(blossom);
            return true;
        }

        false
    }

    fn process_tree_primal(&mut self, root: u32) -> bool {
        if root == NONE
            || root as usize >= self.nodes.len()
            || !self.nodes[root as usize].is_outer
            || !self.nodes[root as usize].is_tree_root
        {
            return false;
        }
        self.ensure_scheduler_tree_slot(root);

        if let Some((e_idx, left, right)) = self.find_tree_step1_augment_edge_from_scheduler(root) {
            self.apply_generic_augment(e_idx, left, right);
            return true;
        }

        let mut progressed = false;
        let tree_num0 = self.tree_num;

        while self.tree_num == tree_num0
            && (root as usize) < self.nodes.len()
            && self.nodes[root as usize].is_outer
            && self.nodes[root as usize].is_tree_root
        {
            let eps_root = self.scheduler_trees[root as usize].eps;
            let shrink_cap = eps_root.saturating_mul(2);

            if let Some((e_idx, plus, free)) = self.find_tree_grow_edge_with_eps(root, eps_root) {
                self.apply_generic_grow(e_idx, plus, free);
                progressed = true;
                continue;
            }

            if let Some((e_idx, _left, _right)) =
                self.find_tree_shrink_edge_with_cap(root, shrink_cap)
            {
                self.clear_generic_queue_state(e_idx);
                if self.process_edge00(e_idx, true) {
                    let left = self.edge_head_outer(e_idx, 0);
                    let right = self.edge_head_outer(e_idx, 1);
                    if left != NONE && right != NONE && left != right {
                        self.apply_generic_shrink(e_idx, left, right);
                    }
                }
                progressed = true;
                continue;
            }

            if let Some(blossom) = self.find_tree_expand_blossom_with_eps(root, eps_root) {
                self.apply_generic_expand(blossom);
                progressed = true;
                continue;
            }

            break;
        }

        if self.tree_num == tree_num0 {
            self.clear_generic_tree_currents_local(root);
        }
        progressed
    }

    fn scheduler_tree_best_pq0_edge(&self, root: u32) -> Option<u32> {
        if (root as usize) >= self.scheduler_trees.len() {
            return None;
        }
        let mut best = None;
        let mut best_slack = i64::MAX;
        let mut best_stamp = 0u64;
        for &e_idx in &self.scheduler_trees[root as usize].pq0 {
            if (e_idx as usize) >= self.edge_num {
                continue;
            }
            if !matches!(self.edge_queue_owner(e_idx), GenericQueueState::Pq0 { root: q_root } if q_root == root)
            {
                continue;
            }
            let slack = self.edges[e_idx as usize].slack;
            let stamp = self.edge_queue_stamp(e_idx);
            if best.is_none() || slack < best_slack || (slack == best_slack && stamp > best_stamp) {
                best = Some(e_idx);
                best_slack = slack;
                best_stamp = stamp;
            }
        }
        best
    }

    fn scheduler_tree_best_pq_blossom_edge(&self, root: u32) -> Option<u32> {
        if (root as usize) >= self.scheduler_trees.len() {
            return None;
        }
        let mut best = None;
        let mut best_slack = i64::MAX;
        let mut best_stamp = 0u64;
        for &e_idx in &self.scheduler_trees[root as usize].pq_blossoms {
            if (e_idx as usize) >= self.edge_num {
                continue;
            }
            if !matches!(self.edge_queue_owner(e_idx), GenericQueueState::PqBlossoms { root: q_root } if q_root == root)
            {
                continue;
            }
            let slack = self.edges[e_idx as usize].slack;
            let stamp = self.edge_queue_stamp(e_idx);
            if best.is_none() || slack < best_slack || (slack == best_slack && stamp > best_stamp) {
                best = Some(e_idx);
                best_slack = slack;
                best_stamp = stamp;
            }
        }
        best
    }

    fn tree_min_pq00_local_for_step3(&mut self, root: u32) -> Option<(u32, u32, u32, i64)> {
        if (root as usize) >= self.scheduler_trees.len() {
            return None;
        }

        loop {
            let Some(e_idx) = self.scheduler_trees[root as usize].pq00_local_heap.get_min() else {
                return None;
            };
            if (e_idx as usize) >= self.edge_num {
                self.remove_edge_from_generic_queue(e_idx);
                continue;
            }

            if !self.process_edge00(e_idx, false) {
                self.remove_edge_from_generic_queue(e_idx);
                continue;
            }

            let left = self.edge_head_outer(e_idx, 0);
            let right = self.edge_head_outer(e_idx, 1);
            if left == NONE
                || right == NONE
                || left == right
                || !self.nodes[left as usize].is_outer
                || !self.nodes[right as usize].is_outer
                || self.nodes[left as usize].flag != PLUS
                || self.nodes[right as usize].flag != PLUS
                || !self.nodes[left as usize].is_processed
                || !self.nodes[right as usize].is_processed
                || self.find_tree_root(left) != root
                || self.find_tree_root(right) != root
            {
                self.remove_edge_from_generic_queue(e_idx);
                continue;
            }

            return Some((e_idx, left, right, self.edges[e_idx as usize].slack));
        }
    }

    fn scheduler_tree_best_pq00_local_edge(&self, root: u32) -> Option<u32> {
        if (root as usize) >= self.scheduler_trees.len() {
            return None;
        }
        let mut best = None;
        let mut best_slack = i64::MAX;
        let mut best_stamp = 0u64;
        for &e_idx in &self.scheduler_trees[root as usize].pq00_local {
            if (e_idx as usize) >= self.edge_num {
                continue;
            }
            if !matches!(self.edge_queue_owner(e_idx), GenericQueueState::Pq00Local { root: q_root } if q_root == root)
            {
                continue;
            }
            let slack = self.edges[e_idx as usize].slack;
            let stamp = self.edge_queue_stamp(e_idx);
            if best.is_none() || slack < best_slack || (slack == best_slack && stamp > best_stamp) {
                best = Some(e_idx);
                best_slack = slack;
                best_stamp = stamp;
            }
        }
        best
    }

    fn scheduler_tree_edge_best_pq00_edge(
        &self,
        pair_idx: usize,
        root: u32,
        other_root: u32,
    ) -> Option<u32> {
        if pair_idx >= self.scheduler_tree_edges.len() {
            return None;
        }
        let mut best = None;
        let mut best_slack = i64::MAX;
        let mut best_stamp = 0u64;
        for &e_idx in &self.scheduler_tree_edges[pair_idx].pq00 {
            if (e_idx as usize) >= self.edge_num {
                continue;
            }
            let left = self.edge_head_outer(e_idx, 0);
            let right = self.edge_head_outer(e_idx, 1);
            if left == NONE
                || right == NONE
                || left == right
                || !self.nodes[left as usize].is_outer
                || !self.nodes[right as usize].is_outer
                || self.nodes[left as usize].flag != PLUS
                || self.nodes[right as usize].flag != PLUS
            {
                continue;
            }
            let root_left = self.find_tree_root(left);
            let root_right = self.find_tree_root(right);
            let matches_pair = (root_left == root && root_right == other_root)
                || (root_left == other_root && root_right == root);
            if !matches_pair {
                continue;
            }
            let slack = self.edges[e_idx as usize].slack;
            let stamp = self.edge_queue_stamp(e_idx);
            if best.is_none() || slack < best_slack || (slack == best_slack && stamp > best_stamp) {
                best = Some(e_idx);
                best_slack = slack;
                best_stamp = stamp;
            }
        }
        best
    }

    fn find_tree_step1_augment_edge_from_scheduler(
        &mut self,
        root: u32,
    ) -> Option<(u32, u32, u32)> {
        if (root as usize) >= self.scheduler_trees.len() {
            return None;
        }
        if (root as usize) < self.scheduler_trees.len() {
            self.scheduler_trees[root as usize].current = SchedulerCurrent::Root;
        }
        let eps_root = self.tree_eps(root);

        for dir in 0..2usize {
            let mut pair_cursor = self.scheduler_trees[root as usize].first[dir];
            while let Some(pair_idx) = pair_cursor {
                let next_pair = self.scheduler_tree_edges[pair_idx].next[dir];
                let Some(other_root) = self.scheduler_tree_edge_other(pair_idx, root) else {
                    pair_cursor = next_pair;
                    continue;
                };
                if other_root == NONE
                    || (other_root as usize) >= self.nodes.len()
                    || !self.nodes[other_root as usize].is_outer
                    || !self.nodes[other_root as usize].is_tree_root
                {
                    pair_cursor = next_pair;
                    continue;
                }
                if (other_root as usize) < self.scheduler_trees.len() {
                    self.scheduler_trees[other_root as usize].current =
                        SchedulerCurrent::Pair { pair_idx, dir };
                }

                if let Some(e_idx) =
                    self.scheduler_tree_edge_best_pq00_edge(pair_idx, root, other_root)
                {
                    let left = self.edge_head_outer(e_idx, 0);
                    let right = self.edge_head_outer(e_idx, 1);
                    let slack = self.edges[e_idx as usize].slack;
                    if slack - eps_root <= self.tree_eps(other_root) {
                        return Some((e_idx, left, right));
                    }
                }

                pair_cursor = next_pair;
            }
        }

        None
    }

    fn find_tree_expand_blossom_with_eps(&mut self, root: u32, eps_root: i64) -> Option<u32> {
        if (root as usize) >= self.scheduler_trees.len() {
            return None;
        }
        loop {
            let Some(e_idx) = self.scheduler_tree_best_pq_blossom_edge(root) else {
                return None;
            };
            if (e_idx as usize) >= self.edge_num {
                self.remove_edge_from_generic_queue(e_idx);
                continue;
            }
            let slack = self.edges[e_idx as usize].slack;
            if slack > eps_root {
                return None;
            }

            let mut blossom = NONE;
            for dir in 0..2usize {
                let cand = self.edge_head_outer(e_idx, dir);
                if cand == NONE {
                    continue;
                }
                if self.nodes[cand as usize].is_blossom
                    && self.nodes[cand as usize].is_outer
                    && self.nodes[cand as usize].flag == MINUS
                    && self.find_tree_root(cand) == root
                {
                    blossom = cand;
                    break;
                }
            }
            if blossom == NONE {
                self.remove_edge_from_generic_queue(e_idx);
                continue;
            }
            return Some(blossom);
        }
    }

    fn find_tree_grow_edge_with_eps(
        &mut self,
        root: u32,
        eps_root: i64,
    ) -> Option<(u32, u32, u32)> {
        if (root as usize) >= self.scheduler_trees.len() {
            return None;
        }
        loop {
            let Some(e_idx) = self.scheduler_tree_best_pq0_edge(root) else {
                return None;
            };
            let slack = self.edges[e_idx as usize].slack;
            if slack > eps_root {
                return None;
            }

            for &(plus, free) in &[
                (self.edge_head_outer(e_idx, 1), self.edge_head_outer(e_idx, 0)),
                (self.edge_head_outer(e_idx, 0), self.edge_head_outer(e_idx, 1)),
            ] {
                if plus == NONE
                    || free == NONE
                    || plus == free
                    || !self.nodes[plus as usize].is_outer
                    || self.nodes[plus as usize].flag != PLUS
                    || self.find_tree_root(plus) != root
                    || !self.nodes[free as usize].is_outer
                    || self.nodes[free as usize].flag != FREE
                {
                    continue;
                }
                return Some((e_idx, plus, free));
            }
            self.remove_edge_from_generic_queue(e_idx);
        }
    }

    #[cfg(test)]
    fn find_tree_grow_edge(&mut self, root: u32) -> Option<(u32, u32, u32)> {
        self::test_support::rebuild_scheduler_tree_mirror(self);
        let eps_root = self.tree_eps(root);
        self.find_tree_grow_edge_with_eps(root, eps_root)
    }

    fn find_tree_shrink_edge_with_cap(
        &mut self,
        root: u32,
        shrink_cap: i64,
    ) -> Option<(u32, u32, u32)> {
        if (root as usize) >= self.scheduler_trees.len() {
            return None;
        }

        loop {
            let Some(e_idx) = self.scheduler_tree_best_pq00_local_edge(root) else {
                return None;
            };
            if (e_idx as usize) >= self.edge_num {
                self.remove_edge_from_generic_queue(e_idx);
                continue;
            }

            let slack = self.edges[e_idx as usize].slack;
            if slack > shrink_cap {
                return None;
            }
            return Some((e_idx, NONE, NONE));
        }
    }

    #[cfg(test)]
    fn find_scheduler_global_augment_edge(&self) -> Option<(u32, u32, u32)> {
        let mut best: Option<(i64, u32, u32, u32)> = None;
        let mut root = self.root_list_head;
        let mut seen = vec![false; self.nodes.len()];
        while root != NONE && (root as usize) < self.nodes.len() && !seen[root as usize] {
            seen[root as usize] = true;
            if self.nodes[root as usize].is_outer && self.nodes[root as usize].is_tree_root {
                let eps_root = self.tree_eps(root);
                for dir in 0..2usize {
                    let mut pair_cursor = if (root as usize) < self.scheduler_trees.len() {
                        self.scheduler_trees[root as usize].first[dir]
                    } else {
                        None
                    };
                    while let Some(pair_idx) = pair_cursor {
                        let next_pair = self.scheduler_tree_edges[pair_idx].next[dir];
                        let Some(other_root) = self.scheduler_tree_edge_other(pair_idx, root)
                        else {
                            pair_cursor = next_pair;
                            continue;
                        };
                        if other_root == NONE
                            || (other_root as usize) >= self.nodes.len()
                            || !self.nodes[other_root as usize].is_outer
                            || !self.nodes[other_root as usize].is_tree_root
                        {
                            pair_cursor = next_pair;
                            continue;
                        }
                        if let Some(e_idx) =
                            self.scheduler_tree_edge_best_pq00_edge(pair_idx, root, other_root)
                        {
                            let eps_other = self.tree_eps(other_root);
                            let slack = self.edges[e_idx as usize].slack;
                            let adjusted = slack - eps_other;
                            if adjusted <= eps_root {
                                let left = self.edge_head_outer(e_idx, 0);
                                let right = self.edge_head_outer(e_idx, 1);
                                match best {
                                    Some((best_adjusted, best_edge, _, _))
                                        if best_adjusted < adjusted
                                            || (best_adjusted == adjusted && best_edge < e_idx) => {
                                    }
                                    _ => best = Some((adjusted, e_idx, left, right)),
                                }
                            }
                        }
                        pair_cursor = next_pair;
                    }
                }
            }
            root = self.nodes[root as usize].tree_sibling_next;
        }
        best.map(|(_, e_idx, left, right)| (e_idx, left, right))
    }

    fn compute_tree_local_eps(&self, root: u32) -> i64 {
        self.compute_tree_local_eps_for_virtual_dual(root)
    }

    fn compute_tree_local_eps_for_virtual_dual(&self, root: u32) -> i64 {
        let mut eps = i64::MAX;

        for e_idx in 0..self.edge_num {
            let u = self.edge_head_outer(e_idx as u32, 0);
            let v = self.edge_head_outer(e_idx as u32, 1);
            if u == NONE || v == NONE || u == v {
                continue;
            }

            let slack = self.edges[e_idx].slack;
            if self.nodes[u as usize].flag == PLUS
                && self.nodes[u as usize].is_processed
                && self.find_tree_root(u) == root
                && self.nodes[v as usize].flag == FREE
            {
                eps = eps.min(slack);
            }
            if self.nodes[v as usize].flag == PLUS
                && self.nodes[v as usize].is_processed
                && self.find_tree_root(v) == root
                && self.nodes[u as usize].flag == FREE
            {
                eps = eps.min(slack);
            }
        }

        for i in 0..self.nodes.len() {
            let node = &self.nodes[i];
            if node.is_blossom
                && node.is_outer
                && node.flag == MINUS
                && self.find_tree_root(i as u32) == root
            {
                let candidate = if node.match_arc != NONE
                    && (arc_edge(node.match_arc) as usize) < self.edge_num
                {
                    self.edges[arc_edge(node.match_arc) as usize].slack
                } else {
                    node.y
                };
                eps = eps.min(candidate);
            }
        }

        eps
    }

    fn find_virtual_dual_augment_edge(&self) -> Option<(u32, u32, u32)> {
        let mut tree_eps = vec![i64::MAX; self.nodes.len()];
        for root in 0..self.nodes.len() {
            if self.nodes[root].is_outer && self.nodes[root].is_tree_root {
                tree_eps[root] = self.compute_tree_local_eps_for_virtual_dual(root as u32);
            }
        }

        let mut best: Option<(i64, u32, u32, u32)> = None;
        for root in 0..self.nodes.len() {
            if !self.nodes[root].is_outer || !self.nodes[root].is_tree_root {
                continue;
            }

            self.for_each_tree_plus(root as u32, |plus| {
                let root_plus = self.find_tree_root(plus);
                if root_plus == NONE {
                    return None::<()>;
                }
                let eps_plus = tree_eps[root_plus as usize];
                for (e_idx, dir) in self.incident_edges(plus) {
                    let slack = self.edges[e_idx as usize].slack;
                    let other = self.edge_head_outer(e_idx, dir);
                    if other == NONE || other == plus || !self.nodes[other as usize].is_outer {
                        continue;
                    }
                    if self.nodes[other as usize].flag != PLUS {
                        continue;
                    }

                    let root_other = self.find_tree_root(other);
                    if root_other == root_plus || root_other == NONE {
                        continue;
                    }

                    let eps_other = tree_eps[root_other as usize];
                    let can_tighten = eps_plus == i64::MAX
                        || eps_other == i64::MAX
                        || slack <= eps_plus + eps_other;
                    if can_tighten {
                        match best {
                            Some((best_slack, _, _, _)) if best_slack <= slack => {}
                            _ => best = Some((slack, e_idx, plus, other)),
                        }
                    }
                }

                None::<()>
            });
        }

        best.map(|(_, e_idx, plus, other)| (e_idx, plus, other))
    }

    fn for_each_tree_plus<T>(&self, root: u32, mut f: impl FnMut(u32) -> Option<T>) -> Option<T> {
        let mut current = root;
        loop {
            if let Some(value) = f(current) {
                return Some(value);
            }
            match self.next_tree_plus(current, root) {
                Some(next) => current = next,
                None => return None,
            }
        }
    }

    /// Iterate edges of node `v` (both directions), calling `f(edge_idx,
    /// head_dir, &edge)`.
    fn for_each_edge(&self, v: u32, mut f: impl FnMut(u32, usize, &Edge)) {
        for start_dir in 0..2usize {
            let first = self.nodes[v as usize].first[start_dir];
            if first == NONE {
                continue;
            }
            let mut e = first;
            let mut safety = self.edge_num + 1;
            loop {
                f(e, start_dir, &self.edges[e as usize]);
                e = self.edges[e as usize].next[start_dir];
                safety -= 1;
                if e == first || safety == 0 {
                    break;
                }
            }
        }
    }

    #[inline]
    fn collect_incident_edges_into(&self, v: u32, out: &mut Vec<(u32, usize)>) {
        out.clear();
        self.for_each_edge(v, |e_idx, dir, _| out.push((e_idx, dir)));
    }

    #[inline]
    fn take_incident_scratch(&mut self) -> Vec<(u32, usize)> {
        core::mem::take(&mut self.scratch.incident_edges)
    }

    #[inline]
    fn restore_incident_scratch(&mut self, mut scratch: Vec<(u32, usize)>) {
        scratch.clear();
        self.scratch.incident_edges = scratch;
    }

    #[inline]
    fn take_queue_edges_scratch(&mut self) -> Vec<u32> {
        core::mem::take(&mut self.scratch.queue_edges)
    }

    #[inline]
    fn restore_queue_edges_scratch(&mut self, mut scratch: Vec<u32>) {
        scratch.clear();
        self.scratch.queue_edges = scratch;
    }

    #[inline]
    fn take_incident_pairs_scratch(&mut self) -> Vec<(usize, usize)> {
        core::mem::take(&mut self.scratch.incident_pairs)
    }

    #[inline]
    fn restore_incident_pairs_scratch(&mut self, mut scratch: Vec<(usize, usize)>) {
        scratch.clear();
        self.scratch.incident_pairs = scratch;
    }

    #[inline]
    fn take_tree_members_u_scratch(&mut self) -> Vec<u32> {
        core::mem::take(&mut self.scratch.tree_members_u)
    }

    #[inline]
    fn restore_tree_members_u_scratch(&mut self, mut scratch: Vec<u32>) {
        scratch.clear();
        self.scratch.tree_members_u = scratch;
    }

    #[inline]
    fn take_tree_members_v_scratch(&mut self) -> Vec<u32> {
        core::mem::take(&mut self.scratch.tree_members_v)
    }

    #[inline]
    fn restore_tree_members_v_scratch(&mut self, mut scratch: Vec<u32>) {
        scratch.clear();
        self.scratch.tree_members_v = scratch;
    }

    #[inline]
    fn take_members_mask_scratch(&mut self) -> Vec<bool> {
        core::mem::take(&mut self.scratch.members_mask)
    }

    #[inline]
    fn restore_members_mask_scratch(&mut self, mut scratch: Vec<bool>) {
        scratch.clear();
        self.scratch.members_mask = scratch;
    }

    #[inline]
    fn take_node_work_a_scratch(&mut self) -> Vec<u32> {
        core::mem::take(&mut self.scratch.node_work_a)
    }

    #[inline]
    fn restore_node_work_a_scratch(&mut self, mut scratch: Vec<u32>) {
        scratch.clear();
        self.scratch.node_work_a = scratch;
    }

    #[inline]
    fn take_node_work_b_scratch(&mut self) -> Vec<u32> {
        core::mem::take(&mut self.scratch.node_work_b)
    }

    #[inline]
    fn restore_node_work_b_scratch(&mut self, mut scratch: Vec<u32>) {
        scratch.clear();
        self.scratch.node_work_b = scratch;
    }

    #[inline]
    fn take_edge_moves_scratch(&mut self) -> Vec<(u32, usize, u32)> {
        core::mem::take(&mut self.scratch.edge_moves)
    }

    #[inline]
    fn restore_edge_moves_scratch(&mut self, mut scratch: Vec<(u32, usize, u32)>) {
        scratch.clear();
        self.scratch.edge_moves = scratch;
    }

    fn incident_edges(&self, v: u32) -> Vec<(u32, usize)> {
        let mut incident = Vec::new();
        self.for_each_edge(v, |e_idx, dir, _| incident.push((e_idx, dir)));
        incident
    }

    fn raw_incident_edges(&self, v: u32) -> Vec<(u32, usize)> {
        let mut incident = Vec::new();
        for (e_idx, edge) in self.edges.iter().take(self.edge_num).enumerate() {
            for dir in 0..2usize {
                if edge.head[dir] == v {
                    incident.push((e_idx as u32, dir));
                }
            }
        }
        incident
    }

    fn process_expand_selfloop(&mut self, e_idx: u32) {
        let mut prev = [NONE; 2];
        for dir in 0..2usize {
            let head = self.edges[e_idx as usize].head[dir];
            if head == NONE {
                return;
            }
            let (penultimate, _) = self.penultimate_blossom_and_outer(head);
            prev[dir] = penultimate;
        }

        if prev[0] == NONE || prev[1] == NONE {
            return;
        }

        if prev[0] != prev[1] {
            edge_list_add(&mut self.nodes, &mut self.edges, prev[0], e_idx, 1);
            edge_list_add(&mut self.nodes, &mut self.edges, prev[1], e_idx, 0);
            self.edges[e_idx as usize].slack -= 2 * self.nodes[prev[0] as usize].blossom_eps;
        } else {
            self.edges[e_idx as usize].next[0] = self.nodes[prev[0] as usize].blossom_selfloops;
            self.nodes[prev[0] as usize].blossom_selfloops = e_idx;
        }
    }

    fn next_tree_plus(&self, mut current: u32, branch_root: u32) -> Option<u32> {
        if self.nodes[current as usize].first_tree_child != NONE {
            return Some(self.nodes[current as usize].first_tree_child);
        }

        while current != branch_root && self.nodes[current as usize].tree_sibling_next == NONE {
            // Match Blossom V's MOVE_NODE_IN_TREE macro:
            //   i = ARC_HEAD(i->match); GET_TREE_PARENT(i, i);
            // The climb first goes to the raw matched MINUS node, and only
            // then resolves the parent arc outward. Resolving the matched
            // endpoint outward too early can jump to an outer blossom root
            // that has no tree_parent arc of its own.
            let minus = self.arc_head_raw(self.nodes[current as usize].match_arc);
            if minus == NONE {
                return None;
            }
            current = self.arc_head_outer(self.nodes[minus as usize].tree_parent_arc);
            if current == NONE {
                return None;
            }
        }

        if current == branch_root {
            None
        } else {
            Some(self.nodes[current as usize].tree_sibling_next)
        }
    }

    fn init_clear_best_edges(&mut self, best_edge: &mut Vec<u32>) {
        if best_edge.len() < self.nodes.len() {
            best_edge.resize(self.nodes.len(), NONE);
        }
        for value in best_edge.iter_mut() {
            *value = NONE;
        }
    }

    fn init_pick_best_edge(&self, best_edge: &[u32], upper_bound: i64) -> Option<u32> {
        let mut best = NONE;
        let mut best_slack = upper_bound;
        for (node_idx, &edge_idx) in best_edge.iter().enumerate() {
            if edge_idx == NONE
                || self.nodes[node_idx].flag != FREE
                || !self.nodes[node_idx].is_outer
            {
                continue;
            }
            let slack = self.edges[edge_idx as usize].slack;
            if slack < best_slack {
                best_slack = slack;
                best = edge_idx;
            }
        }
        (best != NONE).then_some(best)
    }

    fn init_global_cleanup(&mut self, root: u32, eps: i64, best_edge: &mut Vec<u32>) {
        if !self.nodes[root as usize].is_tree_root {
            self.init_clear_best_edges(best_edge);
            return;
        }

        let mut current = root;
        loop {
            if self.nodes[current as usize].is_processed {
                self.nodes[current as usize].y += eps;
                if !self.nodes[current as usize].is_tree_root {
                    let minus = self.arc_head_raw(self.nodes[current as usize].match_arc);
                    self.nodes[minus as usize].y -= eps;
                    let match_slack =
                        self.edges[arc_edge(self.nodes[current as usize].match_arc) as usize].slack;
                    let delta = eps - match_slack;
                    let minus_incident = self.incident_edges(minus);
                    for (e_idx, _) in minus_incident {
                        self.edges[e_idx as usize].slack += delta;
                    }
                }

                let incident = self.incident_edges(current);
                for (e_idx, dir) in incident {
                    let other = self.edges[e_idx as usize].head[dir];
                    if other as usize >= best_edge.len() {
                        best_edge.resize(self.nodes.len(), NONE);
                    }
                    if best_edge[other as usize] == e_idx {
                        best_edge[other as usize] = NONE;
                    }
                    self.edges[e_idx as usize].slack -= eps;
                }
                self.nodes[current as usize].is_processed = false;
            } else if !self.nodes[current as usize].is_tree_root {
                let minus = self.arc_head_raw(self.nodes[current as usize].match_arc);
                if minus as usize >= best_edge.len() {
                    best_edge.resize(self.nodes.len(), NONE);
                }
                best_edge[minus as usize] = NONE;
            }

            if let Some(next) = self.next_tree_plus(current, root) {
                current = next;
            } else {
                break;
            }
        }

        self.init_clear_best_edges(best_edge);
    }

    fn expand_solved_outer_blossoms(&mut self) {
        loop {
            let mut expanded = false;
            for b in self.node_num..self.nodes.len() {
                if self.nodes[b].is_outer && self.nodes[b].is_blossom {
                    self.expand(b as u32);
                    expanded = true;
                }
            }
            if !expanded {
                break;
            }
        }

        self.clear_solved_tree_state();
    }

    fn clear_solved_tree_state(&mut self) {
        self.root_list_head = NONE;
        for v in 0..self.node_num {
            self.nodes[v].flag = FREE;
            self.nodes[v].is_tree_root = false;
            self.nodes[v].tree_eps = 0;
            self.nodes[v].is_processed = false;
            self.nodes[v].tree_root = NONE;
            self.nodes[v].tree_parent_arc = NONE;
            self.nodes[v].first_tree_child = NONE;
            self.nodes[v].tree_sibling_prev = NONE;
            self.nodes[v].tree_sibling_next = NONE;
        }
    }

    fn init_global_finalize(&mut self) {
        let expected_roots = self.tree_num;

        // C++ InitGlobal() hands the generic solver a fresh root-only forest:
        // free outer nodes do not retain stale tree membership from the
        // startup search, and the new roots are rebuilt only from the
        // remaining non-outer original nodes via ExpandInit().
        for r in 0..self.node_num {
            self.nodes[r].is_tree_root = false;
            self.nodes[r].tree_eps = 0;
            self.nodes[r].is_processed = false;
            self.nodes[r].tree_root = NONE;
            self.nodes[r].tree_parent_arc = NONE;
            self.nodes[r].first_tree_child = NONE;
            self.nodes[r].tree_sibling_prev = NONE;
            self.nodes[r].tree_sibling_next = NONE;
            if self.nodes[r].is_outer {
                self.nodes[r].flag = FREE;
            }
        }

        let mut rebuilt_roots = 0usize;
        self.root_list_head = NONE;
        for r in 0..self.node_num {
            if !self.nodes[r].is_outer {
                self.init_expand(r as u32);
                self.nodes[r].is_tree_root = true;
                self.nodes[r].tree_eps = 0;
                self.nodes[r].flag = PLUS;
                self.nodes[r].is_processed = false;
                self.nodes[r].tree_root = r as u32;
                self.nodes[r].tree_parent_arc = NONE;
                self.nodes[r].first_tree_child = NONE;
                self.root_list_append(r as u32);
                self.seed_tree_root_frontier(r as u32);
                rebuilt_roots += 1;
            }
        }
        if rebuilt_roots != expected_roots {
            // The C++ code assumes support-feasible input and has undefined
            // behavior otherwise. Rust should fail nominally here instead of
            // consulting a second exact feasibility oracle.
            self.tree_num = 0;
            return;
        }
        self.tree_num = rebuilt_roots;
    }

    fn init_add_tree_child(&mut self, parent: u32, child: u32) {
        let tree_root = self.nodes[parent as usize].tree_root;
        self.nodes[child as usize].flag = PLUS;
        self.nodes[child as usize].is_tree_root = false;
        self.nodes[child as usize].tree_root = tree_root;
        self.nodes[child as usize].first_tree_child = NONE;
        self.nodes[child as usize].tree_sibling_next = self.nodes[parent as usize].first_tree_child;
        if self.nodes[parent as usize].first_tree_child != NONE {
            let first = self.nodes[parent as usize].first_tree_child;
            self.nodes[child as usize].tree_sibling_prev =
                self.nodes[first as usize].tree_sibling_prev;
            self.nodes[first as usize].tree_sibling_prev = child;
        } else {
            self.nodes[child as usize].tree_sibling_prev = child;
        }
        self.nodes[parent as usize].first_tree_child = child;
    }

    fn init_grow(&mut self, edge_idx: u32, plus_node: u32, free_node: u32) {
        let tree_root = self.nodes[plus_node as usize].tree_root;
        self.nodes[free_node as usize].flag = MINUS;
        self.nodes[free_node as usize].is_tree_root = false;
        self.nodes[free_node as usize].tree_eps = 0;
        self.nodes[free_node as usize].tree_root = tree_root;
        let dir = if self.edges[edge_idx as usize].head[0] == free_node { 1 } else { 0 };
        self.nodes[free_node as usize].tree_parent_arc = make_arc(edge_idx, dir);

        let match_partner = self.arc_head_raw(self.nodes[free_node as usize].match_arc);
        self.init_add_tree_child(plus_node, match_partner);
    }

    fn init_mark_tree_free(&mut self, root: u32) {
        self.nodes[root as usize].flag = FREE;
        let mut plus = self.nodes[root as usize].first_tree_child;
        while plus != NONE {
            let minus = self.arc_head_raw(self.nodes[plus as usize].match_arc);
            self.nodes[minus as usize].flag = FREE;
            self.nodes[plus as usize].flag = FREE;
            match self.next_tree_plus(plus, root) {
                Some(next) => plus = next,
                None => break,
            }
        }
    }

    fn init_find_blossom_root(&mut self, edge_idx: u32) -> u32 {
        let a0 = make_arc(edge_idx, 0);
        let mut branch = 0usize;
        let mut endpoints = [self.arc_head_raw(a0), self.arc_head_raw(arc_rev(a0))];
        let (r, j) = loop {
            if !self.nodes[endpoints[branch] as usize].is_outer {
                break (endpoints[branch], endpoints[1 - branch]);
            }

            self.nodes[endpoints[branch] as usize].is_outer = false;
            if self.nodes[endpoints[branch] as usize].is_tree_root {
                let j = endpoints[branch];
                let mut i = endpoints[1 - branch];
                while self.nodes[i as usize].is_outer {
                    self.nodes[i as usize].is_outer = false;
                    i = self.arc_head_raw(self.nodes[i as usize].match_arc);
                    self.nodes[i as usize].is_outer = false;
                    i = self.arc_head_raw(self.nodes[i as usize].tree_parent_arc);
                }
                break (i, j);
            }

            let i = self.arc_head_raw(self.nodes[endpoints[branch] as usize].match_arc);
            self.nodes[i as usize].is_outer = false;
            endpoints[branch] = self.arc_head_raw(self.nodes[i as usize].tree_parent_arc);
            branch = 1 - branch;
        };

        let mut i = r;
        while i != j {
            i = self.arc_head_raw(self.nodes[i as usize].match_arc);
            self.nodes[i as usize].is_outer = true;
            i = self.arc_head_raw(self.nodes[i as usize].tree_parent_arc);
            self.nodes[i as usize].is_outer = true;
        }
        r
    }

    fn init_shrink(&mut self, edge_idx: u32, tree_root: u32) {
        self.init_mark_tree_free(tree_root);
        let r = self.init_find_blossom_root(edge_idx);

        if !self.nodes[r as usize].is_tree_root {
            let mut j = self.arc_head_raw(self.nodes[r as usize].match_arc);
            let mut aa = self.nodes[j as usize].tree_parent_arc;
            self.nodes[j as usize].match_arc = aa;
            let mut i = self.arc_head_raw(aa);
            while !self.nodes[i as usize].is_tree_root {
                j = self.arc_head_raw(self.nodes[i as usize].match_arc);
                self.nodes[i as usize].match_arc = arc_rev(aa);
                aa = self.nodes[j as usize].tree_parent_arc;
                self.nodes[j as usize].match_arc = aa;
                i = self.arc_head_raw(aa);
            }
            self.nodes[i as usize].match_arc = arc_rev(aa);
        }

        self.nodes[tree_root as usize].is_tree_root = false;
        self.nodes[tree_root as usize].tree_eps = 0;

        let a0 = make_arc(edge_idx, 0);
        let mut branch = 0usize;
        let mut phase = 0usize;
        let mut a_prev = a0;
        let mut i = self.arc_head_raw(a_prev);
        loop {
            let a_next = if phase == 0 {
                self.nodes[i as usize].match_arc
            } else {
                self.nodes[i as usize].tree_parent_arc
            };
            phase = 1 - phase;
            self.nodes[i as usize].flag = PLUS;
            self.nodes[i as usize].match_arc = NONE;

            if branch == 0 {
                self.nodes[i as usize].blossom_sibling_arc = a_next;
                if i == r {
                    branch = 1;
                    phase = 0;
                    a_prev = arc_rev(a0);
                    i = self.arc_head_raw(a_prev);
                    if i == r {
                        break;
                    }
                } else {
                    a_prev = self.nodes[i as usize].blossom_sibling_arc;
                    i = self.arc_head_raw(a_prev);
                }
            } else {
                self.nodes[i as usize].blossom_sibling_arc = arc_rev(a_prev);
                a_prev = a_next;
                i = self.arc_head_raw(a_prev);
                if i == r {
                    break;
                }
            }
        }
        self.nodes[i as usize].blossom_sibling_arc = arc_rev(a_prev);
    }

    fn init_expand(&mut self, k: u32) {
        let mut i = self.arc_head_raw(self.nodes[k as usize].blossom_sibling_arc);
        loop {
            self.nodes[i as usize].flag = FREE;
            self.nodes[i as usize].is_outer = true;
            if i == k {
                break;
            }
            self.nodes[i as usize].match_arc = self.nodes[i as usize].blossom_sibling_arc;
            let j = self.arc_head_raw(self.nodes[i as usize].match_arc);
            self.nodes[j as usize].flag = FREE;
            self.nodes[j as usize].is_outer = true;
            self.nodes[j as usize].match_arc = arc_rev(self.nodes[i as usize].match_arc);
            i = self.arc_head_raw(self.nodes[j as usize].blossom_sibling_arc);
        }
    }

    fn init_augment_branch(&mut self, start: u32, root: u32) {
        self.init_mark_tree_free(root);
        if !self.nodes[start as usize].is_tree_root {
            let mut j = self.arc_head_raw(self.nodes[start as usize].match_arc);
            let mut aa = self.nodes[j as usize].tree_parent_arc;
            self.nodes[j as usize].match_arc = aa;
            let mut i = self.arc_head_raw(aa);
            while !self.nodes[i as usize].is_tree_root {
                j = self.arc_head_raw(self.nodes[i as usize].match_arc);
                self.nodes[i as usize].match_arc = arc_rev(aa);
                aa = self.nodes[j as usize].tree_parent_arc;
                self.nodes[j as usize].match_arc = aa;
                i = self.arc_head_raw(aa);
            }
            self.nodes[i as usize].match_arc = arc_rev(aa);
        }
        if self.nodes[root as usize].is_tree_root {
            self.root_list_remove(root);
            self.nodes[root as usize].is_tree_root = false;
            self.nodes[root as usize].tree_eps = 0;
            self.tree_num -= 1;
        }
    }

    #[cfg(test)]
    fn init_global_event_grow(e_idx: u32, plus: u32, free: u32, edges: &[Edge]) -> InitGlobalEvent {
        let edge = normalized_edge_pair(edges[e_idx as usize].head0);
        InitGlobalEvent::Grow { edge, plus, free }
    }

    #[cfg(test)]
    fn apply_init_global_grow(&mut self, e_idx: u32, plus: u32, free: u32) {
        let event = Self::init_global_event_grow(e_idx, plus, free, &self.edges);
        let before = self.test_strict_parity_snapshot();
        self.init_grow(e_idx, plus, free);
        let after = self.test_strict_parity_snapshot();
        self.init_global_trace.push(event.clone());
        self.init_global_steps.push(InitGlobalStepTrace { event, before, after });
    }

    #[cfg(not(test))]
    fn apply_init_global_grow(&mut self, e_idx: u32, plus: u32, free: u32) {
        self.init_grow(e_idx, plus, free);
        self.maybe_write_debug_trace_snapshot("GROW_AFTER");
    }

    #[cfg(test)]
    fn init_global_event_augment(
        e_idx: u32,
        left: u32,
        right: u32,
        edges: &[Edge],
    ) -> InitGlobalEvent {
        let edge = normalized_edge_pair(edges[e_idx as usize].head0);
        InitGlobalEvent::Augment { edge, left, right }
    }

    #[cfg(test)]
    fn init_global_event_shrink(
        e_idx: u32,
        left: u32,
        right: u32,
        edges: &[Edge],
    ) -> InitGlobalEvent {
        let edge = normalized_edge_pair(edges[e_idx as usize].head0);
        InitGlobalEvent::Shrink { edge, left, right }
    }

    #[cfg(test)]
    fn apply_init_global_augment(&mut self, e_idx: u32, left: u32, right: u32, root: u32) {
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
        self.init_global_trace.push(event.clone());
        self.init_global_steps.push(InitGlobalStepTrace { event, before, after });
    }

    #[cfg(not(test))]
    fn apply_init_global_augment(&mut self, e_idx: u32, left: u32, right: u32, root: u32) {
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
    fn apply_init_global_shrink(&mut self, e_idx: u32, left: u32, right: u32, root: u32) {
        let event = Self::init_global_event_shrink(e_idx, left, right, &self.edges);
        let before = self.test_strict_parity_snapshot();
        self.init_shrink(e_idx, root);
        let after = self.test_strict_parity_snapshot();
        self.init_global_trace.push(event.clone());
        self.init_global_steps.push(InitGlobalStepTrace { event, before, after });
    }

    #[cfg(not(test))]
    fn apply_init_global_shrink(&mut self, e_idx: u32, _left: u32, _right: u32, root: u32) {
        self.init_shrink(e_idx, root);
        self.maybe_write_debug_trace_snapshot("SHRINK_AFTER");
    }

    #[cfg(test)]
    fn apply_generic_grow(&mut self, e_idx: u32, plus: u32, free: u32) {
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
        self.generic_primal_steps.push(GenericPrimalStepTrace { event, before, after });
    }

    #[cfg(not(test))]
    fn apply_generic_grow(&mut self, e_idx: u32, plus: u32, free: u32) {
        self.grow(e_idx, plus, free);
        if let Some((augment_edge, left, right)) = self.grow_tree_after_absorb(plus, free) {
            self.augment(augment_edge, left, right);
        } else {
            self.maybe_write_debug_trace_snapshot("GROW_AFTER");
        }
    }

    #[cfg(test)]
    fn apply_generic_shrink(&mut self, e_idx: u32, left: u32, right: u32) {
        let event = GenericPrimalEvent::Shrink {
            edge: normalized_edge_pair(self.edges[e_idx as usize].head0),
            left,
            right,
        };
        let before = self.test_strict_parity_snapshot();
        self.shrink(e_idx, left, right);
        let after = self.test_strict_parity_snapshot();
        self.generic_primal_steps.push(GenericPrimalStepTrace { event, before, after });
    }

    #[cfg(not(test))]
    fn apply_generic_shrink(&mut self, e_idx: u32, left: u32, right: u32) {
        self.shrink(e_idx, left, right);
        self.maybe_write_debug_trace_snapshot("SHRINK_AFTER");
    }

    #[cfg(test)]
    fn apply_generic_augment(&mut self, e_idx: u32, left: u32, right: u32) {
        let event = GenericPrimalEvent::Augment {
            edge: normalized_edge_pair(self.edges[e_idx as usize].head0),
            left,
            right,
        };
        let before = self.test_strict_parity_snapshot();
        self.augment(e_idx, left, right);
        let after = self.test_strict_parity_snapshot();
        self.generic_primal_steps.push(GenericPrimalStepTrace { event, before, after });
    }

    #[cfg(not(test))]
    fn apply_generic_augment(&mut self, e_idx: u32, left: u32, right: u32) {
        self.augment(e_idx, left, right);
        self.maybe_write_debug_trace_snapshot("AUGMENT_AFTER");
        self.maybe_write_debug_queue_summary("after AUGMENT_AFTER");
    }

    fn perform_generic_expand(&mut self, b: u32) {
        let match_arc = self.nodes[b as usize].match_arc;
        if match_arc != NONE && (arc_edge(match_arc) as usize) < self.edge_num {
            let match_edge = arc_edge(match_arc) as usize;
            core::mem::swap(&mut self.edges[match_edge].slack, &mut self.nodes[b as usize].y);
        }
        self.expand(b);
    }

    #[cfg(test)]
    fn apply_generic_expand(&mut self, b: u32) {
        let event = GenericPrimalEvent::Expand { blossom: b };
        let before = self.test_strict_parity_snapshot();
        self.perform_generic_expand(b);
        let after = self.test_strict_parity_snapshot();
        self.generic_primal_steps.push(GenericPrimalStepTrace { event, before, after });
    }

    #[cfg(not(test))]
    fn apply_generic_expand(&mut self, b: u32) {
        self.perform_generic_expand(b);
        self.maybe_write_debug_trace_snapshot("EXPAND_AFTER");
    }

    pub(super) fn solve(self) -> MatchingResult<M::RowIndex, M::ColumnIndex> {
        // The reference Blossom V code does not impose a small fixed
        // iteration budget on the main loop. The Rust port keeps a bounded
        // entrypoint for focused tests, but the production solver should not
        // synthesize `NoPerfectMatching` from an arbitrary cap.
        self.solve_impl(usize::MAX, usize::MAX, None)
    }

    fn solve_impl(
        mut self,
        max_outer_iters: usize,
        max_inner_iters: usize,
        budget_label: Option<&'static str>,
    ) -> MatchingResult<M::RowIndex, M::ColumnIndex> {
        let n = self.node_num;
        if n == 0 {
            return Ok(Vec::new());
        }
        self.init_global();
        self.maybe_write_debug_trace_snapshot("INIT_GLOBAL_AFTER");

        if self.tree_num == 0 {
            self.maybe_write_debug_trace_snapshot("FINISH_BEFORE");
            return self.into_pairs_checked();
        }

        // Main loop: alternate between primal operations on tight edges
        // and dual updates to create new tight edges.
        let mut iters = 0;
        loop {
            iters += 1;
            if iters > max_outer_iters {
                if let Some(label) = budget_label {
                    #[cfg(test)]
                    panic!(
                        "{label}: outer iteration budget exhausted at iteration {iters} (limit {max_outer_iters})"
                    );
                    #[cfg(not(test))]
                    panic!("{label}: outer iteration budget exhausted");
                }
                return Err(BlossomVError::NoPerfectMatching);
            }

            let mut progress = true;
            let mut inner_iters = 0u32;
            while progress {
                inner_iters += 1;
                if inner_iters as usize > max_inner_iters {
                    if let Some(label) = budget_label {
                        #[cfg(test)]
                        panic!(
                            "{label}: inner iteration budget exhausted at iteration {inner_iters} (limit {max_inner_iters})"
                        );
                        #[cfg(not(test))]
                        panic!("{label}: inner iteration budget exhausted");
                    }
                    return Err(BlossomVError::NoPerfectMatching);
                }
                progress = self.generic_primal_pass_once();
                if self.tree_num == 0 {
                    return self.into_pairs_checked();
                }
            }

            if self.tree_num == 0 {
                self.maybe_write_debug_trace_snapshot("FINISH_BEFORE");
                return self.into_pairs_checked();
            }

            // No progress via tight edges — do a dual update
            self.maybe_write_debug_trace_snapshot("DUAL_UPDATE_BEFORE");
            if !self.update_duals() {
                // Some real cases still need this rescue path: committed HEAD
                // solved honggfuzz case 10 here by augmenting on a virtually
                // tight cross-tree PLUS/PLUS edge after dual update stalled.
                if let Some((e_idx, left, right)) = self.find_virtual_dual_augment_edge() {
                    self.augment(e_idx, left, right);
                    if self.tree_num == 0 {
                        self.maybe_write_debug_trace_snapshot("FINISH_BEFORE");
                        return self.into_pairs_checked();
                    }
                    continue;
                }
                if let Some(blossom) = self.find_global_expand_fallback_blossom() {
                    self.apply_generic_expand(blossom);
                    continue;
                }
                return Err(BlossomVError::NoPerfectMatching);
            }
            self.maybe_write_debug_trace_snapshot("DUAL_UPDATE_AFTER");
            self.maybe_write_debug_queue_summary("after DUAL_UPDATE_AFTER");
        }
    }

    #[cfg(test)]
    pub(super) fn solve_with_test_budget(
        self,
        max_outer_iters: usize,
        max_inner_iters: usize,
    ) -> MatchingResult<M::RowIndex, M::ColumnIndex> {
        self.solve_impl(max_outer_iters, max_inner_iters, Some("BlossomV test budget exhausted"))
    }

    fn seed_tree_root_frontier(&mut self, root: u32) {
        if root == NONE
            || (root as usize) >= self.nodes.len()
            || !self.nodes[root as usize].is_outer
            || !self.nodes[root as usize].is_tree_root
            || self.nodes[root as usize].is_processed
        {
            return;
        }

        for start_dir in 0..2usize {
            let first = self.nodes[root as usize].first[start_dir];
            if first == NONE {
                continue;
            }
            let mut e_idx = first;
            let mut safety = self.edge_num + 1;
            loop {
                let next = self.edges[e_idx as usize].next[start_dir];
                let other = self.edge_head_outer(e_idx, start_dir);
                if other != NONE && other != root && self.nodes[other as usize].is_outer {
                    if self.nodes[other as usize].flag == FREE {
                        self.set_generic_pq0(e_idx, root);
                    } else if self.nodes[other as usize].flag == PLUS
                        && self.nodes[other as usize].is_processed
                    {
                        let other_root = self.find_tree_root(other);
                        if other_root != NONE {
                            self.set_generic_pq00(e_idx, root, other_root);
                        }
                    }
                }
                e_idx = next;
                safety -= 1;
                if e_idx == first || safety == 0 {
                    break;
                }
            }
        }
        self.nodes[root as usize].is_processed = true;
    }

    fn current_root_list(&self) -> Vec<u32> {
        let mut roots = Vec::new();
        let mut seen = Vec::new();
        self.fill_current_root_list(&mut roots, &mut seen);
        roots
    }

    #[inline]
    fn edge_queue_owner(&self, e_idx: u32) -> GenericQueueState {
        self.edge_queue_owner[e_idx as usize]
    }

    #[inline]
    fn edge_queue_stamp(&self, e_idx: u32) -> u64 {
        self.edge_queue_stamp[e_idx as usize]
    }

    #[inline]
    fn set_edge_queue_owner(&mut self, e_idx: u32, owner: GenericQueueState) {
        self.edge_queue_owner[e_idx as usize] = owner;
    }

    #[inline]
    fn set_edge_queue_stamp(&mut self, e_idx: u32, stamp: u64) {
        self.edge_queue_stamp[e_idx as usize] = stamp;
    }

    fn fill_current_root_list(&self, roots: &mut Vec<u32>, seen: &mut Vec<bool>) {
        roots.clear();
        seen.clear();
        seen.resize(self.nodes.len(), false);
        let mut current = self.root_list_head;
        while current != NONE && (current as usize) < self.nodes.len() && !seen[current as usize] {
            seen[current as usize] = true;
            roots.push(current);
            current = self.nodes[current as usize].tree_sibling_next;
        }
    }

    #[inline]
    fn effective_blossom_expand_slack(&self, b: u32) -> i64 {
        let root = self.find_tree_root(b);
        let match_arc = self.nodes[b as usize].match_arc;
        let slack = if match_arc != NONE && (arc_edge(match_arc) as usize) < self.edge_num {
            self.edges[arc_edge(match_arc) as usize].slack
        } else {
            self.nodes[b as usize].y
        };
        slack - self.tree_eps(root)
    }

    fn find_global_expand_fallback_blossom(&self) -> Option<u32> {
        (self.node_num..self.nodes.len())
            .find(|&b| {
                self.nodes[b].is_blossom
                    && self.nodes[b].is_outer
                    && self.nodes[b].flag == MINUS
                    && self.effective_blossom_expand_slack(b as u32) == 0
            })
            .map(|b| b as u32)
    }

    #[inline]
    fn ensure_scheduler_tree_slot(&mut self, root: u32) {
        let needed = root as usize + 1;
        if self.scheduler_trees.len() < needed {
            self.scheduler_trees.resize_with(needed, PersistentTreeState::default);
        }
        self.scheduler_trees[root as usize].root = root;
        self.scheduler_trees[root as usize].eps = self.tree_eps(root);
    }

    #[inline]
    fn ensure_scheduler_tree_edge_slot(&mut self, pair_idx: usize) {
        let needed = pair_idx + 1;
        let was_missing = self.scheduler_tree_edges.len() < needed;
        if self.scheduler_tree_edges.len() < needed {
            self.scheduler_tree_edges.resize_with(needed, PersistentTreeEdgeState::default);
        }
        if was_missing {
            self.scheduler_tree_edges[pair_idx].head = [NONE, NONE];
        }
    }

    #[inline]
    fn clear_generic_tree_currents_local(&mut self, root: u32) {
        if root == NONE || (root as usize) >= self.scheduler_trees.len() {
            return;
        }

        self.scheduler_trees[root as usize].current = SchedulerCurrent::None;

        for dir in 0..2usize {
            let mut pair_cursor = self.scheduler_trees[root as usize].first[dir];
            while let Some(pair_idx) = pair_cursor {
                if pair_idx >= self.scheduler_tree_edges.len() {
                    break;
                }
                let next = self.scheduler_tree_edges[pair_idx].next[dir];
                if let Some(other_root) = self.scheduler_tree_edge_other(pair_idx, root) {
                    if (other_root as usize) < self.scheduler_trees.len() {
                        self.scheduler_trees[other_root as usize].current = SchedulerCurrent::None;
                    }
                }
                pair_cursor = next;
            }
        }
    }

    #[inline]
    fn scheduler_current_pair_dir(&self, root: u32) -> Option<(usize, usize)> {
        if (root as usize) >= self.scheduler_trees.len() {
            return None;
        }
        match self.scheduler_trees[root as usize].current {
            SchedulerCurrent::Pair { pair_idx, dir } => Some((pair_idx, dir)),
            _ => None,
        }
    }

    #[inline]
    fn scheduler_pair_dir_from_active_root(
        &self,
        current_root: u32,
        other_root: u32,
    ) -> Option<(usize, usize)> {
        let pair_idx = self.scheduler_tree_edge_index(current_root, other_root)?;
        let dir = self.scheduler_tree_edge_dir(pair_idx, current_root)?;
        Some((pair_idx, dir))
    }

    fn scheduler_tree_edge_index(&self, left_root: u32, right_root: u32) -> Option<usize> {
        if left_root == NONE || right_root == NONE {
            return None;
        }
        if (left_root as usize) >= self.scheduler_trees.len() {
            return None;
        }
        for dir in 0..2usize {
            let mut pair_cursor = self.scheduler_trees[left_root as usize].first[dir];
            while let Some(pair_idx) = pair_cursor {
                if self.scheduler_tree_edge_other(pair_idx, left_root) == Some(right_root) {
                    return Some(pair_idx);
                }
                pair_cursor = self.scheduler_tree_edges[pair_idx].next[dir];
            }
        }
        None
    }

    #[inline]
    fn scheduler_tree_edge_dir(&self, pair_idx: usize, root: u32) -> Option<usize> {
        if pair_idx >= self.scheduler_tree_edges.len() {
            return None;
        }
        let pair = &self.scheduler_tree_edges[pair_idx];
        if pair.head[1] == root {
            Some(0)
        } else if pair.head[0] == root {
            Some(1)
        } else {
            None
        }
    }

    #[inline]
    fn scheduler_tree_edge_other(&self, pair_idx: usize, root: u32) -> Option<u32> {
        let dir = self.scheduler_tree_edge_dir(pair_idx, root)?;
        let other = self.scheduler_tree_edges[pair_idx].head[dir];
        if other == NONE { None } else { Some(other) }
    }

    fn add_generic_tree_edge(&mut self, current_root: u32, other_root: u32) -> usize {
        let idx = self.scheduler_tree_edges.len();
        #[cfg(test)]
        {
            self.ensure_generic_tree_slot(current_root);
            self.ensure_generic_tree_slot(other_root);
            if self.generic_pairs.len() <= idx {
                self.generic_pairs.resize_with(idx + 1, GenericPairQueues::default);
            }
            self.generic_pairs[idx] = GenericPairQueues::new(current_root, other_root);
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
    fn add_generic_tree_edge_with_other_current_dir(
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

    fn ensure_generic_tree_edge(&mut self, current_root: u32, other_root: u32) -> usize {
        if let Some(pair_idx) = self.scheduler_tree_edge_index(current_root, other_root) {
            pair_idx
        } else {
            self.add_generic_tree_edge(current_root, other_root)
        }
    }

    fn replace_generic_tree_root(&mut self, old_root: u32, new_root: u32) {
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

    fn vec_remove_edge(edges: &mut Vec<u32>, e_idx: u32) {
        if let Some(pos) = edges.iter().position(|&e| e == e_idx) {
            edges.swap_remove(pos);
        }
    }

    #[inline]
    fn clear_generic_queue_state(&mut self, e_idx: u32) {
        self.remove_edge_from_generic_queue(e_idx);
    }

    #[inline]
    fn set_generic_pq0_root_slot(&mut self, e_idx: u32, root: u32, preserve_stamp: bool) {
        if (e_idx as usize) >= self.edge_num || root == NONE {
            return;
        }
        self.ensure_scheduler_tree_slot(root);
        let old_stamp = self.edge_queue_stamp(e_idx);
        self.remove_edge_from_generic_queue(e_idx);
        if !self.scheduler_trees[root as usize].pq0.contains(&e_idx) {
            self.scheduler_trees[root as usize].pq0.push(e_idx);
        }
        self.scheduler_trees[root as usize].pq0_heap.add(
            e_idx,
            self.edges.as_mut_slice(),
            self.pq_nodes.as_mut_slice(),
        );
        self.set_edge_queue_owner(e_idx, GenericQueueState::Pq0 { root });
        if preserve_stamp {
            self.set_edge_queue_stamp(e_idx, old_stamp);
        } else {
            self.generic_queue_epoch = self.generic_queue_epoch.wrapping_add(1);
            self.set_edge_queue_stamp(e_idx, self.generic_queue_epoch);
        }
        #[cfg(test)]
        self.sync_generic_root_queues_from_scheduler(root);
    }

    #[inline]
    fn set_generic_pq0(&mut self, e_idx: u32, root: u32) {
        self.set_generic_pq0_root_slot(e_idx, root, false);
    }

    #[inline]
    fn set_generic_pq_blossoms_root_slot(&mut self, e_idx: u32, root: u32, preserve_stamp: bool) {
        if (e_idx as usize) >= self.edge_num || root == NONE {
            return;
        }
        self.ensure_scheduler_tree_slot(root);
        let old_stamp = self.edge_queue_stamp(e_idx);
        self.remove_edge_from_generic_queue(e_idx);
        if !self.scheduler_trees[root as usize].pq_blossoms.contains(&e_idx) {
            self.scheduler_trees[root as usize].pq_blossoms.push(e_idx);
        }
        self.scheduler_trees[root as usize].pq_blossoms_heap.add(
            e_idx,
            self.edges.as_mut_slice(),
            self.pq_nodes.as_mut_slice(),
        );
        self.set_edge_queue_owner(e_idx, GenericQueueState::PqBlossoms { root });
        if preserve_stamp {
            self.set_edge_queue_stamp(e_idx, old_stamp);
        } else {
            self.generic_queue_epoch = self.generic_queue_epoch.wrapping_add(1);
            self.set_edge_queue_stamp(e_idx, self.generic_queue_epoch);
        }
        #[cfg(test)]
        self.sync_generic_root_queues_from_scheduler(root);
    }

    #[cfg(test)]
    #[inline]
    fn set_generic_pq_blossoms(&mut self, e_idx: u32, root: u32) {
        self.set_generic_pq_blossoms_root_slot(e_idx, root, false);
    }

    #[inline]
    fn set_generic_pq00_local_slot(&mut self, e_idx: u32, root: u32, preserve_stamp: bool) {
        if (e_idx as usize) >= self.edge_num || root == NONE {
            return;
        }
        self.ensure_scheduler_tree_slot(root);
        let old_stamp = self.edge_queue_stamp(e_idx);
        self.remove_edge_from_generic_queue(e_idx);
        if !self.scheduler_trees[root as usize].pq00_local.contains(&e_idx) {
            self.scheduler_trees[root as usize].pq00_local.push(e_idx);
        }
        self.scheduler_trees[root as usize].pq00_local_heap.add(
            e_idx,
            self.edges.as_mut_slice(),
            self.pq_nodes.as_mut_slice(),
        );
        self.set_edge_queue_owner(e_idx, GenericQueueState::Pq00Local { root });
        if preserve_stamp {
            self.set_edge_queue_stamp(e_idx, old_stamp);
        } else {
            self.generic_queue_epoch = self.generic_queue_epoch.wrapping_add(1);
            self.set_edge_queue_stamp(e_idx, self.generic_queue_epoch);
        }
        #[cfg(test)]
        self.sync_generic_root_queues_from_scheduler(root);
    }

    #[inline]
    fn set_generic_pq00(&mut self, e_idx: u32, left_root: u32, right_root: u32) {
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
            if !self.scheduler_tree_edges[pair_idx].pq00.contains(&e_idx) {
                self.scheduler_tree_edges[pair_idx].pq00.push(e_idx);
            }
            self.scheduler_tree_edges[pair_idx].pq00_heap.add(
                e_idx,
                self.edges.as_mut_slice(),
                self.pq_nodes.as_mut_slice(),
            );
            self.set_edge_queue_owner(e_idx, GenericQueueState::Pq00Pair { pair_idx });
            self.generic_queue_epoch = self.generic_queue_epoch.wrapping_add(1);
            self.set_edge_queue_stamp(e_idx, self.generic_queue_epoch);
            #[cfg(test)]
            self.sync_generic_pair_queues_from_scheduler(pair_idx);
        }
    }

    #[inline]
    fn set_generic_pq01(&mut self, e_idx: u32, current_root: u32, other_root: u32) {
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
    fn set_generic_pq01_other_side(&mut self, e_idx: u32, current_root: u32, other_root: u32) {
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
    fn set_generic_pq01_pair_slot(
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
        if !self.scheduler_tree_edges[pair_idx].pq01[dir].contains(&e_idx) {
            self.scheduler_tree_edges[pair_idx].pq01[dir].push(e_idx);
        }
        self.scheduler_tree_edges[pair_idx].pq01_heap[dir].add(
            e_idx,
            self.edges.as_mut_slice(),
            self.pq_nodes.as_mut_slice(),
        );
        self.set_edge_queue_owner(e_idx, GenericQueueState::Pq01Pair { pair_idx, dir });
        if preserve_stamp {
            self.set_edge_queue_stamp(e_idx, old_stamp);
        } else {
            self.generic_queue_epoch = self.generic_queue_epoch.wrapping_add(1);
            self.set_edge_queue_stamp(e_idx, self.generic_queue_epoch);
        }
        #[cfg(test)]
        self.sync_generic_pair_queues_from_scheduler(pair_idx);
    }

    fn remove_edge_from_generic_queue(&mut self, e_idx: u32) {
        if (e_idx as usize) >= self.edge_num {
            return;
        }
        let state = self.edge_queue_owner(e_idx);
        match state {
            GenericQueueState::None => {}
            GenericQueueState::Pq0 { root } => {
                if (root as usize) < self.scheduler_trees.len() {
                    if self.pq_nodes[e_idx as usize].is_in_heap() {
                        self.scheduler_trees[root as usize].pq0_heap.remove(
                            e_idx,
                            self.edges.as_mut_slice(),
                            self.pq_nodes.as_mut_slice(),
                        );
                    }
                    Self::vec_remove_edge(&mut self.scheduler_trees[root as usize].pq0, e_idx);
                    #[cfg(test)]
                    self.sync_generic_root_queues_from_scheduler(root);
                }
            }
            GenericQueueState::Pq00Local { root } => {
                if (root as usize) < self.scheduler_trees.len() {
                    if self.pq_nodes[e_idx as usize].is_in_heap() {
                        self.scheduler_trees[root as usize].pq00_local_heap.remove(
                            e_idx,
                            self.edges.as_mut_slice(),
                            self.pq_nodes.as_mut_slice(),
                        );
                    }
                    Self::vec_remove_edge(
                        &mut self.scheduler_trees[root as usize].pq00_local,
                        e_idx,
                    );
                    #[cfg(test)]
                    self.sync_generic_root_queues_from_scheduler(root);
                }
            }
            GenericQueueState::Pq00Pair { pair_idx } => {
                if pair_idx < self.scheduler_tree_edges.len() {
                    if self.pq_nodes[e_idx as usize].is_in_heap() {
                        self.scheduler_tree_edges[pair_idx].pq00_heap.remove(
                            e_idx,
                            self.edges.as_mut_slice(),
                            self.pq_nodes.as_mut_slice(),
                        );
                    }
                    Self::vec_remove_edge(&mut self.scheduler_tree_edges[pair_idx].pq00, e_idx);
                    #[cfg(test)]
                    self.sync_generic_pair_queues_from_scheduler(pair_idx);
                }
            }
            GenericQueueState::Pq01Pair { pair_idx, dir } => {
                if pair_idx < self.scheduler_tree_edges.len() {
                    if self.pq_nodes[e_idx as usize].is_in_heap() {
                        self.scheduler_tree_edges[pair_idx].pq01_heap[dir].remove(
                            e_idx,
                            self.edges.as_mut_slice(),
                            self.pq_nodes.as_mut_slice(),
                        );
                    }
                    Self::vec_remove_edge(
                        &mut self.scheduler_tree_edges[pair_idx].pq01[dir],
                        e_idx,
                    );
                    #[cfg(test)]
                    self.sync_generic_pair_queues_from_scheduler(pair_idx);
                }
            }
            GenericQueueState::PqBlossoms { root } => {
                if (root as usize) < self.scheduler_trees.len() {
                    if self.pq_nodes[e_idx as usize].is_in_heap() {
                        self.scheduler_trees[root as usize].pq_blossoms_heap.remove(
                            e_idx,
                            self.edges.as_mut_slice(),
                            self.pq_nodes.as_mut_slice(),
                        );
                    }
                    Self::vec_remove_edge(
                        &mut self.scheduler_trees[root as usize].pq_blossoms,
                        e_idx,
                    );
                    #[cfg(test)]
                    self.sync_generic_root_queues_from_scheduler(root);
                }
            }
        }
        self.set_edge_queue_owner(e_idx, GenericQueueState::None);
        self.set_edge_queue_stamp(e_idx, 0);
    }

    fn replace_generic_queue_root(&mut self, old_root: u32, new_root: u32) {
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

    fn detach_generic_root_after_augment(&mut self, root: u32) {
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

    fn detach_scheduler_root_topology(&mut self, root: u32) {
        if root == NONE || (root as usize) >= self.scheduler_trees.len() {
            return;
        }
        for dir in 0..2usize {
            let mut cursor = self.scheduler_trees[root as usize].first[dir];
            while let Some(pair_idx) = cursor {
                if pair_idx >= self.scheduler_tree_edges.len() {
                    break;
                }
                let next = self.scheduler_tree_edges[pair_idx].next[dir];
                if self.scheduler_tree_edge_dir(pair_idx, root) == Some(dir) {
                    self.scheduler_tree_edges[pair_idx].head[1 - dir] = NONE;
                    #[cfg(test)]
                    self.sync_generic_pair_head_from_scheduler(pair_idx);
                    self.scheduler_tree_edges[pair_idx].next[dir] = None;
                }
                cursor = next;
            }
            self.scheduler_trees[root as usize].first[dir] = None;
        }
        self.scheduler_trees[root as usize].current = SchedulerCurrent::None;
    }

    fn root_list_append(&mut self, root: u32) {
        self.nodes[root as usize].tree_sibling_next = NONE;
        if self.root_list_head == NONE {
            self.root_list_head = root;
            self.nodes[root as usize].tree_sibling_prev = root;
            return;
        }

        let mut last = self.root_list_head;
        let mut safety = self.nodes.len();
        while self.nodes[last as usize].tree_sibling_next != NONE && safety > 0 {
            last = self.nodes[last as usize].tree_sibling_next;
            safety -= 1;
        }
        self.nodes[root as usize].tree_sibling_prev = last;
        self.nodes[last as usize].tree_sibling_next = root;
    }

    fn root_list_remove(&mut self, root: u32) {
        if root == NONE || root as usize >= self.nodes.len() {
            return;
        }

        let prev = self.nodes[root as usize].tree_sibling_prev;
        let next = self.nodes[root as usize].tree_sibling_next;

        if self.root_list_head == root {
            self.root_list_head = next;
        }
        if prev != NONE && prev != root && self.nodes[prev as usize].tree_sibling_next == root {
            self.nodes[prev as usize].tree_sibling_next = next;
        }
        if next != NONE {
            self.nodes[next as usize].tree_sibling_prev =
                if prev == NONE || prev == root { next } else { prev };
        }

        self.nodes[root as usize].tree_sibling_prev = NONE;
        self.nodes[root as usize].tree_sibling_next = NONE;
    }

    fn root_list_replace(&mut self, old_root: u32, new_root: u32) {
        let prev = self.nodes[old_root as usize].tree_sibling_prev;
        let next = self.nodes[old_root as usize].tree_sibling_next;

        if self.root_list_head == old_root {
            self.root_list_head = new_root;
        }
        self.nodes[new_root as usize].tree_sibling_prev =
            if prev == NONE || prev == old_root { new_root } else { prev };
        self.nodes[new_root as usize].tree_sibling_next = next;

        if prev != NONE
            && prev != old_root
            && self.nodes[prev as usize].tree_sibling_next == old_root
        {
            self.nodes[prev as usize].tree_sibling_next = new_root;
        }
        if next != NONE {
            self.nodes[next as usize].tree_sibling_prev = new_root;
        }

        self.nodes[old_root as usize].tree_sibling_prev = NONE;
        self.nodes[old_root as usize].tree_sibling_next = NONE;
    }

    // ===== Blossom-aware helpers =====

    /// Find the outermost blossom containing node `v`.
    fn get_outer_node(&self, mut v: u32) -> u32 {
        if v == NONE || v as usize >= self.nodes.len() {
            return NONE;
        }
        let mut limit = self.nodes.len();
        while !self.nodes[v as usize].is_outer {
            let p = self.nodes[v as usize].blossom_parent;
            if p == NONE || p as usize >= self.nodes.len() || limit == 0 {
                return v; // safety: broken chain
            }
            v = p;
            limit -= 1;
        }
        v
    }

    #[inline]
    fn edge_head_outer(&self, edge_idx: u32, dir: usize) -> u32 {
        if (edge_idx as usize) >= self.edge_num {
            return NONE;
        }
        self.get_outer_node(self.edges[edge_idx as usize].head[dir])
    }

    /// Follow an arc and resolve the destination to its outermost blossom.
    #[inline]
    fn arc_head_outer(&self, arc: u32) -> u32 {
        if arc == NONE || (arc_edge(arc) as usize) >= self.edge_num {
            return NONE;
        }
        let e = arc_edge(arc) as usize;
        let d = arc_dir(arc);
        self.get_outer_node(self.edges[e].head[d])
    }

    /// Follow an arc to get the raw destination node (no blossom resolution).
    #[inline]
    fn arc_head_raw(&self, arc: u32) -> u32 {
        if arc == NONE || (arc_edge(arc) as usize) >= self.edge_num {
            return NONE;
        }
        self.edges[arc_edge(arc) as usize].head[arc_dir(arc)]
    }

    fn penultimate_blossom_and_outer(&self, v: u32) -> (u32, u32) {
        if v == NONE || v as usize >= self.nodes.len() {
            return (NONE, NONE);
        }
        if self.nodes[v as usize].is_outer {
            return (NONE, v);
        }

        let mut prev = v;
        let mut limit = self.nodes.len();
        loop {
            let parent = self.nodes[prev as usize].blossom_parent;
            if parent == NONE || parent as usize >= self.nodes.len() || limit == 0 {
                return (prev, parent);
            }
            if self.nodes[parent as usize].is_outer {
                return (prev, parent);
            }
            prev = parent;
            limit -= 1;
        }
    }

    fn normalize_edge_outer_heads(&mut self, e_idx: u32) {
        if (e_idx as usize) >= self.edge_num {
            return;
        }
        for dir in 0..2usize {
            let head = self.edges[e_idx as usize].head[dir];
            if head == NONE || self.nodes[head as usize].is_outer {
                continue;
            }
            let (_, outer) = self.penultimate_blossom_and_outer(head);
            if outer == NONE || outer == head {
                continue;
            }
            edge_list_remove(&mut self.nodes, &mut self.edges, head, e_idx, 1 - dir);
            edge_list_add(&mut self.nodes, &mut self.edges, outer, e_idx, 1 - dir);
        }
    }

    #[allow(clippy::too_many_lines)]
    fn process_edge00(&mut self, e_idx: u32, update_boundary_edge: bool) -> bool {
        if e_idx as usize >= self.edge_num {
            return false;
        }

        let mut prev = [NONE; 2];
        let mut last = [NONE; 2];
        for dir in 0..2usize {
            let head = self.edges[e_idx as usize].head[dir];
            if head == NONE {
                return false;
            }
            if self.nodes[head as usize].is_outer {
                last[dir] = head;
            } else {
                let (penultimate, outer) = self.penultimate_blossom_and_outer(head);
                prev[dir] = penultimate;
                last[dir] = outer;
            }
        }

        if last[0] == NONE || last[1] == NONE {
            return false;
        }

        if last[0] != last[1] {
            for dir in 0..2usize {
                let head = self.edges[e_idx as usize].head[dir];
                if head != last[dir] {
                    edge_list_remove(&mut self.nodes, &mut self.edges, head, e_idx, 1 - dir);
                    edge_list_add(&mut self.nodes, &mut self.edges, last[dir], e_idx, 1 - dir);
                }
            }
            if update_boundary_edge {
                let root = self.find_tree_root(last[0]);
                if root != NONE {
                    self.edges[e_idx as usize].slack -= 2 * self.tree_eps(root);
                }
            }
            return true;
        }

        if prev[0] != prev[1] {
            if prev[0] == NONE || prev[1] == NONE {
                return false;
            }
            for dir in 0..2usize {
                let head = self.edges[e_idx as usize].head[dir];
                if head != prev[dir] {
                    edge_list_remove(&mut self.nodes, &mut self.edges, head, e_idx, 1 - dir);
                    edge_list_add(&mut self.nodes, &mut self.edges, prev[dir], e_idx, 1 - dir);
                }
            }
            self.edges[e_idx as usize].slack -= 2 * self.nodes[prev[0] as usize].blossom_eps;
            return false;
        }

        if prev[0] != NONE {
            for dir in 0..2usize {
                let head = self.edges[e_idx as usize].head[1 - dir];
                if head != NONE {
                    edge_list_remove(&mut self.nodes, &mut self.edges, head, e_idx, dir);
                }
            }
            self.edges[e_idx as usize].next[0] = self.nodes[prev[0] as usize].blossom_selfloops;
            self.nodes[prev[0] as usize].blossom_selfloops = e_idx;
        }
        false
    }

    /// Find the tree root for node `v` (must be outer).
    #[inline]
    fn find_tree_root(&self, v: u32) -> u32 {
        self.nodes[v as usize].tree_root
    }

    #[inline]
    fn tree_eps(&self, root: u32) -> i64 {
        if root == NONE { 0 } else { self.nodes[root as usize].tree_eps }
    }

    // ===== GROW =====

    /// Grow tree: add free node `free_node` (becomes −) and its match
    /// partner (becomes +) to the tree of `plus_node`.
    #[allow(clippy::used_underscore_binding)]
    fn grow(&mut self, _edge_idx: u32, plus_node: u32, free_node: u32) {
        let tree_root = self.nodes[plus_node as usize].tree_root;

        // free_node becomes "−"
        self.nodes[free_node as usize].flag = MINUS;
        self.nodes[free_node as usize].is_tree_root = false;
        self.nodes[free_node as usize].tree_eps = 0;
        self.nodes[free_node as usize].tree_root = tree_root;
        self.nodes[free_node as usize].first_tree_child = NONE;
        self.nodes[free_node as usize].tree_sibling_prev = NONE;
        self.nodes[free_node as usize].tree_sibling_next = NONE;
        // tree_parent points toward plus_node via this edge
        // We need the arc from free_node to plus_node
        let edge_idx = _edge_idx;
        let dir = if self.edges[edge_idx as usize].head[0] == free_node
            || self.edge_head_outer(edge_idx, 0) == free_node
        {
            // free_node is at head[0] side, plus_node at head[1] side
            // Arc from free_node toward plus_node: head[1] = plus_node → dir=1
            1usize
        } else {
            0usize
        };
        self.nodes[free_node as usize].tree_parent_arc = make_arc(edge_idx, dir);

        // Match partner of free_node becomes "+"
        let marc = self.nodes[free_node as usize].match_arc;
        let match_partner = self.arc_head_raw(marc);
        if match_partner == NONE {
            return;
        }
        self.nodes[match_partner as usize].flag = PLUS;
        self.nodes[match_partner as usize].is_tree_root = false;
        self.nodes[match_partner as usize].tree_eps = 0;
        self.nodes[match_partner as usize].tree_root = tree_root;

        // Add match_partner as child of plus_node in tree
        self.nodes[match_partner as usize].tree_sibling_next =
            self.nodes[plus_node as usize].first_tree_child;
        if self.nodes[plus_node as usize].first_tree_child != NONE {
            let old_first = self.nodes[plus_node as usize].first_tree_child;
            self.nodes[match_partner as usize].tree_sibling_prev =
                self.nodes[old_first as usize].tree_sibling_prev;
            self.nodes[old_first as usize].tree_sibling_prev = match_partner;
        } else {
            self.nodes[match_partner as usize].tree_sibling_prev = match_partner;
        }
        self.nodes[match_partner as usize].first_tree_child = NONE;
        self.nodes[plus_node as usize].first_tree_child = match_partner;

        // One less tree (the free node was counted as a potential tree root)
        if self.nodes[free_node as usize].is_tree_root {
            self.nodes[free_node as usize].is_tree_root = false;
            // free_node was a tree root that got absorbed
        }
        // Actually, free nodes from init have is_tree_root=false and
        // are not counted in tree_num. Only unmatched nodes are tree roots.
        // The free_node's match partner was in a matched pair, so neither
        // was a tree root. No tree_num change for GROW.
    }

    fn grow_tree_after_absorb(
        &mut self,
        plus_node: u32,
        free_node: u32,
    ) -> Option<(u32, u32, u32)> {
        let root = self.find_tree_root(plus_node);
        let eps = self.tree_eps(root);
        let new_plus = self.arc_head_raw(self.nodes[free_node as usize].match_arc);

        self.nodes[free_node as usize].y += eps;
        if new_plus != NONE {
            self.nodes[new_plus as usize].y -= eps;
            return self.grow_tree(new_plus, true);
        }

        None
    }

    fn grow_tree(&mut self, branch_root: u32, new_subtree: bool) -> Option<(u32, u32, u32)> {
        let root = self.find_tree_root(branch_root);
        if root == NONE {
            return None;
        }
        let eps = self.tree_eps(root);
        let mut current = branch_root;
        let mut stop = self.nodes[branch_root as usize].tree_sibling_next;
        let mut incident = self.take_incident_scratch();
        if new_subtree && self.nodes[branch_root as usize].first_tree_child != NONE {
            stop = self.nodes[branch_root as usize].first_tree_child;
        }

        let result = loop {
            if !self.nodes[current as usize].is_tree_root {
                let minus = self.arc_head_raw(self.nodes[current as usize].match_arc);
                if minus != NONE {
                    self.collect_incident_edges_into(minus, &mut incident);
                    for &(e_idx, dir) in &incident {
                        let other = self.edge_head_outer(e_idx, dir);
                        if other != NONE
                            && other != minus
                            && self.nodes[other as usize].is_outer
                            && self.nodes[other as usize].flag == PLUS
                            && self.nodes[other as usize].is_processed
                        {
                            let other_root = self.find_tree_root(other);
                            if other_root != NONE && other_root != root {
                                self.remove_edge_from_generic_queue(e_idx);
                                self.edges[e_idx as usize].slack -= eps;
                                let (pair_idx, current_dir) = self
                                    .scheduler_pair_dir_from_active_root(root, other_root)
                                    .or_else(|| self.scheduler_current_pair_dir(other_root))
                                    .unwrap_or_else(|| {
                                        self.add_generic_tree_edge_with_other_current_dir(
                                            root, other_root,
                                        )
                                    });
                                self.set_generic_pq01_pair_slot(
                                    e_idx,
                                    pair_idx,
                                    1 - current_dir,
                                    false,
                                );
                                continue;
                            }
                        }
                        if matches!(self.edge_queue_owner(e_idx), GenericQueueState::Pq0 { .. }) {
                            self.clear_generic_queue_state(e_idx);
                        }
                        self.edges[e_idx as usize].slack -= eps;
                    }
                }
            }

            self.restore_incident_scratch(incident);
            if let Some(augment) = self.grow_node(current) {
                incident = self.take_incident_scratch();
                break Some(augment);
            }
            incident = self.take_incident_scratch();

            if self.nodes[current as usize].first_tree_child != NONE {
                current = self.nodes[current as usize].first_tree_child;
            } else {
                let mut abort_none = false;
                while current != branch_root
                    && self.nodes[current as usize].tree_sibling_next == NONE
                {
                    let minus = self.arc_head_raw(self.nodes[current as usize].match_arc);
                    if minus == NONE {
                        abort_none = true;
                        break;
                    }
                    current = self.arc_head_raw(self.nodes[minus as usize].tree_parent_arc);
                    if current == NONE {
                        abort_none = true;
                        break;
                    }
                }
                if abort_none {
                    break None;
                }
                current = self.nodes[current as usize].tree_sibling_next;
            }

            if current == stop {
                break None;
            }
        };

        self.restore_incident_scratch(incident);
        result
    }

    #[allow(clippy::manual_swap, clippy::too_many_lines)]
    fn grow_node(&mut self, plus_node: u32) -> Option<(u32, u32, u32)> {
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

    fn queue_processed_plus_blossom_match_edge(&mut self, plus_node: u32) {
        if plus_node == NONE
            || (plus_node as usize) >= self.nodes.len()
            || !self.nodes[plus_node as usize].is_outer
            || self.nodes[plus_node as usize].flag != PLUS
            || !self.nodes[plus_node as usize].is_processed
        {
            return;
        }

        let root = self.find_tree_root(plus_node);
        if root == NONE || self.nodes[plus_node as usize].is_tree_root {
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

    fn requeue_processed_plus_edges_after_expand(&mut self, plus_node: u32) {
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

        let mut incident = self.take_incident_scratch();
        self.collect_incident_edges_into(plus_node, &mut incident);
        for &(e_idx, dir) in &incident {
            if !matches!(self.edge_queue_owner(e_idx), GenericQueueState::None) {
                continue;
            }

            let other = self.edge_head_outer(e_idx, dir);
            if other == NONE || other == plus_node || !self.nodes[other as usize].is_outer {
                continue;
            }

            let other_root = self.find_tree_root(other);
            match self.nodes[other as usize].flag {
                FREE => self.set_generic_pq0(e_idx, root),
                PLUS if self.nodes[other as usize].is_processed && other_root != NONE => {
                    self.set_generic_pq00(e_idx, root, other_root);
                }
                MINUS if other_root != NONE && other_root != root => {
                    self.set_generic_pq01(e_idx, root, other_root);
                }
                _ => {}
            }
        }
        self.restore_incident_scratch(incident);

        if !self.nodes[plus_node as usize].is_tree_root {
            let minus = self.arc_head_outer(self.nodes[plus_node as usize].match_arc);
            if minus != NONE
                && (minus as usize) < self.nodes.len()
                && self.nodes[minus as usize].is_blossom
                && self.nodes[minus as usize].is_processed
            {
                let match_edge = arc_edge(self.nodes[plus_node as usize].match_arc);
                if (match_edge as usize) < self.edge_num
                    && matches!(self.edge_queue_owner(match_edge), GenericQueueState::None)
                {
                    self.set_generic_pq_blossoms_root_slot(match_edge, root, false);
                }
            }
        }
    }

    // ===== AUGMENT =====

    /// Augment along the path from `u` (in tree 1) to `v` (in tree 2)
    /// through edge `edge_idx`, flipping the matching along both paths
    /// to the roots, then removing both trees.
    fn augment(&mut self, edge_idx: u32, _u: u32, _v: u32) {
        self.normalize_edge_outer_heads(edge_idx);

        let u = self.edges[edge_idx as usize].head[0];
        let v = self.edges[edge_idx as usize].head[1];
        let root_u = self.find_tree_root(u);
        let root_v = self.find_tree_root(v);

        // Freeze the current tree membership before rewiring matches. The
        // generic tree walk uses the pre-augment alternating structure, and
        // rewriting match arcs first can hide the terminal MINUS nodes.
        let mut members_u = self.take_tree_members_u_scratch();
        let mut members_v = self.take_tree_members_v_scratch();
        self.collect_tree_members_with_scratch(root_u, &mut members_u);
        self.collect_tree_members_with_scratch(root_v, &mut members_v);

        // C++ AugmentBranch() does not bluntly commit lazy eps to every raw
        // edge in the tree. It frees the tree while applying the pending eps
        // only to the tree-node duals and to raw edges incident to processed
        // MINUS nodes. Mirror that here before the tree structure is cleared.
        self.prepare_tree_for_augment(root_u, &members_u);
        self.prepare_tree_for_augment(root_v, &members_v);

        // Flip matching along path from u to root of u's tree
        self.augment_path_to_root(u);
        // Flip matching along path from v to root of v's tree
        self.augment_path_to_root(v);

        // Set the augmenting edge as matched
        self.nodes[u as usize].match_arc = make_arc(edge_idx, 1);
        self.nodes[v as usize].match_arc = make_arc(edge_idx, 0);

        self.root_list_remove(root_u);
        self.root_list_remove(root_v);
        self.detach_generic_root_after_augment(root_u);
        self.detach_generic_root_after_augment(root_v);

        // Free all nodes in both trees using the pre-augment membership.
        self.free_tree_members(&members_u);
        self.free_tree_members(&members_v);
        self.restore_tree_members_u_scratch(members_u);
        self.restore_tree_members_v_scratch(members_v);

        self.tree_num -= 2;
    }

    #[allow(clippy::manual_swap, clippy::too_many_lines)]
    fn prepare_tree_for_augment(&mut self, root: u32, members: &[u32]) {
        if root == NONE || !self.nodes[root as usize].is_outer {
            return;
        }
        let eps = self.tree_eps(root);

        let mut incident_pairs = self.take_incident_pairs_scratch();
        if (root as usize) < self.scheduler_trees.len() {
            for dir in 0..2usize {
                let mut pair_cursor = self.scheduler_trees[root as usize].first[dir];
                while let Some(pair_idx) = pair_cursor {
                    if pair_idx >= self.scheduler_tree_edges.len() {
                        break;
                    }
                    if !incident_pairs.contains(&(pair_idx, dir)) {
                        incident_pairs.push((pair_idx, dir));
                    }
                    pair_cursor = self.scheduler_tree_edges[pair_idx].next[dir];
                }
            }
        }

        // Match C++ AugmentBranch(): before freeing the tree, mark each
        // surviving neighboring tree with the incident TreeEdge and direction
        // that should receive any cross-tree queue traffic during the teardown.
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
        members_mask.clear();
        members_mask.resize(self.nodes.len(), false);
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
        self.restore_queue_edges_scratch(queue_edges);
        self.restore_members_mask_scratch(members_mask);
        self.restore_incident_pairs_scratch(incident_pairs);
    }

    /// Flip matching along the alternating path from node `v` to its tree root.
    fn augment_path_to_root(&mut self, i0: u32) {
        if self.nodes[i0 as usize].is_tree_root {
            return;
        }

        // Follow the C++ AugmentBranch() logic: cache the upward parent arc in
        // a temporary and only overwrite the current plus-node match after the
        // next step has been determined from the old matching.
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

    fn free_tree_members(&mut self, members: &[u32]) {
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

    fn collect_tree_members_with_scratch(&mut self, root: u32, members: &mut Vec<u32>) {
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

    // ===== SHRINK =====

    /// Shrink the odd cycle into a blossom pseudo-node.
    /// Follows C++ PMshrink.cpp: a0 = make_arc(edge_idx, 0) always.
    #[allow(clippy::manual_swap, clippy::too_many_lines)]
    fn shrink(&mut self, edge_idx: u32, _u: u32, _v: u32) {
        // Phase 0: Find LCA, save tree fields
        let a0 = make_arc(edge_idx, 0);
        let endpoint0 = self.arc_head_raw(a0);
        let endpoint1 = self.arc_head_raw(arc_rev(a0));
        let lca = self.find_blossom_root_raw(a0);
        let tree_root = self.nodes[lca as usize].tree_root;
        let shrink_eps = self.tree_eps(tree_root);

        let lca_match_arc = self.nodes[lca as usize].match_arc;
        let lca_is_tree_root = self.nodes[lca as usize].is_tree_root;
        let lca_sibling_prev = self.nodes[lca as usize].tree_sibling_prev;
        let lca_sibling_next = self.nodes[lca as usize].tree_sibling_next;

        // Phase 1: Create blossom
        let b = (self.node_num + self.blossom_count) as u32;
        self.blossom_count += 1;
        while self.nodes.len() <= b as usize {
            self.nodes.push(Node::new_vertex());
        }

        // Phase 2: Collect cycle nodes
        let mut cycle_set = self.take_node_work_a_scratch();
        for &ep in &[endpoint0, endpoint1] {
            let mut node = ep;
            while node != lca {
                cycle_set.push(node);
                let a = if self.nodes[node as usize].flag == PLUS {
                    self.nodes[node as usize].match_arc
                } else {
                    self.nodes[node as usize].tree_parent_arc
                };
                debug_assert_ne!(
                    a,
                    NONE,
                    "shrink phase 2 missing arc: edge_idx={edge_idx} node={node} flag={} is_outer={} match_arc={} tree_parent_arc={} blossom_parent={} lca={lca} endpoint0={endpoint0} endpoint1={endpoint1} tree_root={tree_root}",
                    self.nodes[node as usize].flag,
                    self.nodes[node as usize].is_outer,
                    self.nodes[node as usize].match_arc,
                    self.nodes[node as usize].tree_parent_arc,
                    self.nodes[node as usize].blossom_parent
                );
                node = self.arc_head_raw(a);
            }
        }
        cycle_set.push(lca);
        let mut in_cycle = self.take_members_mask_scratch();
        in_cycle.clear();
        in_cycle.resize(self.nodes.len(), false);
        for &node in &cycle_set {
            in_cycle[node as usize] = true;
        }

        // C++ Shrink() first pass removes matched blossom MINUS nodes from
        // pq_blossoms and swaps the matched-edge slack with the blossom dual
        // before the second pass rewires the cycle.
        for &node in &cycle_set {
            if self.nodes[node as usize].flag != MINUS || !self.nodes[node as usize].is_blossom {
                continue;
            }
            let match_arc = self.nodes[node as usize].match_arc;
            if match_arc == NONE || (arc_edge(match_arc) as usize) >= self.edge_num {
                continue;
            }
            let match_edge = arc_edge(match_arc);
            self.remove_edge_from_generic_queue(match_edge);
            let tmp = self.edges[match_edge as usize].slack;
            self.edges[match_edge as usize].slack = self.nodes[node as usize].y;
            self.nodes[node as usize].y = tmp;
        }

        // Phase 3: Move tree children from cycle PLUS nodes to blossom
        let mut ext_children = self.take_node_work_b_scratch();
        for &node in &cycle_set {
            let fc = self.nodes[node as usize].first_tree_child;
            if fc == NONE {
                continue;
            }
            let mut child = fc;
            loop {
                let nxt = self.nodes[child as usize].tree_sibling_next;
                if !in_cycle[child as usize] {
                    ext_children.push(child);
                }
                if nxt == NONE {
                    break;
                }
                child = nxt;
            }
            self.nodes[node as usize].first_tree_child = NONE;
        }
        if ext_children.is_empty() {
            self.nodes[b as usize].first_tree_child = NONE;
        } else {
            let nc = ext_children.len();
            self.nodes[b as usize].first_tree_child = ext_children[0];
            for ci in 0..nc {
                self.nodes[ext_children[ci] as usize].tree_sibling_next =
                    if ci + 1 < nc { ext_children[ci + 1] } else { NONE };
                self.nodes[ext_children[ci] as usize].tree_sibling_prev =
                    if ci == 0 { ext_children[nc - 1] } else { ext_children[ci - 1] };
            }
        }

        // Phase 4: Set blossom_sibling chain (C++ PMshrink.cpp lines 202-293)
        // Branch 0: endpoint0 → lca, sibling = forward arc (match/tree_parent)
        // Branch 1: endpoint1 → lca, sibling = ARC_REV(a_prev)
        let mut branch = 0usize;
        let mut a_prev = a0;
        let mut i = self.arc_head_raw(a_prev);

        loop {
            let at_lca = i == lca;
            debug_assert_ne!(
                i, NONE,
                "shrink phase 4 reached NONE: edge_idx={edge_idx} a_prev={a_prev} branch={branch} lca={lca} endpoint0={endpoint0} endpoint1={endpoint1} tree_root={tree_root}"
            );
            let a_next = if self.nodes[i as usize].flag == PLUS {
                self.nodes[i as usize].match_arc
            } else {
                self.nodes[i as usize].tree_parent_arc
            };
            debug_assert!(branch <= 1, "shrink phase 4 invalid branch selector {branch}");
            if !at_lca {
                debug_assert_ne!(
                    a_next,
                    NONE,
                    "shrink phase 4 missing next arc: edge_idx={edge_idx} i={i} flag={} is_outer={} match_arc={} tree_parent_arc={} blossom_parent={} a_prev={a_prev} branch={branch} lca={lca} endpoint0={endpoint0} endpoint1={endpoint1} tree_root={tree_root}",
                    self.nodes[i as usize].flag,
                    self.nodes[i as usize].is_outer,
                    self.nodes[i as usize].match_arc,
                    self.nodes[i as usize].tree_parent_arc,
                    self.nodes[i as usize].blossom_parent
                );
            }

            if self.nodes[i as usize].flag == PLUS {
                self.nodes[i as usize].y += shrink_eps;
            } else {
                self.nodes[i as usize].y -= shrink_eps;
            }

            // Mark inner, set parent, clear match
            self.nodes[i as usize].blossom_parent = b;
            self.nodes[i as usize].blossom_grandparent = b;
            self.nodes[i as usize].is_outer = false;
            self.nodes[i as usize].match_arc = NONE;
            self.nodes[i as usize].blossom_eps = shrink_eps;
            self.nodes[i as usize].blossom_selfloops = NONE;
            self.nodes[i as usize].is_processed = false;

            if branch == 0 {
                self.nodes[i as usize].blossom_sibling_arc = a_next;
                if at_lca {
                    branch = 1;
                    a_prev = arc_rev(a0);
                    let next_node = self.arc_head_raw(a_prev);
                    if next_node == lca {
                        break;
                    }
                    i = next_node;
                } else {
                    a_prev = self.nodes[i as usize].blossom_sibling_arc;
                    i = self.arc_head_raw(a_prev);
                }
            } else {
                self.nodes[i as usize].blossom_sibling_arc = arc_rev(a_prev);
                if at_lca || a_next == NONE {
                    break;
                }
                a_prev = a_next;
                let next_node = self.arc_head_raw(a_prev);
                if next_node == lca {
                    break;
                }
                i = next_node;
            }
        }
        // Set LCA's sibling from branch 1 side (overwrites branch 0 setting)
        self.nodes[lca as usize].blossom_sibling_arc = arc_rev(a_prev);

        // Phase 5: Init blossom node
        self.nodes[b as usize].is_outer = true;
        self.nodes[b as usize].is_blossom = true;
        self.nodes[b as usize].flag = PLUS;
        self.nodes[b as usize].y = -shrink_eps;
        self.nodes[b as usize].is_processed = true;
        self.nodes[b as usize].is_tree_root = lca_is_tree_root;
        self.nodes[b as usize].first = [NONE; 2];

        if lca_is_tree_root {
            self.nodes[b as usize].match_arc = NONE;
            self.nodes[b as usize].tree_root = b;
            self.nodes[b as usize].tree_eps = shrink_eps;
            self.nodes[b as usize].tree_sibling_prev = lca_sibling_prev;
            self.nodes[b as usize].tree_sibling_next = lca_sibling_next;
            self.root_list_replace(lca, b);
            self.replace_generic_queue_root(lca, b);
            // Update tree_root for all nodes in this tree
            for j in 0..self.nodes.len() {
                if self.nodes[j].tree_root == lca {
                    self.nodes[j].tree_root = b;
                }
            }
        } else {
            self.nodes[b as usize].match_arc = lca_match_arc;
            self.nodes[b as usize].tree_root = tree_root;
            self.nodes[b as usize].tree_eps = 0;

            // Replace LCA in parent's child list
            self.nodes[b as usize].tree_sibling_prev = lca_sibling_prev;
            self.nodes[b as usize].tree_sibling_next = lca_sibling_next;

            if lca_sibling_prev != NONE
                && self.nodes[lca_sibling_prev as usize].tree_sibling_next == lca
            {
                self.nodes[lca_sibling_prev as usize].tree_sibling_next = b;
            } else {
                // LCA was first child — find parent, update first_tree_child
                let minus = self.arc_head_outer(lca_match_arc);
                let parent = self.arc_head_outer(self.nodes[minus as usize].tree_parent_arc);
                if self.nodes[parent as usize].first_tree_child == lca {
                    self.nodes[parent as usize].first_tree_child = b;
                }
            }

            if lca_sibling_next != NONE {
                self.nodes[lca_sibling_next as usize].tree_sibling_prev = b;
            } else {
                // LCA was last child — update first child's prev
                let minus = self.arc_head_outer(lca_match_arc);
                let parent = self.arc_head_outer(self.nodes[minus as usize].tree_parent_arc);
                let first = self.nodes[parent as usize].first_tree_child;
                if first != NONE {
                    self.nodes[first as usize].tree_sibling_prev = b;
                }
            }
        }

        self.nodes[lca as usize].is_tree_root = false;
        self.nodes[lca as usize].tree_eps = 0;

        // C++ PMshrink.cpp temporarily exposes the partner blossom's y
        // through the matched edge slack during the second pass, then moves
        // that matched edge from the old blossom root onto the new blossom.
        let b_match = self.nodes[b as usize].match_arc;
        let mut b_match_saved_slack = None;
        if b_match != NONE && (arc_edge(b_match) as usize) < self.edge_num {
            let match_edge = arc_edge(b_match) as usize;
            let partner = self.arc_head_outer(b_match);
            if partner != NONE && self.nodes[partner as usize].is_blossom {
                b_match_saved_slack = Some(self.edges[match_edge].slack);
                self.edges[match_edge].slack = self.nodes[partner as usize].y;
            }
        }

        // Phase 6: Port the second pass of PMshrink.cpp more faithfully.
        // Only cycle MINUS nodes rewrite their raw adjacency lists here. The
        // walk is done inline on the live circular list so inner blossom arcs
        // stay attached to the raw cycle node while true boundary arcs move to
        // the new blossom node.
        for &node in &cycle_set {
            if self.nodes[node as usize].flag != MINUS {
                continue;
            }

            for dir in 0..2usize {
                let first = self.nodes[node as usize].first[dir];
                if first == NONE {
                    continue;
                }

                let mut e = first;
                let mut kept_prev = self.edges[e as usize].prev[dir];
                self.edges[kept_prev as usize].next[dir] = NONE;
                let mut new_first = first;
                let mut write_at_head = true;
                let mut kept_tail = NONE;
                let mut safety = self.edge_num + 1;

                while e != NONE && safety > 0 {
                    safety -= 1;
                    let next = self.edges[e as usize].next[dir];

                    if e == edge_idx {
                        if write_at_head {
                            new_first = next;
                        } else {
                            self.edges[kept_tail as usize].next[dir] = next;
                        }
                        e = next;
                        continue;
                    }

                    let other_from = self.edges[e as usize].head[dir];
                    let mut other_to = other_from;
                    let mut limit = self.nodes.len();
                    while other_to != NONE
                        && !self.nodes[other_to as usize].is_outer
                        && !in_cycle[other_to as usize]
                        && limit > 0
                    {
                        other_to = self.nodes[other_to as usize].blossom_parent;
                        limit -= 1;
                    }

                    if other_from != other_to && other_from != NONE && other_to != NONE {
                        edge_list_remove(&mut self.nodes, &mut self.edges, other_from, e, 1 - dir);
                        edge_list_add(&mut self.nodes, &mut self.edges, other_to, e, 1 - dir);
                    }

                    let inner_arc = other_to != NONE && in_cycle[other_to as usize];
                    if inner_arc {
                        self.edges[e as usize].prev[dir] = kept_prev;
                        kept_prev = e;
                        if !write_at_head {
                            self.edges[kept_tail as usize].next[dir] = e;
                        } else {
                            new_first = e;
                            write_at_head = false;
                        }
                        kept_tail = e;

                        if self.nodes[other_to as usize].flag == MINUS {
                            self.edges[e as usize].slack += shrink_eps;
                        }
                    } else {
                        if write_at_head {
                            new_first = next;
                        } else {
                            self.edges[kept_tail as usize].next[dir] = next;
                        }
                        self.edges[e as usize].slack += 2 * shrink_eps;
                        edge_list_add(&mut self.nodes, &mut self.edges, b, e, dir);

                        let blossom_root = self.nodes[b as usize].tree_root;
                        if other_to != NONE && blossom_root != NONE {
                            let other_root = self.find_tree_root(other_to);
                            match self.nodes[other_to as usize].flag {
                                FREE => self.set_generic_pq0(e, blossom_root),
                                PLUS if other_root != NONE => {
                                    self.set_generic_pq00(e, blossom_root, other_root);
                                }
                                MINUS if other_root != NONE && other_root != blossom_root => {
                                    self.set_generic_pq01_other_side(e, blossom_root, other_root);
                                }
                                _ => {}
                            }
                        }
                    }

                    e = next;
                }

                self.nodes[node as usize].first[dir] = new_first;
                if new_first != NONE {
                    self.edges[kept_prev as usize].next[dir] = new_first;
                    self.edges[new_first as usize].prev[dir] = kept_prev;
                }
            }
        }

        if b_match != NONE && (arc_edge(b_match) as usize) < self.edge_num {
            let match_edge = arc_edge(b_match);
            if let Some(saved_slack) = b_match_saved_slack {
                self.edges[match_edge as usize].slack = saved_slack;
            }
            let dir = arc_dir(b_match);
            let old_owner = self.edges[match_edge as usize].head[1 - dir];
            if old_owner != NONE && old_owner != b {
                edge_list_remove(&mut self.nodes, &mut self.edges, old_owner, match_edge, dir);
                edge_list_add(&mut self.nodes, &mut self.edges, b, match_edge, dir);
            }
        }

        // Boundary edges that still sit on raw cycle PLUS nodes must be
        // reachable from the new outer blossom, just like in C++ Shrink().
        self.promote_boundary_edges_to_outer_blossom(b);
        self.rebuild_generic_queue_membership_for_outer_blossom(b);
        self.restore_node_work_b_scratch(ext_children);
        self.restore_node_work_a_scratch(cycle_set);
        self.restore_members_mask_scratch(in_cycle);
    }

    fn find_blossom_root_raw(&self, a0: u32) -> u32 {
        let mut branch_nodes = [self.arc_head_raw(a0), self.arc_head_raw(arc_rev(a0))];
        let mut marked = vec![false; self.nodes.len()];
        let mut branch = 0usize;

        loop {
            let current = branch_nodes[branch];
            if current == NONE || (current as usize) >= self.nodes.len() {
                return NONE;
            }
            if marked[current as usize] {
                return current;
            }
            marked[current as usize] = true;

            if self.nodes[current as usize].is_tree_root {
                let mut other = branch_nodes[1 - branch];
                while other != NONE
                    && (other as usize) < self.nodes.len()
                    && !marked[other as usize]
                {
                    marked[other as usize] = true;
                    let minus = self.arc_head_raw(self.nodes[other as usize].match_arc);
                    if minus == NONE || (minus as usize) >= self.nodes.len() {
                        return NONE;
                    }
                    other = self.arc_head_raw(self.nodes[minus as usize].tree_parent_arc);
                }
                return other;
            }

            let minus = self.arc_head_raw(self.nodes[current as usize].match_arc);
            if minus == NONE || (minus as usize) >= self.nodes.len() {
                return NONE;
            }
            branch_nodes[branch] = self.arc_head_raw(self.nodes[minus as usize].tree_parent_arc);
            branch = 1 - branch;
        }
    }

    fn promote_boundary_edges_to_outer_blossom(&mut self, blossom: u32) {
        if blossom == NONE
            || blossom as usize >= self.nodes.len()
            || !self.nodes[blossom as usize].is_outer
            || !self.nodes[blossom as usize].is_blossom
        {
            return;
        }

        for e_idx in 0..self.edge_num {
            let outer0 = self.edge_head_outer(e_idx as u32, 0);
            let outer1 = self.edge_head_outer(e_idx as u32, 1);

            if outer0 == blossom && outer1 != NONE && outer1 != blossom {
                let raw0 = self.edges[e_idx].head[0];
                if raw0 != NONE && raw0 != blossom {
                    edge_list_remove(&mut self.nodes, &mut self.edges, raw0, e_idx as u32, 1);
                    edge_list_add(&mut self.nodes, &mut self.edges, blossom, e_idx as u32, 1);
                }
            }

            if outer1 == blossom && outer0 != NONE && outer0 != blossom {
                let raw1 = self.edges[e_idx].head[1];
                if raw1 != NONE && raw1 != blossom {
                    edge_list_remove(&mut self.nodes, &mut self.edges, raw1, e_idx as u32, 0);
                    edge_list_add(&mut self.nodes, &mut self.edges, blossom, e_idx as u32, 0);
                }
            }
        }
    }

    fn rebuild_generic_queue_membership_for_outer_blossom(&mut self, blossom: u32) {
        if blossom == NONE
            || blossom as usize >= self.nodes.len()
            || !self.nodes[blossom as usize].is_outer
        {
            return;
        }

        let root = self.find_tree_root(blossom);
        if root == NONE {
            return;
        }

        for e_idx in 0..self.edge_num {
            let outer0 = self.edge_head_outer(e_idx as u32, 0);
            let outer1 = self.edge_head_outer(e_idx as u32, 1);
            if outer0 != blossom && outer1 != blossom {
                continue;
            }

            let old_state = self.edge_queue_owner(e_idx as u32);
            let old_stamp = self.edge_queue_stamp(e_idx as u32);
            self.remove_edge_from_generic_queue(e_idx as u32);

            let pq_blossom_root = [outer0, outer1].into_iter().find_map(|cand| {
                if cand == NONE || (cand as usize) >= self.nodes.len() {
                    return None;
                }
                if !self.nodes[cand as usize].is_blossom || self.nodes[cand as usize].flag != MINUS
                {
                    return None;
                }
                let match_arc = self.nodes[cand as usize].match_arc;
                if match_arc == NONE {
                    return None;
                }
                let match_edge = arc_edge(match_arc);
                if (match_edge as usize) >= self.edge_num || match_edge != e_idx as u32 {
                    return None;
                }
                let cand_root = self.find_tree_root(cand);
                (cand_root != NONE).then_some(cand_root)
            });
            if let Some(pq_root) = pq_blossom_root {
                self.set_generic_pq_blossoms_root_slot(
                    e_idx as u32,
                    pq_root,
                    !matches!(old_state, GenericQueueState::None),
                );
                if !matches!(old_state, GenericQueueState::None) {
                    self.set_edge_queue_stamp(e_idx as u32, old_stamp);
                }
                continue;
            }

            if outer0 == NONE || outer1 == NONE {
                continue;
            }
            if outer0 == outer1 {
                if matches!(old_state, GenericQueueState::Pq00Local { .. })
                    && self.edges[e_idx].slack > 0
                {
                    self.set_generic_pq00(e_idx as u32, root, root);
                    self.set_edge_queue_stamp(e_idx as u32, old_stamp);
                }
                continue;
            }

            let other = if outer0 == blossom { outer1 } else { outer0 };
            match self.nodes[other as usize].flag {
                FREE => {
                    let preserve_stamp = matches!(old_state, GenericQueueState::Pq0 { root: old_root } if old_root == root);
                    self.set_generic_pq0_root_slot(e_idx as u32, root, preserve_stamp);
                    if preserve_stamp {
                        self.set_edge_queue_stamp(e_idx as u32, old_stamp);
                    }
                }
                PLUS => {
                    let other_root = self.find_tree_root(other);
                    if other_root != NONE {
                        self.set_generic_pq00(e_idx as u32, root, other_root);
                    }
                }
                MINUS => {
                    let other_root = self.find_tree_root(other);
                    if other_root != NONE && other_root != root {
                        self.set_generic_pq01(e_idx as u32, root, other_root);
                    }
                }
                _ => {}
            }
        }
    }

    // ===== EXPAND =====

    /// Expand blossom `b` (a "−" blossom with y=0).
    /// Following C++ PMexpand.cpp: resolve matchings, rebuild tree path, splice
    /// child list.
    #[allow(
        clippy::collapsible_match,
        clippy::manual_swap,
        clippy::match_same_arms,
        clippy::too_many_lines
    )]
    fn expand(&mut self, b: u32) {
        let b_match = self.nodes[b as usize].match_arc;
        let b_tp = self.nodes[b as usize].tree_parent_arc;
        let b_tree_root = self.nodes[b as usize].tree_root;
        let eps = if b_tree_root != NONE { self.tree_eps(b_tree_root) } else { 0 };

        // child_plus = PLUS node above blossom (its match partner)
        let child_plus = if b_match != NONE && (arc_edge(b_match) as usize) < self.edge_num {
            self.arc_head_raw(b_match)
        } else {
            NONE
        };
        // grandparent = PLUS node above blossom via tree_parent
        let grandparent = if b_tp != NONE && (arc_edge(b_tp) as usize) < self.edge_num {
            self.arc_head_raw(b_tp)
        } else {
            NONE
        };

        // Collect direct children of b
        let mut children = self.take_node_work_a_scratch();
        for i in 0..self.nodes.len() {
            if self.nodes[i].blossom_parent == b && !self.nodes[i].is_outer {
                children.push(i as u32);
            }
        }

        // C++ Expand() restores each child's blossom selfloops before the
        // inner-arc pass. Without this, minus-to-free edges stay detached from
        // adjacency lists and the later slack updates miss them.
        for &child in &children {
            self.nodes[child as usize].is_outer = true;
            while self.nodes[child as usize].blossom_selfloops != NONE {
                let e_idx = self.nodes[child as usize].blossom_selfloops;
                self.nodes[child as usize].blossom_selfloops = self.edges[e_idx as usize].next[0];
                self.process_expand_selfloop(e_idx);
            }
            self.nodes[child as usize].is_outer = false;
        }

        // Mark all children as outer, initially FREE
        for &c in &children {
            self.nodes[c as usize].is_outer = true;
            self.nodes[c as usize].blossom_parent = NONE;
            self.nodes[c as usize].flag = FREE;
            self.nodes[c as usize].tree_root = NONE;
        }

        // Find k: child containing ARC_TAIL0 of b's match edge
        let mut k = NONE;
        if b_match != NONE && (arc_edge(b_match) as usize) < self.edge_num {
            let md = arc_dir(b_match);
            // head0 has opposite direction from head (edge_list_add swap)
            let tail = self.edges[arc_edge(b_match) as usize].head0[md];
            k = tail;
            while !children.contains(&k) {
                let p = self.nodes[k as usize].blossom_parent;
                if p == NONE {
                    break;
                }
                k = p;
            }

            // Resolve internal matchings via blossom_sibling
            self.nodes[k as usize].match_arc = b_match;
            let k_sib = self.nodes[k as usize].blossom_sibling_arc;
            if k_sib != NONE {
                let mut ci = self.edges[arc_edge(k_sib) as usize].head[arc_dir(k_sib)];
                let mut limit = self.nodes.len();
                while ci != k && limit > 0 {
                    limit -= 1;
                    let ci_sib = self.nodes[ci as usize].blossom_sibling_arc;
                    if ci_sib == NONE {
                        break;
                    }
                    self.nodes[ci as usize].match_arc = ci_sib;
                    let cj = self.edges[arc_edge(ci_sib) as usize].head[arc_dir(ci_sib)];
                    self.nodes[cj as usize].match_arc = arc_rev(ci_sib);
                    let next_sibling_arc = self.nodes[cj as usize].blossom_sibling_arc;
                    if next_sibling_arc == NONE {
                        break;
                    }
                    ci = self.edges[arc_edge(next_sibling_arc) as usize].head
                        [arc_dir(next_sibling_arc)];
                }
            }
        }

        // Find tp and rebuild tree path (C++ PMexpand.cpp lines 107-162)
        let mut j_top = NONE;
        if b_tp != NONE && (arc_edge(b_tp) as usize) < self.edge_num {
            let pd = arc_dir(b_tp);
            let tp_tail = self.edges[arc_edge(b_tp) as usize].head0[pd];
            let mut tp = tp_tail;
            while !children.contains(&tp) {
                let p = self.nodes[tp as usize].blossom_parent;
                if p == NONE {
                    break;
                }
                tp = p;
            }

            // Set tp as MINUS in the tree
            self.nodes[tp as usize].flag = MINUS;
            self.nodes[tp as usize].tree_root = b_tree_root;
            self.nodes[tp as usize].tree_parent_arc = b_tp;
            self.nodes[tp as usize].y += eps;

            if k != NONE && tp != k {
                let tp_match = self.nodes[tp as usize].match_arc;
                let tp_sib = self.nodes[tp as usize].blossom_sibling_arc;

                if tp_match != NONE && tp_match == tp_sib {
                    // Forward: tp → match_partner(PLUS) → sib → MINUS → match → PLUS → ...→ k
                    // Build chain: j_top = first PLUS. Each PLUS links to next via
                    // first_tree_child. Last PLUS.first_tree_child =
                    // child_plus.
                    let mut cur = self.edges[arc_edge(tp_match) as usize].head[arc_dir(tp_match)];
                    j_top = cur;
                    let mut prev_plus: u32 = NONE;
                    let mut limit = self.nodes.len();
                    loop {
                        if limit == 0 {
                            break;
                        }
                        limit -= 1;
                        // cur is PLUS
                        self.nodes[cur as usize].flag = PLUS;
                        self.nodes[cur as usize].tree_root = b_tree_root;
                        self.nodes[cur as usize].y -= eps;
                        self.nodes[cur as usize].first_tree_child = NONE;
                        self.nodes[cur as usize].tree_sibling_prev = cur;
                        self.nodes[cur as usize].tree_sibling_next = NONE;

                        // Link previous PLUS → this PLUS via first_tree_child
                        if prev_plus != NONE {
                            self.nodes[prev_plus as usize].first_tree_child = cur;
                        }

                        let cs = self.nodes[cur as usize].blossom_sibling_arc;
                        if cs == NONE {
                            break;
                        }
                        let nxt = self.arc_head_raw(cs);
                        self.nodes[nxt as usize].flag = MINUS;
                        self.nodes[nxt as usize].tree_root = b_tree_root;
                        self.nodes[nxt as usize].tree_parent_arc = arc_rev(cs);
                        self.nodes[nxt as usize].y += eps;
                        if nxt == k {
                            // Last PLUS connects to child_plus
                            self.nodes[cur as usize].first_tree_child = child_plus;
                            break;
                        }
                        prev_plus = cur;
                        let nm = self.nodes[nxt as usize].match_arc;
                        if nm == NONE {
                            break;
                        }
                        cur = self.arc_head_raw(nm);
                    }
                } else if tp_match != NONE {
                    // Backward: k → sib → PLUS → match → MINUS → sib → PLUS → ... → tp
                    // Build chain bottom-up: start with j = child_plus, walk up.
                    let mut j = child_plus;
                    let mut cur = k;
                    let mut limit = self.nodes.len();
                    loop {
                        if limit == 0 {
                            break;
                        }
                        limit -= 1;
                        let cs = self.nodes[cur as usize].blossom_sibling_arc;
                        if cs == NONE {
                            break;
                        }
                        self.nodes[cur as usize].tree_parent_arc = cs;
                        self.nodes[cur as usize].flag = MINUS;
                        self.nodes[cur as usize].tree_root = b_tree_root;
                        self.nodes[cur as usize].y += eps;

                        let nxt = self.arc_head_raw(cs);
                        self.nodes[nxt as usize].flag = PLUS;
                        self.nodes[nxt as usize].tree_root = b_tree_root;
                        self.nodes[nxt as usize].y -= eps;
                        self.nodes[nxt as usize].first_tree_child = j;
                        j = nxt;
                        self.nodes[nxt as usize].tree_sibling_prev = nxt;
                        self.nodes[nxt as usize].tree_sibling_next = NONE;

                        let nm = self.nodes[nxt as usize].match_arc;
                        if nm == NONE {
                            break;
                        }
                        cur = self.arc_head_raw(nm);
                        if self.nodes[cur as usize].flag == MINUS {
                            break; // reached tp
                        }
                    }
                    j_top = j;
                }
            }

            // Replace child_plus with j_top in grandparent's child list (C++ lines 151-162)
            if j_top != NONE && child_plus != NONE && j_top != child_plus && grandparent != NONE {
                let cp_prev = self.nodes[child_plus as usize].tree_sibling_prev;
                let cp_next = self.nodes[child_plus as usize].tree_sibling_next;

                self.nodes[j_top as usize].tree_sibling_prev = cp_prev;
                self.nodes[j_top as usize].tree_sibling_next = cp_next;

                if cp_prev != NONE && self.nodes[cp_prev as usize].tree_sibling_next == child_plus {
                    self.nodes[cp_prev as usize].tree_sibling_next = j_top;
                } else if self.nodes[grandparent as usize].first_tree_child == child_plus {
                    self.nodes[grandparent as usize].first_tree_child = j_top;
                }

                if cp_next != NONE {
                    self.nodes[cp_next as usize].tree_sibling_prev = j_top;
                } else {
                    // child_plus was last child — update first child's prev
                    let first = self.nodes[grandparent as usize].first_tree_child;
                    if first != NONE {
                        self.nodes[first as usize].tree_sibling_prev = j_top;
                    }
                }

                // Disconnect child_plus from sibling list
                self.nodes[child_plus as usize].tree_sibling_prev = child_plus;
                self.nodes[child_plus as usize].tree_sibling_next = NONE;
            }
        }

        // Recreate the lazy-dual updates that C++ applies while expanding the
        // alternating branch inside the blossom.
        if k != NONE {
            let mut minus = k;
            let mut limit = self.nodes.len();
            loop {
                if limit == 0 {
                    break;
                }
                limit -= 1;

                if self.nodes[minus as usize].is_blossom {
                    let match_edge = arc_edge(self.nodes[minus as usize].match_arc);
                    let tmp = self.edges[match_edge as usize].slack;
                    self.edges[match_edge as usize].slack = self.nodes[minus as usize].y;
                    self.nodes[minus as usize].y = tmp;
                }

                let listed_minus = self.incident_edges(minus);
                let mut seen_minus = vec![false; self.edge_num];
                for (e_idx, dir) in listed_minus {
                    seen_minus[e_idx as usize] = true;
                    let other = self.edge_head_outer(e_idx, dir);
                    if other != NONE && self.nodes[other as usize].flag != PLUS {
                        self.edges[e_idx as usize].slack -= eps;
                    }
                }
                for (e_idx, dir) in self.raw_incident_edges(minus) {
                    if seen_minus[e_idx as usize] {
                        continue;
                    }
                    let other = self.edges[e_idx as usize].head[1 - dir];
                    if other != NONE && self.nodes[other as usize].flag != PLUS {
                        self.edges[e_idx as usize].slack -= eps;
                    }
                }
                self.nodes[minus as usize].is_processed = true;

                if self.nodes[minus as usize].tree_parent_arc == b_tp {
                    break;
                }

                let plus = self.arc_head_raw(self.nodes[minus as usize].tree_parent_arc);
                let listed_plus = self.incident_edges(plus);
                let mut seen_plus = vec![false; self.edge_num];
                for (e_idx, dir) in listed_plus {
                    seen_plus[e_idx as usize] = true;
                    let other = self.edge_head_outer(e_idx, dir);
                    if other == NONE {
                        continue;
                    }
                    if self.nodes[other as usize].flag == FREE {
                        self.edges[e_idx as usize].slack += eps;
                    } else if self.nodes[other as usize].flag == PLUS && plus < other {
                        self.edges[e_idx as usize].slack += 2 * eps;
                    }
                }
                for (e_idx, dir) in self.raw_incident_edges(plus) {
                    if seen_plus[e_idx as usize] {
                        continue;
                    }
                    let other = self.edges[e_idx as usize].head[1 - dir];
                    if other == NONE {
                        continue;
                    }
                    if self.nodes[other as usize].flag == FREE {
                        self.edges[e_idx as usize].slack += eps;
                    } else if self.nodes[other as usize].flag == PLUS && plus < other {
                        self.edges[e_idx as usize].slack += 2 * eps;
                    }
                }
                self.nodes[plus as usize].is_processed = true;

                if self.nodes[plus as usize].match_arc == NONE {
                    break;
                }
                minus = self.arc_head_raw(self.nodes[plus as usize].match_arc);
            }
        }

        // Move edges from blossom back to original nodes
        let mut moves = self.take_edge_moves_scratch();
        for dir in 0..2usize {
            let first = self.nodes[b as usize].first[dir];
            if first == NONE {
                continue;
            }
            let mut e = first;
            let mut safety = self.edge_num + 1;
            loop {
                safety -= 1;
                if safety == 0 {
                    break;
                } // corrupted list guard
                let next = self.edges[e as usize].next[dir];
                let orig = self.edges[e as usize].head0[dir];
                let ov = self.get_outer_node(orig);
                if ov != b {
                    moves.push((e, dir, ov));
                }
                if next == first {
                    break;
                }
                e = next;
            }
        }
        for &(e, dir, new_owner) in &moves {
            edge_list_remove(&mut self.nodes, &mut self.edges, b, e, dir);
            edge_list_add(&mut self.nodes, &mut self.edges, new_owner, e, dir);
            let other = self.edge_head_outer(e, dir);
            let other_root = if other == NONE { NONE } else { self.find_tree_root(other) };
            let old_owner = self.edge_queue_owner(e);
            let new_owner_match_edge = if self.nodes[new_owner as usize].match_arc != NONE {
                let match_edge = arc_edge(self.nodes[new_owner as usize].match_arc);
                ((match_edge as usize) < self.edge_num).then_some(match_edge)
            } else {
                None
            };
            let old_stamp = self.edge_queue_stamp(e);

            self.clear_generic_queue_state(e);

            match self.nodes[new_owner as usize].flag {
                MINUS => {
                    let preserve_same_root_blossom = matches!(
                        old_owner,
                        GenericQueueState::PqBlossoms { root } if root == b_tree_root
                    ) && other != NONE
                        && self.nodes[other as usize].is_blossom;
                    let seed_new_blossom = matches!(old_owner, GenericQueueState::None);
                    if self.nodes[new_owner as usize].is_blossom
                        && b_tree_root != NONE
                        && other != NONE
                        && new_owner_match_edge == Some(e)
                        && (seed_new_blossom || preserve_same_root_blossom)
                    {
                        self.set_generic_pq_blossoms_root_slot(
                            e,
                            b_tree_root,
                            preserve_same_root_blossom,
                        );
                        if preserve_same_root_blossom {
                            self.set_edge_queue_stamp(e, old_stamp);
                        }
                    }
                }
                FREE => {
                    self.edges[e as usize].slack += eps;
                    if other != NONE
                        && self.nodes[other as usize].is_outer
                        && self.nodes[other as usize].flag == PLUS
                        && other_root != NONE
                    {
                        self.set_generic_pq0(e, other_root);
                    }
                }
                PLUS => {
                    self.edges[e as usize].slack += 2 * eps;
                    if other == NONE || !self.nodes[other as usize].is_outer {
                        continue;
                    }
                    match self.nodes[other as usize].flag {
                        FREE => {
                            if b_tree_root != NONE {
                                self.set_generic_pq0(e, b_tree_root);
                            }
                        }
                        PLUS => {
                            if b_tree_root != NONE && other_root != NONE {
                                self.set_generic_pq00(e, b_tree_root, other_root);
                            }
                        }
                        MINUS => {
                            if b_tree_root != NONE
                                && other_root != NONE
                                && other_root != b_tree_root
                            {
                                self.set_generic_pq01(e, b_tree_root, other_root);
                            }
                        }
                        _ => {}
                    }
                }
                _ => {}
            }
        }

        self.nodes[b as usize].is_outer = false;
        self.nodes[b as usize].is_blossom = false;
        self.nodes[b as usize].first = [NONE; 2];

        // Expanding the blossom makes processed PLUS children expose outer
        // boundary edges again. Requeue those now that ownership is visible.
        for &child in &children {
            self.requeue_processed_plus_edges_after_expand(child);
        }

        self.restore_edge_moves_scratch(moves);
        self.restore_node_work_a_scratch(children);
    }

    // ===== DUAL UPDATE =====

    /// Update dual variables uniformly. Returns false if no progress possible.
    #[allow(clippy::too_many_lines)]
    fn update_duals(&mut self) -> bool {
        let mut roots = core::mem::take(&mut self.scratch.dual_roots);
        let mut seen = core::mem::take(&mut self.scratch.dual_seen);
        let mut root_to_var = core::mem::take(&mut self.scratch.dual_root_to_var);
        let mut local_caps = core::mem::take(&mut self.scratch.dual_local_caps);
        let mut pair_eps00 = core::mem::take(&mut self.scratch.dual_pair_eps00);
        let mut pair_eps01 = core::mem::take(&mut self.scratch.dual_pair_eps01);
        let mut deltas = core::mem::take(&mut self.scratch.dual_deltas);
        let mut marks = core::mem::take(&mut self.scratch.dual_marks);
        let mut queue = core::mem::take(&mut self.scratch.dual_queue);
        let mut component = core::mem::take(&mut self.scratch.dual_component);

        self.fill_current_root_list(&mut roots, &mut seen);
        let result = if roots.is_empty() {
            false
        } else {
            for &root in &roots {
                let _ = self.tree_min_pq00_local_for_step3(root);
            }

            root_to_var.clear();
            root_to_var.resize(self.nodes.len(), usize::MAX);
            for (var, &root) in roots.iter().enumerate() {
                root_to_var[root as usize] = var;
            }

            let inf_cap = i64::MAX / 4;
            local_caps.clear();
            local_caps.resize(roots.len(), inf_cap);
            for (var, &root) in roots.iter().enumerate() {
                let local_abs_eps = self.compute_tree_local_eps(root);
                if local_abs_eps != i64::MAX {
                    local_caps[var] = local_caps[var].min(local_abs_eps - self.tree_eps(root));
                }
            }

            let roots_len = roots.len();
            pair_eps00.clear();
            pair_eps01.clear();
            pair_eps00.resize(roots_len * roots_len, inf_cap);
            pair_eps01.resize(roots_len * roots_len, inf_cap);
            let pair_idx = |u: usize, v: usize| u * roots_len + v;

            for e in 0..self.edge_num {
                let u = self.edge_head_outer(e as u32, 0);
                let v = self.edge_head_outer(e as u32, 1);
                if u == NONE || v == NONE || u == v {
                    continue;
                }

                let lu = self.nodes[u as usize].flag;
                let lv = self.nodes[v as usize].flag;
                let root_u = if lu == FREE { NONE } else { self.find_tree_root(u) };
                let root_v = if lv == FREE { NONE } else { self.find_tree_root(v) };
                let var_u = if root_u != NONE { root_to_var[root_u as usize] } else { usize::MAX };
                let var_v = if root_v != NONE { root_to_var[root_v as usize] } else { usize::MAX };
                let slack = self.edges[e].slack;
                let eps_u = self.tree_eps(root_u);
                let eps_v = self.tree_eps(root_v);
                let processed_u = self.nodes[u as usize].is_processed;
                let processed_v = self.nodes[v as usize].is_processed;

                match (lu, lv) {
                    (PLUS, PLUS)
                        if var_u != usize::MAX
                            && var_v != usize::MAX
                            && processed_u
                            && processed_v =>
                    {
                        if var_u == var_v {
                            local_caps[var_u] = local_caps[var_u].min(slack / 2 - eps_u);
                        } else {
                            let eps00 = slack - eps_u - eps_v;
                            let uv = pair_idx(var_u, var_v);
                            let vu = pair_idx(var_v, var_u);
                            pair_eps00[uv] = pair_eps00[uv].min(eps00);
                            pair_eps00[vu] = pair_eps00[vu].min(eps00);
                        }
                    }
                    (PLUS, MINUS)
                        if var_u != usize::MAX
                            && var_v != usize::MAX
                            && var_u != var_v
                            && processed_u =>
                    {
                        let uv = pair_idx(var_u, var_v);
                        pair_eps01[uv] = pair_eps01[uv].min(slack - eps_u + eps_v);
                    }
                    (MINUS, PLUS)
                        if var_u != usize::MAX
                            && var_v != usize::MAX
                            && var_u != var_v
                            && processed_v =>
                    {
                        let vu = pair_idx(var_v, var_u);
                        pair_eps01[vu] = pair_eps01[vu].min(slack - eps_v + eps_u);
                    }
                    _ => {}
                }
            }

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

                    for t2 in 0..roots.len() {
                        if t == t2 {
                            continue;
                        }

                        let eps00 = pair_eps00[pair_idx(t, t2)];
                        if marks[t2] == start {
                            if eps00 < inf_cap {
                                eps = eps.min(eps00 / 2);
                            }
                            continue;
                        }

                        let eps01_forward = pair_eps01[pair_idx(t, t2)];
                        let eps01_reverse = pair_eps01[pair_idx(t2, t)];

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
                            continue;
                        };

                        if eps00 < inf_cap {
                            eps = eps.min(eps00 - eps2);
                        }
                        if eps01_forward < inf_cap {
                            eps = eps.min(eps2 + eps01_forward);
                        }
                    }
                }

                if eps >= inf_cap {
                    unbounded_component = true;
                    break;
                }

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
        self.scratch.dual_pair_eps00 = pair_eps00;
        self.scratch.dual_pair_eps01 = pair_eps01;
        self.scratch.dual_deltas = deltas;
        self.scratch.dual_marks = marks;
        self.scratch.dual_queue = queue;
        self.scratch.dual_component = component;

        result
    }

    /// Resolve blossom internal matchings (C++ Finish(), PMmain.cpp lines
    /// 8-53).
    fn finish(&mut self) {
        for v in 0..self.node_num {
            let marc = self.nodes[v].match_arc;
            if marc != NONE && (arc_edge(marc) as usize) < self.edge_num {
                let mate = self.arc_head_raw(marc);
                if mate != NONE
                    && (mate as usize) < self.node_num
                    && !self.nodes[mate as usize].is_blossom
                {
                    continue;
                }
            }

            // Walk up blossom_parent, reverse-link via blossom_grandparent
            let mut b_prev: u32 = NONE;
            let mut b = v as u32;
            let mut found = false;
            loop {
                self.nodes[b as usize].blossom_grandparent = b_prev;
                b_prev = b;
                let parent = self.nodes[b as usize].blossom_parent;
                if parent == NONE {
                    break;
                }
                b = parent;
                let bm = self.nodes[b as usize].match_arc;
                if bm != NONE && (arc_edge(bm) as usize) < self.edge_num {
                    found = true;
                    break;
                }
            }
            if !found {
                continue;
            }

            let mut b_prev_prev = self.nodes[b_prev as usize].blossom_grandparent;

            loop {
                let bm = self.nodes[b as usize].match_arc;
                if bm == NONE {
                    break;
                }
                let me = arc_edge(bm) as usize;
                if me >= self.edge_num {
                    break;
                }
                let md = arc_dir(bm);
                let tail_orig = self.edges[me].head0[md];

                // Find k: direct child of b containing tail_orig
                let mut k = tail_orig;
                while self.nodes[k as usize].blossom_parent != b {
                    let p = self.nodes[k as usize].blossom_parent;
                    if p == NONE {
                        break;
                    }
                    k = p;
                }

                // Transfer match from b to k
                self.nodes[k as usize].match_arc = bm;

                let k_sib = self.nodes[k as usize].blossom_sibling_arc;
                if k_sib != NONE {
                    let mut i = self.arc_head_raw(k_sib);
                    let mut limit = self.nodes.len();
                    while i != k && limit > 0 {
                        limit -= 1;
                        let i_sib = self.nodes[i as usize].blossom_sibling_arc;
                        if i_sib == NONE {
                            break;
                        }
                        self.nodes[i as usize].match_arc = i_sib;
                        let j = self.arc_head_raw(i_sib);
                        self.nodes[j as usize].match_arc = arc_rev(i_sib);
                        let j_sib = self.nodes[j as usize].blossom_sibling_arc;
                        if j_sib == NONE {
                            break;
                        }
                        i = self.arc_head_raw(j_sib);
                    }
                }

                // Move to next blossom level down
                b = b_prev;
                if !self.nodes[b as usize].is_blossom {
                    break;
                }
                b_prev = b_prev_prev;
                if b_prev == NONE {
                    break;
                }
                b_prev_prev = self.nodes[b_prev as usize].blossom_grandparent;
            }
        }
    }

    /// Convert internal matching to output format.
    fn into_pairs_checked(mut self) -> MatchingResult<M::RowIndex, M::ColumnIndex> {
        self.finish();
        let mut pairs = Vec::with_capacity(self.node_num / 2);
        for v in 0..self.node_num {
            let arc = self.nodes[v].match_arc;
            if arc == NONE {
                continue;
            }
            let e = arc_edge(arc) as usize;
            let u = self.edges[e].head0[0] as usize;
            let w = self.edges[e].head0[1] as usize;
            if v == u && u < w {
                let ri = M::RowIndex::try_from(u).ok().expect("valid row index");
                let ci = M::ColumnIndex::try_from(w).ok().expect("valid col index");
                pairs.push((ri, ci));
            } else if v == w && w < u {
                let ri = M::RowIndex::try_from(w).ok().expect("valid row index");
                let ci = M::ColumnIndex::try_from(u).ok().expect("valid col index");
                pairs.push((ri, ci));
            }
        }
        pairs.sort_unstable();

        if pairs.len() != self.node_num / 2 {
            return Err(BlossomVError::NoPerfectMatching);
        }

        let mut used = vec![false; self.node_num];
        for &(u, v) in &pairs {
            let uu: usize = u.as_();
            let vv: usize = v.as_();
            if uu >= self.node_num || vv >= self.node_num || uu == vv || used[uu] || used[vv] {
                return Err(BlossomVError::NoPerfectMatching);
            }
            used[uu] = true;
            used[vv] = true;
        }

        if used.iter().any(|&seen| !seen) {
            return Err(BlossomVError::NoPerfectMatching);
        }

        Ok(pairs)
    }

    /// Convert internal matching to output format.
    #[cfg(test)]
    fn into_pairs(mut self) -> Vec<(M::RowIndex, M::ColumnIndex)> {
        self.finish();
        let mut pairs = Vec::with_capacity(self.node_num / 2);
        for v in 0..self.node_num {
            let arc = self.nodes[v].match_arc;
            if arc == NONE {
                continue;
            }
            let e = arc_edge(arc) as usize;
            let u = self.edges[e].head0[0] as usize;
            let w = self.edges[e].head0[1] as usize;
            if v == u && u < w {
                let ri = M::RowIndex::try_from(u).ok().expect("valid row index");
                let ci = M::ColumnIndex::try_from(w).ok().expect("valid col index");
                pairs.push((ri, ci));
            } else if v == w && w < u {
                let ri = M::RowIndex::try_from(w).ok().expect("valid row index");
                let ci = M::ColumnIndex::try_from(u).ok().expect("valid col index");
                pairs.push((ri, ci));
            }
        }
        pairs.sort_unstable();
        pairs
    }
}

// ===== Test accessors =====

#[cfg(test)]
#[allow(
    clippy::cast_lossless,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::if_not_else,
    clippy::ignored_unit_patterns,
    clippy::items_after_statements,
    clippy::let_unit_value,
    clippy::map_unwrap_or,
    clippy::match_same_arms,
    clippy::similar_names,
    clippy::stable_sort_primitive,
    clippy::too_many_lines,
    clippy::uninlined_format_args,
    clippy::useless_vec
)]
mod tests {
    use std::boxed::Box;

    use super::{
        test_support::{SchedulerMirrorTestExt, TestAccessorExt},
        *,
    };
    use crate::{
        impls::ValuedCSR2D,
        traits::{MatrixMut, SparseMatrixMut, algorithms::blossom_v::BlossomV},
    };

    type Vcsr = ValuedCSR2D<usize, usize, usize, i32>;

    fn build_graph(n: usize, edges: &[(usize, usize, i32)]) -> Vcsr {
        let mut sorted: Vec<(usize, usize, i32)> = Vec::new();
        for &(i, j, w) in edges {
            if i == j {
                continue;
            }
            let (lo, hi) = if i < j { (i, j) } else { (j, i) };
            sorted.push((lo, hi, w));
            sorted.push((hi, lo, w));
        }
        sorted.sort_unstable();
        sorted.dedup_by(|a, b| a.0 == b.0 && a.1 == b.1);
        let mut vcsr: Vcsr = SparseMatrixMut::with_sparse_shaped_capacity((n, n), sorted.len());
        for (r, c, v) in sorted {
            MatrixMut::add(&mut vcsr, (r, c, v)).unwrap();
        }
        vcsr
    }

    fn case_474_edges() -> Vec<(usize, usize, i32)> {
        vec![
            (0, 1, 11),
            (2, 4, 53),
            (3, 5, -49),
            (1, 4, 88),
            (2, 3, -27),
            (1, 2, -42),
            (0, 5, 96),
            (4, 5, 33),
            (3, 4, -62),
            (0, 2, 43),
        ]
    }

    fn case_9_edges() -> Vec<(usize, usize, i32)> {
        vec![
            (0, 1, 0),
            (0, 3, -21251),
            (0, 6, -2023),
            (0, 9, 14768),
            (0, 12, 12819),
            (0, 14, 0),
            (0, 15, 0),
            (0, 16, -27420),
            (0, 17, -26215),
            (1, 3, -1),
            (1, 5, 32512),
            (1, 9, -30271),
            (1, 10, 5020),
            (1, 13, 12937),
            (2, 3, 2303),
            (2, 4, 100),
            (2, 14, 76),
            (2, 16, 26984),
            (2, 17, -20523),
            (3, 4, 15679),
            (3, 6, -1),
            (3, 12, 3072),
            (3, 15, 22123),
            (3, 16, -13726),
            (4, 5, 2752),
            (4, 8, 26125),
            (4, 17, -18671),
            (5, 8, 12331),
            (5, 14, -10251),
            (6, 7, -30029),
            (6, 10, -10397),
            (6, 11, -23283),
            (7, 9, 13364),
            (8, 9, -2846),
            (8, 10, -1387),
            (8, 12, -24415),
            (8, 15, -18235),
            (9, 10, -26215),
            (9, 13, 21062),
            (9, 14, -26215),
            (9, 16, -18577),
            (10, 11, -12279),
            (10, 13, -8642),
            (11, 13, -7374),
            (11, 14, 32018),
            (12, 14, 14393),
            (12, 15, -24),
            (12, 17, 50),
            (14, 17, 1128),
        ]
    }

    fn case_97_edges() -> Vec<(usize, usize, i32)> {
        vec![
            (0, 1, 94),
            (0, 2, 62),
            (0, 3, -67),
            (0, 4, -71),
            (0, 5, -32),
            (0, 6, 71),
            (0, 7, 47),
            (0, 8, -70),
            (0, 9, 32),
            (0, 10, 85),
            (0, 11, 71),
            (0, 12, -43),
            (1, 2, 99),
            (1, 3, 14),
            (1, 4, 82),
            (1, 5, 71),
            (1, 7, 65),
            (1, 8, 99),
            (1, 9, -85),
            (1, 17, 43),
            (2, 3, -82),
            (2, 4, 74),
            (2, 6, -8),
            (2, 10, 27),
            (2, 11, 40),
            (2, 16, 41),
            (2, 17, -40),
            (3, 4, -6),
            (3, 5, 56),
            (3, 6, -6),
            (3, 7, -12),
            (3, 8, 26),
            (3, 11, 94),
            (3, 12, 19),
            (3, 13, -95),
            (3, 14, -7),
            (3, 15, -77),
            (3, 17, -74),
            (4, 5, 65),
            (4, 6, 23),
            (4, 7, -21),
            (4, 11, 37),
            (4, 12, -83),
            (4, 14, -100),
            (5, 13, -19),
            (5, 15, 57),
            (6, 9, -91),
            (7, 8, -11),
            (7, 9, -16),
            (7, 14, -76),
            (7, 15, 95),
            (8, 10, -86),
            (8, 13, 3),
            (8, 14, -14),
            (8, 16, 11),
            (9, 10, -5),
            (9, 12, 41),
            (9, 15, 36),
            (10, 13, 73),
            (10, 16, 35),
            (11, 16, 74),
            (13, 17, 93),
        ]
    }

    fn case_honggfuzz_sigabrt_4_edges() -> Vec<(usize, usize, i32)> {
        vec![
            (0, 1, -29481),
            (0, 4, -23385),
            (0, 6, -9713),
            (0, 13, 3660),
            (0, 14, 13857),
            (0, 16, 0),
            (0, 18, 0),
            (0, 19, -8688),
            (0, 20, 29128),
            (0, 21, -1),
            (1, 14, 10906),
            (1, 17, -28356),
            (1, 20, 0),
            (2, 4, -27066),
            (2, 6, -9498),
            (2, 11, 17867),
            (2, 12, 0),
            (2, 13, -14016),
            (2, 16, 130),
            (2, 17, 7281),
            (2, 18, 32281),
            (2, 19, -16009),
            (3, 5, 6243),
            (3, 7, -18728),
            (3, 9, 3233),
            (3, 16, 28116),
            (4, 12, -6480),
            (5, 11, -28628),
            (5, 14, -12713),
            (5, 20, 17905),
            (6, 8, -7974),
            (6, 20, -30597),
            (6, 21, -23196),
            (6, 22, -27428),
            (7, 19, 0),
            (7, 21, -61),
            (8, 24, -6547),
            (8, 25, 1),
            (9, 14, -17579),
            (9, 17, 30917),
            (9, 22, -19162),
            (10, 15, -18927),
            (10, 19, -429),
            (10, 24, 12562),
            (11, 19, -19309),
            (11, 21, 0),
            (12, 15, -5228),
            (12, 20, 17077),
            (13, 21, -31234),
            (14, 17, 64),
            (14, 21, 4843),
            (14, 22, -16020),
            (14, 24, 16426),
            (15, 16, 0),
            (15, 20, 2492),
            (16, 18, -29415),
            (16, 23, 10546),
            (17, 24, 30942),
            (18, 23, 9509),
            (19, 21, 0),
            (19, 22, -12607),
            (24, 25, -22204),
        ]
    }

    fn case_honggfuzz_sigabrt_5_edges() -> Vec<(usize, usize, i32)> {
        vec![
            (0, 2, 54),
            (0, 4, 0),
            (0, 7, 364),
            (0, 11, 0),
            (0, 12, 22101),
            (0, 13, 1),
            (0, 16, 0),
            (0, 18, 2816),
            (0, 19, 24275),
            (0, 20, 21398),
            (0, 21, 0),
            (0, 24, 8379),
            (1, 4, 30776),
            (1, 6, 1),
            (1, 7, -628),
            (1, 18, -15828),
            (1, 23, 110),
            (2, 3, 8239),
            (2, 7, -14876),
            (2, 9, 455),
            (2, 11, 17867),
            (2, 16, 13954),
            (2, 17, 3199),
            (2, 22, 4058),
            (3, 7, -18728),
            (3, 9, -13058),
            (3, 22, -15953),
            (4, 5, -16511),
            (5, 7, 21845),
            (5, 11, -15360),
            (5, 22, 2816),
            (5, 24, 0),
            (6, 9, 27985),
            (6, 12, -20450),
            (6, 14, 381),
            (6, 22, 2636),
            (6, 24, 31716),
            (7, 8, 21589),
            (7, 14, -15413),
            (7, 17, 29485),
            (8, 11, 896),
            (8, 15, -318),
            (9, 12, -21845),
            (9, 18, 13613),
            (9, 19, 25273),
            (9, 22, -25404),
            (10, 11, -9253),
            (10, 17, -32074),
            (11, 15, -28291),
            (12, 16, 27181),
            (12, 21, 0),
            (12, 23, -5228),
            (12, 25, -10034),
            (13, 16, 16334),
            (13, 19, 6597),
            (13, 20, -11177),
            (13, 22, 19534),
            (14, 21, -25631),
            (14, 24, -12246),
            (16, 18, -29415),
            (16, 25, -28375),
            (17, 24, 2782),
            (18, 23, 881),
            (21, 24, 124),
            (22, 25, 21),
        ]
    }

    fn case_honggfuzz_sigabrt_6_edges() -> Vec<(usize, usize, i32)> {
        vec![
            (0, 2, 54),
            (0, 4, 0),
            (0, 7, 364),
            (0, 11, 0),
            (0, 12, 22101),
            (0, 13, 1),
            (0, 16, 0),
            (0, 18, 2816),
            (0, 19, 24275),
            (0, 21, 0),
            (0, 24, 8379),
            (1, 4, 30776),
            (1, 6, 1),
            (1, 7, -628),
            (1, 10, 12302),
            (1, 18, -15828),
            (1, 23, 110),
            (2, 3, 8239),
            (2, 7, -14876),
            (2, 9, 455),
            (2, 11, 17867),
            (2, 17, 3199),
            (2, 22, 4058),
            (3, 7, -18728),
            (3, 9, -13058),
            (3, 22, -15953),
            (4, 5, -16511),
            (4, 11, 31704),
            (5, 7, 21845),
            (5, 11, -15360),
            (5, 22, 2816),
            (5, 24, 0),
            (6, 9, 27985),
            (6, 12, -20450),
            (6, 14, 381),
            (6, 22, 2636),
            (6, 24, 31716),
            (7, 8, 21589),
            (7, 14, -15413),
            (8, 11, 896),
            (8, 15, -318),
            (8, 17, 19387),
            (9, 12, -21845),
            (9, 18, 13613),
            (9, 19, 25273),
            (9, 22, -25404),
            (10, 11, 23441),
            (10, 17, -32074),
            (11, 15, -28291),
            (12, 16, 27181),
            (12, 21, 0),
            (12, 23, -5228),
            (12, 25, -10034),
            (13, 16, 16334),
            (13, 19, 6597),
            (13, 20, -11177),
            (13, 22, 19534),
            (13, 24, -10229),
            (14, 21, -25631),
            (14, 24, -12246),
            (16, 18, -29415),
            (16, 25, -28375),
            (17, 24, 2782),
            (18, 23, 881),
            (21, 24, 124),
            (22, 25, 21),
        ]
    }

    fn case_honggfuzz_sigabrt_14_edges() -> Vec<(usize, usize, i32)> {
        vec![
            (0, 1, 0),
            (0, 2, 0),
            (0, 3, -21251),
            (0, 5, -18577),
            (0, 6, 32018),
            (0, 8, -31624),
            (0, 10, -1387),
            (0, 11, -2023),
            (0, 12, 12819),
            (0, 14, 21845),
            (0, 15, 0),
            (0, 16, -26363),
            (0, 17, 0),
            (1, 2, -12194),
            (1, 3, -10864),
            (1, 4, 0),
            (1, 5, 32512),
            (1, 7, 0),
            (1, 11, 0),
            (1, 13, 12937),
            (2, 3, 2303),
            (2, 5, 0),
            (2, 15, 13302),
            (2, 16, 26984),
            (2, 17, -20523),
            (3, 4, 15679),
            (3, 10, -1),
            (3, 12, 3072),
            (3, 14, 2920),
            (3, 15, 22123),
            (3, 16, -13726),
            (4, 16, -320),
            (5, 8, 12331),
            (5, 13, 21356),
            (5, 14, 3053),
            (6, 7, -30029),
            (6, 10, -10397),
            (6, 11, -23283),
            (7, 10, -768),
            (7, 11, -26516),
            (8, 9, 32738),
            (8, 12, -24415),
            (8, 13, 17552),
            (8, 15, -18235),
            (9, 10, -26215),
            (9, 13, 21062),
            (9, 16, 0),
            (10, 11, -12279),
            (10, 13, -8642),
            (12, 15, 0),
            (12, 17, -16846),
            (13, 16, 12857),
            (14, 17, 7981),
        ]
    }

    fn case_honggfuzz_sigabrt_7_edges() -> Vec<(usize, usize, i32)> {
        vec![
            (0, 2, 54),
            (0, 4, 0),
            (0, 7, 364),
            (0, 12, 22101),
            (0, 13, 1),
            (0, 15, 0),
            (0, 16, 0),
            (0, 18, 2816),
            (0, 19, 24275),
            (0, 21, 0),
            (0, 24, 8379),
            (1, 4, 30776),
            (1, 6, 1),
            (1, 7, -628),
            (1, 10, 12302),
            (1, 18, -15828),
            (1, 23, 110),
            (2, 3, 8239),
            (2, 7, -14876),
            (2, 9, 455),
            (2, 11, 17867),
            (2, 16, 14210),
            (2, 22, 4058),
            (3, 7, -18728),
            (3, 9, -13058),
            (3, 22, -15953),
            (4, 5, -16511),
            (4, 11, 31704),
            (5, 7, 21845),
            (5, 11, 0),
            (5, 22, 2816),
            (5, 24, 0),
            (6, 9, 27985),
            (6, 12, -20450),
            (6, 22, 2636),
            (6, 24, 31716),
            (7, 8, 21589),
            (7, 14, -15413),
            (7, 17, 29485),
            (8, 11, 896),
            (8, 15, -318),
            (9, 12, -21845),
            (9, 18, 13613),
            (9, 19, 25273),
            (9, 22, -25404),
            (10, 11, 23441),
            (10, 17, -32074),
            (11, 15, -28291),
            (12, 16, 27181),
            (12, 21, 0),
            (12, 23, -5228),
            (12, 25, -10034),
            (13, 16, 16334),
            (13, 19, 6597),
            (13, 20, -11177),
            (13, 22, 19534),
            (14, 21, -25631),
            (14, 24, -12246),
            (16, 18, -29415),
            (16, 25, -28375),
            (17, 20, 381),
            (17, 24, 2782),
            (18, 20, 21398),
            (18, 23, 881),
            (21, 24, 124),
            (22, 25, 21),
        ]
    }

    fn case_87417_edges() -> Vec<(usize, usize, i32)> {
        vec![
            (0, 11, -88),
            (0, 17, -39),
            (0, 18, 19),
            (0, 21, -22),
            (0, 23, -54),
            (0, 24, -25),
            (0, 27, 12),
            (1, 12, -8),
            (1, 15, -97),
            (1, 16, -32),
            (1, 24, 55),
            (2, 10, -49),
            (2, 15, -74),
            (2, 16, -47),
            (3, 12, 100),
            (3, 21, -65),
            (3, 29, -8),
            (4, 9, -45),
            (4, 11, -73),
            (4, 13, -97),
            (4, 15, 81),
            (4, 16, 4),
            (4, 21, -100),
            (4, 22, -25),
            (4, 27, -27),
            (4, 29, -77),
            (5, 9, 16),
            (5, 13, 57),
            (5, 26, -96),
            (5, 27, 63),
            (6, 9, 67),
            (6, 12, 82),
            (6, 13, 44),
            (6, 15, -49),
            (6, 16, 1),
            (7, 8, -8),
            (7, 11, -77),
            (7, 12, -51),
            (7, 13, -50),
            (7, 15, -77),
            (7, 17, 20),
            (7, 19, -21),
            (7, 20, 53),
            (7, 21, -50),
            (7, 22, 64),
            (7, 23, 81),
            (8, 14, -83),
            (8, 15, -25),
            (8, 18, 99),
            (8, 22, -51),
            (8, 25, -32),
            (8, 27, 84),
            (8, 29, 78),
            (9, 11, 78),
            (9, 18, 32),
            (9, 23, -71),
            (9, 29, 2),
            (10, 14, -57),
            (10, 23, -89),
            (11, 14, -22),
            (11, 16, 10),
            (11, 17, 87),
            (11, 20, -91),
            (11, 23, 17),
            (11, 24, -39),
            (11, 26, 11),
            (11, 27, 22),
            (12, 22, 87),
            (12, 23, -83),
            (12, 24, 75),
            (12, 25, -36),
            (12, 27, 12),
            (12, 28, 51),
            (13, 20, 26),
            (13, 22, -58),
            (13, 27, 26),
            (13, 28, -57),
            (14, 15, 41),
            (14, 22, -73),
            (14, 25, -63),
            (14, 26, 73),
            (14, 29, -50),
            (15, 16, 11),
            (15, 18, 28),
            (15, 19, -19),
            (15, 26, -5),
            (15, 29, 73),
            (16, 19, -67),
            (16, 20, -85),
            (16, 21, -29),
            (16, 25, -99),
            (17, 19, -79),
            (17, 24, 85),
            (18, 23, 5),
            (18, 28, 79),
            (19, 20, -86),
            (19, 21, 77),
            (20, 23, -19),
            (20, 25, -38),
            (20, 26, -83),
            (20, 28, 13),
            (21, 22, 31),
            (21, 27, -12),
            (21, 28, 97),
            (21, 29, -97),
            (22, 23, 41),
            (23, 28, -22),
            (24, 26, -76),
            (25, 28, 57),
            (26, 28, 39),
            (27, 28, 27),
        ]
    }

    fn case_416_edges() -> Vec<(usize, usize, i32)> {
        vec![
            (0, 1, 5),
            (0, 3, 65),
            (0, 5, 96),
            (0, 6, 63),
            (0, 8, -85),
            (0, 9, -65),
            (0, 11, -12),
            (0, 13, 0),
            (0, 15, 34),
            (0, 18, -21),
            (0, 19, 64),
            (1, 2, -3),
            (1, 3, 20),
            (1, 4, -88),
            (1, 6, -5),
            (1, 9, -23),
            (1, 10, 86),
            (1, 12, 56),
            (1, 13, 53),
            (1, 15, -50),
            (1, 18, 54),
            (1, 19, 32),
            (2, 5, -30),
            (2, 8, -82),
            (2, 9, -3),
            (2, 10, 38),
            (2, 12, -1),
            (2, 14, -43),
            (2, 15, 21),
            (2, 16, -61),
            (2, 17, 74),
            (3, 4, 2),
            (3, 5, -51),
            (3, 7, 94),
            (3, 8, 12),
            (3, 9, -48),
            (3, 14, -39),
            (3, 16, -57),
            (4, 6, -59),
            (4, 7, 18),
            (4, 8, -70),
            (4, 9, -92),
            (4, 14, 75),
            (4, 17, -89),
            (4, 18, -81),
            (5, 6, 40),
            (5, 7, -48),
            (5, 8, 17),
            (5, 9, 33),
            (5, 10, -5),
            (5, 13, 25),
            (6, 7, 33),
            (6, 8, -94),
            (6, 9, 66),
            (6, 11, 71),
            (6, 12, 98),
            (6, 15, -47),
            (6, 17, 87),
            (6, 18, 75),
            (7, 8, -15),
            (7, 9, 82),
            (7, 10, 35),
            (7, 12, -46),
            (7, 13, -63),
            (7, 14, 89),
            (7, 15, -79),
            (7, 17, 6),
            (7, 18, 15),
            (8, 9, -50),
            (8, 10, -36),
            (8, 11, -20),
            (8, 12, 74),
            (8, 14, 46),
            (8, 16, 98),
            (8, 19, 33),
            (9, 11, -92),
            (9, 12, 92),
            (9, 13, 85),
            (9, 14, 92),
            (9, 15, 23),
            (9, 17, -5),
            (10, 11, 50),
            (10, 12, -32),
            (10, 13, -14),
            (10, 14, -48),
            (10, 15, -74),
            (10, 16, 2),
            (10, 18, -85),
            (10, 19, -36),
            (11, 12, 59),
            (11, 13, 73),
            (11, 14, -94),
            (11, 15, 70),
            (11, 16, 8),
            (11, 18, 35),
            (11, 19, 71),
            (12, 14, -59),
            (12, 15, -86),
            (12, 16, 59),
            (12, 17, 54),
            (12, 18, 11),
            (12, 19, 57),
            (13, 14, 71),
            (13, 15, 57),
            (13, 16, 17),
            (13, 17, 71),
            (13, 19, -52),
            (14, 16, 73),
            (14, 17, -94),
            (14, 18, -31),
            (14, 19, -90),
            (15, 16, 92),
            (15, 17, 87),
            (15, 19, 77),
            (16, 17, -47),
            (16, 18, 75),
            (16, 19, -23),
            (17, 18, -5),
            (17, 19, 89),
            (18, 19, -93),
        ]
    }

    fn case_1594_edges() -> Vec<(usize, usize, i32)> {
        vec![
            (0, 3, 70),
            (0, 5, -38),
            (0, 6, 0),
            (0, 7, 11),
            (0, 10, 40),
            (0, 12, -37),
            (0, 13, 51),
            (0, 14, -21),
            (0, 15, 88),
            (0, 16, 33),
            (0, 18, -35),
            (0, 19, -77),
            (0, 20, 96),
            (0, 23, -71),
            (0, 24, 34),
            (1, 5, 2),
            (1, 6, 63),
            (1, 7, -38),
            (1, 10, -8),
            (1, 11, -82),
            (1, 14, 23),
            (1, 20, -89),
            (1, 21, -63),
            (1, 22, -29),
            (1, 23, 15),
            (2, 4, 95),
            (2, 6, 35),
            (2, 8, -9),
            (2, 11, 94),
            (2, 12, 85),
            (2, 14, -33),
            (2, 15, 65),
            (2, 16, 32),
            (2, 19, -44),
            (2, 20, -93),
            (2, 24, -19),
            (2, 25, 47),
            (3, 4, 83),
            (3, 5, -4),
            (3, 6, -31),
            (3, 7, 66),
            (3, 8, 82),
            (3, 10, 58),
            (3, 11, -14),
            (3, 14, 87),
            (3, 15, -31),
            (3, 20, 95),
            (3, 22, -48),
            (4, 5, 73),
            (4, 6, -79),
            (4, 7, -63),
            (4, 9, -84),
            (4, 10, 4),
            (4, 11, 12),
            (4, 13, -83),
            (4, 14, -45),
            (4, 16, 33),
            (4, 17, -52),
            (4, 19, 54),
            (4, 21, -22),
            (4, 22, -68),
            (4, 24, 18),
            (4, 25, -61),
            (5, 6, 44),
            (5, 7, 22),
            (5, 10, 24),
            (5, 12, 53),
            (5, 14, -73),
            (5, 19, -45),
            (5, 20, -53),
            (5, 24, -25),
            (6, 10, 44),
            (6, 12, 92),
            (6, 13, 100),
            (6, 14, -63),
            (6, 15, -3),
            (6, 17, 87),
            (6, 19, 96),
            (6, 20, 89),
            (6, 21, 52),
            (6, 22, 89),
            (6, 23, -5),
            (6, 24, -91),
            (6, 25, -55),
            (7, 8, -97),
            (7, 9, -41),
            (7, 10, -95),
            (7, 12, 30),
            (7, 14, 22),
            (7, 15, -22),
            (7, 16, -22),
            (7, 17, 21),
            (7, 20, 79),
            (7, 22, -77),
            (7, 23, 83),
            (7, 24, -28),
            (7, 25, 36),
            (8, 11, 100),
            (8, 12, -83),
            (8, 13, 76),
            (8, 15, -95),
            (8, 16, 63),
            (8, 18, -79),
            (8, 19, 74),
            (8, 20, 63),
            (8, 21, -26),
            (8, 23, 81),
            (8, 25, -63),
            (9, 10, -44),
            (9, 12, -98),
            (9, 13, 17),
            (9, 14, 100),
            (9, 15, 85),
            (9, 16, 38),
            (9, 18, -79),
            (9, 19, -27),
            (9, 20, 53),
            (9, 23, 90),
            (9, 24, -98),
            (9, 25, 27),
            (10, 14, 94),
            (10, 15, 29),
            (10, 16, -64),
            (10, 21, 2),
            (10, 23, 42),
            (10, 25, -85),
            (11, 14, 72),
            (11, 15, 79),
            (11, 16, 99),
            (11, 19, 36),
            (11, 23, -50),
            (11, 25, -61),
            (12, 15, 33),
            (12, 17, 40),
            (12, 19, 94),
            (12, 20, 50),
            (12, 22, 31),
            (12, 25, -23),
            (13, 14, -61),
            (13, 16, 96),
            (13, 17, -58),
            (13, 18, 24),
            (13, 20, -17),
            (13, 22, -42),
            (13, 24, -24),
            (13, 25, 23),
            (14, 15, -51),
            (14, 23, -53),
            (14, 25, -17),
            (15, 16, 6),
            (15, 17, 69),
            (15, 22, 97),
            (15, 24, -79),
            (16, 18, -91),
            (16, 21, 77),
            (16, 22, 9),
            (16, 25, -63),
            (17, 20, 50),
            (17, 21, -64),
            (17, 22, 46),
            (17, 24, 59),
            (17, 25, -19),
            (18, 19, 1),
            (18, 20, -41),
            (18, 22, 63),
            (18, 23, 86),
            (18, 24, 11),
            (18, 25, -55),
            (19, 23, 30),
            (19, 24, 65),
            (20, 23, 18),
            (20, 24, 72),
            (20, 25, -32),
            (21, 23, 41),
            (21, 25, 57),
            (22, 25, 75),
        ]
    }

    fn case_232_edges() -> Vec<(usize, usize, i32)> {
        vec![
            (0, 6, 13),
            (0, 9, 97),
            (0, 13, 80),
            (0, 15, -73),
            (0, 16, 42),
            (0, 18, -15),
            (0, 19, 84),
            (0, 20, -3),
            (0, 21, -31),
            (0, 22, 90),
            (0, 23, -9),
            (0, 26, 82),
            (0, 27, 0),
            (1, 9, 17),
            (1, 11, -69),
            (1, 18, 79),
            (1, 19, -12),
            (1, 20, -12),
            (1, 21, 2),
            (1, 23, 92),
            (1, 24, -86),
            (1, 25, -37),
            (1, 27, 45),
            (2, 3, 32),
            (2, 6, 59),
            (2, 7, 88),
            (2, 8, 63),
            (2, 9, 82),
            (2, 11, 89),
            (2, 15, 25),
            (2, 22, -8),
            (2, 23, -75),
            (2, 24, 15),
            (3, 8, 85),
            (3, 9, 7),
            (3, 10, 80),
            (3, 15, -58),
            (3, 19, 42),
            (3, 20, 29),
            (3, 24, -2),
            (3, 27, -96),
            (4, 12, 96),
            (4, 13, -53),
            (4, 16, -79),
            (4, 21, 93),
            (4, 22, 79),
            (4, 23, -27),
            (4, 24, -95),
            (5, 6, 14),
            (5, 8, -91),
            (5, 9, -20),
            (5, 12, -59),
            (5, 13, 5),
            (5, 16, -21),
            (5, 19, -68),
            (5, 22, -19),
            (5, 23, 74),
            (6, 9, -63),
            (6, 16, -39),
            (6, 17, -77),
            (6, 19, -23),
            (6, 20, -77),
            (6, 23, -69),
            (6, 24, -1),
            (6, 27, -58),
            (7, 8, -97),
            (7, 15, -5),
            (7, 17, 65),
            (7, 18, -88),
            (7, 22, 43),
            (7, 23, -30),
            (7, 26, -31),
            (8, 9, -77),
            (8, 10, 83),
            (8, 11, 27),
            (8, 13, -84),
            (8, 14, 48),
            (8, 18, 52),
            (8, 19, 24),
            (8, 20, 28),
            (8, 21, -5),
            (8, 25, 71),
            (8, 26, 8),
            (8, 27, 23),
            (9, 18, -49),
            (9, 19, -60),
            (9, 20, -60),
            (9, 22, 52),
            (9, 23, 100),
            (9, 24, -5),
            (9, 25, -88),
            (10, 11, -37),
            (10, 12, 54),
            (10, 14, -42),
            (10, 15, -59),
            (10, 17, 88),
            (10, 18, 35),
            (10, 19, 4),
            (10, 20, 15),
            (10, 21, 39),
            (10, 22, -24),
            (10, 24, 84),
            (10, 27, -35),
            (11, 15, -46),
            (11, 16, -98),
            (11, 17, 67),
            (11, 18, -16),
            (11, 23, 95),
            (11, 24, -27),
            (11, 25, 44),
            (12, 15, 58),
            (12, 19, 64),
            (12, 20, -20),
            (12, 25, 30),
            (12, 27, 55),
            (13, 15, -50),
            (13, 17, -24),
            (13, 19, 41),
            (13, 22, 28),
            (13, 25, -83),
            (14, 18, 6),
            (14, 20, -4),
            (14, 22, -97),
            (14, 23, 14),
            (15, 18, -69),
            (15, 19, -20),
            (15, 20, 42),
            (15, 22, 20),
            (15, 25, 41),
            (15, 26, 37),
            (16, 17, 13),
            (16, 19, 80),
            (16, 20, 45),
            (16, 22, 97),
            (16, 24, 96),
            (16, 25, -25),
            (17, 20, -90),
            (17, 22, 47),
            (17, 23, 22),
            (17, 24, 65),
            (18, 19, -17),
            (18, 24, 72),
            (18, 25, -92),
            (18, 26, 95),
            (19, 22, 86),
            (19, 25, -78),
            (20, 23, -84),
            (20, 24, -20),
            (21, 25, 26),
            (21, 27, -99),
            (22, 23, -42),
            (22, 24, -30),
            (22, 25, -58),
            (22, 27, -60),
            (23, 24, 15),
            (23, 27, -88),
            (24, 25, -71),
        ]
    }

    fn case_4666_edges() -> Vec<(usize, usize, i32)> {
        vec![
            (0, 2, -58),
            (0, 3, 17),
            (0, 4, 38),
            (0, 12, 89),
            (0, 17, 78),
            (0, 22, 5),
            (0, 26, 91),
            (0, 27, 67),
            (0, 28, 4),
            (0, 29, -56),
            (1, 2, -96),
            (1, 3, 99),
            (1, 4, 24),
            (1, 5, -18),
            (1, 20, -65),
            (1, 28, 58),
            (1, 29, -80),
            (2, 3, 43),
            (2, 4, 59),
            (2, 5, 44),
            (2, 6, -96),
            (2, 11, -82),
            (2, 27, 15),
            (2, 28, -72),
            (2, 29, 5),
            (3, 4, 76),
            (3, 5, 52),
            (3, 6, 88),
            (3, 7, 59),
            (3, 29, -13),
            (4, 5, -11),
            (4, 6, 81),
            (4, 7, 96),
            (4, 8, 52),
            (5, 6, -32),
            (5, 8, -44),
            (5, 9, -89),
            (5, 29, 20),
            (6, 7, 59),
            (6, 8, -22),
            (6, 9, 47),
            (6, 10, 3),
            (7, 8, -87),
            (7, 9, -50),
            (7, 10, -50),
            (7, 11, -29),
            (7, 19, 69),
            (7, 22, 14),
            (8, 9, -25),
            (8, 10, -86),
            (8, 12, 58),
            (8, 27, -8),
            (9, 10, 61),
            (9, 11, -78),
            (9, 12, 46),
            (9, 13, -10),
            (9, 14, 7),
            (9, 16, 76),
            (10, 11, 49),
            (10, 12, 60),
            (10, 13, 15),
            (10, 14, 68),
            (10, 28, 58),
            (11, 12, 76),
            (11, 14, 78),
            (11, 15, -39),
            (11, 19, -40),
            (12, 13, 88),
            (12, 14, 29),
            (12, 15, -72),
            (12, 16, 17),
            (13, 14, -57),
            (13, 15, 37),
            (13, 16, -81),
            (13, 17, -46),
            (13, 24, -97),
            (13, 26, 75),
            (14, 16, 49),
            (14, 17, 42),
            (15, 17, -61),
            (15, 18, 87),
            (15, 19, 86),
            (15, 23, -94),
            (16, 17, 94),
            (16, 19, -71),
            (16, 24, -63),
            (16, 25, 3),
            (17, 19, -51),
            (17, 20, -2),
            (17, 27, 94),
            (18, 19, 62),
            (18, 22, 56),
            (18, 26, 32),
            (18, 27, -35),
            (19, 21, 31),
            (19, 22, -10),
            (20, 21, 9),
            (20, 23, 87),
            (20, 24, -21),
            (21, 22, 22),
            (21, 23, 15),
            (21, 24, 20),
            (21, 25, -16),
            (22, 25, -39),
            (22, 29, -70),
            (23, 24, -90),
            (23, 25, -42),
            (23, 26, 53),
            (23, 27, 68),
            (24, 25, -21),
            (24, 26, 34),
            (25, 26, -13),
            (25, 27, -24),
            (25, 28, -84),
            (25, 29, -31),
            (26, 28, 57),
            (26, 29, 78),
            (27, 28, 91),
            (27, 29, -46),
            (28, 29, -2),
        ]
    }

    fn case_4666_graph() -> Vcsr {
        build_graph(30, &case_4666_edges())
    }

    fn stall_generic_phase(state: &mut BlossomVState<Vcsr>, context: &str) {
        let mut generic_steps = 0usize;
        while state.generic_primal_pass_once() {
            generic_steps += 1;
            assert!(generic_steps <= 128, "generic primal phase did not stall in {context}");
        }
    }

    fn generic_primal_pass_without_global_expand_fallback_once(
        state: &mut BlossomVState<Vcsr>,
    ) -> bool {
        let mut root = state.root_list_head;
        while root != NONE {
            let root_usize = root as usize;
            let next_root = state.nodes[root_usize].tree_sibling_next;
            let next_next_root = if next_root != NONE {
                state.nodes[next_root as usize].tree_sibling_next
            } else {
                NONE
            };

            if state.nodes[root_usize].is_outer
                && state.nodes[root_usize].is_tree_root
                && state.process_tree_primal(root)
            {
                return true;
            }

            root = next_root;
            if root != NONE && !state.nodes[root as usize].is_tree_root {
                root = next_next_root;
            }
        }

        false
    }

    fn find_global_expand_fallback_blossom_for_test(state: &BlossomVState<Vcsr>) -> Option<u32> {
        (state.node_num..state.nodes.len())
            .find(|&b| {
                state.nodes[b].is_blossom
                    && state.nodes[b].is_outer
                    && state.nodes[b].flag == MINUS
                    && state.effective_blossom_expand_slack(b as u32) == 0
            })
            .map(|b| b as u32)
    }

    fn case_4666_state_at_dual_before(dual_updates_applied: usize) -> BlossomVState<Vcsr> {
        let g = case_4666_graph();
        let leaked = Box::leak(Box::new(g));
        let mut state = BlossomVState::new(leaked);
        state.init_global();

        for dual_idx in 0..dual_updates_applied {
            state.mark_tree_roots_processed();
            stall_generic_phase(&mut state, "before dual update");
            assert!(state.update_duals(), "dual update {} failed for case #4666", dual_idx + 1);
        }

        state.mark_tree_roots_processed();
        stall_generic_phase(&mut state, "after dual updates");
        state
    }

    fn normalize_pairs(pairs: &[(usize, usize)]) -> Vec<(usize, usize)> {
        let mut normalized =
            pairs.iter().map(|&(u, v)| if u < v { (u, v) } else { (v, u) }).collect::<Vec<_>>();
        normalized.sort_unstable();
        normalized
    }

    fn find_edge_idx(state: &BlossomVState<Vcsr>, a: usize, b: usize) -> u32 {
        let endpoints = if a < b { (a, b) } else { (b, a) };
        (0..state.test_edge_count())
            .find_map(|e| {
                let edge = state.test_edge_endpoints(e);
                let edge = if edge.0 < edge.1 {
                    (edge.0 as usize, edge.1 as usize)
                } else {
                    (edge.1 as usize, edge.0 as usize)
                };
                (edge == endpoints).then_some(e as u32)
            })
            .unwrap_or_else(|| panic!("missing edge ({}, {})", endpoints.0, endpoints.1))
    }

    fn validate_matching(n: usize, matching: &[(usize, usize)]) {
        let mut used = vec![false; n];
        for &(u, v) in matching {
            assert!(u < n, "matching endpoint {u} out of range for n={n}");
            assert!(v < n, "matching endpoint {v} out of range for n={n}");
            assert!(!used[u], "vertex {u} used twice");
            assert!(!used[v], "vertex {v} used twice");
            used[u] = true;
            used[v] = true;
        }
        assert_eq!(matching.len(), n / 2, "matching is not perfect");
    }

    fn matching_cost(edges: &[(usize, usize, i32)], matching: &[(usize, usize)]) -> i32 {
        matching
            .iter()
            .map(|&(u, v)| {
                let (lo, hi) = if u < v { (u, v) } else { (v, u) };
                edges
                    .iter()
                    .find_map(|&(a, b, w)| {
                        let (ea, eb) = if a < b { (a, b) } else { (b, a) };
                        (ea == lo && eb == hi).then_some(w)
                    })
                    .unwrap_or_else(|| panic!("edge ({lo}, {hi}) not found in graph"))
            })
            .sum()
    }

    fn assert_edge_list_invariants(state: &BlossomVState<Vcsr>, phase: &str) {
        let mut seen = vec![[0u8; 2]; state.edge_num];
        let mut in_selfloops = vec![false; state.edge_num];

        for v in 0..state.nodes.len() {
            let mut e = state.nodes[v].blossom_selfloops;
            let mut steps = 0usize;
            while e != NONE {
                assert!(
                    (e as usize) < state.edge_num,
                    "{phase}: node {v} has out-of-range blossom selfloop edge {e}",
                );
                assert!(
                    !in_selfloops[e as usize],
                    "{phase}: edge {e} appears more than once in blossom selfloop chains",
                );
                in_selfloops[e as usize] = true;
                e = state.edges[e as usize].next[0];
                steps += 1;
                assert!(
                    steps <= state.edge_num,
                    "{phase}: node {v} blossom selfloop walk exceeded edge count",
                );
            }
        }

        for v in 0..state.nodes.len() {
            for dir in 0..2usize {
                let first = state.nodes[v].first[dir];
                if first == NONE {
                    continue;
                }

                assert!(
                    (first as usize) < state.edge_num,
                    "{phase}: node {v} dir {dir} has out-of-range first edge {first}",
                );

                let mut e = first;
                let mut steps = 0usize;
                loop {
                    assert_ne!(
                        e, NONE,
                        "{phase}: node {v} dir {dir} reached NONE inside adjacency cycle"
                    );
                    assert!(
                        (e as usize) < state.edge_num,
                        "{phase}: node {v} dir {dir} reached out-of-range edge {e}",
                    );

                    let edge = &state.edges[e as usize];
                    assert_eq!(
                        edge.head[1 - dir],
                        v as u32,
                        "{phase}: node {v} dir {dir} contains edge {e} whose stored endpoint is {:?}",
                        edge.head,
                    );

                    let next = edge.next[dir];
                    let prev = edge.prev[dir];
                    assert_ne!(next, NONE, "{phase}: edge {e} has NONE next in dir {dir}");
                    assert_ne!(prev, NONE, "{phase}: edge {e} has NONE prev in dir {dir}");
                    assert!(
                        (next as usize) < state.edge_num,
                        "{phase}: edge {e} has out-of-range next {next} in dir {dir}",
                    );
                    assert!(
                        (prev as usize) < state.edge_num,
                        "{phase}: edge {e} has out-of-range prev {prev} in dir {dir}",
                    );
                    assert_eq!(
                        state.edges[next as usize].prev[dir], e,
                        "{phase}: edge {e} dir {dir} next {next} does not point back",
                    );
                    assert_eq!(
                        state.edges[prev as usize].next[dir], e,
                        "{phase}: edge {e} dir {dir} prev {prev} does not point forward",
                    );

                    seen[e as usize][dir] = seen[e as usize][dir].saturating_add(1);
                    assert_eq!(
                        seen[e as usize][dir], 1,
                        "{phase}: edge {e} appears more than once in dir {dir} adjacency lists",
                    );

                    e = next;
                    steps += 1;
                    assert!(
                        steps <= state.edge_num,
                        "{phase}: node {v} dir {dir} adjacency walk exceeded edge count",
                    );
                    if e == first {
                        break;
                    }
                }
            }
        }

        for (e_idx, edge) in state.edges.iter().take(state.edge_num).enumerate() {
            for dir in 0..2usize {
                let expected = u8::from(!(in_selfloops[e_idx] || edge.head[1 - dir] == NONE));
                assert_eq!(
                    seen[e_idx][dir], expected,
                    "{phase}: edge {e_idx} dir {dir} seen {} times but head is {:?}",
                    seen[e_idx][dir], edge.head,
                );
            }
        }
    }

    fn solve_case_1594_with_edge_list_checks() -> Vec<(usize, usize)> {
        let edges = case_1594_edges();
        let g = build_graph(26, &edges);
        let mut state = BlossomVState::new(&g);
        assert_edge_list_invariants(&state, "case #1594 after new");

        state.init_global();
        assert_edge_list_invariants(&state, "case #1594 after init_global");

        for outer in 0..5_000usize {
            state.mark_tree_roots_processed();
            assert_edge_list_invariants(
                &state,
                &format!("case #1594 after mark_tree_roots_processed outer {outer}"),
            );
            let mut inner_steps = 0usize;
            loop {
                let progressed = state.generic_primal_pass_once();
                assert_edge_list_invariants(
                    &state,
                    &format!("case #1594 after generic pass outer {outer} inner {inner_steps}"),
                );
                inner_steps += 1;
                if !progressed || state.tree_num == 0 {
                    break;
                }
                assert!(
                    inner_steps <= 50_000,
                    "case #1594 exceeded inner-step budget while checking edge lists",
                );
            }

            if state.tree_num == 0 {
                break;
            }

            let dual_ok = state.update_duals();
            assert_edge_list_invariants(
                &state,
                &format!("case #1594 after dual update outer {outer}"),
            );

            assert!(dual_ok, "case #1594 dual update failed during edge-list check");

            assert!(
                outer < 4_999,
                "case #1594 exceeded outer-step budget while checking edge lists",
            );
        }

        normalize_pairs(&state.into_pairs())
    }

    fn solve_case_232_with_edge_list_checks() -> Vec<(usize, usize)> {
        let edges = case_232_edges();
        let g = build_graph(28, &edges);
        let mut state = BlossomVState::new(&g);
        assert_edge_list_invariants(&state, "case #232 after new");

        state.init_global();
        assert_edge_list_invariants(&state, "case #232 after init_global");

        for outer in 0..5_000usize {
            state.mark_tree_roots_processed();
            assert_edge_list_invariants(
                &state,
                &format!("case #232 after mark_tree_roots_processed outer {outer}"),
            );
            let mut inner_steps = 0usize;
            loop {
                let progressed = state.generic_primal_pass_once();
                assert_edge_list_invariants(
                    &state,
                    &format!("case #232 after generic pass outer {outer} inner {inner_steps}"),
                );
                inner_steps += 1;
                if !progressed || state.tree_num == 0 {
                    break;
                }
                assert!(
                    inner_steps <= 50_000,
                    "case #232 exceeded inner-step budget while checking edge lists",
                );
            }

            if state.tree_num == 0 {
                break;
            }

            let dual_ok = state.update_duals();
            assert_edge_list_invariants(
                &state,
                &format!("case #232 after dual update outer {outer}"),
            );

            assert!(dual_ok, "case #232 dual update failed during edge-list check");

            assert!(
                outer < 4_999,
                "case #232 exceeded outer-step budget while checking edge lists",
            );
        }

        normalize_pairs(&state.into_pairs())
    }

    fn solve_case_474_with_edge_list_checks() -> Vec<(usize, usize)> {
        let edges = case_474_edges();
        let g = build_graph(6, &edges);
        let mut state = BlossomVState::new(&g);
        assert_edge_list_invariants(&state, "case #474 after new");

        state.init_global();
        state.mark_tree_roots_processed();
        assert_edge_list_invariants(&state, "case #474 after init_global");

        for outer in 0..100usize {
            let mut inner_steps = 0usize;
            loop {
                let progressed = state.generic_primal_pass_once();
                assert_edge_list_invariants(
                    &state,
                    &format!("case #474 after generic pass outer {outer} inner {inner_steps}"),
                );
                inner_steps += 1;
                if !progressed || state.tree_num == 0 {
                    break;
                }
                assert!(
                    inner_steps <= 10_000,
                    "case #474 exceeded inner-step budget while checking edge lists",
                );
            }

            if state.tree_num == 0 {
                break;
            }

            let dual_ok = state.update_duals();
            assert_edge_list_invariants(
                &state,
                &format!("case #474 after dual update outer {outer}"),
            );

            assert!(dual_ok, "case #474 dual update failed during edge-list check");

            assert!(outer < 99, "case #474 exceeded outer-step budget while checking edge lists");
        }

        normalize_pairs(&state.into_pairs())
    }

    fn case_474_first_marked_generic_steps() -> Vec<GenericPrimalStepTrace> {
        let edges = case_474_edges();
        let g = build_graph(6, &edges);
        let mut state = BlossomVState::new(&g);
        state.init_global();
        state.mark_tree_roots_processed();

        let before = state.test_generic_primal_steps().len();
        assert!(
            state.generic_primal_pass_once(),
            "case #474 should execute the first marked generic call",
        );
        state.test_generic_primal_steps()[before..].to_vec()
    }

    fn edge_slacks_by_endpoints(state: &BlossomVState<Vcsr>) -> Vec<((u32, u32), i64)> {
        let mut edge_slacks = (0..state.test_edge_count())
            .map(|e| {
                let (u, v) = state.test_edge_endpoints(e);
                let endpoints = if u < v { (u, v) } else { (v, u) };
                (endpoints, state.test_edge_slack(e))
            })
            .collect::<Vec<_>>();
        edge_slacks.sort_unstable();
        edge_slacks
    }

    fn detached_matched_node(partner: u32, edge: (u32, u32)) -> NodeParitySnapshot {
        NodeParitySnapshot {
            flag: FREE,
            is_outer: true,
            is_tree_root: false,
            is_processed: false,
            tree_root: None,
            match_partner: Some(partner),
            match_edge: Some(edge),
            tree_parent_edge: None,
            first_tree_child: None,
            tree_sibling_prev: None,
            tree_sibling_next: None,
        }
    }

    #[test]
    fn test_construction_edge_count() {
        let g = build_graph(4, &[(0, 1, 5), (1, 2, 3), (2, 3, 7)]);
        let state = BlossomVState::new(&g);
        assert_eq!(state.test_edge_count(), 3);
    }

    #[test]
    fn test_construction_edge_endpoints() {
        let g = build_graph(4, &[(0, 1, 5), (2, 3, 7)]);
        let state = BlossomVState::new(&g);
        assert_eq!(state.test_edge_endpoints(0), (0, 1));
        assert_eq!(state.test_edge_endpoints(1), (2, 3));
    }

    #[test]
    fn test_construction_adjacency() {
        let g = build_graph(4, &[(0, 1, 5), (0, 2, 3), (0, 3, 7)]);
        let state = BlossomVState::new(&g);
        // Node 0 has 3 incident edges
        assert_eq!(state.test_degree(0), 3);
        // Nodes 1, 2, 3 each have 1 incident edge
        assert_eq!(state.test_degree(1), 1);
        assert_eq!(state.test_degree(2), 1);
        assert_eq!(state.test_degree(3), 1);
    }

    #[test]
    fn test_greedy_feasible_duals() {
        let g = build_graph(
            6,
            &[
                (0, 1, 3),
                (0, 3, 10),
                (0, 4, 7),
                (1, 2, -1),
                (1, 4, 5),
                (1, 5, 4),
                (2, 5, -7),
                (3, 4, 0),
                (4, 5, 4),
            ],
        );
        let state = BlossomVState::new(&g);
        // All slacks must be ≥ 0
        for e in 0..state.test_edge_count() {
            assert!(
                state.test_edge_slack(e) >= 0,
                "Edge {} has negative slack {} (endpoints {:?})",
                e,
                state.test_edge_slack(e),
                state.test_edge_endpoints(e),
            );
        }
    }

    #[test]
    fn test_greedy_valid_matching() {
        let g = build_graph(
            6,
            &[
                (0, 1, 3),
                (0, 3, 10),
                (0, 4, 7),
                (1, 2, -1),
                (1, 4, 5),
                (1, 5, 4),
                (2, 5, -7),
                (3, 4, 0),
                (4, 5, 4),
            ],
        );
        let state = BlossomVState::new(&g);
        // No vertex matched twice
        let mut matched = vec![false; 6];
        for v in 0..6 {
            if state.test_is_matched(v) {
                let partner = state.test_match_partner(v).unwrap() as usize;
                assert!(!matched[v], "Vertex {v} matched twice");
                assert_eq!(
                    state.test_match_partner(partner),
                    Some(v as u32),
                    "Matching not symmetric for {v} <-> {partner}"
                );
                matched[v] = true;
            }
        }
    }

    #[test]
    fn test_greedy_matched_edges_tight() {
        // Matched edges should have slack = 0
        let g = build_graph(4, &[(0, 1, 5), (2, 3, 7)]);
        let state = BlossomVState::new(&g);
        for v in 0..4 {
            let arc = state.nodes[v].match_arc;
            if arc != NONE {
                let e = arc_edge(arc) as usize;
                assert_eq!(
                    state.test_edge_slack(e),
                    0,
                    "Matched edge {e} should have slack 0, got {}",
                    state.test_edge_slack(e),
                );
            }
        }
    }

    #[test]
    fn test_greedy_creates_trees() {
        // 4 vertices, 1 edge → 2 matched, 2 unmatched → 2 trees
        let g = build_graph(4, &[(0, 1, 5)]);
        let state = BlossomVState::new(&g);
        assert_eq!(state.test_tree_num(), 2);
        assert!(state.test_is_matched(0));
        assert!(state.test_is_matched(1));
        assert!(!state.test_is_matched(2));
        assert!(!state.test_is_matched(3));
        assert!(state.test_is_tree_root(2));
        assert!(state.test_is_tree_root(3));
        assert_eq!(state.test_flag(2), PLUS);
        assert_eq!(state.test_flag(3), PLUS);
    }

    #[test]
    fn test_greedy_single_edge_matches() {
        let g = build_graph(2, &[(0, 1, 42)]);
        let state = BlossomVState::new(&g);
        assert_eq!(state.test_tree_num(), 0);
        assert!(state.test_is_matched(0));
        assert!(state.test_is_matched(1));
        assert_eq!(state.test_match_partner(0), Some(1));
        assert_eq!(state.test_match_partner(1), Some(0));
    }

    #[test]
    fn test_greedy_feasible_negative_weights() {
        let g = build_graph(4, &[(0, 1, -10), (2, 3, -20), (0, 2, 5)]);
        let state = BlossomVState::new(&g);
        for e in 0..state.test_edge_count() {
            assert!(
                state.test_edge_slack(e) >= 0,
                "Edge {} has negative slack {}",
                e,
                state.test_edge_slack(e),
            );
        }
    }

    #[test]
    fn test_greedy_prefers_low_cost() {
        // Triangle: (0,1,w=1), (1,2,w=100), (0,2,w=1)
        // Plus (2,3,w=1) to make n even.
        // Greedy should try to match low-cost edges.
        let g = build_graph(4, &[(0, 1, 1), (1, 2, 100), (0, 2, 1), (2, 3, 1)]);
        let state = BlossomVState::new(&g);
        // All 4 vertices should be matched (greedy may or may not be optimal)
        assert_eq!(state.test_tree_num(), 0);
    }

    #[test]
    fn test_greedy_case_909_matches_cpp_init_after() {
        let g = build_graph(4, &[(0, 2, 16), (1, 3, -65), (0, 1, -64), (2, 3, 16)]);
        let state = BlossomVState::new(&g);

        let expected = GreedyInitSnapshot {
            y: vec![-63, -65, 95, -65],
            slacks: vec![0, 0, 0, 2],
            matching: vec![1, 0, -1, -1],
            flags: vec![FREE, FREE, PLUS, PLUS],
            is_outer: vec![true, true, true, true],
            tree_eps_by_node: vec![0, 0, 0, 0],
            tree_num: 2,
        };

        assert_eq!(state.test_init_snapshot(), expected);
    }

    #[test]
    fn test_ground_truth_first_n6_case_with_budget() {
        // First n=6 case from the gzipped ground-truth corpus.
        let g = build_graph(
            6,
            &[
                (0, 3, -35),
                (2, 5, -39),
                (1, 4, 80),
                (3, 5, 71),
                (3, 4, 65),
                (2, 4, -87),
                (0, 5, -9),
                (1, 2, 73),
                (1, 3, 63),
            ],
        );

        let matching = BlossomVState::new(&g)
            .solve_with_test_budget(200, 5_000)
            .expect("should solve first n=6 ground-truth case within budget");

        assert_eq!(normalize_pairs(&matching), vec![(0, 5), (1, 3), (2, 4)],);
    }

    #[test]
    fn test_ground_truth_case_49_with_budget() {
        // First known bad n=6 case from the gzipped ground-truth corpus.
        let edges = [
            (0, 5, 89),
            (3, 4, 82),
            (1, 2, -12),
            (1, 3, 80),
            (0, 2, -78),
            (0, 4, -14),
            (3, 5, 53),
            (0, 1, 50),
            (2, 5, -22),
        ];
        let g = build_graph(6, &edges);

        let matching = BlossomVState::new(&g)
            .solve_with_test_budget(200, 5_000)
            .expect("should solve n=6 corpus case #49 within budget");
        let matching = normalize_pairs(&matching);

        validate_matching(6, &matching);
        assert_eq!(matching_cost(&edges, &matching), 27);
    }

    #[test]
    fn test_case_49_manual_grow_grow_augment() {
        let edges = [
            (0, 5, 89),
            (3, 4, 82),
            (1, 2, -12),
            (1, 3, 80),
            (0, 2, -78),
            (0, 4, -14),
            (3, 5, 53),
            (0, 1, 50),
            (2, 5, -22),
        ];
        let g = build_graph(6, &edges);
        let mut state = BlossomVState::new(&g);

        let e04 = find_edge_idx(&state, 0, 4);
        let e12 = find_edge_idx(&state, 1, 2);
        let e35 = find_edge_idx(&state, 3, 5);

        state.grow(e04, 4, 0);
        state.grow(e12, 2, 1);
        state.augment(e35, 3, 5);

        let pairs = normalize_pairs(&state.into_pairs());
        assert_eq!(pairs, vec![(0, 4), (1, 2), (3, 5)]);
    }

    #[test]
    fn test_greedy_case_49_matches_cpp_init_after() {
        let g = build_graph(
            6,
            &[
                (0, 5, 89),
                (3, 4, 82),
                (1, 2, -12),
                (1, 3, 80),
                (0, 2, -78),
                (0, 4, -14),
                (3, 5, 53),
                (0, 1, 50),
                (2, 5, -22),
            ],
        );
        let state = BlossomVState::new(&g);
        let snapshot = state.test_init_snapshot();

        assert_eq!(snapshot.y, vec![-78, 54, -78, 106, 50, 0]);
        assert_eq!(snapshot.matching, vec![2, 3, 0, 1, -1, -1]);
        assert_eq!(snapshot.flags, vec![FREE, FREE, FREE, FREE, PLUS, PLUS]);
        assert_eq!(snapshot.is_outer, vec![true, true, true, true, true, true]);
        assert_eq!(snapshot.tree_num, 2);
        assert_eq!(
            edge_slacks_by_endpoints(&state),
            vec![
                ((0, 1), 124),
                ((0, 2), 0),
                ((0, 4), 0),
                ((0, 5), 256),
                ((1, 2), 0),
                ((1, 3), 0),
                ((2, 5), 34),
                ((3, 4), 8),
                ((3, 5), 0),
            ],
        );
    }

    #[test]
    fn test_init_global_case_49_matches_cpp_finish_before() {
        let g = build_graph(
            6,
            &[
                (0, 5, 89),
                (3, 4, 82),
                (1, 2, -12),
                (1, 3, 80),
                (0, 2, -78),
                (0, 4, -14),
                (3, 5, 53),
                (0, 1, 50),
                (2, 5, -22),
            ],
        );
        let mut state = BlossomVState::new(&g);
        state.init_global();

        assert_eq!(
            state.test_init_global_trace(),
            &[
                InitGlobalEvent::Grow { edge: (0, 4), plus: 4, free: 0 },
                InitGlobalEvent::Grow { edge: (1, 2), plus: 2, free: 1 },
                InitGlobalEvent::Augment { edge: (3, 5), left: 3, right: 5 },
            ],
        );

        let steps = state.test_init_global_steps();
        assert_eq!(steps.len(), 3);
        assert_eq!(steps[0].event, InitGlobalEvent::Grow { edge: (0, 4), plus: 4, free: 0 },);
        assert_eq!(steps[0].before.tree_num, 2);
        assert_eq!(steps[0].after.tree_num, 2);
        assert_eq!(steps[0].after.nodes[0].flag, MINUS);
        assert_eq!(steps[0].after.nodes[0].tree_root, Some(4));
        assert_eq!(steps[0].after.nodes[2].flag, PLUS);
        assert_eq!(steps[0].after.nodes[2].tree_root, Some(4));

        assert_eq!(steps[1].event, InitGlobalEvent::Grow { edge: (1, 2), plus: 2, free: 1 },);
        assert_eq!(steps[1].after.tree_num, 2);
        assert_eq!(steps[1].after.nodes[1].flag, MINUS);
        assert_eq!(steps[1].after.nodes[1].tree_root, Some(4));
        assert_eq!(steps[1].after.nodes[3].flag, PLUS);
        assert_eq!(steps[1].after.nodes[3].tree_root, Some(4));

        assert_eq!(steps[2].event, InitGlobalEvent::Augment { edge: (3, 5), left: 3, right: 5 },);

        let snapshot = state.test_strict_parity_snapshot();
        assert_eq!(
            snapshot,
            StrictParitySnapshot {
                y: vec![-78, 54, -78, 106, 50, 0],
                edge_slacks: vec![
                    ((0, 1), 124),
                    ((0, 2), 0),
                    ((0, 4), 0),
                    ((0, 5), 256),
                    ((1, 2), 0),
                    ((1, 3), 0),
                    ((2, 5), 34),
                    ((3, 4), 8),
                    ((3, 5), 0),
                ],
                nodes: vec![
                    detached_matched_node(4, (0, 4)),
                    detached_matched_node(2, (1, 2)),
                    detached_matched_node(1, (1, 2)),
                    detached_matched_node(5, (3, 5)),
                    detached_matched_node(0, (0, 4)),
                    detached_matched_node(3, (3, 5)),
                ],
                tree_num: 0,
            },
        );
    }

    #[test]
    fn test_init_global_case_909_matches_cpp_finish_before() {
        let g = build_graph(4, &[(0, 2, 16), (1, 3, -65), (0, 1, -64), (2, 3, 16)]);
        let mut state = BlossomVState::new(&g);
        state.init_global();

        let steps = state.test_init_global_steps();
        assert_eq!(steps.len(), 2);
        assert_eq!(steps[0].event, InitGlobalEvent::Grow { edge: (0, 2), plus: 2, free: 0 },);
        assert_eq!(steps[0].after.nodes[0].flag, MINUS);
        assert_eq!(steps[0].after.nodes[0].tree_root, Some(2));
        assert_eq!(steps[0].after.nodes[1].flag, PLUS);
        assert_eq!(steps[0].after.nodes[1].tree_root, Some(2));

        assert_eq!(steps[1].event, InitGlobalEvent::Augment { edge: (1, 3), left: 1, right: 3 },);

        let snapshot = state.test_strict_parity_snapshot();
        assert_eq!(
            snapshot,
            StrictParitySnapshot {
                y: vec![-63, -65, 95, -65],
                edge_slacks: vec![((0, 1), 0), ((0, 2), 0), ((1, 3), 0), ((2, 3), 2),],
                nodes: vec![
                    detached_matched_node(2, (0, 2)),
                    detached_matched_node(3, (1, 3)),
                    detached_matched_node(0, (0, 2)),
                    detached_matched_node(1, (1, 3)),
                ],
                tree_num: 0,
            },
        );
    }

    #[test]
    fn test_greedy_case_98_matches_cpp_init_after() {
        let g = build_graph(
            6,
            &[
                (2, 3, -49),
                (0, 1, 40),
                (4, 5, -56),
                (1, 5, -32),
                (3, 4, -17),
                (2, 5, -86),
                (1, 4, -59),
                (3, 5, -30),
            ],
        );
        let state = BlossomVState::new(&g);
        let snapshot = state.test_init_snapshot();

        assert_eq!(snapshot.y, vec![139, -59, -86, -12, -59, -86]);
        assert_eq!(snapshot.matching, vec![1, 0, 5, -1, -1, 2]);
        assert_eq!(snapshot.flags, vec![FREE, FREE, FREE, PLUS, PLUS, FREE]);
        assert_eq!(snapshot.is_outer, vec![true, true, true, true, true, true]);
        assert_eq!(snapshot.tree_num, 2);
        assert_eq!(
            edge_slacks_by_endpoints(&state),
            vec![
                ((0, 1), 0),
                ((1, 4), 0),
                ((1, 5), 81),
                ((2, 3), 0),
                ((2, 5), 0),
                ((3, 4), 37),
                ((3, 5), 38),
                ((4, 5), 33),
            ],
        );
    }

    #[test]
    fn test_init_global_case_98_matches_cpp_finish_before() {
        let g = build_graph(
            6,
            &[
                (2, 3, -49),
                (0, 1, 40),
                (4, 5, -56),
                (1, 5, -32),
                (3, 4, -17),
                (2, 5, -86),
                (1, 4, -59),
                (3, 5, -30),
            ],
        );
        let mut state = BlossomVState::new(&g);
        state.init_global();

        assert!(
            state.test_init_global_trace().iter().any(
                |event| matches!(event, InitGlobalEvent::Shrink { edge, .. } if *edge == (3, 5))
            ),
            "case #98 should trigger a startup shrink on edge (3, 5)",
        );

        let snapshot = state.test_strict_parity_snapshot();
        assert_eq!(
            snapshot,
            StrictParitySnapshot {
                y: vec![153, -73, -105, 7, -45, -67],
                edge_slacks: vec![
                    ((0, 1), 0),
                    ((1, 4), 0),
                    ((1, 5), 76),
                    ((2, 3), 0),
                    ((2, 5), 0),
                    ((3, 4), 4),
                    ((3, 5), 0),
                    ((4, 5), 0),
                ],
                nodes: vec![
                    detached_matched_node(1, (0, 1)),
                    detached_matched_node(0, (0, 1)),
                    detached_matched_node(3, (2, 3)),
                    detached_matched_node(2, (2, 3)),
                    detached_matched_node(5, (4, 5)),
                    detached_matched_node(4, (4, 5)),
                ],
                tree_num: 0,
            },
        );
    }

    #[test]
    fn test_ground_truth_case_98_with_budget() {
        let edges = [
            (2, 3, -49),
            (0, 1, 40),
            (4, 5, -56),
            (1, 5, -32),
            (3, 4, -17),
            (2, 5, -86),
            (1, 4, -59),
            (3, 5, -30),
        ];
        let g = build_graph(6, &edges);

        let matching = BlossomVState::new(&g)
            .solve_with_test_budget(200, 5_000)
            .expect("should solve n=6 corpus case #98 within budget");
        let matching = normalize_pairs(&matching);

        validate_matching(6, &matching);
        assert_eq!(matching_cost(&edges, &matching), -65);
        assert_eq!(matching, vec![(0, 1), (2, 3), (4, 5)]);
    }

    #[test]
    fn test_greedy_case_27004_matches_cpp_init_after() {
        let g = build_graph(
            10,
            &[
                (5, 7, 10),
                (0, 8, -18),
                (3, 4, -70),
                (6, 9, -100),
                (1, 2, 51),
                (0, 3, 94),
                (0, 5, -42),
                (2, 8, 66),
                (0, 6, 47),
                (5, 9, 6),
                (2, 7, 38),
                (3, 7, -95),
                (0, 9, -87),
                (0, 1, -56),
                (1, 8, -69),
                (7, 9, 72),
                (1, 7, -86),
                (0, 2, -64),
                (2, 4, 62),
                (6, 8, 55),
                (5, 6, -30),
                (1, 5, -3),
                (1, 4, 23),
                (7, 8, -21),
                (3, 8, -69),
            ],
        );
        let state = BlossomVState::new(&g);
        let snapshot = state.test_init_snapshot();

        assert_eq!(snapshot.y, vec![-74, -77, -54, -95, -45, -10, -100, -95, -61, -100]);
        assert_eq!(snapshot.matching, vec![9, 7, -1, 4, 3, -1, -1, 1, -1, 0]);
        assert_eq!(
            snapshot.flags,
            vec![FREE, FREE, PLUS, FREE, FREE, PLUS, PLUS, FREE, PLUS, FREE],
        );
        assert_eq!(snapshot.is_outer, vec![true; 10]);
        assert_eq!(snapshot.tree_num, 4);
        assert_eq!(
            edge_slacks_by_endpoints(&state),
            vec![
                ((0, 1), 39),
                ((0, 2), 0),
                ((0, 3), 357),
                ((0, 5), 0),
                ((0, 6), 268),
                ((0, 8), 99),
                ((0, 9), 0),
                ((1, 2), 233),
                ((1, 4), 168),
                ((1, 5), 81),
                ((1, 7), 0),
                ((1, 8), 0),
                ((2, 4), 223),
                ((2, 7), 225),
                ((2, 8), 247),
                ((3, 4), 0),
                ((3, 7), 0),
                ((3, 8), 18),
                ((5, 6), 50),
                ((5, 7), 125),
                ((5, 9), 122),
                ((6, 8), 271),
                ((6, 9), 0),
                ((7, 8), 114),
                ((7, 9), 339),
            ],
        );
    }

    #[test]
    fn test_init_global_case_27004_matches_cpp_finish_before() {
        let g = build_graph(
            10,
            &[
                (5, 7, 10),
                (0, 8, -18),
                (3, 4, -70),
                (6, 9, -100),
                (1, 2, 51),
                (0, 3, 94),
                (0, 5, -42),
                (2, 8, 66),
                (0, 6, 47),
                (5, 9, 6),
                (2, 7, 38),
                (3, 7, -95),
                (0, 9, -87),
                (0, 1, -56),
                (1, 8, -69),
                (7, 9, 72),
                (1, 7, -86),
                (0, 2, -64),
                (2, 4, 62),
                (6, 8, 55),
                (5, 6, -30),
                (1, 5, -3),
                (1, 4, 23),
                (7, 8, -21),
                (3, 8, -69),
            ],
        );
        let mut state = BlossomVState::new(&g);
        state.init_global();

        let snapshot = state.test_strict_parity_snapshot();
        assert_eq!(
            snapshot,
            StrictParitySnapshot {
                y: vec![-160, -116, 32, -134, -6, 76, -136, -56, -22, -64],
                edge_slacks: vec![
                    ((0, 1), 164),
                    ((0, 2), 0),
                    ((0, 3), 482),
                    ((0, 5), 0),
                    ((0, 6), 390),
                    ((0, 8), 146),
                    ((0, 9), 50),
                    ((1, 2), 186),
                    ((1, 4), 168),
                    ((1, 5), 34),
                    ((1, 7), 0),
                    ((1, 8), 0),
                    ((2, 4), 98),
                    ((2, 7), 100),
                    ((2, 8), 122),
                    ((3, 4), 0),
                    ((3, 7), 0),
                    ((3, 8), 18),
                    ((5, 6), 0),
                    ((5, 7), 0),
                    ((5, 9), 0),
                    ((6, 8), 268),
                    ((6, 9), 0),
                    ((7, 8), 36),
                    ((7, 9), 264),
                ],
                nodes: vec![
                    detached_matched_node(2, (0, 2)),
                    detached_matched_node(8, (1, 8)),
                    detached_matched_node(0, (0, 2)),
                    detached_matched_node(4, (3, 4)),
                    detached_matched_node(3, (3, 4)),
                    detached_matched_node(7, (5, 7)),
                    detached_matched_node(9, (6, 9)),
                    detached_matched_node(5, (5, 7)),
                    detached_matched_node(1, (1, 8)),
                    detached_matched_node(6, (6, 9)),
                ],
                tree_num: 0,
            },
        );
    }

    #[test]
    fn test_ground_truth_case_26951_with_budget() {
        let edges = [
            (1, 3, 54),
            (4, 6, -95),
            (0, 5, 81),
            (2, 7, -2),
            (6, 7, -23),
            (2, 4, 73),
            (2, 5, -97),
            (1, 4, -86),
            (0, 4, 88),
            (1, 5, 73),
            (1, 7, 10),
            (3, 6, -84),
            (5, 6, 41),
            (3, 7, -34),
            (0, 2, -22),
        ];
        let g = build_graph(8, &edges);

        let matching = BlossomVState::new(&g)
            .solve_with_test_budget(200, 5_000)
            .expect("should solve n=8 corpus case #26951 within budget");
        let matching = normalize_pairs(&matching);

        validate_matching(8, &matching);
        assert_eq!(matching_cost(&edges, &matching), -101);
        assert_eq!(matching, vec![(0, 2), (1, 4), (3, 7), (5, 6)]);
    }

    #[test]
    fn test_ground_truth_case_27373_with_budget() {
        let edges = [
            (0, 1, 5),
            (6, 8, 40),
            (3, 11, -94),
            (4, 5, -88),
            (2, 10, -48),
            (7, 9, 69),
            (1, 3, 98),
            (2, 6, 89),
            (1, 7, 30),
            (9, 10, 8),
            (3, 7, -3),
            (8, 10, -82),
            (6, 9, -26),
            (4, 10, -95),
            (0, 10, -47),
            (2, 3, -46),
            (3, 5, -38),
            (2, 5, -1),
            (1, 5, 82),
            (5, 9, -94),
        ];
        let g = build_graph(12, &edges);

        let matching = BlossomVState::new(&g)
            .solve_with_test_budget(200, 5_000)
            .expect("should solve n=12 corpus case #27373 within budget");
        let matching = normalize_pairs(&matching);

        validate_matching(12, &matching);
        assert_eq!(matching_cost(&edges, &matching), -116);
        assert_eq!(matching, vec![(0, 1), (2, 10), (3, 11), (4, 5), (6, 8), (7, 9)]);
    }

    #[test]
    fn test_ground_truth_case_27004_with_budget() {
        let edges = [
            (5, 7, 10),
            (0, 8, -18),
            (3, 4, -70),
            (6, 9, -100),
            (1, 2, 51),
            (0, 3, 94),
            (0, 5, -42),
            (2, 8, 66),
            (0, 6, 47),
            (5, 9, 6),
            (2, 7, 38),
            (3, 7, -95),
            (0, 9, -87),
            (0, 1, -56),
            (1, 8, -69),
            (7, 9, 72),
            (1, 7, -86),
            (0, 2, -64),
            (2, 4, 62),
            (6, 8, 55),
            (5, 6, -30),
            (1, 5, -3),
            (1, 4, 23),
            (7, 8, -21),
            (3, 8, -69),
        ];
        let g = build_graph(10, &edges);

        let matching = BlossomVState::new(&g)
            .solve_with_test_budget(200, 5_000)
            .expect("should solve n=10 corpus case #27004 within budget");
        let matching = normalize_pairs(&matching);

        validate_matching(10, &matching);
        assert_eq!(matching_cost(&edges, &matching), -293);
        assert_eq!(matching, vec![(0, 2), (1, 8), (3, 4), (5, 7), (6, 9)]);
    }

    #[test]
    fn test_ground_truth_case_91838_with_budget() {
        let edges = [
            (0, 1, 87),
            (0, 2, -14),
            (0, 15, -84),
            (0, 16, -84),
            (0, 17, 11),
            (1, 2, 32),
            (1, 17, 48),
            (2, 3, -65),
            (2, 4, -50),
            (3, 4, 99),
            (3, 5, 41),
            (4, 5, -84),
            (4, 6, 6),
            (5, 6, -53),
            (5, 7, 23),
            (6, 7, 26),
            (6, 8, 2),
            (7, 8, -19),
            (7, 9, -49),
            (8, 9, -65),
            (8, 10, 33),
            (9, 11, -86),
            (9, 12, -31),
            (10, 11, -44),
            (10, 12, 20),
            (11, 12, -48),
            (11, 13, 91),
            (12, 13, 67),
            (12, 14, 56),
            (13, 14, -91),
            (13, 15, 93),
            (14, 15, -11),
            (14, 16, -17),
            (15, 16, -94),
            (15, 17, -15),
            (16, 17, 0),
        ];
        let g = build_graph(18, &edges);

        let matching = BlossomVState::new(&g)
            .solve_with_test_budget(200, 5_000)
            .expect("should solve n=18 corpus case #91838 within budget");
        let matching = normalize_pairs(&matching);

        validate_matching(18, &matching);
        assert_eq!(matching_cost(&edges, &matching), -226);
        assert_eq!(
            matching,
            vec![(0, 15), (1, 17), (2, 3), (4, 5), (6, 8), (7, 9), (10, 11), (12, 13), (14, 16),],
        );
    }

    #[test]
    fn test_ground_truth_case_91838_public_path() {
        let edges = [
            (0, 1, 87),
            (0, 2, -14),
            (0, 15, -84),
            (0, 16, -84),
            (0, 17, 11),
            (1, 2, 32),
            (1, 17, 48),
            (2, 3, -65),
            (2, 4, -50),
            (3, 4, 99),
            (3, 5, 41),
            (4, 5, -84),
            (4, 6, 6),
            (5, 6, -53),
            (5, 7, 23),
            (6, 7, 26),
            (6, 8, 2),
            (7, 8, -19),
            (7, 9, -49),
            (8, 9, -65),
            (8, 10, 33),
            (9, 11, -86),
            (9, 12, -31),
            (10, 11, -44),
            (10, 12, 20),
            (11, 12, -48),
            (11, 13, 91),
            (12, 13, 67),
            (12, 14, 56),
            (13, 14, -91),
            (13, 15, 93),
            (14, 15, -11),
            (14, 16, -17),
            (15, 16, -94),
            (15, 17, -15),
            (16, 17, 0),
        ];
        let g = build_graph(18, &edges);

        let matching =
            g.blossom_v().expect("public BlossomV path should solve n=18 corpus case #91838");
        let matching = normalize_pairs(&matching);

        validate_matching(18, &matching);
        assert_eq!(matching_cost(&edges, &matching), -226);
        assert_eq!(
            matching,
            vec![(0, 15), (1, 17), (2, 3), (4, 5), (6, 8), (7, 9), (10, 11), (12, 13), (14, 16),],
        );
    }

    #[test]
    fn test_ground_truth_case_97_with_budget() {
        let edges = case_97_edges();
        let g = build_graph(18, &edges);

        let matching = BlossomVState::new(&g)
            .solve_with_test_budget(300, 10_000)
            .expect("should solve generated corpus case #97 within budget");
        let matching = normalize_pairs(&matching);

        validate_matching(18, &matching);
        assert_eq!(matching_cost(&edges, &matching), -329);
        assert_eq!(
            matching,
            vec![(0, 12), (1, 9), (2, 17), (3, 15), (4, 6), (5, 13), (7, 14), (8, 10), (11, 16),],
        );
    }

    #[test]
    fn test_ground_truth_case_97_public_path() {
        let edges = case_97_edges();
        let g = build_graph(18, &edges);

        let matching =
            g.blossom_v().expect("public BlossomV path should solve generated corpus case #97");
        let matching = normalize_pairs(&matching);

        validate_matching(18, &matching);
        assert_eq!(matching_cost(&edges, &matching), -329);
        assert_eq!(
            matching,
            vec![(0, 12), (1, 9), (2, 17), (3, 15), (4, 6), (5, 13), (7, 14), (8, 10), (11, 16),],
        );
    }

    #[test]
    fn test_ground_truth_case_honggfuzz_sigabrt_4_with_budget() {
        let edges = case_honggfuzz_sigabrt_4_edges();
        let g = build_graph(26, &edges);

        let matching = BlossomVState::new(&g)
            .solve_with_test_budget(600, 20_000)
            .expect("should solve honggfuzz replay case 4 within budget");
        let matching = normalize_pairs(&matching);

        validate_matching(26, &matching);
        assert_eq!(matching_cost(&edges, &matching), -186717);
        assert_eq!(
            matching,
            vec![
                (0, 16),
                (1, 17),
                (2, 4),
                (3, 7),
                (5, 11),
                (6, 8),
                (9, 14),
                (10, 15),
                (12, 20),
                (13, 21),
                (18, 23),
                (19, 22),
                (24, 25),
            ],
        );
    }

    #[test]
    fn test_ground_truth_case_honggfuzz_sigabrt_5_with_budget() {
        let edges = case_honggfuzz_sigabrt_5_edges();
        let g = build_graph(26, &edges);

        let matching = BlossomVState::new(&g)
            .solve_with_test_budget(800, 20_000)
            .expect("should solve honggfuzz replay case 5 within budget");
        let matching = normalize_pairs(&matching);

        validate_matching(26, &matching);
        assert_eq!(matching_cost(&edges, &matching), -116000);
        assert_eq!(
            matching,
            vec![
                (0, 19),
                (1, 6),
                (2, 7),
                (3, 22),
                (4, 5),
                (8, 15),
                (9, 12),
                (10, 11),
                (13, 20),
                (14, 21),
                (16, 25),
                (17, 24),
                (18, 23),
            ],
        );
    }

    #[test]
    fn test_ground_truth_case_honggfuzz_sigabrt_6_with_budget() {
        let edges = case_honggfuzz_sigabrt_6_edges();
        let g = build_graph(26, &edges);

        let matching = BlossomVState::new(&g)
            .solve_with_test_budget(800, 20_000)
            .expect("should solve honggfuzz replay case 6 within budget");
        let matching = normalize_pairs(&matching);

        validate_matching(26, &matching);
        assert_eq!(matching_cost(&edges, &matching), -114562);
        assert_eq!(
            matching,
            vec![
                (0, 11),
                (1, 18),
                (2, 7),
                (3, 22),
                (4, 5),
                (6, 14),
                (8, 15),
                (9, 19),
                (10, 17),
                (12, 23),
                (13, 20),
                (16, 25),
                (21, 24),
            ],
        );
    }

    #[test]
    fn test_ground_truth_case_honggfuzz_sigabrt_7_with_budget() {
        let edges = case_honggfuzz_sigabrt_7_edges();
        let g = build_graph(26, &edges);

        let matching = BlossomVState::new(&g)
            .solve_with_test_budget(1200, 30_000)
            .expect("should solve honggfuzz replay case 7 within budget");
        let matching = normalize_pairs(&matching);

        validate_matching(26, &matching);
        assert_eq!(matching_cost(&edges, &matching), -113140);
        assert_eq!(
            matching,
            vec![
                (0, 24),
                (1, 23),
                (2, 11),
                (3, 7),
                (4, 5),
                (6, 12),
                (8, 15),
                (9, 22),
                (10, 17),
                (13, 19),
                (14, 21),
                (16, 25),
                (18, 20),
            ],
        );
    }

    #[test]
    fn test_ground_truth_case_416_with_budget() {
        let edges = case_416_edges();
        let g = build_graph(20, &edges);

        let matching = BlossomVState::new(&g)
            .solve_with_test_budget(400, 10_000)
            .expect("should solve generated corpus case #416 within budget");
        let matching = normalize_pairs(&matching);

        validate_matching(20, &matching);
        assert_eq!(matching_cost(&edges, &matching), -707);
        assert_eq!(
            matching,
            vec![
                (0, 8),
                (1, 6),
                (2, 16),
                (3, 5),
                (4, 17),
                (7, 13),
                (9, 11),
                (10, 18),
                (12, 15),
                (14, 19),
            ],
        );
    }

    #[test]
    fn test_ground_truth_case_87417_with_budget() {
        let edges = case_87417_edges();
        let g = build_graph(30, &edges);

        let matching = BlossomVState::new(&g)
            .solve_with_test_budget(800, 20_000)
            .expect("should solve corpus case #87417 within budget");
        let matching = normalize_pairs(&matching);

        validate_matching(30, &matching);
        assert_eq!(matching_cost(&edges, &matching), -771);
        assert_eq!(
            matching,
            vec![
                (0, 24),
                (1, 15),
                (2, 10),
                (3, 21),
                (4, 29),
                (5, 26),
                (6, 9),
                (7, 12),
                (8, 14),
                (11, 20),
                (13, 22),
                (16, 25),
                (17, 19),
                (18, 23),
                (27, 28),
            ],
        );
    }

    fn case_87417_state_after_generic_steps(steps_target: usize) -> BlossomVState<Vcsr> {
        let g = build_graph(30, &case_87417_edges());
        let leaked = Box::leak(Box::new(g));
        let mut state = BlossomVState::new(leaked);
        state.init_global();
        state.mark_tree_roots_processed();
        for step in 0..steps_target {
            assert!(
                state.generic_primal_pass_once(),
                "missing generic step {} on case #87417",
                step
            );
        }
        state
    }

    #[test]
    fn test_case_97_first_generic_grow_edge_is_visible_from_root_0() {
        let edges = case_97_edges();
        let g = build_graph(18, &edges);
        let mut state = BlossomVState::new(&g);
        state.init_global();
        state.mark_tree_roots_processed();

        let mut saw = false;
        for (e_idx, dir) in state.incident_edges(0) {
            let other = state.edge_head_outer(e_idx, dir);
            if normalized_edge_pair(state.edges[e_idx as usize].head) == (0, 12) {
                saw = true;
                assert_eq!(
                    state.edges[e_idx as usize].slack, 0,
                    "edge (0,12) should already be tight"
                );
                assert_eq!(other, 12, "edge (0,12) should point from root 0 to free node 12");
                assert_eq!(state.nodes[other as usize].flag, FREE, "node 12 should still be free");
            }
        }

        assert!(saw, "root 0 should still see the tight edge (0,12) after init_global");
    }

    #[test]
    fn test_case_97_first_generic_grow_candidate_matches_cpp() {
        let edges = case_97_edges();
        let g = build_graph(18, &edges);
        let mut state = BlossomVState::new(&g);
        state.init_global();
        state.mark_tree_roots_processed();

        let grow = state.find_tree_grow_edge(0);
        assert_eq!(
            grow,
            Some((11, 0, 12)),
            "case #97 should start the generic phase by taking the tight edge (0,12) from root 0 before C++ enters GrowNode(4)"
        );
    }

    #[test]
    fn test_case_97_matches_cpp_first_generic_grow_after() {
        let edges = case_97_edges();
        let g = build_graph(18, &edges);
        let mut state = BlossomVState::new(&g);
        state.init_global();
        state.mark_tree_roots_processed();
        assert!(state.generic_primal_pass_once(), "missing first generic event on case #97");
        let step = &state.test_generic_primal_steps()[0];
        assert_eq!(
            step.event,
            GenericPrimalEvent::Grow { edge: (0, 12), plus: 0, free: 12 },
            "case #97 should start the generic phase with the C++ grow on edge (0,12)",
        );
    }

    #[test]
    fn test_ground_truth_case_1594_with_budget() {
        let edges = case_1594_edges();
        let g = build_graph(26, &edges);

        let matching = BlossomVState::new(&g)
            .solve_with_test_budget(400, 30_000)
            .expect("should solve n=26 corpus case #1594 within budget");
        let matching = normalize_pairs(&matching);

        validate_matching(26, &matching);
        assert_eq!(matching_cost(&edges, &matching), -1003);
        assert_eq!(
            matching,
            vec![
                (0, 23),
                (1, 20),
                (2, 19),
                (3, 22),
                (4, 13),
                (5, 14),
                (6, 24),
                (7, 10),
                (8, 15),
                (9, 12),
                (11, 25),
                (16, 18),
                (17, 21),
            ],
        );
    }

    #[test]
    fn test_case_1594_edge_list_invariants_hold_during_solve() {
        let edges = case_1594_edges();
        let matching = solve_case_1594_with_edge_list_checks();

        validate_matching(26, &matching);
        assert_eq!(matching_cost(&edges, &matching), -1003);
    }

    #[test]
    fn test_init_global_case_232_matches_cpp_after() {
        let edges = case_232_edges();
        let g = build_graph(28, &edges);
        let mut state = BlossomVState::new(&g);
        state.init_global();

        let events = state
            .test_init_global_trace()
            .iter()
            .map(|event| {
                match event {
                    InitGlobalEvent::Grow { edge, .. } => ("grow", *edge),
                    InitGlobalEvent::Augment { edge, .. } => ("augment", *edge),
                    InitGlobalEvent::Shrink { edge, .. } => ("shrink", *edge),
                }
            })
            .collect::<Vec<_>>();
        assert_eq!(
            events,
            vec![
                ("grow", (4, 24)),
                ("grow", (4, 16)),
                ("shrink", (1, 11)),
                ("grow", (10, 15)),
                ("grow", (10, 14)),
                ("augment", (10, 11)),
                ("grow", (5, 12)),
                ("grow", (7, 8)),
                ("augment", (8, 13)),
                ("grow", (18, 25)),
                ("grow", (7, 18)),
                ("grow", (6, 9)),
                ("augment", (17, 20)),
                ("grow", (9, 19)),
                ("grow", (6, 17)),
                ("shrink", (6, 20)),
                ("grow", (21, 27)),
                ("grow", (0, 21)),
                ("grow", (10, 15)),
                ("grow", (11, 16)),
                ("grow", (1, 11)),
                ("shrink", (4, 24)),
            ],
        );

        let snapshot = state.test_strict_parity_snapshot();
        let flags = snapshot.nodes.iter().map(|node| node.flag).collect::<Vec<_>>();
        let is_outer = snapshot.nodes.iter().map(|node| node.is_outer).collect::<Vec<_>>();
        let tree_roots = snapshot.nodes.iter().map(|node| node.tree_root).collect::<Vec<_>>();
        let matching = (0..28)
            .map(|v| state.test_match_partner(v).map(|partner| partner as i32).unwrap_or(-1))
            .collect::<Vec<_>>();

        assert_eq!(
            snapshot.y,
            vec![
                -5, -41, -62, -51, -59, -88, -64, -106, -94, -62, 23, -97, -30, -74, -107, -141,
                -99, -90, -70, -58, -90, -57, -87, -88, -131, -114, 44, -141,
            ],
        );
        assert_eq!(
            matching,
            vec![
                21, -1, 23, 27, 24, 12, -1, 26, 13, 19, 15, 16, 5, 8, 22, 10, 11, 20, 25, 9, 17, 0,
                14, 2, 4, 18, 7, 3,
            ],
        );
        assert_eq!(
            flags,
            vec![
                FREE, PLUS, FREE, FREE, FREE, FREE, PLUS, FREE, FREE, FREE, FREE, FREE, FREE, FREE,
                FREE, FREE, FREE, FREE, FREE, FREE, FREE, FREE, FREE, FREE, FREE, FREE, FREE, FREE,
            ],
        );
        assert_eq!(is_outer, vec![true; 28]);
        assert_eq!(snapshot.tree_num, 2);
        assert_eq!(
            tree_roots,
            vec![
                None,
                Some(1),
                None,
                None,
                None,
                None,
                Some(6),
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            ],
        );
    }

    #[test]
    fn test_ground_truth_case_232_with_budget() {
        let edges = case_232_edges();
        let g = build_graph(28, &edges);

        let matching = BlossomVState::new(&g)
            .solve_with_test_budget(400, 30_000)
            .expect("should solve n=28 corpus case #232 within budget");
        let matching = normalize_pairs(&matching);

        validate_matching(28, &matching);
        assert_eq!(matching_cost(&edges, &matching), -982);
        assert_eq!(
            matching,
            vec![
                (0, 21),
                (1, 19),
                (2, 23),
                (3, 27),
                (4, 24),
                (5, 12),
                (6, 9),
                (7, 26),
                (8, 13),
                (10, 15),
                (11, 16),
                (14, 22),
                (17, 20),
                (18, 25),
            ],
        );
    }

    #[test]
    fn test_case_232_edge_list_invariants_hold_during_solve() {
        let edges = case_232_edges();
        let matching = solve_case_232_with_edge_list_checks();

        validate_matching(28, &matching);
        assert_eq!(matching_cost(&edges, &matching), -982);
    }

    #[test]
    fn test_case_474_matches_cpp_first_grow_after_semantics() {
        let steps = case_474_first_marked_generic_steps();
        assert_eq!(
            steps.len(),
            2,
            "the first marked generic call on case #474 should compact to grow+shrink",
        );
        assert_eq!(steps[0].event, GenericPrimalEvent::Grow { edge: (0, 2), plus: 0, free: 2 });
        assert_eq!(steps[1].event, GenericPrimalEvent::Shrink { edge: (0, 1), left: 1, right: 0 });
    }

    #[test]
    fn test_ground_truth_case_474_with_budget() {
        let edges = case_474_edges();
        let g = build_graph(6, &edges);

        let matching = BlossomVState::new(&g)
            .solve_with_test_budget(100, 10_000)
            .expect("should solve n=6 corpus case #474 within budget");
        let matching = normalize_pairs(&matching);

        validate_matching(6, &matching);
        assert_eq!(matching_cost(&edges, &matching), -8);
        assert_eq!(matching, vec![(0, 5), (1, 2), (3, 4)]);
    }

    #[test]
    fn test_case_474_edge_list_invariants_hold_during_solve() {
        let edges = case_474_edges();
        let matching = solve_case_474_with_edge_list_checks();

        validate_matching(6, &matching);
        assert_eq!(matching_cost(&edges, &matching), -8);
    }

    #[test]
    fn test_case_474_default_solve_matches_budgeted_solve() {
        let edges = case_474_edges();
        let g = build_graph(6, &edges);

        let default_matching =
            BlossomVState::new(&g).solve().expect("default solve should succeed");
        let default_matching = normalize_pairs(&default_matching);
        let budgeted_matching = BlossomVState::new(&g)
            .solve_with_test_budget(100, 10_000)
            .expect("budgeted solve should succeed");
        let budgeted_matching = normalize_pairs(&budgeted_matching);

        assert_eq!(default_matching, budgeted_matching);
        assert_eq!(matching_cost(&edges, &default_matching), -8);
    }

    #[test]
    fn test_ground_truth_case_145677_with_budget() {
        let edges = [
            (0, 1, -35),
            (0, 2, -28),
            (0, 24, -5),
            (0, 25, 92),
            (1, 2, -49),
            (1, 3, -18),
            (1, 25, -51),
            (2, 3, -46),
            (2, 4, 15),
            (3, 4, 54),
            (3, 5, 38),
            (4, 5, -39),
            (4, 6, -79),
            (5, 6, -12),
            (5, 7, -64),
            (6, 7, -27),
            (6, 8, 54),
            (7, 8, -70),
            (7, 9, -24),
            (8, 9, 82),
            (8, 12, 80),
            (9, 10, 48),
            (9, 11, -64),
            (10, 11, 72),
            (10, 12, 89),
            (11, 12, 77),
            (11, 13, -28),
            (12, 13, -59),
            (12, 14, 50),
            (13, 14, 28),
            (13, 15, -31),
            (13, 18, 26),
            (14, 15, 26),
            (14, 16, 65),
            (15, 16, -34),
            (15, 17, 36),
            (16, 17, 63),
            (16, 18, 96),
            (17, 18, 16),
            (17, 19, 87),
            (18, 20, -73),
            (19, 20, 18),
            (19, 21, -10),
            (20, 21, -89),
            (20, 22, -21),
            (21, 22, -22),
            (21, 23, -40),
            (22, 23, 96),
            (22, 24, -96),
            (23, 24, -47),
            (23, 25, -66),
            (24, 25, -76),
        ];
        let g = build_graph(26, &edges);

        let matching = BlossomVState::new(&g)
            .solve_with_test_budget(300, 20_000)
            .expect("should solve n=26 corpus case #145677 within budget");
        let matching = normalize_pairs(&matching);

        validate_matching(26, &matching);
        assert_eq!(matching_cost(&edges, &matching), -293);
        assert_eq!(
            matching,
            vec![
                (0, 2),
                (1, 25),
                (3, 5),
                (4, 6),
                (7, 8),
                (9, 11),
                (10, 12),
                (13, 14),
                (15, 16),
                (17, 19),
                (18, 20),
                (21, 23),
                (22, 24),
            ],
        );
    }

    #[test]
    fn test_ground_truth_case_174453_with_budget() {
        let edges = [
            (0, 4, -80),
            (0, 6, 33),
            (0, 9, -67),
            (0, 17, 68),
            (0, 18, 69),
            (0, 19, 24),
            (0, 20, 45),
            (0, 21, 69),
            (0, 23, 99),
            (1, 3, 44),
            (1, 6, 31),
            (1, 8, -14),
            (1, 10, 13),
            (1, 11, 27),
            (1, 13, -36),
            (1, 15, -47),
            (1, 16, 46),
            (1, 17, -23),
            (1, 18, -92),
            (1, 20, -92),
            (1, 21, 71),
            (1, 22, -22),
            (1, 23, -77),
            (2, 10, 83),
            (2, 12, -53),
            (2, 16, -60),
            (2, 22, -99),
            (2, 23, -74),
            (2, 25, 52),
            (3, 4, -58),
            (3, 8, 1),
            (3, 10, 5),
            (3, 14, -78),
            (3, 17, 29),
            (3, 18, -9),
            (3, 23, 79),
            (3, 24, 17),
            (3, 25, 10),
            (4, 5, -74),
            (4, 8, 19),
            (4, 9, 100),
            (4, 10, 29),
            (4, 11, -77),
            (4, 12, 29),
            (4, 13, -52),
            (4, 17, 0),
            (4, 19, -55),
            (4, 20, -99),
            (4, 23, -64),
            (4, 24, -56),
            (5, 6, 14),
            (5, 9, 25),
            (5, 11, 65),
            (5, 14, -18),
            (5, 18, -98),
            (5, 20, -96),
            (5, 21, -69),
            (5, 23, -11),
            (5, 24, -95),
            (6, 10, -85),
            (6, 11, 8),
            (6, 14, -1),
            (6, 15, 31),
            (6, 16, -22),
            (6, 18, 92),
            (6, 20, 89),
            (7, 8, 74),
            (7, 10, -58),
            (7, 14, -34),
            (7, 17, 22),
            (7, 21, 39),
            (7, 23, 25),
            (7, 24, 45),
            (8, 11, 28),
            (8, 12, 47),
            (8, 16, -100),
            (8, 17, 91),
            (8, 22, 9),
            (8, 23, -97),
            (9, 14, 37),
            (9, 16, 81),
            (9, 19, -6),
            (9, 20, 82),
            (9, 21, -65),
            (9, 23, -49),
            (9, 25, -3),
            (10, 11, -94),
            (10, 12, -48),
            (10, 17, 81),
            (10, 23, -21),
            (10, 24, -11),
            (11, 12, -15),
            (11, 18, 6),
            (11, 21, -7),
            (11, 22, -56),
            (12, 13, 77),
            (12, 17, 33),
            (12, 18, 91),
            (12, 19, -73),
            (12, 20, 78),
            (12, 21, -28),
            (13, 20, 44),
            (13, 21, -30),
            (13, 24, -63),
            (14, 15, 49),
            (14, 16, 85),
            (14, 23, -20),
            (15, 18, -11),
            (15, 19, 97),
            (16, 19, 30),
            (16, 21, -33),
            (16, 22, 92),
            (16, 23, -97),
            (17, 19, 62),
            (17, 24, 27),
            (17, 25, 23),
            (18, 19, 20),
            (18, 21, 86),
            (18, 23, -19),
            (18, 24, 83),
            (19, 21, -14),
            (19, 22, -57),
            (21, 22, 80),
            (21, 23, 11),
            (22, 23, -5),
            (22, 24, 62),
        ];
        let g = build_graph(26, &edges);

        let matching = BlossomVState::new(&g)
            .solve_with_test_budget(400, 30_000)
            .expect("should solve n=26 corpus case #174453 within budget");
        let matching = normalize_pairs(&matching);

        validate_matching(26, &matching);
        assert_eq!(matching_cost(&edges, &matching), -785);
        assert_eq!(
            matching,
            vec![
                (0, 9),
                (1, 15),
                (2, 22),
                (3, 14),
                (4, 20),
                (5, 18),
                (6, 16),
                (7, 10),
                (8, 23),
                (11, 21),
                (12, 19),
                (13, 24),
                (17, 25),
            ],
        );
    }

    #[test]
    fn test_ground_truth_case_4666_with_budget() {
        let edges = [
            (0, 2, -58),
            (0, 3, 17),
            (0, 4, 38),
            (0, 12, 89),
            (0, 17, 78),
            (0, 22, 5),
            (0, 26, 91),
            (0, 27, 67),
            (0, 28, 4),
            (0, 29, -56),
            (1, 2, -96),
            (1, 3, 99),
            (1, 4, 24),
            (1, 5, -18),
            (1, 20, -65),
            (1, 28, 58),
            (1, 29, -80),
            (2, 3, 43),
            (2, 4, 59),
            (2, 5, 44),
            (2, 6, -96),
            (2, 11, -82),
            (2, 27, 15),
            (2, 28, -72),
            (2, 29, 5),
            (3, 4, 76),
            (3, 5, 52),
            (3, 6, 88),
            (3, 7, 59),
            (3, 29, -13),
            (4, 5, -11),
            (4, 6, 81),
            (4, 7, 96),
            (4, 8, 52),
            (5, 6, -32),
            (5, 8, -44),
            (5, 9, -89),
            (5, 29, 20),
            (6, 7, 59),
            (6, 8, -22),
            (6, 9, 47),
            (6, 10, 3),
            (7, 8, -87),
            (7, 9, -50),
            (7, 10, -50),
            (7, 11, -29),
            (7, 19, 69),
            (7, 22, 14),
            (8, 9, -25),
            (8, 10, -86),
            (8, 12, 58),
            (8, 27, -8),
            (9, 10, 61),
            (9, 11, -78),
            (9, 12, 46),
            (9, 13, -10),
            (9, 14, 7),
            (9, 16, 76),
            (10, 11, 49),
            (10, 12, 60),
            (10, 13, 15),
            (10, 14, 68),
            (10, 28, 58),
            (11, 12, 76),
            (11, 14, 78),
            (11, 15, -39),
            (11, 19, -40),
            (12, 13, 88),
            (12, 14, 29),
            (12, 15, -72),
            (12, 16, 17),
            (13, 14, -57),
            (13, 15, 37),
            (13, 16, -81),
            (13, 17, -46),
            (13, 24, -97),
            (13, 26, 75),
            (14, 16, 49),
            (14, 17, 42),
            (15, 17, -61),
            (15, 18, 87),
            (15, 19, 86),
            (15, 23, -94),
            (16, 17, 94),
            (16, 19, -71),
            (16, 24, -63),
            (16, 25, 3),
            (17, 19, -51),
            (17, 20, -2),
            (17, 27, 94),
            (18, 19, 62),
            (18, 22, 56),
            (18, 26, 32),
            (18, 27, -35),
            (19, 21, 31),
            (19, 22, -10),
            (20, 21, 9),
            (20, 23, 87),
            (20, 24, -21),
            (21, 22, 22),
            (21, 23, 15),
            (21, 24, 20),
            (21, 25, -16),
            (22, 25, -39),
            (22, 29, -70),
            (23, 24, -90),
            (23, 25, -42),
            (23, 26, 53),
            (23, 27, 68),
            (24, 25, -21),
            (24, 26, 34),
            (25, 26, -13),
            (25, 27, -24),
            (25, 28, -84),
            (25, 29, -31),
            (26, 28, 57),
            (26, 29, 78),
            (27, 28, 91),
            (27, 29, -46),
            (28, 29, -2),
        ];
        let g = build_graph(30, &edges);

        let matching = BlossomVState::new(&g)
            .solve_with_test_budget(500, 50_000)
            .expect("should solve corpus case #4666 within budget");
        let matching = normalize_pairs(&matching);

        validate_matching(30, &matching);
        assert_eq!(matching_cost(&edges, &matching), -641);
        assert_eq!(
            matching,
            vec![
                (0, 3),
                (1, 20),
                (2, 6),
                (4, 5),
                (7, 10),
                (8, 27),
                (9, 11),
                (12, 15),
                (13, 14),
                (16, 24),
                (17, 19),
                (18, 26),
                (21, 23),
                (22, 29),
                (25, 28),
            ],
        );
    }

    #[test]
    fn test_case_4666_first_post_second_dual_event_matches_cpp() {
        let mut state = case_4666_state_at_dual_before(1);
        assert!(state.update_duals(), "second dual update failed");
        state.mark_tree_roots_processed();
        let steps_before = state.test_generic_primal_steps().len();
        assert!(state.generic_primal_pass_once(), "no first post-second-dual generic operation");
        let step = &state.test_generic_primal_steps()[steps_before];
        match &step.event {
            GenericPrimalEvent::Shrink { edge, .. } => assert_eq!(*edge, (7, 11)),
            other => panic!("expected first post-second-dual shrink, got {other:?}"),
        }
    }

    #[test]
    fn test_case_174453_first_post_dual_augment_edge_matches_cpp() {
        let edges = [
            (0, 4, -80),
            (0, 6, 33),
            (0, 9, -67),
            (0, 17, 68),
            (0, 18, 69),
            (0, 19, 24),
            (0, 20, 45),
            (0, 21, 69),
            (0, 23, 99),
            (1, 3, 44),
            (1, 6, 31),
            (1, 8, -14),
            (1, 10, 13),
            (1, 11, 27),
            (1, 13, -36),
            (1, 15, -47),
            (1, 16, 46),
            (1, 17, -23),
            (1, 18, -92),
            (1, 20, -92),
            (1, 21, 71),
            (1, 22, -22),
            (1, 23, -77),
            (2, 10, 83),
            (2, 12, -53),
            (2, 16, -60),
            (2, 22, -99),
            (2, 23, -74),
            (2, 25, 52),
            (3, 4, -58),
            (3, 8, 1),
            (3, 10, 5),
            (3, 14, -78),
            (3, 17, 29),
            (3, 18, -9),
            (3, 23, 79),
            (3, 24, 17),
            (3, 25, 10),
            (4, 5, -74),
            (4, 8, 19),
            (4, 9, 100),
            (4, 10, 29),
            (4, 11, -77),
            (4, 12, 29),
            (4, 13, -52),
            (4, 17, 0),
            (4, 19, -55),
            (4, 20, -99),
            (4, 23, -64),
            (4, 24, -56),
            (5, 6, 14),
            (5, 9, 25),
            (5, 11, 65),
            (5, 14, -18),
            (5, 18, -98),
            (5, 20, -96),
            (5, 21, -69),
            (5, 23, -11),
            (5, 24, -95),
            (6, 10, -85),
            (6, 11, 8),
            (6, 14, -1),
            (6, 15, 31),
            (6, 16, -22),
            (6, 18, 92),
            (6, 20, 89),
            (7, 8, 74),
            (7, 10, -58),
            (7, 14, -34),
            (7, 17, 22),
            (7, 21, 39),
            (7, 23, 25),
            (7, 24, 45),
            (8, 11, 28),
            (8, 12, 47),
            (8, 16, -100),
            (8, 17, 91),
            (8, 22, 9),
            (8, 23, -97),
            (9, 14, 37),
            (9, 16, 81),
            (9, 19, -6),
            (9, 20, 82),
            (9, 21, -65),
            (9, 23, -49),
            (9, 25, -3),
            (10, 11, -94),
            (10, 12, -48),
            (10, 17, 81),
            (10, 23, -21),
            (10, 24, -11),
            (11, 12, -15),
            (11, 18, 6),
            (11, 21, -7),
            (11, 22, -56),
            (12, 13, 77),
            (12, 17, 33),
            (12, 18, 91),
            (12, 19, -73),
            (12, 20, 78),
            (12, 21, -28),
            (13, 20, 44),
            (13, 21, -30),
            (13, 24, -63),
            (14, 15, 49),
            (14, 16, 85),
            (14, 23, -20),
            (15, 18, -11),
            (15, 19, 97),
            (16, 19, 30),
            (16, 21, -33),
            (16, 22, 92),
            (16, 23, -97),
            (17, 19, 62),
            (17, 24, 27),
            (17, 25, 23),
            (18, 19, 20),
            (18, 21, 86),
            (18, 23, -19),
            (18, 24, 83),
            (19, 21, -14),
            (19, 22, -57),
            (21, 22, 80),
            (21, 23, 11),
            (22, 23, -5),
            (22, 24, 62),
        ];
        let g = build_graph(26, &edges);
        let mut state = BlossomVState::new(&g);
        state.init_global();
        state.mark_tree_roots_processed();

        while state.generic_primal_pass_once() {}
        assert!(state.update_duals(), "first dual update failed");
        state.mark_tree_roots_processed();

        let steps_before = state.test_generic_primal_steps().len();
        assert!(state.generic_primal_pass_once(), "no first post-dual generic operation");
        let step = &state.test_generic_primal_steps()[steps_before];
        match &step.event {
            GenericPrimalEvent::Augment { edge, .. } => assert_eq!(*edge, (2, 23)),
            other => panic!("expected first post-dual augment, got {other:?}"),
        }
    }

    #[test]
    fn test_ground_truth_case_224_with_budget() {
        let edges = [
            (1, 2, -65),
            (0, 5, -71),
            (3, 4, 50),
            (3, 5, -90),
            (2, 4, -80),
            (1, 4, -16),
            (0, 3, -31),
            (1, 5, 38),
            (2, 3, -47),
            (2, 5, 46),
        ];
        let g = build_graph(6, &edges);

        let matching = BlossomVState::new(&g)
            .solve_with_test_budget(200, 5_000)
            .expect("should solve n=6 corpus case #224 within budget");
        let matching = normalize_pairs(&matching);

        validate_matching(6, &matching);
        assert_eq!(matching_cost(&edges, &matching), -134);
        assert_eq!(matching, vec![(0, 5), (1, 4), (2, 3)]);
    }

    #[test]
    fn test_ground_truth_case_26924_with_budget() {
        let edges = [
            (1, 7, -98),
            (0, 2, 67),
            (5, 9, 71),
            (3, 6, -45),
            (4, 8, 71),
            (0, 6, 1),
            (1, 3, -19),
            (1, 4, 31),
            (7, 8, -7),
            (1, 5, 7),
            (4, 7, 18),
            (2, 8, -74),
            (3, 4, -13),
            (2, 5, 76),
            (2, 6, 73),
            (2, 9, 30),
            (0, 7, 18),
            (1, 2, 78),
            (8, 9, 58),
            (2, 4, -75),
            (2, 7, 69),
            (4, 9, -39),
            (4, 5, -46),
            (7, 9, -3),
        ];
        let g = build_graph(10, &edges);

        let matching = BlossomVState::new(&g)
            .solve_with_test_budget(200, 5_000)
            .expect("should solve n=10 corpus case #26924 within budget");
        let matching = normalize_pairs(&matching);

        validate_matching(10, &matching);
        assert_eq!(matching_cost(&edges, &matching), -141);
        assert_eq!(matching, vec![(0, 6), (1, 3), (2, 8), (4, 5), (7, 9)]);
    }

    #[test]
    fn test_init_global_case_224_matches_cpp_after() {
        let g = build_graph(
            6,
            &[
                (1, 2, -65),
                (0, 5, -71),
                (3, 4, 50),
                (3, 5, -90),
                (2, 4, -80),
                (1, 4, -16),
                (0, 3, -31),
                (1, 5, 38),
                (2, 3, -47),
                (2, 5, 46),
            ],
        );
        let mut state = BlossomVState::new(&g);
        state.init_global();

        let snapshot = state.test_state_snapshot();
        assert_eq!(snapshot.y, vec![-12, -1, -129, -50, -31, -130]);
        assert_eq!(snapshot.matching, vec![-1, -1, 4, 5, 2, 3]);
        assert_eq!(snapshot.flags, vec![PLUS, PLUS, FREE, FREE, FREE, FREE]);
        assert_eq!(snapshot.is_outer, vec![true; 6]);
        assert_eq!(snapshot.tree_num, 2);
        assert_eq!(
            edge_slacks_by_endpoints(&state),
            vec![
                ((0, 3), 0),
                ((0, 5), 0),
                ((1, 2), 0),
                ((1, 4), 0),
                ((1, 5), 207),
                ((2, 3), 85),
                ((2, 4), 0),
                ((2, 5), 351),
                ((3, 4), 181),
                ((3, 5), 0),
            ],
        );
    }

    fn case_224_pre_first_dual_steps() -> Vec<GenericPrimalStepTrace> {
        let g = build_graph(
            6,
            &[
                (1, 2, -65),
                (0, 5, -71),
                (3, 4, 50),
                (3, 5, -90),
                (2, 4, -80),
                (1, 4, -16),
                (0, 3, -31),
                (1, 5, 38),
                (2, 3, -47),
                (2, 5, 46),
            ],
        );
        let mut state = BlossomVState::new(&g);
        state.init_global();
        state.mark_tree_roots_processed();
        while state.generic_primal_pass_once() {}
        state.test_generic_primal_steps().to_vec()
    }

    fn case_224_state_at_post_first_dual() -> BlossomVState<Vcsr> {
        let g = build_graph(
            6,
            &[
                (1, 2, -65),
                (0, 5, -71),
                (3, 4, 50),
                (3, 5, -90),
                (2, 4, -80),
                (1, 4, -16),
                (0, 3, -31),
                (1, 5, 38),
                (2, 3, -47),
                (2, 5, 46),
            ],
        );
        let leaked = Box::leak(Box::new(g));
        let mut state = BlossomVState::new(leaked);
        state.init_global();
        state.mark_tree_roots_processed();
        while state.generic_primal_pass_once() {}
        assert!(state.update_duals(), "first dual update should succeed on case #224");
        state
    }

    #[test]
    fn test_case_224_matches_sorted_cpp_first_generic_grow_after() {
        let steps = case_224_pre_first_dual_steps();
        assert!(
            steps.len() >= 2,
            "missing compacted first generic pass on case #224: only {} steps recorded",
            steps.len()
        );
        assert_eq!(steps[0].event, GenericPrimalEvent::Grow { edge: (0, 5), plus: 0, free: 5 });
        assert_eq!(steps[1].event, GenericPrimalEvent::Shrink { edge: (0, 3), left: 3, right: 0 });
    }

    #[test]
    fn test_case_224_after_first_grow_exposes_sorted_cpp_shrink_03() {
        let steps = case_224_pre_first_dual_steps();
        assert_eq!(
            steps.get(1).map(|step| step.event.clone()),
            Some(GenericPrimalEvent::Shrink { edge: (0, 3), left: 3, right: 0 }),
            "the compacted first Rust generic pass on case #224 should expose and consume shrink (0,3)"
        );
    }

    #[test]
    fn test_case_224_matches_sorted_cpp_generic_event_prefix() {
        let events = case_224_pre_first_dual_steps()
            .iter()
            .map(|step| step.event.clone())
            .collect::<Vec<_>>();

        assert_eq!(
            events,
            vec![
                GenericPrimalEvent::Grow { edge: (0, 5), plus: 0, free: 5 },
                GenericPrimalEvent::Shrink { edge: (0, 3), left: 3, right: 0 },
                GenericPrimalEvent::Grow { edge: (1, 4), plus: 1, free: 4 },
                GenericPrimalEvent::Shrink { edge: (1, 2), left: 2, right: 1 },
            ]
        );
    }

    #[test]
    fn test_case_224_matches_sorted_cpp_dual_then_augment() {
        let mut state = case_224_state_at_post_first_dual();
        let after_dual = state.test_state_snapshot();
        assert_eq!(after_dual.y, vec![-12, -1, -129, -50, -31, -130]);
        assert_eq!(
            edge_slacks_by_endpoints(&state),
            vec![
                ((0, 3), 0),
                ((0, 5), 0),
                ((1, 2), 0),
                ((1, 4), 0),
                ((1, 5), 207),
                ((2, 3), 85),
                ((2, 4), 0),
                ((2, 5), 351),
                ((3, 4), 181),
                ((3, 5), 0),
            ],
        );

        state.mark_tree_roots_processed();
        assert!(state.generic_primal_pass_once(), "missing sorted-order augment after dual update");
        let last = state.test_generic_primal_steps().last().unwrap();
        match last.event {
            GenericPrimalEvent::Augment { edge, left, right } => {
                assert_eq!(edge, (2, 3));
                assert_ne!(left, right);
                assert!(left as usize >= state.node_num || right as usize >= state.node_num);
            }
            ref other => panic!("expected sorted-order augment on edge (2,3), got {other:?}"),
        }
    }

    #[test]
    fn test_case_224_post_dual_edge_23_is_augment_ready() {
        let state = case_224_state_at_post_first_dual();

        let edge_23 = find_edge_idx(&state, 2, 3);
        let outer_u = state.edge_head_outer(edge_23, 0);
        let outer_v = state.edge_head_outer(edge_23, 1);
        let root_u = state.find_tree_root(outer_u);
        let root_v = state.find_tree_root(outer_v);
        let eps_u = state.tree_eps(root_u);
        let eps_v = state.tree_eps(root_v);
        let slack = state.edges[edge_23 as usize].slack;

        assert_ne!(outer_u, NONE, "edge (2,3) lost its first outer endpoint after the dual step");
        assert_ne!(outer_v, NONE, "edge (2,3) lost its second outer endpoint after the dual step");
        assert_ne!(
            outer_u, outer_v,
            "edge (2,3) collapsed into one outer blossom after the dual step"
        );
        assert_ne!(
            root_u, NONE,
            "edge (2,3) first outer endpoint has no tree root after the dual step"
        );
        assert_ne!(
            root_v, NONE,
            "edge (2,3) second outer endpoint has no tree root after the dual step"
        );
        assert_ne!(
            root_u, root_v,
            "edge (2,3) endpoints ended up in the same tree after the dual step"
        );
        assert!(
            slack <= eps_u + eps_v,
            "edge (2,3) is not augment-ready after dual update: slack={slack}, eps_u={eps_u}, eps_v={eps_v}, outer_u={outer_u}, outer_v={outer_v}, root_u={root_u}, root_v={root_v}"
        );
        assert_eq!(
            state
                .find_scheduler_global_augment_edge()
                .map(|(e_idx, _, _)| normalized_edge_pair(state.edges[e_idx as usize].head0)),
            Some((2, 3)),
            "after the sorted-order dual update, C++ immediately augments on edge (2,3)"
        );
    }

    #[test]
    fn test_case_24_matches_sorted_cpp_first_generic_grow_after() {
        let g = build_graph(
            20,
            &[
                (0, 5, 9),
                (0, 11, -19),
                (1, 4, 13),
                (1, 9, -28),
                (1, 12, -49),
                (1, 13, 84),
                (1, 17, 78),
                (2, 7, 46),
                (2, 10, 92),
                (3, 6, 89),
                (3, 14, 36),
                (4, 17, -91),
                (5, 8, -87),
                (5, 11, -70),
                (5, 15, -39),
                (5, 16, 65),
                (5, 18, -60),
                (5, 19, 60),
                (6, 11, -43),
                (6, 14, 9),
                (7, 10, 86),
                (8, 9, -53),
                (8, 11, 59),
                (8, 13, 27),
                (8, 16, -70),
                (8, 19, 3),
                (9, 10, 95),
                (9, 13, -74),
                (9, 16, -2),
                (9, 18, 0),
                (9, 19, 45),
                (10, 12, 88),
                (10, 13, 9),
                (12, 13, -23),
                (13, 17, -89),
                (15, 16, -37),
                (15, 18, 46),
                (15, 19, -10),
                (16, 18, 94),
                (16, 19, -91),
                (18, 19, 7),
            ],
        );
        let mut state = BlossomVState::new(&g);
        state.init_global();
        state.mark_tree_roots_processed();
        assert!(state.generic_primal_pass_once());
        let steps = state.test_generic_primal_steps();
        assert!(
            steps.len() >= 2,
            "missing compacted first generic pass on case #24: only {} steps recorded",
            steps.len()
        );
        assert_eq!(steps[0].event, GenericPrimalEvent::Grow { edge: (3, 14), plus: 3, free: 14 });
        assert_eq!(steps[1].event, GenericPrimalEvent::Shrink { edge: (3, 6), left: 6, right: 3 });
    }

    #[test]
    fn test_case_24_matches_sorted_cpp_first_shrink_after() {
        let g = build_graph(
            20,
            &[
                (0, 5, 9),
                (0, 11, -19),
                (1, 4, 13),
                (1, 9, -28),
                (1, 12, -49),
                (1, 13, 84),
                (1, 17, 78),
                (2, 7, 46),
                (2, 10, 92),
                (3, 6, 89),
                (3, 14, 36),
                (4, 17, -91),
                (5, 8, -87),
                (5, 11, -70),
                (5, 15, -39),
                (5, 16, 65),
                (5, 18, -60),
                (5, 19, 60),
                (6, 11, -43),
                (6, 14, 9),
                (7, 10, 86),
                (8, 9, -53),
                (8, 11, 59),
                (8, 13, 27),
                (8, 16, -70),
                (8, 19, 3),
                (9, 10, 95),
                (9, 13, -74),
                (9, 16, -2),
                (9, 18, 0),
                (9, 19, 45),
                (10, 12, 88),
                (10, 13, 9),
                (12, 13, -23),
                (13, 17, -89),
                (15, 16, -37),
                (15, 18, 46),
                (15, 19, -10),
                (16, 18, 94),
                (16, 19, -91),
                (18, 19, 7),
            ],
        );
        let mut state = BlossomVState::new(&g);
        state.init_global();
        state.mark_tree_roots_processed();
        assert!(state.generic_primal_pass_once());
        let steps = state.test_generic_primal_steps();
        assert!(
            steps.len() >= 2,
            "missing compacted first generic pass on case #24: only {} steps recorded",
            steps.len()
        );
        assert_eq!(steps[1].event, GenericPrimalEvent::Shrink { edge: (3, 6), left: 6, right: 3 });
    }

    #[test]
    fn test_ground_truth_case_24_with_budget() {
        let g = build_graph(
            20,
            &[
                (0, 5, 9),
                (0, 11, -19),
                (1, 4, 13),
                (1, 9, -28),
                (1, 12, -49),
                (1, 13, 84),
                (1, 17, 78),
                (2, 7, 46),
                (2, 10, 92),
                (3, 6, 89),
                (3, 14, 36),
                (4, 17, -91),
                (5, 8, -87),
                (5, 11, -70),
                (5, 15, -39),
                (5, 16, 65),
                (5, 18, -60),
                (5, 19, 60),
                (6, 11, -43),
                (6, 14, 9),
                (7, 10, 86),
                (8, 9, -53),
                (8, 11, 59),
                (8, 13, 27),
                (8, 16, -70),
                (8, 19, 3),
                (9, 10, 95),
                (9, 13, -74),
                (9, 16, -2),
                (9, 18, 0),
                (9, 19, 45),
                (10, 12, 88),
                (10, 13, 9),
                (12, 13, -23),
                (13, 17, -89),
                (15, 16, -37),
                (15, 18, 46),
                (15, 19, -10),
                (16, 18, 94),
                (16, 19, -91),
                (18, 19, 7),
            ],
        );
        let pairs = BlossomVState::new(&g)
            .solve_with_test_budget(4096, 4096)
            .expect("case #24 should have a perfect matching");
        assert_eq!(
            pairs,
            vec![
                (0, 5),
                (1, 12),
                (2, 7),
                (3, 14),
                (4, 17),
                (6, 11),
                (8, 9),
                (10, 13),
                (15, 18),
                (16, 19),
            ]
        );
        let cost: i32 = pairs
            .iter()
            .map(|&(u, v)| {
                match (u, v) {
                    (0, 5) => 9,
                    (1, 12) => -49,
                    (2, 7) => 46,
                    (3, 14) => 36,
                    (4, 17) => -91,
                    (6, 11) => -43,
                    (8, 9) => -53,
                    (10, 13) => 9,
                    (15, 18) => 46,
                    (16, 19) => -91,
                    _ => panic!("unexpected pair ({u}, {v}) in case #24"),
                }
            })
            .sum();
        assert_eq!(cost, -181);
    }

    #[test]
    fn test_ground_truth_case_214_with_budget() {
        let edges = [
            (0, 1, 90),
            (0, 2, -66),
            (0, 3, -13),
            (0, 4, -83),
            (0, 5, 73),
            (0, 7, -70),
            (0, 9, -67),
            (0, 11, 39),
            (0, 12, 40),
            (0, 13, -57),
            (0, 14, -32),
            (0, 18, -54),
            (0, 19, 73),
            (0, 20, -60),
            (1, 2, -96),
            (1, 3, -17),
            (1, 4, 12),
            (1, 5, -20),
            (1, 6, 42),
            (1, 8, -5),
            (1, 11, 31),
            (1, 14, 0),
            (1, 16, -90),
            (1, 21, -90),
            (2, 3, -88),
            (2, 4, -38),
            (2, 5, 80),
            (2, 7, 7),
            (2, 9, -27),
            (2, 10, -30),
            (2, 12, 66),
            (2, 17, -46),
            (2, 18, -66),
            (2, 19, -61),
            (3, 4, -11),
            (3, 5, 90),
            (3, 6, 77),
            (3, 8, 82),
            (3, 10, 96),
            (3, 11, -90),
            (3, 12, -91),
            (3, 18, 30),
            (4, 6, -93),
            (4, 7, -5),
            (4, 8, -38),
            (4, 10, 2),
            (4, 14, 57),
            (4, 18, -11),
            (5, 6, 98),
            (5, 7, 91),
            (5, 15, -63),
            (5, 21, -83),
            (6, 8, 89),
            (6, 12, 58),
            (6, 13, 98),
            (7, 9, -27),
            (7, 10, -64),
            (7, 15, 59),
            (7, 16, -66),
            (7, 19, -5),
            (8, 9, -28),
            (8, 11, -29),
            (8, 13, -70),
            (8, 15, 38),
            (9, 17, -91),
            (10, 17, -2),
            (10, 20, 28),
            (10, 21, -11),
            (12, 13, -76),
            (12, 16, -8),
            (12, 19, 68),
            (13, 14, -57),
            (13, 20, -42),
            (13, 21, -16),
            (14, 15, 78),
            (14, 16, -2),
            (16, 17, -47),
            (18, 20, -35),
        ];
        let g = build_graph(22, &edges);

        let matching = BlossomVState::new(&g)
            .solve_with_test_budget(4096, 20_000)
            .expect("case #214 should have a perfect matching");
        let matching = normalize_pairs(&matching);

        validate_matching(22, &matching);
        assert_eq!(matching_cost(&edges, &matching), -697);
        assert_eq!(
            matching,
            vec![
                (0, 14),
                (1, 21),
                (2, 19),
                (3, 11),
                (4, 6),
                (5, 15),
                (7, 10),
                (8, 13),
                (9, 17),
                (12, 16),
                (18, 20),
            ]
        );
    }

    #[test]
    fn test_ground_truth_case_214_public_matrix_with_budget() {
        let edges = [
            (0, 1, 90),
            (0, 2, -66),
            (0, 3, -13),
            (0, 4, -83),
            (0, 5, 73),
            (0, 7, -70),
            (0, 9, -67),
            (0, 11, 39),
            (0, 12, 40),
            (0, 13, -57),
            (0, 14, -32),
            (0, 18, -54),
            (0, 19, 73),
            (0, 20, -60),
            (1, 2, -96),
            (1, 3, -17),
            (1, 4, 12),
            (1, 5, -20),
            (1, 6, 42),
            (1, 8, -5),
            (1, 11, 31),
            (1, 14, 0),
            (1, 16, -90),
            (1, 21, -90),
            (2, 3, -88),
            (2, 4, -38),
            (2, 5, 80),
            (2, 7, 7),
            (2, 9, -27),
            (2, 10, -30),
            (2, 12, 66),
            (2, 17, -46),
            (2, 18, -66),
            (2, 19, -61),
            (3, 4, -11),
            (3, 5, 90),
            (3, 6, 77),
            (3, 8, 82),
            (3, 10, 96),
            (3, 11, -90),
            (3, 12, -91),
            (3, 18, 30),
            (4, 6, -93),
            (4, 7, -5),
            (4, 8, -38),
            (4, 10, 2),
            (4, 14, 57),
            (4, 18, -11),
            (5, 6, 98),
            (5, 7, 91),
            (5, 15, -63),
            (5, 21, -83),
            (6, 8, 89),
            (6, 12, 58),
            (6, 13, 98),
            (7, 9, -27),
            (7, 10, -64),
            (7, 15, 59),
            (7, 16, -66),
            (7, 19, -5),
            (8, 9, -28),
            (8, 11, -29),
            (8, 13, -70),
            (8, 15, 38),
            (9, 17, -91),
            (10, 17, -2),
            (10, 20, 28),
            (10, 21, -11),
            (12, 13, -76),
            (12, 16, -8),
            (12, 19, 68),
            (13, 14, -57),
            (13, 20, -42),
            (13, 21, -16),
            (14, 15, 78),
            (14, 16, -2),
            (16, 17, -47),
            (18, 20, -35),
        ];

        type PublicVcsr = crate::impls::ValuedCSR2D<usize, usize, usize, i32>;
        let mut sorted_edges: Vec<(usize, usize, i32)> = Vec::new();
        for &(i, j, w) in &edges {
            if i == j {
                continue;
            }
            let (lo, hi) = if i < j { (i, j) } else { (j, i) };
            sorted_edges.push((lo, hi, w));
            sorted_edges.push((hi, lo, w));
        }
        sorted_edges.sort_unstable();
        sorted_edges.dedup_by(|a, b| a.0 == b.0 && a.1 == b.1);

        let mut g: PublicVcsr = crate::traits::SparseMatrixMut::with_sparse_shaped_capacity(
            (22, 22),
            sorted_edges.len(),
        );
        for (r, c, v) in sorted_edges {
            crate::traits::MatrixMut::add(&mut g, (r, c, v)).unwrap();
        }

        let matching = BlossomVState::new(&g)
            .solve_with_test_budget(4096, 20_000)
            .expect("case #214 should have a perfect matching");
        let matching = normalize_pairs(&matching);

        validate_matching(22, &matching);
        assert_eq!(matching_cost(&edges, &matching), -697);
        assert_eq!(
            matching,
            vec![
                (0, 14),
                (1, 21),
                (2, 19),
                (3, 11),
                (4, 6),
                (5, 15),
                (7, 10),
                (8, 13),
                (9, 17),
                (12, 16),
                (18, 20),
            ]
        );
    }

    fn case_214_edge_list() -> Vec<(usize, usize, i32)> {
        vec![
            (0, 1, 90),
            (0, 2, -66),
            (0, 3, -13),
            (0, 4, -83),
            (0, 5, 73),
            (0, 7, -70),
            (0, 9, -67),
            (0, 11, 39),
            (0, 12, 40),
            (0, 13, -57),
            (0, 14, -32),
            (0, 18, -54),
            (0, 19, 73),
            (0, 20, -60),
            (1, 2, -96),
            (1, 3, -17),
            (1, 4, 12),
            (1, 5, -20),
            (1, 6, 42),
            (1, 8, -5),
            (1, 11, 31),
            (1, 14, 0),
            (1, 16, -90),
            (1, 21, -90),
            (2, 3, -88),
            (2, 4, -38),
            (2, 5, 80),
            (2, 7, 7),
            (2, 9, -27),
            (2, 10, -30),
            (2, 12, 66),
            (2, 17, -46),
            (2, 18, -66),
            (2, 19, -61),
            (3, 4, -11),
            (3, 5, 90),
            (3, 6, 77),
            (3, 8, 82),
            (3, 10, 96),
            (3, 11, -90),
            (3, 12, -91),
            (3, 18, 30),
            (4, 6, -93),
            (4, 7, -5),
            (4, 8, -38),
            (4, 10, 2),
            (4, 14, 57),
            (4, 18, -11),
            (5, 6, 98),
            (5, 7, 91),
            (5, 15, -63),
            (5, 21, -83),
            (6, 8, 89),
            (6, 12, 58),
            (6, 13, 98),
            (7, 9, -27),
            (7, 10, -64),
            (7, 15, 59),
            (7, 16, -66),
            (7, 19, -5),
            (8, 9, -28),
            (8, 11, -29),
            (8, 13, -70),
            (8, 15, 38),
            (9, 17, -91),
            (10, 17, -2),
            (10, 20, 28),
            (10, 21, -11),
            (12, 13, -76),
            (12, 16, -8),
            (12, 19, 68),
            (13, 14, -57),
            (13, 20, -42),
            (13, 21, -16),
            (14, 15, 78),
            (14, 16, -2),
            (16, 17, -47),
            (18, 20, -35),
        ]
    }

    fn case_214_pre_first_dual_steps() -> Vec<GenericPrimalStepTrace> {
        let edges = case_214_edge_list();
        let g = build_graph(22, &edges);
        let mut state = BlossomVState::new(&g);
        state.init_global();
        state.mark_tree_roots_processed();

        let start = state.test_generic_primal_steps().len();
        let mut passes = 0usize;
        while state.generic_primal_pass_once() {
            passes += 1;
            assert!(passes <= 4, "generic phase should stall before first dual on case #214");
        }

        state.test_generic_primal_steps()[start..].to_vec()
    }

    fn case_214_state_before_first_dual() -> BlossomVState<Vcsr> {
        let edges = case_214_edge_list();
        let g = build_graph(22, &edges);
        let leaked = Box::leak(Box::new(g));
        let mut state = BlossomVState::new(leaked);
        state.init_global();
        state.mark_tree_roots_processed();

        let mut passes = 0usize;
        while state.generic_primal_pass_once() {
            passes += 1;
            assert!(passes <= 4, "generic phase should stall before first dual on case #214");
        }

        state
    }

    fn case_24943_edges() -> Vec<(usize, usize, i32)> {
        vec![
            (0, 2, -30),
            (0, 8, 16),
            (0, 28, -13),
            (0, 29, 36),
            (1, 2, -4),
            (1, 3, 5),
            (2, 3, -42),
            (2, 4, -53),
            (3, 5, 74),
            (3, 29, -3),
            (4, 6, 67),
            (4, 20, -76),
            (5, 6, 5),
            (5, 7, -67),
            (6, 7, -55),
            (6, 8, 74),
            (7, 9, 82),
            (7, 19, 44),
            (8, 9, 97),
            (8, 10, 8),
            (8, 16, 97),
            (9, 10, 95),
            (9, 11, 47),
            (10, 11, -21),
            (10, 12, -70),
            (11, 12, -69),
            (11, 13, 24),
            (12, 13, 80),
            (12, 14, -90),
            (13, 14, -44),
            (13, 15, 61),
            (14, 15, 11),
            (14, 16, -72),
            (15, 16, 96),
            (15, 17, 83),
            (16, 17, -67),
            (16, 29, 42),
            (17, 18, -41),
            (17, 19, -89),
            (18, 19, -61),
            (18, 20, 91),
            (19, 20, -10),
            (19, 21, 58),
            (20, 21, -38),
            (20, 22, -39),
            (21, 22, 60),
            (21, 23, 65),
            (22, 23, -41),
            (22, 24, -14),
            (23, 24, 36),
            (23, 25, -24),
            (24, 25, 25),
            (24, 26, -76),
            (25, 26, -86),
            (25, 27, -45),
            (26, 27, -35),
            (26, 28, 57),
            (27, 28, 97),
            (27, 29, -88),
            (28, 29, 15),
        ]
    }

    fn case_24595_edges() -> Vec<(usize, usize, i32)> {
        vec![
            (0, 7, -6),
            (0, 9, 14),
            (0, 16, -17),
            (1, 2, 34),
            (1, 3, 98),
            (1, 6, 58),
            (1, 10, 24),
            (1, 11, -22),
            (1, 15, 49),
            (2, 3, 72),
            (2, 5, 55),
            (2, 6, 72),
            (2, 8, -82),
            (2, 10, 90),
            (2, 11, -13),
            (2, 15, 26),
            (2, 17, 93),
            (3, 5, -51),
            (3, 6, -66),
            (3, 8, 10),
            (3, 9, -84),
            (3, 11, -93),
            (3, 14, 43),
            (3, 15, -9),
            (3, 17, -50),
            (4, 12, 45),
            (4, 13, 43),
            (4, 14, 16),
            (5, 6, 84),
            (5, 7, 32),
            (5, 8, -92),
            (5, 9, -58),
            (5, 14, 66),
            (5, 15, 76),
            (5, 16, 47),
            (5, 17, -59),
            (6, 8, -9),
            (6, 9, -1),
            (6, 10, 70),
            (6, 15, -55),
            (6, 17, -23),
            (7, 9, 42),
            (7, 16, -98),
            (7, 17, 41),
            (8, 9, -71),
            (8, 14, -66),
            (8, 15, 10),
            (8, 17, 24),
            (9, 15, -75),
            (9, 16, -62),
            (9, 17, 35),
            (10, 11, -67),
            (10, 15, 53),
            (12, 13, 17),
            (14, 17, -75),
            (15, 17, -18),
            (16, 17, 20),
        ]
    }

    fn case_28832_edges() -> Vec<(usize, usize, i32)> {
        vec![
            (0, 2, -90),
            (0, 3, 63),
            (0, 5, 80),
            (0, 6, 46),
            (0, 12, -15),
            (0, 17, 98),
            (0, 20, -61),
            (0, 21, -7),
            (1, 2, -79),
            (1, 5, 15),
            (1, 6, -49),
            (1, 7, -2),
            (1, 11, -46),
            (1, 18, -25),
            (1, 20, -5),
            (2, 3, -32),
            (2, 4, -10),
            (2, 5, -39),
            (2, 7, -75),
            (2, 16, 41),
            (2, 19, -11),
            (2, 20, 15),
            (2, 21, -68),
            (3, 4, -91),
            (3, 5, -84),
            (3, 6, 86),
            (3, 7, -96),
            (3, 18, 70),
            (3, 19, 62),
            (3, 20, -34),
            (3, 21, 68),
            (4, 6, -93),
            (4, 7, 66),
            (4, 8, 78),
            (4, 9, 98),
            (4, 19, -72),
            (5, 6, -26),
            (5, 7, -81),
            (5, 9, 20),
            (5, 10, 20),
            (5, 18, -67),
            (5, 19, 12),
            (5, 21, 73),
            (6, 7, 75),
            (6, 8, 78),
            (6, 9, -70),
            (6, 11, 33),
            (6, 19, -80),
            (6, 20, 74),
            (7, 8, -27),
            (7, 10, 98),
            (7, 12, 37),
            (7, 18, -47),
            (7, 19, 9),
            (8, 9, -83),
            (8, 10, -59),
            (8, 11, -25),
            (8, 13, -83),
            (8, 14, -95),
            (8, 16, 27),
            (8, 18, -87),
            (8, 19, -19),
            (9, 10, 88),
            (9, 11, 15),
            (9, 12, -3),
            (9, 13, -56),
            (9, 14, 54),
            (10, 11, -76),
            (10, 12, 72),
            (10, 13, -42),
            (10, 14, 18),
            (10, 15, 17),
            (10, 18, -12),
            (11, 12, -51),
            (11, 14, -86),
            (11, 15, 52),
            (11, 16, 10),
            (12, 13, 19),
            (12, 14, 4),
            (12, 15, 32),
            (12, 16, -75),
            (12, 17, -57),
            (12, 20, 63),
            (13, 14, -67),
            (13, 16, 4),
            (13, 17, 61),
            (13, 18, -61),
            (13, 19, -35),
            (14, 16, 18),
            (14, 17, -82),
            (14, 18, 94),
            (14, 21, -41),
            (15, 16, -36),
            (15, 17, -54),
            (15, 18, 99),
            (15, 19, 3),
            (15, 20, 71),
            (16, 18, -66),
            (16, 19, 0),
            (16, 20, 37),
            (16, 21, -66),
            (17, 18, 30),
            (17, 19, 69),
            (17, 20, 20),
            (17, 21, 86),
            (18, 19, -63),
            (18, 20, 17),
            (18, 21, -24),
            (19, 20, -71),
            (20, 21, 50),
        ]
    }

    fn case_21222_edges() -> Vec<(usize, usize, i32)> {
        vec![
            (0, 2, 6),
            (0, 11, -82),
            (0, 18, 81),
            (0, 25, 69),
            (0, 26, -22),
            (0, 27, -22),
            (1, 2, -51),
            (1, 3, -85),
            (1, 4, 59),
            (1, 7, -50),
            (1, 14, 71),
            (1, 26, -30),
            (1, 27, -21),
            (2, 3, -23),
            (2, 4, 85),
            (2, 5, 52),
            (3, 4, 41),
            (3, 5, -50),
            (3, 6, -23),
            (4, 5, -96),
            (4, 6, -42),
            (4, 7, -29),
            (5, 6, 79),
            (5, 7, 18),
            (5, 8, -35),
            (6, 7, -51),
            (6, 8, 84),
            (6, 9, -91),
            (6, 10, 25),
            (7, 8, -34),
            (7, 10, -48),
            (8, 9, 87),
            (8, 10, -58),
            (8, 11, 46),
            (8, 15, 10),
            (9, 10, 33),
            (9, 11, 18),
            (9, 12, 71),
            (10, 11, -38),
            (10, 12, 8),
            (11, 12, -38),
            (11, 13, -50),
            (11, 14, -79),
            (12, 13, 83),
            (12, 14, -42),
            (12, 15, 12),
            (12, 19, 55),
            (13, 14, 86),
            (13, 15, -66),
            (13, 16, 53),
            (14, 15, -79),
            (14, 17, 75),
            (14, 27, -86),
            (15, 16, 17),
            (15, 18, 14),
            (16, 17, -5),
            (16, 18, -6),
            (16, 19, 19),
            (17, 18, 65),
            (17, 19, -29),
            (17, 20, -92),
            (18, 19, 16),
            (18, 20, -56),
            (18, 21, -59),
            (19, 20, 42),
            (19, 22, 56),
            (20, 21, 73),
            (20, 22, -94),
            (20, 23, -55),
            (21, 22, 90),
            (21, 23, -33),
            (21, 24, 92),
            (22, 23, 71),
            (22, 24, -69),
            (22, 25, 7),
            (23, 24, -74),
            (23, 25, 92),
            (23, 26, -81),
            (24, 25, -79),
            (24, 26, -64),
            (24, 27, -92),
            (25, 26, 79),
            (25, 27, 60),
            (26, 27, -4),
        ]
    }

    fn assert_tree_navigation_invariants<M: SparseValuedMatrix2D + ?Sized>(state: &BlossomVState<M>)
    where
        M::Value: Number + AsPrimitive<i64>,
        M::RowIndex: PositiveInteger,
        M::ColumnIndex: PositiveInteger,
    {
        for (idx, node) in state.nodes.iter().enumerate() {
            if !node.is_outer || node.flag != PLUS || node.is_tree_root {
                continue;
            }
            assert_ne!(
                node.match_arc, NONE,
                "outer non-root PLUS node {idx} lost match_arc during tree navigation"
            );
            let minus = state.arc_head_outer(node.match_arc);
            assert_ne!(
                minus, NONE,
                "outer non-root PLUS node {idx} resolves to NONE minus via match_arc={}",
                node.match_arc
            );
            assert_eq!(
                state.nodes[minus as usize].flag, MINUS,
                "outer non-root PLUS node {idx} points to non-MINUS node {minus}"
            );
            assert_ne!(
                state.nodes[minus as usize].tree_parent_arc, NONE,
                "MINUS node {minus} lost tree_parent_arc while still attached to PLUS node {idx}"
            );
            let parent_plus = state.arc_head_outer(state.nodes[minus as usize].tree_parent_arc);
            assert_ne!(
                parent_plus, NONE,
                "MINUS node {minus} resolves to NONE parent via tree_parent_arc={}",
                state.nodes[minus as usize].tree_parent_arc
            );
        }
    }

    fn solve_case_24943_with_tree_checks<M: SparseValuedMatrix2D + ?Sized>(
        state: &mut BlossomVState<M>,
        max_outer_iters: usize,
        max_inner_iters: usize,
    ) -> Result<(), BlossomVError>
    where
        M::Value: Number + AsPrimitive<i64>,
        M::RowIndex: PositiveInteger,
        M::ColumnIndex: PositiveInteger,
    {
        state.init_global();
        assert_tree_navigation_invariants(state);

        let mut outer_iters = 0usize;
        loop {
            if state.tree_num == 0 {
                break;
            }
            outer_iters += 1;
            assert!(outer_iters <= max_outer_iters, "case #24943 exceeded outer iteration budget");

            state.mark_tree_roots_processed();
            assert_tree_navigation_invariants(state);

            let mut progressed = false;
            let mut inner_iters = 0usize;
            loop {
                inner_iters += 1;
                assert!(
                    inner_iters <= max_inner_iters,
                    "case #24943 exceeded inner iteration budget"
                );
                let step = state.generic_primal_pass_once();
                assert_tree_navigation_invariants(state);
                if !step {
                    break;
                }
                progressed = true;
                if state.tree_num == 0 {
                    break;
                }
            }

            if state.tree_num == 0 {
                break;
            }

            if !progressed && !state.update_duals() {
                return Err(BlossomVError::NoPerfectMatching);
            }
            assert_tree_navigation_invariants(state);
        }

        Ok(())
    }

    #[test]
    fn test_case_214_matches_cpp_first_generic_grow_after() {
        let steps = case_214_pre_first_dual_steps();
        assert!(!steps.is_empty(), "missing first compacted pre-dual step on case #214");
        assert_eq!(steps[0].event, GenericPrimalEvent::Grow { edge: (0, 20), plus: 0, free: 20 });
    }

    #[test]
    fn test_case_214_matches_cpp_second_generic_grow_after() {
        let steps = case_214_pre_first_dual_steps();
        assert!(
            steps.len() >= 3,
            "missing second compacted pre-dual grow on case #214: only {} steps recorded",
            steps.len()
        );
        assert_eq!(steps[2].event, GenericPrimalEvent::Grow { edge: (1, 21), plus: 1, free: 21 });
    }

    #[test]
    fn test_case_214_matches_cpp_first_generic_shrink_after() {
        let steps = case_214_pre_first_dual_steps();
        assert!(
            steps.len() >= 2,
            "missing first compacted pre-dual shrink on case #214: only {} steps recorded",
            steps.len()
        );
        assert_eq!(
            steps[1].event,
            GenericPrimalEvent::Shrink { edge: (0, 18), left: 18, right: 0 }
        );
    }

    #[test]
    fn test_case_214_scheduler_local_eps_matches_visible_scan_before_first_dual() {
        let visible_scan_local_eps = |state: &BlossomVState<_>, root: u32| -> i64 {
            let mut eps = i64::MAX;
            for e_idx in 0..state.edge_num {
                let u = state.edge_head_outer(e_idx as u32, 0);
                let v = state.edge_head_outer(e_idx as u32, 1);
                if u == NONE || v == NONE || u == v {
                    continue;
                }

                let slack = state.edges[e_idx].slack;
                if state.nodes[u as usize].flag == PLUS
                    && state.nodes[u as usize].is_processed
                    && state.find_tree_root(u) == root
                    && state.nodes[v as usize].flag == FREE
                {
                    eps = eps.min(slack);
                }
                if state.nodes[v as usize].flag == PLUS
                    && state.nodes[v as usize].is_processed
                    && state.find_tree_root(v) == root
                    && state.nodes[u as usize].flag == FREE
                {
                    eps = eps.min(slack);
                }
            }

            for i in 0..state.nodes.len() {
                let node = &state.nodes[i];
                if node.is_blossom
                    && node.is_outer
                    && node.flag == MINUS
                    && state.find_tree_root(i as u32) == root
                {
                    let candidate = if node.match_arc != NONE
                        && (arc_edge(node.match_arc) as usize) < state.edge_num
                    {
                        state.edges[arc_edge(node.match_arc) as usize].slack
                    } else {
                        node.y
                    };
                    eps = eps.min(candidate);
                }
            }

            eps
        };

        let mut state = case_214_state_before_first_dual();
        let roots = state.current_root_list();
        assert!(!roots.is_empty(), "case #214 should have active roots before first dual");

        for root in roots {
            let _ = state.tree_min_pq00_local_for_step3(root);
            assert_eq!(
                state.compute_tree_local_eps(root),
                visible_scan_local_eps(&state, root),
                "case #214 local eps mismatch for root {root}",
            );
        }
    }

    #[test]
    fn test_case_9_does_not_expose_missed_local_expand_state() {
        let edges = case_9_edges();
        let g = build_graph(18, &edges);
        let leaked = Box::leak(Box::new(g));
        let mut state = BlossomVState::new(leaked);
        state.init_global();

        for outer_iter in 0..64 {
            let mut progressed = true;
            let mut inner_iters = 0usize;
            while progressed {
                progressed = generic_primal_pass_without_global_expand_fallback_once(&mut state);
                inner_iters += usize::from(progressed);
                assert!(
                    inner_iters <= 512,
                    "case 9 did not stall before finding a local/global expand mismatch"
                );
                if state.tree_num == 0 {
                    return;
                }
            }

            if let Some(blossom) = find_global_expand_fallback_blossom_for_test(&state) {
                let root = state.find_tree_root(blossom);
                let eps_root = state.tree_eps(root);
                let match_edge = arc_edge(state.nodes[blossom as usize].match_arc);
                let queue_owner = state.edge_queue_owner(match_edge);
                let outer0 = state.edge_head_outer(match_edge, 0);
                let outer1 = state.edge_head_outer(match_edge, 1);
                let raw = state.edges[match_edge as usize].head;
                assert_eq!(
                    state.find_tree_expand_blossom_with_eps(root, eps_root),
                    Some(blossom),
                    "case 9 missed local expand: blossom={blossom} root={root} eps_root={eps_root} match_edge={match_edge} owner={queue_owner:?} outer=({outer0},{outer1}) raw={raw:?} pq_blossoms={:?}",
                    state.scheduler_trees[root as usize].pq_blossoms,
                );
                return;
            }

            assert!(
                state.update_duals(),
                "case 9 failed dual update before exposing the missed local expand state at outer_iter={outer_iter}"
            );
        }

        panic!("case 9 did not expose a missed local expand state within the search budget");
    }

    #[test]
    fn test_case_6_does_not_expose_missed_local_expand_state() {
        let edges = case_honggfuzz_sigabrt_6_edges();
        let g = build_graph(26, &edges);
        let leaked = Box::leak(Box::new(g));
        let mut state = BlossomVState::new(leaked);
        state.init_global();

        for outer_iter in 0..128 {
            let mut progressed = true;
            let mut inner_iters = 0usize;
            while progressed {
                progressed = generic_primal_pass_without_global_expand_fallback_once(&mut state);
                inner_iters += usize::from(progressed);
                assert!(
                    inner_iters <= 2048,
                    "case 6 did not stall before finding a local/global expand mismatch"
                );
                if state.tree_num == 0 {
                    return;
                }
            }

            if let Some(blossom) = find_global_expand_fallback_blossom_for_test(&state) {
                let root = state.find_tree_root(blossom);
                let eps_root = state.tree_eps(root);
                let match_edge = arc_edge(state.nodes[blossom as usize].match_arc);
                let queue_owner = state.edge_queue_owner(match_edge);
                let outer0 = state.edge_head_outer(match_edge, 0);
                let outer1 = state.edge_head_outer(match_edge, 1);
                let raw = state.edges[match_edge as usize].head;
                assert_eq!(
                    state.find_tree_expand_blossom_with_eps(root, eps_root),
                    Some(blossom),
                    "case 6 missed local expand: blossom={blossom} root={root} eps_root={eps_root} match_edge={match_edge} owner={queue_owner:?} outer=({outer0},{outer1}) raw={raw:?} pq_blossoms={:?}",
                    state.scheduler_trees[root as usize].pq_blossoms,
                );
                return;
            }

            assert!(
                state.update_duals(),
                "case 6 failed dual update before exposing the missed local expand state at outer_iter={outer_iter}"
            );
        }

        panic!("case 6 did not expose a missed local expand state within the search budget");
    }

    #[test]
    fn test_case_14_exposes_missed_global_expand_fallback_state() {
        let edges = case_honggfuzz_sigabrt_14_edges();
        let g = build_graph(18, &edges);
        let leaked = Box::leak(Box::new(g));
        let mut state = BlossomVState::new(leaked);
        state.init_global();

        for outer_iter in 0..128 {
            let mut progressed = true;
            let mut inner_iters = 0usize;
            while progressed {
                progressed = generic_primal_pass_without_global_expand_fallback_once(&mut state);
                inner_iters += usize::from(progressed);
                assert!(
                    inner_iters <= 2048,
                    "case 14 did not stall before exposing the fallback-dependent expand state"
                );
                assert!(
                    state.tree_num != 0,
                    "case 14 unexpectedly solved without the global expand fallback at outer_iter={outer_iter}"
                );
            }

            if let Some(blossom) = find_global_expand_fallback_blossom_for_test(&state) {
                let root = state.find_tree_root(blossom);
                let eps_root = state.tree_eps(root);
                assert_eq!(
                    state.find_tree_expand_blossom_with_eps(root, eps_root),
                    None,
                    "case 14 no longer needs the global expand fallback"
                );
                return;
            }

            assert!(
                state.update_duals(),
                "case 14 failed dual update before exposing the fallback-dependent expand state at outer_iter={outer_iter}"
            );
        }

        panic!(
            "case 14 did not expose the fallback-dependent expand state within the search budget"
        );
    }

    #[test]
    fn test_case_14_next_primal_event_at_head_checkpoint() {
        let edges = case_honggfuzz_sigabrt_14_edges();
        let g = build_graph(18, &edges);
        let leaked = Box::leak(Box::new(g));
        let mut state = BlossomVState::new(leaked);
        state.init_global();

        for outer_iter in 0..128 {
            let roots = state.current_root_list();
            let at_head_checkpoint = roots == vec![19, 20]
                && state.scheduler_trees[19].pq0 == vec![33]
                && state.scheduler_trees[19].pq00_local == vec![47, 36, 38]
                && state.scheduler_trees[19].pq_blossoms.is_empty()
                && state.scheduler_trees[20].pq0 == vec![32, 21]
                && state.scheduler_trees[20].pq00_local == vec![22]
                && state.scheduler_trees[20].pq_blossoms.is_empty()
                && state.scheduler_tree_edges[5].head == [20, 19]
                && state.scheduler_tree_edges[5].pq00 == vec![42, 40]
                && state.scheduler_tree_edges[5].pq01[0].is_empty()
                && state.scheduler_tree_edges[5].pq01[1].is_empty();

            if at_head_checkpoint {
                let eps19 = state.tree_eps(19);
                let eps20 = state.tree_eps(20);
                let shrink19 = state.find_tree_shrink_edge_with_cap(19, eps19.saturating_mul(2));
                let shrink20 = state.find_tree_shrink_edge_with_cap(20, eps20.saturating_mul(2));
                let expand19 = state.find_tree_expand_blossom_with_eps(19, eps19);
                let expand20 = state.find_tree_expand_blossom_with_eps(20, eps20);
                let fallback = find_global_expand_fallback_blossom_for_test(&state);
                let before = state.test_generic_primal_steps().len();
                assert!(
                    state.generic_primal_pass_once(),
                    "case 14 checkpoint should still make primal progress"
                );
                let step = &state.test_generic_primal_steps()[before];
                assert_eq!(
                    step.event,
                    GenericPrimalEvent::Expand { blossom: 18 },
                    "case 14 diverged from committed HEAD at the checkpoint in outer_iter={outer_iter}; eps19={eps19} eps20={eps20} shrink19={shrink19:?} shrink20={shrink20:?} expand19={expand19:?} expand20={expand20:?} fallback={fallback:?} owner40={:?} owner42={:?} owner44={:?} pq_blossoms19={:?} pq_blossoms20={:?}",
                    state.edge_queue_owner(40),
                    state.edge_queue_owner(42),
                    state.edge_queue_owner(44),
                    state.scheduler_trees[19].pq_blossoms,
                    state.scheduler_trees[20].pq_blossoms,
                );
                return;
            }

            let mut progressed = true;
            let mut inner_iters = 0usize;
            while progressed {
                progressed = state.generic_primal_pass_once();
                inner_iters += usize::from(progressed);
                assert!(inner_iters <= 2048, "case 14 exceeded the inner-step budget");
                assert!(
                    state.tree_num != 0,
                    "case 14 solved before reaching the committed-HEAD checkpoint"
                );
            }

            assert!(
                state.update_duals(),
                "case 14 failed dual update before reaching the committed-HEAD checkpoint at outer_iter={outer_iter}"
            );
        }

        panic!("case 14 never reached the committed-HEAD checkpoint within the search budget");
    }

    #[test]
    fn test_ground_truth_case_24943_with_budget() {
        let edges = case_24943_edges();
        let g = build_graph(30, &edges);
        let state = BlossomVState::new(&g);
        let pairs = state.solve_with_test_budget(400, 2000).expect("case #24943 should solve");
        let mut sorted = pairs;
        for (u, v) in &mut sorted {
            if *u > *v {
                core::mem::swap(u, v);
            }
        }
        sorted.sort_unstable();
        assert_eq!(
            sorted,
            vec![
                (0, 28),
                (1, 2),
                (3, 29),
                (4, 6),
                (5, 7),
                (8, 10),
                (9, 11),
                (12, 14),
                (13, 15),
                (16, 17),
                (18, 19),
                (20, 21),
                (22, 23),
                (24, 26),
                (25, 27),
            ]
        );
        let mut cost = 0i64;
        for &(u, v) in &sorted {
            let uv = edges
                .iter()
                .find_map(|&(a, b, w)| {
                    ((a == u && b == v) || (a == v && b == u)).then_some(w as i64)
                })
                .expect("matching edge must exist");
            cost += uv;
        }
        assert_eq!(cost, -322);
    }

    #[test]
    fn test_ground_truth_case_24595_with_budget() {
        let edges = case_24595_edges();
        let g = build_graph(18, &edges);
        let state = BlossomVState::new(&g);
        let pairs = state.solve_with_test_budget(500, 4000).expect("case #24595 should solve");
        let mut sorted = pairs;
        for (u, v) in &mut sorted {
            if *u > *v {
                core::mem::swap(u, v);
            }
        }
        sorted.sort_unstable();
        assert_eq!(
            sorted,
            vec![(0, 9), (1, 10), (2, 8), (3, 11), (4, 14), (5, 17), (6, 15), (7, 16), (12, 13),]
        );
        let mut cost = 0i64;
        for &(u, v) in &sorted {
            let uv = edges
                .iter()
                .find_map(|&(a, b, w)| {
                    ((a == u && b == v) || (a == v && b == u)).then_some(w as i64)
                })
                .expect("matching edge must exist");
            cost += uv;
        }
        assert_eq!(cost, -316);
    }

    #[test]
    fn test_ground_truth_case_24595_plain_solve() {
        let edges = case_24595_edges();
        let g = build_graph(18, &edges);
        let pairs =
            BlossomVState::new(&g).solve().expect("case #24595 should solve via plain solve()");
        let mut sorted = pairs;
        for (u, v) in &mut sorted {
            if *u > *v {
                core::mem::swap(u, v);
            }
        }
        sorted.sort_unstable();
        assert_eq!(
            sorted,
            vec![(0, 9), (1, 10), (2, 8), (3, 11), (4, 14), (5, 17), (6, 15), (7, 16), (12, 13),]
        );
        let mut cost = 0i64;
        for &(u, v) in &sorted {
            let uv = edges
                .iter()
                .find_map(|&(a, b, w)| {
                    ((a == u && b == v) || (a == v && b == u)).then_some(w as i64)
                })
                .expect("matching edge must exist");
            cost += uv;
        }
        assert_eq!(cost, -316);
    }

    #[test]
    fn test_ground_truth_case_24595_solve_matches_budget_and_public_path() {
        let edges = case_24595_edges();
        let g = build_graph(18, &edges);

        let mut budget_pairs = BlossomVState::new(&g)
            .solve_with_test_budget(500, 4000)
            .expect("case #24595 should solve with budget");
        let mut direct_pairs =
            BlossomVState::new(&g).solve().expect("case #24595 should solve directly");
        let mut public_pairs = g.blossom_v().expect("case #24595 should solve via public path");

        for (u, v) in &mut budget_pairs {
            if *u > *v {
                core::mem::swap(u, v);
            }
        }
        budget_pairs.sort_unstable();

        for (u, v) in &mut direct_pairs {
            if *u > *v {
                core::mem::swap(u, v);
            }
        }
        direct_pairs.sort_unstable();

        for (u, v) in &mut public_pairs {
            if *u > *v {
                core::mem::swap(u, v);
            }
        }
        public_pairs.sort_unstable();

        assert_eq!(direct_pairs, budget_pairs, "solve() diverged from solve_with_test_budget()");
        assert_eq!(
            public_pairs, budget_pairs,
            "public blossom_v() diverged from internal solve path"
        );
    }

    #[test]
    fn test_ground_truth_case_28832_with_budget() {
        let edges = case_28832_edges();
        let g = build_graph(22, &edges);
        let state = BlossomVState::new(&g);
        let pairs = state.solve_with_test_budget(400, 2000).expect("case #28832 should solve");
        let mut sorted = pairs;
        for (u, v) in &mut sorted {
            if *u > *v {
                core::mem::swap(u, v);
            }
        }
        sorted.sort_unstable();
        assert_eq!(
            sorted,
            vec![
                (0, 20),
                (1, 2),
                (3, 4),
                (5, 7),
                (6, 19),
                (8, 9),
                (10, 11),
                (12, 16),
                (13, 18),
                (14, 21),
                (15, 17),
            ]
        );
        let mut cost = 0i64;
        for &(u, v) in &sorted {
            let uv = edges
                .iter()
                .find_map(|&(a, b, w)| {
                    ((a == u && b == v) || (a == v && b == u)).then_some(w as i64)
                })
                .expect("matching edge must exist");
            cost += uv;
        }
        assert_eq!(cost, -782);
    }

    #[test]
    fn test_ground_truth_case_21222_with_budget() {
        let edges = case_21222_edges();
        let g = build_graph(28, &edges);
        let state = BlossomVState::new(&g);
        let pairs = state.solve_with_test_budget(500, 3000).expect("case #21222 should solve");
        let mut sorted = pairs;
        for (u, v) in &mut sorted {
            if *u > *v {
                core::mem::swap(u, v);
            }
        }
        sorted.sort_unstable();
        assert_eq!(
            sorted,
            vec![
                (0, 11),
                (1, 7),
                (2, 3),
                (4, 5),
                (6, 9),
                (8, 10),
                (12, 19),
                (13, 15),
                (14, 27),
                (16, 17),
                (18, 21),
                (20, 22),
                (23, 26),
                (24, 25),
            ]
        );
        let mut cost = 0i64;
        for &(u, v) in &sorted {
            let uv = edges
                .iter()
                .find_map(|&(a, b, w)| {
                    ((a == u && b == v) || (a == v && b == u)).then_some(w as i64)
                })
                .expect("matching edge must exist");
            cost += uv;
        }
        assert_eq!(cost, -815);
    }

    #[test]
    fn test_case_24943_tree_navigation_invariants_hold_during_solve() {
        let g = build_graph(30, &case_24943_edges());
        let mut state = BlossomVState::new(&g);
        let _ = solve_case_24943_with_tree_checks(&mut state, 400, 2000)
            .expect("case #24943 should satisfy tree navigation invariants");
    }

    #[test]
    fn test_init_global_case_26924_matches_cpp_after() {
        let g = build_graph(
            10,
            &[
                (1, 7, -98),
                (0, 2, 67),
                (5, 9, 71),
                (3, 6, -45),
                (4, 8, 71),
                (0, 6, 1),
                (1, 3, -19),
                (1, 4, 31),
                (7, 8, -7),
                (1, 5, 7),
                (4, 7, 18),
                (2, 8, -74),
                (3, 4, -13),
                (2, 5, 76),
                (2, 6, 73),
                (2, 9, 30),
                (0, 7, 18),
                (1, 2, 78),
                (8, 9, 58),
                (2, 4, -75),
                (2, 7, 69),
                (4, 9, -39),
                (4, 5, -46),
                (7, 9, -3),
            ],
        );
        let mut state = BlossomVState::new(&g);
        state.init_global();

        let snapshot = state.test_state_snapshot();
        assert_eq!(snapshot.y, vec![143, -89, -75, 51, -156, 64, -141, -107, -73, 78]);
        assert_eq!(snapshot.matching, vec![-1, 7, 8, 6, -1, 9, 3, 1, 2, 5]);
        assert_eq!(
            snapshot.flags,
            vec![PLUS, FREE, FREE, FREE, PLUS, FREE, FREE, FREE, FREE, FREE]
        );
        assert_eq!(snapshot.is_outer, vec![true; 10]);
        assert_eq!(snapshot.tree_num, 2);
        assert_eq!(
            edge_slacks_by_endpoints(&state),
            vec![
                ((0, 2), 66),
                ((0, 6), 0),
                ((0, 7), 0),
                ((1, 2), 320),
                ((1, 3), 0),
                ((1, 4), 307),
                ((1, 5), 39),
                ((1, 7), 0),
                ((2, 4), 81),
                ((2, 5), 163),
                ((2, 6), 362),
                ((2, 7), 320),
                ((2, 8), 0),
                ((2, 9), 57),
                ((3, 4), 79),
                ((3, 6), 0),
                ((4, 5), 0),
                ((4, 7), 299),
                ((4, 8), 371),
                ((4, 9), 0),
                ((5, 9), 0),
                ((7, 8), 166),
                ((7, 9), 23),
                ((8, 9), 111),
            ],
        );
    }

    #[test]
    fn test_init_global_case_27373_matches_cpp_after() {
        let g = build_graph(
            12,
            &[
                (0, 1, 5),
                (6, 8, 40),
                (3, 11, -94),
                (4, 5, -88),
                (2, 10, -48),
                (7, 9, 69),
                (1, 3, 98),
                (2, 6, 89),
                (1, 7, 30),
                (9, 10, 8),
                (3, 7, -3),
                (8, 10, -82),
                (6, 9, -26),
                (4, 10, -95),
                (0, 10, -47),
                (2, 3, -46),
                (3, 5, -38),
                (2, 5, -1),
                (1, 5, 82),
                (5, 9, -94),
            ],
        );
        let mut state = BlossomVState::new(&g);
        state.init_global();

        let snapshot = state.test_state_snapshot();
        assert_eq!(snapshot.y, vec![160, -150, 158, -250, -16, -160, 20, 210, 60, -72, -254, 62]);
        assert_eq!(snapshot.matching, vec![1, 0, 10, 11, 5, 4, 8, 9, 6, 7, 2, 3]);
        assert_eq!(snapshot.flags, vec![FREE; 12]);
        assert_eq!(snapshot.is_outer, vec![true; 12]);
        assert_eq!(snapshot.tree_num, 0);
        assert_eq!(
            edge_slacks_by_endpoints(&state),
            vec![
                ((0, 1), 0),
                ((0, 10), 0),
                ((1, 3), 596),
                ((1, 5), 474),
                ((1, 7), 0),
                ((2, 3), 0),
                ((2, 5), 0),
                ((2, 6), 0),
                ((2, 10), 0),
                ((3, 5), 334),
                ((3, 7), 34),
                ((3, 11), 0),
                ((4, 5), 0),
                ((4, 10), 80),
                ((5, 9), 44),
                ((6, 8), 0),
                ((6, 9), 0),
                ((7, 9), 0),
                ((8, 10), 30),
                ((9, 10), 342),
            ],
        );
    }

    #[test]
    fn test_case_26924_matches_cpp_first_generic_grow_before() {
        let g = build_graph(
            10,
            &[
                (1, 7, -98),
                (0, 2, 67),
                (5, 9, 71),
                (3, 6, -45),
                (4, 8, 71),
                (0, 6, 1),
                (1, 3, -19),
                (1, 4, 31),
                (7, 8, -7),
                (1, 5, 7),
                (4, 7, 18),
                (2, 8, -74),
                (3, 4, -13),
                (2, 5, 76),
                (2, 6, 73),
                (2, 9, 30),
                (0, 7, 18),
                (1, 2, 78),
                (8, 9, 58),
                (2, 4, -75),
                (2, 7, 69),
                (4, 9, -39),
                (4, 5, -46),
                (7, 9, -3),
            ],
        );
        let mut state = BlossomVState::new(&g);
        state.init_global();
        state.mark_tree_roots_processed();
        assert!(state.generic_primal_pass_once());
        let steps = state.test_generic_primal_steps();
        assert!(
            steps.len() >= 2,
            "missing compacted first generic pass on case #26924: only {} steps recorded",
            steps.len()
        );
        assert_eq!(steps[0].event, GenericPrimalEvent::Grow { edge: (0, 7), plus: 0, free: 7 });
        assert_eq!(steps[1].event, GenericPrimalEvent::Shrink { edge: (0, 6), left: 6, right: 0 });
    }

    #[test]
    fn test_solve_single_edge_via_greedy() {
        // When greedy matches everything, solve should return immediately
        let g = build_graph(2, &[(0, 1, 42)]);
        let result = BlossomVState::new(&g).solve();
        let pairs = result.expect("should succeed");
        assert_eq!(pairs, vec![(0usize, 1usize)]);
    }

    #[test]
    fn test_edge_list_iteration_circular() {
        // Verify circular list integrity
        let g = build_graph(4, &[(0, 1, 1), (0, 2, 2), (0, 3, 3)]);
        let state = BlossomVState::new(&g);
        let mut neighbors = Vec::new();
        state.for_each_edge(0, |_e, dir, edge| {
            neighbors.push(edge.head[dir] as usize);
        });
        neighbors.sort();
        assert_eq!(neighbors, vec![1, 2, 3]);
    }

    #[test]
    fn test_process_expand_selfloop_relinks_distinct_penultimate_nodes() {
        let g = build_graph(4, &[(0, 1, 1)]);
        let mut state = BlossomVState::new(&g);
        let e_idx = 0u32;

        for node in &mut state.nodes {
            node.first = [NONE; 2];
        }
        state.edges[e_idx as usize].next = [NONE; 2];
        state.edges[e_idx as usize].prev = [NONE; 2];
        state.edges[e_idx as usize].head = [0, 1];
        state.edges[e_idx as usize].slack = 20;

        state.nodes[0].is_outer = false;
        state.nodes[0].blossom_parent = 2;
        state.nodes[0].blossom_eps = 7;

        state.nodes[1].is_outer = false;
        state.nodes[1].blossom_parent = 3;
        state.nodes[1].blossom_eps = 11;

        state.nodes[2].is_outer = true;
        state.nodes[3].is_outer = true;

        state.process_expand_selfloop(e_idx);

        assert_eq!(state.nodes[0].first[1], e_idx);
        assert_eq!(state.nodes[1].first[0], e_idx);
        assert_eq!(state.edges[e_idx as usize].prev[0], e_idx);
        assert_eq!(state.edges[e_idx as usize].next[0], e_idx);
        assert_eq!(state.edges[e_idx as usize].prev[1], e_idx);
        assert_eq!(state.edges[e_idx as usize].next[1], e_idx);
        assert_eq!(state.edges[e_idx as usize].slack, 6);
    }

    #[test]
    fn test_process_expand_selfloop_stashes_edge_on_shared_penultimate_node() {
        let g = build_graph(3, &[(0, 1, 1)]);
        let mut state = BlossomVState::new(&g);
        let e_idx = 0u32;

        for node in &mut state.nodes {
            node.first = [NONE; 2];
        }
        state.edges[e_idx as usize].next = [NONE; 2];
        state.edges[e_idx as usize].prev = [NONE; 2];
        state.edges[e_idx as usize].head = [0, 1];

        state.nodes[0].is_outer = false;
        state.nodes[0].blossom_parent = 1;

        state.nodes[1].is_outer = false;
        state.nodes[1].blossom_parent = 2;
        state.nodes[1].blossom_selfloops = NONE;

        state.nodes[2].is_outer = true;

        state.process_expand_selfloop(e_idx);

        assert_eq!(state.nodes[1].blossom_selfloops, e_idx);
        assert_eq!(state.edges[e_idx as usize].next[0], NONE);
    }

    #[test]
    fn test_process_expand_selfloop_returns_when_edge_head_is_none() {
        let g = build_graph(2, &[(0, 1, 1)]);
        let mut state = BlossomVState::new(&g);
        let e_idx = 0u32;

        state.edges[e_idx as usize].head = [NONE, 1];
        state.nodes[0].blossom_selfloops = 9;
        state.nodes[1].blossom_selfloops = 11;

        state.process_expand_selfloop(e_idx);

        assert_eq!(state.nodes[0].blossom_selfloops, 9);
        assert_eq!(state.nodes[1].blossom_selfloops, 11);
    }

    #[test]
    fn test_process_expand_selfloop_returns_when_penultimate_is_missing() {
        let g = build_graph(2, &[(0, 1, 1)]);
        let mut state = BlossomVState::new(&g);
        let e_idx = 0u32;
        let before_first0 = state.nodes[0].first;
        let before_first1 = state.nodes[1].first;

        state.edges[e_idx as usize].head = [0, 1];
        state.edges[e_idx as usize].slack = 12;

        state.process_expand_selfloop(e_idx);

        assert_eq!(state.edges[e_idx as usize].slack, 12);
        assert_eq!(state.nodes[0].first, before_first0);
        assert_eq!(state.nodes[1].first, before_first1);
    }

    #[test]
    fn test_next_tree_plus_returns_none_when_match_arc_has_no_raw_head() {
        let g = build_graph(3, &[(0, 1, 1)]);
        let mut state = BlossomVState::new(&g);
        let root = 0u32;
        let current = 1u32;

        state.nodes[root as usize].flag = PLUS;
        state.nodes[root as usize].is_outer = true;
        state.nodes[root as usize].is_tree_root = true;

        state.nodes[current as usize].flag = PLUS;
        state.nodes[current as usize].is_outer = true;
        state.nodes[current as usize].tree_root = root;
        state.nodes[current as usize].match_arc = make_arc(0, 0);
        state.nodes[current as usize].tree_sibling_next = NONE;
        state.edges[0].head = [NONE, current];

        assert_eq!(state.next_tree_plus(current, root), None);
    }

    #[test]
    fn test_next_tree_plus_returns_none_when_minus_parent_arc_has_no_outer_head() {
        let g = build_graph(4, &[(0, 1, 1), (1, 2, 1)]);
        let mut state = BlossomVState::new(&g);
        let root = 0u32;
        let current = 1u32;
        let minus = 2u32;

        state.nodes[root as usize].flag = PLUS;
        state.nodes[root as usize].is_outer = true;
        state.nodes[root as usize].is_tree_root = true;

        state.nodes[current as usize].flag = PLUS;
        state.nodes[current as usize].is_outer = true;
        state.nodes[current as usize].tree_root = root;
        state.nodes[current as usize].match_arc = make_arc(0, 0);
        state.nodes[current as usize].tree_sibling_next = NONE;

        state.nodes[minus as usize].flag = MINUS;
        state.nodes[minus as usize].is_outer = true;
        state.nodes[minus as usize].tree_parent_arc = NONE;
        state.edges[0].head = [minus, current];

        assert_eq!(state.next_tree_plus(current, root), None);
    }

    #[test]
    fn test_expand_drains_child_selfloops_before_child_marking() {
        let g = build_graph(6, &[(0, 1, 1)]);
        let mut state = BlossomVState::new(&g);
        let b = 2u32;
        let child = 1u32;
        let left_inner = 0u32;
        let right_inner = 3u32;
        let left_outer = 4u32;
        let right_outer = 5u32;
        let e_idx = 0u32;

        for node in &mut state.nodes {
            node.first = [NONE; 2];
            node.is_outer = true;
            node.blossom_parent = NONE;
            node.blossom_selfloops = NONE;
        }

        state.nodes[b as usize].is_blossom = true;
        state.nodes[child as usize].is_blossom = true;
        state.nodes[child as usize].is_outer = false;
        state.nodes[child as usize].blossom_parent = b;
        state.nodes[child as usize].blossom_selfloops = e_idx;

        state.nodes[left_inner as usize].is_outer = false;
        state.nodes[left_inner as usize].blossom_parent = left_outer;
        state.nodes[right_inner as usize].is_outer = false;
        state.nodes[right_inner as usize].blossom_parent = right_outer;

        state.edges[e_idx as usize].head = [left_inner, right_inner];
        state.edges[e_idx as usize].next = [NONE; 2];
        state.edges[e_idx as usize].prev = [NONE; 2];

        state.expand(b);

        assert_eq!(state.nodes[child as usize].blossom_selfloops, NONE);
        assert_eq!(state.nodes[left_inner as usize].first[1], e_idx);
        assert_eq!(state.nodes[right_inner as usize].first[0], e_idx);
    }

    #[test]
    fn test_expand_forward_branch_breaks_when_next_minus_is_unmatched() {
        let g = build_graph(7, &[(0, 1, 1), (1, 2, 1), (2, 3, 1), (3, 4, 1)]);
        let mut state = BlossomVState::new(&g);
        let tp = 0u32;
        let cur = 1u32;
        let nxt = 2u32;
        let k = 3u32;
        let b = 4u32;
        let child_plus = 5u32;
        let grandparent = 6u32;
        let b_match = make_arc(0, 0);
        let b_tp = make_arc(1, 0);
        let tp_match = make_arc(2, 0);
        let cur_sib = make_arc(3, 0);

        for node in &mut state.nodes {
            *node = Node::new_vertex();
        }
        for edge in &mut state.edges {
            edge.head = [0, 1];
            edge.head0 = [0, 1];
            edge.next = [NONE; 2];
            edge.prev = [NONE; 2];
            edge.slack = 0;
        }

        state.nodes[b as usize].is_blossom = true;
        state.nodes[b as usize].flag = MINUS;
        state.nodes[b as usize].tree_root = grandparent;
        state.nodes[b as usize].match_arc = b_match;
        state.nodes[b as usize].tree_parent_arc = b_tp;

        state.nodes[grandparent as usize].tree_eps = 5;
        state.nodes[grandparent as usize].first_tree_child = child_plus;

        state.nodes[child_plus as usize].tree_sibling_prev = child_plus;
        state.nodes[child_plus as usize].tree_sibling_next = NONE;

        for &child in &[tp, cur, nxt, k] {
            state.nodes[child as usize].is_outer = false;
            state.nodes[child as usize].blossom_parent = b;
        }

        state.nodes[tp as usize].match_arc = tp_match;
        state.nodes[tp as usize].blossom_sibling_arc = tp_match;
        state.nodes[cur as usize].blossom_sibling_arc = cur_sib;
        state.nodes[k as usize].tree_parent_arc = b_tp;

        state.edges[0].head = [child_plus, b];
        state.edges[0].head0 = [k, child_plus];
        state.edges[1].head = [grandparent, b];
        state.edges[1].head0 = [tp, grandparent];
        state.edges[2].head = [cur, tp];
        state.edges[2].head0 = [tp, cur];
        state.edges[3].head = [nxt, cur];
        state.edges[3].head0 = [cur, nxt];

        state.expand(b);

        assert_eq!(state.nodes[cur as usize].flag, PLUS);
        assert_eq!(state.nodes[nxt as usize].flag, MINUS);
        assert_eq!(state.nodes[cur as usize].tree_root, grandparent);
        assert_eq!(state.nodes[nxt as usize].tree_parent_arc, arc_rev(cur_sib));
        assert_eq!(state.nodes[grandparent as usize].first_tree_child, cur);
        assert_eq!(state.nodes[child_plus as usize].tree_sibling_prev, child_plus);
        assert_eq!(state.nodes[child_plus as usize].tree_sibling_next, NONE);
    }

    #[test]
    fn test_expand_swaps_blossom_minus_y_with_match_edge_slack() {
        let g = build_graph(6, &[(0, 1, 1), (0, 1, 1)]);
        let mut state = BlossomVState::new(&g);
        let k = 2u32;
        let b = 4u32;
        let child_plus = 3u32;
        let root = 5u32;
        let b_match = make_arc(0, 0);

        for node in &mut state.nodes {
            *node = Node::new_vertex();
        }
        for edge in &mut state.edges {
            edge.head = [0, 1];
            edge.head0 = [0, 1];
            edge.next = [NONE; 2];
            edge.prev = [NONE; 2];
            edge.slack = 0;
        }

        state.nodes[b as usize].is_blossom = true;
        state.nodes[b as usize].flag = MINUS;
        state.nodes[b as usize].tree_root = root;
        state.nodes[b as usize].match_arc = b_match;

        state.nodes[root as usize].tree_eps = 4;

        state.nodes[k as usize].is_blossom = true;
        state.nodes[k as usize].is_outer = false;
        state.nodes[k as usize].blossom_parent = b;
        state.nodes[k as usize].y = 17;

        state.edges[0].head = [child_plus, b];
        state.edges[0].head0 = [k, child_plus];
        state.edges[0].slack = 9;

        state.expand(b);

        assert_eq!(state.edges[0].slack, 17);
        assert_eq!(state.nodes[k as usize].y, 9);
        assert!(state.nodes[k as usize].is_processed);
    }

    #[test]
    fn test_expand_raw_plus_edges_update_free_and_plus_neighbors() {
        let g = build_graph(7, &[(0, 1, 1), (1, 2, 1), (2, 3, 1), (3, 4, 1)]);
        let mut state = BlossomVState::new(&g);
        let free = 0u32;
        let plus = 1u32;
        let k = 2u32;
        let other_plus = 3u32;
        let b = 4u32;
        let child_plus = 5u32;
        let root = 6u32;
        let b_match = make_arc(0, 0);
        let k_parent = make_arc(1, 0);

        for node in &mut state.nodes {
            *node = Node::new_vertex();
        }
        for edge in &mut state.edges {
            edge.head = [0, 1];
            edge.head0 = [0, 1];
            edge.next = [NONE; 2];
            edge.prev = [NONE; 2];
            edge.slack = 0;
        }

        state.nodes[b as usize].is_blossom = true;
        state.nodes[b as usize].flag = MINUS;
        state.nodes[b as usize].tree_root = root;
        state.nodes[b as usize].match_arc = b_match;

        state.nodes[root as usize].tree_eps = 5;

        state.nodes[k as usize].is_outer = false;
        state.nodes[k as usize].blossom_parent = b;
        state.nodes[k as usize].tree_parent_arc = k_parent;

        state.nodes[plus as usize].match_arc = NONE;
        state.nodes[other_plus as usize].flag = PLUS;

        state.edges[0].head = [child_plus, b];
        state.edges[0].head0 = [k, child_plus];
        state.edges[1].head = [plus, k];
        state.edges[1].head0 = [plus, k];
        state.edges[1].slack = 3;
        state.edges[2].head = [plus, free];
        state.edges[2].head0 = [plus, free];
        state.edges[2].slack = 7;
        state.edges[3].head = [plus, other_plus];
        state.edges[3].head0 = [plus, other_plus];
        state.edges[3].slack = 11;

        state.expand(b);

        assert_eq!(state.edges[2].slack, 12);
        assert_eq!(state.edges[3].slack, 21);
        assert!(state.nodes[k as usize].is_processed);
        assert!(state.nodes[plus as usize].is_processed);
    }

    #[test]
    fn test_expand_k_search_breaks_when_match_tail_has_no_child_parent() {
        let g = build_graph(7, &[(0, 1, 1)]);
        let mut state = BlossomVState::new(&g);
        let child = 0u32;
        let b = 4u32;
        let child_plus = 5u32;
        let stray = 6u32;
        let b_match = make_arc(0, 0);

        for node in &mut state.nodes {
            *node = Node::new_vertex();
        }

        state.nodes[b as usize].is_blossom = true;
        state.nodes[b as usize].flag = MINUS;
        state.nodes[b as usize].match_arc = b_match;

        state.nodes[child as usize].is_outer = false;
        state.nodes[child as usize].blossom_parent = b;

        state.edges[0].head = [child_plus, b];
        state.edges[0].head0 = [stray, child_plus];

        state.expand(b);

        assert_eq!(state.nodes[stray as usize].match_arc, b_match);
        assert!(state.nodes[child as usize].is_outer);
    }

    #[test]
    fn test_expand_tp_search_breaks_when_tree_parent_tail_has_no_child_parent() {
        let g = build_graph(7, &[(0, 1, 1)]);
        let mut state = BlossomVState::new(&g);
        let child = 0u32;
        let b = 4u32;
        let root = 5u32;
        let stray = 6u32;
        let b_tp = make_arc(0, 0);

        for node in &mut state.nodes {
            *node = Node::new_vertex();
        }

        state.nodes[b as usize].is_blossom = true;
        state.nodes[b as usize].flag = MINUS;
        state.nodes[b as usize].tree_root = root;
        state.nodes[b as usize].tree_parent_arc = b_tp;
        state.nodes[root as usize].tree_eps = 7;

        state.nodes[child as usize].is_outer = false;
        state.nodes[child as usize].blossom_parent = b;

        state.edges[0].head = [root, b];
        state.edges[0].head0 = [stray, root];

        state.expand(b);

        assert_eq!(state.nodes[stray as usize].flag, MINUS);
        assert_eq!(state.nodes[stray as usize].tree_root, root);
        assert_eq!(state.nodes[stray as usize].tree_parent_arc, b_tp);
        assert_eq!(state.nodes[stray as usize].y, 7);
    }

    #[test]
    fn test_expand_backward_branch_breaks_when_next_plus_is_unmatched_and_relinks_prev_sibling() {
        let g = build_graph(8, &[(0, 1, 1), (1, 2, 1), (2, 3, 1), (3, 4, 1), (4, 5, 1)]);
        let mut state = BlossomVState::new(&g);
        let tp = 0u32;
        let plus_nxt = 1u32;
        let k = 3u32;
        let b = 4u32;
        let child_plus = 5u32;
        let grandparent = 6u32;
        let cp_prev = 7u32;
        let b_match = make_arc(0, 0);
        let b_tp = make_arc(1, 0);
        let tp_match = make_arc(2, 0);
        let tp_sib = make_arc(3, 0);
        let k_sib = make_arc(4, 0);

        for node in &mut state.nodes {
            *node = Node::new_vertex();
        }

        state.nodes[b as usize].is_blossom = true;
        state.nodes[b as usize].flag = MINUS;
        state.nodes[b as usize].tree_root = grandparent;
        state.nodes[b as usize].match_arc = b_match;
        state.nodes[b as usize].tree_parent_arc = b_tp;

        state.nodes[grandparent as usize].tree_eps = 4;
        state.nodes[grandparent as usize].first_tree_child = cp_prev;

        state.nodes[cp_prev as usize].tree_sibling_next = child_plus;
        state.nodes[child_plus as usize].tree_sibling_prev = cp_prev;
        state.nodes[child_plus as usize].tree_sibling_next = NONE;

        for &child in &[tp, plus_nxt, k] {
            state.nodes[child as usize].is_outer = false;
            state.nodes[child as usize].blossom_parent = b;
        }

        state.nodes[tp as usize].match_arc = tp_match;
        state.nodes[tp as usize].blossom_sibling_arc = tp_sib;
        state.nodes[k as usize].blossom_sibling_arc = k_sib;

        state.edges[0].head = [child_plus, b];
        state.edges[0].head0 = [k, child_plus];
        state.edges[1].head = [grandparent, b];
        state.edges[1].head0 = [tp, grandparent];
        state.edges[2].head = [plus_nxt, tp];
        state.edges[2].head0 = [tp, plus_nxt];
        state.edges[3].head = [tp, plus_nxt];
        state.edges[3].head0 = [tp, plus_nxt];
        state.edges[4].head = [plus_nxt, k];
        state.edges[4].head0 = [k, plus_nxt];

        state.expand(b);

        assert_eq!(state.nodes[k as usize].flag, MINUS);
        assert_eq!(state.nodes[plus_nxt as usize].flag, PLUS);
        assert_eq!(state.nodes[plus_nxt as usize].first_tree_child, child_plus);
        assert_eq!(state.nodes[cp_prev as usize].tree_sibling_next, plus_nxt);
        assert_eq!(state.nodes[child_plus as usize].tree_sibling_prev, child_plus);
        assert_eq!(state.nodes[child_plus as usize].tree_sibling_next, NONE);
    }

    #[test]
    fn test_expand_k_search_climbs_nested_parent_and_breaks_when_cj_sib_is_missing() {
        let g = build_graph(9, &[(0, 1, 1), (1, 2, 1), (2, 3, 1)]);
        let mut state = BlossomVState::new(&g);
        let inner = 0u32;
        let k = 1u32;
        let ci = 2u32;
        let cj = 3u32;
        let b = 6u32;
        let child_plus = 7u32;
        let b_match = make_arc(0, 0);
        let k_sib = make_arc(1, 0);
        let ci_sib = make_arc(2, 0);

        for node in &mut state.nodes {
            *node = Node::new_vertex();
        }

        state.nodes[b as usize].is_blossom = true;
        state.nodes[b as usize].flag = MINUS;
        state.nodes[b as usize].match_arc = b_match;

        state.nodes[inner as usize].is_outer = false;
        state.nodes[inner as usize].blossom_parent = k;
        for &child in &[k, ci, cj] {
            state.nodes[child as usize].is_outer = false;
            state.nodes[child as usize].blossom_parent = b;
        }

        state.nodes[k as usize].blossom_sibling_arc = k_sib;
        state.nodes[ci as usize].blossom_sibling_arc = ci_sib;

        state.edges[0].head = [child_plus, b];
        state.edges[0].head0 = [inner, child_plus];
        state.edges[1].head = [ci, k];
        state.edges[1].head0 = [k, ci];
        state.edges[2].head = [cj, ci];
        state.edges[2].head0 = [ci, cj];

        state.expand(b);

        assert_eq!(state.nodes[k as usize].match_arc, b_match);
        assert_eq!(state.nodes[ci as usize].match_arc, ci_sib);
        assert_eq!(state.nodes[cj as usize].match_arc, arc_rev(ci_sib));
        assert!(state.nodes[k as usize].is_outer);
    }

    #[test]
    fn test_expand_tp_search_climbs_nested_parent_and_builds_multi_step_forward_chain() {
        let g =
            build_graph(12, &[(0, 1, 1), (1, 2, 1), (2, 3, 1), (3, 4, 1), (4, 5, 1), (5, 6, 1)]);
        let mut state = BlossomVState::new(&g);
        let tp_inner = 0u32;
        let tp = 1u32;
        let plus1 = 2u32;
        let minus1 = 3u32;
        let plus2 = 4u32;
        let k = 5u32;
        let child_plus = 6u32;
        let root = 7u32;
        let b = 8u32;
        let b_match = make_arc(0, 0);
        let b_tp = make_arc(1, 0);
        let tp_match = make_arc(2, 0);
        let plus1_sib = make_arc(3, 0);
        let minus1_match = make_arc(4, 0);
        let plus2_sib = make_arc(5, 0);

        for node in &mut state.nodes {
            *node = Node::new_vertex();
        }

        state.nodes[b as usize].is_blossom = true;
        state.nodes[b as usize].flag = MINUS;
        state.nodes[b as usize].tree_root = root;
        state.nodes[b as usize].match_arc = b_match;
        state.nodes[b as usize].tree_parent_arc = b_tp;
        state.nodes[root as usize].tree_eps = 4;
        state.nodes[root as usize].first_tree_child = child_plus;
        state.nodes[child_plus as usize].tree_sibling_prev = child_plus;
        state.nodes[child_plus as usize].tree_sibling_next = NONE;

        state.nodes[tp_inner as usize].is_outer = false;
        state.nodes[tp_inner as usize].blossom_parent = tp;
        for &child in &[tp, plus1, minus1, plus2, k] {
            state.nodes[child as usize].is_outer = false;
            state.nodes[child as usize].blossom_parent = b;
        }

        state.nodes[tp as usize].match_arc = tp_match;
        state.nodes[tp as usize].blossom_sibling_arc = tp_match;
        state.nodes[plus1 as usize].blossom_sibling_arc = plus1_sib;
        state.nodes[minus1 as usize].match_arc = minus1_match;
        state.nodes[plus2 as usize].blossom_sibling_arc = plus2_sib;
        state.nodes[k as usize].tree_parent_arc = b_tp;

        state.edges[0].head = [child_plus, b];
        state.edges[0].head0 = [k, child_plus];
        state.edges[1].head = [root, b];
        state.edges[1].head0 = [tp_inner, root];
        state.edges[2].head = [plus1, tp];
        state.edges[2].head0 = [tp, plus1];
        state.edges[3].head = [minus1, plus1];
        state.edges[3].head0 = [plus1, minus1];
        state.edges[4].head = [plus2, minus1];
        state.edges[4].head0 = [minus1, plus2];
        state.edges[5].head = [k, plus2];
        state.edges[5].head0 = [plus2, k];

        state.expand(b);

        assert_eq!(state.nodes[tp as usize].flag, MINUS);
        assert_eq!(state.nodes[plus1 as usize].flag, PLUS);
        assert_eq!(state.nodes[plus2 as usize].flag, PLUS);
        assert_eq!(state.nodes[plus1 as usize].first_tree_child, plus2);
        assert_eq!(state.nodes[plus2 as usize].first_tree_child, child_plus);
        assert_eq!(state.nodes[root as usize].first_tree_child, plus1);
    }

    #[test]
    fn test_expand_late_lazy_dual_pass_skips_detached_plus_neighbors() {
        let g = build_graph(8, &[(0, 1, 1), (1, 2, 1), (2, 3, 1), (3, 4, 1), (4, 5, 1), (5, 6, 1)]);
        let mut state = BlossomVState::new(&g);
        let free = 0u32;
        let plus = 1u32;
        let k = 2u32;
        let other_plus = 3u32;
        let b = 4u32;
        let child_plus = 5u32;
        let root = 6u32;
        let b_match = make_arc(0, 0);
        let k_parent = make_arc(1, 0);
        let listed_detached = 4u32;
        let raw_detached = 5u32;

        for node in &mut state.nodes {
            *node = Node::new_vertex();
        }
        for edge in &mut state.edges {
            edge.head = [0, 1];
            edge.head0 = [0, 1];
            edge.next = [NONE; 2];
            edge.prev = [NONE; 2];
            edge.slack = 0;
        }

        state.nodes[b as usize].is_blossom = true;
        state.nodes[b as usize].flag = MINUS;
        state.nodes[b as usize].tree_root = root;
        state.nodes[b as usize].match_arc = b_match;
        state.nodes[root as usize].tree_eps = 5;

        state.nodes[k as usize].is_outer = false;
        state.nodes[k as usize].blossom_parent = b;
        state.nodes[k as usize].tree_parent_arc = k_parent;
        state.nodes[plus as usize].match_arc = NONE;
        state.nodes[other_plus as usize].flag = PLUS;

        state.edges[0].head = [child_plus, b];
        state.edges[0].head0 = [k, child_plus];
        state.edges[1].head = [plus, k];
        state.edges[1].head0 = [plus, k];
        state.edges[1].slack = 3;
        state.edges[2].head = [plus, free];
        state.edges[2].head0 = [plus, free];
        state.edges[2].slack = 7;
        state.edges[3].head = [plus, other_plus];
        state.edges[3].head0 = [plus, other_plus];
        state.edges[3].slack = 11;

        edge_list_add(&mut state.nodes, &mut state.edges, plus, listed_detached, 0);
        let listed_dir = state
            .incident_edges(plus)
            .into_iter()
            .find(|(e_idx, _)| *e_idx == listed_detached)
            .map(|(_, dir)| dir)
            .expect("listed detached edge should be reachable from plus");
        state.edges[listed_detached as usize].head = [plus, plus];
        state.edges[listed_detached as usize].head[listed_dir] = NONE;
        state.edges[listed_detached as usize].head[1 - listed_dir] = plus;
        state.edges[listed_detached as usize].head0 = state.edges[listed_detached as usize].head;
        state.edges[listed_detached as usize].slack = 19;

        state.edges[raw_detached as usize].head = [plus, NONE];
        state.edges[raw_detached as usize].head0 = [plus, NONE];
        state.edges[raw_detached as usize].slack = 23;

        state.expand(b);

        assert_eq!(state.edges[listed_detached as usize].slack, 19);
        assert_eq!(state.edges[raw_detached as usize].slack, 23);
        assert!(state.nodes[k as usize].is_processed);
        assert!(state.nodes[plus as usize].is_processed);
    }

    #[test]
    fn test_shrink_restores_partner_blossom_match_slack_and_moves_match_edge_to_new_blossom() {
        let g = build_graph(
            9,
            &[
                (0, 1, 1),
                (0, 2, 1),
                (0, 3, 1),
                (0, 4, 1),
                (0, 5, 1),
                (0, 6, 1),
                (0, 7, 1),
                (0, 8, 1),
                (1, 8, 1),
            ],
        );
        let mut state = BlossomVState::new(&g);

        for node in &mut state.nodes {
            *node = Node::new_vertex();
        }
        for edge in &mut state.edges {
            edge.head = [0, 1];
            edge.head0 = [0, 1];
            edge.next = [NONE; 2];
            edge.prev = [NONE; 2];
            edge.slack = 0;
        }

        let root = 5u32;
        let lca = 0u32;
        let ep0 = 1u32;
        let ep1 = 2u32;
        let minus0 = 3u32;
        let minus1 = 4u32;
        let partner = 6u32;
        let shrink_edge = 0u32;
        let b_match_edge = 3u32;
        let new_blossom = state.node_num as u32;

        for &plus in &[root, lca, ep0, ep1] {
            state.nodes[plus as usize].flag = PLUS;
            state.nodes[plus as usize].is_outer = true;
            state.nodes[plus as usize].tree_root = root;
        }
        state.nodes[root as usize].is_tree_root = true;
        state.nodes[root as usize].tree_eps = 7;
        state.nodes[root as usize].first_tree_child = lca;

        for &minus in &[minus0, minus1, partner] {
            state.nodes[minus as usize].flag = MINUS;
            state.nodes[minus as usize].is_outer = true;
            state.nodes[minus as usize].tree_root = root;
        }
        state.nodes[partner as usize].is_blossom = true;
        state.nodes[partner as usize].y = 21;

        state.nodes[lca as usize].match_arc = make_arc(b_match_edge, 0);
        state.nodes[ep0 as usize].match_arc = make_arc(1, 1);
        state.nodes[ep1 as usize].match_arc = make_arc(5, 1);
        state.nodes[minus0 as usize].tree_parent_arc = make_arc(2, 1);
        state.nodes[minus1 as usize].tree_parent_arc = make_arc(6, 1);
        state.nodes[partner as usize].tree_parent_arc = make_arc(4, 1);

        edge_list_add(&mut state.nodes, &mut state.edges, ep0, 1, 1);
        edge_list_add(&mut state.nodes, &mut state.edges, minus0, 1, 0);
        edge_list_add(&mut state.nodes, &mut state.edges, lca, 2, 0);
        edge_list_add(&mut state.nodes, &mut state.edges, minus0, 2, 1);
        edge_list_add(&mut state.nodes, &mut state.edges, lca, b_match_edge, 0);
        edge_list_add(&mut state.nodes, &mut state.edges, partner, b_match_edge, 1);
        edge_list_add(&mut state.nodes, &mut state.edges, root, 4, 0);
        edge_list_add(&mut state.nodes, &mut state.edges, partner, 4, 1);
        edge_list_add(&mut state.nodes, &mut state.edges, ep1, 5, 1);
        edge_list_add(&mut state.nodes, &mut state.edges, minus1, 5, 0);
        edge_list_add(&mut state.nodes, &mut state.edges, lca, 6, 0);
        edge_list_add(&mut state.nodes, &mut state.edges, minus1, 6, 1);

        state.edges[shrink_edge as usize].head = [ep0, ep1];
        state.edges[shrink_edge as usize].head0 = [ep0, ep1];
        state.edges[1].head0 = [ep0, minus0];
        state.edges[2].head0 = [minus0, lca];
        state.edges[b_match_edge as usize].head0 = [lca, partner];
        state.edges[4].head0 = [partner, root];
        state.edges[5].head0 = [ep1, minus1];
        state.edges[6].head0 = [minus1, lca];
        state.edges[b_match_edge as usize].slack = 13;

        state.shrink(shrink_edge, ep0, ep1);

        assert_eq!(state.edges[b_match_edge as usize].slack, 13);
        assert_eq!(state.nodes[partner as usize].y, 21);
        assert_eq!(state.edges[b_match_edge as usize].head[1], new_blossom);
        assert_eq!(state.nodes[new_blossom as usize].match_arc, make_arc(b_match_edge, 0));
    }

    #[test]
    fn test_shrink_second_pass_relinks_inner_edge_and_promotes_boundary_edge() {
        let g = build_graph(
            10,
            &[
                (0, 1, 1),
                (0, 2, 1),
                (0, 3, 1),
                (0, 4, 1),
                (0, 5, 1),
                (0, 6, 1),
                (0, 7, 1),
                (0, 8, 1),
                (0, 9, 1),
            ],
        );
        let mut state = BlossomVState::new(&g);

        for node in &mut state.nodes {
            *node = Node::new_vertex();
        }
        for edge in &mut state.edges {
            edge.head = [0, 1];
            edge.head0 = [0, 1];
            edge.next = [NONE; 2];
            edge.prev = [NONE; 2];
            edge.slack = 0;
        }

        let lca = 0u32;
        let ep0 = 1u32;
        let ep1 = 2u32;
        let minus0 = 3u32;
        let minus1 = 4u32;
        let free = 7u32;
        let hidden = 8u32;
        let shrink_edge = 0u32;
        let boundary_edge = 7u32;
        let hidden_edge = 8u32;
        let new_blossom = state.node_num as u32;

        for &plus in &[lca, ep0, ep1] {
            state.nodes[plus as usize].flag = PLUS;
            state.nodes[plus as usize].is_outer = true;
            state.nodes[plus as usize].tree_root = lca;
        }
        state.nodes[lca as usize].is_tree_root = true;
        state.nodes[lca as usize].tree_eps = 7;

        for &minus in &[minus0, minus1] {
            state.nodes[minus as usize].flag = MINUS;
            state.nodes[minus as usize].is_outer = true;
            state.nodes[minus as usize].tree_root = lca;
        }

        state.nodes[hidden as usize].is_outer = false;
        state.nodes[hidden as usize].blossom_parent = minus1;
        state.nodes[hidden as usize].flag = FREE;

        state.nodes[ep0 as usize].match_arc = make_arc(1, 1);
        state.nodes[ep1 as usize].match_arc = make_arc(5, 1);
        state.nodes[minus0 as usize].tree_parent_arc = make_arc(2, 1);
        state.nodes[minus1 as usize].tree_parent_arc = make_arc(6, 1);

        edge_list_add(&mut state.nodes, &mut state.edges, ep0, 1, 1);
        edge_list_add(&mut state.nodes, &mut state.edges, minus0, 1, 0);
        edge_list_add(&mut state.nodes, &mut state.edges, lca, 2, 0);
        edge_list_add(&mut state.nodes, &mut state.edges, minus0, 2, 1);
        edge_list_add(&mut state.nodes, &mut state.edges, ep1, 5, 1);
        edge_list_add(&mut state.nodes, &mut state.edges, minus1, 5, 0);
        edge_list_add(&mut state.nodes, &mut state.edges, lca, 6, 0);
        edge_list_add(&mut state.nodes, &mut state.edges, minus1, 6, 1);

        edge_list_add(&mut state.nodes, &mut state.edges, free, boundary_edge, 1);
        edge_list_add(&mut state.nodes, &mut state.edges, minus0, boundary_edge, 0);
        edge_list_add(&mut state.nodes, &mut state.edges, hidden, hidden_edge, 1);
        edge_list_add(&mut state.nodes, &mut state.edges, minus0, hidden_edge, 0);

        state.edges[shrink_edge as usize].head = [ep0, ep1];
        state.edges[shrink_edge as usize].head0 = [ep0, ep1];
        state.edges[1].head0 = [ep0, minus0];
        state.edges[2].head0 = [minus0, lca];
        state.edges[5].head0 = [ep1, minus1];
        state.edges[6].head0 = [minus1, lca];
        state.edges[boundary_edge as usize].head0 = [free, minus0];
        state.edges[boundary_edge as usize].slack = 5;
        state.edges[hidden_edge as usize].head0 = [hidden, minus0];
        state.edges[hidden_edge as usize].slack = 11;

        state.shrink(shrink_edge, ep0, ep1);

        assert_eq!(state.edges[boundary_edge as usize].head[1], new_blossom);
        assert_eq!(
            state.edge_queue_owner(boundary_edge),
            GenericQueueState::Pq0 { root: new_blossom }
        );
        assert_eq!(state.edges[boundary_edge as usize].slack, 19);

        assert_eq!(state.edges[hidden_edge as usize].head[0], minus1);
        assert_eq!(state.edges[hidden_edge as usize].slack, 25);
    }

    #[test]
    fn test_rebuild_outer_blossom_queue_membership_preserves_existing_pq0_stamp_order() {
        let g = build_graph(3, &[(0, 2, 1), (1, 2, 1)]);
        let mut state = BlossomVState::new(&g);

        for node in &mut state.nodes {
            *node = Node::new_vertex();
            node.is_outer = true;
        }

        let blossom = 2u32;
        state.nodes[blossom as usize].is_blossom = true;
        state.nodes[blossom as usize].flag = PLUS;
        state.nodes[blossom as usize].is_tree_root = true;
        state.nodes[blossom as usize].tree_root = blossom;
        state.nodes[blossom as usize].tree_eps = 5;

        state.nodes[0].flag = FREE;
        state.nodes[1].flag = FREE;

        state.edges[0].head = [0, blossom];
        state.edges[0].head0 = [0, blossom];
        state.edges[0].slack = 7;
        state.edges[1].head = [1, blossom];
        state.edges[1].head0 = [1, blossom];
        state.edges[1].slack = 7;

        state.set_generic_pq0(0, blossom);
        state.set_generic_pq0(1, blossom);

        state.set_edge_queue_stamp(0, 11);
        state.set_edge_queue_stamp(1, 10);

        assert_eq!(state.scheduler_tree_best_pq0_edge(blossom), Some(0));

        state.rebuild_generic_queue_membership_for_outer_blossom(blossom);

        assert_eq!(state.edge_queue_stamp(0), 11);
        assert_eq!(state.edge_queue_stamp(1), 10);
        assert_eq!(state.scheduler_tree_best_pq0_edge(blossom), Some(0));
    }

    #[test]
    fn test_shrink_branch_switch_breaks_when_reverse_shrink_edge_returns_to_lca() {
        let g = build_graph(4, &[(0, 1, 1), (0, 2, 1), (0, 3, 1)]);
        let mut state = BlossomVState::new(&g);
        let root = 0u32;
        let endpoint0 = 1u32;
        let minus0 = 2u32;
        let shrink_edge = 0u32;
        let match_edge = 1u32;
        let parent_edge = 2u32;
        let new_blossom = state.node_num as u32;

        for node in &mut state.nodes {
            *node = Node::new_vertex();
        }
        for edge in &mut state.edges {
            edge.head = [0, 1];
            edge.head0 = [0, 1];
            edge.next = [NONE; 2];
            edge.prev = [NONE; 2];
            edge.slack = 0;
        }

        state.nodes[root as usize].flag = PLUS;
        state.nodes[root as usize].is_outer = true;
        state.nodes[root as usize].is_tree_root = true;
        state.nodes[root as usize].tree_root = root;
        state.nodes[root as usize].tree_eps = 3;
        state.nodes[root as usize].first_tree_child = endpoint0;

        state.nodes[endpoint0 as usize].flag = PLUS;
        state.nodes[endpoint0 as usize].is_outer = true;
        state.nodes[endpoint0 as usize].tree_root = root;
        state.nodes[endpoint0 as usize].match_arc = make_arc(match_edge, 0);
        state.nodes[endpoint0 as usize].tree_sibling_prev = endpoint0;
        state.nodes[endpoint0 as usize].tree_sibling_next = NONE;

        state.nodes[minus0 as usize].flag = MINUS;
        state.nodes[minus0 as usize].is_outer = true;
        state.nodes[minus0 as usize].tree_root = root;
        state.nodes[minus0 as usize].tree_parent_arc = make_arc(parent_edge, 0);

        state.edges[shrink_edge as usize].head = [endpoint0, root];
        state.edges[shrink_edge as usize].head0 = [endpoint0, root];
        state.edges[match_edge as usize].head = [minus0, endpoint0];
        state.edges[match_edge as usize].head0 = [endpoint0, minus0];
        state.edges[parent_edge as usize].head = [root, minus0];
        state.edges[parent_edge as usize].head0 = [minus0, root];

        state.shrink(shrink_edge, endpoint0, root);

        assert!(state.nodes[new_blossom as usize].is_tree_root);
        assert_eq!(state.nodes[new_blossom as usize].tree_root, new_blossom);
        assert_eq!(state.nodes[root as usize].blossom_parent, new_blossom);
        assert_eq!(state.nodes[endpoint0 as usize].blossom_parent, new_blossom);
    }

    #[test]
    fn test_shrink_branch_one_breaks_when_next_arc_returns_to_lca() {
        let g = build_graph(4, &[(0, 1, 1), (0, 2, 1), (0, 3, 1)]);
        let mut state = BlossomVState::new(&g);
        let root = 0u32;
        let endpoint1 = 1u32;
        let minus1 = 2u32;
        let shrink_edge = 0u32;
        let match_edge = 1u32;
        let parent_edge = 2u32;
        let new_blossom = state.node_num as u32;

        for node in &mut state.nodes {
            *node = Node::new_vertex();
        }
        for edge in &mut state.edges {
            edge.head = [0, 1];
            edge.head0 = [0, 1];
            edge.next = [NONE; 2];
            edge.prev = [NONE; 2];
            edge.slack = 0;
        }

        state.nodes[root as usize].flag = PLUS;
        state.nodes[root as usize].is_outer = true;
        state.nodes[root as usize].is_tree_root = true;
        state.nodes[root as usize].tree_root = root;
        state.nodes[root as usize].tree_eps = 4;
        state.nodes[root as usize].first_tree_child = endpoint1;

        state.nodes[endpoint1 as usize].flag = PLUS;
        state.nodes[endpoint1 as usize].is_outer = true;
        state.nodes[endpoint1 as usize].tree_root = root;
        state.nodes[endpoint1 as usize].match_arc = make_arc(match_edge, 0);
        state.nodes[endpoint1 as usize].tree_sibling_prev = endpoint1;
        state.nodes[endpoint1 as usize].tree_sibling_next = NONE;

        state.nodes[minus1 as usize].flag = MINUS;
        state.nodes[minus1 as usize].is_outer = true;
        state.nodes[minus1 as usize].tree_root = root;
        state.nodes[minus1 as usize].tree_parent_arc = make_arc(parent_edge, 0);

        state.edges[shrink_edge as usize].head = [root, endpoint1];
        state.edges[shrink_edge as usize].head0 = [root, endpoint1];
        state.edges[match_edge as usize].head = [minus1, endpoint1];
        state.edges[match_edge as usize].head0 = [endpoint1, minus1];
        state.edges[parent_edge as usize].head = [root, minus1];
        state.edges[parent_edge as usize].head0 = [minus1, root];

        state.shrink(shrink_edge, root, endpoint1);

        assert!(state.nodes[new_blossom as usize].is_tree_root);
        assert_eq!(state.nodes[new_blossom as usize].tree_root, new_blossom);
        assert_eq!(state.nodes[endpoint1 as usize].blossom_parent, new_blossom);
        assert_eq!(state.nodes[minus1 as usize].blossom_parent, new_blossom);
    }

    #[test]
    fn test_find_global_augment_edge_uses_scheduler_pair_with_adjusted_other_eps() {
        let g = build_graph(4, &[(0, 1, 1)]);
        let mut state = BlossomVState::new(&g);
        let e_idx = 0u32;

        for node in &mut state.nodes {
            node.first = [NONE; 2];
            node.flag = FREE;
            node.is_tree_root = false;
            node.is_processed = false;
            node.tree_root = NONE;
            node.tree_eps = 0;
            node.blossom_parent = NONE;
            node.is_outer = true;
        }

        state.nodes[0].flag = PLUS;
        state.nodes[0].is_tree_root = true;
        state.nodes[0].is_processed = true;
        state.nodes[0].tree_root = 0;
        state.nodes[0].tree_eps = 4;

        state.nodes[1].flag = PLUS;
        state.nodes[1].is_tree_root = true;
        state.nodes[1].is_processed = true;
        state.nodes[1].tree_root = 1;
        state.nodes[1].tree_eps = 2;

        state.root_list_head = 0;
        state.nodes[0].tree_sibling_next = 1;
        state.nodes[1].tree_sibling_next = NONE;

        state.edges[e_idx as usize].head = [0, 1];
        state.edges[e_idx as usize].head0 = [0, 1];
        state.edges[e_idx as usize].slack = 5;
        state.set_generic_pq00(e_idx, 0, 1);

        assert_eq!(state.find_scheduler_global_augment_edge(), Some((e_idx, 0, 1)));
    }

    #[test]
    fn test_prepare_tree_for_augment_requeues_pair_edge_into_other_roots_pq0() {
        let g = build_graph(4, &[(1, 2, 1)]);
        let mut state = BlossomVState::new(&g);
        let e_idx = 0u32;

        for node in &mut state.nodes {
            node.flag = FREE;
            node.is_tree_root = false;
            node.is_processed = false;
            node.tree_root = NONE;
            node.tree_eps = 0;
            node.first_tree_child = NONE;
            node.is_outer = true;
        }

        state.nodes[0].flag = PLUS;
        state.nodes[0].is_tree_root = true;
        state.nodes[0].tree_root = 0;
        state.nodes[0].tree_eps = 4;

        state.nodes[3].flag = PLUS;
        state.nodes[3].is_tree_root = true;
        state.nodes[3].tree_root = 3;

        state.nodes[1].flag = PLUS;
        state.nodes[1].tree_root = 3;
        state.nodes[2].flag = PLUS;
        state.nodes[2].tree_root = 3;

        state.edges[e_idx as usize].head = [1, 2];
        state.set_generic_pq00(e_idx, 0, 3);
        state.generic_pairs[0].pq00.clear();

        state.prepare_tree_for_augment(0, &[0]);

        assert_eq!(state.edge_queue_owner(e_idx), GenericQueueState::Pq0 { root: 3 });
        assert!(state.generic_trees[3].pq0.contains(&e_idx));
        assert!(state.scheduler_trees[3].pq0.contains(&e_idx));
    }

    #[test]
    fn test_prepare_tree_for_augment_uses_scheduler_root_queues_when_shadow_is_stale() {
        let g = build_graph(4, &[(0, 1, 5)]);
        let mut state = BlossomVState::new(&g);
        let e_idx = 0u32;

        for node in &mut state.nodes {
            node.flag = FREE;
            node.is_tree_root = false;
            node.is_processed = false;
            node.tree_root = NONE;
            node.tree_eps = 0;
            node.first_tree_child = NONE;
            node.is_outer = true;
        }

        state.nodes[0].flag = PLUS;
        state.nodes[0].is_tree_root = true;
        state.nodes[0].tree_root = 0;
        state.nodes[0].tree_eps = 4;

        state.nodes[1].flag = FREE;
        state.nodes[1].is_outer = true;
        state.edges[e_idx as usize].head = [0, 1];
        state.edges[e_idx as usize].head0 = [0, 1];
        state.edges[e_idx as usize].slack = 7;

        state.set_generic_pq0(e_idx, 0);
        state.generic_trees[0].pq0.clear();

        state.prepare_tree_for_augment(0, &[0]);

        assert_eq!(state.edge_queue_owner(e_idx), GenericQueueState::None);
        assert!(state.scheduler_trees[0].pq0.is_empty());
        assert!(state.generic_trees[0].pq0.is_empty());
        assert_eq!(state.edges[e_idx as usize].slack, 3);
    }

    #[test]
    fn test_prepare_tree_for_augment_switches_neighbor_current_from_root_to_pair() {
        let g = build_graph(4, &[(1, 2, 1)]);
        let mut state = BlossomVState::new(&g);
        let e_idx = 0u32;

        for node in &mut state.nodes {
            node.flag = FREE;
            node.is_tree_root = false;
            node.is_processed = false;
            node.tree_root = NONE;
            node.tree_eps = 0;
            node.first_tree_child = NONE;
            node.is_outer = true;
        }

        state.nodes[0].flag = PLUS;
        state.nodes[0].is_tree_root = true;
        state.nodes[0].tree_root = 0;
        state.nodes[0].tree_eps = 4;

        state.nodes[3].flag = PLUS;
        state.nodes[3].is_tree_root = true;
        state.nodes[3].tree_root = 3;
        state.nodes[3].tree_eps = 2;

        state.nodes[1].flag = PLUS;
        state.nodes[1].tree_root = 3;
        state.nodes[2].flag = PLUS;
        state.nodes[2].tree_root = 3;

        state.edges[e_idx as usize].head = [1, 2];
        state.set_generic_pq00(e_idx, 0, 3);

        state.scheduler_trees[3].current = SchedulerCurrent::Pair { pair_idx: 17, dir: 1 };

        state.prepare_tree_for_augment(0, &[0]);

        assert_eq!(state.scheduler_trees[3].current, SchedulerCurrent::None);
    }

    #[test]
    fn test_queue_processed_plus_blossom_match_edge_promotes_owned_edge_into_pq_blossoms() {
        let g = build_graph(2, &[(0, 1, 5)]);
        let mut state = BlossomVState::new(&g);
        let plus = 1u32;
        let root = 0u32;
        let blossom = state.nodes.len() as u32;
        let match_edge = 0u32;

        state.nodes[root as usize].flag = PLUS;
        state.nodes[root as usize].is_outer = true;
        state.nodes[root as usize].is_tree_root = true;
        state.nodes[root as usize].is_processed = true;
        state.nodes[root as usize].tree_root = root;

        state.nodes[plus as usize].flag = PLUS;
        state.nodes[plus as usize].is_outer = true;
        state.nodes[plus as usize].is_tree_root = false;
        state.nodes[plus as usize].is_processed = true;
        state.nodes[plus as usize].tree_root = root;
        state.nodes[plus as usize].match_arc = make_arc(match_edge, 0);

        let mut blossom_node = Node::new_vertex();
        blossom_node.is_blossom = true;
        blossom_node.is_outer = true;
        blossom_node.flag = MINUS;
        blossom_node.is_processed = true;
        blossom_node.tree_root = root;
        blossom_node.match_arc = make_arc(match_edge, 1);
        state.nodes.push(blossom_node);

        state.edges[match_edge as usize].head = [blossom, plus];
        state.edges[match_edge as usize].head0 = [blossom, plus];

        state.set_generic_pq0(match_edge, root);

        state.queue_processed_plus_blossom_match_edge(plus);

        assert_eq!(state.edge_queue_owner(match_edge), GenericQueueState::PqBlossoms { root });
        assert!(state.scheduler_trees[root as usize].pq_blossoms.contains(&match_edge));
    }

    #[test]
    fn test_rebuild_outer_blossom_queue_membership_requeues_match_edge_into_pq_blossoms() {
        let g = build_graph(2, &[(0, 1, 5)]);
        let mut state = BlossomVState::new(&g);
        let plus = 1u32;
        let root = 0u32;
        let blossom = state.nodes.len() as u32;
        let match_edge = 0u32;

        state.nodes[root as usize].flag = PLUS;
        state.nodes[root as usize].is_outer = true;
        state.nodes[root as usize].is_tree_root = true;
        state.nodes[root as usize].is_processed = true;
        state.nodes[root as usize].tree_root = root;

        state.nodes[plus as usize].flag = PLUS;
        state.nodes[plus as usize].is_outer = true;
        state.nodes[plus as usize].is_tree_root = false;
        state.nodes[plus as usize].is_processed = true;
        state.nodes[plus as usize].tree_root = root;

        let mut blossom_node = Node::new_vertex();
        blossom_node.is_blossom = true;
        blossom_node.is_outer = true;
        blossom_node.flag = MINUS;
        blossom_node.is_processed = true;
        blossom_node.tree_root = root;
        blossom_node.match_arc = make_arc(match_edge, 1);
        state.nodes.push(blossom_node);

        state.edges[match_edge as usize].head = [blossom, plus];
        state.edges[match_edge as usize].head0 = [blossom, plus];

        state.rebuild_generic_queue_membership_for_outer_blossom(blossom);

        assert_eq!(state.edge_queue_owner(match_edge), GenericQueueState::PqBlossoms { root });
        assert!(state.scheduler_trees[root as usize].pq_blossoms.contains(&match_edge));
    }

    #[test]
    fn test_scheduler_tree_mirror_tracks_queue_mutations_without_rebuild() {
        let g = build_graph(4, &[(0, 1, 5), (1, 2, 7), (2, 3, 9)]);
        let mut state = BlossomVState::new(&g);

        state.rebuild_scheduler_tree_mirror();

        state.set_generic_pq0(0, 0);
        assert!(state.scheduler_trees[0].pq0.contains(&0));
        state.remove_edge_from_generic_queue(0);
        assert!(!state.scheduler_trees[0].pq0.contains(&0));

        state.set_generic_pq00(1, 0, 0);
        assert!(state.scheduler_trees[0].pq00_local.contains(&1));
        state.remove_edge_from_generic_queue(1);
        assert!(!state.scheduler_trees[0].pq00_local.contains(&1));

        state.set_generic_pq_blossoms(2, 0);
        assert!(state.scheduler_trees[0].pq_blossoms.contains(&2));
        state.remove_edge_from_generic_queue(2);
        assert!(!state.scheduler_trees[0].pq_blossoms.contains(&2));

        state.set_generic_pq00(0, 0, 2);
        assert!(state.scheduler_tree_edges[0].pq00.contains(&0));
        state.remove_edge_from_generic_queue(0);
        assert!(!state.scheduler_tree_edges[0].pq00.contains(&0));

        state.set_generic_pq01(1, 0, 2);
        let dir = state.scheduler_tree_edge_dir(0, 0).expect("pair should attach to root 0");
        assert!(state.scheduler_tree_edges[0].pq01[dir].contains(&1));
        state.remove_edge_from_generic_queue(1);
        assert!(!state.scheduler_tree_edges[0].pq01[dir].contains(&1));
    }

    #[test]
    fn test_scheduler_tree_mirror_tracks_root_replacement_topology() {
        let g = build_graph(4, &[(0, 1, 5)]);
        let mut state = BlossomVState::new(&g);

        state.set_generic_pq00(0, 0, 2);
        state.scheduler_trees[0].current = SchedulerCurrent::Pair { pair_idx: 77, dir: 1 };
        state.replace_generic_tree_root(0, 1);

        assert_eq!(state.scheduler_tree_edges[0].head, [2, 1]);
        assert_eq!(state.scheduler_trees[1].first[0], Some(0));
        assert_eq!(state.scheduler_trees[2].first[1], Some(0));
        assert_eq!(state.scheduler_tree_edge_other(0, 1), Some(2));
        assert_eq!(state.scheduler_tree_edge_dir(0, 1), Some(0));
        assert_eq!(state.scheduler_trees[1].current, SchedulerCurrent::None);
    }

    #[test]
    fn test_add_generic_tree_edge_syncs_shadow_from_scheduler_topology() {
        let g = build_graph(4, &[(0, 1, 5)]);
        let mut state = BlossomVState::new(&g);

        state.ensure_generic_tree_slot(0);
        state.ensure_generic_tree_slot(2);
        state.generic_trees[0].tree_edges[0] = vec![77];
        state.generic_trees[2].tree_edges[1] = vec![88];

        let pair_idx = state.add_generic_tree_edge(0, 2);

        assert_eq!(pair_idx, 0);
        assert_eq!(state.scheduler_tree_edges[0].head, [2, 0]);
        assert_eq!(state.scheduler_trees[0].first[0], Some(0));
        assert_eq!(state.scheduler_trees[2].first[1], Some(0));
        assert_eq!(
            state.scheduler_trees[2].current,
            SchedulerCurrent::Pair { pair_idx: 0, dir: 0 }
        );
        assert_eq!(state.generic_trees[0].tree_edges[0], vec![0]);
        assert!(state.generic_trees[0].tree_edges[1].is_empty());
        assert_eq!(state.generic_trees[2].tree_edges[1], vec![0]);
        assert!(state.generic_trees[2].tree_edges[0].is_empty());
    }

    #[test]
    fn test_replace_generic_tree_root_uses_scheduler_topology_when_generic_shadow_is_stale() {
        let g = build_graph(4, &[(0, 1, 5)]);
        let mut state = BlossomVState::new(&g);

        state.set_generic_pq00(0, 0, 2);
        state.generic_trees[0].tree_edges = [Vec::new(), Vec::new()];
        state.generic_trees[2].tree_edges = [Vec::new(), Vec::new()];

        state.replace_generic_tree_root(0, 1);

        assert_eq!(state.scheduler_tree_edges[0].head, [2, 1]);
        assert_eq!(state.generic_pairs[0].head, [2, 1]);
        assert_eq!(state.scheduler_trees[1].first[0], Some(0));
        assert_eq!(state.scheduler_trees[2].first[1], Some(0));
        assert_eq!(state.generic_trees[1].tree_edges[0], vec![0]);
        assert!(state.generic_trees[1].tree_edges[1].is_empty());
        assert!(state.generic_trees[0].tree_edges[0].is_empty());
        assert!(state.generic_trees[0].tree_edges[1].is_empty());
    }

    #[test]
    fn test_detach_scheduler_root_topology_syncs_shadow_heads_and_root_edges() {
        let g = build_graph(4, &[(0, 1, 5)]);
        let mut state = BlossomVState::new(&g);

        state.set_generic_pq00(0, 0, 2);
        state.generic_pairs[0].head = [77, 88];
        state.generic_trees[0].tree_edges[0] = vec![55];

        state.detach_scheduler_root_topology(0);
        state.sync_generic_root_topology_from_scheduler(0);

        assert_eq!(state.scheduler_tree_edges[0].head, [2, NONE]);
        assert_eq!(state.generic_pairs[0].head, [2, NONE]);
        assert!(state.generic_trees[0].tree_edges[0].is_empty());
        assert!(state.generic_trees[0].tree_edges[1].is_empty());
    }

    #[test]
    fn test_clear_generic_queues_for_root_uses_scheduler_pair_heads_when_shadow_is_stale() {
        let g = build_graph(4, &[(0, 1, 5)]);
        let mut state = BlossomVState::new(&g);

        state.set_generic_pq00(0, 0, 2);
        state.generic_pairs[0].head = [77, 88];

        state.clear_generic_queues_for_root(0);

        assert_eq!(state.edge_queue_owner(0), GenericQueueState::None);
        assert_eq!(state.scheduler_tree_edges[0].head, [2, NONE]);
        assert_eq!(state.generic_pairs[0].head, [2, NONE]);
    }

    #[test]
    fn test_replace_generic_queue_root_retargets_only_root_owned_scheduler_queues() {
        let g = build_graph(5, &[(0, 1, 5), (0, 2, 7), (0, 3, 9), (0, 4, 11)]);
        let mut state = BlossomVState::new(&g);

        state.set_generic_pq0(0, 0);
        state.set_generic_pq00(1, 0, 0);
        state.set_generic_pq_blossoms(2, 0);
        state.set_generic_pq00(3, 0, 2);
        let pq0_stamp = state.edge_queue_stamp(0);
        let pq00_local_stamp = state.edge_queue_stamp(1);
        let pq_blossoms_stamp = state.edge_queue_stamp(2);
        let pair_stamp = state.edge_queue_stamp(3);

        state.generic_trees[0].pq0.clear();
        state.generic_trees[0].pq00_local.clear();
        state.generic_trees[0].pq_blossoms.clear();

        state.replace_generic_queue_root(0, 1);

        assert_eq!(state.edge_queue_owner(0), GenericQueueState::Pq0 { root: 1 });
        assert_eq!(state.edge_queue_owner(1), GenericQueueState::Pq00Local { root: 1 });
        assert_eq!(state.edge_queue_owner(2), GenericQueueState::PqBlossoms { root: 1 });
        assert_eq!(state.edge_queue_owner(3), GenericQueueState::Pq00Pair { pair_idx: 0 });
        assert_eq!(state.edge_queue_stamp(0), pq0_stamp);
        assert_eq!(state.edge_queue_stamp(1), pq00_local_stamp);
        assert_eq!(state.edge_queue_stamp(2), pq_blossoms_stamp);
        assert_eq!(state.edge_queue_stamp(3), pair_stamp);
        assert_eq!(state.scheduler_tree_edges[0].head, [2, 1]);
        assert_eq!(state.generic_pairs[0].head, [2, 1]);
    }

    #[test]
    fn test_ensure_generic_tree_edge_prefers_scheduler_topology() {
        let g = build_graph(4, &[(0, 1, 5)]);
        let mut state = BlossomVState::new(&g);

        state.set_generic_pq00(0, 0, 2);
        state.generic_trees[0].tree_edges = [Vec::new(), Vec::new()];
        state.generic_trees[2].tree_edges = [Vec::new(), Vec::new()];

        let pair_idx = state.ensure_generic_tree_edge(0, 2);
        assert_eq!(pair_idx, 0);
        assert_eq!(state.generic_pairs.len(), 1);
    }

    #[test]
    fn test_ensure_scheduler_tree_edge_slot_does_not_clobber_existing_scheduler_head() {
        let g = build_graph(4, &[(0, 1, 5)]);
        let mut state = BlossomVState::new(&g);

        state.set_generic_pq00(0, 0, 2);
        state.generic_pairs[0].head = [77, 88];

        state.ensure_scheduler_tree_edge_slot(0);

        assert_eq!(state.scheduler_tree_edges[0].head, [2, 0]);
        assert_eq!(state.generic_pairs[0].head, [77, 88]);
    }

    #[test]
    fn test_ensure_scheduler_tree_edge_slot_initializes_blank_scheduler_head() {
        let g = build_graph(4, &[(0, 1, 5)]);
        let mut state = BlossomVState::new(&g);

        state.generic_pairs.push(GenericPairQueues::new(0, 2));

        state.ensure_scheduler_tree_edge_slot(0);

        assert_eq!(state.scheduler_tree_edges[0].head, [NONE, NONE]);
        assert_eq!(state.generic_pairs[0].head, [2, 0]);
    }

    #[test]
    fn test_set_generic_pq01_other_side_creates_pair_and_seeds_neighbor_current() {
        let g = build_graph(4, &[(0, 1, 5)]);
        let mut state = BlossomVState::new(&g);

        state.set_generic_pq01_other_side(0, 0, 2);

        assert_eq!(state.generic_pairs.len(), 1);
        assert_eq!(
            state.scheduler_trees[2].current,
            SchedulerCurrent::Pair { pair_idx: 0, dir: 0 }
        );
        assert_eq!(state.edge_queue_owner(0), GenericQueueState::Pq01Pair { pair_idx: 0, dir: 0 });
        assert_eq!(state.generic_pairs[0].pq01[0], vec![0]);
        assert_eq!(state.scheduler_tree_edges[0].pq01[0], vec![0]);
    }

    #[test]
    fn test_add_generic_tree_edge_uses_cxx_head_orientation() {
        let g = build_graph(4, &[(0, 1, 5)]);
        let mut state = BlossomVState::new(&g);

        let pair_idx = state.add_generic_tree_edge(0, 2);

        assert_eq!(state.scheduler_tree_edges[pair_idx].head, [2, 0]);
        assert_eq!(state.generic_pairs[pair_idx].head, [2, 0]);
        assert_eq!(state.scheduler_tree_edge_dir(pair_idx, 0), Some(0));
        assert_eq!(state.scheduler_tree_edge_dir(pair_idx, 2), Some(1));
    }

    #[test]
    fn test_set_generic_pq01_other_side_prefers_scheduler_topology_over_neighbor_current() {
        let g = build_graph(4, &[(0, 1, 5)]);
        let mut state = BlossomVState::new(&g);

        state.set_generic_pq00(0, 0, 2);
        state.remove_edge_from_generic_queue(0);

        state.scheduler_trees[2].current = SchedulerCurrent::Pair { pair_idx: 99, dir: 1 };

        state.set_generic_pq01_other_side(0, 0, 2);

        assert_eq!(state.edge_queue_owner(0), GenericQueueState::Pq01Pair { pair_idx: 0, dir: 0 });
        assert_eq!(state.generic_pairs[0].pq01[0], vec![0]);
        assert_eq!(state.scheduler_tree_edges[0].pq01[0], vec![0]);
    }

    #[test]
    fn test_set_generic_pq01_pair_slot_uses_scheduler_pair_len_when_shadow_is_stale() {
        let g = build_graph(4, &[(0, 1, 5)]);
        let mut state = BlossomVState::new(&g);

        state.set_generic_pq00(0, 0, 2);
        state.remove_edge_from_generic_queue(0);
        state.generic_pairs.clear();

        state.set_generic_pq01_pair_slot(0, 0, 0, false);

        assert_eq!(state.edge_queue_owner(0), GenericQueueState::Pq01Pair { pair_idx: 0, dir: 0 });
        assert_eq!(state.scheduler_tree_edges[0].pq01[0], vec![0]);
    }

    #[test]
    fn test_set_generic_pq00_pair_slot_uses_scheduler_pair_len_when_shadow_is_stale() {
        let g = build_graph(4, &[(0, 1, 5)]);
        let mut state = BlossomVState::new(&g);

        state.set_generic_pq00(0, 0, 2);
        state.remove_edge_from_generic_queue(0);
        state.generic_pairs.clear();

        state.set_generic_pq00_pair_slot(0, 0, false);

        assert_eq!(state.edge_queue_owner(0), GenericQueueState::Pq00Pair { pair_idx: 0 });
        assert_eq!(state.scheduler_tree_edges[0].pq00, vec![0]);
    }

    #[test]
    fn test_tree_min_pq00_local_for_step3_uses_scheduler_state_when_shadow_is_stale() {
        let g = build_graph(4, &[(0, 1, 5)]);
        let mut state = BlossomVState::new(&g);

        state.set_generic_pq00(0, 0, 0);
        state.generic_trees.clear();

        state.nodes[0].is_outer = true;
        state.nodes[0].flag = PLUS;
        state.nodes[0].is_processed = true;
        state.nodes[0].tree_root = 0;
        state.nodes[0].is_tree_root = true;

        state.nodes[1].is_outer = true;
        state.nodes[1].flag = PLUS;
        state.nodes[1].is_processed = true;
        state.nodes[1].tree_root = 0;
        state.nodes[1].is_tree_root = false;

        let best = state.tree_min_pq00_local_for_step3(0);

        let (e_idx, left, right, slack) =
            best.expect("scheduler local pq00 edge should still be found");
        assert_eq!(e_idx, 0);
        assert!((left == 0 && right == 1) || (left == 1 && right == 0));
        assert_eq!(slack, state.edges[0].slack);
        assert_eq!(state.edge_queue_owner(0), GenericQueueState::Pq00Local { root: 0 });
    }

    #[test]
    fn test_clear_generic_tree_currents_local_clears_root_and_incident_scheduler_roots() {
        let g = build_graph(6, &[(0, 1, 5)]);
        let mut state = BlossomVState::new(&g);

        state.generic_trees = vec![GenericTreeQueues::default(); state.nodes.len()];
        state.generic_pairs = vec![
            GenericPairQueues::new(0, 3),
            GenericPairQueues::new(0, 5),
            GenericPairQueues::new(5, 3),
        ];

        state.generic_trees[0].tree_edges[0] = vec![0, 1];
        state.generic_trees[3].tree_edges[1] = vec![0, 2];
        state.generic_trees[5].tree_edges[0] = vec![2];
        state.generic_trees[5].tree_edges[1] = vec![1];
        state.rebuild_scheduler_tree_mirror();

        state.scheduler_trees[0].current = SchedulerCurrent::Pair { pair_idx: 99, dir: 0 };
        state.scheduler_trees[3].current = SchedulerCurrent::Pair { pair_idx: 0, dir: 1 };
        state.scheduler_trees[5].current = SchedulerCurrent::Pair { pair_idx: 1, dir: 1 };
        state.scheduler_trees[4].current = SchedulerCurrent::Pair { pair_idx: 77, dir: 1 };

        state.clear_generic_tree_currents_local(0);

        assert_eq!(state.scheduler_trees[0].current, SchedulerCurrent::None);
        assert_eq!(state.scheduler_trees[3].current, SchedulerCurrent::None);
        assert_eq!(state.scheduler_trees[5].current, SchedulerCurrent::None);
        assert_eq!(
            state.scheduler_trees[4].current,
            SchedulerCurrent::Pair { pair_idx: 77, dir: 1 }
        );
    }

    #[test]
    fn test_seed_tree_root_frontier_seeds_only_one_unprocessed_root() {
        let g = build_graph(4, &[(0, 1, 5), (0, 2, 7)]);
        let mut state = BlossomVState::new(&g);

        state.nodes[0].flag = PLUS;
        state.nodes[0].is_outer = true;
        state.nodes[0].is_tree_root = true;
        state.nodes[0].tree_root = 0;
        state.nodes[0].is_processed = false;

        state.nodes[1].flag = FREE;
        state.nodes[1].is_outer = true;

        state.nodes[2].flag = PLUS;
        state.nodes[2].is_outer = true;
        state.nodes[2].is_tree_root = true;
        state.nodes[2].tree_root = 2;
        state.nodes[2].is_processed = true;

        state.seed_tree_root_frontier(0);

        assert!(state.nodes[0].is_processed);
        assert!(state.scheduler_trees[0].pq0.contains(&0));
        assert_eq!(state.edge_queue_owner(0), GenericQueueState::Pq0 { root: 0 });
        assert_eq!(state.generic_pairs.len(), 1);
        assert_eq!(state.scheduler_tree_edges[0].head, [2, 0]);
        assert!(state.scheduler_tree_edges[0].pq00.contains(&1));
        assert_eq!(state.edge_queue_owner(1), GenericQueueState::Pq00Pair { pair_idx: 0 });
    }

    #[test]
    fn test_step1_scheduler_scan_preserves_unrelated_current_pointers() {
        let g = build_graph(6, &[(0, 1, 5)]);
        let mut state = BlossomVState::new(&g);

        state.generic_trees = vec![GenericTreeQueues::default(); state.nodes.len()];
        state.generic_pairs = vec![GenericPairQueues::new(0, 3)];
        state.generic_trees[0].tree_edges[0] = vec![0];
        state.generic_trees[3].tree_edges[1] = vec![0];

        for &root in &[0usize, 3, 4] {
            state.nodes[root].is_outer = true;
            state.nodes[root].is_tree_root = true;
            state.nodes[root].tree_root = root as u32;
            state.nodes[root].tree_eps = 3;
        }

        state.rebuild_scheduler_tree_mirror();
        state.scheduler_trees[4].current = SchedulerCurrent::Pair { pair_idx: 77, dir: 1 };

        assert_eq!(state.find_tree_step1_augment_edge_from_scheduler(0), None);

        assert_eq!(state.scheduler_trees[0].current, SchedulerCurrent::Root);
        assert_eq!(
            state.scheduler_trees[3].current,
            SchedulerCurrent::Pair { pair_idx: 0, dir: 0 }
        );
        assert_eq!(
            state.scheduler_trees[4].current,
            SchedulerCurrent::Pair { pair_idx: 77, dir: 1 }
        );
    }

    #[test]
    fn test_process_tree_primal_step1_augment_preserves_unrelated_scheduler_currents() {
        let g = build_graph(6, &[(0, 3, 5)]);
        let mut state = BlossomVState::new(&g);

        for node in &mut state.nodes {
            node.flag = FREE;
            node.is_tree_root = false;
            node.is_processed = false;
            node.tree_root = NONE;
            node.tree_eps = 0;
            node.first_tree_child = NONE;
            node.tree_sibling_prev = NONE;
            node.tree_sibling_next = NONE;
            node.tree_parent_arc = NONE;
            node.is_outer = true;
        }

        for &root in &[0usize, 3, 4] {
            state.nodes[root].flag = PLUS;
            state.nodes[root].is_tree_root = true;
            state.nodes[root].is_processed = true;
            state.nodes[root].tree_root = root as u32;
            state.nodes[root].tree_eps = 3;
        }

        state.edges[0].head = [0, 3];
        state.edges[0].head0 = [0, 3];
        state.edges[0].slack = 0;
        state.set_generic_pq00(0, 0, 3);

        state.ensure_scheduler_tree_slot(4);
        state.scheduler_trees[4].current = SchedulerCurrent::Pair { pair_idx: 77, dir: 1 };

        assert!(state.process_tree_primal(0));

        assert_eq!(state.scheduler_trees[0].current, SchedulerCurrent::None);
        assert_eq!(state.scheduler_trees[3].current, SchedulerCurrent::None);
        assert_eq!(
            state.scheduler_trees[4].current,
            SchedulerCurrent::Pair { pair_idx: 77, dir: 1 }
        );
    }

    #[test]
    fn test_process_tree_primal_step1_augment_detaches_roots_via_scheduler_topology() {
        let g = build_graph(6, &[(0, 3, 5)]);
        let mut state = BlossomVState::new(&g);

        for node in &mut state.nodes {
            node.flag = FREE;
            node.is_tree_root = false;
            node.is_processed = false;
            node.tree_root = NONE;
            node.tree_eps = 0;
            node.first_tree_child = NONE;
            node.tree_sibling_prev = NONE;
            node.tree_sibling_next = NONE;
            node.tree_parent_arc = NONE;
            node.is_outer = true;
        }

        for &root in &[0usize, 3] {
            state.nodes[root].flag = PLUS;
            state.nodes[root].is_tree_root = true;
            state.nodes[root].is_processed = true;
            state.nodes[root].tree_root = root as u32;
            state.nodes[root].tree_eps = 3;
        }

        state.edges[0].head = [0, 3];
        state.edges[0].head0 = [0, 3];
        state.edges[0].slack = 0;
        state.set_generic_pq00(0, 0, 3);

        state.generic_trees[0].tree_edges = [Vec::new(), Vec::new()];
        state.generic_trees[3].tree_edges = [Vec::new(), Vec::new()];

        assert!(state.process_tree_primal(0));
        assert_eq!(state.generic_pairs[0].head, [NONE, NONE]);
        assert_eq!(state.scheduler_trees[0].first, [None, None]);
        assert_eq!(state.scheduler_trees[3].first, [None, None]);
    }

    #[test]
    fn test_detach_generic_root_after_augment_uses_scheduler_root_queues_when_shadow_is_stale() {
        let g = build_graph(4, &[(0, 1, 5), (0, 2, 7), (0, 3, 9)]);
        let mut state = BlossomVState::new(&g);

        state.set_generic_pq0(0, 0);
        state.set_generic_pq00(1, 0, 0);
        state.set_generic_pq_blossoms(2, 0);

        state.generic_trees[0].pq0.clear();
        state.generic_trees[0].pq00_local.clear();
        state.generic_trees[0].pq_blossoms.clear();

        state.detach_generic_root_after_augment(0);

        assert_eq!(state.edge_queue_owner(0), GenericQueueState::None);
        assert_eq!(state.edge_queue_owner(1), GenericQueueState::None);
        assert_eq!(state.edge_queue_owner(2), GenericQueueState::None);
        assert!(state.scheduler_trees[0].pq0.is_empty());
        assert!(state.scheduler_trees[0].pq00_local.is_empty());
        assert!(state.scheduler_trees[0].pq_blossoms.is_empty());
    }

    #[test]
    fn test_process_tree_primal_step2_grow_uses_scheduler_pq0_without_rebuild() {
        let g = build_graph(4, &[(0, 1, 5), (1, 2, 7)]);
        let mut state = BlossomVState::new(&g);

        for node in &mut state.nodes {
            node.flag = FREE;
            node.is_tree_root = false;
            node.is_processed = false;
            node.tree_root = NONE;
            node.tree_eps = 0;
            node.first_tree_child = NONE;
            node.tree_sibling_prev = NONE;
            node.tree_sibling_next = NONE;
            node.tree_parent_arc = NONE;
            node.match_arc = NONE;
            node.is_outer = true;
        }

        state.nodes[0].flag = PLUS;
        state.nodes[0].is_tree_root = true;
        state.nodes[0].is_processed = true;
        state.nodes[0].tree_root = 0;
        state.nodes[0].tree_eps = 3;

        state.nodes[1].flag = FREE;
        state.nodes[1].match_arc = make_arc(1, 1);
        state.nodes[2].flag = FREE;
        state.nodes[2].match_arc = make_arc(1, 0);

        state.edges[0].head = [0, 1];
        state.edges[0].head0 = [0, 1];
        state.edges[0].slack = 0;
        state.edges[1].head = [1, 2];
        state.edges[1].head0 = [1, 2];

        state.set_generic_pq0(0, 0);
        state.generic_trees[0].pq0.clear();

        assert!(state.process_tree_primal(0));
        assert_eq!(state.nodes[1].flag, MINUS);
        assert_eq!(state.nodes[1].tree_root, 0);
        assert_eq!(state.nodes[2].flag, PLUS);
        assert_eq!(state.nodes[2].tree_root, 0);
    }

    #[test]
    fn test_into_pairs_checked_accepts_reversed_original_endpoint_order() {
        let g = build_graph(2, &[(0, 1, 5)]);
        let mut state = BlossomVState::new(&g);
        state.edges[0].head0 = [1, 0];

        let pairs = state.into_pairs_checked().expect("single edge should remain a valid matching");
        let normalized =
            pairs.iter().map(|&(r, c)| (r.as_(), c.as_())).collect::<Vec<(usize, usize)>>();

        assert_eq!(normalized, vec![(0, 1)]);
    }

    #[test]
    fn test_into_pairs_checked_rejects_duplicate_vertex_usage() {
        let g = build_graph(4, &[(0, 1, 5), (1, 2, 7)]);
        let mut state = BlossomVState::new(&g);

        state.nodes[0].match_arc = make_arc(0, 0);
        state.nodes[1].match_arc = make_arc(1, 0);
        state.nodes[2].match_arc = make_arc(1, 1);
        state.nodes[3].match_arc = make_arc(0, 1);
        state.edges[0].head0 = [0, 1];
        state.edges[1].head0 = [1, 2];

        let err =
            state.into_pairs_checked().expect_err("duplicate vertex usage should be rejected");
        assert!(matches!(err, BlossomVError::NoPerfectMatching));
    }

    #[test]
    fn test_k4_has_tight_grow_edge() {
        // After greedy: (0,1) matched, 2 and 3 are tree roots
        // Edge (0,2) should be tight (slack=0), enabling GROW
        let g = build_graph(4, &[(0, 1, 1), (0, 2, 2), (0, 3, 3), (1, 2, 4), (1, 3, 5), (2, 3, 6)]);
        let state = BlossomVState::new(&g);
        assert_eq!(state.test_tree_num(), 2);

        // Find a tight (+,free) edge
        let mut found_grow = false;
        for e in 0..state.test_edge_count() {
            if state.test_edge_slack(e) == 0 {
                let (u_orig, v_orig) = state.test_edge_endpoints(e);
                let fu = state.test_flag(u_orig as usize);
                let fv = state.test_flag(v_orig as usize);
                if (fu == PLUS && fv == FREE) || (fu == FREE && fv == PLUS) {
                    found_grow = true;
                }
            }
        }
        assert!(found_grow, "Should have a tight (+,free) edge for GROW");
    }

    #[test]
    fn test_k4_solve_produces_two_pairs() {
        let g = build_graph(4, &[(0, 1, 1), (0, 2, 2), (0, 3, 3), (1, 2, 4), (1, 3, 5), (2, 3, 6)]);
        let result = BlossomVState::new(&g).solve();
        let pairs = result.expect("K4 has a perfect matching");
        assert_eq!(pairs.len(), 2, "K4 should have 2 matched pairs, got {:?}", pairs);
    }

    #[test]
    fn test_case_440_augment_correctness() {
        // Case 440: n=4, edges: (2,3,-20), (0,1,-17), (1,3,-84), (0,3,-28)
        // Expected: [(0,1),(2,3)], cost=-37
        // Our bug: returns [(1,3),(2,3)] — vertex 3 in two pairs
        let g = build_graph(4, &[(2, 3, -20), (0, 1, -17), (1, 3, -84), (0, 3, -28)]);
        let result = BlossomVState::new(&g).solve();
        let pairs = result.expect("should find perfect matching");
        assert_eq!(pairs.len(), 2, "should have 2 pairs, got {:?}", pairs);
        // Check no vertex used twice
        let mut used = [false; 4];
        for &(u, v) in &pairs {
            assert!(!used[u], "vertex {u} used twice in {:?}", pairs);
            assert!(!used[v], "vertex {v} used twice in {:?}", pairs);
            used[u] = true;
            used[v] = true;
        }
    }

    #[test]
    fn test_case15_blossom_k6() {
        // Ground truth case #15: K6 with blossoms, n=6, expected cost=-44
        let g = build_graph(
            6,
            &[
                (0, 1, -14),
                (0, 2, -68),
                (0, 3, -5),
                (0, 4, 83),
                (0, 5, -61),
                (1, 2, 71),
                (1, 3, -13),
                (1, 4, 21),
                (1, 5, 63),
                (2, 3, 17),
                (2, 4, 83),
                (2, 5, -59),
                (3, 4, 87),
                (3, 5, 3),
                (4, 5, 42),
            ],
        );
        let result = BlossomVState::new(&g).solve();
        let pairs = result.expect("should find perfect matching");
        assert_eq!(pairs.len(), 3, "should have 3 pairs, got {:?}", pairs);
    }

    fn assert_scheduler_tree_mirror_matches_generic_state<M>(state: &mut BlossomVState<M>)
    where
        M: SparseValuedMatrix2D + ?Sized,
        M::Value: Number + AsPrimitive<i64>,
        M::RowIndex: PositiveInteger,
        M::ColumnIndex: PositiveInteger,
    {
        state.rebuild_scheduler_tree_mirror();

        assert_eq!(state.scheduler_trees.len(), state.nodes.len());
        assert_eq!(state.scheduler_tree_edges.len(), state.generic_pairs.len());

        for (root_idx, tree) in state.generic_trees.iter().enumerate() {
            let mirror = &state.scheduler_trees[root_idx];
            let root = root_idx as u32;
            assert_eq!(mirror.root, root);
            assert_eq!(mirror.eps, state.tree_eps(root));
            assert_eq!(mirror.first[0], tree.tree_edges[0].first().copied());
            assert_eq!(mirror.first[1], tree.tree_edges[1].first().copied());
            assert_eq!(mirror.pq0, tree.pq0);
            assert_eq!(mirror.pq00_local, tree.pq00_local);
            assert_eq!(mirror.pq_blossoms, tree.pq_blossoms);

            for dir in 0..2usize {
                for (pos, &pair_idx) in tree.tree_edges[dir].iter().enumerate() {
                    let expected_next = tree.tree_edges[dir].get(pos + 1).copied();
                    assert_eq!(state.scheduler_tree_edges[pair_idx].next[dir], expected_next);
                }
            }
        }

        for (pair_idx, pair) in state.generic_pairs.iter().enumerate() {
            let mirror = &state.scheduler_tree_edges[pair_idx];
            assert_eq!(mirror.head, pair.head);
            assert_eq!(mirror.pq00, pair.pq00);
            assert_eq!(mirror.pq01, pair.pq01);
        }
    }

    #[test]
    fn test_scheduler_tree_mirror_tracks_current_tree_edge_topology() {
        let g = build_graph(6, &[(0, 1, 1)]);
        let mut state = BlossomVState::new(&g);
        state.generic_trees = vec![GenericTreeQueues::default(); state.nodes.len()];
        state.generic_pairs = vec![
            GenericPairQueues::new(0, 3),
            GenericPairQueues::new(0, 5),
            GenericPairQueues::new(5, 3),
        ];

        state.nodes[0].tree_eps = 11;
        state.nodes[3].tree_eps = 7;
        state.nodes[5].tree_eps = 5;

        state.generic_trees[0].tree_edges[0] = vec![0, 1];
        state.generic_trees[3].tree_edges[1] = vec![0, 2];
        state.generic_trees[5].tree_edges[0] = vec![2];
        state.generic_trees[5].tree_edges[1] = vec![1];

        state.generic_trees[0].pq0 = vec![4, 9];
        state.generic_trees[0].pq00_local = vec![6];
        state.generic_trees[5].pq_blossoms = vec![8];

        state.generic_pairs[0].pq00 = vec![10];
        state.generic_pairs[0].pq01[0] = vec![11, 12];
        state.generic_pairs[1].pq01[0] = vec![13];
        state.generic_pairs[2].pq01[1] = vec![14, 15];

        state.rebuild_scheduler_tree_mirror();

        assert_eq!(state.scheduler_trees[0].root, 0);
        assert_eq!(state.scheduler_trees[0].eps, 11);
        assert_eq!(state.scheduler_trees[0].first, [Some(0), None]);
        assert_eq!(state.scheduler_trees[0].pq0, vec![4, 9]);
        assert_eq!(state.scheduler_trees[0].pq00_local, vec![6]);

        assert_eq!(state.scheduler_trees[3].root, 3);
        assert_eq!(state.scheduler_trees[3].eps, 7);
        assert_eq!(state.scheduler_trees[3].first, [None, Some(0)]);
        assert_eq!(state.scheduler_trees[3].current, SchedulerCurrent::None);

        assert_eq!(state.scheduler_trees[5].root, 5);
        assert_eq!(state.scheduler_trees[5].eps, 5);
        assert_eq!(state.scheduler_trees[5].first, [Some(2), Some(1)]);
        assert_eq!(state.scheduler_trees[5].current, SchedulerCurrent::None);
        assert_eq!(state.scheduler_trees[5].pq_blossoms, vec![8]);

        assert_eq!(state.scheduler_tree_edges[0].head, [3, 0]);
        assert_eq!(state.scheduler_tree_edges[0].next, [Some(1), Some(2)]);
        assert_eq!(state.scheduler_tree_edges[0].pq00, vec![10]);
        assert_eq!(state.scheduler_tree_edges[0].pq01[0], vec![11, 12]);

        assert_eq!(state.scheduler_tree_edges[1].head, [5, 0]);
        assert_eq!(state.scheduler_tree_edges[1].next, [None, None]);
        assert_eq!(state.scheduler_tree_edges[1].pq01[0], vec![13]);

        assert_eq!(state.scheduler_tree_edges[2].head, [3, 5]);
        assert_eq!(state.scheduler_tree_edges[2].next, [None, None]);
        assert_eq!(state.scheduler_tree_edges[2].pq01[1], vec![14, 15]);
    }

    #[test]
    fn test_scheduler_tree_mirror_matches_live_case_87417_state() {
        let mut state = case_87417_state_after_generic_steps(2);
        assert_scheduler_tree_mirror_matches_generic_state(&mut state);
    }
}
