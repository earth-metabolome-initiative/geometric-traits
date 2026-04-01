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

mod dual_update;
mod generic_queue;
mod init_global_phase;
mod primal_phase;

#[cfg(test)]
mod test_support;

#[cfg(test)]
use self::test_support::{
    BlossomVTestState, GenericPairQueues, GenericPrimalEvent, GenericPrimalStepTrace,
    InitGlobalEvent, InitGlobalStepTrace,
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
    /// A direct child of this blossom, used as a stable entry point when
    /// expanding the blossom back into its cycle.
    blossom_child: u32,
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
            blossom_child: NONE,
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

struct EdgeQueueKeyStore<'a> {
    edges: &'a mut [Edge],
    stamps: &'a [u64],
}

impl<'a> EdgeQueueKeyStore<'a> {
    #[inline]
    fn new(edges: &'a mut [Edge], stamps: &'a [u64]) -> Self {
        Self { edges, stamps }
    }
}

impl PQKeyStore for EdgeQueueKeyStore<'_> {
    #[inline]
    fn get_key(&self, idx: u32) -> i64 {
        self.edges[idx as usize].slack
    }

    #[inline]
    fn set_key(&mut self, idx: u32, key: i64) {
        self.edges[idx as usize].slack = key;
    }

    #[inline]
    fn less(&self, lhs: u32, rhs: u32) -> bool {
        let lhs_slack = self.edges[lhs as usize].slack;
        let rhs_slack = self.edges[rhs as usize].slack;
        lhs_slack < rhs_slack
            || (lhs_slack == rhs_slack && self.stamps[lhs as usize] > self.stamps[rhs as usize])
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
    edge_seen: Vec<bool>,
    node_seen_stamp: Vec<u32>,
    node_seen_epoch: u32,
    incident_pairs: Vec<(usize, usize)>,
    pair_marks: Vec<u8>,
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
    dual_pair_tree_eps00: Vec<i64>,
    dual_pair_tree_eps01_dir0: Vec<i64>,
    dual_pair_tree_eps01_dir1: Vec<i64>,
    dual_pair_tree_ready: Vec<bool>,
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
    edge_queue_slot: Vec<usize>,
    edge_queue_stamp: Vec<u64>,
    pq_nodes: Vec<PQNode>,
    root_list_head: u32,
    #[cfg(test)]
    test_state: BlossomVTestState,
    scheduler_trees: Vec<PersistentTreeState>,
    scheduler_tree_edges: Vec<PersistentTreeEdgeState>,
    scratch: BlossomVScratch,
    generic_queue_epoch: u64,
    tree_num: usize,
    blossom_count: usize,
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
            edge_queue_slot: vec![usize::MAX; edge_num],
            edge_queue_stamp: vec![0; edge_num],
            pq_nodes,
            root_list_head: NONE,
            #[cfg(test)]
            test_state: BlossomVTestState::with_node_capacity(n),
            scheduler_trees: Vec::new(),
            scheduler_tree_edges: Vec::new(),
            scratch: BlossomVScratch::default(),
            generic_queue_epoch: 0,
            tree_num: 0,
            blossom_count: 0,
        };

        if n > 0 {
            state.init_greedy();
        }

        state
    }

    /// Greedy initialization: set initial dual variables and matching.
    ///
    /// For each node `v`, set `y[v]` to the doubled minimum incident edge cost.
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
            self.test_state.init_global_trace.clear();
            self.test_state.init_global_steps.clear();
        }

        if self.tree_num == 0 {
            return;
        }

        self.init_global_ported();
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

    fn scheduler_tree_best_pq0_edge(&mut self, root: u32) -> Option<u32> {
        loop {
            let e_idx = self.scheduler_tree_heap_min_pq0_edge(root)?;
            if (e_idx as usize) >= self.edge_num
                || !matches!(
                    self.edge_queue_owner(e_idx),
                    GenericQueueState::Pq0 { root: q_root } if q_root == root
                )
            {
                self.remove_edge_from_generic_queue(e_idx);
                continue;
            }
            return Some(e_idx);
        }
    }

    #[inline]
    fn scheduler_tree_heap_min_pq0_edge(&self, root: u32) -> Option<u32> {
        if (root as usize) >= self.scheduler_trees.len() {
            None
        } else {
            self.scheduler_trees[root as usize].pq0_heap.get_min()
        }
    }

    fn tree_min_pq0_for_duals(&mut self, root: u32) -> Option<u32> {
        loop {
            let e_idx = self.scheduler_tree_heap_min_pq0_edge(root)?;
            if (e_idx as usize) >= self.edge_num
                || !matches!(
                    self.edge_queue_owner(e_idx),
                    GenericQueueState::Pq0 { root: q_root } if q_root == root
                )
            {
                self.remove_edge_from_generic_queue(e_idx);
                continue;
            }

            let is_valid =
                [(0usize, 1usize), (1usize, 0usize)].into_iter().any(|(plus_dir, free_dir)| {
                    let plus = self.edge_head_outer(e_idx, plus_dir);
                    let free = self.edge_head_outer(e_idx, free_dir);
                    plus != NONE
                        && free != NONE
                        && plus != free
                        && self.nodes[plus as usize].is_outer
                        && self.nodes[plus as usize].flag == PLUS
                        && self.nodes[plus as usize].is_processed
                        && self.find_tree_root(plus) == root
                        && self.nodes[free as usize].is_outer
                        && self.nodes[free as usize].flag == FREE
                });
            if is_valid {
                return Some(e_idx);
            }

            self.remove_edge_from_generic_queue(e_idx);
        }
    }

    fn scheduler_tree_best_pq_blossom_edge(&mut self, root: u32) -> Option<u32> {
        loop {
            let e_idx = self.scheduler_tree_heap_min_pq_blossom_edge(root)?;
            if (e_idx as usize) >= self.edge_num
                || !matches!(
                    self.edge_queue_owner(e_idx),
                    GenericQueueState::PqBlossoms { root: q_root } if q_root == root
                )
            {
                self.remove_edge_from_generic_queue(e_idx);
                continue;
            }
            return Some(e_idx);
        }
    }

    #[inline]
    fn scheduler_tree_heap_min_pq_blossom_edge(&self, root: u32) -> Option<u32> {
        if (root as usize) >= self.scheduler_trees.len() {
            None
        } else {
            self.scheduler_trees[root as usize].pq_blossoms_heap.get_min()
        }
    }

    fn tree_min_pq_blossom_for_duals(&mut self, root: u32) -> Option<u32> {
        loop {
            let e_idx = self.scheduler_tree_heap_min_pq_blossom_edge(root)?;
            if (e_idx as usize) >= self.edge_num
                || !matches!(
                    self.edge_queue_owner(e_idx),
                    GenericQueueState::PqBlossoms { root: q_root } if q_root == root
                )
            {
                self.remove_edge_from_generic_queue(e_idx);
                continue;
            }

            let is_valid = (0..2usize).any(|dir| {
                let blossom = self.edge_head_outer(e_idx, dir);
                blossom != NONE
                    && self.nodes[blossom as usize].is_blossom
                    && self.nodes[blossom as usize].is_outer
                    && self.nodes[blossom as usize].flag == MINUS
                    && self.find_tree_root(blossom) == root
                    && self.nodes[blossom as usize].match_arc != NONE
                    && arc_edge(self.nodes[blossom as usize].match_arc) == e_idx
            });
            if is_valid {
                return Some(e_idx);
            }

            self.remove_edge_from_generic_queue(e_idx);
        }
    }

    fn scheduler_tree_heap_min_pq00_local_edge(&self, root: u32) -> Option<u32> {
        if (root as usize) >= self.scheduler_trees.len() {
            None
        } else {
            self.scheduler_trees[root as usize].pq00_local_heap.get_min()
        }
    }

    fn tree_min_pq00_local_for_duals(&mut self, root: u32) -> Option<u32> {
        loop {
            let e_idx = self.scheduler_tree_heap_min_pq00_local_edge(root)?;
            if self.process_edge00(e_idx, false) {
                return Some(e_idx);
            }
            self.remove_edge_from_generic_queue(e_idx);
        }
    }

    fn scheduler_tree_edge_best_pq00_edge(
        &mut self,
        pair_idx: usize,
        root: u32,
        other_root: u32,
    ) -> Option<u32> {
        loop {
            let e_idx = self.scheduler_tree_edge_heap_min_pq00_edge(pair_idx)?;
            if (e_idx as usize) >= self.edge_num
                || !matches!(
                    self.edge_queue_owner(e_idx),
                    GenericQueueState::Pq00Pair { pair_idx: q_pair_idx } if q_pair_idx == pair_idx
                )
            {
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
            {
                continue;
            }
            let root_left = self.find_tree_root(left);
            let root_right = self.find_tree_root(right);
            let matches_pair = (root_left == root && root_right == other_root)
                || (root_left == other_root && root_right == root);
            if !matches_pair {
                self.remove_edge_from_generic_queue(e_idx);
                continue;
            }
            return Some(e_idx);
        }
    }

    #[inline]
    fn scheduler_tree_edge_heap_min_pq00_edge(&self, pair_idx: usize) -> Option<u32> {
        if pair_idx >= self.scheduler_tree_edges.len() {
            None
        } else {
            self.scheduler_tree_edges[pair_idx].pq00_heap.get_min()
        }
    }

    fn scheduler_tree_edge_min_pq00_edge_for_duals(
        &mut self,
        pair_idx: usize,
        root: u32,
        other_root: u32,
    ) -> Option<u32> {
        loop {
            let e_idx = self.scheduler_tree_edge_heap_min_pq00_edge(pair_idx)?;
            if (e_idx as usize) >= self.edge_num
                || !matches!(
                    self.edge_queue_owner(e_idx),
                    GenericQueueState::Pq00Pair { pair_idx: q_pair_idx } if q_pair_idx == pair_idx
                )
            {
                self.remove_edge_from_generic_queue(e_idx);
                continue;
            }

            let outer0 = self.edge_head_outer(e_idx, 0);
            let outer1 = self.edge_head_outer(e_idx, 1);
            let valid = outer0 != NONE
                && outer1 != NONE
                && outer0 != outer1
                && self.nodes[outer0 as usize].flag == PLUS
                && self.nodes[outer1 as usize].flag == PLUS
                && self.nodes[outer0 as usize].is_processed
                && self.nodes[outer1 as usize].is_processed
                && {
                    let root0 = self.find_tree_root(outer0);
                    let root1 = self.find_tree_root(outer1);
                    (root0 == root && root1 == other_root) || (root0 == other_root && root1 == root)
                };
            if valid {
                return Some(e_idx);
            }

            self.remove_edge_from_generic_queue(e_idx);
        }
    }

    #[inline]
    fn scheduler_tree_edge_heap_min_pq01_edge(&self, pair_idx: usize, dir: usize) -> Option<u32> {
        if pair_idx >= self.scheduler_tree_edges.len() || dir > 1 {
            None
        } else {
            self.scheduler_tree_edges[pair_idx].pq01_heap[dir].get_min()
        }
    }

    fn scheduler_tree_edge_min_pq01_edge_for_duals(
        &mut self,
        pair_idx: usize,
        dir: usize,
        root: u32,
        other_root: u32,
    ) -> Option<u32> {
        loop {
            let e_idx = self.scheduler_tree_edge_heap_min_pq01_edge(pair_idx, dir)?;
            if (e_idx as usize) >= self.edge_num
                || !matches!(
                    self.edge_queue_owner(e_idx),
                    GenericQueueState::Pq01Pair { pair_idx: q_pair_idx, dir: q_dir }
                        if q_pair_idx == pair_idx && q_dir == dir
                )
            {
                self.remove_edge_from_generic_queue(e_idx);
                continue;
            }

            let outer0 = self.edge_head_outer(e_idx, 0);
            let outer1 = self.edge_head_outer(e_idx, 1);
            let valid = outer0 != NONE
                && outer1 != NONE
                && outer0 != outer1
                && [(outer0, outer1), (outer1, outer0)].into_iter().any(|(plus, minus)| {
                    self.nodes[plus as usize].flag == PLUS
                        && self.nodes[plus as usize].is_processed
                        && self.find_tree_root(plus) == root
                        && self.nodes[minus as usize].flag == MINUS
                        && self.find_tree_root(minus) == other_root
                });
            if valid {
                return Some(e_idx);
            }

            self.remove_edge_from_generic_queue(e_idx);
        }
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
        let mut reseeded_missing_match_edges = false;
        loop {
            let Some(e_idx) = self.scheduler_tree_best_pq_blossom_edge(root) else {
                if !reseeded_missing_match_edges
                    && self.reseed_tree_blossom_match_edges_from_processed_plus(root)
                {
                    reseeded_missing_match_edges = true;
                    continue;
                }
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

    fn reseed_tree_blossom_match_edges_from_processed_plus(&mut self, root: u32) -> bool {
        let mut added = false;
        let mut members = self.take_tree_members_u_scratch();
        self.collect_tree_members_with_scratch(root, &mut members);
        for plus in members.iter().copied() {
            if !self.nodes[plus as usize].is_outer
                || self.nodes[plus as usize].flag != PLUS
                || !self.nodes[plus as usize].is_processed
            {
                continue;
            }

            let match_edge = arc_edge(self.nodes[plus as usize].match_arc);
            if (match_edge as usize) >= self.edge_num
                || !matches!(self.edge_queue_owner(match_edge), GenericQueueState::None)
            {
                continue;
            }

            let before = self.scheduler_trees[root as usize].pq_blossoms.len();
            self.queue_processed_plus_blossom_match_edge(plus);
            added |= self.scheduler_trees[root as usize].pq_blossoms.len() != before;
        }
        self.restore_tree_members_u_scratch(members);
        added
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

    fn find_tree_shrink_edge_with_cap(
        &mut self,
        root: u32,
        shrink_cap: i64,
    ) -> Option<(u32, u32, u32)> {
        if (root as usize) >= self.scheduler_trees.len() {
            return None;
        }

        let e_idx = self.scheduler_tree_heap_min_pq00_local_edge(root)?;
        let slack = self.edges[e_idx as usize].slack;
        if slack > shrink_cap { None } else { Some((e_idx, NONE, NONE)) }
    }

    #[cfg(test)]
    fn find_scheduler_global_augment_edge(&mut self) -> Option<(u32, u32, u32)> {
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

    fn compute_tree_local_eps(&mut self, root: u32) -> i64 {
        if (root as usize) >= self.scheduler_trees.len() {
            return i64::MAX;
        }

        let mut eps = i64::MAX;
        if let Some(e_idx) = self.tree_min_pq0_for_duals(root) {
            if (e_idx as usize) < self.edge_num {
                eps = eps.min(self.edges[e_idx as usize].slack);
            }
        }

        if let Some(e_idx) = self.tree_min_pq00_local_for_duals(root) {
            if (e_idx as usize) < self.edge_num {
                eps = eps.min(self.edges[e_idx as usize].slack / 2);
            }
        }

        if let Some(e_idx) = self.tree_min_pq_blossom_for_duals(root) {
            if (e_idx as usize) < self.edge_num {
                eps = eps.min(self.edges[e_idx as usize].slack);
            }
        }

        eps
    }

    #[cfg(test)]
    fn compute_tree_local_eps_visible_scan(&self, root: u32) -> i64 {
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

    #[cfg(test)]
    fn fill_dual_pair_caps_from_scheduler(
        &mut self,
        roots: &[u32],
        root_to_var: &[usize],
        inf_cap: i64,
        pair_eps00: &mut Vec<i64>,
        pair_eps01: &mut Vec<i64>,
    ) {
        let roots_len = roots.len();
        let pair_slot = |u: usize, v: usize| u * roots_len + v;

        pair_eps00.clear();
        pair_eps01.clear();
        pair_eps00.resize(roots_len * roots_len, inf_cap);
        pair_eps01.resize(roots_len * roots_len, inf_cap);

        for pair_idx in 0..self.scheduler_tree_edges.len() {
            let root_left = self.scheduler_tree_edges[pair_idx].head[0];
            let root_right = self.scheduler_tree_edges[pair_idx].head[1];
            if root_left == NONE || root_right == NONE {
                continue;
            }
            if (root_left as usize) >= root_to_var.len()
                || (root_right as usize) >= root_to_var.len()
            {
                continue;
            }
            let var_left = root_to_var[root_left as usize];
            let var_right = root_to_var[root_right as usize];
            if var_left == usize::MAX || var_right == usize::MAX {
                continue;
            }

            let eps_left = self.tree_eps(root_left);
            let eps_right = self.tree_eps(root_right);

            if let Some(e_idx) =
                self.scheduler_tree_edge_min_pq00_edge_for_duals(pair_idx, root_left, root_right)
            {
                let eps00 = self.edges[e_idx as usize].slack - eps_left - eps_right;
                let lr = pair_slot(var_left, var_right);
                let rl = pair_slot(var_right, var_left);
                pair_eps00[lr] = pair_eps00[lr].min(eps00);
                pair_eps00[rl] = pair_eps00[rl].min(eps00);
            }

            if let Some(e_idx) =
                self.scheduler_tree_edge_min_pq01_edge_for_duals(pair_idx, 0, root_right, root_left)
            {
                let eps01 = self.edges[e_idx as usize].slack - eps_right + eps_left;
                let rl = pair_slot(var_right, var_left);
                pair_eps01[rl] = pair_eps01[rl].min(eps01);
            }

            if let Some(e_idx) =
                self.scheduler_tree_edge_min_pq01_edge_for_duals(pair_idx, 1, root_left, root_right)
            {
                let eps01 = self.edges[e_idx as usize].slack - eps_left + eps_right;
                let lr = pair_slot(var_left, var_right);
                pair_eps01[lr] = pair_eps01[lr].min(eps01);
            }
        }
    }

    #[cfg(test)]
    fn fill_dual_pair_caps_visible_scan(
        &self,
        roots: &[u32],
        root_to_var: &[usize],
        inf_cap: i64,
        pair_eps00: &mut Vec<i64>,
        pair_eps01: &mut Vec<i64>,
    ) {
        let roots_len = roots.len();
        let pair_slot = |u: usize, v: usize| u * roots_len + v;

        pair_eps00.clear();
        pair_eps01.clear();
        pair_eps00.resize(roots_len * roots_len, inf_cap);
        pair_eps01.resize(roots_len * roots_len, inf_cap);

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
                        && processed_v
                        && var_u != var_v =>
                {
                    let eps00 = slack - eps_u - eps_v;
                    let uv = pair_slot(var_u, var_v);
                    let vu = pair_slot(var_v, var_u);
                    pair_eps00[uv] = pair_eps00[uv].min(eps00);
                    pair_eps00[vu] = pair_eps00[vu].min(eps00);
                }
                (PLUS, MINUS)
                    if var_u != usize::MAX
                        && var_v != usize::MAX
                        && var_u != var_v
                        && processed_u =>
                {
                    let uv = pair_slot(var_u, var_v);
                    pair_eps01[uv] = pair_eps01[uv].min(slack - eps_u + eps_v);
                }
                (MINUS, PLUS)
                    if var_u != usize::MAX
                        && var_v != usize::MAX
                        && var_u != var_v
                        && processed_v =>
                {
                    let vu = pair_slot(var_v, var_u);
                    pair_eps01[vu] = pair_eps01[vu].min(slack - eps_v + eps_u);
                }
                _ => {}
            }
        }
    }

    #[inline]
    fn build_dual_root_index(&self, roots: &[u32], root_to_var: &mut Vec<usize>) {
        root_to_var.clear();
        root_to_var.resize(self.nodes.len(), usize::MAX);
        for (var, &root) in roots.iter().enumerate() {
            root_to_var[root as usize] = var;
        }
    }

    fn seed_dual_local_caps(&mut self, roots: &[u32], inf_cap: i64, local_caps: &mut Vec<i64>) {
        local_caps.clear();
        local_caps.resize(roots.len(), inf_cap);
        for (var, &root) in roots.iter().enumerate() {
            let local_abs_eps = self.compute_tree_local_eps(root);
            if local_abs_eps != i64::MAX {
                local_caps[var] = local_caps[var].min(local_abs_eps - self.tree_eps(root));
            }
        }
    }

    fn reset_dual_pair_tree_caps(
        &self,
        inf_cap: i64,
        pair_tree_eps00: &mut Vec<i64>,
        pair_tree_eps01_dir0: &mut Vec<i64>,
        pair_tree_eps01_dir1: &mut Vec<i64>,
        pair_tree_ready: &mut Vec<bool>,
    ) {
        pair_tree_eps00.clear();
        pair_tree_eps01_dir0.clear();
        pair_tree_eps01_dir1.clear();
        pair_tree_ready.clear();
        pair_tree_eps00.resize(self.scheduler_tree_edges.len(), inf_cap);
        pair_tree_eps01_dir0.resize(self.scheduler_tree_edges.len(), inf_cap);
        pair_tree_eps01_dir1.resize(self.scheduler_tree_edges.len(), inf_cap);
        pair_tree_ready.resize(self.scheduler_tree_edges.len(), false);
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
    fn collect_raw_incident_edges_into(&self, v: u32, out: &mut Vec<(u32, usize)>) {
        out.clear();
        for start_dir in 0..2usize {
            let first = self.nodes[v as usize].first[start_dir];
            if first == NONE {
                continue;
            }
            let mut e = first;
            let mut safety = self.edge_num + 1;
            loop {
                out.push((e, 1 - start_dir));
                e = self.edges[e as usize].next[start_dir];
                safety -= 1;
                if e == first || safety == 0 {
                    break;
                }
            }
        }
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
    fn take_edge_seen_scratch(&mut self) -> Vec<bool> {
        core::mem::take(&mut self.scratch.edge_seen)
    }

    #[inline]
    fn restore_edge_seen_scratch(&mut self, scratch: Vec<bool>) {
        self.scratch.edge_seen = scratch;
    }

    #[inline]
    fn take_node_seen_stamp_scratch(&mut self) -> Vec<u32> {
        core::mem::take(&mut self.scratch.node_seen_stamp)
    }

    #[inline]
    fn restore_node_seen_stamp_scratch(&mut self, scratch: Vec<u32>) {
        self.scratch.node_seen_stamp = scratch;
    }

    #[inline]
    fn next_node_seen_epoch(&mut self, stamps: &mut [u32]) -> u32 {
        self.scratch.node_seen_epoch = self.scratch.node_seen_epoch.wrapping_add(1);
        if self.scratch.node_seen_epoch == 0 {
            stamps.fill(0);
            self.scratch.node_seen_epoch = 1;
        }
        self.scratch.node_seen_epoch
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
    fn take_pair_marks_scratch(&mut self) -> Vec<u8> {
        core::mem::take(&mut self.scratch.pair_marks)
    }

    #[inline]
    fn restore_pair_marks_scratch(&mut self, scratch: Vec<u8>) {
        self.scratch.pair_marks = scratch;
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
    fn restore_members_mask_scratch(&mut self, scratch: Vec<bool>) {
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

    fn edge_queue_stamp(&self, e_idx: u32) -> u64 {
        self.edge_queue_stamp[e_idx as usize]
    }

    #[inline]
    fn set_edge_queue_owner(&mut self, e_idx: u32, owner: GenericQueueState) {
        self.edge_queue_owner[e_idx as usize] = owner;
    }

    #[inline]
    fn set_edge_queue_slot(&mut self, e_idx: u32, slot: usize) {
        self.edge_queue_slot[e_idx as usize] = slot;
    }

    #[inline]
    fn set_edge_queue_stamp(&mut self, e_idx: u32, stamp: u64) {
        self.edge_queue_stamp[e_idx as usize] = stamp;
    }

    fn fill_current_root_list(&self, roots: &mut Vec<u32>, seen: &mut Vec<bool>) {
        roots.clear();
        if seen.len() != self.nodes.len() {
            seen.resize(self.nodes.len(), false);
        }
        let mut current = self.root_list_head;
        while current != NONE && (current as usize) < self.nodes.len() && !seen[current as usize] {
            seen[current as usize] = true;
            roots.push(current);
            current = self.nodes[current as usize].tree_sibling_next;
        }
        for &root in roots.iter() {
            seen[root as usize] = false;
        }
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
        let mut root_members = if lca_is_tree_root {
            let mut members = self.take_tree_members_u_scratch();
            self.collect_tree_members_with_scratch(lca, &mut members);
            Some(members)
        } else {
            None
        };

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
        if in_cycle.len() != self.nodes.len() {
            in_cycle.resize(self.nodes.len(), false);
        }
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
        self.nodes[b as usize].blossom_child = cycle_set[0];
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
            if let Some(members) = root_members.as_ref() {
                for &node in members {
                    self.nodes[node as usize].tree_root = b;
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
        self.promote_boundary_edges_to_outer_blossom(b, &cycle_set);
        self.rebuild_generic_queue_membership_for_outer_blossom(b);
        for &node in &cycle_set {
            in_cycle[node as usize] = false;
        }
        if let Some(members) = root_members.take() {
            self.restore_tree_members_u_scratch(members);
        }
        self.restore_node_work_b_scratch(ext_children);
        self.restore_node_work_a_scratch(cycle_set);
        self.restore_members_mask_scratch(in_cycle);
    }

    fn find_blossom_root_raw(&mut self, a0: u32) -> u32 {
        let mut branch_nodes = [self.arc_head_raw(a0), self.arc_head_raw(arc_rev(a0))];
        let mut marked = self.take_node_seen_stamp_scratch();
        if marked.len() != self.nodes.len() {
            marked.resize(self.nodes.len(), 0);
        }
        let epoch = self.next_node_seen_epoch(marked.as_mut_slice());
        let mut branch = 0usize;

        let result = loop {
            let current = branch_nodes[branch];
            if current == NONE || (current as usize) >= self.nodes.len() {
                break NONE;
            }
            if marked[current as usize] == epoch {
                break current;
            }
            marked[current as usize] = epoch;

            if self.nodes[current as usize].is_tree_root {
                let mut other = branch_nodes[1 - branch];
                while other != NONE
                    && (other as usize) < self.nodes.len()
                    && marked[other as usize] != epoch
                {
                    marked[other as usize] = epoch;
                    let minus = self.arc_head_raw(self.nodes[other as usize].match_arc);
                    if minus == NONE || (minus as usize) >= self.nodes.len() {
                        other = NONE;
                        break;
                    }
                    other = self.arc_head_raw(self.nodes[minus as usize].tree_parent_arc);
                }
                break other;
            }

            let minus = self.arc_head_raw(self.nodes[current as usize].match_arc);
            if minus == NONE || (minus as usize) >= self.nodes.len() {
                break NONE;
            }
            branch_nodes[branch] = self.arc_head_raw(self.nodes[minus as usize].tree_parent_arc);
            branch = 1 - branch;
        };
        self.restore_node_seen_stamp_scratch(marked);
        result
    }

    fn promote_boundary_edges_to_outer_blossom(&mut self, blossom: u32, cycle_members: &[u32]) {
        if blossom == NONE
            || blossom as usize >= self.nodes.len()
            || !self.nodes[blossom as usize].is_outer
            || !self.nodes[blossom as usize].is_blossom
        {
            return;
        }

        let mut incident = self.take_incident_scratch();
        let mut edges = self.take_queue_edges_scratch();
        let mut edge_seen = self.take_edge_seen_scratch();
        if edge_seen.len() != self.edge_num {
            edge_seen.resize(self.edge_num, false);
        }
        for &node in cycle_members {
            if node == NONE || node as usize >= self.nodes.len() {
                continue;
            }
            self.collect_raw_incident_edges_into(node, &mut incident);
            for &(e_idx, _) in &incident {
                let idx = e_idx as usize;
                if idx < edge_seen.len() && !edge_seen[idx] {
                    edge_seen[idx] = true;
                    edges.push(e_idx);
                }
            }
        }

        for e_idx in edges.iter().copied() {
            let outer0 = self.edge_head_outer(e_idx, 0);
            let outer1 = self.edge_head_outer(e_idx, 1);

            if outer0 == blossom && outer1 != NONE && outer1 != blossom {
                let raw0 = self.edges[e_idx as usize].head[0];
                if raw0 != NONE && raw0 != blossom {
                    edge_list_remove(&mut self.nodes, &mut self.edges, raw0, e_idx, 1);
                    edge_list_add(&mut self.nodes, &mut self.edges, blossom, e_idx, 1);
                }
            }

            if outer1 == blossom && outer0 != NONE && outer0 != blossom {
                let raw1 = self.edges[e_idx as usize].head[1];
                if raw1 != NONE && raw1 != blossom {
                    edge_list_remove(&mut self.nodes, &mut self.edges, raw1, e_idx, 0);
                    edge_list_add(&mut self.nodes, &mut self.edges, blossom, e_idx, 0);
                }
            }
        }

        for &e_idx in &edges {
            edge_seen[e_idx as usize] = false;
        }
        self.restore_edge_seen_scratch(edge_seen);
        self.restore_queue_edges_scratch(edges);
        self.restore_incident_scratch(incident);
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

        let mut incident = self.take_incident_scratch();
        let mut edges = self.take_queue_edges_scratch();
        let mut edge_seen = self.take_edge_seen_scratch();
        if edge_seen.len() != self.edge_num {
            edge_seen.resize(self.edge_num, false);
        }
        self.collect_incident_edges_into(blossom, &mut incident);
        for &(e_idx, _) in &incident {
            let idx = e_idx as usize;
            if idx < edge_seen.len() && !edge_seen[idx] {
                edge_seen[idx] = true;
                edges.push(e_idx);
            }
        }

        for e_idx in edges.iter().copied() {
            let outer0 = self.edge_head_outer(e_idx, 0);
            let outer1 = self.edge_head_outer(e_idx, 1);
            if outer0 != blossom && outer1 != blossom {
                continue;
            }

            let old_state = self.edge_queue_owner(e_idx);
            self.remove_edge_from_generic_queue(e_idx);

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
                if (match_edge as usize) >= self.edge_num || match_edge != e_idx {
                    return None;
                }
                let cand_root = self.find_tree_root(cand);
                (cand_root != NONE).then_some(cand_root)
            });
            if let Some(pq_root) = pq_blossom_root {
                self.set_generic_pq_blossoms_root_slot(
                    e_idx,
                    pq_root,
                    !matches!(old_state, GenericQueueState::None),
                );
                continue;
            }

            if outer0 == NONE || outer1 == NONE {
                continue;
            }
            if outer0 == outer1 {
                if matches!(old_state, GenericQueueState::Pq00Local { .. })
                    && self.edges[e_idx as usize].slack > 0
                {
                    self.set_generic_pq00_local_slot(e_idx, root, true);
                }
                continue;
            }

            let other = if outer0 == blossom { outer1 } else { outer0 };
            match self.nodes[other as usize].flag {
                FREE => {
                    let preserve_stamp = matches!(old_state, GenericQueueState::Pq0 { root: old_root } if old_root == root);
                    self.set_generic_pq0_root_slot(e_idx, root, preserve_stamp);
                }
                PLUS => {
                    let other_root = self.find_tree_root(other);
                    if other_root != NONE {
                        self.set_generic_pq00(e_idx, root, other_root);
                    }
                }
                MINUS => {
                    let other_root = self.find_tree_root(other);
                    if other_root != NONE && other_root != root {
                        self.set_generic_pq01(e_idx, root, other_root);
                    }
                }
                _ => {}
            }
        }

        for &e_idx in &edges {
            edge_seen[e_idx as usize] = false;
        }
        self.restore_edge_seen_scratch(edge_seen);
        self.restore_queue_edges_scratch(edges);
        self.restore_incident_scratch(incident);
    }

    fn find_direct_blossom_child(&self, blossom: u32, mut node: u32) -> u32 {
        let mut limit = self.nodes.len();
        while node != NONE && (node as usize) < self.nodes.len() && limit > 0 {
            if self.nodes[node as usize].blossom_parent == blossom {
                return node;
            }
            let parent = self.nodes[node as usize].blossom_parent;
            if parent == NONE {
                return NONE;
            }
            node = parent;
            limit -= 1;
        }
        NONE
    }

    fn collect_expand_children_by_scan(&self, blossom: u32, children: &mut Vec<u32>) {
        children.clear();
        for i in 0..self.nodes.len() {
            if self.nodes[i].blossom_parent == blossom && !self.nodes[i].is_outer {
                children.push(i as u32);
            }
        }
    }

    fn collect_expand_children(
        &self,
        blossom: u32,
        b_match: u32,
        b_tp: u32,
        children: &mut Vec<u32>,
    ) {
        children.clear();

        let mut start = NONE;
        if b_match != NONE && (arc_edge(b_match) as usize) < self.edge_num {
            let md = arc_dir(b_match);
            let tail = self.edges[arc_edge(b_match) as usize].head0[md];
            start = self.find_direct_blossom_child(blossom, tail);
        }
        if start == NONE && b_tp != NONE && (arc_edge(b_tp) as usize) < self.edge_num {
            let pd = arc_dir(b_tp);
            let tail = self.edges[arc_edge(b_tp) as usize].head0[pd];
            start = self.find_direct_blossom_child(blossom, tail);
        }
        if start == NONE {
            let hint = self.nodes[blossom as usize].blossom_child;
            if hint != NONE
                && (hint as usize) < self.nodes.len()
                && self.nodes[hint as usize].blossom_parent == blossom
                && !self.nodes[hint as usize].is_outer
            {
                start = hint;
            }
        }

        if start == NONE {
            self.collect_expand_children_by_scan(blossom, children);
            return;
        }

        let mut current = start;
        let mut limit = self.nodes.len();
        loop {
            if current == NONE
                || (current as usize) >= self.nodes.len()
                || self.nodes[current as usize].blossom_parent != blossom
                || self.nodes[current as usize].is_outer
                || limit == 0
            {
                self.collect_expand_children_by_scan(blossom, children);
                return;
            }

            children.push(current);
            let sibling_arc = self.nodes[current as usize].blossom_sibling_arc;
            if sibling_arc == NONE {
                self.collect_expand_children_by_scan(blossom, children);
                return;
            }
            let next = self.arc_head_raw(sibling_arc);
            if next == start {
                return;
            }
            current = next;
            limit -= 1;
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

        let mut children = self.take_node_work_a_scratch();
        self.collect_expand_children(b, b_match, b_tp, &mut children);
        let mut child_mask = self.take_members_mask_scratch();
        if child_mask.len() != self.nodes.len() {
            child_mask.resize(self.nodes.len(), false);
        }
        for &child in &children {
            child_mask[child as usize] = true;
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
            while k != NONE && (k as usize) < self.nodes.len() && !child_mask[k as usize] {
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
            while tp != NONE && (tp as usize) < self.nodes.len() && !child_mask[tp as usize] {
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
            let mut incident = self.take_incident_scratch();
            let mut marked_edges = self.take_queue_edges_scratch();
            let mut edge_seen = self.take_edge_seen_scratch();
            if edge_seen.len() != self.edge_num {
                edge_seen.resize(self.edge_num, false);
            }
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

                incident.clear();
                marked_edges.clear();
                self.collect_incident_edges_into(minus, &mut incident);
                for &(e_idx, dir) in &incident {
                    edge_seen[e_idx as usize] = true;
                    marked_edges.push(e_idx);
                    let other = self.edge_head_outer(e_idx, dir);
                    if other != NONE && self.nodes[other as usize].flag != PLUS {
                        self.edges[e_idx as usize].slack -= eps;
                    }
                }
                incident.clear();
                self.collect_raw_incident_edges_into(minus, &mut incident);
                for &(e_idx, dir) in &incident {
                    if edge_seen[e_idx as usize] {
                        continue;
                    }
                    let other = self.edges[e_idx as usize].head[1 - dir];
                    if other != NONE && self.nodes[other as usize].flag != PLUS {
                        self.edges[e_idx as usize].slack -= eps;
                    }
                }
                for &e_idx in &marked_edges {
                    edge_seen[e_idx as usize] = false;
                }
                self.nodes[minus as usize].is_processed = true;

                if self.nodes[minus as usize].tree_parent_arc == b_tp {
                    break;
                }

                let plus = self.arc_head_raw(self.nodes[minus as usize].tree_parent_arc);
                incident.clear();
                marked_edges.clear();
                self.collect_incident_edges_into(plus, &mut incident);
                for &(e_idx, dir) in &incident {
                    edge_seen[e_idx as usize] = true;
                    marked_edges.push(e_idx);
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
                incident.clear();
                self.collect_raw_incident_edges_into(plus, &mut incident);
                for &(e_idx, dir) in &incident {
                    if edge_seen[e_idx as usize] {
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
                for &e_idx in &marked_edges {
                    edge_seen[e_idx as usize] = false;
                }
                self.nodes[plus as usize].is_processed = true;

                if self.nodes[plus as usize].match_arc == NONE {
                    break;
                }
                minus = self.arc_head_raw(self.nodes[plus as usize].match_arc);
            }
            self.restore_edge_seen_scratch(edge_seen);
            self.restore_queue_edges_scratch(marked_edges);
            self.restore_incident_scratch(incident);
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
        self.nodes[b as usize].blossom_child = NONE;
        self.nodes[b as usize].first = [NONE; 2];

        // Expanding the blossom makes processed PLUS children expose outer
        // boundary edges again. Requeue those now that ownership is visible.
        for &child in &children {
            self.requeue_edges_after_expand(child);
        }

        for &child in &children {
            child_mask[child as usize] = false;
        }
        self.restore_members_mask_scratch(child_mask);
        self.restore_edge_moves_scratch(moves);
        self.restore_node_work_a_scratch(children);
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

        let mut used = self.take_members_mask_scratch();
        if used.len() != self.node_num {
            used.resize(self.node_num, false);
        } else {
            used.fill(false);
        }
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
mod tests;
