//! Internal state and algorithm for the Kocay-Stone Balanced Network Search.
//!
//! The algorithm constructs a balanced network N from graph G = (V, E):
//! - Virtual vertices: source `s=0`, sink `t=1`, plus `x_i = 2(i+1)` and `y_i =
//!   2(i+1)+1` for each vertex i.
//! - Complement: `prim(v) = v ^ 1`.
//! - Edges: `s→x_i` (cap = budget[i]), `y_i→t` (cap = budget[i]), and for each
//!   {i,j} ∈ E the pair `x_i→y_j`, `x_j→y_i` (cap = edge capacity, with
//!   balanced flow condition: flow on `x_i→y_j` equals flow on `x_j→y_i`).
//!
//! A balanced flow of value k corresponds to k units of total flow. Each BNS
//! pass: BFS from s, build tree T/T', contract blossoms, find augmenting path,
//! compute delta via `find_path_cap`, augment by delta via `pull_flow`.
//! Repeat until no augmenting path exists.
//!
//! All recursion from the reference C implementation (InChI `ichi_bns.c`) has
//! been converted to iterative form.
use alloc::vec::Vec;

use num_traits::AsPrimitive;

use crate::traits::{PositiveInteger, SparseValuedMatrix2D};

// ── Virtual vertex layout ────────────────────────────────────────────────────
const VTX_S: usize = 0;
const VTX_T: usize = 1;

#[inline]
fn prim(v: usize) -> usize {
    v ^ 1
}

#[inline]
fn virt_x(i: usize) -> usize {
    2 * (i + 1)
}

#[inline]
fn virt_y(i: usize) -> usize {
    2 * (i + 1) + 1
}

#[inline]
fn orig(v: usize) -> usize {
    (v - 2) / 2
}

// ── Union-find sentinels ─────────────────────────────────────────────────────
const NO_VERTEX: usize = usize::MAX;
const BLOSSOM_BASE: usize = usize::MAX - 1;

// ── Tree membership flags ────────────────────────────────────────────────────
const TREE_NOT_IN: u8 = 0;
const TREE_IN_T: u8 = 1;
const TREE_IN_TP: u8 = 2;
const TREE_IN_BLOSSOM: u8 = 3;

#[inline]
fn is_s_reachable(t: u8) -> bool {
    t == TREE_IN_T || t == TREE_IN_BLOSSOM
}

// ── Edge index sentinel ─────────────────────────────────────────────────────
const NO_EDGE: usize = usize::MAX;

// ── Switch-edge record ───────────────────────────────────────────────────────

/// Records the balanced-network edge that made a virtual vertex s-reachable.
/// `vert1` is the endpoint closer to s; `vert2` is the other endpoint.
/// `edge_idx` is the undirected edge index (or `NO_EDGE` for s-t edges).
#[derive(Clone, Copy)]
struct SwitchEdge {
    vert1: usize,
    vert2: usize,
    edge_idx: usize,
}

impl SwitchEdge {
    const NONE: Self = Self { vert1: NO_VERTEX, vert2: NO_VERTEX, edge_idx: NO_EDGE };
}

// ── KocayState ───────────────────────────────────────────────────────────────

/// State for a full run of the Kocay-Stone BNS balanced flow algorithm.
pub(super) struct KocayState<'a, M: SparseValuedMatrix2D + ?Sized> {
    _marker: core::marker::PhantomData<&'a M>,
    /// Number of original vertices.
    n: usize,

    /// Precomputed adjacency: `adj[i]` = list of `(j, edge_idx)`.
    adj: Vec<Vec<(usize, usize)>>,

    /// Edge capacity per undirected edge.
    edge_cap: Vec<usize>,
    /// Edge flow per undirected edge.
    edge_flow: Vec<usize>,

    /// Vertex-to-source/sink capacity (budget) per original vertex.
    st_cap: Vec<usize>,
    /// Vertex-to-source/sink flow per original vertex.
    st_flow: Vec<usize>,

    /// Path marking for `find_path_cap`.
    edge_on_path: Vec<bool>,
    st_on_path: Vec<bool>,

    /// Union-find for blossom bases, indexed by virtual vertex.
    base_ptr: Vec<usize>,
    /// Switch-edge per virtual vertex — the edge that made it s-reachable.
    switch_edge: Vec<SwitchEdge>,
    /// Tree membership per virtual vertex.
    tree: Vec<u8>,
    /// BFS queue of s-reachable virtual vertices.
    scan_q: Vec<usize>,

    /// Scratch buffer for blossom-base path computation (Pu).
    path_u: Vec<usize>,
    /// Scratch buffer for blossom-base path computation (Pv).
    path_v: Vec<usize>,
}

impl<'a, M: SparseValuedMatrix2D + ?Sized> KocayState<'a, M>
where
    M::Value: PositiveInteger,
    M::RowIndex: PositiveInteger,
    M::ColumnIndex: PositiveInteger,
{
    pub(super) fn new(matrix: &'a M, vertex_budgets: &[M::Value]) -> Self {
        let n_rows: usize = matrix.number_of_rows().as_();
        let n = n_rows;
        let nv = 2 * n + 2;

        // Build adjacency list from the upper triangle of the symmetric matrix.
        let mut adj: Vec<Vec<(usize, usize)>> = vec![Vec::new(); n];
        let mut edge_cap: Vec<usize> = Vec::new();

        for i in matrix.row_indices() {
            let iu: usize = i.as_();
            for (col, val) in matrix.sparse_row(i).zip(matrix.sparse_row_values(i)) {
                let ju: usize = col.as_();
                if ju <= iu {
                    continue; // only upper triangle, skip self-loops
                }
                let cap: usize = val.as_();
                if cap == 0 {
                    continue;
                }
                let eidx = edge_cap.len();
                adj[iu].push((ju, eidx));
                adj[ju].push((iu, eidx));
                edge_cap.push(cap);
            }
        }

        let num_edges = edge_cap.len();

        let st_cap: Vec<usize> = vertex_budgets.iter().map(|b| (*b).as_()).collect();

        Self {
            _marker: core::marker::PhantomData,
            n,
            adj,
            edge_cap,
            edge_flow: vec![0; num_edges],
            st_cap,
            st_flow: vec![0; n],
            edge_on_path: vec![false; num_edges],
            st_on_path: vec![false; n],
            base_ptr: vec![NO_VERTEX; nv],
            switch_edge: vec![SwitchEdge::NONE; nv],
            tree: vec![TREE_NOT_IN; nv],
            scan_q: Vec::with_capacity(nv),
            path_u: Vec::with_capacity(nv),
            path_v: Vec::with_capacity(nv),
        }
    }

    /// Runs the algorithm and returns the flow as sorted triples.
    pub(super) fn solve(mut self) -> Vec<(M::RowIndex, M::ColumnIndex, M::Value)> {
        if self.n == 0 {
            return Vec::new();
        }
        loop {
            if !self.balanced_network_search() {
                break;
            }
            self.reinit();
        }
        self.into_flow_triples()
    }

    /// Converts the edge flow into `(row, col, flow)` triples with `row < col`
    /// and `flow > 0`.
    fn into_flow_triples(self) -> Vec<(M::RowIndex, M::ColumnIndex, M::Value)> {
        let mut triples = Vec::new();
        for (i, neighbors) in self.adj.iter().enumerate() {
            for &(j, eidx) in neighbors {
                if j <= i {
                    continue; // emit each edge once, i < j
                }
                let flow = self.edge_flow[eidx];
                if flow > 0 {
                    let i_idx = M::RowIndex::try_from(i).ok().expect("valid vertex index");
                    let j_idx = M::ColumnIndex::try_from(j).ok().expect("valid vertex index");
                    let f_val = M::Value::try_from(flow).ok().expect("valid flow value");
                    triples.push((i_idx, j_idx, f_val));
                }
            }
        }
        triples
    }

    /// Resets per-search state, touching only vertices that were on `scan_q`.
    fn reinit(&mut self) {
        for idx in 0..self.scan_q.len() {
            let u = self.scan_q[idx];
            let v = prim(u);
            self.base_ptr[u] = NO_VERTEX;
            self.base_ptr[v] = NO_VERTEX;
            self.switch_edge[u] = SwitchEdge::NONE;
            self.switch_edge[v] = SwitchEdge::NONE;
            self.tree[u] = TREE_NOT_IN;
            self.tree[v] = TREE_NOT_IN;
        }
        self.scan_q.clear();
    }

    /// Iterative `FindBase` with path compression.
    fn find_base(&mut self, u: usize) -> usize {
        if self.base_ptr[u] == NO_VERTEX {
            return NO_VERTEX;
        }
        // Find root.
        let mut root = u;
        while self.base_ptr[root] != BLOSSOM_BASE {
            root = self.base_ptr[root];
        }
        // Path compression.
        let mut cur = u;
        while self.base_ptr[cur] != BLOSSOM_BASE {
            let next = self.base_ptr[cur];
            self.base_ptr[cur] = root;
            cur = next;
        }
        root
    }

    /// Residual capacity of the edge from `u` to `v` with edge index `eidx`.
    fn rescap(&self, u: usize, v: usize, eidx: usize) -> usize {
        if eidx == NO_EDGE {
            // ST edge: u or v is s or t.
            if u == VTX_S {
                // s → x_i: forward = st_cap[i] - st_flow[i]
                let i = orig(v);
                self.st_cap[i] - self.st_flow[i]
            } else if v == VTX_S {
                // x_i → s: backward = st_flow[i]
                let i = orig(u);
                self.st_flow[i]
            } else if v == VTX_T {
                // y_i → t: forward = st_cap[i] - st_flow[i]
                let i = orig(u);
                self.st_cap[i] - self.st_flow[i]
            } else {
                // t → y_i: backward = st_flow[i]
                assert_eq!(u, VTX_T);
                let i = orig(v);
                self.st_flow[i]
            }
        } else if u >= 2 && u % 2 == 0 {
            // u is x_i (even): forward edge x_i → y_j
            self.edge_cap[eidx] - self.edge_flow[eidx]
        } else {
            // u is y_i (odd): backward edge y_i → x_j, rescap = flow
            self.edge_flow[eidx]
        }
    }

    /// Residual capacity with path marking for `find_path_cap`.
    /// If the edge is already marked (on both P and P'), returns rescap/2
    /// because augmenting both the path and its complement doubles the
    /// change to the shared flow variable.
    fn rescap_mark(&mut self, u: usize, v: usize, eidx: usize) -> usize {
        if eidx == NO_EDGE {
            let i = if u == VTX_S || v == VTX_T {
                // s→x_i or y_i→t: the non-s/t vertex tells us i
                if u == VTX_S { orig(v) } else { orig(u) }
            } else {
                // x_i→s or t→y_i: the non-s/t vertex tells us i
                if v == VTX_S { orig(u) } else { orig(v) }
            };
            if self.st_on_path[i] {
                self.rescap(u, v, eidx) / 2
            } else {
                self.st_on_path[i] = true;
                self.rescap(u, v, eidx)
            }
        } else if self.edge_on_path[eidx] {
            self.rescap(u, v, eidx) / 2
        } else {
            self.edge_on_path[eidx] = true;
            self.rescap(u, v, eidx)
        }
    }

    /// Find the edge index for the undirected edge between original vertices
    /// `i` and `j`.
    fn find_edge_idx(&self, i: usize, j: usize) -> usize {
        for &(neighbor, eidx) in &self.adj[i] {
            if neighbor == j {
                return eidx;
            }
        }
        NO_EDGE
    }

    /// Compute the edge index for a pair of virtual vertices that represent an
    /// atom-atom edge.
    fn edge_idx_for_virtual(&self, u: usize, v: usize) -> usize {
        if u <= 1 || v <= 1 {
            return NO_EDGE;
        }
        let i = orig(u);
        let j = orig(v);
        self.find_edge_idx(i, j)
    }

    /// Clear path marks.
    fn clear_path_marks(&mut self) {
        for m in &mut self.edge_on_path {
            *m = false;
        }
        for m in &mut self.st_on_path {
            *m = false;
        }
    }

    /// Iterative `FindPathCap`: traverse the augmenting path from x to y using
    /// switch-edges, marking edges and computing the minimum residual capacity
    /// (delta).
    fn find_path_cap(&mut self, x: usize, y: usize) -> usize {
        let mut delta = usize::MAX;
        let mut stack: Vec<(usize, usize)> = Vec::new();
        stack.push((x, y));

        while let Some((x, y)) = stack.pop() {
            let se = self.switch_edge[y];
            let w = se.vert1;
            let z = se.vert2;
            let eidx = se.edge_idx;

            let cap = self.rescap_mark(w, z, eidx);
            delta = delta.min(cap);

            if w != x {
                stack.push((x, w));
            }
            if z != y {
                stack.push((prim(y), prim(z)));
            }
        }
        delta
    }

    /// Augment a single edge by `delta`.
    ///
    /// The direction of augmentation is determined solely by the edge type
    /// (forward vs backward), not by the `reverse` flag used for path
    /// decomposition. In the balanced network, s→x_i and y_i→t both map to
    /// the same st_flow[i], and the complement mapping preserves this.
    fn augment_edge(&mut self, u: usize, v: usize, eidx: usize, delta: usize) {
        if eidx == NO_EDGE {
            // ST edge: the complement of s→x_i is t→y_i, both correspond to
            // increasing st_flow[i].
            let is_forward = (u == VTX_S) || (v == VTX_T);
            let i = if u == VTX_S || v == VTX_T {
                if u == VTX_S { orig(v) } else { orig(u) }
            } else {
                if v == VTX_S { orig(u) } else { orig(v) }
            };
            if is_forward {
                self.st_flow[i] += delta;
            } else {
                self.st_flow[i] -= delta;
            }
        } else {
            // Atom-atom edge: direction determined by parity of u.
            // u even (x_i) → forward: edge_flow += delta
            // u odd (y_i) → backward: edge_flow -= delta
            if u >= 2 && u % 2 == 0 {
                self.edge_flow[eidx] += delta;
            } else {
                self.edge_flow[eidx] -= delta;
            }
        }
    }

    /// Iterative `PullFlow`: augment the flow along the augmenting path from s
    /// to t by `delta`.
    fn pull_flow(&mut self, delta: usize) {
        let mut stack: Vec<(usize, usize, bool)> = Vec::new();
        stack.push((VTX_S, VTX_T, false));

        while let Some((x, y, reverse)) = stack.pop() {
            let se = self.switch_edge[y];
            let w = se.vert1;
            let z = se.vert2;
            let eidx = se.edge_idx;

            self.augment_edge(w, z, eidx, delta);

            if reverse {
                if w != x {
                    stack.push((x, w, true));
                }
                if z != y {
                    stack.push((prim(y), prim(z), false));
                }
            } else {
                if z != y {
                    stack.push((prim(y), prim(z), true));
                }
                if w != x {
                    stack.push((x, w, false));
                }
            }
        }
    }

    /// Single BFS pass of the Balanced Network Search. Returns `true` if an
    /// augmenting path was found and the flow was increased.
    fn balanced_network_search(&mut self) -> bool {
        // Initialise: put s on the queue, create the initial {s, t} blossom.
        self.scan_q.clear();
        self.scan_q.push(VTX_S);
        self.base_ptr[VTX_T] = VTX_S;
        self.base_ptr[VTX_S] = BLOSSOM_BASE;
        self.tree[VTX_S] = TREE_IN_T;

        let mut k = 0;
        while k < self.scan_q.len() {
            let u = self.scan_q[k];
            let mut b_u = self.find_base(u);

            if u == VTX_S {
                // Neighbors of s: x_i for each vertex i with st residual > 0.
                for i in 0..self.n {
                    if self.st_cap[i] > self.st_flow[i] {
                        let v = virt_x(i);
                        if self.process_neighbor(u, v, NO_EDGE, &mut b_u) {
                            return true;
                        }
                    }
                }
            } else if u >= 2 && u % 2 == 0 {
                // u = x_i (even virtual vertex).
                let i = orig(u);

                // Neighbor s (backward edge x_i → s, positive rescap iff
                // st_flow[i] > 0).
                if self.st_flow[i] > 0 && self.process_neighbor(u, VTX_S, NO_EDGE, &mut b_u) {
                    return true;
                }

                // Neighbors y_j for each j adjacent to i where forward rescap > 0.
                // We need to snapshot adjacency to avoid borrow issues.
                let adj_i = self.adj[i].clone();
                for &(j, eidx) in &adj_i {
                    if self.edge_cap[eidx] > self.edge_flow[eidx] {
                        let v = virt_y(j);
                        if self.process_neighbor(u, v, eidx, &mut b_u) {
                            return true;
                        }
                    }
                }
            } else if u >= 2 {
                // u = y_i (odd virtual vertex).
                let i = orig(u);

                // Neighbor t (forward edge y_i → t, positive rescap iff
                // st_cap[i] > st_flow[i]).
                if self.st_cap[i] > self.st_flow[i]
                    && self.process_neighbor(u, VTX_T, NO_EDGE, &mut b_u)
                {
                    return true;
                }

                // Neighbors x_j for each j adjacent to i where backward rescap > 0
                // (flow on x_j → y_i > 0).
                let adj_i = self.adj[i].clone();
                for &(j, eidx) in &adj_i {
                    if self.edge_flow[eidx] > 0 {
                        let v = virt_x(j);
                        if self.process_neighbor(u, v, eidx, &mut b_u) {
                            return true;
                        }
                    }
                }
            }
            // u == VTX_T is not normally processed.
            k += 1;
        }
        false
    }

    /// Process a single neighbor `v` of `u` in the balanced network.
    /// Returns `true` if an augmenting path was found and augmented.
    fn process_neighbor(&mut self, u: usize, v: usize, eidx: usize, b_u: &mut usize) -> bool {
        // Avoid the tree edge of u.
        if self.switch_edge[u].vert1 == v
            && self.switch_edge[u].vert2 == u
            && self.switch_edge[u].edge_idx == eidx
        {
            return false;
        }

        let b_v = self.find_base(v);

        if b_v == NO_VERTEX {
            // v not yet in the mirror network M — add v to T, prim(v) to T'.
            self.scan_q.push(v);
            self.tree[v] = TREE_IN_T;
            self.tree[prim(v)] = TREE_IN_TP;
            self.switch_edge[v] = SwitchEdge { vert1: u, vert2: v, edge_idx: eidx };
            self.base_ptr[prim(v)] = v;
            self.base_ptr[v] = BLOSSOM_BASE;

            return false;
        }

        if is_s_reachable(self.tree[prim(v)]) && *b_u != b_v {
            // Avoid complement tree edge.
            let pu = prim(u);
            let pv = prim(v);

            // Compute complement edge index
            let comp_eidx = self.edge_idx_for_virtual(pu, pv);

            if self.switch_edge[pu].vert1 == pv
                && self.switch_edge[pu].vert2 == pu
                && self.switch_edge[pu].edge_idx == comp_eidx
            {
                return false;
            }

            let w = self.make_blossom(u, v, eidx, *b_u, b_v);
            *b_u = w;

            if prim(w) == VTX_T {
                // t is now s-reachable — augment the flow.
                self.clear_path_marks();
                let delta = self.find_path_cap(VTX_S, VTX_T);
                if delta > 0 {
                    self.pull_flow(delta);
                    return true;
                }
            }
        }

        false
    }

    /// Build a path of blossom bases from `x` to `s`, storing into `path`.
    fn find_path_to_s(&mut self, mut x: usize, path: &mut Vec<usize>) {
        path.clear();
        path.push(x);
        while x != VTX_S {
            x = self.find_base(self.switch_edge[x].vert1);
            path.push(x);
        }
    }

    /// Contract a blossom joining the blossoms of `u` and `v` (with bases
    /// `b_u` and `b_v`). Returns the base of the new blossom.
    fn make_blossom(&mut self, u: usize, v: usize, iuv: usize, b_u: usize, b_v: usize) -> usize {
        // Take scratch paths out to avoid borrow conflicts with find_base.
        let mut pu = core::mem::take(&mut self.path_u);
        let mut pv = core::mem::take(&mut self.path_v);

        self.find_path_to_s(b_u, &mut pu);
        self.find_path_to_s(b_v, &mut pv);

        // Find the LCA: compare the two paths from the end (both end at s).
        let mut match_count = 0;
        let min_len = pu.len().min(pv.len());
        while match_count < min_len
            && pu[pu.len() - 1 - match_count] == pv[pv.len() - 1 - match_count]
        {
            match_count += 1;
        }

        let lca_idx = pu.len() - match_count;
        let mut w = pu[lca_idx];

        // Blossom extension: while rescap(switch_edge[w]) >= 2, extend w
        // upward along the path.
        while w != VTX_S {
            let se = self.switch_edge[w];
            if se.vert1 == NO_VERTEX {
                break;
            }
            let rc = self.rescap(se.vert1, se.vert2, se.edge_idx);
            if rc < 2 {
                break;
            }
            let parent_base = self.find_base(se.vert1);
            if parent_base == NO_VERTEX || parent_base == w {
                break;
            }
            w = parent_base;
        }

        // Contract along the Pu path.
        for k in (0..lca_idx).rev() {
            let z = pu[k];
            self.base_ptr[z] = w;
            self.base_ptr[prim(z)] = w;

            let pz = prim(z);
            if !is_s_reachable(self.tree[pz]) {
                // Compute complement edge index for the switch edge
                let comp_eidx = self.edge_idx_for_virtual(prim(v), prim(u));
                self.switch_edge[pz] =
                    SwitchEdge { vert1: prim(v), vert2: prim(u), edge_idx: comp_eidx };
                self.scan_q.push(pz);
                self.tree[pz] = TREE_IN_BLOSSOM;
            }
        }

        // Contract along the Pv path.
        let pv_end = pv.len() - match_count;
        for k in (0..pv_end).rev() {
            let z = pv[k];
            self.base_ptr[z] = w;
            self.base_ptr[prim(z)] = w;

            let pz = prim(z);
            if !is_s_reachable(self.tree[pz]) {
                self.switch_edge[pz] = SwitchEdge { vert1: u, vert2: v, edge_idx: iuv };
                self.scan_q.push(pz);
                self.tree[pz] = TREE_IN_BLOSSOM;
            }
        }

        // Handle prim(w) — the new blossom base's complement.
        let pw = prim(w);
        if !is_s_reachable(self.tree[pw]) {
            self.switch_edge[pw] = SwitchEdge { vert1: u, vert2: v, edge_idx: iuv };
            self.scan_q.push(pw);
            self.tree[pw] = TREE_IN_BLOSSOM;
        }

        // Put scratch paths back for reuse.
        self.path_u = pu;
        self.path_v = pv;

        w
    }
}
