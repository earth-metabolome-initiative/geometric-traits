//! Internal state and algorithm for Blum's maximum matching.
//!
//! Reference: Norbert Blum, "Maximum Matching in General Graphs Without
//! Explicit Consideration of Blossoms Revisited" (arXiv:1509.04927, 2015),
//! with corrections from Dandeh & Lukovszki (ICTCS 2025).
//!
//! The algorithm reduces finding an augmenting path in a general graph G
//! to finding a *strongly simple* s→t path in a directed bipartite graph
//! G_M, using a Modified Depth-First Search (MDFS).
//!
//! The phased approach (Hopcroft-Karp style, Section 3) builds G_M once
//! per phase, runs MBFS (Modified BFS) to compute shortest *strongly
//! simple* distances, then runs MDFS on G_M restricted to the layered
//! subgraph to find multiple vertex-disjoint augmenting paths.  MBFS
//! uses a two-part structure: Part 1 assigns first levels via standard
//! BFS, Part 2 processes bridge pairs via back-path search with
//! union-find to assign second levels.  A single-path MDFS fallback
//! on the full G_M is kept as a safety net.
use alloc::{collections::VecDeque, vec::Vec};

use num_traits::AsPrimitive;

use crate::traits::SparseSquareMatrix;

const fn a_side(v: usize) -> usize {
    2 * v
}
const fn b_side(v: usize) -> usize {
    2 * v + 1
}
const fn orig(gm: usize) -> usize {
    gm / 2
}
const fn twin(gm: usize) -> usize {
    gm ^ 1
}
const fn is_a(gm: usize) -> bool {
    gm & 1 == 0
}

const INF: usize = usize::MAX;

// ── Public entry ────────────────────────────────────────────────────────

pub(super) struct BlumState<'a, M: SparseSquareMatrix + ?Sized> {
    matrix: &'a M,
    n: usize,
    mate: Vec<Option<usize>>,
    // Reusable buffers (allocated once, cleared each phase):
    adj: Vec<Vec<usize>>,
    mbfs_level: Vec<usize>,
    mbfs_level1: Vec<usize>,
    mbfs_par: Vec<usize>,
    mbfs_uf: Vec<usize>,
    mbfs_queue: VecDeque<usize>,
}

impl<'a, M: SparseSquareMatrix + ?Sized> BlumState<'a, M> {
    pub(super) fn new(matrix: &'a M) -> Self {
        let n: usize = matrix.order().as_();
        let sz = 2 * n + 2;
        Self {
            matrix,
            n,
            mate: vec![None; n],
            adj: vec![Vec::new(); sz],
            mbfs_level: vec![INF; sz],
            mbfs_level1: vec![INF; n],
            mbfs_par: vec![usize::MAX; sz],
            mbfs_uf: (0..sz).collect(),
            mbfs_queue: VecDeque::new(),
        }
    }

    pub(super) fn solve(mut self) -> Vec<(M::Index, M::Index)> {
        if self.n == 0 {
            return Vec::new();
        }

        let sz = 2 * self.n + 2;

        for _ in 0..self.n {
            let (s, t) = self.fill_gm();

            // MBFS on G_M to compute shortest strongly simple distances.
            mbfs(
                &self.adj,
                s,
                t,
                self.n,
                &mut self.mbfs_level,
                &mut self.mbfs_level1,
                &mut self.mbfs_par,
                &mut self.mbfs_uf,
                &mut self.mbfs_queue,
            );
            if self.mbfs_level[t] == INF {
                break;
            }

            // Take ownership of adj and level for MDFS.
            let adj = core::mem::take(&mut self.adj);
            let level = core::mem::take(&mut self.mbfs_level);

            let mut mdfs = Mdfs::new_layered(sz, s, t, self.n, adj, level);
            let found = mdfs.run_multi_path(&mut self.mate);

            // Recover buffers for next phase.
            self.adj = mdfs.take_adj();
            self.mbfs_level = mdfs.take_level();

            if found == 0 {
                // Fallback: BFS dist < strongly-simple dist.
                let (s2, t2) = self.fill_gm();
                let adj2 = core::mem::take(&mut self.adj);
                let mut mdfs_full = Mdfs::new(sz, s2, t2, self.n, adj2);
                let ok = mdfs_full.run(&mut self.mate);
                self.adj = mdfs_full.take_adj();
                if !ok {
                    break;
                }
            }
        }

        let indices: Vec<M::Index> = self.matrix.row_indices().collect();
        crate::traits::algorithms::matching_utils::mate_to_pairs(&self.mate, &indices)
    }

    /// Fill `self.adj` with G_M edges for the current matching.
    /// Clears existing adj lists rather than reallocating.
    fn fill_gm(&mut self) -> (usize, usize) {
        let n = self.n;
        let s = 2 * n;
        let t = s + 1;

        for list in &mut self.adj {
            list.clear();
        }

        for u in self.matrix.row_indices() {
            let ui: usize = u.as_();
            if self.mate[ui].is_none() {
                self.adj[s].push(b_side(ui));
                self.adj[a_side(ui)].push(t);
            }
            for v in self.matrix.sparse_row(u) {
                let vi: usize = v.as_();
                if vi <= ui {
                    continue;
                }
                if self.mate[ui] == Some(vi) {
                    self.adj[a_side(ui)].push(b_side(vi));
                    self.adj[a_side(vi)].push(b_side(ui));
                } else {
                    self.adj[b_side(ui)].push(a_side(vi));
                    self.adj[b_side(vi)].push(a_side(ui));
                }
            }
        }
        (s, t)
    }
}

// ── MBFS (Modified Breadth-First Search) ────────────────────────────────
//
// Computes shortest *strongly simple* distances from s in G_M.
//
// Part 1: standard BFS assigning first levels (level₁).
// Part 2: processes bridge pairs E(k) in order via back-path search.
//
// Reference: Blum 2015, Section 3 (full version, Uni Bonn, July 2016).
//
// Union-find choice: standard path-halving (not Gabow-Tarjan).
// See <https://github.com/LucaCappelletti94/incremental-tree-set-union>
// for benchmarks showing Gabow-Tarjan is 6-10× slower in practice.

#[allow(clippy::too_many_arguments)]
fn mbfs(
    adj: &[Vec<usize>],
    s: usize,
    t: usize,
    n: usize,
    level: &mut [usize],
    level1: &mut [usize],
    par: &mut [usize],
    uf: &mut [usize],
    queue: &mut VecDeque<usize>,
) {
    let n2 = 2 * n;

    // Reset buffers.
    for entry in level.iter_mut() {
        *entry = INF;
    }
    for entry in level1.iter_mut() {
        *entry = INF;
    }
    for entry in par.iter_mut() {
        *entry = usize::MAX;
    }
    for (index, entry) in uf.iter_mut().enumerate() {
        *entry = index;
    }
    queue.clear();

    // Flat bridge storage: (x, y, k).  Sorted by k before Part 2.
    let mut bridges: Vec<(usize, usize, usize)> = Vec::new();

    // ── Part 1: BFS for first levels ────────────────────────────────────

    level[s] = 0;
    for &w in &adj[s] {
        if w < n2 && level[w] == INF {
            level[w] = 1;
            par[w] = s;
            let v = orig(w);
            if v < n && level1[v] == INF {
                level1[v] = 1;
            }
            queue.push_back(w);
        }
    }

    while let Some(u) = queue.pop_front() {
        mbfs_scan(u, adj, level, level1, par, queue, &mut bridges, n, n2, t, true);
    }

    // ── Generate A-side bridge pairs from matched edges ─────────────────

    for u in 0..n {
        let u_a = a_side(u);
        let lu = level[u_a];
        if lu == INF {
            continue;
        }
        for &w_b in &adj[u_a] {
            if w_b >= n2 {
                continue;
            }
            let w = orig(w_b);
            if w <= u {
                continue;
            }
            let w_a = a_side(w);
            let lw = level[w_a];
            if lw != INF {
                bridges.push((u_a, w_a, usize::midpoint(lu, lw)));
            }
        }
    }

    // ── Part 2: process bridges in order of k ───────────────────────────
    //
    // Bridges are sorted by k before processing.  New bridges appended
    // during Part 2 (by mbfs_scan) are at the tail and processed in
    // subsequent iterations.  In rare pathological cases a new bridge
    // could have k < current position, causing a suboptimal (but not
    // incorrect) level assignment — the MDFS fallback handles this.

    bridges.sort_unstable_by_key(|&(_, _, k)| k);

    let mut bi = 0;
    while bi < bridges.len() {
        let (x_z, y_z, _k) = bridges[bi];
        bi += 1;

        let lx = level[x_z];
        let ly = level[y_z];
        if lx == INF || ly == INF {
            continue;
        }
        let sum = lx + ly;

        let mut px = uf_find(uf, x_z);
        let mut py = uf_find(uf, y_z);

        while px != py {
            if level[px] == INF || level[py] == INF {
                break;
            }

            let adv = if level[px] >= level[py] { px } else { py };

            if adv < n2 {
                let tw = twin(adv);
                if level[tw] == INF {
                    let new_lev = sum + 1 - level[adv];
                    level[tw] = new_lev;
                    par[tw] = adv;

                    mbfs_scan(tw, adj, level, level1, par, queue, &mut bridges, n, n2, t, false);
                    while let Some(u) = queue.pop_front() {
                        mbfs_scan(u, adj, level, level1, par, queue, &mut bridges, n, n2, t, false);
                    }
                }
            }

            let p = par[adv];
            if p != usize::MAX && p != adv {
                uf[adv] = p;
            } else {
                break;
            }

            if level[px] >= level[py] {
                px = uf_find(uf, px);
            } else {
                py = uf_find(uf, py);
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn mbfs_scan(
    gm_vertex: usize,
    adj: &[Vec<usize>],
    level: &mut [usize],
    level1: &mut [usize],
    par: &mut [usize],
    queue: &mut VecDeque<usize>,
    bridges: &mut Vec<(usize, usize, usize)>,
    vertex_count: usize,
    gm_limit: usize,
    sink: usize,
    first_level_only: bool,
) {
    let vertex_level = level[gm_vertex];
    if vertex_level == INF || gm_vertex == sink {
        return;
    }
    let orig_vertex = orig(gm_vertex);
    if orig_vertex >= vertex_count {
        return;
    }

    if is_a(gm_vertex) {
        if first_level_only && level1[orig_vertex] != vertex_level {
            return;
        }
        for &neighbor in &adj[gm_vertex] {
            if neighbor == sink {
                if level[sink] == INF {
                    level[sink] = vertex_level + 1;
                    par[sink] = gm_vertex;
                }
                continue;
            }
            if neighbor >= gm_limit {
                continue;
            }
            if level[neighbor] == INF {
                level[neighbor] = vertex_level + 1;
                par[neighbor] = gm_vertex;
                let neighbor_vertex = orig(neighbor);
                if neighbor_vertex < vertex_count && level1[neighbor_vertex] == INF {
                    level1[neighbor_vertex] = vertex_level + 1;
                }
                queue.push_back(neighbor);
            }
        }
    } else {
        for &neighbor_a in &adj[gm_vertex] {
            if neighbor_a == sink {
                if level[sink] == INF {
                    level[sink] = vertex_level + 1;
                    par[sink] = gm_vertex;
                }
                continue;
            }
            if neighbor_a >= gm_limit {
                continue;
            }

            let neighbor_vertex = orig(neighbor_a);
            let neighbor_b = b_side(neighbor_vertex);
            let candidate_level = level[neighbor_a];
            let mate_side_level = level[neighbor_b];

            if candidate_level == INF && (mate_side_level == INF || mate_side_level > vertex_level)
            {
                // Case 1: both sides unleveled or both > l → first level.
                level[neighbor_a] = vertex_level + 1;
                par[neighbor_a] = gm_vertex;
                if level1[neighbor_vertex] == INF {
                    level1[neighbor_vertex] = vertex_level + 1;
                }
                queue.push_back(neighbor_a);
            } else if mate_side_level != INF
                && ((candidate_level == INF && mate_side_level <= vertex_level)
                    || (candidate_level != INF && candidate_level <= vertex_level))
            {
                // Cases 2 & 3: bridge (w_b leveled, edge needed for
                // second-level computation).
                bridges.push((
                    gm_vertex,
                    neighbor_b,
                    usize::midpoint(vertex_level, mate_side_level),
                ));
            }
        }
    }
}

/// Union-find with path halving.
fn uf_find(uf: &mut [usize], mut x: usize) -> usize {
    while uf[x] != x {
        uf[x] = uf[uf[x]];
        x = uf[x];
    }
    x
}

// ── MDFS ────────────────────────────────────────────────────────────────

struct Mdfs {
    adj: Vec<Vec<usize>>,
    s: usize,
    t: usize,
    n: usize,
    sz: usize,

    eptr: Vec<usize>,
    ever: Vec<bool>,
    ink: Vec<bool>,
    par: Vec<usize>,
    push_time: Vec<usize>,
    time_counter: usize,

    l: Vec<Option<usize>>,
    l_rev: Vec<Vec<usize>>,
    l_ever: Vec<bool>,

    r: Vec<Vec<usize>>,
    e: Vec<Vec<usize>>,

    p: Vec<Option<(usize, usize)>>,

    drep: Vec<usize>,

    expanded: Vec<Option<(usize, usize)>>,

    k: Vec<usize>,

    // Generation-based visited: vis_stamp[i] == vis_gen means visited.
    vis_stamp: Vec<u32>,
    vis_gen: u32,

    // Reusable temporaries for backward_search (avoid per-call alloc).
    bs_queue: VecDeque<usize>,
    bs_dl: Vec<usize>,

    level: Option<Vec<usize>>,
    deleted: Vec<bool>,
}

impl Mdfs {
    fn new(sz: usize, s: usize, t: usize, n: usize, adj: Vec<Vec<usize>>) -> Self {
        Self {
            adj,
            s,
            t,
            n,
            sz,
            eptr: vec![0; sz],
            ever: vec![false; sz],
            ink: vec![false; sz],
            par: vec![usize::MAX; sz],
            push_time: vec![0; sz],
            time_counter: 0,
            l: vec![None; sz],
            l_rev: vec![Vec::new(); sz],
            l_ever: vec![false; sz],
            r: vec![Vec::new(); sz],
            e: vec![Vec::new(); sz],
            p: vec![None; sz],
            drep: (0..sz).collect(),
            expanded: vec![None; sz],
            k: Vec::with_capacity(sz),
            vis_stamp: vec![0; sz],
            vis_gen: 0,
            bs_queue: VecDeque::new(),
            bs_dl: Vec::new(),
            level: None,
            deleted: vec![false; sz],
        }
    }

    fn new_layered(
        sz: usize,
        s: usize,
        t: usize,
        n: usize,
        adj: Vec<Vec<usize>>,
        level: Vec<usize>,
    ) -> Self {
        let mut mdfs = Self::new(sz, s, t, n, adj);
        mdfs.level = Some(level);
        mdfs
    }

    fn take_adj(&mut self) -> Vec<Vec<usize>> {
        core::mem::take(&mut self.adj)
    }

    fn take_level(&mut self) -> Vec<usize> {
        self.level.take().unwrap_or_default()
    }

    /// Single-path MDFS on full G_M (fallback).
    fn run(&mut self, mate: &mut [Option<usize>]) -> bool {
        self.do_push(self.s, self.s);
        loop {
            let Some(&top) = self.k.last() else {
                return false;
            };
            if top == self.t {
                self.augment(mate);
                return true;
            }
            if !self.step(top) {
                self.do_pop(top);
            }
        }
    }

    /// Multi-path MDFS on the layered subgraph.
    fn run_multi_path(&mut self, mate: &mut [Option<usize>]) -> usize {
        let mut count = 0;
        self.do_push(self.s, self.s);
        loop {
            let Some(&top) = self.k.last() else {
                return count;
            };
            if top == self.t {
                self.augment_and_delete(mate);
                self.retreat_to_s();
                count += 1;
                continue;
            }
            if !self.step(top) {
                self.do_pop(top);
            }
        }
    }

    // ── Edge processing (SEARCH procedure) ──────────────────────────────

    fn step(&mut self, top: usize) -> bool {
        let n2 = 2 * self.n;

        while self.eptr[top] < self.adj[top].len() {
            let w = self.adj[top][self.eptr[top]];
            self.eptr[top] += 1;

            if self.deleted[w] {
                continue;
            }

            if let Some(ref lev) = self.level {
                let lt = lev[top];
                let lw = lev[w];
                if lt == INF || lw == INF || lw != lt + 1 {
                    continue;
                }
            }

            if w == self.t {
                self.do_push(self.t, top);
                return true;
            }

            if top == self.s {
                if !self.ever[w] {
                    self.do_push(w, top);
                    return true;
                }
                continue;
            }

            if is_a(top) {
                if !self.ever[w] {
                    self.do_push(w, top);
                    return true;
                }
                continue;
            }

            // top is B-side, w is A-side: unmatched edge.
            if self.ink[w] {
                if w < n2 {
                    self.e[w].push(top);
                }
                continue;
            }

            if self.ink[twin(w)] {
                if self.ever[w] {
                    if w < n2 {
                        self.e[w].push(top);
                        let v_a = twin(top);
                        let w_b = twin(w);
                        if !self.ever[v_a] || self.push_time[w_b] < self.push_time[v_a] {
                            self.r[w].push(top);
                        }
                    }
                } else if w < n2 {
                    self.r[w].push(top);
                }
                continue;
            }

            if self.ever[w] {
                if let Some(mut u_a) = self.l[w] {
                    while self.ever[u_a] || self.deleted[u_a] {
                        if let Some(next) = self.l[u_a] {
                            u_a = next;
                        } else {
                            self.l[w] = None;
                            break;
                        }
                    }
                    if self.ever[u_a] || self.deleted[u_a] {
                        self.l[w] = None;
                    }
                }
                if let Some(u_a) = self.l[w] {
                    self.expanded[u_a] = Some((top, w));
                    let sources = core::mem::take(&mut self.l_rev[u_a]);
                    for &src in &sources {
                        if self.l[src] == Some(u_a) {
                            self.l[src] = None;
                        }
                    }
                    self.l_rev[u_a] = sources;
                    self.do_push(u_a, top);
                    return true;
                }
                if w < n2 && !self.l_ever[w] {
                    self.e[w].push(top);
                }
            } else {
                self.do_push(w, top);
                return true;
            }
        }
        false
    }

    fn do_push(&mut self, node: usize, parent: usize) {
        self.time_counter += 1;
        self.push_time[node] = self.time_counter;
        self.ever[node] = true;
        self.ink[node] = true;
        self.par[node] = parent;
        self.k.push(node);
    }

    fn do_pop(&mut self, top: usize) {
        if top != self.s && !is_a(top) && top < 2 * self.n && !self.deleted[top] {
            self.backward_search(top);
        }
        self.ink[top] = false;
        self.k.pop();

        if top != self.s && !is_a(top) {
            if let Some(&pa) = self.k.last() {
                if pa != self.s && is_a(pa) && pa < 2 * self.n {
                    self.ink[pa] = false;
                    self.k.pop();
                }
            }
        }
    }

    // ── Backward search ─────────────────────────────────────────────────

    #[inline]
    fn vis_check(&self, i: usize) -> bool {
        self.vis_stamp[i] == self.vis_gen
    }

    #[inline]
    fn vis_set(&mut self, i: usize) {
        self.vis_stamp[i] = self.vis_gen;
    }

    fn backward_search(&mut self, v_b: usize) {
        let v_a = twin(v_b);

        // Generation-based reset: O(1) instead of O(sz).
        self.vis_gen = self.vis_gen.wrapping_add(1);
        if self.vis_gen == 0 {
            self.vis_stamp.fill(0);
            self.vis_gen = 1;
        }

        self.bs_queue.clear();
        self.bs_dl.clear();

        for i in 0..self.r[v_a].len() {
            let q_b = self.r[v_a][i];
            if !self.deleted[q_b] {
                self.constrl(q_b, v_a, v_b, v_a);
            }
        }
        for i in 0..self.e[v_a].len() {
            let q_b = self.e[v_a][i];
            if !self.vis_check(q_b) && !self.deleted[q_b] {
                self.constrl(q_b, v_a, v_b, v_a);
            }
        }

        while let Some(k_a) = self.bs_queue.pop_front() {
            for i in 0..self.e[k_a].len() {
                let q_b = self.e[k_a][i];
                if !self.vis_check(q_b) && !self.deleted[q_b] {
                    self.constrl(q_b, k_a, v_b, v_a);
                }
            }
        }

        if !self.ever[v_a] {
            for i in 0..self.bs_dl.len() {
                let y_a = self.bs_dl[i];
                self.l[y_a] = Some(v_a);
                self.l_ever[y_a] = true;
                self.l_rev[v_a].push(y_a);
            }
        }
    }

    fn constrl(&mut self, start_b: usize, edge_a: usize, stop_b: usize, lcur: usize) {
        let pcur = (start_b, edge_a);
        let lcur_root = self.find_rep(lcur);
        let mut z = start_b;

        loop {
            if z == stop_b || self.vis_check(z) || self.deleted[z] {
                return;
            }
            self.vis_set(z);

            let p = self.par[z];
            if p == usize::MAX || p == self.s || !is_a(p) || p >= 2 * self.n {
                return;
            }

            let y_a = p;
            if self.vis_check(y_a) || self.deleted[y_a] {
                return;
            }
            self.vis_set(y_a);

            if self.l_ever[y_a] {
                let r_a = self.find_rep(y_a);
                if r_a != lcur_root {
                    self.drep[r_a] = lcur_root;
                }
                let r_b = twin(r_a);
                if r_b != stop_b && !self.vis_check(r_b) && !self.deleted[r_b] {
                    z = r_b;
                    continue;
                }
                return;
            }

            self.drep[y_a] = lcur_root;
            self.p[y_a] = Some(pcur);
            self.bs_dl.push(y_a);
            self.bs_queue.push_back(y_a);

            let pp = self.par[y_a];
            if pp == usize::MAX || pp == self.s {
                return;
            }
            z = pp;
        }
    }

    fn find_rep(&self, mut x: usize) -> usize {
        let mut steps = 0;
        while self.drep[x] != x {
            steps += 1;
            if steps > self.sz {
                return x;
            }
            x = self.drep[x];
        }
        x
    }

    // ── Path reconstruction and augmentation ─────────────────────────────

    fn augment(&self, mate: &mut [Option<usize>]) {
        let n2 = 2 * self.n;
        let mut path: Vec<usize> = Vec::new();
        self.reconstr_path(self.t, self.s, &mut path);
        path.reverse();
        let orig_path: Vec<usize> = path.iter().filter(|&&v| v < n2).map(|&v| orig(v)).collect();
        for pair in orig_path.chunks_exact(2) {
            mate[pair[0]] = Some(pair[1]);
            mate[pair[1]] = Some(pair[0]);
        }
    }

    fn augment_and_delete(&mut self, mate: &mut [Option<usize>]) {
        let n2 = 2 * self.n;
        let mut path: Vec<usize> = Vec::new();
        self.reconstr_path(self.t, self.s, &mut path);
        path.reverse();
        for &v in &path {
            if v < n2 {
                self.deleted[v] = true;
                self.deleted[twin(v)] = true;
            }
        }
        let orig_path: Vec<usize> = path.iter().filter(|&&v| v < n2).map(|&v| orig(v)).collect();
        for pair in orig_path.chunks_exact(2) {
            mate[pair[0]] = Some(pair[1]);
            mate[pair[1]] = Some(pair[0]);
        }
    }

    fn retreat_to_s(&mut self) {
        while let Some(&top) = self.k.last() {
            if top == self.s {
                break;
            }
            if !self.deleted[top] && !is_a(top) && top < 2 * self.n {
                self.backward_search(top);
            }
            self.ink[top] = false;
            self.k.pop();
        }
    }

    fn reconstr_path(&self, end: usize, start: usize, out: &mut Vec<usize>) {
        self.reconstr_path_inner(end, start, out, true);
    }

    fn reconstr_path_inner(
        &self,
        end: usize,
        start: usize,
        out: &mut Vec<usize>,
        follow_expanded: bool,
    ) {
        let mut cur = end;
        let mut steps = 0;
        while cur != start && steps < self.sz * 4 {
            steps += 1;
            out.push(cur);
            if follow_expanded {
                if let Some((v_b, w_a)) = self.expanded[cur] {
                    self.reconstr_q(cur, w_a, out);
                    cur = v_b;
                    continue;
                }
            }
            cur = self.par[cur];
        }
        out.push(start);
    }

    fn reconstr_q(&self, u_a: usize, w_a: usize, out: &mut Vec<usize>) {
        let mut blocks: Vec<(usize, usize)> = Vec::new();
        let mut st = w_a;
        for _ in 0..self.sz {
            let Some((p1_b, p2_a)) = self.p[st] else {
                out.push(st);
                return;
            };
            blocks.push((p1_b, st));
            if p2_a == u_a {
                for &(blk_end, blk_start) in blocks.iter().rev() {
                    self.reconstr_path_inner(blk_end, blk_start, out, false);
                }
                return;
            }
            st = p2_a;
        }
    }
}
