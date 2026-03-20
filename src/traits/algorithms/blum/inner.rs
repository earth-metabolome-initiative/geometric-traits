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
//! MDFS maintains the full tree T (POP only moves a pointer, never
//! deletes nodes).  Each node is pushed at most once, so the total work
//! per augmentation is O(n+m).
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

// ── Public entry ────────────────────────────────────────────────────────

pub(super) struct BlumState<'a, M: SparseSquareMatrix + ?Sized> {
    matrix: &'a M,
    n: usize,
    mate: Vec<Option<usize>>,
}

impl<'a, M: SparseSquareMatrix + ?Sized> BlumState<'a, M> {
    pub(super) fn new(matrix: &'a M) -> Self {
        let n: usize = matrix.order().as_();
        Self { matrix, n, mate: vec![None; n] }
    }

    pub(super) fn solve(mut self) -> Vec<(M::Index, M::Index)> {
        for _ in 0..self.n {
            if !self.augment_once() {
                break;
            }
        }
        let indices: Vec<M::Index> = self.matrix.row_indices().collect();
        crate::traits::algorithms::matching_utils::mate_to_pairs(&self.mate, &indices)
    }

    fn augment_once(&mut self) -> bool {
        let n = self.n;
        if n == 0 {
            return false;
        }
        let s = 2 * n;
        let t = s + 1;
        let sz = t + 1;

        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); sz];
        for u in self.matrix.row_indices() {
            let ui: usize = u.as_();
            if self.mate[ui].is_none() {
                adj[s].push(b_side(ui));
                adj[a_side(ui)].push(t);
            }
            for v in self.matrix.sparse_row(u) {
                let vi: usize = v.as_();
                if vi <= ui {
                    continue;
                }
                if self.mate[ui] == Some(vi) {
                    adj[a_side(ui)].push(b_side(vi));
                    adj[a_side(vi)].push(b_side(ui));
                } else {
                    adj[b_side(ui)].push(a_side(vi));
                    adj[b_side(vi)].push(a_side(ui));
                }
            }
        }

        let mut mdfs = Mdfs::new(sz, s, t, n, adj);
        mdfs.run(&mut self.mate)
    }
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

    /// L[w_A]: deferred-push target.  Explicitly stored per the paper.
    /// After POP([w,A]), contains the single [u,A] satisfying:
    ///   1. MDFS found path [w,A]→Q→[u,A] with [u,B]∉Q
    ///   2. PUSH([u,A]) never performed
    ///   3. POP([u,B]) performed
    l: Vec<Option<usize>>,
    /// Reverse index: l_rev[target] lists sources with l[src]==Some(target).
    l_rev: Vec<Vec<usize>>,
    /// Has L[w] ever been non-empty?  (Paper's "L" set membership.)
    l_ever: Vec<bool>,

    /// R[u_A]: set of [v,B] where (v_B→u_A) is a weak back edge.
    r: Vec<Vec<usize>>,
    /// E[q_A]: set of [v,B] where (v_B→q_A) is a back/cross/forward edge.
    e: Vec<Vec<usize>>,

    /// P[v_A] = (source_B, target_A): non-tree edge concluding the block
    /// of tree edges containing v_A.  Set during backward search.
    p: Vec<Option<(usize, usize)>>,

    /// D-set union-find: drep[p_A] points toward the representative.
    drep: Vec<usize>,

    /// Expanded-node record: expanded[u_A] = (v_b, w_a) when Case 2.3.i
    /// pushed u_A via extensible edge from [v,B] through [w,A].
    expanded: Vec<Option<(usize, usize)>>,

    k: Vec<usize>,

    /// Reusable visited buffer for backward_search.
    vis: Vec<bool>,
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
            vis: vec![false; sz],
        }
    }

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

    // ── Edge processing (SEARCH procedure) ──────────────────────────────

    fn step(&mut self, top: usize) -> bool {
        let n2 = 2 * self.n;

        while self.eptr[top] < self.adj[top].len() {
            let w = self.adj[top][self.eptr[top]];
            self.eptr[top] += 1;

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
                // Case 1: matched edge A→B.
                if !self.ever[w] {
                    self.do_push(w, top);
                    return true;
                }
                continue;
            }

            // top is B-side, w is A-side: unmatched edge.
            if self.ink[w] {
                // Case 2.1: back edge.
                if w < n2 {
                    self.e[w].push(top);
                }
                continue;
            }

            if self.ink[twin(w)] {
                // Case 2.2.
                if self.ever[w] {
                    // 2.2.i: [w,A] previously pushed → cross edge.
                    if w < n2 {
                        self.e[w].push(top);
                        // 2025 fix: also add to R if [v,A] not yet pushed
                        // OR [w,B] was pushed before [v,A].
                        let v_a = twin(top);
                        let w_b = twin(w);
                        if !self.ever[v_a] || self.push_time[w_b] < self.push_time[v_a] {
                            self.r[w].push(top);
                        }
                    }
                } else {
                    // 2.2.ii: [w,A] never pushed → weak back edge.
                    if w < n2 {
                        self.r[w].push(top);
                    }
                }
                continue;
            }

            // Case 2.3: [w,A]∉K and [w,B]∉K.
            if self.ever[w] {
                // 2.3.i: [w,A] was previously pushed.
                if let Some(mut u_a) = self.l[w] {
                    // Follow L-chain: if u_a was already pushed,
                    // follow L[u_a] to find an unpushed target.
                    while self.ever[u_a] {
                        if let Some(next) = self.l[u_a] {
                            u_a = next;
                        } else {
                            self.l[w] = None;
                            break;
                        }
                    }
                    if self.ever[u_a] {
                        self.l[w] = None;
                    }
                }
                if let Some(u_a) = self.l[w] {
                    // L[w,A] ≠ ∅: extensible edge.
                    self.expanded[u_a] = Some((top, w));
                    // Invariant 3: clear L for all nodes pointing to u_a.
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
                // L[w,A] = ∅: add to E if [w,A] ∉ L (paper line 757).
                if w < n2 && !self.l_ever[w] {
                    self.e[w].push(top);
                }
            } else {
                // 2.3.ii: [w,A] never pushed → tree edge.
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
        if top != self.s && !is_a(top) && top < 2 * self.n {
            self.backward_search(top);
        }
        self.ink[top] = false;
        self.k.pop();

        // Double-POP: also pop the A-side parent.
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
    //
    // Paper Section 2.5, lines 824-843 (SEARCH after for-loop).
    //
    // Round 1: process R[v,A] (weak back edges) AND E[v,A] (since v_A is
    // never enqueued, its E edges would otherwise never be processed).
    // Subsequent rounds: process E[k,A] for each k_A dequeued.

    fn backward_search(&mut self, v_b: usize) {
        let v_a = twin(v_b);
        self.vis.fill(false);
        let mut queue: VecDeque<usize> = VecDeque::new();
        let mut dl: Vec<usize> = Vec::new();

        // Round 1: R[v,A].
        for i in 0..self.r[v_a].len() {
            let q_b = self.r[v_a][i];
            self.constrl(q_b, v_a, v_b, v_a, &mut queue, &mut dl);
        }
        // Also process E[v,A] in Round 1 (v_A is the root and is never
        // enqueued, so its E edges would be missed in subsequent rounds).
        for i in 0..self.e[v_a].len() {
            let q_b = self.e[v_a][i];
            if !self.vis[q_b] {
                self.constrl(q_b, v_a, v_b, v_a, &mut queue, &mut dl);
            }
        }

        // Subsequent rounds: E[k,A].
        while let Some(k_a) = queue.pop_front() {
            for i in 0..self.e[k_a].len() {
                let q_b = self.e[k_a][i];
                if !self.vis[q_b] {
                    self.constrl(q_b, k_a, v_b, v_a, &mut queue, &mut dl);
                }
            }
        }

        // Set L for discovered nodes if [v,A] was never pushed.
        if !self.ever[v_a] {
            for &y_a in &dl {
                self.l[y_a] = Some(v_a);
                self.l_ever[y_a] = true;
                self.l_rev[v_a].push(y_a);
            }
        }
    }

    // ── CONSTRL (paper lines 848-870) ───────────────────────────────────

    fn constrl(
        &mut self,
        start_b: usize,
        edge_a: usize,
        stop_b: usize,
        lcur: usize,
        queue: &mut VecDeque<usize>,
        dl: &mut Vec<usize>,
    ) {
        let pcur = (start_b, edge_a);
        // Always link to the root of lcur's D-set to avoid cycles.
        let lcur_root = self.find_rep(lcur);
        let mut z = start_b;

        loop {
            if z == stop_b || self.vis[z] {
                return;
            }
            self.vis[z] = true;

            let p = self.par[z];
            if p == usize::MAX || p == self.s || !is_a(p) || p >= 2 * self.n {
                return;
            }

            let y_a = p;
            if self.vis[y_a] {
                return;
            }
            self.vis[y_a] = true;

            if self.l_ever[y_a] {
                // [y,A] ∈ L: merge D-sets, jump to representative.
                // Paper: DLcur ∪= D[r,A]; [z,B] := [r,B]
                let r_a = self.find_rep(y_a);
                if r_a != lcur_root {
                    self.drep[r_a] = lcur_root;
                }
                let r_b = twin(r_a);
                if r_b != stop_b && !self.vis[r_b] {
                    z = r_b;
                    continue;
                }
                return;
            }

            // [y,A] ∉ L: new node for DLcur.
            // Paper: DLcur ∪= {[y,A]}, L ∪= {[y,A]}, P[y,A] := Pcur
            self.drep[y_a] = lcur_root;
            self.p[y_a] = Some(pcur);
            dl.push(y_a);
            queue.push_back(y_a);

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
                return x; // drep cycle — bail
            }
            x = self.drep[x];
        }
        x
    }

    // ── Path reconstruction and augmentation ─────────────────────────────
    //
    // Paper: RECONSTRPATH (lines 895-908) and RECONSTRQ (lines 912-923).

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
        // Collect blocks in forward order (w_a → u_a), then emit in
        // reverse so the final global reversal produces correct order.
        let mut blocks: Vec<(usize, usize)> = Vec::new(); // (p1_b, st)
        let mut st = w_a;
        for _ in 0..self.sz {
            let Some((p1_b, p2_a)) = self.p[st] else {
                out.push(st);
                return;
            };

            blocks.push((p1_b, st));

            if p2_a == u_a {
                // Emit blocks in reverse order.  Blocks are pure tree
                // walks — do NOT follow expanded entries.
                for &(blk_end, blk_start) in blocks.iter().rev() {
                    self.reconstr_path_inner(blk_end, blk_start, out, false);
                }
                return;
            }
            st = p2_a;
        }
    }
}
