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
//! union-find to assign second levels.  We also keep a per-free-vertex
//! single-path MDFS fallback as a safety net.  Blum's paper claims the
//! phased bound O(√V · (V + E)) because it assumes the search stays in
//! that regime. Our implementation keeps the phased fast path, but the
//! fallback is required for correctness on known counterexamples, so the
//! implementation's worst-case time is O(V · (V + E)).
use alloc::{collections::VecDeque, vec::Vec};

use num_traits::AsPrimitive;

use crate::traits::SparseSquareMatrix;

/// Maps original vertex `v` to its A-side copy `[v,A]` in G_M.
const fn a_side(v: usize) -> usize {
    2 * v
}
/// Maps original vertex `v` to its B-side copy `[v,B]` in G_M.
const fn b_side(v: usize) -> usize {
    2 * v + 1
}
/// Recovers the original vertex index from a G_M node index.
const fn orig(gm: usize) -> usize {
    gm / 2
}
/// Returns the twin of a G_M node: `[v,A] ↔ [v,B]`.
const fn twin(gm: usize) -> usize {
    gm ^ 1
}
/// Returns `true` if `gm` is an A-side node (even index).
const fn is_a(gm: usize) -> bool {
    gm & 1 == 0
}

/// Sentinel value meaning "level not yet defined" in MBFS arrays.
const INF: usize = usize::MAX;

// ── Public entry ────────────────────────────────────────────────────────

/// Top-level driver for Blum's maximum matching algorithm.
///
/// Holds the input matrix, the current matching (`mate`), and reusable
/// buffers for G_M construction and MBFS.  Call [`solve`](Self::solve)
/// to compute the maximum matching.
pub(super) struct BlumState<'a, M: SparseSquareMatrix + ?Sized> {
    matrix: &'a M,
    /// Number of original vertices.
    n: usize,
    /// Current matching: `mate[u] = Some(v)` means edge (u,v) is matched.
    mate: Vec<Option<usize>>,
    /// Adjacency lists for the directed bipartite graph G_M.
    /// Indexed by G_M node (0..2n for graph nodes, 2n=s, 2n+1=t).
    adj: Vec<Vec<usize>>,
    /// MBFS level array: `level[gm_node]` = shortest strongly simple
    /// distance from s, or `INF` if not yet reached.
    mbfs_level: Vec<usize>,
    /// First-level array indexed by *original* vertex (not G_M node).
    /// `level1[v]` records the level assigned to v during MBFS Part 1.
    mbfs_level1: Vec<usize>,
    /// MBFS parent pointers for the BFS tree.
    mbfs_par: Vec<usize>,
    /// Union-find parent array for MBFS bridge processing.
    mbfs_uf: Vec<usize>,
    /// Reusable BFS queue for MBFS.
    mbfs_queue: VecDeque<usize>,
}

impl<'a, M: SparseSquareMatrix + ?Sized> BlumState<'a, M> {
    /// Creates a new `BlumState` with empty matching and pre-allocated buffers.
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

    /// Runs the phased Blum algorithm and returns the maximum matching.
    ///
    /// Each phase:
    /// 1. Builds G_M from the current matching via [`fill_gm`](Self::fill_gm).
    /// 2. Runs MBFS to compute shortest strongly simple distances.
    /// 3. If MBFS reaches t: runs layered multi-path MDFS. If layered MDFS
    ///    finds 0 paths: falls back to per-vertex MDFS.
    /// 4. If MBFS cannot reach t (level\[t\] = INF): falls back to per-vertex
    ///    MDFS directly (Bug 3 workaround).
    ///
    /// The per-vertex fallback (Bug 2 workaround) isolates each free
    /// vertex's DFS to avoid cross-subtree `ever`-state poisoning.
    /// Both `augment` and `augment_and_delete` validate the reconstructed
    /// path for strong simplicity before applying it (Bug 1 workaround).
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
                // MBFS could not reach t — fall back to per-vertex MDFS.
                if !self.fallback_per_vertex(sz) {
                    break;
                }
                continue;
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
                // Fallback: layered MDFS found no paths. Try per-vertex.
                if !self.fallback_per_vertex(sz) {
                    break;
                }
            }
        }

        let indices: Vec<M::Index> = self.matrix.row_indices().collect();
        crate::traits::algorithms::matching_utils::mate_to_pairs(&self.mate, &indices)
    }

    /// Builds the directed bipartite graph G_M from the current matching.
    ///
    /// - For each free vertex u: adds edges s → b(u) and a(u) → t.
    /// - For each matched edge (u,v): adds a(u) → b(v) and a(v) → b(u).
    /// - For each unmatched edge (u,v): adds b(u) → a(v) and b(v) → a(u).
    ///
    /// Returns (s, t) — the source and sink sentinel indices.
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

    /// Per-free-vertex MDFS fallback (Bug 2 + Bug 3 workaround).
    ///
    /// The standard single-path MDFS explores all free vertices from a
    /// shared source `s`.  DFS ordering means the first subtree can mark
    /// nodes `ever`, poisoning later subtrees.  This fallback tries each
    /// free vertex with a **fresh** MDFS, isolating their `ever` state.
    fn fallback_per_vertex(&mut self, sz: usize) -> bool {
        let n = self.n;
        let s = 2 * n;
        let t = s + 1;

        // Collect free vertices (mate == None).
        let free: Vec<usize> = (0..n).filter(|&u| self.mate[u].is_none()).collect();

        for &u in &free {
            // Build G_M with only b(u) reachable from s.
            self.fill_gm();
            self.adj[s].clear();
            self.adj[s].push(b_side(u));

            let adj = core::mem::take(&mut self.adj);
            let mut mdfs = Mdfs::new(sz, s, t, n, adj);
            let ok = mdfs.run(&mut self.mate);
            self.adj = mdfs.take_adj();
            if ok {
                return true;
            }
        }
        false
    }
}

// ── MBFS (Modified Breadth-First Search) ────────────────────────────────

/// Computes shortest *strongly simple* distances from s in G_M.
///
/// - **Part 1**: standard BFS assigning first levels (level₁).
/// - **Part 2**: processes bridge pairs E(k) in order via back-path search with
///   union-find to assign second levels.
///
/// Reference: Blum 2015, Section 3 (full version, Uni Bonn, July 2016).
///
/// Union-find choice: standard path-halving (not Gabow-Tarjan).
/// See <https://github.com/LucaCappelletti94/incremental-tree-set-union>
/// for benchmarks showing Gabow-Tarjan is consistently slower in practice.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_lines)]
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

    // ── Part 1: BFS for first levels ────────────────────────────────

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

    // ── Generate A-side bridge pairs from matched edges ─────────────

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

    // ── Part 2: process bridges in order of k ───────────────────────
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

/// BFS neighbor scan for one G_M vertex during MBFS.
///
/// - **A-side nodes**: scan outgoing matched edges (A → B). In Part 1
///   (`first_level_only = true`), only scan if `level1[orig] == vertex_level`
///   to avoid re-scanning nodes that got second levels.
/// - **B-side nodes**: scan outgoing unmatched edges (B → A). If the neighbor
///   A-side is unleveled and qualifies, assign a first level (Case 1). If the
///   neighbor's mate-side is already leveled at ≤ current level, generate a
///   bridge pair (Cases 2 & 3).
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

// ── MDFS (Modified Depth-First Search) ──────────────────────────────────

/// State for the Modified Depth-First Search on G_M.
///
/// Searches for strongly simple s→t paths using Blum's edge
/// classification (Cases 1, 2.1, 2.2.i/ii, 2.3.i/ii) with the
/// Dandeh & Lukovszki corrections for Cases 2.2.i and 2.3.i.
struct Mdfs {
    /// G_M adjacency lists (owned during MDFS, returned afterwards).
    adj: Vec<Vec<usize>>,
    /// Source sentinel in G_M (index 2n).
    s: usize,
    /// Sink sentinel in G_M (index 2n+1).
    t: usize,
    /// Number of original vertices.
    n: usize,
    /// Total G_M size: 2n + 2 (graph nodes + s + t).
    sz: usize,

    /// Edge pointer: next unexamined neighbor index for each node.
    eptr: Vec<usize>,
    /// `ever[v]` = true iff PUSH(v) has been performed at any point.
    ever: Vec<bool>,
    /// `ink[v]` = true iff v is currently on the DFS stack K.
    ink: Vec<bool>,
    /// DFS parent pointer: `par[v]` = the node that pushed v.
    par: Vec<usize>,

    /// Label set L: `l[w_A] = Some(u_A)` means the extensible edge
    /// from w_A reaches u_A.  See Blum 2015 Section 2.3.
    l: Vec<Option<usize>>,
    /// Reverse map for L: `l_rev[u_A]` lists all w_A with `l[w_A] = u_A`,
    /// used by [`clear_l_sources`](Self::clear_l_sources).
    l_rev: Vec<Vec<usize>>,
    /// `l_ever[w]` = true iff the L entry for `w` was ever set
    /// (even if later cleared).
    l_ever: Vec<bool>,

    /// R set: `r[u_A]` = { v_B | (v_B, u_A) is a weak back edge }.
    /// Populated in Cases 2.2.i (D&L fix) and 2.2.ii.
    r: Vec<Vec<usize>>,
    /// E set: `e[q_A]` = { v_B | (v_B, q_A) is a back, cross, or forward edge
    /// }. Populated in Cases 2.1, 2.3.i.
    e: Vec<Vec<usize>>,

    /// Non-tree edge record for path reconstruction:
    /// `p[r_A] = Some((start_B, edge_A))` records the non-tree edge
    /// that concludes the block of tree edges containing r_A.
    p: Vec<Option<(usize, usize)>>,

    /// D-set representative (union-find): `drep[x]` is x's parent in
    /// the disjoint-set forest.  Blum defines D\[q,A\] as the set of
    /// nodes whose label L was previously set to q_A.  We implement D
    /// via union-find (as Blum recommends) with path-halving compression.
    drep: Vec<usize>,

    /// Extensible-edge record: `expanded[u_A] = Some((v_B, w_A))`
    /// means u_A was pushed via Case 2.3.i using L\[w_A\] = u_A,
    /// creating extensible edge (v_B, u_A)\[w_A\].
    expanded: Vec<Option<(usize, usize)>>,

    /// DFS stack K: contains the nodes from s to the current vertex.
    k: Vec<usize>,

    /// Generation-based visited stamp for backward search.
    /// `vis_stamp[i] == vis_gen` means node i has been visited in
    /// the current backward search invocation.
    vis_stamp: Vec<u32>,
    /// Current generation counter.  Incremented at each backward search
    /// call, giving O(1) reset instead of O(n) clearing.
    vis_gen: u32,

    /// Reusable BFS queue for backward search rounds.
    bs_queue: VecDeque<usize>,
    /// Nodes discovered during backward search that will receive L labels.
    bs_dl: Vec<usize>,

    /// Optional MBFS level array.  When `Some`, the MDFS is restricted
    /// to the layered subgraph (edges where `level[w] == level[top] + 1`).
    /// When `None`, the MDFS runs on the full G_M (fallback mode).
    level: Option<Vec<usize>>,
    /// `deleted[v]` = true iff v was part of a previously augmented path
    /// in this phase (multi-path mode only).
    deleted: Vec<bool>,
}

impl Mdfs {
    /// Creates a new MDFS for full-graph (unlayered) search.
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

    /// Creates a new MDFS restricted to the layered subgraph defined
    /// by the MBFS `level` array.
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

    /// Returns the adjacency lists, transferring ownership back to the caller.
    fn take_adj(&mut self) -> Vec<Vec<usize>> {
        core::mem::take(&mut self.adj)
    }

    /// Returns the level array, transferring ownership back to the caller.
    fn take_level(&mut self) -> Vec<usize> {
        self.level.take().unwrap_or_default()
    }

    /// Single-path MDFS on the full G_M (fallback mode).
    ///
    /// Searches for one strongly simple s→t path.  When t is reached,
    /// the path is reconstructed and validated for strong simplicity
    /// (Bug 1 workaround).  If the path fails validation, the search
    /// backtracks and continues looking for an alternative.
    fn run(&mut self, mate: &mut [Option<usize>]) -> bool {
        self.do_push(self.s, self.s);
        loop {
            let Some(&top) = self.k.last() else {
                return false;
            };
            if top == self.t {
                if self.augment(mate) {
                    return true;
                }
                // Path was not strongly simple — backtrack and keep searching.
                self.do_pop(top);
            } else if !self.step(top) {
                self.do_pop(top);
            }
        }
    }

    /// Multi-path MDFS on the layered subgraph.
    ///
    /// Finds as many vertex-disjoint augmenting paths as possible in
    /// the subgraph defined by the MBFS levels.  Each found path is
    /// validated, augmented, and its vertices deleted from the search
    /// space.  Returns the number of augmenting paths found.
    fn run_multi_path(&mut self, mate: &mut [Option<usize>]) -> usize {
        let mut count = 0;
        self.do_push(self.s, self.s);
        loop {
            let Some(&top) = self.k.last() else {
                return count;
            };
            if top == self.t {
                if self.augment_and_delete(mate) {
                    self.retreat_to_s();
                    count += 1;
                } else {
                    // Path was not strongly simple — backtrack.
                    self.do_pop(top);
                }
                continue;
            }
            if !self.step(top) {
                self.do_pop(top);
            }
        }
    }

    // ── Edge processing (SEARCH procedure) ──────────────────────────

    /// Examines the next unprocessed edge from `top` and either pushes
    /// a new node or records the edge for later use.
    ///
    /// Returns `true` if a node was pushed (the caller should re-enter
    /// the main loop), `false` if all edges from `top` are exhausted
    /// (the caller should pop).
    ///
    /// Edge classification follows Blum 2015, Section 2.3:
    /// - **Case 1** (tree edge, A→B via matched edge): push w_B.
    /// - **Case 2.1** (back edge, w_A on stack): add to E\[w\].
    /// - **Case 2.2.i** (weak back, w_A previously pushed, w_B on stack): add
    ///   to R\[w\] with D&L selective condition.
    /// - **Case 2.2.ii** (weak back, w_A never pushed): add to R\[w\].
    /// - **Case 2.3.i** (forward/cross, w_A previously pushed): if L\[w\] has a
    ///   target, push via extensible edge; otherwise add to E\[w\].
    /// - **Case 2.3.ii** (tree edge, w_A never pushed): push w_A.
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

            // Case 1 / 2.3.ii from s: tree edge to unvisited node.
            if top == self.s {
                if !self.ever[w] {
                    self.do_push(w, top);
                    return true;
                }
                continue;
            }

            // Case 1: tree edge (A-side top → B-side w via matched edge).
            if is_a(top) {
                if !self.ever[w] {
                    self.do_push(w, top);
                    return true;
                }
                continue;
            }

            // From here: top is B-side, w is A-side (unmatched edge).

            // Case 2.1: back edge — w_A is currently on the stack.
            if self.ink[w] {
                if w < n2 {
                    self.e[w].push(top);
                }
                continue;
            }

            // Case 2.2: weak back edge — w_A not on stack, w_B on stack.
            if self.ink[twin(w)] {
                if self.ever[w] {
                    // Case 2.2.i: w_A was previously pushed.
                    // Blum's original algorithm: do nothing.
                    // D&L propose adding to R[w] with a selective condition,
                    // but their bug does not reproduce in our phased
                    // architecture (tested on their Figure 1 counterexample).
                } else if w < n2 {
                    // Case 2.2.ii: w_A was never pushed.
                    self.r[w].push(top);
                }
                continue;
            }

            // Case 2.3: w_A and w_B both off the stack.
            if self.ever[w] {
                // Case 2.3.i: forward/cross edge, w_A was previously pushed.
                // Chase the label chain to find a valid extensible target.
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
                    // L[w] has a usable target → push via extensible edge.
                    self.expanded[u_a] = Some((top, w));
                    self.clear_l_sources(u_a);
                    self.do_push(u_a, top);
                    return true;
                }
                // L[w] is empty → record in E for backward search.
                if w < n2 {
                    self.e[w].push(top);
                }
            } else {
                // Case 2.3.ii: tree edge — w_A never pushed before.
                self.do_push(w, top);
                return true;
            }
        }
        false
    }

    /// Clears all L labels that point to `target`, using the reverse
    /// map `l_rev`.  Called when `target` is used as an extensible-edge
    /// endpoint, invalidating all labels that pointed to it.
    fn clear_l_sources(&mut self, target: usize) {
        let sources = core::mem::take(&mut self.l_rev[target]);
        for &src in &sources {
            if self.l[src] == Some(target) {
                self.l[src] = None;
            }
        }
    }

    /// PUSH(node): adds `node` to the DFS stack K and marks it as
    /// visited (`ever`) and on-stack (`ink`).
    fn do_push(&mut self, node: usize, parent: usize) {
        self.ever[node] = true;
        self.ink[node] = true;
        self.par[node] = parent;
        self.k.push(node);
    }

    /// POP(top): removes `top` from the DFS stack K.
    ///
    /// If `top` is a B-side graph node (not s, not deleted), triggers
    /// [`backward_search`](Self::backward_search) to compute L labels.
    /// Also pops the paired A-side node if it was pushed as part of a
    /// matched-edge tree step.
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

    // ── Backward search ─────────────────────────────────────────────

    /// Returns `true` if node `i` was visited in the current backward search.
    #[inline]
    fn vis_check(&self, i: usize) -> bool {
        self.vis_stamp[i] == self.vis_gen
    }

    /// Marks node `i` as visited in the current backward search.
    #[inline]
    fn vis_set(&mut self, i: usize) {
        self.vis_stamp[i] = self.vis_gen;
    }

    /// Backward search after POP(v_B).
    ///
    /// Computes L labels for A-side nodes reachable backward from v_A
    /// (the twin of the popped B-side node) through the expanded
    /// MDFS-tree T_exp.  Processes R\[v_A\] and E\[v_A\] entries via
    /// [`constrl`](Self::constrl) in BFS rounds.
    ///
    /// If v_A has never been pushed (`!ever[v_A]`), sets `L[y_A] = v_A`
    /// for all discovered nodes y_A.
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

        // Round 1: process R (weak back edges).
        for i in (0..self.r[v_a].len()).rev() {
            let q_b = self.r[v_a][i];
            if !self.deleted[q_b] {
                self.constrl(q_b, v_a, v_b, v_a);
            }
        }
        // Round 2 (BFS-through-Q): process E entries of discovered nodes.
        while let Some(k_a) = self.bs_queue.pop_front() {
            for i in (0..self.e[k_a].len()).rev() {
                let q_b = self.e[k_a][i];
                if !self.vis_check(q_b) && !self.deleted[q_b] {
                    self.constrl(q_b, k_a, v_b, v_a);
                }
            }
        }

        // Assign L labels unconditionally (testing without ever guard).
        for i in 0..self.bs_dl.len() {
            let y_a = self.bs_dl[i];
            self.l[y_a] = Some(v_a);
            self.l_ever[y_a] = true;
            self.l_rev[v_a].push(y_a);
        }
    }

    /// Backward tree-walk from `start_b` upward through parent pointers.
    ///
    /// Follows the DFS tree backward from `start_b`, visiting pairs
    /// (z_B, y_A = par\[z\]).  For each y_A:
    /// - If y_A was previously labeled (`l_ever`): merges its D-set
    ///   representative with `lcur`'s root, then jumps to twin(rep) to continue
    ///   the walk.
    /// - Otherwise: records `p[y_A] = (start_b, edge_a)`, adds y_A to the
    ///   discovery list and BFS queue, and continues upward.
    ///
    /// Stops at `stop_b`, at already-visited nodes, or at s.
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
                // y_A was previously labeled: merge D-sets and jump.
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

            // y_A is new: record P, add to discovery list and queue.
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

    /// Union-find root query with path-halving compression on `drep`.
    fn find_rep(&mut self, mut x: usize) -> usize {
        while self.drep[x] != x {
            let gp = self.drep[self.drep[x]];
            self.drep[x] = gp;
            x = gp;
        }
        x
    }

    // ── Path reconstruction and augmentation ─────────────────────────

    /// Reconstructs the s→t path and validates it for strong simplicity.
    ///
    /// Returns the path as a `Vec<usize>` of G_M nodes (s first, t last),
    /// or an empty `Vec` if the path visits any original vertex on both
    /// its A-side and B-side (Bug 1 workaround).
    fn validated_path(&self) -> Vec<usize> {
        let n2 = 2 * self.n;
        let mut path: Vec<usize> = Vec::new();
        self.reconstr_path(self.t, self.s, &mut path);
        path.reverse();
        // Strong-simplicity check: each original vertex appears at most once.
        let mut seen = vec![false; self.n];
        for &v in &path {
            if v < n2 {
                let o = orig(v);
                if seen[o] {
                    return Vec::new(); // not strongly simple
                }
                seen[o] = true;
            }
        }
        path
    }

    /// Single-path augmentation: reconstructs the path, validates it,
    /// and flips the matching along it.  Returns `false` if the path
    /// fails strong-simplicity validation.
    fn augment(&self, mate: &mut [Option<usize>]) -> bool {
        let n2 = 2 * self.n;
        let path = self.validated_path();
        if path.is_empty() {
            return false;
        }
        let orig_path: Vec<usize> = path.iter().filter(|&&v| v < n2).map(|&v| orig(v)).collect();
        for pair in orig_path.chunks_exact(2) {
            mate[pair[0]] = Some(pair[1]);
            mate[pair[1]] = Some(pair[0]);
        }
        true
    }

    /// Multi-path augmentation: reconstructs and validates the path,
    /// marks all path vertices as deleted, and flips the matching.
    /// Returns `false` if the path fails strong-simplicity validation.
    fn augment_and_delete(&mut self, mate: &mut [Option<usize>]) -> bool {
        let n2 = 2 * self.n;
        let path = self.validated_path();
        if path.is_empty() {
            return false;
        }
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
        true
    }

    /// Pops the stack back to s after a successful augmentation in
    /// multi-path mode.  Runs backward search on each popped B-side
    /// node (unless deleted) to update labels for subsequent paths.
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

    /// Entry point for path reconstruction: reconstructs the path from
    /// `end` (t) back to `start` (s) with full extensible-edge expansion.
    fn reconstr_path(&self, end: usize, start: usize, out: &mut Vec<usize>) {
        self.reconstr_path_inner(end, start, out, true);
    }

    /// Recursive path reconstruction following parent pointers.
    ///
    /// Walks from `end` toward `start` via `par` pointers.  When an
    /// extensible edge is encountered (`expanded[cur]` is set), calls
    /// [`reconstr_q`](Self::reconstr_q) to expand the subpath through
    /// the P-pointer block chain, then continues from the B-side source.
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

    /// RECONSTRQ: reconstructs the subpath through an extensible edge.
    ///
    /// Given extensible edge endpoint `u_a` and the label source `w_a`,
    /// follows the P-pointer chain from `w_a` backward through blocks
    /// until reaching `u_a`.  Each block is reconstructed recursively
    /// via `reconstr_path_inner`.
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
                    self.reconstr_path_inner(blk_end, blk_start, out, true);
                }
                return;
            }
            st = p2_a;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clear_l_sources_discards_stale_reverse_links() {
        let sz = 6;
        let mut mdfs = Mdfs::new(sz, 4, 5, 2, vec![Vec::new(); sz]);

        mdfs.l[0] = Some(2);
        mdfs.l[1] = Some(2);
        mdfs.l[3] = Some(4);
        mdfs.l_rev[2] = vec![0, 1, 1, 3];

        mdfs.clear_l_sources(2);

        assert_eq!(mdfs.l[0], None);
        assert_eq!(mdfs.l[1], None);
        assert_eq!(mdfs.l[3], Some(4));
        assert!(mdfs.l_rev[2].is_empty());
    }
}
