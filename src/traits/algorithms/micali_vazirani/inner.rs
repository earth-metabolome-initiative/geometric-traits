//! Internal state and algorithm for the Micali-Vazirani maximum matching.
//!
//! Reference: Peterson & Loui (1988) exposition of Micali-Vazirani.
//! Implementation based on ggawryal/MV-matching (C++).
//! All recursion has been converted to iterative form.

use alloc::{collections::VecDeque, vec, vec::Vec};

use num_traits::AsPrimitive;

use super::MicaliVaziraniError;
use crate::traits::SparseSquareMatrix;

const INF: usize = usize::MAX / 2;

// ────────────────────────────────────────────────────────────────────────────
// Edge types and internal edge representation
// ────────────────────────────────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq, Eq)]
enum EdgeType {
    NotScanned,
    Prop,
    Bridge,
}

#[derive(Clone)]
struct InternalEdge {
    to: usize,
    /// Index of the reverse edge in `adj[to]`.
    reverse_idx: usize,
    edge_type: EdgeType,
}

// ────────────────────────────────────────────────────────────────────────────
// Disjoint Set Union with group root tracking
// ────────────────────────────────────────────────────────────────────────────

struct Dsu {
    link: Vec<usize>,
    direct_parent: Vec<Option<usize>>,
    size: Vec<usize>,
    group_root: Vec<usize>,
}

impl Dsu {
    fn new(n: usize) -> Self {
        Self {
            link: (0..n).collect(),
            direct_parent: vec![None; n],
            size: vec![1; n],
            group_root: (0..n).collect(),
        }
    }

    fn reset(&mut self, n: usize) {
        if self.link.len() == n {
            for (i, v) in self.link.iter_mut().enumerate() {
                *v = i;
            }
            self.direct_parent.fill(None);
            self.size.fill(1);
            for (i, v) in self.group_root.iter_mut().enumerate() {
                *v = i;
            }
        } else {
            *self = Self::new(n);
        }
    }

    /// Iterative find with path compression.
    fn find(&mut self, mut a: usize) -> usize {
        let mut root = a;
        while self.link[root] != root {
            root = self.link[root];
        }
        while self.link[a] != root {
            let next = self.link[a];
            self.link[a] = root;
            a = next;
        }
        root
    }

    /// Returns the group root (bud) for vertex `a`.
    fn bud(&mut self, a: usize) -> usize {
        let r = self.find(a);
        self.group_root[r]
    }

    /// Links `a`'s component into `b`'s component, preserving `b`'s group root.
    fn link_to(&mut self, a: usize, b: usize) {
        debug_assert!(self.direct_parent[a].is_none());
        debug_assert!(self.direct_parent[b].is_none());
        self.direct_parent[a] = Some(b);
        let mut ra = self.find(a);
        let mut rb = self.find(b);
        let gr = self.group_root[rb];
        debug_assert_ne!(ra, rb);

        if self.size[ra] < self.size[rb] {
            core::mem::swap(&mut ra, &mut rb);
        }
        // ra is now the larger root; attach rb under ra.
        self.link[rb] = ra;
        self.size[ra] += self.size[rb];
        self.group_root[ra] = gr;
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Frame types for the iterative augment_path work stack
// ────────────────────────────────────────────────────────────────────────────

enum AugFrame {
    Augment {
        u: usize,
        v: usize,
        initial: bool,
    },
    OpeningDfs {
        cur: usize,
        bcur: usize,
        b: usize,
        child_idx: usize,
    },
    /// Runs after a child `OpeningDfs` returns. If the child succeeded
    /// (`opening_dfs_result == true`), commits the augment+flip and
    /// propagates. Otherwise, continues trying the next children.
    HandleChildResult {
        cur: usize,
        bcur: usize,
        b: usize,
        child_a: usize,
        next_child_idx: usize,
    },
    /// Restores `opening_dfs_result` to `true` after a nested `Augment`
    /// subtree completes, so the parent `HandleChildResult` sees success.
    RestoreResult,
}

// ────────────────────────────────────────────────────────────────────────────
// Main MV state
// ────────────────────────────────────────────────────────────────────────────

pub(super) struct MVState<'a, M: SparseSquareMatrix + ?Sized> {
    matrix: &'a M,
    n: usize,
    adj: Vec<Vec<InternalEdge>>,
    mate: Vec<Option<usize>>,

    // Per-phase state (reset each phase):
    predecessors: Vec<Vec<usize>>,
    ddfs_pred_ptr: Vec<usize>,
    removed: Vec<bool>,
    evenlvl: Vec<usize>,
    oddlvl: Vec<usize>,
    bud: Dsu,
    global_color_counter: usize,
    color: Vec<usize>,
    children_in_ddfs_tree: Vec<Vec<(usize, usize)>>,
    my_bridge: Vec<((usize, usize), (usize, usize))>,
    removed_queue: VecDeque<usize>,
}

impl<'a, M: SparseSquareMatrix + ?Sized> MVState<'a, M> {
    pub(super) fn try_new(matrix: &'a M) -> Result<Self, MicaliVaziraniError> {
        let n: usize = matrix.order().as_();
        let adj = Self::build_adjacency(matrix, n)?;
        Ok(Self {
            matrix,
            n,
            adj,
            mate: vec![None; n],
            predecessors: Vec::new(),
            ddfs_pred_ptr: Vec::new(),
            removed: Vec::new(),
            evenlvl: Vec::new(),
            oddlvl: Vec::new(),
            bud: Dsu::new(0),
            global_color_counter: 1,
            color: Vec::new(),
            children_in_ddfs_tree: Vec::new(),
            my_bridge: Vec::new(),
            removed_queue: VecDeque::new(),
        })
    }

    /// Build internal adjacency list with reverse-edge indices.
    ///
    /// Expects a symmetric matrix. For each undirected edge `{u, v}`, adds
    /// both `(u, v)` and `(v, u)` with linked reverse indices. Edges are
    /// discovered by scanning `sparse_row(u)` for `v > u` only, relying on
    /// symmetry to avoid duplicates.
    fn build_adjacency(
        matrix: &M,
        n: usize,
    ) -> Result<Vec<Vec<InternalEdge>>, MicaliVaziraniError> {
        if !matrix.is_symmetric() {
            for u in matrix.row_indices() {
                let ui: usize = u.as_();
                for v in matrix.sparse_row(u) {
                    let vi: usize = v.as_();
                    if ui != vi && !matrix.has_entry(v, u) {
                        return Err(MicaliVaziraniError::NonSymmetricEdge {
                            source_id: ui,
                            destination_id: vi,
                        });
                    }
                }
            }
        }

        let mut adj: Vec<Vec<InternalEdge>> = vec![Vec::new(); n];

        for u in matrix.row_indices() {
            let ui: usize = u.as_();
            for v in matrix.sparse_row(u) {
                let vi: usize = v.as_();
                if vi <= ui {
                    continue;
                }
                let idx_uv = adj[ui].len();
                let idx_vu = adj[vi].len();
                adj[ui].push(InternalEdge {
                    to: vi,
                    reverse_idx: idx_vu,
                    edge_type: EdgeType::NotScanned,
                });
                adj[vi].push(InternalEdge {
                    to: ui,
                    reverse_idx: idx_uv,
                    edge_type: EdgeType::NotScanned,
                });
            }
        }

        Ok(adj)
    }

    pub(super) fn solve(mut self) -> Vec<(M::Index, M::Index)> {
        loop {
            self.reset_phase();
            if !self.bfs_phase() {
                break;
            }
        }
        self.into_pairs()
    }

    fn reset_phase(&mut self) {
        let n = self.n;

        for edges in &mut self.adj {
            for e in edges {
                e.edge_type = EdgeType::NotScanned;
            }
        }

        // Reuse existing allocations — clear/fill instead of reallocating.
        if self.predecessors.len() == n {
            for v in &mut self.predecessors {
                v.clear();
            }
        } else {
            self.predecessors = vec![Vec::new(); n];
        }
        if self.ddfs_pred_ptr.len() == n {
            self.ddfs_pred_ptr.fill(0);
        } else {
            self.ddfs_pred_ptr = vec![0; n];
        }
        if self.removed.len() == n {
            self.removed.fill(false);
        } else {
            self.removed = vec![false; n];
        }
        if self.evenlvl.len() == n {
            self.evenlvl.fill(INF);
        } else {
            self.evenlvl = vec![INF; n];
        }
        if self.oddlvl.len() == n {
            self.oddlvl.fill(INF);
        } else {
            self.oddlvl = vec![INF; n];
        }
        self.bud.reset(n);
        self.global_color_counter = 1;
        if self.color.len() == n {
            self.color.fill(0);
        } else {
            self.color = vec![0; n];
        }
        if self.children_in_ddfs_tree.len() == n {
            for v in &mut self.children_in_ddfs_tree {
                v.clear();
            }
        } else {
            self.children_in_ddfs_tree = vec![Vec::new(); n];
        }
        if self.my_bridge.len() == n {
            self.my_bridge.fill(((0, 0), (0, 0)));
        } else {
            self.my_bridge = vec![((0, 0), (0, 0)); n];
        }
        self.removed_queue.clear();
    }

    #[inline]
    fn minlvl(&self, u: usize) -> usize {
        self.evenlvl[u].min(self.oddlvl[u])
    }

    #[inline]
    fn tenacity(&self, u: usize, v: usize) -> usize {
        if self.mate[u] == Some(v) {
            self.oddlvl[u].saturating_add(self.oddlvl[v]).saturating_add(1)
        } else {
            self.evenlvl[u].saturating_add(self.evenlvl[v]).saturating_add(1)
        }
    }

    #[inline]
    fn set_lvl(&mut self, u: usize, lev: usize, vertices_at_level: &mut [Vec<usize>]) {
        if lev & 1 == 1 {
            self.oddlvl[u] = lev;
        } else {
            self.evenlvl[u] = lev;
        }
        if lev < vertices_at_level.len() {
            vertices_at_level[lev].push(u);
        }
    }

    #[inline]
    fn remove_and_push(&mut self, u: usize) {
        self.removed[u] = true;
        self.removed_queue.push_back(u);
    }

    #[inline]
    fn flip(&mut self, u: usize, v: usize) {
        if self.removed[u] || self.removed[v] || self.mate[u] == Some(v) {
            return;
        }
        self.remove_and_push(u);
        self.remove_and_push(v);
        self.mate[u] = Some(v);
        self.mate[v] = Some(u);
    }

    // ────────────────────────────────────────────────────────────────────
    // BFS phase
    // ────────────────────────────────────────────────────────────────────

    fn bfs_phase(&mut self) -> bool {
        let n = self.n;
        let mut vertices_at_level: Vec<Vec<usize>> = vec![Vec::new(); n];
        let mut bridges: Vec<Vec<(usize, usize)>> = vec![Vec::new(); 2 * n + 2];
        let mut removed_pred_size: Vec<usize> = vec![0; n];

        for u in 0..n {
            if self.mate[u].is_none() {
                self.set_lvl(u, 0, &mut vertices_at_level);
            }
        }

        let mut found_path = false;
        let mut i = 0;
        while i < n && !found_path {
            self.scan_level_edges(i, &mut vertices_at_level, &mut bridges);

            let bridge_idx = 2 * i + 1;
            if bridge_idx < bridges.len() {
                found_path = self.process_bridges(
                    i,
                    bridge_idx,
                    &mut bridges,
                    &mut vertices_at_level,
                    &mut removed_pred_size,
                );
            }

            i += 1;
        }

        found_path
    }

    /// Scan edges from vertices at the given BFS level, classifying as Prop
    /// or Bridge.
    fn scan_level_edges(
        &mut self,
        level: usize,
        vertices_at_level: &mut [Vec<usize>],
        bridges: &mut [Vec<(usize, usize)>],
    ) {
        let level_vertices = core::mem::take(&mut vertices_at_level[level]);
        for &u in &level_vertices {
            let num_edges = self.adj[u].len();
            for ei in 0..num_edges {
                if self.adj[u][ei].edge_type != EdgeType::NotScanned {
                    continue;
                }
                let to = self.adj[u][ei].to;
                let rev = self.adj[u][ei].reverse_idx;

                let u_at_odd = self.oddlvl[u] == level;
                let is_matched_edge = self.mate[u] == Some(to);
                if u_at_odd != is_matched_edge {
                    continue;
                }

                if self.minlvl(to) > level {
                    self.adj[u][ei].edge_type = EdgeType::Prop;
                    self.adj[to][rev].edge_type = EdgeType::Prop;

                    if self.minlvl(to) > level + 1 {
                        self.set_lvl(to, level + 1, vertices_at_level);
                    }
                    self.predecessors[to].push(u);
                } else {
                    self.adj[u][ei].edge_type = EdgeType::Bridge;
                    self.adj[to][rev].edge_type = EdgeType::Bridge;
                    let t = self.tenacity(u, to);
                    if t < INF && t < bridges.len() {
                        bridges[t].push((u, to));
                    }
                }
            }
        }
        vertices_at_level[level] = level_vertices;
    }

    /// Process all bridges at the given tenacity bucket. Returns true if an
    /// augmenting path was found.
    fn process_bridges(
        &mut self,
        level: usize,
        bridge_idx: usize,
        bridges: &mut [Vec<(usize, usize)>],
        vertices_at_level: &mut [Vec<usize>],
        removed_pred_size: &mut [usize],
    ) -> bool {
        let mut found_path = false;
        let current_bridges = core::mem::take(&mut bridges[bridge_idx]);

        for &(edge_u, edge_v) in &current_bridges {
            if self.removed[self.bud.bud(edge_u)] || self.removed[self.bud.bud(edge_v)] {
                continue;
            }

            let mut support = Vec::new();
            let ddfs_result = self.ddfs((edge_u, edge_v), &mut support);
            let cur_bridge = ((edge_u, edge_v), (self.bud.bud(edge_u), self.bud.bud(edge_v)));

            for &v in &support {
                if v == ddfs_result.1 {
                    continue;
                }
                self.my_bridge[v] = cur_bridge;
                self.bud.link_to(v, ddfs_result.1);

                let new_lvl = (2 * level + 1) - self.minlvl(v);
                self.set_lvl(v, new_lvl, vertices_at_level);

                if self.evenlvl[v] > self.oddlvl[v] {
                    let num_edges = self.adj[v].len();
                    for ei in 0..num_edges {
                        let e = &self.adj[v][ei];
                        if e.edge_type == EdgeType::Bridge && self.mate[v] != Some(e.to) {
                            let t = self.tenacity(v, e.to);
                            if t < INF && t < bridges.len() {
                                bridges[t].push((v, e.to));
                            }
                        }
                    }
                }
            }

            if ddfs_result.0 != ddfs_result.1 {
                self.augment_path(ddfs_result.0, ddfs_result.1, true);
                found_path = true;
                self.propagate_removals(removed_pred_size);
            }
        }

        bridges[bridge_idx] = current_bridges;
        found_path
    }

    /// Cascade removal of vertices whose all predecessors have been removed.
    fn propagate_removals(&mut self, removed_pred_size: &mut [usize]) {
        while let Some(v) = self.removed_queue.pop_front() {
            let num_edges = self.adj[v].len();
            for ei in 0..num_edges {
                let e = &self.adj[v][ei];
                if e.edge_type == EdgeType::Prop
                    && self.minlvl(e.to) > self.minlvl(v)
                    && !self.removed[e.to]
                {
                    let eto = e.to;
                    removed_pred_size[eto] += 1;
                    if removed_pred_size[eto] == self.predecessors[eto].len() {
                        self.remove_and_push(eto);
                    }
                }
            }
        }
    }

    // ────────────────────────────────────────────────────────────────────
    // DDFS (Double Depth-First Search)
    // ────────────────────────────────────────────────────────────────────

    /// Performs double DFS from bridge endpoints. Returns `(r, g)`:
    /// - If `r == g`: bottleneck (blossom found), `r` is the bud.
    /// - If `r != g`: two free vertices found (augmenting path).
    fn ddfs(&mut self, edge: (usize, usize), support: &mut Vec<usize>) -> (usize, usize) {
        let sr0 = self.bud.bud(edge.0);
        let sg0 = self.bud.bud(edge.1);
        if sr0 == sg0 {
            return (sr0, sg0);
        }

        let mut sr = vec![sr0];
        let mut sg = vec![sg0];
        support.push(sr0);
        support.push(sg0);

        self.global_color_counter += 1;
        let new_red = self.global_color_counter;
        self.global_color_counter += 1;
        let new_green = self.global_color_counter;
        debug_assert_eq!(new_red, new_green ^ 1);

        self.color[sr0] = new_red;
        self.color[sg0] = new_green;

        loop {
            if self.minlvl(*sr.last().unwrap()) == 0 && self.minlvl(*sg.last().unwrap()) == 0 {
                return (*sr.last().unwrap(), *sg.last().unwrap());
            }

            let b = if self.minlvl(*sr.last().unwrap()) >= self.minlvl(*sg.last().unwrap()) {
                self.ddfs_move(&mut sr, new_red, &mut sg, new_green, support)
            } else {
                self.ddfs_move(&mut sg, new_green, &mut sr, new_red, support)
            };

            if let Some(bottleneck) = b {
                return (bottleneck, bottleneck);
            }
        }
    }

    /// One step of DDFS: try to advance `stack1` downward. Returns
    /// `Some(bottleneck)` if a bottleneck is found.
    fn ddfs_move(
        &mut self,
        stack1: &mut Vec<usize>,
        color1: usize,
        stack2: &mut Vec<usize>,
        color2: usize,
        support: &mut Vec<usize>,
    ) -> Option<usize> {
        let u = *stack1.last().unwrap();

        while self.ddfs_pred_ptr[u] < self.predecessors[u].len() {
            let a = self.predecessors[u][self.ddfs_pred_ptr[u]];
            self.ddfs_pred_ptr[u] += 1;
            let v = self.bud.bud(a);
            debug_assert_eq!(self.removed[a], self.removed[v]);
            if self.removed[a] {
                continue;
            }
            if self.color[v] == 0 {
                stack1.push(v);
                support.push(v);
                self.children_in_ddfs_tree[u].push((a, v));
                self.color[v] = color1;
                return None;
            } else if v == *stack2.last().unwrap() {
                self.children_in_ddfs_tree[u].push((a, v));
            }
        }

        stack1.pop();

        if stack1.is_empty() {
            if stack2.len() == 1 {
                self.color[stack2[0]] = 0;
                return Some(stack2[0]);
            }
            let top = *stack2.last().unwrap();
            debug_assert_eq!(self.color[top], color2);
            stack1.push(top);
            self.color[top] = color1;
            stack2.pop();
        }

        None
    }

    // ────────────────────────────────────────────────────────────────────
    // Iterative augment_path and opening_dfs
    // ────────────────────────────────────────────────────────────────────

    /// Augments the matching along the path from `u` to `v`.
    /// `initial` is true for the top-level call from `bfs_phase`.
    ///
    /// Fully iterative rewrite of the mutually recursive `augumentPath` /
    /// `openingDfs` from the reference C++ implementation.
    fn augment_path(&mut self, u: usize, v: usize, initial: bool) {
        let mut stack: Vec<AugFrame> = vec![AugFrame::Augment { u, v, initial }];
        let mut opening_dfs_result: bool = false;

        while let Some(frame) = stack.pop() {
            match frame {
                AugFrame::Augment { u: au, v: av, initial: ai } => {
                    if au == av {
                        continue;
                    }
                    if !ai && self.minlvl(au) == self.evenlvl[au] {
                        self.augment_even(au, av, &mut stack);
                    } else {
                        self.augment_through_bridge(au, av, &mut stack);
                    }
                }
                AugFrame::OpeningDfs { cur, bcur, b, child_idx } => {
                    opening_dfs_result =
                        self.handle_opening_dfs(cur, bcur, b, child_idx, &mut stack);
                }
                AugFrame::HandleChildResult { cur, bcur, b, child_a, next_child_idx } => {
                    if opening_dfs_result {
                        // Child succeeded: commit the augment+flip.
                        // Push RestoreResult below Augment so it fires
                        // after any nested augmentation completes, ensuring
                        // the parent HandleChildResult sees success.
                        stack.push(AugFrame::RestoreResult);
                        stack.push(AugFrame::Augment { u: cur, v: bcur, initial: false });
                        self.flip(bcur, child_a);
                    } else {
                        // Child failed: try the next children.
                        opening_dfs_result =
                            self.handle_opening_dfs(cur, bcur, b, next_child_idx, &mut stack);
                    }
                }
                AugFrame::RestoreResult => {
                    opening_dfs_result = true;
                }
            }
        }
    }

    /// Even-level augmentation: follow the unique predecessor chain.
    fn augment_even(&mut self, au: usize, av: usize, stack: &mut Vec<AugFrame>) {
        assert_eq!(
            self.predecessors[au].len(),
            1,
            "MV invariant broken: even-level vertex {au} should have exactly one predecessor, found {}.",
            self.predecessors[au].len()
        );
        let x = self.predecessors[au][0];

        let bud_x = self.bud.bud(x);
        let mut next_u = None;
        for &candidate in &self.predecessors[x] {
            if self.bud.bud(candidate) == bud_x {
                next_u = Some(candidate);
                break;
            }
        }
        let next_u = next_u.unwrap_or_else(|| {
            panic!(
                "MV invariant broken: predecessor chain for vertex {x} could not be reopened inside bud {bud_x}."
            )
        });
        assert!(
            !self.removed[next_u],
            "MV invariant broken: attempted to augment through removed vertex {next_u}."
        );
        self.flip(x, next_u);

        stack.push(AugFrame::Augment { u: next_u, v: av, initial: false });
    }

    /// Odd-level / initial augmentation: go through the bridge.
    fn augment_through_bridge(&mut self, au: usize, av: usize, stack: &mut Vec<AugFrame>) {
        let ((mut u3, mut v3), (mut u2, mut v2)) = self.my_bridge[au];

        if (self.color[u2] ^ 1) == self.color[au] || self.color[v2] == self.color[au] {
            core::mem::swap(&mut u2, &mut v2);
            core::mem::swap(&mut u3, &mut v3);
        }

        self.flip(u3, v3);

        let v4 = self.bud.direct_parent[au].unwrap_or_else(|| {
            panic!(
                "MV invariant broken: blossom vertex {au} has no direct parent during bridge augmentation."
            )
        });

        // Push in reverse order for correct execution.
        stack.push(AugFrame::Augment { u: v4, v: av, initial: false });
        stack.push(AugFrame::OpeningDfs { cur: v3, bcur: v2, b: v4, child_idx: 0 });
        stack.push(AugFrame::OpeningDfs { cur: u3, bcur: u2, b: au, child_idx: 0 });
    }

    /// Handle one `OpeningDfs` frame. Returns the `opening_dfs_result` to
    /// propagate.
    fn handle_opening_dfs(
        &mut self,
        cur: usize,
        bcur: usize,
        b: usize,
        child_idx: usize,
        stack: &mut Vec<AugFrame>,
    ) -> bool {
        if bcur == b {
            stack.push(AugFrame::Augment { u: cur, v: bcur, initial: false });
            return true;
        }

        let len = self.children_in_ddfs_tree[bcur].len();
        for ci in child_idx..len {
            let (a, nd) = self.children_in_ddfs_tree[bcur][ci];
            if nd == b || self.color[nd] == self.color[bcur] {
                stack.push(AugFrame::HandleChildResult {
                    cur,
                    bcur,
                    b,
                    child_a: a,
                    next_child_idx: ci + 1,
                });
                stack.push(AugFrame::OpeningDfs { cur: a, bcur: nd, b, child_idx: 0 });
                return false;
            }
        }

        false
    }

    // ────────────────────────────────────────────────────────────────────
    // Result extraction
    // ────────────────────────────────────────────────────────────────────

    fn into_pairs(self) -> Vec<(M::Index, M::Index)> {
        let indices: Vec<M::Index> = self.matrix.row_indices().collect();
        crate::traits::algorithms::matching_utils::mate_to_pairs(&self.mate, &indices)
    }
}
