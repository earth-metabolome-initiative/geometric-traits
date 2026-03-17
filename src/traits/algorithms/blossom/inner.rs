//! Internal state and algorithm for the Edmonds blossom algorithm.
use alloc::vec::Vec;

use num_traits::AsPrimitive;

use crate::traits::SparseSquareMatrix;

/// State for a single run of the Edmonds blossom algorithm.
pub(super) struct BlossomState<'a, M: SparseSquareMatrix + ?Sized> {
    /// The adjacency matrix representing the graph.
    matrix: &'a M,
    /// Number of vertices.
    n: usize,
    /// Matched partner of each vertex, or `None` if exposed.
    mate: Vec<Option<M::Index>>,
    /// BFS tree parent of each vertex during the current augmenting path
    /// search.
    parent: Vec<Option<M::Index>>,
    /// Base vertex of the blossom containing each vertex.
    base: Vec<M::Index>,
    /// Whether each vertex is an outer (even-level) vertex in the current BFS
    /// tree.
    in_queue: Vec<bool>,
    /// Scratch array for LCA computation and blossom marking.
    used: Vec<bool>,
    /// BFS frontier of outer vertices.
    queue: Vec<M::Index>,
}

impl<'a, M: SparseSquareMatrix + ?Sized> BlossomState<'a, M> {
    /// Creates a new blossom state from a sparse square matrix.
    pub(super) fn new(matrix: &'a M) -> Self {
        let n: usize = matrix.order().as_();
        Self {
            matrix,
            n,
            mate: vec![None; n],
            parent: vec![None; n],
            base: matrix.row_indices().collect(),
            in_queue: vec![false; n],
            used: vec![false; n],
            queue: Vec::new(),
        }
    }

    /// Runs the algorithm and returns the matching as sorted pairs.
    pub(super) fn solve(mut self) -> Vec<(M::Index, M::Index)> {
        for v in self.matrix.row_indices() {
            if self.mate[v.as_()].is_none() {
                self.find_augmenting_path(v);
            }
        }
        self.into_pairs()
    }

    /// Converts the mate array into sorted `(u, v)` pairs with `u < v`.
    fn into_pairs(self) -> Vec<(M::Index, M::Index)> {
        let mut pairs = Vec::with_capacity(self.n / 2);
        for i in self.matrix.row_indices() {
            if let Some(j) = self.mate[i.as_()] {
                if i < j {
                    pairs.push((i, j));
                }
            }
        }
        pairs
    }

    /// Searches for an augmenting path from the given exposed root vertex.
    /// If found, augments the matching and returns `true`.
    fn find_augmenting_path(&mut self, root: M::Index) -> bool {
        for i in self.matrix.row_indices() {
            let idx = i.as_();
            self.parent[idx] = None;
            self.base[idx] = i;
            self.in_queue[idx] = false;
        }

        self.in_queue[root.as_()] = true;
        self.queue.clear();
        self.queue.push(root);
        let mut head = 0;

        while head < self.queue.len() {
            let v = self.queue[head];
            head += 1;

            for w in self.matrix.sparse_row(v) {
                // Skip self-loops and edges within the same contracted blossom.
                if self.base[v.as_()] == self.base[w.as_()] || w == v {
                    continue;
                }

                if self.mate[w.as_()].is_none() && w != root {
                    // Found an augmenting path ending at exposed vertex w.
                    self.parent[w.as_()] = Some(v);
                    self.augment_path(w);
                    return true;
                }

                if self.in_queue[self.base[w.as_()].as_()] {
                    // w (or its blossom base) is outer: odd cycle detected.
                    let lca = self.find_lca(v, w);
                    self.contract_blossom(v, w, lca);
                } else if self.parent[w.as_()].is_none() {
                    // w is unvisited and matched: extend the BFS tree.
                    self.parent[w.as_()] = Some(v);
                    let m = self.mate[w.as_()].unwrap();
                    self.in_queue[m.as_()] = true;
                    self.queue.push(m);
                }
            }
        }

        false
    }

    /// Finds the lowest common ancestor of `a` and `b` in the BFS tree,
    /// accounting for blossom contractions.
    fn find_lca(&mut self, a: M::Index, b: M::Index) -> M::Index {
        self.used.fill(false);

        let mut a = self.base[a.as_()];
        let mut b = self.base[b.as_()];

        // Walk a toward the root, marking each outer base vertex.
        loop {
            self.used[a.as_()] = true;
            match self.mate[a.as_()] {
                None => break,
                Some(ma) => {
                    a = self.base[self.parent[ma.as_()].unwrap().as_()];
                }
            }
        }

        // Walk b toward the root until we hit a marked vertex.
        loop {
            if self.used[b.as_()] {
                return b;
            }
            let mb = self.mate[b.as_()].unwrap();
            b = self.base[self.parent[mb.as_()].unwrap().as_()];
        }
    }

    /// Contracts a blossom rooted at `lca`, detected via the edge `(v, w)`.
    fn contract_blossom(&mut self, v: M::Index, w: M::Index, lca: M::Index) {
        self.used.fill(false);
        self.mark_path(v, lca, w);
        self.mark_path(w, lca, v);

        for i in self.matrix.row_indices() {
            if self.used[self.base[i.as_()].as_()] {
                self.base[i.as_()] = lca;
                if !self.in_queue[i.as_()] {
                    self.in_queue[i.as_()] = true;
                    self.queue.push(i);
                }
            }
        }
    }

    /// Marks vertices on the blossom path from `v` to `b`, updating parent
    /// pointers for correct augmentation through the blossom.
    fn mark_path(&mut self, mut v: M::Index, b: M::Index, mut child: M::Index) {
        while self.base[v.as_()] != b {
            self.used[self.base[v.as_()].as_()] = true;
            let mv = self.mate[v.as_()].unwrap();
            self.used[self.base[mv.as_()].as_()] = true;
            self.parent[v.as_()] = Some(child);
            child = mv;
            v = self.parent[mv.as_()].unwrap();
        }
    }

    /// Augments the matching along the alternating path ending at `u`.
    fn augment_path(&mut self, mut u: M::Index) {
        loop {
            let pv = self.parent[u.as_()].unwrap();
            let ppv = self.mate[pv.as_()];
            self.mate[u.as_()] = Some(pv);
            self.mate[pv.as_()] = Some(u);
            match ppv {
                Some(ppv) => u = ppv,
                None => return,
            }
        }
    }
}
