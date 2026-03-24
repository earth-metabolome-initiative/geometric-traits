//! Internal state and algorithm for Gabow's 1976 maximum matching algorithm.
use alloc::vec::Vec;

use num_traits::AsPrimitive;

use crate::traits::SparseSquareMatrix;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Label {
    Unlabeled,
    Start,
    Vertex(usize),
    Edge(usize, usize),
    Flag(usize, usize),
}

impl Label {
    #[inline]
    fn is_outer(self) -> bool {
        matches!(self, Self::Start | Self::Vertex(_) | Self::Edge(_, _))
    }
}

/// State for a single run of Gabow's 1976 implementation of Edmonds'
/// algorithm.
pub(super) struct Gabow1976State<'a, M: SparseSquareMatrix + ?Sized> {
    /// The adjacency matrix representing the graph.
    matrix: &'a M,
    /// Stable matrix indices in internal `usize` order.
    indices: Vec<M::Index>,
    /// Number of vertices in the graph.
    n: usize,
    /// Dummy sentinel vertex used for the paper's boundary conditions.
    dummy: usize,
    /// Matched partner of each vertex, or `dummy` if unmatched.
    mate: Vec<usize>,
    /// Current label state for the search.
    label: Vec<Label>,
    /// The first nonouter vertex on the current outer path.
    first: Vec<usize>,
    /// BFS queue of outer vertices.
    queue: Vec<usize>,
    /// Tracks whether a vertex has already been queued in the current search.
    queued: Vec<bool>,
}

impl<'a, M: SparseSquareMatrix + ?Sized> Gabow1976State<'a, M> {
    /// Creates a new solver state from a sparse square matrix.
    pub(super) fn new(matrix: &'a M) -> Self {
        let indices: Vec<M::Index> = matrix.row_indices().collect();
        let n = indices.len();
        let dummy = n;
        let mut mate = vec![dummy; n + 1];
        mate[dummy] = dummy;
        let mut first = vec![dummy; n + 1];
        first[dummy] = dummy;
        Self {
            matrix,
            indices,
            n,
            dummy,
            mate,
            label: vec![Label::Unlabeled; n + 1],
            first,
            queue: Vec::with_capacity(n),
            queued: vec![false; n + 1],
        }
    }

    /// Runs the algorithm and returns the matching as sorted pairs.
    pub(super) fn solve(mut self) -> Vec<(M::Index, M::Index)> {
        for root in 0..self.n {
            if self.mate[root] == self.dummy {
                let _ = self.find_augmenting_path(root);
            }
        }
        self.into_pairs()
    }

    /// Converts the mate array into sorted `(u, v)` pairs with `u < v`.
    fn into_pairs(self) -> Vec<(M::Index, M::Index)> {
        let usize_mate: Vec<Option<usize>> =
            self.mate.iter().take(self.n).map(|&mate| (mate < self.n).then_some(mate)).collect();
        crate::traits::algorithms::matching_utils::mate_to_pairs(&usize_mate, &self.indices)
    }

    /// Searches for an augmenting path rooted at the unmatched vertex `root`.
    fn find_augmenting_path(&mut self, root: usize) -> bool {
        self.label.fill(Label::Unlabeled);
        self.first.fill(self.dummy);
        self.first[self.dummy] = self.dummy;
        self.queued.fill(false);
        self.queue.clear();

        self.label[root] = Label::Start;
        self.first[root] = self.dummy;
        self.enqueue(root);

        let mut head = 0;
        while head < self.queue.len() {
            let x = self.queue[head];
            head += 1;

            for neighbor in self.matrix.sparse_row(self.indices[x]) {
                let y = neighbor.as_();
                if y == x {
                    continue;
                }

                if self.mate[y] == self.dummy && y != root {
                    self.mate[y] = x;
                    self.rematch(x, y);
                    return true;
                }

                if self.label[y].is_outer() {
                    self.relabel(x, y);
                    continue;
                }

                let v = self.mate[y];
                if v != self.dummy && !self.label[v].is_outer() {
                    self.label[v] = Label::Vertex(x);
                    self.first[v] = y;
                    self.enqueue(v);
                }
            }
        }

        false
    }

    /// Assigns an edge label to the blossom discovered via edge
    /// `(left_outer, right_outer)`.
    fn relabel(&mut self, left_outer: usize, right_outer: usize) {
        let mut left_first = self.first[left_outer];
        let mut right_first = self.first[right_outer];
        if left_first == right_first {
            return;
        }

        let flag = Label::Flag(left_outer, right_outer);
        self.label[left_first] = flag;
        self.label[right_first] = flag;

        let join = loop {
            if right_first != self.dummy {
                core::mem::swap(&mut left_first, &mut right_first);
            }

            let next_outer = self.anchor(self.label[self.mate[left_first]]);
            left_first = self.first[next_outer];
            if self.label[left_first] == flag {
                break left_first;
            }
            self.label[left_first] = flag;
        };

        for &path_start in &[self.first[left_outer], self.first[right_outer]] {
            let mut path_vertex = path_start;
            while path_vertex != join {
                self.label[path_vertex] = Label::Edge(left_outer, right_outer);
                self.first[path_vertex] = join;
                self.enqueue(path_vertex);

                let next_outer = self.anchor(self.label[self.mate[path_vertex]]);
                path_vertex = self.first[next_outer];
            }
        }

        for vertex in 0..self.n {
            if self.label[vertex].is_outer()
                && self.first[vertex] < self.n
                && self.label[self.first[vertex]].is_outer()
            {
                self.first[vertex] = join;
            }
        }
    }

    /// Recursively rematches the alternating path fragment described in the
    /// paper's `R` routine.
    fn rematch(&mut self, vertex: usize, partner: usize) {
        let previous_partner = self.mate[vertex];
        self.mate[vertex] = partner;
        if self.mate[previous_partner] != vertex {
            return;
        }

        match self.label[vertex] {
            Label::Vertex(parent_outer) => {
                self.mate[previous_partner] = parent_outer;
                self.rematch(parent_outer, previous_partner);
            }
            Label::Edge(left_outer, right_outer) => {
                self.rematch(left_outer, right_outer);
                self.rematch(right_outer, left_outer);
            }
            Label::Start | Label::Unlabeled | Label::Flag(_, _) => {}
        }
    }

    /// Enqueues an outer vertex once per search.
    #[inline]
    fn enqueue(&mut self, vertex: usize) {
        if vertex < self.n && !self.queued[vertex] {
            self.queued[vertex] = true;
            self.queue.push(vertex);
        }
    }

    /// Returns the anchor vertex used to advance `FIRST` through an outer
    /// label.
    #[inline]
    fn anchor(&self, label: Label) -> usize {
        match label {
            Label::Vertex(vertex) | Label::Edge(vertex, _) => vertex,
            Label::Start | Label::Unlabeled | Label::Flag(_, _) => self.dummy,
        }
    }
}
