//! Disjoint-set forest (union-find) with union by rank and iterative path
//! compression. Shared by the minimum-cycle-basis and minimum-spanning-tree
//! algorithms. `find` is iterative so deep chains cannot overflow the stack.

use alloc::vec::Vec;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct UnionFind {
    parents: Vec<usize>,
    ranks: Vec<u8>,
}

impl UnionFind {
    /// Creates a forest of `size` singleton sets.
    pub(crate) fn new(size: usize) -> Self {
        Self { parents: (0..size).collect(), ranks: alloc::vec![0; size] }
    }

    /// Representative of `node`'s set, compressing the path to the root.
    pub(crate) fn find(&mut self, node: usize) -> usize {
        let mut root = node;
        while self.parents[root] != root {
            root = self.parents[root];
        }
        let mut current = node;
        while current != root {
            let next = self.parents[current];
            self.parents[current] = root;
            current = next;
        }
        root
    }

    /// Merges the sets of `left` and `right`. Returns `true` if they were
    /// distinct (a merge happened), `false` if already joined.
    pub(crate) fn union(&mut self, left: usize, right: usize) -> bool {
        let left_root = self.find(left);
        let right_root = self.find(right);
        if left_root == right_root {
            return false;
        }

        match self.ranks[left_root].cmp(&self.ranks[right_root]) {
            core::cmp::Ordering::Less => self.parents[left_root] = right_root,
            core::cmp::Ordering::Greater => self.parents[right_root] = left_root,
            core::cmp::Ordering::Equal => {
                self.parents[right_root] = left_root;
                self.ranks[left_root] += 1;
            }
        }

        true
    }
}
