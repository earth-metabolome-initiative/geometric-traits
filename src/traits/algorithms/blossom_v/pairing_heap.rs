//! Index-based pairing heap for use as priority queues in Blossom V.
//!
//! Each item is identified by an index into an external storage array.
//! The heap stores only structural links (parent, left-child, right-sibling).
//! Keys (slacks) are stored externally and accessed via the index.
//!
//! Supports O(1) merge, add, and the **global offset trick**: a per-heap
//! `offset` value is added to all stored keys conceptually, making bulk
//! updates O(1). When inserting, the key is stored as `key - offset`; when
//! reading, the effective key is `stored_key + offset`.
//!
//! Uses interleaved multipass for delete-min, following Kolmogorov's v2.0+.

/// Sentinel value indicating "no node".
const NONE: u32 = u32::MAX;

pub(crate) trait PQKeyStore {
    fn get_key(&self, idx: u32) -> i64;
    fn set_key(&mut self, idx: u32, key: i64);

    #[inline]
    fn less(&self, lhs: u32, rhs: u32) -> bool {
        self.get_key(lhs) < self.get_key(rhs)
    }

    #[inline]
    fn less_or_equal(&self, lhs: u32, rhs: u32) -> bool {
        !self.less(rhs, lhs)
    }

    #[inline]
    fn add_to_key(&mut self, idx: u32, delta: i64) {
        let key = self.get_key(idx) + delta;
        self.set_key(idx, key);
    }
}

impl PQKeyStore for [i64] {
    #[inline]
    fn get_key(&self, idx: u32) -> i64 {
        self[idx as usize]
    }

    #[inline]
    fn set_key(&mut self, idx: u32, key: i64) {
        self[idx as usize] = key;
    }
}

impl PQKeyStore for alloc::vec::Vec<i64> {
    #[inline]
    fn get_key(&self, idx: u32) -> i64 {
        self[idx as usize]
    }

    #[inline]
    fn set_key(&mut self, idx: u32, key: i64) {
        self[idx as usize] = key;
    }
}

/// Per-node links for the pairing heap. Stored in an external `Vec<PQNode>`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct PQNode {
    /// Parent node (self-referencing for root, NONE if not in any heap).
    pub parent: u32,
    /// Left child (first child in the child list).
    pub left: u32,
    /// Right sibling in the sibling list.
    pub right: u32,
}

impl PQNode {
    pub const RESET: Self = Self { parent: NONE, left: NONE, right: NONE };

    #[inline]
    pub fn is_in_heap(&self) -> bool {
        self.parent != NONE
    }
}

/// A pairing heap that stores indices into an external array.
///
/// Keys (slacks) are stored in a separate `&[i64]` or `&mut [i64]` array
/// and accessed by index. The heap maintains structural links in `nodes`.
///
/// The `offset` field implements the global offset trick: conceptually,
/// every key in the heap is shifted by `offset`. This makes bulk
/// `update(delta)` an O(1) operation.
#[derive(Clone, Debug)]
pub(crate) struct PairingHeap {
    root: u32,
    pub offset: i64,
}

impl PairingHeap {
    /// Creates a new empty heap.
    #[inline]
    pub fn new() -> Self {
        Self { root: NONE, offset: 0 }
    }

    /// Returns `true` if the heap is empty.
    #[cfg(test)]
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.root == NONE
    }

    /// Returns the index of the minimum-key item, or `None` if empty.
    #[inline]
    pub fn get_min(&self) -> Option<u32> {
        if self.root == NONE { None } else { Some(self.root) }
    }

    /// Returns the effective key of the minimum item: `keys[root] + offset`.
    #[cfg(test)]
    #[inline]
    pub fn get_min_key(&self, keys: &(impl PQKeyStore + ?Sized)) -> Option<i64> {
        if self.root == NONE { None } else { Some(keys.get_key(self.root) + self.offset) }
    }

    /// Adds item `i` to the heap. The item's key is `keys[i]`, but we store
    /// `keys[i] - offset` internally (by adjusting the key in-place).
    ///
    /// **Precondition:** `nodes[i]` must not be in any heap (`parent == NONE`).
    #[inline]
    pub fn add(&mut self, i: u32, keys: &mut (impl PQKeyStore + ?Sized), nodes: &mut [PQNode]) {
        debug_assert!(!nodes[i as usize].is_in_heap());
        // Adjust key to account for current offset
        keys.add_to_key(i, -self.offset);
        nodes[i as usize].left = NONE;
        nodes[i as usize].right = NONE;

        if self.root == NONE {
            self.root = i;
            nodes[i as usize].parent = i; // self-referencing = root
        } else if keys.less_or_equal(i, self.root) {
            nodes[self.root as usize].parent = i;
            nodes[i as usize].left = self.root;
            nodes[i as usize].right = NONE;
            self.root = i;
            nodes[i as usize].parent = i;
        } else {
            nodes[i as usize].left = NONE;
            nodes[i as usize].right = nodes[self.root as usize].left;
            if nodes[i as usize].right != NONE {
                nodes[nodes[i as usize].right as usize].parent = i;
            }
            nodes[self.root as usize].left = i;
            nodes[i as usize].parent = self.root;
        }
    }

    /// Removes item `i` from the heap.
    ///
    /// After removal, `keys[i]` is restored to the effective key
    /// (`stored + offset`) and `nodes[i].parent` is set to `NONE`.
    #[inline]
    pub fn remove(&mut self, i: u32, keys: &mut (impl PQKeyStore + ?Sized), nodes: &mut [PQNode]) {
        debug_assert!(nodes[i as usize].is_in_heap());
        let iu = i as usize;

        if nodes[iu].parent == i {
            // i is the root
            self.remove_root(keys, nodes);
        } else {
            // Detach i from its parent
            let p = nodes[iu].parent;
            if nodes[iu].right != NONE {
                nodes[nodes[iu].right as usize].parent = p;
            }
            if nodes[p as usize].left == i {
                nodes[p as usize].left = nodes[iu].right;
            } else {
                nodes[p as usize].right = nodes[iu].right;
            }
            // Merge i's children back into the heap
            if nodes[iu].left != NONE {
                let child_tree = self.multipass_merge(nodes[iu].left, keys, nodes);
                self.root = Self::link(self.root, child_tree, keys, nodes);
            }
        }

        // Restore effective key and mark as removed
        keys.add_to_key(i, self.offset);
        nodes[iu].parent = NONE;
        nodes[iu].left = NONE;
        nodes[iu].right = NONE;
    }

    /// Decrease the key of item `i` to `new_effective_key`.
    ///
    /// The new key must be ≤ the current effective key.
    #[cfg(test)]
    #[inline]
    pub fn decrease_key(
        &mut self,
        i: u32,
        new_effective_key: i64,
        keys: &mut (impl PQKeyStore + ?Sized),
        nodes: &mut [PQNode],
    ) {
        let iu = i as usize;
        debug_assert!(nodes[iu].is_in_heap());
        let new_stored = new_effective_key - self.offset;
        debug_assert!(new_stored <= keys.get_key(i));
        keys.set_key(i, new_stored);

        if nodes[iu].parent == i {
            // Already the root, just update the key
            return;
        }

        // Cut i from its parent
        let p = nodes[iu].parent;
        if nodes[iu].right != NONE {
            nodes[nodes[iu].right as usize].parent = p;
        }
        if nodes[p as usize].left == i {
            nodes[p as usize].left = nodes[iu].right;
        } else {
            nodes[p as usize].right = nodes[iu].right;
        }
        nodes[iu].right = NONE;

        // Re-link with root
        nodes[iu].parent = i; // temporary self-ref
        self.root = Self::link(self.root, i, keys, nodes);
    }

    /// Adds `delta` to all keys conceptually. O(1).
    #[cfg(test)]
    #[inline]
    pub fn update(&mut self, delta: i64) {
        self.offset += delta;
    }

    /// Destructively merges `other` into `self`. After this, `other` is empty.
    #[cfg(test)]
    #[inline]
    pub fn merge(
        &mut self,
        other: &mut Self,
        keys: &mut (impl PQKeyStore + ?Sized),
        nodes: &mut [PQNode],
    ) {
        if other.root == NONE {
            return;
        }
        // Adjust keys in `other` to account for offset difference
        if other.offset != self.offset {
            let delta = other.offset - self.offset;
            Self::adjust_all_keys(other.root, delta, keys, nodes);
        }
        if self.root == NONE {
            self.root = other.root;
        } else {
            if keys.less(other.root, self.root) {
                core::mem::swap(&mut self.root, &mut other.root);
            }
            nodes[other.root as usize].right = nodes[self.root as usize].left;
            if nodes[other.root as usize].right != NONE {
                nodes[nodes[other.root as usize].right as usize].parent = other.root;
            }
            nodes[other.root as usize].parent = self.root;
            nodes[self.root as usize].left = other.root;
        }
        other.root = NONE;
        other.offset = 0;
    }

    /// Iterate all items in the heap (unordered). Calls `f(index)` for each.
    #[cfg(test)]
    #[inline]
    pub fn for_each(&self, nodes: &[PQNode], mut f: impl FnMut(u32)) {
        if self.root == NONE {
            return;
        }
        Self::for_each_recursive(self.root, nodes, &mut f);
    }

    /// Removes all items from the heap, calling `f(index, effective_key)` for
    /// each and resetting their node links.
    #[cfg(test)]
    #[inline]
    pub fn drain(
        &mut self,
        keys: &mut (impl PQKeyStore + ?Sized),
        nodes: &mut [PQNode],
        mut f: impl FnMut(u32, i64),
    ) {
        if self.root == NONE {
            return;
        }
        Self::drain_recursive(self.root, self.offset, keys, nodes, &mut f);
        self.root = NONE;
        self.offset = 0;
    }

    // ===== Internal helpers =====

    /// Links two heap roots. The one with the smaller key becomes the new root.
    /// Returns the new root index.
    #[inline]
    fn link(a: u32, b: u32, keys: &(impl PQKeyStore + ?Sized), nodes: &mut [PQNode]) -> u32 {
        debug_assert_ne!(a, NONE);
        debug_assert_ne!(b, NONE);
        if keys.less_or_equal(a, b) {
            // a becomes root, b becomes child of a
            nodes[b as usize].right = nodes[a as usize].left;
            if nodes[b as usize].right != NONE {
                nodes[nodes[b as usize].right as usize].parent = b;
            }
            nodes[b as usize].parent = a;
            nodes[a as usize].left = b;
            nodes[a as usize].parent = a; // self-ref for root
            a
        } else {
            // b becomes root, a becomes child of b
            nodes[a as usize].right = nodes[b as usize].left;
            if nodes[a as usize].right != NONE {
                nodes[nodes[a as usize].right as usize].parent = a;
            }
            nodes[a as usize].parent = b;
            nodes[b as usize].left = a;
            nodes[b as usize].parent = b; // self-ref for root
            b
        }
    }

    /// Interleaved multipass merge of a sibling list starting at `first`.
    /// Returns the single remaining root.
    #[allow(clippy::if_not_else, clippy::unused_self)]
    fn multipass_merge(
        &self,
        first: u32,
        keys: &mut (impl PQKeyStore + ?Sized),
        nodes: &mut [PQNode],
    ) -> u32 {
        let mut i = first;

        // Keep pairing until one remains
        while nodes[i as usize].right != NONE {
            let mut prev: u32 = NONE;
            let mut cur = i;

            // Forward pass: pair adjacent siblings
            while cur != NONE {
                let next;
                if nodes[cur as usize].right != NONE {
                    let j = nodes[cur as usize].right;
                    next = nodes[j as usize].right;
                    let merged = Self::link(cur, j, keys, nodes);
                    nodes[merged as usize].right = prev;
                    prev = merged;
                    cur = next;
                } else {
                    nodes[cur as usize].right = prev;
                    prev = cur;
                    cur = NONE;
                }
            }

            i = prev;
        }

        nodes[i as usize].parent = i; // mark as root
        i
    }

    /// Remove the root and merge its children via multipass.
    #[inline]
    fn remove_root(&mut self, keys: &mut (impl PQKeyStore + ?Sized), nodes: &mut [PQNode]) {
        let old_root = self.root;
        debug_assert_ne!(old_root, NONE);
        let child = nodes[old_root as usize].left;

        if child == NONE {
            self.root = NONE;
        } else {
            self.root = self.multipass_merge(child, keys, nodes);
        }

        nodes[old_root as usize].parent = NONE;
    }

    /// Recursively adjust all keys by `delta` (used when merging heaps
    /// with different offsets).
    #[cfg(test)]
    fn adjust_all_keys(
        i: u32,
        delta: i64,
        keys: &mut (impl PQKeyStore + ?Sized),
        nodes: &[PQNode],
    ) {
        if i == NONE {
            return;
        }
        keys.add_to_key(i, delta);
        Self::adjust_all_keys(nodes[i as usize].left, delta, keys, nodes);
        Self::adjust_all_keys(nodes[i as usize].right, delta, keys, nodes);
    }

    #[cfg(test)]
    fn for_each_recursive(i: u32, nodes: &[PQNode], f: &mut impl FnMut(u32)) {
        if i == NONE {
            return;
        }
        f(i);
        Self::for_each_recursive(nodes[i as usize].left, nodes, f);
        Self::for_each_recursive(nodes[i as usize].right, nodes, f);
    }

    #[cfg(test)]
    fn drain_recursive(
        i: u32,
        offset: i64,
        keys: &mut (impl PQKeyStore + ?Sized),
        nodes: &mut [PQNode],
        f: &mut impl FnMut(u32, i64),
    ) {
        if i == NONE {
            return;
        }
        let left = nodes[i as usize].left;
        let right = nodes[i as usize].right;
        keys.add_to_key(i, offset);
        f(i, keys.get_key(i));
        nodes[i as usize] = PQNode::RESET;
        Self::drain_recursive(left, offset, keys, nodes, f);
        Self::drain_recursive(right, offset, keys, nodes, f);
    }
}

impl Default for PairingHeap {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
#[allow(clippy::pedantic, clippy::useless_vec)]
mod tests {
    use alloc::{vec, vec::Vec};

    use super::*;

    fn make_heap_with_items(items: &[(u32, i64)]) -> (PairingHeap, Vec<i64>, Vec<PQNode>) {
        let max_idx = items.iter().map(|&(i, _)| i).max().unwrap_or(0) as usize + 1;
        let mut keys = vec![0i64; max_idx];
        let mut nodes = vec![PQNode::RESET; max_idx];
        let mut heap = PairingHeap::new();
        for &(i, k) in items {
            keys[i as usize] = k;
            heap.add(i, &mut keys, &mut nodes);
        }
        (heap, keys, nodes)
    }

    #[test]
    fn test_empty_heap() {
        let heap = PairingHeap::new();
        assert!(heap.is_empty());
        assert_eq!(heap.get_min(), None);
    }

    #[test]
    fn test_default_heap_matches_new() {
        let heap = PairingHeap::default();
        assert!(heap.is_empty());
        assert_eq!(heap.get_min(), None);
    }

    #[test]
    fn test_pq_key_store_slice_and_vec_impls() {
        let mut vec_keys = vec![5, 9];
        assert_eq!(PQKeyStore::get_key(&vec_keys, 1), 9);
        PQKeyStore::set_key(&mut vec_keys, 0, 7);
        assert_eq!(vec_keys[0], 7);

        let slice_keys: &mut [i64] = vec_keys.as_mut_slice();
        assert_eq!(PQKeyStore::get_key(slice_keys, 0), 7);
        PQKeyStore::set_key(slice_keys, 1, 11);
        assert_eq!(slice_keys[1], 11);
    }

    #[test]
    fn test_add_single() {
        let (heap, keys, _nodes) = make_heap_with_items(&[(0, 42)]);
        assert!(!heap.is_empty());
        assert_eq!(heap.get_min(), Some(0));
        assert_eq!(heap.get_min_key(&keys), Some(42));
    }

    #[test]
    fn test_add_multiple_min_correct() {
        let (heap, keys, _nodes) = make_heap_with_items(&[(0, 10), (1, 5), (2, 20), (3, 3)]);
        assert_eq!(heap.get_min(), Some(3));
        assert_eq!(heap.get_min_key(&keys), Some(3));
    }

    #[test]
    fn test_add_equal_key_new_item_becomes_root() {
        let (heap, keys, _nodes) = make_heap_with_items(&[(0, 5), (1, 5)]);
        assert_eq!(heap.get_min(), Some(1));
        assert_eq!(heap.get_min_key(&keys), Some(5));
    }

    #[test]
    fn test_remove_min() {
        let (mut heap, mut keys, mut nodes) =
            make_heap_with_items(&[(0, 10), (1, 5), (2, 20), (3, 3)]);
        // Remove min (3)
        heap.remove(3, &mut keys, &mut nodes);
        assert_eq!(heap.get_min(), Some(1));
        assert_eq!(heap.get_min_key(&keys), Some(5));
        // Remove min (1)
        heap.remove(1, &mut keys, &mut nodes);
        assert_eq!(heap.get_min_key(&keys), Some(10));
        // Remove min (0)
        heap.remove(0, &mut keys, &mut nodes);
        assert_eq!(heap.get_min_key(&keys), Some(20));
        // Remove last
        heap.remove(2, &mut keys, &mut nodes);
        assert!(heap.is_empty());
    }

    #[test]
    fn test_remove_non_min() {
        let (mut heap, mut keys, mut nodes) = make_heap_with_items(&[(0, 10), (1, 5), (2, 20)]);
        // Remove non-min (2)
        heap.remove(2, &mut keys, &mut nodes);
        assert_eq!(heap.get_min_key(&keys), Some(5));
        // Verify effective key restored
        assert_eq!(keys[2], 20);
        assert!(!nodes[2].is_in_heap());
    }

    #[test]
    fn test_decrease_key() {
        let (mut heap, mut keys, mut nodes) = make_heap_with_items(&[(0, 10), (1, 5), (2, 20)]);
        assert_eq!(heap.get_min(), Some(1));
        // Decrease key of 2 from 20 to 1
        heap.decrease_key(2, 1, &mut keys, &mut nodes);
        assert_eq!(heap.get_min(), Some(2));
        assert_eq!(heap.get_min_key(&keys), Some(1));
    }

    #[test]
    fn test_decrease_key_root() {
        let (mut heap, mut keys, mut nodes) = make_heap_with_items(&[(0, 10), (1, 5)]);
        // Decrease root's key
        heap.decrease_key(1, 2, &mut keys, &mut nodes);
        assert_eq!(heap.get_min(), Some(1));
        assert_eq!(heap.get_min_key(&keys), Some(2));
    }

    #[test]
    fn test_offset_update() {
        let (mut heap, keys, _nodes) = make_heap_with_items(&[(0, 10), (1, 5)]);
        heap.update(100);
        assert_eq!(heap.get_min_key(&keys), Some(105));
        heap.update(-50);
        assert_eq!(heap.get_min_key(&keys), Some(55));
    }

    #[test]
    fn test_add_after_offset() {
        let (mut heap, mut keys, mut nodes) = make_heap_with_items(&[(0, 10)]);
        heap.update(100); // effective key of 0 is now 110
        // Add item 1 with effective key 50
        keys.push(50); // index 1
        nodes.push(PQNode::RESET);
        heap.add(1, &mut keys, &mut nodes);
        // Min should be 1 (effective key 50 < 110)
        assert_eq!(heap.get_min(), Some(1));
        assert_eq!(heap.get_min_key(&keys), Some(50));
    }

    #[test]
    fn test_merge_two_heaps() {
        let (mut h1, mut keys, mut nodes) = make_heap_with_items(&[(0, 10), (1, 5)]);
        let mut h2 = PairingHeap::new();
        keys.push(2); // index 2
        nodes.push(PQNode::RESET);
        keys.push(8); // index 3
        nodes.push(PQNode::RESET);
        h2.add(2, &mut keys, &mut nodes);
        h2.add(3, &mut keys, &mut nodes);

        h1.merge(&mut h2, &mut keys, &mut nodes);
        assert!(h2.is_empty());
        assert_eq!(h1.get_min(), Some(2)); // key 2
        assert_eq!(h1.get_min_key(&keys), Some(2));
    }

    #[test]
    fn test_merge_equal_key_keeps_destination_root() {
        let (mut h1, mut keys, mut nodes) = make_heap_with_items(&[(0, 5)]);
        let mut h2 = PairingHeap::new();
        keys.push(5);
        nodes.push(PQNode::RESET);
        h2.add(1, &mut keys, &mut nodes);
        h1.merge(&mut h2, &mut keys, &mut nodes);
        assert_eq!(h1.get_min(), Some(0));
        assert_eq!(h1.get_min_key(&keys), Some(5));
    }

    #[test]
    fn test_merge_with_different_offsets() {
        let (mut h1, mut keys, mut nodes) = make_heap_with_items(&[(0, 10), (1, 5)]);
        h1.update(100); // h1 keys: 110, 105

        let mut h2 = PairingHeap::new();
        keys.push(50); // index 2
        nodes.push(PQNode::RESET);
        h2.add(2, &mut keys, &mut nodes);
        h2.update(200); // h2 key: 250

        h1.merge(&mut h2, &mut keys, &mut nodes);
        // Min should still be item 1 with effective key 105
        assert_eq!(h1.get_min(), Some(1));
        assert_eq!(h1.get_min_key(&keys), Some(105));
    }

    #[test]
    fn test_merge_ignores_empty_other_heap() {
        let (mut h1, mut keys, mut nodes) = make_heap_with_items(&[(0, 10), (1, 5)]);
        let mut h2 = PairingHeap::new();

        h1.merge(&mut h2, &mut keys, &mut nodes);

        assert_eq!(h1.get_min(), Some(1));
        assert_eq!(h1.get_min_key(&keys), Some(5));
        assert!(h2.is_empty());
    }

    #[test]
    fn test_merge_into_empty_heap_takes_other_root() {
        let mut h1 = PairingHeap::new();
        let (mut h2, mut keys, mut nodes) = make_heap_with_items(&[(0, 8), (1, 3)]);

        h1.merge(&mut h2, &mut keys, &mut nodes);

        assert_eq!(h1.get_min(), Some(1));
        assert_eq!(h1.get_min_key(&keys), Some(3));
        assert!(h2.is_empty());
    }

    #[test]
    fn test_for_each() {
        let (heap, _keys, nodes) = make_heap_with_items(&[(0, 10), (1, 5), (2, 20)]);
        let mut visited = vec![false; 3];
        heap.for_each(&nodes, |i| visited[i as usize] = true);
        assert!(visited.iter().all(|&v| v));
    }

    #[test]
    fn test_for_each_empty_heap_does_not_visit_anything() {
        let heap = PairingHeap::new();
        let nodes = vec![PQNode::RESET; 1];
        let mut visits = 0;
        heap.for_each(&nodes, |_| visits += 1);
        assert_eq!(visits, 0);
    }

    #[test]
    fn test_drain() {
        let (mut heap, mut keys, mut nodes) = make_heap_with_items(&[(0, 10), (1, 5), (2, 20)]);
        heap.update(100);
        let mut drained = Vec::new();
        heap.drain(&mut keys, &mut nodes, |i, k| drained.push((i, k)));
        assert!(heap.is_empty());
        drained.sort_by_key(|&(i, _)| i);
        assert_eq!(drained, vec![(0, 110), (1, 105), (2, 120)]);
        // All nodes should be reset
        assert!(nodes.iter().all(|n| !n.is_in_heap()));
    }

    #[test]
    fn test_drain_empty_heap_is_noop() {
        let mut heap = PairingHeap::new();
        let mut keys = vec![1i64];
        let mut nodes = vec![PQNode::RESET; 1];
        let mut drained = Vec::new();

        heap.drain(&mut keys, &mut nodes, |i, k| drained.push((i, k)));

        assert!(drained.is_empty());
        assert!(heap.is_empty());
        assert_eq!(keys, vec![1]);
        assert!(nodes.iter().all(|n| !n.is_in_heap()));
    }

    #[test]
    fn test_remove_restores_effective_key() {
        let (mut heap, mut keys, mut nodes) = make_heap_with_items(&[(0, 10)]);
        heap.update(50); // effective key 60
        heap.remove(0, &mut keys, &mut nodes);
        assert_eq!(keys[0], 60); // restored to effective key
    }

    #[test]
    fn test_stress_ascending_descending() {
        let n = 100u32;
        let mut keys = vec![0i64; n as usize];
        let mut nodes = vec![PQNode::RESET; n as usize];
        let mut heap = PairingHeap::new();

        // Add in ascending order
        for i in 0..n {
            keys[i as usize] = i as i64;
            heap.add(i, &mut keys, &mut nodes);
        }
        assert_eq!(heap.get_min_key(&keys), Some(0));

        // Remove all in order
        for expected in 0..n {
            let min = heap.get_min().unwrap();
            let min_key = heap.get_min_key(&keys).unwrap();
            assert_eq!(min_key, expected as i64);
            heap.remove(min, &mut keys, &mut nodes);
        }
        assert!(heap.is_empty());
    }

    #[test]
    fn test_stress_random_ops() {
        // Simple deterministic pseudo-random sequence
        let mut rng: u64 = 12345;
        let next = |rng: &mut u64| -> u64 {
            *rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            *rng >> 33
        };

        let n = 50u32;
        let mut keys = vec![0i64; n as usize];
        let mut nodes = vec![PQNode::RESET; n as usize];
        let mut heap = PairingHeap::new();
        let mut in_heap = vec![false; n as usize];

        for _ in 0..500 {
            let op = next(&mut rng) % 4;
            match op {
                0 => {
                    // Add random item not in heap
                    let i = (next(&mut rng) % n as u64) as u32;
                    if !in_heap[i as usize] {
                        keys[i as usize] = (next(&mut rng) % 1000) as i64;
                        heap.add(i, &mut keys, &mut nodes);
                        in_heap[i as usize] = true;
                    }
                }
                1 => {
                    // Remove min
                    if let Some(min) = heap.get_min() {
                        heap.remove(min, &mut keys, &mut nodes);
                        in_heap[min as usize] = false;
                    }
                }
                2 => {
                    // Decrease key of random item in heap
                    let i = (next(&mut rng) % n as u64) as u32;
                    if in_heap[i as usize] {
                        let cur = keys[i as usize] + heap.offset;
                        let new_key = cur - (next(&mut rng) % 100) as i64;
                        heap.decrease_key(i, new_key, &mut keys, &mut nodes);
                    }
                }
                _ => {
                    // Update offset
                    heap.update((next(&mut rng) % 10) as i64);
                }
            }

            // Verify min invariant
            if let Some(min) = heap.get_min() {
                let min_eff = keys[min as usize] + heap.offset;
                for (i, &present) in in_heap.iter().enumerate() {
                    if present {
                        let eff = keys[i] + heap.offset;
                        assert!(
                            min_eff <= eff,
                            "min invariant violated: min={min_eff} at {min}, but {eff} at {i}"
                        );
                    }
                }
            }
        }
    }
}
