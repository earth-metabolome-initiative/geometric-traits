//! Shared utilities for matching algorithms.
use alloc::vec::Vec;

/// Convert a usize-indexed mate array into sorted pairs `(u, v)` with `u < v`.
pub(crate) fn mate_to_pairs<I: Copy>(mate: &[Option<usize>], indices: &[I]) -> Vec<(I, I)> {
    let mut pairs = Vec::with_capacity(mate.len() / 2);
    for (i, slot) in mate.iter().enumerate() {
        if let Some(j) = *slot {
            if i < j {
                pairs.push((indices[i], indices[j]));
            }
        }
    }
    pairs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_mate() {
        let pairs: Vec<(usize, usize)> = mate_to_pairs(&[], &[] as &[usize]);
        assert!(pairs.is_empty());
    }

    #[test]
    fn test_all_none() {
        let mate = [None, None, None, None];
        let indices = [0usize, 1, 2, 3];
        let pairs = mate_to_pairs(&mate, &indices);
        assert!(pairs.is_empty());
    }

    #[test]
    fn test_single_pair() {
        let mate = [Some(1), Some(0)];
        let indices = [0usize, 1];
        let pairs = mate_to_pairs(&mate, &indices);
        assert_eq!(pairs, vec![(0, 1)]);
    }

    #[test]
    fn test_two_pairs() {
        // 0<->1, 2<->3
        let mate = [Some(1), Some(0), Some(3), Some(2)];
        let indices = [0usize, 1, 2, 3];
        let pairs = mate_to_pairs(&mate, &indices);
        assert_eq!(pairs, vec![(0, 1), (2, 3)]);
    }

    #[test]
    fn test_mixed_matched_and_exposed() {
        // 0 exposed, 1<->2, 3 exposed, 4<->5
        let mate = [None, Some(2), Some(1), None, Some(5), Some(4)];
        let indices = [0usize, 1, 2, 3, 4, 5];
        let pairs = mate_to_pairs(&mate, &indices);
        assert_eq!(pairs, vec![(1, 2), (4, 5)]);
    }

    #[test]
    fn test_non_identity_indices() {
        // Mate uses 0-based internal indices, but real IDs are offset.
        let mate = [Some(1), Some(0), Some(3), Some(2)];
        let indices = [10u32, 20, 30, 40];
        let pairs = mate_to_pairs(&mate, &indices);
        assert_eq!(pairs, vec![(10, 20), (30, 40)]);
    }

    #[test]
    fn test_odd_count_one_exposed() {
        // 0<->2, 1 exposed
        let mate = [Some(2), None, Some(0)];
        let indices = [0usize, 1, 2];
        let pairs = mate_to_pairs(&mate, &indices);
        assert_eq!(pairs, vec![(0, 2)]);
    }
}
