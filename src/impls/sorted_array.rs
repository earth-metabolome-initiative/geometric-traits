//! Module implementing traits for the Vec type.

use core::{
    iter::Cloned,
    ops::{Index, Range},
};

use crate::{
    errors::SortedError,
    traits::{Symbol, TransmuteFrom},
};

#[derive(Debug, Clone, Copy)]
/// Struct defining a sorted vector and its primary methods.
pub struct SortedArray<V, const N: usize> {
    array: [V; N],
}

impl<V, const N: usize> TryFrom<[V; N]> for SortedArray<V, N>
where
    V: Ord + Clone,
{
    type Error = SortedError<V>;

    fn try_from(array: [V; N]) -> Result<Self, Self::Error> {
        if array.is_sorted() {
            Ok(Self { array })
        } else {
            // We identify the offending entry by returning the first unsorted entry.
            let unsorted_entry = array.windows(2).find_map(|window| {
                if window[0] > window[1] { Some(window[1].clone()) } else { None }
            });
            if let Some(entry) = unsorted_entry {
                Err(SortedError::UnsortedEntry(entry))
            } else {
                unreachable!("The source vector is not sorted.");
            }
        }
    }
}

impl<V: Ord, const N: usize> TransmuteFrom<[V; N]> for SortedArray<V, N> {
    unsafe fn transmute_from(source: [V; N]) -> Self {
        debug_assert!(source.is_sorted(), "The source vector is not sorted.");
        Self { array: source }
    }
}

impl<V, Idx, const N: usize> Index<Idx> for SortedArray<V, N>
where
    [V; N]: Index<Idx>,
{
    type Output = <[V; N] as Index<Idx>>::Output;

    fn index(&self, index: Idx) -> &Self::Output {
        &self.array[index]
    }
}

impl<V, const N: usize> SortedArray<V, N> {
    #[must_use]
    /// Returns the entry at the provided index.
    pub fn get(&self, index: usize) -> Option<&V> {
        self.array.get(index)
    }

    #[must_use]
    /// Returns the length of the vector.
    pub fn len(&self) -> usize {
        self.array.len()
    }

    #[must_use]
    /// Returns whether the vector is empty.
    pub fn is_empty(&self) -> bool {
        self.array.is_empty()
    }

    /// Returns an iterator over the vector.
    pub fn iter(&self) -> core::slice::Iter<'_, V> {
        self.into_iter()
    }
}

impl<V: Ord, const N: usize> SortedArray<V, N> {
    #[must_use]
    /// Returns whether the vector is sorted.
    #[allow(clippy::unused_self)]
    pub fn is_sorted(&self) -> bool {
        true
    }
}

impl<'a, V, const N: usize> IntoIterator for &'a SortedArray<V, N> {
    type Item = &'a V;
    type IntoIter = core::slice::Iter<'a, V>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.array.iter()
    }
}

impl<V, const N: usize> AsRef<[V]> for SortedArray<V, N> {
    fn as_ref(&self) -> &[V] {
        &self.array
    }
}

impl<V: Symbol, const N: usize> crate::traits::Vocabulary for SortedArray<V, N> {
    type SourceSymbol = usize;
    type DestinationSymbol = V;
    type Sources<'a>
        = Range<usize>
    where
        Self: 'a;
    type Destinations<'a>
        = Cloned<core::slice::Iter<'a, Self::DestinationSymbol>>
    where
        Self: 'a;

    fn convert(&self, source: &Self::SourceSymbol) -> Option<Self::DestinationSymbol> {
        self.get(*source).cloned()
    }

    fn len(&self) -> usize {
        self.len()
    }

    fn sources(&self) -> Self::Sources<'_> {
        0..self.len()
    }

    fn destinations(&self) -> Self::Destinations<'_> {
        self.iter().cloned()
    }
}

impl<V: Symbol, const N: usize> crate::traits::VocabularyRef for SortedArray<V, N> {
    type DestinationRefs<'a>
        = core::slice::Iter<'a, Self::DestinationSymbol>
    where
        Self: 'a;

    fn convert_ref(&self, source: &Self::SourceSymbol) -> Option<&Self::DestinationSymbol> {
        self.get(*source)
    }

    fn destination_refs(&self) -> Self::DestinationRefs<'_> {
        self.iter()
    }
}

impl<V: Symbol + Ord, const N: usize> crate::traits::BidirectionalVocabulary for SortedArray<V, N> {
    fn invert(&self, destination: &Self::DestinationSymbol) -> Option<Self::SourceSymbol> {
        self.as_ref().binary_search(destination).ok()
    }
}

#[cfg(all(test, feature = "alloc"))]
mod tests {
    use alloc::vec::Vec;

    use super::*;
    use crate::traits::{BidirectionalVocabulary, Vocabulary, VocabularyRef};

    #[test]
    fn test_sorted_array_try_from_sorted() {
        let arr = [1, 2, 3, 4, 5];
        let sa: SortedArray<i32, 5> = SortedArray::try_from(arr).unwrap();
        assert_eq!(sa.len(), 5);
    }

    #[test]
    fn test_sorted_array_try_from_unsorted() {
        let arr = [1, 3, 2, 4];
        let result: Result<SortedArray<i32, 4>, _> = SortedArray::try_from(arr);
        assert!(result.is_err());
    }

    #[test]
    fn test_sorted_array_get() {
        let sa: SortedArray<i32, 3> = SortedArray::try_from([10, 20, 30]).unwrap();
        assert_eq!(sa.get(0), Some(&10));
        assert_eq!(sa.get(1), Some(&20));
        assert_eq!(sa.get(2), Some(&30));
        assert_eq!(sa.get(3), None);
    }

    #[test]
    fn test_sorted_array_len() {
        let sa: SortedArray<i32, 3> = SortedArray::try_from([1, 2, 3]).unwrap();
        assert_eq!(sa.len(), 3);
    }

    #[test]
    fn test_sorted_array_is_empty() {
        let sa: SortedArray<i32, 0> = SortedArray::try_from([]).unwrap();
        assert!(sa.is_empty());

        let sa2: SortedArray<i32, 1> = SortedArray::try_from([1]).unwrap();
        assert!(!sa2.is_empty());
    }

    #[test]
    fn test_sorted_array_is_sorted() {
        let sa: SortedArray<i32, 3> = SortedArray::try_from([1, 2, 3]).unwrap();
        assert!(sa.is_sorted());
    }

    #[test]
    fn test_sorted_array_iter() {
        let sa: SortedArray<i32, 3> = SortedArray::try_from([1, 2, 3]).unwrap();
        let items: Vec<&i32> = sa.iter().collect();
        assert_eq!(items, vec![&1, &2, &3]);
    }

    #[test]
    fn test_sorted_array_index() {
        let sa: SortedArray<i32, 3> = SortedArray::try_from([10, 20, 30]).unwrap();
        assert_eq!(sa[0], 10);
        assert_eq!(sa[1], 20);
        assert_eq!(sa[2], 30);
    }

    #[test]
    fn test_sorted_array_as_ref() {
        let sa: SortedArray<i32, 3> = SortedArray::try_from([1, 2, 3]).unwrap();
        let slice: &[i32] = sa.as_ref();
        assert_eq!(slice, &[1, 2, 3]);
    }

    #[test]
    fn test_sorted_array_vocabulary_convert() {
        let sa: SortedArray<i32, 3> = SortedArray::try_from([10, 20, 30]).unwrap();
        assert_eq!(Vocabulary::convert(&sa, &0), Some(10));
        assert_eq!(Vocabulary::convert(&sa, &1), Some(20));
        assert_eq!(Vocabulary::convert(&sa, &2), Some(30));
        assert_eq!(Vocabulary::convert(&sa, &3), None);
    }

    #[test]
    fn test_sorted_array_vocabulary_len() {
        let sa: SortedArray<i32, 3> = SortedArray::try_from([1, 2, 3]).unwrap();
        assert_eq!(Vocabulary::len(&sa), 3);
    }

    #[test]
    fn test_sorted_array_vocabulary_sources() {
        let sa: SortedArray<i32, 3> = SortedArray::try_from([10, 20, 30]).unwrap();
        let sources: Vec<usize> = sa.sources().collect();
        assert_eq!(sources, vec![0, 1, 2]);
    }

    #[test]
    fn test_sorted_array_vocabulary_destinations() {
        let sa: SortedArray<i32, 3> = SortedArray::try_from([10, 20, 30]).unwrap();
        let destinations: Vec<i32> = sa.destinations().collect();
        assert_eq!(destinations, vec![10, 20, 30]);
    }

    #[test]
    fn test_sorted_array_vocabulary_ref_convert_ref() {
        let sa: SortedArray<i32, 2> = SortedArray::try_from([100, 200]).unwrap();
        assert_eq!(sa.convert_ref(&0), Some(&100));
        assert_eq!(sa.convert_ref(&1), Some(&200));
        assert_eq!(sa.convert_ref(&2), None);
    }

    #[test]
    fn test_sorted_array_vocabulary_ref_destination_refs() {
        let sa: SortedArray<i32, 2> = SortedArray::try_from([1, 2]).unwrap();
        let refs: Vec<&i32> = sa.destination_refs().collect();
        assert_eq!(refs, vec![&1, &2]);
    }

    #[test]
    fn test_sorted_array_bidirectional_vocabulary_invert() {
        let sa: SortedArray<i32, 3> = SortedArray::try_from([10, 20, 30]).unwrap();
        assert_eq!(sa.invert(&10), Some(0));
        assert_eq!(sa.invert(&20), Some(1));
        assert_eq!(sa.invert(&30), Some(2));
        assert_eq!(sa.invert(&25), None);
    }
}
