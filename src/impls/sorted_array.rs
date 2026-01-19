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
