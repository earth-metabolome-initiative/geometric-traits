//! Module implementing traits for the Vec type.

use core::{iter::Cloned, ops::Range};

use algebra::prelude::Symbol;

use crate::prelude::*;


use crate::error::Error;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
/// Struct defining a sorted vector and its primary methods.
pub struct SortedVec<V> {
    vec: Vec<V>,
}

impl<V> Default for SortedVec<V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<V> TryFrom<Vec<V>> for SortedVec<V>
where
    V: Ord + Clone,
{
    type Error = Error<V>;

    fn try_from(vec: Vec<V>) -> Result<Self, Self::Error> {
        if vec.is_sorted() {
            Ok(Self { vec })
        } else {
            // We identify the offending entry by returning the first unsorted entry.
            let unsorted_entry = vec.windows(2).find_map(|window| {
                if window[0] > window[1] { Some(window[1].clone()) } else { None }
            });
            if let Some(entry) = unsorted_entry {
                Err(Error::UnsortedEntry(entry))
            } else {
                unreachable!("The source vector is not sorted.");
            }
        }
    }
}

impl<V: Ord> TransmuteFrom<Vec<V>> for SortedVec<V> {
    unsafe fn transmute_from(source: Vec<V>) -> Self {
        debug_assert!(source.is_sorted(), "The source vector is not sorted.");
        Self { vec: source }
    }
}

impl<V, Idx> Index<Idx> for SortedVec<V>
where
    Vec<V>: Index<Idx>,
{
    type Output = <Vec<V> as Index<Idx>>::Output;

    fn index(&self, index: Idx) -> &Self::Output {
        &self.vec[index]
    }
}

impl<V> SortedVec<V> {
    #[must_use]
    /// Returns a new instance of the struct.
    pub fn new() -> Self {
        Self { vec: Vec::new() }
    }

    #[must_use]
    /// Returns a new instance of the struct with the provided capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self { vec: Vec::with_capacity(capacity) }
    }

    #[must_use]
    /// Returns the entry at the provided index.
    pub fn get(&self, index: usize) -> Option<&V> {
        self.vec.get(index)
    }

    #[must_use]
    /// Returns the length of the vector.
    pub fn len(&self) -> usize {
        self.vec.len()
    }

    #[must_use]
    /// Returns whether the vector is empty.
    pub fn is_empty(&self) -> bool {
        self.vec.is_empty()
    }

    /// Returns an iterator over the vector.
    pub fn iter(&self) -> core::slice::Iter<'_, V> {
        self.into_iter()
    }

    #[must_use]
    /// Returns a reference to the last node in the vector, if any.
    pub fn last(&self) -> Option<&V> {
        self.vec.last()
    }

    #[must_use]
    /// Returns a reference to the first node in the vector, if any.
    pub fn first(&self) -> Option<&V> {
        self.vec.first()
    }

    /// Binary searches this slice for a given element.
    /// If the slice is not sorted, the returned result is unspecified and
    /// meaningless.
    ///
    /// # Errors
    ///
    /// * `Err(usize)` if the value is not found, containing the index where it
    ///   could be inserted to maintain sorted order.
    /// * `Ok(usize)` if the value is found, containing the index of the value.
    pub fn binary_search(&self, value: &V) -> Result<usize, usize>
    where
        V: Ord,
    {
        self.vec.binary_search(value)
    }

    /// Binary searches this slice with a comparator function.
    /// If the slice is not sorted, the returned result is unspecified and
    /// meaningless.
    ///
    /// # Errors
    ///
    /// * `Err(usize)` if the value is not found, containing the index where it
    ///   could be inserted to maintain sorted order.
    /// * `Ok(usize)` if the value is found, containing the index of the value.
    pub fn binary_search_by<F>(&self, f: F) -> Result<usize, usize>
    where
        F: FnMut(&V) -> core::cmp::Ordering,
    {
        self.vec.binary_search_by(f)
    }
}

impl<V: PartialOrd> SortedVec<V> {
    #[must_use]
    /// Returns whether the vector is sorted.
    pub fn is_sorted(&self) -> bool {
        true
    }

    /// Attempts to push a value to the vector.
    ///
    /// # Arguments
    ///
    /// * `value`: The value to push.
    ///
    /// # Errors
    ///
    /// * `Error::UnsortedPush(v)` if the value is not sorted.
    pub fn push(&mut self, value: V) -> Result<(), Error<V>> {
        if self.last().is_some_and(|last| last >= &value) {
            Err(Error::UnsortedEntry(value))
        } else {
            self.vec.push(value);
            Ok(())
        }
    }
}

impl<'a, V> IntoIterator for &'a SortedVec<V> {
    type Item = &'a V;
    type IntoIter = core::slice::Iter<'a, V>;

    fn into_iter(self) -> Self::IntoIter {
        self.vec.iter()
    }
}

impl<V> AsRef<[V]> for SortedVec<V> {
    fn as_ref(&self) -> &[V] {
        &self.vec
    }
}


impl<V: Symbol> Vocabulary for SortedVec<V> {
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

impl<V: Symbol> VocabularyRef for SortedVec<V> {
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

impl<V: Symbol + Ord> BidirectionalVocabulary for SortedVec<V> {
    fn invert(&self, destination: &Self::DestinationSymbol) -> Option<Self::SourceSymbol> {
        self.as_ref().binary_search(destination).ok()
    }
}

impl<V: Symbol + Ord> GrowableVocabulary for SortedVec<V> {
    fn new() -> Self {
        SortedVec::new()
    }

    fn with_capacity(capacity: usize) -> Self {
        SortedVec::with_capacity(capacity)
    }

    fn add(
        &mut self,
        source: Self::SourceSymbol,
        destination: Self::DestinationSymbol,
    ) -> Result<(), crate::errors::builder::vocabulary::VocabularyBuilderError<Self>> {
        if source != self.len() {
            return Err(
                crate::errors::builder::vocabulary::VocabularyBuilderError::SparseSourceNode(
                    source,
                ),
            );
        }

        self.push(destination).map_err(|err| {
            match err {
                sorted_vec::error::Error::UnsortedEntry(destination) => {
                    crate::errors::builder::vocabulary::VocabularyBuilderError::UnorderedDestinationNode(destination)
                }
            }
        })?;

        Ok(())
    }
}
