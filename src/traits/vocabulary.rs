//! A trait defining a conversion between a source symbol and a destination
//! symbol.

#[cfg(feature = "alloc")]
use alloc::{boxed::Box, rc::Rc};
use core::fmt::Debug;

use crate::{errors::builder::vocabulary::VocabularyBuilderError, traits::Symbol};

/// Trait defining a conversion between a source symbol and a destination
/// symbol.
///
/// # Examples
///
/// ```
/// use geometric_traits::prelude::*;
///
/// // Vec<T> implements Vocabulary where indices are source symbols
/// let vocab: Vec<&str> = vec!["apple", "banana", "cherry"];
///
/// assert_eq!(vocab.len(), 3);
/// assert!(!vocab.is_empty());
///
/// // Convert index to value
/// assert_eq!(vocab.convert(&0), Some("apple"));
/// assert_eq!(vocab.convert(&1), Some("banana"));
/// assert_eq!(vocab.convert(&3), None); // Out of bounds
///
/// // Iterate over sources (indices)
/// let sources: Vec<_> = vocab.sources().collect();
/// assert_eq!(sources, vec![0, 1, 2]);
///
/// // Iterate over destinations (values)
/// let destinations: Vec<_> = vocab.destinations().collect();
/// assert_eq!(destinations, vec!["apple", "banana", "cherry"]);
/// ```
pub trait Vocabulary: Debug {
    /// The source symbol.
    type SourceSymbol: Symbol;
    /// The destination symbol.
    type DestinationSymbol: Symbol;
    /// Iterator over the source symbols.
    type Sources<'a>: Iterator<Item = Self::SourceSymbol>
    where
        Self: 'a;
    /// Iterator over the destination symbols.
    type Destinations<'a>: Iterator<Item = Self::DestinationSymbol>
    where
        Self: 'a;

    /// Converts a source symbol into a destination symbol.
    fn convert(&self, source: &Self::SourceSymbol) -> Option<Self::DestinationSymbol>;

    /// Returns whether the vocabulary is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the number of entries in the vocabulary.
    fn len(&self) -> usize;

    /// Returns an iterator over the source symbols.
    fn sources(&self) -> Self::Sources<'_>;

    /// Returns an iterator over the destination symbols.
    fn destinations(&self) -> Self::Destinations<'_>;
}

impl<V: Vocabulary + ?Sized> Vocabulary for &V {
    type SourceSymbol = V::SourceSymbol;
    type DestinationSymbol = V::DestinationSymbol;
    type Sources<'a>
        = V::Sources<'a>
    where
        Self: 'a;
    type Destinations<'a>
        = V::Destinations<'a>
    where
        Self: 'a;

    fn convert(&self, source: &Self::SourceSymbol) -> Option<Self::DestinationSymbol> {
        (*self).convert(source)
    }

    fn len(&self) -> usize {
        (*self).len()
    }

    fn sources(&self) -> Self::Sources<'_> {
        (*self).sources()
    }

    fn destinations(&self) -> Self::Destinations<'_> {
        (*self).destinations()
    }
}

#[cfg(feature = "alloc")]
impl<V: Vocabulary + ?Sized> Vocabulary for Box<V> {
    type SourceSymbol = V::SourceSymbol;
    type DestinationSymbol = V::DestinationSymbol;
    type Sources<'a>
        = V::Sources<'a>
    where
        Self: 'a;
    type Destinations<'a>
        = V::Destinations<'a>
    where
        Self: 'a;

    fn convert(&self, source: &Self::SourceSymbol) -> Option<Self::DestinationSymbol> {
        (**self).convert(source)
    }

    fn len(&self) -> usize {
        (**self).len()
    }

    fn sources(&self) -> Self::Sources<'_> {
        (**self).sources()
    }

    fn destinations(&self) -> Self::Destinations<'_> {
        (**self).destinations()
    }
}

#[cfg(feature = "alloc")]
impl<V: Vocabulary + ?Sized> Vocabulary for Rc<V> {
    type SourceSymbol = V::SourceSymbol;
    type DestinationSymbol = V::DestinationSymbol;
    type Sources<'a>
        = V::Sources<'a>
    where
        Self: 'a;
    type Destinations<'a>
        = V::Destinations<'a>
    where
        Self: 'a;

    fn convert(&self, source: &Self::SourceSymbol) -> Option<Self::DestinationSymbol> {
        (**self).convert(source)
    }

    fn len(&self) -> usize {
        (**self).len()
    }

    fn sources(&self) -> Self::Sources<'_> {
        (**self).sources()
    }

    fn destinations(&self) -> Self::Destinations<'_> {
        (**self).destinations()
    }
}

/// Trait defining a conversion between a source symbol and a destination symbol
/// reference.
pub trait VocabularyRef: Vocabulary {
    /// Iterator over the references of the destination symbols.
    type DestinationRefs<'a>: Iterator<Item = &'a Self::DestinationSymbol>
    where
        Self: 'a;

    /// Converts a source symbol into a destination symbol.
    fn convert_ref(&self, source: &Self::SourceSymbol) -> Option<&Self::DestinationSymbol>;

    /// Returns an iterator over the references of the destination symbols.
    fn destination_refs(&self) -> Self::DestinationRefs<'_>;
}

impl<V: VocabularyRef + ?Sized> VocabularyRef for &V {
    type DestinationRefs<'a>
        = V::DestinationRefs<'a>
    where
        Self: 'a;

    fn convert_ref(&self, source: &Self::SourceSymbol) -> Option<&Self::DestinationSymbol> {
        (*self).convert_ref(source)
    }

    fn destination_refs(&self) -> Self::DestinationRefs<'_> {
        (*self).destination_refs()
    }
}

/// Trait defining a bidirectional conversion between a source symbol and a
/// destination symbol.
pub trait BidirectionalVocabulary: Vocabulary {
    /// Converts a destination symbol into a source symbol.
    fn invert(&self, destination: &Self::DestinationSymbol) -> Option<Self::SourceSymbol>;
}

impl<V: BidirectionalVocabulary + ?Sized> BidirectionalVocabulary for &V {
    fn invert(&self, destination: &Self::DestinationSymbol) -> Option<Self::SourceSymbol> {
        (*self).invert(destination)
    }
}

#[cfg(feature = "alloc")]
impl<V: BidirectionalVocabulary + ?Sized> BidirectionalVocabulary for Box<V> {
    fn invert(&self, destination: &Self::DestinationSymbol) -> Option<Self::SourceSymbol> {
        (**self).invert(destination)
    }
}

#[cfg(feature = "alloc")]
impl<V: BidirectionalVocabulary + ?Sized> BidirectionalVocabulary for Rc<V> {
    fn invert(&self, destination: &Self::DestinationSymbol) -> Option<Self::SourceSymbol> {
        (**self).invert(destination)
    }
}

/// Trait defining a bidirectional conversion between a destination symbol and a
/// source symbol reference.
pub trait BidirectionalVocabularyRef: BidirectionalVocabulary + VocabularyRef {
    /// Iterator over the references of the source symbols.
    type SourceRefs<'a>: Iterator<Item = &'a Self::SourceSymbol>
    where
        Self: 'a;

    /// Converts a destination symbol into a source symbol.
    fn invert_ref(&self, destination: &Self::DestinationSymbol) -> Option<&Self::SourceSymbol>;

    /// Returns an iterator over the references of the source symbols.
    fn source_refs(&self) -> Self::SourceRefs<'_>;
}

impl<V: BidirectionalVocabularyRef + ?Sized> BidirectionalVocabularyRef for &V {
    type SourceRefs<'a>
        = V::SourceRefs<'a>
    where
        Self: 'a;

    fn invert_ref(&self, destination: &Self::DestinationSymbol) -> Option<&Self::SourceSymbol> {
        (*self).invert_ref(destination)
    }

    fn source_refs(&self) -> Self::SourceRefs<'_> {
        (*self).source_refs()
    }
}

/// Trait defining a growable vocabulary.
pub trait GrowableVocabulary: Vocabulary + Default {
    /// Creates a new growable vocabulary.
    fn new() -> Self;

    /// Creates a new growable vocabulary with the specified capacity.
    fn with_capacity(capacity: usize) -> Self;

    /// Adds a new entry to the vocabulary.
    ///
    /// # Arguments
    ///
    /// * `source` - The source symbol.
    /// * `destination` - The destination symbol.
    ///
    /// # Errors
    ///
    /// * If the source symbol is already in the vocabulary.
    /// * If the destination symbol is already in the vocabulary.
    fn add(
        &mut self,
        source: Self::SourceSymbol,
        destination: Self::DestinationSymbol,
    ) -> Result<(), VocabularyBuilderError<Self>>;
}
