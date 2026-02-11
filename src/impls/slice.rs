use core::{iter::Cloned, ops::Range};

use crate::{prelude::*, traits::Symbol};

impl<V: Symbol> Vocabulary for [V] {
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

impl<V: Symbol> VocabularyRef for [V] {
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

impl<V: Symbol> BidirectionalVocabulary for [V] {
    fn invert(&self, destination: &Self::DestinationSymbol) -> Option<Self::SourceSymbol> {
        self.iter().position(|v| v == destination)
    }
}

#[cfg(all(test, feature = "alloc"))]
mod tests {
    use alloc::vec::Vec;

    use crate::traits::{BidirectionalVocabulary, Vocabulary, VocabularyRef};

    #[test]
    fn test_slice_vocabulary_len() {
        let data = [10, 20, 30];
        let slice: &[i32] = &data;
        assert_eq!(Vocabulary::len(slice), 3);
    }

    #[test]
    fn test_slice_vocabulary_convert() {
        let data = ["a", "b", "c"];
        let slice: &[&str] = &data;
        assert_eq!(slice.convert(&0), Some("a"));
        assert_eq!(slice.convert(&1), Some("b"));
        assert_eq!(slice.convert(&2), Some("c"));
        assert_eq!(slice.convert(&3), None);
    }

    #[test]
    fn test_slice_vocabulary_sources() {
        let data = [1, 2, 3];
        let slice: &[i32] = &data;
        let sources: Vec<usize> = slice.sources().collect();
        assert_eq!(sources, vec![0, 1, 2]);
    }

    #[test]
    fn test_slice_vocabulary_destinations() {
        let data = [10, 20, 30];
        let slice: &[i32] = &data;
        let destinations: Vec<i32> = slice.destinations().collect();
        assert_eq!(destinations, vec![10, 20, 30]);
    }

    #[test]
    fn test_slice_vocabulary_ref_convert_ref() {
        let data = [100, 200];
        let slice: &[i32] = &data;
        assert_eq!(slice.convert_ref(&0), Some(&100));
        assert_eq!(slice.convert_ref(&1), Some(&200));
        assert_eq!(slice.convert_ref(&2), None);
    }

    #[test]
    fn test_slice_vocabulary_ref_destination_refs() {
        let data = [1, 2];
        let slice: &[i32] = &data;
        let refs: Vec<&i32> = slice.destination_refs().collect();
        assert_eq!(refs, vec![&1, &2]);
    }

    #[test]
    fn test_slice_bidirectional_vocabulary_invert() {
        let data = ["x", "y", "z"];
        let slice: &[&str] = &data;
        assert_eq!(slice.invert(&"x"), Some(0));
        assert_eq!(slice.invert(&"y"), Some(1));
        assert_eq!(slice.invert(&"z"), Some(2));
        assert_eq!(slice.invert(&"w"), None);
    }

    #[test]
    fn test_slice_empty() {
        let data: [i32; 0] = [];
        let slice: &[i32] = &data;
        assert_eq!(Vocabulary::len(slice), 0);
        assert!(slice.is_empty());
    }
}
