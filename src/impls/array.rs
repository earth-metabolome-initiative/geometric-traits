//! Module implementing traits for the Vec type.

use core::{iter::Cloned, ops::Range};

use crate::traits::{PositiveInteger, Symbol};

impl<V: Symbol, const N: usize> crate::traits::Vocabulary for [V; N] {
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

    #[inline]
    fn convert(&self, source: &Self::SourceSymbol) -> Option<Self::DestinationSymbol> {
        self.get(*source).cloned()
    }

    #[inline]
    fn len(&self) -> usize {
        N
    }

    #[inline]
    fn sources(&self) -> Self::Sources<'_> {
        0..self.len()
    }

    #[inline]
    fn destinations(&self) -> Self::Destinations<'_> {
        self.iter().cloned()
    }
}

impl<V: Symbol, const N: usize> crate::traits::VocabularyRef for [V; N] {
    type DestinationRefs<'a>
        = core::slice::Iter<'a, Self::DestinationSymbol>
    where
        Self: 'a;

    #[inline]
    fn convert_ref(&self, source: &Self::SourceSymbol) -> Option<&Self::DestinationSymbol> {
        self.get(*source)
    }

    #[inline]
    fn destination_refs(&self) -> Self::DestinationRefs<'_> {
        self.iter()
    }
}

impl<V: Symbol, const N: usize> crate::traits::BidirectionalVocabulary for [V; N] {
    #[inline]
    fn invert(&self, destination: &Self::DestinationSymbol) -> Option<Self::SourceSymbol> {
        self.iter().position(|v| v == destination)
    }
}

impl<NodeId: PositiveInteger> crate::traits::Edge for [NodeId; 2] {
    type SourceNodeId = NodeId;
    type DestinationNodeId = NodeId;

    #[inline]
    fn source(&self) -> Self::SourceNodeId {
        self[0]
    }

    #[inline]
    fn destination(&self) -> Self::DestinationNodeId {
        self[1]
    }
}

#[cfg(all(test, feature = "alloc"))]
mod tests {
    use alloc::vec::Vec;

    use crate::traits::{BidirectionalVocabulary, Edge, Vocabulary, VocabularyRef};

    #[test]
    fn test_array_vocabulary_len() {
        let arr: [i32; 3] = [10, 20, 30];
        assert_eq!(arr.len(), 3);
    }

    #[test]
    fn test_array_vocabulary_convert() {
        let arr: [&str; 3] = ["a", "b", "c"];
        assert_eq!(arr.convert(&0), Some("a"));
        assert_eq!(arr.convert(&1), Some("b"));
        assert_eq!(arr.convert(&2), Some("c"));
        assert_eq!(arr.convert(&3), None);
    }

    #[test]
    fn test_array_vocabulary_sources() {
        let arr: [i32; 3] = [1, 2, 3];
        let sources: Vec<usize> = arr.sources().collect();
        assert_eq!(sources, vec![0, 1, 2]);
    }

    #[test]
    fn test_array_vocabulary_destinations() {
        let arr: [i32; 3] = [10, 20, 30];
        let destinations: Vec<i32> = arr.destinations().collect();
        assert_eq!(destinations, vec![10, 20, 30]);
    }

    #[test]
    fn test_array_vocabulary_ref_convert_ref() {
        let arr: [i32; 2] = [100, 200];
        assert_eq!(arr.convert_ref(&0), Some(&100));
        assert_eq!(arr.convert_ref(&1), Some(&200));
        assert_eq!(arr.convert_ref(&2), None);
    }

    #[test]
    fn test_array_vocabulary_ref_destination_refs() {
        let arr: [i32; 2] = [1, 2];
        let refs: Vec<&i32> = arr.destination_refs().collect();
        assert_eq!(refs, vec![&1, &2]);
    }

    #[test]
    fn test_array_bidirectional_vocabulary_invert() {
        let arr: [&str; 3] = ["x", "y", "z"];
        assert_eq!(arr.invert(&"x"), Some(0));
        assert_eq!(arr.invert(&"y"), Some(1));
        assert_eq!(arr.invert(&"z"), Some(2));
        assert_eq!(arr.invert(&"w"), None);
    }

    #[test]
    fn test_array_edge_source() {
        let edge: [usize; 2] = [5, 10];
        assert_eq!(edge.source(), 5);
    }

    #[test]
    fn test_array_edge_destination() {
        let edge: [usize; 2] = [5, 10];
        assert_eq!(edge.destination(), 10);
    }

    #[test]
    fn test_array_edge_is_self_loop() {
        let edge: [usize; 2] = [5, 5];
        assert!(edge.is_self_loop());

        let edge2: [usize; 2] = [5, 10];
        assert!(!edge2.is_self_loop());
    }
}
