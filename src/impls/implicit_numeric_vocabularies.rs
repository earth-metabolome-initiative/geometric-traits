//! Submodule providing the implementation of implicit numeric vocabularies.

use crate::traits::{BidirectionalVocabulary, Vocabulary};

impl Vocabulary for u8 {
    type SourceSymbol = u8;
    type DestinationSymbol = u8;
    type Sources<'a> = core::ops::Range<u8>;
    type Destinations<'a> = core::ops::Range<u8>;

    fn convert(&self, source: &Self::SourceSymbol) -> Option<Self::DestinationSymbol> {
        Some(*source)
    }

    fn len(&self) -> usize {
        usize::from(*self)
    }

    fn sources(&self) -> Self::Sources<'_> {
        0..*self
    }

    fn destinations(&self) -> Self::Destinations<'_> {
        0..*self
    }
}

impl BidirectionalVocabulary for u8 {
    fn invert(&self, destination: &Self::DestinationSymbol) -> Option<Self::SourceSymbol> {
        Some(*destination)
    }
}

impl Vocabulary for u16 {
    type SourceSymbol = u16;
    type DestinationSymbol = u16;
    type Sources<'a> = core::ops::Range<u16>;
    type Destinations<'a> = core::ops::Range<u16>;

    fn convert(&self, source: &Self::SourceSymbol) -> Option<Self::DestinationSymbol> {
        Some(*source)
    }

    fn len(&self) -> usize {
        usize::from(*self)
    }

    fn sources(&self) -> Self::Sources<'_> {
        0..*self
    }

    fn destinations(&self) -> Self::Destinations<'_> {
        0..*self
    }
}

impl BidirectionalVocabulary for u16 {
    fn invert(&self, destination: &Self::DestinationSymbol) -> Option<Self::SourceSymbol> {
        Some(*destination)
    }
}

impl Vocabulary for u32 {
    type SourceSymbol = u32;
    type DestinationSymbol = u32;
    type Sources<'a> = core::ops::Range<u32>;
    type Destinations<'a> = core::ops::Range<u32>;

    fn convert(&self, source: &Self::SourceSymbol) -> Option<Self::DestinationSymbol> {
        Some(*source)
    }

    fn len(&self) -> usize {
        *self as usize
    }

    fn sources(&self) -> Self::Sources<'_> {
        0..*self
    }

    fn destinations(&self) -> Self::Destinations<'_> {
        0..*self
    }
}

impl BidirectionalVocabulary for u32 {
    fn invert(&self, destination: &Self::DestinationSymbol) -> Option<Self::SourceSymbol> {
        Some(*destination)
    }
}

impl Vocabulary for usize {
    type SourceSymbol = usize;
    type DestinationSymbol = usize;
    type Sources<'a> = core::ops::Range<usize>;
    type Destinations<'a> = core::ops::Range<usize>;

    fn convert(&self, source: &Self::SourceSymbol) -> Option<Self::DestinationSymbol> {
        Some(*source)
    }

    fn len(&self) -> usize {
        *self
    }

    fn sources(&self) -> Self::Sources<'_> {
        0..*self
    }

    fn destinations(&self) -> Self::Destinations<'_> {
        0..*self
    }
}

impl BidirectionalVocabulary for usize {
    fn invert(&self, destination: &Self::DestinationSymbol) -> Option<Self::SourceSymbol> {
        Some(*destination)
    }
}

impl Vocabulary for u64 {
    type SourceSymbol = u64;
    type DestinationSymbol = u64;
    type Sources<'a> = core::ops::Range<u64>;
    type Destinations<'a> = core::ops::Range<u64>;

    fn convert(&self, source: &Self::SourceSymbol) -> Option<Self::DestinationSymbol> {
        Some(*source)
    }

    #[allow(clippy::cast_possible_truncation)]
    fn len(&self) -> usize {
        *self as usize
    }

    fn sources(&self) -> Self::Sources<'_> {
        0..*self
    }

    fn destinations(&self) -> Self::Destinations<'_> {
        0..*self
    }
}

impl BidirectionalVocabulary for u64 {
    fn invert(&self, destination: &Self::DestinationSymbol) -> Option<Self::SourceSymbol> {
        Some(*destination)
    }
}

#[cfg(all(test, feature = "alloc"))]
mod tests {
    use alloc::vec::Vec;

    use super::*;

    #[test]
    fn test_u8_vocabulary_len() {
        let v: u8 = 5;
        assert_eq!(Vocabulary::len(&v), 5);
    }

    #[test]
    fn test_u8_vocabulary_convert() {
        let v: u8 = 10;
        assert_eq!(v.convert(&3), Some(3));
        assert_eq!(v.convert(&0), Some(0));
        assert_eq!(v.convert(&9), Some(9));
    }

    #[test]
    fn test_u8_vocabulary_sources() {
        let v: u8 = 3;
        let sources: Vec<u8> = v.sources().collect();
        assert_eq!(sources, vec![0, 1, 2]);
    }

    #[test]
    fn test_u8_vocabulary_destinations() {
        let v: u8 = 3;
        let destinations: Vec<u8> = v.destinations().collect();
        assert_eq!(destinations, vec![0, 1, 2]);
    }

    #[test]
    fn test_u8_bidirectional_vocabulary_invert() {
        let v: u8 = 10;
        assert_eq!(v.invert(&5), Some(5));
    }

    #[test]
    fn test_u16_vocabulary_len() {
        let v: u16 = 100;
        assert_eq!(Vocabulary::len(&v), 100);
    }

    #[test]
    fn test_u16_vocabulary_sources() {
        let v: u16 = 3;
        let sources: Vec<u16> = v.sources().collect();
        assert_eq!(sources, vec![0, 1, 2]);
    }

    #[test]
    fn test_u16_bidirectional_vocabulary_invert() {
        let v: u16 = 1000;
        assert_eq!(v.invert(&500), Some(500));
    }

    #[test]
    fn test_u32_vocabulary_len() {
        let v: u32 = 1000;
        assert_eq!(Vocabulary::len(&v), 1000);
    }

    #[test]
    fn test_u32_vocabulary_convert() {
        let v: u32 = 100;
        assert_eq!(v.convert(&50), Some(50));
    }

    #[test]
    fn test_u32_vocabulary_sources() {
        let v: u32 = 3;
        let sources: Vec<u32> = v.sources().collect();
        assert_eq!(sources, vec![0, 1, 2]);
    }

    #[test]
    fn test_u32_bidirectional_vocabulary_invert() {
        let v: u32 = 100;
        assert_eq!(v.invert(&50), Some(50));
    }

    #[test]
    fn test_usize_vocabulary_len() {
        let v: usize = 50;
        assert_eq!(Vocabulary::len(&v), 50);
    }

    #[test]
    fn test_usize_vocabulary_convert() {
        let v: usize = 100;
        assert_eq!(v.convert(&25), Some(25));
    }

    #[test]
    fn test_usize_vocabulary_sources() {
        let v: usize = 4;
        let sources: Vec<usize> = v.sources().collect();
        assert_eq!(sources, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_usize_vocabulary_destinations() {
        let v: usize = 4;
        let destinations: Vec<usize> = v.destinations().collect();
        assert_eq!(destinations, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_usize_bidirectional_vocabulary_invert() {
        let v: usize = 100;
        assert_eq!(v.invert(&75), Some(75));
    }

    #[test]
    fn test_u64_vocabulary_len() {
        let v: u64 = 1000;
        assert_eq!(Vocabulary::len(&v), 1000);
    }

    #[test]
    fn test_u64_vocabulary_sources() {
        let v: u64 = 3;
        let sources: Vec<u64> = v.sources().collect();
        assert_eq!(sources, vec![0, 1, 2]);
    }

    #[test]
    fn test_u64_bidirectional_vocabulary_invert() {
        let v: u64 = 1000;
        assert_eq!(v.invert(&999), Some(999));
    }

    #[test]
    fn test_zero_vocabulary() {
        let v: usize = 0;
        assert_eq!(Vocabulary::len(&v), 0);
        let sources: Vec<usize> = v.sources().collect();
        assert!(sources.is_empty());
    }
}
