//! Module implementing traits for the `HashMap` type.

use core::{hash::BuildHasher, iter::Cloned};

use crate::{prelude::*, traits::Symbol};

macro_rules! impl_hashmap {
    ($Map:ident, $mod:ident) => {
        impl<K: Symbol, V: Symbol, S: BuildHasher + Clone> Vocabulary for $Map<K, V, S> {
            type SourceSymbol = K;
            type DestinationSymbol = V;
            type Sources<'a>
                = Cloned<$mod::Keys<'a, K, V>>
            where
                Self: 'a;
            type Destinations<'a>
                = Cloned<$mod::Values<'a, K, V>>
            where
                Self: 'a;

            #[inline]
            fn convert(&self, source: &Self::SourceSymbol) -> Option<Self::DestinationSymbol> {
                self.get(source).cloned()
            }

            #[inline]
            fn len(&self) -> usize {
                self.len()
            }

            #[inline]
            fn sources(&self) -> Self::Sources<'_> {
                self.keys().cloned()
            }

            #[inline]
            fn destinations(&self) -> Self::Destinations<'_> {
                self.values().cloned()
            }
        }

        impl<K: Symbol, V: Symbol, S: BuildHasher + Clone> VocabularyRef for $Map<K, V, S> {
            type DestinationRefs<'a>
                = $mod::Values<'a, K, V>
            where
                Self: 'a;

            #[inline]
            fn convert_ref(&self, source: &Self::SourceSymbol) -> Option<&Self::DestinationSymbol> {
                self.get(source)
            }

            #[inline]
            fn destination_refs(&self) -> Self::DestinationRefs<'_> {
                self.values()
            }
        }

        impl<K: Symbol, V: Symbol, S: BuildHasher + Clone> BidirectionalVocabulary for $Map<K, V, S> {
            #[inline]
            fn invert(&self, destination: &Self::DestinationSymbol) -> Option<Self::SourceSymbol> {
                self.iter().find(|(_, v)| v == &destination).map(|(k, _)| k.clone())
            }
        }

        impl<K: Symbol, V: Symbol, S: BuildHasher + Clone> BidirectionalVocabularyRef for $Map<K, V, S> {
            type SourceRefs<'a>
                = $mod::Keys<'a, K, V>
            where
                Self: 'a;

            #[inline]
            fn invert_ref(&self, destination: &Self::DestinationSymbol) -> Option<&Self::SourceSymbol> {
                self.iter().find(|(_, v)| v == &destination).map(|(k, _)| k)
            }

            #[inline]
            fn source_refs(&self) -> Self::SourceRefs<'_> {
                self.keys()
            }
        }

        impl<K: Symbol, V: Symbol, S: BuildHasher + Default + Clone> GrowableVocabulary
            for $Map<K, V, S>
        {
            #[inline]
            fn new() -> Self {
                // In hashbrown and std, with_hasher is consistent
                $Map::with_hasher(Default::default())
            }

            #[inline]
            fn with_capacity(capacity: usize) -> Self {
                $Map::with_capacity_and_hasher(capacity, Default::default())
            }

            #[inline]
            fn add(
                &mut self,
                source: K,
                destination: V,
            ) -> Result<(), crate::errors::builder::vocabulary::VocabularyBuilderError<Self>> {
                if self.contains_key(&source) {
                    return Err(
                        crate::errors::builder::vocabulary::VocabularyBuilderError::RepeatedSourceSymbol(
                            source,
                        ),
                    );
                }
                if <Self as BidirectionalVocabularyRef>::invert_ref(self, &destination).is_some() {
                    return Err(crate::errors::builder::vocabulary::VocabularyBuilderError::RepeatedDestinationSymbol(destination));
                }
                self.insert(source, destination);
                Ok(())
            }
        }
    }
}

#[cfg(feature = "std")]
use std::collections::{HashMap as StdHashMap, hash_map as std_hash_map};

#[cfg(feature = "std")]
impl_hashmap!(StdHashMap, std_hash_map);

#[cfg(feature = "hashbrown")]
use hashbrown::{HashMap as HashBrownHashMap, hash_map as hashbrown_hash_map};

#[cfg(feature = "hashbrown")]
impl_hashmap!(HashBrownHashMap, hashbrown_hash_map);

#[cfg(all(test, feature = "std"))]
mod tests {
    use std::{collections::HashMap, vec::Vec};

    use crate::traits::{
        BidirectionalVocabulary, BidirectionalVocabularyRef, GrowableVocabulary, Vocabulary,
        VocabularyRef,
    };

    #[test]
    fn test_hashmap_vocabulary_len() {
        let mut map: HashMap<i32, i32> = HashMap::new();
        map.insert(1, 10);
        map.insert(2, 20);
        map.insert(3, 30);
        assert_eq!(Vocabulary::len(&map), 3);
    }

    #[test]
    fn test_hashmap_vocabulary_convert() {
        let mut map: HashMap<i32, i32> = HashMap::new();
        map.insert(1, 10);
        map.insert(2, 20);
        assert_eq!(map.convert(&1), Some(10));
        assert_eq!(map.convert(&2), Some(20));
        assert_eq!(map.convert(&3), None);
    }

    #[test]
    fn test_hashmap_vocabulary_sources() {
        let mut map: HashMap<i32, i32> = HashMap::new();
        map.insert(1, 10);
        map.insert(2, 20);
        let mut sources: Vec<i32> = map.sources().collect();
        sources.sort_unstable();
        assert_eq!(sources, vec![1, 2]);
    }

    #[test]
    fn test_hashmap_vocabulary_destinations() {
        let mut map: HashMap<i32, i32> = HashMap::new();
        map.insert(1, 10);
        map.insert(2, 20);
        let mut destinations: Vec<i32> = map.destinations().collect();
        destinations.sort_unstable();
        assert_eq!(destinations, vec![10, 20]);
    }

    #[test]
    fn test_hashmap_vocabulary_ref_convert_ref() {
        let mut map: HashMap<i32, i32> = HashMap::new();
        map.insert(1, 100);
        map.insert(2, 200);
        assert_eq!(map.convert_ref(&1), Some(&100));
        assert_eq!(map.convert_ref(&2), Some(&200));
        assert_eq!(map.convert_ref(&3), None);
    }

    #[test]
    fn test_hashmap_vocabulary_ref_destination_refs() {
        let mut map: HashMap<i32, i32> = HashMap::new();
        map.insert(1, 10);
        map.insert(2, 20);
        let mut refs: Vec<&i32> = map.destination_refs().collect();
        refs.sort_unstable();
        assert_eq!(refs, vec![&10, &20]);
    }

    #[test]
    fn test_hashmap_bidirectional_vocabulary_invert() {
        let mut map: HashMap<i32, i32> = HashMap::new();
        map.insert(1, 10);
        map.insert(2, 20);
        map.insert(3, 30);
        assert_eq!(map.invert(&10), Some(1));
        assert_eq!(map.invert(&20), Some(2));
        assert_eq!(map.invert(&30), Some(3));
        assert_eq!(map.invert(&40), None);
    }

    #[test]
    fn test_hashmap_bidirectional_vocabulary_ref_invert_ref() {
        let mut map: HashMap<i32, i32> = HashMap::new();
        map.insert(1, 10);
        map.insert(2, 20);
        assert_eq!(map.invert_ref(&10), Some(&1));
        assert_eq!(map.invert_ref(&20), Some(&2));
        assert_eq!(map.invert_ref(&30), None);
    }

    #[test]
    fn test_hashmap_bidirectional_vocabulary_ref_source_refs() {
        let mut map: HashMap<i32, i32> = HashMap::new();
        map.insert(1, 10);
        map.insert(2, 20);
        let mut refs: Vec<&i32> = map.source_refs().collect();
        refs.sort_unstable();
        assert_eq!(refs, vec![&1, &2]);
    }

    #[test]
    fn test_hashmap_growable_vocabulary_new() {
        let map: HashMap<i32, i32> = GrowableVocabulary::new();
        assert!(map.is_empty());
    }

    #[test]
    fn test_hashmap_growable_vocabulary_with_capacity() {
        let map: HashMap<i32, i32> = GrowableVocabulary::with_capacity(10);
        assert!(map.is_empty());
    }

    #[test]
    fn test_hashmap_growable_vocabulary_add() {
        let mut map: HashMap<i32, i32> = GrowableVocabulary::new();
        assert!(map.add(1, 10).is_ok());
        assert!(map.add(2, 20).is_ok());
        assert_eq!(map.len(), 2);
    }

    #[test]
    fn test_hashmap_growable_vocabulary_add_duplicate_source() {
        let mut map: HashMap<i32, i32> = GrowableVocabulary::new();
        assert!(map.add(1, 10).is_ok());
        assert!(map.add(1, 20).is_err());
    }

    #[test]
    fn test_hashmap_growable_vocabulary_add_duplicate_destination() {
        let mut map: HashMap<i32, i32> = GrowableVocabulary::new();
        assert!(map.add(1, 10).is_ok());
        assert!(map.add(2, 10).is_err());
    }
}
