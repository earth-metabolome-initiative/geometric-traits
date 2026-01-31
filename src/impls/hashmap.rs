//! Module implementing traits for the [`HashMap`] type.

use core::iter::Cloned;
use core::hash::BuildHasher;

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

            fn convert(&self, source: &Self::SourceSymbol) -> Option<Self::DestinationSymbol> {
                self.get(source).cloned()
            }

            fn len(&self) -> usize {
                self.len()
            }

            fn sources(&self) -> Self::Sources<'_> {
                self.keys().cloned()
            }

            fn destinations(&self) -> Self::Destinations<'_> {
                self.values().cloned()
            }
        }

        impl<K: Symbol, V: Symbol, S: BuildHasher + Clone> VocabularyRef for $Map<K, V, S> {
            type DestinationRefs<'a>
                = $mod::Values<'a, K, V>
            where
                Self: 'a;

            fn convert_ref(&self, source: &Self::SourceSymbol) -> Option<&Self::DestinationSymbol> {
                self.get(source)
            }

            fn destination_refs(&self) -> Self::DestinationRefs<'_> {
                self.values()
            }
        }

        impl<K: Symbol, V: Symbol, S: BuildHasher + Clone> BidirectionalVocabulary for $Map<K, V, S> {
            fn invert(&self, destination: &Self::DestinationSymbol) -> Option<Self::SourceSymbol> {
                self.iter().find(|(_, v)| v == &destination).map(|(k, _)| k.clone())
            }
        }

        impl<K: Symbol, V: Symbol, S: BuildHasher + Clone> BidirectionalVocabularyRef for $Map<K, V, S> {
            type SourceRefs<'a>
                = $mod::Keys<'a, K, V>
            where
                Self: 'a;

            fn invert_ref(&self, destination: &Self::DestinationSymbol) -> Option<&Self::SourceSymbol> {
                self.iter().find(|(_, v)| v == &destination).map(|(k, _)| k)
            }

            fn source_refs(&self) -> Self::SourceRefs<'_> {
                self.keys()
            }
        }

        impl<K: Symbol, V: Symbol, S: BuildHasher + Default + Clone> GrowableVocabulary
            for $Map<K, V, S>
        {
            fn new() -> Self {
                // In hashbrown and std, with_hasher is consistent
                $Map::with_hasher(Default::default())
            }

            fn with_capacity(capacity: usize) -> Self {
                $Map::with_capacity_and_hasher(capacity, Default::default())
            }

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