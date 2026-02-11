//! Module implementing traits for the Vec type.

#[cfg(feature = "alloc")]
use alloc::vec::Vec;
use core::{iter::Cloned, ops::Range};

use crate::{prelude::*, traits::Symbol};

impl<V: Symbol> Vocabulary for Vec<V> {
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

impl<V: Symbol> VocabularyRef for Vec<V> {
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

impl<V: Symbol> BidirectionalVocabulary for Vec<V> {
    fn invert(&self, destination: &Self::DestinationSymbol) -> Option<Self::SourceSymbol> {
        self.iter().position(|v| v == destination)
    }
}

impl<V: Symbol + Ord> GrowableVocabulary for Vec<V> {
    fn new() -> Self {
        Vec::new()
    }

    fn with_capacity(capacity: usize) -> Self {
        Vec::with_capacity(capacity)
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

        if self.invert(&destination).is_some() {
            return Err(crate::errors::builder::vocabulary::VocabularyBuilderError::RepeatedDestinationSymbol(destination));
        }

        self.push(destination);

        Ok(())
    }
}

use crate::traits::{Matrix, Matrix2D};

/// Implementation of a matrix using a vector.
pub struct VecMatrix2D<V> {
    /// The data of the matrix.
    data: Vec<V>,
    /// The number of rows.
    number_of_rows: usize,
}

impl<V> Matrix for VecMatrix2D<V> {
    type Coordinates = (usize, usize);

    #[inline]
    fn shape(&self) -> Vec<usize> {
        vec![self.number_of_rows(), self.number_of_columns()]
    }
}

impl<V> Matrix2D for VecMatrix2D<V> {
    type RowIndex = usize;
    type ColumnIndex = usize;

    #[inline]
    fn number_of_rows(&self) -> usize {
        self.number_of_rows
    }

    #[inline]
    fn number_of_columns(&self) -> usize {
        self.data.len() / self.number_of_rows
    }
}

#[cfg(all(test, feature = "alloc"))]
mod tests {
    use alloc::{vec, vec::Vec};

    use super::*;
    use crate::traits::{BidirectionalVocabulary, GrowableVocabulary, Vocabulary, VocabularyRef};

    #[test]
    fn test_vec_vocabulary_len() {
        let v: Vec<i32> = vec![10, 20, 30];
        assert_eq!(Vocabulary::len(&v), 3);
    }

    #[test]
    fn test_vec_vocabulary_convert() {
        let v: Vec<&str> = vec!["a", "b", "c"];
        assert_eq!(v.convert(&0), Some("a"));
        assert_eq!(v.convert(&1), Some("b"));
        assert_eq!(v.convert(&2), Some("c"));
        assert_eq!(v.convert(&3), None);
    }

    #[test]
    fn test_vec_vocabulary_sources() {
        let v: Vec<i32> = vec![1, 2, 3];
        let sources: Vec<usize> = v.sources().collect();
        assert_eq!(sources, vec![0, 1, 2]);
    }

    #[test]
    fn test_vec_vocabulary_destinations() {
        let v: Vec<i32> = vec![10, 20, 30];
        let destinations: Vec<i32> = v.destinations().collect();
        assert_eq!(destinations, vec![10, 20, 30]);
    }

    #[test]
    fn test_vec_vocabulary_ref_convert_ref() {
        let v: Vec<i32> = vec![100, 200];
        assert_eq!(v.convert_ref(&0), Some(&100));
        assert_eq!(v.convert_ref(&1), Some(&200));
        assert_eq!(v.convert_ref(&2), None);
    }

    #[test]
    fn test_vec_vocabulary_ref_destination_refs() {
        let v: Vec<i32> = vec![1, 2];
        let refs: Vec<&i32> = v.destination_refs().collect();
        assert_eq!(refs, vec![&1, &2]);
    }

    #[test]
    fn test_vec_bidirectional_vocabulary_invert() {
        let v: Vec<&str> = vec!["x", "y", "z"];
        assert_eq!(v.invert(&"x"), Some(0));
        assert_eq!(v.invert(&"y"), Some(1));
        assert_eq!(v.invert(&"z"), Some(2));
        assert_eq!(v.invert(&"w"), None);
    }

    #[test]
    fn test_vec_growable_vocabulary_new() {
        let v: Vec<i32> = GrowableVocabulary::new();
        assert!(v.is_empty());
    }

    #[test]
    fn test_vec_growable_vocabulary_with_capacity() {
        let v: Vec<i32> = GrowableVocabulary::with_capacity(10);
        assert!(v.is_empty());
    }

    #[test]
    fn test_vec_growable_vocabulary_add() {
        let mut v: Vec<i32> = GrowableVocabulary::new();
        assert!(v.add(0, 10).is_ok());
        assert!(v.add(1, 20).is_ok());
        assert!(v.add(2, 30).is_ok());
        assert_eq!(v.len(), 3);
    }

    #[test]
    fn test_vec_growable_vocabulary_add_sparse_error() {
        let mut v: Vec<i32> = GrowableVocabulary::new();
        // Try to add at index 1 when vec is empty
        assert!(v.add(1, 10).is_err());
    }

    #[test]
    fn test_vec_growable_vocabulary_add_duplicate_error() {
        let mut v: Vec<i32> = GrowableVocabulary::new();
        assert!(v.add(0, 10).is_ok());
        // Try to add duplicate destination
        assert!(v.add(1, 10).is_err());
    }

    #[test]
    fn test_vec_matrix2d_shape() {
        let matrix = VecMatrix2D { data: vec![1, 2, 3, 4, 5, 6], number_of_rows: 2 };
        assert_eq!(matrix.number_of_rows(), 2);
        assert_eq!(matrix.number_of_columns(), 3);
        assert_eq!(matrix.shape(), vec![2, 3]);
    }
}
