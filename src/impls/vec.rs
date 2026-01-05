//! Module implementing traits for the Vec type.

#[cfg(all(not(feature = "std"), feature = "alloc"))]
use alloc::vec::Vec;
use core::{iter::Cloned, ops::Range};

use crate::traits::Symbol;

use crate::prelude::*;

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
