//! Submodule providing the [`Vector`] trait.

use core::ops::Index;

use num_traits::Zero;

use crate::traits::{IntoUsize, PositiveInteger};

/// Trait defining a vector.
///
/// # Examples
///
/// ```
/// use geometric_traits::traits::Vector;
///
/// let vec = vec![10, 20, 30];
/// assert_eq!(vec.len(), 3);
/// assert!(!vec.is_empty());
///
/// let items: Vec<_> = vec.iter().cloned().collect();
/// assert_eq!(items, vec![10, 20, 30]);
///
/// let indices: Vec<_> = vec.indices().collect();
/// assert_eq!(indices, vec![0, 1, 2]);
///
/// let empty: Vec<i32> = Vec::new();
/// assert!(empty.is_empty());
/// ```
pub trait Vector: Index<<Self as Vector>::Index, Output = <Self as Vector>::Value> {
    /// The value of the vector.
    type Value;
    /// Iterator over the values in the vector.
    type Iter<'a>: Iterator<Item = &'a Self::Value> + Clone + DoubleEndedIterator
    where
        Self: 'a;
    /// The index of the vector.
    type Index: PositiveInteger + IntoUsize;
    /// Iterator over the indices of the vector.
    type Indices<'a>: Iterator<Item = Self::Index>
    where
        Self: 'a;

    /// Returns an iterator over the values in the vector.
    fn iter(&self) -> Self::Iter<'_>;

    /// Returns an iterator over the indices of the vector.
    fn indices(&self) -> Self::Indices<'_>;

    /// Returns the number of elements in the vector.
    fn len(&self) -> Self::Index;

    /// Returns whether the vector is empty.
    fn is_empty(&self) -> bool {
        self.len() == <Self::Index as Zero>::zero()
    }
}
