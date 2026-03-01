//! Implementations of the [`Vector`] trait for various types.

use crate::traits::Vector;

impl<V> Vector for [V] {
    type Value = V;
    type Iter<'a>
        = core::slice::Iter<'a, V>
    where
        Self: 'a;
    type Index = usize;
    type Indices<'a>
        = core::ops::Range<usize>
    where
        Self: 'a;

    #[inline]
    fn indices(&self) -> Self::Indices<'_> {
        0..self.len()
    }

    #[inline]
    fn len(&self) -> usize {
        <[V]>::len(self)
    }

    #[inline]
    fn iter(&self) -> Self::Iter<'_> {
        self.iter()
    }
}

#[cfg(feature = "alloc")]
use alloc::vec::Vec;

#[cfg(feature = "alloc")]
impl<V> Vector for Vec<V> {
    type Value = V;
    type Iter<'a>
        = core::slice::Iter<'a, V>
    where
        Self: 'a;
    type Index = usize;
    type Indices<'a>
        = core::ops::Range<usize>
    where
        Self: 'a;

    #[inline]
    fn indices(&self) -> Self::Indices<'_> {
        0..self.len()
    }

    #[inline]
    fn len(&self) -> usize {
        Vec::len(self)
    }

    #[inline]
    fn iter(&self) -> Self::Iter<'_> {
        self.as_slice().iter()
    }
}

impl<V, const N: usize> Vector for [V; N] {
    type Value = V;
    type Iter<'a>
        = core::slice::Iter<'a, V>
    where
        Self: 'a;
    type Index = usize;
    type Indices<'a>
        = core::ops::Range<usize>
    where
        Self: 'a;

    #[inline]
    fn indices(&self) -> Self::Indices<'_> {
        0..N
    }

    #[inline]
    fn len(&self) -> usize {
        N
    }

    #[inline]
    fn iter(&self) -> Self::Iter<'_> {
        <[V]>::iter(self)
    }
}

#[cfg(all(test, feature = "alloc"))]
mod tests {
    use alloc::vec::Vec;

    use super::*;

    #[test]
    fn test_slice_vector_len() {
        let data = [1, 2, 3, 4, 5];
        let slice: &[i32] = &data;
        assert_eq!(Vector::len(slice), 5);
    }

    #[test]
    fn test_slice_vector_indices() {
        let data = [10, 20, 30];
        let slice: &[i32] = &data;
        let indices: Vec<usize> = slice.indices().collect();
        assert_eq!(indices, vec![0, 1, 2]);
    }

    #[test]
    fn test_slice_vector_iter() {
        let data = [1, 2, 3];
        let slice: &[i32] = &data;
        let items: Vec<&i32> = Vector::iter(slice).collect();
        assert_eq!(items, vec![&1, &2, &3]);
    }

    #[test]
    fn test_slice_vector_empty() {
        let data: [i32; 0] = [];
        let slice: &[i32] = &data;
        assert_eq!(Vector::len(slice), 0);
        assert!(slice.is_empty());
    }

    #[test]
    fn test_vec_vector_len() {
        let v: Vec<i32> = alloc::vec![1, 2, 3, 4];
        assert_eq!(Vector::len(&v), 4);
    }

    #[test]
    fn test_vec_vector_indices() {
        let v: Vec<i32> = alloc::vec![10, 20, 30];
        let indices: Vec<usize> = v.indices().collect();
        assert_eq!(indices, vec![0, 1, 2]);
    }

    #[test]
    fn test_vec_vector_iter() {
        let v: Vec<i32> = alloc::vec![1, 2, 3];
        let items: Vec<&i32> = Vector::iter(&v).collect();
        assert_eq!(items, vec![&1, &2, &3]);
    }

    #[test]
    fn test_vec_vector_empty() {
        let v: Vec<i32> = Vec::new();
        assert_eq!(Vector::len(&v), 0);
        assert!(v.is_empty());
    }

    #[test]
    fn test_array_vector_len() {
        let arr: [i32; 3] = [1, 2, 3];
        assert_eq!(Vector::len(&arr), 3);
    }

    #[test]
    fn test_array_vector_indices() {
        let arr: [i32; 4] = [10, 20, 30, 40];
        let indices: Vec<usize> = arr.indices().collect();
        assert_eq!(indices, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_array_vector_iter() {
        let arr: [i32; 3] = [1, 2, 3];
        let items: Vec<&i32> = Vector::iter(&arr).collect();
        assert_eq!(items, vec![&1, &2, &3]);
    }

    #[test]
    fn test_array_vector_empty() {
        let arr: [i32; 0] = [];
        assert_eq!(Vector::len(&arr), 0);
        assert!(arr.is_empty());
    }
}
