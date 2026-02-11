//! Submodule providing a definition of a coordinate.

use core::fmt::Debug;

/// Trait defining coordinates.
///
/// # Examples
///
/// ```
/// use geometric_traits::traits::Coordinates;
///
/// assert_eq!(<()>::dimensions(), 0);
/// assert_eq!(<(i32,)>::dimensions(), 1);
/// assert_eq!(<(i32, i32)>::dimensions(), 2);
/// assert_eq!(<(i32, i32, i32)>::dimensions(), 3);
/// assert_eq!(<[f64; 2]>::dimensions(), 2);
/// assert_eq!(<[f64; 3]>::dimensions(), 3);
/// ```
pub trait Coordinates: core::fmt::Debug {
    /// Returns the number of dimensions of the coordinate.
    fn dimensions() -> usize;
}

impl Coordinates for () {
    #[inline]
    fn dimensions() -> usize {
        0
    }
}

impl<A: Debug> Coordinates for (A,) {
    #[inline]
    fn dimensions() -> usize {
        1
    }
}

impl<A: Debug, B: Debug> Coordinates for (A, B) {
    #[inline]
    fn dimensions() -> usize {
        2
    }
}

impl<A: Debug, B: Debug, C: Debug> Coordinates for (A, B, C) {
    #[inline]
    fn dimensions() -> usize {
        3
    }
}

impl<A: Debug, B: Debug, C: Debug, D: Debug> Coordinates for (A, B, C, D) {
    #[inline]
    fn dimensions() -> usize {
        4
    }
}

impl<A: Debug> Coordinates for [A; 0] {
    #[inline]
    fn dimensions() -> usize {
        0
    }
}

impl<A: Debug> Coordinates for [A; 1] {
    #[inline]
    fn dimensions() -> usize {
        1
    }
}

impl<A: Debug> Coordinates for [A; 2] {
    #[inline]
    fn dimensions() -> usize {
        2
    }
}

impl<A: Debug> Coordinates for [A; 3] {
    #[inline]
    fn dimensions() -> usize {
        3
    }
}

impl<A: Debug> Coordinates for [A; 4] {
    #[inline]
    fn dimensions() -> usize {
        4
    }
}
