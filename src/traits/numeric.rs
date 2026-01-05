//! Numeric traits.

use super::into_usize::IntoUsize as IntoUsizeMethod;
use core::fmt::{Debug, Display};
use core::hash::Hash;
use core::ops::{AddAssign, SubAssign};
use num_traits::{Bounded, ConstOne, ConstZero, Num, SaturatingAdd, Unsigned};

/// Trait for types that can be converted into `usize`.
pub trait IntoUsize: Into<usize> + Copy {}
impl<T: Into<usize> + Copy> IntoUsize for T {}

/// Trait for types that can be created from `usize`.
pub trait TryFromUsize: TryFrom<usize> + Copy {
    /// Tries to create a value from a `usize`.
    ///
    /// # Errors
    ///
    /// Returns an error if the value cannot be converted from `usize`.
    fn try_from_usize(v: usize) -> Result<Self, Self::Error> {
        Self::try_from(v)
    }
}
impl<T: TryFrom<usize> + Copy> TryFromUsize for T {}

/// Trait for positive integers.
pub trait PositiveInteger:
    Unsigned
    + Copy
    + Eq
    + Hash
    + Debug
    + Display
    + Ord
    + IntoUsize
    + TryFromUsize
    + IntoUsizeMethod
    + Bounded
    + AddAssign
    + SubAssign
    + SaturatingAdd
    + ConstZero
    + ConstOne
{
}
impl<
    T: Unsigned
        + Copy
        + Eq
        + Hash
        + Debug
        + Display
        + Ord
        + IntoUsize
        + TryFromUsize
        + IntoUsizeMethod
        + Bounded
        + AddAssign
        + SubAssign
        + SaturatingAdd
        + ConstZero
        + ConstOne,
> PositiveInteger for T
{
}

/// Trait for numbers.
pub trait Number: Num + Copy + PartialOrd + Debug + Bounded + AddAssign + SubAssign {}
impl<T: Num + Copy + PartialOrd + Debug + Bounded + AddAssign + SubAssign> Number for T {}

/// Trait for finite numbers.
pub trait Finite {
    /// Returns `true` if the number is finite.
    fn is_finite(&self) -> bool;
}

impl Finite for f32 {
    fn is_finite(&self) -> bool {
        f32::is_finite(*self)
    }
}

impl Finite for f64 {
    fn is_finite(&self) -> bool {
        f64::is_finite(*self)
    }
}

macro_rules! impl_finite_int {
    ($($t:ty),*) => {
        $(
            impl Finite for $t {
                fn is_finite(&self) -> bool {
                    true
                }
            }
        )*
    };
}

impl_finite_int!(
    u8, u16, u32, u64, u128, usize, i8, i16, i32, i64, i128, isize
);
