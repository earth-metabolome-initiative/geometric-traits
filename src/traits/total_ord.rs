//! Trait for total ordering.

use core::cmp::Ordering;

/// Trait for total ordering.
///
/// # Examples
///
/// ```
/// use core::cmp::Ordering;
///
/// use geometric_traits::traits::TotalOrd;
///
/// // Float comparison handles NaN correctly
/// assert_eq!(1.0_f64.total_cmp(&2.0), Ordering::Less);
/// assert_eq!(2.0_f64.total_cmp(&1.0), Ordering::Greater);
/// assert_eq!(1.0_f64.total_cmp(&1.0), Ordering::Equal);
///
/// // Integer comparison
/// assert_eq!(1_u32.total_cmp(&2), Ordering::Less);
/// assert_eq!((-1_i32).total_cmp(&1), Ordering::Less);
/// ```
pub trait TotalOrd {
    /// Compare two values.
    fn total_cmp(&self, other: &Self) -> Ordering;
}

impl TotalOrd for f32 {
    fn total_cmp(&self, other: &Self) -> Ordering {
        self.total_cmp(other)
    }
}

impl TotalOrd for f64 {
    fn total_cmp(&self, other: &Self) -> Ordering {
        self.total_cmp(other)
    }
}

macro_rules! impl_total_ord {
    ($($t:ty),*) => {
        $(
            impl TotalOrd for $t {
                fn total_cmp(&self, other: &Self) -> Ordering {
                    self.cmp(other)
                }
            }
        )*
    };
}

impl_total_ord!(u8, u16, u32, u64, u128, usize, i8, i16, i32, i64, i128, isize);
