//! Trait for total ordering.

use core::cmp::Ordering;

/// Trait for total ordering.
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

impl_total_ord!(
    u8, u16, u32, u64, u128, usize, i8, i16, i32, i64, i128, isize
);
