//! Submodule providing algorithms for solving the Weighted Assignment Problem.

#[cfg(feature = "alloc")]
mod lapjv;
#[cfg(feature = "alloc")]
pub use lapjv::{LAPJV, LAPJVError, SparseLAPJV};
