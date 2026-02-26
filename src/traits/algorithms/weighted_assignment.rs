//! Submodule providing algorithms for solving the Weighted Assignment Problem.

#[cfg(feature = "alloc")]
mod lapjv;
#[cfg(feature = "alloc")]
pub use lapjv::{LAPJV, LAPJVError, SparseLAPJV};

#[cfg(feature = "alloc")]
mod lapmod;
#[cfg(feature = "alloc")]
pub use lapmod::{LAPMOD, LAPMODError, SparseLAPMOD};

#[cfg(feature = "alloc")]
mod lap_error;
#[cfg(feature = "alloc")]
pub use lap_error::LAPError;
