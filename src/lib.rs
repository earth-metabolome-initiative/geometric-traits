#![no_std]
#![doc = include_str!("../README.md")]

#[cfg(feature = "alloc")]
#[macro_use]
extern crate alloc;

#[cfg(feature = "std")]
extern crate std;

pub mod errors;
pub mod impls;
pub mod naive_structs;
pub mod traits;

/// Prelude module for the graph crate.
pub mod prelude {
    pub use crate::{impls::*, naive_structs::*, traits::*};
}
