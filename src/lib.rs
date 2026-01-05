#![doc = include_str!("../README.md")]
#![cfg(feature = "alloc")]
extern crate alloc;

pub mod errors;
pub mod impls;
pub mod naive_structs;
pub mod traits;

/// Prelude module for the graph crate.
pub mod prelude {
    pub use crate::{impls::*, naive_structs::*, traits::*};
}
