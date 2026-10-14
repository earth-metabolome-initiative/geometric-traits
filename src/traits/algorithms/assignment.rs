//! Submodule providing algorithms for solving the Assignment Problem.

#[cfg(feature = "alloc")]
mod hopcroft_karp;
#[cfg(feature = "alloc")]
pub use hopcroft_karp::*;
mod assignment_state;
pub use assignment_state::AssignmentState;
