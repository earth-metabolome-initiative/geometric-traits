//! Submodule providing traits to generate randomized graphs

#[cfg(all(feature = "alloc", any(feature = "std", feature = "hashbrown")))]
mod randomized_dag;
#[cfg(all(feature = "alloc", any(feature = "std", feature = "hashbrown")))]
pub use randomized_dag::{RandomizedDAG, XorShift64};
