//! Minimum spanning forest traits over an undirected weighted graph:
//! [`Kruskal`] O(E log E), [`Prim`] and [`Boruvka`] O(E log V). Self-loops are
//! ignored and each unordered pair is taken once. The three agree on the total
//! weight, which is unique even when the edge set is not (tied weights).
//!
//! References: Kruskal (1956), Prim (1957), Borůvka (1926).

mod boruvka;
mod common;
mod kruskal;
mod prim;

pub use boruvka::Boruvka;
pub use common::MinimumSpanningTreeError;
pub use kruskal::Kruskal;
pub use prim::Prim;
