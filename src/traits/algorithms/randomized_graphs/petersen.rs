//! Generator for the Petersen graph.
#![cfg(feature = "alloc")]

use crate::impls::{CSR2D, SymmetricCSR2D};

use super::builder_utils::build_symmetric;

/// Returns the Petersen graph (10 vertices, 15 edges).
#[must_use]
pub fn petersen_graph() -> SymmetricCSR2D<CSR2D<usize, usize, usize>> {
    let edges = vec![
        // Outer cycle
        (0, 1),
        (0, 4),
        // Spokes
        (0, 5),
        (1, 2),
        (1, 6),
        (2, 3),
        (2, 7),
        (3, 4),
        (3, 8),
        (4, 9),
        // Inner pentagram
        (5, 7),
        (5, 8),
        (6, 8),
        (6, 9),
        (7, 9),
    ];
    build_symmetric(10, edges)
}
