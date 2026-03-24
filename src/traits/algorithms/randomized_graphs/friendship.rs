//! Generator for friendship graphs (F_n).
#![cfg(feature = "alloc")]

use alloc::vec::Vec;

use super::{builder_utils::build_symmetric, windmill_graph};
use crate::impls::{CSR2D, SymmetricCSR2D};

/// Returns the friendship graph F_n: n triangles sharing a universal hub vertex
/// 0.
///
/// 2n+1 vertices, 3n edges.
///
/// This is the triangular special case of [`windmill_graph`] with
/// `clique_size = 3`.
///
/// For historical compatibility, `friendship_graph(0)` returns a single
/// isolated hub vertex.
#[must_use]
pub fn friendship_graph(n: usize) -> SymmetricCSR2D<CSR2D<usize, usize, usize>> {
    if n == 0 {
        return build_symmetric(1, Vec::new());
    }

    windmill_graph(n, 3)
}
