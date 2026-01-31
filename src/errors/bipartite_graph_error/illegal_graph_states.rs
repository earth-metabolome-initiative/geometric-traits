//! Submodule defining illegal graph states that, if reached, indicate a bug in
//! some implementation of the graph traits.

use crate::traits::BipartiteGraph;

#[derive(Debug, thiserror::Error, Clone, PartialEq, Eq)]
/// Error enumeration relative to illegal graph states.
pub enum IllegalBipartiteGraphState<G: BipartiteGraph + ?Sized> {
    /// The maximal number of left nodes of the graph is larger than the number
    /// of nodes that can be represented by the graph's left node ID type.
    /// This should be impossible to reach, and indicates some bug in the
    /// implementation of the graph traits.
    #[error("The maximal number of left nodes of the graph {number_of_left_nodes} is larger than the number of nodes that can be represented by the graph's left node ID type.")]
    TooManyLeftNodes {
        /// The number of left nodes that was reported.
        number_of_left_nodes: usize,
    },
    /// The maximal number of right nodes of the graph is larger than the number
    /// of nodes that can be represented by the graph's right node ID type.
    /// This should be impossible to reach, and indicates some bug in the
    /// implementation of the graph traits.
    #[error("The maximal number of right nodes of the graph {number_of_right_nodes} is larger than the number of nodes that can be represented by the graph's right node ID type.")]
    TooManyRightNodes {
        /// The number of right nodes that was reported.
        number_of_right_nodes: usize,
    },
    /// `PhantomPlaceholder`
    #[error("PhantomPlaceholder")]
    PhantomPlaceholder(core::marker::PhantomData<G>),
}
