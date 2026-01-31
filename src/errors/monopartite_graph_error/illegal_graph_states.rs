//! Submodule defining illegal graph states that, if reached, indicate a bug in
//! some implementation of the graph traits.

use crate::traits::MonopartiteGraph;

#[derive(Debug, thiserror::Error, Clone, PartialEq, Eq)]
/// Error enumeration relative to illegal graph states.
pub enum IllegalMonopartiteGraphState<G: MonopartiteGraph + ?Sized> {
    /// The maximal number of nodes of the graph is larger than the number of
    /// nodes that can be represented by the graph's node ID type. This
    /// should be impossible to reach, and indicates some bug in the
    /// implementation of the graph traits.
    #[error("The maximal number of nodes of the graph {number_of_nodes} is larger than the number of nodes that can be represented by the graph's node ID type.")]
    TooManyNodes {
        /// The number of nodes that was reported.
        number_of_nodes: usize,
    },
    /// `PhantomPlaceholder`
    #[error("PhantomPlaceholder")]
    PhantomPlaceholder(core::marker::PhantomData<G>),
}
