//! Submodule providing the `ConnectedComponents` trait and its primary methods.
use alloc::vec::Vec;

use num_traits::{One, Zero};

use num_traits::AsPrimitive;

use crate::traits::{
    MonopartiteGraph, PositiveInteger, UndirectedMonopartiteMonoplexGraph,
};

/// Connected components object.
pub struct ConnectedComponentsResult<'a, G: MonopartiteGraph, Marker = usize> {
    /// Identifiers of the connected components.
    component_identifiers: Vec<Marker>,
    /// Underlying graph.
    graph: &'a G,
    /// Precomputed number of connected components.
    number_of_components: Marker,
    /// Size of the largest connected component.
    largest_component_size: G::NodeId,
    /// Size of the smallest connected component.
    smallest_component_size: G::NodeId,
}

impl<G: UndirectedMonopartiteMonoplexGraph, Marker: PositiveInteger>
    ConnectedComponentsResult<'_, G, Marker>
where
    G::NodeId: AsPrimitive<usize>,
{
    /// Returns the number of connected components in the graph.
    pub fn number_of_components(&self) -> Marker {
        self.number_of_components
    }

    /// Returns the size of the largest connected component.
    pub fn largest_component_size(&self) -> G::NodeId {
        self.largest_component_size
    }

    /// Returns the size of the smallest connected component.
    pub fn smallest_component_size(&self) -> G::NodeId {
        self.smallest_component_size
    }

    /// Returns the connected component of a node.
    pub fn component_of_node(&self, node: G::NodeId) -> Marker {
        self.component_identifiers[node.as_()]
    }

    /// Returns an iterator over the connected component identifiers.
    pub fn component_identifiers(&self) -> core::iter::Copied<core::slice::Iter<'_, Marker>> {
        self.component_identifiers.iter().copied()
    }

    /// Returns an iterator over the node ids of a connected component.
    pub fn node_ids_of_component(
        &self,
        component_identifier: Marker,
    ) -> impl Iterator<Item = G::NodeId> + '_ {
        self.graph.node_ids().zip(self.component_identifiers.iter()).filter_map(
            move |(node, &component)| (component == component_identifier).then_some(node),
        )
    }

    /// Returns an iterator over the symbols of the nodes of a connected
    /// component.
    pub fn nodes_of_component(
        &self,
        component_identifier: Marker,
    ) -> impl Iterator<Item = G::NodeSymbol> + '_ {
        self.graph.nodes().zip(self.component_identifiers.iter()).filter_map(
            move |(symbol, &component)| (component == component_identifier).then_some(symbol),
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
/// Error type for connected components.
pub enum ConnectedComponentsError {
    /// The graph has too many connected components for the provided marker
    /// type.
    TooManyComponents,
}

impl core::fmt::Display for ConnectedComponentsError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            ConnectedComponentsError::TooManyComponents => {
                write!(
                    f,
                    "The graph has too many connected components for the provided marker type."
                )
            }
        }
    }
}

impl From<ConnectedComponentsError>
    for crate::errors::monopartite_graph_error::algorithms::MonopartiteAlgorithmError
{
    fn from(error: ConnectedComponentsError) -> Self {
        Self::ConnectedComponentsError(error)
    }
}

impl<G: MonopartiteGraph> From<ConnectedComponentsError> for crate::errors::MonopartiteError<G> {
    fn from(error: ConnectedComponentsError) -> Self {
        Self::AlgorithmError(error.into())
    }
}

/// Trait for connected components.
///
/// # Type parameter
///
/// - `Marker`: The type used to identify the connected components. This should
///   be the smallest positive integer which is expected to be able to represent
///   the number of connected components in the graph. On very large graphs
///   which are expected to be strongly connected, choosing a smaller integer
///   type may save a significant amount of memory.
pub trait ConnectedComponents<Marker: AsPrimitive<usize> + PositiveInteger = usize>:
    UndirectedMonopartiteMonoplexGraph + Sized
{
    /// Returns the number of connected components in the graph.
    ///
    /// # Errors
    ///
    /// * If the graph has too many connected components for the provided marker
    ///   type.
    /// * If the graph has too many nodes.
    ///
    /// # Complexity
    ///
    /// O(V + E) time and O(V) space.
    ///
    /// # Examples
    ///
    /// ```
    /// use geometric_traits::{
    ///     impls::{SortedVec, SymmetricCSR2D},
    ///     prelude::*,
    ///     traits::{
    ///         EdgesBuilder, VocabularyBuilder,
    ///         algorithms::connected_components::ConnectedComponentsResult,
    ///     },
    /// };
    ///
    /// let nodes: Vec<usize> = vec![0, 1, 2, 3];
    /// let edges: Vec<(usize, usize)> = vec![(0, 1), (2, 3)];
    /// let nodes: SortedVec<usize> = GenericVocabularyBuilder::default()
    ///     .expected_number_of_symbols(nodes.len())
    ///     .symbols(nodes.into_iter().enumerate())
    ///     .build()
    ///     .unwrap();
    /// let edges: SymmetricCSR2D<_> = UndiEdgesBuilder::default()
    ///     .expected_number_of_edges(edges.len())
    ///     .expected_shape(nodes.len())
    ///     .edges(edges.into_iter())
    ///     .build()
    ///     .unwrap();
    /// let graph: UndiGraph<usize> = UndiGraph::from((nodes, edges));
    ///
    /// let cc: ConnectedComponentsResult<_, usize> = graph.connected_components().unwrap();
    /// assert_eq!(cc.number_of_components(), 2);
    /// ```
    fn connected_components(
        &self,
    ) -> Result<ConnectedComponentsResult<'_, Self, Marker>, crate::errors::MonopartiteError<Self>>
    {
        let mut component_identifiers: Vec<Marker> =
            vec![Marker::max_value(); self.number_of_nodes().as_()];
        let mut number_of_components: Marker = Marker::zero();
        let mut largest_component_size: Self::NodeId = Self::NodeId::zero();
        let mut smallest_component_size: Self::NodeId = self.number_of_nodes();

        let mut frontier: Vec<Self::NodeId> = Vec::new();
        let mut temporary_frontier: Vec<Self::NodeId> = Vec::new();

        for node in self.node_ids() {
            // If the node is already marked as part of a component, skip it.
            if component_identifiers[node.as_()] != Marker::max_value() {
                continue;
            }
            // Otherwise, we have found a new component and need to mark all nodes in it.
            let mut current_component_size = Self::NodeId::zero();

            // Add the current node to the frontier.
            frontier.push(node);

            while !frontier.is_empty() {
                // For each node in the frontier, mark it and add its neighbors to the frontier.
                for neighbour in frontier.drain(..) {
                    // If the neighbour is already marked as part of a component, skip it.
                    if component_identifiers[neighbour.as_()] != Marker::max_value() {
                        continue;
                    }

                    // Mark the neighbour as part of the current component.
                    component_identifiers[neighbour.as_()] = number_of_components;

                    // Increment the size of the current component.
                    current_component_size += Self::NodeId::one();

                    // Add the neighbors of this node to the temporary frontier.
                    temporary_frontier.extend(self.neighbors(neighbour));
                }

                // We swap the temporary frontier with the frontier to avoid allocating a new
                // vector.
                core::mem::swap(&mut frontier, &mut temporary_frontier);
            }

            // Update the size of the largest and smallest components.
            if current_component_size > largest_component_size {
                largest_component_size = current_component_size;
            }
            if current_component_size < smallest_component_size {
                smallest_component_size = current_component_size;
            }

            number_of_components += Marker::one();

            // If the number of components exceeds the maximum value of the marker type,
            // return an error.
            if number_of_components == Marker::max_value() {
                return Err(ConnectedComponentsError::TooManyComponents.into());
            }
        }

        Ok(ConnectedComponentsResult {
            component_identifiers,
            graph: self,
            number_of_components,
            largest_component_size,
            smallest_component_size,
        })
    }
}

impl<G: UndirectedMonopartiteMonoplexGraph, Marker: AsPrimitive<usize> + PositiveInteger>
    ConnectedComponents<Marker> for G
{
}
