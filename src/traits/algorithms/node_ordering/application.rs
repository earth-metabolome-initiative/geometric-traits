use alloc::vec::Vec;

use multi_ranged::Step;
use num_traits::AsPrimitive;

use crate::{
    impls::{CSR2D, SquareCSR2D, SymmetricCSR2D, UpperTriangularCSR2D, ValuedCSR2D},
    naive_structs::{GenericGraph, GenericUndirectedMonopartiteEdgesBuilder},
    traits::{
        BidirectionalVocabulary, Edge, Edges, EdgesBuilder, GrowableEdges, MonopartiteGraph,
        MonoplexGraph, PositiveInteger, SparseValuedMatrixRef, TryFromUsize,
        algorithms::ModularProductGraph,
    },
};

/// Trait for vocabularies whose entries can be reordered while preserving
/// their own vocabulary semantics.
pub trait PermutableVocabulary: BidirectionalVocabulary + Sized {
    /// Returns the vocabulary permuted according to `order`.
    ///
    /// The slice contains the original source symbol now placed at each new
    /// position.
    #[must_use]
    fn permuted(&self, order: &[Self::SourceSymbol]) -> Self;
}

impl<V> PermutableVocabulary for Vec<V>
where
    V: crate::traits::Symbol,
{
    #[inline]
    fn permuted(&self, order: &[Self::SourceSymbol]) -> Self {
        order.iter().map(|&index| self[index].clone()).collect()
    }
}

/// Validated node permutation derived from a complete node ordering.
#[derive(Clone, Debug, PartialEq, Eq)]
struct NodePermutation {
    new_to_old: Vec<usize>,
}

impl NodePermutation {
    /// Builds a validated permutation from a complete node order.
    ///
    /// # Panics
    ///
    /// Panics if `order` is not a permutation of `0..number_of_nodes`.
    #[must_use]
    fn from_order<NodeId>(order: &[NodeId], number_of_nodes: usize) -> Self
    where
        NodeId: Copy + AsPrimitive<usize>,
    {
        assert_eq!(
            order.len(),
            number_of_nodes,
            "node order must contain exactly one entry per node"
        );

        let mut seen = vec![false; number_of_nodes];
        let new_to_old: Vec<usize> = order.iter().map(|node_id| (*node_id).as_()).collect();
        for &old_index in &new_to_old {
            assert!(
                old_index < number_of_nodes,
                "node order contains out-of-range node {old_index}"
            );
            assert!(!seen[old_index], "node order contains duplicate node {old_index}");
            seen[old_index] = true;
        }

        Self { new_to_old }
    }

    /// Returns the inverse permutation mapping each original node id to its new
    /// position.
    #[must_use]
    fn inverse<NodeId>(&self) -> Vec<NodeId>
    where
        NodeId: PositiveInteger + TryFromUsize,
    {
        let mut old_to_new = vec![NodeId::zero(); self.new_to_old.len()];
        for (new_index, &old_index) in self.new_to_old.iter().enumerate() {
            old_to_new[old_index] = NodeId::try_from_usize(new_index)
                .unwrap_or_else(|_| panic!("node permutation must fit in the node id type"));
        }
        old_to_new
    }
}

/// Trait for graph families that can rebuild themselves after a node
/// permutation.
pub trait NodeOrderApplicableGraph: MonopartiteGraph + MonoplexGraph {
    /// Reordered graph type.
    type ReorderedGraph: MonopartiteGraph<NodeId = Self::NodeId, NodeSymbol = Self::NodeSymbol, Nodes = Self::Nodes>
        + MonoplexGraph<Edge = Self::Edge, Edges = Self::Edges>;

    /// Applies a complete node order to the graph.
    ///
    /// # Panics
    ///
    /// Panics if `order` is not a permutation of the graph node ids.
    #[must_use]
    fn apply_node_order(&self, order: &[Self::NodeId]) -> Self::ReorderedGraph;
}

/// Applies a node ordering to any graph that implements
/// [`NodeOrderApplicableGraph`].
#[must_use]
pub fn apply_node_order_to_graph<G>(graph: &G, order: &[G::NodeId]) -> G::ReorderedGraph
where
    G: NodeOrderApplicableGraph,
{
    graph.apply_node_order(order)
}

impl<Nodes, SparseIndex, NodeId> NodeOrderApplicableGraph
    for GenericGraph<Nodes, SquareCSR2D<CSR2D<SparseIndex, NodeId, NodeId>>>
where
    Nodes: PermutableVocabulary<SourceSymbol = NodeId>,
    Nodes::DestinationSymbol: crate::traits::Symbol,
    SparseIndex: PositiveInteger + AsPrimitive<usize> + TryFromUsize,
    NodeId: Step + PositiveInteger + AsPrimitive<usize> + TryFromUsize + TryFrom<SparseIndex>,
{
    type ReorderedGraph = GenericGraph<Nodes, SquareCSR2D<CSR2D<SparseIndex, NodeId, NodeId>>>;

    fn apply_node_order(&self, order: &[Self::NodeId]) -> Self::ReorderedGraph {
        let permutation = NodePermutation::from_order(order, self.number_of_nodes().as_());
        let number_of_nodes = self.number_of_nodes();
        let old_to_new = permutation.inverse::<NodeId>();
        let nodes = self.nodes_vocabulary().permuted(order);
        let mut edges: Vec<(NodeId, NodeId)> = self
            .sparse_coordinates()
            .map(|(source, destination)| (old_to_new[source.as_()], old_to_new[destination.as_()]))
            .collect();
        edges.sort_unstable();

        let edge_count = SparseIndex::try_from_usize(edges.len())
            .unwrap_or_else(|_| panic!("reordered graph contains too many edges to index"));
        let edges = {
            let mut directed_edges =
                SquareCSR2D::<CSR2D<SparseIndex, NodeId, NodeId>>::with_shaped_capacity(
                    number_of_nodes,
                    edge_count,
                );
            for edge in edges {
                directed_edges.add(edge).expect("reordered directed edges must remain valid");
            }
            directed_edges
        };

        GenericGraph::from((nodes, edges))
    }
}

impl<Nodes, SparseIndex, NodeId> NodeOrderApplicableGraph
    for GenericGraph<Nodes, SymmetricCSR2D<CSR2D<SparseIndex, NodeId, NodeId>>>
where
    Nodes: PermutableVocabulary<SourceSymbol = NodeId>,
    Nodes::DestinationSymbol: crate::traits::Symbol,
    SparseIndex: PositiveInteger + AsPrimitive<usize> + TryFromUsize,
    NodeId: Step + PositiveInteger + AsPrimitive<usize> + TryFromUsize + TryFrom<SparseIndex>,
{
    type ReorderedGraph = GenericGraph<Nodes, SymmetricCSR2D<CSR2D<SparseIndex, NodeId, NodeId>>>;

    fn apply_node_order(&self, order: &[Self::NodeId]) -> Self::ReorderedGraph {
        let permutation = NodePermutation::from_order(order, self.number_of_nodes().as_());
        let number_of_nodes = self.number_of_nodes();
        let old_to_new = permutation.inverse::<NodeId>();
        let nodes = self.nodes_vocabulary().permuted(order);
        let mut edges: Vec<(NodeId, NodeId)> = self
            .sparse_coordinates()
            .map(|(left, right)| {
                let new_left = old_to_new[left.as_()];
                let new_right = old_to_new[right.as_()];
                if new_left <= new_right { (new_left, new_right) } else { (new_right, new_left) }
            })
            .collect();
        edges.sort_unstable();
        edges.dedup();

        let edge_count = SparseIndex::try_from_usize(edges.len())
            .unwrap_or_else(|_| panic!("reordered graph contains too many edges to index"));
        let edges = GenericUndirectedMonopartiteEdgesBuilder::<
            _,
            UpperTriangularCSR2D<CSR2D<SparseIndex, NodeId, NodeId>>,
            SymmetricCSR2D<CSR2D<SparseIndex, NodeId, NodeId>>,
        >::default()
        .expected_number_of_edges(edge_count)
        .expected_shape(number_of_nodes)
        .edges(edges.into_iter())
        .build()
        .expect("reordered undirected edges must remain valid");

        GenericGraph::from((nodes, edges))
    }
}

impl<Nodes, SparseIndex, NodeId, Value> NodeOrderApplicableGraph
    for GenericGraph<Nodes, ValuedCSR2D<SparseIndex, NodeId, NodeId, Value>>
where
    Nodes: PermutableVocabulary<SourceSymbol = NodeId>,
    Nodes::DestinationSymbol: crate::traits::Symbol,
    SparseIndex: PositiveInteger + AsPrimitive<usize> + TryFromUsize,
    NodeId: Step + PositiveInteger + AsPrimitive<usize> + TryFromUsize + TryFrom<SparseIndex>,
    Value: Clone + core::fmt::Debug + 'static,
{
    type ReorderedGraph = GenericGraph<Nodes, ValuedCSR2D<SparseIndex, NodeId, NodeId, Value>>;

    fn apply_node_order(&self, order: &[Self::NodeId]) -> Self::ReorderedGraph {
        let permutation = NodePermutation::from_order(order, self.number_of_nodes().as_());
        let number_of_nodes = self.number_of_nodes();
        let old_to_new = permutation.inverse::<NodeId>();
        let nodes = self.nodes_vocabulary().permuted(order);
        let mut edges: Vec<(NodeId, NodeId, Value)> = self
            .edges()
            .matrix()
            .sparse_entries()
            .map(|((source, destination), value)| {
                (old_to_new[source.as_()], old_to_new[destination.as_()], value.clone())
            })
            .collect();
        edges.sort_unstable_by(|left, right| {
            left.source()
                .cmp(&right.source())
                .then_with(|| left.destination().cmp(&right.destination()))
        });

        let edge_count = SparseIndex::try_from_usize(edges.len())
            .unwrap_or_else(|_| panic!("reordered graph contains too many edges to index"));
        let edges = {
            let mut directed_edges =
                ValuedCSR2D::<SparseIndex, NodeId, NodeId, Value>::with_shaped_capacity(
                    (number_of_nodes, number_of_nodes),
                    edge_count,
                );
            for edge in edges {
                directed_edges.add(edge).expect("reordered directed edges must remain valid");
            }
            directed_edges
        };

        GenericGraph::from((nodes, edges))
    }
}

impl<Nodes, SparseIndex, NodeId, Value> NodeOrderApplicableGraph
    for GenericGraph<Nodes, SymmetricCSR2D<ValuedCSR2D<SparseIndex, NodeId, NodeId, Value>>>
where
    Nodes: PermutableVocabulary<SourceSymbol = NodeId>,
    Nodes::DestinationSymbol: crate::traits::Symbol,
    SparseIndex: PositiveInteger + AsPrimitive<usize> + TryFromUsize,
    NodeId: Step + PositiveInteger + AsPrimitive<usize> + TryFromUsize + TryFrom<SparseIndex>,
    Value: Clone + core::fmt::Debug + 'static,
{
    type ReorderedGraph =
        GenericGraph<Nodes, SymmetricCSR2D<ValuedCSR2D<SparseIndex, NodeId, NodeId, Value>>>;

    fn apply_node_order(&self, order: &[Self::NodeId]) -> Self::ReorderedGraph {
        let permutation = NodePermutation::from_order(order, self.number_of_nodes().as_());
        let number_of_nodes = self.number_of_nodes();
        let old_to_new = permutation.inverse::<NodeId>();
        let nodes = self.nodes_vocabulary().permuted(order);
        let mut edges: Vec<(NodeId, NodeId, Value)> = self
            .edges()
            .matrix()
            .sparse_entries()
            .map(|((left, right), value)| {
                let new_left = old_to_new[left.as_()];
                let new_right = old_to_new[right.as_()];
                if new_left <= new_right {
                    (new_left, new_right, value.clone())
                } else {
                    (new_right, new_left, value.clone())
                }
            })
            .collect();
        edges.sort_unstable_by(|left, right| {
            left.source()
                .cmp(&right.source())
                .then_with(|| left.destination().cmp(&right.destination()))
        });
        edges.dedup_by(|left, right| {
            left.source() == right.source() && left.destination() == right.destination()
        });

        let edges = SymmetricCSR2D::<ValuedCSR2D<SparseIndex, NodeId, NodeId, Value>>::from_sorted_upper_triangular_entries(number_of_nodes, edges)
            .expect("reordered undirected weighted edges must remain valid");

        GenericGraph::from((nodes, edges))
    }
}

impl<I1, I2> NodeOrderApplicableGraph for ModularProductGraph<I1, I2>
where
    I1: crate::traits::Symbol,
    I2: crate::traits::Symbol,
{
    type ReorderedGraph = ModularProductGraph<I1, I2>;

    fn apply_node_order(&self, order: &[Self::NodeId]) -> Self::ReorderedGraph {
        let permutation = NodePermutation::from_order(order, self.number_of_nodes().as_());
        let number_of_nodes = self.number_of_nodes();
        let old_to_new = permutation.inverse::<usize>();
        let nodes = self.nodes_vocabulary().permuted(order);
        let mut edges: Vec<(usize, usize)> = self
            .sparse_coordinates()
            .map(|(left, right)| {
                let new_left = old_to_new[left];
                let new_right = old_to_new[right];
                if new_left <= new_right { (new_left, new_right) } else { (new_right, new_left) }
            })
            .collect();
        edges.sort_unstable();
        edges.dedup();

        let edges = crate::impls::BitSquareMatrix::from_symmetric_edges(number_of_nodes, edges);
        ModularProductGraph::new(edges, nodes)
    }
}
