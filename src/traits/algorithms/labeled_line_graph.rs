//! Submodule providing the `LabeledLineGraph` trait.
//!
//! Given a graph G whose nodes carry types (via [`TypedNode`]), the labeled
//! line graph L(G) has one vertex per edge of G.  Two vertices in L(G) are
//! adjacent iff the corresponding edges in G share an endpoint, and the edge
//! in L(G) carries the **node type of the shared endpoint** as its label.
//!
//! This is the key building block for a *labeled* MCES pipeline: the edge
//! labels in L(G) encode which node type is "between" two original edges,
//! allowing the labeled modular product to filter incompatible pairings.

use alloc::vec::Vec;

use num_traits::AsPrimitive;

use super::line_graph::LineGraphResult;
use crate::{
    impls::{SymmetricCSR2D, ValuedCSR2D},
    traits::{MonoplexMonopartiteGraph, TypedNode},
};

/// Trait providing labeled line graph construction for node-typed
/// monopartite graphs.
pub trait LabeledLineGraph: MonoplexMonopartiteGraph
where
    Self::NodeSymbol: TypedNode,
{
    /// Computes the undirected labeled line graph L(G).
    ///
    /// Each edge in L(G) is labeled with the node type of the shared
    /// endpoint in G. Edges are filtered to `src < dst` (no self-loops).
    ///
    /// # Complexity
    /// O(sum of deg(v)^2) time + O(|E(L(G))| log |E(L(G))|) for sorting.
    #[allow(clippy::type_complexity)]
    fn labeled_line_graph(
        &self,
    ) -> LineGraphResult<
        SymmetricCSR2D<ValuedCSR2D<usize, usize, usize, <Self::NodeSymbol as TypedNode>::NodeType>>,
        Self::NodeId,
    >;
}

impl<G: MonoplexMonopartiteGraph> LabeledLineGraph for G
where
    G::NodeId: AsPrimitive<usize>,
    G::NodeSymbol: TypedNode,
{
    fn labeled_line_graph(
        &self,
    ) -> LineGraphResult<
        SymmetricCSR2D<ValuedCSR2D<usize, usize, usize, <Self::NodeSymbol as TypedNode>::NodeType>>,
        Self::NodeId,
    > {
        type NT<G> = <<G as crate::traits::MonopartiteGraph>::NodeSymbol as TypedNode>::NodeType;

        let n: usize = self.number_of_nodes().as_();

        // Step 1: Collect edges with src < dst (undirected, no self-loops).
        let edge_map: Vec<(Self::NodeId, Self::NodeId)> =
            self.sparse_coordinates().filter(|(src, dst)| src < dst).collect();
        let m = edge_map.len();

        if m == 0 {
            let graph =
                SymmetricCSR2D::<ValuedCSR2D<usize, usize, usize, NT<G>>>::from_sorted_upper_triangular_entries(
                    0,
                    Vec::new(),
                )
                .expect("empty line graph must build successfully");
            return LineGraphResult::new(graph, edge_map);
        }

        // Step 2: Precompute node types for O(1) lookup.
        let node_types: Vec<NT<G>> = self.nodes().map(|sym| sym.node_type()).collect();

        // Step 3: Build incidence lists — for each vertex, the list of edge
        // indices.
        let mut incident: Vec<Vec<usize>> = vec![Vec::new(); n];
        for (idx, &(src, dst)) in edge_map.iter().enumerate() {
            incident[src.as_()].push(idx);
            incident[dst.as_()].push(idx);
        }

        // Step 4: For each vertex, connect all pairs of incident edges and
        // record the shared vertex's node type as the edge label.
        let mut lg_edges: Vec<(usize, usize, NT<G>)> = Vec::new();
        for (v, inc) in incident.iter().enumerate() {
            let len = inc.len();
            if len < 2 {
                continue;
            }
            let v_type = &node_types[v];
            for i in 0..len {
                for j in (i + 1)..len {
                    let a = inc[i];
                    let b = inc[j];
                    let (lo, hi) = if a < b { (a, b) } else { (b, a) };
                    lg_edges.push((lo, hi, v_type.clone()));
                }
            }
        }

        // Step 5: Sort and dedup (two edges can share at most one endpoint in
        // a simple graph, but defensive).
        lg_edges.sort_unstable_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
        lg_edges.dedup_by(|a, b| a.0 == b.0 && a.1 == b.1);

        let graph =
            SymmetricCSR2D::<ValuedCSR2D<usize, usize, usize, NT<G>>>::from_sorted_upper_triangular_entries(
                m,
                lg_edges,
            )
            .expect("labeled line graph edges must be sorted upper-triangular entries");
        LineGraphResult::new(graph, edge_map)
    }
}
