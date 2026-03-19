//! Submodule providing the `LineGraph` trait for computing line graphs.
//!
//! Given a graph G, the line graph L(G) has one vertex per edge of G.
//! Two vertices in L(G) are adjacent iff the corresponding edges in G
//! share an endpoint.
use alloc::vec::Vec;

use num_traits::AsPrimitive;

use crate::{
    impls::{CSR2D, SquareCSR2D, SymmetricCSR2D},
    traits::MonoplexMonopartiteGraph,
};

/// Result of computing a line graph L(G).
///
/// Contains the line graph itself and a mapping from each vertex in L(G)
/// back to the original edge (src, dst) in G.
pub struct LineGraphResult<M, NodeId> {
    graph: M,
    edge_map: Vec<(NodeId, NodeId)>,
}

impl<M, NodeId: Copy> LineGraphResult<M, NodeId> {
    /// Returns a reference to the line graph.
    #[inline]
    pub fn graph(&self) -> &M {
        &self.graph
    }

    /// Consumes the result and returns the line graph.
    #[inline]
    pub fn into_graph(self) -> M {
        self.graph
    }

    /// Returns the mapping from line graph vertices to original edges.
    #[inline]
    pub fn edge_map(&self) -> &[(NodeId, NodeId)] {
        &self.edge_map
    }

    /// Returns the number of vertices in the line graph.
    #[inline]
    pub fn number_of_vertices(&self) -> usize {
        self.edge_map.len()
    }

    /// Returns the original edge corresponding to a vertex in the line graph.
    #[inline]
    pub fn original_edge(&self, vertex: usize) -> (NodeId, NodeId) {
        self.edge_map[vertex]
    }
}

/// Trait providing line graph construction for monoplex monopartite graphs.
pub trait LineGraph: MonoplexMonopartiteGraph {
    /// Computes the undirected line graph L(G).
    ///
    /// Edges are filtered to `src < dst`, which naturally excludes self-loops
    /// (standard for undirected simple graphs per Krausz 1943, Beineke 1968).
    ///
    /// # Complexity
    /// O(sum of deg(v)^2) time + O(|E(L(G))| log |E(L(G))|) for sorting.
    fn line_graph(
        &self,
    ) -> LineGraphResult<SymmetricCSR2D<CSR2D<usize, usize, usize>>, Self::NodeId>;

    /// Computes the directed line digraph L(G).
    ///
    /// Edge i is adjacent to edge j in L(G) iff the head of i equals the tail
    /// of j. Self-loops in G are preserved per Hemminger & Beineke (1978).
    ///
    /// # Complexity
    /// O(sum of in_deg(v) * out_deg(v)) time + sorting.
    fn directed_line_graph(
        &self,
    ) -> LineGraphResult<SquareCSR2D<CSR2D<usize, usize, usize>>, Self::NodeId>;
}

impl<G: MonoplexMonopartiteGraph> LineGraph for G
where
    G::NodeId: AsPrimitive<usize>,
{
    fn line_graph(
        &self,
    ) -> LineGraphResult<SymmetricCSR2D<CSR2D<usize, usize, usize>>, Self::NodeId> {
        let n: usize = self.number_of_nodes().as_();

        // Step 1: Collect edges with src < dst (undirected, no self-loops).
        let edge_map: Vec<(Self::NodeId, Self::NodeId)> =
            self.sparse_coordinates().filter(|(src, dst)| src < dst).collect();
        let m = edge_map.len();

        if m == 0 {
            let graph =
                crate::traits::algorithms::randomized_graphs::builder_utils::build_symmetric(
                    0,
                    Vec::new(),
                );
            return LineGraphResult { graph, edge_map };
        }

        // Step 2: Build incidence lists — for each vertex, the list of edge indices.
        let mut incident: Vec<Vec<usize>> = vec![Vec::new(); n];
        for (idx, &(src, dst)) in edge_map.iter().enumerate() {
            incident[src.as_()].push(idx);
            incident[dst.as_()].push(idx);
        }

        // Step 3: For each vertex, connect all pairs of incident edges.
        let mut lg_edges: Vec<(usize, usize)> = Vec::new();
        for inc in &incident {
            let len = inc.len();
            for i in 0..len {
                for j in (i + 1)..len {
                    lg_edges.push((inc[i], inc[j]));
                }
            }
        }

        // Step 4: Sort and deduplicate, then build.
        lg_edges.sort_unstable();
        lg_edges.dedup();
        let graph = crate::traits::algorithms::randomized_graphs::builder_utils::build_symmetric(
            m, lg_edges,
        );
        LineGraphResult { graph, edge_map }
    }

    fn directed_line_graph(
        &self,
    ) -> LineGraphResult<SquareCSR2D<CSR2D<usize, usize, usize>>, Self::NodeId> {
        let n: usize = self.number_of_nodes().as_();

        // Step 1: Collect all edges (including self-loops).
        let edge_map: Vec<(Self::NodeId, Self::NodeId)> = self.sparse_coordinates().collect();
        let m = edge_map.len();

        if m == 0 {
            let graph = crate::traits::algorithms::randomized_graphs::builder_utils::build_directed(
                0,
                Vec::new(),
            );
            return LineGraphResult { graph, edge_map };
        }

        // Step 2: Build outgoing[v] and incoming[v] edge index lists.
        let mut outgoing: Vec<Vec<usize>> = vec![Vec::new(); n];
        let mut incoming: Vec<Vec<usize>> = vec![Vec::new(); n];
        for (idx, &(src, dst)) in edge_map.iter().enumerate() {
            outgoing[src.as_()].push(idx);
            incoming[dst.as_()].push(idx);
        }

        // Step 3: For each vertex v, connect every incoming edge to every outgoing
        // edge.
        let mut lg_edges: Vec<(usize, usize)> = Vec::new();
        for v in 0..n {
            for &i in &incoming[v] {
                for &j in &outgoing[v] {
                    lg_edges.push((i, j));
                }
            }
        }

        // Step 4: Sort and build.
        lg_edges.sort_unstable();
        let graph = crate::traits::algorithms::randomized_graphs::builder_utils::build_directed(
            m, lg_edges,
        );
        LineGraphResult { graph, edge_map }
    }
}
