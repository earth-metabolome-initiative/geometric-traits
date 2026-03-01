//! Submodule providing the randomized dag trait to generate randomized dag with
//! provided parameters.
#![cfg(feature = "alloc")]
use alloc::vec::Vec;
#[cfg(feature = "std")]
use std::collections::HashSet;

#[cfg(all(feature = "hashbrown", not(feature = "std")))]
use hashbrown::HashSet;

use crate::traits::{GrowableEdges, MonoplexGraph, MonoplexMonopartiteGraph, SparseMatrixMut};

/// Trait providing randomized dag method
pub trait RandomizedDAG: MonoplexGraph {
    /// returns a randomized dag within given parameters
    ///
    /// # Arguments
    /// - `seed`: the random seed the dag is generated from
    /// - `nodes`: number of the nodes the dag is generated from
    fn randomized_dag(seed: u64, nodes: usize) -> Self;
}

impl<G> RandomizedDAG for G
where
    G: MonoplexMonopartiteGraph<Nodes = usize> + From<(G::Nodes, G::MonoplexMonopartiteEdges)>,
    G::MonoplexMonopartiteEdges: GrowableEdges<EdgeId = usize, Edge = (usize, usize)>,
    <G::MonoplexMonopartiteEdges as GrowableEdges>::GrowableMatrix:
        SparseMatrixMut<MinimalShape = usize>,
{
    #[allow(clippy::cast_possible_truncation)]
    fn randomized_dag(seed: u64, nodes: usize) -> Self {
        if nodes <= 1 {
            let edges = G::MonoplexMonopartiteEdges::with_shaped_capacity(nodes, 0);
            return G::from((nodes, edges));
        }

        // XorShift64 with a zero state remains zero forever.
        // Map seed 0 to a fixed non-zero state to avoid degenerate empty DAGs.
        let normalized_seed = if seed == 0 { 0x9E37_79B9_7F4A_7C15 } else { seed };
        let mut xorshift = XorShift64::from(normalized_seed);
        let nodes_u64 = nodes as u64;
        let max_number_of_edges = nodes_u64 * (nodes_u64 - 1) / 2;
        let number_of_edges_u64 = xorshift.next().unwrap() % (max_number_of_edges + 1);
        let number_of_edges = usize::try_from(number_of_edges_u64).unwrap();

        let mut edge_tuples = HashSet::with_capacity(number_of_edges);
        while edge_tuples.len() < number_of_edges {
            let seed1 = xorshift.next().unwrap();
            let seed2 = xorshift.next().unwrap();
            let mut src = (seed1 % nodes_u64) as usize;
            let mut dst = (seed2 % nodes_u64) as usize;
            if src == dst {
                continue;
            }
            if src > dst {
                core::mem::swap(&mut src, &mut dst);
            }
            edge_tuples.insert((src, dst));
        }
        let mut sorted_edge_tuples: Vec<(usize, usize)> = edge_tuples.into_iter().collect();
        sorted_edge_tuples.sort_unstable();
        let mut edges = G::MonoplexMonopartiteEdges::with_shaped_capacity(nodes, number_of_edges);
        for (src, dst) in sorted_edge_tuples {
            edges.add((src, dst)).unwrap();
        }
        G::from((nodes, edges))
    }
}

/// Struct for storing the `XorShift64` state
pub struct XorShift64(u64);

impl From<u64> for XorShift64 {
    fn from(state: u64) -> Self {
        Self(state)
    }
}

impl Iterator for XorShift64 {
    type Item = u64;

    fn next(&mut self) -> Option<Self::Item> {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        Some(x)
    }
}
