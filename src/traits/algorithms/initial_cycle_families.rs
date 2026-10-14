//! Internal helpers for exact cycle-family style algorithms on undirected
//! simple graphs.
//!
//! The current implementation is correctness-first and materializes simple
//! cycles explicitly per cyclic biconnected component. This is sufficient for
//! exact `relevant_cycles` and `essential_cycles` support, while leaving room
//! for a later compact Vismara-style family representation.

use alloc::{collections::BTreeMap, vec, vec::Vec};

use crate::{
    errors::{MonopartiteError, monopartite_graph_error::MonopartiteAlgorithmError},
    traits::{
        BiconnectedComponents, BiconnectedComponentsError, PositiveInteger,
        UndirectedMonopartiteMonoplexGraph,
    },
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CycleCandidate<NodeId: PositiveInteger> {
    pub(crate) nodes: Vec<NodeId>,
    pub(crate) edge_bits: Vec<u64>,
    pub(crate) length: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct LocalCycleComponent<NodeId: PositiveInteger> {
    vertices: Vec<NodeId>,
    adjacency: Vec<Vec<usize>>,
    edge_lookup: Vec<usize>,
    edge_count: usize,
}

impl<NodeId: PositiveInteger> LocalCycleComponent<NodeId> {
    fn from_edges(mut edges: Vec<[NodeId; 2]>) -> Self {
        edges.sort_unstable();
        edges.dedup();

        let mut vertices = edges.iter().flat_map(|edge| [edge[0], edge[1]]).collect::<Vec<_>>();
        vertices.sort_unstable();
        vertices.dedup();

        let mut global_to_local = BTreeMap::new();
        for (local_id, &vertex) in vertices.iter().enumerate() {
            global_to_local.insert(vertex, local_id);
        }

        let mut adjacency = vec![Vec::new(); vertices.len()];
        let mut edge_lookup = vec![usize::MAX; vertices.len() * vertices.len()];
        let mut edge_count = 0usize;
        for [left, right] in edges {
            let left_local = global_to_local[&left];
            let right_local = global_to_local[&right];
            adjacency[left_local].push(right_local);
            adjacency[right_local].push(left_local);
            let edge_id = edge_count;
            edge_count += 1;
            edge_lookup[left_local * vertices.len() + right_local] = edge_id;
            edge_lookup[right_local * vertices.len() + left_local] = edge_id;
        }

        for neighbors in &mut adjacency {
            neighbors.sort_unstable();
        }
        Self { vertices, adjacency, edge_lookup, edge_count }
    }

    fn enumerate_cycle_candidates(&self) -> Vec<CycleCandidate<NodeId>> {
        let mut cycles = Vec::new();
        let mut visited = vec![false; self.vertices.len()];
        let mut path = Vec::new();

        for start in 0..self.vertices.len() {
            visited[start] = true;
            path.clear();
            path.push(start);
            enumerate_from(start, start, &self.adjacency, &mut visited, &mut path, &mut cycles);
            visited[start] = false;
        }

        cycles.sort_unstable();
        cycles.dedup();

        let mut candidates = cycles
            .into_iter()
            .map(|cycle| {
                let mut edge_bits = vec![0_u64; self.edge_count.div_ceil(64)];
                for pair in cycle.windows(2) {
                    set_bit(
                        &mut edge_bits,
                        self.edge_lookup[pair[0] * self.vertices.len() + pair[1]],
                    );
                }
                let last = cycle[cycle.len() - 1];
                let first = cycle[0];
                set_bit(&mut edge_bits, self.edge_lookup[last * self.vertices.len() + first]);

                let nodes = cycle.iter().map(|&node| self.vertices[node]).collect::<Vec<_>>();
                CycleCandidate { length: nodes.len(), nodes, edge_bits }
            })
            .collect::<Vec<_>>();
        candidates.sort_unstable_by(|left, right| {
            left.length.cmp(&right.length).then_with(|| left.nodes.cmp(&right.nodes))
        });
        candidates
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct XorBasis {
    rows_by_pivot: Vec<Option<Vec<u64>>>,
    rank: usize,
}

impl XorBasis {
    pub(crate) fn new(width: usize) -> Self {
        Self { rows_by_pivot: vec![None; width], rank: 0 }
    }

    pub(crate) fn rank(&self) -> usize {
        self.rank
    }

    pub(crate) fn is_independent(&self, row: &[u64]) -> bool {
        highest_set_bit(&self.reduce(row.to_vec())).is_some()
    }

    pub(crate) fn insert(&mut self, row: Vec<u64>) -> bool {
        let reduced = self.reduce(row);
        let Some(pivot) = highest_set_bit(&reduced) else {
            return false;
        };
        self.rows_by_pivot[pivot] = Some(reduced);
        self.rank += 1;
        true
    }

    fn reduce(&self, mut row: Vec<u64>) -> Vec<u64> {
        while let Some(pivot) = highest_set_bit(&row) {
            if let Some(existing) = &self.rows_by_pivot[pivot] {
                xor_assign(&mut row, existing);
            } else {
                break;
            }
        }
        row
    }
}

pub(crate) fn cyclic_component_cycle_candidates<G: UndirectedMonopartiteMonoplexGraph>(
    graph: &G,
) -> Result<Vec<Vec<CycleCandidate<G::NodeId>>>, BiconnectedComponentsError> {
    let components = graph.biconnected_components().map_err(|error| {
        match error {
            MonopartiteError::AlgorithmError(
                MonopartiteAlgorithmError::BiconnectedComponentsError(error),
            ) => error,
            _ => unreachable!("biconnected_components only returns biconnected-components errors"),
        }
    })?;
    let mut candidates = Vec::new();

    for component_id in components.cyclic_biconnected_component_ids() {
        let component = LocalCycleComponent::from_edges(
            components.edge_biconnected_component(component_id).to_vec(),
        );
        let cycles = component.enumerate_cycle_candidates();
        if !cycles.is_empty() {
            candidates.push(cycles);
        }
    }

    Ok(candidates)
}

fn enumerate_from(
    start: usize,
    current: usize,
    adjacency: &[Vec<usize>],
    visited: &mut [bool],
    path: &mut Vec<usize>,
    cycles: &mut Vec<Vec<usize>>,
) {
    for &neighbor in &adjacency[current] {
        if neighbor == start {
            if path.len() >= 3 {
                cycles.push(normalize_cycle(path.clone()));
            }
            continue;
        }
        if visited[neighbor] || neighbor < start {
            continue;
        }
        visited[neighbor] = true;
        path.push(neighbor);
        enumerate_from(start, neighbor, adjacency, visited, path, cycles);
        path.pop();
        visited[neighbor] = false;
    }
}

fn normalize_cycle(mut cycle: Vec<usize>) -> Vec<usize> {
    if cycle.is_empty() {
        return cycle;
    }

    let smallest_position =
        cycle.iter().enumerate().min_by_key(|(_, node)| *node).map_or(0, |(position, _)| position);
    cycle.rotate_left(smallest_position);
    if cycle.len() > 2 && cycle[cycle.len() - 1] < cycle[1] {
        cycle[1..].reverse();
    }
    cycle
}

#[inline]
fn set_bit(bits: &mut [u64], bit: usize) {
    bits[bit / 64] |= 1_u64 << (bit % 64);
}

#[inline]
fn xor_assign(left: &mut [u64], right: &[u64]) {
    for (left_word, right_word) in left.iter_mut().zip(right.iter()) {
        *left_word ^= *right_word;
    }
}

#[inline]
fn highest_set_bit(bits: &[u64]) -> Option<usize> {
    for (word_index, &word) in bits.iter().enumerate().rev() {
        if word != 0 {
            let bit_offset = 63 - word.leading_zeros() as usize;
            return Some(word_index * 64 + bit_offset);
        }
    }
    None
}
