//! Exact-oracle tests for the graph-level minimum-cycle-basis trait.
#![cfg(feature = "std")]

use std::collections::{BTreeSet, VecDeque};

use geometric_traits::{
    impls::{CSR2D, SortedVec, SymmetricCSR2D},
    prelude::*,
    traits::{EdgesBuilder, VocabularyBuilder},
};

fn build_undigraph(node_count: usize, edges: &[[usize; 2]]) -> UndiGraph<usize> {
    let nodes: SortedVec<usize> = GenericVocabularyBuilder::default()
        .expected_number_of_symbols(node_count)
        .symbols((0..node_count).enumerate())
        .build()
        .unwrap();
    let mut normalized_edges = edges
        .iter()
        .copied()
        .map(|[left, right]| if left <= right { (left, right) } else { (right, left) })
        .collect::<Vec<_>>();
    normalized_edges.sort_unstable();
    let matrix: SymmetricCSR2D<CSR2D<usize, usize, usize>> = UndiEdgesBuilder::default()
        .expected_number_of_edges(normalized_edges.len())
        .expected_shape(node_count)
        .edges(normalized_edges.into_iter())
        .build()
        .unwrap();
    UndiGraph::from((nodes, matrix))
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct OracleCycle {
    nodes: Vec<usize>,
    edge_ids: Vec<usize>,
    weight: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct XorBasis {
    rows_by_pivot: Vec<Option<Vec<u64>>>,
}

impl XorBasis {
    fn new(width: usize) -> Self {
        Self { rows_by_pivot: vec![None; width] }
    }

    fn insert(&mut self, mut row: Vec<u64>) -> bool {
        while let Some(pivot) = highest_set_bit(&row) {
            if let Some(existing) = &self.rows_by_pivot[pivot] {
                xor_assign(&mut row, existing);
            } else {
                self.rows_by_pivot[pivot] = Some(row);
                return true;
            }
        }
        false
    }
}

#[test]
fn test_minimum_cycle_basis_matches_exhaustive_order5_oracle() {
    let node_count = 5usize;
    let all_edges = complete_graph_edges(node_count);

    for mask in 0_u32..(1_u32 << all_edges.len()) {
        let edges = all_edges
            .iter()
            .enumerate()
            .filter_map(|(edge_id, edge)| ((mask >> edge_id) & 1 == 1).then_some(*edge))
            .collect::<Vec<_>>();
        let graph = build_undigraph(node_count, &edges);
        let actual = graph.minimum_cycle_basis().unwrap();
        let expected_weight = exact_minimum_cycle_basis_weight(node_count, &edges);

        assert_eq!(
            actual.cycle_rank(),
            cycle_rank(node_count, &edges),
            "cycle-rank mismatch for edge mask {mask:010b}"
        );
        assert_eq!(
            actual.total_weight(),
            expected_weight,
            "minimum-cycle-basis weight mismatch for edge mask {mask:010b}"
        );
        assert!(
            actual.minimum_cycle_basis().all(|cycle| is_simple_cycle_in_graph(cycle, &edges)),
            "returned non-cycle for edge mask {mask:010b}"
        );
    }
}

fn complete_graph_edges(node_count: usize) -> Vec<[usize; 2]> {
    let mut edges = Vec::new();
    for left in 0..node_count {
        for right in (left + 1)..node_count {
            edges.push([left, right]);
        }
    }
    edges
}

fn exact_minimum_cycle_basis_weight(node_count: usize, edges: &[[usize; 2]]) -> usize {
    let rank = cycle_rank(node_count, edges);
    if rank == 0 {
        return 0;
    }

    let candidates = enumerate_simple_cycles(node_count, edges);
    let edge_count = edges.len();
    let edge_lookup = edges
        .iter()
        .copied()
        .enumerate()
        .map(|(edge_id, [left, right])| ((left, right), edge_id))
        .collect::<std::collections::BTreeMap<_, _>>();

    let candidate_rows = candidates
        .iter()
        .map(|cycle| {
            let mut row = vec![0_u64; edge_count.div_ceil(64)];
            for pair in cycle.nodes.windows(2) {
                let left = pair[0].min(pair[1]);
                let right = pair[0].max(pair[1]);
                set_bit(&mut row, edge_lookup[&(left, right)]);
            }
            let last = cycle.nodes[cycle.nodes.len() - 1];
            let first = cycle.nodes[0];
            set_bit(&mut row, edge_lookup[&(last.min(first), last.max(first))]);
            row
        })
        .collect::<Vec<_>>();

    let mut best_weight = usize::MAX;
    let mut chosen = Vec::new();
    search_minimum_basis(&candidate_rows, &candidates, rank, 0, &mut chosen, 0, &mut best_weight);
    best_weight
}

fn search_minimum_basis(
    candidate_rows: &[Vec<u64>],
    candidates: &[OracleCycle],
    rank: usize,
    index: usize,
    chosen: &mut Vec<usize>,
    current_weight: usize,
    best_weight: &mut usize,
) {
    if chosen.len() == rank {
        let mut basis = XorBasis::new(candidate_rows[0].len() * 64);
        if chosen.iter().all(|&candidate_id| basis.insert(candidate_rows[candidate_id].clone()))
            && current_weight < *best_weight
        {
            *best_weight = current_weight;
        }
        return;
    }

    if index == candidates.len() || current_weight >= *best_weight {
        return;
    }

    if chosen.len() + (candidates.len() - index) < rank {
        return;
    }

    chosen.push(index);
    search_minimum_basis(
        candidate_rows,
        candidates,
        rank,
        index + 1,
        chosen,
        current_weight + candidates[index].weight,
        best_weight,
    );
    chosen.pop();

    search_minimum_basis(
        candidate_rows,
        candidates,
        rank,
        index + 1,
        chosen,
        current_weight,
        best_weight,
    );
}

fn enumerate_simple_cycles(node_count: usize, edges: &[[usize; 2]]) -> Vec<OracleCycle> {
    let mut adjacency = vec![Vec::new(); node_count];
    for &[left, right] in edges {
        adjacency[left].push(right);
        adjacency[right].push(left);
    }
    for neighbors in &mut adjacency {
        neighbors.sort_unstable();
    }

    let mut cycles = BTreeSet::new();
    for start in 0..node_count {
        let mut path = vec![start];
        let mut visited = vec![false; node_count];
        visited[start] = true;
        enumerate_from(start, start, &adjacency, &mut visited, &mut path, &mut cycles);
    }

    cycles
        .into_iter()
        .map(|nodes| {
            let weight = nodes.len();
            let edge_ids = cycle_edge_ids(&nodes, edges);
            OracleCycle { nodes, edge_ids, weight }
        })
        .collect()
}

fn enumerate_from(
    start: usize,
    current: usize,
    adjacency: &[Vec<usize>],
    visited: &mut [bool],
    path: &mut Vec<usize>,
    cycles: &mut BTreeSet<Vec<usize>>,
) {
    for &neighbor in &adjacency[current] {
        if neighbor == start {
            if path.len() >= 3 {
                cycles.insert(normalize_cycle(path.clone()));
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

fn cycle_edge_ids(cycle: &[usize], edges: &[[usize; 2]]) -> Vec<usize> {
    let edge_lookup = edges
        .iter()
        .copied()
        .enumerate()
        .map(|(edge_id, [left, right])| ((left, right), edge_id))
        .collect::<std::collections::BTreeMap<_, _>>();
    let mut edge_ids = Vec::new();
    for pair in cycle.windows(2) {
        edge_ids.push(edge_lookup[&(pair[0].min(pair[1]), pair[0].max(pair[1]))]);
    }
    let last = cycle[cycle.len() - 1];
    let first = cycle[0];
    edge_ids.push(edge_lookup[&(last.min(first), last.max(first))]);
    edge_ids.sort_unstable();
    edge_ids
}

fn cycle_rank(node_count: usize, edges: &[[usize; 2]]) -> usize {
    edges.len() + connected_components(node_count, edges) - node_count
}

fn connected_components(node_count: usize, edges: &[[usize; 2]]) -> usize {
    let mut adjacency = vec![Vec::new(); node_count];
    for &[left, right] in edges {
        adjacency[left].push(right);
        adjacency[right].push(left);
    }

    let mut visited = vec![false; node_count];
    let mut components = 0usize;
    for start in 0..node_count {
        if visited[start] {
            continue;
        }
        components += 1;
        visited[start] = true;
        let mut queue = VecDeque::from([start]);
        while let Some(node) = queue.pop_front() {
            for &neighbor in &adjacency[node] {
                if !visited[neighbor] {
                    visited[neighbor] = true;
                    queue.push_back(neighbor);
                }
            }
        }
    }
    components
}

fn is_simple_cycle_in_graph(cycle: &[usize], edges: &[[usize; 2]]) -> bool {
    if cycle.len() < 3 {
        return false;
    }
    let mut unique_nodes = cycle.to_vec();
    unique_nodes.sort_unstable();
    unique_nodes.dedup();
    if unique_nodes.len() != cycle.len() {
        return false;
    }

    let edge_set = edges
        .iter()
        .copied()
        .map(|[left, right]| (left.min(right), left.max(right)))
        .collect::<BTreeSet<_>>();
    for pair in cycle.windows(2) {
        if !edge_set.contains(&(pair[0].min(pair[1]), pair[0].max(pair[1]))) {
            return false;
        }
    }
    let last = cycle[cycle.len() - 1];
    let first = cycle[0];
    edge_set.contains(&(last.min(first), last.max(first)))
}

fn normalize_cycle(mut cycle: Vec<usize>) -> Vec<usize> {
    let smallest_position = cycle
        .iter()
        .enumerate()
        .min_by_key(|(_, node)| *node)
        .map(|(position, _)| position)
        .unwrap();
    cycle.rotate_left(smallest_position);
    if cycle[cycle.len() - 1] < cycle[1] {
        cycle[1..].reverse();
    }
    cycle
}

fn set_bit(bits: &mut [u64], bit: usize) {
    bits[bit / 64] |= 1_u64 << (bit % 64);
}

fn xor_assign(left: &mut [u64], right: &[u64]) {
    for (left_word, right_word) in left.iter_mut().zip(right.iter()) {
        *left_word ^= *right_word;
    }
}

fn highest_set_bit(bits: &[u64]) -> Option<usize> {
    for (word_index, &word) in bits.iter().enumerate().rev() {
        if word != 0 {
            let bit_offset = 63 - word.leading_zeros() as usize;
            return Some(word_index * 64 + bit_offset);
        }
    }
    None
}
