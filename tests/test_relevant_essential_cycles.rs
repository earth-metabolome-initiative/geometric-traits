//! Tests for the graph-level relevant-cycles and essential-cycles traits.
#![cfg(feature = "std")]

use std::collections::BTreeSet;

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
    normalized_edges.dedup();
    let matrix: SymmetricCSR2D<CSR2D<usize, usize, usize>> = UndiEdgesBuilder::default()
        .expected_number_of_edges(normalized_edges.len())
        .expected_shape(node_count)
        .edges(normalized_edges.into_iter())
        .build()
        .unwrap();
    UndiGraph::from((nodes, matrix))
}

#[test]
fn test_triangle_relevant_and_essential_match() {
    let graph = build_undigraph(3, &[[0, 1], [1, 2], [0, 2]]);

    let relevant = graph.relevant_cycles().unwrap();
    let essential = graph.essential_cycles().unwrap();

    assert_eq!(relevant.relevant_cycles().cloned().collect::<Vec<_>>(), vec![vec![0, 1, 2]]);
    assert_eq!(essential.essential_cycles().cloned().collect::<Vec<_>>(), vec![vec![0, 1, 2]]);
}

#[test]
fn test_square_with_diagonal_relevant_and_essential_exclude_square() {
    let graph = build_undigraph(4, &[[0, 1], [1, 2], [2, 3], [3, 0], [0, 2]]);

    let relevant = graph.relevant_cycles().unwrap();
    let essential = graph.essential_cycles().unwrap();

    let expected = vec![vec![0, 1, 2], vec![0, 2, 3]];
    assert_eq!(relevant.relevant_cycles().cloned().collect::<Vec<_>>(), expected);
    assert_eq!(essential.essential_cycles().cloned().collect::<Vec<_>>(), expected);
}

#[test]
fn test_k4_relevant_has_all_triangles_and_essential_is_empty() {
    let graph = build_undigraph(4, &[[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]);

    let relevant = graph.relevant_cycles().unwrap();
    let essential = graph.essential_cycles().unwrap();

    assert_eq!(
        relevant.relevant_cycles().cloned().collect::<Vec<_>>(),
        vec![vec![0, 1, 2], vec![0, 1, 3], vec![0, 2, 3], vec![1, 2, 3]]
    );
    assert!(essential.is_empty());
}

#[test]
fn test_relevant_and_essential_blanket_impl_on_reference() {
    let graph = build_undigraph(3, &[[0, 1], [1, 2], [0, 2]]);

    let relevant = <UndiGraph<usize> as RelevantCycles>::relevant_cycles(&graph).unwrap();
    let essential = <UndiGraph<usize> as EssentialCycles>::essential_cycles(&graph).unwrap();

    assert_eq!(relevant.relevant_cycles().cloned().collect::<Vec<_>>(), vec![vec![0, 1, 2]]);
    assert_eq!(essential.essential_cycles().cloned().collect::<Vec<_>>(), vec![vec![0, 1, 2]]);
}

#[test]
fn test_relevant_and_essential_match_exhaustive_order4_oracle() {
    let node_count = 4usize;
    let all_edges = complete_graph_edges(node_count);

    for mask in 0_u32..(1_u32 << all_edges.len()) {
        let edges = all_edges
            .iter()
            .enumerate()
            .filter_map(|(edge_id, edge)| ((mask >> edge_id) & 1 == 1).then_some(*edge))
            .collect::<Vec<_>>();
        let graph = build_undigraph(node_count, &edges);
        let actual_relevant = normalize_cycles(
            graph.relevant_cycles().unwrap().relevant_cycles().cloned().collect::<Vec<_>>(),
        );
        let actual_essential = normalize_cycles(
            graph.essential_cycles().unwrap().essential_cycles().cloned().collect::<Vec<_>>(),
        );
        let (expected_relevant, expected_essential) =
            exact_relevant_and_essential_cycles(node_count, &edges);

        assert_eq!(
            actual_relevant, expected_relevant,
            "relevant-cycle mismatch for edge mask {mask:06b}"
        );
        assert_eq!(
            actual_essential, expected_essential,
            "essential-cycle mismatch for edge mask {mask:06b}"
        );
    }
}

#[test]
fn test_relevant_and_essential_match_exhaustive_order5_oracle() {
    let node_count = 5usize;
    let all_edges = complete_graph_edges(node_count);

    for mask in 0_u32..(1_u32 << all_edges.len()) {
        let edges = all_edges
            .iter()
            .enumerate()
            .filter_map(|(edge_id, edge)| ((mask >> edge_id) & 1 == 1).then_some(*edge))
            .collect::<Vec<_>>();
        let graph = build_undigraph(node_count, &edges);
        let actual_relevant = normalize_cycles(
            graph.relevant_cycles().unwrap().relevant_cycles().cloned().collect::<Vec<_>>(),
        );
        let actual_essential = normalize_cycles(
            graph.essential_cycles().unwrap().essential_cycles().cloned().collect::<Vec<_>>(),
        );
        let (expected_relevant, expected_essential) =
            exact_relevant_and_essential_cycles(node_count, &edges);

        assert_eq!(
            actual_relevant, expected_relevant,
            "relevant-cycle mismatch for order-5 edge mask {mask:010b}"
        );
        assert_eq!(
            actual_essential, expected_essential,
            "essential-cycle mismatch for order-5 edge mask {mask:010b}"
        );
    }
}

#[test]
#[ignore = "exact exhaustive order-6 oracle; slow but useful for periodic validation"]
fn test_relevant_and_essential_match_exhaustive_order6_oracle() {
    use rayon::prelude::*;

    let node_count = 6usize;
    let all_edges = complete_graph_edges(node_count);

    (0_u32..(1_u32 << all_edges.len())).into_par_iter().for_each(|mask| {
        let edges = all_edges
            .iter()
            .enumerate()
            .filter_map(|(edge_id, edge)| ((mask >> edge_id) & 1 == 1).then_some(*edge))
            .collect::<Vec<_>>();
        let graph = build_undigraph(node_count, &edges);
        let actual_relevant = normalize_cycles(
            graph.relevant_cycles().unwrap().relevant_cycles().cloned().collect::<Vec<_>>(),
        );
        let actual_essential = normalize_cycles(
            graph.essential_cycles().unwrap().essential_cycles().cloned().collect::<Vec<_>>(),
        );
        let (expected_relevant, expected_essential) =
            exact_relevant_and_essential_cycles(node_count, &edges);

        assert_eq!(
            actual_relevant, expected_relevant,
            "relevant-cycle mismatch for order-6 edge mask {mask:015b}"
        );
        assert_eq!(
            actual_essential, expected_essential,
            "essential-cycle mismatch for order-6 edge mask {mask:015b}"
        );
    });
}

fn exact_relevant_and_essential_cycles(
    node_count: usize,
    edges: &[[usize; 2]],
) -> (Vec<Vec<usize>>, Vec<Vec<usize>>) {
    let bases = exact_minimum_cycle_bases(node_count, edges);
    if bases.is_empty() {
        return (Vec::new(), Vec::new());
    }

    let mut relevant = BTreeSet::new();
    let mut essential = bases[0].iter().cloned().collect::<BTreeSet<_>>();
    for basis in &bases {
        for cycle in basis {
            relevant.insert(cycle.clone());
        }
        let basis_set = basis.iter().cloned().collect::<BTreeSet<_>>();
        essential = essential.intersection(&basis_set).cloned().collect();
    }

    (
        normalize_cycles(relevant.into_iter().collect()),
        normalize_cycles(essential.into_iter().collect()),
    )
}

fn exact_minimum_cycle_bases(node_count: usize, edges: &[[usize; 2]]) -> Vec<Vec<Vec<usize>>> {
    let rank = cycle_rank(node_count, edges);
    if rank == 0 {
        return vec![Vec::new()];
    }

    let mut candidates = enumerate_simple_cycles(node_count, edges);
    candidates
        .sort_unstable_by(|left, right| left.len().cmp(&right.len()).then_with(|| left.cmp(right)));
    let candidate_rows = cycle_rows(&candidates, edges);
    let candidate_lengths = candidates.iter().map(Vec::len).collect::<Vec<_>>();
    let mut suffix_min_weight = vec![0usize; candidate_lengths.len() + 1];
    for index in (0..candidate_lengths.len()).rev() {
        suffix_min_weight[index] = suffix_min_weight[index + 1] + candidate_lengths[index];
    }

    let mut best_weight = usize::MAX;
    let mut best_bases = BTreeSet::new();
    let mut chosen = Vec::new();
    let mut search = OracleBasisSearch {
        candidates: &candidates,
        candidate_rows: &candidate_rows,
        candidate_lengths: &candidate_lengths,
        suffix_min_weight: &suffix_min_weight,
        rank,
        best_weight: &mut best_weight,
        best_bases: &mut best_bases,
    };
    search.search(0, &mut XorBasis::new(candidate_rows[0].len() * 64), &mut chosen, 0);

    best_bases.into_iter().collect()
}

struct OracleBasisSearch<'a> {
    candidates: &'a [Vec<usize>],
    candidate_rows: &'a [Vec<u64>],
    candidate_lengths: &'a [usize],
    suffix_min_weight: &'a [usize],
    rank: usize,
    best_weight: &'a mut usize,
    best_bases: &'a mut BTreeSet<Vec<Vec<usize>>>,
}

impl OracleBasisSearch<'_> {
    fn search(
        &mut self,
        index: usize,
        basis: &mut XorBasis,
        chosen: &mut Vec<usize>,
        current_weight: usize,
    ) {
        if basis.rank() == self.rank {
            if current_weight < *self.best_weight {
                *self.best_weight = current_weight;
                self.best_bases.clear();
            }
            if current_weight == *self.best_weight {
                let mut basis_cycles = chosen
                    .iter()
                    .map(|&candidate_id| self.candidates[candidate_id].clone())
                    .collect::<Vec<_>>();
                basis_cycles.sort_unstable_by(|left, right| {
                    left.len().cmp(&right.len()).then_with(|| left.cmp(right))
                });
                self.best_bases.insert(basis_cycles);
            }
            return;
        }

        if index == self.candidates.len() || current_weight > *self.best_weight {
            return;
        }
        let needed = self.rank - basis.rank();
        if self.candidates.len() - index < needed {
            return;
        }
        let optimistic_weight =
            self.suffix_min_weight[index] - self.suffix_min_weight[index + needed];
        if current_weight + optimistic_weight > *self.best_weight {
            return;
        }
        if self.max_reachable_rank(index, basis) < self.rank {
            return;
        }

        let mut with_candidate = basis.clone();
        if with_candidate.insert(self.candidate_rows[index].clone()) {
            chosen.push(index);
            self.search(
                index + 1,
                &mut with_candidate,
                chosen,
                current_weight + self.candidate_lengths[index],
            );
            chosen.pop();
        }

        self.search(index + 1, basis, chosen, current_weight);
    }

    fn max_reachable_rank(&self, index: usize, basis: &XorBasis) -> usize {
        let mut reachable = basis.clone();
        for row in &self.candidate_rows[index..] {
            reachable.insert(row.clone());
        }
        reachable.rank()
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

fn enumerate_simple_cycles(node_count: usize, edges: &[[usize; 2]]) -> Vec<Vec<usize>> {
    let mut adjacency = vec![Vec::new(); node_count];
    for &[left, right] in edges {
        adjacency[left].push(right);
        adjacency[right].push(left);
    }
    for neighbors in &mut adjacency {
        neighbors.sort_unstable();
    }

    let mut cycles = BTreeSet::new();
    let mut visited = vec![false; node_count];
    let mut path = Vec::new();
    for start in 0..node_count {
        visited[start] = true;
        path.clear();
        path.push(start);
        enumerate_from(start, start, &adjacency, &mut visited, &mut path, &mut cycles);
        visited[start] = false;
    }

    cycles.into_iter().collect()
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

fn cycle_rows(cycles: &[Vec<usize>], edges: &[[usize; 2]]) -> Vec<Vec<u64>> {
    let edge_lookup = edges
        .iter()
        .copied()
        .enumerate()
        .map(|(edge_id, [left, right])| ((left, right), edge_id))
        .collect::<std::collections::BTreeMap<_, _>>();

    cycles
        .iter()
        .map(|cycle| {
            let mut row = vec![0_u64; edges.len().div_ceil(64)];
            for pair in cycle.windows(2) {
                set_bit(&mut row, edge_lookup[&(pair[0].min(pair[1]), pair[0].max(pair[1]))]);
            }
            let last = cycle[cycle.len() - 1];
            let first = cycle[0];
            set_bit(&mut row, edge_lookup[&(last.min(first), last.max(first))]);
            row
        })
        .collect()
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
        let mut stack = vec![start];
        visited[start] = true;
        while let Some(node) = stack.pop() {
            for &neighbor in &adjacency[node] {
                if !visited[neighbor] {
                    visited[neighbor] = true;
                    stack.push(neighbor);
                }
            }
        }
    }
    components
}

fn normalize_cycles(mut cycles: Vec<Vec<usize>>) -> Vec<Vec<usize>> {
    for cycle in &mut cycles {
        *cycle = normalize_cycle(cycle.clone());
    }
    cycles
        .sort_unstable_by(|left, right| left.len().cmp(&right.len()).then_with(|| left.cmp(right)));
    cycles
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

#[derive(Debug, Clone, PartialEq, Eq)]
struct XorBasis {
    rows_by_pivot: Vec<Option<Vec<u64>>>,
}

impl XorBasis {
    fn new(width: usize) -> Self {
        Self { rows_by_pivot: vec![None; width] }
    }

    fn rank(&self) -> usize {
        self.rows_by_pivot.iter().flatten().count()
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
