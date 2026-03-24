//! Internal reduction engine for exact Karp-Sipser preprocessing.
use alloc::{
    collections::{BTreeMap, BTreeSet},
    vec::Vec,
};
use core::mem;

use num_traits::AsPrimitive;

use super::{KarpSipserKernel, KarpSipserRules};
use crate::{
    impls::{CSR2D, SymmetricCSR2D},
    naive_structs::UndiEdgesBuilder,
    traits::{EdgesBuilder, SparseMatrix2D, SparseSquareMatrix, SquareMatrix},
};

#[derive(Debug)]
pub(super) struct ContractNeighbor {
    vertex: usize,
    side_mask: u8,
}

const LEFT_NEIGHBOR: u8 = 0b01;
const RIGHT_NEIGHBOR: u8 = 0b10;
const BOTH_NEIGHBORS: u8 = LEFT_NEIGHBOR | RIGHT_NEIGHBOR;

#[derive(Debug)]
pub(super) enum RecoveryStep {
    ForcedMatch {
        left: usize,
        right: usize,
    },
    Contract {
        pivot: usize,
        left: usize,
        right: usize,
        merged: usize,
        neighbor_sides: Vec<ContractNeighbor>,
    },
}

#[derive(Debug)]
struct VertexState {
    active: bool,
    neighbors: BTreeSet<usize>,
    queued_degree_one: bool,
    queued_degree_two: bool,
}

struct ReductionEngine {
    vertices: Vec<VertexState>,
    degree_one: Vec<usize>,
    degree_two: Vec<usize>,
    recovery: Vec<RecoveryStep>,
}

impl ReductionEngine {
    fn new<M: SparseSquareMatrix + ?Sized>(matrix: &M) -> Self {
        let mut vertices = Vec::with_capacity(matrix.order().as_());
        for row in matrix.row_indices() {
            let row_usize = row.as_();
            let mut neighbors = BTreeSet::new();
            for neighbor in matrix.sparse_row(row).map(AsPrimitive::as_) {
                if neighbor != row_usize {
                    neighbors.insert(neighbor);
                }
            }
            vertices.push(VertexState {
                active: true,
                neighbors,
                queued_degree_one: false,
                queued_degree_two: false,
            });
        }

        let mut engine =
            Self { vertices, degree_one: Vec::new(), degree_two: Vec::new(), recovery: Vec::new() };
        for vertex in 0..engine.vertices.len() {
            engine.enqueue_if_reducible(vertex);
        }
        engine
    }

    fn run(&mut self, rules: KarpSipserRules) {
        loop {
            while let Some(vertex) = self.pop_degree_one() {
                self.apply_rule1(vertex);
            }

            if rules == KarpSipserRules::Degree1 {
                break;
            }

            let Some(vertex) = self.pop_degree_two() else {
                break;
            };
            self.apply_rule2(vertex);
        }
    }

    fn into_kernel<I: Copy>(self, original_indices: Vec<I>) -> KarpSipserKernel<I> {
        let total_vertices = self.vertices.len();
        let kernel_to_internal: Vec<usize> = self
            .vertices
            .iter()
            .enumerate()
            .filter_map(|(vertex, state)| {
                (state.active && !state.neighbors.is_empty()).then_some(vertex)
            })
            .collect();

        let kernel = if kernel_to_internal.is_empty() {
            SymmetricCSR2D::<CSR2D<usize, usize, usize>>::default()
        } else {
            let mut internal_to_kernel = vec![usize::MAX; total_vertices];
            for (kernel_vertex, &internal_vertex) in kernel_to_internal.iter().enumerate() {
                internal_to_kernel[internal_vertex] = kernel_vertex;
            }

            let mut edges = Vec::new();
            for &internal_vertex in &kernel_to_internal {
                let kernel_vertex = internal_to_kernel[internal_vertex];
                for &neighbor in &self.vertices[internal_vertex].neighbors {
                    if self.vertices[neighbor].active && internal_vertex < neighbor {
                        edges.push((kernel_vertex, internal_to_kernel[neighbor]));
                    }
                }
            }
            edges.sort_unstable();
            edges.dedup();

            UndiEdgesBuilder::default()
                .expected_number_of_edges(edges.len())
                .expected_shape(kernel_to_internal.len())
                .edges(edges.into_iter())
                .build()
                .expect("kernel edges are constructed in sorted simple-graph order")
        };

        KarpSipserKernel {
            kernel,
            kernel_to_internal,
            original_indices,
            recovery: self.recovery,
            total_vertices,
        }
    }

    fn pop_degree_one(&mut self) -> Option<usize> {
        while let Some(vertex) = self.degree_one.pop() {
            self.vertices[vertex].queued_degree_one = false;
            if self.is_active_degree(vertex, 1) {
                return Some(vertex);
            }
        }
        None
    }

    fn pop_degree_two(&mut self) -> Option<usize> {
        while let Some(vertex) = self.degree_two.pop() {
            self.vertices[vertex].queued_degree_two = false;
            if self.is_active_degree(vertex, 2) {
                return Some(vertex);
            }
        }
        None
    }

    fn is_active_degree(&self, vertex: usize, degree: usize) -> bool {
        self.vertices[vertex].active && self.vertices[vertex].neighbors.len() == degree
    }

    fn enqueue_if_reducible(&mut self, vertex: usize) {
        if !self.vertices[vertex].active {
            return;
        }

        match self.vertices[vertex].neighbors.len() {
            1 if !self.vertices[vertex].queued_degree_one => {
                self.vertices[vertex].queued_degree_one = true;
                self.degree_one.push(vertex);
            }
            2 if !self.vertices[vertex].queued_degree_two => {
                self.vertices[vertex].queued_degree_two = true;
                self.degree_two.push(vertex);
            }
            _ => {}
        }
    }

    fn deactivate_vertex(&mut self, vertex: usize) {
        self.vertices[vertex].active = false;
        self.vertices[vertex].neighbors.clear();
        self.vertices[vertex].queued_degree_one = false;
        self.vertices[vertex].queued_degree_two = false;
    }

    fn contract_neighbor_sides(
        &self,
        left_neighbors: &BTreeSet<usize>,
        right_neighbors: &BTreeSet<usize>,
        pivot: usize,
        left: usize,
        right: usize,
    ) -> Vec<ContractNeighbor> {
        let mut merged_neighbors = BTreeMap::new();

        for &neighbor in left_neighbors {
            if neighbor == pivot || neighbor == right || !self.vertices[neighbor].active {
                continue;
            }
            merged_neighbors.insert(neighbor, LEFT_NEIGHBOR);
        }

        for &neighbor in right_neighbors {
            if neighbor == pivot || neighbor == left || !self.vertices[neighbor].active {
                continue;
            }
            if let Some(side_mask) = merged_neighbors.get_mut(&neighbor) {
                *side_mask |= RIGHT_NEIGHBOR;
            } else {
                merged_neighbors.insert(neighbor, RIGHT_NEIGHBOR);
            }
        }

        merged_neighbors
            .into_iter()
            .map(|(vertex, side_mask)| ContractNeighbor { vertex, side_mask })
            .collect()
    }

    fn apply_rule1(&mut self, vertex: usize) {
        if !self.is_active_degree(vertex, 1) {
            return;
        }

        let neighbor = *self.vertices[vertex]
            .neighbors
            .first()
            .expect("degree-1 vertex must have one neighbor");
        let affected = mem::take(&mut self.vertices[neighbor].neighbors);

        self.deactivate_vertex(vertex);
        self.deactivate_vertex(neighbor);

        for affected_vertex in affected {
            if affected_vertex == vertex || !self.vertices[affected_vertex].active {
                continue;
            }

            let touched = {
                let neighbors = &mut self.vertices[affected_vertex].neighbors;
                let removed_neighbor = neighbors.remove(&neighbor);
                let removed_vertex = neighbors.remove(&vertex);
                removed_neighbor || removed_vertex
            };

            if touched {
                self.enqueue_if_reducible(affected_vertex);
            }
        }

        let (left, right) = if vertex < neighbor { (vertex, neighbor) } else { (neighbor, vertex) };
        self.recovery.push(RecoveryStep::ForcedMatch { left, right });
    }

    fn apply_rule2(&mut self, pivot: usize) {
        if !self.is_active_degree(pivot, 2) {
            return;
        }

        let mut pivot_neighbors = self.vertices[pivot].neighbors.iter().copied();
        let left = pivot_neighbors.next().expect("degree-2 pivot must have two neighbors");
        let right = pivot_neighbors.next().expect("degree-2 pivot must have two neighbors");
        let left_neighbors = mem::take(&mut self.vertices[left].neighbors);
        let right_neighbors = mem::take(&mut self.vertices[right].neighbors);
        // Avoid the naive whole-row rebuild warned about in the design note:
        // degree-2 contractions update only touched incidence sets and keep a
        // single provenance list for recovery.
        let neighbor_sides =
            self.contract_neighbor_sides(&left_neighbors, &right_neighbors, pivot, left, right);
        let merged_neighbors = neighbor_sides.iter().map(|neighbor| neighbor.vertex).collect();

        let merged = self.vertices.len();
        self.vertices.push(VertexState {
            active: true,
            neighbors: merged_neighbors,
            queued_degree_one: false,
            queued_degree_two: false,
        });

        self.deactivate_vertex(pivot);
        self.deactivate_vertex(left);
        self.deactivate_vertex(right);

        self.reconnect_contract_neighbors(left, right, pivot, merged, &neighbor_sides);
        self.enqueue_if_reducible(merged);

        self.recovery.push(RecoveryStep::Contract { pivot, left, right, merged, neighbor_sides });
    }

    fn reconnect_contract_neighbors(
        &mut self,
        left: usize,
        right: usize,
        pivot: usize,
        merged: usize,
        neighbor_sides: &[ContractNeighbor],
    ) {
        for contract_neighbor in neighbor_sides {
            let affected_vertex = contract_neighbor.vertex;
            if !self.vertices[affected_vertex].active {
                continue;
            }

            let touched = {
                let neighbors = &mut self.vertices[affected_vertex].neighbors;
                let removed_left = neighbors.remove(&left);
                let removed_right = neighbors.remove(&right);
                let removed_pivot = neighbors.remove(&pivot);
                if removed_left || removed_right {
                    neighbors.insert(merged);
                }
                removed_left || removed_right || removed_pivot
            };

            if touched {
                self.enqueue_if_reducible(affected_vertex);
            }
        }
    }
}

pub(super) fn build_kernel<M: SparseSquareMatrix + ?Sized>(
    matrix: &M,
    rules: KarpSipserRules,
) -> KarpSipserKernel<M::Index> {
    let original_indices: Vec<M::Index> = matrix.row_indices().collect();
    let mut engine = ReductionEngine::new(matrix);
    engine.run(rules);
    engine.into_kernel(original_indices)
}

pub(super) fn recover_pairs<I: Copy>(
    kernel: &KarpSipserKernel<I>,
    kernel_matching: Vec<(usize, usize)>,
) -> Vec<(I, I)> {
    validate_kernel_matching(&kernel.kernel, &kernel_matching);

    let mut mate = vec![None; kernel.total_vertices];
    for (left, right) in kernel_matching {
        let left = kernel.kernel_to_internal[left];
        let right = kernel.kernel_to_internal[right];
        mate[left] = Some(right);
        mate[right] = Some(left);
    }

    for step in kernel.recovery.iter().rev() {
        match step {
            RecoveryStep::ForcedMatch { left, right } => {
                assert!(
                    mate[*left].is_none() && mate[*right].is_none(),
                    "forced-match recovery reintroduced an already matched vertex",
                );
                mate[*left] = Some(*right);
                mate[*right] = Some(*left);
            }
            RecoveryStep::Contract { pivot, left, right, merged, neighbor_sides } => {
                match mate[*merged] {
                    None => {
                        assert!(
                            mate[*pivot].is_none() && mate[*left].is_none(),
                            "contract recovery reintroduced an already matched vertex",
                        );
                        mate[*pivot] = Some(*left);
                        mate[*left] = Some(*pivot);
                    }
                    Some(partner) => {
                        assert_eq!(
                            mate[partner],
                            Some(*merged),
                            "kernel mate relation must remain symmetric during recovery",
                        );

                        mate[*merged] = None;
                        mate[partner] = None;

                        let side_mask = contract_neighbor_side(neighbor_sides, partner)
                            .unwrap_or_else(|| {
                                panic!("merged vertex matched to a vertex absent from both sides");
                            });
                        let (matched_side, pivot_side) = match side_mask {
                            RIGHT_NEIGHBOR => (*right, *left),
                            LEFT_NEIGHBOR | BOTH_NEIGHBORS => (*left, *right),
                            _ => unreachable!("invalid contraction provenance"),
                        };

                        assert!(
                            mate[matched_side].is_none()
                                && mate[pivot_side].is_none()
                                && mate[*pivot].is_none()
                                && mate[partner].is_none(),
                            "contract recovery reintroduced an already matched vertex",
                        );

                        mate[matched_side] = Some(partner);
                        mate[partner] = Some(matched_side);
                        mate[*pivot] = Some(pivot_side);
                        mate[pivot_side] = Some(*pivot);
                    }
                }
            }
        }
    }

    let original_count = kernel.original_indices.len();
    let mut pairs = Vec::new();
    for (vertex, maybe_neighbor) in mate.iter().copied().enumerate().take(original_count) {
        if let Some(neighbor) = maybe_neighbor {
            assert!(
                neighbor < original_count,
                "original vertex remained matched to a synthetic vertex"
            );
            if vertex < neighbor {
                pairs.push((kernel.original_indices[vertex], kernel.original_indices[neighbor]));
            }
        }
    }
    pairs
}

fn contract_neighbor_side(neighbor_sides: &[ContractNeighbor], partner: usize) -> Option<u8> {
    neighbor_sides
        .binary_search_by_key(&partner, |neighbor| neighbor.vertex)
        .ok()
        .map(|index| neighbor_sides[index].side_mask)
}

fn validate_kernel_matching(
    kernel: &SymmetricCSR2D<CSR2D<usize, usize, usize>>,
    matching: &[(usize, usize)],
) {
    let mut used = vec![false; kernel.order()];
    for &(left, right) in matching {
        assert!(left < right, "kernel matching pairs must satisfy u < v");
        assert!(right < kernel.order(), "kernel matching vertex out of bounds");
        assert!(kernel.has_entry(left, right), "kernel matching contains a non-edge");
        assert!(!used[left], "kernel matching reuses a vertex");
        assert!(!used[right], "kernel matching reuses a vertex");
        used[left] = true;
        used[right] = true;
    }
}

#[cfg(test)]
mod tests {
    use alloc::collections::BTreeSet;

    use super::*;

    fn build_graph(
        n: usize,
        edges: &[(usize, usize)],
    ) -> SymmetricCSR2D<CSR2D<usize, usize, usize>> {
        let mut sorted_edges: Vec<(usize, usize)> = edges.to_vec();
        sorted_edges.sort_unstable();
        UndiEdgesBuilder::default()
            .expected_number_of_edges(sorted_edges.len())
            .expected_shape(n)
            .edges(sorted_edges.into_iter())
            .build()
            .expect("test graphs are valid simple undirected graphs")
    }

    #[test]
    fn test_enqueue_if_reducible_ignores_inactive_vertex() {
        let g = build_graph(3, &[(0, 1), (1, 2), (0, 2)]);
        let mut engine = ReductionEngine::new(&g);
        let original_degree_two_len = engine.degree_two.len();

        engine.deactivate_vertex(0);
        engine.enqueue_if_reducible(0);

        assert_eq!(engine.degree_two.len(), original_degree_two_len);
        assert!(!engine.vertices[0].active);
        assert!(!engine.vertices[0].queued_degree_one);
        assert!(!engine.vertices[0].queued_degree_two);
    }

    #[test]
    fn test_apply_rule1_returns_for_non_degree_one_vertex() {
        let g = build_graph(3, &[(0, 1), (1, 2), (0, 2)]);
        let mut engine = ReductionEngine::new(&g);

        engine.apply_rule1(0);

        assert!(engine.recovery.is_empty());
        assert_eq!(engine.vertices.len(), 3);
        assert!(engine.vertices[0].active);
        assert_eq!(engine.vertices[0].neighbors.len(), 2);
    }

    #[test]
    fn test_apply_rule2_returns_for_non_degree_two_vertex() {
        let g = build_graph(2, &[(0, 1)]);
        let mut engine = ReductionEngine::new(&g);
        let original_degree_one_len = engine.degree_one.len();

        engine.apply_rule2(0);

        assert!(engine.recovery.is_empty());
        assert_eq!(engine.vertices.len(), 2);
        assert_eq!(engine.degree_one.len(), original_degree_one_len);
        assert!(engine.vertices[0].active);
        assert_eq!(engine.vertices[0].neighbors.len(), 1);
    }

    #[test]
    fn test_reconnect_contract_neighbors_skips_inactive_vertices() {
        let inactive_neighbors: BTreeSet<usize> = [0usize, 1, 2].into_iter().collect();
        let mut engine = ReductionEngine {
            vertices: vec![
                VertexState {
                    active: true,
                    neighbors: BTreeSet::new(),
                    queued_degree_one: false,
                    queued_degree_two: false,
                },
                VertexState {
                    active: true,
                    neighbors: BTreeSet::new(),
                    queued_degree_one: false,
                    queued_degree_two: false,
                },
                VertexState {
                    active: true,
                    neighbors: BTreeSet::new(),
                    queued_degree_one: false,
                    queued_degree_two: false,
                },
                VertexState {
                    active: false,
                    neighbors: inactive_neighbors.clone(),
                    queued_degree_one: false,
                    queued_degree_two: false,
                },
                VertexState {
                    active: true,
                    neighbors: BTreeSet::new(),
                    queued_degree_one: false,
                    queued_degree_two: false,
                },
            ],
            degree_one: Vec::new(),
            degree_two: Vec::new(),
            recovery: Vec::new(),
        };

        engine.reconnect_contract_neighbors(
            0,
            1,
            2,
            4,
            &[ContractNeighbor { vertex: 3, side_mask: LEFT_NEIGHBOR }],
        );

        assert_eq!(engine.vertices[3].neighbors, inactive_neighbors);
        assert!(engine.degree_one.is_empty());
        assert!(engine.degree_two.is_empty());
    }
}
