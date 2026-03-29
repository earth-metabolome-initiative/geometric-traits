//! Shared support for ground-truth MCES tests.

pub(super) use std::{
    collections::{BTreeMap, VecDeque},
    fs::File,
    io::{Read as _, Write as _},
    time::Instant,
};

pub(super) use geometric_traits::{
    impls::{
        BitSquareMatrix, CSR2D, EdgeContexts, SortedVec, SquareCSR2D, SymmetricCSR2D, ValuedCSR2D,
    },
    prelude::*,
    traits::{
        EdgesBuilder, MatrixMut, SparseMatrix2D, SparseMatrixMut, SquareMatrix, TypedNode,
        VocabularyBuilder,
    },
};
pub(super) use rayon::prelude::*;

#[path = "support/fixtures.rs"]
mod fixtures;
pub(super) use fixtures::*;

// ============================================================================
// Typed node infrastructure for labeled MCES
// ============================================================================

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(super) struct GroundTruthNodeLabel {
    pub(super) atom_type: u8,
    pub(super) explicit_degree: Option<u8>,
    pub(super) is_aromatic: Option<bool>,
}

/// A node labeled by a generic harness-local node label.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(super) struct AtomNode {
    pub(super) id: usize,
    pub(super) node_label: GroundTruthNodeLabel,
}

impl TypedNode for AtomNode {
    type NodeType = GroundTruthNodeLabel;
    fn node_type(&self) -> Self::NodeType {
        self.node_label
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(super) struct GroundTruthEdgeValue {
    pub(super) bond_order: Option<u32>,
    pub(super) in_ring: Option<bool>,
}

pub(super) type TypedGraph = geometric_traits::naive_structs::GenericGraph<
    SortedVec<AtomNode>,
    SymmetricCSR2D<ValuedCSR2D<usize, usize, usize, GroundTruthEdgeValue>>,
>;
pub(super) type GroundTruthBondLabel =
    (GroundTruthNodeLabel, GroundTruthEdgeValue, GroundTruthNodeLabel);

/// Maps atom type strings across both graphs to a shared sequential u8 space.
pub(super) fn atom_type_to_shared_indices(
    first_atom_types: &[String],
    second_atom_types: &[String],
) -> (Vec<u8>, Vec<u8>) {
    let mut unique: Vec<&str> =
        first_atom_types.iter().chain(second_atom_types.iter()).map(|s| s.as_str()).collect();
    unique.sort_unstable();
    unique.dedup();

    assert!(
        unique.len() <= (u8::MAX as usize) + 1,
        "shared atom label universe exceeds u8 capacity"
    );

    let remap = |atom_types: &[String]| {
        atom_types
            .iter()
            .map(|t| unique.iter().position(|&u| u == t.as_str()).unwrap() as u8)
            .collect::<Vec<_>>()
    };

    (remap(first_atom_types), remap(second_atom_types))
}

pub(super) fn build_typed_graph(
    n_atoms: usize,
    edges: &[[usize; 2]],
    atom_type_indices: &[u8],
    atom_is_aromatic: &[bool],
    bond_types: &[u32],
    ignore_bond_orders: bool,
    ring_matches_ring_only: bool,
    exact_connections_match: bool,
    respect_atom_aromaticity: bool,
) -> TypedGraph {
    assert_eq!(n_atoms, atom_type_indices.len());
    assert_eq!(n_atoms, atom_is_aromatic.len());
    assert_eq!(edges.len(), bond_types.len());
    let mut normalized_edges: Vec<(usize, usize, u32)> = edges
        .iter()
        .zip(bond_types.iter().copied())
        .map(
            |(e, bond_type)| {
                if e[0] < e[1] { (e[0], e[1], bond_type) } else { (e[1], e[0], bond_type) }
            },
        )
        .collect();
    normalized_edges.sort_unstable_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
    normalized_edges.dedup();

    let explicit_degrees = explicit_degrees(n_atoms, &normalized_edges);
    let ring_membership = edge_ring_membership(n_atoms, &normalized_edges);

    let nodes_vec: Vec<AtomNode> = (0..n_atoms)
        .map(|i| {
            AtomNode {
                id: i,
                node_label: GroundTruthNodeLabel {
                    atom_type: atom_type_indices[i],
                    explicit_degree: if exact_connections_match {
                        Some(explicit_degrees[i])
                    } else {
                        None
                    },
                    is_aromatic: if respect_atom_aromaticity {
                        Some(atom_is_aromatic[i])
                    } else {
                        None
                    },
                },
            }
        })
        .collect();
    let nodes: SortedVec<AtomNode> = GenericVocabularyBuilder::default()
        .expected_number_of_symbols(n_atoms)
        .symbols(nodes_vec.into_iter().enumerate())
        .build()
        .unwrap();

    let mut sorted: Vec<(usize, usize, GroundTruthEdgeValue)> = normalized_edges
        .into_iter()
        .zip(ring_membership)
        .map(|((src, dst, bond_order), in_ring)| {
            (
                src,
                dst,
                GroundTruthEdgeValue {
                    bond_order: if ignore_bond_orders { None } else { Some(bond_order) },
                    in_ring: if ring_matches_ring_only { Some(in_ring) } else { None },
                },
            )
        })
        .collect();

    let mut all_entries = Vec::with_capacity(sorted.len() * 2);
    for (src, dst, edge_value) in sorted.drain(..) {
        all_entries.push((src, dst, edge_value));
        all_entries.push((dst, src, edge_value));
    }
    all_entries.sort_unstable_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));

    let mut valued: ValuedCSR2D<usize, usize, usize, GroundTruthEdgeValue> =
        SparseMatrixMut::with_sparse_shaped_capacity((n_atoms, n_atoms), all_entries.len());
    for (src, dst, edge_value) in all_entries {
        MatrixMut::add(&mut valued, (src, dst, edge_value)).unwrap();
    }

    geometric_traits::naive_structs::GenericGraph::from((
        nodes,
        SymmetricCSR2D::from_parts(SquareCSR2D::from_parts(valued, 0)),
    ))
}

pub(super) fn normalized_graph_bonds(graph: &GraphData) -> Vec<(usize, usize, u32)> {
    let mut normalized_edges: Vec<(usize, usize, u32)> = graph
        .edges
        .iter()
        .zip(graph.bond_types.iter().copied())
        .map(|(edge, bond_type)| {
            if edge[0] < edge[1] {
                (edge[0], edge[1], bond_type)
            } else {
                (edge[1], edge[0], bond_type)
            }
        })
        .collect();
    normalized_edges.sort_unstable_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
    normalized_edges.dedup();
    normalized_edges
}

pub(super) fn explicit_degrees(n_atoms: usize, edges: &[(usize, usize, u32)]) -> Vec<u8> {
    let mut degrees = vec![0usize; n_atoms];
    for &(src, dst, _) in edges {
        degrees[src] += 1;
        degrees[dst] += 1;
    }
    degrees
        .into_iter()
        .map(|degree| u8::try_from(degree).expect("explicit degree must fit in u8 for fixtures"))
        .collect()
}

pub(super) fn edge_ring_membership(n_atoms: usize, edges: &[(usize, usize, u32)]) -> Vec<bool> {
    let mut adjacency = vec![Vec::<(usize, usize)>::new(); n_atoms];
    for (edge_index, &(src, dst, _)) in edges.iter().enumerate() {
        adjacency[src].push((dst, edge_index));
        adjacency[dst].push((src, edge_index));
    }

    edges
        .iter()
        .enumerate()
        .map(|(skipped_edge, &(src, dst, _))| {
            let mut stack = vec![src];
            let mut visited = vec![false; n_atoms];
            visited[src] = true;

            while let Some(node) = stack.pop() {
                for &(neighbor, edge_index) in &adjacency[node] {
                    if edge_index == skipped_edge || visited[neighbor] {
                        continue;
                    }
                    visited[neighbor] = true;
                    if neighbor == dst {
                        return true;
                    }
                    stack.push(neighbor);
                }
            }

            false
        })
        .collect()
}

// Also keep unlabeled graph builder for comparison tests.
pub(super) fn build_unlabeled_graph(
    n_atoms: usize,
    edges: &[[usize; 2]],
) -> geometric_traits::naive_structs::UndiGraph<usize> {
    let edge_tuples: Vec<(usize, usize)> =
        edges.iter().map(|e| if e[0] < e[1] { (e[0], e[1]) } else { (e[1], e[0]) }).collect();
    let mut sorted = edge_tuples;
    sorted.sort_unstable();
    sorted.dedup();

    type UndiBuilder<I> = geometric_traits::naive_structs::GenericUndirectedMonopartiteEdgesBuilder<
        I,
        geometric_traits::impls::UpperTriangularCSR2D<CSR2D<usize, usize, usize>>,
        SymmetricCSR2D<CSR2D<usize, usize, usize>>,
    >;

    let undi: SymmetricCSR2D<CSR2D<usize, usize, usize>> = UndiBuilder::default()
        .expected_number_of_edges(sorted.len())
        .expected_shape(n_atoms)
        .edges(sorted.into_iter())
        .build()
        .unwrap();

    let nodes: SortedVec<usize> = GenericVocabularyBuilder::default()
        .expected_number_of_symbols(n_atoms)
        .symbols((0..n_atoms).enumerate())
        .build()
        .unwrap();
    geometric_traits::naive_structs::UndiGraph::from((nodes, undi))
}

pub(super) fn compute_case_bond_labels(
    graph: &TypedGraph,
    edge_map: &[(usize, usize)],
) -> Vec<GroundTruthBondLabel> {
    let node_types: Vec<GroundTruthNodeLabel> =
        graph.nodes().map(|symbol| symbol.node_type()).collect();
    edge_map
        .iter()
        .map(|&(src, dst)| {
            let left = node_types[src];
            let right = node_types[dst];
            let edge_value = geometric_traits::traits::Edges::matrix(graph.edges())
                .sparse_value_at(src, dst)
                .unwrap();
            if left <= right { (left, edge_value, right) } else { (right, edge_value, left) }
        })
        .collect()
}

pub(super) fn intern_case_bond_labels(
    first_labels: &[GroundTruthBondLabel],
    second_labels: &[GroundTruthBondLabel],
) -> (Vec<usize>, Vec<usize>, usize) {
    let mut interned = BTreeMap::<GroundTruthBondLabel, usize>::new();
    let mut next = 0usize;

    let mut intern_slice = |labels: &[GroundTruthBondLabel]| {
        labels
            .iter()
            .map(|label| {
                *interned.entry(*label).or_insert_with(|| {
                    let current = next;
                    next += 1;
                    current
                })
            })
            .collect::<Vec<_>>()
    };

    let first = intern_slice(first_labels);
    let second = intern_slice(second_labels);
    (first, second, next.max(1))
}

pub(super) struct LabeledCaseProductDiagnostics {
    pub(super) matrix: BitSquareMatrix,
    pub(super) vertex_pairs: Vec<(usize, usize)>,
    pub(super) first_edge_map: Vec<(usize, usize)>,
    pub(super) second_edge_map: Vec<(usize, usize)>,
    pub(super) first_bond_labels: Vec<GroundTruthBondLabel>,
    pub(super) second_bond_labels: Vec<GroundTruthBondLabel>,
}

pub(super) fn collect_prepared_labeled_case_product_diagnostics(
    case: &GroundTruthCase,
    prepared: &PreparedLabeledCase,
    use_edge_contexts: bool,
) -> LabeledCaseProductDiagnostics {
    let lg1 = prepared.first.labeled_line_graph();
    let lg2 = prepared.second.labeled_line_graph();
    let first_bond_labels = compute_case_bond_labels(&prepared.first, lg1.edge_map());
    let second_bond_labels = compute_case_bond_labels(&prepared.second, lg2.edge_map());
    let use_edge_contexts = use_edge_contexts && case_uses_complete_aromatic_rings(case);

    let product = lg1.graph().labeled_modular_product_filtered(
        lg2.graph(),
        |i, j| {
            let contexts_match = match (&prepared.first_contexts, &prepared.second_contexts) {
                (Some(first_contexts), Some(second_contexts)) if use_edge_contexts => {
                    first_contexts.compatible_with(i, second_contexts, j)
                }
                _ => true,
            };
            first_bond_labels[i] == second_bond_labels[j] && contexts_match
        },
        |left, right| left == right,
    );

    LabeledCaseProductDiagnostics {
        matrix: product.matrix().clone(),
        vertex_pairs: product.vertex_pairs().to_vec(),
        first_edge_map: lg1.edge_map().to_vec(),
        second_edge_map: lg2.edge_map().to_vec(),
        first_bond_labels,
        second_bond_labels,
    }
}

pub(super) fn collect_labeled_case_product_diagnostics(
    case: &GroundTruthCase,
    use_edge_contexts: bool,
) -> LabeledCaseProductDiagnostics {
    let prepared = prepare_labeled_case(case);
    collect_prepared_labeled_case_product_diagnostics(case, &prepared, use_edge_contexts)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct FixtureBondRecord {
    pub(super) edge: (usize, usize),
    pub(super) bond_type: u32,
    pub(super) canonical_index: usize,
    pub(super) original_index: usize,
}

pub(super) fn fixture_bond_records(graph: &GraphData) -> Vec<FixtureBondRecord> {
    let mut records: Vec<FixtureBondRecord> = graph
        .edges
        .iter()
        .zip(graph.bond_types.iter().copied())
        .enumerate()
        .map(|(canonical_index, (&edge, bond_type))| {
            FixtureBondRecord {
                edge: canonical_edge(edge),
                bond_type,
                canonical_index,
                original_index: graph
                    .bond_original_indices
                    .get(canonical_index)
                    .copied()
                    .unwrap_or(canonical_index),
            }
        })
        .collect();
    records.sort_unstable_by_key(|record| {
        (record.original_index, record.edge.0, record.edge.1, record.canonical_index)
    });
    records
}

pub(super) fn fixture_edge_map(graph: &GraphData) -> Vec<(usize, usize)> {
    fixture_bond_records(graph).into_iter().map(|record| record.edge).collect()
}

pub(super) fn fixture_context_rows(graph: &GraphData) -> Vec<Vec<String>> {
    if graph.aromatic_ring_contexts.is_empty() {
        return Vec::new();
    }
    fixture_bond_records(graph)
        .into_iter()
        .map(|record| graph.aromatic_ring_contexts[record.canonical_index].clone())
        .collect()
}

pub(super) fn graph_node_labels(
    graph: &GraphData,
    atom_type_indices: &[u8],
    exact_connections_match: bool,
    respect_atom_aromaticity: bool,
) -> Vec<GroundTruthNodeLabel> {
    let normalized_edges = normalized_graph_bonds(graph);
    let explicit_degrees = explicit_degrees(graph.n_atoms, &normalized_edges);
    (0..graph.n_atoms)
        .map(|index| {
            GroundTruthNodeLabel {
                atom_type: atom_type_indices[index],
                explicit_degree: if exact_connections_match {
                    Some(explicit_degrees[index])
                } else {
                    None
                },
                is_aromatic: if respect_atom_aromaticity {
                    Some(graph.atom_is_aromatic[index])
                } else {
                    None
                },
            }
        })
        .collect()
}

pub(super) fn fixture_order_bond_labels(
    case: &GroundTruthCase,
    graph: &GraphData,
    atom_type_indices: &[u8],
) -> Vec<GroundTruthBondLabel> {
    let node_labels = graph_node_labels(
        graph,
        atom_type_indices,
        case_uses_exact_connections_match(case),
        case_respects_atom_aromaticity(case),
    );
    let ring_membership = graph_ring_membership_by_edge(graph);
    fixture_bond_records(graph)
        .into_iter()
        .map(|record| {
            let (src, dst) = record.edge;
            let left = node_labels[src];
            let right = node_labels[dst];
            let edge_value = GroundTruthEdgeValue {
                bond_order: if case_ignores_edge_values(case) {
                    None
                } else {
                    Some(record.bond_type)
                },
                in_ring: if case_uses_ring_matches_ring_only(case) {
                    Some(ring_membership[&(src, dst)])
                } else {
                    None
                },
            };
            if left <= right { (left, edge_value, right) } else { (right, edge_value, left) }
        })
        .collect()
}

pub(super) fn fixture_order_line_graph_labels(
    graph: &GraphData,
    atom_type_indices: &[u8],
    exact_connections_match: bool,
    respect_atom_aromaticity: bool,
) -> BTreeMap<(usize, usize), GroundTruthNodeLabel> {
    let node_labels = graph_node_labels(
        graph,
        atom_type_indices,
        exact_connections_match,
        respect_atom_aromaticity,
    );
    let edge_map = fixture_edge_map(graph);
    let mut labels = BTreeMap::new();
    for left in 0..edge_map.len() {
        for right in left + 1..edge_map.len() {
            let left_edge = edge_map[left];
            let right_edge = edge_map[right];
            if let Some(shared) = shared_endpoint(left_edge, right_edge) {
                labels.insert((left, right), node_labels[shared]);
            }
        }
    }
    labels
}

pub(super) fn collect_fixture_order_product_diagnostics(
    case: &GroundTruthCase,
    use_edge_contexts: bool,
) -> LabeledCaseProductDiagnostics {
    let (first_type_indices, second_type_indices) =
        atom_type_to_shared_indices(&case.graph1.atom_types, &case.graph2.atom_types);
    let first_edge_map = fixture_edge_map(&case.graph1);
    let second_edge_map = fixture_edge_map(&case.graph2);
    let first_bond_labels = fixture_order_bond_labels(case, &case.graph1, &first_type_indices);
    let second_bond_labels = fixture_order_bond_labels(case, &case.graph2, &second_type_indices);
    let first_line_graph_labels = fixture_order_line_graph_labels(
        &case.graph1,
        &first_type_indices,
        case_uses_exact_connections_match(case),
        case_respects_atom_aromaticity(case),
    );
    let second_line_graph_labels = fixture_order_line_graph_labels(
        &case.graph2,
        &second_type_indices,
        case_uses_exact_connections_match(case),
        case_respects_atom_aromaticity(case),
    );
    let first_contexts: Option<EdgeContexts<String>> =
        if case.graph1.aromatic_ring_contexts.is_empty() {
            None
        } else {
            Some(EdgeContexts::from_rows(fixture_context_rows(&case.graph1)))
        };
    let second_contexts: Option<EdgeContexts<String>> =
        if case.graph2.aromatic_ring_contexts.is_empty() {
            None
        } else {
            Some(EdgeContexts::from_rows(fixture_context_rows(&case.graph2)))
        };
    let use_edge_contexts = use_edge_contexts && case_uses_complete_aromatic_rings(case);

    let mut vertex_pairs = Vec::new();
    for first in 0..first_edge_map.len() {
        for second in 0..second_edge_map.len() {
            let contexts_match = match (&first_contexts, &second_contexts) {
                (Some(first_contexts), Some(second_contexts)) if use_edge_contexts => {
                    first_contexts.compatible_with(first, second_contexts, second)
                }
                _ => true,
            };
            if first_bond_labels[first] == second_bond_labels[second] && contexts_match {
                vertex_pairs.push((first, second));
            }
        }
    }

    let mut matrix = BitSquareMatrix::new(vertex_pairs.len());
    for left in 0..vertex_pairs.len() {
        let (first_left, second_left) = vertex_pairs[left];
        for right in left + 1..vertex_pairs.len() {
            let (first_right, second_right) = vertex_pairs[right];
            if first_left == first_right || second_left == second_right {
                continue;
            }
            let first_label = first_line_graph_labels
                .get(&(first_left.min(first_right), first_left.max(first_right)))
                .copied();
            let second_label = second_line_graph_labels
                .get(&(second_left.min(second_right), second_left.max(second_right)))
                .copied();
            if first_label == second_label {
                matrix.set_symmetric(left, right);
            }
        }
    }

    LabeledCaseProductDiagnostics {
        matrix,
        vertex_pairs,
        first_edge_map,
        second_edge_map,
        first_bond_labels,
        second_bond_labels,
    }
}

pub(super) fn find_case<'a>(cases: &'a [GroundTruthCase], name: &str) -> &'a GroundTruthCase {
    cases.iter().find(|case| case.name == name).unwrap_or_else(|| panic!("missing case '{name}'"))
}

pub(super) fn reverse_graph_bond_payload(graph: &GraphData) -> GraphData {
    let mut reversed = graph.clone();
    reversed.edges.reverse();
    reversed.bond_types.reverse();
    reversed.aromatic_ring_contexts.reverse();
    reversed.bond_orientations.reverse();
    reversed
}

pub(super) fn permute_product(
    matrix: &BitSquareMatrix,
    vertex_pairs: &[(usize, usize)],
    order: &[usize],
) -> (BitSquareMatrix, Vec<(usize, usize)>) {
    assert_eq!(order.len(), matrix.order());
    let mut seen = vec![false; order.len()];
    for &old in order {
        assert!(old < order.len(), "permutation index out of bounds");
        assert!(!seen[old], "permutation contains duplicate vertex {old}");
        seen[old] = true;
    }

    let mut permuted = BitSquareMatrix::new(order.len());
    for new_left in 0..order.len() {
        for new_right in new_left + 1..order.len() {
            if matrix.has_entry(order[new_left], order[new_right]) {
                permuted.set_symmetric(new_left, new_right);
            }
        }
    }

    let permuted_pairs = order.iter().map(|&old| vertex_pairs[old]).collect();
    (permuted, permuted_pairs)
}

pub(super) fn rank_partitioned_cliques(
    cliques: Vec<Vec<usize>>,
    vertex_pairs: &[(usize, usize)],
    first_edge_map: &[(usize, usize)],
    second_edge_map: &[(usize, usize)],
) -> Vec<EagerCliqueInfo<usize>> {
    let mut infos: Vec<EagerCliqueInfo<usize>> = cliques
        .into_iter()
        .map(|clique| {
            EagerCliqueInfo::new(
                clique,
                vertex_pairs,
                first_edge_map,
                second_edge_map,
                |_, _, _, _| true,
            )
        })
        .collect();
    let ranker =
        FragmentCountRanker.then(LargestFragmentMetricRanker::new(LargestFragmentMetric::Atoms));
    infos.sort_by(|left, right| ranker.compare(left, right));
    infos
}

pub(super) fn info_johnson_similarity(
    case: &GroundTruthCase,
    info: &EagerCliqueInfo<usize>,
) -> f64 {
    geometric_traits::traits::algorithms::johnson_similarity(
        info.matched_edges().len(),
        info.vertex_matches().len(),
        case.graph1.n_atoms,
        case.graph1.edges.len(),
        case.graph2.n_atoms,
        case.graph2.edges.len(),
    )
}

pub(super) fn product_order_identity(vertex_pairs: &[(usize, usize)]) -> Vec<usize> {
    (0..vertex_pairs.len()).collect()
}

pub(super) fn product_order_reverse(vertex_pairs: &[(usize, usize)]) -> Vec<usize> {
    let mut order = product_order_identity(vertex_pairs);
    order.reverse();
    order
}

pub(super) fn product_order_second_then_first(vertex_pairs: &[(usize, usize)]) -> Vec<usize> {
    let mut order = product_order_identity(vertex_pairs);
    order.sort_unstable_by_key(|&index| {
        let (first, second) = vertex_pairs[index];
        (second, first, index)
    });
    order
}

pub(super) fn product_order_reverse_within_first_buckets(
    vertex_pairs: &[(usize, usize)],
) -> Vec<usize> {
    let mut buckets = BTreeMap::<usize, Vec<usize>>::new();
    for (index, &(first, _)) in vertex_pairs.iter().enumerate() {
        buckets.entry(first).or_default().push(index);
    }
    let mut order = Vec::with_capacity(vertex_pairs.len());
    for bucket in buckets.values() {
        order.extend(bucket.iter().rev().copied());
    }
    order
}

pub(super) fn product_order_target_last(
    vertex_pairs: &[(usize, usize)],
    target_clique: &[usize],
) -> Vec<usize> {
    let target: std::collections::BTreeSet<usize> = target_clique.iter().copied().collect();
    let mut order = Vec::with_capacity(vertex_pairs.len());
    order.extend((0..vertex_pairs.len()).filter(|index| !target.contains(index)));
    order.extend(target_clique.iter().copied());
    order
}

pub(super) fn fixture_edge_rank_map(graph: &GraphData) -> BTreeMap<(usize, usize), usize> {
    let mut ranks = BTreeMap::new();
    for record in fixture_bond_records(graph) {
        ranks.entry(record.edge).or_insert(record.original_index);
    }
    ranks
}

pub(super) fn product_order_fixture_edge_indices(
    case: &GroundTruthCase,
    diagnostics: &LabeledCaseProductDiagnostics,
) -> Vec<usize> {
    let first_ranks = fixture_edge_rank_map(&case.graph1);
    let second_ranks = fixture_edge_rank_map(&case.graph2);
    let mut order = product_order_identity(&diagnostics.vertex_pairs);
    order.sort_unstable_by_key(|&index| {
        let (first, second) = diagnostics.vertex_pairs[index];
        (
            first_ranks[&diagnostics.first_edge_map[first]],
            second_ranks[&diagnostics.second_edge_map[second]],
            index,
        )
    });
    order
}

pub(super) fn product_order_fixture_edge_indices_second_then_first(
    case: &GroundTruthCase,
    diagnostics: &LabeledCaseProductDiagnostics,
) -> Vec<usize> {
    let first_ranks = fixture_edge_rank_map(&case.graph1);
    let second_ranks = fixture_edge_rank_map(&case.graph2);
    let mut order = product_order_identity(&diagnostics.vertex_pairs);
    order.sort_unstable_by_key(|&index| {
        let (first, second) = diagnostics.vertex_pairs[index];
        (
            second_ranks[&diagnostics.second_edge_map[second]],
            first_ranks[&diagnostics.first_edge_map[first]],
            index,
        )
    });
    order
}

pub(super) fn product_order_rdkit_raw_pair_order(
    case: &GroundTruthCase,
    diagnostics: &LabeledCaseProductDiagnostics,
) -> Vec<usize> {
    if case.graph1.n_atoms <= case.graph2.n_atoms {
        product_order_fixture_edge_indices(case, diagnostics)
    } else {
        product_order_fixture_edge_indices_second_then_first(case, diagnostics)
    }
}

pub(super) fn configure_rdkit_raw_pair_order<'g, PF, XC, EC, D, R>(
    builder: McesBuilder<'g, TypedGraph, PF, XC, EC, D, R>,
    case: &GroundTruthCase,
) -> McesBuilder<'g, TypedGraph, PF, XC, EC, D, R> {
    let first_ranks = fixture_edge_rank_map(&case.graph1);
    let second_ranks = fixture_edge_rank_map(&case.graph2);
    let second_major = case.graph1.n_atoms > case.graph2.n_atoms;

    builder.with_product_vertex_ordering(move |_first_lg, _second_lg, first_edge, second_edge| {
        let first_rank = first_ranks[&canonical_edge([first_edge.0, first_edge.1])];
        let second_rank = second_ranks[&canonical_edge([second_edge.0, second_edge.1])];
        if second_major { (second_rank, first_rank) } else { (first_rank, second_rank) }
    })
}

pub(super) fn permuted_partitioned_infos(
    case: &GroundTruthCase,
    diagnostics: &LabeledCaseProductDiagnostics,
    order: &[usize],
    partition_side: geometric_traits::traits::algorithms::maximum_clique::PartitionSide,
    search_mode: McesSearchMode,
) -> Vec<EagerCliqueInfo<usize>> {
    let (matrix, permuted_pairs) =
        permute_product(&diagnostics.matrix, &diagnostics.vertex_pairs, order);
    let (g1_label_indices, g2_label_indices, num_labels) =
        intern_case_bond_labels(&diagnostics.first_bond_labels, &diagnostics.second_bond_labels);
    let partition = PartitionInfo {
        pairs: &permuted_pairs,
        g1_labels: &g1_label_indices,
        g2_labels: &g2_label_indices,
        num_labels,
        partition_side,
    };
    let accept = |clique: &[usize]| {
        !clique_has_delta_y_from_product(
            clique,
            &permuted_pairs,
            &diagnostics.first_edge_map,
            &diagnostics.second_edge_map,
            case.graph1.n_atoms,
            case.graph2.n_atoms,
        )
    };
    let cliques = match search_mode {
        McesSearchMode::PartialEnumeration => {
            geometric_traits::traits::algorithms::maximum_clique::partial_search(
                &matrix, &partition, 0, accept,
            )
        }
        McesSearchMode::AllBest => {
            matrix.all_maximum_cliques_with_partition_where(&partition, accept)
        }
    };
    rank_partitioned_cliques(
        cliques,
        &permuted_pairs,
        &diagnostics.first_edge_map,
        &diagnostics.second_edge_map,
    )
}

pub(super) fn run_labeled_case(case: &GroundTruthCase) -> McesResult<usize> {
    run_labeled_case_with_default_orientation(case, true, McesSearchMode::PartialEnumeration)
}

pub(super) fn run_labeled_case_with_contexts(
    case: &GroundTruthCase,
    use_edge_contexts: bool,
) -> McesResult<usize> {
    run_labeled_case_with_search_mode(
        case,
        use_edge_contexts,
        false,
        McesSearchMode::PartialEnumeration,
    )
}

pub(super) fn run_labeled_case_with_options(
    case: &GroundTruthCase,
    use_edge_contexts: bool,
    use_partition_orientation_heuristic: bool,
) -> McesResult<usize> {
    run_labeled_case_with_search_mode(
        case,
        use_edge_contexts,
        use_partition_orientation_heuristic,
        McesSearchMode::PartialEnumeration,
    )
}

pub(super) fn run_labeled_case_with_default_orientation(
    case: &GroundTruthCase,
    use_edge_contexts: bool,
    search_mode: McesSearchMode,
) -> McesResult<usize> {
    let prepared = prepare_labeled_case(case);

    if use_edge_contexts && case_uses_complete_aromatic_rings(case) {
        if let (Some(graph1_contexts), Some(graph2_contexts)) =
            (prepared.first_contexts.as_ref(), prepared.second_contexts.as_ref())
        {
            return configure_rdkit_raw_pair_order(
                McesBuilder::new(&prepared.first, &prepared.second)
                    .with_edge_contexts(graph1_contexts, graph2_contexts)
                    .with_largest_fragment_metric(LargestFragmentMetric::Atoms)
                    .with_search_mode(search_mode),
                case,
            )
            .compute_labeled();
        }
    }

    configure_rdkit_raw_pair_order(
        McesBuilder::new(&prepared.first, &prepared.second)
            .with_largest_fragment_metric(LargestFragmentMetric::Atoms)
            .with_search_mode(search_mode),
        case,
    )
    .compute_labeled()
}

pub(super) fn run_labeled_case_with_search_mode(
    case: &GroundTruthCase,
    use_edge_contexts: bool,
    use_partition_orientation_heuristic: bool,
    search_mode: McesSearchMode,
) -> McesResult<usize> {
    let prepared = prepare_labeled_case(case);

    if use_edge_contexts && case_uses_complete_aromatic_rings(case) {
        if let (Some(graph1_contexts), Some(graph2_contexts)) =
            (prepared.first_contexts.as_ref(), prepared.second_contexts.as_ref())
        {
            return configure_rdkit_raw_pair_order(
                McesBuilder::new(&prepared.first, &prepared.second)
                    .with_edge_contexts(graph1_contexts, graph2_contexts)
                    .with_largest_fragment_metric(LargestFragmentMetric::Atoms)
                    .with_partition_orientation_heuristic(use_partition_orientation_heuristic)
                    .with_search_mode(search_mode),
                case,
            )
            .compute_labeled();
        }
    }

    configure_rdkit_raw_pair_order(
        McesBuilder::new(&prepared.first, &prepared.second)
            .with_largest_fragment_metric(LargestFragmentMetric::Atoms)
            .with_partition_orientation_heuristic(use_partition_orientation_heuristic)
            .with_search_mode(search_mode),
        case,
    )
    .compute_labeled()
}

pub(super) fn assert_labeled_result_matches_ground_truth(
    case: &GroundTruthCase,
    result: &McesResult<usize>,
    context: &str,
) {
    assert_eq!(
        result.matched_edges().len(),
        case.expected_bond_matches,
        "case '{}': {context} matched edges ({}) != expected ({})",
        case.name,
        result.matched_edges().len(),
        case.expected_bond_matches,
    );

    let similarity = result.johnson_similarity();
    assert!(
        (similarity - case.expected_similarity).abs() <= 1e-6,
        "case '{}': {context} similarity ({similarity:.6}) != expected ({:.6})",
        case.name,
        case.expected_similarity,
    );
}

pub(super) fn labeled_result_mismatch(
    case: &GroundTruthCase,
    result: &McesResult<usize>,
) -> Option<String> {
    let matched_edges = result.matched_edges().len();
    let similarity = result.johnson_similarity();
    if matched_edges == case.expected_bond_matches
        && (similarity - case.expected_similarity).abs() <= 1e-6
    {
        return None;
    }

    Some(format!(
        "{}: matched_edges={} expected_edges={} similarity={similarity:.6} expected_similarity={:.6}",
        case.name, matched_edges, case.expected_bond_matches, case.expected_similarity
    ))
}

pub(super) fn labeled_info_mismatch(
    case: &GroundTruthCase,
    info: &EagerCliqueInfo<usize>,
) -> Option<String> {
    let matched_edges = info.matched_edges().len();
    let similarity = info_johnson_similarity(case, info);
    if matched_edges == case.expected_bond_matches
        && (similarity - case.expected_similarity).abs() <= 1e-6
    {
        return None;
    }

    Some(format!(
        "{}: matched_edges={} expected_edges={} similarity={similarity:.6} expected_similarity={:.6}",
        case.name, matched_edges, case.expected_bond_matches, case.expected_similarity
    ))
}

pub(super) fn inferred_atom_count_from_similarity(
    case: &GroundTruthCase,
    common_edges: usize,
    similarity: f64,
) -> usize {
    let denominator = (case.graph1.n_atoms + case.graph1.edges.len())
        * (case.graph2.n_atoms + case.graph2.edges.len());
    let numerator = (similarity * denominator as f64).sqrt();
    let atoms = numerator - common_edges as f64;
    atoms.round() as usize
}

pub(super) fn canonical_edge(edge: [usize; 2]) -> (usize, usize) {
    if edge[0] <= edge[1] { (edge[0], edge[1]) } else { (edge[1], edge[0]) }
}

pub(super) fn normalized_graph_edges(graph: &GraphData) -> Vec<(usize, usize, u32)> {
    let mut normalized_edges: Vec<(usize, usize, u32)> = graph
        .edges
        .iter()
        .zip(graph.bond_types.iter().copied())
        .map(|(edge, bond_type)| {
            if edge[0] <= edge[1] {
                (edge[0], edge[1], bond_type)
            } else {
                (edge[1], edge[0], bond_type)
            }
        })
        .collect();
    normalized_edges.sort_unstable_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
    normalized_edges.dedup();
    normalized_edges
}

pub(super) fn graph_ring_membership_by_edge(graph: &GraphData) -> BTreeMap<(usize, usize), bool> {
    let normalized_edges = normalized_graph_edges(graph);
    normalized_edges
        .iter()
        .copied()
        .zip(edge_ring_membership(graph.n_atoms, &normalized_edges))
        .map(|((src, dst, _), in_ring)| ((src, dst), in_ring))
        .collect()
}

pub(super) fn graph_distance_matrix(graph: &GraphData) -> Vec<Vec<usize>> {
    let mut adjacency = vec![Vec::<usize>::new(); graph.n_atoms];
    for &edge in &graph.edges {
        adjacency[edge[0]].push(edge[1]);
        adjacency[edge[1]].push(edge[0]);
    }

    let mut distances = vec![vec![usize::MAX; graph.n_atoms]; graph.n_atoms];
    for source in 0..graph.n_atoms {
        let mut queue = VecDeque::from([source]);
        distances[source][source] = 0;
        while let Some(node) = queue.pop_front() {
            let next_distance = distances[source][node] + 1;
            for &neighbor in &adjacency[node] {
                if distances[source][neighbor] != usize::MAX {
                    continue;
                }
                distances[source][neighbor] = next_distance;
                queue.push_back(neighbor);
            }
        }
    }

    distances
}

pub(super) fn non_empty_context_row_count(graph: &GraphData) -> usize {
    graph.aromatic_ring_contexts.iter().filter(|contexts| !contexts.is_empty()).count()
}

pub(super) fn matched_edge_fragment_stats(
    matched_edges: &[((usize, usize), (usize, usize))],
) -> (usize, usize) {
    if matched_edges.is_empty() {
        return (0, 0);
    }

    let mut adjacency = BTreeMap::<usize, Vec<usize>>::new();
    for &((src, dst), _) in matched_edges {
        adjacency.entry(src).or_default().push(dst);
        adjacency.entry(dst).or_default().push(src);
    }

    let mut visited = std::collections::BTreeSet::<usize>::new();
    let mut queue = VecDeque::new();
    let mut fragment_count = 0usize;
    let mut largest_fragment_atoms = 0usize;

    for &start in adjacency.keys() {
        if !visited.insert(start) {
            continue;
        }
        queue.push_back(start);
        let mut fragment_atoms = 0usize;
        while let Some(node) = queue.pop_front() {
            fragment_atoms += 1;
            for &neighbor in &adjacency[&node] {
                if visited.insert(neighbor) {
                    queue.push_back(neighbor);
                }
            }
        }
        fragment_count += 1;
        largest_fragment_atoms = largest_fragment_atoms.max(fragment_atoms);
    }

    (fragment_count, largest_fragment_atoms)
}

pub(super) fn clique_degree_sequence(num_vertices: usize, edges: &[(usize, usize)]) -> Vec<usize> {
    let mut counts = vec![0usize; num_vertices];
    for &(src, dst) in edges {
        counts[src] += 1;
        counts[dst] += 1;
    }
    let mut seq: Vec<usize> = counts.into_iter().filter(|&degree| degree > 0).collect();
    seq.sort_unstable();
    seq
}

pub(super) fn clique_has_delta_y_from_product(
    clique: &[usize],
    vertex_pairs: &[(usize, usize)],
    first_edge_map: &[(usize, usize)],
    second_edge_map: &[(usize, usize)],
    first_vertices: usize,
    second_vertices: usize,
) -> bool {
    let first_edges: Vec<(usize, usize)> =
        clique.iter().map(|&k| first_edge_map[vertex_pairs[k].0]).collect();
    let second_edges: Vec<(usize, usize)> =
        clique.iter().map(|&k| second_edge_map[vertex_pairs[k].1]).collect();
    clique_degree_sequence(first_vertices, &first_edges)
        != clique_degree_sequence(second_vertices, &second_edges)
}

pub(super) fn other_endpoint(edge: (usize, usize), shared: usize) -> usize {
    if edge.0 == shared { edge.1 } else { edge.0 }
}

pub(super) fn graph_edge_index(graph: &GraphData, edge: (usize, usize)) -> usize {
    let canonical = canonical_edge([edge.0, edge.1]);
    graph
        .edges
        .iter()
        .position(|&candidate| canonical_edge(candidate) == canonical)
        .unwrap_or_else(|| panic!("missing graph edge for {:?}", edge))
}

pub(super) fn infer_vertex_matches_rdkit_style(
    case: &GroundTruthCase,
    clique: &[usize],
    diagnostics: &LabeledCaseProductDiagnostics,
) -> Vec<(usize, usize)> {
    if clique.is_empty() {
        return Vec::new();
    }

    let matched_edges: Vec<((usize, usize), (usize, usize))> = clique
        .iter()
        .map(|&k| {
            let (left, right) = diagnostics.vertex_pairs[k];
            (diagnostics.first_edge_map[left], diagnostics.second_edge_map[right])
        })
        .collect();
    let edge_indices: Vec<(usize, usize)> = clique
        .iter()
        .map(|&k| {
            let (left, right) = diagnostics.vertex_pairs[k];
            (
                graph_edge_index(&case.graph1, diagnostics.first_edge_map[left]),
                graph_edge_index(&case.graph2, diagnostics.second_edge_map[right]),
            )
        })
        .collect();

    let mut mol1_matches = vec![-1isize; case.graph1.n_atoms];
    for &((src, dst), _) in &matched_edges {
        mol1_matches[src] = -2;
        mol1_matches[dst] = -2;
    }

    for i in 0..matched_edges.len().saturating_sub(1) {
        let (left_edge_1, right_edge_1) = matched_edges[i];
        for j in i + 1..matched_edges.len() {
            let (left_edge_2, right_edge_2) = matched_edges[j];
            if let (Some(left_shared), Some(right_shared)) = (
                shared_endpoint(left_edge_1, left_edge_2),
                shared_endpoint(right_edge_1, right_edge_2),
            ) {
                mol1_matches[left_shared] = right_shared as isize;

                let other_left = other_endpoint(left_edge_1, left_shared);
                let other_right = other_endpoint(right_edge_1, right_shared);
                mol1_matches[other_left] = other_right as isize;

                let other_left = other_endpoint(left_edge_2, left_shared);
                let other_right = other_endpoint(right_edge_2, right_shared);
                mol1_matches[other_left] = other_right as isize;
            }
        }
    }

    if mol1_matches.contains(&-2) {
        for ((left_edge, _), &(left_edge_index, right_edge_index)) in
            matched_edges.iter().zip(edge_indices.iter())
        {
            if mol1_matches[left_edge.0] == -2 && mol1_matches[left_edge.1] == -2 {
                let ((left_first, right_first), (left_second, right_second), _) =
                    rdkit_preferred_isolated_mapping(case, left_edge_index, right_edge_index);
                mol1_matches[left_first] = right_first as isize;
                mol1_matches[left_second] = right_second as isize;
            }
        }
    }

    mol1_matches
        .into_iter()
        .enumerate()
        .filter_map(|(left, right)| (right >= 0).then_some((left, right as usize)))
        .collect()
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct CliqueScorecard {
    pub(super) matched_bonds: usize,
    pub(super) matched_atoms: usize,
    pub(super) fragment_count: usize,
    pub(super) largest_fragment_size: usize,
    pub(super) ring_non_ring_bond_score: usize,
    pub(super) atom_h_score: usize,
    pub(super) max_delta_atom_atom_dist: usize,
}

pub(super) fn score_clique(
    case: &GroundTruthCase,
    info: &EagerCliqueInfo<usize>,
    first_ring_membership: &BTreeMap<(usize, usize), bool>,
    second_ring_membership: &BTreeMap<(usize, usize), bool>,
    first_distances: &[Vec<usize>],
    second_distances: &[Vec<usize>],
) -> CliqueScorecard {
    let (fragment_count, largest_fragment_size) = matched_edge_fragment_stats(info.matched_edges());
    let ring_non_ring_bond_score = info
        .matched_edges()
        .iter()
        .filter(|&&((src1, dst1), (src2, dst2))| {
            let first_in_ring = first_ring_membership[&canonical_edge([src1, dst1])];
            let second_in_ring = second_ring_membership[&canonical_edge([src2, dst2])];
            first_in_ring != second_in_ring
        })
        .count();

    let atom_h_score: usize = info
        .vertex_matches()
        .iter()
        .map(|&(first_atom, second_atom)| {
            let first_hs = case.graph1.atom_total_hs.get(first_atom).copied().unwrap_or(0);
            let second_hs = case.graph2.atom_total_hs.get(second_atom).copied().unwrap_or(0);
            usize::from(first_hs.abs_diff(second_hs))
        })
        .sum();

    let mut max_delta_atom_atom_dist = 0usize;
    for (index, &(first_left, second_left)) in info.vertex_matches().iter().enumerate() {
        for &(first_right, second_right) in &info.vertex_matches()[index + 1..] {
            let first_distance = first_distances[first_left][first_right];
            let second_distance = second_distances[second_left][second_right];
            let delta = first_distance.abs_diff(second_distance);
            max_delta_atom_atom_dist = max_delta_atom_atom_dist.max(delta);
        }
    }

    CliqueScorecard {
        matched_bonds: info.matched_edges().len(),
        matched_atoms: info.vertex_matches().len(),
        fragment_count,
        largest_fragment_size,
        ring_non_ring_bond_score,
        atom_h_score,
        max_delta_atom_atom_dist,
    }
}

pub(super) fn scorecards_for_infos(
    case: &GroundTruthCase,
    infos: &[EagerCliqueInfo<usize>],
) -> Vec<CliqueScorecard> {
    let first_ring_membership = graph_ring_membership_by_edge(&case.graph1);
    let second_ring_membership = graph_ring_membership_by_edge(&case.graph2);
    let first_distances = graph_distance_matrix(&case.graph1);
    let second_distances = graph_distance_matrix(&case.graph2);

    infos
        .iter()
        .map(|info| {
            score_clique(
                case,
                info,
                &first_ring_membership,
                &second_ring_membership,
                &first_distances,
                &second_distances,
            )
        })
        .collect()
}

pub(super) fn ranked_indices_by_scorecards<F>(
    scorecards: &[CliqueScorecard],
    compare: F,
) -> Vec<usize>
where
    F: Fn(&CliqueScorecard, &CliqueScorecard) -> core::cmp::Ordering,
{
    let mut indices: Vec<usize> = (0..scorecards.len()).collect();
    indices.sort_unstable_by(|&left, &right| {
        compare(&scorecards[left], &scorecards[right]).then_with(|| left.cmp(&right))
    });
    indices
}

pub(super) fn first_missing_rdkit_compare(
    left: &CliqueScorecard,
    right: &CliqueScorecard,
) -> core::cmp::Ordering {
    left.fragment_count
        .cmp(&right.fragment_count)
        .then_with(|| right.largest_fragment_size.cmp(&left.largest_fragment_size))
        .then_with(|| left.ring_non_ring_bond_score.cmp(&right.ring_non_ring_bond_score))
}

pub(super) fn reordered_all_best_indices_by_first_missing_ranker(
    case: &GroundTruthCase,
    result: &McesResult<usize>,
) -> Vec<usize> {
    let scorecards = scorecards_for_infos(case, result.all_cliques());
    ranked_indices_by_scorecards(&scorecards, first_missing_rdkit_compare)
}

pub(super) fn approx_rdkit_compare(
    left: &CliqueScorecard,
    right: &CliqueScorecard,
) -> core::cmp::Ordering {
    right
        .matched_bonds
        .cmp(&left.matched_bonds)
        .then_with(|| left.fragment_count.cmp(&right.fragment_count))
        .then_with(|| right.largest_fragment_size.cmp(&left.largest_fragment_size))
        .then_with(|| left.ring_non_ring_bond_score.cmp(&right.ring_non_ring_bond_score))
        .then_with(|| left.atom_h_score.cmp(&right.atom_h_score))
        .then_with(|| left.max_delta_atom_atom_dist.cmp(&right.max_delta_atom_atom_dist))
}

pub(super) fn selected_clique_edge_indices(
    result: &McesResult<usize>,
    diagnostics: &LabeledCaseProductDiagnostics,
) -> Vec<(usize, usize)> {
    result
        .all_cliques()
        .first()
        .map(|info| info.clique().iter().map(|&k| diagnostics.vertex_pairs[k]).collect())
        .unwrap_or_default()
}

pub(super) fn edge_components(
    edge_indices: &[(usize, usize)],
    edges: &[[usize; 2]],
) -> Vec<Vec<(usize, usize)>> {
    let mut visited = vec![false; edge_indices.len()];
    let mut components = Vec::new();

    for start in 0..edge_indices.len() {
        if visited[start] {
            continue;
        }

        let mut stack = vec![start];
        visited[start] = true;
        let mut component = Vec::new();

        while let Some(index) = stack.pop() {
            let pair = edge_indices[index];
            component.push(pair);

            let edge = edges[pair.0];
            for (candidate_index, candidate_pair) in edge_indices.iter().copied().enumerate() {
                if visited[candidate_index] {
                    continue;
                }
                let candidate_edge = edges[candidate_pair.0];
                let shares_vertex = edge[0] == candidate_edge[0]
                    || edge[0] == candidate_edge[1]
                    || edge[1] == candidate_edge[0]
                    || edge[1] == candidate_edge[1];
                if shares_vertex {
                    visited[candidate_index] = true;
                    stack.push(candidate_index);
                }
            }
        }

        components.push(component);
    }

    components
}

pub(super) fn graph_edge_orientation(graph: &GraphData, edge_index: usize) -> [usize; 2] {
    graph.bond_orientations.get(edge_index).copied().unwrap_or(graph.edges[edge_index])
}

pub(super) fn graph_atom_total_hs(graph: &GraphData, atom_index: usize) -> Option<u8> {
    graph.atom_total_hs.get(atom_index).copied()
}

pub(super) fn rdkit_preferred_isolated_mapping(
    case: &GroundTruthCase,
    left_edge_index: usize,
    right_edge_index: usize,
) -> ((usize, usize), (usize, usize), &'static str) {
    let [left_begin, left_end] = graph_edge_orientation(&case.graph1, left_edge_index);
    let [right_begin, right_end] = graph_edge_orientation(&case.graph2, right_edge_index);

    let left_begin_type = case.graph1.atom_types[left_begin].as_str();
    let left_end_type = case.graph1.atom_types[left_end].as_str();

    if left_begin_type != left_end_type {
        if left_begin_type == case.graph2.atom_types[right_begin].as_str() {
            return ((left_begin, right_begin), (left_end, right_end), "atomic_number");
        }
        return ((left_begin, right_end), (left_end, right_begin), "atomic_number");
    }

    match (
        graph_atom_total_hs(&case.graph1, left_begin),
        graph_atom_total_hs(&case.graph1, left_end),
    ) {
        (Some(left_begin_hs), Some(left_end_hs)) if left_begin_hs != left_end_hs => {
            if left_begin_hs > left_end_hs {
                ((left_begin, right_begin), (left_end, right_end), "total_hydrogens")
            } else {
                ((left_begin, right_end), (left_end, right_begin), "total_hydrogens")
            }
        }
        _ => ((left_begin, right_begin), (left_end, right_end), "fallback"),
    }
}

// ============================================================================
// Labeled MCES ground truth tests
// ============================================================================
