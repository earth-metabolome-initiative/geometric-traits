//! Shared support for ground-truth MCES tests.

pub(super) use std::{collections::BTreeMap, fs::File, io::Read as _};

pub(super) use geometric_traits::{
    impls::{CSR2D, EdgeContexts, SortedVec, SquareCSR2D, SymmetricCSR2D, ValuedCSR2D},
    prelude::*,
    traits::{EdgesBuilder, MatrixMut, SparseMatrixMut, TypedNode, VocabularyBuilder},
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

pub(super) struct LabeledCaseProductDiagnostics {
    pub(super) vertex_pairs: Vec<(usize, usize)>,
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
        vertex_pairs: product.vertex_pairs().to_vec(),
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

pub(super) fn find_case<'a>(cases: &'a [GroundTruthCase], name: &str) -> &'a GroundTruthCase {
    cases.iter().find(|case| case.name == name).unwrap_or_else(|| panic!("missing case '{name}'"))
}

pub(super) fn fixture_edge_rank_map(graph: &GraphData) -> BTreeMap<(usize, usize), usize> {
    let mut ranks = BTreeMap::new();
    for record in fixture_bond_records(graph) {
        ranks.entry(record.edge).or_insert(record.original_index);
    }
    ranks
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

pub(super) fn run_labeled_case(case: &GroundTruthCase) -> McesResult<usize> {
    run_labeled_case_with_search_mode(case, true, McesSearchMode::PartialEnumeration)
}

pub(super) fn run_labeled_case_with_search_mode(
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

pub(super) fn canonical_edge(edge: [usize; 2]) -> (usize, usize) {
    if edge[0] <= edge[1] { (edge[0], edge[1]) } else { (edge[1], edge[0]) }
}

// ============================================================================
// Labeled MCES ground truth tests
// ============================================================================
