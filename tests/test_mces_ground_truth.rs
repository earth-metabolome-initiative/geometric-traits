//! Ground truth tests for the MCES pipeline.
//!
//! Loads test cases generated from RDKit's RASCAL test suite
//! (`tests/fixtures/mces_ground_truth.json.gz`) and validates our
//! labeled MCES results against them.
#![cfg(feature = "std")]

use std::{
    io::{Read as _, Write as _},
    time::Instant,
};

use geometric_traits::{
    impls::{CSR2D, EdgeContexts, SortedVec, SquareCSR2D, SymmetricCSR2D, ValuedCSR2D},
    prelude::*,
    traits::{EdgesBuilder, MatrixMut, SparseMatrixMut, TypedNode, VocabularyBuilder},
};

// ============================================================================
// Typed node infrastructure for labeled MCES
// ============================================================================

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct GroundTruthNodeLabel {
    atom_type: u8,
    explicit_degree: Option<u8>,
    is_aromatic: Option<bool>,
}

/// A node labeled by a generic harness-local node label.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct AtomNode {
    id: usize,
    node_label: GroundTruthNodeLabel,
}

impl TypedNode for AtomNode {
    type NodeType = GroundTruthNodeLabel;
    fn node_type(&self) -> Self::NodeType {
        self.node_label
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct GroundTruthEdgeValue {
    bond_order: Option<u32>,
    in_ring: Option<bool>,
}

type TypedGraph = geometric_traits::naive_structs::GenericGraph<
    SortedVec<AtomNode>,
    SymmetricCSR2D<ValuedCSR2D<usize, usize, usize, GroundTruthEdgeValue>>,
>;
type GroundTruthBondLabel = (GroundTruthNodeLabel, GroundTruthEdgeValue, GroundTruthNodeLabel);

/// Maps atom type strings across both graphs to a shared sequential u8 space.
fn atom_type_to_shared_indices(
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

fn build_typed_graph(
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
        .map(|(e, bond_type)| if e[0] < e[1] { (e[0], e[1], bond_type) } else { (e[1], e[0], bond_type) })
        .collect();
    normalized_edges
        .sort_unstable_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
    normalized_edges.dedup();

    let explicit_degrees = explicit_degrees(n_atoms, &normalized_edges);
    let ring_membership = edge_ring_membership(n_atoms, &normalized_edges);

    let nodes_vec: Vec<AtomNode> = (0..n_atoms)
        .map(|i| AtomNode {
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

fn explicit_degrees(n_atoms: usize, edges: &[(usize, usize, u32)]) -> Vec<u8> {
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

fn edge_ring_membership(n_atoms: usize, edges: &[(usize, usize, u32)]) -> Vec<bool> {
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

struct PreparedLabeledCase {
    first: TypedGraph,
    second: TypedGraph,
    first_contexts: Option<EdgeContexts<String>>,
    second_contexts: Option<EdgeContexts<String>>,
}

// Also keep unlabeled graph builder for comparison tests.
fn build_unlabeled_graph(
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

// ============================================================================
// Ground truth loading
// ============================================================================

#[derive(serde::Deserialize)]
struct GroundTruthFile {
    version: u32,
    cases: Vec<GroundTruthCase>,
}

#[derive(serde::Deserialize)]
struct GraphData {
    n_atoms: usize,
    edges: Vec<[usize; 2]>,
    atom_types: Vec<String>,
    bond_types: Vec<u32>,
    #[serde(default)]
    aromatic_ring_contexts: Vec<Vec<String>>,
    #[allow(dead_code)]
    atom_is_aromatic: Vec<bool>,
}

#[derive(serde::Deserialize)]
struct GroundTruthCase {
    name: String,
    #[allow(dead_code)]
    smiles1: String,
    #[allow(dead_code)]
    smiles2: String,
    graph1: GraphData,
    graph2: GraphData,
    expected_bond_matches: usize,
    #[allow(dead_code)]
    expected_atom_matches: usize,
    expected_similarity: f64,
    timed_out: bool,
    #[allow(dead_code)]
    options: Option<serde_json::Value>,
}

static GROUND_TRUTH_GZ: &[u8] = include_bytes!("fixtures/mces_ground_truth.json.gz");

fn load_ground_truth() -> Vec<GroundTruthCase> {
    let mut decoder = flate2::read::GzDecoder::new(GROUND_TRUTH_GZ);
    let mut json_str = String::new();
    decoder.read_to_string(&mut json_str).unwrap();
    let file: GroundTruthFile = serde_json::from_str(&json_str).unwrap();
    assert!(file.version >= 1);
    file.cases
}

fn case_ignores_edge_values(case: &GroundTruthCase) -> bool {
    case.options
        .as_ref()
        .and_then(|options| options.get("ignoreBondOrders"))
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false)
}

fn case_uses_complete_aromatic_rings(case: &GroundTruthCase) -> bool {
    case.options
        .as_ref()
        .and_then(|options| options.get("completeAromaticRings"))
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(true)
}

fn case_uses_ring_matches_ring_only(case: &GroundTruthCase) -> bool {
    case.options
        .as_ref()
        .and_then(|options| options.get("ringMatchesRingOnly"))
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false)
}

fn case_uses_exact_connections_match(case: &GroundTruthCase) -> bool {
    case.options
        .as_ref()
        .and_then(|options| options.get("exactConnectionsMatch"))
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false)
}

fn case_respects_atom_aromaticity(case: &GroundTruthCase) -> bool {
    matches!(
        case.options
            .as_ref()
            .and_then(|options| options.get("ignoreAtomAromaticity"))
            .and_then(serde_json::Value::as_bool),
        Some(false)
    )
}

fn build_edge_contexts(graph: &GraphData) -> Option<EdgeContexts<String>> {
    if graph.aromatic_ring_contexts.is_empty() {
        return None;
    }
    Some(EdgeContexts::from_rows(graph.aromatic_ring_contexts.iter().cloned()))
}

fn prepare_labeled_case(case: &GroundTruthCase) -> PreparedLabeledCase {
    let (first_type_indices, second_type_indices) =
        atom_type_to_shared_indices(&case.graph1.atom_types, &case.graph2.atom_types);

    PreparedLabeledCase {
        first: build_typed_graph(
            case.graph1.n_atoms,
            &case.graph1.edges,
            &first_type_indices,
            &case.graph1.atom_is_aromatic,
            &case.graph1.bond_types,
            case_ignores_edge_values(case),
            case_uses_ring_matches_ring_only(case),
            case_uses_exact_connections_match(case),
            case_respects_atom_aromaticity(case),
        ),
        second: build_typed_graph(
            case.graph2.n_atoms,
            &case.graph2.edges,
            &second_type_indices,
            &case.graph2.atom_is_aromatic,
            &case.graph2.bond_types,
            case_ignores_edge_values(case),
            case_uses_ring_matches_ring_only(case),
            case_uses_exact_connections_match(case),
            case_respects_atom_aromaticity(case),
        ),
        first_contexts: build_edge_contexts(&case.graph1),
        second_contexts: build_edge_contexts(&case.graph2),
    }
}

fn compute_case_bond_labels(
    graph: &TypedGraph,
    edge_map: &[(usize, usize)],
) -> Vec<GroundTruthBondLabel> {
    let node_types: Vec<GroundTruthNodeLabel> = graph.nodes().map(|symbol| symbol.node_type()).collect();
    edge_map
        .iter()
        .map(|&(src, dst)| {
            let left = node_types[src];
            let right = node_types[dst];
            let edge_value =
                geometric_traits::traits::Edges::matrix(graph.edges()).sparse_value_at(src, dst).unwrap();
            if left <= right { (left, edge_value, right) } else { (right, edge_value, left) }
        })
        .collect()
}

struct LabeledCaseProductDiagnostics {
    vertex_pairs: Vec<(usize, usize)>,
    first_bond_labels: Vec<GroundTruthBondLabel>,
    second_bond_labels: Vec<GroundTruthBondLabel>,
}

fn collect_prepared_labeled_case_product_diagnostics(
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

fn collect_labeled_case_product_diagnostics(
    case: &GroundTruthCase,
    use_edge_contexts: bool,
) -> LabeledCaseProductDiagnostics {
    let prepared = prepare_labeled_case(case);
    collect_prepared_labeled_case_product_diagnostics(case, &prepared, use_edge_contexts)
}

fn find_case<'a>(cases: &'a [GroundTruthCase], name: &str) -> &'a GroundTruthCase {
    cases.iter().find(|case| case.name == name).unwrap_or_else(|| panic!("missing case '{name}'"))
}

fn run_labeled_case(case: &GroundTruthCase) -> McesResult<usize> {
    run_labeled_case_with_options(case, true, false)
}

fn run_labeled_case_with_contexts(
    case: &GroundTruthCase,
    use_edge_contexts: bool,
) -> McesResult<usize> {
    run_labeled_case_with_options(case, use_edge_contexts, false)
}

fn run_labeled_case_with_options(
    case: &GroundTruthCase,
    use_edge_contexts: bool,
    use_partition_orientation_heuristic: bool,
) -> McesResult<usize> {
    let prepared = prepare_labeled_case(case);

    if use_edge_contexts && case_uses_complete_aromatic_rings(case) {
        if let (Some(graph1_contexts), Some(graph2_contexts)) =
            (prepared.first_contexts.as_ref(), prepared.second_contexts.as_ref())
        {
            return McesBuilder::new(&prepared.first, &prepared.second)
                .with_edge_contexts(graph1_contexts, graph2_contexts)
                .with_partition_orientation_heuristic(use_partition_orientation_heuristic)
                .compute_labeled();
        }
    }

    McesBuilder::new(&prepared.first, &prepared.second)
        .with_partition_orientation_heuristic(use_partition_orientation_heuristic)
        .compute_labeled()
}

fn assert_labeled_result_matches_ground_truth(
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

// ============================================================================
// Labeled MCES ground truth tests
// ============================================================================

#[test]
fn test_ground_truth_labeled_mces() {
    let cases = load_ground_truth();
    for case in &cases {
        if case.timed_out {
            continue;
        }

        let result = run_labeled_case(case);
        assert_labeled_result_matches_ground_truth(case, &result, "labeled MCES");
    }
}

#[test]
fn test_ground_truth_labeled_mces_with_partition_orientation_heuristic() {
    let cases = load_ground_truth();
    for case in &cases {
        if case.timed_out {
            continue;
        }

        let result = run_labeled_case_with_options(case, true, true);
        assert_labeled_result_matches_ground_truth(
            case,
            &result,
            "labeled MCES with orientation heuristic",
        );
    }
}

#[test]
fn test_ground_truth_fixture_metadata() {
    let cases = load_ground_truth();
    assert!(cases.len() >= 10, "expected at least 10 test cases");

    let very_small = cases.iter().find(|c| c.name == "very_small");
    assert!(very_small.is_some(), "missing 'very_small' case");
    let vs = very_small.unwrap();
    assert_eq!(vs.graph1.n_atoms, 5);
    assert_eq!(vs.graph2.n_atoms, 5);
    assert_eq!(vs.expected_bond_matches, 3);
    assert!((vs.expected_similarity - 0.6049).abs() < 0.001);

    let symmetrical_esters = find_case(&cases, "symmetrical_esters");
    assert_eq!(
        symmetrical_esters.graph1.aromatic_ring_contexts.len(),
        symmetrical_esters.graph1.edges.len()
    );
    assert!(
        symmetrical_esters
            .graph1
            .aromatic_ring_contexts
            .iter()
            .any(|contexts| !contexts.is_empty()),
        "expected aromatic-ring contexts for symmetrical_esters"
    );
}

#[test]
fn test_ground_truth_ignore_bond_orders_option() {
    let cases = load_ground_truth();
    let case = find_case(&cases, "ignore_bond_orders");
    let result = run_labeled_case(case);

    assert_labeled_result_matches_ground_truth(case, &result, "ignore bond orders");
}

#[test]
fn test_ground_truth_ring_matches_ring_only_option() {
    let cases = load_ground_truth();
    for case_name in ["ring_matches_ring", "single_fragment_1", "single_fragment_2"] {
        let case = find_case(&cases, case_name);
        let result = run_labeled_case(case);

        assert_labeled_result_matches_ground_truth(case, &result, "ring matches ring only");
    }
}

#[test]
fn test_ground_truth_exact_connections_option() {
    let cases = load_ground_truth();
    let case = find_case(&cases, "exact_connections_1");
    let result = run_labeled_case(case);

    assert_labeled_result_matches_ground_truth(case, &result, "exact connections match");
}

#[test]
fn test_ground_truth_atom_aromaticity_respect_option() {
    let cases = load_ground_truth();
    let case = find_case(&cases, "atom_aromaticity_respect");
    let result = run_labeled_case(case);

    assert_labeled_result_matches_ground_truth(case, &result, "atom aromaticity respect");
}

#[test]
fn test_ground_truth_bad_aromatics_contexts_change_pair_admission() {
    let cases = load_ground_truth();
    let case = find_case(&cases, "bad_aromatics_1a");

    let without_contexts = collect_labeled_case_product_diagnostics(case, false);
    let with_contexts = collect_labeled_case_product_diagnostics(case, true);

    assert!(with_contexts.vertex_pairs.len() < without_contexts.vertex_pairs.len());
}

#[test]
fn test_ground_truth_exact_connections_refines_node_labels() {
    let cases = load_ground_truth();
    let case = find_case(&cases, "exact_connections_1");
    let prepared = prepare_labeled_case(case);

    let first_labels: Vec<GroundTruthNodeLabel> =
        prepared.first.nodes().map(|node| node.node_type()).collect();
    let second_labels: Vec<GroundTruthNodeLabel> =
        prepared.second.nodes().map(|node| node.node_type()).collect();

    let find_refined_pair = |labels: &[GroundTruthNodeLabel]| {
        labels.iter().enumerate().find_map(|(left_index, left_label)| {
            labels.iter().enumerate().skip(left_index + 1).find_map(|(right_index, right_label)| {
                if left_label.atom_type == right_label.atom_type
                    && left_label.explicit_degree != right_label.explicit_degree
                {
                    Some((left_index, right_index))
                } else {
                    None
                }
            })
        })
    };

    let (first_left, first_right) =
        find_refined_pair(&first_labels).expect("expected a same-type/different-degree witness in graph1");
    let (second_left, second_right) =
        find_refined_pair(&second_labels).expect("expected a same-type/different-degree witness in graph2");

    assert_eq!(first_labels[first_left].atom_type, first_labels[first_right].atom_type);
    assert_ne!(first_labels[first_left].explicit_degree, first_labels[first_right].explicit_degree);
    assert_ne!(first_labels[first_left], first_labels[first_right]);

    assert_eq!(second_labels[second_left].atom_type, second_labels[second_right].atom_type);
    assert_ne!(
        second_labels[second_left].explicit_degree,
        second_labels[second_right].explicit_degree
    );
    assert_ne!(second_labels[second_left], second_labels[second_right]);
}

#[test]
fn test_ground_truth_atom_aromaticity_option_refines_node_labels() {
    let cases = load_ground_truth();
    let ignore_case = find_case(&cases, "atom_aromaticity_ignore");
    let respect_case = find_case(&cases, "atom_aromaticity_respect");
    let prepared_ignore = prepare_labeled_case(ignore_case);
    let prepared_respect = prepare_labeled_case(respect_case);

    let ignore_labels: Vec<GroundTruthNodeLabel> =
        prepared_ignore.first.nodes().map(|node| node.node_type()).collect();
    let respect_labels: Vec<GroundTruthNodeLabel> =
        prepared_respect.first.nodes().map(|node| node.node_type()).collect();

    let witness = ignore_case
        .graph1
        .atom_types
        .iter()
        .enumerate()
        .find_map(|(left_index, left_type)| {
            ignore_case
                .graph1
                .atom_types
                .iter()
                .enumerate()
                .skip(left_index + 1)
                .find_map(|(right_index, right_type)| {
                    if left_type == right_type
                        && ignore_case.graph1.atom_is_aromatic[left_index]
                            != ignore_case.graph1.atom_is_aromatic[right_index]
                    {
                        Some((left_index, right_index))
                    } else {
                        None
                    }
                })
        })
        .expect("expected same-type atoms with different aromaticity in atom_aromaticity fixture");

    let (left_index, right_index) = witness;

    assert_eq!(ignore_labels[left_index], ignore_labels[right_index]);
    assert_eq!(ignore_labels[left_index].is_aromatic, None);

    assert_ne!(respect_labels[left_index], respect_labels[right_index]);
    assert_eq!(
        respect_labels[left_index].is_aromatic,
        Some(ignore_case.graph1.atom_is_aromatic[left_index])
    );
    assert_eq!(
        respect_labels[right_index].is_aromatic,
        Some(ignore_case.graph1.atom_is_aromatic[right_index])
    );
}

#[test]
fn test_ground_truth_bad_aromatics_uses_shared_atom_labels() {
    let cases = load_ground_truth();
    let case = find_case(&cases, "bad_aromatics_1a");
    let prepared = prepare_labeled_case(case);
    let diagnostics = collect_prepared_labeled_case_product_diagnostics(case, &prepared, true);
    let result = run_labeled_case(case);

    let first_lg = prepared.first.labeled_line_graph();
    let second_lg = prepared.second.labeled_line_graph();
    let carbonyl_left =
        first_lg.edge_map().iter().position(|&(src, dst)| (src, dst) == (6, 7)).unwrap();
    let carbonyl_right =
        second_lg.edge_map().iter().position(|&(src, dst)| (src, dst) == (6, 7)).unwrap();

    assert_eq!(
        diagnostics.first_bond_labels[carbonyl_left],
        diagnostics.second_bond_labels[carbonyl_right]
    );
    assert!(diagnostics.vertex_pairs.contains(&(carbonyl_left, carbonyl_right)));
    assert_labeled_result_matches_ground_truth(case, &result, "bad aromatics shared labels");
}

#[test]
fn test_ground_truth_ring_matches_ring_only_blocks_mixed_ring_pairs() {
    let cases = load_ground_truth();
    let case = find_case(&cases, "ring_matches_ring");
    let diagnostics = collect_labeled_case_product_diagnostics(case, true);

    let mismatched_pair = diagnostics
        .first_bond_labels
        .iter()
        .enumerate()
        .find_map(|(i, first_label)| {
            diagnostics.second_bond_labels.iter().enumerate().find_map(|(j, second_label)| {
                let same_primary =
                    first_label.0 == second_label.0
                        && first_label.1.bond_order == second_label.1.bond_order
                        && first_label.2 == second_label.2;
                let different_ring_class = first_label.1.in_ring != second_label.1.in_ring;
                if same_primary && different_ring_class { Some((i, j)) } else { None }
            })
        })
        .expect("expected at least one ring-vs-chain bond pair with matching primary label");

    assert!(
        !diagnostics.vertex_pairs.contains(&mismatched_pair),
        "ringMatchesRingOnly should reject mixed ring/chain bond pairs",
    );
}

#[test]
fn test_ground_truth_unlabeled_identical_topology() {
    let cases = load_ground_truth();
    let ibo = cases.iter().find(|c| c.name == "ignore_bond_orders");
    assert!(ibo.is_some());
    let case = ibo.unwrap();

    let g1 = build_unlabeled_graph(case.graph1.n_atoms, &case.graph1.edges);
    let g2 = build_unlabeled_graph(case.graph2.n_atoms, &case.graph2.edges);

    let result = McesBuilder::new(&g1, &g2).compute_unlabeled();
    assert_eq!(result.matched_edges().len(), 3);
    let j = result.johnson_similarity();
    assert!((j - 1.0).abs() < 1e-6, "identical topology should give similarity 1.0, got {j}");
}

#[test]
fn test_ground_truth_delta_y_cases() {
    let cases = load_ground_truth();
    let dy = cases.iter().find(|c| c.name == "delta_y_small").unwrap();

    let g1 = build_unlabeled_graph(dy.graph1.n_atoms, &dy.graph1.edges);
    let g2 = build_unlabeled_graph(dy.graph2.n_atoms, &dy.graph2.edges);

    let without_dy = McesBuilder::new(&g1, &g2).with_delta_y(false).compute_unlabeled();
    let with_dy = McesBuilder::new(&g1, &g2).with_delta_y(true).compute_unlabeled();

    assert!(with_dy.matched_edges().len() <= without_dy.matched_edges().len());
}

#[test]
#[ignore = "timing harness for manual RDKit/Rust comparisons"]
fn print_labeled_case_timing() {
    let case_name = std::env::var("MCES_CASE").expect("set MCES_CASE to a ground-truth case name");
    let repeats: usize =
        std::env::var("MCES_REPEATS").ok().and_then(|value| value.parse().ok()).unwrap_or(1);
    let use_edge_contexts = std::env::var("MCES_USE_EDGE_CONTEXTS")
        .ok()
        .map(|value| !matches!(value.as_str(), "0" | "false" | "FALSE"))
        .unwrap_or(true);
    let use_partition_orientation_heuristic =
        std::env::var("MCES_USE_PARTITION_ORIENTATION_HEURISTIC")
            .ok()
            .map(|value| !matches!(value.as_str(), "0" | "false" | "FALSE"))
            .unwrap_or(false);

    let cases = load_ground_truth();
    let case = find_case(&cases, &case_name);

    for repeat in 0..repeats {
        let started = Instant::now();
        let result = run_labeled_case_with_options(
            case,
            use_edge_contexts,
            use_partition_orientation_heuristic,
        );
        let elapsed = started.elapsed();
        println!(
            "{} repeat {}: edge_contexts={} partition_orientation_heuristic={} elapsed={elapsed:?} matched_edges={} similarity={:.6}",
            case.name,
            repeat + 1,
            use_edge_contexts,
            use_partition_orientation_heuristic,
            result.matched_edges().len(),
            result.johnson_similarity(),
        );
        std::io::stdout().flush().unwrap();
    }
}

#[test]
#[ignore = "timing harness for manual RDKit/Rust comparisons"]
fn print_all_labeled_case_timings() {
    let cases = load_ground_truth();
    let use_edge_contexts = std::env::var("MCES_USE_EDGE_CONTEXTS")
        .ok()
        .map(|value| !matches!(value.as_str(), "0" | "false" | "FALSE"))
        .unwrap_or(true);
    let use_partition_orientation_heuristic =
        std::env::var("MCES_USE_PARTITION_ORIENTATION_HEURISTIC")
            .ok()
            .map(|value| !matches!(value.as_str(), "0" | "false" | "FALSE"))
            .unwrap_or(false);

    for case in &cases {
        let started = Instant::now();
        let result = run_labeled_case_with_options(
            case,
            use_edge_contexts,
            use_partition_orientation_heuristic,
        );
        let elapsed = started.elapsed();
        println!(
            "{}: edge_contexts={} partition_orientation_heuristic={} elapsed={elapsed:?} matched_edges={} similarity={:.6}",
            case.name,
            use_edge_contexts,
            use_partition_orientation_heuristic,
            result.matched_edges().len(),
            result.johnson_similarity(),
        );
        std::io::stdout().flush().unwrap();
    }
}

#[test]
#[ignore = "manual diagnostic harness for edge-context wiring"]
fn print_labeled_case_context_comparison() {
    let case_name = std::env::var("MCES_CASE").expect("set MCES_CASE to a ground-truth case name");
    let removed_limit: usize =
        std::env::var("MCES_REMOVED_LIMIT").ok().and_then(|value| value.parse().ok()).unwrap_or(20);

    let cases = load_ground_truth();
    let case = find_case(&cases, &case_name);
    let prepared = prepare_labeled_case(case);

    let started_without = Instant::now();
    let result_without = run_labeled_case_with_contexts(case, false);
    let elapsed_without = started_without.elapsed();
    let diagnostics_without =
        collect_prepared_labeled_case_product_diagnostics(case, &prepared, false);

    let started_with = Instant::now();
    let result_with = run_labeled_case_with_contexts(case, true);
    let elapsed_with = started_with.elapsed();
    let diagnostics_with = collect_prepared_labeled_case_product_diagnostics(case, &prepared, true);

    println!("case: {}", case.name);
    println!("options: {:?}", case.options);
    println!(
        "without contexts: elapsed={elapsed_without:?} product_vertices={} matched_edges={} similarity={:.6}",
        diagnostics_without.vertex_pairs.len(),
        result_without.matched_edges().len(),
        result_without.johnson_similarity(),
    );
    println!(
        "with contexts: elapsed={elapsed_with:?} product_vertices={} matched_edges={} similarity={:.6}",
        diagnostics_with.vertex_pairs.len(),
        result_with.matched_edges().len(),
        result_with.johnson_similarity(),
    );

    println!("graph1 non-empty context rows:");
    for (edge_index, contexts) in case.graph1.aromatic_ring_contexts.iter().enumerate() {
        if !contexts.is_empty() {
            println!("  {edge_index}: {:?}", contexts);
        }
    }
    println!("graph2 non-empty context rows:");
    for (edge_index, contexts) in case.graph2.aromatic_ring_contexts.iter().enumerate() {
        if !contexts.is_empty() {
            println!("  {edge_index}: {:?}", contexts);
        }
    }

    let removed_pairs: Vec<(usize, usize)> = diagnostics_without
        .vertex_pairs
        .iter()
        .copied()
        .filter(|pair| !diagnostics_with.vertex_pairs.contains(pair))
        .collect();
    println!("removed product vertices: {}", removed_pairs.len());
    for (i, j) in removed_pairs.iter().copied().take(removed_limit) {
        println!(
            "  ({i}, {j}) left_label={:?} right_label={:?} left_contexts={:?} right_contexts={:?}",
            diagnostics_without.first_bond_labels[i],
            diagnostics_without.second_bond_labels[j],
            case.graph1.aromatic_ring_contexts[i],
            case.graph2.aromatic_ring_contexts[j],
        );
    }
    if removed_pairs.len() > removed_limit {
        println!("  ... {} more removed pairs", removed_pairs.len() - removed_limit);
    }
}
