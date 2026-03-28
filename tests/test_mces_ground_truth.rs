//! Ground truth tests for the MCES pipeline.
//!
//! Loads test cases generated from RDKit's RASCAL test suite
//! (`tests/fixtures/mces_ground_truth.json.gz`) and validates our
//! labeled MCES results against them.
#![cfg(feature = "std")]

use std::{
    collections::{BTreeMap, BTreeSet, VecDeque},
    io::{Read as _, Write as _},
    time::Instant,
};

use geometric_traits::{
    impls::{
        BitSquareMatrix, CSR2D, EdgeContexts, SortedVec, SquareCSR2D, SymmetricCSR2D, ValuedCSR2D,
    },
    prelude::*,
    traits::{
        EdgesBuilder, MatrixMut, SparseMatrix2D, SparseMatrixMut, SquareMatrix, TypedNode,
        VocabularyBuilder,
    },
};
use rayon::prelude::*;

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

fn normalized_graph_bonds(graph: &GraphData) -> Vec<(usize, usize, u32)> {
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

fn heuristic_atom_refinement_colors(graph: &GraphData) -> Vec<usize> {
    let bonds = normalized_graph_bonds(graph);
    let degrees = explicit_degrees(graph.n_atoms, &bonds);
    let ring_membership = edge_ring_membership(graph.n_atoms, &bonds);

    let mut atom_type_ids = BTreeMap::<&str, usize>::new();
    for atom_type in &graph.atom_types {
        let next = atom_type_ids.len();
        atom_type_ids.entry(atom_type.as_str()).or_insert(next);
    }

    let base_labels: Vec<(usize, bool, u8)> = (0..graph.n_atoms)
        .map(|atom| {
            (
                atom_type_ids[graph.atom_types[atom].as_str()],
                graph.atom_is_aromatic[atom],
                degrees[atom],
            )
        })
        .collect();

    let mut adjacency = vec![Vec::<(usize, u32, bool)>::new(); graph.n_atoms];
    for ((src, dst, bond_type), in_ring) in bonds.into_iter().zip(ring_membership.into_iter()) {
        adjacency[src].push((dst, bond_type, in_ring));
        adjacency[dst].push((src, bond_type, in_ring));
    }

    let mut colors = vec![0usize; graph.n_atoms];
    let mut base_color_ids = BTreeMap::<(usize, bool, u8), usize>::new();
    for (atom, base_label) in base_labels.iter().copied().enumerate() {
        let next = base_color_ids.len();
        colors[atom] = *base_color_ids.entry(base_label).or_insert(next);
    }

    loop {
        let mut next_colors = vec![0usize; graph.n_atoms];
        let mut signatures = BTreeMap::<((usize, bool, u8), Vec<(usize, u32, bool)>), usize>::new();
        for atom in 0..graph.n_atoms {
            let mut neighborhood: Vec<(usize, u32, bool)> = adjacency[atom]
                .iter()
                .map(|&(neighbor, bond_type, in_ring)| (colors[neighbor], bond_type, in_ring))
                .collect();
            neighborhood.sort_unstable();
            let signature = (base_labels[atom], neighborhood);
            let next = signatures.len();
            next_colors[atom] = *signatures.entry(signature).or_insert(next);
        }
        if next_colors == colors {
            return colors;
        }
        colors = next_colors;
    }
}

fn heuristic_equivalent_bond_classes(graph: &GraphData, edge_map: &[(usize, usize)]) -> Vec<i32> {
    let colors = heuristic_atom_refinement_colors(graph);
    let bonds = normalized_graph_bonds(graph);
    let mut pair_counts = BTreeMap::<(usize, usize), usize>::new();
    for &(src, dst, _) in &bonds {
        let key = if colors[src] < colors[dst] {
            (colors[src], colors[dst])
        } else {
            (colors[dst], colors[src])
        };
        *pair_counts.entry(key).or_default() += 1;
    }

    let mut class_ids = BTreeMap::<(usize, usize), i32>::new();
    let mut next_class = 0i32;
    for (key, count) in pair_counts {
        if count > 1 {
            class_ids.insert(key, next_class);
            next_class += 1;
        }
    }

    edge_map
        .iter()
        .map(|&(src, dst)| {
            let key = if colors[src] < colors[dst] {
                (colors[src], colors[dst])
            } else {
                (colors[dst], colors[src])
            };
            class_ids.get(&key).copied().unwrap_or(-1)
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

#[derive(Clone, serde::Deserialize)]
struct GraphData {
    n_atoms: usize,
    edges: Vec<[usize; 2]>,
    atom_types: Vec<String>,
    bond_types: Vec<u32>,
    #[serde(default)]
    aromatic_ring_contexts: Vec<Vec<String>>,
    #[allow(dead_code)]
    atom_is_aromatic: Vec<bool>,
    #[serde(default)]
    atom_total_hs: Vec<u8>,
    #[serde(default)]
    bond_orientations: Vec<[usize; 2]>,
    #[serde(default)]
    bond_original_indices: Vec<usize>,
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
    #[allow(dead_code)]
    rdkit_elapsed_seconds: Option<f64>,
    timed_out: bool,
    #[allow(dead_code)]
    options: Option<serde_json::Value>,
}

static GROUND_TRUTH_GZ: &[u8] = include_bytes!("fixtures/mces_ground_truth.json.gz");
static MASSSPECGYM_GROUND_TRUTH_GZ: &[u8] =
    include_bytes!("fixtures/massspecgym_mces_default_100.json.gz");
static MASSSPECGYM_GROUND_TRUTH_1000_GZ: &[u8] =
    include_bytes!("fixtures/massspecgym_mces_default_1000.json.gz");
static MASSSPECGYM_GROUND_TRUTH_10000_GZ: &[u8] =
    include_bytes!("fixtures/massspecgym_mces_default_10000.json.gz");
static MASSSPECGYM_ALL_BEST_GROUND_TRUTH_GZ: &[u8] =
    include_bytes!("fixtures/massspecgym_mces_all_best_100.json.gz");
static MASSSPECGYM_ALL_BEST_GROUND_TRUTH_1000_GZ: &[u8] =
    include_bytes!("fixtures/massspecgym_mces_all_best_1000.json.gz");
static MASSSPECGYM_ALL_BEST_GROUND_TRUTH_10000_GZ: &[u8] =
    include_bytes!("fixtures/massspecgym_mces_all_best_10000.json.gz");
static MASSSPECGYM_ALL_BEST_HOLDOUTS_GZ: &[u8] =
    include_bytes!("fixtures/massspecgym_mces_all_best_holdouts.json.gz");

fn load_ground_truth_from_bytes(gz_bytes: &[u8]) -> Vec<GroundTruthCase> {
    let mut decoder = flate2::read::GzDecoder::new(gz_bytes);
    let mut json_str = String::new();
    decoder.read_to_string(&mut json_str).unwrap();
    let file: GroundTruthFile = serde_json::from_str(&json_str).unwrap();
    assert!(file.version >= 1);
    file.cases
}

fn load_ground_truth() -> Vec<GroundTruthCase> {
    load_ground_truth_from_bytes(GROUND_TRUTH_GZ)
}

fn load_massspecgym_ground_truth() -> Vec<GroundTruthCase> {
    load_ground_truth_from_bytes(MASSSPECGYM_GROUND_TRUTH_GZ)
}

fn load_massspecgym_ground_truth_1000() -> Vec<GroundTruthCase> {
    load_ground_truth_from_bytes(MASSSPECGYM_GROUND_TRUTH_1000_GZ)
}

fn load_massspecgym_ground_truth_10000() -> Vec<GroundTruthCase> {
    load_ground_truth_from_bytes(MASSSPECGYM_GROUND_TRUTH_10000_GZ)
}

fn load_massspecgym_all_best_ground_truth() -> Vec<GroundTruthCase> {
    load_ground_truth_from_bytes(MASSSPECGYM_ALL_BEST_GROUND_TRUTH_GZ)
}

fn load_massspecgym_all_best_ground_truth_1000() -> Vec<GroundTruthCase> {
    load_ground_truth_from_bytes(MASSSPECGYM_ALL_BEST_GROUND_TRUTH_1000_GZ)
}

fn load_massspecgym_all_best_ground_truth_10000() -> Vec<GroundTruthCase> {
    load_ground_truth_from_bytes(MASSSPECGYM_ALL_BEST_GROUND_TRUTH_10000_GZ)
}

fn load_massspecgym_all_best_holdouts() -> Vec<GroundTruthCase> {
    load_ground_truth_from_bytes(MASSSPECGYM_ALL_BEST_HOLDOUTS_GZ)
}

fn load_massspecgym_ground_truth_by_size(size: usize) -> Vec<GroundTruthCase> {
    match size {
        100 => load_massspecgym_ground_truth(),
        1000 => load_massspecgym_ground_truth_1000(),
        10000 => load_massspecgym_ground_truth_10000(),
        _ => panic!("unsupported MassSpecGym corpus size {size}"),
    }
}

fn evenly_spaced_case_indices(len: usize, samples: usize) -> Vec<usize> {
    assert!(len > 0, "cannot sample from an empty corpus");
    let samples = samples.min(len);
    if samples == len {
        return (0..len).collect();
    }

    (0..samples).map(|i| i * (len - 1) / (samples - 1)).collect()
}

fn first_parallel_mismatch<F>(cases: &[GroundTruthCase], run_case: F) -> Option<String>
where
    F: Fn(&GroundTruthCase) -> McesResult<usize> + Sync + Send,
{
    cases
        .par_iter()
        .try_for_each(|case| -> Result<(), String> {
            let result = run_case(case);
            match labeled_result_mismatch(case, &result) {
                Some(mismatch) => Err(mismatch),
                None => Ok(()),
            }
        })
        .err()
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

fn prepare_labeled_case_from_graph_data(
    case: &GroundTruthCase,
    first_graph: &GraphData,
    second_graph: &GraphData,
) -> PreparedLabeledCase {
    let (first_type_indices, second_type_indices) =
        atom_type_to_shared_indices(&first_graph.atom_types, &second_graph.atom_types);

    PreparedLabeledCase {
        first: build_typed_graph(
            first_graph.n_atoms,
            &first_graph.edges,
            &first_type_indices,
            &first_graph.atom_is_aromatic,
            &first_graph.bond_types,
            case_ignores_edge_values(case),
            case_uses_ring_matches_ring_only(case),
            case_uses_exact_connections_match(case),
            case_respects_atom_aromaticity(case),
        ),
        second: build_typed_graph(
            second_graph.n_atoms,
            &second_graph.edges,
            &second_type_indices,
            &second_graph.atom_is_aromatic,
            &second_graph.bond_types,
            case_ignores_edge_values(case),
            case_uses_ring_matches_ring_only(case),
            case_uses_exact_connections_match(case),
            case_respects_atom_aromaticity(case),
        ),
        first_contexts: build_edge_contexts(first_graph),
        second_contexts: build_edge_contexts(second_graph),
    }
}

fn prepare_labeled_case(case: &GroundTruthCase) -> PreparedLabeledCase {
    prepare_labeled_case_from_graph_data(case, &case.graph1, &case.graph2)
}

fn compute_case_bond_labels(
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

fn intern_case_bond_labels(
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

struct LabeledCaseProductDiagnostics {
    matrix: BitSquareMatrix,
    vertex_pairs: Vec<(usize, usize)>,
    first_edge_map: Vec<(usize, usize)>,
    second_edge_map: Vec<(usize, usize)>,
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
        matrix: product.matrix().clone(),
        vertex_pairs: product.vertex_pairs().to_vec(),
        first_edge_map: lg1.edge_map().to_vec(),
        second_edge_map: lg2.edge_map().to_vec(),
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct FixtureBondRecord {
    edge: (usize, usize),
    bond_type: u32,
    canonical_index: usize,
    original_index: usize,
}

fn fixture_bond_records(graph: &GraphData) -> Vec<FixtureBondRecord> {
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

fn fixture_edge_map(graph: &GraphData) -> Vec<(usize, usize)> {
    fixture_bond_records(graph).into_iter().map(|record| record.edge).collect()
}

fn fixture_context_rows(graph: &GraphData) -> Vec<Vec<String>> {
    if graph.aromatic_ring_contexts.is_empty() {
        return Vec::new();
    }
    fixture_bond_records(graph)
        .into_iter()
        .map(|record| graph.aromatic_ring_contexts[record.canonical_index].clone())
        .collect()
}

fn graph_node_labels(
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

fn fixture_order_bond_labels(
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

fn fixture_order_line_graph_labels(
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

fn collect_fixture_order_product_diagnostics(
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

fn find_case<'a>(cases: &'a [GroundTruthCase], name: &str) -> &'a GroundTruthCase {
    cases.iter().find(|case| case.name == name).unwrap_or_else(|| panic!("missing case '{name}'"))
}

fn reverse_graph_bond_payload(graph: &GraphData) -> GraphData {
    let mut reversed = graph.clone();
    reversed.edges.reverse();
    reversed.bond_types.reverse();
    reversed.aromatic_ring_contexts.reverse();
    reversed.bond_orientations.reverse();
    reversed
}

fn permute_product(
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

fn rank_partitioned_cliques(
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

fn info_johnson_similarity(case: &GroundTruthCase, info: &EagerCliqueInfo<usize>) -> f64 {
    geometric_traits::traits::algorithms::johnson_similarity(
        info.matched_edges().len(),
        info.vertex_matches().len(),
        case.graph1.n_atoms,
        case.graph1.edges.len(),
        case.graph2.n_atoms,
        case.graph2.edges.len(),
    )
}

fn product_order_identity(vertex_pairs: &[(usize, usize)]) -> Vec<usize> {
    (0..vertex_pairs.len()).collect()
}

fn product_order_reverse(vertex_pairs: &[(usize, usize)]) -> Vec<usize> {
    let mut order = product_order_identity(vertex_pairs);
    order.reverse();
    order
}

fn product_order_second_then_first(vertex_pairs: &[(usize, usize)]) -> Vec<usize> {
    let mut order = product_order_identity(vertex_pairs);
    order.sort_unstable_by_key(|&index| {
        let (first, second) = vertex_pairs[index];
        (second, first, index)
    });
    order
}

fn product_order_reverse_within_first_buckets(vertex_pairs: &[(usize, usize)]) -> Vec<usize> {
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

fn product_order_target_last(
    vertex_pairs: &[(usize, usize)],
    target_clique: &[usize],
) -> Vec<usize> {
    let target: std::collections::BTreeSet<usize> = target_clique.iter().copied().collect();
    let mut order = Vec::with_capacity(vertex_pairs.len());
    order.extend((0..vertex_pairs.len()).filter(|index| !target.contains(index)));
    order.extend(target_clique.iter().copied());
    order
}

fn fixture_edge_rank_map(graph: &GraphData) -> BTreeMap<(usize, usize), usize> {
    let mut ranks = BTreeMap::new();
    for record in fixture_bond_records(graph) {
        ranks.entry(record.edge).or_insert(record.original_index);
    }
    ranks
}

fn product_order_fixture_edge_indices(
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

fn product_order_fixture_edge_indices_second_then_first(
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

fn product_order_rdkit_raw_pair_order(
    case: &GroundTruthCase,
    diagnostics: &LabeledCaseProductDiagnostics,
) -> Vec<usize> {
    if case.graph1.n_atoms <= case.graph2.n_atoms {
        product_order_fixture_edge_indices(case, diagnostics)
    } else {
        product_order_fixture_edge_indices_second_then_first(case, diagnostics)
    }
}

fn configure_rdkit_raw_pair_order<'g, PF, XC, EC, D, R>(
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

fn permuted_partitioned_infos(
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

fn run_labeled_case(case: &GroundTruthCase) -> McesResult<usize> {
    run_labeled_case_with_default_orientation(case, true, McesSearchMode::PartialEnumeration)
}

fn run_labeled_case_with_contexts(
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

fn run_labeled_case_with_options(
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

fn run_labeled_case_with_default_orientation(
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

fn run_labeled_case_with_search_mode(
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

fn labeled_result_mismatch(case: &GroundTruthCase, result: &McesResult<usize>) -> Option<String> {
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

fn labeled_info_mismatch(case: &GroundTruthCase, info: &EagerCliqueInfo<usize>) -> Option<String> {
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

fn inferred_atom_count_from_similarity(
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

fn canonical_edge(edge: [usize; 2]) -> (usize, usize) {
    if edge[0] <= edge[1] { (edge[0], edge[1]) } else { (edge[1], edge[0]) }
}

fn normalized_graph_edges(graph: &GraphData) -> Vec<(usize, usize, u32)> {
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

fn graph_ring_membership_by_edge(graph: &GraphData) -> BTreeMap<(usize, usize), bool> {
    let normalized_edges = normalized_graph_edges(graph);
    normalized_edges
        .iter()
        .copied()
        .zip(edge_ring_membership(graph.n_atoms, &normalized_edges))
        .map(|((src, dst, _), in_ring)| ((src, dst), in_ring))
        .collect()
}

fn graph_distance_matrix(graph: &GraphData) -> Vec<Vec<usize>> {
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

fn non_empty_context_row_count(graph: &GraphData) -> usize {
    graph.aromatic_ring_contexts.iter().filter(|contexts| !contexts.is_empty()).count()
}

fn matched_edge_fragment_stats(
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

fn clique_degree_sequence(num_vertices: usize, edges: &[(usize, usize)]) -> Vec<usize> {
    let mut counts = vec![0usize; num_vertices];
    for &(src, dst) in edges {
        counts[src] += 1;
        counts[dst] += 1;
    }
    let mut seq: Vec<usize> = counts.into_iter().filter(|&degree| degree > 0).collect();
    seq.sort_unstable();
    seq
}

fn clique_has_delta_y_from_product(
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

fn other_endpoint(edge: (usize, usize), shared: usize) -> usize {
    if edge.0 == shared { edge.1 } else { edge.0 }
}

fn graph_edge_index(graph: &GraphData, edge: (usize, usize)) -> usize {
    let canonical = canonical_edge([edge.0, edge.1]);
    graph
        .edges
        .iter()
        .position(|&candidate| canonical_edge(candidate) == canonical)
        .unwrap_or_else(|| panic!("missing graph edge for {:?}", edge))
}

fn infer_vertex_matches_rdkit_style(
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
struct CliqueScorecard {
    matched_bonds: usize,
    matched_atoms: usize,
    fragment_count: usize,
    largest_fragment_size: usize,
    ring_non_ring_bond_score: usize,
    atom_h_score: usize,
    max_delta_atom_atom_dist: usize,
}

fn score_clique(
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

fn scorecards_for_infos(
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

fn ranked_indices_by_scorecards<F>(scorecards: &[CliqueScorecard], compare: F) -> Vec<usize>
where
    F: Fn(&CliqueScorecard, &CliqueScorecard) -> core::cmp::Ordering,
{
    let mut indices: Vec<usize> = (0..scorecards.len()).collect();
    indices.sort_unstable_by(|&left, &right| {
        compare(&scorecards[left], &scorecards[right]).then_with(|| left.cmp(&right))
    });
    indices
}

fn first_missing_rdkit_compare(
    left: &CliqueScorecard,
    right: &CliqueScorecard,
) -> core::cmp::Ordering {
    left.fragment_count
        .cmp(&right.fragment_count)
        .then_with(|| right.largest_fragment_size.cmp(&left.largest_fragment_size))
        .then_with(|| left.ring_non_ring_bond_score.cmp(&right.ring_non_ring_bond_score))
}

fn reordered_all_best_indices_by_first_missing_ranker(
    case: &GroundTruthCase,
    result: &McesResult<usize>,
) -> Vec<usize> {
    let scorecards = scorecards_for_infos(case, result.all_cliques());
    ranked_indices_by_scorecards(&scorecards, first_missing_rdkit_compare)
}

fn approx_rdkit_compare(left: &CliqueScorecard, right: &CliqueScorecard) -> core::cmp::Ordering {
    right
        .matched_bonds
        .cmp(&left.matched_bonds)
        .then_with(|| left.fragment_count.cmp(&right.fragment_count))
        .then_with(|| right.largest_fragment_size.cmp(&left.largest_fragment_size))
        .then_with(|| left.ring_non_ring_bond_score.cmp(&right.ring_non_ring_bond_score))
        .then_with(|| left.atom_h_score.cmp(&right.atom_h_score))
        .then_with(|| left.max_delta_atom_atom_dist.cmp(&right.max_delta_atom_atom_dist))
}

fn selected_clique_edge_indices(
    result: &McesResult<usize>,
    diagnostics: &LabeledCaseProductDiagnostics,
) -> Vec<(usize, usize)> {
    result
        .all_cliques()
        .first()
        .map(|info| info.clique().iter().map(|&k| diagnostics.vertex_pairs[k]).collect())
        .unwrap_or_default()
}

fn edge_components(
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

fn graph_edge_orientation(graph: &GraphData, edge_index: usize) -> [usize; 2] {
    graph.bond_orientations.get(edge_index).copied().unwrap_or(graph.edges[edge_index])
}

fn graph_atom_total_hs(graph: &GraphData, atom_index: usize) -> Option<u8> {
    graph.atom_total_hs.get(atom_index).copied()
}

fn rdkit_preferred_isolated_mapping(
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
#[ignore = "large-corpus parity check against MassSpecGym-derived RDKit fixtures"]
fn test_massspecgym_ground_truth_labeled_mces() {
    let cases = load_massspecgym_ground_truth();
    assert_eq!(cases.len(), 100, "expected exactly 100 large-corpus cases");
    assert!(
        cases.iter().all(|case| !case.timed_out),
        "large-corpus fixture must exclude timed-out RDKit pairs",
    );

    let mismatch = first_parallel_mismatch(&cases, run_labeled_case);

    println!(
        "checked {} MassSpecGym default-config cases; mismatches={}",
        cases.len(),
        usize::from(mismatch.is_some())
    );

    assert!(
        mismatch.is_none(),
        "found mismatch in MassSpecGym default-config corpus:\n{}",
        mismatch.unwrap()
    );
}

#[test]
#[ignore = "1K-corpus parity check against fast 1-second MassSpecGym-derived RDKit default fixtures"]
fn test_massspecgym_ground_truth_labeled_mces_1000() {
    let cases = load_massspecgym_ground_truth_1000();
    assert_eq!(cases.len(), 1000, "expected exactly 1000 fast large-corpus cases");
    assert!(
        cases.iter().all(|case| !case.timed_out),
        "fast 1K default fixture must exclude timed-out RDKit pairs",
    );

    let mismatch = first_parallel_mismatch(&cases, run_labeled_case);

    println!(
        "checked {} MassSpecGym default-config 1K fast cases; mismatches={}",
        cases.len(),
        usize::from(mismatch.is_some())
    );

    assert!(
        mismatch.is_none(),
        "found mismatch in MassSpecGym default-config 1K fast corpus:\n{}",
        mismatch.unwrap()
    );
}

#[test]
#[ignore = "10K-corpus parity check against fast 1-second MassSpecGym-derived RDKit default fixtures"]
fn test_massspecgym_ground_truth_labeled_mces_10000() {
    let cases = load_massspecgym_ground_truth_10000();
    assert_eq!(cases.len(), 10000, "expected exactly 10000 fast large-corpus cases");
    assert!(
        cases.iter().all(|case| !case.timed_out),
        "fast 10K default fixture must exclude timed-out RDKit pairs",
    );

    let mismatch = first_parallel_mismatch(&cases, run_labeled_case);

    println!(
        "checked {} MassSpecGym default-config 10K fast cases; mismatches={}",
        cases.len(),
        usize::from(mismatch.is_some())
    );

    assert!(
        mismatch.is_none(),
        "found mismatch in MassSpecGym default-config 10K fast corpus:\n{}",
        mismatch.unwrap()
    );
}

#[test]
fn test_massspecgym_ground_truth_labeled_mces_smoke() {
    let cases = load_massspecgym_ground_truth();
    let indices = evenly_spaced_case_indices(cases.len(), 25);
    let mismatch = indices
        .par_iter()
        .try_for_each(|&index| -> Result<(), String> {
            let case = &cases[index];
            let result = run_labeled_case(case);
            match labeled_result_mismatch(case, &result) {
                Some(mismatch) => Err(format!("[{index}] {mismatch}")),
                None => Ok(()),
            }
        })
        .err();

    assert!(
        mismatch.is_none(),
        "found mismatch in MassSpecGym default-config smoke sample:\n{}",
        mismatch.unwrap()
    );
}

#[test]
#[ignore = "large-corpus parity check against MassSpecGym-derived RDKit allBestMCESs=true fixtures"]
fn test_massspecgym_ground_truth_labeled_mces_all_best() {
    let cases = load_massspecgym_all_best_ground_truth();
    assert!(
        cases.iter().all(|case| !case.timed_out),
        "fast allBest fixture must exclude timed-out RDKit pairs",
    );
    assert!(
        !cases.is_empty(),
        "fast allBest fixture must retain at least one non-timeout RDKit pair",
    );

    let mismatch = first_parallel_mismatch(&cases, |case| {
        run_labeled_case_with_search_mode(case, true, false, McesSearchMode::AllBest)
    });

    println!(
        "checked {} MassSpecGym allBestMCESs=true cases in AllBest mode; mismatches={}",
        cases.len(),
        usize::from(mismatch.is_some())
    );

    assert!(
        mismatch.is_none(),
        "found mismatch in MassSpecGym allBestMCESs=true corpus with AllBest:\n{}",
        mismatch.unwrap()
    );
}

#[test]
#[ignore = "1K-corpus parity check against fast 1-second MassSpecGym-derived RDKit allBestMCESs=true fixtures"]
fn test_massspecgym_ground_truth_labeled_mces_all_best_1000() {
    let cases = load_massspecgym_all_best_ground_truth_1000();
    assert!(
        !cases.is_empty(),
        "fast 1K allBest fixture must retain at least one non-timeout RDKit pair",
    );
    assert!(
        cases.iter().all(|case| !case.timed_out),
        "fast 1K allBest fixture must exclude timed-out RDKit pairs",
    );

    let mismatch = first_parallel_mismatch(&cases, |case| {
        run_labeled_case_with_search_mode(case, true, false, McesSearchMode::AllBest)
    });

    println!(
        "checked {} MassSpecGym allBestMCESs=true 1K fast cases in AllBest mode; mismatches={}",
        cases.len(),
        usize::from(mismatch.is_some())
    );

    assert!(
        mismatch.is_none(),
        "found mismatch in MassSpecGym allBestMCESs=true 1K fast corpus with AllBest:\n{}",
        mismatch.unwrap()
    );
}

#[test]
#[ignore = "10K-corpus parity check against fast 1-second MassSpecGym-derived RDKit allBestMCESs=true fixtures"]
fn test_massspecgym_ground_truth_labeled_mces_all_best_10000() {
    let cases = load_massspecgym_all_best_ground_truth_10000();
    assert!(
        !cases.is_empty(),
        "fast 10K allBest fixture must retain at least one non-timeout RDKit pair",
    );
    assert!(
        cases.iter().all(|case| !case.timed_out),
        "fast 10K allBest fixture must exclude timed-out RDKit pairs",
    );

    let mismatch = first_parallel_mismatch(&cases, |case| {
        run_labeled_case_with_search_mode(case, true, false, McesSearchMode::AllBest)
    });

    println!(
        "checked {} MassSpecGym allBestMCESs=true 10K fast cases in AllBest mode; mismatches={}",
        cases.len(),
        usize::from(mismatch.is_some())
    );

    assert!(
        mismatch.is_none(),
        "found mismatch in MassSpecGym allBestMCESs=true 10K fast corpus with AllBest:\n{}",
        mismatch.unwrap()
    );
}

#[test]
fn test_massspecgym_ground_truth_labeled_mces_all_best_smoke() {
    let cases = load_massspecgym_all_best_ground_truth();
    assert!(
        cases.iter().all(|case| !case.timed_out),
        "fast allBest fixture must exclude timed-out RDKit pairs",
    );
    let indices = evenly_spaced_case_indices(cases.len(), 25);
    let mismatch = indices
        .par_iter()
        .try_for_each(|&index| -> Result<(), String> {
            let case = &cases[index];
            let result =
                run_labeled_case_with_search_mode(case, true, false, McesSearchMode::AllBest);
            match labeled_result_mismatch(case, &result) {
                Some(mismatch) => Err(format!("[{}:{}] {}", index, case.name, mismatch)),
                None => Ok(()),
            }
        })
        .err();

    assert!(
        mismatch.is_none(),
        "found mismatch in MassSpecGym allBestMCESs=true smoke sample:\n{}",
        mismatch.unwrap()
    );
}

#[test]
#[ignore = "focused probe showing where AllBest differs from the RDKit default-path fixture"]
fn test_massspecgym_ground_truth_labeled_mces_all_best_known_differences_from_default_fixture() {
    let mismatch_cases = [
        "massspecgym_default_0006",
        "massspecgym_default_0010",
        "massspecgym_default_0018",
        "massspecgym_default_0029",
        "massspecgym_default_0038",
        "massspecgym_default_0054",
        "massspecgym_default_0086",
        "massspecgym_default_0092",
    ];

    let cases = load_massspecgym_ground_truth();
    let mut mismatches = Vec::new();

    for case_name in mismatch_cases {
        let case = find_case(&cases, case_name);
        let result = run_labeled_case_with_search_mode(case, true, false, McesSearchMode::AllBest);
        println!(
            "{}: bonds={} expected_bonds={} similarity={:.6} expected_similarity={:.6}",
            case.name,
            result.matched_edges().len(),
            case.expected_bond_matches,
            result.johnson_similarity(),
            case.expected_similarity,
        );
        if let Some(mismatch) = labeled_result_mismatch(case, &result) {
            mismatches.push(mismatch);
        }
    }

    println!(
        "checked {} known default-fixture difference cases in AllBest mode; mismatches={}",
        8,
        mismatches.len()
    );

    assert!(
        mismatches.is_empty(),
        "found {} differences from the RDKit default-path fixture among known AllBest cases:\n{}",
        mismatches.len(),
        mismatches.join("\n")
    );
}

#[test]
#[ignore = "focused parity check against RDKit allBestMCESs=true for the known holdouts"]
fn test_massspecgym_ground_truth_labeled_mces_all_best_rdkit_holdouts() {
    let cases = load_massspecgym_all_best_holdouts();
    assert_eq!(cases.len(), 3, "expected exactly 3 focused allBest holdout cases");

    for case in &cases {
        let result = run_labeled_case_with_search_mode(case, true, false, McesSearchMode::AllBest);
        assert_labeled_result_matches_ground_truth(case, &result, "AllBest vs RDKit allBestMCESs");
    }
}

#[test]
#[ignore = "focused PartialEnumeration regression for a known MassSpecGym tie-retention case"]
fn test_massspecgym_ground_truth_labeled_mces_partial_enumeration_case_0006() {
    let cases = load_massspecgym_ground_truth();
    let case = find_case(&cases, "massspecgym_default_0006");
    let result =
        run_labeled_case_with_default_orientation(case, true, McesSearchMode::PartialEnumeration);
    assert_labeled_result_matches_ground_truth(case, &result, "PartialEnumeration");
}

#[test]
#[ignore = "focused PartialEnumeration regressions for smaller-first partition-side cases"]
fn test_massspecgym_ground_truth_labeled_mces_partial_enumeration_smaller_first_cases() {
    let cases = load_massspecgym_ground_truth();
    for case_name in
        ["massspecgym_default_0018", "massspecgym_default_0029", "massspecgym_default_0054"]
    {
        let case = find_case(&cases, case_name);
        let result = run_labeled_case(case);
        assert_labeled_result_matches_ground_truth(case, &result, "PartialEnumeration");
    }
}

#[test]
#[ignore = "focused red regression for a remaining 1K default mismatch"]
fn test_massspecgym_ground_truth_labeled_mces_1000_case_0719() {
    let cases = load_massspecgym_ground_truth_1000();
    let case = find_case(&cases, "massspecgym_default_0719");
    let result = run_labeled_case(case);
    assert_labeled_result_matches_ground_truth(case, &result, "PartialEnumeration");
}

#[test]
#[ignore = "focused red regression for a remaining 1K default mismatch"]
fn test_massspecgym_ground_truth_labeled_mces_1000_case_0585() {
    let cases = load_massspecgym_ground_truth_1000();
    let case = find_case(&cases, "massspecgym_default_0585");
    let result = run_labeled_case(case);
    assert_labeled_result_matches_ground_truth(case, &result, "PartialEnumeration");
}

#[test]
#[ignore = "focused red regression for a remaining 1K default mismatch"]
fn test_massspecgym_ground_truth_labeled_mces_1000_case_0702() {
    let cases = load_massspecgym_ground_truth_1000();
    let case = find_case(&cases, "massspecgym_default_0702");
    let result = run_labeled_case(case);
    assert_labeled_result_matches_ground_truth(case, &result, "PartialEnumeration");
}

#[test]
#[ignore = "focused red regression for a remaining 1K default mismatch"]
fn test_massspecgym_ground_truth_labeled_mces_1000_case_0911() {
    let cases = load_massspecgym_ground_truth_1000();
    let case = find_case(&cases, "massspecgym_default_0911");
    let result = run_labeled_case(case);
    assert_labeled_result_matches_ground_truth(case, &result, "PartialEnumeration");
}

#[test]
#[ignore = "focused red regression for a remaining 10K allBest mismatch"]
fn test_massspecgym_ground_truth_labeled_mces_all_best_10000_case_5585() {
    let cases = load_massspecgym_all_best_ground_truth_10000();
    let case = find_case(&cases, "massspecgym_default_5585");
    let result = run_labeled_case_with_search_mode(case, true, false, McesSearchMode::AllBest);
    assert_labeled_result_matches_ground_truth(case, &result, "AllBest");
}

#[test]
#[ignore = "focused proof that the first missing RDKit ranker flips the 5585 allBest top clique"]
fn test_massspecgym_ground_truth_labeled_mces_all_best_10000_case_5585_first_missing_ranker() {
    let cases = load_massspecgym_all_best_ground_truth_10000();
    let case = find_case(&cases, "massspecgym_default_5585");
    let result = run_labeled_case_with_search_mode(case, true, false, McesSearchMode::AllBest);
    let expected_index = result
        .all_cliques()
        .iter()
        .position(|info| labeled_info_mismatch(case, info).is_none())
        .expect("expected RDKit allBest clique must be present in the Rust AllBest set");
    let reordered = reordered_all_best_indices_by_first_missing_ranker(case, &result);

    assert_ne!(
        expected_index, 0,
        "raw AllBest should still disagree on 5585 before the test-only ranker is applied"
    );
    assert_eq!(
        reordered.first().copied(),
        Some(expected_index),
        "the first missing RDKit ranker should move the RDKit allBest clique to the top"
    );
}

#[test]
#[ignore = "exploratory parity check for the first missing RDKit allBest ranker over the fast 10K corpus"]
fn test_massspecgym_ground_truth_labeled_mces_all_best_10000_first_missing_ranker() {
    let cases = load_massspecgym_all_best_ground_truth_10000();
    assert!(
        !cases.is_empty(),
        "fast 10K allBest fixture must retain at least one non-timeout RDKit pair",
    );
    assert!(
        cases.iter().all(|case| !case.timed_out),
        "fast 10K allBest fixture must exclude timed-out RDKit pairs",
    );

    let mismatch = cases
        .par_iter()
        .try_for_each(|case| -> Result<(), String> {
            let result =
                run_labeled_case_with_search_mode(case, true, false, McesSearchMode::AllBest);
            let reordered = reordered_all_best_indices_by_first_missing_ranker(case, &result);
            let top = reordered
                .first()
                .map(|&index| &result.all_cliques()[index])
                .expect("AllBest should retain at least one clique");
            match labeled_info_mismatch(case, top) {
                Some(mismatch) => Err(mismatch),
                None => Ok(()),
            }
        })
        .err();

    println!(
        "checked {} MassSpecGym allBestMCESs=true 10K fast cases with the first missing RDKit ranker; mismatches={}",
        cases.len(),
        usize::from(mismatch.is_some())
    );

    assert!(
        mismatch.is_none(),
        "found mismatch in MassSpecGym allBestMCESs=true 10K fast corpus with the first missing RDKit ranker:\n{}",
        mismatch.unwrap()
    );
}

#[test]
#[ignore = "focused diagnostic for the remaining 10K allBest mismatch 5585"]
fn print_massspecgym_case_10000_all_best_5585() {
    let default_cases = load_massspecgym_ground_truth_10000();
    let all_best_cases = load_massspecgym_all_best_ground_truth_10000();
    let default_case = find_case(&default_cases, "massspecgym_default_5585");
    let all_best_case = find_case(&all_best_cases, "massspecgym_default_5585");

    let partial = run_labeled_case_with_search_mode(
        all_best_case,
        true,
        false,
        McesSearchMode::PartialEnumeration,
    );
    let all_best =
        run_labeled_case_with_search_mode(all_best_case, true, false, McesSearchMode::AllBest);

    let matches_default = |info: &EagerCliqueInfo<usize>| {
        info.matched_edges().len() == default_case.expected_bond_matches
            && info.vertex_matches().len() == default_case.expected_atom_matches
            && (info_johnson_similarity(default_case, info) - default_case.expected_similarity)
                .abs()
                <= 1e-6
    };
    let matches_all_best = |info: &EagerCliqueInfo<usize>| {
        info.matched_edges().len() == all_best_case.expected_bond_matches
            && info.vertex_matches().len() == all_best_case.expected_atom_matches
            && (info_johnson_similarity(all_best_case, info) - all_best_case.expected_similarity)
                .abs()
                <= 1e-6
    };

    let partial_default_index = partial.all_cliques().iter().position(matches_default);
    let partial_all_best_index = partial.all_cliques().iter().position(matches_all_best);
    let all_best_default_index = all_best.all_cliques().iter().position(matches_default);
    let all_best_all_best_index = all_best.all_cliques().iter().position(matches_all_best);

    println!("case: {}", all_best_case.name);
    println!(
        "default_fixture bonds={} atoms={} similarity={:.6}",
        default_case.expected_bond_matches,
        default_case.expected_atom_matches,
        default_case.expected_similarity,
    );
    println!(
        "all_best_fixture bonds={} atoms={} similarity={:.6}",
        all_best_case.expected_bond_matches,
        all_best_case.expected_atom_matches,
        all_best_case.expected_similarity,
    );
    println!(
        "partial: retained={} top_bonds={} top_atoms={} top_similarity={:.6} default_index={partial_default_index:?} all_best_index={partial_all_best_index:?}",
        partial.all_cliques().len(),
        partial.matched_edges().len(),
        partial.vertex_matches().len(),
        partial.johnson_similarity(),
    );
    println!(
        "all_best: retained={} top_bonds={} top_atoms={} top_similarity={:.6} default_index={all_best_default_index:?} all_best_index={all_best_all_best_index:?}",
        all_best.all_cliques().len(),
        all_best.matched_edges().len(),
        all_best.vertex_matches().len(),
        all_best.johnson_similarity(),
    );

    for (index, info) in all_best.all_cliques().iter().take(12).enumerate() {
        println!(
            "  #{index}: bonds={} atoms={} similarity={:.6} fragments={} largest_fragment_atoms={}",
            info.matched_edges().len(),
            info.vertex_matches().len(),
            info_johnson_similarity(all_best_case, info),
            info.fragment_count(),
            info.largest_fragment_atom_count(),
        );
    }
}

#[test]
#[ignore = "focused scorecard diagnostic for the remaining 10K allBest mismatch 5585"]
fn print_massspecgym_case_10000_all_best_5585_scorecards() {
    let cases = load_massspecgym_all_best_ground_truth_10000();
    let case = find_case(&cases, "massspecgym_default_5585");
    let result = run_labeled_case_with_search_mode(case, true, false, McesSearchMode::AllBest);
    let scorecards = scorecards_for_infos(case, result.all_cliques());

    let expected_index = scorecards.iter().position(|scorecard| {
        scorecard.matched_bonds == case.expected_bond_matches
            && scorecard.matched_atoms == case.expected_atom_matches
            && (geometric_traits::traits::algorithms::johnson_similarity(
                scorecard.matched_bonds,
                scorecard.matched_atoms,
                case.graph1.n_atoms,
                case.graph1.edges.len(),
                case.graph2.n_atoms,
                case.graph2.edges.len(),
            ) - case.expected_similarity)
                .abs()
                <= 1e-6
    });

    let mut approx_rdkit_order: Vec<usize> = (0..scorecards.len()).collect();
    approx_rdkit_order.sort_unstable_by(|&left, &right| {
        approx_rdkit_compare(&scorecards[left], &scorecards[right]).then_with(|| left.cmp(&right))
    });
    let first_missing_order =
        ranked_indices_by_scorecards(&scorecards, first_missing_rdkit_compare);

    println!("case: {}", case.name);
    println!(
        "fixture all_best bonds={} atoms={} similarity={:.6} expected_index={expected_index:?}",
        case.expected_bond_matches, case.expected_atom_matches, case.expected_similarity,
    );
    println!("rust_order:");
    for (index, scorecard) in scorecards.iter().take(12).enumerate() {
        println!(
            "  #{index}: bonds={} atoms={} fragments={} largest_fragment={} ring_non_ring={} atom_h={} max_delta={}",
            scorecard.matched_bonds,
            scorecard.matched_atoms,
            scorecard.fragment_count,
            scorecard.largest_fragment_size,
            scorecard.ring_non_ring_bond_score,
            scorecard.atom_h_score,
            scorecard.max_delta_atom_atom_dist,
        );
    }
    println!("first_missing_ranker_order:");
    for &index in first_missing_order.iter().take(12) {
        let scorecard = &scorecards[index];
        println!(
            "  #{index}: bonds={} atoms={} fragments={} largest_fragment={} ring_non_ring={} atom_h={} max_delta={}",
            scorecard.matched_bonds,
            scorecard.matched_atoms,
            scorecard.fragment_count,
            scorecard.largest_fragment_size,
            scorecard.ring_non_ring_bond_score,
            scorecard.atom_h_score,
            scorecard.max_delta_atom_atom_dist,
        );
    }
    println!("approx_rdkit_order:");
    for &index in approx_rdkit_order.iter().take(12) {
        let scorecard = &scorecards[index];
        println!(
            "  #{index}: bonds={} atoms={} fragments={} largest_fragment={} ring_non_ring={} atom_h={} max_delta={}",
            scorecard.matched_bonds,
            scorecard.matched_atoms,
            scorecard.fragment_count,
            scorecard.largest_fragment_size,
            scorecard.ring_non_ring_bond_score,
            scorecard.atom_h_score,
            scorecard.max_delta_atom_atom_dist,
        );
    }
}

#[test]
#[ignore = "focused diagnostic for the remaining 1K default mismatch 0719"]
fn print_massspecgym_case_1000_0719_partial_vs_all_best() {
    let cases = load_massspecgym_ground_truth_1000();
    let case = find_case(&cases, "massspecgym_default_0719");
    let partial =
        run_labeled_case_with_search_mode(case, true, false, McesSearchMode::PartialEnumeration);
    let partial_with_orientation =
        run_labeled_case_with_search_mode(case, true, true, McesSearchMode::PartialEnumeration);
    let all_best = run_labeled_case_with_search_mode(case, true, false, McesSearchMode::AllBest);

    let matches_expected = |info: &EagerCliqueInfo<usize>| {
        info.matched_edges().len() == case.expected_bond_matches
            && info.vertex_matches().len() == case.expected_atom_matches
            && (info_johnson_similarity(case, info) - case.expected_similarity).abs() <= 1e-6
    };

    let partial_expected = partial.all_cliques().iter().position(matches_expected);
    let partial_orientation_expected =
        partial_with_orientation.all_cliques().iter().position(matches_expected);
    let all_best_expected = all_best.all_cliques().iter().position(matches_expected);

    println!("case: {}", case.name);
    println!(
        "expected bonds={} atoms={} similarity={:.6}",
        case.expected_bond_matches, case.expected_atom_matches, case.expected_similarity
    );
    println!(
        "partial: retained={} top_bonds={} top_atoms={} top_similarity={:.6} expected_index={partial_expected:?}",
        partial.all_cliques().len(),
        partial.matched_edges().len(),
        partial.vertex_matches().len(),
        partial.johnson_similarity(),
    );
    println!(
        "partial+orientation: retained={} top_bonds={} top_atoms={} top_similarity={:.6} expected_index={partial_orientation_expected:?}",
        partial_with_orientation.all_cliques().len(),
        partial_with_orientation.matched_edges().len(),
        partial_with_orientation.vertex_matches().len(),
        partial_with_orientation.johnson_similarity(),
    );
    println!(
        "all_best: retained={} top_bonds={} top_atoms={} top_similarity={:.6} expected_index={all_best_expected:?}",
        all_best.all_cliques().len(),
        all_best.matched_edges().len(),
        all_best.vertex_matches().len(),
        all_best.johnson_similarity(),
    );

    println!("partial_order:");
    for (index, info) in partial.all_cliques().iter().take(8).enumerate() {
        println!(
            "  #{index}: bonds={} atoms={} similarity={:.6} fragments={} largest_fragment_atoms={}",
            info.matched_edges().len(),
            info.vertex_matches().len(),
            info_johnson_similarity(case, info),
            info.fragment_count(),
            info.largest_fragment_atom_count(),
        );
    }

    println!("all_best_order:");
    for (index, info) in all_best.all_cliques().iter().take(8).enumerate() {
        println!(
            "  #{index}: bonds={} atoms={} similarity={:.6} fragments={} largest_fragment_atoms={}",
            info.matched_edges().len(),
            info.vertex_matches().len(),
            info_johnson_similarity(case, info),
            info.fragment_count(),
            info.largest_fragment_atom_count(),
        );
    }
}

#[test]
#[ignore = "focused diagnostic for the remaining 1K default mismatches 0702 and 0911"]
fn print_massspecgym_case_1000_remaining_partial_partition_side_probe() {
    let cases = load_massspecgym_ground_truth_1000();

    for case_name in ["massspecgym_default_0702", "massspecgym_default_0911"] {
        let case = find_case(&cases, case_name);
        let partial = run_labeled_case_with_search_mode(
            case,
            true,
            false,
            McesSearchMode::PartialEnumeration,
        );
        let partial_with_orientation =
            run_labeled_case_with_search_mode(case, true, true, McesSearchMode::PartialEnumeration);
        let all_best =
            run_labeled_case_with_search_mode(case, true, false, McesSearchMode::AllBest);
        let diagnostics = collect_fixture_order_product_diagnostics(case, true);
        let order = product_order_rdkit_raw_pair_order(case, &diagnostics);

        println!("case: {}", case.name);
        println!(
            "expected bonds={} atoms={} similarity={:.6}",
            case.expected_bond_matches, case.expected_atom_matches, case.expected_similarity
        );
        println!(
            "partial: retained={} top_bonds={} top_atoms={} top_similarity={:.6}",
            partial.all_cliques().len(),
            partial.matched_edges().len(),
            partial.vertex_matches().len(),
            partial.johnson_similarity(),
        );
        println!(
            "partial+orientation: retained={} top_bonds={} top_atoms={} top_similarity={:.6}",
            partial_with_orientation.all_cliques().len(),
            partial_with_orientation.matched_edges().len(),
            partial_with_orientation.vertex_matches().len(),
            partial_with_orientation.johnson_similarity(),
        );
        println!(
            "all_best: retained={} top_bonds={} top_atoms={} top_similarity={:.6}",
            all_best.all_cliques().len(),
            all_best.matched_edges().len(),
            all_best.vertex_matches().len(),
            all_best.johnson_similarity(),
        );

        let expected_in_all_best = all_best.all_cliques().iter().position(|info| {
            info.matched_edges().len() == case.expected_bond_matches
                && info.vertex_matches().len() == case.expected_atom_matches
                && (info_johnson_similarity(case, info) - case.expected_similarity).abs() <= 1e-6
        });
        println!("expected_index_in_all_best={expected_in_all_best:?}");

        for side in [
            geometric_traits::traits::algorithms::maximum_clique::PartitionSide::First,
            geometric_traits::traits::algorithms::maximum_clique::PartitionSide::Second,
        ] {
            let infos = permuted_partitioned_infos(
                case,
                &diagnostics,
                &order,
                side,
                McesSearchMode::PartialEnumeration,
            );
            let expected_index = infos.iter().position(|info| {
                info.matched_edges().len() == case.expected_bond_matches
                    && info.vertex_matches().len() == case.expected_atom_matches
                    && (info_johnson_similarity(case, info) - case.expected_similarity).abs()
                        <= 1e-6
            });
            println!(
                "  side={side:?} retained={} top_bonds={} top_atoms={} top_similarity={:.6} expected_index={expected_index:?}",
                infos.len(),
                infos.first().map_or(0, |info| info.matched_edges().len()),
                infos.first().map_or(0, |info| info.vertex_matches().len()),
                infos.first().map(|info| info_johnson_similarity(case, info)).unwrap_or(0.0),
            );
        }
    }
}

#[test]
#[ignore = "focused diagnostic for lower-bound and root-pruning effects on the remaining 1K default mismatches"]
fn print_massspecgym_case_1000_remaining_partial_search_probe() {
    let cases = load_massspecgym_ground_truth_1000();

    for case_name in ["massspecgym_default_0702", "massspecgym_default_0911"] {
        let case = find_case(&cases, case_name);
        let diagnostics = collect_fixture_order_product_diagnostics(case, true);
        let order = product_order_rdkit_raw_pair_order(case, &diagnostics);
        let (matrix, vertex_pairs) =
            permute_product(&diagnostics.matrix, &diagnostics.vertex_pairs, &order);
        let (g1_label_indices, g2_label_indices, num_labels) = intern_case_bond_labels(
            &diagnostics.first_bond_labels,
            &diagnostics.second_bond_labels,
        );
        let partition_side =
            geometric_traits::traits::algorithms::maximum_clique::choose_partition_side(
                &vertex_pairs,
                diagnostics.first_edge_map.len(),
                diagnostics.second_edge_map.len(),
            );
        let partition = PartitionInfo {
            pairs: &vertex_pairs,
            g1_labels: &g1_label_indices,
            g2_labels: &g2_label_indices,
            num_labels,
            partition_side,
        };
        let first_equiv =
            heuristic_equivalent_bond_classes(&case.graph1, &diagnostics.first_edge_map);
        let second_equiv =
            heuristic_equivalent_bond_classes(&case.graph2, &diagnostics.second_edge_map);
        let mut seen_root_classes = BTreeSet::<(i32, i32)>::new();
        let root_pruned =
            geometric_traits::traits::algorithms::maximum_clique::partial_search_with_root_pruning(
                &matrix,
                &partition,
                0,
                |clique| {
                    !clique_has_delta_y_from_product(
                        clique,
                        &vertex_pairs,
                        &diagnostics.first_edge_map,
                        &diagnostics.second_edge_map,
                        case.graph1.n_atoms,
                        case.graph2.n_atoms,
                    )
                },
                |root_vertex| {
                    let (g1, g2) = vertex_pairs[root_vertex];
                    let classes = (first_equiv[g1], second_equiv[g2]);
                    classes.0 >= 0 && classes.1 >= 0 && !seen_root_classes.insert(classes)
                },
            );
        let root_pruned_infos = rank_partitioned_cliques(
            root_pruned,
            &vertex_pairs,
            &diagnostics.first_edge_map,
            &diagnostics.second_edge_map,
        );
        let root_pruned_expected = root_pruned_infos.iter().position(|info| {
            info.matched_edges().len() == case.expected_bond_matches
                && info.vertex_matches().len() == case.expected_atom_matches
                && (info_johnson_similarity(case, info) - case.expected_similarity).abs() <= 1e-6
        });

        let mut lower_bound_hit = None;
        for lower_bound in 0..=case.expected_bond_matches {
            let rerun = geometric_traits::traits::algorithms::maximum_clique::partial_search(
                &matrix,
                &partition,
                lower_bound,
                |clique| {
                    !clique_has_delta_y_from_product(
                        clique,
                        &vertex_pairs,
                        &diagnostics.first_edge_map,
                        &diagnostics.second_edge_map,
                        case.graph1.n_atoms,
                        case.graph2.n_atoms,
                    )
                },
            );
            let reranked = rank_partitioned_cliques(
                rerun,
                &vertex_pairs,
                &diagnostics.first_edge_map,
                &diagnostics.second_edge_map,
            );
            let expected_index = reranked.iter().position(|info| {
                info.matched_edges().len() == case.expected_bond_matches
                    && info.vertex_matches().len() == case.expected_atom_matches
                    && (info_johnson_similarity(case, info) - case.expected_similarity).abs()
                        <= 1e-6
            });
            if expected_index.is_some() {
                lower_bound_hit =
                    Some((lower_bound, expected_index, reranked[0].vertex_matches().len()));
                break;
            }
        }

        println!("case: {}", case.name);
        println!(
            "partition_side={partition_side:?} root_pruned_expected={root_pruned_expected:?} lower_bound_hit={lower_bound_hit:?}"
        );
        println!(
            "root_pruned_top_atoms={} root_pruned_top_similarity={:.6}",
            root_pruned_infos.first().map_or(0, |info| info.vertex_matches().len()),
            root_pruned_infos
                .first()
                .map(|info| info_johnson_similarity(case, info))
                .unwrap_or(0.0),
        );
    }
}

#[test]
#[ignore = "focused context-admission diagnostic for the remaining 1K default mismatch 0719"]
fn print_massspecgym_case_1000_0719_context_comparison() {
    let cases = load_massspecgym_ground_truth_1000();
    let case = find_case(&cases, "massspecgym_default_0719");
    let prepared = prepare_labeled_case(case);

    let diagnostics_without =
        collect_prepared_labeled_case_product_diagnostics(case, &prepared, false);
    let result_without =
        run_labeled_case_with_search_mode(case, false, false, McesSearchMode::AllBest);

    let diagnostics_with = collect_prepared_labeled_case_product_diagnostics(case, &prepared, true);
    let result_with = run_labeled_case_with_search_mode(case, true, false, McesSearchMode::AllBest);

    println!("case: {}", case.name);
    println!(
        "expected bonds={} atoms={} similarity={:.6}",
        case.expected_bond_matches, case.expected_atom_matches, case.expected_similarity
    );
    println!(
        "without contexts: product_vertices={} retained={} top_bonds={} top_atoms={} top_similarity={:.6}",
        diagnostics_without.vertex_pairs.len(),
        result_without.all_cliques().len(),
        result_without.matched_edges().len(),
        result_without.vertex_matches().len(),
        result_without.johnson_similarity(),
    );
    println!(
        "with contexts: product_vertices={} retained={} top_bonds={} top_atoms={} top_similarity={:.6}",
        diagnostics_with.vertex_pairs.len(),
        result_with.all_cliques().len(),
        result_with.matched_edges().len(),
        result_with.vertex_matches().len(),
        result_with.johnson_similarity(),
    );

    let removed_pairs: Vec<(usize, usize)> = diagnostics_without
        .vertex_pairs
        .iter()
        .copied()
        .filter(|pair| !diagnostics_with.vertex_pairs.contains(pair))
        .collect();
    println!("removed product vertices: {}", removed_pairs.len());
    for (i, j) in removed_pairs.iter().copied().take(20) {
        println!(
            "  ({i}, {j}) left_label={:?} right_label={:?} left_contexts={:?} right_contexts={:?}",
            diagnostics_without.first_bond_labels[i],
            diagnostics_without.second_bond_labels[j],
            case.graph1.aromatic_ring_contexts[i],
            case.graph2.aromatic_ring_contexts[j],
        );
    }
}

#[test]
#[ignore = "focused diagnostic proving the remaining default mismatches are RDKit raw-pair-order effects"]
fn test_massspecgym_ground_truth_labeled_mces_partial_enumeration_rdkit_raw_pair_order_cases() {
    let cases = load_massspecgym_ground_truth();
    for case_name in ["massspecgym_default_0038", "massspecgym_default_0092"] {
        let case = find_case(&cases, case_name);
        let diagnostics = collect_fixture_order_product_diagnostics(case, true);
        let order = product_order_rdkit_raw_pair_order(case, &diagnostics);
        let partition_side =
            geometric_traits::traits::algorithms::maximum_clique::choose_partition_side(
                &diagnostics.vertex_pairs,
                diagnostics.first_edge_map.len(),
                diagnostics.second_edge_map.len(),
            );
        let infos = permuted_partitioned_infos(
            case,
            &diagnostics,
            &order,
            partition_side,
            McesSearchMode::PartialEnumeration,
        );
        let top = infos.first().expect("expected at least one retained clique");
        assert_eq!(top.matched_edges().len(), case.expected_bond_matches, "{case_name}");
        assert_eq!(top.vertex_matches().len(), case.expected_atom_matches, "{case_name}");
        let similarity = info_johnson_similarity(case, top);
        assert!(
            (similarity - case.expected_similarity).abs() <= 1e-6,
            "{case_name}: expected similarity {:.6}, got {:.6}",
            case.expected_similarity,
            similarity
        );
    }
}

#[test]
#[ignore = "manual diagnostic harness for 1K MassSpecGym PartialEnumeration vs AllBest retention"]
fn print_massspecgym_case_1000_partial_vs_all_best() {
    let case_name =
        std::env::var("MCES_CASE").expect("set MCES_CASE to a MassSpecGym 1K case name");
    let limit: usize =
        std::env::var("MCES_COMPARE_LIMIT").ok().and_then(|value| value.parse().ok()).unwrap_or(8);

    let cases = load_massspecgym_ground_truth_1000();
    let case = find_case(&cases, &case_name);
    let partial =
        run_labeled_case_with_search_mode(case, true, false, McesSearchMode::PartialEnumeration);
    let partial_with_orientation =
        run_labeled_case_with_search_mode(case, true, true, McesSearchMode::PartialEnumeration);
    let all_best = run_labeled_case_with_search_mode(case, true, false, McesSearchMode::AllBest);

    println!("case: {}", case.name);
    println!(
        "expected bonds={} atoms={} similarity={:.6}",
        case.expected_bond_matches, case.expected_atom_matches, case.expected_similarity
    );
    println!(
        "partial: retained={} top_bonds={} top_atoms={} top_similarity={:.6}",
        partial.all_cliques().len(),
        partial.matched_edges().len(),
        partial.vertex_matches().len(),
        partial.johnson_similarity(),
    );
    println!(
        "partial+orientation: retained={} top_bonds={} top_atoms={} top_similarity={:.6}",
        partial_with_orientation.all_cliques().len(),
        partial_with_orientation.matched_edges().len(),
        partial_with_orientation.vertex_matches().len(),
        partial_with_orientation.johnson_similarity(),
    );
    println!(
        "all_best: retained={} top_bonds={} top_atoms={} top_similarity={:.6}",
        all_best.all_cliques().len(),
        all_best.matched_edges().len(),
        all_best.vertex_matches().len(),
        all_best.johnson_similarity(),
    );

    println!("partial_order:");
    for (index, info) in partial.all_cliques().iter().take(limit).enumerate() {
        println!(
            "  #{index}: bonds={} atoms={} similarity={:.6} fragments={} largest_fragment_atoms={}",
            info.matched_edges().len(),
            info.vertex_matches().len(),
            info_johnson_similarity(case, info),
            info.fragment_count(),
            info.largest_fragment_atom_count(),
        );
        println!("    vertex_matches={:?}", info.vertex_matches());
    }

    println!("all_best_order:");
    for (index, info) in all_best.all_cliques().iter().take(limit).enumerate() {
        println!(
            "  #{index}: bonds={} atoms={} similarity={:.6} fragments={} largest_fragment_atoms={}",
            info.matched_edges().len(),
            info.vertex_matches().len(),
            info_johnson_similarity(case, info),
            info.fragment_count(),
            info.largest_fragment_atom_count(),
        );
        println!("    vertex_matches={:?}", info.vertex_matches());
    }
}

#[test]
#[ignore = "manual timing harness for one MassSpecGym case"]
fn print_massspecgym_case_timing() {
    let case_name = std::env::var("MCES_CASE").expect("set MCES_CASE to a MassSpecGym case name");
    let corpus_size: usize =
        std::env::var("MCES_CORPUS_SIZE").ok().and_then(|value| value.parse().ok()).unwrap_or(1000);

    let cases = load_massspecgym_ground_truth_by_size(corpus_size);
    let case = find_case(&cases, &case_name);

    println!("case: {}", case.name);
    println!(
        "fixture_expected: bonds={} atoms={} similarity={:.6}",
        case.expected_bond_matches, case.expected_atom_matches, case.expected_similarity
    );
    println!(
        "graph_sizes: atoms=({},{}) edges=({},{})",
        case.graph1.n_atoms,
        case.graph2.n_atoms,
        case.graph1.edges.len(),
        case.graph2.edges.len(),
    );
    println!("rdkit_default_elapsed_seconds={:.6}", case.rdkit_elapsed_seconds.unwrap_or_default());

    let started = Instant::now();
    let partial =
        run_labeled_case_with_search_mode(case, true, false, McesSearchMode::PartialEnumeration);
    let partial_elapsed = started.elapsed();
    println!(
        "partial: wall_ms={} retained={} top_bonds={} top_atoms={} top_similarity={:.6}",
        partial_elapsed.as_millis(),
        partial.all_cliques().len(),
        partial.matched_edges().len(),
        partial.vertex_matches().len(),
        partial.johnson_similarity(),
    );

    let started = Instant::now();
    let partial_orientation =
        run_labeled_case_with_search_mode(case, true, true, McesSearchMode::PartialEnumeration);
    let partial_orientation_elapsed = started.elapsed();
    println!(
        "partial+orientation: wall_ms={} retained={} top_bonds={} top_atoms={} top_similarity={:.6}",
        partial_orientation_elapsed.as_millis(),
        partial_orientation.all_cliques().len(),
        partial_orientation.matched_edges().len(),
        partial_orientation.vertex_matches().len(),
        partial_orientation.johnson_similarity(),
    );

    let started = Instant::now();
    let all_best = run_labeled_case_with_search_mode(case, true, false, McesSearchMode::AllBest);
    let all_best_elapsed = started.elapsed();
    println!(
        "all_best: wall_ms={} retained={} top_bonds={} top_atoms={} top_similarity={:.6}",
        all_best_elapsed.as_millis(),
        all_best.all_cliques().len(),
        all_best.matched_edges().len(),
        all_best.vertex_matches().len(),
        all_best.johnson_similarity(),
    );
}

#[test]
#[ignore = "manual timing breakdown harness for one MassSpecGym case"]
fn print_massspecgym_case_timing_breakdown() {
    let case_name = std::env::var("MCES_CASE").expect("set MCES_CASE to a MassSpecGym case name");
    let corpus_size: usize =
        std::env::var("MCES_CORPUS_SIZE").ok().and_then(|value| value.parse().ok()).unwrap_or(1000);
    let search_mode =
        match std::env::var("MCES_SEARCH_MODE").unwrap_or_else(|_| "all_best".to_string()).as_str()
        {
            "partial" => McesSearchMode::PartialEnumeration,
            "all_best" => McesSearchMode::AllBest,
            other => panic!("unsupported MCES_SEARCH_MODE '{other}'"),
        };
    let use_partition_orientation =
        matches!(std::env::var("MCES_PARTITION_ORIENTATION").as_deref(), Ok("1" | "true" | "yes"));
    let initial_lower_bound_override =
        std::env::var("MCES_INITIAL_LOWER_BOUND").ok().and_then(|value| value.parse().ok());
    let state_lower_bound_override =
        std::env::var("MCES_STATE_LOWER_BOUND").ok().and_then(|value| value.parse().ok());
    let best_size_seed_override =
        std::env::var("MCES_BEST_SIZE_SEED").ok().and_then(|value| value.parse().ok());

    let cases = load_massspecgym_ground_truth_by_size(corpus_size);
    let case = find_case(&cases, &case_name);

    println!("case: {}", case.name);
    println!(
        "fixture_expected: bonds={} atoms={} similarity={:.6}",
        case.expected_bond_matches, case.expected_atom_matches, case.expected_similarity
    );
    println!(
        "graph_sizes: atoms=({},{}) edges=({},{})",
        case.graph1.n_atoms,
        case.graph2.n_atoms,
        case.graph1.edges.len(),
        case.graph2.edges.len(),
    );
    println!("rdkit_default_elapsed_seconds={:.6}", case.rdkit_elapsed_seconds.unwrap_or_default());
    println!("mode={search_mode:?} partition_orientation={use_partition_orientation}");
    println!("initial_lower_bound_override={initial_lower_bound_override:?}");

    let started = Instant::now();
    let prepared = prepare_labeled_case(case);
    let prepare_elapsed = started.elapsed();

    let started = Instant::now();
    let diagnostics = collect_prepared_labeled_case_product_diagnostics(case, &prepared, true);
    let diagnostics_elapsed = started.elapsed();

    let started = Instant::now();
    let order = product_order_rdkit_raw_pair_order(case, &diagnostics);
    let (matrix, permuted_pairs) =
        permute_product(&diagnostics.matrix, &diagnostics.vertex_pairs, &order);
    let reorder_elapsed = started.elapsed();

    let started = Instant::now();
    let (g1_label_indices, g2_label_indices, num_labels) =
        intern_case_bond_labels(&diagnostics.first_bond_labels, &diagnostics.second_bond_labels);
    let partition_side = if use_partition_orientation {
        geometric_traits::traits::algorithms::maximum_clique::choose_partition_side(
            &permuted_pairs,
            diagnostics.first_edge_map.len(),
            diagnostics.second_edge_map.len(),
        )
    } else {
        geometric_traits::traits::algorithms::maximum_clique::PartitionSide::First
    };
    let partition = PartitionInfo {
        pairs: &permuted_pairs,
        g1_labels: &g1_label_indices,
        g2_labels: &g2_label_indices,
        num_labels,
        partition_side,
    };
    let partition_elapsed = started.elapsed();

    let baseline_lower_bound = initial_lower_bound_override.unwrap_or_else(|| {
        usize::from(matches!(search_mode, McesSearchMode::PartialEnumeration) && matrix.order() > 0)
    });
    let state_lower_bound = state_lower_bound_override.unwrap_or(baseline_lower_bound);
    let best_size_seed = best_size_seed_override.unwrap_or(baseline_lower_bound);

    let started = Instant::now();
    let greedy_lower_bound =
        geometric_traits::traits::algorithms::maximum_clique::greedy_lower_bound(
            &matrix,
            &partition,
            baseline_lower_bound,
            |clique| {
                !clique_has_delta_y_from_product(
                    clique,
                    &permuted_pairs,
                    &diagnostics.first_edge_map,
                    &diagnostics.second_edge_map,
                    case.graph1.n_atoms,
                    case.graph2.n_atoms,
                )
            },
        );
    let greedy_elapsed = started.elapsed();

    let started = Instant::now();
    let unfiltered_max_size =
        geometric_traits::traits::algorithms::maximum_clique::PartitionedMaximumClique::maximum_clique_with_partition(
            &matrix,
            &partition,
        )
        .len();
    let unfiltered_elapsed = started.elapsed();
    println!(
        "baseline_lower_bound={} state_lower_bound={} best_size_seed={} greedy_lower_bound={} greedy_ms={} unfiltered_max_size={} unfiltered_ms={}",
        baseline_lower_bound,
        state_lower_bound,
        best_size_seed,
        greedy_lower_bound,
        greedy_elapsed.as_millis(),
        unfiltered_max_size,
        unfiltered_elapsed.as_millis(),
    );

    let started = Instant::now();
    let mut accept_calls = 0usize;
    let mut accept_nanos = 0u128;
    let accept = |clique: &[usize]| {
        accept_calls += 1;
        let started = Instant::now();
        let accepted = !clique_has_delta_y_from_product(
            clique,
            &permuted_pairs,
            &diagnostics.first_edge_map,
            &diagnostics.second_edge_map,
            case.graph1.n_atoms,
            case.graph2.n_atoms,
        );
        accept_nanos += started.elapsed().as_nanos();
        accepted
    };
    let profile = geometric_traits::traits::algorithms::maximum_clique::profile_search_with_bounds(
        &matrix,
        &partition,
        matches!(search_mode, McesSearchMode::AllBest),
        state_lower_bound,
        best_size_seed,
        accept,
    );
    let search_elapsed = started.elapsed();
    let retained = profile.best_cliques.len();

    let started = Instant::now();
    let infos = rank_partitioned_cliques(
        profile.best_cliques,
        &permuted_pairs,
        &diagnostics.first_edge_map,
        &diagnostics.second_edge_map,
    );
    let ranking_elapsed = started.elapsed();

    let top = infos.first().expect("expected at least one retained clique");
    println!(
        "product: vertices={} first_edges={} second_edges={}",
        diagnostics.vertex_pairs.len(),
        diagnostics.first_edge_map.len(),
        diagnostics.second_edge_map.len(),
    );
    println!(
        "timing_ms: prepare={} diagnostics={} reorder={} partition={} greedy={} search={} ranking={}",
        prepare_elapsed.as_millis(),
        diagnostics_elapsed.as_millis(),
        reorder_elapsed.as_millis(),
        partition_elapsed.as_millis(),
        greedy_elapsed.as_millis(),
        search_elapsed.as_millis(),
        ranking_elapsed.as_millis(),
    );
    println!(
        "search_stats: dfs_calls={} empty_returns={} max_depth={} parts_prunes={} parts_equal_best={} label_prunes={} label_equal_best={} popped={} pruned_vertices={} restored_vertices={} prune_candidate_checks={} selected_part_scans={} upper_bound_labels_scanned={} take_branches={} skip_branches={} maybe_updates={} best_improvements={} last_best_improvement_call={} retained_best_cliques={} state_clones={} cloned_partitions={} cloned_vertices={} accept_calls={} accept_ms={}",
        profile.stats.dfs_calls,
        profile.stats.empty_state_returns,
        profile.stats.max_depth,
        profile.stats.parts_bound_prunes,
        profile.stats.parts_bound_equal_best,
        profile.stats.label_bound_prunes,
        profile.stats.label_bound_equal_best,
        profile.stats.vertices_popped,
        profile.stats.vertices_pruned,
        profile.stats.restored_vertices,
        profile.stats.prune_candidate_checks,
        profile.stats.selected_part_scans,
        profile.stats.upper_bound_labels_scanned,
        profile.stats.take_branches,
        profile.stats.skip_branches,
        profile.stats.maybe_update_calls,
        profile.stats.best_size_improvements,
        profile.stats.last_best_improvement_dfs_call,
        profile.stats.retained_best_cliques,
        profile.stats.state_clones,
        profile.stats.cloned_partitions,
        profile.stats.cloned_vertices,
        accept_calls,
        accept_nanos / 1_000_000,
    );
    println!(
        "result: retained={} top_bonds={} top_atoms={} top_similarity={:.6}",
        retained,
        top.matched_edges().len(),
        top.vertex_matches().len(),
        info_johnson_similarity(case, top),
    );
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

    let (first_left, first_right) = find_refined_pair(&first_labels)
        .expect("expected a same-type/different-degree witness in graph1");
    let (second_left, second_right) = find_refined_pair(&second_labels)
        .expect("expected a same-type/different-degree witness in graph2");

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
            ignore_case.graph1.atom_types.iter().enumerate().skip(left_index + 1).find_map(
                |(right_index, right_type)| {
                    if left_type == right_type
                        && ignore_case.graph1.atom_is_aromatic[left_index]
                            != ignore_case.graph1.atom_is_aromatic[right_index]
                    {
                        Some((left_index, right_index))
                    } else {
                        None
                    }
                },
            )
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
                let same_primary = first_label.0 == second_label.0
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
#[ignore = "manual diagnostic harness for MassSpecGym atom-count mismatches"]
fn print_massspecgym_case_atom_mismatch_details() {
    let case_name = std::env::var("MCES_CASE").expect("set MCES_CASE to a MassSpecGym case name");
    let cases = load_massspecgym_ground_truth();
    let case = find_case(&cases, &case_name);
    let prepared = prepare_labeled_case(case);
    let diagnostics = collect_prepared_labeled_case_product_diagnostics(case, &prepared, true);
    let result = run_labeled_case(case);
    let selected_edge_indices = selected_clique_edge_indices(&result, &diagnostics);
    let inferred_atoms = inferred_atom_count_from_similarity(
        case,
        result.matched_edges().len(),
        result.johnson_similarity(),
    );

    println!("case: {}", case.name);
    println!("smiles1: {}", case.smiles1);
    println!("smiles2: {}", case.smiles2);
    println!(
        "expected: bonds={} atoms={} similarity={:.6}",
        case.expected_bond_matches, case.expected_atom_matches, case.expected_similarity
    );
    println!(
        "rust: bonds={} atoms={} similarity={:.6}",
        result.matched_edges().len(),
        result.vertex_matches().len(),
        result.johnson_similarity(),
    );
    println!("inferred_atoms_from_similarity: {inferred_atoms}");
    println!(
        "atom_delta: {}",
        result.vertex_matches().len() as isize - case.expected_atom_matches as isize
    );
    println!(
        "selected clique members: {}",
        result.all_cliques().first().map(|info| info.clique().len()).unwrap_or(0)
    );
    println!("selected edge indices: {:?}", selected_edge_indices);
    println!("matched edge pairs: {:?}", result.matched_edges());
    println!("vertex matches: {:?}", result.vertex_matches());
}

#[test]
#[ignore = "manual diagnostic harness for MassSpecGym isolated-bond orientation"]
fn print_massspecgym_case_isolated_bond_analysis() {
    let case_name = std::env::var("MCES_CASE").expect("set MCES_CASE to a MassSpecGym case name");
    let cases = load_massspecgym_ground_truth();
    let case = find_case(&cases, &case_name);
    let prepared = prepare_labeled_case(case);
    let diagnostics = collect_prepared_labeled_case_product_diagnostics(case, &prepared, true);
    let result = run_labeled_case(case);
    let selected_edge_indices = selected_clique_edge_indices(&result, &diagnostics);
    let components = edge_components(&selected_edge_indices, &case.graph1.edges);

    println!("case: {}", case.name);
    println!(
        "expected atoms={} rust atoms={}",
        case.expected_atom_matches,
        result.vertex_matches().len()
    );
    println!(
        "component sizes on graph1 matched-edge subgraph: {:?}",
        components.iter().map(Vec::len).collect::<Vec<_>>()
    );

    for component in components.into_iter().filter(|component| component.len() == 1) {
        let (left_edge_index, right_edge_index) = component[0];
        let left_edge = case.graph1.edges[left_edge_index];
        let right_edge = case.graph2.edges[right_edge_index];
        let left_orientation = graph_edge_orientation(&case.graph1, left_edge_index);
        let right_orientation = graph_edge_orientation(&case.graph2, right_edge_index);
        let rdkit_mapping =
            rdkit_preferred_isolated_mapping(case, left_edge_index, right_edge_index);

        println!(
            "isolated bond pair ({left_edge_index}, {right_edge_index}) \
             left_edge={left_edge:?} right_edge={right_edge:?} \
             left_orientation={left_orientation:?} right_orientation={right_orientation:?}"
        );
        println!(
            "  left atom types: {:?} / {:?}, total_hs: {:?} / {:?}",
            case.graph1.atom_types[left_orientation[0]],
            case.graph1.atom_types[left_orientation[1]],
            graph_atom_total_hs(&case.graph1, left_orientation[0]),
            graph_atom_total_hs(&case.graph1, left_orientation[1]),
        );
        println!(
            "  right atom types: {:?} / {:?}, total_hs: {:?} / {:?}",
            case.graph2.atom_types[right_orientation[0]],
            case.graph2.atom_types[right_orientation[1]],
            graph_atom_total_hs(&case.graph2, right_orientation[0]),
            graph_atom_total_hs(&case.graph2, right_orientation[1]),
        );
        println!(
            "  rust default mapping: ({}, {}) ({}, {})",
            left_edge[0], right_edge[0], left_edge[1], right_edge[1]
        );
        println!(
            "  rdkit preferred mapping [{}]: {:?} {:?}",
            rdkit_mapping.2, rdkit_mapping.0, rdkit_mapping.1
        );
    }
}

#[test]
#[ignore = "manual diagnostic harness for MassSpecGym all-best clique summaries"]
fn print_massspecgym_case_all_best_atom_counts() {
    let case_name = std::env::var("MCES_CASE").expect("set MCES_CASE to a MassSpecGym case name");
    let limit: usize = std::env::var("MCES_ALL_BEST_LIMIT")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(20);

    let cases = load_massspecgym_ground_truth();
    let case = find_case(&cases, &case_name);
    let result = run_labeled_case_with_search_mode(case, true, false, McesSearchMode::AllBest);

    println!("case: {}", case.name);
    println!(
        "expected bonds={} atoms={} similarity={:.6}",
        case.expected_bond_matches, case.expected_atom_matches, case.expected_similarity
    );
    println!("all_best_cliques={}", result.all_cliques().len());

    for (index, info) in result.all_cliques().iter().take(limit).enumerate() {
        println!(
            "#{index}: bonds={} atoms={} fragments={} largest_fragment={}",
            info.matched_edges().len(),
            info.vertex_matches().len(),
            info.fragment_count(),
            info.largest_fragment_size(),
        );
        println!("  matched_edges={:?}", info.matched_edges());
        println!("  vertex_matches={:?}", info.vertex_matches());
    }
}

#[test]
#[ignore = "manual diagnostic harness for MassSpecGym AllBest scorecards"]
fn print_massspecgym_case_all_best_scorecards() {
    let case_name = std::env::var("MCES_CASE").expect("set MCES_CASE to a MassSpecGym case name");
    let limit: usize = std::env::var("MCES_SCORECARD_LIMIT")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(8);

    let cases = load_massspecgym_ground_truth();
    let case = find_case(&cases, &case_name);
    let result = run_labeled_case_with_search_mode(case, true, false, McesSearchMode::AllBest);
    let first_ring_membership = graph_ring_membership_by_edge(&case.graph1);
    let second_ring_membership = graph_ring_membership_by_edge(&case.graph2);
    let first_distances = graph_distance_matrix(&case.graph1);
    let second_distances = graph_distance_matrix(&case.graph2);

    let scorecards: Vec<CliqueScorecard> = result
        .all_cliques()
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
        .collect();

    let mut approx_rdkit_order: Vec<usize> = (0..scorecards.len()).collect();
    approx_rdkit_order.sort_by(|&left, &right| {
        approx_rdkit_compare(&scorecards[left], &scorecards[right]).then_with(|| left.cmp(&right))
    });

    println!("case: {}", case.name);
    println!(
        "expected bonds={} atoms={} similarity={:.6}",
        case.expected_bond_matches, case.expected_atom_matches, case.expected_similarity
    );
    println!("all_best_cliques={}", result.all_cliques().len());

    println!("current_rust_order:");
    for (index, (info, scorecard)) in
        result.all_cliques().iter().zip(scorecards.iter()).take(limit).enumerate()
    {
        println!(
            "#{index}: bonds={} atoms={} fragments={} largest_fragment={} ring_non_ring={} atom_h={} max_delta_dist={}",
            scorecard.matched_bonds,
            scorecard.matched_atoms,
            scorecard.fragment_count,
            scorecard.largest_fragment_size,
            scorecard.ring_non_ring_bond_score,
            scorecard.atom_h_score,
            scorecard.max_delta_atom_atom_dist,
        );
        println!("  matched_edges={:?}", info.matched_edges());
        println!("  vertex_matches={:?}", info.vertex_matches());
    }

    println!("approx_rdkit_order:");
    for (rank, &index) in approx_rdkit_order.iter().take(limit).enumerate() {
        let info = &result.all_cliques()[index];
        let scorecard = &scorecards[index];
        println!(
            "#{rank} [all_best_index={index}]: bonds={} atoms={} fragments={} largest_fragment={} ring_non_ring={} atom_h={} max_delta_dist={}",
            scorecard.matched_bonds,
            scorecard.matched_atoms,
            scorecard.fragment_count,
            scorecard.largest_fragment_size,
            scorecard.ring_non_ring_bond_score,
            scorecard.atom_h_score,
            scorecard.max_delta_atom_atom_dist,
        );
        println!("  matched_edges={:?}", info.matched_edges());
        println!("  vertex_matches={:?}", info.vertex_matches());
    }
}

#[test]
#[ignore = "manual diagnostic harness for MassSpecGym AllBest RDKit-style atom materialization"]
fn print_massspecgym_case_all_best_rdkit_materialization() {
    let case_name = std::env::var("MCES_CASE").expect("set MCES_CASE to a MassSpecGym case name");
    let limit: usize = std::env::var("MCES_RDKit_MATERIALIZATION_LIMIT")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(8);

    let cases = load_massspecgym_ground_truth();
    let case = find_case(&cases, &case_name);
    let diagnostics = collect_labeled_case_product_diagnostics(case, true);
    let result = run_labeled_case_with_search_mode(case, true, false, McesSearchMode::AllBest);
    let rdkit_matches: Vec<Vec<(usize, usize)>> = result
        .all_cliques()
        .iter()
        .map(|info| infer_vertex_matches_rdkit_style(case, info.clique(), &diagnostics))
        .collect();

    let first_rust_expected = result
        .all_cliques()
        .iter()
        .position(|info| info.vertex_matches().len() == case.expected_atom_matches);
    let first_rdkit_expected =
        rdkit_matches.iter().position(|matches| matches.len() == case.expected_atom_matches);

    println!("case: {}", case.name);
    println!(
        "expected bonds={} atoms={} similarity={:.6}",
        case.expected_bond_matches, case.expected_atom_matches, case.expected_similarity
    );
    println!("all_best_cliques={}", result.all_cliques().len());
    println!("first_rust_expected_index={first_rust_expected:?}");
    println!("first_rdkit_expected_index={first_rdkit_expected:?}");

    for (index, info) in result.all_cliques().iter().take(limit).enumerate() {
        let rdkit_like = &rdkit_matches[index];
        let rust_only: Vec<(usize, usize)> = info
            .vertex_matches()
            .iter()
            .copied()
            .filter(|pair| !rdkit_like.contains(pair))
            .collect();
        let rdkit_only: Vec<(usize, usize)> = rdkit_like
            .iter()
            .copied()
            .filter(|pair| !info.vertex_matches().contains(pair))
            .collect();

        println!(
            "#{index}: bonds={} rust_atoms={} rdkit_atoms={} fragments={} largest_fragment_atoms={}",
            info.matched_edges().len(),
            info.vertex_matches().len(),
            rdkit_like.len(),
            info.fragment_count(),
            info.largest_fragment_atom_count(),
        );
        println!("  clique={:?}", info.clique());
        println!("  matched_edges={:?}", info.matched_edges());
        println!("  rust_vertex_matches={:?}", info.vertex_matches());
        println!("  rdkit_vertex_matches={:?}", rdkit_like);
        if !rust_only.is_empty() || !rdkit_only.is_empty() {
            println!("  rust_only={:?}", rust_only);
            println!("  rdkit_only={:?}", rdkit_only);
        }
    }
}

#[test]
#[ignore = "manual diagnostic harness for MassSpecGym edge-context admission comparison"]
fn print_massspecgym_case_context_comparison() {
    let case_name = std::env::var("MCES_CASE").expect("set MCES_CASE to a MassSpecGym case name");
    let removed_limit: usize =
        std::env::var("MCES_REMOVED_LIMIT").ok().and_then(|value| value.parse().ok()).unwrap_or(20);

    let cases = load_massspecgym_ground_truth();
    let case = find_case(&cases, &case_name);
    let prepared = prepare_labeled_case(case);

    let started_without = Instant::now();
    let result_without =
        run_labeled_case_with_search_mode(case, false, false, McesSearchMode::AllBest);
    let elapsed_without = started_without.elapsed();
    let diagnostics_without =
        collect_prepared_labeled_case_product_diagnostics(case, &prepared, false);

    let started_with = Instant::now();
    let result_with = run_labeled_case_with_search_mode(case, true, false, McesSearchMode::AllBest);
    let elapsed_with = started_with.elapsed();
    let diagnostics_with = collect_prepared_labeled_case_product_diagnostics(case, &prepared, true);

    println!("case: {}", case.name);
    println!("options: {:?}", case.options);
    println!(
        "graph1 non-empty aromatic context rows: {} / {}",
        non_empty_context_row_count(&case.graph1),
        case.graph1.aromatic_ring_contexts.len()
    );
    println!(
        "graph2 non-empty aromatic context rows: {} / {}",
        non_empty_context_row_count(&case.graph2),
        case.graph2.aromatic_ring_contexts.len()
    );
    println!(
        "without contexts: elapsed={elapsed_without:?} product_vertices={} all_cliques={} matched_edges={} atoms={} similarity={:.6}",
        diagnostics_without.vertex_pairs.len(),
        result_without.all_cliques().len(),
        result_without.matched_edges().len(),
        result_without.vertex_matches().len(),
        result_without.johnson_similarity(),
    );
    println!(
        "with contexts: elapsed={elapsed_with:?} product_vertices={} all_cliques={} matched_edges={} atoms={} similarity={:.6}",
        diagnostics_with.vertex_pairs.len(),
        result_with.all_cliques().len(),
        result_with.matched_edges().len(),
        result_with.vertex_matches().len(),
        result_with.johnson_similarity(),
    );

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

#[test]
#[ignore = "manual diagnostic harness for MassSpecGym PartialEnumeration vs AllBest retention"]
fn print_massspecgym_case_partial_vs_all_best() {
    let case_name = std::env::var("MCES_CASE").expect("set MCES_CASE to a MassSpecGym case name");
    let limit: usize =
        std::env::var("MCES_COMPARE_LIMIT").ok().and_then(|value| value.parse().ok()).unwrap_or(8);

    let cases = load_massspecgym_ground_truth();
    let case = find_case(&cases, &case_name);
    let partial =
        run_labeled_case_with_search_mode(case, true, false, McesSearchMode::PartialEnumeration);
    let partial_with_orientation =
        run_labeled_case_with_search_mode(case, true, true, McesSearchMode::PartialEnumeration);
    let all_best = run_labeled_case_with_search_mode(case, true, false, McesSearchMode::AllBest);

    let partial_expected = partial
        .all_cliques()
        .iter()
        .position(|info| info.vertex_matches().len() == case.expected_atom_matches);
    let partial_orientation_expected = partial_with_orientation
        .all_cliques()
        .iter()
        .position(|info| info.vertex_matches().len() == case.expected_atom_matches);
    let all_best_expected = all_best
        .all_cliques()
        .iter()
        .position(|info| info.vertex_matches().len() == case.expected_atom_matches);

    println!("case: {}", case.name);
    println!(
        "expected bonds={} atoms={} similarity={:.6}",
        case.expected_bond_matches, case.expected_atom_matches, case.expected_similarity
    );
    println!(
        "partial: retained={} top_atoms={} top_similarity={:.6} expected_index={partial_expected:?}",
        partial.all_cliques().len(),
        partial.vertex_matches().len(),
        partial.johnson_similarity(),
    );
    println!(
        "partial+orientation: retained={} top_atoms={} top_similarity={:.6} expected_index={partial_orientation_expected:?}",
        partial_with_orientation.all_cliques().len(),
        partial_with_orientation.vertex_matches().len(),
        partial_with_orientation.johnson_similarity(),
    );
    println!(
        "all_best: retained={} top_atoms={} top_similarity={:.6} expected_index={all_best_expected:?}",
        all_best.all_cliques().len(),
        all_best.vertex_matches().len(),
        all_best.johnson_similarity(),
    );

    println!("partial_order:");
    for (index, info) in partial.all_cliques().iter().take(limit).enumerate() {
        println!(
            "  #{index}: bonds={} atoms={} fragments={} largest_fragment_atoms={}",
            info.matched_edges().len(),
            info.vertex_matches().len(),
            info.fragment_count(),
            info.largest_fragment_atom_count(),
        );
    }

    println!("all_best_order:");
    for (index, info) in all_best.all_cliques().iter().take(limit).enumerate() {
        println!(
            "  #{index}: bonds={} atoms={} fragments={} largest_fragment_atoms={}",
            info.matched_edges().len(),
            info.vertex_matches().len(),
            info.fragment_count(),
            info.largest_fragment_atom_count(),
        );
    }
}

#[test]
#[ignore = "manual diagnostic harness for modular-product vertex-order sensitivity"]
fn print_massspecgym_case_product_order_sensitivity() {
    let case_name = std::env::var("MCES_CASE").expect("set MCES_CASE to a MassSpecGym case name");
    let partition_side = match std::env::var("MCES_PARTITION_SIDE").ok().as_deref() {
        Some("second") | Some("Second") | Some("SECOND") => {
            geometric_traits::traits::algorithms::maximum_clique::PartitionSide::Second
        }
        _ => geometric_traits::traits::algorithms::maximum_clique::PartitionSide::First,
    };

    let cases = load_massspecgym_ground_truth();
    let case = find_case(&cases, &case_name);
    let diagnostics = collect_labeled_case_product_diagnostics(case, true);
    let identity_all_best = permuted_partitioned_infos(
        case,
        &diagnostics,
        &product_order_identity(&diagnostics.vertex_pairs),
        partition_side,
        McesSearchMode::AllBest,
    );
    let target_clique = identity_all_best
        .iter()
        .find(|info| info.vertex_matches().len() == case.expected_atom_matches)
        .map(|info| info.clique().to_vec())
        .unwrap_or_default();
    let strategies = [
        ("identity", product_order_identity(&diagnostics.vertex_pairs)),
        ("reverse", product_order_reverse(&diagnostics.vertex_pairs)),
        ("second_then_first", product_order_second_then_first(&diagnostics.vertex_pairs)),
        (
            "reverse_within_first_buckets",
            product_order_reverse_within_first_buckets(&diagnostics.vertex_pairs),
        ),
        ("fixture_edge_indices", product_order_fixture_edge_indices(case, &diagnostics)),
        (
            "fixture_edge_indices_second_then_first",
            product_order_fixture_edge_indices_second_then_first(case, &diagnostics),
        ),
        ("target_last", product_order_target_last(&diagnostics.vertex_pairs, &target_clique)),
    ];

    println!("case: {}", case.name);
    println!(
        "expected bonds={} atoms={} similarity={:.6} partition_side={partition_side:?}",
        case.expected_bond_matches, case.expected_atom_matches, case.expected_similarity
    );
    println!("product_vertices={}", diagnostics.vertex_pairs.len());

    for (name, order) in strategies {
        let partial = permuted_partitioned_infos(
            case,
            &diagnostics,
            &order,
            partition_side,
            McesSearchMode::PartialEnumeration,
        );
        let all_best = permuted_partitioned_infos(
            case,
            &diagnostics,
            &order,
            partition_side,
            McesSearchMode::AllBest,
        );
        let partial_expected = partial
            .iter()
            .position(|info| info.vertex_matches().len() == case.expected_atom_matches);
        let all_best_expected = all_best
            .iter()
            .position(|info| info.vertex_matches().len() == case.expected_atom_matches);
        let partial_top_similarity =
            partial.first().map(|info| info_johnson_similarity(case, info)).unwrap_or(0.0);
        let all_best_top_similarity =
            all_best.first().map(|info| info_johnson_similarity(case, info)).unwrap_or(0.0);

        println!(
            "{name}: partial_retained={} partial_top_atoms={} partial_top_similarity={:.6} partial_expected_index={partial_expected:?}",
            partial.len(),
            partial.first().map_or(0, |info| info.vertex_matches().len()),
            partial_top_similarity,
        );
        println!(
            "  {name}: all_best_retained={} all_best_top_atoms={} all_best_top_similarity={:.6} all_best_expected_index={all_best_expected:?}",
            all_best.len(),
            all_best.first().map_or(0, |info| info.vertex_matches().len()),
            all_best_top_similarity,
        );
    }
}

#[test]
#[ignore = "manual diagnostic harness for fixture-order line-graph and product construction"]
fn print_massspecgym_case_fixture_order_line_graph_product() {
    let case_name = std::env::var("MCES_CASE").expect("set MCES_CASE to a MassSpecGym case name");
    let cases = load_massspecgym_ground_truth();
    let case = find_case(&cases, &case_name);

    let row_major = collect_labeled_case_product_diagnostics(case, true);
    let fixture_order = collect_fixture_order_product_diagnostics(case, true);
    let partition_side =
        geometric_traits::traits::algorithms::maximum_clique::choose_partition_side(
            &fixture_order.vertex_pairs,
            fixture_order.first_edge_map.len(),
            fixture_order.second_edge_map.len(),
        );

    let row_major_partial = permuted_partitioned_infos(
        case,
        &row_major,
        &product_order_identity(&row_major.vertex_pairs),
        partition_side,
        McesSearchMode::PartialEnumeration,
    );
    let row_major_all_best = permuted_partitioned_infos(
        case,
        &row_major,
        &product_order_identity(&row_major.vertex_pairs),
        partition_side,
        McesSearchMode::AllBest,
    );
    let fixture_partial = permuted_partitioned_infos(
        case,
        &fixture_order,
        &product_order_identity(&fixture_order.vertex_pairs),
        partition_side,
        McesSearchMode::PartialEnumeration,
    );
    let fixture_all_best = permuted_partitioned_infos(
        case,
        &fixture_order,
        &product_order_identity(&fixture_order.vertex_pairs),
        partition_side,
        McesSearchMode::AllBest,
    );

    let row_major_partial_expected = row_major_partial
        .iter()
        .position(|info| info.vertex_matches().len() == case.expected_atom_matches);
    let row_major_all_best_expected = row_major_all_best
        .iter()
        .position(|info| info.vertex_matches().len() == case.expected_atom_matches);
    let fixture_partial_expected = fixture_partial
        .iter()
        .position(|info| info.vertex_matches().len() == case.expected_atom_matches);
    let fixture_all_best_expected = fixture_all_best
        .iter()
        .position(|info| info.vertex_matches().len() == case.expected_atom_matches);

    println!("case: {}", case.name);
    println!(
        "expected bonds={} atoms={} similarity={:.6}",
        case.expected_bond_matches, case.expected_atom_matches, case.expected_similarity
    );
    println!(
        "row_major: product_vertices={} partial_top_atoms={} partial_expected_index={row_major_partial_expected:?} all_best_top_atoms={} all_best_expected_index={row_major_all_best_expected:?}",
        row_major.vertex_pairs.len(),
        row_major_partial.first().map(|info| info.vertex_matches().len()).unwrap_or(0),
        row_major_all_best.first().map(|info| info.vertex_matches().len()).unwrap_or(0),
    );
    let first_edge_order_diff = row_major
        .first_edge_map
        .iter()
        .zip(fixture_order.first_edge_map.iter())
        .filter(|(left, right)| left != right)
        .count();
    let second_edge_order_diff = row_major
        .second_edge_map
        .iter()
        .zip(fixture_order.second_edge_map.iter())
        .filter(|(left, right)| left != right)
        .count();
    println!(
        "fixture_order: product_vertices={} partial_top_atoms={} partial_expected_index={fixture_partial_expected:?} all_best_top_atoms={} all_best_expected_index={fixture_all_best_expected:?} first_edge_order_diff={} second_edge_order_diff={}",
        fixture_order.vertex_pairs.len(),
        fixture_partial.first().map(|info| info.vertex_matches().len()).unwrap_or(0),
        fixture_all_best.first().map(|info| info.vertex_matches().len()).unwrap_or(0),
        first_edge_order_diff,
        second_edge_order_diff,
    );
}

#[test]
#[ignore = "manual diagnostic harness for fixture edge-order canonicalization"]
fn print_massspecgym_case_fixture_order_canonicalization() {
    let case_name = std::env::var("MCES_CASE").expect("set MCES_CASE to a MassSpecGym case name");

    let cases = load_massspecgym_ground_truth();
    let case = find_case(&cases, &case_name);
    let prepared_original = prepare_labeled_case(case);
    let diagnostics_original =
        collect_prepared_labeled_case_product_diagnostics(case, &prepared_original, true);

    let reversed_graph1 = reverse_graph_bond_payload(&case.graph1);
    let reversed_graph2 = reverse_graph_bond_payload(&case.graph2);
    let prepared_reversed =
        prepare_labeled_case_from_graph_data(case, &reversed_graph1, &reversed_graph2);
    let diagnostics_reversed =
        collect_prepared_labeled_case_product_diagnostics(case, &prepared_reversed, true);

    println!("case: {}", case.name);
    println!(
        "graph1_edge_map_equal={} graph2_edge_map_equal={} product_pairs_equal={}",
        diagnostics_original.first_edge_map == diagnostics_reversed.first_edge_map,
        diagnostics_original.second_edge_map == diagnostics_reversed.second_edge_map,
        diagnostics_original.vertex_pairs == diagnostics_reversed.vertex_pairs,
    );
    println!("product_matrix_equal={}", diagnostics_original.matrix == diagnostics_reversed.matrix,);
}

#[test]
#[ignore = "manual diagnostic harness for comparing retained and expected clique members"]
fn print_massspecgym_case_clique_member_differences() {
    let case_name = std::env::var("MCES_CASE").expect("set MCES_CASE to a MassSpecGym case name");

    let cases = load_massspecgym_ground_truth();
    let case = find_case(&cases, &case_name);
    let diagnostics = collect_labeled_case_product_diagnostics(case, true);
    let partial =
        run_labeled_case_with_search_mode(case, true, false, McesSearchMode::PartialEnumeration);
    let all_best = run_labeled_case_with_search_mode(case, true, false, McesSearchMode::AllBest);
    let selected = partial.all_cliques().first().expect("partial result should retain a clique");
    let expected = all_best
        .all_cliques()
        .iter()
        .find(|info| info.vertex_matches().len() == case.expected_atom_matches)
        .expect("AllBest should contain the expected clique");

    let selected_only: Vec<usize> =
        selected.clique().iter().copied().filter(|v| !expected.clique().contains(v)).collect();
    let expected_only: Vec<usize> =
        expected.clique().iter().copied().filter(|v| !selected.clique().contains(v)).collect();

    println!("case: {}", case.name);
    println!("selected_only_vertices={selected_only:?}");
    for vertex in selected_only {
        let (left, right) = diagnostics.vertex_pairs[vertex];
        println!(
            "  selected_only vertex={vertex} pair=({left},{right}) g1_edge={:?} g2_edge={:?}",
            diagnostics.first_edge_map[left], diagnostics.second_edge_map[right],
        );
    }

    println!("expected_only_vertices={expected_only:?}");
    for vertex in expected_only {
        let (left, right) = diagnostics.vertex_pairs[vertex];
        println!(
            "  expected_only vertex={vertex} pair=({left},{right}) g1_edge={:?} g2_edge={:?}",
            diagnostics.first_edge_map[left], diagnostics.second_edge_map[right],
        );
    }
}

#[test]
#[ignore = "manual diagnostic harness for rerunning PartialEnumeration after rejecting the current winner"]
fn print_massspecgym_case_partial_after_rejecting_current_winner() {
    let case_name = std::env::var("MCES_CASE").expect("set MCES_CASE to a MassSpecGym case name");

    let cases = load_massspecgym_ground_truth();
    let case = find_case(&cases, &case_name);
    let diagnostics = collect_labeled_case_product_diagnostics(case, true);
    let partial =
        run_labeled_case_with_search_mode(case, true, false, McesSearchMode::PartialEnumeration);
    let rejected = partial
        .all_cliques()
        .first()
        .expect("partial result should retain a clique")
        .clique()
        .to_vec();
    let (g1_label_indices, g2_label_indices, num_labels) =
        intern_case_bond_labels(&diagnostics.first_bond_labels, &diagnostics.second_bond_labels);
    let partition = PartitionInfo {
        pairs: &diagnostics.vertex_pairs,
        g1_labels: &g1_label_indices,
        g2_labels: &g2_label_indices,
        num_labels,
        partition_side: geometric_traits::traits::algorithms::maximum_clique::PartitionSide::First,
    };
    let rerun = geometric_traits::traits::algorithms::maximum_clique::partial_search(
        &diagnostics.matrix,
        &partition,
        0,
        |clique| {
            clique != rejected
                && !clique_has_delta_y_from_product(
                    clique,
                    &diagnostics.vertex_pairs,
                    &diagnostics.first_edge_map,
                    &diagnostics.second_edge_map,
                    case.graph1.n_atoms,
                    case.graph2.n_atoms,
                )
        },
    );
    let reranked = rank_partitioned_cliques(
        rerun,
        &diagnostics.vertex_pairs,
        &diagnostics.first_edge_map,
        &diagnostics.second_edge_map,
    );

    println!("case: {}", case.name);
    println!("rejected_clique={rejected:?}");
    println!(
        "rerun_retained={} top_atoms={} top_similarity={:.6}",
        reranked.len(),
        reranked.first().map_or(0, |info| info.vertex_matches().len()),
        reranked.first().map(|info| info_johnson_similarity(case, info)).unwrap_or(0.0),
    );
    if let Some(info) = reranked.first() {
        println!("rerun_top_clique={:?}", info.clique());
        println!("rerun_top_vertex_matches={:?}", info.vertex_matches());
    }
}

#[test]
#[ignore = "manual diagnostic harness for rerunning PartialEnumeration after rejecting the retained set"]
fn print_massspecgym_case_partial_after_rejecting_retained_set() {
    let case_name = std::env::var("MCES_CASE").expect("set MCES_CASE to a MassSpecGym case name");

    let cases = load_massspecgym_ground_truth();
    let case = find_case(&cases, &case_name);
    let diagnostics = collect_labeled_case_product_diagnostics(case, true);
    let partial =
        run_labeled_case_with_search_mode(case, true, false, McesSearchMode::PartialEnumeration);
    let rejected: Vec<Vec<usize>> =
        partial.all_cliques().iter().map(|info| info.clique().to_vec()).collect();
    let (g1_label_indices, g2_label_indices, num_labels) =
        intern_case_bond_labels(&diagnostics.first_bond_labels, &diagnostics.second_bond_labels);
    let partition = PartitionInfo {
        pairs: &diagnostics.vertex_pairs,
        g1_labels: &g1_label_indices,
        g2_labels: &g2_label_indices,
        num_labels,
        partition_side: geometric_traits::traits::algorithms::maximum_clique::PartitionSide::First,
    };
    let rerun = geometric_traits::traits::algorithms::maximum_clique::partial_search(
        &diagnostics.matrix,
        &partition,
        0,
        |clique| {
            !rejected.contains(&clique.to_vec())
                && !clique_has_delta_y_from_product(
                    clique,
                    &diagnostics.vertex_pairs,
                    &diagnostics.first_edge_map,
                    &diagnostics.second_edge_map,
                    case.graph1.n_atoms,
                    case.graph2.n_atoms,
                )
        },
    );
    let reranked = rank_partitioned_cliques(
        rerun,
        &diagnostics.vertex_pairs,
        &diagnostics.first_edge_map,
        &diagnostics.second_edge_map,
    );

    println!("case: {}", case.name);
    println!("rejected_retained_count={}", rejected.len());
    println!(
        "rerun_retained={} top_atoms={} top_similarity={:.6}",
        reranked.len(),
        reranked.first().map_or(0, |info| info.vertex_matches().len()),
        reranked.first().map(|info| info_johnson_similarity(case, info)).unwrap_or(0.0),
    );
    if let Some(info) = reranked.first() {
        println!("rerun_top_clique={:?}", info.clique());
        println!("rerun_top_vertex_matches={:?}", info.vertex_matches());
    }
}

#[test]
#[ignore = "manual diagnostic harness for walking successive PartialEnumeration retained horizons"]
fn print_massspecgym_case_partial_horizon_walk() {
    let case_name = std::env::var("MCES_CASE").expect("set MCES_CASE to a MassSpecGym case name");
    let max_steps: usize =
        std::env::var("MCES_HORIZON_STEPS").ok().and_then(|value| value.parse().ok()).unwrap_or(8);

    let cases = load_massspecgym_ground_truth();
    let case = find_case(&cases, &case_name);
    let diagnostics = collect_labeled_case_product_diagnostics(case, true);
    let (g1_label_indices, g2_label_indices, num_labels) =
        intern_case_bond_labels(&diagnostics.first_bond_labels, &diagnostics.second_bond_labels);
    let partition = PartitionInfo {
        pairs: &diagnostics.vertex_pairs,
        g1_labels: &g1_label_indices,
        g2_labels: &g2_label_indices,
        num_labels,
        partition_side: geometric_traits::traits::algorithms::maximum_clique::PartitionSide::First,
    };
    let mut rejected: Vec<Vec<usize>> = Vec::new();

    println!("case: {}", case.name);
    println!(
        "expected bonds={} atoms={} similarity={:.6}",
        case.expected_bond_matches, case.expected_atom_matches, case.expected_similarity
    );

    for step in 0..max_steps {
        let rerun = geometric_traits::traits::algorithms::maximum_clique::partial_search(
            &diagnostics.matrix,
            &partition,
            0,
            |clique| {
                !rejected.contains(&clique.to_vec())
                    && !clique_has_delta_y_from_product(
                        clique,
                        &diagnostics.vertex_pairs,
                        &diagnostics.first_edge_map,
                        &diagnostics.second_edge_map,
                        case.graph1.n_atoms,
                        case.graph2.n_atoms,
                    )
            },
        );
        let reranked = rank_partitioned_cliques(
            rerun,
            &diagnostics.vertex_pairs,
            &diagnostics.first_edge_map,
            &diagnostics.second_edge_map,
        );

        if reranked.is_empty() {
            println!("step={step} retained=0 rejected_total={}", rejected.len());
            break;
        }

        let expected_index = reranked.iter().position(|info| {
            info.matched_edges().len() == case.expected_bond_matches
                && info.vertex_matches().len() == case.expected_atom_matches
                && (info_johnson_similarity(case, info) - case.expected_similarity).abs() < 1e-6
        });
        let top = &reranked[0];
        let top_similarity = info_johnson_similarity(case, top);
        println!(
            "step={step} retained={} rejected_total={} top_bonds={} top_atoms={} top_similarity={:.6} expected_index={expected_index:?}",
            reranked.len(),
            rejected.len(),
            top.matched_edges().len(),
            top.vertex_matches().len(),
            top_similarity,
        );
        if let Some(index) = expected_index {
            println!("expected_found_at_step={step} retained_index={index}");
            println!("expected_clique={:?}", reranked[index].clique());
            break;
        }

        rejected.extend(reranked.into_iter().map(|info| info.clique().to_vec()));
    }
}

#[test]
#[ignore = "manual diagnostic harness for counting distinct result signatures in PartialEnumeration"]
fn print_massspecgym_case_partial_result_signature_counts() {
    use std::collections::BTreeMap;

    let case_name = std::env::var("MCES_CASE").expect("set MCES_CASE to a MassSpecGym case name");

    let cases = load_massspecgym_ground_truth();
    let case = find_case(&cases, &case_name);
    let partial =
        run_labeled_case_with_search_mode(case, true, false, McesSearchMode::PartialEnumeration);
    let mut counts: BTreeMap<Vec<(usize, usize)>, (usize, usize, f64)> = BTreeMap::new();

    for info in partial.all_cliques() {
        let key = info.vertex_matches().to_vec();
        let entry = counts.entry(key.clone()).or_insert((
            0,
            info.matched_edges().len(),
            info_johnson_similarity(case, info),
        ));
        entry.0 += 1;
    }

    println!("case: {}", case.name);
    println!(
        "retained_cliques={} distinct_result_signatures={}",
        partial.all_cliques().len(),
        counts.len(),
    );
    for (index, (vertex_matches, (count, bonds, similarity))) in counts.iter().enumerate() {
        println!(
            "  signature#{index}: cliques={} bonds={} atoms={} similarity={:.6}",
            count,
            bonds,
            vertex_matches.len(),
            similarity,
        );
        if index >= 9 {
            break;
        }
    }
}

#[test]
#[ignore = "manual diagnostic harness for equivalent-root pruning on PartialEnumeration"]
fn print_massspecgym_case_equivalent_root_pruning_probe() {
    let case_name = std::env::var("MCES_CASE").expect("set MCES_CASE to a MassSpecGym case name");

    let cases = load_massspecgym_ground_truth();
    let case = find_case(&cases, &case_name);
    let diagnostics = collect_labeled_case_product_diagnostics(case, true);
    let partial =
        run_labeled_case_with_search_mode(case, true, false, McesSearchMode::PartialEnumeration);
    let first_equiv = heuristic_equivalent_bond_classes(&case.graph1, &diagnostics.first_edge_map);
    let second_equiv =
        heuristic_equivalent_bond_classes(&case.graph2, &diagnostics.second_edge_map);
    let equivalent_root_vertices = diagnostics
        .vertex_pairs
        .iter()
        .filter(|&&(g1, g2)| first_equiv[g1] >= 0 && second_equiv[g2] >= 0)
        .count();
    let (g1_label_indices, g2_label_indices, num_labels) =
        intern_case_bond_labels(&diagnostics.first_bond_labels, &diagnostics.second_bond_labels);
    let partition = PartitionInfo {
        pairs: &diagnostics.vertex_pairs,
        g1_labels: &g1_label_indices,
        g2_labels: &g2_label_indices,
        num_labels,
        partition_side: geometric_traits::traits::algorithms::maximum_clique::PartitionSide::First,
    };

    let mut seen_root_classes = BTreeSet::<(i32, i32)>::new();
    let mut skipped_roots = 0usize;
    let rerun =
        geometric_traits::traits::algorithms::maximum_clique::partial_search_with_root_pruning(
            &diagnostics.matrix,
            &partition,
            0,
            |clique| {
                !clique_has_delta_y_from_product(
                    clique,
                    &diagnostics.vertex_pairs,
                    &diagnostics.first_edge_map,
                    &diagnostics.second_edge_map,
                    case.graph1.n_atoms,
                    case.graph2.n_atoms,
                )
            },
            |root_vertex| {
                let (g1, g2) = diagnostics.vertex_pairs[root_vertex];
                let classes = (first_equiv[g1], second_equiv[g2]);
                if classes.0 < 0 || classes.1 < 0 {
                    return false;
                }
                let duplicate = !seen_root_classes.insert(classes);
                if duplicate {
                    skipped_roots += 1;
                }
                duplicate
            },
        );
    let reranked = rank_partitioned_cliques(
        rerun,
        &diagnostics.vertex_pairs,
        &diagnostics.first_edge_map,
        &diagnostics.second_edge_map,
    );
    let expected_index = reranked.iter().position(|info| {
        info.matched_edges().len() == case.expected_bond_matches
            && info.vertex_matches().len() == case.expected_atom_matches
            && (info_johnson_similarity(case, info) - case.expected_similarity).abs() < 1e-6
    });

    println!("case: {}", case.name);
    println!(
        "equivalent_root_vertices={} skipped_roots={}",
        equivalent_root_vertices, skipped_roots
    );
    println!(
        "baseline_retained={} baseline_top_atoms={} baseline_top_similarity={:.6}",
        partial.all_cliques().len(),
        partial.all_cliques().first().map_or(0, |info| info.vertex_matches().len()),
        partial
            .all_cliques()
            .first()
            .map(|info| info_johnson_similarity(case, info))
            .unwrap_or(0.0),
    );
    println!(
        "root_pruned_retained={} root_pruned_top_atoms={} root_pruned_top_similarity={:.6} expected_index={expected_index:?}",
        reranked.len(),
        reranked.first().map_or(0, |info| info.vertex_matches().len()),
        reranked.first().map(|info| info_johnson_similarity(case, info)).unwrap_or(0.0),
    );
    if let Some(info) = reranked.first() {
        println!("root_pruned_top_clique={:?}", info.clique());
        println!("root_pruned_top_vertex_matches={:?}", info.vertex_matches());
    }
}

#[test]
#[ignore = "manual diagnostic harness for sweeping the initial lower bound in PartialEnumeration"]
fn print_massspecgym_case_partial_lower_bound_sweep() {
    let case_name = std::env::var("MCES_CASE").expect("set MCES_CASE to a MassSpecGym case name");
    let max_bound: usize = std::env::var("MCES_MAX_LOWER_BOUND")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or_else(|| {
            load_massspecgym_ground_truth()
                .iter()
                .find(|case| case.name == case_name)
                .map(|case| case.expected_bond_matches)
                .unwrap_or(0)
        });

    let cases = load_massspecgym_ground_truth();
    let case = find_case(&cases, &case_name);
    let diagnostics = collect_labeled_case_product_diagnostics(case, true);
    let (g1_label_indices, g2_label_indices, num_labels) =
        intern_case_bond_labels(&diagnostics.first_bond_labels, &diagnostics.second_bond_labels);
    let partition = PartitionInfo {
        pairs: &diagnostics.vertex_pairs,
        g1_labels: &g1_label_indices,
        g2_labels: &g2_label_indices,
        num_labels,
        partition_side: geometric_traits::traits::algorithms::maximum_clique::PartitionSide::First,
    };

    println!("case: {}", case.name);
    println!(
        "expected bonds={} atoms={} similarity={:.6}",
        case.expected_bond_matches, case.expected_atom_matches, case.expected_similarity
    );

    for lower_bound in 0..=max_bound {
        let rerun = geometric_traits::traits::algorithms::maximum_clique::partial_search(
            &diagnostics.matrix,
            &partition,
            lower_bound,
            |clique| {
                !clique_has_delta_y_from_product(
                    clique,
                    &diagnostics.vertex_pairs,
                    &diagnostics.first_edge_map,
                    &diagnostics.second_edge_map,
                    case.graph1.n_atoms,
                    case.graph2.n_atoms,
                )
            },
        );
        let reranked = rank_partitioned_cliques(
            rerun,
            &diagnostics.vertex_pairs,
            &diagnostics.first_edge_map,
            &diagnostics.second_edge_map,
        );
        let expected_index = reranked.iter().position(|info| {
            info.matched_edges().len() == case.expected_bond_matches
                && info.vertex_matches().len() == case.expected_atom_matches
                && (info_johnson_similarity(case, info) - case.expected_similarity).abs() < 1e-6
        });
        println!(
            "lower_bound={lower_bound} retained={} top_bonds={} top_atoms={} top_similarity={:.6} expected_index={expected_index:?}",
            reranked.len(),
            reranked.first().map_or(0, |info| info.matched_edges().len()),
            reranked.first().map_or(0, |info| info.vertex_matches().len()),
            reranked.first().map(|info| info_johnson_similarity(case, info)).unwrap_or(0.0),
        );
    }
}

#[test]
#[ignore = "manual diagnostic harness for sweeping partition side in PartialEnumeration"]
fn print_massspecgym_case_partial_partition_side_sweep() {
    let case_name = std::env::var("MCES_CASE").expect("set MCES_CASE to a MassSpecGym case name");
    let lower_bound: usize = std::env::var("MCES_PARTITION_SIDE_LOWER_BOUND")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(0);

    let cases = load_massspecgym_ground_truth();
    let case = find_case(&cases, &case_name);
    let diagnostics = collect_labeled_case_product_diagnostics(case, true);
    let (g1_label_indices, g2_label_indices, num_labels) =
        intern_case_bond_labels(&diagnostics.first_bond_labels, &diagnostics.second_bond_labels);
    let heuristic_side =
        geometric_traits::traits::algorithms::maximum_clique::choose_partition_side(
            &diagnostics.vertex_pairs,
            diagnostics.first_edge_map.len(),
            diagnostics.second_edge_map.len(),
        );

    println!("case: {}", case.name);
    println!(
        "expected bonds={} atoms={} similarity={:.6} lower_bound={lower_bound} heuristic_side={heuristic_side:?}",
        case.expected_bond_matches, case.expected_atom_matches, case.expected_similarity
    );

    for side in [
        geometric_traits::traits::algorithms::maximum_clique::PartitionSide::First,
        geometric_traits::traits::algorithms::maximum_clique::PartitionSide::Second,
    ] {
        let partition = PartitionInfo {
            pairs: &diagnostics.vertex_pairs,
            g1_labels: &g1_label_indices,
            g2_labels: &g2_label_indices,
            num_labels,
            partition_side: side,
        };
        let rerun = geometric_traits::traits::algorithms::maximum_clique::partial_search(
            &diagnostics.matrix,
            &partition,
            lower_bound,
            |clique| {
                !clique_has_delta_y_from_product(
                    clique,
                    &diagnostics.vertex_pairs,
                    &diagnostics.first_edge_map,
                    &diagnostics.second_edge_map,
                    case.graph1.n_atoms,
                    case.graph2.n_atoms,
                )
            },
        );
        let reranked = rank_partitioned_cliques(
            rerun,
            &diagnostics.vertex_pairs,
            &diagnostics.first_edge_map,
            &diagnostics.second_edge_map,
        );
        let expected_index = reranked.iter().position(|info| {
            info.matched_edges().len() == case.expected_bond_matches
                && info.vertex_matches().len() == case.expected_atom_matches
                && (info_johnson_similarity(case, info) - case.expected_similarity).abs() < 1e-6
        });
        println!(
            "side={side:?} retained={} top_bonds={} top_atoms={} top_similarity={:.6} expected_index={expected_index:?}",
            reranked.len(),
            reranked.first().map_or(0, |info| info.matched_edges().len()),
            reranked.first().map_or(0, |info| info.vertex_matches().len()),
            reranked.first().map(|info| info_johnson_similarity(case, info)).unwrap_or(0.0),
        );
    }
}

#[test]
#[ignore = "manual diagnostic harness for rerunning a case with the graph order swapped"]
fn print_massspecgym_case_swapped_graph_order() {
    let case_name = std::env::var("MCES_CASE").expect("set MCES_CASE to a MassSpecGym case name");

    let cases = load_massspecgym_ground_truth();
    let case = find_case(&cases, &case_name);
    let original = run_labeled_case(case);
    let prepared_swapped = prepare_labeled_case_from_graph_data(case, &case.graph2, &case.graph1);
    let swapped = McesBuilder::new(&prepared_swapped.first, &prepared_swapped.second)
        .with_largest_fragment_metric(LargestFragmentMetric::Atoms)
        .compute_labeled();

    println!("case: {}", case.name);
    println!(
        "graph1 atoms={} bonds={} graph2 atoms={} bonds={}",
        case.graph1.n_atoms,
        case.graph1.edges.len(),
        case.graph2.n_atoms,
        case.graph2.edges.len(),
    );
    println!(
        "expected bonds={} atoms={} similarity={:.6}",
        case.expected_bond_matches, case.expected_atom_matches, case.expected_similarity
    );
    println!(
        "original bonds={} atoms={} similarity={:.6}",
        original.matched_edges().len(),
        original.vertex_matches().len(),
        original.johnson_similarity(),
    );
    println!(
        "swapped bonds={} atoms={} similarity={:.6}",
        swapped.matched_edges().len(),
        swapped.vertex_matches().len(),
        swapped.johnson_similarity(),
    );
}

#[test]
#[ignore = "manual diagnostic harness for tracing PartialEnumeration branch order toward a target clique"]
fn print_massspecgym_case_partial_trace_to_target() {
    let case_name = std::env::var("MCES_CASE").expect("set MCES_CASE to a MassSpecGym case name");
    let event_limit: usize =
        std::env::var("MCES_EVENT_LIMIT").ok().and_then(|value| value.parse().ok()).unwrap_or(120);
    let use_partition_orientation_heuristic =
        std::env::var("MCES_USE_PARTITION_ORIENTATION_HEURISTIC")
            .ok()
            .map(|value| !matches!(value.as_str(), "0" | "false" | "FALSE"))
            .unwrap_or(false);

    let cases = load_massspecgym_ground_truth();
    let case = find_case(&cases, &case_name);
    let prepared = prepare_labeled_case(case);
    let lg1 = prepared.first.labeled_line_graph();
    let lg2 = prepared.second.labeled_line_graph();
    let first_bond_labels = compute_case_bond_labels(&prepared.first, lg1.edge_map());
    let second_bond_labels = compute_case_bond_labels(&prepared.second, lg2.edge_map());
    let (g1_label_indices, g2_label_indices, num_labels) =
        intern_case_bond_labels(&first_bond_labels, &second_bond_labels);
    let use_edge_contexts = case_uses_complete_aromatic_rings(case);
    let mp = lg1.graph().labeled_modular_product_filtered(
        lg2.graph(),
        |i, j| {
            let contexts_match = match (&prepared.first_contexts, &prepared.second_contexts) {
                (Some(first_contexts), Some(second_contexts)) if use_edge_contexts => {
                    first_contexts.compatible_with(i, second_contexts, j)
                }
                _ => true,
            };
            g1_label_indices[i] == g2_label_indices[j] && contexts_match
        },
        |left, right| left == right,
    );

    let partition = PartitionInfo {
        pairs: mp.vertex_pairs(),
        g1_labels: &g1_label_indices,
        g2_labels: &g2_label_indices,
        num_labels,
        partition_side: if use_partition_orientation_heuristic {
            geometric_traits::traits::algorithms::maximum_clique::choose_partition_side(
                mp.vertex_pairs(),
                g1_label_indices.len(),
                g2_label_indices.len(),
            )
        } else {
            geometric_traits::traits::algorithms::maximum_clique::PartitionSide::First
        },
    };

    let all_best = run_labeled_case_with_search_mode(case, true, false, McesSearchMode::AllBest);
    let target_index = all_best
        .all_cliques()
        .iter()
        .position(|info| info.vertex_matches().len() == case.expected_atom_matches)
        .expect("expected clique must be present in AllBest");
    let target_clique = all_best.all_cliques()[target_index].clique().to_vec();

    let trace =
        geometric_traits::traits::algorithms::maximum_clique::trace_partial_search_to_target(
            mp.matrix(),
            &partition,
            0,
            &target_clique,
            |clique| {
                !clique_has_delta_y_from_product(
                    clique,
                    mp.vertex_pairs(),
                    lg1.edge_map(),
                    lg2.edge_map(),
                    case.graph1.n_atoms,
                    case.graph2.n_atoms,
                )
            },
        );

    println!("case: {}", case.name);
    println!(
        "target_index={} target_clique={:?} target_atoms={} orientation_heuristic={}",
        target_index,
        target_clique,
        case.expected_atom_matches,
        use_partition_orientation_heuristic,
    );
    println!("trace best_cliques={:?}", trace.best_cliques);
    println!("trace events={}", trace.events.len());
    for event in trace.events.iter().take(event_limit) {
        println!("{event}");
    }
    if trace.events.len() > event_limit {
        println!("... {} more events", trace.events.len() - event_limit);
    }
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
