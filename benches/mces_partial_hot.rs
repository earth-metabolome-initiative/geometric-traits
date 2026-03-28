//! Criterion benchmarks for hot partitioned partial-search MCES cases.

use std::{
    collections::BTreeMap,
    hint::black_box,
    io::Read as _,
    time::Duration,
};

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use geometric_traits::{
    impls::{BitSquareMatrix, EdgeContexts, SortedVec, SquareCSR2D, SymmetricCSR2D, ValuedCSR2D},
    prelude::*,
    traits::{MatrixMut, SparseMatrixMut, TypedNode, VocabularyBuilder},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct GroundTruthNodeLabel {
    atom_type: u8,
    explicit_degree: Option<u8>,
    is_aromatic: Option<bool>,
}

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

#[derive(serde::Deserialize)]
struct GroundTruthFile {
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
    atom_is_aromatic: Vec<bool>,
    #[serde(default)]
    bond_original_indices: Vec<usize>,
}

#[derive(Clone, serde::Deserialize)]
struct GroundTruthCase {
    name: String,
    graph1: GraphData,
    graph2: GraphData,
    timed_out: bool,
    options: Option<serde_json::Value>,
}

struct PreparedLabeledCase {
    first: TypedGraph,
    second: TypedGraph,
    first_contexts: Option<EdgeContexts<String>>,
    second_contexts: Option<EdgeContexts<String>>,
}

struct LabeledCaseProductDiagnostics {
    matrix: BitSquareMatrix,
    vertex_pairs: Vec<(usize, usize)>,
    first_edge_map: Vec<(usize, usize)>,
    second_edge_map: Vec<(usize, usize)>,
    first_bond_labels: Vec<GroundTruthBondLabel>,
    second_bond_labels: Vec<GroundTruthBondLabel>,
}

struct PreparedHotBenchCase {
    name: String,
    matrix: BitSquareMatrix,
    vertex_pairs: Vec<(usize, usize)>,
    first_edge_map: Vec<(usize, usize)>,
    second_edge_map: Vec<(usize, usize)>,
    g1_labels: Vec<usize>,
    g2_labels: Vec<usize>,
    num_labels: usize,
    partition_side: geometric_traits::traits::algorithms::maximum_clique::PartitionSide,
    initial_lower_bound: usize,
    first_vertices: usize,
    second_vertices: usize,
}

impl PreparedHotBenchCase {
    fn partition_info(
        &self,
    ) -> geometric_traits::traits::algorithms::maximum_clique::PartitionInfo<'_> {
        geometric_traits::traits::algorithms::maximum_clique::PartitionInfo {
            pairs: &self.vertex_pairs,
            g1_labels: &self.g1_labels,
            g2_labels: &self.g2_labels,
            num_labels: self.num_labels,
            partition_side: self.partition_side,
        }
    }
}

static HOT_CASE_NAMES: &[&str] = &[
    "massspecgym_default_0594",
    "massspecgym_default_0631",
    "massspecgym_default_0939",
];

static MASSSPECGYM_GROUND_TRUTH_1000_GZ: &[u8] =
    include_bytes!("../tests/fixtures/massspecgym_mces_default_1000.json.gz");

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

fn explicit_degrees(n_atoms: usize, edges: &[(usize, usize, u32)]) -> Vec<u8> {
    let mut degrees = vec![0usize; n_atoms];
    for &(src, dst, _) in edges {
        degrees[src] += 1;
        degrees[dst] += 1;
    }
    degrees
        .into_iter()
        .map(|degree| u8::try_from(degree).expect("explicit degree must fit in u8"))
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
    let normalized_edges: Vec<(usize, usize, u32)> = edges
        .iter()
        .zip(bond_types.iter().copied())
        .map(|(edge, bond_type)| {
            if edge[0] < edge[1] {
                (edge[0], edge[1], bond_type)
            } else {
                (edge[1], edge[0], bond_type)
            }
        })
        .collect();
    let mut normalized_edges = normalized_edges;
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

fn build_edge_contexts(graph: &GraphData) -> Option<EdgeContexts<String>> {
    if graph.aromatic_ring_contexts.is_empty() {
        return None;
    }
    Some(EdgeContexts::from_rows(graph.aromatic_ring_contexts.iter().cloned()))
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

fn collect_prepared_labeled_case_product_diagnostics(
    case: &GroundTruthCase,
    prepared: &PreparedLabeledCase,
) -> LabeledCaseProductDiagnostics {
    let lg1 = prepared.first.labeled_line_graph();
    let lg2 = prepared.second.labeled_line_graph();
    let first_bond_labels = compute_case_bond_labels(&prepared.first, lg1.edge_map());
    let second_bond_labels = compute_case_bond_labels(&prepared.second, lg2.edge_map());
    let use_edge_contexts = case_uses_complete_aromatic_rings(case);

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

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct FixtureBondRecord {
    edge: (usize, usize),
    canonical_index: usize,
    original_index: usize,
}

fn canonical_edge(edge: [usize; 2]) -> (usize, usize) {
    if edge[0] <= edge[1] { (edge[0], edge[1]) } else { (edge[1], edge[0]) }
}

fn fixture_bond_records(graph: &GraphData) -> Vec<FixtureBondRecord> {
    let mut records: Vec<FixtureBondRecord> = graph
        .edges
        .iter()
        .enumerate()
        .map(|(canonical_index, &edge)| FixtureBondRecord {
            edge: canonical_edge(edge),
            canonical_index,
            original_index: graph
                .bond_original_indices
                .get(canonical_index)
                .copied()
                .unwrap_or(canonical_index),
        })
        .collect();
    records.sort_unstable_by_key(|record| (record.original_index, record.edge.0, record.edge.1));
    records
}

fn fixture_edge_rank_map(graph: &GraphData) -> BTreeMap<(usize, usize), usize> {
    let mut ranks = BTreeMap::new();
    for record in fixture_bond_records(graph) {
        ranks.entry(record.edge).or_insert(record.original_index);
    }
    ranks
}

fn product_order_rdkit_raw_pair_order(
    case: &GroundTruthCase,
    diagnostics: &LabeledCaseProductDiagnostics,
) -> Vec<usize> {
    let first_ranks = fixture_edge_rank_map(&case.graph1);
    let second_ranks = fixture_edge_rank_map(&case.graph2);
    let mut order: Vec<usize> = (0..diagnostics.vertex_pairs.len()).collect();
    order.sort_unstable_by_key(|&index| {
        let (first, second) = diagnostics.vertex_pairs[index];
        if case.graph1.n_atoms <= case.graph2.n_atoms {
            (
                first_ranks[&diagnostics.first_edge_map[first]],
                second_ranks[&diagnostics.second_edge_map[second]],
                index,
            )
        } else {
            (
                second_ranks[&diagnostics.second_edge_map[second]],
                first_ranks[&diagnostics.first_edge_map[first]],
                index,
            )
        }
    });
    order
}

fn permute_product(
    matrix: &BitSquareMatrix,
    vertex_pairs: &[(usize, usize)],
    order: &[usize],
) -> (BitSquareMatrix, Vec<(usize, usize)>) {
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

fn load_hot_cases() -> Vec<GroundTruthCase> {
    let mut decoder = flate2::read::GzDecoder::new(MASSSPECGYM_GROUND_TRUTH_1000_GZ);
    let mut json = String::new();
    decoder.read_to_string(&mut json).unwrap();
    let file: GroundTruthFile = serde_json::from_str(&json).unwrap();
    HOT_CASE_NAMES
        .iter()
        .map(|name| {
            file.cases
                .iter()
                .find(|case| !case.timed_out && case.name == *name)
                .unwrap_or_else(|| panic!("missing hot bench case '{name}'"))
                .clone()
        })
        .collect()
}

fn prepare_hot_bench_case(case: &GroundTruthCase) -> PreparedHotBenchCase {
    let prepared = prepare_labeled_case(case);
    let diagnostics = collect_prepared_labeled_case_product_diagnostics(case, &prepared);
    let order = product_order_rdkit_raw_pair_order(case, &diagnostics);
    let (matrix, vertex_pairs) = permute_product(&diagnostics.matrix, &diagnostics.vertex_pairs, &order);
    let (g1_labels, g2_labels, num_labels) =
        intern_case_bond_labels(&diagnostics.first_bond_labels, &diagnostics.second_bond_labels);
    let partition_side = geometric_traits::traits::algorithms::maximum_clique::choose_partition_side(
        &vertex_pairs,
        diagnostics.first_edge_map.len(),
        diagnostics.second_edge_map.len(),
    );
    PreparedHotBenchCase {
        name: case.name.clone(),
        matrix,
        vertex_pairs,
        first_edge_map: diagnostics.first_edge_map,
        second_edge_map: diagnostics.second_edge_map,
        g1_labels,
        g2_labels,
        num_labels,
        partition_side,
        initial_lower_bound: usize::from(!order.is_empty()),
        first_vertices: case.graph1.n_atoms,
        second_vertices: case.graph2.n_atoms,
    }
}

fn prepared_hot_cases() -> Vec<PreparedHotBenchCase> {
    load_hot_cases().iter().map(prepare_hot_bench_case).collect()
}

fn run_scalar(case: &PreparedHotBenchCase) -> geometric_traits::traits::algorithms::maximum_clique::PartitionSearchProfile {
    let partition = case.partition_info();
    geometric_traits::traits::algorithms::maximum_clique::profile_search_with_bounds(
        &case.matrix,
        &partition,
        false,
        case.initial_lower_bound,
        case.initial_lower_bound,
        |clique| {
            !clique_has_delta_y_from_product(
                clique,
                &case.vertex_pairs,
                &case.first_edge_map,
                &case.second_edge_map,
                case.first_vertices,
                case.second_vertices,
            )
        },
    )
}

fn run_u32(case: &PreparedHotBenchCase) -> geometric_traits::traits::algorithms::maximum_clique::PartitionSearchProfile {
    let partition = case.partition_info();
    geometric_traits::traits::algorithms::maximum_clique::experimental_profile_partial_search_u32(
        &case.matrix,
        &partition,
        case.initial_lower_bound,
        |clique| {
            !clique_has_delta_y_from_product(
                clique,
                &case.vertex_pairs,
                &case.first_edge_map,
                &case.second_edge_map,
                case.first_vertices,
                case.second_vertices,
            )
        },
    )
}

fn bench_mces_partial_hot(c: &mut Criterion) {
    let cases = prepared_hot_cases();
    let mut group = c.benchmark_group("mces_partial_hot");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(5));

    for case in &cases {
        group.bench_with_input(BenchmarkId::new("scalar", &case.name), case, |b, case| {
            b.iter(|| black_box(run_scalar(case)));
        });
        group.bench_with_input(BenchmarkId::new("u32", &case.name), case, |b, case| {
            b.iter(|| black_box(run_u32(case)));
        });
    }

    group.finish();
}

criterion_group!(benches, bench_mces_partial_hot);
criterion_main!(benches);
