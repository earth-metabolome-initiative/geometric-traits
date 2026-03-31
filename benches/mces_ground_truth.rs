//! Criterion benchmarks for labeled MCES on the RDKit-derived ground-truth
//! fixture suite.

use std::{hint::black_box, io::Read as _, time::Duration};

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use geometric_traits::{
    impls::{EdgeContexts, SortedVec, SquareCSR2D, SymmetricCSR2D, ValuedCSR2D},
    prelude::*,
    traits::{MatrixMut, SparseMatrixMut, TypedNode, VocabularyBuilder},
};

/// A node labeled by a shared atom-type index.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct AtomNode {
    id: usize,
    atom_type: u8,
}

impl TypedNode for AtomNode {
    type NodeType = u8;

    fn node_type(&self) -> Self::NodeType {
        self.atom_type
    }
}

type TypedGraph = geometric_traits::naive_structs::GenericGraph<
    SortedVec<AtomNode>,
    SymmetricCSR2D<ValuedCSR2D<usize, usize, usize, u32>>,
>;

#[derive(serde::Deserialize)]
struct GroundTruthFile {
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
}

#[derive(serde::Deserialize)]
struct GroundTruthCase {
    name: String,
    graph1: GraphData,
    graph2: GraphData,
    timed_out: bool,
    options: Option<serde_json::Value>,
}

struct PreparedLabeledCase {
    name: String,
    first: TypedGraph,
    second: TypedGraph,
    first_contexts: Option<EdgeContexts<String>>,
    second_contexts: Option<EdgeContexts<String>>,
    ignore_edge_values: bool,
    use_edge_contexts: bool,
}

static GROUND_TRUTH_GZ: &[u8] = include_bytes!("../tests/fixtures/mces_ground_truth.json.gz");

fn load_ground_truth() -> Vec<GroundTruthCase> {
    let mut decoder = flate2::read::GzDecoder::new(GROUND_TRUTH_GZ);
    let mut json = String::new();
    decoder.read_to_string(&mut json).unwrap();
    serde_json::from_str::<GroundTruthFile>(&json).unwrap().cases
}

fn atom_type_to_shared_indices(
    first_atom_types: &[String],
    second_atom_types: &[String],
) -> (Vec<u8>, Vec<u8>) {
    let mut unique: Vec<&str> =
        first_atom_types.iter().chain(second_atom_types.iter()).map(String::as_str).collect();
    unique.sort_unstable();
    unique.dedup();
    assert!(unique.len() <= (u8::MAX as usize) + 1);

    let remap = |atom_types: &[String]| {
        atom_types
            .iter()
            .map(|atom_type| {
                unique.iter().position(|candidate| *candidate == atom_type.as_str()).unwrap() as u8
            })
            .collect::<Vec<_>>()
    };

    (remap(first_atom_types), remap(second_atom_types))
}

fn build_typed_graph(
    n_atoms: usize,
    edges: &[[usize; 2]],
    atom_type_indices: &[u8],
    bond_types: &[u32],
) -> TypedGraph {
    let nodes_vec: Vec<AtomNode> =
        (0..n_atoms).map(|id| AtomNode { id, atom_type: atom_type_indices[id] }).collect();
    let nodes: SortedVec<AtomNode> = GenericVocabularyBuilder::default()
        .expected_number_of_symbols(n_atoms)
        .symbols(nodes_vec.into_iter().enumerate())
        .build()
        .unwrap();

    let mut sorted: Vec<(usize, usize, u32)> = edges
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
    sorted.sort_unstable_by(|left, right| {
        left.0.cmp(&right.0).then(left.1.cmp(&right.1)).then(left.2.cmp(&right.2))
    });
    sorted.dedup();

    let mut all_entries = Vec::with_capacity(sorted.len() * 2);
    for (src, dst, bond_type) in sorted {
        all_entries.push((src, dst, bond_type));
        all_entries.push((dst, src, bond_type));
    }
    all_entries.sort_unstable_by(|left, right| {
        left.0.cmp(&right.0).then(left.1.cmp(&right.1)).then(left.2.cmp(&right.2))
    });

    let mut valued: ValuedCSR2D<usize, usize, usize, u32> =
        SparseMatrixMut::with_sparse_shaped_capacity((n_atoms, n_atoms), all_entries.len());
    for (src, dst, bond_type) in all_entries {
        MatrixMut::add(&mut valued, (src, dst, bond_type)).unwrap();
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

fn prepare_labeled_case(case: GroundTruthCase) -> PreparedLabeledCase {
    let ignore_edge_values = case_ignores_edge_values(&case);
    let use_edge_contexts = case_uses_complete_aromatic_rings(&case);
    let (first_type_indices, second_type_indices) =
        atom_type_to_shared_indices(&case.graph1.atom_types, &case.graph2.atom_types);

    PreparedLabeledCase {
        name: case.name,
        first: build_typed_graph(
            case.graph1.n_atoms,
            &case.graph1.edges,
            &first_type_indices,
            &case.graph1.bond_types,
        ),
        second: build_typed_graph(
            case.graph2.n_atoms,
            &case.graph2.edges,
            &second_type_indices,
            &case.graph2.bond_types,
        ),
        first_contexts: build_edge_contexts(&case.graph1),
        second_contexts: build_edge_contexts(&case.graph2),
        ignore_edge_values,
        use_edge_contexts,
    }
}

fn run_prepared_labeled_case(case: &PreparedLabeledCase) -> McesResult<usize> {
    if case.use_edge_contexts {
        if let (Some(first_contexts), Some(second_contexts)) =
            (case.first_contexts.as_ref(), case.second_contexts.as_ref())
        {
            let builder = McesBuilder::new(&case.first, &case.second)
                .with_edge_contexts(first_contexts, second_contexts);
            let builder = if case.ignore_edge_values {
                builder.with_ignore_edge_values(true)
            } else {
                builder
            };
            return builder.compute_labeled();
        }
    }

    let builder = McesBuilder::new(&case.first, &case.second);
    let builder =
        if case.ignore_edge_values { builder.with_ignore_edge_values(true) } else { builder };
    builder.compute_labeled()
}

fn prepared_cases() -> Vec<PreparedLabeledCase> {
    load_ground_truth()
        .into_iter()
        .filter(|case| !case.timed_out)
        .map(prepare_labeled_case)
        .collect()
}

fn bench_mces_ground_truth(c: &mut Criterion) {
    let cases = prepared_cases();
    let mut group = c.benchmark_group("mces_ground_truth_labeled");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(2));

    for case in &cases {
        group.bench_with_input(BenchmarkId::new("stable", &case.name), case, |b, case| {
            b.iter(|| black_box(run_prepared_labeled_case(case)));
        });
    }

    group.finish();
}

criterion_group!(benches, bench_mces_ground_truth);
criterion_main!(benches);
