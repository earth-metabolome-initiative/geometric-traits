//! Fuzz the canonizer against a local `bliss` CLI oracle.
//!
//! This target is intentionally slower than the pure in-process fuzzers: every
//! input is checked both against the Rust canonizer and an external `bliss`
//! process. In return, it can keep pressure on behavioral alignment while
//! exploring larger simple undirected fully labeled graphs.

use std::{
    env, fs,
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
    time::{SystemTime, UNIX_EPOCH},
};

use arbitrary::Arbitrary;
use geometric_traits::{
    impls::{SortedVec, SymmetricCSR2D, ValuedCSR2D},
    naive_structs::GenericGraph,
    prelude::{GenericVocabularyBuilder, canonical_label_labeled_simple_graph},
    traits::{Edges, MonoplexGraph, SparseValuedMatrix2D, VocabularyBuilder},
};
use honggfuzz::fuzz;

mod canon_bliss_support;

use canon_bliss_support::{
    encode_labeled_simple_graph_as_dimacs, run_bliss_on_dimacs_file,
};

type LabeledUndirectedEdges = SymmetricCSR2D<ValuedCSR2D<usize, usize, usize, u8>>;
type LabeledUndirectedGraph = GenericGraph<SortedVec<usize>, LabeledUndirectedEdges>;

const MAX_VERTEX_COUNT: usize = 25;
const MAX_LABEL_ALPHABET: u8 = 6;
static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

#[derive(Arbitrary, Debug)]
struct FuzzCanonBlissCase {
    vertex_count_hint: u8,
    vertex_label_alphabet_hint: u8,
    edge_label_alphabet_hint: u8,
    edge_density_hint: u8,
    vertex_label_bytes: Vec<u8>,
    edge_bytes: Vec<u8>,
}

fn main() {
    let bliss = locate_bliss_binary().expect(
        "canon_bliss requires a local bliss executable; set GEOMETRIC_TRAITS_BLISS_BIN or build /tmp/bliss-build/bliss",
    );
    loop {
        fuzz!(|case: FuzzCanonBlissCase| {
            check_case(&bliss, &case);
        });
    }
}

fn locate_bliss_binary() -> Option<PathBuf> {
    if let Ok(path) = env::var("GEOMETRIC_TRAITS_BLISS_BIN") {
        let candidate = PathBuf::from(path);
        if candidate.is_file() {
            return Some(candidate);
        }
    }

    let fuzz_manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let repo_root = fuzz_manifest_dir
        .parent()
        .expect("fuzz crate should live under the repo root");
    let candidates = [
        PathBuf::from("/tmp/bliss-build/bliss"),
        repo_root.join("papers/software/bliss-0.77/build/bliss"),
        repo_root.join("papers/software/bliss-0.77/bliss"),
    ];

    candidates.into_iter().find(|candidate| candidate.is_file())
}

fn check_case(bliss: &Path, case: &FuzzCanonBlissCase) {
    let (vertex_labels, edges) = materialize_case(case);
    let graph = build_bidirectional_labeled_graph(vertex_labels.len(), &edges);
    let matrix = Edges::matrix(graph.edges());
    let rust_result = canonical_label_labeled_simple_graph(
        &graph,
        |node| vertex_labels[node],
        |left, right| matrix.sparse_value_at(left, right).unwrap(),
    );
    let rust_certificate = certificate_from_order(&graph, &vertex_labels, &rust_result.order);

    let encoded = encode_labeled_simple_graph_as_dimacs(&vertex_labels, &edges)
        .expect("normalized fuzz case should encode as bliss DIMACS");
    let temp_dir = make_temp_dir();
    let input_path = temp_dir.join("input.dimacs");
    let canonical_path = temp_dir.join("canonical.dimacs");
    fs::write(&input_path, &encoded.dimacs).expect("fuzz oracle input should write");
    let bliss_result = run_bliss_on_dimacs_file(
        bliss,
        &input_path,
        &canonical_path,
        encoded.expanded_vertex_count,
        encoded.original_vertex_count,
    )
    .expect("bliss oracle run should succeed");
    let _ = fs::remove_dir_all(&temp_dir);

    let bliss_certificate =
        certificate_from_order(&graph, &vertex_labels, &bliss_result.original_canonical_order);
    assert_eq!(
        rust_certificate,
        bliss_certificate,
        "canon_bliss certificate mismatch: vertex_labels={vertex_labels:?} edges={edges:?} rust_order={:?} bliss_order={:?}",
        rust_result.order,
        bliss_result.original_canonical_order,
    );

    let bliss_nodes = bliss_result
        .stats
        .nodes
        .expect("bliss fuzz oracle should report search-node count");
    let bliss_leaf_nodes = bliss_result
        .stats
        .leaf_nodes
        .expect("bliss fuzz oracle should report leaf-node count");
    assert_eq!(
        (rust_result.stats.search_nodes, rust_result.stats.leaf_nodes),
        (bliss_nodes, bliss_leaf_nodes),
        "canon_bliss stats mismatch: vertex_labels={vertex_labels:?} edges={edges:?} rust_stats={:?} bliss_stats={:?}",
        rust_result.stats,
        bliss_result.stats,
    );
}

fn make_temp_dir() -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock should be after unix epoch")
        .as_nanos();
    let counter = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    let temp_dir = env::temp_dir().join(format!(
        "geometric-traits-canon-bliss-fuzz-{nanos}-{}-{counter}",
        std::process::id()
    ));
    fs::create_dir_all(&temp_dir).expect("canon_bliss temp dir should be creatable");
    temp_dir
}

fn materialize_case(case: &FuzzCanonBlissCase) -> (Vec<u8>, Vec<(usize, usize, u8)>) {
    let vertex_count = usize::from(case.vertex_count_hint) % MAX_VERTEX_COUNT + 1;
    let vertex_label_mod = case.vertex_label_alphabet_hint % MAX_LABEL_ALPHABET + 1;
    let edge_label_mod = case.edge_label_alphabet_hint % MAX_LABEL_ALPHABET + 1;

    let vertex_labels = if case.vertex_label_bytes.is_empty() {
        vec![0_u8; vertex_count]
    } else {
        (0..vertex_count)
            .map(|index| case.vertex_label_bytes[index % case.vertex_label_bytes.len()] % vertex_label_mod)
            .collect::<Vec<_>>()
    };

    let mut edges = Vec::new();
    if case.edge_bytes.is_empty() {
        return (vertex_labels, edges);
    }

    let mut cursor = 0usize;
    for left in 0..vertex_count {
        for right in (left + 1)..vertex_count {
            let selector = case.edge_bytes[cursor % case.edge_bytes.len()];
            cursor += 1;
            if selector > case.edge_density_hint {
                continue;
            }
            let label = case.edge_bytes[cursor % case.edge_bytes.len()] % edge_label_mod;
            cursor += 1;
            edges.push((left, right, label));
        }
    }

    (vertex_labels, edges)
}

fn build_bidirectional_labeled_graph(
    number_of_nodes: usize,
    edges: &[(usize, usize, u8)],
) -> LabeledUndirectedGraph {
    let nodes: SortedVec<usize> = GenericVocabularyBuilder::default()
        .expected_number_of_symbols(number_of_nodes)
        .symbols((0..number_of_nodes).enumerate())
        .build()
        .expect("dense graph vocabulary should build");
    let mut upper_edges: Vec<(usize, usize, u8)> = edges
        .iter()
        .map(|&(left, right, label)| {
            if left <= right {
                (left, right, label)
            } else {
                (right, left, label)
            }
        })
        .collect();
    upper_edges.sort_unstable_by(|left, right| {
        left.0.cmp(&right.0).then(left.1.cmp(&right.1)).then(left.2.cmp(&right.2))
    });
    upper_edges.dedup();
    let edges: LabeledUndirectedEdges =
        SymmetricCSR2D::from_sorted_upper_triangular_entries(number_of_nodes, upper_edges)
            .expect("fuzz target should build a simple undirected graph");

    GenericGraph::from((nodes, edges))
}

fn certificate_from_order(
    graph: &LabeledUndirectedGraph,
    vertex_labels: &[u8],
    order: &[usize],
) -> (Vec<u8>, Vec<Option<u8>>) {
    let matrix = Edges::matrix(graph.edges());
    let ordered_vertex_labels =
        order.iter().map(|&vertex| vertex_labels[vertex]).collect::<Vec<_>>();
    let mut upper_triangle_edge_labels = Vec::new();

    for left in 0..order.len() {
        for right in (left + 1)..order.len() {
            upper_triangle_edge_labels.push(matrix.sparse_value_at(order[left], order[right]));
        }
    }

    (ordered_vertex_labels, upper_triangle_edge_labels)
}
