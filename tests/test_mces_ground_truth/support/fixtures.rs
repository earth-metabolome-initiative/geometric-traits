use super::*;

/// Fixture-loading helpers for the MCES ground-truth test suite.

#[derive(serde::Deserialize)]
pub(crate) struct GroundTruthFile {
    pub(crate) version: u32,
    pub(crate) cases: Vec<GroundTruthCase>,
}

#[derive(Clone, serde::Deserialize)]
pub(crate) struct GraphData {
    pub(crate) n_atoms: usize,
    pub(crate) edges: Vec<[usize; 2]>,
    pub(crate) atom_types: Vec<String>,
    pub(crate) bond_types: Vec<u32>,
    #[serde(default)]
    pub(crate) aromatic_ring_contexts: Vec<Vec<String>>,
    pub(crate) atom_is_aromatic: Vec<bool>,
    #[serde(default)]
    pub(crate) bond_original_indices: Vec<usize>,
}

#[derive(serde::Deserialize)]
pub(crate) struct GroundTruthCase {
    pub(crate) name: String,
    pub(crate) graph1: GraphData,
    pub(crate) graph2: GraphData,
    pub(crate) expected_bond_matches: usize,
    pub(crate) expected_similarity: f64,
    pub(crate) timed_out: bool,
    pub(crate) options: Option<serde_json::Value>,
}

pub(crate) static GROUND_TRUTH_GZ: &[u8] =
    include_bytes!("../../fixtures/mces_ground_truth.json.gz");
const MASSSPECGYM_GROUND_TRUTH_100_PATH: &str =
    concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/massspecgym_mces_default_100.json.gz");
const MASSSPECGYM_GROUND_TRUTH_1000_PATH: &str =
    concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/massspecgym_mces_default_1000.json.gz");
const MASSSPECGYM_GROUND_TRUTH_10000_PATH: &str =
    concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/massspecgym_mces_default_10000.json.gz");
const MASSSPECGYM_ALL_BEST_GROUND_TRUTH_100_PATH: &str =
    concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/massspecgym_mces_all_best_100.json.gz");
pub(crate) fn load_ground_truth_from_bytes(gz_bytes: &[u8]) -> Vec<GroundTruthCase> {
    let mut decoder = flate2::read::GzDecoder::new(gz_bytes);
    let mut json_str = String::new();
    decoder.read_to_string(&mut json_str).unwrap();
    let file: GroundTruthFile = serde_json::from_str(&json_str).unwrap();
    assert!(file.version >= 1);
    file.cases
}

pub(crate) fn load_ground_truth_from_path(path: &str) -> Vec<GroundTruthCase> {
    let file =
        File::open(path).unwrap_or_else(|error| panic!("failed to open fixture {path}: {error}"));
    let mut decoder = flate2::read::GzDecoder::new(file);
    let mut json_str = String::new();
    decoder.read_to_string(&mut json_str).unwrap();
    let file: GroundTruthFile = serde_json::from_str(&json_str).unwrap();
    assert!(file.version >= 1);
    file.cases
}

pub(crate) fn load_ground_truth() -> Vec<GroundTruthCase> {
    load_ground_truth_from_bytes(GROUND_TRUTH_GZ)
}

pub(crate) fn load_massspecgym_ground_truth() -> Vec<GroundTruthCase> {
    load_ground_truth_from_path(MASSSPECGYM_GROUND_TRUTH_100_PATH)
}

pub(crate) fn load_massspecgym_ground_truth_1000() -> Vec<GroundTruthCase> {
    load_ground_truth_from_path(MASSSPECGYM_GROUND_TRUTH_1000_PATH)
}

pub(crate) fn load_massspecgym_ground_truth_10000() -> Vec<GroundTruthCase> {
    load_ground_truth_from_path(MASSSPECGYM_GROUND_TRUTH_10000_PATH)
}

pub(crate) fn load_massspecgym_all_best_ground_truth() -> Vec<GroundTruthCase> {
    load_ground_truth_from_path(MASSSPECGYM_ALL_BEST_GROUND_TRUTH_100_PATH)
}

pub(crate) fn evenly_spaced_case_indices(len: usize, samples: usize) -> Vec<usize> {
    assert!(len > 0, "cannot sample from an empty corpus");
    let samples = samples.min(len);
    if samples == len {
        return (0..len).collect();
    }

    (0..samples).map(|i| i * (len - 1) / (samples - 1)).collect()
}

pub(crate) fn first_parallel_mismatch<F>(cases: &[GroundTruthCase], run_case: F) -> Option<String>
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

pub(crate) fn case_ignores_edge_values(case: &GroundTruthCase) -> bool {
    case.options
        .as_ref()
        .and_then(|options| options.get("ignoreBondOrders"))
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false)
}

pub(crate) fn case_uses_complete_aromatic_rings(case: &GroundTruthCase) -> bool {
    case.options
        .as_ref()
        .and_then(|options| options.get("completeAromaticRings"))
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(true)
}

pub(crate) fn case_uses_ring_matches_ring_only(case: &GroundTruthCase) -> bool {
    case.options
        .as_ref()
        .and_then(|options| options.get("ringMatchesRingOnly"))
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false)
}

pub(crate) fn case_uses_exact_connections_match(case: &GroundTruthCase) -> bool {
    case.options
        .as_ref()
        .and_then(|options| options.get("exactConnectionsMatch"))
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false)
}

pub(crate) fn case_respects_atom_aromaticity(case: &GroundTruthCase) -> bool {
    matches!(
        case.options
            .as_ref()
            .and_then(|options| options.get("ignoreAtomAromaticity"))
            .and_then(serde_json::Value::as_bool),
        Some(false)
    )
}

pub(crate) fn case_similarity_threshold(case: &GroundTruthCase) -> Option<f64> {
    case.options
        .as_ref()
        .and_then(|options| options.get("similarityThreshold"))
        .and_then(serde_json::Value::as_f64)
}

pub(crate) fn build_edge_contexts(graph: &GraphData) -> Option<EdgeContexts<String>> {
    if graph.aromatic_ring_contexts.is_empty() {
        return None;
    }
    Some(EdgeContexts::from_rows(graph.aromatic_ring_contexts.iter().cloned()))
}

pub(crate) struct PreparedLabeledCase {
    pub(crate) first: TypedGraph,
    pub(crate) second: TypedGraph,
    pub(crate) first_contexts: Option<EdgeContexts<String>>,
    pub(crate) second_contexts: Option<EdgeContexts<String>>,
}

pub(crate) fn prepare_labeled_case_from_graph_data(
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

pub(crate) fn prepare_labeled_case(case: &GroundTruthCase) -> PreparedLabeledCase {
    prepare_labeled_case_from_graph_data(case, &case.graph1, &case.graph2)
}
