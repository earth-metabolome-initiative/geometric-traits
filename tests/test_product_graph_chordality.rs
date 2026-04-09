//! Chordality summaries for MCES labeled product graphs.
#![cfg(feature = "std")]
#![allow(dead_code)]

#[path = "test_mces_ground_truth/support.rs"]
mod support;

use std::{collections::BTreeMap, env, fs::File, io::Read as _, path::Path};

use flate2::read::GzDecoder;
use geometric_traits::traits::{
    ChordalDetection, ModularProduct, MonoplexGraph, SizedSparseMatrix,
};
use indicatif::{ProgressBar, ProgressStyle};
use serde::Deserialize;
use support::*;

const DEFAULT_K300_INPUT_PATH: &str = "/tmp/massspecgym_pairwise_mces_input_k300.json.gz";
const DEFAULT_LIPID_PAIR_IDS: [usize; 3] = [40865, 42569, 42702];

#[derive(Clone, Copy, Debug, Default)]
struct ProductGraphChordalitySummary {
    total: usize,
    empty: usize,
    chordal: usize,
    nonempty_chordal: usize,
    non_chordal: usize,
    max_order: usize,
    max_edge_count: usize,
}

impl ProductGraphChordalitySummary {
    fn record(&mut self, order: usize, edge_count: usize, chordal: bool) {
        self.total += 1;
        if order == 0 {
            self.empty += 1;
        }
        if chordal {
            self.chordal += 1;
            if order > 0 {
                self.nonempty_chordal += 1;
            }
        } else {
            self.non_chordal += 1;
        }
        self.max_order = self.max_order.max(order);
        self.max_edge_count = self.max_edge_count.max(edge_count);
    }

    fn merge(self, other: Self) -> Self {
        Self {
            total: self.total + other.total,
            empty: self.empty + other.empty,
            chordal: self.chordal + other.chordal,
            nonempty_chordal: self.nonempty_chordal + other.nonempty_chordal,
            non_chordal: self.non_chordal + other.non_chordal,
            max_order: self.max_order.max(other.max_order),
            max_edge_count: self.max_edge_count.max(other.max_edge_count),
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct SingleProductGraphSummary {
    order: usize,
    edge_count: usize,
    chordal: bool,
}

#[derive(Clone, Deserialize)]
struct PairwiseDataset {
    molecules: Vec<PairwiseMoleculeRecord>,
    pairs: Vec<PairwisePairRecord>,
}

#[derive(Clone, Deserialize)]
struct PairwiseMoleculeRecord {
    molecule_id: usize,
    graph: PairwiseGraphRecord,
}

#[derive(Clone, Deserialize)]
struct PairwiseGraphRecord {
    n_atoms: usize,
    n_bonds: usize,
    atoms: Vec<PairwiseAtomRecord>,
    bonds: Vec<PairwiseBondRecord>,
}

#[derive(Clone, Deserialize)]
struct PairwiseAtomRecord {
    atom_index: usize,
    element: String,
}

#[derive(Clone, Deserialize)]
struct PairwiseBondRecord {
    canonical_bond_index: usize,
    src_atom: usize,
    dst_atom: usize,
    bond_order: u32,
    #[serde(default)]
    aromatic_ring_contexts: Vec<String>,
}

#[derive(Clone, Deserialize)]
struct PairwisePairRecord {
    pair_id: usize,
    left_molecule_id: usize,
    right_molecule_id: usize,
    #[serde(default)]
    fingerprint_similarity: f64,
}

fn parse_env_usize(name: &str) -> Option<usize> {
    env::var(name).ok().map(|raw| {
        raw.parse::<usize>().unwrap_or_else(|error| panic!("invalid {name} value '{raw}': {error}"))
    })
}

fn parse_env_pair_ids(name: &str, default: &[usize]) -> Vec<usize> {
    env::var(name)
        .ok()
        .map(|raw| {
            raw.split(',')
                .filter(|item| !item.trim().is_empty())
                .map(|item| {
                    item.trim().parse::<usize>().unwrap_or_else(|error| {
                        panic!("invalid {name} pair id '{}': {error}", item.trim())
                    })
                })
                .collect()
        })
        .unwrap_or_else(|| default.to_vec())
}

fn edge_contexts_match(
    first_contexts: Option<&EdgeContexts<String>>,
    second_contexts: Option<&EdgeContexts<String>>,
    use_edge_contexts: bool,
    first_edge: usize,
    second_edge: usize,
) -> bool {
    match (first_contexts, second_contexts) {
        (Some(first_contexts), Some(second_contexts)) if use_edge_contexts => {
            first_contexts.compatible_with(first_edge, second_contexts, second_edge)
        }
        _ => true,
    }
}

fn summarize_prepared_labeled_product_graph(
    case: &GroundTruthCase,
    prepared: &PreparedLabeledCase,
) -> SingleProductGraphSummary {
    let lg1 = prepared.first.labeled_line_graph();
    let lg2 = prepared.second.labeled_line_graph();
    let first_bond_labels = compute_case_bond_labels(&prepared.first, lg1.edge_map());
    let second_bond_labels = compute_case_bond_labels(&prepared.second, lg2.edge_map());
    let use_edge_contexts = case_uses_complete_aromatic_rings(case);

    let product = lg1.graph().labeled_modular_product_filtered(
        lg2.graph(),
        |i, j| {
            first_bond_labels[i] == second_bond_labels[j]
                && edge_contexts_match(
                    prepared.first_contexts.as_ref(),
                    prepared.second_contexts.as_ref(),
                    use_edge_contexts,
                    i,
                    j,
                )
        },
        |left, right| left == right,
    );

    let graph = product.into_graph();
    let order = graph.number_of_nodes();
    let edge_count = graph.edges().number_of_defined_values() / 2;
    let chordal = graph.is_chordal();
    SingleProductGraphSummary { order, edge_count, chordal }
}

fn pairwise_graph_to_ground_truth(graph: &PairwiseGraphRecord) -> GraphData {
    let mut atom_types = vec![String::new(); graph.n_atoms];
    for atom in &graph.atoms {
        atom_types[atom.atom_index] = atom.element.clone();
    }

    let mut edges = vec![[0usize; 2]; graph.n_bonds];
    let mut bond_types = vec![0u32; graph.n_bonds];
    let mut aromatic_ring_contexts = vec![Vec::<String>::new(); graph.n_bonds];
    for bond in &graph.bonds {
        edges[bond.canonical_bond_index] = [bond.src_atom, bond.dst_atom];
        bond_types[bond.canonical_bond_index] = bond.bond_order;
        aromatic_ring_contexts[bond.canonical_bond_index] = bond.aromatic_ring_contexts.clone();
    }

    GraphData {
        n_atoms: graph.n_atoms,
        edges,
        atom_types,
        bond_types,
        aromatic_ring_contexts,
        atom_is_aromatic: vec![false; graph.n_atoms],
    }
}

fn summarize_pairwise_product_graph(
    pair: &PairwisePairRecord,
    molecules_by_id: &BTreeMap<usize, PairwiseMoleculeRecord>,
) -> SingleProductGraphSummary {
    let left = molecules_by_id.get(&pair.left_molecule_id).unwrap_or_else(|| {
        panic!("missing left molecule {} for pair {}", pair.left_molecule_id, pair.pair_id)
    });
    let right = molecules_by_id.get(&pair.right_molecule_id).unwrap_or_else(|| {
        panic!("missing right molecule {} for pair {}", pair.right_molecule_id, pair.pair_id)
    });

    let synthetic_case = GroundTruthCase {
        name: format!("k300_pair_{}", pair.pair_id),
        graph1: pairwise_graph_to_ground_truth(&left.graph),
        graph2: pairwise_graph_to_ground_truth(&right.graph),
        expected_bond_matches: 0,
        expected_similarity: 0.0,
        timed_out: false,
        options: None,
    };
    let prepared = prepare_labeled_case_from_graph_data(
        &synthetic_case,
        &synthetic_case.graph1,
        &synthetic_case.graph2,
    );
    summarize_prepared_labeled_product_graph(&synthetic_case, &prepared)
}

fn read_pairwise_dataset(path: &Path) -> PairwiseDataset {
    let file = File::open(path).unwrap_or_else(|error| {
        panic!("failed to open pairwise input {}: {error}", path.display())
    });
    let mut json = String::new();
    if path.extension().is_some_and(|ext| ext == "gz") {
        let mut decoder = GzDecoder::new(file);
        decoder.read_to_string(&mut json).unwrap_or_else(|error| {
            panic!("failed to read pairwise input {}: {error}", path.display())
        });
    } else {
        let mut reader = std::io::BufReader::new(file);
        reader.read_to_string(&mut json).unwrap_or_else(|error| {
            panic!("failed to read pairwise input {}: {error}", path.display())
        });
    }
    serde_json::from_str(&json).unwrap_or_else(|error| {
        panic!("failed to parse pairwise input {}: {error}", path.display())
    })
}

#[test]
#[ignore = "summary of chordality across MassSpecGym 200K labeled product graphs"]
fn test_massspecgym_product_graph_chordality_summary_200000() {
    let cases = load_massspecgym_ground_truth_200000();
    let case_offset = parse_env_usize("CASE_OFFSET").unwrap_or(0);
    let case_limit =
        parse_env_usize("CASE_LIMIT").unwrap_or(cases.len().saturating_sub(case_offset));
    assert!(
        case_offset <= cases.len(),
        "CASE_OFFSET {case_offset} exceeds corpus length {}",
        cases.len()
    );
    let end = case_offset.saturating_add(case_limit).min(cases.len());
    let selected = &cases[case_offset..end];

    let progress = ProgressBar::new(selected.len() as u64);
    progress.set_style(
        ProgressStyle::with_template(
            "{spinner:.green} 200k chordal [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta_precise})",
        )
        .expect("progress style template must be valid")
        .progress_chars("=> "),
    );

    let summary = selected
        .par_iter()
        .map(|case| {
            let item = summarize_prepared_labeled_product_graph(case, &prepare_labeled_case(case));
            progress.inc(1);
            let mut summary = ProductGraphChordalitySummary::default();
            summary.record(item.order, item.edge_count, item.chordal);
            summary
        })
        .reduce(ProductGraphChordalitySummary::default, ProductGraphChordalitySummary::merge);
    progress.finish_with_message("200k chordality complete");

    println!("checked {} cases (offset={} end={})", selected.len(), case_offset, end);
    println!(
        "product_graphs: total={} chordal={} nonempty_chordal={} non_chordal={} empty={} max_order={} max_edge_count={}",
        summary.total,
        summary.chordal,
        summary.nonempty_chordal,
        summary.non_chordal,
        summary.empty,
        summary.max_order,
        summary.max_edge_count
    );
}

#[test]
#[ignore = "summary of chordality for the pathological K300 lipid pairs"]
fn test_k300_lipid_product_graph_chordality_summary() {
    let input_path = env::var("K300_INPUT").unwrap_or_else(|_| DEFAULT_K300_INPUT_PATH.to_string());
    let path = Path::new(&input_path);
    if !path.exists() {
        println!("skipping lipid chordality summary: missing {}", path.display());
        return;
    }

    let dataset = read_pairwise_dataset(path);
    let pair_ids = parse_env_pair_ids("K300_PAIR_IDS", &DEFAULT_LIPID_PAIR_IDS);
    let molecules_by_id: BTreeMap<usize, PairwiseMoleculeRecord> =
        dataset.molecules.into_iter().map(|molecule| (molecule.molecule_id, molecule)).collect();
    let pairs_by_id: BTreeMap<usize, PairwisePairRecord> =
        dataset.pairs.into_iter().map(|pair| (pair.pair_id, pair)).collect();

    let mut totals = ProductGraphChordalitySummary::default();
    for pair_id in pair_ids {
        let pair = pairs_by_id
            .get(&pair_id)
            .unwrap_or_else(|| panic!("missing pair {pair_id} in {}", path.display()));
        let summary = summarize_pairwise_product_graph(pair, &molecules_by_id);
        totals.record(summary.order, summary.edge_count, summary.chordal);
        println!(
            "pair_id={} left={} right={} fingerprint_similarity={:.6} order={} edges={} chordal={}",
            pair.pair_id,
            pair.left_molecule_id,
            pair.right_molecule_id,
            pair.fingerprint_similarity,
            summary.order,
            summary.edge_count,
            summary.chordal
        );
    }

    println!(
        "lipid_product_graphs: total={} chordal={} nonempty_chordal={} non_chordal={} empty={} max_order={} max_edge_count={}",
        totals.total,
        totals.chordal,
        totals.nonempty_chordal,
        totals.non_chordal,
        totals.empty,
        totals.max_order,
        totals.max_edge_count
    );
}
