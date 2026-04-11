#![cfg(feature = "std")]
#![allow(clippy::pedantic)]

use std::{
    collections::{BTreeMap, BTreeSet},
    env, fs,
    path::{Path, PathBuf},
    process::Command,
    sync::atomic::{AtomicU64, Ordering},
    time::{SystemTime, UNIX_EPOCH},
};

static TEMP_DIR_COUNTER: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct BlissStats {
    pub(crate) nodes: Option<usize>,
    pub(crate) leaf_nodes: Option<usize>,
    pub(crate) bad_nodes: Option<usize>,
    pub(crate) canrep_updates: Option<usize>,
    pub(crate) generators: Option<usize>,
    pub(crate) max_level: Option<usize>,
    pub(crate) group_size: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct BlissOracleResult {
    pub(crate) expanded_canonical_labeling: Vec<usize>,
    pub(crate) original_canonical_order: Vec<usize>,
    pub(crate) canonical_dimacs: String,
    pub(crate) stats: BlissStats,
    pub(crate) stdout: String,
}

#[derive(Clone, Debug)]
pub(crate) struct EncodedLabeledSimpleGraph {
    pub(crate) dimacs: String,
    pub(crate) expanded_vertex_count: usize,
    pub(crate) original_vertex_count: usize,
}

pub(crate) fn locate_bliss_binary() -> Option<PathBuf> {
    if let Ok(path) = env::var("GEOMETRIC_TRAITS_BLISS_BIN") {
        let candidate = PathBuf::from(path);
        if candidate.is_file() {
            return Some(candidate);
        }
    }

    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let candidates = [
        PathBuf::from("/tmp/bliss-build/bliss"),
        manifest_dir.join("papers/software/bliss-0.77/build/bliss"),
        manifest_dir.join("papers/software/bliss-0.77/bliss"),
    ];

    candidates.into_iter().find(|candidate| candidate.is_file())
}

pub(crate) fn canonicalize_labeled_simple_graph<VertexLabel, EdgeLabel>(
    vertex_labels: &[VertexLabel],
    edges: &[(usize, usize, EdgeLabel)],
) -> Result<BlissOracleResult, String>
where
    VertexLabel: Ord + Clone,
    EdgeLabel: Ord + Clone,
{
    let bliss = locate_bliss_binary().ok_or_else(|| {
        "no bliss executable found; set GEOMETRIC_TRAITS_BLISS_BIN or build /tmp/bliss-build/bliss"
            .to_owned()
    })?;

    let encoded = encode_labeled_simple_graph_as_dimacs(vertex_labels, edges)?;
    let temp_dir = make_temp_dir()?;
    let input_path = temp_dir.join("input.dimacs");
    let canonical_path = temp_dir.join("canonical.dimacs");

    fs::write(&input_path, &encoded.dimacs).map_err(|error| error.to_string())?;
    let result = run_bliss_on_dimacs_file(
        &bliss,
        &input_path,
        &canonical_path,
        encoded.expanded_vertex_count,
        encoded.original_vertex_count,
    )?;

    let _ = fs::remove_dir_all(temp_dir);

    Ok(result)
}

fn make_temp_dir() -> Result<PathBuf, String> {
    let nanos =
        SystemTime::now().duration_since(UNIX_EPOCH).map_err(|error| error.to_string())?.as_nanos();
    let counter = TEMP_DIR_COUNTER.fetch_add(1, Ordering::Relaxed);
    let temp_dir = env::temp_dir()
        .join(format!("geometric-traits-bliss-{nanos}-{}-{counter}", std::process::id()));
    fs::create_dir_all(&temp_dir).map_err(|error| error.to_string())?;
    Ok(temp_dir)
}

pub(crate) fn encode_labeled_simple_graph_as_dimacs<VertexLabel, EdgeLabel>(
    vertex_labels: &[VertexLabel],
    edges: &[(usize, usize, EdgeLabel)],
) -> Result<EncodedLabeledSimpleGraph, String>
where
    VertexLabel: Ord + Clone,
    EdgeLabel: Ord + Clone,
{
    let normalized_edges = normalize_edges(vertex_labels.len(), edges)?;
    let encoded = encode_as_vertex_colored_dimacs(vertex_labels, &normalized_edges);
    Ok(EncodedLabeledSimpleGraph {
        dimacs: encoded.dimacs,
        expanded_vertex_count: encoded.expanded_vertex_count,
        original_vertex_count: vertex_labels.len(),
    })
}

fn normalize_edges<EdgeLabel: Ord + Clone>(
    vertex_count: usize,
    edges: &[(usize, usize, EdgeLabel)],
) -> Result<Vec<(usize, usize, EdgeLabel)>, String> {
    let mut normalized = Vec::with_capacity(edges.len());
    for &(left, right, ref label) in edges {
        if left >= vertex_count || right >= vertex_count {
            return Err(format!(
                "edge ({left}, {right}) is out of range for vertex count {vertex_count}"
            ));
        }
        if left == right {
            return Err(format!("self-loop ({left}, {right}) is not allowed for simple graphs"));
        }
        let (src, dst) = if left < right { (left, right) } else { (right, left) };
        normalized.push((src, dst, label.clone()));
    }
    normalized.sort_unstable_by(|left, right| {
        left.0.cmp(&right.0).then(left.1.cmp(&right.1)).then(left.2.cmp(&right.2))
    });
    for pair in normalized.windows(2) {
        if pair[0].0 == pair[1].0 && pair[0].1 == pair[1].1 {
            return Err(format!("parallel edge ({}, {}) is not allowed", pair[0].0, pair[0].1));
        }
    }
    Ok(normalized)
}

fn dense_label_ids<Label>(labels: impl Iterator<Item = Label>) -> BTreeMap<Label, usize>
where
    Label: Ord,
{
    labels
        .collect::<BTreeSet<_>>()
        .into_iter()
        .enumerate()
        .map(|(index, label)| (label, index))
        .collect()
}

fn encode_as_vertex_colored_dimacs<VertexLabel, EdgeLabel>(
    vertex_labels: &[VertexLabel],
    edges: &[(usize, usize, EdgeLabel)],
) -> EncodedLabeledSimpleGraph
where
    VertexLabel: Ord + Clone,
    EdgeLabel: Ord + Clone,
{
    let vertex_color_ids = dense_label_ids(vertex_labels.iter().cloned());
    let edge_color_ids = dense_label_ids(edges.iter().map(|(_, _, label)| label.clone()));
    let vertex_color_count = vertex_color_ids.len();
    let expanded_vertex_count = vertex_labels.len() + edges.len();
    let expanded_edge_count = edges.len() * 2;

    let mut dimacs = String::new();
    dimacs.push_str(&format!("p edge {expanded_vertex_count} {expanded_edge_count}\n"));

    for (index, label) in vertex_labels.iter().enumerate() {
        let color = vertex_color_ids[label] + 1;
        dimacs.push_str(&format!("n {} {color}\n", index + 1));
    }

    for (edge_index, (_, _, label)) in edges.iter().enumerate() {
        let color = vertex_color_count + edge_color_ids[label] + 1;
        dimacs.push_str(&format!("n {} {color}\n", vertex_labels.len() + edge_index + 1));
    }

    for (edge_index, &(src, dst, _)) in edges.iter().enumerate() {
        let edge_vertex = vertex_labels.len() + edge_index;
        dimacs.push_str(&format!("e {} {}\n", src + 1, edge_vertex + 1));
        dimacs.push_str(&format!("e {} {}\n", dst + 1, edge_vertex + 1));
    }

    EncodedLabeledSimpleGraph {
        dimacs,
        expanded_vertex_count,
        original_vertex_count: vertex_labels.len(),
    }
}

pub(crate) fn run_bliss_on_dimacs_file(
    bliss: &Path,
    input_path: &Path,
    canonical_path: &Path,
    expanded_vertex_count: usize,
    original_vertex_count: usize,
) -> Result<BlissOracleResult, String> {
    run_bliss_on_dimacs_file_with_options(
        bliss,
        input_path,
        canonical_path,
        expanded_vertex_count,
        original_vertex_count,
        true,
        true,
    )
}

pub(crate) fn run_bliss_on_dimacs_file_with_options(
    bliss: &Path,
    input_path: &Path,
    canonical_path: &Path,
    expanded_vertex_count: usize,
    original_vertex_count: usize,
    failure_recording: bool,
    component_recursion: bool,
) -> Result<BlissOracleResult, String> {
    let output = Command::new(bliss)
        .arg("-can")
        .arg("-v=1")
        .arg("-sh=fsm")
        .arg(if failure_recording { "-fr=y" } else { "-fr=n" })
        .arg(if component_recursion { "-cr=y" } else { "-cr=n" })
        .arg(format!("-ocan={}", canonical_path.display()))
        .arg(input_path)
        .output()
        .map_err(|error| error.to_string())?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("bliss failed with status {}: {stderr}", output.status));
    }

    let stdout = String::from_utf8(output.stdout).map_err(|error| error.to_string())?;
    let canonical_dimacs = fs::read_to_string(canonical_path).map_err(|error| error.to_string())?;
    let expanded_canonical_labeling = parse_canonical_labeling(&stdout, expanded_vertex_count)?;
    let original_canonical_order =
        original_vertex_order(&expanded_canonical_labeling, original_vertex_count);
    let stats = parse_stats(&stdout);

    Ok(BlissOracleResult {
        expanded_canonical_labeling,
        original_canonical_order,
        canonical_dimacs,
        stats,
        stdout,
    })
}

fn parse_canonical_labeling(stdout: &str, vertex_count: usize) -> Result<Vec<usize>, String> {
    let line = stdout
        .lines()
        .find(|line| line.starts_with("Canonical labeling:"))
        .ok_or_else(|| "bliss output did not contain a canonical labeling".to_owned())?;
    let cycles = line.split_once(':').map(|(_, suffix)| suffix.trim()).unwrap_or("");
    let mut permutation: Vec<usize> = (0..vertex_count).collect();
    let mut current_cycle = Vec::new();
    let mut number = String::new();

    for ch in cycles.chars() {
        match ch {
            '(' => {
                current_cycle.clear();
                number.clear();
            }
            '0'..='9' => number.push(ch),
            ',' | ')' => {
                if !number.is_empty() {
                    let parsed = number.parse::<usize>().map_err(|error| error.to_string())?;
                    if parsed == 0 || parsed > vertex_count {
                        return Err(format!(
                            "cycle element {parsed} is out of range for {vertex_count} vertices"
                        ));
                    }
                    current_cycle.push(parsed - 1);
                    number.clear();
                }
                if ch == ')' && current_cycle.len() > 1 {
                    for index in 0..current_cycle.len() {
                        permutation[current_cycle[index]] =
                            current_cycle[(index + 1) % current_cycle.len()];
                    }
                }
            }
            _ => {}
        }
    }

    Ok(permutation)
}

fn original_vertex_order(canonical_labeling: &[usize], original_vertex_count: usize) -> Vec<usize> {
    let mut order: Vec<usize> = (0..original_vertex_count).collect();
    order.sort_unstable_by_key(|&vertex| canonical_labeling[vertex]);
    order
}

fn parse_stats(stdout: &str) -> BlissStats {
    fn parse_stat(stdout: &str, prefix: &str) -> Option<usize> {
        stdout.lines().find_map(|line| {
            let suffix = line.strip_prefix(prefix)?.trim();
            suffix.parse::<usize>().ok()
        })
    }

    let group_size = stdout
        .lines()
        .find_map(|line| line.strip_prefix("|Aut|:").map(|suffix| suffix.trim().to_owned()));

    BlissStats {
        nodes: parse_stat(stdout, "Nodes:"),
        leaf_nodes: parse_stat(stdout, "Leaf nodes:"),
        bad_nodes: parse_stat(stdout, "Bad nodes:"),
        canrep_updates: parse_stat(stdout, "Canrep updates:"),
        generators: parse_stat(stdout, "Generators:"),
        max_level: parse_stat(stdout, "Max level:"),
        group_size,
    }
}

#[allow(dead_code)]
fn _assert_file_exists(path: &Path) {
    assert!(path.is_file(), "expected file {}", path.display());
}
