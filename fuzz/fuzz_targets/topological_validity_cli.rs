//! Fuzz harness that cross-checks the crate's topology predicates against the
//! Boyer `planarity` CLI in specific-graph mode.
//!
//! The harness builds a small simple undirected graph from the fuzz input and
//! compares the following booleans against the CLI:
//!
//! - `-p`: planarity
//! - `-o`: outerplanarity
//! - `-2`: `K_{2,3}` homeomorph detection
//! - `-3`: `K_{3,3}` homeomorph detection
//! - `-4`: `K_4` homeomorph detection

use std::{
    collections::BTreeSet,
    env, fs,
    path::{Path, PathBuf},
    process::{self, Command, Stdio},
    sync::{
        atomic::{AtomicU64, Ordering},
        OnceLock,
    },
};

use arbitrary::Arbitrary;
use geometric_traits::{
    impls::{SymmetricCSR2D, UpperTriangularCSR2D, CSR2D},
    naive_structs::{GenericGraph, GenericUndirectedMonopartiteEdgesBuilder},
    traits::{
        EdgesBuilder, K23HomeomorphDetection, K33HomeomorphDetection, K4HomeomorphDetection,
        OuterplanarityDetection, PlanarityDetection,
    },
};
use honggfuzz::fuzz;

type UndirectedGraph = GenericGraph<Vec<u8>, SymmetricCSR2D<CSR2D<usize, usize, usize>>>;

const MAX_ORDER: usize = 16;
const MAX_RAW_EDGES: usize = 128;
const DEV_NULL: &str = "/dev/null";
const DEFAULT_PLANARITY_BIN: &str = "/tmp/edge-addition-planarity-suite-master/planarity";

static NEXT_TEMP_ID: AtomicU64 = AtomicU64::new(0);
static PLANARITY_BIN: OnceLock<PathBuf> = OnceLock::new();

#[derive(Arbitrary, Debug)]
struct FuzzTopologicalValidityCase {
    order: u8,
    edges: Vec<(u8, u8, u8)>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct TopologyOutcomes {
    is_planar: bool,
    is_outerplanar: bool,
    has_k23_homeomorph: bool,
    has_k33_homeomorph: bool,
    has_k4_homeomorph: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TopologyCliMode {
    Planarity,
    Outerplanarity,
    K23Homeomorph,
    K33Homeomorph,
    K4Homeomorph,
}

impl TopologyCliMode {
    fn flag(self) -> &'static str {
        match self {
            Self::Planarity => "-p",
            Self::Outerplanarity => "-o",
            Self::K23Homeomorph => "-2",
            Self::K33Homeomorph => "-3",
            Self::K4Homeomorph => "-4",
        }
    }

    fn decode_exit_status(self, code: i32) -> Option<bool> {
        match self {
            Self::Planarity | Self::Outerplanarity => {
                match code {
                    0 => Some(true),
                    1 => Some(false),
                    _ => None,
                }
            }
            Self::K23Homeomorph | Self::K33Homeomorph | Self::K4Homeomorph => {
                match code {
                    0 => Some(false),
                    1 => Some(true),
                    _ => None,
                }
            }
        }
    }
}

struct TempPlanarityInput {
    path: PathBuf,
}

impl Drop for TempPlanarityInput {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

fn main() {
    loop {
        fuzz!(|case: FuzzTopologicalValidityCase| {
            let graph_data = build_graph_data(&case);
            if env::var_os("TRACE_TOPOLOGY_FUZZ").is_some() {
                eprintln!(
                    "[trace-topology-fuzz] order={} edges={:?}",
                    graph_data.order, graph_data.edges
                );
            }
            let graph = build_graph(&graph_data);
            let crate_outcomes = crate_topology_outcomes(&graph);
            let cli_outcomes = cli_topology_outcomes(&graph_data);

            assert_eq!(
                crate_outcomes, cli_outcomes,
                "topology mismatch order={} edges={:?}",
                graph_data.order, graph_data.edges
            );
        });
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct GraphData {
    order: usize,
    edges: Vec<(usize, usize)>,
}

fn build_graph_data(case: &FuzzTopologicalValidityCase) -> GraphData {
    let order =
        usize::from(case.order % u8::try_from(MAX_ORDER).expect("MAX_ORDER fits into u8")) + 1;
    let mut edges = BTreeSet::new();

    for &(left, right, keep) in case.edges.iter().take(MAX_RAW_EDGES) {
        if keep & 1 == 0 {
            continue;
        }
        let left = usize::from(left) % order;
        let right = usize::from(right) % order;
        if left == right {
            continue;
        }
        edges.insert(normalize_edge(left, right));
    }

    GraphData { order, edges: edges.into_iter().collect() }
}

fn normalize_edge(left: usize, right: usize) -> (usize, usize) {
    if left <= right {
        (left, right)
    } else {
        (right, left)
    }
}

fn build_graph(graph_data: &GraphData) -> UndirectedGraph {
    let nodes: Vec<u8> = (0..graph_data.order)
        .map(|index| u8::try_from(index).expect("node index should fit into u8"))
        .collect();
    let adjacency = GenericUndirectedMonopartiteEdgesBuilder::<
        _,
        UpperTriangularCSR2D<CSR2D<usize, usize, usize>>,
        SymmetricCSR2D<CSR2D<usize, usize, usize>>,
    >::default()
    .expected_number_of_edges(graph_data.edges.len())
    .expected_shape(graph_data.order)
    .edges(graph_data.edges.iter().copied())
    .build()
    .unwrap();
    GenericGraph::from((nodes, adjacency))
}

fn crate_topology_outcomes(graph: &UndirectedGraph) -> TopologyOutcomes {
    TopologyOutcomes {
        is_planar: graph.is_planar().expect("simple fuzz graph should be valid for planarity"),
        is_outerplanar: graph
            .is_outerplanar()
            .expect("simple fuzz graph should be valid for outerplanarity"),
        has_k23_homeomorph: graph
            .has_k23_homeomorph()
            .expect("simple fuzz graph should be valid for K23 detection"),
        has_k33_homeomorph: graph
            .has_k33_homeomorph()
            .expect("simple fuzz graph should be valid for K33 detection"),
        has_k4_homeomorph: graph
            .has_k4_homeomorph()
            .expect("simple fuzz graph should be valid for K4 detection"),
    }
}

fn cli_topology_outcomes(graph_data: &GraphData) -> TopologyOutcomes {
    let input = write_planarity_input(graph_data);
    TopologyOutcomes {
        is_planar: classify_planarity_mode(TopologyCliMode::Planarity, &input.path),
        is_outerplanar: classify_planarity_mode(TopologyCliMode::Outerplanarity, &input.path),
        has_k23_homeomorph: classify_planarity_mode(TopologyCliMode::K23Homeomorph, &input.path),
        has_k33_homeomorph: classify_planarity_mode(TopologyCliMode::K33Homeomorph, &input.path),
        has_k4_homeomorph: classify_planarity_mode(TopologyCliMode::K4Homeomorph, &input.path),
    }
}

fn write_planarity_input(graph_data: &GraphData) -> TempPlanarityInput {
    let mut adjacency = vec![Vec::new(); graph_data.order];
    for &(left, right) in &graph_data.edges {
        adjacency[left].push(right);
        adjacency[right].push(left);
    }
    for neighbors in &mut adjacency {
        neighbors.sort_unstable();
    }

    let temp_id = NEXT_TEMP_ID.fetch_add(1, Ordering::Relaxed);
    let path = env::temp_dir().join(format!(
        "geometric_traits_topology_fuzz_{}_{}.txt",
        process::id(),
        temp_id
    ));

    let mut contents = String::new();
    contents.push_str(&format!("N={}\n", graph_data.order));
    for (vertex, neighbors) in adjacency.iter().enumerate() {
        contents.push_str(&format!("{vertex}:"));
        if neighbors.is_empty() {
            contents.push_str(" -1\n");
            continue;
        }
        for &neighbor in neighbors {
            contents.push(' ');
            contents.push_str(&neighbor.to_string());
        }
        contents.push_str(" -1\n");
    }

    fs::write(&path, contents).expect("failed to write temporary planarity input");
    TempPlanarityInput { path }
}

fn classify_planarity_mode(mode: TopologyCliMode, input_path: &Path) -> bool {
    let completed = Command::new(planarity_bin())
        .args(["-s", "-q", mode.flag()])
        .arg(input_path)
        .arg(DEV_NULL)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .unwrap_or_else(|error| {
            panic!(
                "failed to invoke planarity CLI {:?}: {error}; set PLANARITY_BIN to the upstream 4.x binary",
                planarity_bin()
            )
        });

    match completed.code() {
        Some(code) => {
            mode.decode_exit_status(code).unwrap_or_else(|| {
                panic!(
                    "planarity CLI mode {} exited with unexpected code {code} on {}",
                    mode.flag(),
                    input_path.display()
                )
            })
        }
        None => {
            panic!(
                "planarity CLI mode {} terminated by signal on {}",
                mode.flag(),
                input_path.display()
            )
        }
    }
}

fn planarity_bin() -> &'static Path {
    PLANARITY_BIN
        .get_or_init(|| {
            if let Some(path) = env::var_os("PLANARITY_BIN") {
                return PathBuf::from(path);
            }
            let candidate = PathBuf::from(DEFAULT_PLANARITY_BIN);
            if candidate.is_file() {
                candidate
            } else {
                PathBuf::from("planarity")
            }
        })
        .as_path()
}
