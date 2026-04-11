use std::{
    env, fs,
    path::PathBuf,
    time::{SystemTime, UNIX_EPOCH},
};

use arbitrary::{Arbitrary, Unstructured};

#[path = "../../fuzz_targets/canon_bliss_support.rs"]
mod canon_bliss_support;

use canon_bliss_support::{encode_labeled_simple_graph_as_dimacs, run_bliss_on_dimacs_file};

const MAX_VERTEX_COUNT: usize = 25;
const MAX_LABEL_ALPHABET: u8 = 6;

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
    let mut args = env::args().skip(1);
    let input_path = PathBuf::from(args.next().expect("usage: decode_canon_bliss <artifact>"));
    let bliss = locate_bliss_binary().expect("bliss binary not found");

    let bytes = fs::read(&input_path).expect("artifact should read");
    let mut unstructured = Unstructured::new(&bytes);
    let case = FuzzCanonBlissCase::arbitrary(&mut unstructured).expect("artifact should decode");

    println!("decoded_case={case:#?}");
    let (vertex_labels, edges) = materialize_case(&case);
    println!("vertex_labels={vertex_labels:?}");
    println!("edges={edges:?}");

    let encoded = encode_labeled_simple_graph_as_dimacs(&vertex_labels, &edges)
        .expect("encoding should work");
    let temp_dir = make_temp_dir();
    let input_dimacs = temp_dir.join("input.dimacs");
    let output_dimacs = temp_dir.join("canonical.dimacs");
    fs::write(&input_dimacs, &encoded.dimacs).expect("dimacs should write");
    println!("dimacs:\n{}", encoded.dimacs);

    let raw = std::process::Command::new(&bliss)
        .arg("-can")
        .arg("-v=1")
        .arg("-sh=fsm")
        .arg("-fr=y")
        .arg("-cr=y")
        .arg(format!("-ocan={}", output_dimacs.display()))
        .arg(&input_dimacs)
        .output()
        .expect("bliss should run");
    println!("bliss_status={}", raw.status);
    println!("bliss_stdout:\n{}", String::from_utf8_lossy(&raw.stdout));
    println!("bliss_stderr:\n{}", String::from_utf8_lossy(&raw.stderr));

    let parsed = run_bliss_on_dimacs_file(
        &bliss,
        &input_dimacs,
        &output_dimacs,
        encoded.expanded_vertex_count,
        encoded.original_vertex_count,
    );
    println!("parsed={parsed:#?}");
    let _ = fs::remove_dir_all(&temp_dir);
}

fn locate_bliss_binary() -> Option<PathBuf> {
    if let Ok(path) = env::var("GEOMETRIC_TRAITS_BLISS_BIN") {
        let candidate = PathBuf::from(path);
        if candidate.is_file() {
            return Some(candidate);
        }
    }

    let fuzz_manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let repo_root = fuzz_manifest_dir.parent().expect("fuzz crate should live under the repo root");
    let candidates = [
        PathBuf::from("/tmp/bliss-build/bliss"),
        repo_root.join("papers/software/bliss-0.77/build/bliss"),
        repo_root.join("papers/software/bliss-0.77/bliss"),
    ];

    candidates.into_iter().find(|candidate| candidate.is_file())
}

fn materialize_case(case: &FuzzCanonBlissCase) -> (Vec<u8>, Vec<(usize, usize, u8)>) {
    let vertex_count = usize::from(case.vertex_count_hint) % MAX_VERTEX_COUNT + 1;
    let vertex_label_mod = case.vertex_label_alphabet_hint % MAX_LABEL_ALPHABET + 1;
    let edge_label_mod = case.edge_label_alphabet_hint % MAX_LABEL_ALPHABET + 1;

    let vertex_labels = if case.vertex_label_bytes.is_empty() {
        vec![0_u8; vertex_count]
    } else {
        (0..vertex_count)
            .map(|index| {
                case.vertex_label_bytes[index % case.vertex_label_bytes.len()] % vertex_label_mod
            })
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

fn make_temp_dir() -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock should be after unix epoch")
        .as_nanos();
    let temp_dir = env::temp_dir()
        .join(format!("geometric-traits-canon-bliss-decode-{nanos}-{}", std::process::id()));
    fs::create_dir_all(&temp_dir).expect("decode temp dir should be creatable");
    temp_dir
}
