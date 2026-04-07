//! Parallel validation of the local 500k power-iteration eigenvector corpus.
#![cfg(feature = "std")]

mod common;

use std::{fs::File, io::BufReader};

use flate2::read::GzDecoder;
use geometric_traits::{
    impls::{CSR2D, SortedVec, SymmetricCSR2D},
    prelude::*,
    traits::{EdgesBuilder, VocabularyBuilder},
};
use rayon::prelude::*;
use serde::{
    Deserialize,
    de::{DeserializeSeed, IgnoredAny, MapAccess, SeqAccess, Visitor},
};

const LOCAL_CORPUS_PATH: &str = "local/node_ordering_power_iteration_500k.json.gz";
const CASE_TOLERANCE: f64 = 2.0e-12;
const EXPECTED_CASE_COUNT: usize = 500_000;
const BATCH_SIZE: usize = 512;

type UndirectedGraph = SymmetricCSR2D<CSR2D<usize, usize, usize>>;

#[derive(Debug, Deserialize)]
struct Case {
    name: String,
    family: String,
    n: usize,
    edges: Vec<[usize; 2]>,
    max_iter: usize,
    scores: Vec<f64>,
}

#[derive(Debug)]
struct BatchSummary {
    case_count: usize,
    max_abs_error: f64,
    max_iter: usize,
}

#[derive(Debug, Default)]
struct ValidationSummary {
    case_count: usize,
    max_abs_error: f64,
    max_iter: usize,
}

fn build_graph(case: &Case) -> UndiGraph<usize> {
    let nodes: SortedVec<usize> = GenericVocabularyBuilder::default()
        .expected_number_of_symbols(case.n)
        .symbols((0..case.n).enumerate())
        .build()
        .unwrap();
    let mut edges: Vec<(usize, usize)> =
        case.edges.iter().map(|&[left, right]| (left, right)).collect();
    edges.sort_unstable();
    let matrix: UndirectedGraph = UndiEdgesBuilder::default()
        .expected_number_of_edges(edges.len())
        .expected_shape(case.n)
        .edges(edges.into_iter())
        .build()
        .unwrap();
    UndiGraph::from((nodes, matrix))
}

fn validate_case(case: &Case) -> f64 {
    let graph = build_graph(case);
    let actual = PowerIterationEigenvectorCentralityScorerBuilder::default()
        .max_iter(case.max_iter)
        .tolerance(1.0e-6)
        .build()
        .score_nodes(&graph);

    assert_eq!(
        actual.len(),
        case.scores.len(),
        "case {} ({}) returned a wrong score-vector length",
        case.name,
        case.family
    );

    actual
        .iter()
        .zip(&case.scores)
        .enumerate()
        .map(|(index, (got, want))| {
            let delta = (got - want).abs();
            assert!(
                delta <= CASE_TOLERANCE,
                "case {} ({}) differed at node {}: got={}, want={}, delta={}",
                case.name,
                case.family,
                index,
                got,
                want,
                delta
            );
            delta
        })
        .fold(0.0, f64::max)
}

fn process_batch(batch: &mut Vec<Case>, summary: &mut ValidationSummary) {
    let batch_summary = batch
        .par_iter()
        .map(|case| (validate_case(case), case.max_iter))
        .fold(
            || BatchSummary { case_count: 0, max_abs_error: 0.0, max_iter: 0 },
            |mut acc, (max_abs_error, max_iter)| {
                acc.case_count += 1;
                acc.max_abs_error = acc.max_abs_error.max(max_abs_error);
                acc.max_iter = acc.max_iter.max(max_iter);
                acc
            },
        )
        .reduce(
            || BatchSummary { case_count: 0, max_abs_error: 0.0, max_iter: 0 },
            |left, right| {
                BatchSummary {
                    case_count: left.case_count + right.case_count,
                    max_abs_error: left.max_abs_error.max(right.max_abs_error),
                    max_iter: left.max_iter.max(right.max_iter),
                }
            },
        );

    summary.case_count += batch_summary.case_count;
    summary.max_abs_error = summary.max_abs_error.max(batch_summary.max_abs_error);
    summary.max_iter = summary.max_iter.max(batch_summary.max_iter);
    batch.clear();
}

struct CasesSeed<'a> {
    summary: &'a mut ValidationSummary,
}

impl<'de> DeserializeSeed<'de> for CasesSeed<'_> {
    type Value = ();

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        deserializer.deserialize_seq(CasesVisitor { summary: self.summary })
    }
}

struct CasesVisitor<'a> {
    summary: &'a mut ValidationSummary,
}

impl<'de> Visitor<'de> for CasesVisitor<'_> {
    type Value = ();

    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("a cases array")
    }

    fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        let mut batch = Vec::with_capacity(BATCH_SIZE);
        while let Some(case) = seq.next_element::<Case>()? {
            batch.push(case);
            if batch.len() == BATCH_SIZE {
                process_batch(&mut batch, self.summary);
            }
        }
        if !batch.is_empty() {
            process_batch(&mut batch, self.summary);
        }
        Ok(())
    }
}

struct FixtureSeed<'a> {
    summary: &'a mut ValidationSummary,
}

impl<'de> DeserializeSeed<'de> for FixtureSeed<'_> {
    type Value = ();

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        deserializer.deserialize_map(FixtureVisitor { summary: self.summary })
    }
}

struct FixtureVisitor<'a> {
    summary: &'a mut ValidationSummary,
}

impl<'de> Visitor<'de> for FixtureVisitor<'_> {
    type Value = ();

    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("a power-iteration fixture object")
    }

    fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        while let Some(key) = map.next_key::<String>()? {
            match key.as_str() {
                "cases" => map.next_value_seed(CasesSeed { summary: self.summary })?,
                _ => {
                    let _: IgnoredAny = map.next_value()?;
                }
            }
        }
        Ok(())
    }
}

#[test]
#[ignore = "requires tests/fixtures/local/node_ordering_power_iteration_500k.json.gz"]
fn test_local_power_iteration_reference_corpus_500k() {
    let path = common::fixture_path(LOCAL_CORPUS_PATH);
    assert!(path.exists(), "local 500k fixture missing at {}; generate it first", path.display());

    let file = File::open(&path).unwrap_or_else(|error| {
        panic!("failed to open {}: {error}", path.display());
    });
    let reader = BufReader::new(file);
    let decoder = GzDecoder::new(reader);
    let mut summary = ValidationSummary::default();

    let mut deserializer = serde_json::Deserializer::from_reader(decoder);
    FixtureSeed { summary: &mut summary }.deserialize(&mut deserializer).unwrap();

    assert_eq!(summary.case_count, EXPECTED_CASE_COUNT);
    eprintln!(
        "[node-ordering-power-iteration-500k] validated {} cases, max_abs_error={:.3e}, max_iter={}",
        summary.case_count, summary.max_abs_error, summary.max_iter
    );
}
