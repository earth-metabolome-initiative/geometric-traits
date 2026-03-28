//! Parallel validation of the local 1M GTH reference corpus.
#![cfg(feature = "std")]

mod common;

use std::{fs::File, io::BufReader};

use flate2::read::GzDecoder;
use geometric_traits::{impls::VecMatrix2D, prelude::*};
use rayon::prelude::*;
use serde::{
    Deserialize,
    de::{DeserializeSeed, IgnoredAny, MapAccess, SeqAccess, Visitor},
};

use crate::common::{dense_gth_residual_l1, flatten_dense_rows};

const LOCAL_CORPUS_PATH: &str = "local/gth_reference_corpus_1m.json.gz";
const CASE_TOLERANCE: f64 = 1e-10;
const BATCH_SIZE: usize = 1024;
const EXPECTED_CASE_COUNT: usize = 1_000_000;

#[derive(Debug, Deserialize)]
struct Case {
    name: String,
    shape: [usize; 2],
    matrix: Vec<Vec<f64>>,
    stationary: Vec<f64>,
}

#[derive(Debug)]
struct BatchSummary {
    case_count: usize,
    max_abs_error: f64,
    max_residual_l1: f64,
}

#[derive(Debug, Default)]
struct ValidationSummary {
    case_count: usize,
    max_abs_error: f64,
    max_residual_l1: f64,
}

fn validate_case(case: &Case) -> (f64, f64) {
    let matrix = VecMatrix2D::new(case.shape[0], case.shape[1], flatten_dense_rows(&case.matrix));
    let result = matrix.gth(&GthConfig::default()).unwrap_or_else(|error| {
        panic!("case {}: GTH failed with error {error}", case.name);
    });

    let max_abs_error = result
        .stationary()
        .iter()
        .zip(&case.stationary)
        .map(|(got, want)| (got - want).abs())
        .fold(0.0, f64::max);

    assert!(
        max_abs_error <= CASE_TOLERANCE,
        "case {} exceeded tolerance: max_abs_error={max_abs_error:.3e}",
        case.name
    );

    let residual = dense_gth_residual_l1(&matrix, result.stationary());
    (max_abs_error, residual)
}

fn process_batch(batch: &mut Vec<Case>, summary: &mut ValidationSummary) {
    let batch_summary = batch
        .par_iter()
        .map(validate_case)
        .fold(
            || BatchSummary { case_count: 0, max_abs_error: 0.0, max_residual_l1: 0.0 },
            |mut acc, (max_abs_error, residual)| {
                acc.case_count += 1;
                acc.max_abs_error = acc.max_abs_error.max(max_abs_error);
                acc.max_residual_l1 = acc.max_residual_l1.max(residual);
                acc
            },
        )
        .reduce(
            || BatchSummary { case_count: 0, max_abs_error: 0.0, max_residual_l1: 0.0 },
            |left, right| {
                BatchSummary {
                    case_count: left.case_count + right.case_count,
                    max_abs_error: left.max_abs_error.max(right.max_abs_error),
                    max_residual_l1: left.max_residual_l1.max(right.max_residual_l1),
                }
            },
        );

    summary.case_count += batch_summary.case_count;
    summary.max_abs_error = summary.max_abs_error.max(batch_summary.max_abs_error);
    summary.max_residual_l1 = summary.max_residual_l1.max(batch_summary.max_residual_l1);
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
        formatter.write_str("a GTH fixture object")
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
#[ignore = "requires tests/fixtures/local/gth_reference_corpus_1m.json.gz"]
fn test_local_reference_corpus_1m_parallel() {
    let path = common::fixture_path(LOCAL_CORPUS_PATH);
    assert!(path.exists(), "local 1M fixture missing at {}; generate it first", path.display());

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
        "[gth-1m-parallel] validated {} cases, max_abs_error={:.3e}, max_residual_l1={:.3e}",
        summary.case_count, summary.max_abs_error, summary.max_residual_l1
    );
}
