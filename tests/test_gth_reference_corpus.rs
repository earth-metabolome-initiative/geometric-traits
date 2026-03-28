//! Regression tests against the 1k dense GTH reference corpus.
#![cfg(feature = "std")]

mod common;

use std::io::Read as _;

use flate2::read::GzDecoder;
use geometric_traits::{impls::VecMatrix2D, prelude::*};
use serde::Deserialize;

use crate::common::flatten_dense_rows;

#[derive(Deserialize)]
struct Fixture {
    schema_version: u32,
    summary: Summary,
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Summary {
    case_count: usize,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    shape: [usize; 2],
    matrix: Vec<Vec<f64>>,
    stationary: Vec<f64>,
}

fn fixture() -> Fixture {
    let ground_truth_gz = common::read_fixture("gth_reference_corpus_1k.json.gz");
    let mut json = String::new();
    GzDecoder::new(ground_truth_gz.as_slice())
        .read_to_string(&mut json)
        .expect("gzip decompression failed");
    serde_json::from_str(&json)
        .expect("`tests/fixtures/gth_reference_corpus_1k.json.gz` must contain valid JSON")
}

#[test]
fn test_reference_corpus_metadata() {
    let fixture = fixture();
    assert_eq!(fixture.schema_version, 1);
    assert_eq!(fixture.summary.case_count, 1_000);
    assert_eq!(fixture.cases.len(), 1_000);
}

#[test]
fn test_reference_corpus_cases() {
    let fixture = fixture();
    let config = GthConfig::default();
    for case in fixture.cases {
        let matrix =
            VecMatrix2D::new(case.shape[0], case.shape[1], flatten_dense_rows(&case.matrix));
        let result = matrix.gth(&config).unwrap();
        assert_eq!(result.order(), case.shape[0], "case {}", case.name);
        for (index, (got, want)) in result.stationary().iter().zip(&case.stationary).enumerate() {
            assert!(
                (got - want).abs() <= 1e-10,
                "case {} index {} expected {} got {}",
                case.name,
                index,
                want,
                got
            );
        }
    }
}
