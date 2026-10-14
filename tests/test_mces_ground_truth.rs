//! Ground truth tests for the MCES pipeline.
//!
//! Loads test cases generated from RDKit's RASCAL test suite
//! (`tests/fixtures/mces_ground_truth.json.gz`) and validates our
//! labeled MCES results against them.
//!
//! The `1000` fixture is the committed large-corpus parity asset and is
//! ignored by default.
//! The `10000` fixture is the committed heavier parity asset.
#![cfg(feature = "std")]

#[path = "test_mces_ground_truth/fixture_options.rs"]
mod fixture_options;
#[path = "test_mces_ground_truth/parity.rs"]
mod parity;
#[path = "test_mces_ground_truth/support.rs"]
mod support;
