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

mod fixture_options;
mod parity;
mod support;
