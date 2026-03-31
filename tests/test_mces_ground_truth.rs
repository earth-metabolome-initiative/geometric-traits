//! Ground truth tests for the MCES pipeline.
//!
//! Loads test cases generated from RDKit's RASCAL test suite
//! (`tests/fixtures/mces_ground_truth.json.gz`) and validates our
//! labeled MCES results against them.
//!
//! MassSpecGym default-corpus regeneration:
//!
//! ```bash
//! PYTHONPATH=tests/fixtures uv run --with rdkit --with tqdm python3 \
//!   tests/fixtures/generate_massspecgym_mces_ground_truth.py \
//!   --target-cases 1000 \
//!   --output tests/fixtures/massspecgym_mces_default_1000.json.gz
//!
//! PYTHONPATH=tests/fixtures uv run --with rdkit --with tqdm python3 \
//!   tests/fixtures/generate_massspecgym_mces_ground_truth.py \
//!   --target-cases 10000 \
//!   --output tests/fixtures/massspecgym_mces_default_10000.json.gz
//!
//! PYTHONPATH=tests/fixtures uv run --with rdkit --with tqdm python3 \
//!   tests/fixtures/generate_massspecgym_mces_ground_truth.py \
//!   --target-cases 200000 \
//!   --output tests/fixtures/massspecgym_mces_default_200000.json.gz
//! ```
//!
//! The `1000` fixture is the committed always-on large-corpus parity asset.
//! The `10000` fixture is the committed heavier parity asset.
//! The `200000` fixture is local-only and gitignored.
#![cfg(feature = "std")]

#[path = "test_mces_ground_truth/fixture_options.rs"]
mod fixture_options;
#[path = "test_mces_ground_truth/parity.rs"]
mod parity;
#[path = "test_mces_ground_truth/support.rs"]
mod support;
