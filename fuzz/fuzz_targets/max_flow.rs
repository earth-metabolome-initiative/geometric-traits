//! Fuzz harness for the maximum s-t flow algorithms `Dinic` and `EdmondsKarp`.
//!
//! The arbitrary valued matrix is sanitized two complementary ways:
//!
//! * as a square directed capacity graph, where both algorithms must produce a
//!   feasible conserved flow with a saturated minimum cut equal to the flow (a
//!   self-contained optimality certificate), agree with each other on the flow
//!   value, and be deterministic;
//! * as a bipartite graph, where the unit-capacity flow value of each algorithm
//!   must equal the maximum matching reported by the already-implemented
//!   Hopcroft-Karp matcher.
//!
//! Running both checkers to completion (no panic) is the baseline oracle.

#![no_main]
use arbitrary::Unstructured;
use geometric_traits::{
    impls::ValuedCSR2D,
    test_utils::{check_max_flow_invariants, check_max_flow_matches_hopcroft_karp},
};
use libfuzzer_sys::fuzz_target;

/// Arbitrary input matrix type, mirroring the other valued-graph fuzz targets.
type Csr = ValuedCSR2D<u16, u8, u8, u32>;

// `arbitrary`, not `arbitrary_take_rest`, so crash files replay in tests
fuzz_target!(|bytes: &[u8]| {
    let Ok(csr) = Unstructured::new(bytes).arbitrary::<Csr>() else {
        return;
    };
    check_max_flow_invariants(&csr);
    check_max_flow_matches_hopcroft_karp(&csr);
});
