//! Tests for graph similarity metrics.

use geometric_traits::prelude::*;

// ---------------------------------------------------------------------------
// Test helper: a simple struct implementing GraphSimilarities
// ---------------------------------------------------------------------------

struct MockResult {
    common_e: usize,
    common_v: usize,
    v1: usize,
    e1: usize,
    v2: usize,
    e2: usize,
}

impl GraphSimilarities for MockResult {
    fn common_edges(&self) -> usize {
        self.common_e
    }
    fn common_vertices(&self) -> usize {
        self.common_v
    }
    fn first_graph_vertices(&self) -> usize {
        self.v1
    }
    fn first_graph_edges(&self) -> usize {
        self.e1
    }
    fn second_graph_vertices(&self) -> usize {
        self.v2
    }
    fn second_graph_edges(&self) -> usize {
        self.e2
    }
}

// ---------------------------------------------------------------------------
// Helper
// ---------------------------------------------------------------------------

fn approx_eq(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-12
}

// ===========================================================================
// Free function tests
// ===========================================================================

// --- Tanimoto / Jaccard ---

#[test]
fn test_tanimoto_basic() {
    // intersection=3, a=5, b=4 → union=6 → 3/6 = 0.5
    assert!(approx_eq(tanimoto_similarity(3, 5, 4), 0.5));
}

#[test]
fn test_tanimoto_identical() {
    // Both sets identical: intersection = size_a = size_b
    assert!(approx_eq(tanimoto_similarity(7, 7, 7), 1.0));
}

#[test]
fn test_tanimoto_disjoint() {
    // No overlap: intersection=0, a=3, b=5 → 0/8 = 0
    assert!(approx_eq(tanimoto_similarity(0, 3, 5), 0.0));
}

#[test]
fn test_tanimoto_both_empty() {
    assert!(approx_eq(tanimoto_similarity(0, 0, 0), 1.0));
}

#[test]
fn test_tanimoto_one_empty() {
    assert!(approx_eq(tanimoto_similarity(0, 0, 5), 0.0));
    assert!(approx_eq(tanimoto_similarity(0, 5, 0), 0.0));
}

#[test]
fn test_tanimoto_subset() {
    // A ⊆ B: intersection = size_a = 3, size_b = 7 → 3/(3+7-3) = 3/7
    assert!(approx_eq(tanimoto_similarity(3, 3, 7), 3.0 / 7.0));
}

// --- Dice / Sorensen-Dice ---

#[test]
fn test_dice_basic() {
    // intersection=3, a=5, b=4 → 6/9
    assert!(approx_eq(dice_similarity(3, 5, 4), 6.0 / 9.0));
}

#[test]
fn test_dice_identical() {
    assert!(approx_eq(dice_similarity(7, 7, 7), 1.0));
}

#[test]
fn test_dice_disjoint() {
    assert!(approx_eq(dice_similarity(0, 3, 5), 0.0));
}

#[test]
fn test_dice_both_empty() {
    assert!(approx_eq(dice_similarity(0, 0, 0), 1.0));
}

#[test]
fn test_dice_one_empty() {
    assert!(approx_eq(dice_similarity(0, 0, 5), 0.0));
    assert!(approx_eq(dice_similarity(0, 5, 0), 0.0));
}

// --- Overlap / Szymkiewicz-Simpson ---

#[test]
fn test_overlap_basic() {
    // intersection=3, a=5, b=4 → 3/min(5,4) = 3/4 = 0.75
    assert!(approx_eq(overlap_similarity(3, 5, 4), 0.75));
}

#[test]
fn test_overlap_identical() {
    assert!(approx_eq(overlap_similarity(7, 7, 7), 1.0));
}

#[test]
fn test_overlap_disjoint() {
    assert!(approx_eq(overlap_similarity(0, 3, 5), 0.0));
}

#[test]
fn test_overlap_both_empty() {
    assert!(approx_eq(overlap_similarity(0, 0, 0), 1.0));
}

#[test]
fn test_overlap_one_empty() {
    // min(0,5) = 0, but one is non-empty → 0.0
    assert!(approx_eq(overlap_similarity(0, 0, 5), 0.0));
    assert!(approx_eq(overlap_similarity(0, 5, 0), 0.0));
}

#[test]
fn test_overlap_subset() {
    // A ⊆ B: intersection=3, a=3, b=7 → 3/min(3,7) = 3/3 = 1.0
    assert!(approx_eq(overlap_similarity(3, 3, 7), 1.0));
}

// --- Cosine ---

#[test]
fn test_cosine_basic() {
    // intersection=3, a=5, b=4 → 3/sqrt(20)
    let expected = 3.0 / (20.0_f64).sqrt();
    assert!(approx_eq(cosine_similarity(3, 5, 4), expected));
}

#[test]
fn test_cosine_identical() {
    // intersection=7, a=7, b=7 → 7/sqrt(49) = 7/7 = 1.0
    assert!(approx_eq(cosine_similarity(7, 7, 7), 1.0));
}

#[test]
fn test_cosine_disjoint() {
    assert!(approx_eq(cosine_similarity(0, 3, 5), 0.0));
}

#[test]
fn test_cosine_both_empty() {
    assert!(approx_eq(cosine_similarity(0, 0, 0), 1.0));
}

#[test]
fn test_cosine_one_empty() {
    // sqrt(0 * 5) = 0, one non-empty → 0.0
    assert!(approx_eq(cosine_similarity(0, 0, 5), 0.0));
    assert!(approx_eq(cosine_similarity(0, 5, 0), 0.0));
}

// --- Johnson ---

#[test]
fn test_johnson_basic() {
    // common_e=2, common_v=3, v1=4, e1=3, v2=5, e2=4
    // (2+3)^2 / ((4+3)*(5+4)) = 25 / 63
    let expected = 25.0 / 63.0;
    assert!(approx_eq(johnson_similarity(2, 3, 4, 3, 5, 4), expected));
}

#[test]
fn test_johnson_identical() {
    // Two identical graphs: v=5, e=7, all matched.
    // (7+5)^2 / ((5+7)*(5+7)) = 144/144 = 1.0
    assert!(approx_eq(johnson_similarity(7, 5, 5, 7, 5, 7), 1.0));
}

#[test]
fn test_johnson_no_match() {
    // No common edges or vertices.
    // (0+0)^2 / ((5+3)*(7+4)) = 0/88 = 0.0
    assert!(approx_eq(johnson_similarity(0, 0, 5, 3, 7, 4), 0.0));
}

#[test]
fn test_johnson_both_empty() {
    assert!(approx_eq(johnson_similarity(0, 0, 0, 0, 0, 0), 1.0));
}

#[test]
fn test_johnson_one_empty() {
    // First graph empty, second non-empty → denom_first=0 → denominator=0 → 1.0?
    // Actually: (0+0)^2 / (0 * (5+3)) = 0/0 → 1.0
    // This is a degenerate case: comparing an empty graph with a non-empty one.
    // denominator = (0+0) * (5+3) = 0 → returns 1.0 by our convention.
    assert!(approx_eq(johnson_similarity(0, 0, 0, 0, 5, 3), 1.0));
}

// --- Tversky ---

#[test]
fn test_tversky_recovers_jaccard() {
    // alpha=beta=1 → Jaccard
    let t = tversky_similarity(3, 5, 4, 1.0, 1.0);
    let j = tanimoto_similarity(3, 5, 4);
    assert!(approx_eq(t, j));
}

#[test]
fn test_tversky_recovers_dice() {
    // alpha=beta=0.5 → Dice
    let t = tversky_similarity(3, 5, 4, 0.5, 0.5);
    let d = dice_similarity(3, 5, 4);
    assert!(approx_eq(t, d));
}

#[test]
fn test_tversky_asymmetric() {
    // With different alpha/beta, swapping a and b changes the result.
    let t1 = tversky_similarity(3, 5, 4, 0.2, 0.8);
    let t2 = tversky_similarity(3, 4, 5, 0.2, 0.8);
    assert!(!approx_eq(t1, t2), "tversky should be asymmetric when alpha != beta");
}

#[test]
fn test_tversky_substructure_like() {
    // Small alpha: permissive to query being a subset.
    // intersection=3, a=3 (query, fully matched), b=10 (large target)
    // alpha=0.05, beta=0.95 → 3/(3 + 0.05*0 + 0.95*7) = 3/(3+6.65) = 3/9.65
    let t = tversky_similarity(3, 3, 10, 0.05, 0.95);
    let expected = 3.0 / (3.0 + 0.05 * 0.0 + 0.95 * 7.0);
    assert!(approx_eq(t, expected));
}

#[test]
fn test_tversky_both_empty() {
    assert!(approx_eq(tversky_similarity(0, 0, 0, 1.0, 1.0), 1.0));
}

#[test]
fn test_tversky_identical() {
    // When a=b=intersection, diff_a=diff_b=0, denom=intersection → 1.0
    assert!(approx_eq(tversky_similarity(7, 7, 7, 0.3, 0.8), 1.0));
}

// --- Kulczynski ---

#[test]
fn test_kulczynski_basic() {
    // intersection=3, a=5, b=4 → 0.5*(3/5 + 3/4) = 0.5*(0.6+0.75) = 0.675
    assert!(approx_eq(kulczynski_similarity(3, 5, 4), 0.675));
}

#[test]
fn test_kulczynski_identical() {
    // 0.5*(7/7 + 7/7) = 1.0
    assert!(approx_eq(kulczynski_similarity(7, 7, 7), 1.0));
}

#[test]
fn test_kulczynski_disjoint() {
    assert!(approx_eq(kulczynski_similarity(0, 3, 5), 0.0));
}

#[test]
fn test_kulczynski_both_empty() {
    assert!(approx_eq(kulczynski_similarity(0, 0, 0), 1.0));
}

#[test]
fn test_kulczynski_one_empty() {
    assert!(approx_eq(kulczynski_similarity(0, 0, 5), 0.0));
    assert!(approx_eq(kulczynski_similarity(0, 5, 0), 0.0));
}

#[test]
fn test_kulczynski_symmetric() {
    assert!(approx_eq(kulczynski_similarity(3, 5, 4), kulczynski_similarity(3, 4, 5)));
}

// --- Braun-Blanquet ---

#[test]
fn test_braun_blanquet_basic() {
    // intersection=3, a=5, b=4 → 3/max(5,4) = 3/5 = 0.6
    assert!(approx_eq(braun_blanquet_similarity(3, 5, 4), 0.6));
}

#[test]
fn test_braun_blanquet_identical() {
    assert!(approx_eq(braun_blanquet_similarity(7, 7, 7), 1.0));
}

#[test]
fn test_braun_blanquet_disjoint() {
    assert!(approx_eq(braun_blanquet_similarity(0, 3, 5), 0.0));
}

#[test]
fn test_braun_blanquet_both_empty() {
    assert!(approx_eq(braun_blanquet_similarity(0, 0, 0), 1.0));
}

#[test]
fn test_braun_blanquet_leq_overlap() {
    // Braun-Blanquet (max) <= Overlap (min) always.
    let cases: &[(usize, usize, usize)] =
        &[(3, 5, 4), (1, 5, 3), (2, 10, 10), (0, 0, 0), (7, 7, 7)];
    for &(i, a, b) in cases {
        let bb = braun_blanquet_similarity(i, a, b);
        let o = overlap_similarity(i, a, b);
        assert!(bb <= o + 1e-12, "braun_blanquet({i},{a},{b})={bb} > overlap({i},{a},{b})={o}");
    }
}

// --- Sokal-Sneath ---

#[test]
fn test_sokal_sneath_basic() {
    // intersection=3, a=5, b=4 → mismatches=5+4-6=3
    // 3/(3 + 2*3) = 3/9 = 1/3
    assert!(approx_eq(sokal_sneath_similarity(3, 5, 4), 1.0 / 3.0));
}

#[test]
fn test_sokal_sneath_identical() {
    // mismatches=0 → 7/(7+0) = 1.0
    assert!(approx_eq(sokal_sneath_similarity(7, 7, 7), 1.0));
}

#[test]
fn test_sokal_sneath_disjoint() {
    // intersection=0, mismatches=3+5=8 → 0/(0+16)=0
    assert!(approx_eq(sokal_sneath_similarity(0, 3, 5), 0.0));
}

#[test]
fn test_sokal_sneath_both_empty() {
    assert!(approx_eq(sokal_sneath_similarity(0, 0, 0), 1.0));
}

#[test]
fn test_sokal_sneath_leq_tanimoto() {
    // Sokal-Sneath <= Tanimoto always (stricter).
    let cases: &[(usize, usize, usize)] =
        &[(3, 5, 4), (1, 5, 3), (2, 10, 10), (0, 0, 0), (7, 7, 7), (0, 3, 5)];
    for &(i, a, b) in cases {
        let ss = sokal_sneath_similarity(i, a, b);
        let t = tanimoto_similarity(i, a, b);
        assert!(ss <= t + 1e-12, "sokal_sneath({i},{a},{b})={ss} > tanimoto({i},{a},{b})={t}");
    }
}

// --- McConnaughey ---

#[test]
fn test_mcconnaughey_basic() {
    // intersection=3, a=5, b=4 → (9 - 2*1)/(5*4) = 7/20 = 0.35
    assert!(approx_eq(mcconnaughey_similarity(3, 5, 4), 0.35));
}

#[test]
fn test_mcconnaughey_identical() {
    // intersection=7, a=7, b=7 → (49 - 0)/(49) = 1.0
    assert!(approx_eq(mcconnaughey_similarity(7, 7, 7), 1.0));
}

#[test]
fn test_mcconnaughey_disjoint() {
    // intersection=0, a=3, b=5 → (0 - 15)/(15) = -1.0
    assert!(approx_eq(mcconnaughey_similarity(0, 3, 5), -1.0));
}

#[test]
fn test_mcconnaughey_both_empty() {
    assert!(approx_eq(mcconnaughey_similarity(0, 0, 0), 1.0));
}

#[test]
fn test_mcconnaughey_one_empty() {
    assert!(approx_eq(mcconnaughey_similarity(0, 0, 5), 0.0));
    assert!(approx_eq(mcconnaughey_similarity(0, 5, 0), 0.0));
}

#[test]
fn test_mcconnaughey_negative() {
    // Small overlap, large sets → negative
    // intersection=1, a=10, b=10 → (1 - 81)/100 = -80/100 = -0.8
    assert!(approx_eq(mcconnaughey_similarity(1, 10, 10), -0.8));
}

#[test]
fn test_mcconnaughey_range() {
    // McConnaughey is in [-1, 1]
    let cases: &[(usize, usize, usize)] =
        &[(0, 0, 0), (0, 1, 1), (1, 1, 1), (1, 5, 3), (3, 5, 4), (0, 3, 5), (1, 10, 10)];
    for &(i, a, b) in cases {
        let m = mcconnaughey_similarity(i, a, b);
        assert!((-1.0..=1.0 + 1e-12).contains(&m), "mcconnaughey({i},{a},{b}) = {m} out of [-1,1]");
    }
}

// ===========================================================================
// Mathematical property tests
// ===========================================================================

#[test]
fn test_unit_range_metrics() {
    // All metrics except McConnaughey should be in [0, 1].
    let cases: &[(usize, usize, usize)] = &[
        (0, 0, 0),
        (0, 1, 1),
        (1, 1, 1),
        (1, 5, 3),
        (3, 5, 4),
        (2, 10, 10),
        (10, 10, 10),
        (0, 100, 200),
    ];
    for &(i, a, b) in cases {
        for (name, val) in [
            ("tanimoto", tanimoto_similarity(i, a, b)),
            ("dice", dice_similarity(i, a, b)),
            ("overlap", overlap_similarity(i, a, b)),
            ("cosine", cosine_similarity(i, a, b)),
            ("tversky(1,1)", tversky_similarity(i, a, b, 1.0, 1.0)),
            ("kulczynski", kulczynski_similarity(i, a, b)),
            ("braun_blanquet", braun_blanquet_similarity(i, a, b)),
            ("sokal_sneath", sokal_sneath_similarity(i, a, b)),
        ] {
            assert!((0.0..=1.0 + 1e-12).contains(&val), "{name}({i},{a},{b}) = {val} out of [0,1]");
        }
    }
}

#[test]
fn test_symmetry() {
    // All symmetric metrics should produce the same result when swapping a and b.
    let cases: &[(usize, usize, usize)] = &[(3, 5, 4), (0, 3, 7), (1, 1, 10), (2, 8, 2), (0, 0, 0)];
    for &(i, a, b) in cases {
        for (name, v1, v2) in [
            ("tanimoto", tanimoto_similarity(i, a, b), tanimoto_similarity(i, b, a)),
            ("dice", dice_similarity(i, a, b), dice_similarity(i, b, a)),
            ("overlap", overlap_similarity(i, a, b), overlap_similarity(i, b, a)),
            ("cosine", cosine_similarity(i, a, b), cosine_similarity(i, b, a)),
            ("kulczynski", kulczynski_similarity(i, a, b), kulczynski_similarity(i, b, a)),
            (
                "braun_blanquet",
                braun_blanquet_similarity(i, a, b),
                braun_blanquet_similarity(i, b, a),
            ),
            ("sokal_sneath", sokal_sneath_similarity(i, a, b), sokal_sneath_similarity(i, b, a)),
            ("mcconnaughey", mcconnaughey_similarity(i, a, b), mcconnaughey_similarity(i, b, a)),
        ] {
            assert!(approx_eq(v1, v2), "{name} not symmetric for ({i},{a},{b})");
        }
    }
}

#[test]
fn test_johnson_symmetry() {
    // Swapping (v1,e1) <-> (v2,e2) must give the same result.
    let j1 = johnson_similarity(3, 4, 5, 6, 7, 8);
    let j2 = johnson_similarity(3, 4, 7, 8, 5, 6);
    assert!(approx_eq(j1, j2));
}

#[test]
fn test_dice_geq_tanimoto() {
    // Dice >= Tanimoto for all valid inputs (provable inequality).
    let cases: &[(usize, usize, usize)] =
        &[(0, 0, 0), (0, 1, 1), (1, 1, 1), (1, 5, 3), (3, 5, 4), (2, 10, 10)];
    for &(i, a, b) in cases {
        let t = tanimoto_similarity(i, a, b);
        let d = dice_similarity(i, a, b);
        assert!(d >= t - 1e-12, "dice({i},{a},{b})={d} < tanimoto({i},{a},{b})={t}");
    }
}

#[test]
fn test_overlap_geq_tanimoto() {
    // Overlap >= Tanimoto for all valid inputs.
    let cases: &[(usize, usize, usize)] =
        &[(0, 0, 0), (0, 1, 1), (1, 1, 1), (1, 5, 3), (3, 5, 4), (2, 10, 10)];
    for &(i, a, b) in cases {
        let t = tanimoto_similarity(i, a, b);
        let o = overlap_similarity(i, a, b);
        assert!(o >= t - 1e-12, "overlap({i},{a},{b})={o} < tanimoto({i},{a},{b})={t}");
    }
}

// ===========================================================================
// Trait default method tests
// ===========================================================================

#[test]
fn test_trait_edge_jaccard() {
    let r = MockResult { common_e: 3, common_v: 0, v1: 0, e1: 5, v2: 0, e2: 4 };
    assert!(approx_eq(r.edge_jaccard_similarity(), tanimoto_similarity(3, 5, 4)));
}

#[test]
fn test_trait_edge_dice() {
    let r = MockResult { common_e: 3, common_v: 0, v1: 0, e1: 5, v2: 0, e2: 4 };
    assert!(approx_eq(r.edge_dice_similarity(), dice_similarity(3, 5, 4)));
}

#[test]
fn test_trait_edge_overlap() {
    let r = MockResult { common_e: 3, common_v: 0, v1: 0, e1: 5, v2: 0, e2: 4 };
    assert!(approx_eq(r.edge_overlap_similarity(), overlap_similarity(3, 5, 4)));
}

#[test]
fn test_trait_edge_cosine() {
    let r = MockResult { common_e: 3, common_v: 0, v1: 0, e1: 5, v2: 0, e2: 4 };
    assert!(approx_eq(r.edge_cosine_similarity(), cosine_similarity(3, 5, 4)));
}

#[test]
fn test_trait_vertex_jaccard() {
    let r = MockResult { common_e: 0, common_v: 4, v1: 6, e1: 0, v2: 8, e2: 0 };
    assert!(approx_eq(r.vertex_jaccard_similarity(), tanimoto_similarity(4, 6, 8)));
}

#[test]
fn test_trait_vertex_dice() {
    let r = MockResult { common_e: 0, common_v: 4, v1: 6, e1: 0, v2: 8, e2: 0 };
    assert!(approx_eq(r.vertex_dice_similarity(), dice_similarity(4, 6, 8)));
}

#[test]
fn test_trait_vertex_overlap() {
    let r = MockResult { common_e: 0, common_v: 4, v1: 6, e1: 0, v2: 8, e2: 0 };
    assert!(approx_eq(r.vertex_overlap_similarity(), overlap_similarity(4, 6, 8)));
}

#[test]
fn test_trait_vertex_cosine() {
    let r = MockResult { common_e: 0, common_v: 4, v1: 6, e1: 0, v2: 8, e2: 0 };
    assert!(approx_eq(r.vertex_cosine_similarity(), cosine_similarity(4, 6, 8)));
}

#[test]
fn test_trait_johnson() {
    let r = MockResult { common_e: 2, common_v: 3, v1: 4, e1: 3, v2: 5, e2: 4 };
    assert!(approx_eq(r.johnson_similarity(), johnson_similarity(2, 3, 4, 3, 5, 4)));
}

#[test]
fn test_trait_edge_tversky() {
    let r = MockResult { common_e: 3, common_v: 0, v1: 0, e1: 5, v2: 0, e2: 4 };
    assert!(approx_eq(r.edge_tversky_similarity(0.3, 0.7), tversky_similarity(3, 5, 4, 0.3, 0.7)));
}

#[test]
fn test_trait_vertex_tversky() {
    let r = MockResult { common_e: 0, common_v: 4, v1: 6, e1: 0, v2: 8, e2: 0 };
    assert!(approx_eq(
        r.vertex_tversky_similarity(1.0, 1.0),
        tversky_similarity(4, 6, 8, 1.0, 1.0)
    ));
}

#[test]
fn test_trait_edge_kulczynski() {
    let r = MockResult { common_e: 3, common_v: 0, v1: 0, e1: 5, v2: 0, e2: 4 };
    assert!(approx_eq(r.edge_kulczynski_similarity(), kulczynski_similarity(3, 5, 4)));
}

#[test]
fn test_trait_edge_braun_blanquet() {
    let r = MockResult { common_e: 3, common_v: 0, v1: 0, e1: 5, v2: 0, e2: 4 };
    assert!(approx_eq(r.edge_braun_blanquet_similarity(), braun_blanquet_similarity(3, 5, 4)));
}

#[test]
fn test_trait_edge_sokal_sneath() {
    let r = MockResult { common_e: 3, common_v: 0, v1: 0, e1: 5, v2: 0, e2: 4 };
    assert!(approx_eq(r.edge_sokal_sneath_similarity(), sokal_sneath_similarity(3, 5, 4)));
}

#[test]
fn test_trait_edge_mcconnaughey() {
    let r = MockResult { common_e: 3, common_v: 0, v1: 0, e1: 5, v2: 0, e2: 4 };
    assert!(approx_eq(r.edge_mcconnaughey_similarity(), mcconnaughey_similarity(3, 5, 4)));
}

#[test]
fn test_trait_all_empty() {
    let r = MockResult { common_e: 0, common_v: 0, v1: 0, e1: 0, v2: 0, e2: 0 };
    assert!(approx_eq(r.edge_jaccard_similarity(), 1.0));
    assert!(approx_eq(r.edge_dice_similarity(), 1.0));
    assert!(approx_eq(r.edge_overlap_similarity(), 1.0));
    assert!(approx_eq(r.edge_cosine_similarity(), 1.0));
    assert!(approx_eq(r.edge_tversky_similarity(1.0, 1.0), 1.0));
    assert!(approx_eq(r.edge_kulczynski_similarity(), 1.0));
    assert!(approx_eq(r.edge_braun_blanquet_similarity(), 1.0));
    assert!(approx_eq(r.edge_sokal_sneath_similarity(), 1.0));
    assert!(approx_eq(r.edge_mcconnaughey_similarity(), 1.0));
    assert!(approx_eq(r.vertex_jaccard_similarity(), 1.0));
    assert!(approx_eq(r.vertex_dice_similarity(), 1.0));
    assert!(approx_eq(r.vertex_overlap_similarity(), 1.0));
    assert!(approx_eq(r.vertex_cosine_similarity(), 1.0));
    assert!(approx_eq(r.vertex_tversky_similarity(0.5, 0.5), 1.0));
    assert!(approx_eq(r.vertex_kulczynski_similarity(), 1.0));
    assert!(approx_eq(r.vertex_braun_blanquet_similarity(), 1.0));
    assert!(approx_eq(r.vertex_sokal_sneath_similarity(), 1.0));
    assert!(approx_eq(r.vertex_mcconnaughey_similarity(), 1.0));
    assert!(approx_eq(r.johnson_similarity(), 1.0));
}
