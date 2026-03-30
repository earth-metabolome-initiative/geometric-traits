use std::boxed::Box;

use super::{
    test_support::{SchedulerMirrorTestExt, TestAccessorExt},
    *,
};
use crate::{
    impls::ValuedCSR2D,
    traits::{MatrixMut, SparseMatrixMut, algorithms::blossom_v::BlossomV},
};

type Vcsr = ValuedCSR2D<usize, usize, usize, i32>;

fn build_graph(n: usize, edges: &[(usize, usize, i32)]) -> Vcsr {
    let mut sorted: Vec<(usize, usize, i32)> = Vec::new();
    for &(i, j, w) in edges {
        if i == j {
            continue;
        }
        let (lo, hi) = if i < j { (i, j) } else { (j, i) };
        sorted.push((lo, hi, w));
        sorted.push((hi, lo, w));
    }
    sorted.sort_unstable();
    sorted.dedup_by(|a, b| a.0 == b.0 && a.1 == b.1);
    let mut vcsr: Vcsr = SparseMatrixMut::with_sparse_shaped_capacity((n, n), sorted.len());
    for (r, c, v) in sorted {
        MatrixMut::add(&mut vcsr, (r, c, v)).unwrap();
    }
    vcsr
}

fn case_474_edges() -> Vec<(usize, usize, i32)> {
    vec![
        (0, 1, 11),
        (2, 4, 53),
        (3, 5, -49),
        (1, 4, 88),
        (2, 3, -27),
        (1, 2, -42),
        (0, 5, 96),
        (4, 5, 33),
        (3, 4, -62),
        (0, 2, 43),
    ]
}

fn case_9_edges() -> Vec<(usize, usize, i32)> {
    vec![
        (0, 1, 0),
        (0, 3, -21251),
        (0, 6, -2023),
        (0, 9, 14768),
        (0, 12, 12819),
        (0, 14, 0),
        (0, 15, 0),
        (0, 16, -27420),
        (0, 17, -26215),
        (1, 3, -1),
        (1, 5, 32512),
        (1, 9, -30271),
        (1, 10, 5020),
        (1, 13, 12937),
        (2, 3, 2303),
        (2, 4, 100),
        (2, 14, 76),
        (2, 16, 26984),
        (2, 17, -20523),
        (3, 4, 15679),
        (3, 6, -1),
        (3, 12, 3072),
        (3, 15, 22123),
        (3, 16, -13726),
        (4, 5, 2752),
        (4, 8, 26125),
        (4, 17, -18671),
        (5, 8, 12331),
        (5, 14, -10251),
        (6, 7, -30029),
        (6, 10, -10397),
        (6, 11, -23283),
        (7, 9, 13364),
        (8, 9, -2846),
        (8, 10, -1387),
        (8, 12, -24415),
        (8, 15, -18235),
        (9, 10, -26215),
        (9, 13, 21062),
        (9, 14, -26215),
        (9, 16, -18577),
        (10, 11, -12279),
        (10, 13, -8642),
        (11, 13, -7374),
        (11, 14, 32018),
        (12, 14, 14393),
        (12, 15, -24),
        (12, 17, 50),
        (14, 17, 1128),
    ]
}

fn case_97_edges() -> Vec<(usize, usize, i32)> {
    vec![
        (0, 1, 94),
        (0, 2, 62),
        (0, 3, -67),
        (0, 4, -71),
        (0, 5, -32),
        (0, 6, 71),
        (0, 7, 47),
        (0, 8, -70),
        (0, 9, 32),
        (0, 10, 85),
        (0, 11, 71),
        (0, 12, -43),
        (1, 2, 99),
        (1, 3, 14),
        (1, 4, 82),
        (1, 5, 71),
        (1, 7, 65),
        (1, 8, 99),
        (1, 9, -85),
        (1, 17, 43),
        (2, 3, -82),
        (2, 4, 74),
        (2, 6, -8),
        (2, 10, 27),
        (2, 11, 40),
        (2, 16, 41),
        (2, 17, -40),
        (3, 4, -6),
        (3, 5, 56),
        (3, 6, -6),
        (3, 7, -12),
        (3, 8, 26),
        (3, 11, 94),
        (3, 12, 19),
        (3, 13, -95),
        (3, 14, -7),
        (3, 15, -77),
        (3, 17, -74),
        (4, 5, 65),
        (4, 6, 23),
        (4, 7, -21),
        (4, 11, 37),
        (4, 12, -83),
        (4, 14, -100),
        (5, 13, -19),
        (5, 15, 57),
        (6, 9, -91),
        (7, 8, -11),
        (7, 9, -16),
        (7, 14, -76),
        (7, 15, 95),
        (8, 10, -86),
        (8, 13, 3),
        (8, 14, -14),
        (8, 16, 11),
        (9, 10, -5),
        (9, 12, 41),
        (9, 15, 36),
        (10, 13, 73),
        (10, 16, 35),
        (11, 16, 74),
        (13, 17, 93),
    ]
}

fn case_honggfuzz_sigabrt_4_edges() -> Vec<(usize, usize, i32)> {
    vec![
        (0, 1, -29481),
        (0, 4, -23385),
        (0, 6, -9713),
        (0, 13, 3660),
        (0, 14, 13857),
        (0, 16, 0),
        (0, 18, 0),
        (0, 19, -8688),
        (0, 20, 29128),
        (0, 21, -1),
        (1, 14, 10906),
        (1, 17, -28356),
        (1, 20, 0),
        (2, 4, -27066),
        (2, 6, -9498),
        (2, 11, 17867),
        (2, 12, 0),
        (2, 13, -14016),
        (2, 16, 130),
        (2, 17, 7281),
        (2, 18, 32281),
        (2, 19, -16009),
        (3, 5, 6243),
        (3, 7, -18728),
        (3, 9, 3233),
        (3, 16, 28116),
        (4, 12, -6480),
        (5, 11, -28628),
        (5, 14, -12713),
        (5, 20, 17905),
        (6, 8, -7974),
        (6, 20, -30597),
        (6, 21, -23196),
        (6, 22, -27428),
        (7, 19, 0),
        (7, 21, -61),
        (8, 24, -6547),
        (8, 25, 1),
        (9, 14, -17579),
        (9, 17, 30917),
        (9, 22, -19162),
        (10, 15, -18927),
        (10, 19, -429),
        (10, 24, 12562),
        (11, 19, -19309),
        (11, 21, 0),
        (12, 15, -5228),
        (12, 20, 17077),
        (13, 21, -31234),
        (14, 17, 64),
        (14, 21, 4843),
        (14, 22, -16020),
        (14, 24, 16426),
        (15, 16, 0),
        (15, 20, 2492),
        (16, 18, -29415),
        (16, 23, 10546),
        (17, 24, 30942),
        (18, 23, 9509),
        (19, 21, 0),
        (19, 22, -12607),
        (24, 25, -22204),
    ]
}

fn case_honggfuzz_sigabrt_5_edges() -> Vec<(usize, usize, i32)> {
    vec![
        (0, 2, 54),
        (0, 4, 0),
        (0, 7, 364),
        (0, 11, 0),
        (0, 12, 22101),
        (0, 13, 1),
        (0, 16, 0),
        (0, 18, 2816),
        (0, 19, 24275),
        (0, 20, 21398),
        (0, 21, 0),
        (0, 24, 8379),
        (1, 4, 30776),
        (1, 6, 1),
        (1, 7, -628),
        (1, 18, -15828),
        (1, 23, 110),
        (2, 3, 8239),
        (2, 7, -14876),
        (2, 9, 455),
        (2, 11, 17867),
        (2, 16, 13954),
        (2, 17, 3199),
        (2, 22, 4058),
        (3, 7, -18728),
        (3, 9, -13058),
        (3, 22, -15953),
        (4, 5, -16511),
        (5, 7, 21845),
        (5, 11, -15360),
        (5, 22, 2816),
        (5, 24, 0),
        (6, 9, 27985),
        (6, 12, -20450),
        (6, 14, 381),
        (6, 22, 2636),
        (6, 24, 31716),
        (7, 8, 21589),
        (7, 14, -15413),
        (7, 17, 29485),
        (8, 11, 896),
        (8, 15, -318),
        (9, 12, -21845),
        (9, 18, 13613),
        (9, 19, 25273),
        (9, 22, -25404),
        (10, 11, -9253),
        (10, 17, -32074),
        (11, 15, -28291),
        (12, 16, 27181),
        (12, 21, 0),
        (12, 23, -5228),
        (12, 25, -10034),
        (13, 16, 16334),
        (13, 19, 6597),
        (13, 20, -11177),
        (13, 22, 19534),
        (14, 21, -25631),
        (14, 24, -12246),
        (16, 18, -29415),
        (16, 25, -28375),
        (17, 24, 2782),
        (18, 23, 881),
        (21, 24, 124),
        (22, 25, 21),
    ]
}

fn case_honggfuzz_sigabrt_6_edges() -> Vec<(usize, usize, i32)> {
    vec![
        (0, 2, 54),
        (0, 4, 0),
        (0, 7, 364),
        (0, 11, 0),
        (0, 12, 22101),
        (0, 13, 1),
        (0, 16, 0),
        (0, 18, 2816),
        (0, 19, 24275),
        (0, 21, 0),
        (0, 24, 8379),
        (1, 4, 30776),
        (1, 6, 1),
        (1, 7, -628),
        (1, 10, 12302),
        (1, 18, -15828),
        (1, 23, 110),
        (2, 3, 8239),
        (2, 7, -14876),
        (2, 9, 455),
        (2, 11, 17867),
        (2, 17, 3199),
        (2, 22, 4058),
        (3, 7, -18728),
        (3, 9, -13058),
        (3, 22, -15953),
        (4, 5, -16511),
        (4, 11, 31704),
        (5, 7, 21845),
        (5, 11, -15360),
        (5, 22, 2816),
        (5, 24, 0),
        (6, 9, 27985),
        (6, 12, -20450),
        (6, 14, 381),
        (6, 22, 2636),
        (6, 24, 31716),
        (7, 8, 21589),
        (7, 14, -15413),
        (8, 11, 896),
        (8, 15, -318),
        (8, 17, 19387),
        (9, 12, -21845),
        (9, 18, 13613),
        (9, 19, 25273),
        (9, 22, -25404),
        (10, 11, 23441),
        (10, 17, -32074),
        (11, 15, -28291),
        (12, 16, 27181),
        (12, 21, 0),
        (12, 23, -5228),
        (12, 25, -10034),
        (13, 16, 16334),
        (13, 19, 6597),
        (13, 20, -11177),
        (13, 22, 19534),
        (13, 24, -10229),
        (14, 21, -25631),
        (14, 24, -12246),
        (16, 18, -29415),
        (16, 25, -28375),
        (17, 24, 2782),
        (18, 23, 881),
        (21, 24, 124),
        (22, 25, 21),
    ]
}

fn case_honggfuzz_sigabrt_14_edges() -> Vec<(usize, usize, i32)> {
    vec![
        (0, 1, 0),
        (0, 2, 0),
        (0, 3, -21251),
        (0, 5, -18577),
        (0, 6, 32018),
        (0, 8, -31624),
        (0, 10, -1387),
        (0, 11, -2023),
        (0, 12, 12819),
        (0, 14, 21845),
        (0, 15, 0),
        (0, 16, -26363),
        (0, 17, 0),
        (1, 2, -12194),
        (1, 3, -10864),
        (1, 4, 0),
        (1, 5, 32512),
        (1, 7, 0),
        (1, 11, 0),
        (1, 13, 12937),
        (2, 3, 2303),
        (2, 5, 0),
        (2, 15, 13302),
        (2, 16, 26984),
        (2, 17, -20523),
        (3, 4, 15679),
        (3, 10, -1),
        (3, 12, 3072),
        (3, 14, 2920),
        (3, 15, 22123),
        (3, 16, -13726),
        (4, 16, -320),
        (5, 8, 12331),
        (5, 13, 21356),
        (5, 14, 3053),
        (6, 7, -30029),
        (6, 10, -10397),
        (6, 11, -23283),
        (7, 10, -768),
        (7, 11, -26516),
        (8, 9, 32738),
        (8, 12, -24415),
        (8, 13, 17552),
        (8, 15, -18235),
        (9, 10, -26215),
        (9, 13, 21062),
        (9, 16, 0),
        (10, 11, -12279),
        (10, 13, -8642),
        (12, 15, 0),
        (12, 17, -16846),
        (13, 16, 12857),
        (14, 17, 7981),
    ]
}

fn case_honggfuzz_sigabrt_7_edges() -> Vec<(usize, usize, i32)> {
    vec![
        (0, 2, 54),
        (0, 4, 0),
        (0, 7, 364),
        (0, 12, 22101),
        (0, 13, 1),
        (0, 15, 0),
        (0, 16, 0),
        (0, 18, 2816),
        (0, 19, 24275),
        (0, 21, 0),
        (0, 24, 8379),
        (1, 4, 30776),
        (1, 6, 1),
        (1, 7, -628),
        (1, 10, 12302),
        (1, 18, -15828),
        (1, 23, 110),
        (2, 3, 8239),
        (2, 7, -14876),
        (2, 9, 455),
        (2, 11, 17867),
        (2, 16, 14210),
        (2, 22, 4058),
        (3, 7, -18728),
        (3, 9, -13058),
        (3, 22, -15953),
        (4, 5, -16511),
        (4, 11, 31704),
        (5, 7, 21845),
        (5, 11, 0),
        (5, 22, 2816),
        (5, 24, 0),
        (6, 9, 27985),
        (6, 12, -20450),
        (6, 22, 2636),
        (6, 24, 31716),
        (7, 8, 21589),
        (7, 14, -15413),
        (7, 17, 29485),
        (8, 11, 896),
        (8, 15, -318),
        (9, 12, -21845),
        (9, 18, 13613),
        (9, 19, 25273),
        (9, 22, -25404),
        (10, 11, 23441),
        (10, 17, -32074),
        (11, 15, -28291),
        (12, 16, 27181),
        (12, 21, 0),
        (12, 23, -5228),
        (12, 25, -10034),
        (13, 16, 16334),
        (13, 19, 6597),
        (13, 20, -11177),
        (13, 22, 19534),
        (14, 21, -25631),
        (14, 24, -12246),
        (16, 18, -29415),
        (16, 25, -28375),
        (17, 20, 381),
        (17, 24, 2782),
        (18, 20, 21398),
        (18, 23, 881),
        (21, 24, 124),
        (22, 25, 21),
    ]
}

fn case_87417_edges() -> Vec<(usize, usize, i32)> {
    vec![
        (0, 11, -88),
        (0, 17, -39),
        (0, 18, 19),
        (0, 21, -22),
        (0, 23, -54),
        (0, 24, -25),
        (0, 27, 12),
        (1, 12, -8),
        (1, 15, -97),
        (1, 16, -32),
        (1, 24, 55),
        (2, 10, -49),
        (2, 15, -74),
        (2, 16, -47),
        (3, 12, 100),
        (3, 21, -65),
        (3, 29, -8),
        (4, 9, -45),
        (4, 11, -73),
        (4, 13, -97),
        (4, 15, 81),
        (4, 16, 4),
        (4, 21, -100),
        (4, 22, -25),
        (4, 27, -27),
        (4, 29, -77),
        (5, 9, 16),
        (5, 13, 57),
        (5, 26, -96),
        (5, 27, 63),
        (6, 9, 67),
        (6, 12, 82),
        (6, 13, 44),
        (6, 15, -49),
        (6, 16, 1),
        (7, 8, -8),
        (7, 11, -77),
        (7, 12, -51),
        (7, 13, -50),
        (7, 15, -77),
        (7, 17, 20),
        (7, 19, -21),
        (7, 20, 53),
        (7, 21, -50),
        (7, 22, 64),
        (7, 23, 81),
        (8, 14, -83),
        (8, 15, -25),
        (8, 18, 99),
        (8, 22, -51),
        (8, 25, -32),
        (8, 27, 84),
        (8, 29, 78),
        (9, 11, 78),
        (9, 18, 32),
        (9, 23, -71),
        (9, 29, 2),
        (10, 14, -57),
        (10, 23, -89),
        (11, 14, -22),
        (11, 16, 10),
        (11, 17, 87),
        (11, 20, -91),
        (11, 23, 17),
        (11, 24, -39),
        (11, 26, 11),
        (11, 27, 22),
        (12, 22, 87),
        (12, 23, -83),
        (12, 24, 75),
        (12, 25, -36),
        (12, 27, 12),
        (12, 28, 51),
        (13, 20, 26),
        (13, 22, -58),
        (13, 27, 26),
        (13, 28, -57),
        (14, 15, 41),
        (14, 22, -73),
        (14, 25, -63),
        (14, 26, 73),
        (14, 29, -50),
        (15, 16, 11),
        (15, 18, 28),
        (15, 19, -19),
        (15, 26, -5),
        (15, 29, 73),
        (16, 19, -67),
        (16, 20, -85),
        (16, 21, -29),
        (16, 25, -99),
        (17, 19, -79),
        (17, 24, 85),
        (18, 23, 5),
        (18, 28, 79),
        (19, 20, -86),
        (19, 21, 77),
        (20, 23, -19),
        (20, 25, -38),
        (20, 26, -83),
        (20, 28, 13),
        (21, 22, 31),
        (21, 27, -12),
        (21, 28, 97),
        (21, 29, -97),
        (22, 23, 41),
        (23, 28, -22),
        (24, 26, -76),
        (25, 28, 57),
        (26, 28, 39),
        (27, 28, 27),
    ]
}

fn case_416_edges() -> Vec<(usize, usize, i32)> {
    vec![
        (0, 1, 5),
        (0, 3, 65),
        (0, 5, 96),
        (0, 6, 63),
        (0, 8, -85),
        (0, 9, -65),
        (0, 11, -12),
        (0, 13, 0),
        (0, 15, 34),
        (0, 18, -21),
        (0, 19, 64),
        (1, 2, -3),
        (1, 3, 20),
        (1, 4, -88),
        (1, 6, -5),
        (1, 9, -23),
        (1, 10, 86),
        (1, 12, 56),
        (1, 13, 53),
        (1, 15, -50),
        (1, 18, 54),
        (1, 19, 32),
        (2, 5, -30),
        (2, 8, -82),
        (2, 9, -3),
        (2, 10, 38),
        (2, 12, -1),
        (2, 14, -43),
        (2, 15, 21),
        (2, 16, -61),
        (2, 17, 74),
        (3, 4, 2),
        (3, 5, -51),
        (3, 7, 94),
        (3, 8, 12),
        (3, 9, -48),
        (3, 14, -39),
        (3, 16, -57),
        (4, 6, -59),
        (4, 7, 18),
        (4, 8, -70),
        (4, 9, -92),
        (4, 14, 75),
        (4, 17, -89),
        (4, 18, -81),
        (5, 6, 40),
        (5, 7, -48),
        (5, 8, 17),
        (5, 9, 33),
        (5, 10, -5),
        (5, 13, 25),
        (6, 7, 33),
        (6, 8, -94),
        (6, 9, 66),
        (6, 11, 71),
        (6, 12, 98),
        (6, 15, -47),
        (6, 17, 87),
        (6, 18, 75),
        (7, 8, -15),
        (7, 9, 82),
        (7, 10, 35),
        (7, 12, -46),
        (7, 13, -63),
        (7, 14, 89),
        (7, 15, -79),
        (7, 17, 6),
        (7, 18, 15),
        (8, 9, -50),
        (8, 10, -36),
        (8, 11, -20),
        (8, 12, 74),
        (8, 14, 46),
        (8, 16, 98),
        (8, 19, 33),
        (9, 11, -92),
        (9, 12, 92),
        (9, 13, 85),
        (9, 14, 92),
        (9, 15, 23),
        (9, 17, -5),
        (10, 11, 50),
        (10, 12, -32),
        (10, 13, -14),
        (10, 14, -48),
        (10, 15, -74),
        (10, 16, 2),
        (10, 18, -85),
        (10, 19, -36),
        (11, 12, 59),
        (11, 13, 73),
        (11, 14, -94),
        (11, 15, 70),
        (11, 16, 8),
        (11, 18, 35),
        (11, 19, 71),
        (12, 14, -59),
        (12, 15, -86),
        (12, 16, 59),
        (12, 17, 54),
        (12, 18, 11),
        (12, 19, 57),
        (13, 14, 71),
        (13, 15, 57),
        (13, 16, 17),
        (13, 17, 71),
        (13, 19, -52),
        (14, 16, 73),
        (14, 17, -94),
        (14, 18, -31),
        (14, 19, -90),
        (15, 16, 92),
        (15, 17, 87),
        (15, 19, 77),
        (16, 17, -47),
        (16, 18, 75),
        (16, 19, -23),
        (17, 18, -5),
        (17, 19, 89),
        (18, 19, -93),
    ]
}

fn case_1594_edges() -> Vec<(usize, usize, i32)> {
    vec![
        (0, 3, 70),
        (0, 5, -38),
        (0, 6, 0),
        (0, 7, 11),
        (0, 10, 40),
        (0, 12, -37),
        (0, 13, 51),
        (0, 14, -21),
        (0, 15, 88),
        (0, 16, 33),
        (0, 18, -35),
        (0, 19, -77),
        (0, 20, 96),
        (0, 23, -71),
        (0, 24, 34),
        (1, 5, 2),
        (1, 6, 63),
        (1, 7, -38),
        (1, 10, -8),
        (1, 11, -82),
        (1, 14, 23),
        (1, 20, -89),
        (1, 21, -63),
        (1, 22, -29),
        (1, 23, 15),
        (2, 4, 95),
        (2, 6, 35),
        (2, 8, -9),
        (2, 11, 94),
        (2, 12, 85),
        (2, 14, -33),
        (2, 15, 65),
        (2, 16, 32),
        (2, 19, -44),
        (2, 20, -93),
        (2, 24, -19),
        (2, 25, 47),
        (3, 4, 83),
        (3, 5, -4),
        (3, 6, -31),
        (3, 7, 66),
        (3, 8, 82),
        (3, 10, 58),
        (3, 11, -14),
        (3, 14, 87),
        (3, 15, -31),
        (3, 20, 95),
        (3, 22, -48),
        (4, 5, 73),
        (4, 6, -79),
        (4, 7, -63),
        (4, 9, -84),
        (4, 10, 4),
        (4, 11, 12),
        (4, 13, -83),
        (4, 14, -45),
        (4, 16, 33),
        (4, 17, -52),
        (4, 19, 54),
        (4, 21, -22),
        (4, 22, -68),
        (4, 24, 18),
        (4, 25, -61),
        (5, 6, 44),
        (5, 7, 22),
        (5, 10, 24),
        (5, 12, 53),
        (5, 14, -73),
        (5, 19, -45),
        (5, 20, -53),
        (5, 24, -25),
        (6, 10, 44),
        (6, 12, 92),
        (6, 13, 100),
        (6, 14, -63),
        (6, 15, -3),
        (6, 17, 87),
        (6, 19, 96),
        (6, 20, 89),
        (6, 21, 52),
        (6, 22, 89),
        (6, 23, -5),
        (6, 24, -91),
        (6, 25, -55),
        (7, 8, -97),
        (7, 9, -41),
        (7, 10, -95),
        (7, 12, 30),
        (7, 14, 22),
        (7, 15, -22),
        (7, 16, -22),
        (7, 17, 21),
        (7, 20, 79),
        (7, 22, -77),
        (7, 23, 83),
        (7, 24, -28),
        (7, 25, 36),
        (8, 11, 100),
        (8, 12, -83),
        (8, 13, 76),
        (8, 15, -95),
        (8, 16, 63),
        (8, 18, -79),
        (8, 19, 74),
        (8, 20, 63),
        (8, 21, -26),
        (8, 23, 81),
        (8, 25, -63),
        (9, 10, -44),
        (9, 12, -98),
        (9, 13, 17),
        (9, 14, 100),
        (9, 15, 85),
        (9, 16, 38),
        (9, 18, -79),
        (9, 19, -27),
        (9, 20, 53),
        (9, 23, 90),
        (9, 24, -98),
        (9, 25, 27),
        (10, 14, 94),
        (10, 15, 29),
        (10, 16, -64),
        (10, 21, 2),
        (10, 23, 42),
        (10, 25, -85),
        (11, 14, 72),
        (11, 15, 79),
        (11, 16, 99),
        (11, 19, 36),
        (11, 23, -50),
        (11, 25, -61),
        (12, 15, 33),
        (12, 17, 40),
        (12, 19, 94),
        (12, 20, 50),
        (12, 22, 31),
        (12, 25, -23),
        (13, 14, -61),
        (13, 16, 96),
        (13, 17, -58),
        (13, 18, 24),
        (13, 20, -17),
        (13, 22, -42),
        (13, 24, -24),
        (13, 25, 23),
        (14, 15, -51),
        (14, 23, -53),
        (14, 25, -17),
        (15, 16, 6),
        (15, 17, 69),
        (15, 22, 97),
        (15, 24, -79),
        (16, 18, -91),
        (16, 21, 77),
        (16, 22, 9),
        (16, 25, -63),
        (17, 20, 50),
        (17, 21, -64),
        (17, 22, 46),
        (17, 24, 59),
        (17, 25, -19),
        (18, 19, 1),
        (18, 20, -41),
        (18, 22, 63),
        (18, 23, 86),
        (18, 24, 11),
        (18, 25, -55),
        (19, 23, 30),
        (19, 24, 65),
        (20, 23, 18),
        (20, 24, 72),
        (20, 25, -32),
        (21, 23, 41),
        (21, 25, 57),
        (22, 25, 75),
    ]
}

fn case_232_edges() -> Vec<(usize, usize, i32)> {
    vec![
        (0, 6, 13),
        (0, 9, 97),
        (0, 13, 80),
        (0, 15, -73),
        (0, 16, 42),
        (0, 18, -15),
        (0, 19, 84),
        (0, 20, -3),
        (0, 21, -31),
        (0, 22, 90),
        (0, 23, -9),
        (0, 26, 82),
        (0, 27, 0),
        (1, 9, 17),
        (1, 11, -69),
        (1, 18, 79),
        (1, 19, -12),
        (1, 20, -12),
        (1, 21, 2),
        (1, 23, 92),
        (1, 24, -86),
        (1, 25, -37),
        (1, 27, 45),
        (2, 3, 32),
        (2, 6, 59),
        (2, 7, 88),
        (2, 8, 63),
        (2, 9, 82),
        (2, 11, 89),
        (2, 15, 25),
        (2, 22, -8),
        (2, 23, -75),
        (2, 24, 15),
        (3, 8, 85),
        (3, 9, 7),
        (3, 10, 80),
        (3, 15, -58),
        (3, 19, 42),
        (3, 20, 29),
        (3, 24, -2),
        (3, 27, -96),
        (4, 12, 96),
        (4, 13, -53),
        (4, 16, -79),
        (4, 21, 93),
        (4, 22, 79),
        (4, 23, -27),
        (4, 24, -95),
        (5, 6, 14),
        (5, 8, -91),
        (5, 9, -20),
        (5, 12, -59),
        (5, 13, 5),
        (5, 16, -21),
        (5, 19, -68),
        (5, 22, -19),
        (5, 23, 74),
        (6, 9, -63),
        (6, 16, -39),
        (6, 17, -77),
        (6, 19, -23),
        (6, 20, -77),
        (6, 23, -69),
        (6, 24, -1),
        (6, 27, -58),
        (7, 8, -97),
        (7, 15, -5),
        (7, 17, 65),
        (7, 18, -88),
        (7, 22, 43),
        (7, 23, -30),
        (7, 26, -31),
        (8, 9, -77),
        (8, 10, 83),
        (8, 11, 27),
        (8, 13, -84),
        (8, 14, 48),
        (8, 18, 52),
        (8, 19, 24),
        (8, 20, 28),
        (8, 21, -5),
        (8, 25, 71),
        (8, 26, 8),
        (8, 27, 23),
        (9, 18, -49),
        (9, 19, -60),
        (9, 20, -60),
        (9, 22, 52),
        (9, 23, 100),
        (9, 24, -5),
        (9, 25, -88),
        (10, 11, -37),
        (10, 12, 54),
        (10, 14, -42),
        (10, 15, -59),
        (10, 17, 88),
        (10, 18, 35),
        (10, 19, 4),
        (10, 20, 15),
        (10, 21, 39),
        (10, 22, -24),
        (10, 24, 84),
        (10, 27, -35),
        (11, 15, -46),
        (11, 16, -98),
        (11, 17, 67),
        (11, 18, -16),
        (11, 23, 95),
        (11, 24, -27),
        (11, 25, 44),
        (12, 15, 58),
        (12, 19, 64),
        (12, 20, -20),
        (12, 25, 30),
        (12, 27, 55),
        (13, 15, -50),
        (13, 17, -24),
        (13, 19, 41),
        (13, 22, 28),
        (13, 25, -83),
        (14, 18, 6),
        (14, 20, -4),
        (14, 22, -97),
        (14, 23, 14),
        (15, 18, -69),
        (15, 19, -20),
        (15, 20, 42),
        (15, 22, 20),
        (15, 25, 41),
        (15, 26, 37),
        (16, 17, 13),
        (16, 19, 80),
        (16, 20, 45),
        (16, 22, 97),
        (16, 24, 96),
        (16, 25, -25),
        (17, 20, -90),
        (17, 22, 47),
        (17, 23, 22),
        (17, 24, 65),
        (18, 19, -17),
        (18, 24, 72),
        (18, 25, -92),
        (18, 26, 95),
        (19, 22, 86),
        (19, 25, -78),
        (20, 23, -84),
        (20, 24, -20),
        (21, 25, 26),
        (21, 27, -99),
        (22, 23, -42),
        (22, 24, -30),
        (22, 25, -58),
        (22, 27, -60),
        (23, 24, 15),
        (23, 27, -88),
        (24, 25, -71),
    ]
}

fn generic_primal_pass_without_global_expand_fallback_once(
    state: &mut BlossomVState<Vcsr>,
) -> bool {
    let mut root = state.root_list_head;
    while root != NONE {
        let root_usize = root as usize;
        let next_root = state.nodes[root_usize].tree_sibling_next;
        let next_next_root = if next_root != NONE {
            state.nodes[next_root as usize].tree_sibling_next
        } else {
            NONE
        };

        if state.nodes[root_usize].is_outer
            && state.nodes[root_usize].is_tree_root
            && state.process_tree_primal(root)
        {
            return true;
        }

        root = next_root;
        if root != NONE && !state.nodes[root as usize].is_tree_root {
            root = next_next_root;
        }
    }

    false
}

fn find_global_expand_fallback_blossom_for_test(state: &BlossomVState<Vcsr>) -> Option<u32> {
    (state.node_num..state.nodes.len())
        .find(|&b| {
            let root = state.find_tree_root(b as u32);
            let match_arc = state.nodes[b].match_arc;
            let slack = if match_arc != NONE && (arc_edge(match_arc) as usize) < state.edge_num {
                state.edges[arc_edge(match_arc) as usize].slack
            } else {
                state.nodes[b].y
            };
            state.nodes[b].is_blossom
                && state.nodes[b].is_outer
                && state.nodes[b].flag == MINUS
                && slack - state.tree_eps(root) == 0
        })
        .map(|b| b as u32)
}

fn normalize_pairs(pairs: &[(usize, usize)]) -> Vec<(usize, usize)> {
    let mut normalized =
        pairs.iter().map(|&(u, v)| if u < v { (u, v) } else { (v, u) }).collect::<Vec<_>>();
    normalized.sort_unstable();
    normalized
}

fn find_edge_idx(state: &BlossomVState<Vcsr>, a: usize, b: usize) -> u32 {
    let endpoints = if a < b { (a, b) } else { (b, a) };
    (0..state.test_edge_count())
        .find_map(|e| {
            let edge = state.test_edge_endpoints(e);
            let edge = if edge.0 < edge.1 {
                (edge.0 as usize, edge.1 as usize)
            } else {
                (edge.1 as usize, edge.0 as usize)
            };
            (edge == endpoints).then_some(e as u32)
        })
        .unwrap_or_else(|| panic!("missing edge ({}, {})", endpoints.0, endpoints.1))
}

fn validate_matching(n: usize, matching: &[(usize, usize)]) {
    let mut used = vec![false; n];
    for &(u, v) in matching {
        assert!(u < n, "matching endpoint {u} out of range for n={n}");
        assert!(v < n, "matching endpoint {v} out of range for n={n}");
        assert!(!used[u], "vertex {u} used twice");
        assert!(!used[v], "vertex {v} used twice");
        used[u] = true;
        used[v] = true;
    }
    assert_eq!(matching.len(), n / 2, "matching is not perfect");
}

fn matching_cost(edges: &[(usize, usize, i32)], matching: &[(usize, usize)]) -> i32 {
    matching
        .iter()
        .map(|&(u, v)| {
            let (lo, hi) = if u < v { (u, v) } else { (v, u) };
            edges
                .iter()
                .find_map(|&(a, b, w)| {
                    let (ea, eb) = if a < b { (a, b) } else { (b, a) };
                    (ea == lo && eb == hi).then_some(w)
                })
                .unwrap_or_else(|| panic!("edge ({lo}, {hi}) not found in graph"))
        })
        .sum()
}

fn assert_edge_list_invariants(state: &BlossomVState<Vcsr>, phase: &str) {
    let mut seen = vec![[0u8; 2]; state.edge_num];
    let mut in_selfloops = vec![false; state.edge_num];

    for v in 0..state.nodes.len() {
        let mut e = state.nodes[v].blossom_selfloops;
        let mut steps = 0usize;
        while e != NONE {
            assert!(
                (e as usize) < state.edge_num,
                "{phase}: node {v} has out-of-range blossom selfloop edge {e}",
            );
            assert!(
                !in_selfloops[e as usize],
                "{phase}: edge {e} appears more than once in blossom selfloop chains",
            );
            in_selfloops[e as usize] = true;
            e = state.edges[e as usize].next[0];
            steps += 1;
            assert!(
                steps <= state.edge_num,
                "{phase}: node {v} blossom selfloop walk exceeded edge count",
            );
        }
    }

    for v in 0..state.nodes.len() {
        for dir in 0..2usize {
            let first = state.nodes[v].first[dir];
            if first == NONE {
                continue;
            }

            assert!(
                (first as usize) < state.edge_num,
                "{phase}: node {v} dir {dir} has out-of-range first edge {first}",
            );

            let mut e = first;
            let mut steps = 0usize;
            loop {
                assert_ne!(
                    e, NONE,
                    "{phase}: node {v} dir {dir} reached NONE inside adjacency cycle"
                );
                assert!(
                    (e as usize) < state.edge_num,
                    "{phase}: node {v} dir {dir} reached out-of-range edge {e}",
                );

                let edge = &state.edges[e as usize];
                assert_eq!(
                    edge.head[1 - dir],
                    v as u32,
                    "{phase}: node {v} dir {dir} contains edge {e} whose stored endpoint is {:?}",
                    edge.head,
                );

                let next = edge.next[dir];
                let prev = edge.prev[dir];
                assert_ne!(next, NONE, "{phase}: edge {e} has NONE next in dir {dir}");
                assert_ne!(prev, NONE, "{phase}: edge {e} has NONE prev in dir {dir}");
                assert!(
                    (next as usize) < state.edge_num,
                    "{phase}: edge {e} has out-of-range next {next} in dir {dir}",
                );
                assert!(
                    (prev as usize) < state.edge_num,
                    "{phase}: edge {e} has out-of-range prev {prev} in dir {dir}",
                );
                assert_eq!(
                    state.edges[next as usize].prev[dir], e,
                    "{phase}: edge {e} dir {dir} next {next} does not point back",
                );
                assert_eq!(
                    state.edges[prev as usize].next[dir], e,
                    "{phase}: edge {e} dir {dir} prev {prev} does not point forward",
                );

                seen[e as usize][dir] = seen[e as usize][dir].saturating_add(1);
                assert_eq!(
                    seen[e as usize][dir], 1,
                    "{phase}: edge {e} appears more than once in dir {dir} adjacency lists",
                );

                e = next;
                steps += 1;
                assert!(
                    steps <= state.edge_num,
                    "{phase}: node {v} dir {dir} adjacency walk exceeded edge count",
                );
                if e == first {
                    break;
                }
            }
        }
    }

    for (e_idx, edge) in state.edges.iter().take(state.edge_num).enumerate() {
        for dir in 0..2usize {
            let expected = u8::from(!(in_selfloops[e_idx] || edge.head[1 - dir] == NONE));
            assert_eq!(
                seen[e_idx][dir], expected,
                "{phase}: edge {e_idx} dir {dir} seen {} times but head is {:?}",
                seen[e_idx][dir], edge.head,
            );
        }
    }
}

fn solve_case_1594_with_edge_list_checks() -> Vec<(usize, usize)> {
    let edges = case_1594_edges();
    let g = build_graph(26, &edges);
    let mut state = BlossomVState::new(&g);
    assert_edge_list_invariants(&state, "case #1594 after new");

    state.init_global();
    assert_edge_list_invariants(&state, "case #1594 after init_global");

    for outer in 0..5_000usize {
        state.mark_tree_roots_processed();
        assert_edge_list_invariants(
            &state,
            &format!("case #1594 after mark_tree_roots_processed outer {outer}"),
        );
        let mut inner_steps = 0usize;
        loop {
            let progressed = state.generic_primal_pass_once();
            assert_edge_list_invariants(
                &state,
                &format!("case #1594 after generic pass outer {outer} inner {inner_steps}"),
            );
            inner_steps += 1;
            if !progressed || state.tree_num == 0 {
                break;
            }
            assert!(
                inner_steps <= 50_000,
                "case #1594 exceeded inner-step budget while checking edge lists",
            );
        }

        if state.tree_num == 0 {
            break;
        }

        let dual_ok = state.update_duals();
        assert_edge_list_invariants(&state, &format!("case #1594 after dual update outer {outer}"));

        assert!(dual_ok, "case #1594 dual update failed during edge-list check");

        assert!(outer < 4_999, "case #1594 exceeded outer-step budget while checking edge lists");
    }

    normalize_pairs(&state.into_pairs())
}

fn solve_case_232_with_edge_list_checks() -> Vec<(usize, usize)> {
    let edges = case_232_edges();
    let g = build_graph(28, &edges);
    let mut state = BlossomVState::new(&g);
    assert_edge_list_invariants(&state, "case #232 after new");

    state.init_global();
    assert_edge_list_invariants(&state, "case #232 after init_global");

    for outer in 0..5_000usize {
        state.mark_tree_roots_processed();
        assert_edge_list_invariants(
            &state,
            &format!("case #232 after mark_tree_roots_processed outer {outer}"),
        );
        let mut inner_steps = 0usize;
        loop {
            let progressed = state.generic_primal_pass_once();
            assert_edge_list_invariants(
                &state,
                &format!("case #232 after generic pass outer {outer} inner {inner_steps}"),
            );
            inner_steps += 1;
            if !progressed || state.tree_num == 0 {
                break;
            }
            assert!(
                inner_steps <= 50_000,
                "case #232 exceeded inner-step budget while checking edge lists",
            );
        }

        if state.tree_num == 0 {
            break;
        }

        let dual_ok = state.update_duals();
        assert_edge_list_invariants(&state, &format!("case #232 after dual update outer {outer}"));

        assert!(dual_ok, "case #232 dual update failed during edge-list check");

        assert!(outer < 4_999, "case #232 exceeded outer-step budget while checking edge lists");
    }

    normalize_pairs(&state.into_pairs())
}

fn solve_case_474_with_edge_list_checks() -> Vec<(usize, usize)> {
    let edges = case_474_edges();
    let g = build_graph(6, &edges);
    let mut state = BlossomVState::new(&g);
    assert_edge_list_invariants(&state, "case #474 after new");

    state.init_global();
    state.mark_tree_roots_processed();
    assert_edge_list_invariants(&state, "case #474 after init_global");

    for outer in 0..100usize {
        let mut inner_steps = 0usize;
        loop {
            let progressed = state.generic_primal_pass_once();
            assert_edge_list_invariants(
                &state,
                &format!("case #474 after generic pass outer {outer} inner {inner_steps}"),
            );
            inner_steps += 1;
            if !progressed || state.tree_num == 0 {
                break;
            }
            assert!(
                inner_steps <= 10_000,
                "case #474 exceeded inner-step budget while checking edge lists",
            );
        }

        if state.tree_num == 0 {
            break;
        }

        let dual_ok = state.update_duals();
        assert_edge_list_invariants(&state, &format!("case #474 after dual update outer {outer}"));

        assert!(dual_ok, "case #474 dual update failed during edge-list check");

        assert!(outer < 99, "case #474 exceeded outer-step budget while checking edge lists");
    }

    normalize_pairs(&state.into_pairs())
}

#[test]
fn test_construction_edge_count() {
    let g = build_graph(4, &[(0, 1, 5), (1, 2, 3), (2, 3, 7)]);
    let state = BlossomVState::new(&g);
    assert_eq!(state.test_edge_count(), 3);
}

#[test]
fn test_construction_edge_endpoints() {
    let g = build_graph(4, &[(0, 1, 5), (2, 3, 7)]);
    let state = BlossomVState::new(&g);
    assert_eq!(state.test_edge_endpoints(0), (0, 1));
    assert_eq!(state.test_edge_endpoints(1), (2, 3));
}

#[test]
fn test_construction_adjacency() {
    let g = build_graph(4, &[(0, 1, 5), (0, 2, 3), (0, 3, 7)]);
    let state = BlossomVState::new(&g);
    // Node 0 has 3 incident edges
    assert_eq!(state.test_degree(0), 3);
    // Nodes 1, 2, 3 each have 1 incident edge
    assert_eq!(state.test_degree(1), 1);
    assert_eq!(state.test_degree(2), 1);
    assert_eq!(state.test_degree(3), 1);
}

#[test]
fn test_greedy_feasible_duals() {
    let g = build_graph(
        6,
        &[
            (0, 1, 3),
            (0, 3, 10),
            (0, 4, 7),
            (1, 2, -1),
            (1, 4, 5),
            (1, 5, 4),
            (2, 5, -7),
            (3, 4, 0),
            (4, 5, 4),
        ],
    );
    let state = BlossomVState::new(&g);
    // All slacks must be ≥ 0
    for e in 0..state.test_edge_count() {
        assert!(
            state.test_edge_slack(e) >= 0,
            "Edge {} has negative slack {} (endpoints {:?})",
            e,
            state.test_edge_slack(e),
            state.test_edge_endpoints(e),
        );
    }
}

#[test]
fn test_greedy_valid_matching() {
    let g = build_graph(
        6,
        &[
            (0, 1, 3),
            (0, 3, 10),
            (0, 4, 7),
            (1, 2, -1),
            (1, 4, 5),
            (1, 5, 4),
            (2, 5, -7),
            (3, 4, 0),
            (4, 5, 4),
        ],
    );
    let state = BlossomVState::new(&g);
    // No vertex matched twice
    let mut matched = vec![false; 6];
    for v in 0..6 {
        if state.test_is_matched(v) {
            let partner = state.test_match_partner(v).unwrap() as usize;
            assert!(!matched[v], "Vertex {v} matched twice");
            assert_eq!(
                state.test_match_partner(partner),
                Some(v as u32),
                "Matching not symmetric for {v} <-> {partner}"
            );
            matched[v] = true;
        }
    }
}

#[test]
fn test_greedy_matched_edges_tight() {
    // Matched edges should have slack = 0
    let g = build_graph(4, &[(0, 1, 5), (2, 3, 7)]);
    let state = BlossomVState::new(&g);
    for v in 0..4 {
        let arc = state.nodes[v].match_arc;
        if arc != NONE {
            let e = arc_edge(arc) as usize;
            assert_eq!(
                state.test_edge_slack(e),
                0,
                "Matched edge {e} should have slack 0, got {}",
                state.test_edge_slack(e),
            );
        }
    }
}

#[test]
fn test_greedy_creates_trees() {
    // 4 vertices, 1 edge → 2 matched, 2 unmatched → 2 trees
    let g = build_graph(4, &[(0, 1, 5)]);
    let state = BlossomVState::new(&g);
    assert_eq!(state.test_tree_num(), 2);
    assert!(state.test_is_matched(0));
    assert!(state.test_is_matched(1));
    assert!(!state.test_is_matched(2));
    assert!(!state.test_is_matched(3));
    assert!(state.test_is_tree_root(2));
    assert!(state.test_is_tree_root(3));
    assert_eq!(state.test_flag(2), PLUS);
    assert_eq!(state.test_flag(3), PLUS);
}

#[test]
fn test_greedy_single_edge_matches() {
    let g = build_graph(2, &[(0, 1, 42)]);
    let state = BlossomVState::new(&g);
    assert_eq!(state.test_tree_num(), 0);
    assert!(state.test_is_matched(0));
    assert!(state.test_is_matched(1));
    assert_eq!(state.test_match_partner(0), Some(1));
    assert_eq!(state.test_match_partner(1), Some(0));
}

#[test]
fn test_greedy_feasible_negative_weights() {
    let g = build_graph(4, &[(0, 1, -10), (2, 3, -20), (0, 2, 5)]);
    let state = BlossomVState::new(&g);
    for e in 0..state.test_edge_count() {
        assert!(
            state.test_edge_slack(e) >= 0,
            "Edge {} has negative slack {}",
            e,
            state.test_edge_slack(e),
        );
    }
}

#[test]
fn test_greedy_prefers_low_cost() {
    // Triangle: (0,1,w=1), (1,2,w=100), (0,2,w=1)
    // Plus (2,3,w=1) to make n even.
    // Greedy should try to match low-cost edges.
    let g = build_graph(4, &[(0, 1, 1), (1, 2, 100), (0, 2, 1), (2, 3, 1)]);
    let state = BlossomVState::new(&g);
    // All 4 vertices should be matched (greedy may or may not be optimal)
    assert_eq!(state.test_tree_num(), 0);
}

#[test]
fn test_ground_truth_first_n6_case_with_budget() {
    // First n=6 case from the gzipped ground-truth corpus.
    let g = build_graph(
        6,
        &[
            (0, 3, -35),
            (2, 5, -39),
            (1, 4, 80),
            (3, 5, 71),
            (3, 4, 65),
            (2, 4, -87),
            (0, 5, -9),
            (1, 2, 73),
            (1, 3, 63),
        ],
    );

    let matching = BlossomVState::new(&g)
        .solve_with_test_budget(200, 5_000)
        .expect("should solve first n=6 ground-truth case within budget");

    assert_eq!(normalize_pairs(&matching), vec![(0, 5), (1, 3), (2, 4)],);
}

#[test]
fn test_ground_truth_case_49_with_budget() {
    // First known bad n=6 case from the gzipped ground-truth corpus.
    let edges = [
        (0, 5, 89),
        (3, 4, 82),
        (1, 2, -12),
        (1, 3, 80),
        (0, 2, -78),
        (0, 4, -14),
        (3, 5, 53),
        (0, 1, 50),
        (2, 5, -22),
    ];
    let g = build_graph(6, &edges);

    let matching = BlossomVState::new(&g)
        .solve_with_test_budget(200, 5_000)
        .expect("should solve n=6 corpus case #49 within budget");
    let matching = normalize_pairs(&matching);

    validate_matching(6, &matching);
    assert_eq!(matching_cost(&edges, &matching), 27);
}

#[test]
fn test_case_49_manual_grow_grow_augment() {
    let edges = [
        (0, 5, 89),
        (3, 4, 82),
        (1, 2, -12),
        (1, 3, 80),
        (0, 2, -78),
        (0, 4, -14),
        (3, 5, 53),
        (0, 1, 50),
        (2, 5, -22),
    ];
    let g = build_graph(6, &edges);
    let mut state = BlossomVState::new(&g);

    let e04 = find_edge_idx(&state, 0, 4);
    let e12 = find_edge_idx(&state, 1, 2);
    let e35 = find_edge_idx(&state, 3, 5);

    state.grow(e04, 4, 0);
    state.grow(e12, 2, 1);
    state.augment(e35, 3, 5);

    let pairs = normalize_pairs(&state.into_pairs());
    assert_eq!(pairs, vec![(0, 4), (1, 2), (3, 5)]);
}

#[test]
fn test_ground_truth_case_98_with_budget() {
    let edges = [
        (2, 3, -49),
        (0, 1, 40),
        (4, 5, -56),
        (1, 5, -32),
        (3, 4, -17),
        (2, 5, -86),
        (1, 4, -59),
        (3, 5, -30),
    ];
    let g = build_graph(6, &edges);

    let matching = BlossomVState::new(&g)
        .solve_with_test_budget(200, 5_000)
        .expect("should solve n=6 corpus case #98 within budget");
    let matching = normalize_pairs(&matching);

    validate_matching(6, &matching);
    assert_eq!(matching_cost(&edges, &matching), -65);
    assert_eq!(matching, vec![(0, 1), (2, 3), (4, 5)]);
}

#[test]
fn test_ground_truth_case_26951_with_budget() {
    let edges = [
        (1, 3, 54),
        (4, 6, -95),
        (0, 5, 81),
        (2, 7, -2),
        (6, 7, -23),
        (2, 4, 73),
        (2, 5, -97),
        (1, 4, -86),
        (0, 4, 88),
        (1, 5, 73),
        (1, 7, 10),
        (3, 6, -84),
        (5, 6, 41),
        (3, 7, -34),
        (0, 2, -22),
    ];
    let g = build_graph(8, &edges);

    let matching = BlossomVState::new(&g)
        .solve_with_test_budget(200, 5_000)
        .expect("should solve n=8 corpus case #26951 within budget");
    let matching = normalize_pairs(&matching);

    validate_matching(8, &matching);
    assert_eq!(matching_cost(&edges, &matching), -101);
    assert_eq!(matching, vec![(0, 2), (1, 4), (3, 7), (5, 6)]);
}

#[test]
fn test_ground_truth_case_27373_with_budget() {
    let edges = [
        (0, 1, 5),
        (6, 8, 40),
        (3, 11, -94),
        (4, 5, -88),
        (2, 10, -48),
        (7, 9, 69),
        (1, 3, 98),
        (2, 6, 89),
        (1, 7, 30),
        (9, 10, 8),
        (3, 7, -3),
        (8, 10, -82),
        (6, 9, -26),
        (4, 10, -95),
        (0, 10, -47),
        (2, 3, -46),
        (3, 5, -38),
        (2, 5, -1),
        (1, 5, 82),
        (5, 9, -94),
    ];
    let g = build_graph(12, &edges);

    let matching = BlossomVState::new(&g)
        .solve_with_test_budget(200, 5_000)
        .expect("should solve n=12 corpus case #27373 within budget");
    let matching = normalize_pairs(&matching);

    validate_matching(12, &matching);
    assert_eq!(matching_cost(&edges, &matching), -116);
    assert_eq!(matching, vec![(0, 1), (2, 10), (3, 11), (4, 5), (6, 8), (7, 9)]);
}

#[test]
fn test_ground_truth_case_27004_with_budget() {
    let edges = [
        (5, 7, 10),
        (0, 8, -18),
        (3, 4, -70),
        (6, 9, -100),
        (1, 2, 51),
        (0, 3, 94),
        (0, 5, -42),
        (2, 8, 66),
        (0, 6, 47),
        (5, 9, 6),
        (2, 7, 38),
        (3, 7, -95),
        (0, 9, -87),
        (0, 1, -56),
        (1, 8, -69),
        (7, 9, 72),
        (1, 7, -86),
        (0, 2, -64),
        (2, 4, 62),
        (6, 8, 55),
        (5, 6, -30),
        (1, 5, -3),
        (1, 4, 23),
        (7, 8, -21),
        (3, 8, -69),
    ];
    let g = build_graph(10, &edges);

    let matching = BlossomVState::new(&g)
        .solve_with_test_budget(200, 5_000)
        .expect("should solve n=10 corpus case #27004 within budget");
    let matching = normalize_pairs(&matching);

    validate_matching(10, &matching);
    assert_eq!(matching_cost(&edges, &matching), -293);
    assert_eq!(matching, vec![(0, 2), (1, 8), (3, 4), (5, 7), (6, 9)]);
}

#[test]
fn test_ground_truth_case_91838_with_budget() {
    let edges = [
        (0, 1, 87),
        (0, 2, -14),
        (0, 15, -84),
        (0, 16, -84),
        (0, 17, 11),
        (1, 2, 32),
        (1, 17, 48),
        (2, 3, -65),
        (2, 4, -50),
        (3, 4, 99),
        (3, 5, 41),
        (4, 5, -84),
        (4, 6, 6),
        (5, 6, -53),
        (5, 7, 23),
        (6, 7, 26),
        (6, 8, 2),
        (7, 8, -19),
        (7, 9, -49),
        (8, 9, -65),
        (8, 10, 33),
        (9, 11, -86),
        (9, 12, -31),
        (10, 11, -44),
        (10, 12, 20),
        (11, 12, -48),
        (11, 13, 91),
        (12, 13, 67),
        (12, 14, 56),
        (13, 14, -91),
        (13, 15, 93),
        (14, 15, -11),
        (14, 16, -17),
        (15, 16, -94),
        (15, 17, -15),
        (16, 17, 0),
    ];
    let g = build_graph(18, &edges);

    let matching = BlossomVState::new(&g)
        .solve_with_test_budget(200, 5_000)
        .expect("should solve n=18 corpus case #91838 within budget");
    let matching = normalize_pairs(&matching);

    validate_matching(18, &matching);
    assert_eq!(matching_cost(&edges, &matching), -226);
    assert_eq!(
        matching,
        vec![(0, 15), (1, 17), (2, 3), (4, 5), (6, 8), (7, 9), (10, 11), (12, 13), (14, 16),],
    );
}

#[test]
fn test_ground_truth_case_91838_public_path() {
    let edges = [
        (0, 1, 87),
        (0, 2, -14),
        (0, 15, -84),
        (0, 16, -84),
        (0, 17, 11),
        (1, 2, 32),
        (1, 17, 48),
        (2, 3, -65),
        (2, 4, -50),
        (3, 4, 99),
        (3, 5, 41),
        (4, 5, -84),
        (4, 6, 6),
        (5, 6, -53),
        (5, 7, 23),
        (6, 7, 26),
        (6, 8, 2),
        (7, 8, -19),
        (7, 9, -49),
        (8, 9, -65),
        (8, 10, 33),
        (9, 11, -86),
        (9, 12, -31),
        (10, 11, -44),
        (10, 12, 20),
        (11, 12, -48),
        (11, 13, 91),
        (12, 13, 67),
        (12, 14, 56),
        (13, 14, -91),
        (13, 15, 93),
        (14, 15, -11),
        (14, 16, -17),
        (15, 16, -94),
        (15, 17, -15),
        (16, 17, 0),
    ];
    let g = build_graph(18, &edges);

    let matching =
        g.blossom_v().expect("public BlossomV path should solve n=18 corpus case #91838");
    let matching = normalize_pairs(&matching);

    validate_matching(18, &matching);
    assert_eq!(matching_cost(&edges, &matching), -226);
    assert_eq!(
        matching,
        vec![(0, 15), (1, 17), (2, 3), (4, 5), (6, 8), (7, 9), (10, 11), (12, 13), (14, 16),],
    );
}

#[test]
fn test_ground_truth_case_97_with_budget() {
    let edges = case_97_edges();
    let g = build_graph(18, &edges);

    let matching = BlossomVState::new(&g)
        .solve_with_test_budget(300, 10_000)
        .expect("should solve generated corpus case #97 within budget");
    let matching = normalize_pairs(&matching);

    validate_matching(18, &matching);
    assert_eq!(matching_cost(&edges, &matching), -329);
    assert_eq!(
        matching,
        vec![(0, 12), (1, 9), (2, 17), (3, 15), (4, 6), (5, 13), (7, 14), (8, 10), (11, 16),],
    );
}

#[test]
fn test_ground_truth_case_97_public_path() {
    let edges = case_97_edges();
    let g = build_graph(18, &edges);

    let matching =
        g.blossom_v().expect("public BlossomV path should solve generated corpus case #97");
    let matching = normalize_pairs(&matching);

    validate_matching(18, &matching);
    assert_eq!(matching_cost(&edges, &matching), -329);
    assert_eq!(
        matching,
        vec![(0, 12), (1, 9), (2, 17), (3, 15), (4, 6), (5, 13), (7, 14), (8, 10), (11, 16),],
    );
}

#[test]
fn test_ground_truth_case_honggfuzz_sigabrt_4_with_budget() {
    let edges = case_honggfuzz_sigabrt_4_edges();
    let g = build_graph(26, &edges);

    let matching = BlossomVState::new(&g)
        .solve_with_test_budget(600, 20_000)
        .expect("should solve honggfuzz replay case 4 within budget");
    let matching = normalize_pairs(&matching);

    validate_matching(26, &matching);
    assert_eq!(matching_cost(&edges, &matching), -186717);
    assert_eq!(
        matching,
        vec![
            (0, 16),
            (1, 17),
            (2, 4),
            (3, 7),
            (5, 11),
            (6, 8),
            (9, 14),
            (10, 15),
            (12, 20),
            (13, 21),
            (18, 23),
            (19, 22),
            (24, 25),
        ],
    );
}

#[test]
fn test_ground_truth_case_honggfuzz_sigabrt_5_with_budget() {
    let edges = case_honggfuzz_sigabrt_5_edges();
    let g = build_graph(26, &edges);

    let matching = BlossomVState::new(&g)
        .solve_with_test_budget(800, 20_000)
        .expect("should solve honggfuzz replay case 5 within budget");
    let matching = normalize_pairs(&matching);

    validate_matching(26, &matching);
    assert_eq!(matching_cost(&edges, &matching), -116000);
    assert_eq!(
        matching,
        vec![
            (0, 19),
            (1, 6),
            (2, 7),
            (3, 22),
            (4, 5),
            (8, 15),
            (9, 12),
            (10, 11),
            (13, 20),
            (14, 21),
            (16, 25),
            (17, 24),
            (18, 23),
        ],
    );
}

#[test]
fn test_ground_truth_case_honggfuzz_sigabrt_6_with_budget() {
    let edges = case_honggfuzz_sigabrt_6_edges();
    let g = build_graph(26, &edges);

    let matching = BlossomVState::new(&g)
        .solve_with_test_budget(800, 20_000)
        .expect("should solve honggfuzz replay case 6 within budget");
    let matching = normalize_pairs(&matching);

    validate_matching(26, &matching);
    assert_eq!(matching_cost(&edges, &matching), -114562);
    assert_eq!(
        matching,
        vec![
            (0, 11),
            (1, 18),
            (2, 7),
            (3, 22),
            (4, 5),
            (6, 14),
            (8, 15),
            (9, 19),
            (10, 17),
            (12, 23),
            (13, 20),
            (16, 25),
            (21, 24),
        ],
    );
}

#[test]
fn test_ground_truth_case_honggfuzz_sigabrt_7_with_budget() {
    let edges = case_honggfuzz_sigabrt_7_edges();
    let g = build_graph(26, &edges);

    let matching = BlossomVState::new(&g)
        .solve_with_test_budget(1200, 30_000)
        .expect("should solve honggfuzz replay case 7 within budget");
    let matching = normalize_pairs(&matching);

    validate_matching(26, &matching);
    assert_eq!(matching_cost(&edges, &matching), -113140);
    assert_eq!(
        matching,
        vec![
            (0, 24),
            (1, 23),
            (2, 11),
            (3, 7),
            (4, 5),
            (6, 12),
            (8, 15),
            (9, 22),
            (10, 17),
            (13, 19),
            (14, 21),
            (16, 25),
            (18, 20),
        ],
    );
}

#[test]
fn test_ground_truth_case_416_with_budget() {
    let edges = case_416_edges();
    let g = build_graph(20, &edges);

    let matching = BlossomVState::new(&g)
        .solve_with_test_budget(400, 10_000)
        .expect("should solve generated corpus case #416 within budget");
    let matching = normalize_pairs(&matching);

    validate_matching(20, &matching);
    assert_eq!(matching_cost(&edges, &matching), -707);
    assert_eq!(
        matching,
        vec![
            (0, 8),
            (1, 6),
            (2, 16),
            (3, 5),
            (4, 17),
            (7, 13),
            (9, 11),
            (10, 18),
            (12, 15),
            (14, 19),
        ],
    );
}

#[test]
fn test_ground_truth_case_87417_with_budget() {
    let edges = case_87417_edges();
    let g = build_graph(30, &edges);

    let matching = BlossomVState::new(&g)
        .solve_with_test_budget(800, 20_000)
        .expect("should solve corpus case #87417 within budget");
    let matching = normalize_pairs(&matching);

    validate_matching(30, &matching);
    assert_eq!(matching_cost(&edges, &matching), -771);
    assert_eq!(
        matching,
        vec![
            (0, 24),
            (1, 15),
            (2, 10),
            (3, 21),
            (4, 29),
            (5, 26),
            (6, 9),
            (7, 12),
            (8, 14),
            (11, 20),
            (13, 22),
            (16, 25),
            (17, 19),
            (18, 23),
            (27, 28),
        ],
    );
}

#[test]
fn test_case_97_first_generic_grow_edge_is_visible_from_root_0() {
    let edges = case_97_edges();
    let g = build_graph(18, &edges);
    let mut state = BlossomVState::new(&g);
    state.init_global();
    state.mark_tree_roots_processed();

    let mut saw = false;
    for (e_idx, dir) in state.incident_edges(0) {
        let other = state.edge_head_outer(e_idx, dir);
        if normalized_edge_pair(state.edges[e_idx as usize].head) == (0, 12) {
            saw = true;
            assert_eq!(state.edges[e_idx as usize].slack, 0, "edge (0,12) should already be tight");
            assert_eq!(other, 12, "edge (0,12) should point from root 0 to free node 12");
            assert_eq!(state.nodes[other as usize].flag, FREE, "node 12 should still be free");
        }
    }

    assert!(saw, "root 0 should still see the tight edge (0,12) after init_global");
}

#[test]
fn test_ground_truth_case_1594_with_budget() {
    let edges = case_1594_edges();
    let g = build_graph(26, &edges);

    let matching = BlossomVState::new(&g)
        .solve_with_test_budget(400, 30_000)
        .expect("should solve n=26 corpus case #1594 within budget");
    let matching = normalize_pairs(&matching);

    validate_matching(26, &matching);
    assert_eq!(matching_cost(&edges, &matching), -1003);
    assert_eq!(
        matching,
        vec![
            (0, 23),
            (1, 20),
            (2, 19),
            (3, 22),
            (4, 13),
            (5, 14),
            (6, 24),
            (7, 10),
            (8, 15),
            (9, 12),
            (11, 25),
            (16, 18),
            (17, 21),
        ],
    );
}

#[test]
fn test_case_1594_edge_list_invariants_hold_during_solve() {
    let edges = case_1594_edges();
    let matching = solve_case_1594_with_edge_list_checks();

    validate_matching(26, &matching);
    assert_eq!(matching_cost(&edges, &matching), -1003);
}

#[test]
fn test_ground_truth_case_232_with_budget() {
    let edges = case_232_edges();
    let g = build_graph(28, &edges);

    let matching = BlossomVState::new(&g)
        .solve_with_test_budget(400, 30_000)
        .expect("should solve n=28 corpus case #232 within budget");
    let matching = normalize_pairs(&matching);

    validate_matching(28, &matching);
    assert_eq!(matching_cost(&edges, &matching), -982);
    assert_eq!(
        matching,
        vec![
            (0, 21),
            (1, 19),
            (2, 23),
            (3, 27),
            (4, 24),
            (5, 12),
            (6, 9),
            (7, 26),
            (8, 13),
            (10, 15),
            (11, 16),
            (14, 22),
            (17, 20),
            (18, 25),
        ],
    );
}

#[test]
fn test_case_232_edge_list_invariants_hold_during_solve() {
    let edges = case_232_edges();
    let matching = solve_case_232_with_edge_list_checks();

    validate_matching(28, &matching);
    assert_eq!(matching_cost(&edges, &matching), -982);
}

#[test]
fn test_ground_truth_case_474_with_budget() {
    let edges = case_474_edges();
    let g = build_graph(6, &edges);

    let matching = BlossomVState::new(&g)
        .solve_with_test_budget(100, 10_000)
        .expect("should solve n=6 corpus case #474 within budget");
    let matching = normalize_pairs(&matching);

    validate_matching(6, &matching);
    assert_eq!(matching_cost(&edges, &matching), -8);
    assert_eq!(matching, vec![(0, 5), (1, 2), (3, 4)]);
}

#[test]
fn test_case_474_edge_list_invariants_hold_during_solve() {
    let edges = case_474_edges();
    let matching = solve_case_474_with_edge_list_checks();

    validate_matching(6, &matching);
    assert_eq!(matching_cost(&edges, &matching), -8);
}

#[test]
fn test_case_474_default_solve_matches_budgeted_solve() {
    let edges = case_474_edges();
    let g = build_graph(6, &edges);

    let default_matching = BlossomVState::new(&g).solve().expect("default solve should succeed");
    let default_matching = normalize_pairs(&default_matching);
    let budgeted_matching = BlossomVState::new(&g)
        .solve_with_test_budget(100, 10_000)
        .expect("budgeted solve should succeed");
    let budgeted_matching = normalize_pairs(&budgeted_matching);

    assert_eq!(default_matching, budgeted_matching);
    assert_eq!(matching_cost(&edges, &default_matching), -8);
}

#[test]
fn test_ground_truth_case_145677_with_budget() {
    let edges = [
        (0, 1, -35),
        (0, 2, -28),
        (0, 24, -5),
        (0, 25, 92),
        (1, 2, -49),
        (1, 3, -18),
        (1, 25, -51),
        (2, 3, -46),
        (2, 4, 15),
        (3, 4, 54),
        (3, 5, 38),
        (4, 5, -39),
        (4, 6, -79),
        (5, 6, -12),
        (5, 7, -64),
        (6, 7, -27),
        (6, 8, 54),
        (7, 8, -70),
        (7, 9, -24),
        (8, 9, 82),
        (8, 12, 80),
        (9, 10, 48),
        (9, 11, -64),
        (10, 11, 72),
        (10, 12, 89),
        (11, 12, 77),
        (11, 13, -28),
        (12, 13, -59),
        (12, 14, 50),
        (13, 14, 28),
        (13, 15, -31),
        (13, 18, 26),
        (14, 15, 26),
        (14, 16, 65),
        (15, 16, -34),
        (15, 17, 36),
        (16, 17, 63),
        (16, 18, 96),
        (17, 18, 16),
        (17, 19, 87),
        (18, 20, -73),
        (19, 20, 18),
        (19, 21, -10),
        (20, 21, -89),
        (20, 22, -21),
        (21, 22, -22),
        (21, 23, -40),
        (22, 23, 96),
        (22, 24, -96),
        (23, 24, -47),
        (23, 25, -66),
        (24, 25, -76),
    ];
    let g = build_graph(26, &edges);

    let matching = BlossomVState::new(&g)
        .solve_with_test_budget(300, 20_000)
        .expect("should solve n=26 corpus case #145677 within budget");
    let matching = normalize_pairs(&matching);

    validate_matching(26, &matching);
    assert_eq!(matching_cost(&edges, &matching), -293);
    assert_eq!(
        matching,
        vec![
            (0, 2),
            (1, 25),
            (3, 5),
            (4, 6),
            (7, 8),
            (9, 11),
            (10, 12),
            (13, 14),
            (15, 16),
            (17, 19),
            (18, 20),
            (21, 23),
            (22, 24),
        ],
    );
}

#[test]
fn test_ground_truth_case_174453_with_budget() {
    let edges = [
        (0, 4, -80),
        (0, 6, 33),
        (0, 9, -67),
        (0, 17, 68),
        (0, 18, 69),
        (0, 19, 24),
        (0, 20, 45),
        (0, 21, 69),
        (0, 23, 99),
        (1, 3, 44),
        (1, 6, 31),
        (1, 8, -14),
        (1, 10, 13),
        (1, 11, 27),
        (1, 13, -36),
        (1, 15, -47),
        (1, 16, 46),
        (1, 17, -23),
        (1, 18, -92),
        (1, 20, -92),
        (1, 21, 71),
        (1, 22, -22),
        (1, 23, -77),
        (2, 10, 83),
        (2, 12, -53),
        (2, 16, -60),
        (2, 22, -99),
        (2, 23, -74),
        (2, 25, 52),
        (3, 4, -58),
        (3, 8, 1),
        (3, 10, 5),
        (3, 14, -78),
        (3, 17, 29),
        (3, 18, -9),
        (3, 23, 79),
        (3, 24, 17),
        (3, 25, 10),
        (4, 5, -74),
        (4, 8, 19),
        (4, 9, 100),
        (4, 10, 29),
        (4, 11, -77),
        (4, 12, 29),
        (4, 13, -52),
        (4, 17, 0),
        (4, 19, -55),
        (4, 20, -99),
        (4, 23, -64),
        (4, 24, -56),
        (5, 6, 14),
        (5, 9, 25),
        (5, 11, 65),
        (5, 14, -18),
        (5, 18, -98),
        (5, 20, -96),
        (5, 21, -69),
        (5, 23, -11),
        (5, 24, -95),
        (6, 10, -85),
        (6, 11, 8),
        (6, 14, -1),
        (6, 15, 31),
        (6, 16, -22),
        (6, 18, 92),
        (6, 20, 89),
        (7, 8, 74),
        (7, 10, -58),
        (7, 14, -34),
        (7, 17, 22),
        (7, 21, 39),
        (7, 23, 25),
        (7, 24, 45),
        (8, 11, 28),
        (8, 12, 47),
        (8, 16, -100),
        (8, 17, 91),
        (8, 22, 9),
        (8, 23, -97),
        (9, 14, 37),
        (9, 16, 81),
        (9, 19, -6),
        (9, 20, 82),
        (9, 21, -65),
        (9, 23, -49),
        (9, 25, -3),
        (10, 11, -94),
        (10, 12, -48),
        (10, 17, 81),
        (10, 23, -21),
        (10, 24, -11),
        (11, 12, -15),
        (11, 18, 6),
        (11, 21, -7),
        (11, 22, -56),
        (12, 13, 77),
        (12, 17, 33),
        (12, 18, 91),
        (12, 19, -73),
        (12, 20, 78),
        (12, 21, -28),
        (13, 20, 44),
        (13, 21, -30),
        (13, 24, -63),
        (14, 15, 49),
        (14, 16, 85),
        (14, 23, -20),
        (15, 18, -11),
        (15, 19, 97),
        (16, 19, 30),
        (16, 21, -33),
        (16, 22, 92),
        (16, 23, -97),
        (17, 19, 62),
        (17, 24, 27),
        (17, 25, 23),
        (18, 19, 20),
        (18, 21, 86),
        (18, 23, -19),
        (18, 24, 83),
        (19, 21, -14),
        (19, 22, -57),
        (21, 22, 80),
        (21, 23, 11),
        (22, 23, -5),
        (22, 24, 62),
    ];
    let g = build_graph(26, &edges);

    let matching = BlossomVState::new(&g)
        .solve_with_test_budget(400, 30_000)
        .expect("should solve n=26 corpus case #174453 within budget");
    let matching = normalize_pairs(&matching);

    validate_matching(26, &matching);
    assert_eq!(matching_cost(&edges, &matching), -785);
    assert_eq!(
        matching,
        vec![
            (0, 9),
            (1, 15),
            (2, 22),
            (3, 14),
            (4, 20),
            (5, 18),
            (6, 16),
            (7, 10),
            (8, 23),
            (11, 21),
            (12, 19),
            (13, 24),
            (17, 25),
        ],
    );
}

#[test]
fn test_ground_truth_case_4666_with_budget() {
    let edges = [
        (0, 2, -58),
        (0, 3, 17),
        (0, 4, 38),
        (0, 12, 89),
        (0, 17, 78),
        (0, 22, 5),
        (0, 26, 91),
        (0, 27, 67),
        (0, 28, 4),
        (0, 29, -56),
        (1, 2, -96),
        (1, 3, 99),
        (1, 4, 24),
        (1, 5, -18),
        (1, 20, -65),
        (1, 28, 58),
        (1, 29, -80),
        (2, 3, 43),
        (2, 4, 59),
        (2, 5, 44),
        (2, 6, -96),
        (2, 11, -82),
        (2, 27, 15),
        (2, 28, -72),
        (2, 29, 5),
        (3, 4, 76),
        (3, 5, 52),
        (3, 6, 88),
        (3, 7, 59),
        (3, 29, -13),
        (4, 5, -11),
        (4, 6, 81),
        (4, 7, 96),
        (4, 8, 52),
        (5, 6, -32),
        (5, 8, -44),
        (5, 9, -89),
        (5, 29, 20),
        (6, 7, 59),
        (6, 8, -22),
        (6, 9, 47),
        (6, 10, 3),
        (7, 8, -87),
        (7, 9, -50),
        (7, 10, -50),
        (7, 11, -29),
        (7, 19, 69),
        (7, 22, 14),
        (8, 9, -25),
        (8, 10, -86),
        (8, 12, 58),
        (8, 27, -8),
        (9, 10, 61),
        (9, 11, -78),
        (9, 12, 46),
        (9, 13, -10),
        (9, 14, 7),
        (9, 16, 76),
        (10, 11, 49),
        (10, 12, 60),
        (10, 13, 15),
        (10, 14, 68),
        (10, 28, 58),
        (11, 12, 76),
        (11, 14, 78),
        (11, 15, -39),
        (11, 19, -40),
        (12, 13, 88),
        (12, 14, 29),
        (12, 15, -72),
        (12, 16, 17),
        (13, 14, -57),
        (13, 15, 37),
        (13, 16, -81),
        (13, 17, -46),
        (13, 24, -97),
        (13, 26, 75),
        (14, 16, 49),
        (14, 17, 42),
        (15, 17, -61),
        (15, 18, 87),
        (15, 19, 86),
        (15, 23, -94),
        (16, 17, 94),
        (16, 19, -71),
        (16, 24, -63),
        (16, 25, 3),
        (17, 19, -51),
        (17, 20, -2),
        (17, 27, 94),
        (18, 19, 62),
        (18, 22, 56),
        (18, 26, 32),
        (18, 27, -35),
        (19, 21, 31),
        (19, 22, -10),
        (20, 21, 9),
        (20, 23, 87),
        (20, 24, -21),
        (21, 22, 22),
        (21, 23, 15),
        (21, 24, 20),
        (21, 25, -16),
        (22, 25, -39),
        (22, 29, -70),
        (23, 24, -90),
        (23, 25, -42),
        (23, 26, 53),
        (23, 27, 68),
        (24, 25, -21),
        (24, 26, 34),
        (25, 26, -13),
        (25, 27, -24),
        (25, 28, -84),
        (25, 29, -31),
        (26, 28, 57),
        (26, 29, 78),
        (27, 28, 91),
        (27, 29, -46),
        (28, 29, -2),
    ];
    let g = build_graph(30, &edges);

    let matching = BlossomVState::new(&g)
        .solve_with_test_budget(500, 50_000)
        .expect("should solve corpus case #4666 within budget");
    let matching = normalize_pairs(&matching);

    validate_matching(30, &matching);
    assert_eq!(matching_cost(&edges, &matching), -641);
    assert_eq!(
        matching,
        vec![
            (0, 3),
            (1, 20),
            (2, 6),
            (4, 5),
            (7, 10),
            (8, 27),
            (9, 11),
            (12, 15),
            (13, 14),
            (16, 24),
            (17, 19),
            (18, 26),
            (21, 23),
            (22, 29),
            (25, 28),
        ],
    );
}

#[test]
fn test_ground_truth_case_224_with_budget() {
    let edges = [
        (1, 2, -65),
        (0, 5, -71),
        (3, 4, 50),
        (3, 5, -90),
        (2, 4, -80),
        (1, 4, -16),
        (0, 3, -31),
        (1, 5, 38),
        (2, 3, -47),
        (2, 5, 46),
    ];
    let g = build_graph(6, &edges);

    let matching = BlossomVState::new(&g)
        .solve_with_test_budget(200, 5_000)
        .expect("should solve n=6 corpus case #224 within budget");
    let matching = normalize_pairs(&matching);

    validate_matching(6, &matching);
    assert_eq!(matching_cost(&edges, &matching), -134);
    assert_eq!(matching, vec![(0, 5), (1, 4), (2, 3)]);
}

#[test]
fn test_ground_truth_case_26924_with_budget() {
    let edges = [
        (1, 7, -98),
        (0, 2, 67),
        (5, 9, 71),
        (3, 6, -45),
        (4, 8, 71),
        (0, 6, 1),
        (1, 3, -19),
        (1, 4, 31),
        (7, 8, -7),
        (1, 5, 7),
        (4, 7, 18),
        (2, 8, -74),
        (3, 4, -13),
        (2, 5, 76),
        (2, 6, 73),
        (2, 9, 30),
        (0, 7, 18),
        (1, 2, 78),
        (8, 9, 58),
        (2, 4, -75),
        (2, 7, 69),
        (4, 9, -39),
        (4, 5, -46),
        (7, 9, -3),
    ];
    let g = build_graph(10, &edges);

    let matching = BlossomVState::new(&g)
        .solve_with_test_budget(200, 5_000)
        .expect("should solve n=10 corpus case #26924 within budget");
    let matching = normalize_pairs(&matching);

    validate_matching(10, &matching);
    assert_eq!(matching_cost(&edges, &matching), -141);
    assert_eq!(matching, vec![(0, 6), (1, 3), (2, 8), (4, 5), (7, 9)]);
}

#[test]
fn test_ground_truth_case_24_with_budget() {
    let g = build_graph(
        20,
        &[
            (0, 5, 9),
            (0, 11, -19),
            (1, 4, 13),
            (1, 9, -28),
            (1, 12, -49),
            (1, 13, 84),
            (1, 17, 78),
            (2, 7, 46),
            (2, 10, 92),
            (3, 6, 89),
            (3, 14, 36),
            (4, 17, -91),
            (5, 8, -87),
            (5, 11, -70),
            (5, 15, -39),
            (5, 16, 65),
            (5, 18, -60),
            (5, 19, 60),
            (6, 11, -43),
            (6, 14, 9),
            (7, 10, 86),
            (8, 9, -53),
            (8, 11, 59),
            (8, 13, 27),
            (8, 16, -70),
            (8, 19, 3),
            (9, 10, 95),
            (9, 13, -74),
            (9, 16, -2),
            (9, 18, 0),
            (9, 19, 45),
            (10, 12, 88),
            (10, 13, 9),
            (12, 13, -23),
            (13, 17, -89),
            (15, 16, -37),
            (15, 18, 46),
            (15, 19, -10),
            (16, 18, 94),
            (16, 19, -91),
            (18, 19, 7),
        ],
    );
    let pairs = BlossomVState::new(&g)
        .solve_with_test_budget(4096, 4096)
        .expect("case #24 should have a perfect matching");
    assert_eq!(
        pairs,
        vec![
            (0, 5),
            (1, 12),
            (2, 7),
            (3, 14),
            (4, 17),
            (6, 11),
            (8, 9),
            (10, 13),
            (15, 18),
            (16, 19),
        ]
    );
    let cost: i32 = pairs
        .iter()
        .map(|&(u, v)| {
            match (u, v) {
                (0, 5) => 9,
                (1, 12) => -49,
                (2, 7) => 46,
                (3, 14) => 36,
                (4, 17) => -91,
                (6, 11) => -43,
                (8, 9) => -53,
                (10, 13) => 9,
                (15, 18) => 46,
                (16, 19) => -91,
                _ => panic!("unexpected pair ({u}, {v}) in case #24"),
            }
        })
        .sum();
    assert_eq!(cost, -181);
}

#[test]
fn test_ground_truth_case_214_with_budget() {
    let edges = [
        (0, 1, 90),
        (0, 2, -66),
        (0, 3, -13),
        (0, 4, -83),
        (0, 5, 73),
        (0, 7, -70),
        (0, 9, -67),
        (0, 11, 39),
        (0, 12, 40),
        (0, 13, -57),
        (0, 14, -32),
        (0, 18, -54),
        (0, 19, 73),
        (0, 20, -60),
        (1, 2, -96),
        (1, 3, -17),
        (1, 4, 12),
        (1, 5, -20),
        (1, 6, 42),
        (1, 8, -5),
        (1, 11, 31),
        (1, 14, 0),
        (1, 16, -90),
        (1, 21, -90),
        (2, 3, -88),
        (2, 4, -38),
        (2, 5, 80),
        (2, 7, 7),
        (2, 9, -27),
        (2, 10, -30),
        (2, 12, 66),
        (2, 17, -46),
        (2, 18, -66),
        (2, 19, -61),
        (3, 4, -11),
        (3, 5, 90),
        (3, 6, 77),
        (3, 8, 82),
        (3, 10, 96),
        (3, 11, -90),
        (3, 12, -91),
        (3, 18, 30),
        (4, 6, -93),
        (4, 7, -5),
        (4, 8, -38),
        (4, 10, 2),
        (4, 14, 57),
        (4, 18, -11),
        (5, 6, 98),
        (5, 7, 91),
        (5, 15, -63),
        (5, 21, -83),
        (6, 8, 89),
        (6, 12, 58),
        (6, 13, 98),
        (7, 9, -27),
        (7, 10, -64),
        (7, 15, 59),
        (7, 16, -66),
        (7, 19, -5),
        (8, 9, -28),
        (8, 11, -29),
        (8, 13, -70),
        (8, 15, 38),
        (9, 17, -91),
        (10, 17, -2),
        (10, 20, 28),
        (10, 21, -11),
        (12, 13, -76),
        (12, 16, -8),
        (12, 19, 68),
        (13, 14, -57),
        (13, 20, -42),
        (13, 21, -16),
        (14, 15, 78),
        (14, 16, -2),
        (16, 17, -47),
        (18, 20, -35),
    ];
    let g = build_graph(22, &edges);

    let matching = BlossomVState::new(&g)
        .solve_with_test_budget(4096, 20_000)
        .expect("case #214 should have a perfect matching");
    let matching = normalize_pairs(&matching);

    validate_matching(22, &matching);
    assert_eq!(matching_cost(&edges, &matching), -697);
    assert_eq!(
        matching,
        vec![
            (0, 14),
            (1, 21),
            (2, 19),
            (3, 11),
            (4, 6),
            (5, 15),
            (7, 10),
            (8, 13),
            (9, 17),
            (12, 16),
            (18, 20),
        ]
    );
}

#[test]
fn test_ground_truth_case_214_public_matrix_with_budget() {
    let edges = [
        (0, 1, 90),
        (0, 2, -66),
        (0, 3, -13),
        (0, 4, -83),
        (0, 5, 73),
        (0, 7, -70),
        (0, 9, -67),
        (0, 11, 39),
        (0, 12, 40),
        (0, 13, -57),
        (0, 14, -32),
        (0, 18, -54),
        (0, 19, 73),
        (0, 20, -60),
        (1, 2, -96),
        (1, 3, -17),
        (1, 4, 12),
        (1, 5, -20),
        (1, 6, 42),
        (1, 8, -5),
        (1, 11, 31),
        (1, 14, 0),
        (1, 16, -90),
        (1, 21, -90),
        (2, 3, -88),
        (2, 4, -38),
        (2, 5, 80),
        (2, 7, 7),
        (2, 9, -27),
        (2, 10, -30),
        (2, 12, 66),
        (2, 17, -46),
        (2, 18, -66),
        (2, 19, -61),
        (3, 4, -11),
        (3, 5, 90),
        (3, 6, 77),
        (3, 8, 82),
        (3, 10, 96),
        (3, 11, -90),
        (3, 12, -91),
        (3, 18, 30),
        (4, 6, -93),
        (4, 7, -5),
        (4, 8, -38),
        (4, 10, 2),
        (4, 14, 57),
        (4, 18, -11),
        (5, 6, 98),
        (5, 7, 91),
        (5, 15, -63),
        (5, 21, -83),
        (6, 8, 89),
        (6, 12, 58),
        (6, 13, 98),
        (7, 9, -27),
        (7, 10, -64),
        (7, 15, 59),
        (7, 16, -66),
        (7, 19, -5),
        (8, 9, -28),
        (8, 11, -29),
        (8, 13, -70),
        (8, 15, 38),
        (9, 17, -91),
        (10, 17, -2),
        (10, 20, 28),
        (10, 21, -11),
        (12, 13, -76),
        (12, 16, -8),
        (12, 19, 68),
        (13, 14, -57),
        (13, 20, -42),
        (13, 21, -16),
        (14, 15, 78),
        (14, 16, -2),
        (16, 17, -47),
        (18, 20, -35),
    ];

    type PublicVcsr = crate::impls::ValuedCSR2D<usize, usize, usize, i32>;
    let mut sorted_edges: Vec<(usize, usize, i32)> = Vec::new();
    for &(i, j, w) in &edges {
        if i == j {
            continue;
        }
        let (lo, hi) = if i < j { (i, j) } else { (j, i) };
        sorted_edges.push((lo, hi, w));
        sorted_edges.push((hi, lo, w));
    }
    sorted_edges.sort_unstable();
    sorted_edges.dedup_by(|a, b| a.0 == b.0 && a.1 == b.1);

    let mut g: PublicVcsr =
        crate::traits::SparseMatrixMut::with_sparse_shaped_capacity((22, 22), sorted_edges.len());
    for (r, c, v) in sorted_edges {
        crate::traits::MatrixMut::add(&mut g, (r, c, v)).unwrap();
    }

    let matching = BlossomVState::new(&g)
        .solve_with_test_budget(4096, 20_000)
        .expect("case #214 should have a perfect matching");
    let matching = normalize_pairs(&matching);

    validate_matching(22, &matching);
    assert_eq!(matching_cost(&edges, &matching), -697);
    assert_eq!(
        matching,
        vec![
            (0, 14),
            (1, 21),
            (2, 19),
            (3, 11),
            (4, 6),
            (5, 15),
            (7, 10),
            (8, 13),
            (9, 17),
            (12, 16),
            (18, 20),
        ]
    );
}

fn case_214_edge_list() -> Vec<(usize, usize, i32)> {
    vec![
        (0, 1, 90),
        (0, 2, -66),
        (0, 3, -13),
        (0, 4, -83),
        (0, 5, 73),
        (0, 7, -70),
        (0, 9, -67),
        (0, 11, 39),
        (0, 12, 40),
        (0, 13, -57),
        (0, 14, -32),
        (0, 18, -54),
        (0, 19, 73),
        (0, 20, -60),
        (1, 2, -96),
        (1, 3, -17),
        (1, 4, 12),
        (1, 5, -20),
        (1, 6, 42),
        (1, 8, -5),
        (1, 11, 31),
        (1, 14, 0),
        (1, 16, -90),
        (1, 21, -90),
        (2, 3, -88),
        (2, 4, -38),
        (2, 5, 80),
        (2, 7, 7),
        (2, 9, -27),
        (2, 10, -30),
        (2, 12, 66),
        (2, 17, -46),
        (2, 18, -66),
        (2, 19, -61),
        (3, 4, -11),
        (3, 5, 90),
        (3, 6, 77),
        (3, 8, 82),
        (3, 10, 96),
        (3, 11, -90),
        (3, 12, -91),
        (3, 18, 30),
        (4, 6, -93),
        (4, 7, -5),
        (4, 8, -38),
        (4, 10, 2),
        (4, 14, 57),
        (4, 18, -11),
        (5, 6, 98),
        (5, 7, 91),
        (5, 15, -63),
        (5, 21, -83),
        (6, 8, 89),
        (6, 12, 58),
        (6, 13, 98),
        (7, 9, -27),
        (7, 10, -64),
        (7, 15, 59),
        (7, 16, -66),
        (7, 19, -5),
        (8, 9, -28),
        (8, 11, -29),
        (8, 13, -70),
        (8, 15, 38),
        (9, 17, -91),
        (10, 17, -2),
        (10, 20, 28),
        (10, 21, -11),
        (12, 13, -76),
        (12, 16, -8),
        (12, 19, 68),
        (13, 14, -57),
        (13, 20, -42),
        (13, 21, -16),
        (14, 15, 78),
        (14, 16, -2),
        (16, 17, -47),
        (18, 20, -35),
    ]
}

fn build_public_valued_graph(
    order: usize,
    edges: &[(usize, usize, i32)],
) -> crate::impls::ValuedCSR2D<usize, usize, usize, i32> {
    type PublicVcsr = crate::impls::ValuedCSR2D<usize, usize, usize, i32>;
    let mut sorted_edges: Vec<(usize, usize, i32)> = Vec::new();
    for &(i, j, w) in edges {
        if i == j {
            continue;
        }
        let (lo, hi) = if i < j { (i, j) } else { (j, i) };
        sorted_edges.push((lo, hi, w));
        sorted_edges.push((hi, lo, w));
    }
    sorted_edges.sort_unstable();
    sorted_edges.dedup_by(|a, b| a.0 == b.0 && a.1 == b.1);

    let mut g: PublicVcsr = crate::traits::SparseMatrixMut::with_sparse_shaped_capacity(
        (order, order),
        sorted_edges.len(),
    );
    for (r, c, v) in sorted_edges {
        crate::traits::MatrixMut::add(&mut g, (r, c, v)).unwrap();
    }
    g
}

fn case_214_state_before_first_dual() -> BlossomVState<Vcsr> {
    let edges = case_214_edge_list();
    let g = build_graph(22, &edges);
    let leaked = Box::leak(Box::new(g));
    let mut state = BlossomVState::new(leaked);
    state.init_global();
    state.mark_tree_roots_processed();

    let mut passes = 0usize;
    while state.generic_primal_pass_once() {
        passes += 1;
        assert!(passes <= 4, "generic phase should stall before first dual on case #214");
    }

    state
}

fn case_24943_edges() -> Vec<(usize, usize, i32)> {
    vec![
        (0, 2, -30),
        (0, 8, 16),
        (0, 28, -13),
        (0, 29, 36),
        (1, 2, -4),
        (1, 3, 5),
        (2, 3, -42),
        (2, 4, -53),
        (3, 5, 74),
        (3, 29, -3),
        (4, 6, 67),
        (4, 20, -76),
        (5, 6, 5),
        (5, 7, -67),
        (6, 7, -55),
        (6, 8, 74),
        (7, 9, 82),
        (7, 19, 44),
        (8, 9, 97),
        (8, 10, 8),
        (8, 16, 97),
        (9, 10, 95),
        (9, 11, 47),
        (10, 11, -21),
        (10, 12, -70),
        (11, 12, -69),
        (11, 13, 24),
        (12, 13, 80),
        (12, 14, -90),
        (13, 14, -44),
        (13, 15, 61),
        (14, 15, 11),
        (14, 16, -72),
        (15, 16, 96),
        (15, 17, 83),
        (16, 17, -67),
        (16, 29, 42),
        (17, 18, -41),
        (17, 19, -89),
        (18, 19, -61),
        (18, 20, 91),
        (19, 20, -10),
        (19, 21, 58),
        (20, 21, -38),
        (20, 22, -39),
        (21, 22, 60),
        (21, 23, 65),
        (22, 23, -41),
        (22, 24, -14),
        (23, 24, 36),
        (23, 25, -24),
        (24, 25, 25),
        (24, 26, -76),
        (25, 26, -86),
        (25, 27, -45),
        (26, 27, -35),
        (26, 28, 57),
        (27, 28, 97),
        (27, 29, -88),
        (28, 29, 15),
    ]
}

fn case_24595_edges() -> Vec<(usize, usize, i32)> {
    vec![
        (0, 7, -6),
        (0, 9, 14),
        (0, 16, -17),
        (1, 2, 34),
        (1, 3, 98),
        (1, 6, 58),
        (1, 10, 24),
        (1, 11, -22),
        (1, 15, 49),
        (2, 3, 72),
        (2, 5, 55),
        (2, 6, 72),
        (2, 8, -82),
        (2, 10, 90),
        (2, 11, -13),
        (2, 15, 26),
        (2, 17, 93),
        (3, 5, -51),
        (3, 6, -66),
        (3, 8, 10),
        (3, 9, -84),
        (3, 11, -93),
        (3, 14, 43),
        (3, 15, -9),
        (3, 17, -50),
        (4, 12, 45),
        (4, 13, 43),
        (4, 14, 16),
        (5, 6, 84),
        (5, 7, 32),
        (5, 8, -92),
        (5, 9, -58),
        (5, 14, 66),
        (5, 15, 76),
        (5, 16, 47),
        (5, 17, -59),
        (6, 8, -9),
        (6, 9, -1),
        (6, 10, 70),
        (6, 15, -55),
        (6, 17, -23),
        (7, 9, 42),
        (7, 16, -98),
        (7, 17, 41),
        (8, 9, -71),
        (8, 14, -66),
        (8, 15, 10),
        (8, 17, 24),
        (9, 15, -75),
        (9, 16, -62),
        (9, 17, 35),
        (10, 11, -67),
        (10, 15, 53),
        (12, 13, 17),
        (14, 17, -75),
        (15, 17, -18),
        (16, 17, 20),
    ]
}

fn case_honggfuzz_sigabrt_10_edges() -> Vec<(usize, usize, i32)> {
    vec![
        (0, 1, 0),
        (0, 2, -18577),
        (0, 4, 0),
        (0, 7, -22022),
        (0, 8, -31624),
        (0, 10, 512),
        (0, 11, 0),
        (0, 12, 0),
        (0, 15, 0),
        (0, 16, -27420),
        (0, 17, -26215),
        (1, 3, -1),
        (1, 5, 32512),
        (1, 8, 1),
        (1, 9, -30271),
        (1, 10, 5020),
        (1, 13, 12937),
        (2, 3, 2303),
        (2, 4, 100),
        (2, 9, -2846),
        (2, 14, 76),
        (2, 16, 26984),
        (2, 17, -20523),
        (3, 4, 15679),
        (3, 6, -1),
        (3, 12, 3072),
        (3, 13, 511),
        (3, 15, 22123),
        (3, 16, -13726),
        (4, 5, 2752),
        (4, 8, 26125),
        (4, 17, -18671),
        (5, 8, 12331),
        (5, 14, -10251),
        (6, 7, -30029),
        (6, 10, -10397),
        (6, 11, 0),
        (7, 9, 13364),
        (8, 10, -1387),
        (8, 12, -24415),
        (8, 15, -18235),
        (9, 10, -26215),
        (9, 13, 21062),
        (9, 14, -26215),
        (9, 16, -18577),
        (10, 11, -12279),
        (10, 13, -8642),
        (11, 13, -7374),
        (11, 14, 26851),
        (12, 14, 14393),
        (12, 15, -24),
        (12, 17, 50),
        (16, 17, 1128),
    ]
}

fn case_28832_edges() -> Vec<(usize, usize, i32)> {
    vec![
        (0, 2, -90),
        (0, 3, 63),
        (0, 5, 80),
        (0, 6, 46),
        (0, 12, -15),
        (0, 17, 98),
        (0, 20, -61),
        (0, 21, -7),
        (1, 2, -79),
        (1, 5, 15),
        (1, 6, -49),
        (1, 7, -2),
        (1, 11, -46),
        (1, 18, -25),
        (1, 20, -5),
        (2, 3, -32),
        (2, 4, -10),
        (2, 5, -39),
        (2, 7, -75),
        (2, 16, 41),
        (2, 19, -11),
        (2, 20, 15),
        (2, 21, -68),
        (3, 4, -91),
        (3, 5, -84),
        (3, 6, 86),
        (3, 7, -96),
        (3, 18, 70),
        (3, 19, 62),
        (3, 20, -34),
        (3, 21, 68),
        (4, 6, -93),
        (4, 7, 66),
        (4, 8, 78),
        (4, 9, 98),
        (4, 19, -72),
        (5, 6, -26),
        (5, 7, -81),
        (5, 9, 20),
        (5, 10, 20),
        (5, 18, -67),
        (5, 19, 12),
        (5, 21, 73),
        (6, 7, 75),
        (6, 8, 78),
        (6, 9, -70),
        (6, 11, 33),
        (6, 19, -80),
        (6, 20, 74),
        (7, 8, -27),
        (7, 10, 98),
        (7, 12, 37),
        (7, 18, -47),
        (7, 19, 9),
        (8, 9, -83),
        (8, 10, -59),
        (8, 11, -25),
        (8, 13, -83),
        (8, 14, -95),
        (8, 16, 27),
        (8, 18, -87),
        (8, 19, -19),
        (9, 10, 88),
        (9, 11, 15),
        (9, 12, -3),
        (9, 13, -56),
        (9, 14, 54),
        (10, 11, -76),
        (10, 12, 72),
        (10, 13, -42),
        (10, 14, 18),
        (10, 15, 17),
        (10, 18, -12),
        (11, 12, -51),
        (11, 14, -86),
        (11, 15, 52),
        (11, 16, 10),
        (12, 13, 19),
        (12, 14, 4),
        (12, 15, 32),
        (12, 16, -75),
        (12, 17, -57),
        (12, 20, 63),
        (13, 14, -67),
        (13, 16, 4),
        (13, 17, 61),
        (13, 18, -61),
        (13, 19, -35),
        (14, 16, 18),
        (14, 17, -82),
        (14, 18, 94),
        (14, 21, -41),
        (15, 16, -36),
        (15, 17, -54),
        (15, 18, 99),
        (15, 19, 3),
        (15, 20, 71),
        (16, 18, -66),
        (16, 19, 0),
        (16, 20, 37),
        (16, 21, -66),
        (17, 18, 30),
        (17, 19, 69),
        (17, 20, 20),
        (17, 21, 86),
        (18, 19, -63),
        (18, 20, 17),
        (18, 21, -24),
        (19, 20, -71),
        (20, 21, 50),
    ]
}

fn case_21222_edges() -> Vec<(usize, usize, i32)> {
    vec![
        (0, 2, 6),
        (0, 11, -82),
        (0, 18, 81),
        (0, 25, 69),
        (0, 26, -22),
        (0, 27, -22),
        (1, 2, -51),
        (1, 3, -85),
        (1, 4, 59),
        (1, 7, -50),
        (1, 14, 71),
        (1, 26, -30),
        (1, 27, -21),
        (2, 3, -23),
        (2, 4, 85),
        (2, 5, 52),
        (3, 4, 41),
        (3, 5, -50),
        (3, 6, -23),
        (4, 5, -96),
        (4, 6, -42),
        (4, 7, -29),
        (5, 6, 79),
        (5, 7, 18),
        (5, 8, -35),
        (6, 7, -51),
        (6, 8, 84),
        (6, 9, -91),
        (6, 10, 25),
        (7, 8, -34),
        (7, 10, -48),
        (8, 9, 87),
        (8, 10, -58),
        (8, 11, 46),
        (8, 15, 10),
        (9, 10, 33),
        (9, 11, 18),
        (9, 12, 71),
        (10, 11, -38),
        (10, 12, 8),
        (11, 12, -38),
        (11, 13, -50),
        (11, 14, -79),
        (12, 13, 83),
        (12, 14, -42),
        (12, 15, 12),
        (12, 19, 55),
        (13, 14, 86),
        (13, 15, -66),
        (13, 16, 53),
        (14, 15, -79),
        (14, 17, 75),
        (14, 27, -86),
        (15, 16, 17),
        (15, 18, 14),
        (16, 17, -5),
        (16, 18, -6),
        (16, 19, 19),
        (17, 18, 65),
        (17, 19, -29),
        (17, 20, -92),
        (18, 19, 16),
        (18, 20, -56),
        (18, 21, -59),
        (19, 20, 42),
        (19, 22, 56),
        (20, 21, 73),
        (20, 22, -94),
        (20, 23, -55),
        (21, 22, 90),
        (21, 23, -33),
        (21, 24, 92),
        (22, 23, 71),
        (22, 24, -69),
        (22, 25, 7),
        (23, 24, -74),
        (23, 25, 92),
        (23, 26, -81),
        (24, 25, -79),
        (24, 26, -64),
        (24, 27, -92),
        (25, 26, 79),
        (25, 27, 60),
        (26, 27, -4),
    ]
}

fn assert_tree_navigation_invariants<M: SparseValuedMatrix2D + ?Sized>(state: &BlossomVState<M>)
where
    M::Value: Number + AsPrimitive<i64>,
    M::RowIndex: PositiveInteger,
    M::ColumnIndex: PositiveInteger,
{
    for (idx, node) in state.nodes.iter().enumerate() {
        if !node.is_outer || node.flag != PLUS || node.is_tree_root {
            continue;
        }
        assert_ne!(
            node.match_arc, NONE,
            "outer non-root PLUS node {idx} lost match_arc during tree navigation"
        );
        let minus = state.arc_head_outer(node.match_arc);
        assert_ne!(
            minus, NONE,
            "outer non-root PLUS node {idx} resolves to NONE minus via match_arc={}",
            node.match_arc
        );
        assert_eq!(
            state.nodes[minus as usize].flag, MINUS,
            "outer non-root PLUS node {idx} points to non-MINUS node {minus}"
        );
        assert_ne!(
            state.nodes[minus as usize].tree_parent_arc, NONE,
            "MINUS node {minus} lost tree_parent_arc while still attached to PLUS node {idx}"
        );
        let parent_plus = state.arc_head_outer(state.nodes[minus as usize].tree_parent_arc);
        assert_ne!(
            parent_plus, NONE,
            "MINUS node {minus} resolves to NONE parent via tree_parent_arc={}",
            state.nodes[minus as usize].tree_parent_arc
        );
    }
}

fn solve_case_24943_with_tree_checks<M: SparseValuedMatrix2D + ?Sized>(
    state: &mut BlossomVState<M>,
    max_outer_iters: usize,
    max_inner_iters: usize,
) -> Result<(), BlossomVError>
where
    M::Value: Number + AsPrimitive<i64>,
    M::RowIndex: PositiveInteger,
    M::ColumnIndex: PositiveInteger,
{
    state.init_global();
    assert_tree_navigation_invariants(state);

    let mut outer_iters = 0usize;
    loop {
        if state.tree_num == 0 {
            break;
        }
        outer_iters += 1;
        assert!(outer_iters <= max_outer_iters, "case #24943 exceeded outer iteration budget");

        state.mark_tree_roots_processed();
        assert_tree_navigation_invariants(state);

        let mut progressed = false;
        let mut inner_iters = 0usize;
        loop {
            inner_iters += 1;
            assert!(inner_iters <= max_inner_iters, "case #24943 exceeded inner iteration budget");
            let step = state.generic_primal_pass_once();
            assert_tree_navigation_invariants(state);
            if !step {
                break;
            }
            progressed = true;
            if state.tree_num == 0 {
                break;
            }
        }

        if state.tree_num == 0 {
            break;
        }

        if !progressed && !state.update_duals() {
            return Err(BlossomVError::NoPerfectMatching);
        }
        assert_tree_navigation_invariants(state);
    }

    Ok(())
}

#[test]
fn test_case_214_scheduler_local_eps_matches_visible_scan_before_first_dual() {
    let mut state = case_214_state_before_first_dual();
    let roots = state.current_root_list();
    assert!(!roots.is_empty(), "case #214 should have active roots before first dual");

    for root in roots {
        assert_eq!(
            state.compute_tree_local_eps(root),
            state.compute_tree_local_eps_visible_scan(root),
            "case #214 local eps mismatch for root {root}",
        );
    }
}

#[test]
fn test_case_214_scheduler_pair_caps_match_visible_scan_before_first_dual() {
    let state = case_214_state_before_first_dual();
    let roots = state.current_root_list();
    assert!(!roots.is_empty(), "case #214 should have active roots before first dual");

    let mut root_to_var = vec![usize::MAX; state.nodes.len()];
    for (var, &root) in roots.iter().enumerate() {
        root_to_var[root as usize] = var;
    }

    let inf_cap = i64::MAX / 4;
    let mut visible_pair_eps00 = Vec::new();
    let mut visible_pair_eps01 = Vec::new();
    state.fill_dual_pair_caps_visible_scan(
        &roots,
        &root_to_var,
        inf_cap,
        &mut visible_pair_eps00,
        &mut visible_pair_eps01,
    );

    let mut scheduler_state = case_214_state_before_first_dual();
    let mut scheduler_pair_eps00 = Vec::new();
    let mut scheduler_pair_eps01 = Vec::new();
    scheduler_state.fill_dual_pair_caps_from_scheduler(
        &roots,
        &root_to_var,
        inf_cap,
        &mut scheduler_pair_eps00,
        &mut scheduler_pair_eps01,
    );

    assert_eq!(scheduler_pair_eps00, visible_pair_eps00);
    assert_eq!(scheduler_pair_eps01, visible_pair_eps01);
}

#[test]
fn test_case_9_does_not_expose_missed_local_expand_state() {
    let edges = case_9_edges();
    let g = build_graph(18, &edges);
    let leaked = Box::leak(Box::new(g));
    let mut state = BlossomVState::new(leaked);
    state.init_global();

    for outer_iter in 0..64 {
        let mut progressed = true;
        let mut inner_iters = 0usize;
        while progressed {
            progressed = generic_primal_pass_without_global_expand_fallback_once(&mut state);
            inner_iters += usize::from(progressed);
            assert!(
                inner_iters <= 512,
                "case 9 did not stall before finding a local/global expand mismatch"
            );
            if state.tree_num == 0 {
                return;
            }
        }

        if let Some(blossom) = find_global_expand_fallback_blossom_for_test(&state) {
            let root = state.find_tree_root(blossom);
            let eps_root = state.tree_eps(root);
            let match_edge = arc_edge(state.nodes[blossom as usize].match_arc);
            let queue_owner = state.edge_queue_owner(match_edge);
            let outer0 = state.edge_head_outer(match_edge, 0);
            let outer1 = state.edge_head_outer(match_edge, 1);
            let raw = state.edges[match_edge as usize].head;
            assert_eq!(
                state.find_tree_expand_blossom_with_eps(root, eps_root),
                Some(blossom),
                "case 9 missed local expand: blossom={blossom} root={root} eps_root={eps_root} match_edge={match_edge} owner={queue_owner:?} outer=({outer0},{outer1}) raw={raw:?} pq_blossoms={:?}",
                state.scheduler_trees[root as usize].pq_blossoms,
            );
            return;
        }

        assert!(
            state.update_duals(),
            "case 9 failed dual update before exposing the missed local expand state at outer_iter={outer_iter}"
        );
    }

    panic!("case 9 did not expose a missed local expand state within the search budget");
}

#[test]
fn test_case_6_does_not_expose_missed_local_expand_state() {
    let edges = case_honggfuzz_sigabrt_6_edges();
    let g = build_graph(26, &edges);
    let leaked = Box::leak(Box::new(g));
    let mut state = BlossomVState::new(leaked);
    state.init_global();

    for outer_iter in 0..128 {
        let mut progressed = true;
        let mut inner_iters = 0usize;
        while progressed {
            progressed = generic_primal_pass_without_global_expand_fallback_once(&mut state);
            inner_iters += usize::from(progressed);
            assert!(
                inner_iters <= 2048,
                "case 6 did not stall before finding a local/global expand mismatch"
            );
            if state.tree_num == 0 {
                return;
            }
        }

        if let Some(blossom) = find_global_expand_fallback_blossom_for_test(&state) {
            let root = state.find_tree_root(blossom);
            let eps_root = state.tree_eps(root);
            let match_edge = arc_edge(state.nodes[blossom as usize].match_arc);
            let queue_owner = state.edge_queue_owner(match_edge);
            let outer0 = state.edge_head_outer(match_edge, 0);
            let outer1 = state.edge_head_outer(match_edge, 1);
            let raw = state.edges[match_edge as usize].head;
            assert_eq!(
                state.find_tree_expand_blossom_with_eps(root, eps_root),
                Some(blossom),
                "case 6 missed local expand: blossom={blossom} root={root} eps_root={eps_root} match_edge={match_edge} owner={queue_owner:?} outer=({outer0},{outer1}) raw={raw:?} pq_blossoms={:?}",
                state.scheduler_trees[root as usize].pq_blossoms,
            );
            return;
        }

        assert!(
            state.update_duals(),
            "case 6 failed dual update before exposing the missed local expand state at outer_iter={outer_iter}"
        );
    }

    panic!("case 6 did not expose a missed local expand state within the search budget");
}

#[test]
fn test_case_14_no_longer_exposes_missed_global_expand_fallback_state() {
    let edges = case_honggfuzz_sigabrt_14_edges();
    let g = build_graph(18, &edges);
    let leaked = Box::leak(Box::new(g));
    let mut state = BlossomVState::new(leaked);
    state.init_global();

    for outer_iter in 0..128 {
        let mut progressed = true;
        let mut inner_iters = 0usize;
        while progressed {
            progressed = generic_primal_pass_without_global_expand_fallback_once(&mut state);
            inner_iters += usize::from(progressed);
            assert!(
                inner_iters <= 2048,
                "case 14 exceeded the inner-step budget without the global expand fallback"
            );
            if state.tree_num == 0 {
                return;
            }
        }

        if let Some(blossom) = find_global_expand_fallback_blossom_for_test(&state) {
            let root = state.find_tree_root(blossom);
            let eps_root = state.tree_eps(root);
            let before = state.test_generic_primal_steps().len();
            assert_eq!(
                state.find_tree_expand_blossom_with_eps(root, eps_root),
                Some(blossom),
                "case 14 still exposes a blossom that only the fallback can see: blossom={blossom} root={root} eps_root={eps_root} pq_blossoms={:?}",
                state.scheduler_trees[root as usize].pq_blossoms,
            );
            assert!(
                generic_primal_pass_without_global_expand_fallback_once(&mut state),
                "case 14 found a fallback blossom {blossom} that still did not produce local primal progress"
            );
            let step = &state.test_generic_primal_steps()[before];
            assert_eq!(
                step.event,
                GenericPrimalEvent::Expand { blossom },
                "case 14 local no-fallback path diverged after rediscovering blossom {blossom}"
            );
            return;
        }

        assert!(
            state.update_duals(),
            "case 14 failed dual update before local no-fallback progress at outer_iter={outer_iter}"
        );
    }

    panic!("case 14 did not solve or rediscover the local expand within the search budget");
}

#[test]
fn test_case_14_next_primal_event_at_head_checkpoint() {
    let edges = case_honggfuzz_sigabrt_14_edges();
    let g = build_graph(18, &edges);
    let leaked = Box::leak(Box::new(g));
    let mut state = BlossomVState::new(leaked);
    state.init_global();

    for outer_iter in 0..128 {
        let roots = state.current_root_list();
        let at_head_checkpoint = roots == vec![19, 20]
            && state.scheduler_trees[19].pq0 == vec![33]
            && state.scheduler_trees[19].pq00_local == vec![47, 36, 38]
            && state.scheduler_trees[19].pq_blossoms.is_empty()
            && state.scheduler_trees[20].pq0 == vec![32, 21]
            && state.scheduler_trees[20].pq00_local == vec![22]
            && state.scheduler_trees[20].pq_blossoms.is_empty()
            && state.scheduler_tree_edges[5].head == [20, 19]
            && state.scheduler_tree_edges[5].pq00 == vec![42, 40]
            && state.scheduler_tree_edges[5].pq01[0].is_empty()
            && state.scheduler_tree_edges[5].pq01[1].is_empty();

        if at_head_checkpoint {
            let eps19 = state.tree_eps(19);
            let eps20 = state.tree_eps(20);
            let shrink19 = state.find_tree_shrink_edge_with_cap(19, eps19.saturating_mul(2));
            let shrink20 = state.find_tree_shrink_edge_with_cap(20, eps20.saturating_mul(2));
            let expand19 = state.find_tree_expand_blossom_with_eps(19, eps19);
            let expand20 = state.find_tree_expand_blossom_with_eps(20, eps20);
            let fallback = find_global_expand_fallback_blossom_for_test(&state);
            let match_edge_18 = arc_edge(state.nodes[18].match_arc);
            let match_partner_18 = state.arc_head_outer(state.nodes[18].match_arc);
            let before = state.test_generic_primal_steps().len();
            assert!(
                generic_primal_pass_without_global_expand_fallback_once(&mut state),
                "case 14 checkpoint should still make local primal progress; eps19={eps19} eps20={eps20} shrink19={shrink19:?} shrink20={shrink20:?} expand19={expand19:?} expand20={expand20:?} fallback={fallback:?} match_edge18={match_edge_18} owner18={:?} match_partner18={match_partner_18} partner_flag={:?} partner_root={:?} partner_is_tree_root={:?} partner_is_processed={:?} blossom18_root={:?} blossom18_is_processed={:?} owner40={:?} owner42={:?} owner44={:?} pq_blossoms19={:?} pq_blossoms20={:?}",
                state.edge_queue_owner(match_edge_18),
                if match_partner_18 != NONE {
                    Some(state.nodes[match_partner_18 as usize].flag)
                } else {
                    None
                },
                if match_partner_18 != NONE {
                    Some(state.find_tree_root(match_partner_18))
                } else {
                    None
                },
                if match_partner_18 != NONE {
                    Some(state.nodes[match_partner_18 as usize].is_tree_root)
                } else {
                    None
                },
                if match_partner_18 != NONE {
                    Some(state.nodes[match_partner_18 as usize].is_processed)
                } else {
                    None
                },
                Some(state.find_tree_root(18)),
                Some(state.nodes[18].is_processed),
                state.edge_queue_owner(40),
                state.edge_queue_owner(42),
                state.edge_queue_owner(44),
                state.scheduler_trees[19].pq_blossoms,
                state.scheduler_trees[20].pq_blossoms,
            );
            let step = &state.test_generic_primal_steps()[before];
            assert_eq!(
                step.event,
                GenericPrimalEvent::Expand { blossom: 18 },
                "case 14 diverged from committed HEAD at the checkpoint in outer_iter={outer_iter}; eps19={eps19} eps20={eps20} shrink19={shrink19:?} shrink20={shrink20:?} expand19={expand19:?} expand20={expand20:?} fallback={fallback:?} owner40={:?} owner42={:?} owner44={:?} pq_blossoms19={:?} pq_blossoms20={:?}",
                state.edge_queue_owner(40),
                state.edge_queue_owner(42),
                state.edge_queue_owner(44),
                state.scheduler_trees[19].pq_blossoms,
                state.scheduler_trees[20].pq_blossoms,
            );
            return;
        }

        let mut progressed = true;
        let mut inner_iters = 0usize;
        while progressed {
            progressed = generic_primal_pass_without_global_expand_fallback_once(&mut state);
            inner_iters += usize::from(progressed);
            assert!(inner_iters <= 2048, "case 14 exceeded the inner-step budget");
            if state.tree_num == 0 {
                return;
            }
        }

        assert!(
            state.update_duals(),
            "case 14 failed dual update before reaching the committed-HEAD checkpoint at outer_iter={outer_iter}"
        );
    }

    panic!("case 14 never reached the committed-HEAD checkpoint within the search budget");
}

#[test]
fn test_ground_truth_case_24943_with_budget() {
    let edges = case_24943_edges();
    let g = build_graph(30, &edges);
    let state = BlossomVState::new(&g);
    let pairs = state.solve_with_test_budget(400, 2000).expect("case #24943 should solve");
    let mut sorted = pairs;
    for (u, v) in &mut sorted {
        if *u > *v {
            core::mem::swap(u, v);
        }
    }
    sorted.sort_unstable();
    assert_eq!(
        sorted,
        vec![
            (0, 28),
            (1, 2),
            (3, 29),
            (4, 6),
            (5, 7),
            (8, 10),
            (9, 11),
            (12, 14),
            (13, 15),
            (16, 17),
            (18, 19),
            (20, 21),
            (22, 23),
            (24, 26),
            (25, 27),
        ]
    );
    let mut cost = 0i64;
    for &(u, v) in &sorted {
        let uv = edges
            .iter()
            .find_map(|&(a, b, w)| ((a == u && b == v) || (a == v && b == u)).then_some(w as i64))
            .expect("matching edge must exist");
        cost += uv;
    }
    assert_eq!(cost, -322);
}

#[test]
fn test_ground_truth_case_24595_with_budget() {
    let edges = case_24595_edges();
    let g = build_graph(18, &edges);
    let state = BlossomVState::new(&g);
    let pairs = state.solve_with_test_budget(500, 4000).expect("case #24595 should solve");
    let mut sorted = pairs;
    for (u, v) in &mut sorted {
        if *u > *v {
            core::mem::swap(u, v);
        }
    }
    sorted.sort_unstable();
    assert_eq!(
        sorted,
        vec![(0, 9), (1, 10), (2, 8), (3, 11), (4, 14), (5, 17), (6, 15), (7, 16), (12, 13),]
    );
    let mut cost = 0i64;
    for &(u, v) in &sorted {
        let uv = edges
            .iter()
            .find_map(|&(a, b, w)| ((a == u && b == v) || (a == v && b == u)).then_some(w as i64))
            .expect("matching edge must exist");
        cost += uv;
    }
    assert_eq!(cost, -316);
}

#[test]
fn test_ground_truth_case_24595_plain_solve() {
    let edges = case_24595_edges();
    let g = build_graph(18, &edges);
    let pairs = BlossomVState::new(&g).solve().expect("case #24595 should solve via plain solve()");
    let mut sorted = pairs;
    for (u, v) in &mut sorted {
        if *u > *v {
            core::mem::swap(u, v);
        }
    }
    sorted.sort_unstable();
    assert_eq!(
        sorted,
        vec![(0, 9), (1, 10), (2, 8), (3, 11), (4, 14), (5, 17), (6, 15), (7, 16), (12, 13),]
    );
    let mut cost = 0i64;
    for &(u, v) in &sorted {
        let uv = edges
            .iter()
            .find_map(|&(a, b, w)| ((a == u && b == v) || (a == v && b == u)).then_some(w as i64))
            .expect("matching edge must exist");
        cost += uv;
    }
    assert_eq!(cost, -316);
}

#[test]
fn test_case_10_scheduler_pair_caps_match_visible_scan_before_failed_dual() {
    let edges = case_honggfuzz_sigabrt_10_edges();
    let g = build_public_valued_graph(18, &edges);
    let leaked = Box::leak(Box::new(g));
    let mut state = BlossomVState::new(leaked);
    state.init_global();

    for outer_iter in 0..128 {
        let mut progressed = true;
        let mut inner_iters = 0usize;
        while progressed {
            progressed = state.generic_primal_pass_once();
            inner_iters += usize::from(progressed);
            assert!(inner_iters <= 2048, "case 10 exceeded the inner-step budget");
            if state.tree_num == 0 {
                return;
            }
        }

        let roots = state.current_root_list();
        if !roots.is_empty() {
            let mut root_to_var = vec![usize::MAX; state.nodes.len()];
            for (var, &root) in roots.iter().enumerate() {
                root_to_var[root as usize] = var;
            }
            let inf_cap = i64::MAX / 4;
            let mut visible_pair_eps00 = Vec::new();
            let mut visible_pair_eps01 = Vec::new();
            state.fill_dual_pair_caps_visible_scan(
                &roots,
                &root_to_var,
                inf_cap,
                &mut visible_pair_eps00,
                &mut visible_pair_eps01,
            );
            let mut scheduler_pair_eps00 = Vec::new();
            let mut scheduler_pair_eps01 = Vec::new();
            state.fill_dual_pair_caps_from_scheduler(
                &roots,
                &root_to_var,
                inf_cap,
                &mut scheduler_pair_eps00,
                &mut scheduler_pair_eps01,
            );
            assert_eq!(
                scheduler_pair_eps00, visible_pair_eps00,
                "case 10 pair_eps00 mismatch at outer_iter={outer_iter} roots={roots:?}"
            );
            assert_eq!(
                scheduler_pair_eps01, visible_pair_eps01,
                "case 10 pair_eps01 mismatch at outer_iter={outer_iter} roots={roots:?}"
            );
        }

        assert!(
            state.update_duals(),
            "case 10 failed dual update without exposing a scheduler/visible pair-cap mismatch at outer_iter={outer_iter}"
        );
    }

    panic!("case 10 did not solve within the search budget");
}

#[test]
fn test_case_10_public_api_still_solves_in_unit_build() {
    let edges = case_honggfuzz_sigabrt_10_edges();
    let g = build_public_valued_graph(18, &edges);
    let matching = crate::traits::algorithms::BlossomV::blossom_v(&g)
        .expect("case 10 should still solve through the public API in unit tests");
    validate_matching(18, &normalize_pairs(&matching));
}

#[test]
fn test_ground_truth_case_24595_solve_matches_budget_and_public_path() {
    let edges = case_24595_edges();
    let g = build_graph(18, &edges);

    let mut budget_pairs = BlossomVState::new(&g)
        .solve_with_test_budget(500, 4000)
        .expect("case #24595 should solve with budget");
    let mut direct_pairs =
        BlossomVState::new(&g).solve().expect("case #24595 should solve directly");
    let mut public_pairs = g.blossom_v().expect("case #24595 should solve via public path");

    for (u, v) in &mut budget_pairs {
        if *u > *v {
            core::mem::swap(u, v);
        }
    }
    budget_pairs.sort_unstable();

    for (u, v) in &mut direct_pairs {
        if *u > *v {
            core::mem::swap(u, v);
        }
    }
    direct_pairs.sort_unstable();

    for (u, v) in &mut public_pairs {
        if *u > *v {
            core::mem::swap(u, v);
        }
    }
    public_pairs.sort_unstable();

    assert_eq!(direct_pairs, budget_pairs, "solve() diverged from solve_with_test_budget()");
    assert_eq!(public_pairs, budget_pairs, "public blossom_v() diverged from internal solve path");
}

#[test]
fn test_ground_truth_case_28832_with_budget() {
    let edges = case_28832_edges();
    let g = build_graph(22, &edges);
    let state = BlossomVState::new(&g);
    let pairs = state.solve_with_test_budget(400, 2000).expect("case #28832 should solve");
    let mut sorted = pairs;
    for (u, v) in &mut sorted {
        if *u > *v {
            core::mem::swap(u, v);
        }
    }
    sorted.sort_unstable();
    assert_eq!(
        sorted,
        vec![
            (0, 20),
            (1, 2),
            (3, 4),
            (5, 7),
            (6, 19),
            (8, 9),
            (10, 11),
            (12, 16),
            (13, 18),
            (14, 21),
            (15, 17),
        ]
    );
    let mut cost = 0i64;
    for &(u, v) in &sorted {
        let uv = edges
            .iter()
            .find_map(|&(a, b, w)| ((a == u && b == v) || (a == v && b == u)).then_some(w as i64))
            .expect("matching edge must exist");
        cost += uv;
    }
    assert_eq!(cost, -782);
}

#[test]
fn test_ground_truth_case_21222_with_budget() {
    let edges = case_21222_edges();
    let g = build_graph(28, &edges);
    let state = BlossomVState::new(&g);
    let pairs = state.solve_with_test_budget(500, 3000).expect("case #21222 should solve");
    let mut sorted = pairs;
    for (u, v) in &mut sorted {
        if *u > *v {
            core::mem::swap(u, v);
        }
    }
    sorted.sort_unstable();
    assert_eq!(
        sorted,
        vec![
            (0, 11),
            (1, 7),
            (2, 3),
            (4, 5),
            (6, 9),
            (8, 10),
            (12, 19),
            (13, 15),
            (14, 27),
            (16, 17),
            (18, 21),
            (20, 22),
            (23, 26),
            (24, 25),
        ]
    );
    let mut cost = 0i64;
    for &(u, v) in &sorted {
        let uv = edges
            .iter()
            .find_map(|&(a, b, w)| ((a == u && b == v) || (a == v && b == u)).then_some(w as i64))
            .expect("matching edge must exist");
        cost += uv;
    }
    assert_eq!(cost, -815);
}

#[test]
fn test_case_24943_tree_navigation_invariants_hold_during_solve() {
    let g = build_graph(30, &case_24943_edges());
    let mut state = BlossomVState::new(&g);
    let _ = solve_case_24943_with_tree_checks(&mut state, 400, 2000)
        .expect("case #24943 should satisfy tree navigation invariants");
}

#[test]
fn test_solve_single_edge_via_greedy() {
    // When greedy matches everything, solve should return immediately
    let g = build_graph(2, &[(0, 1, 42)]);
    let result = BlossomVState::new(&g).solve();
    let pairs = result.expect("should succeed");
    assert_eq!(pairs, vec![(0usize, 1usize)]);
}

#[test]
fn test_edge_list_iteration_circular() {
    // Verify circular list integrity
    let g = build_graph(4, &[(0, 1, 1), (0, 2, 2), (0, 3, 3)]);
    let state = BlossomVState::new(&g);
    let mut neighbors = Vec::new();
    state.for_each_edge(0, |_e, dir, edge| {
        neighbors.push(edge.head[dir] as usize);
    });
    neighbors.sort();
    assert_eq!(neighbors, vec![1, 2, 3]);
}

#[test]
fn test_process_expand_selfloop_relinks_distinct_penultimate_nodes() {
    let g = build_graph(4, &[(0, 1, 1)]);
    let mut state = BlossomVState::new(&g);
    let e_idx = 0u32;

    for node in &mut state.nodes {
        node.first = [NONE; 2];
    }
    state.edges[e_idx as usize].next = [NONE; 2];
    state.edges[e_idx as usize].prev = [NONE; 2];
    state.edges[e_idx as usize].head = [0, 1];
    state.edges[e_idx as usize].slack = 20;

    state.nodes[0].is_outer = false;
    state.nodes[0].blossom_parent = 2;
    state.nodes[0].blossom_eps = 7;

    state.nodes[1].is_outer = false;
    state.nodes[1].blossom_parent = 3;
    state.nodes[1].blossom_eps = 11;

    state.nodes[2].is_outer = true;
    state.nodes[3].is_outer = true;

    state.process_expand_selfloop(e_idx);

    assert_eq!(state.nodes[0].first[1], e_idx);
    assert_eq!(state.nodes[1].first[0], e_idx);
    assert_eq!(state.edges[e_idx as usize].prev[0], e_idx);
    assert_eq!(state.edges[e_idx as usize].next[0], e_idx);
    assert_eq!(state.edges[e_idx as usize].prev[1], e_idx);
    assert_eq!(state.edges[e_idx as usize].next[1], e_idx);
    assert_eq!(state.edges[e_idx as usize].slack, 6);
}

#[test]
fn test_process_expand_selfloop_stashes_edge_on_shared_penultimate_node() {
    let g = build_graph(3, &[(0, 1, 1)]);
    let mut state = BlossomVState::new(&g);
    let e_idx = 0u32;

    for node in &mut state.nodes {
        node.first = [NONE; 2];
    }
    state.edges[e_idx as usize].next = [NONE; 2];
    state.edges[e_idx as usize].prev = [NONE; 2];
    state.edges[e_idx as usize].head = [0, 1];

    state.nodes[0].is_outer = false;
    state.nodes[0].blossom_parent = 1;

    state.nodes[1].is_outer = false;
    state.nodes[1].blossom_parent = 2;
    state.nodes[1].blossom_selfloops = NONE;

    state.nodes[2].is_outer = true;

    state.process_expand_selfloop(e_idx);

    assert_eq!(state.nodes[1].blossom_selfloops, e_idx);
    assert_eq!(state.edges[e_idx as usize].next[0], NONE);
}

#[test]
fn test_process_expand_selfloop_returns_when_edge_head_is_none() {
    let g = build_graph(2, &[(0, 1, 1)]);
    let mut state = BlossomVState::new(&g);
    let e_idx = 0u32;

    state.edges[e_idx as usize].head = [NONE, 1];
    state.nodes[0].blossom_selfloops = 9;
    state.nodes[1].blossom_selfloops = 11;

    state.process_expand_selfloop(e_idx);

    assert_eq!(state.nodes[0].blossom_selfloops, 9);
    assert_eq!(state.nodes[1].blossom_selfloops, 11);
}

#[test]
fn test_process_expand_selfloop_returns_when_penultimate_is_missing() {
    let g = build_graph(2, &[(0, 1, 1)]);
    let mut state = BlossomVState::new(&g);
    let e_idx = 0u32;
    let before_first0 = state.nodes[0].first;
    let before_first1 = state.nodes[1].first;

    state.edges[e_idx as usize].head = [0, 1];
    state.edges[e_idx as usize].slack = 12;

    state.process_expand_selfloop(e_idx);

    assert_eq!(state.edges[e_idx as usize].slack, 12);
    assert_eq!(state.nodes[0].first, before_first0);
    assert_eq!(state.nodes[1].first, before_first1);
}

#[test]
fn test_next_tree_plus_returns_none_when_match_arc_has_no_raw_head() {
    let g = build_graph(3, &[(0, 1, 1)]);
    let mut state = BlossomVState::new(&g);
    let root = 0u32;
    let current = 1u32;

    state.nodes[root as usize].flag = PLUS;
    state.nodes[root as usize].is_outer = true;
    state.nodes[root as usize].is_tree_root = true;

    state.nodes[current as usize].flag = PLUS;
    state.nodes[current as usize].is_outer = true;
    state.nodes[current as usize].tree_root = root;
    state.nodes[current as usize].match_arc = make_arc(0, 0);
    state.nodes[current as usize].tree_sibling_next = NONE;
    state.edges[0].head = [NONE, current];

    assert_eq!(state.next_tree_plus(current, root), None);
}

#[test]
fn test_next_tree_plus_returns_none_when_minus_parent_arc_has_no_outer_head() {
    let g = build_graph(4, &[(0, 1, 1), (1, 2, 1)]);
    let mut state = BlossomVState::new(&g);
    let root = 0u32;
    let current = 1u32;
    let minus = 2u32;

    state.nodes[root as usize].flag = PLUS;
    state.nodes[root as usize].is_outer = true;
    state.nodes[root as usize].is_tree_root = true;

    state.nodes[current as usize].flag = PLUS;
    state.nodes[current as usize].is_outer = true;
    state.nodes[current as usize].tree_root = root;
    state.nodes[current as usize].match_arc = make_arc(0, 0);
    state.nodes[current as usize].tree_sibling_next = NONE;

    state.nodes[minus as usize].flag = MINUS;
    state.nodes[minus as usize].is_outer = true;
    state.nodes[minus as usize].tree_parent_arc = NONE;
    state.edges[0].head = [minus, current];

    assert_eq!(state.next_tree_plus(current, root), None);
}

#[test]
fn test_expand_drains_child_selfloops_before_child_marking() {
    let g = build_graph(6, &[(0, 1, 1)]);
    let mut state = BlossomVState::new(&g);
    let b = 2u32;
    let child = 1u32;
    let left_inner = 0u32;
    let right_inner = 3u32;
    let left_outer = 4u32;
    let right_outer = 5u32;
    let e_idx = 0u32;

    for node in &mut state.nodes {
        node.first = [NONE; 2];
        node.is_outer = true;
        node.blossom_parent = NONE;
        node.blossom_selfloops = NONE;
    }

    state.nodes[b as usize].is_blossom = true;
    state.nodes[child as usize].is_blossom = true;
    state.nodes[child as usize].is_outer = false;
    state.nodes[child as usize].blossom_parent = b;
    state.nodes[child as usize].blossom_selfloops = e_idx;

    state.nodes[left_inner as usize].is_outer = false;
    state.nodes[left_inner as usize].blossom_parent = left_outer;
    state.nodes[right_inner as usize].is_outer = false;
    state.nodes[right_inner as usize].blossom_parent = right_outer;

    state.edges[e_idx as usize].head = [left_inner, right_inner];
    state.edges[e_idx as usize].next = [NONE; 2];
    state.edges[e_idx as usize].prev = [NONE; 2];

    state.expand(b);

    assert_eq!(state.nodes[child as usize].blossom_selfloops, NONE);
    assert_eq!(state.nodes[left_inner as usize].first[1], e_idx);
    assert_eq!(state.nodes[right_inner as usize].first[0], e_idx);
}

#[test]
fn test_expand_forward_branch_breaks_when_next_minus_is_unmatched() {
    let g = build_graph(7, &[(0, 1, 1), (1, 2, 1), (2, 3, 1), (3, 4, 1)]);
    let mut state = BlossomVState::new(&g);
    let tp = 0u32;
    let cur = 1u32;
    let nxt = 2u32;
    let k = 3u32;
    let b = 4u32;
    let child_plus = 5u32;
    let grandparent = 6u32;
    let b_match = make_arc(0, 0);
    let b_tp = make_arc(1, 0);
    let tp_match = make_arc(2, 0);
    let cur_sib = make_arc(3, 0);

    for node in &mut state.nodes {
        *node = Node::new_vertex();
    }
    for edge in &mut state.edges {
        edge.head = [0, 1];
        edge.head0 = [0, 1];
        edge.next = [NONE; 2];
        edge.prev = [NONE; 2];
        edge.slack = 0;
    }

    state.nodes[b as usize].is_blossom = true;
    state.nodes[b as usize].flag = MINUS;
    state.nodes[b as usize].tree_root = grandparent;
    state.nodes[b as usize].match_arc = b_match;
    state.nodes[b as usize].tree_parent_arc = b_tp;

    state.nodes[grandparent as usize].tree_eps = 5;
    state.nodes[grandparent as usize].first_tree_child = child_plus;

    state.nodes[child_plus as usize].tree_sibling_prev = child_plus;
    state.nodes[child_plus as usize].tree_sibling_next = NONE;

    for &child in &[tp, cur, nxt, k] {
        state.nodes[child as usize].is_outer = false;
        state.nodes[child as usize].blossom_parent = b;
    }

    state.nodes[tp as usize].match_arc = tp_match;
    state.nodes[tp as usize].blossom_sibling_arc = tp_match;
    state.nodes[cur as usize].blossom_sibling_arc = cur_sib;
    state.nodes[k as usize].tree_parent_arc = b_tp;

    state.edges[0].head = [child_plus, b];
    state.edges[0].head0 = [k, child_plus];
    state.edges[1].head = [grandparent, b];
    state.edges[1].head0 = [tp, grandparent];
    state.edges[2].head = [cur, tp];
    state.edges[2].head0 = [tp, cur];
    state.edges[3].head = [nxt, cur];
    state.edges[3].head0 = [cur, nxt];

    state.expand(b);

    assert_eq!(state.nodes[cur as usize].flag, PLUS);
    assert_eq!(state.nodes[nxt as usize].flag, MINUS);
    assert_eq!(state.nodes[cur as usize].tree_root, grandparent);
    assert_eq!(state.nodes[nxt as usize].tree_parent_arc, arc_rev(cur_sib));
    assert_eq!(state.nodes[grandparent as usize].first_tree_child, cur);
    assert_eq!(state.nodes[child_plus as usize].tree_sibling_prev, child_plus);
    assert_eq!(state.nodes[child_plus as usize].tree_sibling_next, NONE);
}

#[test]
fn test_expand_swaps_blossom_minus_y_with_match_edge_slack() {
    let g = build_graph(6, &[(0, 1, 1), (0, 1, 1)]);
    let mut state = BlossomVState::new(&g);
    let k = 2u32;
    let b = 4u32;
    let child_plus = 3u32;
    let root = 5u32;
    let b_match = make_arc(0, 0);

    for node in &mut state.nodes {
        *node = Node::new_vertex();
    }
    for edge in &mut state.edges {
        edge.head = [0, 1];
        edge.head0 = [0, 1];
        edge.next = [NONE; 2];
        edge.prev = [NONE; 2];
        edge.slack = 0;
    }

    state.nodes[b as usize].is_blossom = true;
    state.nodes[b as usize].flag = MINUS;
    state.nodes[b as usize].tree_root = root;
    state.nodes[b as usize].match_arc = b_match;

    state.nodes[root as usize].tree_eps = 4;

    state.nodes[k as usize].is_blossom = true;
    state.nodes[k as usize].is_outer = false;
    state.nodes[k as usize].blossom_parent = b;
    state.nodes[k as usize].y = 17;

    state.edges[0].head = [child_plus, b];
    state.edges[0].head0 = [k, child_plus];
    state.edges[0].slack = 9;

    state.expand(b);

    assert_eq!(state.edges[0].slack, 17);
    assert_eq!(state.nodes[k as usize].y, 9);
    assert!(state.nodes[k as usize].is_processed);
}

#[test]
fn test_expand_raw_plus_edges_update_free_and_plus_neighbors() {
    let g = build_graph(7, &[(0, 1, 1), (1, 2, 1), (2, 3, 1), (3, 4, 1)]);
    let mut state = BlossomVState::new(&g);
    let free = 0u32;
    let plus = 1u32;
    let k = 2u32;
    let other_plus = 3u32;
    let b = 4u32;
    let child_plus = 5u32;
    let root = 6u32;
    let b_match = make_arc(0, 0);
    let k_parent = make_arc(1, 0);

    for node in &mut state.nodes {
        *node = Node::new_vertex();
    }
    for edge in &mut state.edges {
        edge.head = [0, 1];
        edge.head0 = [0, 1];
        edge.next = [NONE; 2];
        edge.prev = [NONE; 2];
        edge.slack = 0;
    }

    state.nodes[b as usize].is_blossom = true;
    state.nodes[b as usize].flag = MINUS;
    state.nodes[b as usize].tree_root = root;
    state.nodes[b as usize].match_arc = b_match;

    state.nodes[root as usize].tree_eps = 5;

    state.nodes[k as usize].is_outer = false;
    state.nodes[k as usize].blossom_parent = b;
    state.nodes[k as usize].tree_parent_arc = k_parent;

    state.nodes[plus as usize].match_arc = NONE;
    state.nodes[other_plus as usize].flag = PLUS;

    state.edges[0].head = [child_plus, b];
    state.edges[0].head0 = [k, child_plus];
    state.edges[1].head = [plus, k];
    state.edges[1].head0 = [plus, k];
    state.edges[1].slack = 3;
    state.edges[2].head = [plus, free];
    state.edges[2].head0 = [plus, free];
    state.edges[2].slack = 7;
    state.edges[3].head = [plus, other_plus];
    state.edges[3].head0 = [plus, other_plus];
    state.edges[3].slack = 11;

    edge_list_add(&mut state.nodes, &mut state.edges, child_plus, 0, 1);
    edge_list_add(&mut state.nodes, &mut state.edges, b, 0, 0);
    edge_list_add(&mut state.nodes, &mut state.edges, plus, 1, 1);
    edge_list_add(&mut state.nodes, &mut state.edges, k, 1, 0);
    edge_list_add(&mut state.nodes, &mut state.edges, plus, 2, 1);
    edge_list_add(&mut state.nodes, &mut state.edges, free, 2, 0);
    edge_list_add(&mut state.nodes, &mut state.edges, plus, 3, 1);
    edge_list_add(&mut state.nodes, &mut state.edges, other_plus, 3, 0);

    state.expand(b);

    assert_eq!(state.edges[2].slack, 12);
    assert_eq!(state.edges[3].slack, 21);
    assert!(state.nodes[k as usize].is_processed);
    assert!(state.nodes[plus as usize].is_processed);
}

#[test]
fn test_expand_k_search_breaks_when_match_tail_has_no_child_parent() {
    let g = build_graph(7, &[(0, 1, 1)]);
    let mut state = BlossomVState::new(&g);
    let child = 0u32;
    let b = 4u32;
    let child_plus = 5u32;
    let stray = 6u32;
    let b_match = make_arc(0, 0);

    for node in &mut state.nodes {
        *node = Node::new_vertex();
    }

    state.nodes[b as usize].is_blossom = true;
    state.nodes[b as usize].flag = MINUS;
    state.nodes[b as usize].match_arc = b_match;

    state.nodes[child as usize].is_outer = false;
    state.nodes[child as usize].blossom_parent = b;

    state.edges[0].head = [child_plus, b];
    state.edges[0].head0 = [stray, child_plus];

    state.expand(b);

    assert_eq!(state.nodes[stray as usize].match_arc, b_match);
    assert!(state.nodes[child as usize].is_outer);
}

#[test]
fn test_expand_tp_search_breaks_when_tree_parent_tail_has_no_child_parent() {
    let g = build_graph(7, &[(0, 1, 1)]);
    let mut state = BlossomVState::new(&g);
    let child = 0u32;
    let b = 4u32;
    let root = 5u32;
    let stray = 6u32;
    let b_tp = make_arc(0, 0);

    for node in &mut state.nodes {
        *node = Node::new_vertex();
    }

    state.nodes[b as usize].is_blossom = true;
    state.nodes[b as usize].flag = MINUS;
    state.nodes[b as usize].tree_root = root;
    state.nodes[b as usize].tree_parent_arc = b_tp;
    state.nodes[root as usize].tree_eps = 7;

    state.nodes[child as usize].is_outer = false;
    state.nodes[child as usize].blossom_parent = b;

    state.edges[0].head = [root, b];
    state.edges[0].head0 = [stray, root];

    state.expand(b);

    assert_eq!(state.nodes[stray as usize].flag, MINUS);
    assert_eq!(state.nodes[stray as usize].tree_root, root);
    assert_eq!(state.nodes[stray as usize].tree_parent_arc, b_tp);
    assert_eq!(state.nodes[stray as usize].y, 7);
}

#[test]
fn test_expand_backward_branch_breaks_when_next_plus_is_unmatched_and_relinks_prev_sibling() {
    let g = build_graph(8, &[(0, 1, 1), (1, 2, 1), (2, 3, 1), (3, 4, 1), (4, 5, 1)]);
    let mut state = BlossomVState::new(&g);
    let tp = 0u32;
    let plus_nxt = 1u32;
    let k = 3u32;
    let b = 4u32;
    let child_plus = 5u32;
    let grandparent = 6u32;
    let cp_prev = 7u32;
    let b_match = make_arc(0, 0);
    let b_tp = make_arc(1, 0);
    let tp_match = make_arc(2, 0);
    let tp_sib = make_arc(3, 0);
    let k_sib = make_arc(4, 0);

    for node in &mut state.nodes {
        *node = Node::new_vertex();
    }

    state.nodes[b as usize].is_blossom = true;
    state.nodes[b as usize].flag = MINUS;
    state.nodes[b as usize].tree_root = grandparent;
    state.nodes[b as usize].match_arc = b_match;
    state.nodes[b as usize].tree_parent_arc = b_tp;

    state.nodes[grandparent as usize].tree_eps = 4;
    state.nodes[grandparent as usize].first_tree_child = cp_prev;

    state.nodes[cp_prev as usize].tree_sibling_next = child_plus;
    state.nodes[child_plus as usize].tree_sibling_prev = cp_prev;
    state.nodes[child_plus as usize].tree_sibling_next = NONE;

    for &child in &[tp, plus_nxt, k] {
        state.nodes[child as usize].is_outer = false;
        state.nodes[child as usize].blossom_parent = b;
    }

    state.nodes[tp as usize].match_arc = tp_match;
    state.nodes[tp as usize].blossom_sibling_arc = tp_sib;
    state.nodes[k as usize].blossom_sibling_arc = k_sib;

    state.edges[0].head = [child_plus, b];
    state.edges[0].head0 = [k, child_plus];
    state.edges[1].head = [grandparent, b];
    state.edges[1].head0 = [tp, grandparent];
    state.edges[2].head = [plus_nxt, tp];
    state.edges[2].head0 = [tp, plus_nxt];
    state.edges[3].head = [tp, plus_nxt];
    state.edges[3].head0 = [tp, plus_nxt];
    state.edges[4].head = [plus_nxt, k];
    state.edges[4].head0 = [k, plus_nxt];

    state.expand(b);

    assert_eq!(state.nodes[k as usize].flag, MINUS);
    assert_eq!(state.nodes[plus_nxt as usize].flag, PLUS);
    assert_eq!(state.nodes[plus_nxt as usize].first_tree_child, child_plus);
    assert_eq!(state.nodes[cp_prev as usize].tree_sibling_next, plus_nxt);
    assert_eq!(state.nodes[child_plus as usize].tree_sibling_prev, child_plus);
    assert_eq!(state.nodes[child_plus as usize].tree_sibling_next, NONE);
}

#[test]
fn test_expand_k_search_climbs_nested_parent_and_breaks_when_cj_sib_is_missing() {
    let g = build_graph(9, &[(0, 1, 1), (1, 2, 1), (2, 3, 1)]);
    let mut state = BlossomVState::new(&g);
    let inner = 0u32;
    let k = 1u32;
    let ci = 2u32;
    let cj = 3u32;
    let b = 6u32;
    let child_plus = 7u32;
    let b_match = make_arc(0, 0);
    let k_sib = make_arc(1, 0);
    let ci_sib = make_arc(2, 0);

    for node in &mut state.nodes {
        *node = Node::new_vertex();
    }

    state.nodes[b as usize].is_blossom = true;
    state.nodes[b as usize].flag = MINUS;
    state.nodes[b as usize].match_arc = b_match;

    state.nodes[inner as usize].is_outer = false;
    state.nodes[inner as usize].blossom_parent = k;
    for &child in &[k, ci, cj] {
        state.nodes[child as usize].is_outer = false;
        state.nodes[child as usize].blossom_parent = b;
    }

    state.nodes[k as usize].blossom_sibling_arc = k_sib;
    state.nodes[ci as usize].blossom_sibling_arc = ci_sib;

    state.edges[0].head = [child_plus, b];
    state.edges[0].head0 = [inner, child_plus];
    state.edges[1].head = [ci, k];
    state.edges[1].head0 = [k, ci];
    state.edges[2].head = [cj, ci];
    state.edges[2].head0 = [ci, cj];

    state.expand(b);

    assert_eq!(state.nodes[k as usize].match_arc, b_match);
    assert_eq!(state.nodes[ci as usize].match_arc, ci_sib);
    assert_eq!(state.nodes[cj as usize].match_arc, arc_rev(ci_sib));
    assert!(state.nodes[k as usize].is_outer);
}

#[test]
fn test_expand_tp_search_climbs_nested_parent_and_builds_multi_step_forward_chain() {
    let g = build_graph(12, &[(0, 1, 1), (1, 2, 1), (2, 3, 1), (3, 4, 1), (4, 5, 1), (5, 6, 1)]);
    let mut state = BlossomVState::new(&g);
    let tp_inner = 0u32;
    let tp = 1u32;
    let plus1 = 2u32;
    let minus1 = 3u32;
    let plus2 = 4u32;
    let k = 5u32;
    let child_plus = 6u32;
    let root = 7u32;
    let b = 8u32;
    let b_match = make_arc(0, 0);
    let b_tp = make_arc(1, 0);
    let tp_match = make_arc(2, 0);
    let plus1_sib = make_arc(3, 0);
    let minus1_match = make_arc(4, 0);
    let plus2_sib = make_arc(5, 0);

    for node in &mut state.nodes {
        *node = Node::new_vertex();
    }

    state.nodes[b as usize].is_blossom = true;
    state.nodes[b as usize].flag = MINUS;
    state.nodes[b as usize].tree_root = root;
    state.nodes[b as usize].match_arc = b_match;
    state.nodes[b as usize].tree_parent_arc = b_tp;
    state.nodes[root as usize].tree_eps = 4;
    state.nodes[root as usize].first_tree_child = child_plus;
    state.nodes[child_plus as usize].tree_sibling_prev = child_plus;
    state.nodes[child_plus as usize].tree_sibling_next = NONE;

    state.nodes[tp_inner as usize].is_outer = false;
    state.nodes[tp_inner as usize].blossom_parent = tp;
    for &child in &[tp, plus1, minus1, plus2, k] {
        state.nodes[child as usize].is_outer = false;
        state.nodes[child as usize].blossom_parent = b;
    }

    state.nodes[tp as usize].match_arc = tp_match;
    state.nodes[tp as usize].blossom_sibling_arc = tp_match;
    state.nodes[plus1 as usize].blossom_sibling_arc = plus1_sib;
    state.nodes[minus1 as usize].match_arc = minus1_match;
    state.nodes[plus2 as usize].blossom_sibling_arc = plus2_sib;
    state.nodes[k as usize].tree_parent_arc = b_tp;

    state.edges[0].head = [child_plus, b];
    state.edges[0].head0 = [k, child_plus];
    state.edges[1].head = [root, b];
    state.edges[1].head0 = [tp_inner, root];
    state.edges[2].head = [plus1, tp];
    state.edges[2].head0 = [tp, plus1];
    state.edges[3].head = [minus1, plus1];
    state.edges[3].head0 = [plus1, minus1];
    state.edges[4].head = [plus2, minus1];
    state.edges[4].head0 = [minus1, plus2];
    state.edges[5].head = [k, plus2];
    state.edges[5].head0 = [plus2, k];

    state.expand(b);

    assert_eq!(state.nodes[tp as usize].flag, MINUS);
    assert_eq!(state.nodes[plus1 as usize].flag, PLUS);
    assert_eq!(state.nodes[plus2 as usize].flag, PLUS);
    assert_eq!(state.nodes[plus1 as usize].first_tree_child, plus2);
    assert_eq!(state.nodes[plus2 as usize].first_tree_child, child_plus);
    assert_eq!(state.nodes[root as usize].first_tree_child, plus1);
}

#[test]
fn test_expand_late_lazy_dual_pass_skips_detached_plus_neighbors() {
    let g = build_graph(8, &[(0, 1, 1), (1, 2, 1), (2, 3, 1), (3, 4, 1), (4, 5, 1), (5, 6, 1)]);
    let mut state = BlossomVState::new(&g);
    let free = 0u32;
    let plus = 1u32;
    let k = 2u32;
    let other_plus = 3u32;
    let b = 4u32;
    let child_plus = 5u32;
    let root = 6u32;
    let b_match = make_arc(0, 0);
    let k_parent = make_arc(1, 0);
    let listed_detached = 4u32;
    let raw_detached = 5u32;

    for node in &mut state.nodes {
        *node = Node::new_vertex();
    }
    for edge in &mut state.edges {
        edge.head = [0, 1];
        edge.head0 = [0, 1];
        edge.next = [NONE; 2];
        edge.prev = [NONE; 2];
        edge.slack = 0;
    }

    state.nodes[b as usize].is_blossom = true;
    state.nodes[b as usize].flag = MINUS;
    state.nodes[b as usize].tree_root = root;
    state.nodes[b as usize].match_arc = b_match;
    state.nodes[root as usize].tree_eps = 5;

    state.nodes[k as usize].is_outer = false;
    state.nodes[k as usize].blossom_parent = b;
    state.nodes[k as usize].tree_parent_arc = k_parent;
    state.nodes[plus as usize].match_arc = NONE;
    state.nodes[other_plus as usize].flag = PLUS;

    state.edges[0].head = [child_plus, b];
    state.edges[0].head0 = [k, child_plus];
    state.edges[1].head = [plus, k];
    state.edges[1].head0 = [plus, k];
    state.edges[1].slack = 3;
    state.edges[2].head = [plus, free];
    state.edges[2].head0 = [plus, free];
    state.edges[2].slack = 7;
    state.edges[3].head = [plus, other_plus];
    state.edges[3].head0 = [plus, other_plus];
    state.edges[3].slack = 11;

    edge_list_add(&mut state.nodes, &mut state.edges, plus, listed_detached, 0);
    let listed_dir = state
        .incident_edges(plus)
        .into_iter()
        .find(|(e_idx, _)| *e_idx == listed_detached)
        .map(|(_, dir)| dir)
        .expect("listed detached edge should be reachable from plus");
    state.edges[listed_detached as usize].head = [plus, plus];
    state.edges[listed_detached as usize].head[listed_dir] = NONE;
    state.edges[listed_detached as usize].head[1 - listed_dir] = plus;
    state.edges[listed_detached as usize].head0 = state.edges[listed_detached as usize].head;
    state.edges[listed_detached as usize].slack = 19;

    state.edges[raw_detached as usize].head = [plus, NONE];
    state.edges[raw_detached as usize].head0 = [plus, NONE];
    state.edges[raw_detached as usize].slack = 23;

    state.expand(b);

    assert_eq!(state.edges[listed_detached as usize].slack, 19);
    assert_eq!(state.edges[raw_detached as usize].slack, 23);
    assert!(state.nodes[k as usize].is_processed);
    assert!(state.nodes[plus as usize].is_processed);
}

#[test]
fn test_shrink_restores_partner_blossom_match_slack_and_moves_match_edge_to_new_blossom() {
    let g = build_graph(
        9,
        &[
            (0, 1, 1),
            (0, 2, 1),
            (0, 3, 1),
            (0, 4, 1),
            (0, 5, 1),
            (0, 6, 1),
            (0, 7, 1),
            (0, 8, 1),
            (1, 8, 1),
        ],
    );
    let mut state = BlossomVState::new(&g);

    for node in &mut state.nodes {
        *node = Node::new_vertex();
    }
    for edge in &mut state.edges {
        edge.head = [0, 1];
        edge.head0 = [0, 1];
        edge.next = [NONE; 2];
        edge.prev = [NONE; 2];
        edge.slack = 0;
    }

    let root = 5u32;
    let lca = 0u32;
    let ep0 = 1u32;
    let ep1 = 2u32;
    let minus0 = 3u32;
    let minus1 = 4u32;
    let partner = 6u32;
    let shrink_edge = 0u32;
    let b_match_edge = 3u32;
    let new_blossom = state.node_num as u32;

    for &plus in &[root, lca, ep0, ep1] {
        state.nodes[plus as usize].flag = PLUS;
        state.nodes[plus as usize].is_outer = true;
        state.nodes[plus as usize].tree_root = root;
    }
    state.nodes[root as usize].is_tree_root = true;
    state.nodes[root as usize].tree_eps = 7;
    state.nodes[root as usize].first_tree_child = lca;

    for &minus in &[minus0, minus1, partner] {
        state.nodes[minus as usize].flag = MINUS;
        state.nodes[minus as usize].is_outer = true;
        state.nodes[minus as usize].tree_root = root;
    }
    state.nodes[partner as usize].is_blossom = true;
    state.nodes[partner as usize].y = 21;

    state.nodes[lca as usize].match_arc = make_arc(b_match_edge, 0);
    state.nodes[ep0 as usize].match_arc = make_arc(1, 1);
    state.nodes[ep1 as usize].match_arc = make_arc(5, 1);
    state.nodes[minus0 as usize].tree_parent_arc = make_arc(2, 1);
    state.nodes[minus1 as usize].tree_parent_arc = make_arc(6, 1);
    state.nodes[partner as usize].tree_parent_arc = make_arc(4, 1);

    edge_list_add(&mut state.nodes, &mut state.edges, ep0, 1, 1);
    edge_list_add(&mut state.nodes, &mut state.edges, minus0, 1, 0);
    edge_list_add(&mut state.nodes, &mut state.edges, lca, 2, 0);
    edge_list_add(&mut state.nodes, &mut state.edges, minus0, 2, 1);
    edge_list_add(&mut state.nodes, &mut state.edges, lca, b_match_edge, 0);
    edge_list_add(&mut state.nodes, &mut state.edges, partner, b_match_edge, 1);
    edge_list_add(&mut state.nodes, &mut state.edges, root, 4, 0);
    edge_list_add(&mut state.nodes, &mut state.edges, partner, 4, 1);
    edge_list_add(&mut state.nodes, &mut state.edges, ep1, 5, 1);
    edge_list_add(&mut state.nodes, &mut state.edges, minus1, 5, 0);
    edge_list_add(&mut state.nodes, &mut state.edges, lca, 6, 0);
    edge_list_add(&mut state.nodes, &mut state.edges, minus1, 6, 1);

    state.edges[shrink_edge as usize].head = [ep0, ep1];
    state.edges[shrink_edge as usize].head0 = [ep0, ep1];
    state.edges[1].head0 = [ep0, minus0];
    state.edges[2].head0 = [minus0, lca];
    state.edges[b_match_edge as usize].head0 = [lca, partner];
    state.edges[4].head0 = [partner, root];
    state.edges[5].head0 = [ep1, minus1];
    state.edges[6].head0 = [minus1, lca];
    state.edges[b_match_edge as usize].slack = 13;

    state.shrink(shrink_edge, ep0, ep1);

    assert_eq!(state.edges[b_match_edge as usize].slack, 13);
    assert_eq!(state.nodes[partner as usize].y, 21);
    assert_eq!(state.edges[b_match_edge as usize].head[1], new_blossom);
    assert_eq!(state.nodes[new_blossom as usize].match_arc, make_arc(b_match_edge, 0));
}

#[test]
fn test_shrink_second_pass_relinks_inner_edge_and_promotes_boundary_edge() {
    let g = build_graph(
        10,
        &[
            (0, 1, 1),
            (0, 2, 1),
            (0, 3, 1),
            (0, 4, 1),
            (0, 5, 1),
            (0, 6, 1),
            (0, 7, 1),
            (0, 8, 1),
            (0, 9, 1),
        ],
    );
    let mut state = BlossomVState::new(&g);

    for node in &mut state.nodes {
        *node = Node::new_vertex();
    }
    for edge in &mut state.edges {
        edge.head = [0, 1];
        edge.head0 = [0, 1];
        edge.next = [NONE; 2];
        edge.prev = [NONE; 2];
        edge.slack = 0;
    }

    let lca = 0u32;
    let ep0 = 1u32;
    let ep1 = 2u32;
    let minus0 = 3u32;
    let minus1 = 4u32;
    let free = 7u32;
    let hidden = 8u32;
    let shrink_edge = 0u32;
    let boundary_edge = 7u32;
    let hidden_edge = 8u32;
    let new_blossom = state.node_num as u32;

    for &plus in &[lca, ep0, ep1] {
        state.nodes[plus as usize].flag = PLUS;
        state.nodes[plus as usize].is_outer = true;
        state.nodes[plus as usize].tree_root = lca;
    }
    state.nodes[lca as usize].is_tree_root = true;
    state.nodes[lca as usize].tree_eps = 7;

    for &minus in &[minus0, minus1] {
        state.nodes[minus as usize].flag = MINUS;
        state.nodes[minus as usize].is_outer = true;
        state.nodes[minus as usize].tree_root = lca;
    }

    state.nodes[hidden as usize].is_outer = false;
    state.nodes[hidden as usize].blossom_parent = minus1;
    state.nodes[hidden as usize].flag = FREE;

    state.nodes[ep0 as usize].match_arc = make_arc(1, 1);
    state.nodes[ep1 as usize].match_arc = make_arc(5, 1);
    state.nodes[minus0 as usize].tree_parent_arc = make_arc(2, 1);
    state.nodes[minus1 as usize].tree_parent_arc = make_arc(6, 1);

    edge_list_add(&mut state.nodes, &mut state.edges, ep0, 1, 1);
    edge_list_add(&mut state.nodes, &mut state.edges, minus0, 1, 0);
    edge_list_add(&mut state.nodes, &mut state.edges, lca, 2, 0);
    edge_list_add(&mut state.nodes, &mut state.edges, minus0, 2, 1);
    edge_list_add(&mut state.nodes, &mut state.edges, ep1, 5, 1);
    edge_list_add(&mut state.nodes, &mut state.edges, minus1, 5, 0);
    edge_list_add(&mut state.nodes, &mut state.edges, lca, 6, 0);
    edge_list_add(&mut state.nodes, &mut state.edges, minus1, 6, 1);

    edge_list_add(&mut state.nodes, &mut state.edges, free, boundary_edge, 1);
    edge_list_add(&mut state.nodes, &mut state.edges, minus0, boundary_edge, 0);
    edge_list_add(&mut state.nodes, &mut state.edges, hidden, hidden_edge, 1);
    edge_list_add(&mut state.nodes, &mut state.edges, minus0, hidden_edge, 0);

    state.edges[shrink_edge as usize].head = [ep0, ep1];
    state.edges[shrink_edge as usize].head0 = [ep0, ep1];
    state.edges[1].head0 = [ep0, minus0];
    state.edges[2].head0 = [minus0, lca];
    state.edges[5].head0 = [ep1, minus1];
    state.edges[6].head0 = [minus1, lca];
    state.edges[boundary_edge as usize].head0 = [free, minus0];
    state.edges[boundary_edge as usize].slack = 5;
    state.edges[hidden_edge as usize].head0 = [hidden, minus0];
    state.edges[hidden_edge as usize].slack = 11;

    state.shrink(shrink_edge, ep0, ep1);

    assert_eq!(state.edges[boundary_edge as usize].head[1], new_blossom);
    assert_eq!(state.edge_queue_owner(boundary_edge), GenericQueueState::Pq0 { root: new_blossom });
    assert_eq!(state.edges[boundary_edge as usize].slack, 19);

    assert_eq!(state.edges[hidden_edge as usize].head[0], minus1);
    assert_eq!(state.edges[hidden_edge as usize].slack, 25);
}

#[test]
fn test_rebuild_outer_blossom_queue_membership_preserves_existing_pq0_stamp_order() {
    let g = build_graph(3, &[(0, 2, 1), (1, 2, 1)]);
    let mut state = BlossomVState::new(&g);

    for node in &mut state.nodes {
        *node = Node::new_vertex();
        node.is_outer = true;
    }

    let blossom = 2u32;
    state.nodes[blossom as usize].is_blossom = true;
    state.nodes[blossom as usize].flag = PLUS;
    state.nodes[blossom as usize].is_tree_root = true;
    state.nodes[blossom as usize].tree_root = blossom;
    state.nodes[blossom as usize].tree_eps = 5;

    state.nodes[0].flag = FREE;
    state.nodes[1].flag = FREE;

    state.edges[0].head = [0, blossom];
    state.edges[0].head0 = [0, blossom];
    state.edges[0].slack = 7;
    state.edges[1].head = [1, blossom];
    state.edges[1].head0 = [1, blossom];
    state.edges[1].slack = 7;

    state.generic_queue_epoch = 9;
    state.set_generic_pq0(1, blossom);
    state.set_generic_pq0(0, blossom);

    assert_eq!(state.scheduler_tree_best_pq0_edge(blossom), Some(0));

    state.rebuild_generic_queue_membership_for_outer_blossom(blossom);

    assert_eq!(state.edge_queue_stamp(0), 11);
    assert_eq!(state.edge_queue_stamp(1), 10);
    assert_eq!(state.scheduler_tree_best_pq0_edge(blossom), Some(0));
}

#[test]
fn test_scheduler_tree_best_pq_blossom_edge_preserves_existing_stamp_order() {
    let g = build_graph(3, &[(0, 2, 1), (1, 2, 1)]);
    let mut state = BlossomVState::new(&g);

    for node in &mut state.nodes {
        *node = Node::new_vertex();
        node.is_outer = true;
    }

    let root = 2u32;
    state.nodes[root as usize].is_blossom = true;
    state.nodes[root as usize].flag = MINUS;
    state.nodes[root as usize].is_tree_root = true;
    state.nodes[root as usize].tree_root = root;
    state.nodes[root as usize].is_processed = true;
    state.nodes[root as usize].tree_eps = 5;

    state.nodes[0].flag = PLUS;
    state.nodes[0].tree_root = root;
    state.nodes[0].is_processed = true;
    state.nodes[0].match_arc = make_arc(0, 0);

    state.nodes[1].flag = PLUS;
    state.nodes[1].tree_root = root;
    state.nodes[1].is_processed = true;
    state.nodes[1].match_arc = make_arc(1, 0);

    state.edges[0].head = [root, 0];
    state.edges[0].head0 = [root, 0];
    state.edges[0].slack = 7;
    state.edges[1].head = [root, 1];
    state.edges[1].head0 = [root, 1];
    state.edges[1].slack = 7;

    state.generic_queue_epoch = 9;
    state.set_generic_pq_blossoms_root_slot(1, root, false);
    state.set_generic_pq_blossoms_root_slot(0, root, false);

    assert_eq!(state.scheduler_tree_best_pq_blossom_edge(root), Some(0));
}

#[test]
fn test_scheduler_tree_edge_best_pq00_edge_preserves_existing_stamp_order() {
    let g = build_graph(4, &[(0, 3, 1), (1, 2, 1)]);
    let mut state = BlossomVState::new(&g);

    for node in &mut state.nodes {
        *node = Node::new_vertex();
        node.is_outer = true;
    }

    let left_root = 0u32;
    let right_root = 3u32;
    state.nodes[left_root as usize].flag = PLUS;
    state.nodes[left_root as usize].is_tree_root = true;
    state.nodes[left_root as usize].tree_root = left_root;
    state.nodes[left_root as usize].is_processed = true;

    state.nodes[right_root as usize].flag = PLUS;
    state.nodes[right_root as usize].is_tree_root = true;
    state.nodes[right_root as usize].tree_root = right_root;
    state.nodes[right_root as usize].is_processed = true;

    state.edges[0].head = [left_root, right_root];
    state.edges[0].head0 = [left_root, right_root];
    state.edges[0].slack = 9;
    state.edges[1].head = [left_root, right_root];
    state.edges[1].head0 = [left_root, right_root];
    state.edges[1].slack = 9;

    state.generic_queue_epoch = 9;
    state.set_generic_pq00(1, left_root, right_root);
    state.set_generic_pq00(0, left_root, right_root);

    let GenericQueueState::Pq00Pair { pair_idx } = state.edge_queue_owner(0) else {
        panic!("edge 0 should be in a pq00 pair queue");
    };

    assert_eq!(state.scheduler_tree_edge_best_pq00_edge(pair_idx, left_root, right_root), Some(0));
}

#[test]
fn test_shrink_branch_switch_breaks_when_reverse_shrink_edge_returns_to_lca() {
    let g = build_graph(4, &[(0, 1, 1), (0, 2, 1), (0, 3, 1)]);
    let mut state = BlossomVState::new(&g);
    let root = 0u32;
    let endpoint0 = 1u32;
    let minus0 = 2u32;
    let shrink_edge = 0u32;
    let match_edge = 1u32;
    let parent_edge = 2u32;
    let new_blossom = state.node_num as u32;

    for node in &mut state.nodes {
        *node = Node::new_vertex();
    }
    for edge in &mut state.edges {
        edge.head = [0, 1];
        edge.head0 = [0, 1];
        edge.next = [NONE; 2];
        edge.prev = [NONE; 2];
        edge.slack = 0;
    }

    state.nodes[root as usize].flag = PLUS;
    state.nodes[root as usize].is_outer = true;
    state.nodes[root as usize].is_tree_root = true;
    state.nodes[root as usize].tree_root = root;
    state.nodes[root as usize].tree_eps = 3;
    state.nodes[root as usize].first_tree_child = endpoint0;

    state.nodes[endpoint0 as usize].flag = PLUS;
    state.nodes[endpoint0 as usize].is_outer = true;
    state.nodes[endpoint0 as usize].tree_root = root;
    state.nodes[endpoint0 as usize].match_arc = make_arc(match_edge, 0);
    state.nodes[endpoint0 as usize].tree_sibling_prev = endpoint0;
    state.nodes[endpoint0 as usize].tree_sibling_next = NONE;

    state.nodes[minus0 as usize].flag = MINUS;
    state.nodes[minus0 as usize].is_outer = true;
    state.nodes[minus0 as usize].tree_root = root;
    state.nodes[minus0 as usize].tree_parent_arc = make_arc(parent_edge, 0);

    state.edges[shrink_edge as usize].head = [endpoint0, root];
    state.edges[shrink_edge as usize].head0 = [endpoint0, root];
    state.edges[match_edge as usize].head = [minus0, endpoint0];
    state.edges[match_edge as usize].head0 = [endpoint0, minus0];
    state.edges[parent_edge as usize].head = [root, minus0];
    state.edges[parent_edge as usize].head0 = [minus0, root];

    state.shrink(shrink_edge, endpoint0, root);

    assert!(state.nodes[new_blossom as usize].is_tree_root);
    assert_eq!(state.nodes[new_blossom as usize].tree_root, new_blossom);
    assert_eq!(state.nodes[root as usize].blossom_parent, new_blossom);
    assert_eq!(state.nodes[endpoint0 as usize].blossom_parent, new_blossom);
}

#[test]
fn test_shrink_branch_one_breaks_when_next_arc_returns_to_lca() {
    let g = build_graph(4, &[(0, 1, 1), (0, 2, 1), (0, 3, 1)]);
    let mut state = BlossomVState::new(&g);
    let root = 0u32;
    let endpoint1 = 1u32;
    let minus1 = 2u32;
    let shrink_edge = 0u32;
    let match_edge = 1u32;
    let parent_edge = 2u32;
    let new_blossom = state.node_num as u32;

    for node in &mut state.nodes {
        *node = Node::new_vertex();
    }
    for edge in &mut state.edges {
        edge.head = [0, 1];
        edge.head0 = [0, 1];
        edge.next = [NONE; 2];
        edge.prev = [NONE; 2];
        edge.slack = 0;
    }

    state.nodes[root as usize].flag = PLUS;
    state.nodes[root as usize].is_outer = true;
    state.nodes[root as usize].is_tree_root = true;
    state.nodes[root as usize].tree_root = root;
    state.nodes[root as usize].tree_eps = 4;
    state.nodes[root as usize].first_tree_child = endpoint1;

    state.nodes[endpoint1 as usize].flag = PLUS;
    state.nodes[endpoint1 as usize].is_outer = true;
    state.nodes[endpoint1 as usize].tree_root = root;
    state.nodes[endpoint1 as usize].match_arc = make_arc(match_edge, 0);
    state.nodes[endpoint1 as usize].tree_sibling_prev = endpoint1;
    state.nodes[endpoint1 as usize].tree_sibling_next = NONE;

    state.nodes[minus1 as usize].flag = MINUS;
    state.nodes[minus1 as usize].is_outer = true;
    state.nodes[minus1 as usize].tree_root = root;
    state.nodes[minus1 as usize].tree_parent_arc = make_arc(parent_edge, 0);

    state.edges[shrink_edge as usize].head = [root, endpoint1];
    state.edges[shrink_edge as usize].head0 = [root, endpoint1];
    state.edges[match_edge as usize].head = [minus1, endpoint1];
    state.edges[match_edge as usize].head0 = [endpoint1, minus1];
    state.edges[parent_edge as usize].head = [root, minus1];
    state.edges[parent_edge as usize].head0 = [minus1, root];

    state.shrink(shrink_edge, root, endpoint1);

    assert!(state.nodes[new_blossom as usize].is_tree_root);
    assert_eq!(state.nodes[new_blossom as usize].tree_root, new_blossom);
    assert_eq!(state.nodes[endpoint1 as usize].blossom_parent, new_blossom);
    assert_eq!(state.nodes[minus1 as usize].blossom_parent, new_blossom);
}

#[test]
fn test_find_global_augment_edge_uses_scheduler_pair_with_adjusted_other_eps() {
    let g = build_graph(4, &[(0, 1, 1)]);
    let mut state = BlossomVState::new(&g);
    let e_idx = 0u32;

    for node in &mut state.nodes {
        node.first = [NONE; 2];
        node.flag = FREE;
        node.is_tree_root = false;
        node.is_processed = false;
        node.tree_root = NONE;
        node.tree_eps = 0;
        node.blossom_parent = NONE;
        node.is_outer = true;
    }

    state.nodes[0].flag = PLUS;
    state.nodes[0].is_tree_root = true;
    state.nodes[0].is_processed = true;
    state.nodes[0].tree_root = 0;
    state.nodes[0].tree_eps = 4;

    state.nodes[1].flag = PLUS;
    state.nodes[1].is_tree_root = true;
    state.nodes[1].is_processed = true;
    state.nodes[1].tree_root = 1;
    state.nodes[1].tree_eps = 2;

    state.root_list_head = 0;
    state.nodes[0].tree_sibling_next = 1;
    state.nodes[1].tree_sibling_next = NONE;

    state.edges[e_idx as usize].head = [0, 1];
    state.edges[e_idx as usize].head0 = [0, 1];
    state.edges[e_idx as usize].slack = 5;
    state.set_generic_pq00(e_idx, 0, 1);

    assert_eq!(state.find_scheduler_global_augment_edge(), Some((e_idx, 0, 1)));
}

#[test]
fn test_prepare_tree_for_augment_requeues_pair_edge_into_other_roots_pq0() {
    let g = build_graph(4, &[(1, 2, 1)]);
    let mut state = BlossomVState::new(&g);
    let e_idx = 0u32;

    for node in &mut state.nodes {
        node.flag = FREE;
        node.is_tree_root = false;
        node.is_processed = false;
        node.tree_root = NONE;
        node.tree_eps = 0;
        node.first_tree_child = NONE;
        node.is_outer = true;
    }

    state.nodes[0].flag = PLUS;
    state.nodes[0].is_tree_root = true;
    state.nodes[0].tree_root = 0;
    state.nodes[0].tree_eps = 4;

    state.nodes[3].flag = PLUS;
    state.nodes[3].is_tree_root = true;
    state.nodes[3].tree_root = 3;

    state.nodes[1].flag = PLUS;
    state.nodes[1].tree_root = 3;
    state.nodes[2].flag = PLUS;
    state.nodes[2].tree_root = 3;

    state.edges[e_idx as usize].head = [1, 2];
    state.set_generic_pq00(e_idx, 0, 3);
    state.generic_pairs[0].pq00.clear();

    state.prepare_tree_for_augment(0, &[0]);

    assert_eq!(state.edge_queue_owner(e_idx), GenericQueueState::Pq0 { root: 3 });
    assert!(state.generic_trees[3].pq0.contains(&e_idx));
    assert!(state.scheduler_trees[3].pq0.contains(&e_idx));
}

#[test]
fn test_prepare_tree_for_augment_uses_scheduler_root_queues_when_shadow_is_stale() {
    let g = build_graph(4, &[(0, 1, 5)]);
    let mut state = BlossomVState::new(&g);
    let e_idx = 0u32;

    for node in &mut state.nodes {
        node.flag = FREE;
        node.is_tree_root = false;
        node.is_processed = false;
        node.tree_root = NONE;
        node.tree_eps = 0;
        node.first_tree_child = NONE;
        node.is_outer = true;
    }

    state.nodes[0].flag = PLUS;
    state.nodes[0].is_tree_root = true;
    state.nodes[0].tree_root = 0;
    state.nodes[0].tree_eps = 4;

    state.nodes[1].flag = FREE;
    state.nodes[1].is_outer = true;
    state.edges[e_idx as usize].head = [0, 1];
    state.edges[e_idx as usize].head0 = [0, 1];
    state.edges[e_idx as usize].slack = 7;

    state.set_generic_pq0(e_idx, 0);
    state.generic_trees[0].pq0.clear();

    state.prepare_tree_for_augment(0, &[0]);

    assert_eq!(state.edge_queue_owner(e_idx), GenericQueueState::None);
    assert!(state.scheduler_trees[0].pq0.is_empty());
    assert!(state.generic_trees[0].pq0.is_empty());
    assert_eq!(state.edges[e_idx as usize].slack, 3);
}

#[test]
fn test_prepare_tree_for_augment_switches_neighbor_current_from_root_to_pair() {
    let g = build_graph(4, &[(1, 2, 1)]);
    let mut state = BlossomVState::new(&g);
    let e_idx = 0u32;

    for node in &mut state.nodes {
        node.flag = FREE;
        node.is_tree_root = false;
        node.is_processed = false;
        node.tree_root = NONE;
        node.tree_eps = 0;
        node.first_tree_child = NONE;
        node.is_outer = true;
    }

    state.nodes[0].flag = PLUS;
    state.nodes[0].is_tree_root = true;
    state.nodes[0].tree_root = 0;
    state.nodes[0].tree_eps = 4;

    state.nodes[3].flag = PLUS;
    state.nodes[3].is_tree_root = true;
    state.nodes[3].tree_root = 3;
    state.nodes[3].tree_eps = 2;

    state.nodes[1].flag = PLUS;
    state.nodes[1].tree_root = 3;
    state.nodes[2].flag = PLUS;
    state.nodes[2].tree_root = 3;

    state.edges[e_idx as usize].head = [1, 2];
    state.set_generic_pq00(e_idx, 0, 3);

    state.scheduler_trees[3].current = SchedulerCurrent::Pair { pair_idx: 17, dir: 1 };

    state.prepare_tree_for_augment(0, &[0]);

    assert_eq!(state.scheduler_trees[3].current, SchedulerCurrent::None);
}

#[test]
fn test_queue_processed_plus_blossom_match_edge_promotes_owned_edge_into_pq_blossoms() {
    let g = build_graph(2, &[(0, 1, 5)]);
    let mut state = BlossomVState::new(&g);
    let plus = 1u32;
    let root = 0u32;
    let blossom = state.nodes.len() as u32;
    let match_edge = 0u32;

    state.nodes[root as usize].flag = PLUS;
    state.nodes[root as usize].is_outer = true;
    state.nodes[root as usize].is_tree_root = true;
    state.nodes[root as usize].is_processed = true;
    state.nodes[root as usize].tree_root = root;

    state.nodes[plus as usize].flag = PLUS;
    state.nodes[plus as usize].is_outer = true;
    state.nodes[plus as usize].is_tree_root = false;
    state.nodes[plus as usize].is_processed = true;
    state.nodes[plus as usize].tree_root = root;
    state.nodes[plus as usize].match_arc = make_arc(match_edge, 0);

    let mut blossom_node = Node::new_vertex();
    blossom_node.is_blossom = true;
    blossom_node.is_outer = true;
    blossom_node.flag = MINUS;
    blossom_node.is_processed = true;
    blossom_node.tree_root = root;
    blossom_node.match_arc = make_arc(match_edge, 1);
    state.nodes.push(blossom_node);

    state.edges[match_edge as usize].head = [blossom, plus];
    state.edges[match_edge as usize].head0 = [blossom, plus];

    state.set_generic_pq0(match_edge, root);

    state.queue_processed_plus_blossom_match_edge(plus);

    assert_eq!(state.edge_queue_owner(match_edge), GenericQueueState::PqBlossoms { root });
    assert!(state.scheduler_trees[root as usize].pq_blossoms.contains(&match_edge));
}

#[test]
fn test_queue_processed_root_plus_blossom_match_edge_promotes_owned_edge_into_pq_blossoms() {
    let g = build_graph(2, &[(0, 1, 5)]);
    let mut state = BlossomVState::new(&g);
    let root = 0u32;
    let blossom = state.nodes.len() as u32;
    let match_edge = 0u32;

    state.nodes[root as usize].flag = PLUS;
    state.nodes[root as usize].is_outer = true;
    state.nodes[root as usize].is_tree_root = true;
    state.nodes[root as usize].is_processed = true;
    state.nodes[root as usize].tree_root = root;
    state.nodes[root as usize].match_arc = make_arc(match_edge, 0);

    let mut blossom_node = Node::new_vertex();
    blossom_node.is_blossom = true;
    blossom_node.is_outer = true;
    blossom_node.flag = MINUS;
    blossom_node.is_processed = true;
    blossom_node.tree_root = root;
    blossom_node.match_arc = make_arc(match_edge, 1);
    state.nodes.push(blossom_node);

    state.edges[match_edge as usize].head = [blossom, root];
    state.edges[match_edge as usize].head0 = [blossom, root];

    state.queue_processed_plus_blossom_match_edge(root);

    assert_eq!(state.edge_queue_owner(match_edge), GenericQueueState::PqBlossoms { root });
    assert!(state.scheduler_trees[root as usize].pq_blossoms.contains(&match_edge));
}

#[test]
fn test_rebuild_outer_blossom_queue_membership_requeues_match_edge_into_pq_blossoms() {
    let g = build_graph(2, &[(0, 1, 5)]);
    let mut state = BlossomVState::new(&g);
    let plus = 1u32;
    let root = 0u32;
    let blossom = state.nodes.len() as u32;
    let match_edge = 0u32;

    state.nodes[root as usize].flag = PLUS;
    state.nodes[root as usize].is_outer = true;
    state.nodes[root as usize].is_tree_root = true;
    state.nodes[root as usize].is_processed = true;
    state.nodes[root as usize].tree_root = root;

    state.nodes[plus as usize].flag = PLUS;
    state.nodes[plus as usize].is_outer = true;
    state.nodes[plus as usize].is_tree_root = false;
    state.nodes[plus as usize].is_processed = true;
    state.nodes[plus as usize].tree_root = root;

    let mut blossom_node = Node::new_vertex();
    blossom_node.is_blossom = true;
    blossom_node.is_outer = true;
    blossom_node.flag = MINUS;
    blossom_node.is_processed = true;
    blossom_node.tree_root = root;
    blossom_node.match_arc = make_arc(match_edge, 1);
    state.nodes.push(blossom_node);

    edge_list_remove(&mut state.nodes, &mut state.edges, root, match_edge, 1);
    edge_list_add(&mut state.nodes, &mut state.edges, blossom, match_edge, 1);
    state.edges[match_edge as usize].head = [blossom, plus];
    state.edges[match_edge as usize].head0 = [blossom, plus];

    state.rebuild_generic_queue_membership_for_outer_blossom(blossom);

    assert_eq!(state.edge_queue_owner(match_edge), GenericQueueState::PqBlossoms { root });
    assert!(state.scheduler_trees[root as usize].pq_blossoms.contains(&match_edge));
}

#[test]
fn test_promote_boundary_edges_to_outer_blossom_moves_raw_cycle_boundary_edge() {
    let g = build_graph(3, &[(0, 2, 5)]);
    let mut state = BlossomVState::new(&g);
    let blossom = state.nodes.len() as u32;
    state.nodes.push(Node::new_vertex());

    state.nodes[blossom as usize].is_outer = true;
    state.nodes[blossom as usize].is_blossom = true;
    state.nodes[blossom as usize].flag = PLUS;

    state.nodes[0].is_outer = false;
    state.nodes[0].blossom_parent = blossom;
    state.nodes[0].blossom_grandparent = blossom;
    state.nodes[0].flag = PLUS;

    let edge = find_edge_idx(&state, 0, 2);
    let raw_dir = if state.edges[edge as usize].head[0] == 0 { 0 } else { 1 };
    assert_eq!(state.edges[edge as usize].head[raw_dir], 0);
    assert_eq!(state.edges[edge as usize].head[1 - raw_dir], 2);
    assert_eq!(state.nodes[0].first[1 - raw_dir], edge);

    state.promote_boundary_edges_to_outer_blossom(blossom, &[0]);

    assert_eq!(state.edges[edge as usize].head[raw_dir], blossom);
    assert_eq!(state.nodes[0].first[1 - raw_dir], NONE);
    assert_eq!(state.nodes[blossom as usize].first[1 - raw_dir], edge);
}

#[test]
fn test_into_pairs_checked_accepts_reversed_original_endpoint_order() {
    let g = build_graph(2, &[(0, 1, 5)]);
    let mut state = BlossomVState::new(&g);
    state.edges[0].head0 = [1, 0];

    let pairs = state.into_pairs_checked().expect("single edge should remain a valid matching");
    let normalized =
        pairs.iter().map(|&(r, c)| (r.as_(), c.as_())).collect::<Vec<(usize, usize)>>();

    assert_eq!(normalized, vec![(0, 1)]);
}

#[test]
fn test_into_pairs_checked_rejects_duplicate_vertex_usage() {
    let g = build_graph(4, &[(0, 1, 5), (1, 2, 7)]);
    let mut state = BlossomVState::new(&g);

    state.nodes[0].match_arc = make_arc(0, 0);
    state.nodes[1].match_arc = make_arc(1, 0);
    state.nodes[2].match_arc = make_arc(1, 1);
    state.nodes[3].match_arc = make_arc(0, 1);
    state.edges[0].head0 = [0, 1];
    state.edges[1].head0 = [1, 2];

    let err = state.into_pairs_checked().expect_err("duplicate vertex usage should be rejected");
    assert!(matches!(err, BlossomVError::NoPerfectMatching));
}

#[test]
fn test_k4_has_tight_grow_edge() {
    // After greedy: (0,1) matched, 2 and 3 are tree roots
    // Edge (0,2) should be tight (slack=0), enabling GROW
    let g = build_graph(4, &[(0, 1, 1), (0, 2, 2), (0, 3, 3), (1, 2, 4), (1, 3, 5), (2, 3, 6)]);
    let state = BlossomVState::new(&g);
    assert_eq!(state.test_tree_num(), 2);

    // Find a tight (+,free) edge
    let mut found_grow = false;
    for e in 0..state.test_edge_count() {
        if state.test_edge_slack(e) == 0 {
            let (u_orig, v_orig) = state.test_edge_endpoints(e);
            let fu = state.test_flag(u_orig as usize);
            let fv = state.test_flag(v_orig as usize);
            if (fu == PLUS && fv == FREE) || (fu == FREE && fv == PLUS) {
                found_grow = true;
            }
        }
    }
    assert!(found_grow, "Should have a tight (+,free) edge for GROW");
}

#[test]
fn test_k4_solve_produces_two_pairs() {
    let g = build_graph(4, &[(0, 1, 1), (0, 2, 2), (0, 3, 3), (1, 2, 4), (1, 3, 5), (2, 3, 6)]);
    let result = BlossomVState::new(&g).solve();
    let pairs = result.expect("K4 has a perfect matching");
    assert_eq!(pairs.len(), 2, "K4 should have 2 matched pairs, got {:?}", pairs);
}

#[test]
fn test_case_440_augment_correctness() {
    // Case 440: n=4, edges: (2,3,-20), (0,1,-17), (1,3,-84), (0,3,-28)
    // Expected: [(0,1),(2,3)], cost=-37
    // Our bug: returns [(1,3),(2,3)] — vertex 3 in two pairs
    let g = build_graph(4, &[(2, 3, -20), (0, 1, -17), (1, 3, -84), (0, 3, -28)]);
    let result = BlossomVState::new(&g).solve();
    let pairs = result.expect("should find perfect matching");
    assert_eq!(pairs.len(), 2, "should have 2 pairs, got {:?}", pairs);
    // Check no vertex used twice
    let mut used = [false; 4];
    for &(u, v) in &pairs {
        assert!(!used[u], "vertex {u} used twice in {:?}", pairs);
        assert!(!used[v], "vertex {v} used twice in {:?}", pairs);
        used[u] = true;
        used[v] = true;
    }
}

#[test]
fn test_case15_blossom_k6() {
    // Ground truth case #15: K6 with blossoms, n=6, expected cost=-44
    let g = build_graph(
        6,
        &[
            (0, 1, -14),
            (0, 2, -68),
            (0, 3, -5),
            (0, 4, 83),
            (0, 5, -61),
            (1, 2, 71),
            (1, 3, -13),
            (1, 4, 21),
            (1, 5, 63),
            (2, 3, 17),
            (2, 4, 83),
            (2, 5, -59),
            (3, 4, 87),
            (3, 5, 3),
            (4, 5, 42),
        ],
    );
    let result = BlossomVState::new(&g).solve();
    let pairs = result.expect("should find perfect matching");
    assert_eq!(pairs.len(), 3, "should have 3 pairs, got {:?}", pairs);
}
