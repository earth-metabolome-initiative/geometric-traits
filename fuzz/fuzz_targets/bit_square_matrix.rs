//! Fuzzing target for [`BitSquareMatrix`].
//!
//! Builds a `BitSquareMatrix` from arbitrary edge data using multiple
//! construction paths, then validates comprehensive invariants including
//! bitwise operations, transpose roundtrip, and iterator contracts.

use geometric_traits::{prelude::*, test_utils::check_bit_square_matrix_invariants};
use honggfuzz::fuzz;

type FuzzInput = (u8, u8, Vec<u8>, Vec<(u8, u8, u8)>);

fn main() {
    loop {
        fuzz!(|data: FuzzInput| {
            let (order_byte, constructor_byte, mask_bytes, ops) = data;
            // Cap order to keep matrices small but exercise multi-word bitvec paths.
            let order = order_byte as usize % 128;

            // Select construction method from fuzz data.
            let mut m = match constructor_byte % 3 {
                0 => {
                    // Incremental set/set_symmetric/clear.
                    let mut m = BitSquareMatrix::new(order);
                    if order > 0 {
                        for &(r, c, op) in &ops {
                            let r = r as usize % order;
                            let c = c as usize % order;
                            match op % 3 {
                                0 => m.set(r, c),
                                1 => m.set_symmetric(r, c),
                                _ => m.clear(r, c),
                            }
                        }
                    }
                    m
                }
                1 => {
                    // from_edges
                    let edges: Vec<(usize, usize)> = if order > 0 {
                        ops.iter()
                            .map(|&(r, c, _)| (r as usize % order, c as usize % order))
                            .collect()
                    } else {
                        Vec::new()
                    };
                    BitSquareMatrix::from_edges(order, edges.iter().copied())
                }
                _ => {
                    // from_symmetric_edges
                    let edges: Vec<(usize, usize)> = if order > 0 {
                        ops.iter()
                            .map(|&(r, c, _)| (r as usize % order, c as usize % order))
                            .collect()
                    } else {
                        Vec::new()
                    };
                    BitSquareMatrix::from_symmetric_edges(order, edges.iter().copied())
                }
            };

            // Apply some additional mutations for from_edges/from_symmetric_edges paths.
            if constructor_byte % 3 != 0 && order > 0 {
                for &(r, c, op) in ops.iter().rev().take(ops.len().min(8)) {
                    let r = r as usize % order;
                    let c = c as usize % order;
                    match op % 3 {
                        0 => m.set(r, c),
                        1 => m.set_symmetric(r, c),
                        _ => m.clear(r, c),
                    }
                }
            }

            check_bit_square_matrix_invariants(&m, &mask_bytes);
        });
    }
}
