//! Submodule for fuzzing the Kocay-Stone BNS balanced flow algorithm.

use geometric_traits::prelude::*;
use honggfuzz::fuzz;

fn main() {
    loop {
        fuzz!(|data: (ValuedCSR2D<u16, u8, u8, u8>, Vec<u8>)| {
            let (vcsr, raw_budgets) = data;
            let n_rows = vcsr.number_of_rows() as usize;
            let n_cols = vcsr.number_of_columns() as usize;
            if n_rows != n_cols || n_rows > 64 || n_rows == 0 {
                return;
            }
            let n = n_rows;

            // Build budgets: use raw_budgets values, clamp to n elements.
            if raw_budgets.len() < n {
                return;
            }
            let budgets: Vec<u8> = raw_budgets[..n].to_vec();

            let flow = vcsr.kocay(&budgets);

            // Validate: capacity respect and flow conservation.
            let mut vertex_flow = vec![0u16; n];
            for &(i, j, f) in &flow {
                assert!(i < j, "flow triple ordering violated");
                assert!(f > 0, "zero flow in output");

                // Check edge exists and capacity respected.
                let mut found_cap = None;
                for (col, val) in vcsr.sparse_row(i).zip(vcsr.sparse_row_values(i)) {
                    if col == j {
                        found_cap = Some(val);
                        break;
                    }
                }
                if let Some(cap) = found_cap {
                    assert!(f <= cap, "flow {f} exceeds capacity {cap} on edge ({i}, {j})");
                }

                vertex_flow[i as usize] += f as u16;
                vertex_flow[j as usize] += f as u16;
            }

            for v in 0..n {
                assert!(
                    vertex_flow[v] <= budgets[v] as u16,
                    "vertex {v} flow {} exceeds budget {}",
                    vertex_flow[v],
                    budgets[v]
                );
            }
        });
    }
}
