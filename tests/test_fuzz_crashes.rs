//! Regression tests that replay raw fuzzer crash artifacts against the LAP
//! invariant-checking functions to diagnose assertion failures.

#[cfg(feature = "arbitrary")]
mod crashes {
    use std::panic::AssertUnwindSafe;

    use arbitrary::{Arbitrary, Unstructured};
    use geometric_traits::{
        impls::ValuedCSR2D,
        prelude::*,
        test_utils::{check_lap_sparse_wrapper_invariants, check_lap_square_invariants},
    };

    type Csr = ValuedCSR2D<u16, u8, u8, f64>;

    fn deserialize(data: &[u8]) -> Option<Csr> {
        let mut u = Unstructured::new(data);
        Csr::arbitrary(&mut u).ok()
    }

    /// Deserialize the crash artifact and run both invariant checkers,
    /// catching and reporting panics from each independently.
    fn reproduce_crash(name: &str, data: &[u8]) {
        let Some(csr) = deserialize(data) else {
            eprintln!("[{name}] Could not deserialize input into ValuedCSR2D");
            return;
        };

        eprintln!("=== {name} ===");
        eprintln!("Matrix: {csr:?}");
        eprintln!("Shape: {} rows x {} columns", csr.number_of_rows(), csr.number_of_columns());
        eprintln!();

        // Test check_lap_sparse_wrapper_invariants
        {
            let csr = deserialize(data).unwrap();
            let csr = AssertUnwindSafe(csr);
            let result = std::panic::catch_unwind(move || {
                check_lap_sparse_wrapper_invariants(&csr);
            });
            match result {
                Ok(()) => eprintln!("[{name}] check_lap_sparse_wrapper_invariants: OK"),
                Err(payload) => {
                    let msg = if let Some(s) = payload.downcast_ref::<&str>() {
                        s.to_string()
                    } else if let Some(s) = payload.downcast_ref::<String>() {
                        s.clone()
                    } else {
                        "<non-string panic payload>".to_string()
                    };
                    eprintln!("[{name}] check_lap_sparse_wrapper_invariants PANICKED: {msg}");
                }
            }
        }

        // Test check_lap_square_invariants
        {
            let csr = deserialize(data).unwrap();
            let csr = AssertUnwindSafe(csr);
            let result = std::panic::catch_unwind(move || {
                check_lap_square_invariants(&csr);
            });
            match result {
                Ok(()) => eprintln!("[{name}] check_lap_square_invariants: OK"),
                Err(payload) => {
                    let msg = if let Some(s) = payload.downcast_ref::<&str>() {
                        s.to_string()
                    } else if let Some(s) = payload.downcast_ref::<String>() {
                        s.clone()
                    } else {
                        "<non-string panic payload>".to_string()
                    };
                    eprintln!("[{name}] check_lap_square_invariants PANICKED: {msg}");
                }
            }
        }

        eprintln!();
    }

    #[test]
    fn crash_1() {
        reproduce_crash(
            "crash_1",
            &[
                0x32, 0x38, 0x31, 0x34, 0x00, 0x00, 0xac, 0x77, 0x91, 0xd4, 0x24, 0x24, 0x87, 0x2b,
                0xb1, 0x38, 0x66, 0x55, 0x4d, 0x18, 0x5f, 0x6c, 0x50, 0x57, 0xd5, 0xad,
            ],
        );
    }

    #[test]
    fn crash_2() {
        reproduce_crash(
            "crash_2",
            &[
                0x01, 0x01, 0x01, 0x5d, 0x5d, 0x5d, 0x5d, 0x37, 0x36, 0x5d, 0x5d, 0x5d, 0x5d, 0x5d,
                0x5d, 0x5e, 0x5d, 0x5d, 0x5d, 0x5d, 0x79, 0x7e, 0x3f, 0x61, 0x5d, 0x5d, 0x00, 0x00,
                0x00, 0x15, 0x00, 0x00,
            ],
        );
    }

    #[test]
    fn crash_3() {
        reproduce_crash(
            "crash_3",
            &[
                0x01, 0x01, 0x01, 0x5d, 0x5c, 0x5d, 0x5d, 0x5d, 0x5d, 0x5d, 0x5d, 0x5d, 0x5d, 0x5d,
                0xa0, 0x5d, 0x5d, 0x5d, 0x5d, 0x5d, 0x5d, 0x5d, 0x5d, 0x5d, 0x5d, 0x5d, 0x5d, 0x00,
                0x00, 0x00, 0x15, 0x00, 0x00,
            ],
        );
    }

    #[test]
    fn crash_4() {
        reproduce_crash(
            "crash_4",
            &[
                0x01, 0x83, 0x83, 0x00, 0x01, 0x5d, 0x5d, 0x5d, 0x5d, 0x5d, 0x5d, 0x5d, 0x5d, 0x5d,
                0x5d, 0x5d, 0x5d, 0x5d, 0x5d, 0x5d, 0x5d, 0x5d, 0x5d, 0x5d, 0x5d, 0x5d, 0x5d, 0x5d,
                0x5d, 0x5d, 0x5d, 0x5d, 0x5d, 0x5d, 0x5d, 0x00, 0x5d, 0x00, 0x5d, 0x5d, 0x5d, 0x5d,
                0x5d, 0x5d, 0x5d, 0x5d, 0x00, 0x00, 0x5d, 0x15, 0x5d, 0x00,
            ],
        );
    }

    #[test]
    fn crash_5_from_file() {
        let crash_path = "fuzz/hfuzz_workspace/lap/SIGABRT.PC.7ffff7c9eb2c.STACK.c0598fbd4.CODE.-6.ADDR.0.INSTR.mov____%eax,%r14d.fuzz";
        let Ok(data) = std::fs::read(crash_path) else {
            eprintln!("[crash_5] Crash file not found at {crash_path}, skipping");
            return;
        };
        reproduce_crash("crash_5", &data);
    }
}
