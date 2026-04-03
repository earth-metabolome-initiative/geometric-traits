# Fuzz Regression Fixtures

These fixtures replace the previous test dependency on a local honggfuzz
workspace.

The shared replay seed for the generic regression tests now lives inline in
`tests/test_fuzz_regression.rs`, so there is no checked-in `sample.bin`
anymore.

`lap/crash_5.fuzz` is a separate checked-in crash reproducer used by the LAP
crash regression test.
