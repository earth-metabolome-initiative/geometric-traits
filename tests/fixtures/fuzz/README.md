# Fuzz Regression Fixtures

These files replace the previous test dependency on a local honggfuzz
workspace.

`sample.bin` is the shared raw byte fixture used by the replay tests. It
deserializes through the same `Arbitrary` path used by the fuzz target
regressions for the covered types.

`lap/crash_5.fuzz` is a separate checked-in crash reproducer used by the LAP
crash regression test.
