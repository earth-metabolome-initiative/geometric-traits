#!/bin/bash
set -eu

cd "$SRC/geometric-traits"
# the base image exports its own nightly as RUSTUP_TOOLCHAIN, so no toolchain is named here
cargo fuzz build -O --debug-assertions --fuzz-dir fuzz

targets=$(cargo fuzz list --fuzz-dir fuzz)
if [[ -z "$targets" ]]; then
    echo "cargo fuzz list named no target" >&2
    exit 1
fi

target_dir=fuzz/target/x86_64-unknown-linux-gnu/release
for name in $targets; do
    cp "$target_dir/$name" "$OUT/"
done
