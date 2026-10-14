#!/usr/bin/env bash
set -euo pipefail

SOURCE_FILE="${SOURCE_FILE:-}"
SHARD_INDEX="${SHARD_INDEX:-1}"
BATCH_SIZE="${BATCH_SIZE:-20}"
WALL_CLOCK_LIMIT="${WALL_CLOCK_LIMIT:-2700}"
MUTANTS_BUILD_TIMEOUT="${MUTANTS_BUILD_TIMEOUT:-900}"
MUTANTS_TEST_TIMEOUT="${MUTANTS_TEST_TIMEOUT:-300}"
CLEANUP_GRACE="${CLEANUP_GRACE:-5}"
SUMMARY_FILE="mutation-audit-summary.md"

fail() {
  echo "::error::[mutation-audit] $*" >&2
  exit 1
}

for value in "$SHARD_INDEX" "$BATCH_SIZE" "$WALL_CLOCK_LIMIT" "$MUTANTS_BUILD_TIMEOUT" "$MUTANTS_TEST_TIMEOUT" "$CLEANUP_GRACE"; do
  [[ "$value" =~ ^[1-9][0-9]{0,5}$ ]] || fail "limits must be positive integers below 1000000"
done
(( BATCH_SIZE <= 20 )) || fail "BATCH_SIZE must be at most 20"
for tool in cargo jq timeout nproc nice taskset grep date mktemp git; do
  command -v "$tool" >/dev/null || fail "required tool $tool is not on PATH"
done

cd "$(git rev-parse --show-toplevel)"
tmp="$(mktemp -d)"

# Only descendants carrying this run's marker may be signalled.
owned_pids() {
  local path pid
  for path in /proc/[0-9]*/environ; do
    if grep -zFxq "GEOMETRIC_MUTATION_AUDIT_ID=$tmp" "$path" 2>/dev/null; then
      pid="${path#/proc/}"
      echo "${pid%/environ}"
    fi
  done
}

signal_audit() {
  local pid
  while IFS= read -r pid; do
    kill "-$1" "$pid" 2>/dev/null || true
  done < <(owned_pids)
}

cleanup() {
  local remaining
  remaining="$(owned_pids)"
  if [[ -n "$remaining" ]]; then
    signal_audit INT
    sleep "$CLEANUP_GRACE"
    signal_audit KILL
  fi
  [[ -z "$(owned_pids)" ]] || return 1
}

trap 'cleanup; rm -rf "$tmp"' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

build_jobs="$(nproc)"
(( build_jobs <= 8 )) || build_jobs=8
export CARGO_BUILD_JOBS="$build_jobs" RUST_TEST_THREADS="$build_jobs"
allowed=""
while read -r key value; do
  if [[ "$key" == Cpus_allowed_list: ]]; then
    allowed="$value"
    break
  fi
done < /proc/self/status
cpu_ids=()
IFS=, read -ra ranges <<< "$allowed"
for range in "${ranges[@]}"; do
  first="${range%-*}"
  last="${range#*-}"
  for ((cpu=first; cpu<=last && ${#cpu_ids[@]}<build_jobs; cpu++)); do
    cpu_ids+=("$cpu")
  done
  (( ${#cpu_ids[@]} < build_jobs )) || break
done
(( ${#cpu_ids[@]} == build_jobs )) || fail "cannot determine the available CPU affinity"
mask="$(IFS=,; echo "${cpu_ids[*]}")"

# Compiler brokers could escape the caller's affinity and priority.
audit_env=(env "GEOMETRIC_MUTATION_AUDIT_ID=$tmp" "TMPDIR=$tmp" RUSTC_WRAPPER= RUSTC_WORKSPACE_WRAPPER=)
mutants=(taskset -c "$mask" nice -n 19 cargo mutants -p geometric-traits --jobs 1 --jobserver-tasks "$build_jobs" --cargo-arg=--locked)
list_mutants() {
  "${audit_env[@]}" timeout --signal=INT --kill-after=30s 120s "${mutants[@]}" --all-features --list --json "$@"
}

if [[ -n "$SOURCE_FILE" ]]; then
  [[ "$SOURCE_FILE" == src/*.rs && "$SOURCE_FILE" != *..* && -f "$SOURCE_FILE" ]] || fail "SOURCE_FILE must name a Rust source file under src"
  scope="$SOURCE_FILE"
else
  scope='src/traits/algorithms/**/*.rs'
fi
list_mutants --file "$scope" > "$tmp/candidates.json"
files="$(jq -c --argjson batch "$BATCH_SIZE" '
  group_by(.file) | map({file: .[0].file, count: length, shards: ((length + $batch - 1) / $batch | floor)})
  | sort_by(.file)
' "$tmp/candidates.json")"
shard0=$((SHARD_INDEX - 1))
if [[ -z "$SOURCE_FILE" ]]; then
  week=$(( $(date -u +%s) / 604800 ))
  selection="$(jq -nc --argjson files "$files" --argjson week "$week" '
    [$files[] | . as $file | range(0; .shards) | $file + {shard0: .}] as $pool
    | if ($pool | length) == 0 then null else $pool[$week % ($pool | length)] end
  ')"
  if [[ "$selection" != null ]]; then
    SOURCE_FILE="$(jq -r '.file' <<< "$selection")"
    shard0="$(jq -r '.shard0' <<< "$selection")"
  fi
else
  selection="$(jq -c '.[0] // null' <<< "$files")"
fi

status="empty scope"
rc=0
elapsed=0
selected=0
shard_count=0
caught=0
missed=0
unviable=0
timed_out=0
tested=0
baseline="not run"
if [[ "$selection" != null ]]; then
  shard_count="$(jq -r '.shards' <<< "$selection")"
  (( shard0 < shard_count )) || fail "SHARD_INDEX exceeds the file's shard count"
  list_mutants --file "$SOURCE_FILE" --shard "$shard0/$shard_count" --sharding round-robin > "$tmp/selected.json"
  selected="$(jq length "$tmp/selected.json")"
  (( selected > 0 && selected <= BATCH_SIZE )) || fail "selected shard is outside the batch limit"

  # Explicit targets omit benches and doctests without narrowing consumer tests.
  test_flags=(--lib)
  for file in tests/*.rs; do
    [[ -f "$file" ]] || continue
    name="${file#tests/}"
    test_flags+=(--test "${name%.rs}")
  done
  (( ${#test_flags[@]} > 1 )) || fail "no integration test targets found"
  report_dir="$(mktemp -d mutation-audit.XXXXXX)"
  baseline="not-recorded"
  read -r uptime _ < /proc/uptime
  start="${uptime%.*}"
  "${audit_env[@]}" timeout --signal=INT --kill-after=30s "${WALL_CLOCK_LIMIT}s" "${mutants[@]}" \
    --file "$SOURCE_FILE" --shard "$shard0/$shard_count" --sharding round-robin \
    --all-features --output "$report_dir" --build-timeout "$MUTANTS_BUILD_TIMEOUT" --timeout "$MUTANTS_TEST_TIMEOUT" \
    -- "${test_flags[@]}" || rc=$?
  read -r uptime _ < /proc/uptime
  elapsed=$(( ${uptime%.*} - start ))
  case "$rc" in
    0 | 2 | 3) status="advisory outcomes" ;;
    4) status="baseline failure" ;;
    124 | 137) status="incomplete, budget exhausted" ;;
    *) status="infrastructure failure" ;;
  esac
  if [[ -s "$report_dir/mutants.out/outcomes.json" ]] && jq -e '.outcomes | type == "array"' "$report_dir/mutants.out/outcomes.json" >/dev/null; then
    read -r caught missed unviable timed_out tested baseline < <(jq -r '
      [.caught // 0, .missed // 0, .unviable // 0, .timeout // 0,
       ([.outcomes[] | select(.scenario != "Baseline")] | length),
       ([.outcomes[] | select(.scenario == "Baseline")][0].summary // "not-recorded")] | @tsv
    ' "$report_dir/mutants.out/outcomes.json")
  elif [[ "$status" == "advisory outcomes" ]]; then
    status="infrastructure failure, missing outcomes"
  fi
  if [[ "$status" == "advisory outcomes" && ( "$baseline" != Success || "$tested" != "$selected" ) ]]; then
    status="infrastructure failure, incomplete outcomes"
  fi
fi

cleanup || fail "audit descendants survived cleanup"
shard_label="none"
(( shard_count == 0 )) || shard_label="$((shard0 + 1))/$shard_count"
{
  echo "# Mutation audit"
  echo
  echo "| Item | Result |"
  echo "|---|---|"
  echo "| Source | \`$SOURCE_FILE\` |"
  echo "| Shard | $shard_label, at most $BATCH_SIZE candidates |"
  echo "| Test selection | All unit and integration targets, \`--all-features --locked\` |"
  echo "| Outcomes | $caught caught, $missed missed, $unviable unviable, $timed_out timeout |"
  echo "| Completed | $tested/$selected |"
  echo "| Baseline | $baseline |"
  echo "| Budget | $elapsed/$WALL_CLOCK_LIMIT seconds, $build_jobs CPUs, one worker |"
  echo "| Status | $status, tool exit $rc |"
  echo
  echo 'Review survivors for observable differences, equivalent behavior or test gaps. Investigate timeouts before changing limits.'
  if [[ "$status" == "empty scope" ]]; then
    echo "No mutation candidates were selected."
  else
    if [[ -d "$report_dir/mutants.out" ]]; then
      echo "Logs, diffs and per-candidate outcomes are in \`$report_dir/mutants.out/\`."
    else
      echo "No tool report was initialized. See the run log."
    fi
  fi
} > "$SUMMARY_FILE"
cat "$SUMMARY_FILE"
case "$status" in
  "empty scope" | "advisory outcomes") exit 0 ;;
  *) exit 1 ;;
esac
