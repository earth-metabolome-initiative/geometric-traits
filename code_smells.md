# Code Smells and Anti-patterns

**Coverage Status:** 30.85% → 44.58% (+13.73%)

---

## 1. Overly Broad Blanket Implementations

**Location:** `src/traits/directed_graph.rs`, `src/traits/complete_graph.rs`

These marker traits have blanket implementations that apply to ALL types satisfying the super trait:

```rust
impl<E: Edges> DirectedEdges for E {}
impl<G> CompleteGraph for G where G: Graph {}
impl<G> DirectedGraph for G where G: super::Graph {}
```

This means every graph is considered both "directed" and "complete" by default, which is semantically incorrect. A graph should explicitly opt-in to being complete or directed.

**Recommendation:** Remove blanket implementations and require explicit implementation, or add actual constraints that verify the property.

---

## 2. Typo in Documentation (FIXED)

**Location:** `src/traits/transposed_graph.rs:60`

Was: "in-boynd degrees" → Fixed to: "in-bound degrees"

---

## 3. Incomplete Type Coverage in `IntoUsize`

**Location:** `src/traits/into_usize.rs`

The trait implements conversions for `u8`, `u16`, `u32`, `u64` (64-bit only), and `usize`, but:
- Missing `u128` implementation
- No signed integer implementations (though this may be intentional)

---

## 4. Redundant Associated Types as Type Anchors

**Location:** Multiple trait files

Many traits define associated types that simply re-export a super trait's associated type with additional bounds. While necessary for Rust's type inference, this creates verbosity:

```rust
pub trait TransposedMonoplexGraph: MonoplexGraph<Edges = Self::TransposedEdges> {
    type TransposedEdges: TransposedEdges;
    // ...
}
```

This is a known Rust limitation, not a code smell per se, but worth noting for documentation.

---

## 5. Inconsistent Clippy Allow Attributes

**Location:** Various files

Some files use `#[allow(clippy::cast_possible_truncation)]` inline while others might benefit from the same treatment. The usage is inconsistent.

---

## 6. Empty/Minimal Trait Implementations

**Location:** `src/traits/symbol.rs`

The `Symbol` trait is just a marker combining other traits with no additional methods:

```rust
pub trait Symbol: PartialEq + Eq + Clone + core::hash::Hash + core::fmt::Debug {}
impl<T> Symbol for T where T: PartialEq + Eq + Clone + core::hash::Hash + core::fmt::Debug {}
```

While valid, this could be replaced with a type alias if Rust supported trait aliases in stable.

---

## 7. Potential Missing `#[must_use]` Attributes

**Location:** Various trait methods

Methods that return values (like `dimensions()`, `len()`, `is_empty()`) could benefit from `#[must_use]` attributes to catch cases where return values are accidentally ignored.

---

## 8. Magic Numbers in Test Assertions

**Location:** Various doc tests

Some doc tests use magic numbers without explanation:

```rust
assert!(wu_palmer.similarity(&0, &0) > 0.99);
```

The threshold `0.99` should be documented or use a named constant.

---

## 9. Duplicated Builder Patterns

**Location:** `src/naive_structs/`

Multiple builder types follow the same pattern but are implemented separately. Could potentially use a macro or generic builder trait to reduce duplication.

---

## 10. Files with 0% Test Coverage (Updated)

After adding tests, coverage improved from 30.85% to 44.58%. The following files still have 0% coverage:
- `src/traits/coordinates.rs` (0/20 lines) - tests exist but coverage tool may not detect
- `src/traits/transposed_weighted_graph.rs` (0/6 lines) - no types implement required traits
- `src/traits/transposed_monoplex_monopartite_graph.rs` (0/2 lines) - complex trait bounds
- `src/traits/matrix/transposed_valued_matrix2d.rs` (0/3 lines)
- `src/traits/matrix/valued_matrix2d/dense_valued_matrix2d.rs` (0/10 lines)

Files now at 100% coverage:
- `transposed_graph.rs`: 10/10 lines
- `transposed_monoplex_graph.rs`: 10/10 lines
- `vector.rs`: 2/2 lines
- `monoplex_monopartite_graph.rs`: 60/60 lines
- `monoplex_bipartite_graph.rs`: 11/11 lines
- `directory_tree.rs`: 36/36 lines
- `tuple.rs`: 10/10 lines
- `edge.rs`: 2/2 lines (is_self_loop tested)
- `lapjv/errors.rs`: 12/12 lines

Files with improved coverage:
- `vocabulary.rs`: 0% → 55% (22/40 lines)
- `weighted_monoplex_graph.rs`: 0% → 76.92% (20/26 lines)
- `into_usize.rs`: 30% → 80% (8/10 lines)
- `total_ord.rs`: 33% → 66% (4/6 lines)
- `bipartite_graph.rs`: 0% → 100% (20/20 lines)

---

## 11. Ambiguous Method Resolution

**Location:** Multiple graph trait files

When a type implements both a trait and its associated "graph" wrapper trait (e.g., `TransposedEdges` and `TransposedMonoplexGraph`), calling methods like `predecessors()` requires explicit trait qualification because both traits define the same method names:

```rust
// This causes "multiple applicable items in scope" error:
// bimatrix.predecessors(2)

// Must use explicit qualification:
TransposedEdges::predecessors(&bimatrix, 2)
```

This is a design choice but can be confusing for users. Consider either:
- Renaming methods in one of the traits
- Using different naming conventions for edge-level vs graph-level methods

---

## 12. Variable Shadowing with Different Types

**Location:** Test helper functions, builder patterns

Variables are frequently shadowed with different types in the same scope:

```rust
let nodes: Vec<usize> = vec![0, 1, 2, 3, 4];
let nodes: SortedVec<usize> = GenericVocabularyBuilder::default()
    .symbols(nodes.into_iter().enumerate())
    .build()
    .unwrap();
```

While valid Rust, this can be confusing. Consider using more descriptive names like `node_ids` and `nodes_vocab`.
