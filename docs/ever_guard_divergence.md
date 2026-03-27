# The `!ever[vA]` Guard: Removal Causes Non-Termination

## Summary

Removing the `!ever[v_a]` guard from backward search label assignment (to match the
paper's unconditional assignment, Section 2.5) causes the MDFS to loop infinitely on
at least 3 graphs. The guard is load-bearing for termination, not just a performance
optimization. This constitutes a bug in the paper's specification.

## The guard

In `backward_search`, after CONSTRL discovers nodes and the BFS-through-E loop
completes, labels are assigned:

```rust
if !self.ever[v_a] {
    for y_a in discovered {
        L[y_a] = v_a;
    }
}
```

When `ever[v_a]` is true, the label assignment is skipped entirely. The paper
(Blum 2016, Section 2.5, p.19) performs this assignment unconditionally.

## What happens when the guard is removed

With unconditional label assignment, labels can point to nodes (`v_a`) whose DFS
subtrees have already been fully explored and popped. When Case 2.3.i later encounters
such a label, it pushes `v_a` via an extensible edge. From `v_a`, the DFS re-explores
neighbors, but it can cycle back into the same pattern: explore, pop, assign more labels,
push via extensible edge, explore again -- without ever reaching `t`.

53 of 56 tests in `test_blum.rs` pass. Three tests hang (infinite loop in the MDFS):

## Graph 1: K_7 (Complete graph on 7 vertices)

```
n = 7, 21 edges
(0,1) (0,2) (0,3) (0,4) (0,5) (0,6)
(1,2) (1,3) (1,4) (1,5) (1,6)
(2,3) (2,4) (2,5) (2,6)
(3,4) (3,5) (3,6)
(4,5) (4,6)
(5,6)
```

Expected maximum matching size: 3.

This is the simplest counterexample -- a dense graph where every vertex is adjacent to
every other. The high connectivity means backward search discovers many nodes, and
without the guard, labels proliferate, creating extensible edge cycles.

## Graph 2: Regression graph (n=12, 16 edges)

```
n = 12, 16 edges
(0,1) (0,3) (0,5) (1,2) (1,6) (2,10) (3,5) (3,6)
(4,7) (4,11) (5,10) (6,9) (7,9) (7,11) (8,9) (8,10)
```

Expected maximum matching size: 6.

Test: `test_regression_invalid_non_edge_from_degree1_kernel`

## Graph 3: Regression graph (n=12, 19 edges)

```
n = 12, 19 edges
(0,1) (0,3) (0,5) (1,2) (1,3) (1,6) (2,3) (2,10) (3,5) (3,6)
(4,7) (4,9) (4,11) (5,10) (6,9) (7,9) (7,11) (8,9) (8,10)
```

Expected maximum matching size: 6.

Test: `test_regression_non_edge_from_degree1_kernel_corpus_two`

## Why unconditional assignment breaks termination

The MDFS termination argument relies on the invariant that each node is pushed at most
once in a given search (tracked by the `ever` array). The extensible edge mechanism
(Case 2.3.i) pushes a node `u_a` that was NEVER pushed before -- the label `L[w] = u_a`
is only useful if `ever[u_a] = false`, because the Case 2.3.i code checks
`self.ever[u_a] || self.deleted[u_a]` and skips used targets.

When the guard is present (`if !self.ever[v_a]`), labels only point to nodes that were
never pushed. This guarantees the extensible edge target is fresh, maintaining the
"each node pushed at most once" invariant.

When the guard is removed, labels can point to nodes with `ever[v_a] = true`. The
Case 2.3.i label chain chase skips these (the `while self.ever[u_a]` loop), but this
chase can redirect to other nodes that ARE fresh. The problem is that this redirection
can create circular dependencies: node A's label chain leads to node B, which gets
pushed and popped, triggering backward search that sets labels pointing back toward
subtrees containing A's predecessors, leading to another extensible edge push, and so on.

The `ever` flag prevents a single node from being pushed twice, but it does NOT prevent
the DFS from making zero net progress -- the search can cycle through different nodes
indefinitely, each pushed once but collectively forming a loop in the search strategy.

## Relationship to D&L Bug A

The Lean formalization found that removing the guard fixes D&L's Figure 1 counterexample
(Case 2.2.i). This is correct for that specific graph -- the extra labels provide the
extensible edge path that the guard blocks. But the fix is not general: it trades
correctness on D&L's Figure 1 for non-termination on K_7 and other graphs.

The D&L R-set correction for Case 2.2.i remains the correct fix. It provides the missing
backward search information through a different mechanism (adding to R sets instead of
unconditional labeling) that does not break the termination invariant.

## Conclusion

The `!ever[v_a]` guard is a necessary correction to the paper. The paper's unconditional
label assignment (Section 2.5) is Bug 4 -- an implicit assumption that labels pointing
to already-explored nodes are harmless, when in fact they can create non-terminating
search cycles.

The three counterexamples above (K_7, and two n=12 regression graphs) are witnesses
to this bug. They can be added to the Lean formalization as non-termination evidence.
