# Bug 1 Reproduction Report: Exact Execution Trace from Rust Implementation

## Executive Summary

Bug 1 fires at **Phase 1** with matching `{(0,1), (2,3), (4,7), (5,10), (6,9)}` (5 pairs).
The layered MDFS produces a non-strongly-simple path because the `reconstr_path_inner`
function, when following a P-pointer block, encounters an `expanded` marker set by a
*different* extensible edge. This sends the reconstruction into the wrong DFS subtree,
where it cycles at `s` without reaching the target node, producing a path that visits
`[11,B]` twice.

## Graph and Matching

**Graph** (n=12, 15 edges):
```
(0,1) (0,4) (0,5) (1,6) (2,3) (2,6) (3,6) (3,7)
(4,7) (4,11) (5,10) (6,9) (7,11) (8,9) (8,10)
```

**Matching at Phase 1** (the phase where Bug 1 fires):
```
M = {(0,1), (2,3), (4,7), (5,10), (6,9)}
```
(This is the result of Phase 0's layered MDFS, which found 5 paths trivially.)

**Free vertices**: 8 and 11.

## G_M Construction at Phase 1

With `a_side(v) = 2v`, `b_side(v) = 2v+1`, `s = 24`, `t = 25`:

```
s(24) → {[8,B](17), [11,B](23)}           -- free vertex B-side entries

[0,A](0)  → {[0,B](3)}                     -- matched: a(0)→b(1), but mate(0)=1 so a(0)→b(1)
```

Full G_M adjacency lists:
```
[0,A](0):  [3]          -- matched edge (0,1): a(0)→b(1)
[0,B](1):  [8, 10]      -- unmatched: b(0)→a(4), b(0)→a(5)
[1,A](2):  [1]          -- matched edge (0,1): a(1)→b(0)
[1,B](3):  [12]         -- unmatched: b(1)→a(6)
[2,A](4):  [7]          -- matched edge (2,3): a(2)→b(3)
[2,B](5):  [12]         -- unmatched: b(2)→a(6)
[3,A](6):  [5]          -- matched edge (2,3): a(3)→b(2)
[3,B](7):  [12, 14]     -- unmatched: b(3)→a(6), b(3)→a(7)
[4,A](8):  [15]         -- matched edge (4,7): a(4)→b(7)
[4,B](9):  [0, 22]      -- unmatched: b(4)→a(0), b(4)→a(11)
[5,A](10): [21]         -- matched edge (5,10): a(5)→b(10)
[5,B](11): [0]          -- unmatched: b(5)→a(0)
[6,A](12): [19]         -- matched edge (6,9): a(6)→b(9)
[6,B](13): [2, 4, 6]    -- unmatched: b(6)→a(1), b(6)→a(2), b(6)→a(3)
[7,A](14): [9]          -- matched edge (4,7): a(7)→b(4)
[7,B](15): [6, 22]      -- unmatched: b(7)→a(3), b(7)→a(11)
[8,A](16): [25]         -- free: a(8)→t
[8,B](17): [18, 20]     -- unmatched: b(8)→a(9), b(8)→a(10)
[9,A](18): [13]         -- matched edge (6,9): a(9)→b(6)
[9,B](19): [16]         -- unmatched: b(9)→a(8)
[10,A](20):[11]         -- matched edge (5,10): a(10)→b(5)
[10,B](21):[16]         -- unmatched: b(10)→a(8)
[11,A](22):[25]         -- free: a(11)→t
[11,B](23):[8, 14]      -- unmatched: b(11)→a(4), b(11)→a(7)
s(24):     [17, 23]     -- free: s→b(8), s→b(11)
```

MBFS assigns `level[t] = 9`, so the layered MDFS runs.

## MDFS Execution Trace (Phase 1)

### Subtree 1: From s via [8,B](17)

The DFS explores deep into the graph:

```
PUSH s(24)
PUSH [8,B](17)   par=s     -- tree_from_s
PUSH [9,A](18)   par=17    -- Case 2.3.ii (tree, never pushed)
PUSH [6,B](13)   par=18    -- Case 1 (A→B via matched edge)
```

From [6,B](13), three subtrees are explored:

**Branch 1a**: [6,B](13) → [1,A](2) → [0,B](1) → [5,A](10) → [10,B](21)
```
PUSH [1,A](2)    par=13    -- Case 2.3.ii
PUSH [0,B](1)    par=2     -- Case 1
PUSH [5,A](10)   par=1     -- Case 2.3.ii
PUSH [10,B](21)  par=10    -- Case 1
  step: 21→16  Case 2.2.ii(weak_back,new) → R[16] ← 21
POP [10,B](21):  backward_search → L not assigned (ever[20]=false, discovered=[])
POP_PAIRED [5,A](10)
POP [0,B](1):    backward_search → L not assigned (ever[0]=false, discovered=[])
POP_PAIRED [1,A](2)
```

**Branch 1b**: [6,B](13) → [2,A](4) → [3,B](7)
```
PUSH [2,A](4)    par=13    -- Case 2.3.ii
PUSH [3,B](7)    par=4     -- Case 1
  step: 7→12  Case 2.2.ii(weak_back,new) → R[12] ← 7
POP [3,B](7):    backward_search → L not assigned (ever[6]=false, discovered=[])
POP_PAIRED [2,A](4)
```

**Branch 1c**: [6,B](13) → [3,A](6) → [2,B](5)
```
PUSH [3,A](6)    par=13    -- Case 2.3.ii
PUSH [2,B](5)    par=6     -- Case 1
  step: 5→12  Case 2.2.ii(weak_back,new) → R[12] ← 5
POP [2,B](5):    backward_search → v_a=4=[2,A], ever[4]=true → no labels
POP_PAIRED [3,A](6)
```

**POP [6,B](13)**: backward_search on v_a=12=[6,A], ever[12]=false.
R[12]=[7, 5]. constrl walks up from 7 and 5 through parent pointers:
- constrl(5, 12, 13, 12): z=5→y_a=6→stop_b(13). Discovers y_a=6. p[6]=(5,12).
- constrl(7, 12, 13, 12): z=7→y_a=4→stop_b(13). Discovers y_a=4. p[4]=(7,12).
- **Result**: L[6]=12, L[4]=12. l_ever[4]=true, l_ever[6]=true.

POP_PAIRED [9,A](18).

### Second branch from [8,B](17): via [10,A](20)

```
PUSH [10,A](20)  par=17    -- Case 2.3.ii
PUSH [5,B](11)   par=20    -- Case 1
PUSH [0,A](0)    par=11    -- Case 2.3.ii
PUSH [1,B](3)    par=0     -- Case 1
PUSH [6,A](12)   par=3     -- Case 2.3.ii
PUSH [9,B](19)   par=12    -- Case 1
  step: 19→16  Case 2.2.ii(weak_back,new) → R[16] ← 19
POP [9,B](19):   backward_search → v_a=18, ever[18]=true → no labels
POP_PAIRED [6,A](12)
POP [1,B](3):    backward_search → v_a=2, ever[2]=true → no labels
POP_PAIRED [0,A](0)
POP [5,B](11):   backward_search → v_a=10, ever[10]=true → no labels
POP_PAIRED [10,A](20)
```

### POP [8,B](17): The critical backward_search

**backward_search(v_b=17)**: v_a=16=[8,A], ever[16]=false.
R[16]=[21, 19]. (Reverse-iterated: 19 first, then 21.)

**constrl(start_b=19, edge_a=16, stop_b=17, lcur=16)**:
Walks the par chain from 19:
```
z=19 → y_a=12=[6,A]  (l_ever[12]=false!)  → p[12]=(19,16), discovered
z=3  → y_a=0=[0,A]   (l_ever[0]=false)    → p[0]=(19,16),  discovered
z=11 → y_a=20=[10,A]  (l_ever[20]=false)   → p[20]=(19,16), discovered
z=17 == stop_b → return
```

**CRITICAL**: `l_ever[12]=false` because l_ever is only set for *discovered* nodes (nodes
added to bs_dl), not for the backward_search's *target* (v_a). In backward_search(v_b=13),
v_a=12 was the label target — L[4]=12 and L[6]=12 were assigned, but l_ever[12] was never
set. So when constrl encounters y_a=12 here, it treats it as a NEW node, not a previously
labeled one.

**constrl(start_b=21, edge_a=16, stop_b=17, lcur=16)**:
```
z=21 → y_a=10=[5,A]   (l_ever[10]=false)   → p[10]=(21,16), discovered
z=1  → y_a=2=[1,A]    (l_ever[2]=false)    → p[2]=(21,16),  discovered
z=13 (already visited in first constrl)     → return
     → but 13 was visited, so we continue to y_a=18:
z=13 → y_a=18=[9,A]   (l_ever[18]=false)   → p[18]=(21,16), discovered
z=17 == stop_b → return
```

**Result**: `discovered = [12, 0, 20, 10, 2, 18]`. All get L[·]=16.

This is the moment that creates the conditions for Bug 1.

### P-pointer state after backward_search(v_b=17)

Two backward_search rounds have set P-pointers:

| Node | P-pointer | Set by |
|------|-----------|--------|
| p[4]  | (7, 12)   | backward_search(v_b=13), constrl from R[12]={7} |
| p[6]  | (5, 12)   | backward_search(v_b=13), constrl from R[12]={5} |
| p[12] | (19, 16)  | backward_search(v_b=17), constrl from R[16]={19} |
| p[0]  | (19, 16)  | backward_search(v_b=17), constrl from R[16]={19} |
| p[20] | (19, 16)  | backward_search(v_b=17), constrl from R[16]={19} |
| p[10] | (21, 16)  | backward_search(v_b=17), constrl from R[16]={21} |
| p[2]  | (21, 16)  | backward_search(v_b=17), constrl from R[16]={21} |
| p[18] | (21, 16)  | backward_search(v_b=17), constrl from R[16]={21} |

### DFS parent pointers at this point

| Node | par | Set by |
|------|-----|--------|
| [8,B](17) | s(24)    | first PUSH from s |
| [9,A](18) | [8,B](17) | subtree 1 |
| [6,B](13) | [9,A](18) | subtree 1 |
| [1,A](2)  | [6,B](13) | subtree 1, branch 1a |
| [0,B](1)  | [1,A](2)  | subtree 1, branch 1a |
| [5,A](10) | [0,B](1)  | subtree 1, branch 1a |
| [10,B](21)| [5,A](10) | subtree 1, branch 1a |
| [2,A](4)  | [6,B](13) | subtree 1, branch 1b |
| [3,B](7)  | [2,A](4)  | subtree 1, branch 1b |
| [3,A](6)  | [6,B](13) | subtree 1, branch 1c |
| [2,B](5)  | [3,A](6)  | subtree 1, branch 1c |
| [10,A](20)| [8,B](17) | subtree 2 |
| [5,B](11) | [10,A](20)| subtree 2 |
| [0,A](0)  | [5,B](11) | subtree 2 |
| [1,B](3)  | [0,A](0)  | subtree 2 |
| [6,A](12) | [1,B](3)  | subtree 2 |
| [9,B](19) | [6,A](12) | subtree 2 |
| s(24)     | s(24)    | self |

### Subtree 2: From s via [11,B](23) — where the bug fires

```
PUSH [11,B](23)   par=s     -- tree_from_s
PUSH [4,A](8)     par=23    -- Case 2.3.ii
PUSH [7,B](15)    par=8     -- Case 1
  step: 15→6  L[6]=12, ever[12]=false → Case 2.3.i(extensible→12)
              expanded[12] = (15, 6). PUSH [6,A](12) par=15.
POP [6,A](12)  -- dead end (no further edges pass level filter)
POP [7,B](15): backward_search → L[14]=14, discovered=[]
POP_PAIRED [4,A](8)
```

Note: `expanded[12] = (15, 6)` is set here. This is the FIRST extensible edge.

```
  step: 23→14  Case 2.3.ii (tree, never pushed)
PUSH [7,A](14)   par=23
PUSH [4,B](9)    par=14    -- Case 1
  step: 9→0  L[0]=16, ever[16]=false → Case 2.3.i(extensible→16)
             expanded[16] = (9, 0). PUSH [8,A](16) par=9.
  step: 16→25  reached t
PUSH t(25)       par=16
```

This is the SECOND extensible edge: `expanded[16] = (9, 0)`.

The DFS has reached t. Stack = [s(24), [11,B](23), [7,A](14), [4,B](9), [8,A](16), t(25)].

## Path Reconstruction: Where it breaks

`reconstr_path(end=25, start=24)` walks backward from t to s:

### Step 1: t → [8,A](16)

cur=25. Push 25. par[25]=16. cur=16.

### Step 2: [8,A](16) has expanded marker

cur=16. Push 16. **expanded[16] = (9, 0)** → call `reconstr_q(u_a=16, w_a=0)`.

### Step 3: reconstr_q(u_a=16, w_a=0)

Follow P-pointer chain from w_a=0:
- p[0] = (19, 16). blocks = [(19, 0)]. p2_a = 16 == u_a. Done.
- Reconstruct block: `reconstr_path_inner(end=19, start=0, out, true)`.

### Step 4: reconstr_path_inner(19, 0) — the broken reconstruction

Walk from 19 toward 0 via par pointers:

```
cur=19  → push 19. par[19]=12. cur=12.
cur=12  → push 12. expanded[12] = (15, 6)!
          → call reconstr_q(u_a=12, w_a=6)
```

**THE BUG**: `expanded[12]` was set during the SECOND subtree's DFS exploration (when
[7,B](15)→[3,A](6) triggered Case 2.3.i). But `reconstr_path_inner(19, 0)` is
reconstructing a path segment from the FIRST subtree's backward_search (the P-pointer
p[0]=(19,16) was set by constrl walking the par chain 19→12→3→0 which goes through the
FIRST subtree's DFS tree).

The expansion of `expanded[12]=(15, 6)` calls `reconstr_q(12, 6)`:
- p[6] = (5, 12). blocks=[(5, 6)]. p2_a=12 == u_a. Done.
- reconstr_path_inner(end=5, start=6): walks 5→par[5]=6. Output: [5, 6]. OK.

After reconstr_q(12, 6) returns, `reconstr_path_inner(19, 0)` continues from
`v_b = 15` (from expanded[12]=(15, 6)):

```
cur=15  → push 15. par[15]=8.  cur=8.
cur=8   → push 8.  par[8]=23.  cur=23.
cur=23  → push 23. par[23]=24. cur=24.
cur=24  → push 24. par[24]=24. cur=24.  ← INFINITE LOOP
cur=24  → push 24. par[24]=24. cur=24.
... (repeats until the steps < sz*4 guard kicks in at ~104 iterations)
```

The reconstruction went from node 12 (in the first subtree: par chain 19→12→3→0)
into node 15 (in the second subtree: par chain 15→8→23→s). It never reaches node 0
(the target of this inner reconstruction), so it loops at s until the guard fires.

### Result

The full reversed path is:
```
s(24), [11,B](23), [7,A](14), [4,B](9), [0,A](0), s(24)×~98,
[11,B](23), [4,A](8), [7,B](15), [3,A](6), [2,B](5), [6,A](12),
[9,B](19), [8,A](16), t(25)
```

**Vertex 11 appears as [11,B](23) at positions 1 and ~104 → not strongly simple.**

## Root Cause Analysis

The bug is in the interaction between two mechanisms:

1. **P-pointers record tree-path blocks**: When backward_search(v_b=17) runs constrl
   from start_b=19, it walks the par chain 19→12→3→0→11→20→17(stop). It records
   p[12]=(19, 16), meaning "to reconstruct the block ending at 12, start from node 19".

2. **Expanded markers record extensible edges**: When the DFS processes edge [7,B](15)→
   [3,A](6) as Case 2.3.i, it sets expanded[12]=(15, 6), meaning "node 12 was reached
   via an extensible edge from 15 through label w_a=6".

3. **Reconstruction conflates the two**: When reconstr_path_inner(19, 0) encounters
   expanded[12], it follows the extensible edge expansion instead of the par pointer.
   This sends the reconstruction from node 12 (in subtree 1's par chain) into node 15
   (in subtree 2's par chain), where the par pointers lead to s, not to node 0.

The fundamental problem: **the reconstruction assumes that `expanded[u_a]` and `par[u_a]`
are part of the same tree path, but in this execution they are set by different DFS
subtrees.** The par chain 19→12→3→0 was built during the first subtree's exploration,
while expanded[12]=(15, 6) was set during the second subtree's exploration.

## MDFS State at the Point of Bug

This is the exact state for constructing the Lean proof:

```
n = 12,  s = 24,  t = 25,  sz = 26

expanded = {12: (15, 6), 16: (9, 0)}

p = {0: (19, 16), 2: (21, 16), 4: (7, 12), 6: (5, 12),
     10: (21, 16), 12: (19, 16), 18: (21, 16), 20: (19, 16)}

par = {0: 11, 1: 2, 2: 13, 3: 0, 4: 13, 5: 6, 6: 13, 7: 4,
       8: 23, 9: 14, 10: 1, 11: 20, 12: 15, 13: 18, 14: 23,
       15: 8, 16: 9, 17: 24, 18: 17, 19: 12, 20: 17, 21: 10,
       23: 24, 24: 24, 25: 16}

stack K = [24, 23, 14, 9, 16, 25]
```

**NOTE on `par[12]`**: The par dump shows `par[12]=15`, which was set by the second
PUSH of node 12 (line 140 of trace: `PUSH(12) par=15`). But the P-pointer `p[12]=(19,16)`
was set during backward_search(v_b=17) when par[12] was still 3 (from the first PUSH).
The reconstruction uses `par[12]=15` (the latest value), which is from the wrong subtree.

This is an additional wrinkle: **the par pointer for node 12 was overwritten** by the
second PUSH. The P-pointer chain was built assuming par[12]=3 (leading to par chain
12→3→0→11→20→17), but by the time reconstruction runs, par[12]=15 (leading to 15→8→23→24).

## What the Lean Prover Needs

### Approach 1: Prove reconstruction produces non-strongly-simple path from MDFS state

Given the MDFS state above, prove that `reconstr_path(end=25, start=24)` produces a
path containing `[11,B](23)` twice:

```lean
/-- The MDFS state at the point where Bug 1 manifests. -/
def bug1State : MdfsState where
  n := 12
  s := 24
  t := 25
  sz := 26
  par := ![11, 2, 13, 0, 13, 6, 13, 4, 23, 14, 1, 20, 15, 18, 23, 8, 9, 24, 17, 12, 17, 10, default, 24, 24, 16]
  expanded := ![none, none, none, none, none, none, none, none, none, none, none, none, some (15, 6), none, none, none, some (9, 0), ...]
  p := ![some (19, 16), none, some (21, 16), none, some (7, 12), none, some (5, 12), none, none, none, some (21, 16), none, some (19, 16), none, none, none, none, none, some (21, 16), none, some (19, 16), ...]
  -- (fill remaining with none/default)

/-- The reconstruction from this state is not strongly simple. -/
theorem bug1_not_strongly_simple :
    let path := reconstruct bug1State
    ∃ v, v < 12 ∧ (a_side v ∈ path ∨ b_side v ∈ path) ∧ countOccurrences (orig · ) path v > 1 := by
  native_decide
```

### Approach 2: Prove the reconstruction loop diverges

A cleaner proof: show that `reconstr_path_inner(19, 0)` with the given state enters a
cycle. Specifically, show that the par-chain from node 15 (entered via expanded[12])
never reaches node 0:

```
par[15]=8 → par[8]=23 → par[23]=24 → par[24]=24 (cycle)
```

Node 0 is not in the set {15, 8, 23, 24}, so the walk never terminates.

### Approach 3: Prove the P-pointer/expanded conflict directly

Show that when backward_search(v_b=17) calls constrl(start_b=19, edge_a=16, stop_b=17,
lcur=16), and constrl discovers y_a=12 as a NEW node (l_ever[12]=false), it records
p[12]=(19, 16). But later, when the DFS processes edge [7,B](15)→[3,A](6) as Case 2.3.i,
it sets expanded[12]=(15, 6), which overwrites the "meaning" of node 12 in the
reconstruction. The P-pointer says "walk from 19 to find 12 via par", but the expanded
marker says "12 was reached from 15 via extensible edge" — these are contradictory because
par[12] is overwritten from 3 (first subtree) to 15 (second subtree).

## Key Insight for the Prover

The bug does NOT require reproducing the entire MDFS from scratch. It requires:

1. Constructing the MDFS state snapshot above (par, expanded, p arrays).
2. Running `reconstr_path(25, 24)` on that state.
3. Showing the result contains orig(23)=11 twice (or more directly, that the inner call
   `reconstr_path_inner(19, 0)` never reaches 0).

This is a **pure functional property** of the reconstruction algorithm given a specific
state — no DFS simulation needed.

## Adjacency List Ordering

The CSR-based adjacency lists determine the DFS exploration order. The G_M adjacency
lists above are the exact lists used. The critical ordering is:

- `s(24) → [17, 23]`: [8,B] is explored before [11,B]
- `[6,B](13) → [2, 4, 6]`: branches from [6,B] are explored in order [1,A], [2,A], [3,A]
- `[11,B](23) → [8, 14]`: [4,A] is explored before [7,A]
- `[7,B](15) → [6, 22]`: [3,A] is explored before [11,A]

These orderings are determined by the CSR storage of the input graph and the `fill_gm`
construction, which iterates `u < v` edges adding both directions.

## Bug Reproduction Confirmation

Disabling the `validated_path` strong-simplicity check (replacing it with a pass-through
that returns the reconstructed path unconditionally) causes the test to fail with:

```
thread 'test_regression_large_karp_sipser_fixture_replays_blum_invalid_matching' panicked at tests/test_karp_sipser.rs:27:9:
vertex 4 matched twice
```

This confirms the bug is real: without the workaround, the algorithm returns a matching
where vertex 4 is matched to two different partners. The non-strongly-simple reconstructed
path causes the augmentation to put vertex 4 into two matched pairs simultaneously.

Restoring the workaround makes the test pass again. The workaround detects the bad path,
rejects it (returns empty), the layered MDFS finds 0 valid paths, the per-vertex fallback
is invoked, and it finds the correct 6th augmenting path through a fresh DFS.
