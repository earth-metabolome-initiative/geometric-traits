# Experimental Evaluation of Blum’s Maximum Matching Algorithm in General Graphs
Ahmad Dandeh1,* , Tamás Lukovszki1
1

Eötvös Loránd University (ELTE), Budapest, Hungary

Abstract

We describe an implementation and experimental evaluation of Blum’s maximum matching algorithm in general
graphs. Blum’s algorithm finds augmenting paths in general graphs without explicitly analyzing blossoms.
Although there are many implementations and performance studies of Edmonds’ blossom algorithm and its
variants, we are not aware of any implementation of Blum’s approach. We compare three implementations:
Blossom I, Blossom V, and Blum’s modified depth-first search (MDFS). We extend Blum’s algorithm with a
preprocessing step that computes an initial random matching. This significantly reduces the number of augmenting
path searches. We prepared a Java implementation of MDFS. We describe how to handle some cases that are
not fully discussed in Blum’s article. We created a unified experimental framework for testing the algorithms
on random graphs, geometric graphs, complete graphs, DIMACS benchmark graphs, and overlapping cycles.
Our experiments show that Blossom V is always faster than Blossom I and very efficient on all inputs. Blum’s
algorithm with preprocessing competes with Blossom V in several classes of graphs and outperforms it in some
cases, which confirms the resilience of MDFS if properly initialized. These results suggest that preprocessing
makes reachability-based algorithms a good alternative to blossom algorithms.

Keywords

Maximum Matching, Graph Algorithms, Algorithm Engineering, Performance Evaluation

1. Introduction
The problem of computing a maximum cardinality matching in general undirected graphs is a fundamental problem in graph theory and combinatorial optimization. A matching is a set of edges such that
no two edges share a common vertex. A matching is maximum if it contains the largest possible number
of such edges. The theoretical foundation for modern algorithms is Berge’s theorem [1], which states
that a matching in a graph is maximum if and only if there exists no augmenting path relative to that
matching. Augmenting paths are sequences of edges of the graph, which alternate between edges in
the matching and edges not in the matching, such that the first and last edges are not in the matching.
In bipartite graphs, augmenting paths can be found by a BFS or DFS like method, leading to efficient
√
algorithms such as the Hopcroft–Karp algorithm for maximum matching in 𝑂( 𝑛𝑚) time [2], where
𝑛 and 𝑚 are the number of vertices and edges, respectively. For non-bipartite (general) graphs, odd
cycles pose a problem. Augmenting paths can overlap or intersect themselves, and need to be dealt
with carefully to achieve correctness.
This challenge was overcome by Edmonds [3], who introduced the blossom shrinking technique to
manage odd-length cycles and preserve augmenting path structure. His algorithm was the first to solve
the maximum matching problem in general graphs in polynomial time. The original algorithm has a
worst-case complexity of 𝑂(𝑛3 ).
This bound is maintained in initial implementations like Blossom I [4]. More recently, Blossom
V introduced implementation-level enhancements by Kolmogorov [5], including the use of priority
queues, an auxiliary graph for managing alternating trees, and a variable dual update strategy [6]. These
techniques yield significant practical speedups, particularly on large or structured inputs. However,

ICTCS 2025: Italian Conference on Theoretical Computer Science, September 10–12, 2025, Pescara, Italy
Corresponding author.
$ dandehahmad@inf.elte.hu (A. Dandeh); lukovszki@inf.elte.hu (T. Lukovszki)
0009-0000-3217-648X (A. Dandeh); 0000-0001-8878-3246 (T. Lukovszki)

*

© 2025 Copyright for this paper by its authors. Use permitted under Creative Commons License Attribution 4.0 International (CC BY 4.0).

CEUR

Workshop
Proceedings

ceur-ws.org
ISSN 1613-0073



---

due to the flexibility of the dual update logic, the worst-case time complexity of Blossom V is estimated
to be 𝑂(𝑛2 𝑚) [5].
Blum suggested an effective alternative solution using graph transformation and reachability [7, 8].
Blum’s MDFS algorithm avoids blossom contraction by transforming the matching problem into a
reachability problem in an altered, directed bipartite graph. MDFS can be implemented in 𝑂(𝑛 + 𝑚)
time [8], and the maximum matching in 𝑂(𝑛(𝑛 + 𝑚)) complexity time, but it is conceptually easier to
comprehend and implement, as it eliminates blossom detection and shrinking. Despite this, MDFS has
received limited attention in practical settings, and we are not aware of any implementations.
In this paper, we present an experimental evaluation of three algorithms for maximum cardinality
matching:
• Blossom I, with a public domain Java implementation; [9]
• Blossom V, using the JGraphT library; [10]
• MDFS, derived from our implementation and optimization of Blum’s algorithm.
We compare their performance on a range of graph instances and analyze the effect of providing
initial matching. We also identify some cases that are not completely covered by the MDFS in [8]. We
describe the necessary modifications that handle these cases correctly.
Our key contributions are the following:
• A unified experimental framework in Java for comparing Blossom I, Blossom V, and MDFS on
unweighted graphs.
• An MDFS algorithm implementation of Blum’s algorithm with practical robustness enhancements
for specific cases that are not completely covered in [8].
• An empirical assessment with correctness behavior, and running time across a broad spectrum of
graph types.
The remainder of this paper is structured as follows. Section 2 provides background and necessary
algorithmic concepts, and our extension of MDFS. Section 3 presents the experimental setup and
comparative results. Section 4 concludes with a discussion and future work.

2. Background: Blum’s Algorithm
We use publicly available Java implementations of Blossom I and Blossom V to contrast classical
blossom-based algorithms in this paper. For Blum’s algorithm, we used the reformulation in [8], which
framed the maximum matching problem as a reachability problem in an appropriately transformed
directed bipartite graph. We call this method Modified Depth-First Search (MDFS), and it does not
perform actual blossom contraction but instead searches for strongly simple augmenting paths. While
Blum’s paper includes a conceptual algorithm and suggestions for implementation, our experience in
coding revealed some edge cases where the behavior is ambiguous or leads to the wrong traversal.
The remainder of this section gives the basic definitions used in MDFS, and then describes in detail
the specific cases we discovered and how we corrected them.

2.1. Terminology and Algorithm Overview
Let 𝐺 = (𝑉, 𝐸) be an undirected, and unweighted graph where 𝑉 is the set of vertices and 𝐸 is the
set of edges 𝐸 ⊆ {{𝑢, 𝑣} | 𝑢, 𝑣 ∈ 𝑉, 𝑢 ̸= 𝑣}. A bipartite graph is a graph whose vertex set can be
partitioned into two disjoint sets 𝑈 and 𝑊 , and whose edges join vertices of 𝑈 to those of 𝑊 . A
matching 𝑀 ⊆ 𝐸 is a set of edges such that no two edges in 𝑀 share a common vertex. A vertex is
matched if it is incident on an edge in 𝑀 ; otherwise, it is free.
An augmenting path is an elementary path in 𝐺 such that it starts and ends at free vertices and keeps
alternating between non-matching edges and matching edges.



---

Following Berge’s theorem [1], a matching 𝑀 is maximum if and only if there is no augmenting
path concerning 𝑀 . Moreover, whenever an augmenting path is found, its edges can be reversed (the
matched edges become unmatched and vice versa), and a new matching of one larger size than 𝑀 is
found.
Blum’s algorithm formulates the maximum matching problem as a reachability problem in an induced
directed bipartite graph 𝐺′ , constructed from the given undirected graph 𝐺 = (𝑉, 𝐸), and a given
matching 𝑀 ⊆ 𝐸.
For each vertex 𝑣 ∈ 𝑉 , two vertices 𝑣𝐵 and 𝑣𝐴 are constructed, effectively making a copy of the set
of vertices and generating a bipartition 𝐵 ∪ 𝐴. The key concept utilized in this formulation is that of
the strongly simple path, a directed path in 𝐺′ with the following conditions:
1. Has no pair of vertices 𝑣𝐵 , 𝑣𝐴 (having the same original vertex 𝑣).
2. Is simple (no repeated vertices).
3. Follows edge direction rules expressing the existing matching status.
For each undirected edge {𝑢, 𝑣} ∈ 𝐸, the transformation adds directed edges to 𝐺′ depending on
whether the edge is part of the matching 𝑀 :
• If {𝑢, 𝑣} ∈
/ 𝑀 , two forward edges are added: 𝑢𝐵 → 𝑣𝐴 and 𝑣𝐵 → 𝑢𝐴 . These represent steps
along unmatched edges and are directed from 𝐵 to 𝐴.
• If {𝑢, 𝑣} ∈ 𝑀 , two backward edges are added: 𝑢𝐴 → 𝑣𝐵 and 𝑣𝐴 → 𝑢𝐵 . These represent matched
edges and are directed from 𝐴 to 𝐵.
This construction ensures that traversing the 𝐴 → 𝐵 edges corresponds to traversing the existing
matching, and that the 𝐵 → 𝐴 edges represent candidate moves toward establishing new matches. A
Modified Depth-First Search (MDFS) on 𝐺′ is started from a source vertex 𝑠 connecting all free vertices
of side 𝐵, and attempts to find a strongly simple path to a sink vertex 𝑡 connecting all free vertices of
side 𝐴. Upon finding such a path, it translates to an alternating path in the original graph 𝐺 that can be
applied to augment the existing matching. In each augmentation that succeeds, the size of the matching
is incremented by one [8]. The MDFS procedure uses strict rules on labeling to avoid visiting both 𝑣𝐵 ,
𝑣𝐴 for any 𝑣 ∈ 𝑉 , thus maintaining correctness throughout the search.
In practice, the edges traversed by MDFS can be classified into five categories: tree edges, forward
edges, cross edges, back edges, and weak back edges [8]. These categories determine how the search
progresses and how labeling and path constraints are maintained during execution. Blum’s formulation
provides a structured decision process for classifying edges [8]. MDFS constructs the MDFS-Tree 𝑇 .
For the construction, a stack 𝐾 is used that contains the vertices from the root of 𝑇 to the currently
visited vertex. The top element of the stack, denoted 𝑇 𝑂𝑃 (𝐾), guides the exploration. At each step,
the algorithm inspects an edge of the form (𝑇 𝑂𝑃 (𝐾), 𝑤𝑋 ), where 𝑋 ∈ {𝐴, 𝐵}, that has not yet been
examined. Assuming an edge (𝑣𝑋 , 𝑤𝑋 ) is under consideration, where the overline denotes the opposite
label, i.e., 𝑣𝐴 = 𝑣𝐵 and 𝑣𝐵 = 𝑣𝐴 , the edge cases are summarized as follows [8]:
Case 1: Tree edge — 𝑋 = 𝐴 and (𝑣, 𝑤) ∈ 𝑀
Case 2: 𝑋 = 𝐵 and (𝑣, 𝑤) ∈ 𝐸 ∖ 𝑀
2.1: Back edge; 𝑤𝐴 ∈ 𝐾
2.2: 𝑤𝐴 ∈
/ 𝐾, 𝑤𝐵 ∈ 𝐾
i. weak back edge; 𝑤𝐴 has been in 𝐾 previously
ii. weak back edge; 𝑤𝐴 has not been in 𝐾 previously
2.3: 𝑤𝐴 ∈
/ 𝐾, 𝑤𝐵 ∈
/𝐾
i. Forward or cross edge; 𝑤𝐴 has been in 𝐾 previously
ii. Tree edge; 𝑤𝐴 has not been in 𝐾 previously



---

While implementing the algorithm, whenever a vertex 𝑢𝐵 is popped from the stack, the algorithm
looks for reachable vertices 𝑣𝐴 from where it can further build a valid strongly simple path. These
vertices are then stored in the set:
𝐿𝑣𝐴 := {𝑢𝐴 ∈ 𝑉 ′ | ∃ path 𝑃 = (𝑣𝐴 , 𝑄, 𝑢𝐴 ) ∧ 𝑢𝐵 ∈
/𝑄
∧ PUSH(𝑢𝐴 ) has never been performed
∧ POP(𝑢𝐵 ) has been performed }
According to Blum’s [8], a vertex is pushed onto the stack in three situations: Case 1, Case 2.3.ii,
and Case 2.3.i when 𝐿𝑣𝐴 = 𝑢𝐴 . In the latter case, the vertex at the top of the stack is updated by
introducing an extensible vertex 𝑇 𝑂𝑃 (𝐾)[𝑣𝐴 ] , and we push 𝑢𝐴 onto the stack. Consequently, we obtain
an extensible edge (𝑇 𝑂𝑃 (𝐾), 𝑢𝐴 )[𝑣𝐴 ] .
We also rely on the following data structures introduced by Blum [8]:
𝑅𝑢𝐴 := {𝑣𝐵 ∈ 𝑉 ′ | (𝑣𝐵 , 𝑢𝐴 ) is a weak back edge}
𝐸𝑞𝐴 := {𝑣𝐵 ∈ 𝑉 ′ | (𝑣𝐵 , 𝑞𝐴 ) is a cross, forward, or back edge}
𝐷𝑞𝐴 := {𝑝𝐴 ∈ 𝑉 ′ | 𝐿𝑝𝐴 = 𝑞𝐴 previously}
We define the expanded MDFS-tree 𝑇𝑒𝑥𝑝 as the tree obtained from the constructed MDFS-tree 𝑇 by
adding all forward, back, cross, and weak back edges, together with every extensible edge. To determine
the vertices in 𝐿 when 𝑢𝐵 is popped, a backward search is performed on 𝑇𝑒𝑥𝑝 . This search is carried
out using a standard graph traversal, such as depth-first search, starting from vertex 𝑢𝐴 and exploring
the considered edges in reverse until 𝑢𝐵 is reached.
To support the reconstruction of a strongly simple path when reaching 𝑡, the algorithm maintains a
variable 𝑃𝑣𝐴 that records the most recent non-tree edge terminating at vertex 𝑣𝐴 .

2.2. Observed Cases and Modifications
While implementing Blum’s MDFS algorithm [8], we identified that certain cases – particularly 2.2.i and
2.3.i when 𝐿𝑤𝐴 = ∅ – were not handled robustly in the original formulation. Although such cases are
rare in typical graphs, ignoring them may cause the algorithm to skip essential edge relationships, fail
to update sets, or even lose parts of valid augmenting paths. As a result, the algorithm risks violating
the constraint of a strongly simple path or failing to detect an augmenting path when one exists. To
ensure correctness and stability across all graph instances, we introduced selective modifications in
the traversal mechanism. The examples below illustrate the specific problems we encountered and the
adaptations we employed to rectify them.
2.2.1. Case 2.2.i, Weak back edge
When the traversal encounters a weak back edge (𝑣𝐵 , 𝑢𝐴 ) – classified as Case 2.2.i – the original
algorithm does not perform any action. Specifically, it does not update 𝐷𝑢𝐴 and 𝑃𝑤𝐴 , such that there is
a path 𝑃 = 𝑤𝐴 , 𝑄, 𝑢𝐴 and 𝑢𝐵 ∈
/ 𝑄 has been found by the MDFS, even though the edge may play a
structurally important role in the reconstruction of a strongly simple path.
Updating the value of 𝑃𝑤𝐴 is essential for correctly reconstructing the augmenting path from 𝑠 to 𝑡.
Without this update, the algorithm may terminate prematurely or fail to identify an existing augmenting
path, violating its correctness guarantee in these specific configurations.
To solve this issue, whenever the algorithm encounters a weak back edge (𝑣𝐵 , 𝑢𝐴 ), we add 𝑣𝐵 to
𝑅𝑢𝐴 . This operation ensures that structurally significant relationships are preserved for future use. In
order to improve the correctness and efficiency of this operation, we selectively choose which vertices
are added based on the traversal history.
Specifically, we allow 𝑣𝐵 to be added to 𝑅𝑢𝐴 under either of the following two situations:
• 𝑣𝐴 has not yet been visited by MDFS
• 𝑤𝐵 was pushed onto the stack before 𝑣𝐴



---

(a) Graph 𝐺 with current matching (shown in red)

(b) Graph 𝐺 with new matching after successful
augmentation (shown in red)

(c) Tree 𝑇𝑒𝑥𝑝 built by Blum’s MDFS. Edges to green vertices represent weak back edges from Case 2.2.ii, edges
to purple vertices represent a forward or cross edge (Case 2.3.i) used to connect an extensible vertex via an
extensible edge

(d) Corrected tree 𝑇𝑒𝑥𝑝 with our fix. Edges to green vertices represent weak back edges from Case 2.2.ii, edges to
blue vertices correspond to Case 2.2.i, edges to purple vertices represent a forward or cross edge (Case 2.3.i)
used to connect an extensible vertex via an extensible edge
Figure 1: Illustration of Case 2.2.1: Weak back edge. Our modification recovers a valid augmenting path missed
by Blum’s original MDFS.

After the execution of 𝑃 𝑂𝑃 (𝑢𝐵 ), there is a backward traversal of the current expanded tree 𝑇𝑒𝑥𝑝
from the vertex 𝑢𝐴 along edges in the opposite direction. Whenever a vertex 𝑤𝐴 is visited through this
traversal, it is added to the set 𝐷𝑢𝐴 , but no value is given to 𝐿𝑤𝐴 .
The behavior of this case is illustrated in Figure 1. The original graph 𝐺, along with its current
matching, is presented in Figure 1a.
The tree 𝑇𝑒𝑥𝑝 constructed by Blum’s original MDFS is presented in Figure 1c, When 𝑡 is found then
𝑃8𝐴 = (9𝐵 , 7𝐴 ), 𝑃9𝐴 = (8𝐵 , 7𝐴 ), 𝑃4𝐴 = (5𝐵 , 3𝐴 ), 𝑃5𝐴 = (4𝐵 , 3𝐴 ), 𝑃6𝐴 = (8𝐵 , 3𝐴 ). To reconstruct



---

the strongly simple path from 𝑠 to 𝑡, we start with the last vertex 𝑡 and follow the parents in 𝑇𝑒𝑥𝑝 . We
obtain the path segment 3𝐴 → 2𝐵 → 10𝐴 → 𝑡.
At this point, the parent of 3𝐴 in 𝑇𝑒𝑥𝑝 is an extensible vertex 1𝐵 [9𝐴 ] , meaning it was added via Case
2.3.i, using an extensible edge (1𝐵 , 3𝐴 )[9𝐴 ] . Now we have 𝑃9𝐴 = (8𝐵 , 7𝐴 ). So we reconstruct the path
from 9𝐴 to 8𝐵 : 9𝐴 → 8𝐵 , next we search for 𝑃7𝐴 . Since 𝑃7𝐴 is undefined, the reconstruction process
fails at this point.
In contrast, our modified version of the algorithm builds the tree 𝑇𝑒𝑥𝑝 shown in Figure 1d. First
we pop 7𝐵 , and then we get 𝑃8𝐴 = (9𝐵 , 7𝐴 ), 𝑃9𝐴 = (8𝐵 , 7𝐴 ), next we pop 4𝐵 , and then we get
𝑃6𝐴 = (7𝐵 , 4𝐴 ), 𝑃7𝐴 = (6𝐵 , 4𝐴 ), next we pop 3𝐵 , and we get 𝑃4𝐴 = (5𝐵 , 3𝐴 ), 𝑃5𝐴 = (4𝐵 , 3𝐴 ),
after that we find 𝑡. The reconstruction of the strongly simple path from 𝑠 to 𝑡 starts with the last vertex
𝑡, and it follows the parents in 𝑇𝑒𝑥𝑝 . It obtains the path segment 3𝐴 → 2𝐵 → 10𝐴 → 𝑡.
At this point, the parent of 3𝐴 in 𝑇𝑒𝑥𝑝 is an extensible vertex 1𝐵 [9𝐴 ] , meaning it was added via
Case 2.3.i, using an extensible edge (1𝐵 , 3𝐴 )[9𝐴 ] . Now we have 𝑃9𝐴 = (8𝐵 , 7𝐴 ). We reconstruct the
path from 9𝐴 to 8𝐵 : 9𝐴 → 8𝐵 . Then we have 𝑃7𝐴 = (6𝐵 , 4𝐴 ), so we reconstruct path from 7𝐴 to 6𝐵 :
7𝐴 → 6𝐵 . Then we have 𝑃4𝐴 = (5𝐵 , 3𝐴 ), so we reconstruct path from 4𝐴 to 5𝐵 : 4𝐴 → 5𝐵 . Then we
get 3𝐴 , so we continue from 1𝐵 : 𝑠 → 1𝐵 . The strongly simple path will be:
𝑠 → 1𝐵 → 9𝐴 → 8𝐵 → 7𝐴 → 6𝐵 → 4𝐴 → 5𝐵 → 3𝐴 → 2𝐵 → 10𝐴 → 𝑡
The outcome is shown in Figure 1b, where the algorithm correctly finds an augmenting path and
increases the cardinality of the matching by one. The example confirms the need to handle weak back
edges differently to ensure MDFS correctness for any case.
2.2.2. Case 2.3.i, cross or forward edge
When the traversal encounters a forward edge (𝑣𝐵 , 𝑢𝐴 ) of Case 2.3.i with 𝐿𝑢𝐴 = ∅, the original
algorithm does nothing. This may result in the absence of valid path segments and, therefore, incomplete
augmentation or reconstruction.
To address this issue, the algorithm records each forward or cross edge when 𝐿𝑢𝐴 = ∅ and adds it to
a new set WC:
𝑊 𝐶𝑢𝐴 := { 𝑣𝐵 ∈ 𝑉 ′ | (𝑣𝐵 , 𝑢𝐴 ) is a forward or cross edge and 𝐿𝑢𝐴 = ∅ }
To improve the efficiency of this operation, we include 𝑣𝐵 in 𝑊 𝐶𝑢𝐴 subject to the condition that
either of the following holds in the extended tree:
• 𝑢𝐵 is a parent of 𝑣𝐴
• 𝑢𝐴 and 𝑣𝐴 are not on the same path
The behavior of this case is illustrated in Figure 2. The original graph 𝐺, along with its current
matching, is presented in Figure 2a.
The tree 𝑇𝑒𝑥𝑝 constructed by Blum’s original MDFS is presented in Figure 2c, When 𝑡 is found,
then 𝑃10𝐴 = (11𝐵 , 9𝐴 ), 𝑃11𝐴 = (10𝐵 , 9𝐴 ), 𝑃6𝐴 = (7𝐵 , 5𝐴 ), 𝑃7𝐴 = (6𝐵 , 5𝐴 ), 𝑃5𝐴 = (4𝐵 , 3𝐴 ),
𝑃4𝐴 = (7𝐵 , 3𝐴 ), 𝑃8𝐴 = (9𝐵 , 4𝐴 ). To reconstruct the strongly simple path from 𝑠 to 𝑡, we start with
the last vertex 𝑡 and follow the parents in 𝑇𝑒𝑥𝑝 . We obtain the path segment 3𝐴 → 2𝐵 → 12𝐴 → 𝑡.
At this point, the parent of 3𝐴 is an extensible vertex 1𝐵 [11𝐴 ] , meaning it was added via Case 2.3.i,
using an extensible edge (1𝐵 , 3𝐴 )[11𝐴 ] . Now we have 𝑃11𝐴 = (10𝐵 , 9𝐴 ), so we reconstruct path from
11𝐴 to 10𝐵 : 11𝐴 → 10𝐵 , next we search for 𝑃9𝐴 . Since 𝑃9𝐴 is undefined, the reconstruction process
fails at this point.
In contrast, our modified version of the algorithm builds 𝑇𝑒𝑥𝑝 as shown in Figure 2d. First we
pop 9𝐵 , and then we get 𝑃10𝐴 = (11𝐵 , 9𝐴 ), 𝑃11𝐴 = (10𝐵 , 9𝐴 ). Next we pop 5𝐵 , and then we get
𝑃6𝐴 = (7𝐵 , 5𝐴 ), 𝑃7𝐴 = (6𝐵 , 5𝐴 ). Next we pop 3𝐵 , and we get 𝑃4𝐴 = (7𝐵 , 3𝐴 ), 𝑃5𝐴 = (4𝐵 , 3𝐴 ),
𝑃8𝐴 = (9𝐵 , 4𝐴 ), 𝑃9𝐴 = (8𝐵 , 6𝐴 ). After that, we find 𝑡. The reconstruction of the strongly simple



---

(a) Graph 𝐺 with current matching (shown in red)

(b) Graph 𝐺 with new matching after successful
augmentation (shown in red)

(c) Tree 𝑇𝑒𝑥𝑝 built by Blum’s MDFS, edges to green vertices represent weak back edges, edges to red vertices
represent back, cross, or forward edges, edges to purple vertices represent a forward or cross edge (Case 2.3.i)
used to connect an extensible vertex via an extensible edge.

(d) Corrected tree 𝑇𝑒𝑥𝑝 with our fix. Edges to green vertices represent weak back edges, edges to red vertices
represent back, cross, or forward edges. Edges to blue vertices correspond to Case 2.3.i when 𝐿𝑢𝐴 = ∅. Edges
to purple vertices represent a forward or cross edge (Case 2.3.i) used to connect an extensible vertex via an
extensible edge
Figure 2: Illustration of Case 2.3.1: Forward or cross edge when 𝐿𝑢𝐴 = ∅. Our modification recovers a valid
augmenting path missed by Blum’s original MDFS.

path from 𝑠 to 𝑡 starts with the last vertex 𝑡 and follows the parents in 𝑇𝑒𝑥𝑝 . It obtains the path segment
3𝐴 → 2𝐵 → 12𝐴 → 𝑡.
At this point,the parent of 3𝐴 is an extensible vertex 1𝐵 [11𝐴 ] , meaning it was added via Case 2.3.i,
using an extensible edge (1𝐵 , 3𝐴 )[11𝐴 ] . Now we have 𝑃11𝐴 = (10𝐵 , 9𝐴 ), so we reconstruct path from
11𝐴 to 10𝐵 : 11𝐴 → 10𝐵 , then we have 𝑃9𝐴 = (8𝐵 , 6𝐴 ), so we reconstruct path from 9𝐴 to 8𝐵 :
9𝐴 → 8𝐵 . Then we have 𝑃6𝐴 = (7𝐵 , 5𝐴 ), so we reconstruct path from 6𝐴 to 7𝐵 : 6𝐴 → 7𝐵 . Next
we have 𝑃5𝐴 = (4𝐵 , 3𝐴 ), so we reconstruct path from 5𝐴 to 4𝐵 : 5𝐴 → 4𝐵 . Then we get 3𝐴 , so we
continue from 1𝐵 : 𝑠 → 1𝐵 Finally, the strongly simple path will be:
𝑠 → 1𝐵 → 11𝐴 → 10𝐵 → 9𝐴 → 8𝐵 → 6𝐴 → 7𝐵 → 5𝐴 → 4𝐵 → 3𝐴 → 2𝐵 → 12𝐴 → 𝑡



---

The outcome is shown in Figure 2b, where the algorithm correctly finds an augmenting path and
increases the cardinality of the matching by one. The example confirms the need to handle forward or
cross edges differently when 𝐿𝑤𝐴 = ∅ to ensure MDFS correctness for any case.

3. Experimental Results
In this section, we present a comparative analysis of the performance of three maximum matching
algorithms: Blossom I, Blossom V, and Blum’s MDFS algorithm. Our experiments are divided into two
stages. In the first stage, we compare Blossom I and Blossom V using a diverse set of unweighted,
undirected graphs. This comparison highlights the effect of algorithmic optimizations and internal data
structure choices, such as queue-based event handling, on runtime performance.
In the second stage, we evaluate Blum’s MDFS algorithm against Blossom V. We test two versions of
Blum’s algorithm: one running from scratch, and the other initialized with a random greedy matching
(denoted as Blum Preparatory), which processes the edges in a random order and adds the edge to
the matching iff both end vertices are unmatched. This variation allows us to assess how providing
an initial matching affects the performance and convergence speed of MDFS. The initial matching is
applied only to MDFS, as Blossom V already employs a greedy initialization [5]. We then compare the
results across both variants and analyze how well MDFS performs relative to Blossom V in terms of
total runtime under different graph topologies. In all of our experiments, all algorithms – Blossom I,
Blossom V, Blum, and Blum Preparatory – consistently produced maximum matchings of the same size.
All the experiments were performed on a computer with an Intel(R) Core(TM) i7-7700HQ CPU
(2.80GHz), 16GB RAM, under Windows 10 (64-bit). The code was written in Java and executed under
the NetBeans 20.0 integrated development environment.
The Java runtime environment was Java 21.0.2 (LTS), provided by Oracle Corporation, under the
Java HotSpot™ 64-Bit Server VM (version 21.0.2+13-LTS-58). Run times were gathered with System.currentTimeMillis(), and we only counted the time taken in the core matching function. The
construction of the bipartite graph and the initialization of the random matching were included in the
time measurements, while I/O accesses and graph construction were excluded.
We evaluated the algorithms on eight distinct types of problem instances to ensure diversity in both
structure and complexity. Several of these instance generators incorporate randomness; in such cases,
we report the average runtime over 𝑡 independent runs using different random seeds. The value of 𝑡
is indicated in the figure captions. All graphs are described by their number of vertices 𝑛, which are
shown along the horizontal axes of the runtime plots.
Delaunay triangulations: We generated 𝑛 points at random with a uniform distribution in a 220 × 220
square and then computed the Delaunay triangulation of the point set. For constructing Delaunay
triangulations, we employed a Java procedure that conformed to the key ideas and geometric ideas of
Shewchuk’s 2D Delaunay triangulation algorithm [11].
Erdős-Rényi Random Graphs: We generated random graphs with 𝑛 vertices and 𝑚 edges by randomly selecting pairs of distinct vertices. We examined two cases 𝑚 = 6𝑛 and 𝑚 = 80𝑛.
Complete Graphs: We also included full graphs in our set of benchmarks, where every graph has all
the possible edges between 𝑛 vertices. The number of edges in those graphs is 𝑚 = 𝑛(𝑛−1)
.
2
DIMACS Benchmark Graphs: We used families of examples from the First DIMACS Implementation
Challenge [12], which are widely employed to test the performance of matching algorithms in challenging conditions. We used particularly the following three generators available in the benchmark
package:
• hardcard (hardcard.f): These graphs were first examined by Gabow and are proven to be challenging for Edmonds-type algorithms; they are constructed to enforce worst-case situations in
blossom shrinking.
• T (t.f) and TT (tt.f): These generators create sequences of one-connected and tri-connected
triangles, respectively (see [5]).



---

Each instance is parameterized by an integer 𝐾, which determines the size and complexity of the
generated graph.
Overlapping Cycles: For testing the resilience of the algorithms, we built our own generator that
generates graphs composed of overlapping cycles of odd and even size. These instances are specifically
designed in a way that would challenge matching algorithms to their limits by inserting complex
structures such as nested and intersecting cycles. The presence of crossing odd-length cycles is hardest
for blossom-based methods, while even-length cycles test the algorithm to detect and follow alternative
paths. These are structurally dense test cases that put a strain on the limits of contraction, labeling, and
path-reconstruction logic and reveal a lot about the correctness and performance of the algorithm in
adversarial-like instances.

3.1. Comparison of Blossom I and Blossom V
We compared Blossom I and Blossom V’s run-time behavior on several graph families, namely, Delaunay
triangulations, random graphs, complete graphs, overlapping cycles, and instances of the DIMACS
benchmarks. In all cases, Blossom V behaves uniformly better than Blossom I.
Figure 3 summarizes the results, where the running time is plotted on a logarithmic scale to clearly
illustrate performance differences of the algorithms. Blossom I exhibits explosive growth in running
time with graph size, particularly for dense graphs. In contrast, Blossom V has reliable and efficient
performance since it utilizes priority queues, optimized blossom management, and improved data
structures.

3.2. Comparison of Blossom V, Blum, and Blum Preparatory
This section compares the performance of three algorithms: Blossom V, Blum’s MDFS, and Blum
Preparatory, a variant of MDFS that starts with an initial matching. Our goal is to assess both the raw
performance and the practical implications of Blum’s avoidance of blossom shrinking, especially when
combined with a random greedy initial matching. Figure 4 summarizes the results, where the running
time is plotted on a logarithmic scale. Figure 4a shows the results for Delaunay triangulations. Here,
the standard Blum code is consistently slower than both competitors. Nevertheless, Blum Preparatory
performs much better and often outperforms Blossom V on instances of mid-to-large size. This result
shows the strength of MDFS when combined with a precomputed matching.
On random graphs with 𝑚 = 6𝑛 (Figure 4b), Blossom V is always the fastest algorithm. While in
random graphs with 𝑚 = 80𝑛 (Figure 4c), all algorithms perform reasonably well, but the preparatory
version of Blum consistently tracks or slightly outperforms Blossom V, particularly as 𝑛 increases. This
suggests that the combination of simplicity and a warm start is effective.
The results on complete graphs (Figure 4d) are more predictable. Blum Preparatory handles the
complete graphs better overall.
The results on overlapping cycle instances are shown in Figure 4e. Here, the performance gap between
Blum and Blum Preparatory is significant, demonstrating the burden of full search in blossom-heavy
graphs. Blum Preparatory is very competitive with Blossom V.
Figures 4f to 4h show the results on DIMACS graphs. As sample cases, these are chosen to be
challenging, with wild blossom production. These instances also represent worst-case scenarios for
MDFS due to the existence of numerous alternative paths. Blossom V performs best on the hardcard
graph instances, and Blum Preparatory is the fastest on the T and TT instances.
Summary: Blossom V has the best overall runtime for the most graph families. Blum Preparatory is
the fastest on complete graphs, T, and TT instances of the DIMACS graphs. It is highly competitive
in many structured graphs. It does not require blossom shrinking to beat or nearly match Blossom V.
These results suggest that, in real-world scenarios where an approximation for a matching is known or
is inexpensively computable, MDFS-based algorithms represent a good and clean alternative.



---

(a) Delaunay triangulation graphs (𝑡 = 50)

(b) Random graphs (𝑡 = 50 and 𝑚 = 6𝑛)

(c) Random graphs (𝑡 = 50 and 𝑚 = 80𝑛)

(d) Complete graphs

(e) Overlapping cycles (𝑡 = 50)

(f) DIMACS hardcard graphs

(g) DIMACS T graphs

(h) DIMACS TT graphs

Figure 3: Runtime comparison between Blossom I and Blossom V across different graph families. The runtime
is given in milliseconds.

4. Conclusion
We gave a comparative analysis of the three maximum matching algorithms in unweighted, undirected
graphs: Blossom I, Blossom V, and Blum’s MDFS algorithm, along with a version that initializes
MDFS with a random greedy matching. We compared both theoretical and experimental differences of
performance over a large set of graph families: random graphs, geometric graphs, complete graphs,
DIMACS benchmarks, and overlapping cycles.
The results show that Blossom V performs substantially better than Blossom I due to improved
data structures and heuristics. More importantly, our results show that Blum Preparatory, due to not



---

(a) Delaunay triangulations (𝑡 = 50)

(b) Random graphs (𝑡 = 50 and 𝑚 = 6𝑛)

(c) Random graphs (𝑡 = 50 and 𝑚 = 80𝑛)

(d) Complete graphs

(e) Overlapping cycles (𝑡 = 50)

(f) DIMACS hardcard graphs

(g) DIMACS T graphs

(h) DIMACS TT graphs

Figure 4: Runtime comparison of Blossom V, Blum MDFS, and Blum Preparatory across diverse graph families.
The runtime is given in milliseconds.

being complex and blossom shrinking overhead free, can achieve highly competitive performance, even
beating Blossom V in some cases, particularly for structured and sparse graphs. This confirms the
practical applicability of reachability-based matching when given a good initial matching.
We implemented of Blum’s MDFS in Java and proposed some corrections for the previously found
edge cases, rendering it correct and more versatile. Not only do these modifications enhance Blum’s
algorithm for real-world applications, but they also shed light on how initialization and structural
characteristics of the input graph affect the efficiency of matching.



---

In conclusion, the paper confirms the effectiveness of augmenting path algorithms and draws out the
inner strength of other techniques like MDFS. Some avenues for research work may include exploring
hybrid techniques, parallel algorithms, and expanding analysis to the dynamic, weighted graphs or
maximum 2-matching problems.

Declaration on Generative AI
During the preparation of this work, the author(s) used Grammarly solely for grammar and spelling
correction. After using this tool, the author(s) carefully reviewed and edited the content as needed and
take(s) full responsibility for the publication’s content.

References
[1] C. Berge, Two theorems in graph theory, Proceedings of the National Academy of Sciences 43
(1957) 842–844.
[2] J. E. Hopcroft, R. M. Karp, An nˆ5/2 algorithm for maximum matchings in bipartite graphs, SIAM
Journal on computing 2 (1973) 225–231.
[3] J. Edmonds, Paths, trees, and flowers, Canadian Journal of mathematics 17 (1965) 449–467.
[4] J. Edmonds, E. L. Johnson, S. C. Lockhart, Blossom i: a computer code for the matching problem,
IBM TJ Watson Research Center, Yorktown Heights, New York 294 (1969).
[5] V. Kolmogorov, Blossom v: a new implementation of a minimum cost perfect matching algorithm,
Mathematical Programming Computation 1 (2009) 43–67.
[6] W. Cook, A. Rohe, Computing minimum-weight perfect matchings, INFORMS journal on computing 11 (1999) 138–148.
[7] N. Blum, A new approach to maximum matching in general graphs, in: Automata, Languages and
Programming, 17th International Colloquium, ICALP90, volume 443 of LNCS, Springer, 1990, pp.
586–597.
[8] N. Blum, Maximum matching in general graphs without explicit consideration of blossoms revisited, arXiv preprint arXiv:1509.04927, Updated version: https://theory.cs.unibonn.de/blum/papers/gmatching.pdf (2015).
[9] K. Schwarz, Edmonds’ matching algorithm code, https://www.keithschwarz.com/interesting/code/
?dir=edmonds-matching.
[10] JGraphT Project, Jgrapht - a java graph library, https://jgrapht.org.
[11] J. R. Shewchuk, Triangle: Engineering a 2d quality mesh generator and delaunay triangulator, in:
Workshop on applied computational geometry, Springer, 1996, pp. 203–222.
[12] D. S. Johnson, C. C. McGeoch, et al., Network flows and matching: first DIMACS implementation
challenge, volume 12, American Mathematical Soc., 1993.



---
