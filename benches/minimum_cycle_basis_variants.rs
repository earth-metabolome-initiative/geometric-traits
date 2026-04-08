//! Criterion benchmarks comparing candidate minimum-cycle-basis
//! implementations.

#[path = "../tests/support/minimum_cycle_basis_fixture.rs"]
mod minimum_cycle_basis_fixture;

use std::{collections::BinaryHeap, hint::black_box, time::Duration};

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use geometric_traits::{
    impls::{CSR2D, SortedVec, SymmetricCSR2D},
    prelude::*,
    traits::{
        MonoplexGraph, SquareMatrix, VocabularyBuilder,
        algorithms::randomized_graphs::{cycle_graph, friendship_graph, grid_graph, path_graph},
    },
};
use minimum_cycle_basis_fixture::{build_undigraph, load_fixture_suite};

const FIXTURE_NAME: &str = "minimum_cycle_basis_networkx_1000.json.gz";

type UndirectedAdjacency = SymmetricCSR2D<CSR2D<usize, usize, usize>>;

#[derive(Clone)]
struct FixtureBenchCase {
    name: String,
    graph: UndiGraph<usize>,
    expected_basis: Vec<Vec<usize>>,
}

#[derive(Clone)]
struct ScalingBenchCase {
    name: String,
    logical_edge_count: usize,
    graph: UndiGraph<usize>,
}

#[derive(Clone, Copy, Debug)]
enum Variant {
    BaselinePublic,
    ParentsMaterialized,
    ParentsImplicit,
}

impl Variant {
    fn label(self) -> &'static str {
        match self {
            Self::BaselinePublic => "baseline_public",
            Self::ParentsMaterialized => "parents_materialized",
            Self::ParentsImplicit => "parents_implicit",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct BenchBasisResult {
    basis: Vec<Vec<usize>>,
    cycle_rank: usize,
    total_weight: usize,
}

impl BenchBasisResult {
    fn checksum(&self) -> u64 {
        self.basis
            .iter()
            .fold(0u64, |checksum, cycle| {
                cycle.iter().fold(checksum.wrapping_mul(1_099_511_628_211), |acc, &node| {
                    acc.wrapping_mul(257).wrapping_add(node as u64 + 1)
                })
            })
            .wrapping_add(self.cycle_rank as u64)
            .wrapping_add((self.total_weight as u64) << 32)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct LocalGraph {
    vertices: Vec<usize>,
    edges: Vec<[usize; 2]>,
    adjacency: Vec<Vec<usize>>,
    edge_lookup: Vec<usize>,
}

type MeetingState = (usize, usize, Vec<Option<usize>>, Vec<Option<usize>>);

impl LocalGraph {
    fn from_component(vertices: Vec<usize>, edges: Vec<[usize; 2]>, total_nodes: usize) -> Self {
        let mut global_to_local = vec![usize::MAX; total_nodes];
        for (local_id, &vertex) in vertices.iter().enumerate() {
            global_to_local[vertex] = local_id;
        }

        let order = vertices.len();
        let mut local_edges = Vec::with_capacity(edges.len());
        let mut adjacency = vec![Vec::new(); order];
        let mut edge_lookup = vec![usize::MAX; order * order];

        for [left, right] in edges {
            let left_local = global_to_local[left];
            let right_local = global_to_local[right];
            let edge_id = local_edges.len();
            local_edges.push([left_local, right_local]);
            adjacency[left_local].push(right_local);
            adjacency[right_local].push(left_local);
            edge_lookup[left_local * order + right_local] = edge_id;
            edge_lookup[right_local * order + left_local] = edge_id;
        }

        Self { vertices, edges: local_edges, adjacency, edge_lookup }
    }

    #[inline]
    fn order(&self) -> usize {
        self.vertices.len()
    }

    #[inline]
    fn edge_count(&self) -> usize {
        self.edges.len()
    }

    #[inline]
    fn cycle_rank(&self) -> usize {
        self.edge_count() - self.order() + 1
    }

    #[inline]
    fn edge_id(&self, left_local: usize, right_local: usize) -> usize {
        self.edge_lookup[left_local * self.order() + right_local]
    }
}

fn wrap_undi(graph: UndirectedAdjacency) -> UndiGraph<usize> {
    let order = graph.order();
    let nodes: SortedVec<usize> = GenericVocabularyBuilder::default()
        .expected_number_of_symbols(order)
        .symbols((0..order).enumerate())
        .build()
        .unwrap();
    UndiGraph::from((nodes, graph))
}

fn logical_edge_count(graph: &UndiGraph<usize>) -> usize {
    graph.sparse_coordinates().filter(|&(source, destination)| source <= destination).count()
}

fn normalize_cycles(mut cycles: Vec<Vec<usize>>) -> Vec<Vec<usize>> {
    for cycle in &mut cycles {
        *cycle = normalize_cycle(cycle.clone());
    }
    cycles
        .sort_unstable_by(|left, right| left.len().cmp(&right.len()).then_with(|| left.cmp(right)));
    cycles
}

fn normalize_cycle(mut cycle: Vec<usize>) -> Vec<usize> {
    if cycle.is_empty() {
        return cycle;
    }
    let start =
        cycle.iter().enumerate().min_by_key(|(_, node)| **node).map_or(0, |(index, _)| index);
    cycle.rotate_left(start);
    if cycle.len() > 2 && cycle[cycle.len() - 1] < cycle[1] {
        cycle[1..].reverse();
    }
    cycle
}

fn prepare_fixture_cases(relative_path: &str) -> Vec<FixtureBenchCase> {
    let suite = load_fixture_suite(relative_path);
    suite
        .cases
        .into_iter()
        .map(|case| {
            FixtureBenchCase {
                name: case.name.clone(),
                graph: build_undigraph(&case),
                expected_basis: normalize_cycles(case.minimum_cycle_basis),
            }
        })
        .collect()
}

fn prepare_scaling_cases() -> Vec<ScalingBenchCase> {
    [
        ("path_1024", wrap_undi(path_graph(1024))),
        ("cycle_256", wrap_undi(cycle_graph(256))),
        ("grid_12x12", wrap_undi(grid_graph(12, 12))),
        ("friendship_64", wrap_undi(friendship_graph(64))),
    ]
    .into_iter()
    .map(|(name, graph)| {
        ScalingBenchCase {
            name: name.to_string(),
            logical_edge_count: logical_edge_count(&graph),
            graph,
        }
    })
    .collect()
}

fn evaluate_variant(variant: Variant, graph: &UndiGraph<usize>) -> BenchBasisResult {
    match variant {
        Variant::BaselinePublic => {
            let result = graph.minimum_cycle_basis().unwrap();
            BenchBasisResult {
                basis: normalize_cycles(result.minimum_cycle_basis().cloned().collect()),
                cycle_rank: result.cycle_rank(),
                total_weight: result.total_weight(),
            }
        }
        Variant::ParentsMaterialized => exact_mcb_variant(graph, false),
        Variant::ParentsImplicit => exact_mcb_variant(graph, true),
    }
}

fn exact_mcb_variant(graph: &UndiGraph<usize>, implicit_lifted: bool) -> BenchBasisResult {
    let components = graph.connected_components().unwrap();
    let mut basis = Vec::new();

    let number_of_components: usize = components.number_of_components();
    for component_id in 0..number_of_components {
        let mut vertices = components.node_ids_of_component(component_id).collect::<Vec<_>>();
        if vertices.len() < 3 {
            continue;
        }
        vertices.sort_unstable();

        let edges = collect_component_edges_dense(graph, &vertices);
        if edges.len() < vertices.len() {
            continue;
        }

        let component = LocalGraph::from_component(vertices, edges, graph.number_of_nodes());
        basis.extend(minimum_cycle_basis_for_component_variant(&component, implicit_lifted));
    }

    basis
        .sort_unstable_by(|left, right| left.len().cmp(&right.len()).then_with(|| left.cmp(right)));
    let total_weight = basis.iter().map(Vec::len).sum::<usize>();
    let cycle_rank = basis.len();
    BenchBasisResult { basis, cycle_rank, total_weight }
}

fn collect_component_edges_dense(graph: &UndiGraph<usize>, vertices: &[usize]) -> Vec<[usize; 2]> {
    let mut vertex_set = vec![false; graph.number_of_nodes()];
    for &vertex in vertices {
        vertex_set[vertex] = true;
    }

    let mut edges = Vec::new();
    for &left in vertices {
        for right in graph.neighbors(left) {
            if left < right && vertex_set[right] {
                edges.push([left, right]);
            }
        }
    }
    edges.sort_unstable();
    edges
}

fn minimum_cycle_basis_for_component_variant(
    component: &LocalGraph,
    implicit_lifted: bool,
) -> Vec<Vec<usize>> {
    let rank = component.cycle_rank();
    if rank == 0 {
        return Vec::new();
    }

    let tree_edges = kruskal_spanning_tree_flags(component);
    let chords = tree_edges
        .iter()
        .enumerate()
        .filter_map(|(edge_id, &in_tree)| (!in_tree).then_some(edge_id))
        .collect::<Vec<_>>();
    let mut orthogonal_vectors = chords
        .iter()
        .copied()
        .map(|edge_id| singleton_edge_vector(component.edge_count(), edge_id))
        .collect::<Vec<_>>();

    let mut basis = Vec::with_capacity(rank);
    while let Some(base) = orthogonal_vectors.pop() {
        let cycle = if implicit_lifted {
            minimum_orthogonal_cycle_implicit(component, &base)
        } else {
            minimum_orthogonal_cycle_materialized(component, &base)
        };
        basis.push(cycle.cycle);

        for orthogonal in &mut orthogonal_vectors {
            if odd_intersection(orthogonal, &cycle.edge_bits) {
                xor_assign(orthogonal, &base);
            }
        }
    }

    basis
}

fn kruskal_spanning_tree_flags(component: &LocalGraph) -> Vec<bool> {
    let mut parents = (0..component.order()).collect::<Vec<_>>();
    let mut ranks = vec![0u8; component.order()];
    let mut tree_edges = vec![false; component.edge_count()];

    for (edge_id, [left, right]) in component.edges.iter().copied().enumerate() {
        if union_find_union(&mut parents, &mut ranks, left, right) {
            tree_edges[edge_id] = true;
        }
    }

    tree_edges
}

fn union_find_find(parents: &mut [usize], node: usize) -> usize {
    if parents[node] != node {
        let parent = parents[node];
        parents[node] = union_find_find(parents, parent);
    }
    parents[node]
}

fn union_find_union(parents: &mut [usize], ranks: &mut [u8], left: usize, right: usize) -> bool {
    let left_root = union_find_find(parents, left);
    let right_root = union_find_find(parents, right);
    if left_root == right_root {
        return false;
    }
    match ranks[left_root].cmp(&ranks[right_root]) {
        std::cmp::Ordering::Less => parents[left_root] = right_root,
        std::cmp::Ordering::Greater => parents[right_root] = left_root,
        std::cmp::Ordering::Equal => {
            parents[right_root] = left_root;
            ranks[left_root] += 1;
        }
    }
    true
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct OrthogonalCycle {
    cycle: Vec<usize>,
    edge_bits: Vec<u64>,
}

fn minimum_orthogonal_cycle_materialized(
    component: &LocalGraph,
    orthogonal: &[u64],
) -> OrthogonalCycle {
    let lifted = build_lifted_graph(component, orthogonal);
    let plane_size = component.order();

    let mut best_distance = None;
    let mut best_meeting = None;
    let mut best_parents0 = Vec::new();
    let mut best_parents1 = Vec::new();
    for start in 0..plane_size {
        if let Some((distance, meeting, parents0, parents1)) =
            bidirectional_shortest_path_parents_materialized(&lifted, start, plane_size + start)
        {
            if best_distance.is_none_or(|best| distance < best) {
                best_distance = Some(distance);
                best_meeting = Some(meeting);
                best_parents0 = parents0;
                best_parents1 = parents1;
            }
        }
    }

    let meeting = best_meeting.expect("orthogonal cycle should exist");
    let path = reconstruct_bidirectional_path(
        meeting,
        &best_parents0,
        &best_parents1,
        plane_size,
        meeting % plane_size + plane_size * usize::from(meeting >= plane_size),
    );
    collapse_path_into_cycle(component, path)
}

fn minimum_orthogonal_cycle_implicit(
    component: &LocalGraph,
    orthogonal: &[u64],
) -> OrthogonalCycle {
    let plane_size = component.order();

    let mut best_distance = None;
    let mut best_meeting = None;
    let mut best_parents0 = Vec::new();
    let mut best_parents1 = Vec::new();
    let mut best_source = 0usize;
    let mut best_target = 0usize;
    for start in 0..plane_size {
        let source = start;
        let target = plane_size + start;
        if let Some((distance, meeting, parents0, parents1)) =
            bidirectional_shortest_path_parents_implicit(component, orthogonal, source, target)
        {
            if best_distance.is_none_or(|best| distance < best) {
                best_distance = Some(distance);
                best_meeting = Some(meeting);
                best_parents0 = parents0;
                best_parents1 = parents1;
                best_source = source;
                best_target = target;
            }
        }
    }

    let meeting = best_meeting.expect("orthogonal cycle should exist");
    let path = reconstruct_bidirectional_path(
        meeting,
        &best_parents0,
        &best_parents1,
        best_source,
        best_target,
    );
    collapse_path_into_cycle(component, path)
}

fn build_lifted_graph(component: &LocalGraph, orthogonal: &[u64]) -> Vec<Vec<usize>> {
    let plane_size = component.order();
    let mut lifted = vec![Vec::new(); plane_size * 2];
    for (edge_id, [left, right]) in component.edges.iter().copied().enumerate() {
        if is_bit_set(orthogonal, edge_id) {
            add_undirected_edge(&mut lifted, left, plane_size + right);
            add_undirected_edge(&mut lifted, plane_size + left, right);
        } else {
            add_undirected_edge(&mut lifted, left, right);
            add_undirected_edge(&mut lifted, plane_size + left, plane_size + right);
        }
    }
    lifted
}

fn add_undirected_edge(adjacency: &mut [Vec<usize>], left: usize, right: usize) {
    adjacency[left].push(right);
    adjacency[right].push(left);
}

fn bidirectional_shortest_path_parents_materialized(
    adjacency: &[Vec<usize>],
    source: usize,
    target: usize,
) -> Option<MeetingState> {
    bidirectional_shortest_path_core(adjacency.len(), source, target, |node, out| {
        out.extend_from_slice(&adjacency[node]);
    })
}

fn bidirectional_shortest_path_parents_implicit(
    component: &LocalGraph,
    orthogonal: &[u64],
    source: usize,
    target: usize,
) -> Option<MeetingState> {
    let plane_size = component.order();
    bidirectional_shortest_path_core(plane_size * 2, source, target, |node, out| {
        let plane = usize::from(node >= plane_size);
        let base = node % plane_size;
        for &neighbor_base in &component.adjacency[base] {
            let edge_id = component.edge_id(base, neighbor_base);
            let next_plane = if is_bit_set(orthogonal, edge_id) { 1 - plane } else { plane };
            out.push(next_plane * plane_size + neighbor_base);
        }
    })
}

fn bidirectional_shortest_path_core<F>(
    node_count: usize,
    source: usize,
    target: usize,
    mut neighbors: F,
) -> Option<MeetingState>
where
    F: FnMut(usize, &mut Vec<usize>),
{
    if source == target {
        let parents0 = vec![None; node_count];
        let parents1 = vec![None; node_count];
        return Some((0, source, parents0, parents1));
    }

    let mut dists0 = vec![None; node_count];
    let mut dists1 = vec![None; node_count];
    let mut seen0 = vec![None; node_count];
    let mut seen1 = vec![None; node_count];
    let mut parents0 = vec![None; node_count];
    let mut parents1 = vec![None; node_count];
    let mut fringe0 = BinaryHeap::new();
    let mut fringe1 = BinaryHeap::new();
    let mut counter = 0usize;

    seen0[source] = Some(0);
    seen1[target] = Some(0);
    fringe0.push(std::cmp::Reverse((0usize, counter, source)));
    counter += 1;
    fringe1.push(std::cmp::Reverse((0usize, counter, target)));
    counter += 1;

    let mut final_distance = None;
    let mut final_meeting = None;
    let mut neighbor_buffer = Vec::new();
    let mut direction = 1usize;

    while !fringe0.is_empty() && !fringe1.is_empty() {
        direction = 1 - direction;

        if direction == 0 {
            let Some((distance, node)) = pop_next(&mut fringe0, &dists0) else {
                continue;
            };
            dists0[node] = Some(distance);
            if dists1[node].is_some() {
                return final_distance
                    .map(|length| (length, final_meeting.unwrap(), parents0, parents1));
            }

            neighbor_buffer.clear();
            neighbors(node, &mut neighbor_buffer);
            for &neighbor in &neighbor_buffer {
                let next_distance = distance + 1;
                if dists0[neighbor].is_none()
                    && seen0[neighbor].is_none_or(|seen| next_distance < seen)
                {
                    seen0[neighbor] = Some(next_distance);
                    parents0[neighbor] = Some(node);
                    fringe0.push(std::cmp::Reverse((next_distance, counter, neighbor)));
                    counter += 1;

                    if let (Some(left_distance), Some(right_distance)) =
                        (seen0[neighbor], seen1[neighbor])
                    {
                        let total_distance = left_distance + right_distance;
                        if final_distance.is_none_or(|best| total_distance < best) {
                            final_distance = Some(total_distance);
                            final_meeting = Some(neighbor);
                        }
                    }
                }
            }
        } else {
            let Some((distance, node)) = pop_next(&mut fringe1, &dists1) else {
                continue;
            };
            dists1[node] = Some(distance);
            if dists0[node].is_some() {
                return final_distance
                    .map(|length| (length, final_meeting.unwrap(), parents0, parents1));
            }

            neighbor_buffer.clear();
            neighbors(node, &mut neighbor_buffer);
            for &neighbor in &neighbor_buffer {
                let next_distance = distance + 1;
                if dists1[neighbor].is_none()
                    && seen1[neighbor].is_none_or(|seen| next_distance < seen)
                {
                    seen1[neighbor] = Some(next_distance);
                    parents1[neighbor] = Some(node);
                    fringe1.push(std::cmp::Reverse((next_distance, counter, neighbor)));
                    counter += 1;

                    if let (Some(left_distance), Some(right_distance)) =
                        (seen0[neighbor], seen1[neighbor])
                    {
                        let total_distance = left_distance + right_distance;
                        if final_distance.is_none_or(|best| total_distance < best) {
                            final_distance = Some(total_distance);
                            final_meeting = Some(neighbor);
                        }
                    }
                }
            }
        }
    }

    None
}

fn pop_next(
    fringe: &mut BinaryHeap<std::cmp::Reverse<(usize, usize, usize)>>,
    finalized: &[Option<usize>],
) -> Option<(usize, usize)> {
    while let Some(std::cmp::Reverse((distance, _, node))) = fringe.pop() {
        if finalized[node].is_none() {
            return Some((distance, node));
        }
    }
    None
}

fn reconstruct_bidirectional_path(
    meeting: usize,
    parents0: &[Option<usize>],
    parents1: &[Option<usize>],
    source: usize,
    target: usize,
) -> Vec<usize> {
    let mut left = Vec::new();
    let mut current = Some(meeting);
    while let Some(node) = current {
        left.push(node);
        if node == source {
            break;
        }
        current = parents0[node];
    }
    left.reverse();

    let mut right = Vec::new();
    let mut current = parents1[meeting];
    while let Some(node) = current {
        right.push(node);
        if node == target {
            break;
        }
        current = parents1[node];
    }

    left.extend(right);
    left
}

fn collapse_path_into_cycle(component: &LocalGraph, path: Vec<usize>) -> OrthogonalCycle {
    let plane_size = component.order();
    let normalized_path = path
        .into_iter()
        .map(|node| if node < plane_size { node } else { node - plane_size })
        .collect::<Vec<_>>();
    let cycle_edges = collapse_lifted_path_edges(&normalized_path);
    let mut edge_bits = vec![0_u64; component.edge_count().div_ceil(64)];
    for &[left, right] in &cycle_edges {
        set_bit(&mut edge_bits, component.edge_id(left, right));
    }
    let cycle = normalize_cycle(
        cycle_edges.into_iter().map(|[_, right]| component.vertices[right]).collect::<Vec<_>>(),
    );
    OrthogonalCycle { cycle, edge_bits }
}

fn collapse_lifted_path_edges(path: &[usize]) -> Vec<[usize; 2]> {
    let node_count = path.iter().max().copied().unwrap_or(0) + 1;
    let mut counts = vec![0u8; node_count * node_count];

    for pair in path.windows(2) {
        let left = pair[0];
        let right = pair[1];
        counts[left * node_count + right] ^= 1;
    }

    let mut cycle_edges = Vec::new();
    for pair in path.windows(2) {
        let left = pair[0];
        let right = pair[1];
        let idx = left * node_count + right;
        let rev = right * node_count + left;
        if counts[idx] == 1 {
            counts[idx] = 0;
            cycle_edges.push([left, right]);
        } else if counts[rev] == 1 {
            counts[rev] = 0;
            cycle_edges.push([right, left]);
        }
    }

    cycle_edges
}

fn singleton_edge_vector(edge_count: usize, edge_id: usize) -> Vec<u64> {
    let mut bits = vec![0_u64; edge_count.div_ceil(64)];
    set_bit(&mut bits, edge_id);
    bits
}

fn odd_intersection(left: &[u64], right: &[u64]) -> bool {
    let mut parity = false;
    for (&left_word, &right_word) in left.iter().zip(right.iter()) {
        parity ^= ((left_word & right_word).count_ones() & 1) == 1;
    }
    parity
}

fn set_bit(bits: &mut [u64], bit: usize) {
    bits[bit / 64] |= 1_u64 << (bit % 64);
}

fn is_bit_set(bits: &[u64], bit: usize) -> bool {
    ((bits[bit / 64] >> (bit % 64)) & 1) == 1
}

fn xor_assign(left: &mut [u64], right: &[u64]) {
    for (left_word, right_word) in left.iter_mut().zip(right.iter()) {
        *left_word ^= *right_word;
    }
}

fn assert_variants_match_reference() {
    let cases = prepare_fixture_cases(FIXTURE_NAME);
    for case in cases {
        for variant in
            [Variant::BaselinePublic, Variant::ParentsMaterialized, Variant::ParentsImplicit]
        {
            let result = evaluate_variant(variant, &case.graph);
            assert_eq!(
                result.basis,
                case.expected_basis,
                "{}: variant {} mismatched fixture basis",
                case.name,
                variant.label()
            );
        }
    }
}

fn bench_reference_fixture_total(c: &mut Criterion) {
    assert_variants_match_reference();
    let cases = prepare_fixture_cases(FIXTURE_NAME);
    let variants =
        [Variant::BaselinePublic, Variant::ParentsMaterialized, Variant::ParentsImplicit];

    let mut group = c.benchmark_group("minimum_cycle_basis_variants_networkx_1000_total");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(300));
    group.measurement_time(Duration::from_secs(2));
    group.throughput(Throughput::Elements(
        u64::try_from(cases.len()).expect("fixture count should fit into u64"),
    ));

    for variant in variants {
        group.bench_function(BenchmarkId::new("variant", variant.label()), |b| {
            b.iter(|| {
                let checksum = cases
                    .iter()
                    .map(|case| evaluate_variant(variant, &case.graph).checksum())
                    .fold(0u64, u64::wrapping_add);
                black_box(checksum);
            });
        });
    }
    group.finish();
}

fn bench_scaling_cases(c: &mut Criterion) {
    let cases = prepare_scaling_cases();
    let variants =
        [Variant::BaselinePublic, Variant::ParentsMaterialized, Variant::ParentsImplicit];

    let mut group = c.benchmark_group("minimum_cycle_basis_variants_scaling");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(300));
    group.measurement_time(Duration::from_secs(2));

    for case in &cases {
        group.throughput(Throughput::Elements(
            u64::try_from(case.logical_edge_count).expect("edge count should fit into u64"),
        ));
        for variant in variants {
            group.bench_function(
                BenchmarkId::new(
                    format!("{}::{}", case.name, variant.label()),
                    case.logical_edge_count,
                ),
                |b| {
                    b.iter(|| {
                        let checksum = evaluate_variant(variant, &case.graph).checksum();
                        black_box(checksum);
                    });
                },
            );
        }
    }
    group.finish();
}

criterion_group!(benches, bench_reference_fixture_total, bench_scaling_cases);
criterion_main!(benches);
