//! Labeled equitable refinement for simple undirected graphs.
//!
//! The current implementation uses queue-based partition refinement over an
//! existing ordered partition. Starting from the partition's current cells, it
//! repeatedly refines non-singleton cells using splitter cells until an
//! equitable partition is reached.

use alloc::{collections::VecDeque, vec::Vec};

use num_traits::AsPrimitive;

use super::{BacktrackableOrderedPartition, PartitionCellId};
use crate::traits::MonoplexMonopartiteGraph;

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct NeighbourCountSignature<EdgeLabel> {
    pub(crate) counts: Vec<(EdgeLabel, usize)>,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum RefinementTraceEvent<EdgeLabel> {
    Split { first: usize, length: usize },
    Edge { unit_cell_first: usize, target_index: usize, edge_label: EdgeLabel },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum RefinementTraceStorage<EdgeLabel> {
    Packed(Vec<u32>),
    Events(Vec<RefinementTraceEvent<EdgeLabel>>),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct RefinementTrace<EdgeLabel> {
    pub(crate) storage: RefinementTraceStorage<EdgeLabel>,
    pub(crate) subcertificate_length: usize,
    pub(crate) eqref_hash: u64,
}

const CERT_SPLIT: u32 = 0;
const CERT_EDGE: u32 = 1;

#[inline]
#[allow(clippy::cast_possible_truncation)]
fn packed_word(value: usize) -> u32 {
    debug_assert!(u32::try_from(value).is_ok());
    value as u32
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
struct EqRefHash {
    state: u32,
}

impl EqRefHash {
    const RTAB: [u32; 256] = [
        0xAEAA35B8, 0x65632E16, 0x155EDBA9, 0x01349B39, 0x8EB8BD97, 0x8E4C5367, 0x8EA78B35,
        0x2B1B4072, 0xC1163893, 0x269A8642, 0xC79D7F6D, 0x6A32DEA0, 0xD4D2DA56, 0xD96D4F47,
        0x47B5F48A, 0x2587C6BF, 0x642B71D8, 0x5DBBAF58, 0x5C178169, 0xA16D9279, 0x75CDA063,
        0x291BC48B, 0x01AC2F47, 0x5416DF7C, 0x45307514, 0xB3E1317B, 0xE1C7A8DE, 0x3ACDAC96,
        0x11B96831, 0x32DE22DD, 0x6A1DA93B, 0x58B62381, 0x283810E2, 0xBC30E6A6, 0x8EE51705,
        0xB06E8DFB, 0x729AB12A, 0xA9634922, 0x1A6E8525, 0x49DD4E19, 0xE5DB3D44, 0x8C5B3A02,
        0xEBDE2864, 0xA9146D9F, 0x736D2CB4, 0xF5229F42, 0x712BA846, 0x20631593, 0x89C02603,
        0xD5A5BF6A, 0x823F4E18, 0x5BE5DEFF, 0x1C4EBBFA, 0x5FAB8490, 0x6E559B0C, 0x1FE528D6,
        0xB3198066, 0x4A965EB5, 0xFE8BB3D5, 0x4D2F6234, 0x5F125AA4, 0xBCC640FA, 0x4F8BC191,
        0xA447E537, 0xAC474D3C, 0x703BFA2C, 0x617DC0E7, 0xF26299D7, 0xC90FD835, 0x33B71C7B,
        0x6D83E138, 0xCBB1BB14, 0x029CF5FF, 0x7CBD093D, 0x4C9825EF, 0x845C4D6D, 0x124349A5,
        0x53942D21, 0x800E60DA, 0x2BA6EB7F, 0xCEBF30D3, 0xEB18D449, 0xE281F724, 0x58B1CB09,
        0xD469A13D, 0x9C7495C3, 0xE53A7810, 0xA866C08E, 0x832A038B, 0xDDDCA484, 0xD5FE0DDE,
        0x0756002B, 0x2FF51342, 0x60FEC9C8, 0x061A53E3, 0x47B1884E, 0xDC17E461, 0xA17A6A37,
        0x3158E7E2, 0xA40D873B, 0x45AE2140, 0xC8F36149, 0x63A4EE2D, 0xD7107447, 0x6F90994F,
        0x5006770F, 0xC1F3CA9A, 0x91B317B2, 0xF61B4406, 0xA8C9EE8F, 0xC6939B75, 0xB28BBC3B,
        0x36BF4AEF, 0x3B12118D, 0x4D536ECF, 0x9CF4B46B, 0xE8AB1E03, 0x8225A360, 0x7AE4A130,
        0xC4EE8B50, 0x50651797, 0x5BB4C59F, 0xD120EE47, 0x24F3A386, 0xBE579B45, 0x3A378EFC,
        0xC5AB007B, 0x3668942B, 0x2DBDCC3A, 0x6F37F64C, 0xC24F862A, 0xB6F97FCF, 0x9E4FA23D,
        0x551AE769, 0x46A8A5A6, 0xDC1BCFDD, 0x8F684CF9, 0x501D811B, 0x84279F80, 0x2614E0AC,
        0x86445276, 0xAEA0CE71, 0x0812250F, 0xB586D18A, 0xC68D721B, 0x44514E1D, 0x37CDB99A,
        0x24731F89, 0xFA72E589, 0x81E6EBA2, 0x15452965, 0x55523D9D, 0x2DC47E14, 0x2E7FA107,
        0xA7790F23, 0x40EBFDBB, 0x77E7906B, 0x6C1DB960, 0x1A8B9898, 0x65FA0D90, 0xED28B4D8,
        0x34C3ED75, 0x768FD2EC, 0xFAB60BCB, 0x962C75F4, 0x304F0498, 0x0A41A36B, 0xF7DE2A4A,
        0xF4770FE2, 0x73C93BBB, 0xD21C82C5, 0x6C387447, 0x8CDB4CB9, 0x2CC243E8, 0x41859E3D,
        0xB667B9CB, 0x89681E8A, 0x61A0526C, 0x883EDDDC, 0x539DE9A4, 0xC29E1DEC, 0x97C71EC5,
        0x4A560A66, 0xBD7ECACF, 0x576AE998, 0x31CE5616, 0x97172A6C, 0x83D047C4, 0x274EA9A8,
        0xEB31A9DA, 0x327209B5, 0x14D1F2CB, 0x00FE1D96, 0x817DBE08, 0xD3E55AED, 0xF2D30AFC,
        0xFB072660, 0x866687D6, 0x92552EB9, 0xEA8219CD, 0xF7927269, 0xF1948483, 0x694C1DF5,
        0xB7D8B7BF, 0xFFBC5D2F, 0x2E88B849, 0x883FD32B, 0xA0331192, 0x8CB244DF, 0x41FAF895,
        0x16902220, 0x97FB512A, 0x2BEA3CC4, 0xAF9CAE61, 0x41ACD0D5, 0xFD2F28FF, 0xE780ADFA,
        0xB3A3A76E, 0x7112AD87, 0x7C3D6058, 0x69E64FFF, 0xE5F8617C, 0x8580727C, 0x41F54F04,
        0xD72BE498, 0x653D1795, 0x1275A327, 0x14B499D4, 0x4E34D553, 0x4687AA39, 0x68B64292,
        0x5C18ABC3, 0x41EABFCC, 0x92A85616, 0x82684CF8, 0x5B9F8A4E, 0x35382FFE, 0xFB936318,
        0x52C08E15, 0x80918B2E, 0x199EDEE0, 0xA9470163, 0xEC44ACDD, 0x612D6735, 0x8F88EA7D,
        0x759F5EA4, 0xE5CC7240, 0x68CFEB8B, 0x04725601, 0x0C22C23E, 0x5BC97174, 0x89965841,
        0x5D939479, 0x690F338A, 0x3C2D4380, 0xDAE97F2B,
    ];

    #[inline]
    #[allow(clippy::cast_possible_truncation)]
    fn update(&mut self, value: usize) {
        let mut input = (value as u32).wrapping_add(1);
        while input > 0 {
            self.state ^= Self::RTAB[(input & 0xff) as usize];
            let carry = self.state >> 31;
            input >>= 8;
            self.state = (self.state << 1) | carry;
        }
    }

    #[inline]
    fn value(self) -> u64 {
        u64::from(self.state)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SplitQueueMode {
    Binary01,
    General,
}

enum NonUnitCellSplitAnalysis<EdgeLabel> {
    UnsignedCounts {
        max_invariant: usize,
        max_invariant_count: usize,
        all_equal: bool,
        counts_in_current_order: Vec<usize>,
    },
    General {
        all_equal: bool,
        signatures_in_cell_order: Vec<(usize, NeighbourCountSignature<EdgeLabel>)>,
    },
}

#[derive(Clone, Debug)]
struct DenseMarker {
    marks: Vec<u32>,
    generation: u32,
}

impl DenseMarker {
    #[inline]
    fn new(order: usize) -> Self {
        Self { marks: vec![0; order], generation: 1 }
    }

    #[inline]
    fn clear(&mut self) {
        self.generation = self.generation.wrapping_add(1);
        if self.generation == 0 {
            self.marks.fill(0);
            self.generation = 1;
        }
    }

    #[inline]
    fn mark(&mut self, index: usize) {
        self.marks[index] = self.generation;
    }

    #[inline]
    fn contains(&self, index: usize) -> bool {
        self.marks[index] == self.generation
    }
}

#[derive(Clone, Debug)]
struct DenseCellBuckets<T> {
    marker: DenseMarker,
    touched_cells: Vec<PartitionCellId>,
    buckets: Vec<Vec<T>>,
}

impl<T> DenseCellBuckets<T> {
    fn new(order: usize) -> Self {
        Self {
            marker: DenseMarker::new(order),
            touched_cells: Vec::new(),
            buckets: (0..order).map(|_| Vec::new()).collect(),
        }
    }

    fn clear(&mut self) {
        for &cell in &self.touched_cells {
            self.buckets[cell.index()].clear();
        }
        self.touched_cells.clear();
        self.marker.clear();
    }

    fn push(&mut self, cell: PartitionCellId, value: T) {
        if !self.marker.contains(cell.index()) {
            self.marker.mark(cell.index());
            self.touched_cells.push(cell);
        }
        self.buckets[cell.index()].push(value);
    }

    fn sort_touched_cells(&mut self, partition: &BacktrackableOrderedPartition) {
        self.touched_cells.sort_unstable_by_key(|&cell| partition.cell_first(cell));
    }

    fn touched_cells(&self) -> &[PartitionCellId] {
        &self.touched_cells
    }

    fn bucket(&self, cell: PartitionCellId) -> &[T] {
        &self.buckets[cell.index()]
    }
}

#[derive(Clone, Debug)]
pub(crate) struct RefinementWorkspace<EdgeLabel> {
    in_queue: Vec<bool>,
    splitter_queue: VecDeque<PartitionCellId>,
    splitter_marker: DenseMarker,
    touched_cell_marker: DenseMarker,
    unit_touched_elements: DenseCellBuckets<usize>,
    unit_touched_labelled_elements: DenseCellBuckets<(usize, EdgeLabel)>,
    splitter_elements: Vec<usize>,
    candidate_cells: Vec<PartitionCellId>,
    splitter_touched_vertices: Vec<usize>,
    splitter_vertex_counts: Vec<usize>,
}

impl<EdgeLabel> RefinementWorkspace<EdgeLabel> {
    pub(crate) fn new(order: usize) -> Self {
        Self {
            in_queue: vec![false; order],
            splitter_queue: VecDeque::new(),
            splitter_marker: DenseMarker::new(order),
            touched_cell_marker: DenseMarker::new(order),
            unit_touched_elements: DenseCellBuckets::new(order),
            unit_touched_labelled_elements: DenseCellBuckets::new(order),
            splitter_elements: Vec::new(),
            candidate_cells: Vec::new(),
            splitter_touched_vertices: Vec::new(),
            splitter_vertex_counts: vec![0; order],
        }
    }

    fn reset(&mut self) {
        self.in_queue.fill(false);
        self.splitter_queue.clear();
        self.splitter_marker.clear();
        self.touched_cell_marker.clear();
        self.unit_touched_elements.clear();
        self.unit_touched_labelled_elements.clear();
        self.splitter_elements.clear();
        self.candidate_cells.clear();
        self.splitter_touched_vertices.clear();
    }
}

/// Refines `partition` to the stable labeled equitable partition induced by
/// `graph` and `edge_label`.
///
/// The graph is assumed to use dense node identifiers
/// `0..graph.number_of_nodes()`. The partition must have the same order as the
/// graph.
///
/// Returns `true` if the partition changed.
pub fn refine_partition_to_labeled_equitable<G, EdgeLabel, F>(
    graph: &G,
    partition: &mut BacktrackableOrderedPartition,
    edge_label: F,
) -> bool
where
    G: MonoplexMonopartiteGraph,
    G::NodeId: AsPrimitive<usize>,
    EdgeLabel: Ord + Clone,
    F: FnMut(G::NodeId, G::NodeId) -> EdgeLabel,
{
    refine_partition_to_labeled_equitable_with_trace(graph, partition, edge_label).0
}

pub(crate) fn refine_partition_to_labeled_equitable_with_trace<G, EdgeLabel, F>(
    graph: &G,
    partition: &mut BacktrackableOrderedPartition,
    edge_label: F,
) -> (bool, RefinementTrace<EdgeLabel>)
where
    G: MonoplexMonopartiteGraph,
    G::NodeId: AsPrimitive<usize>,
    EdgeLabel: Ord + Clone,
    F: FnMut(G::NodeId, G::NodeId) -> EdgeLabel,
{
    let order = graph.number_of_nodes().as_();
    let nodes: Vec<G::NodeId> = graph.node_ids().collect();
    let mut workspace = RefinementWorkspace::new(order);
    refine_partition_to_labeled_equitable_with_trace_from_splitters_impl(
        graph,
        &nodes,
        partition,
        edge_label,
        partition.cells().map(super::partition::PartitionCellView::id).collect::<Vec<_>>(),
        None,
        &mut workspace,
    )
}

pub(crate) fn refine_partition_to_labeled_equitable_with_trace_from_splitters_in_workspace<
    G,
    EdgeLabel,
    F,
    I,
>(
    graph: &G,
    nodes: &[G::NodeId],
    partition: &mut BacktrackableOrderedPartition,
    edge_label: F,
    initial_splitters: I,
    workspace: &mut RefinementWorkspace<EdgeLabel>,
) -> (bool, RefinementTrace<EdgeLabel>)
where
    G: MonoplexMonopartiteGraph,
    G::NodeId: AsPrimitive<usize>,
    EdgeLabel: Ord + Clone,
    F: FnMut(G::NodeId, G::NodeId) -> EdgeLabel,
    I: IntoIterator<Item = PartitionCellId>,
{
    refine_partition_to_labeled_equitable_with_trace_from_splitters_impl(
        graph,
        nodes,
        partition,
        edge_label,
        initial_splitters,
        None,
        workspace,
    )
}

#[allow(clippy::too_many_lines)]
fn refine_partition_to_labeled_equitable_with_trace_from_splitters_impl<G, EdgeLabel, F, I>(
    graph: &G,
    nodes: &[G::NodeId],
    partition: &mut BacktrackableOrderedPartition,
    mut edge_label: F,
    initial_splitters: I,
    mut pop_sequence: Option<&mut Vec<(usize, usize)>>,
    workspace: &mut RefinementWorkspace<EdgeLabel>,
) -> (bool, RefinementTrace<EdgeLabel>)
where
    G: MonoplexMonopartiteGraph,
    G::NodeId: AsPrimitive<usize>,
    EdgeLabel: Ord + Clone,
    F: FnMut(G::NodeId, G::NodeId) -> EdgeLabel,
    I: IntoIterator<Item = PartitionCellId>,
{
    let order = graph.number_of_nodes().as_();
    assert!(
        partition.order() == order,
        "partition order {} must match graph order {order}",
        partition.order()
    );

    debug_assert_eq!(nodes.len(), order);
    debug_assert!(nodes.iter().enumerate().all(|(index, node)| node.as_() == index));

    workspace.reset();
    let RefinementWorkspace {
        in_queue,
        splitter_queue,
        splitter_marker,
        touched_cell_marker,
        unit_touched_elements,
        unit_touched_labelled_elements,
        splitter_elements,
        candidate_cells,
        splitter_touched_vertices,
        splitter_vertex_counts,
    } = workspace;

    let mut changed_any = false;
    let mut events = Vec::new();
    let mut packed_events = (core::mem::size_of::<EdgeLabel>() == 0).then(Vec::new);
    let mut eqref_hash = EqRefHash::default();
    for cell in initial_splitters {
        enqueue_cell_like_bliss(cell, partition, splitter_queue, in_queue);
    }

    while let Some(splitter) = splitter_queue.pop_front() {
        in_queue[splitter.index()] = false;
        splitter_elements.clear();
        splitter_elements.extend_from_slice(partition.cell_elements(splitter));
        if splitter_elements.is_empty() {
            continue;
        }
        let splitter_first = partition.cell_first(splitter);
        if let Some(pop_sequence) = &mut pop_sequence {
            pop_sequence.push((splitter_first, splitter_elements.len()));
        }

        splitter_marker.clear();
        for &element in splitter_elements.iter() {
            splitter_marker.mark(element);
        }

        if splitter_elements.len() == 1 {
            eqref_hash.update(0x8765_4321);
            eqref_hash.update(splitter_first);
            eqref_hash.update(1);
            let splitter_vertex = nodes[splitter_elements[0]];
            if packed_events.is_some() {
                unit_touched_elements.clear();
                for neighbour in graph.successors(splitter_vertex) {
                    unit_touched_elements.push(partition.cell_of(neighbour.as_()), neighbour.as_());
                }
                unit_touched_elements.sort_touched_cells(partition);

                for &cell in unit_touched_elements.touched_cells() {
                    let was_in_queue = in_queue[cell.index()];
                    let touched_elements = unit_touched_elements.bucket(cell);
                    let produced = if partition.cell_elements(cell).len() > 1 {
                        let produced =
                            partition.split_cell_by_tail_elements_in_order(cell, touched_elements);
                        if let Some(touched_cell) = produced.last().copied() {
                            let produced_first = partition.cell_first(touched_cell);
                            if let Some(packed) = packed_events.as_mut() {
                                for offset in 0..partition.cell_len(touched_cell) {
                                    packed.extend_from_slice(&[
                                        CERT_EDGE,
                                        packed_word(splitter_first),
                                        packed_word(produced_first + offset),
                                    ]);
                                }
                            }
                        }
                        produced
                    } else {
                        if let Some(packed) = packed_events.as_mut() {
                            packed.extend_from_slice(&[
                                CERT_EDGE,
                                packed_word(splitter_first),
                                packed_word(partition.cell_first(cell)),
                            ]);
                        }
                        vec![cell]
                    };
                    if produced.len() > 1 {
                        eqref_hash.update(partition.cell_first(cell));
                        eqref_hash.update(partition.cell_len(cell));
                        eqref_hash.update(produced.len());
                        for &produced_cell in &produced {
                            eqref_hash.update(partition.cell_len(produced_cell));
                        }
                        changed_any = true;
                        let queue_mode = if produced.len() == 2 {
                            SplitQueueMode::Binary01
                        } else {
                            SplitQueueMode::General
                        };
                        enqueue_split_result_like_bliss(
                            &produced,
                            was_in_queue,
                            queue_mode,
                            None,
                            partition,
                            splitter_queue,
                            in_queue,
                        );
                    }
                    for &produced_cell in &produced {
                        eqref_hash.update(partition.cell_first(produced_cell));
                        eqref_hash.update(partition.cell_len(produced_cell));
                    }
                }
            } else {
                unit_touched_labelled_elements.clear();
                for neighbour in graph.successors(splitter_vertex) {
                    let cell = partition.cell_of(neighbour.as_());
                    unit_touched_labelled_elements
                        .push(cell, (neighbour.as_(), edge_label(splitter_vertex, neighbour)));
                }
                unit_touched_labelled_elements.sort_touched_cells(partition);

                for &cell in unit_touched_labelled_elements.touched_cells() {
                    let was_in_queue = in_queue[cell.index()];
                    let touched_entries = unit_touched_labelled_elements.bucket(cell);
                    let produced = if partition.cell_elements(cell).len() > 1 {
                        if let Some((label, touched_elements)) =
                            unit_touched_elements_if_single_label(touched_entries)
                        {
                            let produced = partition
                                .split_cell_by_tail_elements_in_order(cell, &touched_elements);
                            if let Some(touched_cell) = produced.last().copied() {
                                let produced_first = partition.cell_first(touched_cell);
                                for offset in 0..partition.cell_len(touched_cell) {
                                    events.push(RefinementTraceEvent::Edge {
                                        unit_cell_first: splitter_first,
                                        target_index: produced_first + offset,
                                        edge_label: label.clone(),
                                    });
                                }
                            }
                            produced
                        } else {
                            let produced = partition.split_cell_by_key(cell, |vertex| {
                                labelled_signature_to_splitter(
                                    graph,
                                    nodes,
                                    splitter_marker,
                                    vertex,
                                    &mut edge_label,
                                )
                            });
                            for &produced_cell in &produced {
                                let elements = partition.cell_elements(produced_cell);
                                let signature = labelled_signature_to_splitter(
                                    graph,
                                    nodes,
                                    splitter_marker,
                                    elements[0],
                                    &mut edge_label,
                                );
                                if signature.counts.is_empty() {
                                    continue;
                                }
                                debug_assert_eq!(signature.counts.len(), 1);
                                debug_assert_eq!(signature.counts[0].1, 1);
                                let produced_first = partition.cell_first(produced_cell);
                                for offset in 0..elements.len() {
                                    events.push(RefinementTraceEvent::Edge {
                                        unit_cell_first: splitter_first,
                                        target_index: produced_first + offset,
                                        edge_label: signature.counts[0].0.clone(),
                                    });
                                }
                            }
                            produced
                        }
                    } else {
                        let touched_label = touched_entries
                            .first()
                            .expect(
                                "unit or singleton touched cells must contain one touched element",
                            )
                            .1
                            .clone();
                        events.push(RefinementTraceEvent::Edge {
                            unit_cell_first: splitter_first,
                            target_index: partition.cell_first(cell),
                            edge_label: touched_label,
                        });
                        vec![cell]
                    };
                    if produced.len() > 1 {
                        eqref_hash.update(partition.cell_first(cell));
                        eqref_hash.update(partition.cell_len(cell));
                        eqref_hash.update(produced.len());
                        for &produced_cell in &produced {
                            eqref_hash.update(partition.cell_len(produced_cell));
                        }
                        changed_any = true;
                        let queue_mode = if produced.len() == 2 {
                            SplitQueueMode::Binary01
                        } else {
                            SplitQueueMode::General
                        };
                        enqueue_split_result_like_bliss(
                            &produced,
                            was_in_queue,
                            queue_mode,
                            None,
                            partition,
                            splitter_queue,
                            in_queue,
                        );
                    }
                    for &produced_cell in &produced {
                        eqref_hash.update(partition.cell_first(produced_cell));
                        eqref_hash.update(partition.cell_len(produced_cell));
                    }
                }
            }
            continue;
        }

        let use_unsigned_counts = core::mem::size_of::<EdgeLabel>() == 0;
        touched_cell_marker.clear();
        candidate_cells.clear();
        if use_unsigned_counts {
            splitter_touched_vertices.clear();
        }
        for &splitter_vertex in splitter_elements.iter() {
            let source = nodes[splitter_vertex];
            for neighbour in graph.successors(source) {
                let neighbour_index = neighbour.as_();
                if use_unsigned_counts {
                    if splitter_vertex_counts[neighbour_index] == 0 {
                        splitter_touched_vertices.push(neighbour_index);
                    }
                    splitter_vertex_counts[neighbour_index] += 1;
                }
                let neighbour_cell = partition.cell_of(neighbour_index);
                if partition.cell_elements(neighbour_cell).len() > 1
                    && !touched_cell_marker.contains(neighbour_cell.index())
                {
                    touched_cell_marker.mark(neighbour_cell.index());
                    candidate_cells.push(neighbour_cell);
                }
            }
        }

        candidate_cells.sort_unstable_by_key(|&cell| partition.cell_first(cell));
        for &cell in candidate_cells.iter() {
            let was_in_queue = in_queue[cell.index()];
            let analysis = if use_unsigned_counts {
                analyse_non_unit_cell_to_splitter_unsigned_from_counts(
                    partition,
                    cell,
                    splitter_vertex_counts,
                )
            } else {
                analyse_non_unit_cell_to_splitter(
                    partition,
                    graph,
                    nodes,
                    splitter_marker,
                    cell,
                    &mut edge_label,
                )
            };
            let (produced, queue_mode) = match analysis {
                NonUnitCellSplitAnalysis::UnsignedCounts { all_equal: true, .. }
                | NonUnitCellSplitAnalysis::General { all_equal: true, .. } => continue,
                NonUnitCellSplitAnalysis::UnsignedCounts {
                    max_invariant,
                    max_invariant_count,
                    counts_in_current_order,
                    ..
                } => (
                    partition
                        .split_cell_by_unsigned_invariants_in_current_order_like_bliss_with_summary(
                            cell,
                            &counts_in_current_order,
                            max_invariant,
                            max_invariant_count,
                        ),
                    if max_invariant == 1 {
                        SplitQueueMode::Binary01
                    } else {
                        SplitQueueMode::General
                    },
                ),
                NonUnitCellSplitAnalysis::General { signatures_in_cell_order, .. } => {
                    (
                        partition.split_cell_by_precomputed_keys(cell, &signatures_in_cell_order),
                        SplitQueueMode::General,
                    )
                }
            };
            eqref_hash.update(partition.cell_first(cell));
            eqref_hash.update(partition.cell_len(cell));
            eqref_hash.update(produced.len());
            for &produced_cell in &produced {
                eqref_hash.update(partition.cell_len(produced_cell));
            }
            for &produced_cell in &produced {
                eqref_hash.update(partition.cell_first(produced_cell));
                eqref_hash.update(partition.cell_len(produced_cell));
                if let Some(packed) = packed_events.as_mut() {
                    packed.extend_from_slice(&[
                        CERT_SPLIT,
                        packed_word(partition.cell_first(produced_cell)),
                        packed_word(partition.cell_len(produced_cell)),
                    ]);
                } else {
                    events.push(RefinementTraceEvent::Split {
                        first: partition.cell_first(produced_cell),
                        length: partition.cell_len(produced_cell),
                    });
                }
            }
            if produced.len() <= 1 {
                continue;
            }

            changed_any = true;
            let creation_suffix_lengths =
                if was_in_queue && matches!(queue_mode, SplitQueueMode::General) {
                    Some(suffix_creation_lengths_from_cells(&produced, partition))
                } else {
                    None
                };

            enqueue_split_result_like_bliss(
                &produced,
                was_in_queue,
                queue_mode,
                creation_suffix_lengths.as_deref(),
                partition,
                splitter_queue,
                in_queue,
            );
        }

        if use_unsigned_counts {
            for &vertex in splitter_touched_vertices.iter() {
                splitter_vertex_counts[vertex] = 0;
            }
            splitter_touched_vertices.clear();
        }
    }

    let storage = if let Some(packed) = packed_events {
        RefinementTraceStorage::Packed(packed)
    } else {
        RefinementTraceStorage::Events(events)
    };
    let subcertificate_length = match &storage {
        RefinementTraceStorage::Packed(words) => words.len(),
        RefinementTraceStorage::Events(events) => events.len(),
    };

    (
        changed_any,
        RefinementTrace { storage, subcertificate_length, eqref_hash: eqref_hash.value() },
    )
}

#[cfg(test)]
#[allow(dead_code)]
pub(crate) fn refine_partition_to_labeled_equitable_with_trace_and_pop_sequence<
    G,
    EdgeLabel,
    F,
    I,
>(
    graph: &G,
    partition: &mut BacktrackableOrderedPartition,
    edge_label: F,
    initial_splitters: I,
    pop_sequence: &mut Vec<(usize, usize)>,
) -> (bool, RefinementTrace<EdgeLabel>)
where
    G: MonoplexMonopartiteGraph,
    G::NodeId: AsPrimitive<usize>,
    EdgeLabel: Ord + Clone,
    F: FnMut(G::NodeId, G::NodeId) -> EdgeLabel,
    I: IntoIterator<Item = PartitionCellId>,
{
    let order = graph.number_of_nodes().as_();
    let nodes: Vec<G::NodeId> = graph.node_ids().collect();
    let mut workspace = RefinementWorkspace::new(order);
    refine_partition_to_labeled_equitable_with_trace_from_splitters_impl(
        graph,
        &nodes,
        partition,
        edge_label,
        initial_splitters,
        Some(pop_sequence),
        &mut workspace,
    )
}

fn unit_touched_elements_if_single_label<EdgeLabel>(
    touched_entries: &[(usize, EdgeLabel)],
) -> Option<(EdgeLabel, Vec<usize>)>
where
    EdgeLabel: Ord + Clone,
{
    let (_, first_label) = touched_entries.first()?;
    if touched_entries.iter().all(|(_, label)| label == first_label) {
        let elements = touched_entries.iter().map(|(element, _)| *element).collect::<Vec<_>>();
        return Some((first_label.clone(), elements));
    }
    None
}

fn analyse_non_unit_cell_to_splitter<G, EdgeLabel, F>(
    partition: &BacktrackableOrderedPartition,
    graph: &G,
    nodes: &[G::NodeId],
    in_splitter: &DenseMarker,
    cell: PartitionCellId,
    edge_label: &mut F,
) -> NonUnitCellSplitAnalysis<EdgeLabel>
where
    G: MonoplexMonopartiteGraph,
    G::NodeId: AsPrimitive<usize>,
    EdgeLabel: Ord + Clone,
    F: FnMut(G::NodeId, G::NodeId) -> EdgeLabel,
{
    if core::mem::size_of::<EdgeLabel>() == 0 {
        return analyse_non_unit_cell_to_splitter_unsigned(
            partition,
            graph,
            nodes,
            in_splitter,
            cell,
        );
    }

    let mut signatures_in_cell_order = Vec::with_capacity(partition.cell_len(cell));
    let mut common_label = None::<EdgeLabel>;
    let mut counts_in_current_order = Vec::with_capacity(partition.cell_len(cell));
    let mut max_invariant = 0usize;
    let mut max_invariant_count = 0usize;
    let mut first_count = None::<usize>;
    let mut all_equal = true;
    let mut all_single_label_counts = true;
    let mut first_signature = None::<NeighbourCountSignature<EdgeLabel>>;

    for &vertex in partition.cell_elements(cell) {
        let signature =
            labelled_signature_to_splitter(graph, nodes, in_splitter, vertex, edge_label);
        if all_single_label_counts {
            let count = match signature.counts.as_slice() {
                [] => 0,
                [(label, count)] => {
                    match &common_label {
                        None => common_label = Some(label.clone()),
                        Some(existing) if existing == label => {}
                        Some(_) => all_single_label_counts = false,
                    }
                    *count
                }
                _ => {
                    all_single_label_counts = false;
                    0
                }
            };
            if all_single_label_counts {
                if let Some(first) = first_count {
                    if count != first {
                        all_equal = false;
                    }
                } else {
                    first_count = Some(count);
                }
                if count > max_invariant {
                    max_invariant = count;
                    max_invariant_count = 1;
                } else if count == max_invariant {
                    max_invariant_count += 1;
                }
                counts_in_current_order.push(count);
            }
        }
        if let Some(first) = &first_signature {
            if first != &signature {
                all_equal = false;
            }
        } else {
            first_signature = Some(signature.clone());
        }
        signatures_in_cell_order.push((vertex, signature));
    }

    if all_single_label_counts {
        return NonUnitCellSplitAnalysis::UnsignedCounts {
            max_invariant,
            max_invariant_count,
            all_equal,
            counts_in_current_order,
        };
    }

    NonUnitCellSplitAnalysis::General { all_equal, signatures_in_cell_order }
}

fn analyse_non_unit_cell_to_splitter_unsigned<G, EdgeLabel>(
    partition: &BacktrackableOrderedPartition,
    graph: &G,
    nodes: &[G::NodeId],
    in_splitter: &DenseMarker,
    cell: PartitionCellId,
) -> NonUnitCellSplitAnalysis<EdgeLabel>
where
    G: MonoplexMonopartiteGraph,
    G::NodeId: AsPrimitive<usize>,
{
    let mut counts_in_current_order = Vec::with_capacity(partition.cell_len(cell));
    let mut max_invariant = 0usize;
    let mut max_invariant_count = 0usize;
    let mut first_count = None::<usize>;
    let mut all_equal = true;
    for &vertex in partition.cell_elements(cell) {
        let source = nodes[vertex];
        let count = graph
            .successors(source)
            .filter(|neighbour| in_splitter.contains(neighbour.as_()))
            .count();
        if let Some(first) = first_count {
            if count != first {
                all_equal = false;
            }
        } else {
            first_count = Some(count);
        }
        if count > max_invariant {
            max_invariant = count;
            max_invariant_count = 1;
        } else if count == max_invariant {
            max_invariant_count += 1;
        }
        counts_in_current_order.push(count);
    }
    NonUnitCellSplitAnalysis::UnsignedCounts {
        max_invariant,
        max_invariant_count,
        all_equal,
        counts_in_current_order,
    }
}

fn analyse_non_unit_cell_to_splitter_unsigned_from_counts<EdgeLabel>(
    partition: &BacktrackableOrderedPartition,
    cell: PartitionCellId,
    splitter_vertex_counts: &[usize],
) -> NonUnitCellSplitAnalysis<EdgeLabel> {
    let mut counts_in_current_order = Vec::with_capacity(partition.cell_len(cell));
    let mut max_invariant = 0usize;
    let mut max_invariant_count = 0usize;
    let mut first_count = None::<usize>;
    let mut all_equal = true;
    for &vertex in partition.cell_elements(cell) {
        let count = splitter_vertex_counts[vertex];
        if let Some(first) = first_count {
            if count != first {
                all_equal = false;
            }
        } else {
            first_count = Some(count);
        }
        if count > max_invariant {
            max_invariant = count;
            max_invariant_count = 1;
        } else if count == max_invariant {
            max_invariant_count += 1;
        }
        counts_in_current_order.push(count);
    }
    NonUnitCellSplitAnalysis::UnsignedCounts {
        max_invariant,
        max_invariant_count,
        all_equal,
        counts_in_current_order,
    }
}

fn enqueue_split_result_like_bliss(
    produced: &[PartitionCellId],
    original_was_in_queue: bool,
    mode: SplitQueueMode,
    creation_suffix_lengths: Option<&[usize]>,
    partition: &BacktrackableOrderedPartition,
    splitter_queue: &mut VecDeque<PartitionCellId>,
    in_queue: &mut [bool],
) {
    if original_was_in_queue {
        if let Some(lengths) = creation_suffix_lengths {
            debug_assert_eq!(lengths.len(), produced.len().saturating_sub(1));
            for (&cell, &length_at_creation) in produced[1..].iter().zip(lengths.iter()) {
                enqueue_cell_like_bliss_with_len(
                    cell,
                    length_at_creation,
                    splitter_queue,
                    in_queue,
                );
            }
        } else {
            for &cell in &produced[1..] {
                enqueue_cell_like_bliss(cell, partition, splitter_queue, in_queue);
            }
        }
        return;
    }

    match mode {
        SplitQueueMode::Binary01 => {
            debug_assert_eq!(produced.len(), 2);
            let left_cell = produced[0];
            let right_cell = produced[1];
            let (min_cell, max_cell) =
                if partition.cell_len(left_cell) <= partition.cell_len(right_cell) {
                    (left_cell, right_cell)
                } else {
                    (right_cell, left_cell)
                };
            enqueue_cell_like_bliss(min_cell, partition, splitter_queue, in_queue);
            if partition.cell_len(max_cell) == 1 {
                enqueue_cell_like_bliss(max_cell, partition, splitter_queue, in_queue);
            }
        }
        SplitQueueMode::General => {
            debug_assert!(!produced.is_empty());
            let mut largest_cell = produced[0];

            for &cell in produced.iter().take(produced.len().saturating_sub(1)).skip(1) {
                if partition.cell_len(cell) > partition.cell_len(largest_cell) {
                    enqueue_cell_like_bliss(largest_cell, partition, splitter_queue, in_queue);
                    largest_cell = cell;
                } else {
                    enqueue_cell_like_bliss(cell, partition, splitter_queue, in_queue);
                }
            }

            let last_cell = *produced.last().expect("split result must contain at least one cell");
            if partition.cell_len(last_cell) > partition.cell_len(largest_cell) {
                enqueue_cell_like_bliss(largest_cell, partition, splitter_queue, in_queue);
                largest_cell = last_cell;
            } else {
                enqueue_cell_like_bliss(last_cell, partition, splitter_queue, in_queue);
            }

            if partition.cell_len(largest_cell) == 1 {
                enqueue_cell_like_bliss(largest_cell, partition, splitter_queue, in_queue);
            }
        }
    }
}

fn enqueue_cell_like_bliss(
    cell: PartitionCellId,
    partition: &BacktrackableOrderedPartition,
    splitter_queue: &mut VecDeque<PartitionCellId>,
    in_queue: &mut [bool],
) {
    enqueue_cell_like_bliss_with_len(cell, partition.cell_len(cell), splitter_queue, in_queue);
}

fn enqueue_cell_like_bliss_with_len(
    cell: PartitionCellId,
    cell_len: usize,
    splitter_queue: &mut VecDeque<PartitionCellId>,
    in_queue: &mut [bool],
) {
    if in_queue[cell.index()] {
        return;
    }
    in_queue[cell.index()] = true;
    if cell_len <= 1 {
        splitter_queue.push_front(cell);
    } else {
        splitter_queue.push_back(cell);
    }
}

fn suffix_creation_lengths_from_cells(
    produced: &[PartitionCellId],
    partition: &BacktrackableOrderedPartition,
) -> Vec<usize> {
    let mut suffix_sum = 0usize;
    let mut suffix_sums = vec![0usize; produced.len().saturating_sub(1)];
    for (index, &cell) in produced.iter().enumerate().rev() {
        suffix_sum += partition.cell_len(cell);
        if index > 0 {
            suffix_sums[index - 1] = suffix_sum;
        }
    }
    suffix_sums
}

fn labelled_signature_to_splitter<G, EdgeLabel, F>(
    graph: &G,
    nodes: &[G::NodeId],
    in_splitter: &DenseMarker,
    vertex: usize,
    edge_label: &mut F,
) -> NeighbourCountSignature<EdgeLabel>
where
    G: MonoplexMonopartiteGraph,
    G::NodeId: AsPrimitive<usize>,
    EdgeLabel: Ord + Clone,
    F: FnMut(G::NodeId, G::NodeId) -> EdgeLabel,
{
    let source = nodes[vertex];
    let mut neighbour_pairs = graph
        .successors(source)
        .filter(|neighbour| in_splitter.contains(neighbour.as_()))
        .map(|neighbour| edge_label(source, neighbour))
        .collect::<Vec<_>>();
    neighbour_pairs.sort_unstable();

    let mut counts = Vec::new();
    let mut index = 0usize;
    while index < neighbour_pairs.len() {
        let label = neighbour_pairs[index].clone();
        let mut count = 1usize;
        index += 1;
        while index < neighbour_pairs.len() && neighbour_pairs[index] == label {
            count += 1;
            index += 1;
        }
        counts.push((label, count));
    }

    NeighbourCountSignature { counts }
}
