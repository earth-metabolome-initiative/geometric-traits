use alloc::{vec, vec::Vec};

use num_traits::AsPrimitive;

use crate::{
    impls::ValuedCSR2D,
    traits::{Number, PositiveInteger, SparseValuedMatrix2D},
};

pub(super) type ReducedCostGraph = ValuedCSR2D<usize, usize, usize, i64>;

#[derive(Clone, Copy)]
pub(super) enum ExpandedEdgeKind {
    Original(usize),
    Slack,
}

#[derive(Clone, Copy)]
pub(super) struct ExpandedEdge {
    pub(super) u: usize,
    pub(super) v: usize,
    pub(super) cost: i64,
    pub(super) kind: ExpandedEdgeKind,
}

#[derive(Clone, Copy)]
pub(super) struct OriginalEdge {
    pub(super) u: usize,
    pub(super) v: usize,
    pub(super) capacity: usize,
    pub(super) cost: i64,
}

pub(super) fn find_cross_edge(
    cross_edges: &[(usize, usize, usize)],
    u: usize,
    v: usize,
) -> Option<usize> {
    let pair = if u < v { (u, v) } else { (v, u) };
    cross_edges
        .binary_search_by_key(&(pair.0, pair.1), |&(left, right, _)| (left, right))
        .ok()
        .map(|position| cross_edges[position].2)
}

pub(super) fn convert_index<I: PositiveInteger>(index: usize) -> I {
    I::try_from_usize(index).ok().expect("index must fit into the target integer type")
}

pub(super) fn symmetric_cost_at<C>(edge_costs: &C, i: usize, j: usize) -> i64
where
    C: SparseValuedMatrix2D,
    C::Value: Number + AsPrimitive<i64>,
    C::RowIndex: PositiveInteger,
    C::ColumnIndex: PositiveInteger,
{
    let i_row = convert_index::<C::RowIndex>(i);
    let j_row = convert_index::<C::RowIndex>(j);
    let i_col = convert_index::<C::ColumnIndex>(i);
    let j_col = convert_index::<C::ColumnIndex>(j);

    match (edge_costs.sparse_value_at(i_row, j_col), edge_costs.sparse_value_at(j_row, i_col)) {
        (Some(forward), Some(backward)) => {
            let forward_cost = forward.as_();
            let backward_cost = backward.as_();
            assert!(
                forward_cost == backward_cost,
                "cost matrix is not symmetric on edge ({i}, {j}): {forward_cost} != {backward_cost}"
            );
            forward_cost
        }
        (Some(cost), None) | (None, Some(cost)) => cost.as_(),
        (None, None) => panic!("cost matrix is missing a value for capacity edge ({i}, {j})"),
    }
}

pub(super) fn connected_components_from_capacity_matrix<M>(capacities: &M) -> (usize, Vec<usize>)
where
    M: SparseValuedMatrix2D,
    M::Value: PositiveInteger,
    M::RowIndex: PositiveInteger,
    M::ColumnIndex: PositiveInteger,
{
    let n_rows: usize = capacities.number_of_rows().as_();
    let mut component_of_vertex = vec![usize::MAX; n_rows];
    let mut stack = Vec::new();
    let mut component_count = 0usize;

    for start in 0..n_rows {
        if component_of_vertex[start] != usize::MAX {
            continue;
        }

        component_of_vertex[start] = component_count;
        stack.push(start);

        while let Some(vertex) = stack.pop() {
            let row = convert_index::<M::RowIndex>(vertex);
            for column in capacities.sparse_row(row) {
                let neighbour = column.as_();
                if component_of_vertex[neighbour] != usize::MAX {
                    continue;
                }
                component_of_vertex[neighbour] = component_count;
                stack.push(neighbour);
            }
        }

        component_count += 1;
    }

    (component_count, component_of_vertex)
}
