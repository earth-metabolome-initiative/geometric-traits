//! Submodule providing the `BipartiteDetection` trait and its blanket
//! implementation for undirected graphs.

use alloc::{collections::VecDeque, vec, vec::Vec};

use num_traits::AsPrimitive;

use crate::traits::{
    PositiveInteger, SparseValuedMatrix2D, TryFromUsize, UndirectedMonopartiteMonoplexGraph,
};

#[inline]
fn convert_index<I: PositiveInteger + TryFromUsize>(index: usize) -> I {
    I::try_from_usize(index).ok().expect("index must fit into the target integer type")
}

fn bipartite_coloring_with<F>(number_of_vertices: usize, mut visit_neighbors: F) -> Option<Vec<u8>>
where
    F: FnMut(usize, &mut dyn FnMut(usize)),
{
    let mut colors = vec![u8::MAX; number_of_vertices];
    let mut queue = VecDeque::new();

    for start in 0..number_of_vertices {
        if colors[start] != u8::MAX {
            continue;
        }

        colors[start] = 0;
        queue.push_back(start);

        while let Some(vertex) = queue.pop_front() {
            let expected_neighbor_color = colors[vertex] ^ 1;
            let mut found_conflict = false;
            visit_neighbors(vertex, &mut |neighbor| {
                if found_conflict {
                    return;
                }
                if colors[neighbor] == u8::MAX {
                    colors[neighbor] = expected_neighbor_color;
                    queue.push_back(neighbor);
                    return;
                }
                if colors[neighbor] != expected_neighbor_color {
                    found_conflict = true;
                }
            });
            if found_conflict {
                return None;
            }
        }
    }

    Some(colors)
}

pub(crate) fn sparse_matrix_bipartite_coloring<M>(matrix: &M) -> Option<Vec<u8>>
where
    M: SparseValuedMatrix2D,
    M::RowIndex: PositiveInteger,
    M::ColumnIndex: PositiveInteger,
{
    bipartite_coloring_with(matrix.number_of_rows().as_(), |vertex, visit_neighbor| {
        let row = convert_index::<M::RowIndex>(vertex);
        for column in matrix.sparse_row(row) {
            visit_neighbor(column.as_());
        }
    })
}

/// Trait providing predicates and a certificate for bipartiteness in
/// undirected graphs.
pub trait BipartiteDetection: UndirectedMonopartiteMonoplexGraph
where
    Self::NodeId: PositiveInteger + AsPrimitive<usize> + TryFromUsize,
{
    /// Returns a valid 2-coloring of the graph if it is bipartite.
    ///
    /// Each entry is either `0` or `1`, and adjacent vertices always receive
    /// opposite colors.
    ///
    /// # Complexity
    ///
    /// O(V + E) time and O(V) space.
    #[inline]
    fn bipartite_coloring(&self) -> Option<Vec<u8>> {
        bipartite_coloring_with(self.number_of_nodes().as_(), |vertex, visit_neighbor| {
            let node = convert_index::<Self::NodeId>(vertex);
            for neighbor in self.neighbors(node) {
                visit_neighbor(neighbor.as_());
            }
        })
    }

    /// Returns true if the graph is bipartite.
    ///
    /// # Complexity
    ///
    /// O(V + E) time and O(V) space.
    #[inline]
    fn is_bipartite(&self) -> bool {
        self.bipartite_coloring().is_some()
    }
}

impl<G> BipartiteDetection for G
where
    G: UndirectedMonopartiteMonoplexGraph,
    G::NodeId: PositiveInteger + AsPrimitive<usize> + TryFromUsize,
{
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use super::sparse_matrix_bipartite_coloring;
    use crate::{impls::ValuedCSR2D, naive_structs::GenericEdgesBuilder, traits::EdgesBuilder};

    type Weighted = ValuedCSR2D<usize, usize, usize, i32>;

    fn build(node_count: usize, undirected_edges: &[(usize, usize)]) -> Weighted {
        let mut edges = Vec::new();
        for &(left, right) in undirected_edges {
            edges.push((left, right, 1));
            edges.push((right, left, 1));
        }
        edges.sort_unstable_by(|(ls, ld, _), (rs, rd, _)| (ls, ld).cmp(&(rs, rd)));
        GenericEdgesBuilder::<_, Weighted>::default()
            .expected_number_of_edges(edges.len())
            .expected_shape((node_count, node_count))
            .edges(edges.into_iter())
            .build()
            .unwrap()
    }

    #[test]
    fn test_sparse_matrix_bipartite_coloring() {
        // An even cycle is bipartite: adjacent vertices must get opposite colors.
        let even_cycle = build(4, &[(0, 1), (1, 2), (2, 3), (3, 0)]);
        let coloring =
            sparse_matrix_bipartite_coloring(&even_cycle).expect("an even cycle is bipartite");
        assert_eq!(coloring.len(), 4);
        for &(left, right) in &[(0, 1), (1, 2), (2, 3), (3, 0)] {
            assert_ne!(coloring[left], coloring[right]);
        }

        // An odd cycle is not bipartite.
        let odd_cycle = build(3, &[(0, 1), (1, 2), (2, 0)]);
        assert!(sparse_matrix_bipartite_coloring(&odd_cycle).is_none());
    }
}
