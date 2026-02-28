//! Implementation of the `Arbitrary` trait for the `GenericGraph` struct.

use arbitrary::{Arbitrary, Unstructured};
use num_traits::AsPrimitive;

use super::GenericGraph;
use crate::{
    impls::{CSR2D, SquareCSR2D},
    traits::{Matrix2D, PositiveInteger, TryFromUsize},
};

impl<'a, SparseIndex, Idx> Arbitrary<'a>
    for GenericGraph<Idx, SquareCSR2D<CSR2D<SparseIndex, Idx, Idx>>>
where
    SparseIndex: TryFromUsize + AsPrimitive<usize> + PositiveInteger,
    Idx: PositiveInteger
        + for<'b> Arbitrary<'b>
        + TryFrom<SparseIndex>
        + AsPrimitive<usize>
        + TryFromUsize
        + num_traits::ConstOne
        + num_traits::ConstZero
        + core::ops::MulAssign
        + num_traits::CheckedMul
        + num_traits::ToPrimitive
        + num_traits::SaturatingSub
        + 'static,
{
    fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        let edges = SquareCSR2D::arbitrary(u)?;
        let nodes: Idx = edges.number_of_rows();
        Ok(Self { nodes, edges })
    }
}
