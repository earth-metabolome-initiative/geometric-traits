//! Implementation of the `Arbitrary` trait for the `GenericGraph` struct.

use crate::impls::{CSR2D, SquareCSR2D};
use crate::traits::Matrix2D;
use crate::traits::{IntoUsize, PositiveInteger, TryFromUsize};
use arbitrary::{Arbitrary, Unstructured};

use super::GenericGraph;

impl<'a, SparseIndex, Idx> Arbitrary<'a>
    for GenericGraph<Idx, SquareCSR2D<CSR2D<SparseIndex, Idx, Idx>>>
where
    SparseIndex: TryFromUsize + IntoUsize + PositiveInteger,
    Idx: PositiveInteger
        + for<'b> Arbitrary<'b>
        + TryFrom<SparseIndex>
        + IntoUsize
        + TryFromUsize
        + num_traits::ConstOne
        + num_traits::ConstZero
        + std::ops::MulAssign
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
