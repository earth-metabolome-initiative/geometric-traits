//! Submodule providing an implementation of the `Arbitrary` trait for the
//! `GenericMatrix2DWithPaddedDiagonal` struct.

use arbitrary::{Arbitrary, Unstructured};
use num_traits::{AsPrimitive, One};

use crate::{
    impls::GenericMatrix2DWithPaddedDiagonal,
    traits::{Matrix2D, SparseMatrix2D, TryFromUsize, ValuedMatrix2D},
};

fn one<R, V: One>(_a: R) -> V {
    V::one()
}

impl<'a, M> Arbitrary<'a> for GenericMatrix2DWithPaddedDiagonal<M, fn(M::RowIndex) -> M::Value>
where
    M: for<'b> Arbitrary<'b> + ValuedMatrix2D + Matrix2D + SparseMatrix2D,
    M::Value: One,
    M::ColumnIndex: AsPrimitive<usize> + TryFromUsize,
{
    fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        let matrix = M::arbitrary(u)?;

        let padded = GenericMatrix2DWithPaddedDiagonal::new(
            matrix,
            one::<M::RowIndex, M::Value> as fn(M::RowIndex) -> M::Value,
        )
        .map_err(|_| arbitrary::Error::IncorrectFormat)?;

        Ok(padded)
    }
}
