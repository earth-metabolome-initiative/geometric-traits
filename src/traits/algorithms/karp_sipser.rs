//! Exact Karp-Sipser preprocessing for maximum matching in general graphs.
use alloc::vec::Vec;

use crate::{
    impls::{CSR2D, SymmetricCSR2D},
    traits::{Blossom, Blum, MicaliVazirani, SparseSquareMatrix},
};

mod inner;

use inner::{RecoveryStep, build_kernel, recover_pairs};

/// Exact Karp-Sipser reduction rules to apply before calling an exact matcher.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum KarpSipserRules {
    /// Apply only degree-1 reductions.
    Degree1,
    /// Apply degree-1 and degree-2 reductions, always prioritizing degree-1.
    Degree1And2,
}

/// A compact kernel graph together with the metadata required to recover a
/// matching on the original graph.
pub struct KarpSipserKernel<I> {
    kernel: SymmetricCSR2D<CSR2D<usize, usize, usize>>,
    kernel_to_internal: Vec<usize>,
    original_indices: Vec<I>,
    recovery: Vec<RecoveryStep>,
    total_vertices: usize,
}

impl<I: Copy> KarpSipserKernel<I> {
    /// Returns the compact kernel graph.
    #[must_use]
    #[inline]
    pub fn graph(&self) -> &SymmetricCSR2D<CSR2D<usize, usize, usize>> {
        &self.kernel
    }

    /// Recovers a matching on the original graph from a matching on the kernel.
    ///
    /// # Panics
    ///
    /// Panics if `kernel_matching` contains out-of-bounds vertices, repeated
    /// vertices, unsorted pairs, or edges that are not present in the kernel.
    #[must_use]
    #[inline]
    pub fn recover(self, kernel_matching: Vec<(usize, usize)>) -> Vec<(I, I)> {
        recover_pairs(&self, kernel_matching)
    }

    /// Solves the kernel with a caller-provided exact matcher and recovers a
    /// matching on the original graph.
    #[inline]
    pub fn solve_with<F>(self, solver: F) -> Vec<(I, I)>
    where
        F: FnOnce(&SymmetricCSR2D<CSR2D<usize, usize, usize>>) -> Vec<(usize, usize)>,
    {
        let kernel_matching = solver(&self.kernel);
        self.recover(kernel_matching)
    }
}

/// Exact Karp-Sipser preprocessing for general-graph maximum matching.
pub trait KarpSipser: SparseSquareMatrix {
    /// Builds a compact exact kernel using the selected Karp-Sipser rules.
    #[inline]
    fn karp_sipser_kernel(&self, rules: KarpSipserRules) -> KarpSipserKernel<Self::Index> {
        build_kernel(self, rules)
    }

    /// Runs Edmonds blossom on the Karp-Sipser kernel and recovers the result.
    #[inline]
    fn blossom_with_karp_sipser(&self, rules: KarpSipserRules) -> Vec<(Self::Index, Self::Index)> {
        self.karp_sipser_kernel(rules).solve_with(Blossom::blossom)
    }

    /// Runs Micali-Vazirani on the Karp-Sipser kernel and recovers the result.
    #[inline]
    fn micali_vazirani_with_karp_sipser(
        &self,
        rules: KarpSipserRules,
    ) -> Vec<(Self::Index, Self::Index)> {
        self.karp_sipser_kernel(rules).solve_with(MicaliVazirani::micali_vazirani)
    }

    /// Runs Blum's algorithm on the Karp-Sipser kernel and recovers the
    /// result.
    #[inline]
    fn blum_with_karp_sipser(&self, rules: KarpSipserRules) -> Vec<(Self::Index, Self::Index)> {
        self.karp_sipser_kernel(rules).solve_with(Blum::blum)
    }
}

impl<M: SparseSquareMatrix + ?Sized> KarpSipser for M {}
