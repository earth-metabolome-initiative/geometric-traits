//! Set and graph similarity metrics.
//!
//! Provides standalone free functions for generic set similarity coefficients
//! (Tanimoto/Jaccard, Dice/Sorensen, overlap, cosine, Tversky, Kulczynski,
//! Braun-Blanquet, Sokal-Sneath, McConnaughey) and the Johnson graph
//! similarity metric used in MCES result scoring.
//!
//! Also provides the [`GraphSimilarities`] trait: any type that can report
//! matched edge/vertex counts and original graph sizes gets all similarity
//! methods via default implementations.

/// Tanimoto (Jaccard) similarity coefficient.
///
/// `T = intersection / (size_a + size_b - intersection)`
///
/// The de facto standard for comparing molecular fingerprints in
/// cheminformatics. Also widely used in ecology (species overlap),
/// information retrieval, and recommendation systems. Provides a
/// balanced penalty for both shared and unique elements.
///
/// Returns 1.0 when both sets are empty (denominator is zero).
///
/// # Examples
///
/// ```
/// use geometric_traits::prelude::tanimoto_similarity;
///
/// assert!((tanimoto_similarity(3, 5, 4) - 0.5).abs() < f64::EPSILON);
/// assert!((tanimoto_similarity(0, 0, 0) - 1.0).abs() < f64::EPSILON);
/// ```
#[inline]
#[must_use]
pub fn tanimoto_similarity(intersection: usize, size_a: usize, size_b: usize) -> f64 {
    debug_assert!(
        intersection <= size_a,
        "intersection ({intersection}) must be <= size_a ({size_a})"
    );
    debug_assert!(
        intersection <= size_b,
        "intersection ({intersection}) must be <= size_b ({size_b})"
    );
    let union = size_a + size_b - intersection;
    if union == 0 {
        return 1.0;
    }
    #[allow(clippy::cast_precision_loss)]
    {
        intersection as f64 / union as f64
    }
}

/// Dice (Sorensen-Dice) similarity coefficient.
///
/// `D = 2 * intersection / (size_a + size_b)`
///
/// More forgiving than Jaccard — emphasizes overlap and is less sensitive
/// to imbalanced set sizes. Mathematically equivalent to the F1 score.
/// Commonly used in ecology (vegetation surveys), medical image
/// segmentation, and NLP (token overlap).
///
/// Returns 1.0 when both sets are empty (denominator is zero).
///
/// # Examples
///
/// ```
/// use geometric_traits::prelude::dice_similarity;
///
/// assert!((dice_similarity(3, 5, 4) - 6.0 / 9.0).abs() < f64::EPSILON);
/// assert!((dice_similarity(0, 0, 0) - 1.0).abs() < f64::EPSILON);
/// ```
#[inline]
#[must_use]
pub fn dice_similarity(intersection: usize, size_a: usize, size_b: usize) -> f64 {
    debug_assert!(
        intersection <= size_a,
        "intersection ({intersection}) must be <= size_a ({size_a})"
    );
    debug_assert!(
        intersection <= size_b,
        "intersection ({intersection}) must be <= size_b ({size_b})"
    );
    let sum = size_a + size_b;
    if sum == 0 {
        return 1.0;
    }
    #[allow(clippy::cast_precision_loss)]
    {
        (2 * intersection) as f64 / sum as f64
    }
}

/// Overlap (Szymkiewicz-Simpson) similarity coefficient.
///
/// `O = intersection / min(size_a, size_b)`
///
/// Insensitive to set size differences — measures what fraction of the
/// smaller set is shared. Useful when sets differ greatly in size, e.g.,
/// genomic region overlaps, link prediction in graphs (RAPIDS cuGraph),
/// and containment queries in information retrieval.
///
/// Returns 1.0 when both sets are empty (denominator is zero).
///
/// # Examples
///
/// ```
/// use geometric_traits::prelude::overlap_similarity;
///
/// assert!((overlap_similarity(3, 5, 4) - 0.75).abs() < f64::EPSILON);
/// assert!((overlap_similarity(0, 0, 0) - 1.0).abs() < f64::EPSILON);
/// ```
#[inline]
#[must_use]
pub fn overlap_similarity(intersection: usize, size_a: usize, size_b: usize) -> f64 {
    debug_assert!(
        intersection <= size_a,
        "intersection ({intersection}) must be <= size_a ({size_a})"
    );
    debug_assert!(
        intersection <= size_b,
        "intersection ({intersection}) must be <= size_b ({size_b})"
    );
    let min = size_a.min(size_b);
    if min == 0 {
        return if size_a == 0 && size_b == 0 { 1.0 } else { 0.0 };
    }
    #[allow(clippy::cast_precision_loss)]
    {
        intersection as f64 / min as f64
    }
}

/// Cosine similarity coefficient.
///
/// `C = intersection / sqrt(size_a * size_b)`
///
/// Magnitude-invariant — measures direction rather than scale. The
/// standard metric for document and text similarity in NLP and
/// information retrieval. Also used for sparse high-dimensional feature
/// comparison and recommendation systems.
///
/// Returns 1.0 when both sets are empty (denominator is zero).
///
/// # Examples
///
/// ```
/// use geometric_traits::prelude::cosine_similarity;
///
/// let expected = 3.0 / (5.0_f64 * 4.0).sqrt();
/// assert!((cosine_similarity(3, 5, 4) - expected).abs() < f64::EPSILON);
/// assert!((cosine_similarity(0, 0, 0) - 1.0).abs() < f64::EPSILON);
/// ```
#[inline]
#[must_use]
pub fn cosine_similarity(intersection: usize, size_a: usize, size_b: usize) -> f64 {
    debug_assert!(
        intersection <= size_a,
        "intersection ({intersection}) must be <= size_a ({size_a})"
    );
    debug_assert!(
        intersection <= size_b,
        "intersection ({intersection}) must be <= size_b ({size_b})"
    );
    let product = size_a * size_b;
    if product == 0 {
        return if size_a == 0 && size_b == 0 { 1.0 } else { 0.0 };
    }
    #[allow(clippy::cast_precision_loss)]
    {
        intersection as f64 / (product as f64).sqrt()
    }
}

/// Johnson graph similarity metric.
///
/// `J = (common_edges + common_vertices)^2 / ((vertices_first + edges_first) *
/// (vertices_second + edges_second))`
///
/// This is the similarity metric used by the RASCAL MCES algorithm
/// (Raymond, Gardiner, Willett 2002).
///
/// Returns 1.0 when both graphs are empty (denominator is zero).
///
/// # Examples
///
/// ```
/// use geometric_traits::prelude::johnson_similarity;
///
/// // Two graphs: G1 has 4 vertices, 3 edges; G2 has 5 vertices, 4 edges.
/// // MCES found 2 common edges and 3 common vertices.
/// let j = johnson_similarity(2, 3, 4, 3, 5, 4);
/// let expected = 25.0 / (7.0 * 9.0);
/// assert!((j - expected).abs() < f64::EPSILON);
/// ```
#[inline]
#[must_use]
pub fn johnson_similarity(
    common_edges: usize,
    common_vertices: usize,
    vertices_first: usize,
    edges_first: usize,
    vertices_second: usize,
    edges_second: usize,
) -> f64 {
    let numerator = common_edges + common_vertices;
    let denom_first = vertices_first + edges_first;
    let denom_second = vertices_second + edges_second;
    let denominator = denom_first * denom_second;
    if denominator == 0 {
        return 1.0;
    }
    #[allow(clippy::cast_precision_loss)]
    {
        let numerator_sq = (numerator * numerator) as f64;
        numerator_sq / denominator as f64
    }
}

/// Tversky similarity index (Tversky 1977).
///
/// `T = intersection / (intersection + alpha * (size_a - intersection) + beta *
/// (size_b - intersection))`
///
/// An asymmetric generalization of Jaccard and Dice. Originally from
/// cognitive psychology ("Features of Similarity"), widely used in
/// cheminformatics for substructure and scaffold-hopping searches.
///
/// - `alpha = beta = 1.0` recovers Jaccard/Tanimoto.
/// - `alpha = beta = 0.5` recovers Dice/Sorensen.
/// - Small `alpha`, large `beta` gives substructure-like matching (permissive
///   toward the query being a subset).
///
/// Returns 1.0 when both sets are empty (denominator is zero).
///
/// # Examples
///
/// ```
/// use geometric_traits::prelude::tversky_similarity;
///
/// // With alpha=beta=1 it's Jaccard: 3/(3 + 1*2 + 1*1) = 3/6 = 0.5
/// assert!((tversky_similarity(3, 5, 4, 1.0, 1.0) - 0.5).abs() < f64::EPSILON);
/// // With alpha=beta=0.5 it's Dice: 3/(3 + 0.5*2 + 0.5*1) = 3/4.5 = 2/3
/// assert!((tversky_similarity(3, 5, 4, 0.5, 0.5) - 2.0 / 3.0).abs() < f64::EPSILON);
/// ```
#[inline]
#[must_use]
pub fn tversky_similarity(
    intersection: usize,
    size_a: usize,
    size_b: usize,
    alpha: f64,
    beta: f64,
) -> f64 {
    debug_assert!(
        intersection <= size_a,
        "intersection ({intersection}) must be <= size_a ({size_a})"
    );
    debug_assert!(
        intersection <= size_b,
        "intersection ({intersection}) must be <= size_b ({size_b})"
    );
    debug_assert!(alpha >= 0.0, "alpha ({alpha}) must be >= 0");
    debug_assert!(beta >= 0.0, "beta ({beta}) must be >= 0");
    let diff_a = size_a - intersection;
    let diff_b = size_b - intersection;
    #[allow(clippy::cast_precision_loss)]
    let denom = intersection as f64 + alpha * diff_a as f64 + beta * diff_b as f64;
    if denom == 0.0 {
        return 1.0;
    }
    #[allow(clippy::cast_precision_loss)]
    {
        intersection as f64 / denom
    }
}

/// Kulczynski similarity (second coefficient).
///
/// `K = 0.5 * (intersection / size_a + intersection / size_b)`
///
/// The arithmetic mean of the two containment ratios. Originally from
/// botanical community comparison (Stanislaw Kulczynski). Provides a
/// balanced view when sets differ greatly in size.
///
/// Returns 1.0 when both sets are empty. Returns 0.0 when one set is empty
/// and the other is not (one ratio is 0/0, the other is 0/n; we define the
/// empty-set ratio as 0.0 since intersection is necessarily 0).
///
/// # Examples
///
/// ```
/// use geometric_traits::prelude::kulczynski_similarity;
///
/// // intersection=3, a=5, b=4 → 0.5*(3/5 + 3/4) = 0.5*(0.6 + 0.75) = 0.675
/// assert!((kulczynski_similarity(3, 5, 4) - 0.675).abs() < f64::EPSILON);
/// assert!((kulczynski_similarity(0, 0, 0) - 1.0).abs() < f64::EPSILON);
/// ```
#[inline]
#[must_use]
pub fn kulczynski_similarity(intersection: usize, size_a: usize, size_b: usize) -> f64 {
    debug_assert!(
        intersection <= size_a,
        "intersection ({intersection}) must be <= size_a ({size_a})"
    );
    debug_assert!(
        intersection <= size_b,
        "intersection ({intersection}) must be <= size_b ({size_b})"
    );
    if size_a == 0 && size_b == 0 {
        return 1.0;
    }
    if size_a == 0 || size_b == 0 {
        return 0.0;
    }
    #[allow(clippy::cast_precision_loss)]
    {
        let ratio_a = intersection as f64 / size_a as f64;
        let ratio_b = intersection as f64 / size_b as f64;
        0.5 * (ratio_a + ratio_b)
    }
}

/// Braun-Blanquet similarity coefficient.
///
/// `BB = intersection / max(size_a, size_b)`
///
/// The conservative counterpart to the overlap coefficient (which uses min).
/// Originally from phytosociology (Josias Braun-Blanquet, 1928). Measures
/// what fraction of the *larger* set is shared — hard to get a high score
/// unless one set nearly subsumes the other.
///
/// Returns 1.0 when both sets are empty (denominator is zero).
///
/// # Examples
///
/// ```
/// use geometric_traits::prelude::braun_blanquet_similarity;
///
/// // intersection=3, a=5, b=4 → 3/max(5,4) = 3/5 = 0.6
/// assert!((braun_blanquet_similarity(3, 5, 4) - 0.6).abs() < f64::EPSILON);
/// assert!((braun_blanquet_similarity(0, 0, 0) - 1.0).abs() < f64::EPSILON);
/// ```
#[inline]
#[must_use]
pub fn braun_blanquet_similarity(intersection: usize, size_a: usize, size_b: usize) -> f64 {
    debug_assert!(
        intersection <= size_a,
        "intersection ({intersection}) must be <= size_a ({size_a})"
    );
    debug_assert!(
        intersection <= size_b,
        "intersection ({intersection}) must be <= size_b ({size_b})"
    );
    let max = size_a.max(size_b);
    if max == 0 {
        return 1.0;
    }
    #[allow(clippy::cast_precision_loss)]
    {
        intersection as f64 / max as f64
    }
}

/// Sokal-Sneath similarity (first coefficient).
///
/// `SS = intersection / (intersection + 2 * (size_a + size_b - 2 *
/// intersection))`
///
/// A stricter version of Jaccard that double-penalizes mismatches. From
/// numerical taxonomy (Sokal & Sneath, 1963). Use when disagreements are
/// considered twice as bad as missing agreements.
///
/// Returns 1.0 when both sets are empty (denominator is zero).
///
/// # Examples
///
/// ```
/// use geometric_traits::prelude::sokal_sneath_similarity;
///
/// // intersection=3, a=5, b=4 → 3/(3 + 2*(5+4-6)) = 3/(3+6) = 1/3
/// assert!((sokal_sneath_similarity(3, 5, 4) - 1.0 / 3.0).abs() < f64::EPSILON);
/// assert!((sokal_sneath_similarity(0, 0, 0) - 1.0).abs() < f64::EPSILON);
/// ```
#[inline]
#[must_use]
pub fn sokal_sneath_similarity(intersection: usize, size_a: usize, size_b: usize) -> f64 {
    debug_assert!(
        intersection <= size_a,
        "intersection ({intersection}) must be <= size_a ({size_a})"
    );
    debug_assert!(
        intersection <= size_b,
        "intersection ({intersection}) must be <= size_b ({size_b})"
    );
    let mismatches = size_a + size_b - 2 * intersection;
    let denom = intersection + 2 * mismatches;
    if denom == 0 {
        return 1.0;
    }
    #[allow(clippy::cast_precision_loss)]
    {
        intersection as f64 / denom as f64
    }
}

/// McConnaughey similarity coefficient.
///
/// `M = (intersection^2 - (size_a - intersection) * (size_b - intersection)) /
/// (size_a * size_b)`
///
/// A correlation-like metric with range \[-1, 1\]. Originally from chemistry
/// and metabolomics. The only metric here that can go negative, indicating
/// anti-correlation (the sets tend to have opposite membership). A value of
/// 0 means the sets are uncorrelated.
///
/// Returns 1.0 when both sets are empty. Returns 0.0 when exactly one set
/// is empty (the product `size_a * size_b` is zero but one is non-empty).
///
/// # Examples
///
/// ```
/// use geometric_traits::prelude::mcconnaughey_similarity;
///
/// // intersection=3, a=5, b=4 → (9 - 2*1) / (5*4) = 7/20 = 0.35
/// assert!((mcconnaughey_similarity(3, 5, 4) - 0.35).abs() < f64::EPSILON);
/// assert!((mcconnaughey_similarity(0, 0, 0) - 1.0).abs() < f64::EPSILON);
/// ```
#[inline]
#[must_use]
pub fn mcconnaughey_similarity(intersection: usize, size_a: usize, size_b: usize) -> f64 {
    debug_assert!(
        intersection <= size_a,
        "intersection ({intersection}) must be <= size_a ({size_a})"
    );
    debug_assert!(
        intersection <= size_b,
        "intersection ({intersection}) must be <= size_b ({size_b})"
    );
    let product = size_a * size_b;
    if product == 0 {
        return if size_a == 0 && size_b == 0 { 1.0 } else { 0.0 };
    }
    let diff_a = size_a - intersection;
    let diff_b = size_b - intersection;
    #[allow(clippy::cast_precision_loss)]
    {
        let numerator = (intersection * intersection) as f64 - (diff_a * diff_b) as f64;
        numerator / product as f64
    }
}

/// Trait for types that carry graph comparison results and can compute
/// similarity metrics.
///
/// Implementors provide 6 required accessor methods returning counts.
/// All similarity metrics are provided as default methods.
///
/// # Examples
///
/// ```
/// use geometric_traits::prelude::GraphSimilarities;
///
/// struct MyResult {
///     common_e: usize,
///     common_v: usize,
///     v1: usize,
///     e1: usize,
///     v2: usize,
///     e2: usize,
/// }
///
/// impl GraphSimilarities for MyResult {
///     fn common_edges(&self) -> usize {
///         self.common_e
///     }
///     fn common_vertices(&self) -> usize {
///         self.common_v
///     }
///     fn first_graph_vertices(&self) -> usize {
///         self.v1
///     }
///     fn first_graph_edges(&self) -> usize {
///         self.e1
///     }
///     fn second_graph_vertices(&self) -> usize {
///         self.v2
///     }
///     fn second_graph_edges(&self) -> usize {
///         self.e2
///     }
/// }
///
/// let result = MyResult { common_e: 3, common_v: 4, v1: 5, e1: 6, v2: 7, e2: 8 };
/// let j = result.johnson_similarity();
/// let expected = 49.0 / (11.0 * 15.0);
/// assert!((j - expected).abs() < f64::EPSILON);
/// ```
pub trait GraphSimilarities {
    /// Number of edges in common between the two graphs.
    fn common_edges(&self) -> usize;

    /// Number of vertices in common between the two graphs.
    fn common_vertices(&self) -> usize;

    /// Number of vertices in the first graph.
    fn first_graph_vertices(&self) -> usize;

    /// Number of edges in the first graph.
    fn first_graph_edges(&self) -> usize;

    /// Number of vertices in the second graph.
    fn second_graph_vertices(&self) -> usize;

    /// Number of edges in the second graph.
    fn second_graph_edges(&self) -> usize;

    /// Jaccard (Tanimoto) similarity on edge sets.
    #[inline]
    fn edge_jaccard_similarity(&self) -> f64 {
        tanimoto_similarity(
            self.common_edges(),
            self.first_graph_edges(),
            self.second_graph_edges(),
        )
    }

    /// Dice (Sorensen-Dice) similarity on edge sets.
    #[inline]
    fn edge_dice_similarity(&self) -> f64 {
        dice_similarity(self.common_edges(), self.first_graph_edges(), self.second_graph_edges())
    }

    /// Overlap (Szymkiewicz-Simpson) similarity on edge sets.
    #[inline]
    fn edge_overlap_similarity(&self) -> f64 {
        overlap_similarity(self.common_edges(), self.first_graph_edges(), self.second_graph_edges())
    }

    /// Cosine similarity on edge sets.
    #[inline]
    fn edge_cosine_similarity(&self) -> f64 {
        cosine_similarity(self.common_edges(), self.first_graph_edges(), self.second_graph_edges())
    }

    /// Jaccard (Tanimoto) similarity on vertex sets.
    #[inline]
    fn vertex_jaccard_similarity(&self) -> f64 {
        tanimoto_similarity(
            self.common_vertices(),
            self.first_graph_vertices(),
            self.second_graph_vertices(),
        )
    }

    /// Dice (Sorensen-Dice) similarity on vertex sets.
    #[inline]
    fn vertex_dice_similarity(&self) -> f64 {
        dice_similarity(
            self.common_vertices(),
            self.first_graph_vertices(),
            self.second_graph_vertices(),
        )
    }

    /// Overlap (Szymkiewicz-Simpson) similarity on vertex sets.
    #[inline]
    fn vertex_overlap_similarity(&self) -> f64 {
        overlap_similarity(
            self.common_vertices(),
            self.first_graph_vertices(),
            self.second_graph_vertices(),
        )
    }

    /// Cosine similarity on vertex sets.
    #[inline]
    fn vertex_cosine_similarity(&self) -> f64 {
        cosine_similarity(
            self.common_vertices(),
            self.first_graph_vertices(),
            self.second_graph_vertices(),
        )
    }

    /// Tversky similarity on edge sets (Tversky 1977).
    ///
    /// Asymmetric: `alpha` weights edges unique to the first graph,
    /// `beta` weights edges unique to the second graph.
    #[inline]
    fn edge_tversky_similarity(&self, alpha: f64, beta: f64) -> f64 {
        tversky_similarity(
            self.common_edges(),
            self.first_graph_edges(),
            self.second_graph_edges(),
            alpha,
            beta,
        )
    }

    /// Tversky similarity on vertex sets (Tversky 1977).
    ///
    /// Asymmetric: `alpha` weights vertices unique to the first graph,
    /// `beta` weights vertices unique to the second graph.
    #[inline]
    fn vertex_tversky_similarity(&self, alpha: f64, beta: f64) -> f64 {
        tversky_similarity(
            self.common_vertices(),
            self.first_graph_vertices(),
            self.second_graph_vertices(),
            alpha,
            beta,
        )
    }

    /// Kulczynski similarity on edge sets (2nd coefficient).
    #[inline]
    fn edge_kulczynski_similarity(&self) -> f64 {
        kulczynski_similarity(
            self.common_edges(),
            self.first_graph_edges(),
            self.second_graph_edges(),
        )
    }

    /// Kulczynski similarity on vertex sets (2nd coefficient).
    #[inline]
    fn vertex_kulczynski_similarity(&self) -> f64 {
        kulczynski_similarity(
            self.common_vertices(),
            self.first_graph_vertices(),
            self.second_graph_vertices(),
        )
    }

    /// Braun-Blanquet similarity on edge sets.
    #[inline]
    fn edge_braun_blanquet_similarity(&self) -> f64 {
        braun_blanquet_similarity(
            self.common_edges(),
            self.first_graph_edges(),
            self.second_graph_edges(),
        )
    }

    /// Braun-Blanquet similarity on vertex sets.
    #[inline]
    fn vertex_braun_blanquet_similarity(&self) -> f64 {
        braun_blanquet_similarity(
            self.common_vertices(),
            self.first_graph_vertices(),
            self.second_graph_vertices(),
        )
    }

    /// Sokal-Sneath similarity on edge sets (first coefficient).
    #[inline]
    fn edge_sokal_sneath_similarity(&self) -> f64 {
        sokal_sneath_similarity(
            self.common_edges(),
            self.first_graph_edges(),
            self.second_graph_edges(),
        )
    }

    /// Sokal-Sneath similarity on vertex sets (first coefficient).
    #[inline]
    fn vertex_sokal_sneath_similarity(&self) -> f64 {
        sokal_sneath_similarity(
            self.common_vertices(),
            self.first_graph_vertices(),
            self.second_graph_vertices(),
        )
    }

    /// McConnaughey similarity on edge sets. Range: \[-1, 1\].
    #[inline]
    fn edge_mcconnaughey_similarity(&self) -> f64 {
        mcconnaughey_similarity(
            self.common_edges(),
            self.first_graph_edges(),
            self.second_graph_edges(),
        )
    }

    /// McConnaughey similarity on vertex sets. Range: \[-1, 1\].
    #[inline]
    fn vertex_mcconnaughey_similarity(&self) -> f64 {
        mcconnaughey_similarity(
            self.common_vertices(),
            self.first_graph_vertices(),
            self.second_graph_vertices(),
        )
    }

    /// Johnson graph similarity (Raymond, Gardiner, Willett 2002).
    ///
    /// `J = (common_edges + common_vertices)^2 / ((V1 + E1) * (V2 + E2))`
    #[inline]
    fn johnson_similarity(&self) -> f64 {
        johnson_similarity(
            self.common_edges(),
            self.common_vertices(),
            self.first_graph_vertices(),
            self.first_graph_edges(),
            self.second_graph_vertices(),
            self.second_graph_edges(),
        )
    }
}
