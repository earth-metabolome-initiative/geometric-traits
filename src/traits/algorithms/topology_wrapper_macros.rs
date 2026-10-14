macro_rules! define_planarity_derived_error {
    (
        $(#[$enum_meta:meta])*
        $error_vis:vis enum $error_name:ident => $algorithm_variant:ident,
        mapper = $mapper_name:ident,
        self_loop = $self_loop_message:literal,
        parallel = $parallel_message:literal
    ) => {
        #[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
        $(#[$enum_meta])*
        $error_vis enum $error_name {
            /// The graph contains a self-loop, which the topology detector does not support.
            #[error($self_loop_message)]
            SelfLoopsUnsupported,
            /// The graph contains parallel edges, which the topology detector does not support.
            #[error($parallel_message)]
            ParallelEdgesUnsupported,
            /// The graph exposed an edge endpoint that is out of range for the node count.
            #[error(
                "The graph exposed edge endpoint {endpoint}, which is out of range for node_count={node_count}."
            )]
            InvalidEdgeEndpoint {
                /// The out-of-range edge endpoint.
                endpoint: usize,
                /// The node count that the endpoint exceeded.
                node_count: usize,
            },
        }

        impl From<$error_name>
            for crate::errors::monopartite_graph_error::algorithms::MonopartiteAlgorithmError
        {
            #[inline]
            fn from(error: $error_name) -> Self {
                Self::$algorithm_variant(error)
            }
        }

        impl<G: crate::traits::MonopartiteGraph> From<$error_name>
            for crate::errors::MonopartiteError<G>
        {
            #[inline]
            fn from(error: $error_name) -> Self {
                Self::AlgorithmError(error.into())
            }
        }

        #[allow(clippy::needless_pass_by_value)]
        #[inline]
        fn $mapper_name<G: crate::traits::MonopartiteGraph>(
            error: crate::traits::algorithms::PlanarityError,
        ) -> crate::errors::MonopartiteError<G> {
            match error {
                crate::traits::algorithms::PlanarityError::SelfLoopsUnsupported => {
                    $error_name::SelfLoopsUnsupported.into()
                }
                crate::traits::algorithms::PlanarityError::ParallelEdgesUnsupported => {
                    $error_name::ParallelEdgesUnsupported.into()
                }
                crate::traits::algorithms::PlanarityError::InvalidEdgeEndpoint {
                    endpoint,
                    node_count,
                } => $error_name::InvalidEdgeEndpoint { endpoint, node_count }.into(),
            }
        }
    };
}

pub(crate) use define_planarity_derived_error;
