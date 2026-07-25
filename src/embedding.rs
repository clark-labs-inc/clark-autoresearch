//! Host-implemented text embedding.
//!
//! Producing an embedding requires a model, which is execution — so the host
//! implements [`Embedder`]. clark never runs a model directly. The resulting
//! `Vec<f32>` feeds the optional `similarity` feature: clark-hash sketches for
//! the research ledger ([`crate::lm_loop::SemanticSketches`]) and the semantic
//! candidate cache ([`crate::cache::SemanticCandidateCache`]).
//!
//! This module is dep-free and always available; the clark-hash backend that
//! *consumes* embeddings is gated behind the `similarity` Cargo feature.

use anyhow::Result;

/// A text embedder the host provides.
///
/// Implement this over an HTTP embedding API, a local model, or a CLI. clark
/// calls only [`Self::embed_text`]; it never loads a model or touches the
/// network itself.
pub trait Embedder {
    /// Embed `text` into a dense float vector. Higher dimensions improve the
    /// quality of the clark-hash sketch; the ledger's [`SemanticSketches`] is
    /// configured with a matching input dimension.
    fn embed_text(&self, text: &str) -> Result<Vec<f32>>;
}
