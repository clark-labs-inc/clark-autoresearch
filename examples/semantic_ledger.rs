//! Semantic ledger retrieval with clark-hash (requires the `similarity` feature).
//!
//! Builds a [`SemanticSketches`] index over a few observations using a
//! model-free bag-of-characters embedder, then queries it. On a real run,
//! swap [`CharBagEmbedder`] for an [`Embedder`] backed by a real sentence
//! model — the sketches and search stay the same.
//!
//! Run with: `cargo run --example semantic_ledger --features similarity`

use anyhow::Result;
use clark_autoresearch::Embedder;
use clark_autoresearch::lm_loop::SemanticSketches;

/// A model-free embedder: a normalized bag-of-characters hashed into `dim`
/// buckets. Texts sharing characters produce similar vectors; disjoint texts
/// do not. Good enough to demonstrate the sketch + search pipeline.
struct CharBagEmbedder {
    dim: usize,
}

impl Embedder for CharBagEmbedder {
    fn embed_text(&self, text: &str) -> Result<Vec<f32>> {
        let mut v = vec![0.0_f32; self.dim];
        for ch in text.chars() {
            v[(ch as usize) % self.dim] += 1.0;
        }
        let norm = v
            .iter()
            .map(|x| x * x)
            .sum::<f32>()
            .sqrt()
            .max(f32::MIN_POSITIVE);
        for x in &mut v {
            *x /= norm;
        }
        Ok(v)
    }
}

fn main() -> Result<()> {
    let embedder = CharBagEmbedder { dim: 64 };
    let mut sketches = SemanticSketches::new(64);

    let observations = [
        "csrf token is missing on the login form",
        "session cookie lacks secure flag",
        "rate limiting is not enforced on /api/login",
        "the database index on users.email is missing",
    ];
    for obs in observations {
        sketches.add_observation_text(obs, &embedder)?;
    }

    let query = embedder.embed_text("the login endpoint has no csrf protection")?;
    println!("query: 'the login endpoint has no csrf protection'");
    println!("top-2 similar observations:");
    for (idx, score) in sketches.find_similar(&query, 2)? {
        println!("  [{idx:.0}] score={score:.3}  {}", observations[idx]);
    }

    let max_sim = sketches.max_similarity(&query)?;
    println!("novelty (1 - max_similarity) = {:.3}", 1.0 - max_sim);
    Ok(())
}
