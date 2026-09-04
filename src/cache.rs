//! Evaluation cache: skip redundant rollouts for (candidate, example) pairs.
//!
//! Mirrors GEPA's `EvaluationCache` (`gepa.core.state.EvaluationCache`). When
//! the frontier re-samples a parent, the same (candidate, example) pair is
//! often re-evaluated; the cache returns the stored result instead of calling
//! the adapter again, saving metric calls and budget. the crate's cache is exact
//! on the candidate hash now; the optional `similarity` feature (Phase 5)
//! extends it to *semantic* near-duplicate rejection.

use std::collections::BTreeMap;

use anyhow::Result;
use sha1::{Digest, Sha1};

use crate::adapter::{Candidate, EvaluationBatch, ObjectiveScores};

/// A stable hash of a candidate, over its sorted (component, text) pairs.
///
/// `Candidate` is a `BTreeMap`, so iteration is in sorted key order — the hash
/// is independent of insertion order. This is a deduplication key, not a
/// security primitive.
pub fn candidate_hash(candidate: &Candidate) -> String {
    let mut hasher = Sha1::new();
    for (component, text) in candidate {
        hasher.update(component.as_bytes());
        hasher.update([0]);
        hasher.update(text.as_bytes());
        hasher.update([0]);
    }
    let digest = hasher.finalize();
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct CacheKey {
    candidate_hash: String,
    example_id: String,
}

#[derive(Clone, Debug)]
struct CachedEval {
    score: f64,
    output: serde_json::Value,
    objective_scores: Option<ObjectiveScores>,
}

/// An exact-match cache of (candidate, example) evaluation results.
#[derive(Default)]
pub struct EvaluationCache {
    entries: BTreeMap<CacheKey, CachedEval>,
    hits: u32,
    misses: u32,
}

impl EvaluationCache {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// How many lookups were served from the cache.
    pub fn hits(&self) -> u32 {
        self.hits
    }

    /// How many lookups required a fresh evaluation.
    pub fn misses(&self) -> u32 {
        self.misses
    }

    fn get(&mut self, candidate: &Candidate, example_id: &str) -> Option<CachedEval> {
        let key = CacheKey {
            candidate_hash: candidate_hash(candidate),
            example_id: example_id.to_string(),
        };
        let found = self.entries.get(&key).cloned();
        if found.is_some() {
            self.hits += 1;
        } else {
            self.misses += 1;
        }
        found
    }

    fn insert(
        &mut self,
        candidate: &Candidate,
        example_id: String,
        score: f64,
        output: serde_json::Value,
        objective_scores: Option<ObjectiveScores>,
    ) {
        let key = CacheKey {
            candidate_hash: candidate_hash(candidate),
            example_id,
        };
        self.entries.insert(
            key,
            CachedEval {
                score,
                output,
                objective_scores,
            },
        );
    }
}

/// Evaluate `candidate` on `inputs`, serving cached (candidate, example)
/// results where available and calling `adapter` only for the misses.
///
/// Returns a full [`EvaluationBatch`] aligned with `inputs`/`example_ids`.
/// `num_metric_calls` reflects only the examples actually evaluated (the
/// misses), so the loop's budget tracking credits the cache.
pub fn cached_evaluate<A: crate::adapter::ResearchAdapter>(
    adapter: &A,
    cache: &mut EvaluationCache,
    inputs: &[serde_json::Value],
    example_ids: &[String],
    candidate: &Candidate,
    capture_traces: bool,
) -> Result<EvaluationBatch> {
    debug_assert_eq!(inputs.len(), example_ids.len());

    // Partition into cached hits and misses, preserving positions.
    let mut cached: Vec<Option<CachedEval>> = Vec::with_capacity(inputs.len());
    let mut miss_indices: Vec<usize> = Vec::new();
    for (i, example_id) in example_ids.iter().enumerate() {
        match cache.get(candidate, example_id) {
            Some(entry) => cached.push(Some(entry)),
            None => {
                cached.push(None);
                miss_indices.push(i);
            }
        }
    }

    let num_metric_calls = miss_indices.len() as u32;

    if miss_indices.is_empty() {
        // Everything cached: reassemble without calling the adapter.
        return Ok(reassemble(cached, None, &[], num_metric_calls));
    }

    let miss_inputs: Vec<serde_json::Value> =
        miss_indices.iter().map(|&i| inputs[i].clone()).collect();
    let fresh = adapter.evaluate(&miss_inputs, candidate, capture_traces)?;

    // Store the fresh per-example results in the cache.
    for (k, &i) in miss_indices.iter().enumerate() {
        let score = fresh.scores.get(k).copied().unwrap_or(0.0);
        let output = fresh.outputs.get(k).cloned().unwrap_or_default();
        let objective_scores = fresh
            .objective_scores
            .as_ref()
            .and_then(|scores| scores.get(k).cloned());
        cache.insert(
            candidate,
            example_ids[i].clone(),
            score,
            output.clone(),
            objective_scores.clone(),
        );
        cached[i] = Some(CachedEval {
            score,
            output,
            objective_scores,
        });
    }

    Ok(reassemble(
        cached,
        Some(fresh),
        &miss_indices,
        num_metric_calls,
    ))
}

fn reassemble(
    cached: Vec<Option<CachedEval>>,
    fresh: Option<EvaluationBatch>,
    miss_indices: &[usize],
    num_metric_calls: u32,
) -> EvaluationBatch {
    let n = cached.len();
    let mut scores = Vec::with_capacity(n);
    let mut outputs = Vec::with_capacity(n);
    let mut any_trajectory = fresh.as_ref().and_then(|f| f.trajectories.clone());
    let mut trajectories: Vec<Option<serde_json::Value>> = vec![None; n];
    let mut any_objective = false;
    let mut objective_scores: Vec<Option<ObjectiveScores>> = vec![None; n];

    for (i, entry) in cached.into_iter().enumerate() {
        let Some(entry) = entry else {
            // Should not happen — misses were filled above.
            scores.push(0.0);
            outputs.push(serde_json::Value::Null);
            continue;
        };
        scores.push(entry.score);
        outputs.push(entry.output);
        if let Some(obj) = entry.objective_scores {
            any_objective = true;
            objective_scores[i] = Some(obj);
        }
    }

    // Splice fresh trajectories back into position for misses (best-effort:
    // traces are only meaningful for fresh evals; cached entries carry none).
    if let (Some(_fresh), Some(ref mut fresh_traces)) = (fresh.as_ref(), any_trajectory.as_mut()) {
        for (k, &i) in miss_indices.iter().enumerate() {
            if let Some(trace) = fresh_traces.get(k).cloned() {
                trajectories[i] = Some(trace);
            }
        }
        // Drop the trajectory channel entirely if no fresh traces were captured.
        if !trajectories.iter().any(Option::is_some) {
            any_trajectory = None;
        }
    }

    let trajectories = any_trajectory.map(|_| {
        trajectories
            .into_iter()
            .map(|opt| opt.unwrap_or(serde_json::Value::Null))
            .collect()
    });
    let objective_scores = if any_objective {
        Some(
            objective_scores
                .into_iter()
                .map(|opt| opt.unwrap_or_default())
                .collect(),
        )
    } else {
        None
    };

    EvaluationBatch {
        scores,
        outputs,
        trajectories,
        objective_scores,
        num_metric_calls,
    }
}

// ---------------------------------------------------------------------------
// Semantic candidate cache (optional, behind the `similarity` feature).
// ---------------------------------------------------------------------------

#[cfg(feature = "similarity")]
mod semantic_cache {
    use anyhow::Result;

    use crate::adapter::Candidate;
    use crate::embedding::Embedder;

    /// A semantic near-duplicate guard over evaluated candidates.
    ///
    /// GEPA's `EvaluationCache` is exact-match on the candidate dict — it misses
    /// paraphrases. This store keeps a clark-hash sketch of each evaluated
    /// candidate's text and rejects a proposed candidate whose sketch is
    /// near-identical to an existing one, so the frontier does not re-evaluate
    /// semantic duplicates.
    #[derive(Clone, Debug)]
    pub struct SemanticCandidateCache {
        input_dim: usize,
        sketches: Vec<clark_hash::QuantizedVector>,
    }

    impl SemanticCandidateCache {
        pub fn new(input_dim: usize) -> Self {
            Self {
                input_dim,
                sketches: Vec::new(),
            }
        }

        pub fn len(&self) -> usize {
            self.sketches.len()
        }

        pub fn is_empty(&self) -> bool {
            self.sketches.is_empty()
        }

        /// Embed `candidate`'s text via `embedder`, encode a sketch, and store it.
        pub fn add<E: Embedder>(&mut self, candidate: &Candidate, embedder: &E) -> Result<()> {
            let text = candidate.values().cloned().collect::<Vec<_>>().join(" ");
            self.add_text(&text, embedder)
        }

        /// Encode and store a sketch for an arbitrary text (e.g. a candidate
        /// description rather than its raw components).
        pub fn add_text<E: Embedder>(&mut self, text: &str, embedder: &E) -> Result<()> {
            let embedding = embedder.embed_text(text)?;
            let codec = clark_hash::ClarkHash::new(clark_hash::SQuaJLConfig::new(self.input_dim))?;
            self.sketches.push(codec.encode(&embedding)?);
            Ok(())
        }

        /// Returns `true` if `candidate`'s text is within `threshold` similarity
        /// of any stored sketch (i.e. it is a near-duplicate worth rejecting).
        pub fn is_near_duplicate<E: Embedder>(
            &self,
            candidate: &Candidate,
            embedder: &E,
            threshold: f64,
        ) -> Result<bool> {
            if self.sketches.is_empty() {
                return Ok(false);
            }
            let text = candidate.values().cloned().collect::<Vec<_>>().join(" ");
            let embedding = embedder.embed_text(&text)?;
            let codec = clark_hash::ClarkHash::new(clark_hash::SQuaJLConfig::new(self.input_dim))?;
            let prepared = codec.sketch_query(&embedding)?;
            for sketch in &self.sketches {
                let score = codec.score(&prepared, sketch)? as f64;
                if score >= threshold {
                    return Ok(true);
                }
            }
            Ok(false)
        }
    }
}

#[cfg(feature = "similarity")]
pub use semantic_cache::SemanticCandidateCache;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adapter::ResearchAdapter;

    /// An adapter that charges one metric call per example and scores by a
    /// "score" component, so the cache's effect on num_metric_calls is visible.
    struct CountingAdapter;
    impl ResearchAdapter for CountingAdapter {
        fn evaluate(
            &self,
            batch: &[serde_json::Value],
            candidate: &Candidate,
            _capture_traces: bool,
        ) -> Result<EvaluationBatch> {
            let base: f64 = candidate
                .get("score")
                .and_then(|s| s.parse().ok())
                .unwrap_or(0.0);
            Ok(EvaluationBatch {
                scores: batch.iter().map(|_| base).collect(),
                outputs: batch.to_vec(),
                trajectories: None,
                objective_scores: None,
                num_metric_calls: batch.len() as u32,
            })
        }
    }

    #[test]
    fn repeated_evaluation_is_served_from_cache() {
        // Reproduction for Phase 3: the second evaluation of the same
        // (candidate, example) pair costs zero metric calls (a cache hit).
        let adapter = CountingAdapter;
        let mut cache = EvaluationCache::new();
        let candidate: Candidate = [("score".to_string(), "0.5".to_string())]
            .into_iter()
            .collect();
        let inputs = vec![serde_json::json!("q1"), serde_json::json!("q2")];
        let ids = vec!["e1".to_string(), "e2".to_string()];

        let first =
            cached_evaluate(&adapter, &mut cache, &inputs, &ids, &candidate, false).unwrap();
        assert_eq!(first.num_metric_calls, 2);
        assert_eq!(cache.misses(), 2);

        let second =
            cached_evaluate(&adapter, &mut cache, &inputs, &ids, &candidate, false).unwrap();
        assert_eq!(
            second.num_metric_calls, 0,
            "second call should be all cache hits"
        );
        assert_eq!(cache.hits(), 2);
        assert_eq!(second.scores, first.scores);
    }

    #[test]
    fn candidate_hash_is_order_independent_and_stable() {
        let a: Candidate = [
            ("x".to_string(), "1".to_string()),
            ("y".to_string(), "2".to_string()),
        ]
        .into_iter()
        .collect();
        let b: Candidate = [
            ("y".to_string(), "2".to_string()),
            ("x".to_string(), "1".to_string()),
        ]
        .into_iter()
        .collect();
        assert_eq!(candidate_hash(&a), candidate_hash(&b));
        assert!(!candidate_hash(&a).is_empty());
    }

    #[test]
    fn distinct_candidates_get_distinct_hashes() {
        let a: Candidate = [("x".to_string(), "1".to_string())].into_iter().collect();
        let b: Candidate = [("x".to_string(), "2".to_string())].into_iter().collect();
        assert_ne!(candidate_hash(&a), candidate_hash(&b));
    }
}
