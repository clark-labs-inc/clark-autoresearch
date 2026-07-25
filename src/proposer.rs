//! Proposers generate new candidates from a parent and a reflective dataset.
//!
//! Mirrors GEPA's `ReflectiveMutationProposer`, but the language model is
//! *injected* via the [`LanguageModel`] trait rather than bundled. clark never
//! calls a provider directly; the host supplies the LM. GEPA's
//! `gepa.oa.AutoResearchEngine` instead shells out to `claude --print` inside
//! the optimizer — clark keeps that execution behind the trait so the same
//! loop runs against any provider.

use anyhow::Result;

use crate::adapter::{Candidate, ReflectiveDataset};

/// A language model the host provides. clark never calls a provider directly;
/// the host implements this (over an HTTP API, a local model, a CLI, …).
pub trait LanguageModel {
    /// Complete the prompt and return the raw text response.
    fn complete(&self, prompt: &str) -> Result<String>;
}

/// A proposer generates a new candidate from a parent and a reflective dataset.
pub trait Proposer {
    /// Produce a new candidate from `parent`, reflecting on `reflective_dataset`
    /// for the named `components`.
    ///
    /// `history` is the rendered research dossier (the whole ledger history),
    /// so reflection can read past observations/hypotheses/results — not just
    /// the last minibatch's traces. This is clark's unique lever over GEPA,
    /// whose reflective dataset is per-iteration and per-component only.
    fn propose(
        &self,
        parent: &Candidate,
        reflective_dataset: &ReflectiveDataset,
        components: &[String],
        history: Option<&str>,
    ) -> Result<Candidate>;
}

/// The default reflection prompt template. Placeholders:
/// - `<curr_param>`: the current text of the component being evolved.
/// - `<side_info>`: the per-example reflective entries (serialized JSON).
/// - `<history>`: the rendered research dossier (past ledger knowledge).
///
/// Matches GEPA's `InstructionProposalSignature` placeholder contract, plus
/// the `<history>` slot clark adds for ledger-backed reflection.
pub const DEFAULT_REFLECTION_TEMPLATE: &str = "\
You are improving a component of a system. Analyze the execution feedback and
propose an improved version.

## Current component text
<curr_param>

## Execution feedback (inputs, outputs, scores; higher score is better)
<side_info>

## Prior research history (observations, hypotheses, results)
<history>

## Task
Propose an improved version of the component text. Reuse prior findings before
rediscovering them. Return ONLY a JSON object mapping the component name to the
new text, e.g. {\"prompt\": \"...\"}. No markdown fences, no explanation.";

/// GEPA's `ReflectiveMutationProposer`, but with the LM injected.
///
/// For each component to update, this builds a reflection prompt from the
/// component's current text and its reflective entries, asks the LM for an
/// improved version, and parses the response into the new candidate.
pub struct ReflectiveMutation<L: LanguageModel> {
    lm: L,
    template: String,
}

impl<L: LanguageModel> ReflectiveMutation<L> {
    pub fn new(lm: L) -> Self {
        Self {
            lm,
            template: DEFAULT_REFLECTION_TEMPLATE.to_string(),
        }
    }

    pub fn with_template(mut self, template: impl Into<String>) -> Self {
        self.template = template.into();
        self
    }
}

impl<L: LanguageModel> Proposer for ReflectiveMutation<L> {
    fn propose(
        &self,
        parent: &Candidate,
        reflective_dataset: &ReflectiveDataset,
        components: &[String],
        history: Option<&str>,
    ) -> Result<Candidate> {
        let mut new_candidate = parent.clone();
        let history = history.unwrap_or("(none yet)");
        for component in components {
            let curr = parent.get(component).cloned().unwrap_or_default();
            let side_info = reflective_dataset
                .get(component)
                .map(|entries| serde_json::to_string_pretty(entries).unwrap_or_default())
                .unwrap_or_default();
            let prompt = self
                .template
                .replace("<curr_param>", &curr)
                .replace("<side_info>", &side_info)
                .replace("<history>", history);
            let response = self.lm.complete(&prompt)?;
            let new_text = parse_component_response(&response, component).unwrap_or_else(|| {
                // Fall back to the trimmed raw response when the LM ignores the
                // JSON contract — keeps the loop progressing on real models.
                response.trim().to_string()
            });
            new_candidate.insert(component.clone(), new_text);
        }
        Ok(new_candidate)
    }
}

/// Parse the LM response into the new text for `component`.
///
/// Accepts either a JSON object `{"component": "new text"}` or, when a single
/// component is in play, a plain string.
fn parse_component_response(response: &str, component: &str) -> Option<String> {
    let trimmed = response.trim();
    // Strip a leading/trailing ``` fence if present (models often add one).
    let body = trimmed
        .strip_prefix("```json")
        .or_else(|| trimmed.strip_prefix("```"))
        .map(|s| s.trim_start())
        .unwrap_or(trimmed)
        .trim_end_matches("```")
        .trim();
    if let Ok(value) = serde_json::from_str::<serde_json::Value>(body) {
        if let Some(text) = value.get(component).and_then(|v| v.as_str()) {
            return Some(text.to_string());
        }
        if let Some(text) = value.as_str() {
            return Some(text.to_string());
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adapter::ReflectiveEntry;

    /// A deterministic LM that echoes a fixed JSON response, simulating a
    /// reflection model that proposes a longer prompt each call.
    struct FakeLm {
        responses: Vec<String>,
        calls: std::cell::Cell<usize>,
    }

    impl FakeLm {
        fn new(responses: Vec<String>) -> Self {
            Self {
                responses,
                calls: std::cell::Cell::new(0),
            }
        }
    }

    impl LanguageModel for FakeLm {
        fn complete(&self, _prompt: &str) -> Result<String> {
            let i = self.calls.get();
            self.calls.set(i + 1);
            Ok(self
                .responses
                .get(i)
                .cloned()
                .unwrap_or_else(|| r#"{"prompt": "improved"}"#.to_string()))
        }
    }

    #[test]
    fn reflective_mutation_updates_component_from_json() {
        let lm = FakeLm::new(vec![r#"{"prompt": "be explicit and concise"}"#.to_string()]);
        let proposer = ReflectiveMutation::new(lm);
        let parent: Candidate = [("prompt".to_string(), "be brief".to_string())]
            .into_iter()
            .collect();
        let mut dataset = ReflectiveDataset::new();
        dataset.insert(
            "prompt".to_string(),
            vec![ReflectiveEntry {
                score: 0.4,
                ..Default::default()
            }],
        );
        let child = proposer
            .propose(&parent, &dataset, &["prompt".to_string()], None)
            .unwrap();
        assert_eq!(child.get("prompt").unwrap(), "be explicit and concise");
    }

    #[test]
    fn parse_handles_fenced_json() {
        let text = parse_component_response("```json\n{\"prompt\": \"x\"}\n```", "prompt");
        assert_eq!(text.as_deref(), Some("x"));
    }

    #[test]
    fn parse_falls_back_to_none_for_garbage() {
        assert!(parse_component_response("not json at all", "prompt").is_none());
    }
}
