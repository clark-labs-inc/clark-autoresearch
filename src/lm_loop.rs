use std::collections::BTreeSet;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::ids::stable_id;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceConfidence {
    Low,
    #[default]
    Medium,
    High,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum HypothesisState {
    #[default]
    Proposed,
    Active,
    Supported,
    Refuted,
    Inconclusive,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ExperimentState {
    #[default]
    Proposed,
    Running,
    Complete,
    Blocked,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ResultVerdict {
    Supports,
    Refutes,
    Expands,
    #[default]
    Inconclusive,
    Blocked,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct WorkBudget {
    #[serde(default)]
    pub max_agent_turns: Option<u32>,
    #[serde(default)]
    pub max_tool_calls: Option<u32>,
    #[serde(default)]
    pub expected_duration: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Observation {
    pub id: String,
    pub target: String,
    pub source: String,
    pub summary: String,
    #[serde(default)]
    pub detail: String,
    #[serde(default)]
    pub artifact_refs: Vec<String>,
    #[serde(default)]
    pub confidence: EvidenceConfidence,
    pub created_at_ms: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct HypothesisCandidate {
    pub id: String,
    pub statement: String,
    #[serde(default)]
    pub target: Option<String>,
    #[serde(default)]
    pub rationale: String,
    #[serde(default)]
    pub status: HypothesisState,
    #[serde(default)]
    pub evidence_ids: Vec<String>,
    pub created_at_ms: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExperimentPlan {
    pub id: String,
    #[serde(default)]
    pub hypothesis_id: Option<String>,
    pub objective: String,
    #[serde(default)]
    pub target: Option<String>,
    #[serde(default)]
    pub required_tools: Vec<String>,
    #[serde(default)]
    pub budget: WorkBudget,
    #[serde(default)]
    pub status: ExperimentState,
    pub created_at_ms: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResearchResult {
    pub id: String,
    #[serde(default)]
    pub experiment_id: Option<String>,
    pub verdict: ResultVerdict,
    pub summary: String,
    #[serde(default)]
    pub evidence_ids: Vec<String>,
    pub created_at_ms: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LedgerEvent {
    pub id: String,
    pub kind: String,
    pub ref_id: String,
    pub summary: String,
    pub created_at_ms: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResearchDossier {
    pub run_id: String,
    pub targets: Vec<String>,
    pub observations: Vec<Observation>,
    pub hypotheses: Vec<HypothesisCandidate>,
    pub experiments: Vec<ExperimentPlan>,
    pub results: Vec<ResearchResult>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContextPack {
    pub run_id: String,
    pub current_target: String,
    pub current_node_id: String,
    pub mission: String,
    pub dossier: ResearchDossier,
    pub rendered: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResearchLedger {
    pub run_id: String,
    pub targets: Vec<String>,
    #[serde(default)]
    pub observations: Vec<Observation>,
    #[serde(default)]
    pub hypotheses: Vec<HypothesisCandidate>,
    #[serde(default)]
    pub experiments: Vec<ExperimentPlan>,
    #[serde(default)]
    pub results: Vec<ResearchResult>,
    #[serde(default)]
    pub events: Vec<LedgerEvent>,
    pub updated_at_ms: u64,
}

impl ResearchLedger {
    pub fn new(run_id: impl Into<String>, targets: Vec<String>) -> Self {
        Self {
            run_id: run_id.into(),
            targets,
            observations: Vec::new(),
            hypotheses: Vec::new(),
            experiments: Vec::new(),
            results: Vec::new(),
            events: Vec::new(),
            updated_at_ms: now_millis(),
        }
    }

    pub fn record_observation(
        &mut self,
        target: impl Into<String>,
        source: impl Into<String>,
        summary: impl Into<String>,
        detail: impl Into<String>,
        confidence: EvidenceConfidence,
    ) -> String {
        let target = target.into();
        let source = source.into();
        let summary = clean_line(summary.into());
        let detail = clean_detail(detail.into());
        let id = stable_id(
            "obs",
            [
                self.run_id.as_str(),
                target.as_str(),
                source.as_str(),
                summary.as_str(),
                detail.as_str(),
            ],
        );
        if !self.observations.iter().any(|item| item.id == id) {
            let now = now_millis();
            self.observations.push(Observation {
                id: id.clone(),
                target,
                source,
                summary: summary.clone(),
                detail,
                artifact_refs: Vec::new(),
                confidence,
                created_at_ms: now,
            });
            self.record_event("observation", &id, &summary, now);
        }
        self.updated_at_ms = now_millis();
        id
    }

    pub fn record_hypothesis(
        &mut self,
        statement: impl Into<String>,
        target: Option<String>,
        rationale: impl Into<String>,
        evidence_ids: Vec<String>,
    ) -> String {
        let statement = clean_line(statement.into());
        let rationale = clean_detail(rationale.into());
        let target_key = target.clone().unwrap_or_default();
        let id = stable_id(
            "hyp",
            [
                self.run_id.as_str(),
                target_key.as_str(),
                statement.as_str(),
            ],
        );
        if !self.hypotheses.iter().any(|item| item.id == id) {
            let now = now_millis();
            self.hypotheses.push(HypothesisCandidate {
                id: id.clone(),
                statement: statement.clone(),
                target,
                rationale,
                status: HypothesisState::Proposed,
                evidence_ids: unique(evidence_ids),
                created_at_ms: now,
            });
            self.record_event("hypothesis", &id, &statement, now);
        }
        self.updated_at_ms = now_millis();
        id
    }

    pub fn record_experiment(&mut self, plan: ExperimentPlan) -> String {
        let id = plan.id.clone();
        if !self.experiments.iter().any(|item| item.id == id) {
            let now = plan.created_at_ms;
            let summary = plan.objective.clone();
            self.experiments.push(plan);
            self.record_event("experiment", &id, &summary, now);
        }
        self.updated_at_ms = now_millis();
        id
    }

    pub fn record_result(
        &mut self,
        experiment_id: Option<String>,
        verdict: ResultVerdict,
        summary: impl Into<String>,
        evidence_ids: Vec<String>,
    ) -> String {
        let summary = clean_line(summary.into());
        let experiment_key = experiment_id.clone().unwrap_or_default();
        let evidence_key = evidence_ids.join(",");
        let verdict_key = format!("{verdict:?}");
        let id = stable_id(
            "result",
            [
                self.run_id.as_str(),
                experiment_key.as_str(),
                verdict_key.as_str(),
                summary.as_str(),
                evidence_key.as_str(),
            ],
        );
        if !self.results.iter().any(|item| item.id == id) {
            let now = now_millis();
            self.results.push(ResearchResult {
                id: id.clone(),
                experiment_id,
                verdict,
                summary: summary.clone(),
                evidence_ids: unique(evidence_ids),
                created_at_ms: now,
            });
            self.record_event("result", &id, &summary, now);
        }
        self.updated_at_ms = now_millis();
        id
    }

    pub fn absorb_agent_output(
        &mut self,
        node_id: &str,
        target: &str,
        agent_id: &str,
        output: &str,
    ) -> AbsorbReport {
        let mut report = AbsorbReport::default();
        let Some(json) = extract_json_value(output) else {
            let id = self.record_observation(
                target,
                agent_id,
                format!("unstructured output from {node_id}"),
                output.chars().take(1000).collect::<String>(),
                EvidenceConfidence::Low,
            );
            report.observation_ids.push(id);
            return report;
        };
        let json = unwrap_output_value(&json);
        let source = format!("{agent_id}:{node_id}");
        let mut evidence_ids = Vec::new();

        if let Some(surfaces) = json.get("surfaces").and_then(Value::as_array) {
            for surface in surfaces {
                if let Some((summary, detail)) = surface_observation(surface) {
                    let id = self.record_observation(
                        target,
                        &source,
                        summary,
                        detail,
                        EvidenceConfidence::Medium,
                    );
                    report.observation_ids.push(id.clone());
                    evidence_ids.push(id);
                }
            }
        }

        if let Some(evidence) = json.get("evidence").and_then(Value::as_array) {
            for item in evidence {
                if let Some((summary, detail)) = evidence_observation(item) {
                    let id = self.record_observation(
                        target,
                        &source,
                        summary,
                        detail,
                        EvidenceConfidence::High,
                    );
                    report.observation_ids.push(id.clone());
                    evidence_ids.push(id);
                }
            }
        }

        if let Some(findings) = json.get("findings").and_then(Value::as_array) {
            for finding in findings {
                if let Some((summary, detail, confidence)) = finding_observation(finding) {
                    let id = self.record_observation(target, &source, summary, detail, confidence);
                    report.observation_ids.push(id.clone());
                    evidence_ids.push(id);
                }
            }
        }

        if let Some(hypotheses) = json.get("hypotheses").and_then(Value::as_array) {
            for hypothesis in hypotheses {
                if let Some((statement, hyp_target, rationale)) = hypothesis_candidate(hypothesis) {
                    let id = self.record_hypothesis(
                        statement,
                        hyp_target.or_else(|| Some(target.to_string())),
                        rationale,
                        evidence_ids.clone(),
                    );
                    report.hypothesis_ids.push(id);
                }
            }
        }

        let verdict_summary = json
            .get("verdict")
            .and_then(Value::as_str)
            .map(clean_line)
            .filter(|value| !value.is_empty());
        if let Some(summary) = verdict_summary {
            let verdict = verdict_from_text(&summary);
            let id = self.record_result(Some(node_id.to_string()), verdict, summary, evidence_ids);
            report.result_ids.push(id);
        }

        report
    }

    pub fn dossier(&self, max_items: usize) -> ResearchDossier {
        ResearchDossier {
            run_id: self.run_id.clone(),
            targets: self.targets.clone(),
            observations: newest(self.observations.clone(), max_items),
            hypotheses: newest(self.hypotheses.clone(), max_items),
            experiments: newest(self.experiments.clone(), max_items),
            results: newest(self.results.clone(), max_items),
        }
    }

    pub fn build_context_pack(
        &self,
        current_target: impl Into<String>,
        current_node_id: impl Into<String>,
        mission: impl Into<String>,
        max_items: usize,
    ) -> ContextPack {
        let current_target = current_target.into();
        let current_node_id = current_node_id.into();
        let mission = mission.into();
        let dossier = self.dossier(max_items);
        let rendered =
            render_dossier_with_header(&dossier, &current_target, &current_node_id, &mission);
        ContextPack {
            run_id: self.run_id.clone(),
            current_target,
            current_node_id,
            mission,
            dossier,
            rendered,
        }
    }

    pub fn render_dossier(&self, max_items: usize) -> String {
        render_dossier(&self.dossier(max_items))
    }

    fn record_event(&mut self, kind: &str, ref_id: &str, summary: &str, created_at_ms: u64) {
        let id = stable_id("event", [self.run_id.as_str(), kind, ref_id, summary]);
        if self.events.iter().any(|item| item.id == id) {
            return;
        }
        self.events.push(LedgerEvent {
            id,
            kind: kind.to_string(),
            ref_id: ref_id.to_string(),
            summary: summary.to_string(),
            created_at_ms,
        });
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct AbsorbReport {
    pub observation_ids: Vec<String>,
    pub hypothesis_ids: Vec<String>,
    pub result_ids: Vec<String>,
}

pub fn render_dossier(dossier: &ResearchDossier) -> String {
    render_dossier_with_header(dossier, "", "", "Continue the research loop.")
}

fn render_dossier_with_header(
    dossier: &ResearchDossier,
    current_target: &str,
    current_node_id: &str,
    mission: &str,
) -> String {
    let mut out = String::new();
    out.push_str("## LM Research Dossier\n");
    out.push_str("Read in order: observations, hypotheses, experiments, results, then decide the next smallest useful action.\n");
    out.push_str(&format!("Run: {}\n", dossier.run_id));
    if !dossier.targets.is_empty() {
        out.push_str(&format!("Targets: {}\n", dossier.targets.join(", ")));
    }
    if !current_target.is_empty() {
        out.push_str(&format!("Current target: {current_target}\n"));
    }
    if !current_node_id.is_empty() {
        out.push_str(&format!("Current node: {current_node_id}\n"));
    }
    out.push_str(&format!("Mission: {mission}\n"));

    out.push_str("\n### 1. Observations\n");
    if dossier.observations.is_empty() {
        out.push_str("- none yet\n");
    } else {
        for observation in &dossier.observations {
            out.push_str(&format!(
                "- [{}] target={} source={} confidence={:?}: {}\n",
                observation.id,
                observation.target,
                observation.source,
                observation.confidence,
                observation.summary
            ));
            if !observation.detail.is_empty() {
                out.push_str(&format!(
                    "  detail: {}\n",
                    truncate(&observation.detail, 260)
                ));
            }
        }
    }

    out.push_str("\n### 2. Hypotheses\n");
    if dossier.hypotheses.is_empty() {
        out.push_str("- none yet\n");
    } else {
        for hypothesis in &dossier.hypotheses {
            let target = hypothesis.target.as_deref().unwrap_or("unknown");
            out.push_str(&format!(
                "- [{}] status={:?} target={}: {}\n",
                hypothesis.id, hypothesis.status, target, hypothesis.statement
            ));
            if !hypothesis.rationale.is_empty() {
                out.push_str(&format!(
                    "  rationale: {}\n",
                    truncate(&hypothesis.rationale, 220)
                ));
            }
        }
    }

    out.push_str("\n### 3. Experiments\n");
    if dossier.experiments.is_empty() {
        out.push_str("- none yet\n");
    } else {
        for experiment in &dossier.experiments {
            let target = experiment.target.as_deref().unwrap_or("unknown");
            out.push_str(&format!(
                "- [{}] status={:?} target={}: {}\n",
                experiment.id, experiment.status, target, experiment.objective
            ));
            if !experiment.required_tools.is_empty() {
                out.push_str(&format!(
                    "  tools: {}\n",
                    experiment.required_tools.join(", ")
                ));
            }
        }
    }

    out.push_str("\n### 4. Results\n");
    if dossier.results.is_empty() {
        out.push_str("- none yet\n");
    } else {
        for result in &dossier.results {
            out.push_str(&format!(
                "- [{}] verdict={:?}: {}\n",
                result.id, result.verdict, result.summary
            ));
            if !result.evidence_ids.is_empty() {
                out.push_str(&format!("  evidence: {}\n", result.evidence_ids.join(", ")));
            }
        }
    }

    out.push_str("\n### 5. Next-action policy\n");
    out.push_str("- Reuse receipts before rediscovery.\n");
    out.push_str("- Generate new hypotheses only from observations above.\n");
    out.push_str("- Prefer the smallest executable experiment that changes belief.\n");
    out.push_str("- Mark false positives and blocked work explicitly so other workers stop repeating them.\n");
    out
}

fn surface_observation(value: &Value) -> Option<(String, String)> {
    if let Some(url) = value.as_str().map(str::trim).filter(|url| !url.is_empty()) {
        return Some((format!("surface observed {url}"), String::new()));
    }
    let url = find_str(value, &["url", "endpoint", "href", "path"])?;
    let method = find_str(value, &["method"]).unwrap_or("GET");
    let status = find_str(value, &["status", "status_code"]).unwrap_or("");
    let mut detail = format!("method={method}");
    if !status.is_empty() {
        detail.push_str(&format!(" status={status}"));
    }
    if let Some(title) = find_str(value, &["title", "summary", "description"]) {
        detail.push_str(&format!(" detail={title}"));
    }
    Some((format!("surface observed {url}"), detail))
}

fn evidence_observation(value: &Value) -> Option<(String, String)> {
    if let Some(detail) = value
        .as_str()
        .map(clean_detail)
        .filter(|item| !item.is_empty())
    {
        return Some((truncate(&detail, 120), detail));
    }
    let detail = find_str(value, &["detail", "evidence", "summary", "body", "output"])
        .map(clean_detail)
        .filter(|item| !item.is_empty())?;
    let kind = find_str(value, &["kind", "type"]).unwrap_or("evidence");
    Some((format!("{kind}: {}", truncate(&detail, 100)), detail))
}

fn finding_observation(value: &Value) -> Option<(String, String, EvidenceConfidence)> {
    if let Some(detail) = value
        .as_str()
        .map(clean_detail)
        .filter(|item| !item.is_empty())
    {
        return Some((
            format!("finding: {}", truncate(&detail, 100)),
            detail,
            EvidenceConfidence::Medium,
        ));
    }
    let title = find_str(value, &["title", "vulnerability", "name", "summary"])?;
    let severity = find_str(value, &["severity"]).unwrap_or("unknown");
    let endpoint = find_str(value, &["endpoint", "url", "endpoint_url"]).unwrap_or("unknown");
    let verdict = find_str(value, &["verdict", "status"]).unwrap_or("suspected");
    let confidence = if verdict.contains("confirmed") || verdict.contains("validated") {
        EvidenceConfidence::High
    } else {
        EvidenceConfidence::Medium
    };
    Some((
        format!("finding {title} severity={severity}"),
        format!("endpoint={endpoint} verdict={verdict}"),
        confidence,
    ))
}

fn hypothesis_candidate(value: &Value) -> Option<(String, Option<String>, String)> {
    if let Some(statement) = value
        .as_str()
        .map(clean_line)
        .filter(|item| !item.is_empty())
    {
        return Some((statement, None, String::new()));
    }
    let statement = find_str(value, &["statement", "hypothesis", "title", "idea"])
        .map(clean_line)
        .filter(|item| !item.is_empty())?;
    let target = find_str(value, &["target", "url", "endpoint"]).map(str::to_string);
    let rationale = find_str(value, &["reasoning", "rationale", "why", "evidence"])
        .map(clean_detail)
        .unwrap_or_default();
    Some((statement, target, rationale))
}

fn verdict_from_text(value: &str) -> ResultVerdict {
    let lower = value.to_ascii_lowercase();
    if lower.contains("needs_more_budget")
        || lower.contains("blocked")
        || lower.contains("unreachable")
        || lower.contains("timeout")
    {
        ResultVerdict::Blocked
    } else if lower.contains("false_positive")
        || lower.contains("false positive")
        || lower.contains("refuted")
        || lower.contains("not vulnerable")
        || lower.contains("no vulnerabilities")
        || lower.contains("no vulnerability")
        || lower.contains("no confirmed vulnerability")
        || lower.contains("no exploitable")
        || lower.contains("no attack surface")
        || lower.contains("no concrete")
        || lower.contains("zero vulnerability")
        || lower.contains("zero attack surface")
    {
        ResultVerdict::Refutes
    } else if lower.contains("confirmed")
        || lower.contains("validated")
        || lower.contains("supports")
        || lower.contains("exploitable")
    {
        ResultVerdict::Supports
    } else if lower.contains("suspected") || lower.contains("new") {
        ResultVerdict::Expands
    } else {
        ResultVerdict::Inconclusive
    }
}

fn extract_json_value(output: &str) -> Option<Value> {
    serde_json::from_str::<Value>(output).ok().or_else(|| {
        let start = output.find('{')?;
        let end = output.rfind('}')?;
        if end <= start {
            return None;
        }
        serde_json::from_str::<Value>(&output[start..=end]).ok()
    })
}

fn unwrap_output_value(value: &Value) -> &Value {
    value
        .get("output")
        .filter(|inner| inner.is_object())
        .unwrap_or(value)
}

fn find_str<'a>(value: &'a Value, keys: &[&str]) -> Option<&'a str> {
    for key in keys {
        if let Some(found) = value.get(*key).and_then(Value::as_str) {
            let trimmed = found.trim();
            if !trimmed.is_empty() {
                return Some(trimmed);
            }
        }
    }
    None
}

fn clean_line(value: impl AsRef<str>) -> String {
    value
        .as_ref()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn clean_detail(value: impl AsRef<str>) -> String {
    value
        .as_ref()
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .collect::<Vec<_>>()
        .join(" | ")
}

fn truncate(value: &str, limit: usize) -> String {
    let mut out = String::new();
    for ch in value.chars().take(limit) {
        out.push(ch);
    }
    if value.chars().count() > limit {
        out.push_str("...");
    }
    out
}

fn newest<T>(mut values: Vec<T>, max_items: usize) -> Vec<T> {
    if values.len() > max_items {
        values.drain(0..(values.len() - max_items));
    }
    values
}

fn unique(values: Vec<String>) -> Vec<String> {
    let mut seen = BTreeSet::new();
    values
        .into_iter()
        .filter(|value| seen.insert(value.clone()))
        .collect()
}

fn now_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64
}

// ---------------------------------------------------------------------------
// Semantic similarity (optional, behind the `similarity` feature).
// ---------------------------------------------------------------------------

#[cfg(feature = "similarity")]
mod similarity {
    use anyhow::Result;

    use crate::embedding::Embedder;

    /// Quantized sketches kept 1:1 with the ledger's observations and
    /// hypotheses, backed by clark-hash's stateless sparse-JL codec.
    ///
    /// The host embeds text (via [`Embedder`]); the crate encodes the embedding into
    /// a compact [`clark_hash::QuantizedVector`] sketch. [`SemanticSketches::find_similar`]
    /// then retrieves the most relevant past items for a query embedding —
    /// turning the ledger into a semantic index so reflection reuses receipts
    /// before rediscovering them (the dossier's stated policy, now enforced).
    #[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
    pub struct SemanticSketches {
        /// Input embedding dimension; the codec is reconstructed from this
        /// (clark-hash's codec is stateless, so it need not be persisted).
        input_dim: usize,
        #[serde(default)]
        observation_sketches: Vec<clark_hash::QuantizedVector>,
        #[serde(default)]
        hypothesis_sketches: Vec<clark_hash::QuantizedVector>,
    }

    impl SemanticSketches {
        /// Create an empty sketch store for `input_dim`-dimensional embeddings.
        pub fn new(input_dim: usize) -> Self {
            Self {
                input_dim,
                observation_sketches: Vec::new(),
                hypothesis_sketches: Vec::new(),
            }
        }

        pub fn observation_count(&self) -> usize {
            self.observation_sketches.len()
        }

        /// Encode and store a sketch for an observation's text.
        pub fn add_observation_text(&mut self, text: &str, embedder: &dyn Embedder) -> Result<()> {
            let embedding = embedder.embed_text(text)?;
            self.add_observation_embedding(&embedding)
        }

        /// Encode and store a sketch for a raw observation embedding.
        pub fn add_observation_embedding(&mut self, embedding: &[f32]) -> Result<()> {
            let codec = self.codec()?;
            self.observation_sketches.push(codec.encode(embedding)?);
            Ok(())
        }

        /// Encode and store a sketch for a hypothesis embedding.
        pub fn add_hypothesis_embedding(&mut self, embedding: &[f32]) -> Result<()> {
            let codec = self.codec()?;
            self.hypothesis_sketches.push(codec.encode(embedding)?);
            Ok(())
        }

        /// Find the `k` most similar observations to `query`, returning
        /// `(observation_index, similarity_score)` pairs sorted by descending
        /// similarity. Uses clark-hash's `FlatIndex::search_prepared`.
        pub fn find_similar(&self, query: &[f32], k: usize) -> Result<Vec<(usize, f64)>> {
            if self.observation_sketches.is_empty() {
                return Ok(Vec::new());
            }
            let codec = self.codec()?;
            let mut index = clark_hash::FlatIndex::new(codec);
            for sketch in &self.observation_sketches {
                index.add_encoded(sketch.clone())?;
            }
            let prepared = index.codec().sketch_query(query)?;
            let hits = index.search_prepared(&prepared, k.max(1))?;
            Ok(hits
                .into_iter()
                .map(|hit| (hit.index, hit.score as f64))
                .collect())
        }

        /// The highest similarity between `query` and any stored observation.
        /// Opportunity novelty is `1.0 - max_similarity`.
        pub fn max_similarity(&self, query: &[f32]) -> Result<f64> {
            Ok(self
                .find_similar(query, 1)?
                .into_iter()
                .map(|(_, score)| score)
                .next()
                .unwrap_or(0.0))
        }

        fn codec(&self) -> Result<clark_hash::ClarkHash> {
            Ok(clark_hash::ClarkHash::new(clark_hash::SQuaJLConfig::new(
                self.input_dim,
            ))?)
        }
    }

    /// Re-rank a dossier's observations and hypotheses by semantic similarity
    /// to `query`, returning the top `max_items` of each (instead of newest-N).
    /// This is the `similarity`-feature counterpart to
    /// [`ResearchLedger::dossier`], enforcing the "reuse receipts before
    /// rediscovery" policy with actual retrieval.
    pub fn relevance_ranked_dossier(
        ledger: &super::ResearchLedger,
        sketches: &SemanticSketches,
        query: &[f32],
        max_items: usize,
    ) -> Result<super::ResearchDossier> {
        let obs = rank_observations(&ledger.observations, sketches, query, max_items)?;
        let hyps = rank_hypotheses(&ledger.hypotheses, sketches, query, max_items)?;
        Ok(super::ResearchDossier {
            run_id: ledger.run_id.clone(),
            targets: ledger.targets.clone(),
            observations: obs,
            hypotheses: hyps,
            experiments: super::newest(ledger.experiments.clone(), max_items),
            results: super::newest(ledger.results.clone(), max_items),
        })
    }

    fn rank_observations(
        items: &[super::Observation],
        sketches: &SemanticSketches,
        query: &[f32],
        max_items: usize,
    ) -> Result<Vec<super::Observation>> {
        if items.is_empty() || sketches.observation_count() < items.len() {
            return Ok(super::newest(items.to_vec(), max_items));
        }
        let k = max_items.min(items.len());
        let hits = sketches.find_similar(query, k)?;
        let ranked: Vec<super::Observation> = hits
            .into_iter()
            .map(|(idx, _)| items[idx].clone())
            .collect();
        Ok(if ranked.is_empty() {
            super::newest(items.to_vec(), max_items)
        } else {
            ranked
        })
    }

    fn rank_hypotheses(
        items: &[super::HypothesisCandidate],
        sketches: &SemanticSketches,
        query: &[f32],
        max_items: usize,
    ) -> Result<Vec<super::HypothesisCandidate>> {
        if items.is_empty() || sketches.hypothesis_sketches.len() < items.len() {
            return Ok(super::newest(items.to_vec(), max_items));
        }
        // Score each hypothesis sketch against the query and keep the top-k.
        let codec = sketches_codec(sketches)?;
        let prepared = codec.sketch_query(query)?;
        let mut scored: Vec<(usize, f32)> = Vec::with_capacity(items.len());
        for (idx, sketch) in sketches.hypothesis_sketches.iter().enumerate() {
            let score = codec.score(&prepared, sketch)?;
            scored.push((idx, score));
        }
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let ranked: Vec<super::HypothesisCandidate> = scored
            .into_iter()
            .take(max_items)
            .map(|(idx, _)| items[idx].clone())
            .collect();
        Ok(ranked)
    }

    fn sketches_codec(sketches: &SemanticSketches) -> Result<clark_hash::ClarkHash> {
        Ok(clark_hash::ClarkHash::new(clark_hash::SQuaJLConfig::new(
            sketches.input_dim,
        ))?)
    }
}

#[cfg(feature = "similarity")]
pub use similarity::{SemanticSketches, relevance_ranked_dossier};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn absorb_agent_output_records_observations_hypotheses_and_result() {
        let mut ledger = ResearchLedger::new("run", vec!["https://example.com".into()]);
        let output = serde_json::json!({
            "surfaces": [{"url": "https://example.com/api", "method": "GET", "status": "200"}],
            "evidence": ["GET /api returned JSON with csrf token"],
            "hypotheses": [{"statement": "csrf token may be reusable", "target": "https://example.com/api", "reasoning": "token observed in static response"}],
            "verdict": "suspected: needs CodeAct validation"
        })
        .to_string();

        let report = ledger.absorb_agent_output("node:1", "https://example.com", "agent", &output);

        assert_eq!(report.observation_ids.len(), 2);
        assert_eq!(report.hypothesis_ids.len(), 1);
        assert_eq!(report.result_ids.len(), 1);
        assert_eq!(ledger.results[0].verdict, ResultVerdict::Expands);
    }

    #[test]
    fn render_dossier_keeps_autoregressive_order() {
        let mut ledger = ResearchLedger::new("run", vec!["https://example.com".into()]);
        ledger.record_observation(
            "https://example.com",
            "agent",
            "surface observed https://example.com/login",
            "",
            EvidenceConfidence::Medium,
        );
        ledger.record_hypothesis(
            "login csrf token is missing",
            Some("https://example.com/login".into()),
            "form has no hidden token",
            Vec::new(),
        );
        ledger.record_result(
            Some("node:1".into()),
            ResultVerdict::Inconclusive,
            "manual confirmation still needed",
            Vec::new(),
        );

        let rendered = ledger.render_dossier(10);
        assert!(
            rendered.find("### 1. Observations").unwrap()
                < rendered.find("### 2. Hypotheses").unwrap()
        );
        assert!(
            rendered.find("### 2. Hypotheses").unwrap() < rendered.find("### 4. Results").unwrap()
        );
    }

    #[test]
    fn verdict_negation_beats_exploitable_keyword() {
        assert_eq!(
            verdict_from_text("No exploitable surface — static build artifact"),
            ResultVerdict::Refutes
        );
        assert_eq!(
            verdict_from_text("No vulnerabilities found — zero attack surface"),
            ResultVerdict::Refutes
        );
        assert_eq!(
            verdict_from_text(
                "deterministic fallback endpoint probe completed; no confirmed vulnerability from bounded GET"
            ),
            ResultVerdict::Refutes
        );
        assert_eq!(
            verdict_from_text("No attack surface. False positive."),
            ResultVerdict::Refutes
        );
    }

    // Semantic-similarity tests: only compiled with the `similarity` feature.
    #[cfg(feature = "similarity")]
    mod similarity_tests {
        use super::*;
        use crate::embedding::Embedder;

        /// A deterministic, model-free embedder: a bag-of-characters hashed into
        /// a fixed dimension. Texts that share characters produce similar
        /// (high-cosine) vectors; disjoint texts do not. Sufficient to exercise
        /// clark-hash's sketch + search without a real model.
        struct CharBagEmbedder {
            dim: usize,
        }
        impl Embedder for CharBagEmbedder {
            fn embed_text(&self, text: &str) -> anyhow::Result<Vec<f32>> {
                let mut v = vec![0.0_f32; self.dim];
                for ch in text.chars() {
                    let bucket = (ch as usize) % self.dim;
                    v[bucket] += 1.0;
                }
                // L2-normalize so cosine scoring is meaningful.
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

        #[test]
        fn find_similar_returns_semantically_close_observation() {
            // Reproduction for Phase 5: encode two similar + one dissimilar
            // observation, then assert find_similar returns a similar one (not
            // the dissimilar one). This fails without clark-hash (the module
            // is absent without the `similarity` feature).
            let embedder = CharBagEmbedder { dim: 32 };
            let mut sketches = SemanticSketches::new(32);
            sketches
                .add_observation_text("hello world", &embedder)
                .unwrap();
            sketches
                .add_observation_text("hello there", &embedder)
                .unwrap();
            sketches.add_observation_text("zzz qqq", &embedder).unwrap();

            let query = embedder.embed_text("hello friend").unwrap();
            let hits = sketches.find_similar(&query, 1).unwrap();
            assert_eq!(hits.len(), 1);
            // The closest must be one of the two "hello" observations (idx 0 or
            // 1), never the dissimilar "zzz qqq" (idx 2).
            assert!(
                hits[0].0 < 2,
                "expected a similar observation, got idx {}",
                hits[0].0
            );
        }

        #[test]
        fn novelty_is_one_minus_max_similarity() {
            let embedder = CharBagEmbedder { dim: 32 };
            let mut sketches = SemanticSketches::new(32);
            sketches
                .add_observation_text("hello world", &embedder)
                .unwrap();
            let query = embedder.embed_text("hello world").unwrap();
            let max_sim = sketches.max_similarity(&query).unwrap();
            let novelty = 1.0 - max_sim;
            // An exact-ish match (same text) has high similarity → low novelty.
            assert!(
                novelty < 0.5,
                "novelty for a near-duplicate should be low, got {novelty}"
            );
        }
    }
}
