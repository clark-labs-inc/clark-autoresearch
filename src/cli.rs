use std::collections::BTreeMap;
use std::fs;
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command as ProcessCommand, Stdio};

use anyhow::{Context, Result, anyhow, bail};
use clap::{Parser, Subcommand, ValueEnum};
use clark_autoresearch::proposer::Proposer;
use clark_autoresearch::{
    AcceptanceCriterion, Candidate, EvaluationBatch, EvaluationCache, ExperimentGraph,
    ExperimentStatus, FrontierStrategy, FrontierType, Gate, Hypothesis, Metric, MetricDirection,
    OptimizationState, OutcomeStatus, ReflectiveDataset, ResearchAdapter, ResearchBias,
    ResearchLedger, ResearchMode, ResearchOpportunity, ResearchPolicy, StopCondition, TaskScore,
    TrialOutcome, enforce_acceptance, optimize, rank_frontier, rank_opportunities,
};
use serde::{Deserialize, Serialize};

const DEFAULT_STATE_PATH: &str = ".autoresearch/state.json";

#[derive(Debug, Parser)]
#[command(
    name = "clark-autoresearch",
    version,
    about = "Track experiment graphs and rank the next research frontier.",
    after_help = "Examples:\n  clark-autoresearch init --metric accuracy --direction maximize\n  clark-autoresearch spawn \"try a smaller prompt\" --mode explore\n  clark-autoresearch record exp_0000 0.82 --status passed --summary \"accuracy improved\"\n  clark-autoresearch frontier --strategy top-k --k 3\n  clark-autoresearch status"
)]
struct Cli {
    #[arg(long, global = true, default_value = DEFAULT_STATE_PATH)]
    state: PathBuf,

    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Create a new project state file.
    Init {
        #[arg(long, default_value = "run")]
        run_id: String,
        #[arg(long, default_value = "score")]
        metric: String,
        #[arg(long, value_enum, default_value_t = DirectionArg::Maximize)]
        direction: DirectionArg,
        #[arg(long, default_value_t = 3)]
        max_attempts: u32,
        #[arg(long = "gate")]
        gates: Vec<String>,
        #[arg(long)]
        force: bool,
    },

    /// Add a child hypothesis to the experiment graph.
    Spawn {
        hypothesis: String,
        #[arg(long, default_value = "root")]
        parent: String,
        #[arg(long, value_enum, default_value_t = ModeArg::Explore)]
        mode: ModeArg,
        #[arg(long)]
        target: Option<String>,
        #[arg(long, default_value = "")]
        rationale: String,
    },

    /// Record an evaluation outcome for an experiment.
    Record {
        id: String,
        score: f64,
        #[arg(long, value_enum, default_value_t = StatusArg::Passed)]
        status: StatusArg,
        #[arg(long, default_value = "")]
        summary: String,
        #[arg(long = "task-score")]
        task_scores: Vec<String>,
        #[arg(long = "meta")]
        metadata: Vec<String>,
    },

    /// Mark an evaluated experiment as committed.
    Commit {
        id: String,
        #[arg(long)]
        commit: Option<String>,
    },

    /// Mark an experiment as discarded.
    Discard { id: String, reason: String },

    /// Print the ranked frontier.
    Frontier {
        #[arg(long, value_enum, default_value_t = StrategyArg::TopK)]
        strategy: StrategyArg,
        #[arg(long, default_value_t = 5)]
        k: usize,
        #[arg(long, default_value_t = 0.10)]
        epsilon: f64,
        #[arg(long, default_value_t = 1.0)]
        temperature: f64,
        #[arg(long, default_value_t = 42)]
        seed: u64,
        #[arg(long, default_value_t = 0.0)]
        task_floor: f64,
        #[arg(long, value_enum, default_value_t = FrontierTypeArg::Instance)]
        frontier_type: FrontierTypeArg,
        /// Enforce an acceptance gate on the ranked frontier: reject candidates
        /// that do not improve their parent, and report why. "none" (default)
        /// leaves the ranking unfiltered.
        #[arg(long, value_enum, default_value_t = AcceptanceArg::None)]
        acceptance: AcceptanceArg,
        #[arg(long)]
        json: bool,
    },

    /// Print a compact project summary.
    Status {
        #[arg(long)]
        json: bool,
    },

    /// Print the full state JSON.
    Export,

    /// Rank generic research opportunities from JSON.
    OpportunityRank {
        input: PathBuf,
        #[arg(long)]
        json: bool,
    },

    /// Create a reusable LM research ledger.
    LedgerInit {
        #[arg(long, default_value = ".autoresearch/ledger.json")]
        ledger: PathBuf,
        #[arg(long, default_value = "run")]
        run_id: String,
        #[arg(long = "target")]
        targets: Vec<String>,
        #[arg(long)]
        force: bool,
    },

    /// Absorb a worker/agent JSON output into a ledger.
    LedgerAbsorb {
        #[arg(long, default_value = ".autoresearch/ledger.json")]
        ledger: PathBuf,
        #[arg(long)]
        node_id: String,
        #[arg(long)]
        target: String,
        #[arg(long)]
        agent_id: String,
        #[arg(long)]
        output: Option<String>,
        #[arg(long)]
        output_file: Option<PathBuf>,
    },

    /// Render the compact autoregressive dossier from a ledger.
    LedgerDossier {
        #[arg(long, default_value = ".autoresearch/ledger.json")]
        ledger: PathBuf,
        #[arg(long, default_value_t = 12)]
        max_items: usize,
        #[arg(long)]
        json: bool,
    },

    /// Run an execution-agnostic optimization loop.
    ///
    /// Evaluation is delegated to a host-run eval server via HTTP POST
    /// (`--eval-url`), and proposals are produced by a host command
    /// (`--proposer-cmd`) that reads the parent + reflective dataset on stdin
    /// and returns a candidate JSON on stdout. clark owns the loop body
    /// (propose → minibatch-eval → accept → full-eval → Pareto-update); the
    /// provider, sandbox, and evaluator stay in the host — à la GEPA's
    /// `optimize_anything` eval server, but execution-agnostic.
    Optimize {
        /// Seed candidate as JSON, e.g. '{"prompt":"You are a helpful assistant"}'.
        #[arg(long)]
        seed: String,
        /// Eval server URL, e.g. http://127.0.0.1:8081/evaluate.
        #[arg(long)]
        eval_url: String,
        /// Shell command that proposes a new candidate. It receives a JSON
        /// object {parent, reflective_dataset, components, history} on stdin
        /// and must print a JSON candidate on stdout.
        #[arg(long)]
        proposer_cmd: String,
        /// Comma-separated training examples (JSON), or @path to read a file.
        #[arg(long)]
        trainset: String,
        /// Comma-separated validation examples (JSON), or @path to read a file.
        #[arg(long)]
        valset: String,
        #[arg(long, default_value_t = 3)]
        minibatch_size: usize,
        #[arg(long, default_value_t = 20)]
        max_metric_calls: u32,
        /// Hard cap on loop iterations. Always terminates the loop even if the
        /// proposer converges to a fixed candidate (cached evals cost zero
        /// metric calls, so MaxMetricCalls alone cannot bound that case).
        #[arg(long, default_value_t = 50)]
        max_iterations: u32,
        #[arg(long, value_enum, default_value_t = AcceptanceArg::Strict)]
        acceptance: AcceptanceArg,
        #[arg(long, default_value_t = 0.0)]
        explore_weight: f64,
        #[arg(long, default_value_t = 1.0)]
        exploit_weight: f64,
        #[arg(long, default_value_t = 0.0)]
        validation_weight: f64,
        #[arg(long, default_value_t = 0)]
        seed_value: u64,
    },
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum DirectionArg {
    Maximize,
    Minimize,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum ModeArg {
    Explore,
    Exploit,
    Validate,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum StatusArg {
    Passed,
    Failed,
    Inconclusive,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum StrategyArg {
    ArgMax,
    TopK,
    EpsilonGreedy,
    Softmax,
    ParetoPerTask,
    Pareto,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum FrontierTypeArg {
    Instance,
    Objective,
    Hybrid,
    Cartesian,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum AcceptanceArg {
    Strict,
    OrEqual,
    None,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct ProjectState {
    policy: ResearchPolicy,
    graph: ExperimentGraph,
}

#[derive(Clone, Debug, Deserialize)]
struct OpportunityRankInput {
    opportunities: Vec<ResearchOpportunity>,
    #[serde(default)]
    bias: Option<ResearchBias>,
}

pub fn run() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Command::Init {
            run_id,
            metric,
            direction,
            max_attempts,
            gates,
            force,
        } => {
            if cli.state.exists() && !force {
                bail!(
                    "{} already exists. Pass --force to replace it.",
                    cli.state.display()
                );
            }
            let mut policy = ResearchPolicy::new(Metric {
                name: metric,
                direction: direction.into(),
            });
            policy.max_attempts = max_attempts;
            for gate in gates {
                policy.gates.push(parse_gate(&gate)?);
            }
            let state = ProjectState {
                policy,
                graph: ExperimentGraph::new(run_id),
            };
            save_state(&cli.state, &state)?;
            println!("Initialized {}", cli.state.display());
        }
        Command::Spawn {
            hypothesis,
            parent,
            mode,
            target,
            rationale,
        } => {
            let mut state = load_state(&cli.state)?;
            let mut hypothesis = Hypothesis::new(hypothesis).with_mode(mode.into());
            hypothesis.target = target;
            hypothesis.rationale = rationale;
            let id = state.graph.allocate_child(&parent, hypothesis)?;
            state.graph.set_status(&id, ExperimentStatus::Active)?;
            save_state(&cli.state, &state)?;
            println!("Spawned {id}");
        }
        Command::Record {
            id,
            score,
            status,
            summary,
            task_scores,
            metadata,
        } => {
            let mut state = load_state(&cli.state)?;
            let mut outcome = TrialOutcome {
                score,
                status: status.into(),
                summary,
                task_scores: task_scores
                    .iter()
                    .map(|value| parse_task_score(value))
                    .collect::<Result<Vec<_>>>()?,
                metadata: metadata
                    .iter()
                    .map(|value| parse_metadata(value))
                    .collect::<Result<BTreeMap<_, _>>>()?,
                objective_scores: BTreeMap::new(),
                val_subscores: BTreeMap::new(),
                objective_subscores: BTreeMap::new(),
            };
            if outcome.summary.is_empty() {
                outcome.summary = format!("{:?} with score {score}", outcome.status);
            }
            state.graph.record_outcome(&id, outcome)?;
            save_state(&cli.state, &state)?;
            println!("Recorded outcome for {id}");
        }
        Command::Commit { id, commit } => {
            let mut state = load_state(&cli.state)?;
            state
                .graph
                .commit(&id, commit.unwrap_or_else(|| "manual".to_string()))?;
            save_state(&cli.state, &state)?;
            println!("Committed {id}");
        }
        Command::Discard { id, reason } => {
            let mut state = load_state(&cli.state)?;
            state.graph.discard(&id, reason)?;
            save_state(&cli.state, &state)?;
            println!("Discarded {id}");
        }
        Command::Frontier {
            strategy,
            k,
            epsilon,
            temperature,
            seed,
            task_floor,
            frontier_type,
            acceptance,
            json,
        } => {
            let state = load_state(&cli.state)?;
            let strategy =
                strategy.into_strategy(k, epsilon, temperature, seed, task_floor, frontier_type);
            let mut ranked = rank_frontier(&state.graph, &state.policy.metric, &strategy);
            if let Some(criterion) = acceptance.into_criterion() {
                enforce_acceptance(&state.graph, &state.policy.metric, criterion, &mut ranked);
            }
            if json {
                print_json(&ranked)?;
            } else if ranked.is_empty() {
                println!("No frontier candidates.");
            } else {
                for candidate in ranked {
                    if let Some(reason) = &candidate.reject_reason {
                        println!(
                            "#{:<2} {:<10} score={} mode={:?} reason={} REJECTED: {}",
                            candidate.rank,
                            candidate.id,
                            fmt_score(candidate.score),
                            candidate.mode,
                            candidate.reason,
                            reason
                        );
                    } else {
                        println!(
                            "#{:<2} {:<10} score={} mode={:?} reason={}",
                            candidate.rank,
                            candidate.id,
                            fmt_score(candidate.score),
                            candidate.mode,
                            candidate.reason
                        );
                    }
                }
            }
        }
        Command::Status { json } => {
            let state = load_state(&cli.state)?;
            if json {
                print_json(&state)?;
            } else {
                print_status(&state);
            }
        }
        Command::Export => {
            let state = load_state(&cli.state)?;
            print_json(&state)?;
        }
        Command::OpportunityRank { input, json } => {
            let input = read_opportunity_input(&input)?;
            let ranked = rank_opportunities(
                &input.opportunities,
                input.bias.as_ref().unwrap_or(&ResearchBias::default()),
            );
            if json {
                print_json(&ranked)?;
            } else {
                for hint in ranked {
                    println!(
                        "{:<16} score={:.3} mode={:?} dispatch={:?} focus={}",
                        hint.node_id, hint.score, hint.mode, hint.dispatch_class, hint.focus
                    );
                }
            }
        }
        Command::LedgerInit {
            ledger,
            run_id,
            targets,
            force,
        } => {
            if ledger.exists() && !force {
                bail!(
                    "{} already exists. Pass --force to replace it.",
                    ledger.display()
                );
            }
            let ledger_state = ResearchLedger::new(run_id, targets);
            save_ledger(&ledger, &ledger_state)?;
            println!("Initialized {}", ledger.display());
        }
        Command::LedgerAbsorb {
            ledger,
            node_id,
            target,
            agent_id,
            output,
            output_file,
        } => {
            let mut ledger_state = load_ledger(&ledger)?;
            let output = read_agent_output(output, output_file.as_deref())?;
            let report = ledger_state.absorb_agent_output(&node_id, &target, &agent_id, &output);
            save_ledger(&ledger, &ledger_state)?;
            print_json(&report)?;
        }
        Command::LedgerDossier {
            ledger,
            max_items,
            json,
        } => {
            let ledger_state = load_ledger(&ledger)?;
            if json {
                print_json(&ledger_state.dossier(max_items))?;
            } else {
                println!("{}", ledger_state.render_dossier(max_items));
            }
        }
        Command::Optimize {
            seed,
            eval_url,
            proposer_cmd,
            trainset,
            valset,
            minibatch_size,
            max_metric_calls,
            max_iterations,
            acceptance,
            explore_weight,
            exploit_weight,
            validation_weight,
            seed_value,
        } => {
            let seed_candidate: Candidate =
                serde_json::from_str(&seed).context("parse --seed JSON")?;
            let trainset = read_json_examples(&trainset)?;
            let valset = read_json_examples(&valset)?;
            if valset.is_empty() {
                bail!("--valset must contain at least one example");
            }
            let metric = Metric::maximize("score");
            let mut state = OptimizationState::new(metric, seed_candidate)
                .with_acceptance(
                    acceptance
                        .into_criterion()
                        .unwrap_or(AcceptanceCriterion::StrictImprovement),
                )
                .with_bias(ResearchBias {
                    explore_weight,
                    exploit_weight,
                    validation_weight,
                    require_in_scope: false,
                });
            let adapter = HttpEvalAdapter::new(&eval_url)?;
            let proposer = CommandProposer {
                command: proposer_cmd,
            };
            let stop = StopCondition::Composite {
                conditions: vec![
                    StopCondition::MaxMetricCalls {
                        max: max_metric_calls,
                    },
                    StopCondition::MaxIterations {
                        max: max_iterations,
                    },
                ],
            };
            let mut cache = EvaluationCache::new();
            optimize(
                &adapter,
                &proposer,
                &mut state,
                &stop,
                &mut cache,
                None,
                &trainset,
                &valset,
                minibatch_size,
                seed_value,
            )?;
            println!("best score: {:?}", state.best_score);
            println!("iterations: {}", state.snapshot.iteration);
            println!("metric calls: {}", state.snapshot.metric_calls);
            println!("rejections: {}", state.rejections.len());
            if let Some(best) = state.graph.best_committed(&state.metric)
                && let Some(candidate) = state.candidates.get(&best.id)
            {
                println!("best candidate: {}", serde_json::to_string(candidate)?);
            }
        }
    }

    Ok(())
}

fn load_state(path: &Path) -> Result<ProjectState> {
    let contents =
        fs::read_to_string(path).with_context(|| format!("read state file {}", path.display()))?;
    serde_json::from_str(&contents).with_context(|| format!("parse state file {}", path.display()))
}

fn save_state(path: &Path, state: &ProjectState) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
    }
    let contents = serde_json::to_string_pretty(state)?;
    fs::write(path, format!("{contents}\n")).with_context(|| format!("write {}", path.display()))
}

fn load_ledger(path: &Path) -> Result<ResearchLedger> {
    let contents =
        fs::read_to_string(path).with_context(|| format!("read ledger file {}", path.display()))?;
    serde_json::from_str(&contents).with_context(|| format!("parse ledger file {}", path.display()))
}

fn save_ledger(path: &Path, ledger: &ResearchLedger) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
    }
    let contents = serde_json::to_string_pretty(ledger)?;
    fs::write(path, format!("{contents}\n")).with_context(|| format!("write {}", path.display()))
}

fn print_json(value: &impl Serialize) -> Result<()> {
    println!("{}", serde_json::to_string_pretty(value)?);
    Ok(())
}

fn print_status(state: &ProjectState) {
    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    for node in state.graph.nodes.values() {
        *counts.entry(format!("{:?}", node.status)).or_default() += 1;
    }

    println!("Run: {}", state.graph.run_id);
    println!(
        "Metric: {} ({:?})",
        state.policy.metric.name, state.policy.metric.direction
    );
    println!("Nodes: {}", state.graph.nodes.len());
    println!("Frontier: {}", state.graph.frontier_nodes().len());
    for (status, count) in counts {
        println!("  {status}: {count}");
    }
    if let Some(best) = state.graph.best_committed(&state.policy.metric) {
        println!(
            "Best committed: {} score={}",
            best.id,
            fmt_score(best.score)
        );
    }
}

fn read_opportunity_input(path: &Path) -> Result<OpportunityRankInput> {
    let contents = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    if let Ok(input) = serde_json::from_str::<OpportunityRankInput>(&contents) {
        return Ok(input);
    }
    let opportunities = serde_json::from_str::<Vec<ResearchOpportunity>>(&contents)
        .with_context(|| format!("parse opportunity JSON {}", path.display()))?;
    Ok(OpportunityRankInput {
        opportunities,
        bias: None,
    })
}

fn read_agent_output(output: Option<String>, output_file: Option<&Path>) -> Result<String> {
    match (output, output_file) {
        (Some(_), Some(_)) => bail!("pass only one of --output or --output-file"),
        (Some(output), None) => Ok(output),
        (None, Some(path)) => {
            fs::read_to_string(path).with_context(|| format!("read {}", path.display()))
        }
        (None, None) => {
            let mut buffer = String::new();
            io::stdin()
                .read_to_string(&mut buffer)
                .context("read agent output from stdin")?;
            if buffer.trim().is_empty() {
                bail!("no agent output provided; pass --output, --output-file, or stdin")
            }
            Ok(buffer)
        }
    }
}

fn parse_gate(value: &str) -> Result<Gate> {
    let (name, command) = value
        .split_once('=')
        .ok_or_else(|| anyhow!("gate must be name=command"))?;
    if name.trim().is_empty() || command.trim().is_empty() {
        bail!("gate must include a non-empty name and command");
    }
    Ok(Gate::new(name.trim(), command.trim()))
}

fn parse_task_score(value: &str) -> Result<TaskScore> {
    let (task_id, rest) = value
        .split_once('=')
        .ok_or_else(|| anyhow!("task score must be task_id=score[:maximize|minimize]"))?;
    let mut parts = rest.split(':');
    let score = parts
        .next()
        .ok_or_else(|| anyhow!("missing task score"))?
        .parse::<f64>()
        .with_context(|| format!("parse score in {value}"))?;
    let direction = match parts.next() {
        None => None,
        Some("maximize" | "max") => Some(MetricDirection::Maximize),
        Some("minimize" | "min") => Some(MetricDirection::Minimize),
        Some(other) => bail!("unknown task score direction: {other}"),
    };
    if parts.next().is_some() {
        bail!("task score must be task_id=score[:maximize|minimize]");
    }
    let mut score = TaskScore::new(task_id.trim(), score);
    score.direction = direction;
    Ok(score)
}

fn parse_metadata(value: &str) -> Result<(String, serde_json::Value)> {
    let (key, raw) = value
        .split_once('=')
        .ok_or_else(|| anyhow!("metadata must be key=value"))?;
    if key.trim().is_empty() {
        bail!("metadata key cannot be empty");
    }
    let parsed =
        serde_json::from_str(raw).unwrap_or_else(|_| serde_json::Value::String(raw.into()));
    Ok((key.trim().to_string(), parsed))
}

fn fmt_score(score: Option<f64>) -> String {
    score
        .map(|score| format!("{score:.4}"))
        .unwrap_or_else(|| "-".to_string())
}

impl From<DirectionArg> for MetricDirection {
    fn from(value: DirectionArg) -> Self {
        match value {
            DirectionArg::Maximize => Self::Maximize,
            DirectionArg::Minimize => Self::Minimize,
        }
    }
}

impl From<ModeArg> for ResearchMode {
    fn from(value: ModeArg) -> Self {
        match value {
            ModeArg::Explore => Self::Explore,
            ModeArg::Exploit => Self::Exploit,
            ModeArg::Validate => Self::Validate,
        }
    }
}

impl From<StatusArg> for OutcomeStatus {
    fn from(value: StatusArg) -> Self {
        match value {
            StatusArg::Passed => Self::Passed,
            StatusArg::Failed => Self::Failed,
            StatusArg::Inconclusive => Self::Inconclusive,
        }
    }
}

impl StrategyArg {
    fn into_strategy(
        self,
        k: usize,
        epsilon: f64,
        temperature: f64,
        seed: u64,
        task_floor: f64,
        frontier_type: FrontierTypeArg,
    ) -> FrontierStrategy {
        match self {
            StrategyArg::ArgMax => FrontierStrategy::ArgMax,
            StrategyArg::TopK => FrontierStrategy::TopK { k },
            StrategyArg::EpsilonGreedy => FrontierStrategy::EpsilonGreedy { epsilon, seed },
            StrategyArg::Softmax => FrontierStrategy::Softmax {
                temperature,
                k,
                seed,
            },
            StrategyArg::ParetoPerTask => FrontierStrategy::ParetoPerTask { k, task_floor },
            StrategyArg::Pareto => FrontierStrategy::Pareto {
                frontier_type: frontier_type.into(),
                objectives: Vec::new(),
            },
        }
    }
}

impl From<FrontierTypeArg> for FrontierType {
    fn from(value: FrontierTypeArg) -> Self {
        match value {
            FrontierTypeArg::Instance => Self::Instance,
            FrontierTypeArg::Objective => Self::Objective,
            FrontierTypeArg::Hybrid => Self::Hybrid,
            FrontierTypeArg::Cartesian => Self::Cartesian,
        }
    }
}

impl AcceptanceArg {
    fn into_criterion(self) -> Option<AcceptanceCriterion> {
        match self {
            AcceptanceArg::Strict => Some(AcceptanceCriterion::StrictImprovement),
            AcceptanceArg::OrEqual => Some(AcceptanceCriterion::ImprovementOrEqual),
            AcceptanceArg::None => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Execution-agnostic optimize subcommand: HTTP eval adapter + command proposer.
// Both are std-only so the CLI stays dependency-light. The eval server and the
// proposer command live in the host; clark owns only the loop body.
// ---------------------------------------------------------------------------

/// A thin HTTP eval adapter: POSTs `{candidate, batch, capture_traces}` to a
/// host-run eval server and parses the `{scores, outputs, ...}` response.
/// Mirrors GEPA's `optimize_anything` eval server contract. std-only
/// (`TcpStream`); intended for a local eval server (`http://127.0.0.1:PORT`).
struct HttpEvalAdapter {
    host: String,
    port: u16,
    path: String,
}

impl HttpEvalAdapter {
    fn new(url: &str) -> Result<Self> {
        let rest = url
            .strip_prefix("http://")
            .ok_or_else(|| anyhow!("--eval-url must start with http:// (TLS is not supported)"))?;
        let (authority, path) = rest.split_once('/').unwrap_or((rest, ""));
        let (host, port) = match authority.rsplit_once(':') {
            Some((h, p)) => (h.to_string(), p.parse().context("parse eval-url port")?),
            None => (authority.to_string(), 80),
        };
        Ok(Self {
            host,
            port,
            path: format!("/{}", path),
        })
    }

    fn post(&self, body: &str) -> Result<String> {
        use std::io::Read;
        use std::net::TcpStream;
        let mut stream = TcpStream::connect((self.host.as_str(), self.port))
            .with_context(|| format!("connect to eval server {}:{}", self.host, self.port))?;
        let request = format!(
            "POST {path} HTTP/1.1\r\nHost: {host}\r\nContent-Type: application/json\r\nContent-Length: {len}\r\nConnection: close\r\n\r\n{body}",
            path = self.path,
            host = self.host,
            len = body.len(),
            body = body,
        );
        stream
            .write_all(request.as_bytes())
            .context("write eval request")?;
        let mut response = String::new();
        stream
            .read_to_string(&mut response)
            .context("read eval response")?;
        let body_start = response
            .find("\r\n\r\n")
            .ok_or_else(|| anyhow!("malformed HTTP response (no header/body boundary)"))?
            + 4;
        Ok(response[body_start..].to_string())
    }
}

impl ResearchAdapter for HttpEvalAdapter {
    fn evaluate(
        &self,
        batch: &[serde_json::Value],
        candidate: &Candidate,
        capture_traces: bool,
    ) -> Result<EvaluationBatch> {
        let body = serde_json::json!({
            "candidate": candidate,
            "batch": batch,
            "capture_traces": capture_traces,
        })
        .to_string();
        let response = self.post(&body)?;
        let value: serde_json::Value =
            serde_json::from_str(&response).context("parse eval-server response as JSON")?;
        let scores = value
            .get("scores")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow!("eval-server response missing 'scores' array"))?
            .iter()
            .map(|v| v.as_f64().unwrap_or(0.0))
            .collect();
        let outputs = value
            .get("outputs")
            .and_then(|v| v.as_array())
            .map(|arr| arr.to_vec())
            .unwrap_or_else(|| batch.to_vec());
        let trajectories = value
            .get("trajectories")
            .and_then(|v| v.as_array())
            .map(|arr| arr.to_vec());
        let objective_scores = value
            .get("objective_scores")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .map(|m| {
                        m.as_object()
                            .map(|o| {
                                o.iter()
                                    .filter_map(|(k, v)| v.as_f64().map(|f| (k.clone(), f)))
                                    .collect()
                            })
                            .unwrap_or_default()
                    })
                    .collect()
            });
        let num_metric_calls = value
            .get("num_metric_calls")
            .and_then(|v| v.as_u64())
            .unwrap_or(batch.len() as u64) as u32;
        Ok(EvaluationBatch {
            scores,
            outputs,
            trajectories,
            objective_scores,
            num_metric_calls,
        })
    }
}

/// A proposer that shells out to a host command. The command receives a JSON
/// object `{parent, reflective_dataset, components, history}` on stdin and
/// must print a JSON candidate on stdout. This keeps the LM/provider in the
/// host while clark owns the proposal orchestration.
struct CommandProposer {
    command: String,
}

impl Proposer for CommandProposer {
    fn propose(
        &self,
        parent: &Candidate,
        reflective_dataset: &ReflectiveDataset,
        components: &[String],
        history: Option<&str>,
    ) -> Result<Candidate> {
        let request = serde_json::json!({
            "parent": parent,
            "reflective_dataset": reflective_dataset,
            "components": components,
            "history": history,
        })
        .to_string();
        let mut child = ProcessCommand::new("sh")
            .arg("-c")
            .arg(&self.command)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .with_context(|| format!("spawn proposer command: {}", self.command))?;
        {
            let stdin = child
                .stdin
                .as_mut()
                .ok_or_else(|| anyhow!("proposer stdin unavailable"))?;
            stdin
                .write_all(request.as_bytes())
                .context("write proposer stdin")?;
        }
        let output = child
            .wait_with_output()
            .context("wait for proposer command")?;
        if !output.status.success() {
            bail!(
                "proposer command failed ({}): {}",
                output.status,
                String::from_utf8_lossy(&output.stderr).trim()
            );
        }
        let stdout = String::from_utf8_lossy(&output.stdout);
        let candidate: Candidate = serde_json::from_str(stdout.trim()).with_context(|| {
            format!("parse proposer stdout as candidate JSON: {}", stdout.trim())
        })?;
        Ok(candidate)
    }
}

/// Read examples for the optimize loop: a comma-separated list of JSON values,
/// or `@path` to read a JSON array from a file.
fn read_json_examples(spec: &str) -> Result<Vec<serde_json::Value>> {
    if let Some(path) = spec.strip_prefix('@') {
        let contents = fs::read_to_string(path).with_context(|| format!("read {path}"))?;
        return serde_json::from_str(&contents)
            .with_context(|| format!("parse {path} as a JSON array of examples"));
    }
    // Wrap the comma-separated values in a JSON array so individual examples can
    // be scalars or objects.
    let wrapped = format!("[{spec}]");
    serde_json::from_str(&wrapped)
        .with_context(|| format!("parse --trainset/--valset as comma-separated JSON: {spec}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_task_score_with_direction() {
        let score = parse_task_score("task_a=0.42:minimize").unwrap();
        assert_eq!(score.task_id, "task_a");
        assert_eq!(score.score, 0.42);
        assert_eq!(score.direction, Some(MetricDirection::Minimize));
    }

    #[test]
    fn parses_gate() {
        let gate = parse_gate("test=cargo test").unwrap();
        assert_eq!(gate.name, "test");
        assert_eq!(gate.command, "cargo test");
    }
}
