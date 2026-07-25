use clark_autoresearch::{
    ExperimentGraph, FrontierStrategy, Hypothesis, Metric, TaskScore, TrialOutcome, rank_frontier,
};

fn main() -> anyhow::Result<()> {
    let metric = Metric::maximize("accuracy");
    let mut graph = ExperimentGraph::new("demo");

    let baseline = graph.allocate_child("root", Hypothesis::new("baseline prompt"))?;
    graph.record_outcome(
        &baseline,
        TrialOutcome::passed(0.72, "baseline eval").with_task_scores(vec![
            TaskScore::new("math", 0.70),
            TaskScore::new("code", 0.74),
        ]),
    )?;
    graph.commit(&baseline, "baseline-commit")?;

    let candidate = graph.allocate_child(
        &baseline,
        Hypothesis::new("shorter prompt with explicit checks"),
    )?;
    graph.record_outcome(
        &candidate,
        TrialOutcome::passed(0.81, "improved validation pass").with_task_scores(vec![
            TaskScore::new("math", 0.82),
            TaskScore::new("code", 0.80),
        ]),
    )?;
    graph.commit(&candidate, "candidate-commit")?;

    for item in rank_frontier(&graph, &metric, &FrontierStrategy::TopK { k: 3 }) {
        println!(
            "#{:<2} {:<10} score={:?} mode={:?} reason={}",
            item.rank, item.id, item.score, item.mode, item.reason
        );
    }

    Ok(())
}
