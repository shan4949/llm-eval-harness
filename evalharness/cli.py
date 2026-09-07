import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="evalharness",
    help="LLM evaluation and regression-testing framework.",
    no_args_is_help=True,
)
console = Console()


@app.command()
def run(
    eval_set: str = typer.Option("golden_qa", "--eval-set", "-e", help="Eval set name"),
    baseline: bool = typer.Option(False, "--baseline", help="Mark this run as the new baseline"),
    no_judge: bool = typer.Option(False, "--no-judge", help="Skip LLM judge (faster, cheaper)"),
):
    """Run the eval suite against the SUT and store results to the database."""
    from evalharness.runner import run_eval

    console.print(f"[bold green]▶ Running eval set:[/] {eval_set}")
    if baseline:
        console.print("  [yellow]Marking as baseline[/]")

    run_id = run_eval(eval_set_name=eval_set, is_baseline=baseline, use_llm_judge=not no_judge)
    console.print(f"[bold green]✓ Done.[/] run_id=[bold]{run_id}[/]")
    console.print(f"  Run [cyan]evalharness report --run-id {run_id}[/] to see results.")


@app.command()
def compare(
    baseline_id: int = typer.Option(..., "--baseline", "-b", help="Baseline run ID"),
    candidate_id: int = typer.Option(..., "--candidate", "-c", help="Candidate run ID"),
):
    """Compare a candidate run against a baseline. Exits non-zero on regression."""
    from evalharness.regression import compare as do_compare

    report = do_compare(baseline_id, candidate_id)

    table = Table(title=f"Run {candidate_id} vs baseline {baseline_id}")
    table.add_column("Metric")
    table.add_column("Baseline", justify="right")
    table.add_column("Candidate", justify="right")
    table.add_column("Delta", justify="right")

    table.add_row(
        "Pass rate (keyword)",
        f"{report.baseline.pass_rate:.1%}",
        f"{report.candidate.pass_rate:.1%}",
        f"{report.pass_rate_delta:+.1%}",
    )
    table.add_row(
        "P95 latency (ms)",
        f"{report.baseline.latency_p95_ms:.0f}",
        f"{report.candidate.latency_p95_ms:.0f}",
        f"{report.latency_increase_pct:+.1%}",
    )
    table.add_row(
        "Cost/query (USD)",
        f"{report.baseline.cost_per_query_usd:.6f}",
        f"{report.candidate.cost_per_query_usd:.6f}",
        f"{report.cost_increase_pct:+.1%}",
    )
    table.add_row(
        "Retrieval sim median",
        f"{report.baseline.retrieval_sim_median:.4f}",
        f"{report.candidate.retrieval_sim_median:.4f}",
        "",
    )
    console.print(table)

    if report.is_regression:
        console.print("\n[bold red]✗ REGRESSION DETECTED[/]")
        for reason in report.reasons:
            console.print(f"  • {reason}")
        raise typer.Exit(1)
    else:
        console.print("\n[bold green]✓ No regression detected[/]")


@app.command()
def report(
    run_id: int = typer.Option(..., "--run-id", "-r", help="Run ID to report on"),
):
    """Print a per-test-case summary for a completed eval run."""
    from sqlalchemy import select

    from evalharness.storage.db import get_session
    from evalharness.storage.models import EvalResult, EvalScore

    with get_session() as session:
        results = (
            session.execute(
                select(EvalResult).where(EvalResult.run_id == run_id)
            )
            .scalars()
            .all()
        )

        if not results:
            console.print(f"[red]No results found for run {run_id}[/]")
            raise typer.Exit(1)

        table = Table(title=f"Eval run {run_id}", show_lines=True)
        table.add_column("Test case", style="cyan")
        table.add_column("Answer (truncated)")
        table.add_column("KW", justify="center")
        table.add_column("IDK", justify="center")
        table.add_column("Correctness", justify="center")
        table.add_column("ms", justify="right")

        for r in results:
            scores = {
                s.metric_name: s
                for s in session.execute(
                    select(EvalScore).where(EvalScore.result_id == r.result_id)
                ).scalars().all()
            }

            def fmt(metric: str) -> str:
                s = scores.get(metric)
                if s is None:
                    return "–"
                return "[green]✓[/]" if s.passed else "[red]✗[/]"

            table.add_row(
                r.test_case_id,
                (r.model_response or "")[:80] + ("…" if len(r.model_response or "") > 80 else ""),
                fmt("keyword_match"),
                fmt("idk_check"),
                fmt("correctness"),
                str(r.latency_ms or "–"),
            )

        console.print(table)


@app.command()
def db_check():
    """Verify database connectivity and schema."""
    from evalharness.storage.db import check_connection

    if check_connection():
        console.print("[green]✓ Database connection OK[/]")
    else:
        console.print("[red]✗ Cannot connect to database[/]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
