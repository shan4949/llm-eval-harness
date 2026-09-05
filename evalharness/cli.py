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
    eval_set: str = typer.Option(..., "--eval-set", "-e", help="Eval set name (e.g. golden_qa)"),
    pipeline_config: str = typer.Option("config.yaml", "--pipeline-config", "-p", help="Pipeline config file"),
    baseline: bool = typer.Option(False, "--baseline", help="Mark this run as the new baseline"),
):
    """Run the eval suite against the SUT and store results."""
    console.print(f"[bold green]▶ Running eval set:[/] {eval_set}")
    console.print(f"  Pipeline config: {pipeline_config}")
    console.print("[yellow]  (Phase 1–4 not yet implemented)[/]")


@app.command()
def compare(
    baseline_id: int = typer.Option(..., "--baseline", "-b", help="Baseline run ID"),
    candidate_id: int = typer.Option(..., "--candidate", "-c", help="Candidate run ID to compare"),
):
    """Compare a candidate run against a baseline. Exits non-zero on regression."""
    console.print(f"[bold]Comparing run {candidate_id} against baseline {baseline_id}[/]")
    console.print("[yellow]  (Phase 6 not yet implemented)[/]")


@app.command()
def report(
    run_id: int = typer.Option(..., "--run-id", "-r", help="Run ID to report on"),
):
    """Print a summary report for a completed eval run."""
    console.print(f"[bold]Report for run {run_id}[/]")
    console.print("[yellow]  (Phase 6 not yet implemented)[/]")


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
