"""Test Pareto scoring mechanism command."""

import typer


def test_scoring(
    n_miners: int = typer.Option(5, help="Number of simulated miners"),
    n_envs: int = typer.Option(3, help="Number of environments"),
):
    """
    Test the Pareto scoring mechanism with simulated data.
    """
    import numpy as np

    from kinitro.scoring.pareto import compute_pareto_frontier
    from kinitro.scoring.winners_take_all import compute_full_scoring

    typer.echo(f"Testing Pareto scoring with {n_miners} miners, {n_envs} environments\n")

    # Generate random scores
    env_ids = [f"env_{i}" for i in range(n_envs)]
    miner_scores = {}

    for uid in range(n_miners):
        miner_scores[uid] = {env_id: float(np.random.uniform(0.3, 0.9)) for env_id in env_ids}

    # Display scores
    typer.echo("Miner scores:")
    for uid, scores in miner_scores.items():
        scores_str = ", ".join(f"{k}: {v:.2f}" for k, v in scores.items())
        typer.echo(f"  UID {uid}: {scores_str}")

    # Compute Pareto frontier
    pareto = compute_pareto_frontier(miner_scores, env_ids, n_samples_per_env=50)
    typer.echo(f"\nPareto frontier: {pareto.frontier_uids}")

    # Compute weights
    weights = compute_full_scoring(miner_scores, env_ids)
    typer.echo("\nFinal weights:")
    for uid, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True):
        typer.echo(f"  UID {uid}: {weight:.4f}")
