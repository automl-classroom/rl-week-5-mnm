import ast
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rliable import metrics, plotting


def load_returns_from_csv(files):
    """Lädt und parst CSV-Dateien mit 'seed' und 'return'-Spalte."""
    all_returns = {}
    for file in files:
        algo_name = os.path.splitext(os.path.basename(file))[
            0
        ]  # z. B. "ppo_runs.csv" → "ppo"
        df = pd.read_csv(file)

        if "seed" not in df.columns or "return" not in df.columns:
            raise ValueError(f"CSV {file} braucht 'seed' und 'return' Spalten.")

        returns = []
        for _, row in df.iterrows():
            try:
                parsed_returns = ast.literal_eval(row["return"])
                returns.append(parsed_returns)
            except:
                raise ValueError(f"Ungültiges Format in 'return': {row['return']}")

        all_returns[algo_name] = returns

    return all_returns


def plot_rliable_metrics(returns_dict, optimal_score=None, save_path=None):
    """Berechnet und plottet Median, IQM, Mean, Optimality Gap für mehrere Algorithmen."""
    # returns_dict[algo] = List of [List of returns per seed]
    aggregate_fns = {
        "IQM": metrics.aggregate_iqm,
        "Median": metrics.aggregate_median,
        "Mean": metrics.aggregate_mean,
    }

    if optimal_score is not None:
        aggregate_fns["Optimality Gap"] = lambda x: metrics.aggregate_optimality_gap(
            x, optimal_score
        )

    # Bootstrap aggregierte Metriken
    score_distributions = {
        k: np.array([np.mean(run) for run in v])  # Mittelwert pro Seed
        for k, v in returns_dict.items()
    }

    results, cis = metrics.aggregate_and_bootstrap(
        score_distributions,
        aggregate_fns,
        num_trials=1000,
        confidence_interval_size=0.95,
    )

    # Plotten
    fig, ax = plotting.plot_score_distribution(
        results,
        cis,
        algorithms=list(returns_dict.keys()),
        metric_names=list(aggregate_fns.keys()),
        xlabel="Aggregate Score",
        title="Aggregate Performance Metrics with 95% CIs",
    )

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"Plot gespeichert unter: {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    # Hier deine CSV-Dateien angeben
    csv_files = [
        "date/eval_100_episodes.csv",
        "date/eval_500_episodes.csv",
        "date/eval_default.csv",
        "date/eval_lr_1e-1.csv",
        "date/eval_lr_1e-3.csv",
        "date/eval_New_hidden_layer.csv",
        "date/eval_trajectory_length_2.csv",
        "date/eval_trajectory_length_10.csv",
    ]

    # Optional: Maximal erreichbarer Score für Optimality Gap
    optimal_return = 100.0

    # Daten laden
    returns_dict = load_returns_from_csv(csv_files)

    # Plot generieren
    plot_rliable_metrics(returns_dict, optimal_score=optimal_return)
