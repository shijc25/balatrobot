from __future__ import annotations

import csv
import datetime as dt
import json
from pathlib import Path
from typing import Any, Dict, List

from rl_train import DEFAULT_CONFIG_PATH, run_training

RESULTS_DIR = Path("experiment_results")


EXPERIMENTS: List[Dict[str, Any]] = [
    {
        "name": "throughput_baseline",
        "config": DEFAULT_CONFIG_PATH,
        "overrides": [
            "run.name=throughput_baseline",
            "run.stop.training_iteration=20",
        ],
    },
    {
        "name": "throughput_low_env_runners",
        "config": DEFAULT_CONFIG_PATH,
        "overrides": [
            "run.name=throughput_low_env_runners",
            "run.stop.training_iteration=20",
            "ppo_overrides.blind_shop.env_runners.num_env_runners=4",
            "ppo_overrides.blind_shop.env_runners.num_envs_per_env_runner=2",
        ],
    },
    {
        "name": "throughput_small_batch",
        "config": DEFAULT_CONFIG_PATH,
        "overrides": [
            "run.name=throughput_small_batch",
            "run.stop.training_iteration=20",
            "ppo_overrides.blind_shop.training.train_batch_size=8192",
            "ppo_overrides.blind_shop.training.sgd_minibatch_size=1024",
        ],
    },
]


def _summarize_trial(experiment: str, trial) -> Dict[str, Any]:
    last = trial.last_result or {}
    sampler_results = last.get("sampler_results") or {}
    perf = sampler_results.get("perf") or {}

    summary = {
        "experiment": experiment,
        "trial_id": trial.trial_id,
        "status": trial.status,
        "logdir": trial.logdir,
        "training_iteration": last.get("training_iteration"),
        "time_total_s": last.get("time_total_s"),
        "episode_reward_mean": last.get("episode_reward_mean"),
        "episode_len_mean": last.get("episode_len_mean"),
        "custom_metrics_round_mean": last.get("custom_metrics/round_mean"),
        "env_steps_per_sec": perf.get("mean_env_steps_per_sec"),
        "agent_steps_per_sec": perf.get("mean_agent_steps_per_sec"),
        "throughput_samples_per_sec": last.get("throughput_samples_per_sec"),
    }
    return summary


def run_experiments() -> List[Dict[str, Any]]:
    RESULTS_DIR.mkdir(exist_ok=True)
    summaries: List[Dict[str, Any]] = []

    for spec in EXPERIMENTS:
        name = spec["name"]
        config_path = spec.get("config", DEFAULT_CONFIG_PATH)
        overrides = list(spec.get("overrides", []))
        print(f"[{dt.datetime.now().isoformat()}] Starting experiment '{name}'")
        analysis = run_training(config_path, overrides, dry_run=False)
        if analysis is None:
            print(f"[{dt.datetime.now().isoformat()}] Experiment '{name}' returned no analysis (dry run?).")
            continue

        for trial in analysis.trials:
            summaries.append(_summarize_trial(name, trial))

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = RESULTS_DIR / f"summary_{timestamp}.csv"
    json_path = RESULTS_DIR / f"summary_{timestamp}.json"

    if summaries:
        fieldnames = list(summaries[0].keys())
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summaries)

        with json_path.open("w") as handle:
            json.dump(summaries, handle, indent=2)

        print(f"Saved summary CSV to {csv_path}")
        print(f"Saved summary JSON to {json_path}")
    else:
        print("No experiment summaries recorded.")

    return summaries


if __name__ == "__main__":
    run_experiments()
