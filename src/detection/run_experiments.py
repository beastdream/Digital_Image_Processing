"""Preprocessing experiment dataset preparation entry point."""
from __future__ import annotations

from src.detection.experiment_suite import load_experiment_suite, suite_experiments
from src.detection.run_preprocessing_experiments import generate_preprocessed_dataset


def run_experiment_comparison() -> None:
    suite = load_experiment_suite(); source_yaml = suite["dataset"]["source_yaml"]
    for name, config in suite_experiments(suite):
        print(f"{name}: {generate_preprocessed_dataset(name, config, source_yaml)}")


if __name__ == "__main__":
    run_experiment_comparison()
