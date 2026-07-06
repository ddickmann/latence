"""Airflow Runner adapter for the Latence framework (S12, ADR-0003, ADR-0010)."""

from __future__ import annotations

from latence_runner_airflow.runner import RUNNER_NAME, AirflowRunner, XComBackend

__all__ = ["RUNNER_NAME", "AirflowRunner", "XComBackend"]
