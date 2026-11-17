
import pytest
import mlflow

from src.monitor_drift import check_data_drift

from src.retrain_pipeline import (
    train_and_register_model,
    run_pipeline,
    MODEL_NAME
)


def test_data_drift_detection():
    """Verify that drift detection returns a boolean."""

    drift_status = check_data_drift()

    assert isinstance(drift_status, bool)
    assert drift_status is True


def test_train_and_register_model_execution():
    """Verify training and MLflow experiment logging."""

    mlflow.set_tracking_uri("sqlite:///mlflow.db")

    train_and_register_model()

    experiment = mlflow.get_experiment_by_name(
        "iris-continuous-retraining"
    )

    assert experiment is not None

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id]
    )

    assert len(runs) > 0

    assert "metrics.accuracy" in runs.columns

    assert runs["metrics.accuracy"].max() >= 0.90


def test_full_retrain_pipeline_trigger(capsys):
    """Verify the complete retraining pipeline execution."""

    run_pipeline()

    captured = capsys.readouterr()

    assert "Step 1: Checking data drift..." in captured.out

    assert (
        "Data drift confirmed! Triggering retraining pipeline..."
        in captured.out
    )
