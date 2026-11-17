
from pathlib import Path
from sklearn.datasets import load_iris

from evidently import Report
from evidently.presets import DataDriftPreset


def check_data_drift():

    # Load Iris dataset
    iris = load_iris(as_frame=True)
    data = iris.data

    # Reference data
    reference = data.sample(n=100, random_state=42)

    # Simulated production data
    current = data.sample(n=100, random_state=99).copy()

    # Artificially introduce drift
    current["sepal length (cm)"] *= 1.5

    print("Running data drift detection...")

    # Configure Evidently
    report = Report(
        metrics=[
            DataDriftPreset(drift_share=0.25)
        ]
    )

    # Run report
    snapshot = report.run(
        current_data=current,
        reference_data=reference
    )

    # Save HTML report
    project_root = Path(__file__).resolve().parent.parent
    report_path = project_root / "drift_report.html"
    snapshot.save_html(str(report_path))

    # Extract metric results
    results = snapshot.dict()
    metrics = results.get("metrics", [])

    drift_detected = None
    drifted_count = 0
    drift_share = 0.0

    for metric in metrics:

        metric_name = metric.get("metric_name", "")

        if "DriftedColumnsCount" in metric_name:

            value = metric.get("value", {})

            if not isinstance(value, dict):
                raise ValueError("Unexpected drift metric format")

            drifted_count = value["count"]
            drift_share = value["share"]

            drift_detected = drift_share >= 0.25
            break

    if drift_detected is None:
        raise RuntimeError(
            "Could not find DriftedColumnsCount in Evidently results. "
            "Inspect snapshot.dict() for the installed version."
        )

    print("\n========== DRIFT RESULTS ==========")
    print(f"Drifted columns: {drifted_count}")
    print(f"Drift share: {drift_share:.2%}")
    print(f"Dataset drift: {drift_detected}")
    print(f"HTML report: {report_path}")

    return drift_detected


if __name__ == "__main__":
    check_data_drift()
