
import mlflow
import mlflow.sklearn

from mlflow import MlflowClient

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Support both direct execution and pytest imports
try:
    from .monitor_drift import check_data_drift
except ImportError:
    from monitor_drift import check_data_drift


# ==========================================
# 1. MLFLOW CONFIGURATION
# ==========================================

mlflow.set_tracking_uri("sqlite:///mlflow.db")

mlflow.set_experiment("iris-continuous-retraining")

client = MlflowClient()

MODEL_NAME = "IrisRandomForestModel"

ACCURACY_THRESHOLD = 0.90

MODEL_ALIAS = "champion"


# ==========================================
# 2. GET CURRENT PRODUCTION MODEL ACCURACY
# ==========================================

def get_production_accuracy():

    try:

        model_version = client.get_model_version_by_alias(
            MODEL_NAME,
            MODEL_ALIAS
        )

        run_id = model_version.run_id

        run = client.get_run(run_id)

        accuracy = run.data.metrics.get("accuracy")

        print(f"Current production accuracy: {accuracy}")

        return accuracy

    except Exception as e:

        print(f"No existing production model found: {e}")

        return None


# ==========================================
# 3. TRAIN AND REGISTER MODEL
# ==========================================

def train_and_register_model():

    print("\nInitiating retraining process...")

    # Load Iris dataset
    iris = load_iris()

    X_train, X_test, y_train, y_test = train_test_split(
        iris.data,
        iris.target,
        test_size=0.2,
        random_state=42
    )

    # Model hyperparameters
    n_estimators = 100
    max_depth = 5

    # Train Random Forest
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )

    model.fit(X_train, y_train)

    # Evaluate candidate model
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print(f"Candidate Model Trained. Accuracy: {accuracy:.4f}")

    # Get existing production accuracy
    production_accuracy = get_production_accuracy()

    # ======================================
    # 4. QUALITY GATE
    # ======================================

    if accuracy < ACCURACY_THRESHOLD:

        print(
            f"Candidate accuracy {accuracy:.4f} "
            f"is below threshold {ACCURACY_THRESHOLD:.2f}."
        )

        print("Model promotion skipped.")

        return

    if production_accuracy is not None:

        if accuracy <= production_accuracy:

            print("Candidate model did not outperform production model.")

            print("Model promotion skipped.")

            return

    # ======================================
    # 5. MLFLOW LOGGING AND REGISTRATION
    # ======================================

    with mlflow.start_run() as run:

        mlflow.log_param("n_estimators", n_estimators)

        mlflow.log_param("max_depth", max_depth)

        mlflow.log_param(
            "model_type",
            "RandomForestClassifier"
        )

        mlflow.log_metric("accuracy", accuracy)

        # Log and register model
        model_info = mlflow.sklearn.log_model(

            sk_model=model,

            name="model",

            registered_model_name=MODEL_NAME,

            skops_trusted_types=[
                "sklearn.tree._tree.Tree"
            ]
        )

        print("\nModel successfully logged and registered.")

        print(f"Model URI: {model_info.model_uri}")

    # ======================================
    # 6. PROMOTE REGISTERED MODEL
    # ======================================

    # Retrieve the version created by this run
    versions = client.search_model_versions(
        f"name='{MODEL_NAME}'"
    )

    registered_version = next(
        (
            version for version in versions
            if version.run_id == run.info.run_id
        ),
        None
    )

    if registered_version is None:

        raise RuntimeError(
            "Registered model version could not be found."
        )

    # Assign champion alias
    client.set_registered_model_alias(
        name=MODEL_NAME,
        alias=MODEL_ALIAS,
        version=registered_version.version
    )

    print("\n========== MODEL PROMOTION ==========")

    print(
        f"Registered version: {registered_version.version}"
    )

    print(f"Production alias: {MODEL_ALIAS}")

    print("Model successfully promoted!")


# ==========================================
# 7. COMPLETE RETRAINING PIPELINE
# ==========================================

def run_pipeline():

    print("\n========== CONTINUOUS TRAINING PIPELINE ==========")

    print("\nStep 1: Checking data drift...")

    drift_detected = check_data_drift()

    if drift_detected:

        print(
            "\nData drift confirmed! "
            "Triggering retraining pipeline..."
        )

        train_and_register_model()

    else:

        print("\nNo significant drift detected.")

        print("Model operating within normal bounds.")


# ==========================================
# 8. ENTRY POINT
# ==========================================

if __name__ == "__main__":

    run_pipeline()
