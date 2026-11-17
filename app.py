
import mlflow
import mlflow.sklearn

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


app = FastAPI(title="Iris MLOps Level 2 API")

MODEL_NAME = "IrisRandomForestModel"
MODEL_ALIAS = "champion"

# Configure MLflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")


# ==========================================
# 1. LOAD CHAMPION MODEL FROM MLFLOW
# ==========================================

def load_latest_model():

    try:

        model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"

        loaded_model = mlflow.sklearn.load_model(model_uri)

        print("Champion model loaded successfully!")

        return loaded_model

    except Exception as e:

        print(f"Model loading warning: {e}")

        return None


model = load_latest_model()


# ==========================================
# 2. INPUT DATA FORMAT
# ==========================================

class IrisInput(BaseModel):

    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


# ==========================================
# 3. HEALTH CHECK API
# ==========================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": model is not None
    }


# ==========================================
# 4. PREDICTION API
# ==========================================

@app.post("/predict")
def predict(data: IrisInput):

    if model is None:

        raise HTTPException(
            status_code=503,
            detail="Model artifact unavailable."
        )

    features = [[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]]

    prediction = model.predict(features)

    return {
        "prediction": int(prediction[0])
    }
