from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware  # Add this import
from pydantic import BaseModel, Field
import joblib
import numpy as np

# Define the LinearRegressionGD class
class LinearRegressionGD:
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.iterations):
            y_predicted = np.dot(X, self.weights) + self.bias
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

# Initialize FastAPI app
app = FastAPI(
    title="Prediction API",
    description="An API for making predictions using a trained model.",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (replace "*" with specific origins in production)
    allow_methods=["*"],  # Allow all methods (GET, POST, OPTIONS, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Define input schema using actual column names with aliases
class PredictionInput(BaseModel):
    Age: float
    G: float
    GS: float
    MP: float
    FG: float
    FGA: float
    FG_percent: float = Field(alias="FG%")  # Use alias for "FG%"
    threeP: float = Field(alias="3P")       # Use alias for "3P"
    threePA: float = Field(alias="3PA")     # Use alias for "3PA"
    threeP_percent: float = Field(alias="3P%")  # Use alias for "3P%"
    twoP: float = Field(alias="2P")         # Use alias for "2P"
    twoPA: float = Field(alias="2PA")       # Use alias for "2PA"
    twoP_percent: float = Field(alias="2P%")  # Use alias for "2P%"
    eFG_percent: float = Field(alias="eFG%")  # Use alias for "eFG%"
    FT: float
    FTA: float
    FT_percent: float = Field(alias="FT%")  # Use alias for "FT%"
    ORB: float
    DRB: float
    TRB: float
    AST: float
    STL: float
    BLK: float
    TOV: float
    PF: float

    class Config:
        allow_population_by_field_name = True  # Allow populating by field name or alias

# Load the trained model
model = joblib.load('best_model.pkl')

# Define the prediction endpoint
@app.post('/predict', response_model=dict)
def predict(input_data: PredictionInput):
    try:
        # Convert input data to a numpy array
        input_array = np.array([[
            input_data.Age, input_data.G, input_data.GS, input_data.MP,
            input_data.FG, input_data.FGA, input_data.FG_percent,
            input_data.threeP, input_data.threePA, input_data.threeP_percent,
            input_data.twoP, input_data.twoPA, input_data.twoP_percent,
            input_data.eFG_percent, input_data.FT, input_data.FTA,
            input_data.FT_percent, input_data.ORB, input_data.DRB,
            input_data.TRB, input_data.AST, input_data.STL, input_data.BLK,
            input_data.TOV, input_data.PF
        ]])

        # Make a prediction
        prediction = model.predict(input_array)
        return {"prediction": prediction.tolist()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Run the app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)