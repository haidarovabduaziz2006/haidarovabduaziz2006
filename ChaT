from fastapi import FastAPI
from pydantic import BaseModel
from app.neural_network import model_predict

app = FastAPI()

class InputData(BaseModel):
    feature1: float
    feature2: float

@app.post("/predict")
async def predict(data: InputData):
    prediction = model_predict(data.feature1, data.feature2)
    return {"prediction": prediction}
