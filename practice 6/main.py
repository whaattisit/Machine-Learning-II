from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn 
from bin.load import predict_input

app = FastAPI()

class Review(BaseModel):
    description: str = ""


@app.get("/")
def get_status():
    return({"msg" : "ML API is running"})

@app.post("/predict/")
def post_predict(input: Review):
    if not input.description.strip():
        raise HTTPException(status_code=400, detail="Field shouldn't be null")
    try:
        return predict_input(input.description)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction error {exc}") from exc

if __name__ == "__main__":
    uvicorn.run(app=app, host="127.0.0.1", port=8000)