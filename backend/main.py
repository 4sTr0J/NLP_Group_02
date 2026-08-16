from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from predictor import predict_code


# ============================================================
# Create FastAPI application
# ============================================================

app = FastAPI(
    title="Source Code Vulnerability Detection API",
    description="API for Random Forest and CNN vulnerability detection",
    version="1.0.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Request model
# ============================================================

class CodeRequest(BaseModel):
    code: str


# ============================================================
# Health check
# ============================================================

@app.get("/")
def root():

    return {
        "message": "Source Code Vulnerability Detection API",
        "status": "running"
    }


@app.get("/api/health")
def health():

    return {
        "status": "ok",
        "message": "API is running successfully"
    }


# ============================================================
# Prediction endpoint
# ============================================================

@app.post("/api/predict")
def predict(request: CodeRequest):

    # Validate source code
    if not request.code.strip():

        raise HTTPException(
            status_code=400,
            detail="Source code cannot be empty."
        )

    try:

        result = predict_code(request.code)

        return result

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )