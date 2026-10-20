"""Input contract for inference (shared by predict.py and the FastAPI app).

Codes follow the UCI documentation of processed.cleveland.data.
`ca` and `thal` may be null because they are missing in the source data;
the pipeline imputes them. All other fields are mandatory.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class PatientFeatures(BaseModel):
    model_config = ConfigDict(
        extra="forbid",  # unknown fields -> 422 instead of being silently ignored
        json_schema_extra={
            "example": {
                "age": 58, "sex": 1, "cp": 4, "trestbps": 140, "chol": 260,
                "fbs": 0, "restecg": 2, "thalach": 130, "exang": 1,
                "oldpeak": 1.8, "slope": 2, "ca": 1, "thal": 7,
            }
        },
    )

    age: int = Field(..., ge=1, le=120, description="Age in years")
    sex: Literal[0, 1] = Field(..., description="1 = male, 0 = female")
    cp: Literal[1, 2, 3, 4] = Field(
        ..., description="Chest pain: 1 typical, 2 atypical, 3 non-anginal, "
                         "4 asymptomatic")
    trestbps: float = Field(..., ge=50, le=250, description="Resting BP (mm Hg)")
    chol: float = Field(..., ge=100, le=700, description="Serum cholesterol mg/dl")
    fbs: Literal[0, 1] = Field(..., description="Fasting blood sugar > 120 mg/dl")
    restecg: Literal[0, 1, 2] = Field(..., description="Resting ECG result")
    thalach: float = Field(..., ge=50, le=250, description="Max heart rate")
    exang: Literal[0, 1] = Field(..., description="Exercise-induced angina")
    oldpeak: float = Field(..., ge=0, le=10, description="ST depression")
    slope: Literal[1, 2, 3] = Field(..., description="Slope of peak ST segment")
    ca: Optional[int] = Field(None, ge=0, le=3, description="Major vessels 0-3")
    thal: Optional[Literal[3, 6, 7]] = Field(
        None, description="3 normal, 6 fixed defect, 7 reversible defect")


class PredictionResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    prediction: int = Field(..., description="1 = heart disease, 0 = no disease")
    label: str
    probability_disease: float = Field(..., ge=0, le=1)
    confidence: float = Field(..., ge=0, le=1,
                              description="Probability of the predicted class")
    model_name: str
    model_version: str
    disclaimer: str = ("Educational model - not a clinically validated "
                       "diagnostic tool.")
