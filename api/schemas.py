# api/schemas.py
from datetime import date
from typing import Optional

from pydantic import BaseModel, Field


class CreditApplication(BaseModel):
    customer_id: Optional[str] = Field(None, description="ID del cliente")
    application_date: Optional[date] = Field(None, description="Fecha de solicitud")

    age: int = Field(..., ge=18, le=100)
    gender: str
    marital_status: str
    dependents: int = Field(..., ge=0, le=10)

    monthly_income: float = Field(..., gt=0)
    employment_type: str
    employment_months: int = Field(..., ge=0)

    requested_amount: float = Field(..., gt=0)
    loan_term_months: int = Field(..., gt=0)
    interest_rate: float = Field(..., gt=0)
    installment: float = Field(..., gt=0)
    debt_to_income: float = Field(..., ge=0)

    num_open_loans: int = Field(..., ge=0)
    num_credit_cards: int = Field(..., ge=0)
    has_mortgage: int = Field(..., ge=0, le=1)

    channel: str
    region: str


class PredictionResult(BaseModel):
    default_probability: float = Field(..., ge=0, le=1)
    default_class: int = Field(..., ge=0, le=1)
