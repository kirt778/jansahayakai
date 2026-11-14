from typing import Optional, Literal, Any
from pydantic import BaseModel, Field

class ProfilePatch(BaseModel):
    age: Optional[int] = Field(default=None, ge=0, le=120)
    gender: Optional[Literal["male", "female", "other"]] = None
    state: Optional[str] = None
    occupation: Optional[str] = None
    income: Optional[int] = Field(default=None, ge=0)
    category: Optional[str] = None
    owns_land: Optional[bool] = None

class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: str = Field(min_length=1, max_length=4000)
    language: str = "en"
    profile: Optional[ProfilePatch] = None

class EligibilityRequest(BaseModel):
    profile: ProfilePatch

class SpeechRequest(BaseModel):
    text: str = Field(min_length=1, max_length=4000)
    language: str = "en"
