from pydantic import BaseModel, Field

class MentalHealthPredictionModel(BaseModel):
    Age: int = Field(..., alias="Age", example=35)
    MaritalStatus: str = Field(..., alias="Marital Status", example="Married")
    EducationLevel: str = Field(..., alias="Education Level", example="Bachelor's Degree")
    NumberOfChildren: int = Field(..., alias="Number of Children", example=2)
    SmokingStatus: str = Field(..., alias="Smoking Status", example="Non-smoker")
    PhysicalActivityLevel: str = Field(..., alias="Physical Activity Level", example="Active")
    EmploymentStatus: str = Field(..., alias="Employment Status", example="Employed")
    Income: float = Field(..., alias="Income", example=60000.0)
    AlcoholConsumption: str = Field(..., alias="Alcohol Consumption", example="Moderate")
    DietaryHabits: str = Field(..., alias="Dietary Habits", example="Healthy")
    SleepPatterns: str = Field(..., alias="Sleep Patterns", example="Good")
    HistoryOfSubstanceAbuse: str = Field(..., alias="History of Substance Abuse", example="No")
    FamilyHistoryOfDepression: str = Field(..., alias="Family History of Depression", example="Yes")
    ChronicMedicalConditions: str = Field(..., alias="Chronic Medical Conditions", example="No")

    class Config:
        populate_by_name = True
        json_schema_extra = {
                "example": {
        "Age": 35,
        "Marital Status": "Married",
        "Education Level": "Bachelor's Degree",
        "Number of Children": 2,
        "Smoking Status": "Non-smoker",
        "Physical Activity Level": "Active",
        "Employment Status": "Employed",
        "Income": 60000.0,
        "Alcohol Consumption": "Moderate",
        "Dietary Habits": "Healthy",
        "Sleep Patterns": "Good",
        "History of Substance Abuse": "No",
        "Family History of Depression": "Yes",
        "Chronic Medical Conditions": "No"
    }

        }
