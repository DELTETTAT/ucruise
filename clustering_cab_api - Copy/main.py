from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from assignment import run_assignment

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AssignmentRequest(BaseModel):
    source_id: str

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/assign-drivers/{source_id}")
def assign_drivers(source_id: str):
    result = run_assignment(source_id)

    if result["status"] == "true":
        with open("drivers_and_routes.json", "w") as f:
            import json
            json.dump(result["data"], f, indent=2)

    return result

@app.get("/routes")
def get_routes():
    return FileResponse("drivers_and_routes.json", media_type="application/json")
