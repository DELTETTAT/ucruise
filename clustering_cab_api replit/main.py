from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from assignment import run_assignment
import os

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
    try:
        print(f"🚗 Starting assignment for source_id: {source_id}")
        result = run_assignment(source_id)

        if result["status"] == "true":
            print(f"✅ Assignment successful. Routes: {len(result['data'])}")
            with open("drivers_and_routes.json", "w") as f:
                import json
                json.dump(result["data"], f, indent=2)
        else:
            print(f"❌ Assignment failed: {result.get('details', 'Unknown error')}")

        return result

    except Exception as e:
        print(f"❌ Server error: {e}")
        return {"status": "false", "details": f"Server error: {str(e)}", "data": []}

@app.get("/routes")
def get_routes():
    if os.path.exists("drivers_and_routes.json"):
        return FileResponse("drivers_and_routes.json", media_type="application/json")
    else:
        return {"status": "false", "message": "No routes data available. Run assignment first.", "data": []}

@app.get("/visualize", response_class=HTMLResponse)
def get_visualization():
    return FileResponse("visualize.html")

@app.get("/")
def root():
    return {"message": "Driver Assignment API", "endpoints": ["/assign-drivers/{source_id}", "/routes", "/visualize", "/health"]}
