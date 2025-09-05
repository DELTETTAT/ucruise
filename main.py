from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import os
from assignment import run_assignment
from logger_config import get_logger  # Import the logger

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

@app.post("/assign-drivers/{source_id}/{parameter}/{string_param}")
def assign_drivers(source_id: str, parameter: int, string_param: str):
    logger = get_logger()  # Initialize the logger
    try:
        logger.info(f"🚗 Starting assignment for source_id: {source_id}, parameter: {parameter}, string_param: {string_param}")

        # Use automatic API-based routing like run_and_view.py
        result = run_assignment(source_id, parameter, string_param)

        if result["status"] == "true":
            logger.info(f"✅ Assignment successful. Routes: {len(result['data'])}")
            with open("drivers_and_routes.json", "w") as f:
                import json
                json.dump(result["data"], f, indent=2)
        else:
            logger.error(f"❌ Assignment failed: {result.get('details', 'Unknown error')}")

        return result

    except Exception as e:
        logger.error(f"❌ Server error: {e}", exc_info=True)
        return {"status": "false", "details": f"Server error: {str(e)}", "data": [], "parameter": 1, "string_param": ""}

@app.get("/routes")
def get_routes():
    if os.path.exists("drivers_and_routes.json"):
        return FileResponse("drivers_and_routes.json", media_type="application/json")
    else:
        return {"status": "false", "message": "No routes data available. Run assignment first.", "data": []}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)