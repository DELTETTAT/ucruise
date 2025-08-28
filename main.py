from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from assignment_engine import run_assignment
from utils import load_env_and_fetch_data, ValidationError
import os
import json
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Driver Assignment API", version="2.0.0")

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
    return {"status": "ok", "version": "2.0.0"}

@app.post("/assign-drivers/{source_id}/{parameter}/{string_param}")
def assign_drivers(source_id: str, parameter: int, string_param: str):
    try:
        logger.info(f"Starting assignment for source_id: {source_id}, parameter: {parameter}, string_param: {string_param}")

        # Determine assignment type based on API data
        assignment_type = determine_assignment_type(source_id, parameter, string_param)
        logger.info(f"Using assignment type: {assignment_type}")

        # Run unified assignment
        result = run_assignment(source_id, parameter, string_param, assignment_type)

        if result["status"] == "true":
            logger.info(f"Assignment successful. Routes: {len(result['data'])}")

            # Save results for visualization
            with open("drivers_and_routes.json", "w") as f:
                json.dump(result["data"], f, indent=2)
            logger.info("Route data saved for visualization")
        else:
            logger.error(f"Assignment failed: {result.get('details', 'Unknown error')}")

        return result

    except Exception as e:
        logger.error(f"Server error: {e}", exc_info=True)
        return {
            "status": "false",
            "error": "Server error",
            "details": str(e),
            "data": [],
            "parameter": parameter,
            "string_param": string_param
        }

def determine_assignment_type(source_id: str, parameter: int, string_param: str) -> str:
    """Determine assignment type based on API data"""
    try:
        logger.info("Fetching API data to determine assignment type...")
        api_data = load_env_and_fetch_data(source_id, parameter, string_param)

        ride_settings = api_data.get("ride_settings", {})
        route_priority = ride_settings.get("route_priority", 0)
        zigzag_priority = ride_settings.get("zigzag_priority", 0)

        logger.info(f"API ride_settings - route_priority: {route_priority}, zigzag_priority: {zigzag_priority}")

        if route_priority == 1 and zigzag_priority != 1:
            return "route_assignment"
        elif zigzag_priority == 1 and route_priority != 1:
            return "balance_assignment"
        elif route_priority == 1 and zigzag_priority == 1:
            logger.warning("Both priorities set to 1 - defaulting to route_assignment")
            return "route_assignment"
        else:
            logger.info("No specific priority set - defaulting to route_assignment")
            return "route_assignment"

    except Exception as e:
        logger.warning(f"Error determining assignment type: {e}")
        logger.info("Falling back to route_assignment")
        return "route_assignment"

@app.get("/routes")
def get_routes():
    """Get saved routes data"""
    if os.path.exists("drivers_and_routes.json"):
        return FileResponse("drivers_and_routes.json", media_type="application/json")
    else:
        return {
            "status": "false",
            "message": "No routes data available. Run assignment first.",
            "data": []
        }

@app.get("/visualize", response_class=HTMLResponse)
def get_visualization():
    """Serve visualization page"""
    return FileResponse("visualize.html")

@app.get("/")
def root():
    """API root endpoint"""
    return {
        "message": "Driver Assignment API v2.0",
        "endpoints": [
            "/assign-drivers/{source_id}/{parameter}/{string_param}",
            "/routes",
            "/visualize",
            "/health"
        ],
        "features": [
            "Unified assignment engine",
            "Dynamic assignment type detection",
            "Standardized error handling",
            "Route constraint validation"
        ]
    }

# Hardcoded values for easier testing
SOURCE_ID = "your_source_id"  # Replace with your actual source ID
PARAMETER = 10                # Replace with your desired parameter
STRING_PARAM = "example_string" # Replace with your desired string_param

def run_and_view():
    """
    Runs the assignment engine with hardcoded values and serves the visualization.
    """
    try:
        logger.info("Running assignment with hardcoded values...")
        assignment_result = assign_drivers(SOURCE_ID, PARAMETER, STRING_PARAM)
        logger.info(f"Assignment process completed. Status: {assignment_result.get('status')}")

        # If assignment was successful, then show the visualization
        if assignment_result.get("status") == "true":
            logger.info("Assignment successful. Serving visualization...")
            # This part is just for printing the URLs, the actual serving is handled by FastAPI
            # You would typically run 'uvicorn main:app --reload' in a real scenario.
            # For Replit, the server runs automatically.

            # This print statement is intended to show where to access the visualization.
            # For Replit, the internal URL will be different from the public one.
            # The following lines are updated to reflect Replit's environment.
            repl_name = os.environ.get('REPL_SLUG', 'my-repl')
            username = os.environ.get('REPL_OWNER', 'user')
            print(f"📱 Manual URL: https://{repl_name}-{username}.repl.co/visualize")
            print(f"📊 API Endpoint: https://{repl_name}-{username}.repl.co/routes")
            print(f"📱 Local URL: http://0.0.0.0:3000/visualize")

        else:
            logger.error("Assignment failed with hardcoded values. Cannot serve visualization.")
            print("Assignment failed with hardcoded values. Cannot serve visualization.")

    except Exception as e:
        logger.error(f"Error in run_and_view: {e}", exc_info=True)
        print(f"An error occurred during the run_and_view process: {e}")

# Example of how to run the assignment and view it (e.g., for testing)
# You can uncomment the line below to run this function when the script starts.
# In a typical FastAPI deployment on Replit, you don't call run_and_view directly,
# as FastAPI itself handles the server startup and requests.
# However, if you want to pre-run the assignment and then access the results, you could use this.

# if __name__ == "__main__":
#     run_and_view()