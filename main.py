# Updated the assign_drivers function to include the string parameter in the API response.
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from route_assignment import run_assignment as run_route_assignment
from balance_assignment import run_assignment as run_balance_assignment
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

@app.post("/assign-drivers/{source_id}/{parameter}/{string_param}")
def assign_drivers(source_id: str, parameter: int, string_param: str):
    try:
        print(f"🚗 Starting assignment for source_id: {source_id}, parameter: {parameter}, string_param: {string_param}")
        
        # ALWAYS fetch API data first to determine assignment type
        print("📡 Fetching API data to determine assignment type...")
        
        # Use a shared function to fetch API data
        from route_assignment import load_env_and_fetch_data
        
        try:
            api_data = load_env_and_fetch_data(source_id, parameter, string_param)
            
            # Check for ride_settings to determine which assignment method to use
            ride_settings = api_data.get("ride_settings", {})
            route_priority = ride_settings.get("route_priority", 0)
            zigzag_priority = ride_settings.get("zigzag_priority", 0)
            
            print(f"📊 API ride_settings detected:")
            print(f"   - route_priority: {route_priority}")
            print(f"   - zigzag_priority: {zigzag_priority}")
            
            # Determine which assignment method to use based on API response
            if route_priority == 1 and zigzag_priority != 1:
                print("🎯 Using ROUTE ASSIGNMENT (route_priority = 1)")
                result = run_route_assignment(source_id, parameter, string_param)
                assignment_type = "ROUTE ASSIGNMENT"
            elif zigzag_priority == 1 and route_priority != 1:
                print("⚖️ Using BALANCE ASSIGNMENT (zigzag_priority = 1)")
                result = run_balance_assignment(source_id, parameter, string_param)
                assignment_type = "BALANCE ASSIGNMENT"
            elif route_priority == 1 and zigzag_priority == 1:
                # Both set to 1 - conflict resolution (prefer route_priority)
                print("⚠️ Both priorities set to 1 - defaulting to ROUTE ASSIGNMENT")
                result = run_route_assignment(source_id, parameter, string_param)
                assignment_type = "ROUTE ASSIGNMENT (conflict resolution)"
            else:
                # Default to route assignment if no specific priority is set
                print("🎯 Using ROUTE ASSIGNMENT (default - no priorities set)")
                result = run_route_assignment(source_id, parameter, string_param)
                assignment_type = "ROUTE ASSIGNMENT (default)"
                
        except Exception as data_error:
            print(f"⚠️ Error fetching API data: {data_error}")
            print("🎯 Falling back to ROUTE ASSIGNMENT")
            result = run_route_assignment(source_id, parameter, string_param)
            assignment_type = "ROUTE ASSIGNMENT (API error fallback)"

        if result["status"] == "true":
            print(f"✅ {assignment_type} successful. Routes: {len(result['data'])}")
            print(f"📋 Parameter value: {result.get('parameter', 'Not provided')}")
            print(f"📋 String parameter value: {result.get('string_param', 'Not provided')}")
            print(f"📋 Optimization mode: {result.get('optimization_mode', 'Unknown')}")
            
            with open("drivers_and_routes.json", "w") as f:
                import json
                json.dump(result["data"], f, indent=2)
        else:
            print(f"❌ {assignment_type} failed: {result.get('details', 'Unknown error')}")

        return result

    except Exception as e:
        print(f"❌ Server error: {e}")
        return {"status": "false", "details": f"Server error: {str(e)}", "data": [], "parameter": parameter, "string_param": string_param}

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
    return {"message": "Driver Assignment API", "endpoints": ["/assign-drivers/{source_id}/{parameter}/{string_param}", "/routes", "/visualize", "/health"]}