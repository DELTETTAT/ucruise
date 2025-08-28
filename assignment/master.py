
import os
import requests
import json
from dotenv import load_dotenv
import logging
from assignment import assignment_route, assignment_balance

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_env_and_fetch_data(source_id: str, parameter: int = 1, string_param: str = ""):
    """Load environment variables and fetch data from API"""
    env_path = ".env"
    if os.path.exists(env_path):
        load_dotenv(env_path)
    else:
        raise FileNotFoundError(f".env file not found at {env_path}")

    BASE_API_URL = os.getenv("API_URL")
    API_AUTH_TOKEN = os.getenv("API_AUTH_TOKEN")
    if not BASE_API_URL or not API_AUTH_TOKEN:
        raise ValueError("Both API_URL and API_AUTH_TOKEN must be set in .env")

    # Send both parameters along with source_id in the API URL
    API_URL = f"{BASE_API_URL.rstrip('/')}/{source_id}/{parameter}/{string_param}"
    headers = {
        "Authorization": f"Bearer {API_AUTH_TOKEN}",
        "Content-Type": "application/json"
    }

    print(f"📡 Making API call to: {API_URL}")
    resp = requests.get(API_URL, headers=headers)
    resp.raise_for_status()
    
    # Check if response body is empty
    if len(resp.text.strip()) == 0:
        raise ValueError(
            f"API returned empty response body. "
            f"Status: {resp.status_code}, "
            f"Content-Type: {resp.headers.get('content-type', 'unknown')}, "
            f"URL: {API_URL}"
        )
    
    try:
        payload = resp.json()
    except json.JSONDecodeError as e:
        raise ValueError(
            f"API returned invalid JSON. "
            f"Status: {resp.status_code}, "
            f"Content-Type: {resp.headers.get('content-type', 'unknown')}, "
            f"Response body: '{resp.text[:200]}...', "
            f"JSON Error: {str(e)}"
        )

    if not payload.get("status") or "data" not in payload:
        raise ValueError(
            "Unexpected response format: 'status' or 'data' missing")

    # Use the provided parameters
    data = payload["data"]
    data["_parameter"] = parameter
    data["_string_param"] = string_param

    # Handle nested drivers structure
    if "drivers" in data:
        drivers_data = data["drivers"]
        data["driversUnassigned"] = drivers_data.get("driversUnassigned", [])
        data["driversAssigned"] = drivers_data.get("driversAssigned", [])
    else:
        # Fallback for old structure
        data["driversUnassigned"] = data.get("driversUnassigned", [])
        data["driversAssigned"] = data.get("driversAssigned", [])

    # Log the data structure for debugging
    print(f"📊 API Response structure:")
    print(f"   - users: {len(data.get('users', []))}")
    print(f"   - driversUnassigned: {len(data.get('driversUnassigned', []))}")
    print(f"   - driversAssigned: {len(data.get('driversAssigned', []))}")

    return data


def determine_assignment_strategy(data):
    """
    Determine which assignment algorithm to use based on ride_settings
    
    Priority logic:
    - If zigzag_priority = 1: use assignment_balance (focus on utilization balance)
    - If route_priority = 1: use assignment_route (focus on efficient routes)
    - Default: use assignment_route
    """
    
    # Extract ride_settings from the data
    ride_settings = data.get("ride_settings", {})
    zigzag_priority = ride_settings.get("zigzag_priority", 0)
    route_priority = ride_settings.get("route_priority", 0)
    
    print(f"🎯 Ride Settings Analysis:")
    print(f"   - zigzag_priority: {zigzag_priority}")
    print(f"   - route_priority: {route_priority}")
    
    # Decision logic
    if zigzag_priority == 1:
        strategy = "balance"
        algorithm_name = "assignment_balance"
        print(f"🔄 Selected: BALANCE EFFICIENCY (zigzag_priority=1)")
    elif route_priority == 1:
        strategy = "route"
        algorithm_name = "assignment_route"
        print(f"🛣️ Selected: ROUTE EFFICIENCY (route_priority=1)")
    else:
        # Default to route efficiency
        strategy = "route"
        algorithm_name = "assignment_route"
        print(f"🛣️ Default: ROUTE EFFICIENCY (no explicit priority set)")
    
    return strategy, algorithm_name


def run_assignment(source_id: str, parameter: int = 1, string_param: str = ""):
    """
    Master assignment function that determines strategy and delegates to appropriate algorithm
    """
    try:
        # Ensure source_id is clean and not JSON data
        if isinstance(source_id, str) and source_id.startswith('{'):
            print(f"⚠️ WARNING: source_id contains JSON data, extracting proper source_id")
            source_id = "UC_unify_dev"  # Use a safe default
        
        print(f"🧠 MASTER CONTROLLER: Starting assignment for source_id: {source_id}")
        print(f"📋 Parameters: {parameter}, String: {string_param}")
        
        # Fetch data from API
        data = load_env_and_fetch_data(source_id, parameter, string_param)
        
        # Determine which assignment strategy to use
        strategy, algorithm_name = determine_assignment_strategy(data)
        
        # Execute appropriate assignment algorithm
        if strategy == "balance":
            print(f"🎯 Executing: {algorithm_name} for balanced utilization...")
            result = assignment_balance.run_assignment(data)
        else:  # strategy == "route"
            print(f"🎯 Executing: {algorithm_name} for route efficiency...")
            result = assignment_route.run_assignment(data)
        
        # Add metadata about the strategy used
        if result.get("status") == "true":
            result["assignment_strategy"] = strategy
            result["algorithm_used"] = algorithm_name
            result["parameter"] = parameter
            result["string_param"] = string_param
            
            print(f"✅ MASTER CONTROLLER: Assignment completed successfully")
            print(f"📊 Strategy: {strategy.upper()}, Algorithm: {algorithm_name}")
            print(f"🚗 Routes created: {len(result.get('data', []))}")
            print(f"👥 Users assigned: {sum(len(route.get('assigned_users', [])) for route in result.get('data', []))}")
        else:
            print(f"❌ MASTER CONTROLLER: Assignment failed")
        
        return result
        
    except requests.exceptions.RequestException as req_err:
        logger.error(f"API request failed: {req_err}")
        return {
            "status": "false", 
            "details": str(req_err), 
            "data": [],
            "assignment_strategy": "unknown",
            "algorithm_used": "none"
        }
    except Exception as e:
        logger.error(f"Master assignment failed: {e}", exc_info=True)
        return {
            "status": "false", 
            "details": str(e), 
            "data": [],
            "assignment_strategy": "unknown",
            "algorithm_used": "none"
        }
