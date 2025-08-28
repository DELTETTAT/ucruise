
import os
import math
import json
import requests
import numpy as np
import pandas as pd
from functools import lru_cache
from urllib.parse import quote
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)

class ConfigManager:
    """Unified configuration management system"""
    
    def __init__(self):
        self._config = None
        self._load_config()
    
    def _load_config(self):
        """Load configuration from file and environment"""
        try:
            with open('config.json') as f:
                self._config = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.warning(f"Could not load config.json: {e}")
            self._config = {}
    
    def get_config(self, assignment_type: str):
        """Get configuration for specific assignment type"""
        # Office coordinates with environment variable fallbacks
        office_lat = float(os.getenv("OFFICE_LAT", self._config.get("office_latitude", 30.6810489)))
        office_lon = float(os.getenv("OFFICE_LON", self._config.get("office_longitude", 76.7260711)))
        
        # Validate coordinates
        if not (-90 <= office_lat <= 90):
            logger.warning(f"Invalid office latitude {office_lat}, using default")
            office_lat = 30.6810489
        if not (-180 <= office_lon <= 180):
            logger.warning(f"Invalid office longitude {office_lon}, using default")
            office_lon = 76.7260711
        
        config_key = f"{assignment_type}_config"
        type_config = self._config.get(config_key, {})
        
        # Base configuration
        config = {
            'OFFICE_LAT': office_lat,
            'OFFICE_LON': office_lon,
            'LAT_TO_KM': 111.0,
            'LON_TO_KM': 111.0 * math.cos(math.radians(office_lat)),
            'assignment_type': assignment_type
        }
        
        # Assignment-specific defaults
        if assignment_type == "route_assignment":
            defaults = {
                'MAX_FILL_DISTANCE_KM': 3.5,
                'MERGE_DISTANCE_KM': 2.0,
                'DBSCAN_EPS_KM': 1.2,
                'MIN_SAMPLES_DBSCAN': 2,
                'MAX_BEARING_DIFFERENCE': 12,
                'MAX_TURNING_ANGLE': 20,
                'MIN_UTIL_THRESHOLD': 0.4,
                'LOW_UTILIZATION_THRESHOLD': 0.4,
                'UTILIZATION_PENALTY_PER_SEAT': 0.5,
                'OVERFLOW_PENALTY_KM': 8.0,
                'DISTANCE_ISSUE_THRESHOLD': 6.0,
                'SWAP_IMPROVEMENT_THRESHOLD': 0.3,
                'MAX_SWAP_ITERATIONS': 2,
                'optimization_mode': 'route_efficiency',
                'direction_weight': 4.0,
                'capacity_weight': 1.0,
                'zigzag_penalty_weight': 5.0
            }
        else:  # balance_assignment
            defaults = {
                'MAX_FILL_DISTANCE_KM': 5.0,
                'MERGE_DISTANCE_KM': 3.0,
                'DBSCAN_EPS_KM': 3.0,
                'MIN_SAMPLES_DBSCAN': 2,
                'MAX_BEARING_DIFFERENCE': 30,
                'MIN_UTIL_THRESHOLD': 0.5,
                'LOW_UTILIZATION_THRESHOLD': 0.5,
                'UTILIZATION_PENALTY_PER_SEAT': 2.0,
                'OVERFLOW_PENALTY_KM': 10.0,
                'DISTANCE_ISSUE_THRESHOLD': 8.0,
                'SWAP_IMPROVEMENT_THRESHOLD': 0.5,
                'MAX_SWAP_ITERATIONS': 3,
                'optimization_mode': 'balance_assignment',
                'direction_weight': 1.0,
                'capacity_weight': 2.0
            }
        
        # Apply configuration values
        for key, default_value in defaults.items():
            config[key] = type_config.get(key.lower(), default_value)
        
        return config

# Global configuration manager
config_manager = ConfigManager()

class ValidationError(Exception):
    """Custom validation error"""
    pass

def validate_coordinates(lat: float, lon: float, context: str = "") -> tuple:
    """Validate and return coordinates"""
    try:
        lat = float(lat)
        lon = float(lon)
        if not (-90 <= lat <= 90):
            raise ValidationError(f"{context} invalid latitude: {lat} (must be -90 to 90)")
        if not (-180 <= lon <= 180):
            raise ValidationError(f"{context} invalid longitude: {lon} (must be -180 to 180)")
        return lat, lon
    except (ValueError, TypeError) as e:
        raise ValidationError(f"{context} invalid coordinates: {e}")

def validate_input_data(data: dict) -> None:
    """Comprehensive data validation"""
    if not isinstance(data, dict):
        raise ValidationError("API response must be a dictionary")

    # Check for users
    users = data.get("users", [])
    if not isinstance(users, list):
        raise ValidationError("Users must be a list")
    
    if len(users) == 0:
        raise ValidationError("No users found in API response")

    # Validate each user
    for i, user in enumerate(users):
        if not isinstance(user, dict):
            raise ValidationError(f"User {i} must be a dictionary")

        required_fields = ["id", "latitude", "longitude"]
        for field in required_fields:
            if field not in user or user[field] is None or user[field] == "":
                raise ValidationError(f"User {i} missing or empty required field: {field}")

        validate_coordinates(user["latitude"], user["longitude"], f"User {i}")

    # Get all drivers
    all_drivers = []
    if "drivers" in data:
        drivers_data = data["drivers"]
        all_drivers.extend(drivers_data.get("driversUnassigned", []))
        all_drivers.extend(drivers_data.get("driversAssigned", []))
    else:
        all_drivers.extend(data.get("driversUnassigned", []))
        all_drivers.extend(data.get("driversAssigned", []))

    if not all_drivers:
        raise ValidationError("No drivers found in API response")

    # Validate drivers
    for i, driver in enumerate(all_drivers):
        if not isinstance(driver, dict):
            raise ValidationError(f"Driver {i} must be a dictionary")

        required_fields = ["id", "capacity", "latitude", "longitude"]
        for field in required_fields:
            if field not in driver or driver[field] is None or driver[field] == "":
                raise ValidationError(f"Driver {i} missing or empty required field: {field}")

        validate_coordinates(driver["latitude"], driver["longitude"], f"Driver {i}")
        
        try:
            capacity = int(driver["capacity"])
            if capacity <= 0:
                raise ValidationError(f"Driver {i} invalid capacity: {capacity} (must be > 0)")
        except (ValueError, TypeError) as e:
            raise ValidationError(f"Driver {i} invalid capacity: {e}")

@lru_cache(maxsize=1000)
def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate haversine distance between two points in kilometers"""
    if lat1 == lat2 and lon1 == lon2:
        return 0.0
        
    R = 6371.0  # Earth radius in kilometers
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    a = min(1.0, abs(a))  # Clamp to [0,1] to prevent domain errors
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return abs(R * c)

def calculate_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate bearing from point A to B in degrees"""
    lat1, lat2 = map(math.radians, [lat1, lat2])
    dlon = math.radians(lon2 - lon1)
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    return (math.degrees(math.atan2(x, y)) + 360) % 360

def calculate_bearing_vectorized(lat1, lon1, lat2, lon2):
    """Vectorized calculation of bearing from point A to B in degrees"""
    lat1, lat2 = np.radians(lat1), np.radians(lat2)
    dlon = np.radians(lon2 - lon1)
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    return (np.degrees(np.arctan2(x, y)) + 360) % 360

def bearing_difference(b1: float, b2: float) -> float:
    """Compute minimum difference between two bearings"""
    diff = abs(b1 - b2) % 360
    return min(diff, 360 - diff)

def standardize_driver_priority(drivers_list: list, assignment_type: str) -> list:
    """Standardize driver priority calculation across assignment types"""
    standardized_drivers = []
    seen_drivers = {}
    
    # First pass: collect all drivers with priority assignment
    for i, driver in enumerate(drivers_list):
        driver_copy = driver.copy()
        driver_id = str(driver.get('id', ''))
        shift_type_id = int(driver.get('shift_type_id', 2))
        is_unassigned = driver.get('source', 'unassigned') == 'unassigned'
        
        # Unified priority system
        if is_unassigned and shift_type_id in [1, 3]:
            priority = 1  # Highest priority
        elif is_unassigned and shift_type_id == 2:
            priority = 2
        elif not is_unassigned and shift_type_id in [1, 3]:
            priority = 3
        else:
            priority = 4  # Lowest priority
        
        driver_copy['priority'] = priority
        driver_copy['source_index'] = i
        
        # Keep highest priority version of each driver
        if driver_id not in seen_drivers or seen_drivers[driver_id]['priority'] > priority:
            seen_drivers[driver_id] = driver_copy
    
    return list(seen_drivers.values())

def load_env_and_fetch_data(source_id: str, parameter: int = 1, string_param: str = ""):
    """Load environment variables and fetch data from API with proper encoding"""
    env_path = ".env"
    if os.path.exists(env_path):
        load_dotenv(env_path)
    else:
        raise FileNotFoundError(f".env file not found at {env_path}")

    BASE_API_URL = os.getenv("API_URL")
    API_AUTH_TOKEN = os.getenv("API_AUTH_TOKEN")
    if not BASE_API_URL or not API_AUTH_TOKEN:
        raise ValueError("Both API_URL and API_AUTH_TOKEN must be set in .env")

    # Proper URL encoding for string parameters
    encoded_string_param = quote(str(string_param))
    API_URL = f"{BASE_API_URL.rstrip('/')}/{source_id}/{parameter}/{encoded_string_param}"
    
    headers = {
        "Authorization": f"Bearer {API_AUTH_TOKEN}",
        "Content-Type": "application/json"
    }

    logger.info(f"Making API call to: {API_URL}")
    resp = requests.get(API_URL, headers=headers, timeout=30)
    resp.raise_for_status()
    
    if len(resp.text.strip()) == 0:
        raise ValueError(f"API returned empty response body. Status: {resp.status_code}")
    
    try:
        payload = resp.json()
    except json.JSONDecodeError as e:
        raise ValueError(f"API returned invalid JSON: {str(e)}")

    if not payload.get("status") or "data" not in payload:
        raise ValueError("Unexpected response format: 'status' or 'data' missing")

    data = payload["data"]
    data["_parameter"] = parameter
    data["_string_param"] = string_param

    # Handle nested drivers structure and mark source
    if "drivers" in data:
        drivers_data = data["drivers"]
        unassigned = drivers_data.get("driversUnassigned", [])
        assigned = drivers_data.get("driversAssigned", [])
        
        # Mark driver sources
        for driver in unassigned:
            driver['source'] = 'unassigned'
        for driver in assigned:
            driver['source'] = 'assigned'
            
        data["driversUnassigned"] = unassigned
        data["driversAssigned"] = assigned
    else:
        # Fallback for old structure
        unassigned = data.get("driversUnassigned", [])
        assigned = data.get("driversAssigned", [])
        
        for driver in unassigned:
            driver['source'] = 'unassigned'
        for driver in assigned:
            driver['source'] = 'assigned'
            
        data["driversUnassigned"] = unassigned
        data["driversAssigned"] = assigned

    return data

def extract_office_coordinates(data: dict):
    """Extract dynamic office coordinates from API data"""
    company_data = data.get("company", {})
    config = config_manager.get_config("route_assignment")  # Default for extraction
    office_lat = float(company_data.get("latitude", config['OFFICE_LAT']))
    office_lon = float(company_data.get("longitude", config['OFFICE_LON']))
    return office_lat, office_lon

def prepare_dataframes(data: dict, assignment_type: str):
    """Prepare and clean user and driver dataframes"""
    users_list = data.get("users", [])
    all_drivers = []
    all_drivers.extend(data.get("driversUnassigned", []))
    all_drivers.extend(data.get("driversAssigned", []))

    # Prepare user DataFrame
    user_data = []
    for user in users_list:
        try:
            user_data.append({
                'user_id': str(user.get('id', '')),
                'latitude': float(user.get('latitude', 0.0)),
                'longitude': float(user.get('longitude', 0.0)),
                'first_name': str(user.get('first_name', '')),
                'email': str(user.get('email', '')),
                'office_distance': float(user.get('office_distance', 0.0))
            })
        except (ValueError, TypeError) as e:
            logger.warning(f"Skipping invalid user {user.get('id', 'unknown')}: {e}")
            continue

    user_df = pd.DataFrame(user_data)

    # Standardize driver priorities
    standardized_drivers = standardize_driver_priority(all_drivers, assignment_type)

    # Prepare driver DataFrame
    driver_data = []
    for driver in standardized_drivers:
        try:
            driver_data.append({
                'driver_id': str(driver.get('id', '')),
                'latitude': float(driver.get('latitude', 0.0)),
                'longitude': float(driver.get('longitude', 0.0)),
                'capacity': int(driver.get('capacity', 1)),
                'vehicle_id': str(driver.get('vehicle_id', '')),
                'priority': int(driver.get('priority', 5)),
                'source_index': int(driver.get('source_index', 0))
            })
        except (ValueError, TypeError) as e:
            logger.warning(f"Skipping invalid driver {driver.get('id', 'unknown')}: {e}")
            continue

    driver_df = pd.DataFrame(driver_data)
    
    # Sort drivers by priority, then by capacity, then by source index for deterministic behavior
    driver_df = driver_df.sort_values(['priority', 'capacity', 'source_index'], 
                                      ascending=[True, False, True])

    return user_df, driver_df

def calculate_bearings_and_features(user_df: pd.DataFrame, office_lat: float, office_lon: float):
    """Calculate bearings and add geographical features"""
    user_df = user_df.copy()
    user_df['bearing_from_office'] = calculate_bearing_vectorized(
        office_lat, office_lon, user_df['latitude'], user_df['longitude'])
    user_df['bearing_sin'] = np.sin(np.radians(user_df['bearing_from_office']))
    user_df['bearing_cos'] = np.cos(np.radians(user_df['bearing_from_office']))
    return user_df

def create_standardized_error_response(error_message: str, details: str = "", 
                                     parameter: int = 1, string_param: str = "") -> dict:
    """Create standardized error response format"""
    return {
        "status": "false",
        "error": error_message,
        "details": details,
        "data": [],
        "unassignedUsers": [],
        "unassignedDrivers": [],
        "parameter": parameter,
        "string_param": string_param,
        "timestamp": pd.Timestamp.now().isoformat()
    }

def validate_route_constraints(routes: list, config: dict) -> list:
    """Validate that created routes satisfy constraints"""
    validated_routes = []
    
    for route in routes:
        # Check capacity constraint
        if len(route['assigned_users']) > route['vehicle_type']:
            logger.warning(f"Route {route['driver_id']} exceeds capacity: "
                         f"{len(route['assigned_users'])} > {route['vehicle_type']}")
            continue
            
        # Check distance constraints
        valid_route = True
        driver_pos = (route['latitude'], route['longitude'])
        
        for user in route['assigned_users']:
            distance = haversine_distance(driver_pos[0], driver_pos[1],
                                        user['lat'], user['lng'])
            if distance > config.get('DISTANCE_ISSUE_THRESHOLD', 8.0):
                logger.warning(f"Route {route['driver_id']} has excessive distance to user "
                             f"{user['user_id']}: {distance:.2f}km")
                valid_route = False
                break
        
        if valid_route:
            validated_routes.append(route)
    
    return validated_routes
