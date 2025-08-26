import os
import math
import json
import requests
import numpy as np
import pandas as pd
import time
from functools import lru_cache
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist
from dotenv import load_dotenv
import warnings
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
import logging
from road_network import RoadNetwork

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Load and validate configuration with robust error handling
def load_and_validate_config():
    """Load configuration with validation and environment fallbacks"""
    try:
        with open('config.json') as f:
            cfg = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(
            f"Warning: Could not load config.json, using defaults. Error: {e}")
        cfg = {}

    # Validate and set configuration with type checking and bounds
    config = {}

    # Distance configurations (must be positive floats)
    config['MAX_FILL_DISTANCE_KM'] = max(
        0.1, float(cfg.get("max_fill_distance_km", 5.0)))
    config['MERGE_DISTANCE_KM'] = max(0.1,
                                      float(cfg.get("merge_distance_km", 3.0)))
    config['DBSCAN_EPS_KM'] = max(0.1, float(cfg.get("dbscan_eps_km", 3.0)))
    config['OVERFLOW_PENALTY_KM'] = max(
        0.0, float(cfg.get("overflow_penalty_km", 10.0)))
    config['DISTANCE_ISSUE_THRESHOLD'] = max(
        0.1, float(cfg.get("distance_issue_threshold_km", 8.0)))
    config['SWAP_IMPROVEMENT_THRESHOLD'] = max(
        0.0, float(cfg.get("swap_improvement_threshold_km", 0.5)))

    # Utilization thresholds (0-1 range)
    config['MIN_UTIL_THRESHOLD'] = max(
        0.0, min(1.0, float(cfg.get("min_util_threshold", 0.5))))
    config['LOW_UTILIZATION_THRESHOLD'] = max(
        0.0, min(1.0, float(cfg.get("low_utilization_threshold", 0.5))))

    # Integer configurations (must be positive)
    config['MIN_SAMPLES_DBSCAN'] = max(1, int(cfg.get("min_samples_dbscan",
                                                      2)))
    config['MAX_SWAP_ITERATIONS'] = max(1,
                                        int(cfg.get("max_swap_iterations", 3)))
    config['MAX_USERS_FOR_FALLBACK'] = max(
        1, int(cfg.get("max_users_for_fallback", 3)))
    config['FALLBACK_MIN_USERS'] = max(1, int(cfg.get("fallback_min_users",
                                                      2)))
    config['FALLBACK_MAX_USERS'] = max(1, int(cfg.get("fallback_max_users",
                                                      7)))

    # Angle configurations (0-180 degrees) - tighter for better route coherence
    config['MAX_BEARING_DIFFERENCE'] = max(
        0, min(180, float(cfg.get("max_bearing_difference", 15))))
    config['STRICT_BEARING_DIFFERENCE'] = max(
        0, min(180, float(cfg.get("strict_bearing_difference", 10))))
    # Update config with more reasonable constraints
    if 'strict_bearing_difference' not in cfg:
        config['STRICT_BEARING_DIFFERENCE'] = 15  # More lenient for splitting
    if 'route_coherence_threshold' not in cfg:
        config['route_coherence_threshold'] = 0.65  # More reasonable base threshold

    # Cost penalties (must be non-negative)
    config['UTILIZATION_PENALTY_PER_SEAT'] = max(
        0.0, float(cfg.get("utilization_penalty_per_seat", 2.0)))

    # Office coordinates with environment variable fallbacks
    office_lat = float(
        os.getenv("OFFICE_LAT", cfg.get("office_latitude", 30.6810489)))
    office_lon = float(
        os.getenv("OFFICE_LON", cfg.get("office_longitude", 76.7260711)))

    # Validate coordinate bounds
    if not (-90 <= office_lat <= 90):
        print(f"Warning: Invalid office latitude {office_lat}, using default")
        office_lat = 30.6810489
    if not (-180 <= office_lon <= 180):
        print(f"Warning: Invalid office longitude {office_lon}, using default")
        office_lon = 76.7260711

    config['OFFICE_LAT'] = office_lat
    config['OFFICE_LON'] = office_lon

    # Road network related configurations
    config['use_road_network'] = cfg.get("use_road_network", False)
    config['road_network_path'] = cfg.get("road_network_path", 'tricity_main_roads.graphml')
    config['min_cluster_size'] = max(2, int(cfg.get("min_cluster_size", 2)))
    config['max_extra_distance_km'] = max(1.0, float(cfg.get("max_extra_distance_km", 3.0)))
    config['fallback_assign_enabled'] = cfg.get("fallback_assign_enabled", True)
    config['min_road_coherence_score'] = max(0.0, min(1.0, float(cfg.get("min_road_coherence_score", 0.85))))


    return config


# Load validated configuration
_config = load_and_validate_config()

# Initialize road network
ROAD_NETWORK = None
if _config.get('use_road_network', False):
    road_network_path = _config.get('road_network_path', 'tricity_main_roads.graphml')
    if os.path.exists(road_network_path):
        try:
            from road_network import RoadNetwork
            ROAD_NETWORK = RoadNetwork(road_network_path)
            print("✅ Road network loaded successfully")
        except Exception as e:
            print(f"⚠️ Failed to load road network: {e}")
            _config['use_road_network'] = False # Disable if loading fails
    else:
        print(f"⚠️ Road network file not found: {road_network_path}")
        _config['use_road_network'] = False # Disable if file not found

MAX_FILL_DISTANCE_KM = max(5.0, float(_config.get('MAX_FILL_DISTANCE_KM', 5.0)))
MERGE_DISTANCE_KM = max(3.0, float(_config.get('MERGE_DISTANCE_KM', 3.0)))
MIN_UTIL_THRESHOLD = _config['MIN_UTIL_THRESHOLD']
DBSCAN_EPS_KM = _config['DBSCAN_EPS_KM']
MIN_SAMPLES_DBSCAN = _config['MIN_SAMPLES_DBSCAN']
MAX_BEARING_DIFFERENCE = _config['MAX_BEARING_DIFFERENCE']
STRICT_BEARING_DIFFERENCE = _config['STRICT_BEARING_DIFFERENCE']
SWAP_IMPROVEMENT_THRESHOLD = _config['SWAP_IMPROVEMENT_THRESHOLD']
MAX_SWAP_ITERATIONS = _config['MAX_SWAP_ITERATIONS']
UTILIZATION_PENALTY_PER_SEAT = _config['UTILIZATION_PENALTY_PER_SEAT']
OVERFLOW_PENALTY_KM = _config['OVERFLOW_PENALTY_KM']
DISTANCE_ISSUE_THRESHOLD = _config['DISTANCE_ISSUE_THRESHOLD']
LOW_UTILIZATION_THRESHOLD = _config['LOW_UTILIZATION_THRESHOLD']
MAX_USERS_FOR_FALLBACK = _config['MAX_USERS_FOR_FALLBACK']
FALLBACK_MIN_USERS = _config['FALLBACK_MIN_USERS']
FALLBACK_MAX_USERS = _config['FALLBACK_MAX_USERS']
OFFICE_LAT = _config['OFFICE_LAT']
OFFICE_LON = _config['OFFICE_LON']




def validate_input_data(data):
    """Comprehensive data validation with bounds checking"""
    if not isinstance(data, dict):
        raise ValueError("API response must be a dictionary")

    # Check for users
    users = data.get("users", [])
    if not users:
        raise ValueError("No users found in API response")

    if not isinstance(users, list):
        raise ValueError("Users must be a list")

    # Special handling for empty users
    if len(users) == 0:
        raise ValueError("Empty users list")

    # Validate each user comprehensively
    for i, user in enumerate(users):
        if not isinstance(user, dict):
            raise ValueError(f"User {i} must be a dictionary")

        required_fields = ["id", "latitude", "longitude"]
        for field in required_fields:
            if field not in user:
                raise ValueError(f"User {i} missing required field: {field}")
            if user[field] is None or user[field] == "":
                raise ValueError(f"User {i} has null/empty {field}")

        # Validate coordinate bounds
        try:
            lat = float(user["latitude"])
            lon = float(user["longitude"])
            if not (-90 <= lat <= 90):
                raise ValueError(
                    f"User {i} invalid latitude: {lat} (must be -90 to 90)")
            if not (-180 <= lon <= 180):
                raise ValueError(
                    f"User {i} invalid longitude: {lon} (must be -180 to 180)")
        except (ValueError, TypeError) as e:
            raise ValueError(f"User {i} invalid coordinates: {e}")

    # Get all drivers from both sources
    all_drivers = []

    # Check nested format first
    if "drivers" in data:
        drivers_data = data["drivers"]
        all_drivers.extend(drivers_data.get("driversUnassigned", []))
        all_drivers.extend(drivers_data.get("driversAssigned", []))

    # Check flat format
    if not all_drivers:
        all_drivers.extend(data.get("driversUnassigned", []))
        all_drivers.extend(data.get("driversAssigned", []))

    if not all_drivers:
        raise ValueError("No drivers found in API response")

    # Validate drivers comprehensively (allow duplicates for pick/drop scenarios)
    duplicate_driver_count = 0
    for i, driver in enumerate(all_drivers):
        if not isinstance(driver, dict):
            raise ValueError(f"Driver {i} must be a dictionary")

        required_fields = ["id", "capacity", "latitude", "longitude"]
        for field in required_fields:
            if field not in driver:
                raise ValueError(f"Driver {i} missing required field: {field}")
            if driver[field] is None or driver[field] == "":
                raise ValueError(f"Driver {i} has null/empty {field}")

        # Count duplicates but don't error (legitimate for pick/drop scenarios)
        driver_id = str(driver.get("id", ""))
        duplicate_count = sum(1 for d in all_drivers
                              if str(d.get("id", "")) == driver_id)
        if duplicate_count > 1:
            duplicate_driver_count += 1

        # Validate driver coordinates
        try:
            lat = float(driver["latitude"])
            lon = float(driver["longitude"])
            if not (-90 <= lat <= 90):
                raise ValueError(f"Driver {i} invalid latitude: {lat}")
            if not (-180 <= lon <= 180):
                raise ValueError(f"Driver {i} invalid longitude: {lon}")
        except (ValueError, TypeError) as e:
            raise ValueError(f"Driver {i} invalid coordinates: {e}")

        # Validate capacity
        try:
            capacity = int(driver["capacity"])
            if capacity <= 0:
                raise ValueError(
                    f"Driver {i} invalid capacity: {capacity} (must be > 0)")
        except (ValueError, TypeError) as e:
            raise ValueError(f"Driver {i} invalid capacity: {e}")

    if duplicate_driver_count > 0:
        print(
            f"ℹ️ INFO: Found {duplicate_driver_count} duplicate driver entries (normal for pick/drop scenarios)"
        )

    print(
        f"✅ Input data validation passed - {len(users)} users, {len(all_drivers)} drivers"
    )


def load_env_and_fetch_data(source_id: str,
                            parameter: int = 1,
                            string_param: str = ""):
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
            f"URL: {API_URL}")

    try:
        payload = resp.json()
    except json.JSONDecodeError as e:
        raise ValueError(
            f"API returned invalid JSON. "
            f"Status: {resp.status_code}, "
            f"Content-Type: {resp.headers.get('content-type', 'unknown')}, "
            f"Response body: '{resp.text[:200]}...', "
            f"JSON Error: {str(e)}")

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


def extract_office_coordinates(data):
    """Extract dynamic office coordinates from API data"""
    company_data = data.get("company", {})
    office_lat = float(company_data.get("latitude", OFFICE_LAT))
    office_lon = float(company_data.get("longitude", OFFICE_LON))
    return office_lat, office_lon


def bearing_difference(b1, b2):
    """Compute minimum difference between two bearings"""
    diff = abs(b1 - b2) % 360
    return min(diff, 360 - diff)


def is_user_along_route_path(driver_pos, existing_users, candidate_user,
                             office_pos):
    """
    Check if candidate user is along the logical driving path
    Uses road network if available, otherwise falls back to bearing-based logic
    """
    driver_lat, driver_lon = driver_pos
    office_lat, office_lon = office_pos
    # Handle different key formats for coordinates
    candidate_lat = candidate_user.get('latitude',
                                       candidate_user.get('lat', 0))
    candidate_lon = candidate_user.get('longitude',
                                       candidate_user.get('lng', 0))

    # Use road network analysis if available
    if ROAD_NETWORK is not None:
        try:
            existing_positions = []
            for u in existing_users:
                lat = u.get('latitude', u.get('lat', 0))
                lng = u.get('longitude', u.get('lng', 0))
                existing_positions.append((lat, lng))

            # More strict road network validation
            is_on_path = ROAD_NETWORK.is_user_on_route_path(
                (driver_lat, driver_lon),
                existing_positions,
                (candidate_lat, candidate_lon),
                (office_lat, office_lon),
                max_detour_ratio=_config.get('max_detour_ratio',
                                             1.08)  # Much stricter ratio
            )

            # Additional road coherence check with stricter thresholds
            if is_on_path:
                test_positions = existing_positions + [
                    (candidate_lat, candidate_lon)
                ]
                coherence_score = ROAD_NETWORK.get_route_coherence_score(
                    (driver_lat, driver_lon), test_positions,
                    (office_lat, office_lon))

                # Require excellent coherence (route follows roads very well)
                min_coherence = _config.get('route_coherence_threshold', 0.9)
                if coherence_score < min_coherence:
                    print(
                        f"      ❌ User rejected due to poor road coherence: {coherence_score:.2f} < {min_coherence}"
                    )
                    return False

                # Additional check: ensure no excessive directional spread
                if len(test_positions) > 1:
                    bearings = []
                    for pos in test_positions:
                        bearing = calculate_bearing(driver_lat, driver_lon,
                                                    pos[0], pos[1])
                        bearings.append(bearing)

                    max_spread = max(bearings) - min(bearings)
                    if max_spread > 180:  # Handle wrap-around
                        max_spread = 360 - max_spread

                    max_allowed_spread = _config.get('max_directional_spread',
                                                     30)
                    if max_spread > max_allowed_spread:
                        print(
                            f"      ❌ User rejected due to excessive directional spread: {max_spread:.1f}° > {max_allowed_spread}°"
                        )
                        return False

            return is_on_path
        except Exception as e:
            print(
                f"      ⚠️ Road network error: {e}, falling back to bearing logic"
            )
            pass  # Fallback to bearing-based logic

    # Enhanced bearing-based logic fallback
    if not existing_users:
        # First user - check if generally in same direction as office
        driver_to_office_bearing = calculate_bearing(driver_lat, driver_lon,
                                                     office_lat, office_lon)
        driver_to_candidate_bearing = calculate_bearing(
            driver_lat, driver_lon, candidate_lat, candidate_lon)
        bearing_diff = bearing_difference(driver_to_office_bearing,
                                          driver_to_candidate_bearing)

        # Also check distance reasonableness
        distance_to_candidate = haversine_distance(driver_lat, driver_lon,
                                                   candidate_lat,
                                                   candidate_lon)
        if distance_to_candidate > MAX_FILL_DISTANCE_KM:
            return False

        return bearing_diff <= MAX_BEARING_DIFFERENCE

    # Calculate the main route direction (driver -> existing users -> office)
    existing_positions = []
    for u in existing_users:
        lat = u.get('latitude', u.get('lat', 0))
        lng = u.get('longitude', u.get('lng', 0))
        existing_positions.append((lat, lng))

    # Find the centroid of existing users
    centroid_lat = sum(pos[0]
                       for pos in existing_positions) / len(existing_positions)
    centroid_lon = sum(pos[1]
                       for pos in existing_positions) / len(existing_positions)

    # Enhanced bearing consistency check
    main_route_bearing = calculate_bearing(driver_lat, driver_lon,
                                           centroid_lat, centroid_lon)
    candidate_bearing = calculate_bearing(driver_lat, driver_lon,
                                          candidate_lat, candidate_lon)
    bearing_diff = bearing_difference(main_route_bearing, candidate_bearing)

    # Stricter bearing tolerance for route coherence
    if bearing_diff > STRICT_BEARING_DIFFERENCE:
        return False

    # Check distance penalty - candidate shouldn't create major detours
    direct_distance = haversine_distance(driver_lat, driver_lon, office_lat,
                                         office_lon)

    # Distance if we go: driver -> candidate -> existing centroid -> office
    detour_distance = (
        haversine_distance(driver_lat, driver_lon, candidate_lat,
                           candidate_lon) +
        haversine_distance(candidate_lat, candidate_lon, centroid_lat,
                           centroid_lon) +
        haversine_distance(centroid_lat, centroid_lon, office_lat, office_lon))

    # Calculate detour ratio
    detour_ratio = detour_distance / max(direct_distance, 1.0)

    # Improvement: Check for straight roads vs winding roads
    # If the detour ratio is good (e.g., <= 1.1), we can be more lenient with deviation from the direct path,
    # especially if the road network suggests a straighter path for the user.
    # Otherwise, stick to stricter deviation limits.
    # Additional check: ensure user is not too far from direct path
    straight_line_distance = haversine_distance(*driver_pos, *office_pos)
    try:
        user_to_path_distance = ROAD_NETWORK._point_to_line_distance(
            (candidate_lat, candidate_lon), driver_pos, office_pos) if ROAD_NETWORK else 0
    except Exception:
        user_to_path_distance = 0  # Fallback if method fails

    # More lenient for straight roads - if detour ratio is very good, allow more deviation
    max_deviation_ratio = 0.4 if detour_ratio <= 1.1 else 0.25  # 40% for straight roads, 25% otherwise
    if user_to_path_distance > straight_line_distance * max_deviation_ratio:
        return False

    return detour_ratio <= _config.get('max_detour_ratio', 1.15)  # Stricter default


def validate_route_path_coherence(route, office_pos):
    """
    Validate that all users in a route form a coherent path
    Remove users that create significant detours
    """
    if len(route['assigned_users']) <= 1:
        return route

    driver_pos = (route['latitude'], route['longitude'])
    office_lat, office_lon = office_pos

    # Sort users by distance from driver (closest first)
    users_with_distance = []
    for user in route['assigned_users']:
        distance = haversine_distance(driver_pos[0], driver_pos[1],
                                      user['lat'], user['lng'])
        users_with_distance.append((distance, user))

    users_with_distance.sort(key=lambda x: x[0])

    # Build coherent path incrementally
    coherent_users = []

    for distance, user in users_with_distance:
        # Ensure user has proper coordinate format for path validation
        user_for_path_check = {
            'latitude': user.get('lat', user.get('latitude', 0)),
            'longitude': user.get('lng', user.get('longitude', 0)),
            'user_id': user.get('user_id', 'unknown')
        }
        if is_user_along_route_path(driver_pos, coherent_users,
                                    user_for_path_check,
                                    (office_lat, office_lon)):
            coherent_users.append(user)
        else:
            print(
                f"    🚫 Removed user {user.get('user_id', 'unknown')} - not along coherent path"
            )

    # Update route with coherent users only
    route['assigned_users'] = coherent_users
    return route


def prepare_user_driver_dataframes(data):
    """Prepare and clean user and driver dataframes with priority system and deduplication"""
    users_list = data.get("users", [])
    drivers_unassigned_list = data.get("driversUnassigned", [])
    drivers_assigned_list = data.get("driversAssigned", [])

    df_users = pd.DataFrame(users_list)

    # Combine driversUnassigned and driversAssigned with priority flags and deduplication
    all_drivers = []
    seen_drivers = {}  # Track highest priority version of each driver

    # Priority 1: driversUnassigned with shift_type_id 1 and 3 (highest priority)
    priority_1_count = 0
    priority_2_count = 0
    for driver in drivers_unassigned_list:
        driver_copy = driver.copy()
        driver_id = str(driver.get('id', ''))
        shift_type_id = int(driver.get('shift_type_id', 2))

        if shift_type_id in [1, 3]:
            priority = 1
            priority_1_count += 1
        else:
            priority = 2
            priority_2_count += 1

        driver_copy['priority'] = priority
        driver_copy['is_assigned'] = False

        # Keep the highest priority version (lower number = higher priority)
        if driver_id not in seen_drivers or seen_drivers[driver_id][
                'priority'] > priority:
            seen_drivers[driver_id] = driver_copy

    # Priority 3: driversAssigned with shift_type_id 1 and 3 (lower priority)
    # Priority 4: driversAssigned with shift_type_id 2 (lowest priority)
    priority_3_count = 0
    priority_4_count = 0
    for driver in drivers_assigned_list:
        driver_copy = driver.copy()
        driver_id = str(driver.get('id', ''))
        shift_type_id = int(driver.get('shift_type_id', 2))

        if shift_type_id in [1, 3]:
            priority = 3
            priority_3_count += 1
        else:
            priority = 4
            priority_4_count += 1

        driver_copy['priority'] = priority
        driver_copy['is_assigned'] = True

        # Only add if we haven't seen this driver or this has higher priority
        if driver_id not in seen_drivers or seen_drivers[driver_id][
                'priority'] > priority:
            seen_drivers[driver_id] = driver_copy

    # Convert to list
    all_drivers = list(seen_drivers.values())

    # Recalculate counts after deduplication
    final_priority_counts = {1: 0, 2: 0, 3: 0, 4: 0}
    for driver in all_drivers:
        final_priority_counts[driver['priority']] += 1

    print(f"🔍 Driver Priority Assignment Debug (after deduplication):")
    print(
        f"   Priority 1 (driversUnassigned ST:1,3): {final_priority_counts[1]}"
    )
    print(
        f"   Priority 2 (driversUnassigned ST:2): {final_priority_counts[2]}")
    print(
        f"   Priority 3 (driversAssigned ST:1,3): {final_priority_counts[3]}")
    print(f"   Priority 4 (driversAssigned ST:2): {final_priority_counts[4]}")

    original_total = priority_1_count + priority_2_count + priority_3_count + priority_4_count
    final_total = sum(final_priority_counts.values())
    if original_total != final_total:
        print(
            f"   📊 Deduplicated: {original_total} → {final_total} drivers ({original_total - final_total} duplicates resolved)"
        )

    df_drivers = pd.DataFrame(all_drivers)

    numeric_cols = [
        "office_distance", "latitude", "longitude", "capacity",
        "shift_type_id", "priority"
    ]
    for col in numeric_cols:
        if col in df_users.columns:
            df_users[col] = pd.to_numeric(df_users[col], errors="coerce")
        if col in df_drivers.columns:
            df_drivers[col] = pd.to_numeric(df_drivers[col], errors="coerce")

    user_df = df_users.rename(columns={"id": "user_id"})[[
        'user_id', 'latitude', 'longitude', 'office_distance', 'shift_type',
        'first_name', 'email'
    ]]
    user_df['user_id'] = user_df['user_id'].astype(str)

    driver_df = df_drivers.rename(columns={"id": "driver_id"})[[
        'driver_id', 'capacity', 'vehicle_id', 'latitude', 'longitude',
        'shift_type_id', 'priority', 'is_assigned'
    ]]
    driver_df['driver_id'] = driver_df['driver_id'].astype(str)

    # Sort drivers by priority (lower number = higher priority)
    driver_df = driver_df.sort_values(['priority', 'capacity'],
                                      ascending=[True, False])

    for col in ['latitude', 'longitude', 'office_distance']:
        user_df[col] = user_df[col].astype(float)
    user_df['shift_type'] = user_df['shift_type'].astype(int)

    return user_df, driver_df


@lru_cache(maxsize=2000)
def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two points - uses road network if available, otherwise haversine"""
    # Handle edge cases
    if lat1 == lat2 and lon1 == lon2:
        return 0.0

    # Use road network distance if available
    if ROAD_NETWORK is not None:
        try:
            return ROAD_NETWORK.get_road_distance(lat1, lon1, lat2, lon2)
        except Exception:
            pass  # Fallback to haversine

    # Fallback to straight-line haversine distance
    R = 6371.0  # Earth radius in kilometers
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # Use abs() to handle potential floating point precision issues
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    a = min(1.0, abs(a))  # Clamp to [0,1] to prevent domain errors

    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return abs(R * c)  # Ensure positive distance


def calculate_bearing(lat1, lon1, lat2, lon2):
    """Calculate bearing from point A to B in degrees"""
    lat1, lat2 = map(math.radians, [lat1, lat2])
    dlon = math.radians(lon2 - lon1)
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(
        lat2) * math.cos(dlon)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def calculate_bearings_and_features(user_df, office_lat, office_lon):
    """Calculate bearings and add geographical features - OFFICE TO USER direction"""
    user_df = user_df.copy()
    user_df[['office_latitude', 'office_longitude']] = office_lat, office_lon

    def calculate_bearing_vectorized(lat1, lon1, lat2, lon2):
        lat1, lat2 = np.radians(lat1), np.radians(lat2)
        dlon = np.radians(lon2 - lon1)
        x = np.sin(dlon) * np.cos(lat2)
        y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(
            dlon)
        return (np.degrees(np.arctan2(x, y)) + 360) % 360

    # Calculate bearing FROM OFFICE TO USER (standardized direction)
    user_df['bearing'] = calculate_bearing_vectorized(office_lat, office_lon,
                                                      user_df['latitude'],
                                                      user_df['longitude'])
    user_df['bearing_sin'] = np.sin(np.radians(user_df['bearing']))
    user_df['bearing_cos'] = np.cos(np.radians(user_df['bearing']))

    return user_df


# STEP 1: STRAIGHTFORWARD GEOGRAPHIC CLUSTERING
def build_road_distance_matrix(users_df, road_network):
    """Build distance matrix using road network distances"""
    n = len(users_df)
    distance_matrix = np.zeros((n, n))

    coordinates = users_df[['latitude', 'longitude']].values

    for i in range(n):
        for j in range(i + 1, n):
            lat1, lon1 = coordinates[i]
            lat2, lon2 = coordinates[j]

            # Use road network distance
            road_dist = road_network.get_road_distance(lat1, lon1, lat2, lon2)
            distance_matrix[i, j] = road_dist
            distance_matrix[j, i] = road_dist

    return distance_matrix


def create_geographic_clusters(user_df, driver_df, config):
    """Create initial geographic clusters using road-network aware DBSCAN"""
    print("🗂️ Step 1: Creating geographic clusters...")

    if len(user_df) < config.get('min_cluster_size', 2):
        print(f"   ⚠️ Too few users ({len(user_df)}) for clustering, creating single cluster")
        user_df['cluster'] = 0
        user_df['geo_cluster'] = 0
        return user_df, {"method": "Single Cluster", "clusters": 1}

    # Use road network for clustering if available
    if ROAD_NETWORK is not None and config.get('use_road_network', False):
        try:
            print("   🛣️ Using road network for clustering...")

            # Build road distance matrix
            road_distance_matrix = build_road_distance_matrix(user_df, ROAD_NETWORK)

            # Convert km to meters for DBSCAN eps parameter
            eps_km = config.get('max_extra_distance_km', 3.0)
            eps_meters = eps_km * 1000  # Convert to meters

            # Use precomputed road distance matrix for clustering
            clustering = DBSCAN(
                eps=eps_meters,  # Distance in meters
                min_samples=max(2, config.get('min_cluster_size', 2)),
                metric='precomputed'
            ).fit(road_distance_matrix)

            user_df['cluster'] = clustering.labels_
            user_df['geo_cluster'] = clustering.labels_

            n_clusters = len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)
            print(f"   ✅ Road network clustering completed: {n_clusters} clusters")

            return user_df, {
                "method": "Road Network DBSCAN",
                "clusters": n_clusters,
                "eps_km": eps_km,
                "road_network_used": True
            }

        except Exception as e:
            print(f"   ⚠️ Road network clustering failed: {e}, falling back to geographic")
            # Continue to fallback geographic clustering

    # Standard geographic clustering (fallback or when road network not available)
    print("   📍 Using geographic coordinates for clustering...")
    coordinates = user_df[['latitude', 'longitude']].values

    # Convert km to degrees (approximate)
    eps_km = config.get('max_extra_distance_km', 3.0)
    eps_degrees = eps_km / 111.32  # Rough conversion: 1 degree ≈ 111.32 km

    clustering = DBSCAN(
        eps=eps_degrees,
        min_samples=max(2, config.get('min_cluster_size', 2))
    ).fit(coordinates)

    user_df['cluster'] = clustering.labels_
    user_df['geo_cluster'] = clustering.labels_
    n_clusters = len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)

    print(f"   ✅ Geographic clustering completed: {n_clusters} clusters")

    return user_df, {
        "method": "Geographic DBSCAN",
        "clusters": n_clusters,
        "eps_km": eps_km,
        "road_network_used": False
    }


# STEP 2: SUB-CLUSTERING BY CAPACITY
def create_capacity_subclusters(user_df, driver_df):
    """
    Step 2: Within each geographic cluster, create sub-clusters based on driver capacity
    """
    print("🔧 Step 2: Creating capacity-based sub-clusters...")

    # Calculate average driver capacity for sizing
    avg_capacity = int(driver_df['capacity'].mean())
    max_capacity = int(driver_df['capacity'].max())

    user_df = user_df.copy()
    user_df['sub_cluster'] = 0
    sub_cluster_counter = 0

    for geo_cluster_id, geo_cluster_users in user_df.groupby('geo_cluster'):
        cluster_size = len(geo_cluster_users)

        if cluster_size <= max_capacity:
            # Small cluster, no need to split
            user_df.loc[geo_cluster_users.index,
                        'sub_cluster'] = sub_cluster_counter
            sub_cluster_counter += 1
        else:
            # Large cluster, split by capacity
            coords = geo_cluster_users[['latitude', 'longitude']].values
            n_subclusters = min(math.ceil(cluster_size / avg_capacity),
                                cluster_size)

            if n_subclusters > 1:
                kmeans = KMeans(n_clusters=n_subclusters, random_state=42)
                sub_labels = kmeans.fit_predict(coords)

                for i, sub_label in enumerate(sub_labels):
                    user_idx = geo_cluster_users.index[i]
                    user_df.loc[
                        user_idx,
                        'sub_cluster'] = sub_cluster_counter + sub_label

                sub_cluster_counter += n_subclusters
            else:
                user_df.loc[geo_cluster_users.index,
                            'sub_cluster'] = sub_cluster_counter
                sub_cluster_counter += 1

    print(
        f"  ✅ Created {user_df['sub_cluster'].nunique()} capacity-based sub-clusters"
    )
    return user_df


# STEP 3: PRIORITY-BASED DRIVER ASSIGNMENT
def assign_drivers_by_priority(user_df, driver_df, office_lat, office_lon):
    """
    Step 3: Assign drivers based on priority and proximity
    """
    print("🚗 Step 3: Assigning drivers by priority...")

    routes = []
    assigned_user_ids = set()
    used_driver_ids = set()

    # Sort drivers by priority
    available_drivers = driver_df.sort_values(['priority', 'capacity'], ascending=[True, False])

    # Process each sub-cluster
    for sub_cluster_id, cluster_users in user_df.groupby('sub_cluster'):
        unassigned_in_cluster = cluster_users[~cluster_users['user_id'].
                                              isin(assigned_user_ids)]

        if unassigned_in_cluster.empty:
            continue

        cluster_center = unassigned_in_cluster[['latitude',
                                                'longitude']].mean()
        cluster_size = len(unassigned_in_cluster)

        # Check bearing coherence with stricter constraints
        bearings = unassigned_in_cluster['bearing'].values
        if len(bearings) > 1:
            max_bearing_diff = max([
                bearing_difference(bearings[i], bearings[j])
                for i in range(len(bearings))
                for j in range(i + 1, len(bearings))
            ])

            # If users are too spread in direction, split the cluster (stricter threshold)
            if max_bearing_diff > STRICT_BEARING_DIFFERENCE:
                print(
                    f"  📍 Splitting sub-cluster {sub_cluster_id} due to bearing spread ({max_bearing_diff:.1f}° > {STRICT_BEARING_DIFFERENCE}°)"
                )
                # Split into smaller, more coherent groups
                coords_with_bearing = np.column_stack([
                    unassigned_in_cluster[['latitude', 'longitude']].values,
                    unassigned_in_cluster[['bearing_sin', 'bearing_cos'
                                           ]].values *
                    2  # Weight bearing more heavily
                ])

                n_splits = min(3, max(2,
                                      len(unassigned_in_cluster) //
                                      2))  # Create smaller groups
                kmeans = KMeans(n_clusters=n_splits, random_state=42)
                split_labels = kmeans.fit_predict(coords_with_bearing)

                # Process each split separately
                for split_id in range(n_splits):
                    split_users = unassigned_in_cluster[split_labels ==
                                                        split_id]
                    if len(split_users) > 0:
                        route = assign_best_driver_to_cluster(split_users,
                                                              available_drivers,
                                                              used_driver_ids,
                                                              office_lat,
                                                              office_lon)
                        if route:
                            # Ensure driver_id is set
                            if 'driver_id' not in route:
                                route['driver_id'] = str(route['driver']['driver_id'])
                            if 'vehicle_type' not in route:
                                route['vehicle_type'] = int(route['driver']['capacity'])

                            # Validate path coherence before adding
                            route = validate_route_path_coherence(
                                route, (office_lat, office_lon))
                            if route[
                                    'assigned_users']:  # Only add if users remain after validation
                                routes.append(route)
                                assigned_user_ids.update(
                                    u['user_id'] for u in route['assigned_users'])
                continue

        # Assign best driver to this cluster
        route = assign_best_driver_to_cluster(unassigned_in_cluster,
                                              available_drivers,
                                              used_driver_ids, office_lat,
                                              office_lon)

        if route:
            # Ensure driver_id is set
            if 'driver_id' not in route:
                route['driver_id'] = str(route['driver']['driver_id'])
            if 'vehicle_type' not in route:
                route['vehicle_type'] = int(route['driver']['capacity'])

            # Validate path coherence before adding
            route = validate_route_path_coherence(route,
                                                  (office_lat, office_lon))
            if route[
                    'assigned_users']:  # Only add if users remain after validation
                routes.append(route)
                assigned_user_ids.update(u['user_id']
                                         for u in route['assigned_users'])

    print(
        f"  ✅ Created {len(routes)} initial routes with priority-based assignment"
    )
    return routes, assigned_user_ids


def assign_best_driver_to_cluster(cluster_users, available_drivers,
                                  used_driver_ids, office_lat, office_lon):
    """Find and assign the best available driver to a cluster of users with road coherence validation"""
    cluster_center = cluster_users[['latitude', 'longitude']].mean()
    cluster_size = len(cluster_users)

    # Check if cluster needs splitting for better road coherence
    if len(cluster_users) > 1 and ROAD_NETWORK is not None:
        try:
            # Test road coherence of the cluster
            user_positions = [(row['latitude'], row['longitude'])
                              for _, row in cluster_users.iterrows()]

            # If cluster spans multiple disconnected road areas, split it
            if should_split_cluster_for_roads(user_positions, office_lat,
                                              office_lon):
                print(f"      🛣️ Splitting cluster for better road coherence")
                return split_cluster_by_road_network(cluster_users,
                                                     available_drivers,
                                                     used_driver_ids,
                                                     office_lat, office_lon)
        except Exception as e:
            print(f"      ⚠️ Road coherence check failed: {e}")

    best_driver = None
    min_cost = float('inf')

    for _, driver in available_drivers.iterrows():
        if driver['driver_id'] in used_driver_ids:
            continue

        # Check capacity
        if driver['capacity'] < cluster_size:
            continue  # Skip drivers without enough capacity

        # Calculate cost: distance + priority penalty + road coherence penalty
        driver_pos = (driver['latitude'], driver['longitude'])
        distance = haversine_distance(driver_pos[0], driver_pos[1],
                                      cluster_center['latitude'],
                                      cluster_center['longitude'])

        # Priority penalty (lower priority = higher penalty)
        priority_penalty = driver[
            'priority'] * 2.0  # 2km penalty per priority level

        # Utilization bonus (prefer fuller vehicles)
        utilization = cluster_size / driver['capacity']
        utilization_bonus = utilization * 3.0  # 3km bonus for full utilization

        # Road coherence penalty
        road_penalty = 0.0
        if ROAD_NETWORK is not None:
            try:
                user_positions = [(row['latitude'], row['longitude'])
                                  for _, row in cluster_users.iterrows()]
                coherence_score = ROAD_NETWORK.get_route_coherence_score(
                    driver_pos, user_positions, (office_lat, office_lon))
                # Penalty for poor road coherence (0-5km penalty)
                road_penalty = (1.0 - coherence_score) * 5.0
            except Exception:
                pass

        total_cost = distance + priority_penalty - utilization_bonus + road_penalty

        if total_cost < min_cost:
            min_cost = total_cost
            best_driver = driver

    if best_driver is not None:
        used_driver_ids.add(best_driver['driver_id'])

        # Create route with intelligent user selection
        route = {
            'driver': best_driver.to_dict(),
            'latitude': float(best_driver['latitude']),
            'longitude': float(best_driver['longitude']),
            'assigned_users': []
        }

        # Intelligent user assignment considering road paths
        users_to_assign = select_coherent_users_for_driver(cluster_users, best_driver, office_lat,
                                                          office_lon)

        for _, user in users_to_assign.iterrows():
            user_data = {
                'user_id': str(user['user_id']),
                'lat': float(user['latitude']),
                'lng': float(user['longitude']),
                'office_distance': float(user.get('office_distance', 0))
            }

            if pd.notna(user.get('first_name')):
                user_data['first_name'] = str(user['first_name'])
            if pd.notna(user.get('email')):
                user_data['email'] = str(user['email'])

            route['assigned_users'].append(user_data)

        # Calculate route metrics
        if route['assigned_users']:
            lats = [u['lat'] for u in route['assigned_users']]
            lngs = [u['lng'] for u in route['assigned_users']]
            route['centroid'] = [np.mean(lats), np.mean(lngs)]
            route['utilization'] = len(
                route['assigned_users']) / best_driver['capacity']
            route['bearing'] = calculate_bearing(route['centroid'][0],
                                                 route['centroid'][1], OFFICE_LAT,
                                                 OFFICE_LON)

        return route

    return None


def should_split_cluster_for_roads(user_positions, office_lat, office_lon):
    """Check if a cluster should be split based on road network coherence"""
    if len(user_positions) < 3:
        return False

    try:
        # Calculate pairwise road distances vs straight-line distances
        total_road_detour = 0
        comparisons = 0

        for i in range(len(user_positions)):
            for j in range(i + 1, len(user_positions)):
                pos1, pos2 = user_positions[i], user_positions[j]
                road_dist = ROAD_NETWORK.get_road_distance(
                    pos1[0], pos1[1], pos2[0], pos2[1])
                straight_dist = ROAD_NETWORK._haversine_distance(
                    pos1[0], pos1[1], pos2[0], pos2[1])

                if straight_dist > 0:
                    detour_ratio = road_dist / straight_dist
                    total_road_detour += detour_ratio
                    comparisons += 1

        if comparisons > 0:
            avg_detour = total_road_detour / comparisons
            # Split if average detour is > 1.5 (roads are 50% longer than straight line)
            return avg_detour > 1.5

    except Exception:
        pass

    return False


def split_cluster_by_road_network(cluster_users, available_drivers,
                                  used_driver_ids, office_lat, office_lon):
    """Split a cluster into more coherent sub-clusters based on road network"""
    if len(cluster_users) < 2:
        return None

    # For now, simple split by distance - can be enhanced with more sophisticated road-based clustering
    cluster_center = cluster_users[['latitude', 'longitude']].mean()

    # Split users into two groups: closer and farther from center
    distances = []
    for _, user in cluster_users.iterrows():
        dist = haversine_distance(user['latitude'], user['longitude'],
                                  cluster_center['latitude'],
                                  cluster_center['longitude'])
        distances.append((dist, user))

    distances.sort(key=lambda x: x[0])
    mid_point = len(distances) // 2

    # Create two sub-clusters
    closer_users = pd.DataFrame([user for _, user in distances[:mid_point]])
    farther_users = pd.DataFrame([user for _, user in distances[mid_point:]])

    # Try to assign drivers to both sub-clusters
    routes = []
    for sub_cluster in [closer_users, farther_users]:
        if len(sub_cluster) > 0:
            route = assign_best_driver_to_cluster(sub_cluster,
                                                  available_drivers,
                                                  used_driver_ids, office_lat,
                                                  office_lon)
            if route:
                routes.append(route)

    return routes[0] if routes else None


def select_coherent_users_for_driver(cluster_users, driver, office_lat,
                                     office_lon):
    """Select the most coherent subset of users for a driver based on road paths"""
    if len(cluster_users) <= driver['capacity']:
        return cluster_users

    driver_pos = (driver['latitude'], driver['longitude'])

    # Sort users by road-based suitability
    user_scores = []
    for _, user in cluster_users.iterrows():
        user_pos = (user['latitude'], user['longitude'])

        # Base score: distance from driver
        distance = haversine_distance(driver_pos[0], driver_pos[1],
                                      user_pos[0], user_pos[1])
        score = distance

        # Road coherence bonus
        if ROAD_NETWORK is not None:
            try:
                coherence = ROAD_NETWORK.get_route_coherence_score(
                    driver_pos, [user_pos], (office_lat, office_lon))
                score -= coherence * 3.0  # 3km bonus for good coherence
            except Exception:
                pass

        user_scores.append((score, user))

    # Sort by score and take the best ones up to capacity
    user_scores.sort(key=lambda x: x[0])
    selected_users = [user for _, user in user_scores[:driver['capacity']]]

    return pd.DataFrame(selected_users)


# STEP 4: LOCAL OPTIMIZATION
def local_optimization(routes):
    """
    Enhanced Step 4: Local optimization with road network awareness and multiple improvement strategies
    """
    print("🔧 Step 4: Enhanced local optimization...")

    improved = True
    iterations = 0
    total_improvements = 0

    while improved and iterations < MAX_SWAP_ITERATIONS * 2:  # More iterations
        improved = False
        iterations += 1
        iteration_improvements = 0

        # Strategy 1: Optimize individual route sequences using road network
        for i, route in enumerate(routes):
            if len(route['assigned_users']) < 2:
                continue

            # Calculate current route cost
            current_cost = calculate_route_total_cost(route)

            # Optimize user sequence within route
            old_sequence = route['assigned_users'].copy()
            route = optimize_route_sequence(route)

            # Check if optimization improved the route
            new_cost = calculate_route_total_cost(route)
            if new_cost < current_cost - 0.1:  # Improved by at least 100m
                iteration_improvements += 1
                improved = True
                driver_id = route.get('driver_id', route.get('driver', {}).get('driver_id', 'unknown'))
                print(f"      ✅ Route {driver_id} sequence optimized: {current_cost:.1f}km → {new_cost:.1f}km")

        # Strategy 2: Smart user swapping between nearby routes with road network validation
        for i, route_a in enumerate(routes):
            for j, route_b in enumerate(routes[i + 1:], start=i + 1):
                if not route_a['assigned_users'] or not route_b['assigned_users']:
                    continue

                # Check if routes are nearby (extended range for more opportunities)
                center_a = route_a.get('centroid', [route_a['latitude'], route_a['longitude']])
                center_b = route_b.get('centroid', [route_b['latitude'], route_b['longitude']])
                distance = haversine_distance(center_a[0], center_a[1], center_b[0], center_b[1])

                if distance <= MERGE_DISTANCE_KM * 1.5:  # Extended range
                    # Try intelligent user swapping
                    if try_intelligent_user_swap(route_a, route_b):
                        iteration_improvements += 1
                        improved = True

        # Strategy 3: User redistribution between underutilized and overutilized routes
        underutilized_routes = [r for r in routes if r.get('utilization', 0) < 0.6]
        overutilized_routes = [r for r in routes if r.get('utilization', 0) > 0.9]

        for under_route in underutilized_routes:
            for over_route in overutilized_routes:
                if not over_route['assigned_users']:
                    continue

                # Check if routes are geographically compatible
                center_under = under_route.get('centroid', [under_route['latitude'], under_route['longitude']])
                center_over = over_route.get('centroid', [over_route['latitude'], over_route['longitude']])
                distance = haversine_distance(center_under[0], center_under[1], center_over[0], center_over[1])

                if distance <= MERGE_DISTANCE_KM * 2:  # Wider range for rebalancing
                    if try_user_redistribution(under_route, over_route):
                        iteration_improvements += 1
                        improved = True

        # Strategy 4: Cross-route optimization for better road network utilization
        if ROAD_NETWORK is not None:
            for route in routes:
                if len(route['assigned_users']) > 2:
                    if optimize_route_with_road_network(route):
                        iteration_improvements += 1
                        improved = True

        total_improvements += iteration_improvements
        print(f"    📊 Iteration {iterations}: {iteration_improvements} improvements made")

    print(f"  ✅ Enhanced local optimization completed in {iterations} iterations with {total_improvements} total improvements")
    return routes


def optimize_route_sequence(route):
    """Optimize pickup sequence within a route to minimize zig-zag patterns"""
    if len(route['assigned_users']) <= 2:
        return route

    users = route['assigned_users'].copy()
    driver_pos = (route['latitude'], route['longitude'])
    office_pos = (OFFICE_LAT, OFFICE_LON)

    # Use road network optimization if available
    if ROAD_NETWORK is not None and len(users) > 1:
        try:
            user_positions = [(u['lat'], u['lng']) for u in users]
            optimal_indices = ROAD_NETWORK.get_optimal_pickup_sequence(
                driver_pos, user_positions, office_pos)

            # Reorder users based on optimal sequence
            optimized_sequence = [users[i] for i in optimal_indices]
            route['assigned_users'] = optimized_sequence

            print(
                f"      🛣️ Road-optimized pickup sequence for route {route.get('driver_id', 'unknown')}"
            )
            return route
        except Exception as e:
            print(f"      ⚠️ Road optimization failed: {e}, using fallback")

    # Fallback: Progressive nearest neighbor with office direction bias
    optimized_sequence = []
    remaining_users = users.copy()
    current_pos = driver_pos

    while remaining_users:
        best_user = None
        best_score = float('inf')

        for user in remaining_users:
            user_pos = (user['lat'], user['lng'])

            # Distance from current position
            distance = haversine_distance(current_pos[0], current_pos[1],
                                          user_pos[0], user_pos[1])

            # Progress toward office (bonus for users closer to office)
            current_to_office = haversine_distance(current_pos[0],
                                                   current_pos[1],
                                                   office_pos[0],
                                                   office_pos[1])
            user_to_office = haversine_distance(user_pos[0], user_pos[1],
                                                office_pos[0],
                                                office_pos[1])
            progress_bonus = max(0, current_to_office - user_to_office) * 0.5

            # Directional consistency bonus
            if len(optimized_sequence) > 0:
                prev_user_pos = (optimized_sequence[-1]['lat'],
                                 optimized_sequence[-1]['lng'])
                prev_bearing = calculate_bearing(current_pos[0],
                                                 current_pos[1],
                                                 prev_user_pos[0],
                                                 prev_user_pos[1])
                current_bearing = calculate_bearing(current_pos[0],
                                                    current_pos[1],
                                                    user_pos[0], user_pos[1])
                bearing_diff = abs(prev_bearing - current_bearing)
                if bearing_diff > 180:
                    bearing_diff = 360 - bearing_diff

                direction_penalty = bearing_diff / 180.0 * 2.0  # 0-2 penalty
            else:
                direction_penalty = 0

            total_score = distance - progress_bonus + direction_penalty

            if total_score < best_score:
                best_score = total_score
                best_user = user

        if best_user:
            optimized_sequence.append(best_user)
            remaining_users.remove(best_user)
            current_pos = (best_user['lat'], best_user['lng'])

    route['assigned_users'] = optimized_sequence
    return route


def try_user_swap(route_a, route_b):
    """Try swapping users between two routes if it improves overall efficiency"""
    if len(route_a['assigned_users']) == 0 or len(
            route_b['assigned_users']) == 0:
        return False

    # Simple swap: try swapping one user from each route
    for user_a in route_a['assigned_users']:
        for user_b in route_b['assigned_users']:
            # Check if swap improves total distance
            current_cost_a = calculate_route_cost(route_a)
            current_cost_b = calculate_route_cost(route_b)
            current_total = current_cost_a + current_cost_b

            # Perform temporary swap
            route_a['assigned_users'].remove(user_a)
            route_b['assigned_users'].remove(user_b)
            route_a['assigned_users'].append(user_b)
            route_b['assigned_users'].append(user_a)

            new_cost_a = calculate_route_cost(route_a)
            new_cost_b = calculate_route_cost(route_b)
            new_total = new_cost_a + new_cost_b

            if new_total < current_total - SWAP_IMPROVEMENT_THRESHOLD:
                # Keep the swap
                update_route_metrics(route_a)
                update_route_metrics(route_b)
                return True
            else:
                # Revert the swap
                route_a['assigned_users'].remove(user_b)
                route_b['assigned_users'].remove(user_a)
                route_a['assigned_users'].append(user_a)
                route_b['assigned_users'].append(user_b)

    return False


def calculate_route_cost(route):
    """Calculate cost of a route based on distances"""
    if not route['assigned_users']:
        return 0

    total_cost = 0
    driver_pos = (route['latitude'], route['longitude'])

    # Cost from driver to users
    for user in route['assigned_users']:
        cost = haversine_distance(driver_pos[0], driver_pos[1], user['lat'],
                                  user['lng'])
        total_cost += cost

    return total_cost


def update_route_metrics(route):
    """Update route metrics after modifications"""
    if route['assigned_users']:
        lats = [u['lat'] for u in route['assigned_users']]
        lngs = [u['lng'] for u in route['assigned_users']]
        route['centroid'] = [np.mean(lats), np.mean(lngs)]
        route['utilization'] = len(
            route['assigned_users']) / route['driver']['capacity']
        route['bearing'] = calculate_bearing(route['centroid'][0],
                                             route['centroid'][1], OFFICE_LAT,
                                             OFFICE_LON)
    else:
        route['centroid'] = [route['latitude'], route['longitude']]
        route['utilization'] = 0
        route['bearing'] = 0


def calculate_route_total_cost(route):
    """Calculate total cost of a route including driver to users and users to office"""
    if not route['assigned_users']:
        return 0

    total_cost = 0
    driver_pos = (route['latitude'], route['longitude'])

    # Cost from driver to first user
    first_user = route['assigned_users'][0]
    total_cost += haversine_distance(driver_pos[0], driver_pos[1], first_user['lat'],
                                     first_user['lng'])

    # Cost between consecutive users
    for i in range(len(route['assigned_users']) - 1):
        user_a = route['assigned_users'][i]
        user_b = route['assigned_users'][i + 1]
        total_cost += haversine_distance(user_a['lat'], user_a['lng'], user_b['lat'], user_b['lng'])

    # Cost from last user to office
    last_user = route['assigned_users'][-1]
    total_cost += haversine_distance(last_user['lat'], last_user['lng'], OFFICE_LAT, OFFICE_LON)

    return total_cost


def try_intelligent_user_swap(route_a, route_b):
    """Try swapping users between two routes with intelligent criteria"""
    if len(route_a['assigned_users']) == 0 or len(route_b['assigned_users']) == 0:
        return False

    best_improvement = 0
    best_swap = None

    # Calculate current costs
    current_cost_a = calculate_route_total_cost(route_a)
    current_cost_b = calculate_route_total_cost(route_b)
    current_total = current_cost_a + current_cost_b

    # Try swapping each user from route_a with each user from route_b
    for i, user_a in enumerate(route_a['assigned_users']):
        for j, user_b in enumerate(route_b['assigned_users']):
            # Temporarily swap users
            route_a['assigned_users'][i] = user_b
            route_b['assigned_users'][j] = user_a

            # Check if swap maintains route coherence
            driver_pos_a = (route_a['latitude'], route_a['longitude'])
            driver_pos_b = (route_b['latitude'], route_b['longitude'])

            # Basic coherence check - user should be closer to new driver
            dist_a_new = haversine_distance(driver_pos_a[0], driver_pos_a[1], user_b['lat'], user_b['lng'])
            dist_a_old = haversine_distance(driver_pos_a[0], driver_pos_a[1], user_a['lat'], user_a['lng'])
            dist_b_new = haversine_distance(driver_pos_b[0], driver_pos_b[1], user_a['lat'], user_a['lng'])
            dist_b_old = haversine_distance(driver_pos_b[0], driver_pos_b[1], user_b['lat'], user_b['lng'])

            # Calculate new costs
            new_cost_a = calculate_route_total_cost(route_a)
            new_cost_b = calculate_route_total_cost(route_b)
            new_total = new_cost_a + new_cost_b

            improvement = current_total - new_total

            # Additional road network validation if available
            coherence_bonus = 0
            if ROAD_NETWORK is not None:
                try:
                    positions_a = [(u['lat'], u['lng']) for u in route_a['assigned_users']]
                    positions_b = [(u['lat'], u['lng']) for u in route_b['assigned_users']]

                    coherence_a = ROAD_NETWORK.get_route_coherence_score(driver_pos_a, positions_a, (OFFICE_LAT, OFFICE_LON))
                    coherence_b = ROAD_NETWORK.get_route_coherence_score(driver_pos_b, positions_b, (OFFICE_LAT, OFFICE_LON))

                    if coherence_a > 0.7 and coherence_b > 0.7:  # Both routes maintain good coherence
                        coherence_bonus = (coherence_a + coherence_b) * 0.5  # Up to 1km bonus
                except Exception:
                    pass

            total_improvement = improvement + coherence_bonus

            if total_improvement > best_improvement and total_improvement > SWAP_IMPROVEMENT_THRESHOLD:
                best_improvement = total_improvement
                best_swap = (i, j, user_a, user_b)

            # Revert swap for next iteration
            route_a['assigned_users'][i] = user_a
            route_b['assigned_users'][j] = user_b

    # Apply best swap if found
    if best_swap is not None:
        i, j, user_a, user_b = best_swap
        route_a['assigned_users'][i] = user_b
        route_b['assigned_users'][j] = user_a

        # Optimize sequences after swap
        route_a = optimize_route_sequence(route_a)
        route_b = optimize_route_sequence(route_b)

        update_route_metrics(route_a)
        update_route_metrics(route_b)

        print(f"      🔄 Intelligent swap between routes {route_a['driver_id']} and {route_b['driver_id']} (improvement: {best_improvement:.1f}km)")
        return True

    return False


def try_user_redistribution(under_route, over_route):
    """Try redistributing users from overutilized to underutilized routes"""
    if not over_route['assigned_users'] or len(under_route['assigned_users']) >= under_route['driver']['capacity']:
        return False

    available_capacity = under_route['driver']['capacity'] - len(under_route['assigned_users'])
    if available_capacity <= 0:
        return False

    driver_pos_under = (under_route['latitude'], under_route['longitude'])

    # Find users in over_route that could fit better in under_route
    redistribution_candidates = []

    for user in over_route['assigned_users']:
        # Check distance to underutilized driver
        distance_to_under = haversine_distance(
            driver_pos_under[0], driver_pos_under[1], user['lat'], user['lng']
        )

        if distance_to_under <= MAX_FILL_DISTANCE_KM * 1.2:  # Slightly extended range
            # Check if user maintains coherence with under_route
            temp_user = {
                'latitude': user['lat'],
                'longitude': user['lng'],
                'user_id': user['user_id']
            }

            is_coherent = False
            if under_route['assigned_users']:
                is_coherent = is_user_along_route_path(
                    driver_pos_under, under_route['assigned_users'], temp_user, (OFFICE_LAT, OFFICE_LON)
                )
            else:
                is_coherent = True  # Empty route can take any user

            if is_coherent:
                redistribution_candidates.append((distance_to_under, user))

    # Sort by distance and redistribute up to available capacity
    redistribution_candidates.sort(key=lambda x: x[0])
    users_to_move = min(available_capacity, len(redistribution_candidates))

    if users_to_move > 0:
        moved_users = []
        for i in range(users_to_move):
            distance, user = redistribution_candidates[i]
            moved_users.append(user)
            under_route['assigned_users'].append(user)
            over_route['assigned_users'].remove(user)

        # Optimize both routes after redistribution
        under_route = optimize_route_sequence(under_route)
        over_route = optimize_route_sequence(over_route)

        update_route_metrics(under_route)
        update_route_metrics(over_route)

        print(f"      ⚖️ Redistributed {users_to_move} users from route {over_route['driver_id']} to {under_route['driver_id']}")
        return True

    return False


def optimize_route_with_road_network(route):
    """Use road network to optimize route for better coherence"""
    if ROAD_NETWORK is None or len(route['assigned_users']) < 2:
        return False

    driver_pos = (route['latitude'], route['longitude'])
    user_positions = [(u['lat'], u['lng']) for u in route['assigned_users']]

    # Calculate current coherence
    try:
        current_coherence = ROAD_NETWORK.get_route_coherence_score(
            driver_pos, user_positions, (OFFICE_LAT, OFFICE_LON)
        )

        # Get optimal sequence from road network
        optimal_sequence = ROAD_NETWORK.get_optimal_pickup_sequence(
            driver_pos, user_positions, (OFFICE_LAT, OFFICE_LON)
        )

        # Apply optimal sequence
        optimized_users = [route['assigned_users'][i] for i in optimal_sequence]
        route['assigned_users'] = optimized_users

        # Check new coherence
        new_user_positions = [(u['lat'], u['lng']) for u in route['assigned_users']]
        new_coherence = ROAD_NETWORK.get_route_coherence_score(
            driver_pos, new_user_positions, (OFFICE_LAT, OFFICE_LON)
        )

        if new_coherence > current_coherence + 0.05:  # At least 5% improvement
            print(f"      🛣️ Road network optimization improved route {route['driver_id']} coherence: {current_coherence:.2f} → {new_coherence:.2f}")
            return True
        else:
            # Revert if no significant improvement
            route['assigned_users'] = [route['assigned_users'][optimal_sequence.index(i)] for i in range(len(optimal_sequence))]

    except Exception as e:
        print(f"      ⚠️ Road network optimization failed for route {route['driver_id']}: {e}")

    return False


# STEP 5: GLOBAL OPTIMIZATION
def global_optimization(routes, user_df, assigned_user_ids, driver_df):
    """
    Global optimization to merge under-utilized routes with directional filtering
    """
    if len(routes) <= 1:
        return routes, []

    print("🌍 Step 5: Running global optimization...")

    # Calculate utilization for each route
    for route in routes:
        route['utilization'] = len(route['assigned_users']) / route['driver']['capacity']

    # Sort routes by utilization (lowest first for merging candidates)
    routes.sort(key=lambda x: x['utilization'])

    merged_routes = []
    skip_indices = set()

    for i, route1 in enumerate(routes):
        if i in skip_indices:
            continue

        # Try to merge with other low-utilization routes
        for j, route2 in enumerate(routes[i+1:], i+1):
            if j in skip_indices:
                continue

            # Check if routes can be merged
            total_users = len(route1['assigned_users']) + len(route2['assigned_users'])
            max_capacity = max(route1['driver']['capacity'], route2['driver']['capacity'])

            if total_users <= max_capacity:
                # Check directional compatibility before merging
                route1_bearing = calculate_average_route_bearing(route1)
                route2_bearing = calculate_average_route_bearing(route2)

                bearing_diff = abs(route1_bearing - route2_bearing)
                if bearing_diff > 180:
                    bearing_diff = 360 - bearing_diff

                # Only merge if routes are roughly in the same direction
                if bearing_diff <= 30:  # Within 30 degrees
                    # Choose driver with higher capacity or priority
                    if route1['driver']['capacity'] >= route2['driver']['capacity']:
                        primary_driver = route1['driver']
                        secondary_driver = route2['driver']
                    else:
                        primary_driver = route2['driver']
                        secondary_driver = route1['driver']

                    # Merge users
                    merged_users = route1['assigned_users'] + route2['assigned_users']

                    # Create merged route
                    merged_route = {
                        'driver': primary_driver,
                        'driver_id': str(primary_driver['driver_id']),
                        'latitude': float(primary_driver['latitude']),
                        'longitude': float(primary_driver['longitude']),
                        'assigned_users': merged_users,
                        'utilization': total_users / primary_driver['capacity'],
                        'vehicle_type': int(primary_driver['capacity'])
                    }

                    merged_routes.append(merged_route)
                    skip_indices.add(i)
                    skip_indices.add(j)

                    print(f"   ✅ Merged directionally compatible routes ({len(route1['assigned_users'])} + {len(route2['assigned_users'])} users)")
                    break
                else:
                    print(f"   ⚠️ Skipped merge due to directional incompatibility ({bearing_diff:.1f}° difference)")

        if i not in skip_indices:
            merged_routes.append(route1)

    print(f"   📊 Optimization complete: {len(routes)} → {len(merged_routes)} routes")

    return merged_routes, []

def calculate_average_route_bearing(route):
    """Calculate the average bearing direction of a route"""
    if not route['assigned_users']:
        return 0.0

    driver_pos = (route['driver']['latitude'], route['driver']['longitude'])
    office_pos = (OFFICE_LAT, OFFICE_LON)

    # Calculate bearing from driver to office as main direction
    driver_lat, driver_lon = driver_pos
    office_lat, office_lon = office_pos

    import math
    lat1, lat2 = map(math.radians, [driver_lat, office_lat])
    dlon = math.radians(office_lon - driver_lon)

    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)

    bearing = math.degrees(math.atan2(x, y))
    return (bearing + 360) % 360


# STEP 5.5: Route quality validation and reallocation
def validate_and_reallocate_poor_routes(routes, unassigned_users, driver_df):
    """Identify poor quality routes and reassign users to better drivers"""
    good_routes = []
    users_to_reassign = []
    freed_drivers = []

    for route in routes:
        is_poor_route = False
        poor_reasons = []

        if not route['assigned_users']:
            continue

        driver_pos = (route['latitude'], route['longitude'])

        # Check 1: Driver too far from users
        avg_distance_to_users = 0
        max_distance_to_user = 0
        for user in route['assigned_users']:
            distance = haversine_distance(
                driver_pos[0], driver_pos[1], user['lat'], user['lng']
            )
            avg_distance_to_users += distance
            max_distance_to_user = max(max_distance_to_user, distance)

        avg_distance_to_users /= len(route['assigned_users'])

        # Poor route indicators
        if avg_distance_to_users > MAX_FILL_DISTANCE_KM * 2:
            is_poor_route = True
            poor_reasons.append(f"avg distance {avg_distance_to_users:.1f}km > {MAX_FILL_DISTANCE_KM * 2}km")

        if max_distance_to_user > MAX_FILL_DISTANCE_KM * 3:
            is_poor_route = True
            poor_reasons.append(f"max distance {max_distance_to_user:.1f}km > {MAX_FILL_DISTANCE_KM * 3}km")

        # Check 2: Poor utilization with excessive distance
        utilization = len(route['assigned_users']) / route['driver']['capacity']
        if utilization < 0.5 and avg_distance_to_users > MAX_FILL_DISTANCE_KM:
            is_poor_route = True
            poor_reasons.append(f"low utilization {utilization:.1%} with high distance {avg_distance_to_users:.1f}km")

        # Check 3: Route coherence issues
        if ROAD_NETWORK is not None and len(route['assigned_users']) > 1:
            try:
                user_positions = [(u['lat'], u['lng']) for u in route['assigned_users']]
                coherence_score = ROAD_NETWORK.get_route_coherence_score(
                    driver_pos, user_positions, (OFFICE_LAT, OFFICE_LON))
                if coherence_score < 0.6:  # Poor coherence
                    is_poor_route = True
                    poor_reasons.append(f"poor road coherence {coherence_score:.2f}")
            except Exception as e:
                print(f"      ⚠️ Road coherence check failed: {e}")
                pass

        if is_poor_route:
            print(f"    🚫 Rejecting poor route {route['driver_id']}: {', '.join(poor_reasons)}")
            # Add users back to reassignment pool
            for user in route['assigned_users']:
                users_to_reassign.append({
                    'user_id': user['user_id'],
                    'latitude': user['lat'],
                    'longitude': user['lng'],
                    'office_distance': user.get('office_distance', 0),
                    'first_name': user.get('first_name', ''),
                    'email': user.get('email', '')
                })

            # Mark driver as available
            freed_drivers.append({
                'driver_id': route['driver_id'],
                'capacity': route['driver']['capacity'],
                'vehicle_id': route['driver'].get('vehicle_id', ''),
                'latitude': route['latitude'],
                'longitude': route['longitude']
            })
        else:
            good_routes.append(route)

    # If we have users to reassigned and freed drivers, try better assignment
    if users_to_reassign and freed_drivers:
        print(f"    🔄 Reassigning {len(users_to_reassign)} users using {len(freed_drivers)} freed drivers...")

        # Get all available drivers (freed + originally unassigned)
        assigned_driver_ids = {route['driver']['driver_id'] for route in good_routes}
        available_drivers = driver_df[~driver_df['driver_id'].isin(assigned_driver_ids)]

        # Create user DataFrame for reassignment
        reassign_df = pd.DataFrame(users_to_reassign)
        if not reassign_df.empty:
            reassign_df = calculate_bearings_and_features(reassign_df, OFFICE_LAT, OFFICE_LON)

        # Try to create better routes
        assigned_user_ids = set()
        for route in good_routes:
            for user in route['assigned_users']:
                assigned_user_ids.add(user['user_id'])

        # Use comprehensive fallback assignment with stricter criteria
        better_routes, better_assigned_ids = comprehensive_fallback_assignment(
            good_routes.copy(), reassign_df, available_drivers, assigned_user_ids, assigned_driver_ids
        )

        # Update unassigned users
        still_unassigned = []
        for user in users_to_reassign:
            if user['user_id'] not in better_assigned_ids:
                still_unassigned.append(user)

        print(f"    ✅ Reassignment complete: {len(users_to_reassign) - len(still_unassigned)} users reassigned, {len(still_unassigned)} remain unassigned")

        return better_routes, still_unassigned

    # Convert unassigned_users back to list format if needed
    if hasattr(unassigned_users, 'to_dict'):
        unassigned_list = unassigned_users.to_dict('records')
    else:
        unassigned_list = unassigned_users

    # Add users from poor routes to unassigned
    unassigned_list.extend(users_to_reassign)

    return good_routes, unassigned_list


def comprehensive_fallback_assignment(existing_routes, unassigned_df, available_drivers, assigned_user_ids, assigned_driver_ids):
    """
    Comprehensive fallback assignment for users that couldn't be assigned initially
    """
    print("🔄 Running comprehensive fallback assignment...")

    new_routes = existing_routes.copy()
    newly_assigned_user_ids = assigned_user_ids.copy()

    if unassigned_df.empty or available_drivers.empty:
        return new_routes, newly_assigned_user_ids

    # Filter available drivers
    truly_available = available_drivers[~available_drivers['driver_id'].isin(assigned_driver_ids)]

    if truly_available.empty:
        print("   ⚠️ No available drivers for fallback assignment")
        return new_routes, newly_assigned_user_ids

    # Group nearby unassigned users
    unassigned_coords = unassigned_df[['latitude', 'longitude']].values

    if len(unassigned_coords) > 1:
        # Use DBSCAN to group nearby unassigned users
        clustering = DBSCAN(eps=0.03, min_samples=2).fit(unassigned_coords)  # ~3.3km
        unassigned_df['fallback_cluster'] = clustering.labels_
    else:
        unassigned_df['fallback_cluster'] = 0

    # Process each cluster of unassigned users
    for cluster_id in unassigned_df['fallback_cluster'].unique():
        cluster_users = unassigned_df[unassigned_df['fallback_cluster'] == cluster_id]

        if cluster_users.empty:
            continue

        # Find best available driver for this cluster
        cluster_center = cluster_users[['latitude', 'longitude']].mean()

        best_driver = None
        min_cost = float('inf')

        for _, driver in truly_available.iterrows():
            if driver['driver_id'] in assigned_driver_ids:
                continue

            # Calculate cost
            distance = haversine_distance(
                driver['latitude'], driver['longitude'],
                cluster_center['latitude'], cluster_center['longitude']
            )

            # Capacity check
            if driver['capacity'] < len(cluster_users):
                continue

            # Priority penalty
            priority_penalty = driver.get('priority', 1) * 2.0
            cost = distance + priority_penalty

            if cost < min_cost:
                min_cost = cost
                best_driver = driver

        if best_driver is not None:
            # Create new route
            route_users = []
            for _, user in cluster_users.iterrows():
                user_data = {
                    'user_id': str(user['user_id']),
                    'lat': float(user['latitude']),
                    'lng': float(user['longitude']),
                    'office_distance': float(user.get('office_distance', 0))
                }

                if pd.notna(user.get('first_name')):
                    user_data['first_name'] = str(user['first_name'])
                if pd.notna(user.get('email')):
                    user_data['email'] = str(user['email'])

                route_users.append(user_data)
                newly_assigned_user_ids.add(user['user_id'])

            new_route = {
                'driver': best_driver.to_dict(),
                'latitude': float(best_driver['latitude']),
                'longitude': float(best_driver['longitude']),
                'assigned_users': route_users,
                'utilization': len(route_users) / best_driver['capacity'],
                'driver_id': str(best_driver['driver_id']),
                'vehicle_type': int(best_driver['capacity'])
            }

            # Optimize the route sequence
            new_route = optimize_route_sequence(new_route)
            update_route_metrics(new_route)
            new_routes.append(new_route)
            assigned_driver_ids.add(best_driver['driver_id'])

            print(f"   ✅ Created fallback route with {len(route_users)} users")

    return new_routes, newly_assigned_user_ids


def calculate_route_distance_increase(route, user, config):
    """Calculate how much the route distance would increase by adding this user"""
    if not route['assigned_users']:
        # Empty route - just distance from driver to user + user to office
        driver_pos = (route['driver']['latitude'], route['driver']['longitude'])
        user_pos = (user['latitude'], user['longitude'])
        office_pos = (config['OFFICE_LAT'], config['OFFICE_LON'])

        return (haversine_distance(driver_pos[0], driver_pos[1], user_pos[0], user_pos[1]) +
                haversine_distance(user_pos[0], user_pos[1], office_pos[0], office_pos[1]))

    # Calculate current route distance
    current_distance = calculate_route_total_cost(route)

    # Test adding user at the best position
    best_increase = float('inf')

    # Try inserting user at each position
    for insert_pos in range(len(route['assigned_users']) + 1):
        # Create temporary route with user inserted
        temp_users = route['assigned_users'].copy()
        temp_user = {
            'lat': user['latitude'],
            'lng': user['longitude'],
            'user_id': user.get('user_id', 'temp')
        }
        temp_users.insert(insert_pos, temp_user)

        temp_route = route.copy()
        temp_route['assigned_users'] = temp_users

        new_distance = calculate_route_total_cost(temp_route)
        increase = new_distance - current_distance

        if increase < best_increase:
            best_increase = increase

    return best_increase


def handle_remaining_users(unassigned_users, routes, driver_df):
    """Handle users that couldn't be assigned to any route"""
    unassigned_list = []

    # Handle both DataFrame and list inputs
    if hasattr(unassigned_users, 'iterrows'):
        # DataFrame input
        for _, user in unassigned_users.iterrows():
            user_data = {
                'user_id': str(user['user_id']),
                'lat': float(user['latitude']),
                'lng': float(user['longitude']),
                'office_distance': float(user.get('office_distance', 0))
            }

            if pd.notna(user.get('first_name')):
                user_data['first_name'] = str(user['first_name'])
            if pd.notna(user.get('email')):
                user_data['email'] = str(user['email'])

            unassigned_list.append(user_data)
    else:
        # List input (from validate_and_reallocate_poor_routes)
        for user in unassigned_users:
            user_data = {
                'user_id': str(user.get('user_id', '')),
                'lat': float(user.get('latitude', user.get('lat', 0))),
                'lng': float(user.get('longitude', user.get('lng', 0))),
                'office_distance': float(user.get('office_distance', 0))
            }

            if user.get('first_name'):
                user_data['first_name'] = str(user['first_name'])
            if user.get('email'):
                user_data['email'] = str(user['email'])

            unassigned_list.append(user_data)

    return unassigned_list


# MAIN STRAIGHTFORWARD ASSIGNMENT FUNCTION
def run_assignment(source_id: str, parameter: int = 1, string_param: str = ""):
    """
    Main assignment function with straightforward approach:
    1. Geographic clustering
    2. Capacity-based sub-clustering
    3. Priority-based driver assignment
    4. Local optimization
    5. Global optimization
    6. Route quality validation and reallocation
    """
    start_time = time.time()

    try:
        print(
            f"🚀 Starting straightforward assignment for source_id: {source_id}"
        )
        print(f"📋 Parameter: {parameter}, String parameter: {string_param}")

        # Load and validate data
        data = load_env_and_fetch_data(source_id, parameter, string_param)

        # Edge case handling
        users = data.get('users', [])
        if not users:
            print("⚠️ No users found - returning empty assignment")
            return {
                "status": "true",
                "execution_time": time.time() - start_time,
                "data": [],
                "unassignedUsers": [],
                "unassignedDrivers": _get_all_drivers_as_unassigned(data),
                "clustering_analysis": {
                    "method": "No Users",
                    "clusters": 0
                },
                "parameter": parameter,
                "string_param": string_param
            }

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
            print("⚠️ No drivers available - all users unassigned")
            unassigned_users = _convert_users_to_unassigned_format(users)
            return {
                "status": "true",
                "execution_time": time.time() - start_time,
                "data": [],
                "unassignedUsers": unassigned_users,
                "unassignedDrivers": [],
                "clustering_analysis": {
                    "method": "No Drivers",
                    "clusters": 0
                },
                "parameter": parameter,
                "string_param": string_param
            }

        print(
            f"📥 Data loaded - Users: {len(users)}, Total Drivers: {len(all_drivers)}"
        )

        # Extract office coordinates and validate data
        office_lat, office_lon = extract_office_coordinates(data)
        validate_input_data(data)
        print("✅ Data validation passed")

        # Prepare dataframes
        user_df, driver_df = prepare_user_driver_dataframes(data)
        user_df = calculate_bearings_and_features(user_df, office_lat,
                                                  office_lon)

        print(
            f"📊 DataFrames prepared - Users: {len(user_df)}, Drivers: {len(driver_df)}"
        )

        # STEP 1: Geographic clustering
        user_df, clustering_results = create_geographic_clusters(
            user_df, driver_df, _config)

        # STEP 2: Capacity-based sub-clustering
        user_df = create_capacity_subclusters(user_df, driver_df)

        # STEP 3: Priority-based driver assignment
        routes, assigned_user_ids = assign_drivers_by_priority(
            user_df, driver_df, office_lat, office_lon)

        # STEP 4: Local optimization
        routes = local_optimization(routes)

        # STEP 5: Global optimization
        routes, unassigned_users = global_optimization(routes, user_df,
                                                       assigned_user_ids,
                                                       driver_df)

        # STEP 5.5: Route quality validation and reallocation
        print("🔍 Validating route quality and reallocating poor routes...")
        routes, unassigned_users = validate_and_reallocate_poor_routes(
            routes, unassigned_users, driver_df
        )

        # STEP 6: Handle unassigned users with smart fallback strategy
        if unassigned_users and _config.get('fallback_assign_enabled', True):
            print("🔄 Step 6: Applying smart fallback assignment for remaining users...")

            # First try to append unassigned users to existing routes
            appended_users = []
            for user in unassigned_users[:]:
                user_pos = (user['latitude'], user['longitude'])
                office_pos = (_config['OFFICE_LAT'], _config['OFFICE_LON'])

                best_route = None
                best_score = float('inf')

                # Try to append to existing routes
                for route in routes:
                    if len(route['assigned_users']) >= route['driver']['capacity']:
                        continue  # Route is full

                    driver_pos = (route['driver']['latitude'], route['driver']['longitude'])
                    existing_users = [(u['latitude'], u['longitude']) for u in route['assigned_users']]

                    # Check if user can be added to this route with reasonable detour
                    if hasattr(_config, 'road_network') and _config['road_network']:
                        if _config['road_network'].is_user_on_route_path(
                            driver_pos, existing_users, user_pos, office_pos,
                            max_detour_ratio=1.4  # More lenient for fallback
                        ):
                            # Calculate route coherence with this user added
                            test_users = existing_users + [user_pos]
                            coherence = _config['road_network'].get_route_coherence_score(
                                driver_pos, test_users, office_pos
                            )

                            if coherence > _config.get('route_coherence_threshold', 0.75) * 0.85:  # 85% of threshold
                                detour_score = calculate_route_distance_increase(route, user, _config)
                                if detour_score < best_score:
                                    best_score = detour_score
                                    best_route = route

                # Append user to best matching existing route
                if best_route is not None:
                    best_route['assigned_users'].append(user)
                    appended_users.append(user)
                    print(f"   ✅ Appended user {user.get('user_id', 'unknown')} to existing route")

            # Remove appended users from unassigned list
            unassigned_users = [u for u in unassigned_users if u not in appended_users]

            # For remaining unassigned users, create new routes
            if unassigned_users:
                fallback_routes, remaining_unassigned = fallback_assignment(
                    unassigned_users, [], user_df, driver_df, _config
                )
                routes.extend(fallback_routes)
                unassigned_users = remaining_unassigned

            # Update unassigned drivers
            assigned_driver_ids = {r['driver']['driver_id'] for r in routes}
            unassigned_drivers = [d for d in driver_df.to_dict('records') if d['driver_id'] not in assigned_driver_ids]

        # STEP 7: Fix problematic routes (split zig-zag routes)
        print("🔧 Fixing problematic routes...")
        routes = fix_problematic_routes(routes, driver_df)

        # STEP 8: Final local optimization after all assignments
        print("🔧 Final local optimization after all assignments...")
        routes = local_optimization(routes)

        # STEP 9: Final global optimization pass
        print("🌍 Final global optimization pass...")
        # Update assigned_user_ids after fallback assignments
        final_assigned_user_ids = set()
        for route in routes:
            for user in route['assigned_users']:
                final_assigned_user_ids.add(user['user_id'])

        routes, unassigned_users = global_optimization(routes, user_df,
                                                       final_assigned_user_ids,
                                                       driver_df)

        # Build unassigned drivers list
        assigned_driver_ids = {route['driver']['driver_id'] for route in routes}
        unassigned_drivers_df = driver_df[~driver_df['driver_id'].
                                          isin(assigned_driver_ids)]
        unassigned_drivers = []
        for _, driver in unassigned_drivers_df.iterrows():
            driver_data = {
                'driver_id': str(driver['driver_id']),
                'capacity': int(driver['capacity']),
                'vehicle_id': str(driver.get('vehicle_id', '')),
                'latitude': float(driver['latitude']),
                'longitude': float(driver['longitude'])
            }
            unassigned_drivers.append(driver_data)

        # Final metrics update
        for route in routes:
            update_route_metrics(route)

        execution_time = time.time() - start_time

        print(
            f"✅ Straightforward assignment complete in {execution_time:.2f}s")
        print(f"📊 Final routes: {len(routes)}")
        print(
            f"🎯 Users assigned: {sum(len(r['assigned_users']) for r in routes)}"
        )
        print(f"👥 Users unassigned: {len(unassigned_users)}")

        return {
            "status": "true",
            "execution_time": execution_time,
            "data": routes,
            "unassignedUsers": unassigned_users,
            "unassignedDrivers": unassigned_drivers,
            "clustering_analysis": clustering_results,
            "parameter": parameter,
            "string_param": string_param
        }

    except requests.exceptions.RequestException as req_err:
        logger.error(f"API request failed: {req_err}")
        return {"status": "false", "details": str(req_err), "data": []}
    except ValueError as val_err:
        logger.error(f"Data validation error: {val_err}")
        return {"status": "false", "details": str(val_err), "data": []}
    except Exception as e:
        logger.error(f"Assignment failed: {e}", exc_info=True)
        return {"status": "false", "details": str(e), "data": []}


def _get_all_drivers_as_unassigned(data):
    """Helper to get all drivers in the unassigned format"""
    all_drivers = []
    if "drivers" in data:
        drivers_data = data["drivers"]
        all_drivers.extend(drivers_data.get("driversUnassigned", []))
        all_drivers.extend(drivers_data.get("driversAssigned", []))
    else:
        all_drivers.extend(data.get("driversUnassigned", []))
        all_drivers.extend(data.get("driversAssigned", []))

    unassigned_drivers = []
    for driver in all_drivers:
        unassigned_drivers.append({
            'driver_id':
            str(driver.get('id', '')),
            'capacity':
            int(driver.get('capacity', 0)),
            'vehicle_id':
            str(driver.get('vehicle_id', '')),
            'latitude':
            float(driver.get('latitude', 0.0)),
            'longitude':
            float(driver.get('longitude', 0.0))
        })
    return unassigned_drivers


def fallback_assignment(unassigned_users, existing_routes, user_df, driver_df, config):
    """
    Fallback assignment for users that couldn't be assigned in main algorithm
    Creates emergency routes for isolated users
    """
    print(f"🚨 Running fallback assignment for {len(unassigned_users)} users...")

    new_routes = []
    remaining_unassigned = []

    # Get available drivers (not already used)
    used_driver_ids = {route['driver']['driver_id'] for route in existing_routes}
    available_drivers = driver_df[~driver_df['driver_id'].isin(used_driver_ids)]

    if available_drivers.empty:
        print("   ⚠️ No available drivers for fallback assignment")
        return [], unassigned_users

    # Sort available drivers by priority
    available_drivers = available_drivers.sort_values(['priority', 'capacity'], ascending=[True, False])

    # Group nearby unassigned users using simple clustering
    if len(unassigned_users) > 1:
        user_coords = [[u['latitude'], u['longitude']] for u in unassigned_users]

        try:
            from sklearn.cluster import DBSCAN
            clustering = DBSCAN(eps=0.05, min_samples=1).fit(user_coords)  # ~5.5km

            # Group users by cluster
            user_clusters = {}
            for i, label in enumerate(clustering.labels_):
                if label not in user_clusters:
                    user_clusters[label] = []
                user_clusters[label].append(unassigned_users[i])
        except Exception:
            # Fallback: treat each user as separate cluster
            user_clusters = {i: [user] for i, user in enumerate(unassigned_users)}
    else:
        user_clusters = {0: unassigned_users}

    # Assign drivers to clusters
    used_fallback_drivers = set()

    for cluster_id, cluster_users in user_clusters.items():
        if not cluster_users:
            continue

        # Find best available driver for this cluster
        cluster_center_lat = sum(u['latitude'] for u in cluster_users) / len(cluster_users)
        cluster_center_lng = sum(u['longitude'] for u in cluster_users) / len(cluster_users)

        best_driver = None
        min_cost = float('inf')

        for _, driver in available_drivers.iterrows():
            if driver['driver_id'] in used_fallback_drivers:
                continue

            # Check capacity
            if driver['capacity'] < len(cluster_users):
                continue

            # Calculate assignment cost
            distance = haversine_distance(
                driver['latitude'], driver['longitude'],
                cluster_center_lat, cluster_center_lng
            )

            # Add priority penalty
            priority_penalty = driver.get('priority', 1) * 3.0
            cost = distance + priority_penalty

            if cost < min_cost:
                min_cost = cost
                best_driver = driver

        if best_driver is not None:
            # Create emergency route
            route_users = []
            for user in cluster_users:
                user_data = {
                    'user_id': str(user.get('user_id', '')),
                    'lat': float(user.get('latitude', user.get('lat', 0))),
                    'lng': float(user.get('longitude', user.get('lng', 0))),
                    'office_distance': float(user.get('office_distance', 0))
                }

                if user.get('first_name'):
                    user_data['first_name'] = str(user['first_name'])
                if user.get('email'):
                    user_data['email'] = str(user['email'])

                route_users.append(user_data)

            emergency_route = {
                'driver': best_driver.to_dict(),
                'driver_id': str(best_driver['driver_id']),
                'latitude': float(best_driver['latitude']),
                'longitude': float(best_driver['longitude']),
                'assigned_users': route_users,
                'utilization': len(route_users) / best_driver['capacity'],
                'vehicle_type': int(best_driver['capacity']),
                'is_emergency_route': True
            }

            # Optimize route sequence
            emergency_route = optimize_route_sequence(emergency_route)
            update_route_metrics(emergency_route)

            new_routes.append(emergency_route)
            used_fallback_drivers.add(best_driver['driver_id'])

            print(f"   🚑 Created emergency route for {len(route_users)} users with driver {best_driver['driver_id']}")
        else:
            # No available driver for this cluster
            remaining_unassigned.extend(cluster_users)
            print(f"   ❌ Could not assign {len(cluster_users)} users - no suitable driver available")

    print(f"   ✅ Fallback assignment complete: {len(new_routes)} emergency routes created")
    return new_routes, remaining_unassigned


def _convert_users_to_unassigned_format(users):
    """Helper to convert user data to unassigned format"""
    unassigned_users = []
    for user in users:
        unassigned_users.append({
            'user_id':
            str(user.get('id', '')),
            'lat':
            float(user.get('latitude', 0.0)),
            'lng':
            float(user.get('longitude', 0.0)),
            'office_distance':
            float(user.get('office_distance', 0.0)),
            'first_name':
            str(user.get('first_name', '')),
            'email':
            str(user.get('email', ''))
        })
    return unassigned_users


def fix_problematic_routes(routes, driver_df):
    """Identify and fix routes with excessive zig-zag patterns"""
    fixed_routes = []
    used_driver_ids = {route['driver']['driver_id'] for route in routes}
    available_drivers = driver_df[~driver_df['driver_id'].isin(used_driver_ids
                                                               )]

    for route in routes:
        if len(route['assigned_users']) < 3:
            fixed_routes.append(route)
            continue

        # Check route coherence
        driver_pos = (route['latitude'], route['longitude'])
        user_positions = [(u['lat'], u['lng'])
                          for u in route['assigned_users']]
        office_pos = (OFFICE_LAT, OFFICE_LON)

        coherence_score = 1.0  # Default if no road network
        if ROAD_NETWORK is not None:
            try:
                coherence_score = ROAD_NETWORK.get_route_coherence_score(
                    driver_pos, user_positions, office_pos)
            except Exception:
                pass

        # If route has poor coherence, try to split it
        min_coherence = _config.get('min_road_coherence_score', 0.85)
        if coherence_score < min_coherence and len(
                route['assigned_users']) > 2:
            print(
                f"    🔧 Fixing problematic route {route['driver_id']} (coherence: {coherence_score:.2f})"
            )

            # Try to split route into more coherent sub-routes
            split_routes = split_problematic_route(route, available_drivers,
                                                   used_driver_ids)

            if split_routes and len(split_routes) > 1:
                print(
                    f"       ✅ Split into {len(split_routes)} more coherent routes"
                )
                fixed_routes.extend(split_routes)
            else:
                # If splitting failed, at least optimize the sequence
                route = optimize_route_sequence(route)
                fixed_routes.append(route)
        else:
            fixed_routes.append(route)

    return fixed_routes


def split_problematic_route(route, available_drivers, used_driver_ids):
    """Split a problematic route into more coherent sub-routes"""
    users = route['assigned_users']
    driver_pos = (route['latitude'], route['longitude'])
    office_pos = (OFFICE_LAT, OFFICE_LON)

    if len(users) < 3:
        return [route]

    # Group users by proximity and direction
    user_groups = cluster_users_by_coherence(users, driver_pos, office_pos)

    if len(user_groups) < 2:
        return [route]  # Can't split meaningfully

    split_routes = []

    # Keep the original route with the largest group
    largest_group_idx = max(range(len(user_groups)),
                            key=lambda i: len(user_groups[i]))
    route['assigned_users'] = user_groups[largest_group_idx]
    route = optimize_route_sequence(route)
    split_routes.append(route)

    # Try to assign new drivers to other groups
    for i, user_group in enumerate(user_groups):
        if i == largest_group_idx:
            continue

        if not available_drivers.empty and user_group:
            # Find best available driver for this group
            group_center_lat = sum(u['lat']
                                   for u in user_group) / len(user_group)
            group_center_lng = sum(u['lng']
                                   for u in user_group) / len(user_group)

            best_driver = None
            min_distance = float('inf')

            for _, driver in available_drivers.iterrows():
                if driver['driver_id'] in used_driver_ids:
                    continue
                if driver['capacity'] < len(user_group):
                    continue

                distance = haversine_distance(driver['latitude'],
                                              driver['longitude'],
                                              group_center_lat,
                                              group_center_lng)
                if distance < min_distance:
                    min_distance = distance
                    best_driver = driver

            if best_driver is not None:
                new_route = {
                    'driver': best_driver.to_dict(),
                    'latitude': float(best_driver['latitude']),
                    'longitude': float(best_driver['longitude']),
                    'assigned_users': user_group
                }

                new_route = optimize_route_sequence(new_route)
                update_route_metrics(new_route)
                split_routes.append(new_route)
                used_driver_ids.add(best_driver['driver_id'])
                available_drivers = available_drivers[
                    available_drivers['driver_id'] != best_driver['driver_id']]

    return split_routes


def cluster_users_by_coherence(users, driver_pos, office_pos):
    """Cluster users into coherent groups based on direction and proximity"""
    if len(users) < 3:
        return [users]

    # Calculate bearings from driver to each user
    user_bearings = []
    for user in users:
        bearing = calculate_bearing(driver_pos[0], driver_pos[1], user['lat'],
                                    user['lng'])
        user_bearings.append((bearing, user))

    # Sort by bearing
    user_bearings.sort(key=lambda x: x[0])

    # Group users with similar bearings (within 30 degrees)
    groups = []
    current_group = [user_bearings[0][1]]
    current_bearing = user_bearings[0][0]

    for bearing, user in user_bearings[1:]:
        bearing_diff = abs(bearing - current_bearing)
        if bearing_diff > 180:
            bearing_diff = 360 - bearing_diff

        if bearing_diff <= 30:  # Within 30 degrees
            current_group.append(user)
        else:
            groups.append(current_group)
            current_group = [user]
            current_bearing = bearing

    groups.append(current_group)

    # Merge small groups with nearby larger groups
    final_groups = []
    for group in groups:
        if len(group) >= 2 or not final_groups:
            final_groups.append(group)
        else:
            # Add single user to nearest group
            final_groups[-1].extend(group)

    return final_groups


def analyze_assignment_quality(result):
    """Analyze the quality of the assignment with enhanced metrics"""
    if result["status"] != "true":
        return "Assignment failed"

    total_routes = len(result["data"])
    total_assigned = sum(
        len(route["assigned_users"]) for route in result["data"])
    total_unassigned = len(result["unassignedUsers"])

    utilizations = []
    distance_issues = []

    for route in result["data"]:
        if route["assigned_users"]:
            util = len(route["assigned_users"]) / route["driver"]["capacity"]
            utilizations.append(util)

            # Check distances
            driver_pos = (route["latitude"], route["longitude"])
            for user in route["assigned_users"]:
                dist = haversine_distance(driver_pos[0], driver_pos[1],
                                          user["lat"], user["lng"])
                if dist > DISTANCE_ISSUE_THRESHOLD:
                    distance_issues.append({
                        "driver_id": route["driver_id"],
                        "user_id": user["user_id"],
                        "distance_km": round(dist, 2)
                    })

    analysis = {
        "total_routes":
        total_routes,
        "total_assigned_users":
        total_assigned,
        "total_unassigned_users":
        total_unassigned,
        "assignment_rate":
        round(total_assigned / (total_assigned + total_unassigned) * 100, 1) if
        (total_assigned + total_unassigned) > 0 else 0,
        "avg_utilization":
        round(np.mean(utilizations) * 100, 1) if utilizations else 0,
        "min_utilization":
        round(np.min(utilizations) * 100, 1) if utilizations else 0,
        "max_utilization":
        round(np.max(utilizations) * 100, 1) if utilizations else 0,
        "routes_below_80_percent":
        sum(1 for u in utilizations if u < 0.8),
        "distance_issues":
        distance_issues,
        "clustering_method":
        result.get("clustering_analysis", {}).get("method", "Unknown")
    }

    return analysis