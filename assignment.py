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

    # Angle configurations (0-180 degrees) - more lenient for better coverage
    config['MAX_BEARING_DIFFERENCE'] = max(
        0, min(180, float(cfg.get("max_bearing_difference", 45))))  # Increased from 15 to 45
    config['STRICT_BEARING_DIFFERENCE'] = max(
        0, min(180, float(cfg.get("strict_bearing_difference", 35))))  # Increased from 10 to 35

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

    # Road network related configurations with much more lenient defaults
    config['use_road_network'] = cfg.get("use_road_network", False)
    config['road_network_path'] = cfg.get("road_network_path", 'tricity_main_roads.graphml')
    config['min_cluster_size'] = max(2, int(cfg.get("min_cluster_size", 2)))
    config['max_extra_distance_km'] = max(1.0, float(cfg.get("max_extra_distance_km", 3.0)))
    config['fallback_assign_enabled'] = cfg.get("fallback_assign_enabled", True)
    config['min_road_coherence_score'] = max(0.0, min(1.0, float(cfg.get("min_road_coherence_score", 0.15))))  # Much more lenient
    config['min_driver_capacity_util'] = max(0.0, min(1.0, float(cfg.get("min_driver_capacity_util", 0.2)))) # Minimum utilization for driver assignment


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


def is_user_along_route_path(driver_pos, existing_users, candidate_user, office_pos):
    """
    Simplified path check - just use distance
    """
    driver_lat, driver_lon = driver_pos
    candidate_lat = candidate_user.get('latitude', candidate_user.get('lat', 0))
    candidate_lon = candidate_user.get('longitude', candidate_user.get('lng', 0))

    # Simple distance check
    distance = haversine_distance(driver_lat, driver_lon, candidate_lat, candidate_lon)
    
    return distance <= MAX_FILL_DISTANCE_KM


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
    driver_df = driver_df.sort_values(['priority', 'capacity'], ascending=[True, False])

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
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def calculate_bearings_and_features(user_df, office_lat, office_lon):
    """Calculate bearings and add geographical features - OFFICE TO USER direction"""
    user_df = user_df.copy()
    user_df[['office_latitude', 'office_longitude']] = office_lat, office_lon

    def calculate_bearing_vectorized(lat1, lon1, lat2, lon2):
        lat1, lat2 = np.radians(lat1), np.radians(lat2)
        dlon = np.radians(lon2 - lon1)
        x = np.sin(dlon) * np.cos(lat2)
        y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
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
    """Direction-aware geographic clustering from office"""
    print("🗂️ Step 1: Creating direction-aware geographic clusters...")

    if len(user_df) < 2:
        user_df['cluster'] = 0
        user_df['geo_cluster'] = 0
        return user_df, {"method": "Single Cluster", "clusters": 1}

    # Group users by directional sectors from office
    office_lat, office_lon = config['OFFICE_LAT'], config['OFFICE_LON']
    
    # Calculate bearing from office to each user
    user_df['bearing_from_office'] = user_df.apply(
        lambda row: calculate_bearing(office_lat, office_lon, row['latitude'], row['longitude']), 
        axis=1
    )
    
    # Calculate distance from office
    user_df['distance_from_office'] = user_df.apply(
        lambda row: haversine_distance(office_lat, office_lon, row['latitude'], row['longitude']), 
        axis=1
    )
    
    # Create directional clusters (8 sectors of 45 degrees each)
    sector_size = 45  # degrees
    user_df['direction_sector'] = (user_df['bearing_from_office'] / sector_size).astype(int)
    
    # Within each sector, create distance-based sub-clusters
    clusters = []
    cluster_id = 0
    
    for sector in sorted(user_df['direction_sector'].unique()):
        sector_users = user_df[user_df['direction_sector'] == sector]
        
        if len(sector_users) == 1:
            # Single user in sector
            user_df.loc[sector_users.index, 'cluster'] = cluster_id
            user_df.loc[sector_users.index, 'geo_cluster'] = cluster_id
            clusters.append(cluster_id)
            cluster_id += 1
        else:
            # Multiple users in sector - cluster by distance
            distances = sector_users['distance_from_office'].values.reshape(-1, 1)
            
            # Use distance-based clustering within the directional sector
            max_distance_km = config.get('max_extra_distance_km', 8.0)
            eps_degrees = max_distance_km / 111.32
            
            coordinates = sector_users[['latitude', 'longitude']].values
            sector_clustering = DBSCAN(eps=eps_degrees, min_samples=1).fit(coordinates)
            
            for i, label in enumerate(sector_clustering.labels_):
                actual_cluster_id = cluster_id + (label if label != -1 else 0)
                user_idx = sector_users.index[i]
                user_df.loc[user_idx, 'cluster'] = actual_cluster_id
                user_df.loc[user_idx, 'geo_cluster'] = actual_cluster_id
                
                if actual_cluster_id not in clusters:
                    clusters.append(actual_cluster_id)
            
            cluster_id += len(set(sector_clustering.labels_)) - (1 if -1 in sector_clustering.labels_ else 0) + (1 if -1 in sector_clustering.labels_ else 0)
    
    n_clusters = len(clusters)
    print(f"   ✅ Created {n_clusters} direction-aware clusters across {len(user_df['direction_sector'].unique())} directional sectors")
    
    # Log cluster distribution by direction
    for sector in sorted(user_df['direction_sector'].unique()):
        sector_clusters = user_df[user_df['direction_sector'] == sector]['cluster'].nunique()
        bearing_range = f"{sector * sector_size}-{(sector + 1) * sector_size}°"
        print(f"      📍 Sector {bearing_range}: {sector_clusters} clusters")
    
    return user_df, {"method": "Direction-Aware DBSCAN", "clusters": n_clusters}


# STEP 2: Create capacity-based sub-clusters
def create_capacity_subclusters(cluster_users, driver_df, config):
    """
    Step 2: Within each geographic cluster, create sub-clusters based on driver capacity
    """
    print("🔧 Step 2: Creating capacity-based sub-clusters...")
    print(f"   📊 Processing {len(cluster_users)} users in a geographic cluster")

    # Calculate average driver capacity for sizing
    avg_capacity = int(driver_df['capacity'].mean()) if not driver_df.empty else 5
    max_capacity = int(driver_df['capacity'].max()) if not driver_df.empty else 10
    if avg_capacity == 0: avg_capacity = 5
    if max_capacity == 0: max_capacity = 10

    user_df = pd.DataFrame(cluster_users) # Ensure we work with a DataFrame
    user_df['sub_cluster'] = 0
    sub_cluster_counter = 0

    if len(user_df) <= max_capacity:
        # Small cluster, no need to split
        user_df['sub_cluster'] = sub_cluster_counter
        print(f"   ✅ Cluster is small ({len(user_df)} users), assigned to single sub-cluster {sub_cluster_counter}")
        return [user_df.to_dict('records')]
    else:
        # Large cluster, split by capacity
        coords = user_df[['latitude', 'longitude']].values
        n_subclusters = max(1, min(math.ceil(len(user_df) / avg_capacity), len(user_df))) # Ensure at least 1 subcluster

        # Use KMeans to find sub-clusters, aiming for roughly 'avg_capacity' per cluster
        try:
            kmeans = KMeans(n_clusters=n_subclusters, random_state=42, n_init=10)
            sub_labels = kmeans.fit_predict(coords)

            split_subclusters = []
            for i in range(n_subclusters):
                sub_cluster_users = user_df[sub_labels == i]
                if not sub_cluster_users.empty:
                    split_subclusters.append(sub_cluster_users.to_dict('records'))

            print(f"      ➡️ Split into {len(split_subclusters)} sub-clusters (target avg size: {avg_capacity})")
            return split_subclusters

        except ValueError as ve:
            print(f"      ⚠️ KMeans failed for splitting: {ve}. Assigning to single sub-cluster.")
            user_df['sub_cluster'] = sub_cluster_counter
            return [user_df.to_dict('records')]
        except Exception as e:
            print(f"      ⚠️ Unexpected error during KMeans splitting: {e}. Assigning to single sub-cluster.")
            user_df['sub_cluster'] = sub_cluster_counter
            return [user_df.to_dict('records')]


# STEP 3: Assign drivers by priority
def assign_drivers_by_priority(user_df, driver_df, office_lat, office_lon):
    """
    Step 3: Assign drivers based on priority and proximity
    """
    print("🚗 Step 3: Assigning drivers by priority...")
    print(f"   📊 Starting with {len(user_df)} users and {len(driver_df)} available drivers")

    routes = []
    available_drivers = list(driver_df.to_dict('records'))

    # Log available drivers by priority
    priority_counts = {}
    for driver in available_drivers:
        priority = driver.get('priority_group', 'unknown')
        priority_counts[priority] = priority_counts.get(priority, 0) + 1
    print(f"   🎯 Available drivers by priority: {priority_counts}")

    # Group users by geographic cluster first
    clusters = {}
    for _, row in user_df.iterrows():
        geo_cluster_id = row['geo_cluster']
        if geo_cluster_id not in clusters:
            clusters[geo_cluster_id] = []
        clusters[geo_cluster_id].append(row.to_dict())

    print(f"   🌍 Found {len(clusters)} geographic clusters")

    # Process each cluster with performance optimization
    all_subclusters = []
    for cluster_idx, cluster in enumerate(clusters.values()):
        print(f"   🌍 Geographic cluster {cluster_idx}: {len(cluster)} users")

        # Skip empty clusters
        if not cluster:
            continue

        # Step 2: Create capacity-based sub-clusters within each geographic cluster
        try:
            cluster_subclusters = create_capacity_subclusters(cluster, driver_df, _config)
            if cluster_subclusters:
                all_subclusters.extend(cluster_subclusters)
                print(f"      ➡️ Split into {len(cluster_subclusters)} capacity-based sub-clusters")
            else:
                print(f"      ⚠️ No valid sub-clusters created for geographic cluster {cluster_idx}")
        except Exception as e:
            print(f"      ❌ Error creating sub-clusters for cluster {cluster_idx}: {e}")
            # Add original cluster as fallback
            if cluster:
                all_subclusters.append(cluster)

    print(f"  ✅ Created {len(all_subclusters)} total capacity-based sub-clusters")

    # Log sub-cluster sizes
    subcluster_sizes = [len(sc) for sc in all_subclusters if sc]
    if subcluster_sizes:
        avg_size = sum(subcluster_sizes) / len(subcluster_sizes)
        print(f"     📊 Sub-cluster sizes: min={min(subcluster_sizes)}, max={max(subcluster_sizes)}, avg={avg_size:.1f}")
    else:
        print(f"     ⚠️  No valid sub-clusters created!")

    unassigned_users = [] # Initialize unassigned_users list

    total_users_in_subclusters = sum(len(subcluster) for subcluster in all_subclusters)
    print(f"   👥 Total users in sub-clusters: {total_users_in_subclusters}")

    for i, subcluster in enumerate(all_subclusters):
        if not subcluster:
            print(f"  📍 Sub-cluster {i}: Empty, skipping")
            continue

        print(f"  📍 Sub-cluster {i}: Processing {len(subcluster)} users")

        # Quick road coherence check (optimized)
        print(f"     🛣️ Checking road coherence...")
        try:
            final_subcluster = split_cluster_for_road_coherence(subcluster, _config)
        except Exception as e:
            print(f"     ⚠️ Road coherence check failed: {e}, using original subcluster")
            final_subcluster = subcluster

        if not final_subcluster:
            print(f"     ❌ Sub-cluster {i}: No users remaining after road coherence check")
            continue

        if len(final_subcluster) < len(subcluster):
            removed_count = len(subcluster) - len(final_subcluster)
            print(f"     ⚠️ Sub-cluster {i}: {removed_count} users removed for road coherence")
            # Add removed users to unassigned list
            removed_users = [user for user in subcluster if user not in final_subcluster]
            unassigned_users.extend(removed_users)

        print(f"     👥 Final sub-cluster {i}: {len(final_subcluster)} users ready for driver assignment")

        # Assign best available driver to this subcluster (optimized)
        print(f"     🚗 Attempting driver assignment for sub-cluster {i}...")
        start_assign_time = time.time()

        try:
            assigned_driver, available_drivers = assign_best_driver_to_cluster(
                final_subcluster, available_drivers, _config
            )
            assign_time = time.time() - start_assign_time

            if assigned_driver:
                route = create_route_from_assignment(assigned_driver, final_subcluster)
                routes.append(route)
                print(f"     ✅ Sub-cluster {i}: Driver {assigned_driver['driver_id']} assigned to {len(final_subcluster)} users ({assign_time:.2f}s)")
            else:
                # Add unassigned users to the list
                unassigned_users.extend(final_subcluster)
                print(f"     ❌ Sub-cluster {i}: No driver assigned, {len(final_subcluster)} users remain unassigned ({assign_time:.2f}s)")

                # Log why assignment failed (simplified for performance)
                if not available_drivers:
                    print(f"        🚫 Reason: No more available drivers")
                else:
                    min_capacity_needed = len(final_subcluster)
                    suitable_count = sum(1 for d in available_drivers if d.get('capacity', d.get('vehicle_type', 0)) >= min_capacity_needed)
                    print(f"        🚫 Reason: No suitable driver found (need capacity ≥{min_capacity_needed}, {suitable_count} available)")

        except Exception as e:
            print(f"     ❌ Sub-cluster {i}: Driver assignment failed with error: {e}")
            unassigned_users.extend(final_subcluster)

    print(f"   📈 Driver assignment summary:")
    total_assigned = sum(len(route['assigned_users']) for route in routes)
    print(f"      ✅ Routes created: {len(routes)}")
    print(f"      👥 Users assigned: {total_assigned}")
    print(f"      ❌ Users unassigned: {len(unassigned_users)}")
    print(f"      🚗 Drivers used: {len(routes)}")
    print(f"      💤 Drivers remaining: {len(available_drivers)}")

    # Track assignment success rate
    total_users = len(user_df)
    assignment_rate = (total_assigned / total_users * 100) if total_users > 0 else 0
    print(f"      📊 Assignment rate: {assignment_rate:.1f}% ({total_assigned}/{total_users})")

    # Log unassigned user details for debugging
    if unassigned_users:
        print(f"      🔍 Unassigned user details:")
        for i, user in enumerate(unassigned_users[:5]):  # Show first 5
            user_id = user.get('user_id', 'unknown')
            lat = user.get('latitude', user.get('lat', 0))
            lng = user.get('longitude', user.get('lng', 0))
            print(f"         {i+1}. User {user_id} at ({lat:.4f}, {lng:.4f})")
        if len(unassigned_users) > 5:
            print(f"         ... and {len(unassigned_users) - 5} more")
    return routes, unassigned_users


def assign_best_driver_to_cluster(cluster_users, available_drivers, config):
    """
    Direction-aware driver assignment - prioritize drivers that create coherent routes
    """
    if not cluster_users or not available_drivers:
        return None, available_drivers

    # Calculate cluster centroid and direction
    cluster_lat = sum(float(user['lat']) for user in cluster_users) / len(cluster_users)
    cluster_lng = sum(float(user['lng']) for user in cluster_users) / len(cluster_users)
    
    # Calculate cluster bearing from office
    office_lat, office_lon = config['OFFICE_LAT'], config['OFFICE_LON']
    cluster_bearing = calculate_bearing(office_lat, office_lon, cluster_lat, cluster_lng)

    print(f"  🎯 Assigning driver to cluster: {len(cluster_users)} users (bearing: {cluster_bearing:.0f}°)")

    best_driver = None
    best_score = float('inf')

    for driver in available_drivers:
        # Capacity check
        if driver['capacity'] < len(cluster_users):
            continue

        driver_lat = float(driver['latitude'])
        driver_lng = float(driver['longitude'])
        
        # Distance score
        distance = haversine_distance(driver_lat, driver_lng, cluster_lat, cluster_lng)
        
        # Direction coherence score - prefer drivers that create logical route progression
        driver_bearing = calculate_bearing(office_lat, office_lon, driver_lat, driver_lng)
        bearing_diff = abs(driver_bearing - cluster_bearing)
        if bearing_diff > 180:
            bearing_diff = 360 - bearing_diff
        
        # Directional penalty: prefer drivers in same general direction
        direction_penalty = bearing_diff / 180.0 * 10  # 0-10 km penalty
        
        # Route coherence bonus if using road network
        road_coherence_bonus = 0
        if ROAD_NETWORK is not None:
            try:
                # Test route coherence
                user_positions = [(float(user['lat']), float(user['lng'])) for user in cluster_users]
                driver_pos = (driver_lat, driver_lng)
                coherence = ROAD_NETWORK.get_route_coherence_score(driver_pos, user_positions, (office_lat, office_lon))
                road_coherence_bonus = (1.0 - coherence) * 5  # 0-5 km penalty for poor coherence
            except:
                pass
        
        # Priority penalty
        priority = driver.get('priority', 4)
        priority_penalty = priority * 2  # Lower priority = higher penalty
        
        # Combined score (lower is better)
        score = distance + direction_penalty + road_coherence_bonus + priority_penalty
        
        if score < best_score:
            best_score = score
            best_driver = driver

    if best_driver:
        driver_bearing = calculate_bearing(office_lat, office_lon, float(best_driver['latitude']), float(best_driver['longitude']))
        bearing_diff = abs(driver_bearing - cluster_bearing)
        if bearing_diff > 180:
            bearing_diff = 360 - bearing_diff
        
        print(f"  ✅ Assigned driver {best_driver['driver_id']} (bearing diff: {bearing_diff:.0f}°, score: {best_score:.1f})")
        updated_drivers = [d for d in available_drivers if d['driver_id'] != best_driver['driver_id']]
        return best_driver, updated_drivers

    print(f"  ❌ No suitable driver found")
    return None, available_drivers


def create_route_from_assignment(driver, users):
    """Create a route dictionary from a driver and assigned users"""
    route = {
        'driver': driver,
        'driver_id': str(driver['driver_id']),
        'latitude': float(driver['latitude']),
        'longitude': float(driver['longitude']),
        'assigned_users': [],
        'vehicle_type': int(driver['capacity']),
        'utilization': len(users) / driver['capacity'] if driver['capacity'] > 0 else 0,
        'priority_group': driver.get('priority_group', 2)
    }

    # Prepare user data for the route
    for user in users:
        user_data = {
            'user_id': str(user.get('user_id', user.get('id', ''))),
            'lat': float(user.get('latitude', user.get('lat', 0))),
            'lng': float(user.get('longitude', user.get('lng', 0))),
            'office_distance': float(user.get('office_distance', 0)),
            'first_name': str(user.get('first_name', '')),
            'email': str(user.get('email', ''))
        }
        route['assigned_users'].append(user_data)

    # Update route metrics (centroid, bearing)
    if route['assigned_users']:
        lats = [u['lat'] for u in route['assigned_users']]
        lngs = [u['lng'] for u in route['assigned_users']]
        route['centroid'] = [np.mean(lats), np.mean(lngs)]
        route['bearing'] = calculate_bearing(route['centroid'][0],
                                             route['centroid'][1], OFFICE_LAT,
                                             OFFICE_LON)
    else:
        route['centroid'] = [route['latitude'], route['longitude']]
        route['bearing'] = calculate_bearing(route['latitude'],
                                             route['longitude'], OFFICE_LAT, OFFICE_LON)

    return route


def split_cluster_for_road_coherence(cluster_users, config):
    """
    Check if a cluster needs to be split for better road network adherence.
    Returns the cluster (potentially split) or None if all users are removed.
    """
    if len(cluster_users) < 2:
        return cluster_users

    office_pos = (config['OFFICE_LAT'], config['OFFICE_LON'])

    # Simple check: if users are too spread out geographically, suggest splitting.
    if len(cluster_users) > 5: # Heuristic for larger clusters
        try:
            # Check the bearing spread of users relative to office
            bearings = []
            for user in cluster_users:
                # Fix coordinate access - handle both formats
                user_lat = user.get('latitude', user.get('lat', 0))
                user_lng = user.get('longitude', user.get('lng', 0))
                bearings.append(calculate_bearing(office_pos[0], office_pos[1], user_lat, user_lng))

            if len(bearings) > 1:
                bearings.sort()
                # Calculate max spread in bearings
                max_bearing_spread = bearings[-1] - bearings[0]
                if max_bearing_spread > 180: 
                    max_bearing_spread = 360 - max_bearing_spread

                # If bearing spread is large, note it but don't split here
                if max_bearing_spread > _config.get('MAX_BEARING_DIFFERENCE', 45):
                    print(f"     🛣️ Cluster has wide bearing spread: {max_bearing_spread:.1f} degrees")

        except Exception as e:
            print(f"     ⚠️ Error during road coherence check: {e}")

    return cluster_users


# STEP 4: Local optimization
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
            route = optimize_route_sequence(route)

            # Check if optimization improved the route
            new_cost = calculate_route_total_cost(route)
            if new_cost < current_cost - 0.1:  # Improved by at least 100m
                iteration_improvements += 1
                improved = True
                driver_id = route.get('driver_id', route.get('driver', {}).get('driver_id', 'unknown'))
                print(
                    f"      ✅ Route {driver_id} sequence optimized: {current_cost:.1f}km → {new_cost:.1f}km")

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
    """Optimize pickup sequence with strong direction awareness and road network integration"""
    if len(route['assigned_users']) <= 2:
        return route

    users = route['assigned_users'].copy()
    driver_pos = (route['latitude'], route['longitude'])
    office_pos = (OFFICE_LAT, OFFICE_LON)

    # Always try road network optimization first
    if ROAD_NETWORK is not None and len(users) > 1:
        try:
            user_positions = []
            for u in users:
                lat = u.get('lat', u.get('latitude', 0))
                lng = u.get('lng', u.get('longitude', 0))
                user_positions.append((lat, lng))

            optimal_indices = ROAD_NETWORK.get_optimal_pickup_sequence(
                driver_pos, user_positions, office_pos)

            # Reorder users based on optimal sequence
            optimized_sequence = [users[i] for i in optimal_indices]
            route['assigned_users'] = optimized_sequence

            print(f"      🛣️ Road-optimized pickup sequence for route {route.get('driver_id', 'unknown')}")
            return route
        except Exception as e:
            print(f"      ⚠️ Road optimization failed: {e}, using direction-aware fallback")

    # Direction-aware fallback: Sort by bearing from driver towards office
    office_bearing = calculate_bearing(driver_pos[0], driver_pos[1], office_pos[0], office_pos[1])
    
    # Calculate bearing and distance for each user from driver
    user_metrics = []
    for user in users:
        user_pos = (user['lat'], user['lng'])
        user_bearing = calculate_bearing(driver_pos[0], driver_pos[1], user_pos[0], user_pos[1])
        distance_from_driver = haversine_distance(driver_pos[0], driver_pos[1], user_pos[0], user_pos[1])
        distance_to_office = haversine_distance(user_pos[0], user_pos[1], office_pos[0], office_pos[1])
        
        # Calculate bearing difference from office direction
        bearing_diff = abs(user_bearing - office_bearing)
        if bearing_diff > 180:
            bearing_diff = 360 - bearing_diff
        
        user_metrics.append({
            'user': user,
            'bearing': user_bearing,
            'bearing_diff_from_office': bearing_diff,
            'distance_from_driver': distance_from_driver,
            'distance_to_office': distance_to_office,
            'progress_score': distance_from_driver - distance_to_office  # Positive = making progress to office
        })
    
    # Sort users by a combination of:
    # 1. Bearing alignment with office direction (primary)
    # 2. Distance from driver (secondary) 
    # 3. Progress toward office (tertiary)
    user_metrics.sort(key=lambda x: (
        x['bearing_diff_from_office'],  # Smaller bearing difference first
        x['distance_from_driver'],      # Closer users first
        -x['progress_score']            # Better progress toward office first
    ))
    
    # Extract optimized sequence
    optimized_sequence = [metric['user'] for metric in user_metrics]
    route['assigned_users'] = optimized_sequence
    
    # Log optimization result
    bearing_diffs = [m['bearing_diff_from_office'] for m in user_metrics]
    avg_bearing_diff = sum(bearing_diffs) / len(bearing_diffs) if bearing_diffs else 0
    print(f"      🧭 Direction-optimized sequence (avg bearing diff: {avg_bearing_diff:.0f}°)")
    
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
            current_cost_a = calculate_route_total_cost(route_a)
            current_cost_b = calculate_route_total_cost(route_b)
            current_total = current_cost_a + current_cost_b

            # Perform temporary swap
            route_a['assigned_users'].remove(user_a)
            route_b['assigned_users'].remove(user_b)
            route_a['assigned_users'].append(user_b)
            route_b['assigned_users'].append(user_a)

            new_cost_a = calculate_route_total_cost(route_a)
            new_cost_b = calculate_route_total_cost(route_b)
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
    print("🔍 Step 5.5: Validating route quality with detailed analysis...")

    good_routes = []
    users_to_reassign = []
    freed_drivers = []

    for route in routes:
        is_poor_route = False
        poor_reasons = []
        route_id = route.get('driver_id', 'unknown')

        if not route['assigned_users']:
            print(f"    ℹ️ Route {route_id}: Empty route, skipping")
            continue

        driver_pos = (route['latitude'], route['longitude'])
        user_count = len(route['assigned_users'])

        print(f"    🔍 Analyzing route {route_id} with {user_count} users...")

        # Check 1: Driver distance to users with much more lenient thresholds
        avg_distance_to_users = 0
        max_distance_to_user = 0
        distances = []

        for user in route['assigned_users']:
            distance = haversine_distance(
                driver_pos[0], driver_pos[1], user['lat'], user['lng']
            )
            distances.append(distance)
            avg_distance_to_users += distance
            max_distance_to_user = max(max_distance_to_user, distance)

        avg_distance_to_users /= len(route['assigned_users'])

        print(f"         📏 Distance metrics - Avg: {avg_distance_to_users:.1f}km, Max: {max_distance_to_user:.1f}km")
        print(f"         📏 Individual distances: {[f'{d:.1f}km' for d in distances]}")

        # MUCH more lenient distance thresholds - only reject truly extreme cases
        extreme_avg_threshold = MAX_FILL_DISTANCE_KM * 10  # 250km+ average
        extreme_max_threshold = MAX_FILL_DISTANCE_KM * 15  # 375km+ max

        if avg_distance_to_users > extreme_avg_threshold:
            is_poor_route = True
            poor_reasons.append(f"extreme avg distance {avg_distance_to_users:.1f}km > {extreme_avg_threshold}km")

        if max_distance_to_user > extreme_max_threshold:
            is_poor_route = True
            poor_reasons.append(f"extreme max distance {max_distance_to_user:.1f}km > {extreme_max_threshold}km")

        # Check 2: Very lenient utilization check - only for extremely wasteful routes
        utilization = len(route['assigned_users']) / route['driver']['capacity']
        print(f"         🚗 Utilization: {utilization:.1%} ({user_count}/{route['driver']['capacity']})")

        # Only reject if utilization is extremely low AND distance is extremely high
        if utilization < 0.15 and avg_distance_to_users > MAX_FILL_DISTANCE_KM * 8:  # Very extreme cases only
            is_poor_route = True
            poor_reasons.append(f"extremely low utilization {utilization:.1%} with extreme distance {avg_distance_to_users:.1f}km")

        # Check 3: Road coherence with very lenient threshold
        coherence_score = 1.0  # Default good score
        if ROAD_NETWORK is not None and len(route['assigned_users']) > 1:
            try:
                user_positions = [(u['lat'], u['lng']) for u in route['assigned_users']]
                coherence_score = ROAD_NETWORK.get_route_coherence_score(
                    driver_pos, user_positions, (OFFICE_LAT, OFFICE_LON))
                print(f"         🛣️ Road coherence: {coherence_score:.2f}")

                # Only reject if coherence is extremely poor (below 0.1)
                if coherence_score < 0.1:  # Extremely poor coherence only
                    is_poor_route = True
                    poor_reasons.append(f"extremely poor road coherence {coherence_score:.2f}")

            except Exception as e:
                print(f"         ⚠️ Road coherence check failed: {e}")
                coherence_score = 0.5  # Assume moderate if check fails

        # Final decision
        if is_poor_route:
            print(f"    🚫 REJECTING poor route {route_id}: {', '.join(poor_reasons)}")
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
            print(f"    ✅ ACCEPTING route {route_id}: Avg:{avg_distance_to_users:.1f}km, Max:{max_distance_to_user:.1f}km, Util:{utilization:.1%}, Coherence:{coherence_score:.2f}")
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
        print("   ⚠️ No available drivers or unassigned users for fallback assignment")
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

        # STEP 3: Priority-based driver assignment
        routes, unassigned_users = assign_drivers_by_priority(
            user_df, driver_df, office_lat, office_lon)

        # STEP 4: Local optimization
        routes = local_optimization(routes)

        # STEP 5: Global optimization
        assigned_user_ids = set()
        for route in routes:
            for user in route['assigned_users']:
                assigned_user_ids.add(user['user_id'])

        routes, unassigned_users_from_global = global_optimization(routes, user_df,
                                                       assigned_user_ids,
                                                       driver_df)
        unassigned_users.extend(unassigned_users_from_global)

        # STEP 5.5: Route quality validation and reallocation
        print("🔍 Validating route quality and reallocating poor routes...")
        routes, unassigned_users = validate_and_reallocate_poor_routes(
            routes, unassigned_users, driver_df
        )

        # STEP 6: DIRECTION-AWARE EMERGENCY ASSIGNMENT MODE
        if unassigned_users and _config.get('fallback_assign_enabled', True):
            print("🚨 Step 6: DIRECTION-AWARE EMERGENCY ASSIGNMENT MODE...")
            print(f"     🎯 Emergency target: {len(unassigned_users)} unassigned users")

            assigned_driver_ids = {r['driver']['driver_id'] for r in routes}
            available_drivers = driver_df[~driver_df['driver_id'].isin(assigned_driver_ids)]

            print(f"     📊 Available drivers for emergency assignment: {len(available_drivers)}")

            emergency_assigned = []
            emergency_routes_created = 0

            # Group unassigned users by direction from office
            for user in unassigned_users[:]:
                if available_drivers.empty:
                    print(f"     ⚠️ No more available drivers, trying to add to existing coherent routes...")
                    break

                user_pos = (user.get('latitude', user.get('lat', 0)), user.get('longitude', user.get('lng', 0)))
                user_id = user.get('user_id', 'unknown')
                
                # Calculate user bearing from office
                user_bearing = calculate_bearing(office_lat, office_lon, user_pos[0], user_pos[1])
                
                best_driver = None
                best_score = float('inf')

                print(f"     🔍 Finding direction-aware emergency driver for user {user_id} (bearing: {user_bearing:.0f}°)...")

                # Find best driver considering direction coherence even in emergency mode
                for _, driver in available_drivers.iterrows():
                    driver_pos = (driver['latitude'], driver['longitude'])
                    driver_bearing = calculate_bearing(office_lat, office_lon, driver_pos[0], driver_pos[1])
                    
                    distance = haversine_distance(driver_pos[0], driver_pos[1], user_pos[0], user_pos[1])
                    
                    # Direction coherence bonus
                    bearing_diff = abs(user_bearing - driver_bearing)
                    if bearing_diff > 180:
                        bearing_diff = 360 - bearing_diff
                    
                    direction_penalty = bearing_diff / 180.0 * 20  # Up to 20km penalty for opposite directions
                    
                    # Combined score prioritizing direction coherence
                    score = distance + direction_penalty
                    
                    if score < best_score:
                        best_score = score
                        best_driver = driver

                if best_driver is not None:
                    driver_bearing = calculate_bearing(office_lat, office_lon, best_driver['latitude'], best_driver['longitude'])
                    bearing_diff = abs(user_bearing - driver_bearing)
                    if bearing_diff > 180:
                        bearing_diff = 360 - bearing_diff
                    
                    # Create emergency route
                    emergency_route = {
                        'driver': best_driver.to_dict(),
                        'driver_id': str(best_driver['driver_id']),
                        'latitude': float(best_driver['latitude']),
                        'longitude': float(best_driver['longitude']),
                        'assigned_users': [{
                            'user_id': str(user.get('user_id', '')),
                            'lat': float(user.get('latitude', user.get('lat', 0))),
                            'lng': float(user.get('longitude', user.get('lng', 0))),
                            'office_distance': float(user.get('office_distance', 0)),
                            'first_name': str(user.get('first_name', '')),
                            'email': str(user.get('email', ''))
                        }],
                        'utilization': 1 / best_driver['capacity'],
                        'vehicle_type': int(best_driver['capacity']),
                        'is_emergency_route': True
                    }

                    routes.append(emergency_route)
                    emergency_assigned.append(user)
                    emergency_routes_created += 1
                    available_drivers = available_drivers[available_drivers['driver_id'] != best_driver['driver_id']]
                    print(f"       🚨 EMERGENCY: User {user_id} → Driver {best_driver['driver_id']} (bearing diff: {bearing_diff:.0f}°)")

            # For remaining users, add to existing routes with similar direction
            remaining_users = [u for u in unassigned_users if u not in emergency_assigned]
            if remaining_users:
                print(f"     🚨 CRITICAL: {len(remaining_users)} users still unassigned - adding to direction-compatible routes...")

                for user in remaining_users[:]:
                    user_id = user.get('user_id', 'unknown')
                    user_pos = (user.get('latitude', user.get('lat', 0)), user.get('longitude', user.get('lng', 0)))
                    user_bearing = calculate_bearing(office_lat, office_lon, user_pos[0], user_pos[1])

                    best_route = None
                    best_score = float('inf')

                    for route in routes:
                        # Calculate route average bearing
                        if route['assigned_users']:
                            route_lats = [u['lat'] for u in route['assigned_users']]
                            route_lngs = [u['lng'] for u in route['assigned_users']]
                            route_center_lat = sum(route_lats) / len(route_lats)
                            route_center_lng = sum(route_lngs) / len(route_lngs)
                            route_bearing = calculate_bearing(office_lat, office_lon, route_center_lat, route_center_lng)
                            
                            # Calculate bearing difference
                            bearing_diff = abs(user_bearing - route_bearing)
                            if bearing_diff > 180:
                                bearing_diff = 360 - bearing_diff
                            
                            distance = haversine_distance(user_pos[0], user_pos[1], route_center_lat, route_center_lng)
                            score = distance + (bearing_diff / 180.0 * 10)  # Prioritize direction coherence
                            
                            if score < best_score:
                                best_score = score
                                best_route = route

                    if best_route is not None:
                        # Add user to directionally compatible route
                        best_route['assigned_users'].append({
                            'user_id': str(user.get('user_id', '')),
                            'lat': float(user.get('latitude', user.get('lat', 0))),
                            'lng': float(user.get('longitude', user.get('lng', 0))),
                            'office_distance': float(user.get('office_distance', 0)),
                            'first_name': str(user.get('first_name', '')),
                            'email': str(user.get('email', ''))
                        })

                        best_route['utilization'] = len(best_route['assigned_users']) / best_route['driver']['capacity']
                        emergency_assigned.append(user)

                        print(f"       🚨 DIRECTION-MATCH: User {user_id} added to compatible route {best_route['driver_id']}")

            # Remove emergency assigned users
            unassigned_users = [u for u in unassigned_users if u not in emergency_assigned]

            print(f"     ✅ Direction-aware emergency assignment complete:")
            print(f"       🚨 Emergency routes created: {emergency_routes_created}")
            print(f"       📊 Users successfully assigned: {len(emergency_assigned)}")
            print(f"       ⚠️ Users still unassigned: {len(unassigned_users)}")

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

        routes, unassigned_users_from_global = global_optimization(routes, user_df,
                                                       final_assigned_user_ids,
                                                       driver_df)
        unassigned_users.extend(unassigned_users_from_global)


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
            f"🎯 Users assigned: {sum(len(r['assigned_users']) for r in routes)}")
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


def fallback_assignment(unassigned_users, available_drivers, config):
    """
    Fallback assignment for unassigned users using nearest driver approach
    """
    routes = []
    remaining_unassigned = []
    drivers_copy = available_drivers.copy()

    office_lat = config.get('office_latitude')
    office_lng = config.get('office_longitude')
    fallback_max_dist = config.get('fallback_max_dist_km', 8.0)

    print(f"   🎯 Fallback assignment parameters: max_distance={fallback_max_dist}km")

    # Sort users by distance from office (prioritize closer users)
    if office_lat and office_lng:
        unassigned_users.sort(key=lambda u: haversine_distance(
            float(u['lat']), float(u['lng']), office_lat, office_lng
        ))
        print(f"   📍 Sorted {len(unassigned_users)} users by distance from office")

    for i, user in enumerate(unassigned_users):
        user_lat = float(user['lat'])
        user_lng = float(user['lng'])

        # Check if user is within fallback distance from office
        office_distance = None
        if office_lat and office_lng:
            office_distance = haversine_distance(user_lat, user_lng, office_lat, office_lng)
            if office_distance > fallback_max_dist:
                print(f"      ❌ User {user.get('user_id', 'unknown')}: {office_distance:.2f}km > {fallback_max_dist}km from office")
                remaining_unassigned.append(user)
                continue

        # Find nearest available driver
        best_driver = None
        best_distance = float('inf')

        eligible_drivers = 0
        for driver in drivers_copy:
            if driver['vehicle_type'] < 1:  # Driver must have at least 1 seat
                continue
            eligible_drivers += 1

            distance = haversine_distance(
                user_lat, user_lng,
                float(driver['latitude']), float(driver['longitude'])
            )

            if distance < best_distance:
                best_distance = distance
                best_driver = driver

        if best_driver and best_distance <= fallback_max_dist:
            # Create route with this single user
            route = create_route_from_assignment(best_driver, [user])
            routes.append(route)
            drivers_copy.remove(best_driver)
            office_str = f" (office: {office_distance:.2f}km)" if office_distance else ""
            print(f"      ✅ User {user.get('user_id', 'unknown')}: assigned to driver {best_driver['driver_id']} at {best_distance:.2f}km{office_str}")
        else:
            remaining_unassigned.append(user)
            if not best_driver:
                print(f"      ❌ User {user.get('user_id', 'unknown')}: no eligible drivers found (checked {eligible_drivers} drivers)")
            else:
                print(f"      ❌ User {user.get('user_id', 'unknown')}: nearest driver {best_distance:.2f}km > {fallback_max_dist}km limit")

    print(f"   📊 Fallback results: {len(routes)} routes created, {len(remaining_unassigned)} users still unassigned")
    return routes, remaining_unassigned


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
        if len(route['assigned_users']) < 3: # Need at least 3 users for meaningful zig-zag detection
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

    # Keep the route with the largest group, possibly optimized
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