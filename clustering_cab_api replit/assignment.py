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
        print(f"Warning: Could not load config.json, using defaults. Error: {e}")
        cfg = {}

    # Validate and set configuration with type checking and bounds
    config = {}

    # Distance configurations (must be positive floats)
    config['MAX_FILL_DISTANCE_KM'] = max(0.1, float(cfg.get("max_fill_distance_km", 3.0)))  # Reduced for better clustering
    config['MERGE_DISTANCE_KM'] = max(0.1, float(cfg.get("merge_distance_km", 2.0)))  # Reduced for tighter clusters
    config['DBSCAN_EPS_KM'] = max(0.1, float(cfg.get("dbscan_eps_km", 2.0)))  # Reduced for tighter clusters
    config['OVERFLOW_PENALTY_KM'] = max(0.0, float(cfg.get("overflow_penalty_km", 15.0)))  # Increased penalty
    config['DISTANCE_ISSUE_THRESHOLD'] = max(0.1, float(cfg.get("distance_issue_threshold_km", 6.0)))  # Reduced threshold
    config['SWAP_IMPROVEMENT_THRESHOLD'] = max(0.0, float(cfg.get("swap_improvement_threshold_km", 0.3)))  # More sensitive

    # Utilization thresholds (0-1 range) - Made more strict for route quality
    config['MIN_UTIL_THRESHOLD'] = max(0.0, min(1.0, float(cfg.get("min_util_threshold", 0.6))))  # Increased
    config['LOW_UTILIZATION_THRESHOLD'] = max(0.0, min(1.0, float(cfg.get("low_utilization_threshold", 0.6))))  # Increased

    # Integer configurations (must be positive)
    config['MIN_SAMPLES_DBSCAN'] = max(1, int(cfg.get("min_samples_dbscan", 2)))
    config['MAX_SWAP_ITERATIONS'] = max(1, int(cfg.get("max_swap_iterations", 5)))  # More iterations
    config['MAX_USERS_FOR_FALLBACK'] = max(1, int(cfg.get("max_users_for_fallback", 3)))
    config['FALLBACK_MIN_USERS'] = max(1, int(cfg.get("fallback_min_users", 2)))
    config['FALLBACK_MAX_USERS'] = max(1, int(cfg.get("fallback_max_users", 7)))

    # CRITICAL: Much stricter directional constraints
    config['MAX_BEARING_DIFFERENCE'] = max(0, min(180, float(cfg.get("max_bearing_difference", 30))))  # Reduced from 45
    config['MERGE_BEARING_TOLERANCE'] = max(0, min(180, float(cfg.get("merge_bearing_tolerance", 75))))  # Expanded for merging
    config['DIRECTIONAL_CLUSTERING_WEIGHT'] = max(0.0, float(cfg.get("directional_clustering_weight", 0.8)))  # New parameter
    config['ROUTE_EFFICIENCY_THRESHOLD'] = max(0.0, float(cfg.get("route_efficiency_threshold", 0.7)))  # New parameter

    # Cost penalties (must be non-negative) - Made more aggressive
    config['UTILIZATION_PENALTY_PER_SEAT'] = max(0.0, float(cfg.get("utilization_penalty_per_seat", 0.5)))  # Reduced penalty
    config['DIRECTIONAL_SCATTER_PENALTY'] = max(0.0, float(cfg.get("directional_scatter_penalty", 30.0)))  # Reduced penalty

    # Global optimization parameters
    config['GLOBAL_MERGE_DISTANCE_KM'] = max(0.1, float(cfg.get("global_merge_distance_km", 8.0)))
    config['MIN_UTILIZATION_FOR_MERGE'] = max(0.0, min(1.0, float(cfg.get("min_utilization_for_merge", 0.5))))
    config['PRIORITY_WEIGHT_MULTIPLIER'] = max(0.0, float(cfg.get("priority_weight_multiplier", 5.0)))
    config['REGIONAL_MERGE_THRESHOLD_KM'] = max(0.1, float(cfg.get("regional_merge_threshold_km", 7.0)))
    config['MIN_ROUTE_UTILIZATION'] = max(0.0, min(1.0, float(cfg.get("min_route_utilization", 0.6))))

    # Office coordinates with environment variable fallbacks
    office_lat = float(os.getenv("OFFICE_LAT", cfg.get("office_latitude", 30.6810489)))
    office_lon = float(os.getenv("OFFICE_LON", cfg.get("office_longitude", 76.7260711)))

    # Validate coordinate bounds
    if not (-90 <= office_lat <= 90):
        print(f"Warning: Invalid office latitude {office_lat}, using default")
        office_lat = 30.6810489
    if not (-180 <= office_lon <= 180):
        print(f"Warning: Invalid office longitude {office_lon}, using default")
        office_lon = 76.7260711

    config['OFFICE_LAT'] = office_lat
    config['OFFICE_LON'] = office_lon

    return config

# Load validated configuration
_config = load_and_validate_config()
MAX_FILL_DISTANCE_KM = _config['MAX_FILL_DISTANCE_KM']
MERGE_DISTANCE_KM = _config['MERGE_DISTANCE_KM']
MIN_UTIL_THRESHOLD = _config['MIN_UTIL_THRESHOLD']
DBSCAN_EPS_KM = _config['DBSCAN_EPS_KM']
MIN_SAMPLES_DBSCAN = _config['MIN_SAMPLES_DBSCAN']
MAX_BEARING_DIFFERENCE = _config['MAX_BEARING_DIFFERENCE']
MERGE_BEARING_TOLERANCE = _config['MERGE_BEARING_TOLERANCE']
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
DIRECTIONAL_CLUSTERING_WEIGHT = _config['DIRECTIONAL_CLUSTERING_WEIGHT']
ROUTE_EFFICIENCY_THRESHOLD = _config['ROUTE_EFFICIENCY_THRESHOLD']
DIRECTIONAL_SCATTER_PENALTY = _config['DIRECTIONAL_SCATTER_PENALTY']
GLOBAL_MERGE_DISTANCE_KM = _config['GLOBAL_MERGE_DISTANCE_KM']
MIN_UTILIZATION_FOR_MERGE = _config['MIN_UTILIZATION_FOR_MERGE']
PRIORITY_WEIGHT_MULTIPLIER = _config['PRIORITY_WEIGHT_MULTIPLIER']
REGIONAL_MERGE_THRESHOLD_KM = _config['REGIONAL_MERGE_THRESHOLD_KM']
MIN_ROUTE_UTILIZATION = _config['MIN_ROUTE_UTILIZATION']


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
                raise ValueError(f"User {i} invalid latitude: {lat} (must be -90 to 90)")
            if not (-180 <= lon <= 180):
                raise ValueError(f"User {i} invalid longitude: {lon} (must be -180 to 180)")
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
        driver_id = str(driver["id"])
        duplicate_count = sum(1 for d in all_drivers if str(d.get("id", "")) == driver_id)
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
                raise ValueError(f"Driver {i} invalid capacity: {capacity} (must be > 0)")
        except (ValueError, TypeError) as e:
            raise ValueError(f"Driver {i} invalid capacity: {e}")

    if duplicate_driver_count > 0:
        print(f"ℹ️ INFO: Found {duplicate_driver_count} duplicate driver entries (normal for pick/drop scenarios)")

    print(f"✅ Input data validation passed - {len(users)} users, {len(all_drivers)} drivers")


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
    payload = resp.json()

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


def calculate_bearing_vectorized(lat1, lon1, lat2, lon2):
    """Vectorized bearing calculation"""
    lat1, lat2 = np.radians(lat1), np.radians(lat2)
    dlon = np.radians(lon2 - lon1)
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    return (np.degrees(np.arctan2(x, y)) + 360) % 360

def calculate_bearing(lat1, lon1, lat2, lon2):
    """Calculate bearing from point A to B in degrees"""
    lat1, lat2 = map(math.radians, [lat1, lat2])
    dlon = math.radians(lon2 - lon1)
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


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
        if driver_id not in seen_drivers or seen_drivers[driver_id]['priority'] > priority:
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
        if driver_id not in seen_drivers or seen_drivers[driver_id]['priority'] > priority:
            seen_drivers[driver_id] = driver_copy

    # Convert to list
    all_drivers = list(seen_drivers.values())

    # Recalculate counts after deduplication
    final_priority_counts = {1: 0, 2: 0, 3: 0, 4: 0}
    for driver in all_drivers:
        final_priority_counts[driver['priority']] += 1

    print(f"🔍 Driver Priority Assignment Debug (after deduplication):")
    print(f"   Priority 1 (driversUnassigned ST:1,3): {final_priority_counts[1]}")
    print(f"   Priority 2 (driversUnassigned ST:2): {final_priority_counts[2]}")
    print(f"   Priority 3 (driversAssigned ST:1,3): {final_priority_counts[3]}")
    print(f"   Priority 4 (driversAssigned ST:2): {final_priority_counts[4]}")

    original_total = priority_1_count + priority_2_count + priority_3_count + priority_4_count
    final_total = sum(final_priority_counts.values())
    if original_total != final_total:
        print(f"   📊 Deduplicated: {original_total} → {final_total} drivers ({original_total - final_total} duplicates resolved)")

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
    """Calculate haversine distance between two points in kilometers (cached) with edge case handling"""
    R = 6371.0  # Earth radius in kilometers

    # Handle edge cases
    if lat1 == lat2 and lon1 == lon2:
        return 0.0

    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # Use abs() to handle potential floating point precision issues
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    a = min(1.0, abs(a))  # Clamp to [0,1] to prevent domain errors

    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return abs(R * c)  # Ensure positive distance


def calculate_bearings_and_features(user_df, office_lat, office_lon):
    """Calculate bearings and add geographical features - OFFICE TO USER direction"""
    user_df = user_df.copy()
    user_df[['office_latitude', 'office_longitude']] = office_lat, office_lon

    # Calculate bearing FROM OFFICE TO USER (standardized direction)
    user_df['bearing'] = calculate_bearing_vectorized(office_lat, office_lon,
                                                      user_df['latitude'], user_df['longitude'])
    user_df['bearing_sin'] = np.sin(np.radians(user_df['bearing']))
    user_df['bearing_cos'] = np.cos(np.radians(user_df['bearing']))

    # Geographic quadrant analysis - simplified
    user_df['quadrant'] = 'CENTER'
    user_df.loc[(user_df['latitude'] > office_lat) &
                (user_df['longitude'] > office_lon), 'quadrant'] = 'NE'
    user_df.loc[(user_df['latitude'] > office_lat) &
                (user_df['longitude'] <= office_lon), 'quadrant'] = 'NW'
    user_df.loc[(user_df['latitude'] <= office_lat) &
                (user_df['longitude'] > office_lon), 'quadrant'] = 'SE'
    user_df.loc[(user_df['latitude'] <= office_lat) &
                (user_df['longitude'] <= office_lon), 'quadrant'] = 'SW'

    return user_df


def directional_first_clustering(user_df, office_lat, office_lon):
    """
    ULTRA-STRICT DIRECTIONAL CLUSTERING: Prevents any directional violations
    """
    coords = user_df[['latitude', 'longitude']].values

    # Handle single user case
    if len(user_df) == 1:
        user_df = user_df.copy()
        user_df['directional_cluster'] = 0
        return user_df, {
            'method': 'Single User Assignment',
            'clusters': 1
        }

    # STEP 1: ULTRA-STRICT DIRECTIONAL SECTORS
    bearings = user_df['bearing'].values
    print(f"🧭 ULTRA-STRICT DIRECTIONAL CLUSTERING: {len(user_df)} users")

    # Create bearing-based sectors with STRICT enforcement
    sector_size = MAX_BEARING_DIFFERENCE  # 30 degrees per sector
    n_sectors = int(360 / sector_size)  # 12 sectors of 30 degrees each
    
    # Assign users to sectors based on bearing
    sectors = []
    for i in range(n_sectors):
        sector_start = i * sector_size
        sector_end = (i + 1) * sector_size
        sector_users = []
        
        for idx, bearing in enumerate(bearings):
            # Handle wrap-around at 360/0 degrees
            if sector_start <= bearing < sector_end:
                sector_users.append(idx)
            elif i == n_sectors - 1 and bearing >= sector_end:  # Last sector includes 360
                sector_users.append(idx)
        
        if sector_users:
            sectors.append(sector_users)
    
    # Handle 0-degree wrap-around specially
    if sectors:
        first_sector = sectors[0] if sectors else []
        last_sector = sectors[-1] if len(sectors) > 1 else []
        
        # Check if first and last sectors can be merged (near 0/360 degrees)
        if first_sector and last_sector:
            first_bearing_avg = np.mean([bearings[i] for i in first_sector])
            last_bearing_avg = np.mean([bearings[i] for i in last_sector])
            
            # Calculate wrap-around distance
            wrap_distance = min(
                abs(first_bearing_avg - last_bearing_avg),
                360 - abs(first_bearing_avg - last_bearing_avg)
            )
            
            if wrap_distance <= MAX_BEARING_DIFFERENCE:
                # Merge first and last sectors
                sectors[0].extend(sectors[-1])
                sectors.pop()

    print(f"🎯 STEP 1: Created {len(sectors)} ultra-strict directional sectors")

    # STEP 2: VALIDATE DIRECTIONAL COHERENCE
    # Ensure no cluster has users > MAX_BEARING_DIFFERENCE apart
    refined_clusters = []
    cluster_id_counter = 0

    for cluster_id in np.unique(directional_clusters):
        cluster_mask = user_df['directional_cluster'] == cluster_id
        cluster_users = user_df[cluster_mask]
        cluster_bearings = cluster_users['bearing'].values

        if len(cluster_bearings) <= 1:
            # Single user clusters are perfect
            refined_clusters.extend([(idx, cluster_id_counter) for idx in cluster_users.index])
            cluster_id_counter += 1
            continue

        # Check if this directional cluster is coherent
        max_bearing_spread = 0
        for i in range(len(cluster_bearings)):
            for j in range(i + 1, len(cluster_bearings)):
                spread = bearing_difference(cluster_bearings[i], cluster_bearings[j])
                max_bearing_spread = max(max_bearing_spread, spread)

        print(f"   📐 Cluster {cluster_id}: {len(cluster_users)} users, max spread: {max_bearing_spread:.1f}°")

        if max_bearing_spread <= MAX_BEARING_DIFFERENCE:
            # Cluster is directionally coherent - keep it
            refined_clusters.extend([(idx, cluster_id_counter) for idx in cluster_users.index])
            cluster_id_counter += 1
        else:
            # Split this cluster further by bearing similarity
            if len(cluster_bearings) >= 2:
                # Use hierarchical bearing-based splitting
                n_splits = min(3, len(cluster_bearings))  # Max 3 splits per cluster
                sub_kmeans = KMeans(n_clusters=n_splits, random_state=42)
                sub_bearing_vectors = np.column_stack([
                    np.cos(np.radians(cluster_bearings)),
                    np.sin(np.radians(cluster_bearings))
                ])
                sub_clusters = sub_kmeans.fit_predict(sub_bearing_vectors)

                for sub_id in range(n_splits):
                    sub_mask = sub_clusters == sub_id
                    sub_indices = cluster_users.index[sub_mask]
                    if len(sub_indices) > 0:
                        refined_clusters.extend([(idx, cluster_id_counter) for idx in sub_indices])
                        cluster_id_counter += 1
            else:
                # Single user fallback
                refined_clusters.extend([(idx, cluster_id_counter) for idx in cluster_users.index])
                cluster_id_counter += 1

    # Apply refined directional clustering
    for idx, new_cluster_id in refined_clusters:
        user_df.loc[idx, 'directional_cluster'] = new_cluster_id

    # STEP 3: GEOGRAPHICAL REFINEMENT WITHIN DIRECTIONAL CLUSTERS
    # Now split large directional clusters by geography, but NEVER mix directions
    final_refined_clusters = []
    final_cluster_id = 0

    for cluster_id in user_df['directional_cluster'].unique():
        cluster_users = user_df[user_df['directional_cluster'] == cluster_id]

        if len(cluster_users) <= 6:  # Small clusters stay as-is
            final_refined_clusters.extend([(idx, final_cluster_id) for idx in cluster_users.index])
            final_cluster_id += 1
        else:
            # Large directional cluster - split by geography
            cluster_coords = cluster_users[['latitude', 'longitude']].values

            # Determine number of geographical sub-clusters
            n_geo_splits = min(3, len(cluster_users) // 3)  # Each sub-cluster has ~3+ users
            n_geo_splits = max(1, n_geo_splits)

            if n_geo_splits > 1:
                geo_kmeans = KMeans(n_clusters=n_geo_splits, random_state=42)
                geo_clusters = geo_kmeans.fit_predict(cluster_coords)

                for geo_id in range(n_geo_splits):
                    geo_mask = geo_clusters == geo_id
                    geo_indices = cluster_users.index[geo_mask]
                    if len(geo_indices) > 0:
                        final_refined_clusters.extend([(idx, final_cluster_id) for idx in geo_indices])
                        final_cluster_id += 1
            else:
                final_refined_clusters.extend([(idx, final_cluster_id) for idx in cluster_users.index])
                final_cluster_id += 1

    # Apply final clustering
    for idx, new_cluster_id in final_refined_clusters:
        user_df.loc[idx, 'directional_cluster'] = new_cluster_id

    final_clusters = len(user_df['directional_cluster'].unique())
    print(f"🎯 FINAL: {final_clusters} directionally-coherent geographical clusters")

    # STEP 4: FINAL VALIDATION
    # Verify no cluster violates directional constraints
    validation_passed = True
    for cluster_id in user_df['directional_cluster'].unique():
        cluster_users = user_df[user_df['directional_cluster'] == cluster_id]
        if len(cluster_users) > 1:
            cluster_bearings = cluster_users['bearing'].values
            max_spread = max(bearing_difference(cluster_bearings[i], cluster_bearings[j])
                           for i in range(len(cluster_bearings))
                           for j in range(i + 1, len(cluster_bearings)))

            if max_spread > MAX_BEARING_DIFFERENCE:
                print(f"⚠️ WARNING: Cluster {cluster_id} has bearing spread {max_spread:.1f}° > {MAX_BEARING_DIFFERENCE}°")
                validation_passed = False

    if validation_passed:
        print("✅ All clusters pass directional validation")

    return user_df, {
        'method': 'Directional-First Clustering',
        'clusters': final_clusters,
        'validation_passed': validation_passed
    }


def calculate_route_efficiency_score(user_positions, office_lat, office_lon):
    """
    Calculate a route efficiency score (0-1, higher is better)
    Based on directional coherence and geographical compactness
    """
    if len(user_positions) <= 1:
        return 1.0  # Perfect efficiency for single user

    # Calculate directional coherence
    user_bearings = [calculate_bearing(office_lat, office_lon, pos[0], pos[1]) for pos in user_positions]

    # Convert to unit vectors for coherence calculation
    vectors = [np.array([np.cos(np.radians(b)), np.sin(np.radians(b))]) for b in user_bearings]
    mean_vector = np.mean(vectors, axis=0)
    directional_coherence = np.linalg.norm(mean_vector)  # 0 = scattered, 1 = aligned

    # Calculate geographical compactness
    center = np.mean(user_positions, axis=0)
    distances_from_center = [haversine_distance(center[0], center[1], pos[0], pos[1]) for pos in user_positions]
    avg_distance_from_center = np.mean(distances_from_center)

    # Normalize compactness (assume 10km is very spread out)
    geographical_compactness = max(0, 1 - (avg_distance_from_center / 10.0))

    # Combined efficiency score (directional coherence is more important)
    efficiency_score = (0.7 * directional_coherence) + (0.3 * geographical_compactness)

    return min(1.0, max(0.0, efficiency_score))


def calculate_optimized_cost(route_data, driver_pos, user_positions, office_lat, office_lon):
    """
    COMPLETELY REVAMPED cost function that prioritizes route efficiency
    Heavy penalties for directionally scattered routes
    """
    if not user_positions:
        return float('inf')

    # EFFICIENCY SCORE (Primary factor - 60% weight)
    efficiency_score = calculate_route_efficiency_score(user_positions, office_lat, office_lon)
    efficiency_cost = (1 - efficiency_score) * DIRECTIONAL_SCATTER_PENALTY  # Heavy penalty for poor efficiency

    # DISTANCE COST (Secondary factor - 25% weight)
    centroid = np.mean(user_positions, axis=0)
    distance_to_centroid = haversine_distance(driver_pos[0], driver_pos[1], centroid[0], centroid[1])

    # Average distance from driver to all users
    avg_distance_to_users = np.mean([
        haversine_distance(driver_pos[0], driver_pos[1], pos[0], pos[1])
        for pos in user_positions
    ])

    distance_cost = (distance_to_centroid + avg_distance_to_users) / 2

    # UTILIZATION COST (Tertiary factor - 15% weight)
    capacity = route_data.get('capacity', 1)
    utilization = len(user_positions) / capacity
    utilization_cost = (1 - utilization) * UTILIZATION_PENALTY_PER_SEAT

    # TOTAL ROUTE COMPACTNESS (check total travel distance within route)
    total_internal_distance = 0
    if len(user_positions) > 1:
        for i in range(len(user_positions)):
            for j in range(i + 1, len(user_positions)):
                total_internal_distance += haversine_distance(
                    user_positions[i][0], user_positions[i][1],
                    user_positions[j][0], user_positions[j][1]
                )

        # Average internal distance - penalty for spread out routes
        avg_internal_distance = total_internal_distance / (len(user_positions) * (len(user_positions) - 1) / 2)
        compactness_cost = avg_internal_distance * 0.5  # Moderate penalty for spread
    else:
        compactness_cost = 0

    # WEIGHTED TOTAL COST
    total_cost = (
        efficiency_cost * 0.60 +     # Route efficiency is MOST important
        distance_cost * 0.25 +       # Distance is secondary
        utilization_cost * 0.10 +    # Utilization is tertiary
        compactness_cost * 0.05      # Compactness is minor factor
    )

    # Additional penalty for violating directional constraints
    if len(user_positions) > 1:
        user_bearings = [calculate_bearing(office_lat, office_lon, pos[0], pos[1]) for pos in user_positions]
        max_bearing_spread = max(bearing_difference(user_bearings[i], user_bearings[j])
                               for i in range(len(user_bearings))
                               for j in range(i + 1, len(user_bearings)))

        if max_bearing_spread > MAX_BEARING_DIFFERENCE:
            # MASSIVE penalty for violating directional constraints
            directional_violation_penalty = (max_bearing_spread - MAX_BEARING_DIFFERENCE) * 10.0
            total_cost += directional_violation_penalty

    return total_cost


def priority_driver_assignment(user_df, driver_df):
    """
    ENHANCED PRIORITY-FIRST assignment with strict seat filling and directional coherence
    """
    routes = []
    available_drivers = driver_df.sort_values(['priority', 'capacity'], ascending=[True, False]).copy()
    assigned_user_ids = set()
    used_driver_indices = set()

    print(f"🎯 ENHANCED PRIORITY-FIRST ASSIGNMENT")
    print(f"   Available drivers: {len(available_drivers)}")

    # Process drivers by STRICT priority order
    for priority_level in [1, 2, 3, 4]:
        priority_drivers = available_drivers[
            (available_drivers['priority'] == priority_level) & 
            (~available_drivers.index.isin(used_driver_indices))
        ].copy()

        if priority_drivers.empty or user_df[~user_df['user_id'].isin(assigned_user_ids)].empty:
            continue

        print(f"\n🥇 Processing Priority {priority_level} drivers: {len(priority_drivers)} available")

        # For each priority level, assign drivers to maximize seat utilization
        for _, driver in priority_drivers.iterrows():
            unassigned_users = user_df[~user_df['user_id'].isin(assigned_user_ids)].copy()

            if unassigned_users.empty:
                break

            driver_pos = (driver['latitude'], driver['longitude'])
            capacity = driver['capacity']
            
            # Find the BEST combination of users for maximum utilization
            best_user_combination = None
            best_score = float('inf')
            best_utilization = 0

            # Try each directional cluster
            for cluster_id, cluster_users in unassigned_users.groupby('directional_cluster'):
                # STRICT: Only consider clusters with good directional coherence
                if len(cluster_users) > 1:
                    cluster_bearings = cluster_users['bearing'].values
                    max_spread = max(bearing_difference(cluster_bearings[i], cluster_bearings[j])
                                   for i in range(len(cluster_bearings))
                                   for j in range(i + 1, len(cluster_bearings)))
                
                    if max_spread > MAX_BEARING_DIFFERENCE:
                        print(f"     ❌ Cluster {cluster_id} rejected: bearing spread {max_spread:.1f}° > {MAX_BEARING_DIFFERENCE}°")
                        continue

                # Calculate distance to cluster centroid
                cluster_centroid = (cluster_users['latitude'].mean(), cluster_users['longitude'].mean())
                distance_to_cluster = haversine_distance(driver_pos[0], driver_pos[1], 
                                                           cluster_centroid[0], cluster_centroid[1])

                # Try different combinations to maximize seat utilization
                cluster_size = len(cluster_users)
                
                if cluster_size >= capacity:
                    # Perfect or over-utilization - take best capacity users by distance
                    distances = [haversine_distance(driver_pos[0], driver_pos[1], 
                                                    row['latitude'], row['longitude']) 
                                 for _, row in cluster_users.iterrows()]
                    sorted_indices = sorted(range(len(distances)), key=lambda i: distances[i])
                    users_to_take = cluster_users.iloc[sorted_indices[:capacity]]
                    utilization = 1.0  # Perfect utilization
                    
                else:
                    # Under-utilization - try to fill from nearby clusters
                    users_to_take = cluster_users.copy()
                    
                    # Look for compatible users from nearby clusters within same direction
                    for other_cluster_id, other_cluster in unassigned_users.groupby('directional_cluster'):
                        if other_cluster_id == cluster_id or len(users_to_take) >= capacity:
                            continue
                        
                        # Check if other cluster is in compatible direction
                        if len(other_cluster) > 0:
                            primary_bearing = cluster_users['bearing'].mean()
                            other_bearing = other_cluster['bearing'].mean()
                            bearing_diff = bearing_difference(primary_bearing, other_bearing)
                            
                            if bearing_diff <= MAX_BEARING_DIFFERENCE:
                                # Add compatible users by distance
                                other_distances = [haversine_distance(driver_pos[0], driver_pos[1], 
                                                                        row['latitude'], row['longitude']) 
                                                   for _, row in other_cluster.iterrows()]
                                sorted_other = other_cluster.iloc[sorted(range(len(other_distances)), 
                                                                        key=lambda i: other_distances[i])]
                                
                                slots_remaining = capacity - len(users_to_take)
                                compatible_users = sorted_other.head(min(slots_remaining, len(sorted_other)))
                                
                                # Verify final directional coherence
                                test_users = pd.concat([users_to_take, compatible_users])
                                if len(test_users) > 1:
                                    test_bearings = test_users['bearing'].values
                                    final_spread = max(bearing_difference(test_bearings[i], test_bearings[j])
                                                     for i in range(len(test_bearings))
                                                     for j in range(i + 1, len(test_bearings)))
                                    
                                    if final_spread <= MAX_BEARING_DIFFERENCE:
                                        users_to_take = test_users
                

                # Calculate route quality score
                avg_distance = np.mean([haversine_distance(driver_pos[0], driver_pos[1], 
                                                         row['latitude'], row['longitude']) 
                                      for _, row in users_to_take.iterrows()])
                
                # Score prioritizes: high utilization, low distance, directional coherence
                utilization_bonus = (utilization ** 2) * 10  # Heavily reward high utilization
                distance_penalty = avg_distance
                
                score = distance_penalty - utilization_bonus
                
                if utilization > best_utilization or (utilization == best_utilization and score < best_score):
                    best_score = score
                    best_utilization = utilization
                    best_user_combination = users_to_take

            if best_user_combination is not None and len(best_user_combination) > 0:
                print(f"   ✅ Driver {driver.name} (P{priority_level}): {len(best_user_combination)} users, {best_utilization:.1%} utilization")

                # Create route
                driver_data = driver.to_dict()
                if 'driver_id' not in driver_data and hasattr(driver, 'name'):
                    driver_data['driver_id'] = str(driver.name)

                route = create_route_from_users(
                    driver_data,
                    best_user_combination.to_dict('records'),
                    user_df.iloc[0]['office_latitude'],
                    user_df.iloc[0]['office_longitude'])

                # Track assigned users
                for _, user in best_user_combination.iterrows():
                    assigned_user_ids.add(user['user_id'])

                routes.append(route)
                used_driver_indices.add(driver.name)

    print(f"\n🎯 ENHANCED ASSIGNMENT COMPLETE:")
    print(f"   Routes created: {len(routes)}")
    print(f"   Users assigned: {len(assigned_user_ids)}")
    print(f"   Average utilization: {np.mean([len(r['assigned_users'])/r['vehicle_type'] for r in routes]):.1%}")

    return routes, assigned_user_ids


def optimal_cluster_assignment(user_df, driver_df):
    """
    OPTIMAL cluster assignment that respects both priority and route efficiency
    """
    routes = []
    available_drivers = driver_df.sort_values(['priority', 'capacity'], ascending=[True, False]).copy()
    assigned_user_ids = set()
    used_driver_indices = set()

    print(f"🎯 OPTIMAL CLUSTER ASSIGNMENT")

    # Process each directional cluster
    for cluster_id, cluster_users in user_df.groupby('directional_cluster'):
        unassigned_in_cluster = cluster_users[~cluster_users['user_id'].isin(assigned_user_ids)].copy()

        if unassigned_in_cluster.empty or len(used_driver_indices) >= len(available_drivers):
            continue

        print(f"\n🧭 Processing directional cluster {cluster_id}: {len(unassigned_in_cluster)} users")

        # For each cluster, create optimal sub-routes based on capacity constraints
        available_driver_indices = [i for i in available_drivers.index if i not in used_driver_indices]
        remaining_drivers = available_drivers.loc[available_driver_indices]

        if remaining_drivers.empty:
            print(f"   ⚠️ No drivers available for cluster {cluster_id}")
            continue

        # Determine how many sub-routes we need for this cluster
        total_users = len(unassigned_in_cluster)
        avg_capacity = remaining_drivers['capacity'].mean()
        max_capacity = remaining_drivers['capacity'].max()

        # Calculate optimal number of routes for this cluster
        if total_users <= max_capacity:
            target_routes = 1
        else:
            target_routes = min(
                math.ceil(total_users / avg_capacity),
                len(available_driver_indices),
                math.ceil(total_users / 6)  # Max 6 users per route for efficiency
            )

        print(f"   📊 Target routes for cluster: {target_routes}")

        # Create geographically tight sub-clusters within this directional cluster
        if target_routes > 1 and len(unassigned_in_cluster) > 1:
            coords = unassigned_in_cluster[['latitude', 'longitude']].values
            actual_subclusters = min(target_routes, len(unassigned_in_cluster))

            if actual_subclusters > 1:
                kmeans = KMeans(n_clusters=actual_subclusters, random_state=42)
                unassigned_in_cluster['sub_cluster'] = kmeans.fit_predict(coords)
            else:
                unassigned_in_cluster['sub_cluster'] = 0
        else:
            unassigned_in_cluster['sub_cluster'] = 0

        # Process each sub-cluster to find the optimal driver assignment
        for sub_id, sub_users in unassigned_in_cluster.groupby('sub_cluster'):
            if sub_users.empty or len(used_driver_indices) >= len(available_drivers):
                continue

            user_positions = [(row['latitude'], row['longitude']) for _, row in sub_users.iterrows()]
            users_needed = len(sub_users)

            print(f"     📍 Sub-cluster {sub_id}: {users_needed} users")

            # Find the OPTIMAL driver for this specific user group
            best_driver_idx = None
            best_cost = float('inf')
            best_efficiency_score = 0

            available_driver_indices_current = [i for i in available_drivers.index if i not in used_driver_indices]

            # Evaluate ALL available drivers for this route to find the optimal one
            driver_candidates = []

            for driver_idx in available_driver_indices_current:
                driver = available_drivers.loc[driver_idx]
                driver_pos = (driver['latitude'], driver['longitude'])

                # Calculate route efficiency if this driver takes these users
                route_efficiency = calculate_route_efficiency_score(user_positions,
                                                                  user_df.iloc[0]['office_latitude'],
                                                                  user_df.iloc[0]['office_longitude'])

                # Calculate total cost with new optimized cost function
                mock_route = {'capacity': driver['capacity']}
                total_cost = calculate_optimized_cost(mock_route, driver_pos, user_positions,
                                                    user_df.iloc[0]['office_latitude'],
                                                    user_df.iloc[0]['office_longitude'])

                # Apply capacity penalty if needed
                if users_needed > driver['capacity']:
                    total_cost += OVERFLOW_PENALTY_KM

                driver_candidates.append({
                    'driver_idx': driver_idx,
                    'driver': driver,
                    'cost': total_cost,
                    'efficiency': route_efficiency,
                    'priority': driver['priority'],
                    'capacity_fit': users_needed <= driver['capacity']
                })

            # Sort candidates by route optimality FIRST, then by driver priority
            driver_candidates.sort(key=lambda x: (
                not x['capacity_fit'],     # Drivers with sufficient capacity first
                x['cost'],                 # Then by lowest cost (best route efficiency)
                x['priority'],             # Then by driver priority (lower = better)
                -x['efficiency']           # Then by efficiency score (higher = better)
            ))

            if driver_candidates:
                best_candidate = driver_candidates[0]
                best_driver_idx = best_candidate['driver_idx']
                best_driver = best_candidate['driver']
                best_cost = best_candidate['cost']
                best_efficiency = best_candidate['efficiency']

                print(f"       ✅ Selected Driver {best_driver.name}: Priority {best_driver['priority']}, "
                      f"Cost {best_cost:.2f}, Efficiency {best_efficiency:.2f}")

                # Check if we should reject this assignment due to poor route quality
                if best_efficiency < ROUTE_EFFICIENCY_THRESHOLD and len(driver_candidates) > 1:
                    print(f"       ⚠️ Route efficiency {best_efficiency:.2f} below threshold {ROUTE_EFFICIENCY_THRESHOLD}")
                    # Try next best driver if available
                    for alt_candidate in driver_candidates[1:3]:  # Try up to 2 alternatives
                        if alt_candidate['efficiency'] > best_efficiency * 1.2:  # At least 20% better
                            print(f"       🔄 Switching to Driver {alt_candidate['driver'].name} for better efficiency")
                            best_candidate = alt_candidate
                            best_driver_idx = alt_candidate['driver_idx']
                            best_driver = alt_candidate['driver']
                            break

                # Create the route
                capacity = best_driver['capacity']
                users_to_assign = sub_users.head(capacity)  # Assign up to capacity

                # Ensure driver data has driver_id
                driver_data = best_driver.to_dict()
                # Ensure driver_id is properly set
                if 'driver_id' not in driver_data and hasattr(best_driver, 'name'):
                    driver_data['driver_id'] = str(best_driver.name)

                route = create_route_from_users(
                    driver_data,
                    users_to_assign.to_dict('records'),
                    user_df.iloc[0]['office_latitude'],
                    user_df.iloc[0]['office_longitude'])

                # Track assigned users
                for _, user in users_to_assign.iterrows():
                    assigned_user_ids.add(user['user_id'])

                routes.append(route)
                used_driver_indices.add(best_driver_idx)

                print(f"       ✅ Route created: {len(users_to_assign)} users assigned")

    print(f"\n🎯 OPTIMAL ASSIGNMENT COMPLETE:")
    print(f"   Routes created: {len(routes)}")
    print(f"   Users assigned: {len(assigned_user_ids)}")
    print(f"   Drivers used: {len(used_driver_indices)}")

    return routes, assigned_user_ids


def create_route_from_users(driver, users, office_lat, office_lon):
    """Create a route object from driver and assigned users"""
    route = {
        'driver_id': str(driver['driver_id']),
        'vehicle_id': str(driver.get('vehicle_id', '')),
        'vehicle_type': int(driver['capacity']),
        'latitude': float(driver['latitude']),
        'longitude': float(driver['longitude']),
        'assigned_users': []
    }

    for user in users:
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

    # Calculate route metrics including efficiency score
    if route['assigned_users']:
        lats = [u['lat'] for u in route['assigned_users']]
        lngs = [u['lng'] for u in route['assigned_users']]
        route['centroid'] = [np.mean(lats), np.mean(lngs)]
        route['utilization'] = len(route['assigned_users']) / route['vehicle_type']
        route['stops'] = [[u['lat'], u['lng']] for u in route['assigned_users']]
        route['bearing'] = calculate_bearing(route['centroid'][0], route['centroid'][1], office_lat, office_lon)

        # Calculate and store route efficiency
        user_positions = [(u['lat'], u['lng']) for u in route['assigned_users']]
        route['efficiency_score'] = calculate_route_efficiency_score(user_positions, office_lat, office_lon)
    else:
        route['centroid'] = [route['latitude'], route['longitude']]
        route['utilization'] = 0
        route['stops'] = []
        route['bearing'] = 0
        route['efficiency_score'] = 1.0

    return route


def optimized_local_search(routes, max_iterations=MAX_SWAP_ITERATIONS):
            """
            Enhanced local search that maintains directional coherence
            Only allows swaps that improve overall route efficiency
            """
            print("🔧 Starting efficiency-focused local search...")

            improved = True
            iterations = 0

            while improved and iterations < max_iterations:
                improved = False
                iterations += 1

                for i, route_a in enumerate(routes):
                    for j, route_b in enumerate(routes[i + 1:], start=i + 1):
                        if not route_a['assigned_users'] or not route_b['assigned_users']:
                            continue

                        # Try swapping users between routes
                        for user_a in route_a['assigned_users'][:]:
                            for user_b in route_b['assigned_users'][:]:
                                # Calculate current efficiency scores
                                current_eff_a = route_a.get('efficiency_score', 0)
                                current_eff_b = route_b.get('efficiency_score', 0)
                                current_total_efficiency = current_eff_a + current_eff_b

                                # Simulate swap and calculate new efficiency
                                new_efficiency = calculate_swap_efficiency_improvement(
                                    route_a, route_b, user_a, user_b)

                                # OLD CODE APPROACH: Calculate actual distance improvement
                                current_dist_a = haversine_distance(route_a['centroid'][0], route_a['centroid'][1], 
                                                                   user_a['lat'], user_a['lng'])
                                current_dist_b = haversine_distance(route_b['centroid'][0], route_b['centroid'][1], 
                                                                   user_b['lat'], user_b['lng'])

                                swap_dist_a = haversine_distance(route_a['centroid'][0], route_a['centroid'][1], 
                                                                user_b['lat'], user_b['lng'])
                                swap_dist_b = haversine_distance(route_b['centroid'][0], route_b['centroid'][1], 
                                                                user_a['lat'], user_a['lng'])

                                distance_improvement = (current_dist_a + current_dist_b) - (swap_dist_a + swap_dist_b)

                                # Swap if improves distance by at least 200m (like old code) AND efficiency is better
                                if distance_improvement >= 0.2 and new_efficiency > current_total_efficiency * 0.95:
                                    # Verify swap doesn't violate directional constraints
                                    if validate_swap_directional_coherence(route_a, route_b, user_a, user_b):
                                        # Perform swap
                                        route_a['assigned_users'].remove(user_a)
                                        route_b['assigned_users'].remove(user_b)
                                        route_a['assigned_users'].append(user_b)
                                        route_b['assigned_users'].append(user_a)

                                        # Update route metrics
                                        update_route_metrics([route_a, route_b])
                                        improved = True
                                        print(f"  🔄 Beneficial swap performed (efficiency: {current_total_efficiency:.2f} → {new_efficiency:.2f})")
                                        break
                            if improved:
                                break
                        if improved:
                            break
                    if improved:
                        break

                if improved:
                    print(f"  🔄 Improvement found in iteration {iterations}")

            print(f"✅ Efficiency-focused local search complete after {iterations} iterations")
            return routes


def calculate_swap_efficiency_improvement(route_a, route_b, user_a, user_b):
            """Calculate the efficiency improvement from swapping two users"""
            # Get current user positions
            users_a = [u for u in route_a['assigned_users'] if u != user_a] + [user_b]
            users_b = [u for u in route_b['assigned_users'] if u != user_b] + [user_a]

            positions_a = [(u['lat'], u['lng']) for u in users_a]
            positions_b = [(u['lat'], u['lng']) for u in users_b]

            # Calculate efficiency scores for swapped routes
            eff_a = calculate_route_efficiency_score(positions_a, OFFICE_LAT, OFFICE_LON)
            eff_b = calculate_route_efficiency_score(positions_b, OFFICE_LAT, OFFICE_LON)

            return eff_a + eff_b


def validate_swap_directional_coherence(route_a, route_b, user_a, user_b):
            """Validate that a swap doesn't violate directional coherence"""
            # Check if swap maintains directional coherence in both routes
            users_a_after = [u for u in route_a['assigned_users'] if u != user_a] + [user_b]
            users_b_after = [u for u in route_b['assigned_users'] if u != user_b] + [user_a]

            # Check directional spread in route A after swap
            if len(users_a_after) > 1:
                bearings_a = [calculate_bearing(OFFICE_LAT, OFFICE_LON, u['lat'], u['lng']) for u in users_a_after]
                max_spread_a = max(bearing_difference(bearings_a[i], bearings_a[j])
                                  for i in range(len(bearings_a))
                                  for j in range(i + 1, len(bearings_a)))
                if max_spread_a > MAX_BEARING_DIFFERENCE:
                    return False

            # Check directional spread in route B after swap
            if len(users_b_after) > 1:
                bearings_b = [calculate_bearing(OFFICE_LAT, OFFICE_LON, u['lat'], u['lng']) for u in users_b_after]
                max_spread_b = max(bearing_difference(bearings_b[i], bearings_b[j])
                                  for i in range(len(bearings_b))
                                  for j in range(i + 1, len(bearings_b)))
                if max_spread_b > MAX_BEARING_DIFFERENCE:
                    return False

            return True


def update_route_metrics(routes):
            """Update route metrics after modifications"""
            for route in routes:
                if route['assigned_users']:
                    lats = [u['lat'] for u in route['assigned_users']]
                    lngs = [u['lng'] for u in route['assigned_users']]
                    route['centroid'] = [np.mean(lats), np.mean(lngs)]
                    route['utilization'] = len(route['assigned_users']) / route['vehicle_type']
                    route['stops'] = [[u['lat'], u['lng']] for u in route['assigned_users']]
                    route['bearing'] = calculate_bearing(route['centroid'][0], route['centroid'][1], OFFICE_LAT, OFFICE_LON)

                    # Update efficiency score
                    user_positions = [(u['lat'], u['lng']) for u in route['assigned_users']]
                    route['efficiency_score'] = calculate_route_efficiency_score(user_positions, OFFICE_LAT, OFFICE_LON)
                else:
                    route['centroid'] = [route['latitude'], route['longitude']]
                    route['utilization'] = 0
                    route['stops'] = []
                    route['bearing'] = 0
                    route['efficiency_score'] = 1.0


def strict_fill_routes(routes, user_df, assigned_user_ids):
            """
            STRICT route filling that maintains directional coherence
            Only adds users that improve or maintain route efficiency
            """
            unassigned_users = user_df[~user_df['user_id'].isin(assigned_user_ids)].copy()

            if unassigned_users.empty:
                return assigned_user_ids

            print(f"🔍 Strict filling: {len(unassigned_users)} unassigned users")

            for route in routes:
                capacity = route['vehicle_type']
                current_size = len(route['assigned_users'])

                if current_size < capacity and not unassigned_users.empty:
                    if route['assigned_users']:
                        centroid_lat = np.mean([u['lat'] for u in route['assigned_users']])
                        centroid_lng = np.mean([u['lng'] for u in route['assigned_users']])
                        current_efficiency = route.get('efficiency_score', 0)
                    else:
                        centroid_lat, centroid_lng = route['latitude'], route['longitude']
                        current_efficiency = 1.0

                    # Find compatible users within distance who improve efficiency
                    compatible_users = []

                    for _, user in unassigned_users.iterrows():
                        dist = haversine_distance(centroid_lat, centroid_lng, user['latitude'], user['longitude'])

                        if dist <= MAX_FILL_DISTANCE_KM:
                            # Test if adding this user maintains directional coherence
                            test_positions = [(u['lat'], u['lng']) for u in route['assigned_users']]
                            test_positions.append((user['latitude'], user['longitude']))

                            # Check directional coherence
                            if len(test_positions) > 1:
                                test_bearings = [calculate_bearing(OFFICE_LAT, OFFICE_LON, pos[0], pos[1]) for pos in test_positions]
                                max_spread = max(bearing_difference(test_bearings[i], test_bearings[j])
                                               for i in range(len(test_bearings))
                                               for j in range(i + 1, len(test_bearings)))

                                if max_spread > MAX_BEARING_DIFFERENCE:
                                    continue  # Skip users that violate directional coherence

                            # Calculate efficiency with this user added
                            new_efficiency = calculate_route_efficiency_score(test_positions, OFFICE_LAT, OFFICE_LON)

                            # Only add if efficiency is maintained or improved
                            if new_efficiency >= current_efficiency * 0.85:  # Allow 15% efficiency drop for better utilization
                                compatible_users.append((user, dist, new_efficiency))

                    # Sort by efficiency improvement, then by distance
                    compatible_users.sort(key=lambda x: (-x[2], x[1]))  # Best efficiency first, then closest

                    slots_available = capacity - current_size
                    for user, dist, new_efficiency in compatible_users[:slots_available]:
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
                        assigned_user_ids.add(user['user_id'])
                        unassigned_users = unassigned_users[unassigned_users['user_id'] != user['user_id']]

                        # Update route efficiency
                        user_positions = [(u['lat'], u['lng']) for u in route['assigned_users']]
                        route['efficiency_score'] = calculate_route_efficiency_score(user_positions, OFFICE_LAT, OFFICE_LON)

                        print(f"   ✅ Added user {user['user_id']} to route (efficiency: {route['efficiency_score']:.2f})")

            return assigned_user_ids


def reassign_underutilized_routes(routes):
            """
            Reassign users from underutilized routes to nearby better routes (from old code)
            """
            routes_to_remove = []
            reassigned_count = 0

            for i, source_route in enumerate(routes):
                if source_route.get('utilization', 0) >= MIN_UTIL_THRESHOLD:
                    continue

                source_bearing = source_route.get('bearing', 0)
                remaining_users = []

                for user in source_route['assigned_users']:
                    best_target = None
                    best_cost = float('inf')

                    for j, target_route in enumerate(routes):
                        if (j == i or 
                            len(target_route['assigned_users']) >= target_route['vehicle_type']):
                            continue

                        centroid = target_route.get('centroid', [target_route['latitude'], target_route['longitude']])
                        distance = haversine_distance(centroid[0], centroid[1], user['lat'], user['lng'])
                        target_bearing = target_route.get('bearing', 0)
                        bearing_diff = bearing_difference(source_bearing, target_bearing)

                        # OLD CODE LOGIC: Use distance and bearing for reassignment
                        if distance <= MERGE_DISTANCE_KM and bearing_diff <= 45:
                            cost = distance + (bearing_diff / 90.0)  # Normalize bearing difference
                            if cost < best_cost:
                                best_cost = cost
                                best_target = target_route

                    if best_target:
                        best_target['assigned_users'].append(user)
                        reassigned_count += 1
                    else:
                        remaining_users.append(user)

                if not remaining_users:
                    routes_to_remove.append(i)
                else:
                    source_route['assigned_users'] = remaining_users

            # Remove empty routes
            for i in sorted(routes_to_remove, reverse=True):
                routes.pop(i)

            print(f"📈 Reassigned {reassigned_count} users from underutilized routes")
            return routes


def handle_remaining_users_optimally(user_df, routes, assigned_user_ids, driver_df):
            """Handle remaining users while maintaining optimal route quality"""
            unassigned_users = user_df[~user_df['user_id'].isin(assigned_user_ids)].copy()

            # Get assigned driver IDs from routes
            assigned_driver_ids = {route['driver_id'] for route in routes}

            # Filter available drivers - check if driver_id is column or index
            if 'driver_id' in driver_df.columns:
                available_drivers = driver_df[~driver_df['driver_id'].isin(assigned_driver_ids)].copy()
            else:
                # If driver_id is the index, use index-based filtering
                available_drivers = driver_df[~driver_df.index.astype(str).isin(assigned_driver_ids)].copy()

            if unassigned_users.empty:
                return []

            print(f"🎯 Handling {len(unassigned_users)} remaining users optimally")

            # First try to fit into existing routes (only if it maintains efficiency)
            for _, user in unassigned_users.copy().iterrows():
                best_route = None
                best_efficiency_loss = float('inf')

                for route in routes:
                    if len(route['assigned_users']) < route['vehicle_type']:
                        # Test adding this user
                        test_positions = [(u['lat'], u['lng']) for u in route['assigned_users']]
                        test_positions.append((user['latitude'], user['longitude']))

                        # Check directional coherence
                        if len(test_positions) > 1:
                            test_bearings = [calculate_bearing(OFFICE_LAT, OFFICE_LON, u['lat'], u['lng']) for u in test_positions]
                            max_spread = max(bearing_difference(test_bearings[i], test_bearings[j])
                                           for i in range(len(test_bearings))
                                           for j in range(i + 1, len(test_bearings)))

                            if max_spread > MAX_BEARING_DIFFERENCE:
                                continue  # Skip routes that would violate directional coherence

                        current_efficiency = route.get('efficiency_score', 0)
                        new_efficiency = calculate_route_efficiency_score(test_positions, OFFICE_LAT, OFFICE_LON)
                        efficiency_loss = current_efficiency - new_efficiency

                        # Check distance compatibility
                        if route['assigned_users']:
                            centroid_lat = np.mean([u['lat'] for u in route['assigned_users']])
                            centroid_lng = np.mean([u['lng'] for u in route['assigned_users']])
                            dist = haversine_distance(centroid_lat, centroid_lng, user['latitude'], user['longitude'])

                            if dist <= MAX_FILL_DISTANCE_KM and efficiency_loss < best_efficiency_loss and efficiency_loss <= 0.1:
                                best_efficiency_loss = efficiency_loss
                                best_route = route

                if best_route:
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

                    best_route['assigned_users'].append(user_data)
                    assigned_user_ids.add(user['user_id'])
                    unassigned_users = unassigned_users[unassigned_users['user_id'] != user['user_id']]

                    # Update route efficiency
                    user_positions = [(u['lat'], u['lng']) for u in best_route['assigned_users']]
                    best_route['efficiency_score'] = calculate_route_efficiency_score(user_positions, OFFICE_LAT, OFFICE_LON)

            # Create new efficient routes for remaining isolated users
            remaining_unassigned = user_df[~user_df['user_id'].isin(assigned_user_ids)].copy()

            if not remaining_unassigned.empty and not available_drivers.empty:
                # Re-cluster remaining users directionally
                if len(remaining_unassigned) > 1:
                    remaining_bearings = [calculate_bearing(OFFICE_LAT, OFFICE_LON, row['latitude'], row['longitude'])
                                        for _, row in remaining_unassigned.iterrows()]

                    # Create small directional clusters
                    n_clusters = min(len(available_drivers), max(1, len(remaining_unassigned) // 2))

                    if n_clusters > 1:
                        bearing_vectors = np.column_stack([
                            np.cos(np.radians(remaining_bearings)),
                            np.sin(np.radians(remaining_bearings))
                        ])
                        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                        remaining_unassigned['final_cluster'] = kmeans.fit_predict(bearing_vectors)
                    else:
                        remaining_unassigned['final_cluster'] = 0
                else:
                    remaining_unassigned['final_cluster'] = 0

                # Convert available drivers to list of dictionaries
                available_drivers_list = []
                for idx, driver in available_drivers.iterrows():
                    driver_dict = driver.to_dict()
                    # Ensure driver_id is in the dictionary
                    if 'driver_id' not in driver_dict:
                        driver_dict['driver_id'] = str(idx)
                    available_drivers_list.append(driver_dict)

                for cluster_id, cluster_users in remaining_unassigned.groupby('final_cluster'):
                    if not available_drivers_list:
                        break

                    # Find the best driver for this cluster
                    cluster_positions = [(row['latitude'], row['longitude']) for _, row in cluster_users.iterrows()]
                    cluster_centroid = np.mean(cluster_positions, axis=0)

                    best_driver = min(available_drivers_list,
                                    key=lambda d: haversine_distance(cluster_centroid[0], cluster_centroid[1],
                                                                   d['latitude'], d['longitude']))

                    capacity = best_driver['capacity']
                    users_to_assign = cluster_users.head(capacity)

                    route = create_route_from_users(
                        best_driver, users_to_assign.to_dict('records'),
                        OFFICE_LAT,
                        OFFICE_LON)

                    for _, user in users_to_assign.iterrows():
                        assigned_user_ids.add(user['user_id'])

                    routes.append(route)
                    available_drivers_list.remove(best_driver)

            # Return final unassigned users
            final_unassigned = user_df[~user_df['user_id'].isin(assigned_user_ids)].copy()
            unassigned_list = []

            for _, user in final_unassigned.iterrows():
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

            return unassigned_list


def smart_route_merging_with_path_optimization(routes):
            """
            SMART route merging that considers route paths and bearing corridors
            This addresses the issue where users could be merged based on route efficiency rather than just distance
            """
            print(f"🌍 Starting SMART route merging with path optimization...")

            # Load config values
            global_merge_distance = _config.get('global_merge_distance_km', 8.0)
            bearing_corridor_threshold = 75  # Expanded to 75 degrees for better same-direction merging
            route_path_efficiency_threshold = 0.6  # Slightly relaxed for better utilization

            merged_routes = []
            routes_to_merge = routes.copy()

            # Sort routes by utilization (prioritize low utilization routes for merging)
            routes_to_merge.sort(key=lambda r: r['utilization'])

            while routes_to_merge:
                current_route = routes_to_merge.pop(0)

                print(f"   🔍 Analyzing route {current_route['driver_id']} (utilization: {current_route['utilization']:.1%})")

                # Find merge candidates using multiple criteria
                merge_candidates = []

                for i, other_route in enumerate(routes_to_merge):
                    combined_users = len(current_route['assigned_users']) + len(other_route['assigned_users'])
                    max_capacity = max(current_route['vehicle_type'], other_route['vehicle_type'])

                    if combined_users <= max_capacity:
                        # Get route bearings and positions
                        current_bearing = current_route.get('bearing', 0)
                        other_bearing = other_route.get('bearing', 0)
                        bearing_diff = bearing_difference(current_bearing, other_bearing)

                        # Check if routes are in similar bearing corridors
                        in_bearing_corridor = bearing_diff <= bearing_corridor_threshold

                        # Calculate geographical compatibility
                        current_centroid = current_route['centroid']
                        other_centroid = other_route['centroid']
                        centroid_distance = haversine_distance(current_centroid[0], current_centroid[1],
                                                             other_centroid[0], other_centroid[1])

                        # Add bearing-based merging logic from old code
                        bearing_compatibility = bearing_diff <= 45  # Similar to old code's 45-degree threshold

                        # Calculate route path efficiency if merged
                        all_users = current_route['assigned_users'] + other_route['assigned_users']
                        user_positions = [(u['lat'], u['lng']) for u in all_users]
                        merged_efficiency = calculate_route_efficiency_score(user_positions, OFFICE_LAT, OFFICE_LON)

                        # Check directional coherence of merged route
                        user_bearings = [calculate_bearing(OFFICE_LAT, OFFICE_LON, u['lat'], u['lng']) for u in all_users]
                        max_bearing_spread = 0
                        if len(user_bearings) > 1:
                            max_bearing_spread = max(bearing_difference(user_bearings[i], user_bearings[j])
                                                   for i in range(len(user_bearings))
                                                   for j in range(i + 1, len(user_bearings)))

                        # NEW: Route path optimization logic
                        # Consider merging if:
                        # 1. Routes are in similar bearing corridors AND geographically compatible, OR
                        # 2. Merged route has good efficiency even if not perfectly aligned, OR  
                        # 3. One route has very low utilization and merging improves overall efficiency

                        merge_score = 0
                        merge_reasons = []

                        # Criterion 1: Similar bearing corridors + geographical proximity
                        if in_bearing_corridor and centroid_distance <= global_merge_distance:
                            merge_score += 3
                            merge_reasons.append(f"bearing_corridor({bearing_diff:.1f}°)")

                        # Criterion 2: High efficiency even with wider spread (path optimization)
                        if merged_efficiency >= route_path_efficiency_threshold and max_bearing_spread <= bearing_corridor_threshold:
                            merge_score += 2
                            merge_reasons.append(f"path_efficiency({merged_efficiency:.2f})")

                        # Criterion 3: Low utilization improvement with expanded directional tolerance
                        min_utilization = min(current_route['utilization'], other_route['utilization'])
                        if min_utilization < 0.5 and merged_efficiency >= 0.5 and max_bearing_spread <= bearing_corridor_threshold:
                            merge_score += 2
                            merge_reasons.append(f"low_util_improvement({min_utilization:.1%})")

                        # OLD CODE CRITERION: Smart distance + bearing combination
                        if centroid_distance <= MERGE_DISTANCE_KM and bearing_compatibility:
                            merge_score += 2
                            merge_reasons.append(f"distance_bearing({centroid_distance:.1f}km,{bearing_diff:.1f}°)")

                        # Criterion 4: Same-direction optimization (new criterion for 75-degree tolerance)
                        if max_bearing_spread <= bearing_corridor_threshold and merged_efficiency >= 0.5:
                            merge_score += 2
                            merge_reasons.append(f"same_direction({max_bearing_spread:.1f}°)")

                        # Criterion 5: Smart distance consideration (relaxed for good paths)
                        if centroid_distance <= global_merge_distance * 1.5 and merged_efficiency >= 0.7:
                            merge_score += 1
                            merge_reasons.append(f"extended_range({centroid_distance:.1f}km)")

                        # Only consider routes with sufficient merge score and within expanded bearing tolerance
                        if merge_score >= 2 and max_bearing_spread <= bearing_corridor_threshold:
                            merge_candidates.append({
                                'index': i,
                                'route': other_route,
                                'score': merge_score,
                                'distance': centroid_distance,
                                'efficiency': merged_efficiency,
                                'bearing_spread': max_bearing_spread,
                                'reasons': merge_reasons
                            })

                if merge_candidates:
                    # Sort by merge score (descending), then by efficiency (descending), then by distance (ascending)
                    merge_candidates.sort(key=lambda x: (-x['score'], -x['efficiency'], x['distance']))
                    best_candidate = merge_candidates[0]

                    merge_route = best_candidate['route']
                    merge_idx = best_candidate['index']

                    print(f"   ✅ SMART MERGE: Route {current_route.get('driver_id', 'Unknown')} + {merge_route.get('driver_id', 'Unknown')}")
                    print(f"      Reasons: {', '.join(best_candidate['reasons'])}")
                    print(f"      Efficiency: {best_candidate['efficiency']:.3f}, Spread: {best_candidate['bearing_spread']:.1f}°")

                    # Use the higher capacity vehicle for the merged route
                    if current_route['vehicle_type'] >= merge_route['vehicle_type']:
                        primary_route = current_route
                        secondary_route = merge_route
                    else:
                        primary_route = merge_route
                        secondary_route = current_route

                    # Merge users
                    primary_route['assigned_users'].extend(secondary_route['assigned_users'])

                    # Update route metrics
                    if primary_route['assigned_users']:
                        lats = [u['lat'] for u in primary_route['assigned_users']]
                        lngs = [u['lng'] for u in primary_route['assigned_users']]
                        primary_route['centroid'] = [np.mean(lats), np.mean(lngs)]
                        primary_route['utilization'] = len(primary_route['assigned_users']) / primary_route['vehicle_type']
                        primary_route['stops'] = [[u['lat'], u['lng']] for u in primary_route['assigned_users']]
                        primary_route['bearing'] = calculate_bearing(primary_route['centroid'][0], 
                                                                   primary_route['centroid'][1], OFFICE_LAT, OFFICE_LON)

                        # Update efficiency score
                        user_positions = [(u['lat'], u['lng']) for u in primary_route['assigned_users']]
                        primary_route['efficiency_score'] = calculate_route_efficiency_score(user_positions, OFFICE_LAT, OFFICE_LON)

                        # Update bearing spread
                        user_bearings = [calculate_bearing(OFFICE_LAT, OFFICE_LON, u['lat'], u['lng']) for u in primary_route['assigned_users']]
                        if len(user_bearings) > 1:
                            primary_route['bearing_spread'] = max(bearing_difference(user_bearings[i], user_bearings[j])
                                                                 for i in range(len(user_bearings))
                                                                 for j in range(i + 1, len(user_bearings)))
                            primary_route['directional_coherence'] = primary_route['bearing_spread'] <= MAX_BEARING_DIFFERENCE
                        else:
                            primary_route['bearing_spread'] = 0
                            primary_route['directional_coherence'] = True

                    print(f"      Result: {len(primary_route['assigned_users'])} users, {primary_route['utilization']:.1%} utilization")

                    # Remove the merged route
                    routes_to_merge.pop(merge_idx)
                    merged_routes.append(primary_route)
                else:
                    merged_routes.append(current_route)

            print(f"🌍 SMART merging complete: {len(routes)} → {len(merged_routes)} routes")
            return merged_routes


def finalize_routes_with_efficiency(routes, office_lat, office_lon):
            """Finalize routes with efficiency scores and validation"""
            unique_routes = []
            driver_ids_seen = set()

            for route in routes:
                driver_id = route['driver_id']
                if driver_id in driver_ids_seen:
                    print(f"⚠️ WARNING: Duplicate driver {driver_id} detected and removed!")
                    continue

                driver_ids_seen.add(driver_id)

                # Update all route metrics including efficiency
                if route['assigned_users']:
                    lats = [u['lat'] for u in route['assigned_users']]
                    lngs = [u['lng'] for u in route['assigned_users']]
                    route['centroid'] = [np.mean(lats), np.mean(lngs)]
                    route['utilization'] = len(route['assigned_users']) / route['vehicle_type']
                    route['stops'] = [[u['lat'], u['lng']] for u in route['assigned_users']]
                    route['bearing'] = calculate_bearing(route['centroid'][0], route['centroid'][1], office_lat, office_lon)

                    # Calculate final efficiency score
                    user_positions = [(u['lat'], u['lng']) for u in route['assigned_users']]
                    route['efficiency_score'] = calculate_route_efficiency_score(user_positions, office_lat, office_lon)

                    # Validate directional coherence
                    if len(route['assigned_users']) > 1:
                        user_bearings = [calculate_bearing(office_lat, office_lon, u['lat'], u['lng']) for u in route['assigned_users']]
                        max_spread = max(bearing_difference(user_bearings[i], user_bearings[j])
                                       for i in range(len(user_bearings))
                                       for j in range(i + 1, len(user_bearings)))
                        route['bearing_spread'] = max_spread
                        route['directional_coherence'] = max_spread <= MAX_BEARING_DIFFERENCE
                    else:
                        route['bearing_spread'] = 0
                        route['directional_coherence'] = True
                else:
                    route['centroid'] = [route['latitude'], route['longitude']]
                    route['utilization'] = 0
                    route['stops'] = []
                    route['bearing'] = 0
                    route['efficiency_score'] = 1.0
                    route['bearing_spread'] = 0
                    route['directional_coherence'] = True

                unique_routes.append(route)

            # Print efficiency summary
            if unique_routes:
                avg_efficiency = np.mean([r['efficiency_score'] for r in unique_routes])
                coherent_routes = sum(1 for r in unique_routes if r['directional_coherence'])
                print(f"📊 FINAL ROUTE QUALITY:")
                print(f"   Average efficiency score: {avg_efficiency:.3f}")
                print(f"   Directionally coherent routes: {coherent_routes}/{len(unique_routes)}")
                print(f"   Routes below efficiency threshold: {sum(1 for r in unique_routes if r['efficiency_score'] < ROUTE_EFFICIENCY_THRESHOLD)}")

            return unique_routes


        # MAIN OPTIMIZED ASSIGNMENT FUNCTION
def run_assignment(source_id: str, parameter: int = 1, string_param: str = ""):
            """
            COMPLETELY OPTIMIZED assignment that prioritizes route efficiency above all else
            """
            start_time = time.time()

            try:
                print(f"🚀 Starting ROUTE-OPTIMIZED assignment for source_id: {source_id}")
                print(f"🎯 PRIMARY GOAL: Optimal directional routes")
                print(f"🔧 SECONDARY GOAL: Driver priority and capacity utilization")

                # Load and validate data
                data = load_env_and_fetch_data(source_id, parameter, string_param)

                # EDGE CASE: No users
                users = data.get('users', [])
                if not users:
                    print("⚠️ No users found - returning empty assignment")
                    return {
                        "status": "true",
                        "execution_time": time.time() - start_time,
                        "data": [],
                        "unassignedUsers": [],
                        "unassignedDrivers": _get_all_drivers_as_unassigned(data),
                        "clustering_analysis": {"method": "No Users", "clusters": 0},
                        "parameter": parameter,
                        "string_param": string_param,
                        "warnings": ["No users to assign"]
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

                # EDGE CASE: No drivers
                if not all_drivers:
                    print("⚠️ No drivers available - all users unassigned")
                    unassigned_users = _convert_users_to_unassigned_format(users)
                    return {
                        "status": "true",
                        "execution_time": time.time() - start_time,
                        "data": [],
                        "unassignedUsers": unassigned_users,
                        "unassignedDrivers": [],
                        "clustering_analysis": {"method": "No Drivers", "clusters": 0},
                        "parameter": parameter,
                        "string_param": string_param,
                        "warnings": ["No drivers available for assignment"]
                    }

                total_drivers = len(all_drivers)
                print(f"📥 Data loaded - Users: {len(users)}, Total Drivers: {total_drivers}")

                # Extract dynamic office coordinates with validation
                office_lat, office_lon = extract_office_coordinates(data)
                if not (-90 <= office_lat <= 90) or not (-180 <= office_lon <= 180):
                    print(f"⚠️ Invalid office coordinates, using defaults")
                    office_lat, office_lon = OFFICE_LAT, OFFICE_LON

                print(f"🏢 Office coordinates - Lat: {office_lat}, Lon: {office_lon}")

                validate_input_data(data)
                print("✅ Data validation passed")

                user_df, driver_df = prepare_user_driver_dataframes(data)
                print(f"📊 DataFrames prepared - Users: {len(user_df)}, Drivers: {len(driver_df)}")

                user_df = calculate_bearings_and_features(user_df, office_lat, office_lon)
                print("🧭 Bearings and geographical features calculated")

                # Add office coordinates to user_df for later use
                user_df['office_latitude'] = office_lat
                user_df['office_longitude'] = office_lon

                # REVOLUTIONARY DIRECTIONAL-FIRST CLUSTERING
                user_df, clustering_results = directional_first_clustering(user_df, office_lat, office_lon)
                clusters_found = user_df['directional_cluster'].nunique()
                print(f"🎯 DIRECTIONAL-FIRST clustering complete - {clusters_found} clusters found")

                # PRIORITY-FIRST DRIVER ASSIGNMENT (Driver priority strictly enforced)
                routes, assigned_user_ids = priority_driver_assignment(user_df, driver_df)
                print(f"🚗 PRIORITY assignment complete - {len(routes)} routes, {len(assigned_user_ids)} users assigned")

                # SMART ROUTE MERGING WITH PATH OPTIMIZATION (merge routes considering path efficiency)
                routes = smart_route_merging_with_path_optimization(routes)
                print(f"🌍 Smart route merging complete - {len(routes)} routes after merging")

                # STRICT route filling (maintains efficiency)
                assigned_user_ids = strict_fill_routes(routes, user_df, assigned_user_ids)
                print(f"📈 Strict route filling complete - {len(assigned_user_ids)} total users assigned")

                # EFFICIENCY-FOCUSED local search
                routes = optimized_local_search(routes)

                # REASSIGN underutilized routes (from old code approach)
                routes = reassign_underutilized_routes(routes)

                # Handle remaining users optimally
                unassigned_users = handle_remaining_users_optimally(user_df, routes, assigned_user_ids, driver_df)
                print(f"👥 Remaining users handled - {len(unassigned_users)} unassigned")

                # Build unassigned drivers list
                assigned_driver_ids = {route['driver_id'] for route in routes}
                unassigned_drivers_df = driver_df[~driver_df['driver_id'].isin(assigned_driver_ids)]
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

                # Finalize routes with efficiency metrics
                routes = finalize_routes_with_efficiency(routes, office_lat, office_lon)
                print("✅ Routes finalized with efficiency scoring")

                execution_time = time.time() - start_time

                print(f"✅ ROUTE-OPTIMIZED assignment complete in {execution_time:.2f}s")
                print(f"📊 Final routes: {len(routes)}")
                print(f"🎯 Clustering method: {clustering_results['method']}")

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
                    'driver_id': str(driver.get('id', '')),
                    'capacity': int(driver.get('capacity', 0)),
                    'vehicle_id': str(driver.get('vehicle_id', '')),
                    'latitude': float(driver.get('latitude', 0.0)),
                    'longitude': float(driver.get('longitude', 0.0))
                })
            return unassigned_drivers


def _convert_users_to_unassigned_format(users):
            """Helper to convert user data to unassigned format"""
            unassigned_users = []
            for user in users:
                unassigned_users.append({
                    'user_id': str(user.get('id', '')),
                    'lat': float(user.get('latitude', 0.0)),
                    'lng': float(user.get('longitude', 0.0)),
                    'office_distance': float(user.get('office_distance', 0.0)),
                    'first_name': str(user.get('first_name', '')),
                    'email': str(user.get('email', ''))
                })
            return unassigned_users


def analyze_assignment_quality(result):
            """Analyze the quality of the assignment with enhanced metrics including efficiency"""
            if result["status"] != "true":
                return "Assignment failed"

            total_routes = len(result["data"])
            total_assigned = sum(len(route["assigned_users"]) for route in result["data"])
            total_unassigned = len(result["unassignedUsers"])

            utilizations = []
            efficiency_scores = []
            distance_issues = []
            directional_issues = []

            for route in result["data"]:
                if route["assigned_users"]:
                    util = len(route["assigned_users"]) / route["vehicle_type"]
                    utilizations.append(util)

                    efficiency = route.get('efficiency_score', 0)
                    efficiency_scores.append(efficiency)

                    # Check for directional coherence issues
                    if not route.get('directional_coherence', True):
                        directional_issues.append({
                            "driver_id": route["driver_id"],
                            "bearing_spread": route.get('bearing_spread', 0),
                            "users_count": len(route["assigned_users"])
                        })

                    # Check distances
                    driver_pos = (route["latitude"], route["longitude"])
                    for user in route["assigned_users"]:
                        dist = haversine_distance(driver_pos[0], driver_pos[1], user["lat"], user["lng"])
                        if dist > DISTANCE_ISSUE_THRESHOLD:
                            distance_issues.append({
                                "driver_id": route["driver_id"],
                                "user_id": user["user_id"],
                                "distance_km": round(dist, 2)
                            })

            analysis = {
                "total_routes": total_routes,
                "total_assigned_users": total_assigned,
                "total_unassigned_users": total_unassigned,
                "assignment_rate": round(total_assigned / (total_assigned + total_unassigned) * 100, 1) if (total_assigned + total_unassigned) > 0 else 0,
                "avg_utilization": round(np.mean(utilizations) * 100, 1) if utilizations else 0,
                "avg_efficiency_score": round(np.mean(efficiency_scores), 3) if efficiency_scores else 0,
                "min_efficiency_score": round(np.min(efficiency_scores), 3) if efficiency_scores else 0,
                "routes_below_efficiency_threshold": sum(1 for e in efficiency_scores if e < ROUTE_EFFICIENCY_THRESHOLD),
                "routes_below_80_percent_utilization": sum(1 for u in utilizations if u < 0.8),
                "distance_issues": len(distance_issues),
                "directional_issues": len(directional_issues),
                "directional_coherence_rate": round((total_routes - len(directional_issues)) / total_routes * 100, 1) if total_routes > 0 else 100,
                "clustering_method": result.get("clustering_analysis", {}).get("method", "Unknown")
            }

            return analysis