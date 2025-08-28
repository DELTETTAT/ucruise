
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
    """Load configuration for balance assignment"""
    try:
        with open('config.json') as f:
            cfg = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Warning: Could not load config.json, using defaults. Error: {e}")
        cfg = {}

    # Get balance assignment specific configuration
    balance_cfg = cfg.get("balance_assignment_config", {})

    print(f"🎯 Using optimization mode: BALANCE ASSIGNMENT")
    
    # Validate and set configuration
    config = {}

    # Distance configurations (more lenient for balance assignment)
    config['MAX_FILL_DISTANCE_KM'] = max(0.1, float(balance_cfg.get("max_fill_distance_km", 5.0)))
    config['MERGE_DISTANCE_KM'] = max(0.1, float(balance_cfg.get("merge_distance_km", 3.0)))
    config['DBSCAN_EPS_KM'] = max(0.1, float(balance_cfg.get("dbscan_eps_km", 3.0)))
    config['OVERFLOW_PENALTY_KM'] = max(0.0, float(balance_cfg.get("overflow_penalty_km", 10.0)))
    config['DISTANCE_ISSUE_THRESHOLD'] = max(0.1, float(balance_cfg.get("distance_issue_threshold_km", 8.0)))
    config['SWAP_IMPROVEMENT_THRESHOLD'] = max(0.0, float(balance_cfg.get("swap_improvement_threshold_km", 0.5)))

    # Utilization thresholds (0-1 range)
    config['MIN_UTIL_THRESHOLD'] = max(0.0, min(1.0, float(balance_cfg.get("min_util_threshold", 0.5))))
    config['LOW_UTILIZATION_THRESHOLD'] = max(0.0, min(1.0, float(balance_cfg.get("low_utilization_threshold", 0.5))))

    # Integer configurations
    config['MIN_SAMPLES_DBSCAN'] = max(1, int(balance_cfg.get("min_samples_dbscan", 2)))
    config['MAX_SWAP_ITERATIONS'] = max(1, int(balance_cfg.get("max_swap_iterations", 3)))
    config['MAX_USERS_FOR_FALLBACK'] = max(1, int(balance_cfg.get("max_users_for_fallback", 3)))
    config['FALLBACK_MIN_USERS'] = max(1, int(balance_cfg.get("fallback_min_users", 2)))
    config['FALLBACK_MAX_USERS'] = max(1, int(balance_cfg.get("fallback_max_users", 7)))

    # Angle configurations (more lenient for balance assignment)
    config['MAX_BEARING_DIFFERENCE'] = max(0, min(180, float(balance_cfg.get("max_bearing_difference", 30))))

    # Cost penalties
    config['UTILIZATION_PENALTY_PER_SEAT'] = max(0.0, float(balance_cfg.get("utilization_penalty_per_seat", 2.0)))

    # Office coordinates
    office_lat = float(os.getenv("OFFICE_LAT", cfg.get("office_latitude", 30.6810489)))
    office_lon = float(os.getenv("OFFICE_LON", cfg.get("office_longitude", 76.7260711)))

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
        driver_id = str(driver["id"])
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
def create_geographic_clusters(user_df, office_lat, office_lon):
    """
    Step 1: Create geographic clusters based on location and bearing
    Simple and straightforward approach for balance assignment
    """
    print("🗂️ Step 1: Creating geographic clusters...")

    coords = user_df[['latitude', 'longitude']].values

    # Handle single user case
    if len(user_df) == 1:
        user_df = user_df.copy()
        user_df['geo_cluster'] = 0
        return user_df, {"method": "Single User", "clusters": 1}

    # Use simple DBSCAN for geographic clustering
    scaler = StandardScaler()
    scaled_coords = scaler.fit_transform(coords)

    # Try DBSCAN first
    clustering = DBSCAN(eps=0.3,
                        min_samples=MIN_SAMPLES_DBSCAN).fit(scaled_coords)
    user_df = user_df.copy()
    user_df['geo_cluster'] = clustering.labels_

    # Handle noise points by assigning to nearest cluster
    noise_mask = user_df['geo_cluster'] == -1
    if noise_mask.any():
        # Assign noise points to nearest valid cluster or create new clusters
        for idx in user_df[noise_mask].index:
            user_pos = coords[user_df.index.get_loc(idx)]
            min_dist = float('inf')
            best_cluster = 0

            # Find nearest cluster
            for cluster_id in user_df[user_df['geo_cluster'] !=
                                      -1]['geo_cluster'].unique():
                cluster_center = user_df[user_df['geo_cluster'] == cluster_id][
                    ['latitude', 'longitude']].mean()
                dist = haversine_distance(user_pos[0], user_pos[1],
                                          cluster_center['latitude'],
                                          cluster_center['longitude'])
                if dist < min_dist:
                    min_dist = dist
                    best_cluster = cluster_id

            user_df.loc[idx, 'geo_cluster'] = best_cluster

    # If DBSCAN produces too few clusters, use K-means
    n_clusters = user_df['geo_cluster'].nunique()
    if n_clusters <= 1:
        print("  📊 DBSCAN produced too few clusters, using K-means...")
        optimal_clusters = min(6, max(2, len(user_df) // 4))
        kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
        user_df['geo_cluster'] = kmeans.fit_predict(scaled_coords)
        method_used = "K-means"
    else:
        method_used = "DBSCAN"

    print(
        f"  ✅ Created {user_df['geo_cluster'].nunique()} geographic clusters using {method_used}"
    )
    return user_df, {
        "method": method_used,
        "clusters": user_df['geo_cluster'].nunique()
    }


def create_capacity_subclusters(user_df, driver_df):
    """Create capacity-based sub-clusters for balance assignment"""
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


def assign_drivers_by_priority(user_df, driver_df, office_lat, office_lon):
    """Assign drivers based on priority and balanced assignment approach"""
    print("🚗 Step 3: Assigning drivers by priority (BALANCE ASSIGNMENT)...")

    routes = []
    assigned_user_ids = set()
    used_driver_ids = set()

    # Sort drivers by priority
    available_drivers = driver_df.sort_values(['priority', 'capacity'],
                                              ascending=[True, False])

    # Process each sub-cluster
    for sub_cluster_id, cluster_users in user_df.groupby('sub_cluster'):
        unassigned_in_cluster = cluster_users[~cluster_users['user_id'].
                                              isin(assigned_user_ids)]

        if unassigned_in_cluster.empty:
            continue

        cluster_center = unassigned_in_cluster[['latitude',
                                                'longitude']].mean()
        cluster_size = len(unassigned_in_cluster)

        # For balance assignment, we're more flexible with bearing differences
        bearings = unassigned_in_cluster['bearing'].values
        if len(bearings) > 1:
            max_bearing_diff = max([
                bearing_difference(bearings[i], bearings[j])
                for i in range(len(bearings))
                for j in range(i + 1, len(bearings))
            ])

            # More lenient bearing threshold for balance assignment
            if max_bearing_diff > MAX_BEARING_DIFFERENCE * 1.5:  # 50% more lenient
                print(
                    f"  📍 Splitting sub-cluster {sub_cluster_id} due to bearing spread ({max_bearing_diff:.1f}°)"
                )
                # Split into 2 sub-groups based on bearing
                coords_with_bearing = np.column_stack([
                    unassigned_in_cluster[['latitude', 'longitude']].values,
                    unassigned_in_cluster[['bearing_sin',
                                           'bearing_cos']].values
                ])
                kmeans = KMeans(n_clusters=2, random_state=42)
                split_labels = kmeans.fit_predict(coords_with_bearing)

                # Process each split separately
                for split_id in range(2):
                    split_users = unassigned_in_cluster[split_labels ==
                                                        split_id]
                    if len(split_users) > 0:
                        route = assign_best_driver_to_cluster(
                            split_users, available_drivers, used_driver_ids,
                            office_lat, office_lon)
                        if route:
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
            routes.append(route)
            assigned_user_ids.update(u['user_id']
                                     for u in route['assigned_users'])

    print(
        f"  ✅ Created {len(routes)} initial routes with priority-based assignment"
    )
    return routes, assigned_user_ids


def assign_best_driver_to_cluster(cluster_users, available_drivers,
                                  used_driver_ids, office_lat, office_lon):
    """Find and assign the best available driver to a cluster with balance optimization"""
    cluster_center = cluster_users[['latitude', 'longitude']].mean()
    cluster_size = len(cluster_users)

    best_driver = None
    min_cost = float('inf')

    for _, driver in available_drivers.iterrows():
        if driver['driver_id'] in used_driver_ids:
            continue

        # Check capacity
        if driver['capacity'] < cluster_size:
            continue  # Skip drivers without enough capacity

        # Calculate cost: distance + priority penalty
        driver_pos = (driver['latitude'], driver['longitude'])
        distance = haversine_distance(driver_pos[0], driver_pos[1],
                                      cluster_center['latitude'],
                                      cluster_center['longitude'])

        # Reduced priority penalty for balance assignment
        priority_penalty = driver['priority'] * 1.0  # Reduced from 2.0

        # Enhanced utilization bonus for balance assignment
        utilization = cluster_size / driver['capacity']
        utilization_bonus = utilization * 4.0  # Increased from 3.0

        total_cost = distance + priority_penalty - utilization_bonus

        if total_cost < min_cost:
            min_cost = total_cost
            best_driver = driver

    if best_driver is not None:
        used_driver_ids.add(best_driver['driver_id'])

        # Create route
        route = {
            'driver_id': str(best_driver['driver_id']),
            'vehicle_id': str(best_driver.get('vehicle_id', '')),
            'vehicle_type': int(best_driver['capacity']),
            'latitude': float(best_driver['latitude']),
            'longitude': float(best_driver['longitude']),
            'assigned_users': []
        }

        # Add users to route (up to capacity)
        users_to_assign = cluster_users.head(best_driver['capacity'])
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
                route['assigned_users']) / route['vehicle_type']
            route['bearing'] = calculate_bearing(route['centroid'][0],
                                                 route['centroid'][1],
                                                 office_lat, office_lon)

        return route

    return None


def local_optimization(routes):
    """Local optimization for balance assignment"""
    print("🔧 Step 4: Local optimization...")

    improved = True
    iterations = 0

    while improved and iterations < MAX_SWAP_ITERATIONS:
        improved = False
        iterations += 1

        # Try to improve individual routes
        for route in routes:
            if len(route['assigned_users']) < 2:
                continue

            # Optimize user sequence within route
            route = optimize_route_sequence(route)

        # Try swapping users between nearby routes
        for i, route_a in enumerate(routes):
            for j, route_b in enumerate(routes[i + 1:], start=i + 1):
                if not route_a['assigned_users'] or not route_b[
                        'assigned_users']:
                    continue

                # Check if routes are nearby
                center_a = route_a.get(
                    'centroid', [route_a['latitude'], route_a['longitude']])
                center_b = route_b.get(
                    'centroid', [route_b['latitude'], route_b['longitude']])
                distance = haversine_distance(center_a[0], center_a[1],
                                              center_b[0], center_b[1])

                if distance <= MERGE_DISTANCE_KM:
                    # Try swapping users
                    if try_user_swap(route_a, route_b):
                        improved = True

    print(f"  ✅ Local optimization completed in {iterations} iterations")
    return routes


def optimize_route_sequence(route):
    """Optimize pickup sequence within a route"""
    if len(route['assigned_users']) <= 2:
        return route

    # Simple nearest neighbor optimization
    users = route['assigned_users'].copy()
    driver_pos = (route['latitude'], route['longitude'])

    optimized_sequence = []
    remaining_users = users.copy()
    current_pos = driver_pos

    while remaining_users:
        # Find nearest user to current position
        nearest_user = min(
            remaining_users,
            key=lambda u: haversine_distance(current_pos[0], current_pos[1], u[
                'lat'], u['lng']))

        optimized_sequence.append(nearest_user)
        remaining_users.remove(nearest_user)
        current_pos = (nearest_user['lat'], nearest_user['lng'])

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
            route['assigned_users']) / route['vehicle_type']
        route['bearing'] = calculate_bearing(route['centroid'][0],
                                             route['centroid'][1], OFFICE_LAT,
                                             OFFICE_LON)
    else:
        route['centroid'] = [route['latitude'], route['longitude']]
        route['utilization'] = 0
        route['bearing'] = 0


def global_optimization(routes, user_df, assigned_user_ids, driver_df):
    """Global optimization for balance assignment - focuses on maximizing utilization"""
    print("🌍 Step 5: Global optimization (BALANCE ASSIGNMENT)...")

    # Fill underutilized routes with nearby unassigned users
    unassigned_users = user_df[~user_df['user_id'].isin(assigned_user_ids)]

    for route in routes:
        if len(route['assigned_users']
               ) < route['vehicle_type'] and not unassigned_users.empty:
            route_center = route.get('centroid',
                                     [route['latitude'], route['longitude']])

            # Find nearby unassigned users - more lenient for balance assignment
            nearby_users = []
            for _, user in unassigned_users.iterrows():
                distance = haversine_distance(route_center[0], route_center[1],
                                              user['latitude'],
                                              user['longitude'])

                # More lenient distance threshold for balance assignment
                if distance <= MAX_FILL_DISTANCE_KM * 1.5:
                    # Check bearing compatibility with relaxed constraints
                    if route['assigned_users']:
                        user_bearing = user['bearing']
                        route_bearing = route.get('bearing', 0)
                        bearing_diff = bearing_difference(
                            user_bearing, route_bearing)

                        # More lenient bearing threshold for balance assignment
                        if bearing_diff <= MAX_BEARING_DIFFERENCE * 1.5:
                            nearby_users.append((user, distance))

            # Sort by distance and add users up to capacity
            nearby_users.sort(key=lambda x: x[1])
            slots_available = route['vehicle_type'] - len(
                route['assigned_users'])

            for user, distance in nearby_users[:slots_available]:
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
                unassigned_users = unassigned_users[unassigned_users['user_id']
                                                    != user['user_id']]

    # Merge underutilized nearby routes
    routes = merge_compatible_routes(routes)

    # Handle remaining unassigned users
    remaining_unassigned = user_df[~user_df['user_id'].isin(assigned_user_ids)]
    unassigned_list = handle_remaining_users(remaining_unassigned, routes,
                                             driver_df)

    print("  ✅ Global optimization completed")
    return routes, unassigned_list


def merge_compatible_routes(routes):
    """Merge compatible underutilized routes - more aggressive for balance assignment"""
    current_routes = routes.copy()
    merged_count = 0
    max_passes = 4  # More passes for balance assignment

    for pass_num in range(max_passes):
        merged_routes = []
        used_route_indices = set()
        pass_merges = 0

        # Sort routes by utilization (lowest first for aggressive merging)
        sorted_routes = sorted(enumerate(current_routes),
                               key=lambda x: x[1].get('utilization', 1))

        for orig_i, route_a in sorted_routes:
            if orig_i in used_route_indices:
                continue

            # Look for the best merge candidate
            best_merge_candidate = None
            best_merge_score = float('inf')
            best_candidate_index = None

            for orig_j, route_b in sorted_routes:
                if orig_j in used_route_indices or orig_j == orig_i:
                    continue

                # Check if routes can be merged - more lenient for balance assignment
                if can_merge_routes_balance(route_a, route_b):
                    # Calculate merge score (lower is better)
                    center_a = route_a.get(
                        'centroid',
                        [route_a['latitude'], route_a['longitude']])
                    center_b = route_b.get(
                        'centroid',
                        [route_b['latitude'], route_b['longitude']])
                    distance = haversine_distance(center_a[0], center_a[1],
                                                  center_b[0], center_b[1])

                    # Prefer merging with closer, less utilized routes
                    util_penalty = (route_b.get('utilization', 0) * 1.5
                                    )  # Reduced penalty for balance assignment
                    merge_score = distance + util_penalty

                    if merge_score < best_merge_score:
                        best_merge_score = merge_score
                        best_merge_candidate = route_b
                        best_candidate_index = orig_j

            if best_merge_candidate is not None:
                # Merge with best candidate
                target_route = best_merge_candidate if best_merge_candidate[
                    'vehicle_type'] >= route_a['vehicle_type'] else route_a
                source_route = route_a if target_route == best_merge_candidate else best_merge_candidate

                # Create new merged route
                merged_route = target_route.copy()
                merged_route['assigned_users'] = target_route[
                    'assigned_users'] + source_route['assigned_users']
                update_route_metrics(merged_route)

                merged_routes.append(merged_route)
                used_route_indices.add(orig_i)
                used_route_indices.add(best_candidate_index)
                pass_merges += 1
                merged_count += 1

                print(
                    f"    🔗 Pass {pass_num + 1}: Merged routes with utilizations {route_a.get('utilization', 0):.1%} + {best_merge_candidate.get('utilization', 0):.1%} → {merged_route.get('utilization', 0):.1%}"
                )
            else:
                # No merge candidate found, keep original route
                merged_routes.append(route_a)
                used_route_indices.add(orig_i)

        current_routes = merged_routes
        print(f"    📊 Pass {pass_num + 1}: {pass_merges} merges completed")

        # If no merges happened this pass, we're done
        if pass_merges == 0:
            break

    if merged_count > 0:
        print(
            f"  ✅ Total merges: {merged_count}, Final routes: {len(current_routes)}"
        )

    return current_routes


def can_merge_routes_balance(route_a, route_b):
    """Check if two routes can be merged - more lenient for balance assignment"""
    total_users = len(route_a['assigned_users']) + len(
        route_b['assigned_users'])
    max_capacity = max(route_a['vehicle_type'], route_b['vehicle_type'])

    if total_users > max_capacity:
        return False

    # More lenient distance threshold for balance assignment
    center_a = route_a.get('centroid',
                           [route_a['latitude'], route_a['longitude']])
    center_b = route_b.get('centroid',
                           [route_b['latitude'], route_b['longitude']])
    distance = haversine_distance(center_a[0], center_a[1], center_b[0],
                                  center_b[1])

    if distance > MERGE_DISTANCE_KM * 1.5:  # 50% more lenient
        return False

    # More lenient bearing compatibility for balance assignment
    bearing_a = route_a.get('bearing', 0)
    bearing_b = route_b.get('bearing', 0)
    bearing_diff = bearing_difference(bearing_a, bearing_b)

    # Get utilizations
    util_a = route_a.get('utilization', 0)
    util_b = route_b.get('utilization', 0)
    avg_util = (util_a + util_b) / 2

    # Much more lenient bearing threshold for balance assignment
    if avg_util < 0.5:  # Both routes are underutilized
        max_bearing_diff = 90  # Allow up to 90° difference
    elif avg_util < 0.7:  # Moderately utilized
        max_bearing_diff = 75  # Allow up to 75° difference
    else:
        max_bearing_diff = MAX_BEARING_DIFFERENCE * 2  # Double the default

    # Additional check: if routes are very close, be even more lenient
    if distance < 1.5:
        max_bearing_diff = min(max_bearing_diff + 30,
                               120)  # Extra 30° tolerance

    return bearing_diff <= max_bearing_diff


def handle_remaining_users(unassigned_users, routes, driver_df):
    """Handle users that couldn't be assigned to any route"""
    unassigned_list = []

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

    return unassigned_list


def run_assignment(source_id: str, parameter: int = 1, string_param: str = ""):
    """Balance assignment - focuses on maximizing utilization with flexible routing"""
    return run_balance_assignment(source_id, parameter, string_param)


def run_balance_assignment(source_id: str, parameter: int = 1, string_param: str = ""):
    """
    Main balance assignment function:
    1. Geographic clustering
    2. Capacity-based sub-clustering  
    3. Priority-based driver assignment (with balance optimization)
    4. Local optimization
    5. Global optimization (aggressive merging and filling)
    """
    start_time = time.time()

    try:
        print(
            f"🚀 Starting BALANCE assignment for source_id: {source_id}"
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
                "optimization_mode": "balance_assignment",
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
                "optimization_mode": "balance_assignment",
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
            user_df, office_lat, office_lon)

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

        # Build unassigned drivers list
        assigned_driver_ids = {route['driver_id'] for route in routes}
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
            f"✅ Balance assignment complete in {execution_time:.2f}s")
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
            "optimization_mode": "balance_assignment",
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
        logger.error(f"Balance assignment failed: {e}", exc_info=True)
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
            util = len(route["assigned_users"]) / route["vehicle_type"]
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
