
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

    # Angle configurations (0-180 degrees)
    config['MAX_BEARING_DIFFERENCE'] = max(
        0, min(180, float(cfg.get("max_bearing_difference", 30))))

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
    Simple and straightforward approach with balanced utilization focus
    """
    print("🗂️ Step 1: Creating balanced geographic clusters...")

    coords = user_df[['latitude', 'longitude']].values

    # Handle single user case
    if len(user_df) == 1:
        user_df = user_df.copy()
        user_df['geo_cluster'] = 0
        return user_df, {"method": "Single User", "clusters": 1}

    # Use balanced approach - prefer more, smaller clusters for better utilization
    scaler = StandardScaler()
    scaled_coords = scaler.fit_transform(coords)

    # Try DBSCAN with tighter clustering for balance
    clustering = DBSCAN(eps=0.25,  # Tighter clusters
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

    # If DBSCAN produces too few clusters, use K-means with more clusters
    n_clusters = user_df['geo_cluster'].nunique()
    if n_clusters <= 1:
        print("  📊 DBSCAN produced too few clusters, using K-means for balance...")
        # Prefer more clusters for better utilization balance
        optimal_clusters = min(8, max(3, len(user_df) // 3))  # More aggressive clustering
        kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
        user_df['geo_cluster'] = kmeans.fit_predict(scaled_coords)
        method_used = "K-means (Balanced)"
    else:
        method_used = "DBSCAN (Balanced)"

    print(
        f"  ✅ Created {user_df['geo_cluster'].nunique()} balanced geographic clusters using {method_used}"
    )
    return user_df, {
        "method": method_used,
        "clusters": user_df['geo_cluster'].nunique()
    }


def create_balanced_routes(user_df, driver_df, office_lat, office_lon):
    """Create routes with emphasis on balanced utilization"""
    routes = []
    used_driver_ids = set()
    assigned_user_ids = set()

    # Sort drivers by capacity and priority for balanced assignment
    available_drivers = driver_df.sort_values(['priority', 'capacity'], ascending=[True, True])

    # Process each cluster with balanced approach
    for cluster_id in user_df['geo_cluster'].unique():
        cluster_users = user_df[user_df['geo_cluster'] == cluster_id]
        unassigned_in_cluster = cluster_users[~cluster_users['user_id'].isin(assigned_user_ids)]

        if unassigned_in_cluster.empty:
            continue

        # Split large clusters to ensure balanced utilization
        if len(unassigned_in_cluster) > 6:  # Split larger clusters
            coords = unassigned_in_cluster[['latitude', 'longitude']].values
            n_subclusters = math.ceil(len(unassigned_in_cluster) / 4)  # Aim for ~4 users per subcluster
            
            if n_subclusters > 1:
                kmeans = KMeans(n_clusters=n_subclusters, random_state=42)
                sub_labels = kmeans.fit_predict(coords)

                for sub_id in range(n_subclusters):
                    sub_users = unassigned_in_cluster.iloc[sub_labels == sub_id]
                    if len(sub_users) > 0:
                        route = assign_balanced_driver_to_cluster(sub_users, available_drivers, 
                                                                used_driver_ids, office_lat, office_lon)
                        if route:
                            routes.append(route)
                            for user in route['assigned_users']:
                                assigned_user_ids.add(user['user_id'])
            else:
                route = assign_balanced_driver_to_cluster(unassigned_in_cluster, available_drivers,
                                                        used_driver_ids, office_lat, office_lon)
                if route:
                    routes.append(route)
                    for user in route['assigned_users']:
                        assigned_user_ids.add(user['user_id'])
        else:
            # Normal assignment for smaller clusters
            route = assign_balanced_driver_to_cluster(unassigned_in_cluster, available_drivers,
                                                    used_driver_ids, office_lat, office_lon)
            if route:
                routes.append(route)
                for user in route['assigned_users']:
                    assigned_user_ids.add(user['user_id'])

    return routes, assigned_user_ids


def assign_balanced_driver_to_cluster(cluster_users, available_drivers, used_driver_ids, office_lat, office_lon):
    """Find and assign the best driver with balanced utilization in mind"""
    cluster_center = cluster_users[['latitude', 'longitude']].mean()
    cluster_size = len(cluster_users)

    best_driver = None
    best_score = float('inf')

    for _, driver in available_drivers.iterrows():
        if driver['driver_id'] in used_driver_ids:
            continue

        # Check capacity
        if driver['capacity'] < cluster_size:
            continue

        # Calculate balanced score: prefer better utilization and closer distance
        driver_pos = (driver['latitude'], driver['longitude'])
        distance = haversine_distance(driver_pos[0], driver_pos[1],
                                      cluster_center['latitude'],
                                      cluster_center['longitude'])

        # Priority penalty (lower priority = higher penalty)
        priority_penalty = driver['priority'] * 1.5

        # Utilization scoring - prefer drivers that will have good utilization
        utilization = cluster_size / driver['capacity']
        # Bonus for utilization between 60-90% (sweet spot for balance)
        if 0.6 <= utilization <= 0.9:
            utilization_bonus = 3.0
        elif utilization >= 0.5:
            utilization_bonus = 1.5
        else:
            utilization_bonus = 0.0

        # Penalty for oversize vehicles (waste)
        if driver['capacity'] > cluster_size * 1.5:
            oversize_penalty = (driver['capacity'] - cluster_size) * 0.5
        else:
            oversize_penalty = 0

        balanced_score = distance + priority_penalty - utilization_bonus + oversize_penalty

        if balanced_score < best_score:
            best_score = balanced_score
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

        # Add users to route
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

            route['assigned_users'].append(user_data)

        # Calculate route metrics
        if route['assigned_users']:
            lats = [u['lat'] for u in route['assigned_users']]
            lngs = [u['lng'] for u in route['assigned_users']]
            route['centroid'] = [np.mean(lats), np.mean(lngs)]
            route['utilization'] = len(route['assigned_users']) / route['vehicle_type']
            route['bearing'] = calculate_bearing(route['centroid'][0],
                                                 route['centroid'][1],
                                                 office_lat, office_lon)

        return route

    return None


def run_assignment(data):
    """
    Balance-optimized assignment function with emphasis on utilization balance
    """
    start_time = time.time()

    try:
        print("🚀 Starting BALANCE EFFICIENCY assignment...")

        # Extract office coordinates and validate data
        office_lat, office_lon = extract_office_coordinates(data)
        validate_input_data(data)
        print("✅ Data validation passed")

        # Prepare dataframes
        user_df, driver_df = prepare_user_driver_dataframes(data)
        user_df = calculate_bearings_and_features(user_df, office_lat, office_lon)

        print(f"📊 DataFrames prepared - Users: {len(user_df)}, Drivers: {len(driver_df)}")

        # STEP 1: Geographic clustering with balance focus
        user_df, clustering_results = create_geographic_clusters(user_df, office_lat, office_lon)

        # STEP 2: Create balanced routes
        routes, assigned_user_ids = create_balanced_routes(user_df, driver_df, office_lat, office_lon)

        # Build unassigned users and drivers
        unassigned_users = []
        for _, user in user_df.iterrows():
            if user['user_id'] not in assigned_user_ids:
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
                unassigned_users.append(user_data)

        assigned_driver_ids = {route['driver_id'] for route in routes}
        unassigned_drivers = []
        for _, driver in driver_df.iterrows():
            if driver['driver_id'] not in assigned_driver_ids:
                driver_data = {
                    'driver_id': str(driver['driver_id']),
                    'capacity': int(driver['capacity']),
                    'vehicle_id': str(driver.get('vehicle_id', '')),
                    'latitude': float(driver['latitude']),
                    'longitude': float(driver['longitude'])
                }
                unassigned_drivers.append(driver_data)

        execution_time = time.time() - start_time

        print(f"✅ BALANCE EFFICIENCY assignment complete in {execution_time:.2f}s")
        print(f"📊 Final routes: {len(routes)}")
        print(f"🎯 Users assigned: {sum(len(r['assigned_users']) for r in routes)}")
        print(f"👥 Users unassigned: {len(unassigned_users)}")

        # Calculate utilization statistics
        utilizations = [len(r['assigned_users']) / r['vehicle_type'] for r in routes if r['assigned_users']]
        avg_util = sum(utilizations) / len(utilizations) if utilizations else 0
        print(f"📈 Average utilization: {avg_util:.1%}")

        return {
            "status": "true",
            "execution_time": execution_time,
            "data": routes,
            "unassignedUsers": unassigned_users,
            "unassignedDrivers": unassigned_drivers,
            "clustering_analysis": clustering_results,
            "optimization_mode": "balance_efficiency"
        }

    except Exception as e:
        logger.error(f"Assignment failed: {e}", exc_info=True)
        return {"status": "false", "details": str(e), "data": []}
