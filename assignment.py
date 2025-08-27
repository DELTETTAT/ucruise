import os
import math
import requests
import numpy as np
import pandas as pd
import time
import json
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

    # Clustering and optimization parameters
    config['clustering_method'] = cfg.get('clustering_method', 'adaptive')
    config['min_cluster_size'] = max(2, cfg.get('min_cluster_size', 3))
    config['use_sweep_algorithm'] = cfg.get('use_sweep_algorithm', True)
    config['angular_sectors'] = cfg.get('angular_sectors', 8)
    config['max_users_per_initial_cluster'] = cfg.get('max_users_per_initial_cluster', 8)
    config['max_users_per_cluster'] = cfg.get('max_users_per_cluster', 7)
    config['max_allowed_turning_score'] = cfg.get('max_allowed_turning_score', 45)
    config['max_tortuosity_ratio'] = cfg.get('max_tortuosity_ratio', 1.8)
    config['zigzag_penalty_weight'] = cfg.get('zigzag_penalty_weight', 2.0)


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


def prepare_user_driver_dataframes(data):
    """Prepare user and driver dataframes from API data"""
    # Prepare user DataFrame
    users = data.get("users", [])
    user_data = []
    for user in users:
        user_data.append({
            'user_id': str(user.get('id', '')),
            'latitude': float(user.get('latitude', 0.0)),
            'longitude': float(user.get('longitude', 0.0)),
            'first_name': str(user.get('first_name', '')),
            'email': str(user.get('email', '')),
            'office_distance': float(user.get('office_distance', 0.0))
        })
    
    user_df = pd.DataFrame(user_data)
    
    # Prepare driver DataFrame
    all_drivers = []
    if "drivers" in data:
        drivers_data = data["drivers"]
        all_drivers.extend(drivers_data.get("driversUnassigned", []))
        all_drivers.extend(drivers_data.get("driversAssigned", []))
    else:
        all_drivers.extend(data.get("driversUnassigned", []))
        all_drivers.extend(data.get("driversAssigned", []))
    
    driver_data = []
    for i, driver in enumerate(all_drivers):
        driver_data.append({
            'driver_id': str(driver.get('id', '')),
            'latitude': float(driver.get('latitude', 0.0)),
            'longitude': float(driver.get('longitude', 0.0)),
            'capacity': int(driver.get('capacity', 1)),
            'vehicle_id': str(driver.get('vehicle_id', '')),
            'priority': i + 1  # Simple priority based on order
        })
    
    driver_df = pd.DataFrame(driver_data)
    
    return user_df, driver_df


def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate the great circle distance between two points on Earth"""
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    
    # Radius of earth in kilometers
    r = 6371
    return c * r


def bearing_difference(b1, b2):
    """Compute minimum difference between two bearings"""
    diff = abs(b1 - b2) % 360
    return min(diff, 360 - diff)


def calculate_bearing_vectorized(lat1, lon1, lat2, lon2):
    """Vectorized calculation of bearing from point A to B in degrees"""
    lat1, lat2 = np.radians(lat1), np.radians(lat2)
    dlon = np.radians(lon2 - lon1)
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(
        dlon)
    return (np.degrees(np.arctan2(x, y)) + 360) % 360


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

    # Calculate bearing FROM OFFICE TO USER
    user_df['bearing_from_office'] = calculate_bearing_vectorized(
        office_lat, office_lon, user_df['latitude'], user_df['longitude'])
    user_df['bearing_sin'] = np.sin(np.radians(user_df['bearing_from_office']))
    user_df['bearing_cos'] = np.cos(np.radians(user_df['bearing_from_office']))

    return user_df


# Helper functions for clustering
def dbscan_clustering(user_df, eps_km, min_samples):
    """Perform DBSCAN clustering on user data"""
    coords = user_df[['latitude', 'longitude']].values

    # Scale coordinates for DBSCAN
    scaler = StandardScaler()
    scaled_coords = scaler.fit_transform(coords)

    # Perform DBSCAN
    dbscan = DBSCAN(eps=eps_km, min_samples=min_samples)
    labels = dbscan.fit_predict(scaled_coords)

    # Handle noise points: assign to nearest cluster if possible
    noise_mask = labels == -1
    if noise_mask.any():
        valid_labels = labels[~noise_mask]
        if len(valid_labels) > 0:
            # Find nearest cluster for each noise point
            for i in np.where(noise_mask)[0]:
                dist_to_noise_point = scaled_coords[i]
                distances = cdist([dist_to_noise_point], scaled_coords[~noise_mask])[0]
                nearest_cluster_idx = np.argmin(distances)
                labels[i] = valid_labels[nearest_cluster_idx]
        else:
            # If all points are noise, assign to a single cluster
            labels[:] = 0
    return labels

def kmeans_clustering(user_df, n_clusters):
    """Perform KMeans clustering"""
    coords = user_df[['latitude', 'longitude']].values
    scaler = StandardScaler()
    scaled_coords = scaler.fit_transform(coords)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    return kmeans.fit_predict(scaled_coords)

def estimate_clusters(user_df, config):
    """Estimate optimal number of clusters using silhouette score"""
    coords = user_df[['latitude', 'longitude']].values
    scaler = StandardScaler()
    scaled_coords = scaler.fit_transform(coords)

    max_clusters = min(10, len(user_df) // 2)
    if max_clusters < 2:
        return 1

    scores = []
    for n_clusters in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(scaled_coords)
        if len(set(cluster_labels)) > 1:
            score = silhouette_score(scaled_coords, cluster_labels)
            scores.append((n_clusters, score))

    if not scores:
        return 1

    best_n_clusters = max(scores, key=lambda item: item[1])[0]
    return best_n_clusters


# STEP 1: DIRECTION-AWARE GEOGRAPHIC CLUSTERING
def create_geographic_clusters(user_df, office_lat, office_lon, config):
    """Create direction-aware geographic clusters using polar coordinates"""
    if len(user_df) == 0:
        return user_df

    print("  🗺️  Creating direction-aware geographic clusters...")

    # Calculate features including bearings
    user_df = calculate_bearings_and_features(user_df, office_lat, office_lon)

    # Use sector-based clustering for direction awareness
    use_sweep = config.get('use_sweep_algorithm', True)
    max_bearing_diff = config.get('max_bearing_difference', 15)  # Reduced from 30

    if use_sweep and len(user_df) > 3:
        labels = sweep_clustering(user_df, config)
    else:
        labels = polar_sector_clustering(user_df, office_lat, office_lon, config)

    user_df['geo_cluster'] = labels
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    print(f"  ✅ Created {n_clusters} direction-aware geographic clusters")
    return user_df

def sweep_clustering(user_df, config):
    """Sweep algorithm: sort by bearing and group by capacity"""
    # Sort users by bearing from office
    sorted_df = user_df.sort_values('bearing_from_office')

    labels = []
    current_cluster = 0
    current_capacity = 0
    max_capacity = config.get('max_users_per_initial_cluster', 8)

    for idx, user in sorted_df.iterrows():
        if current_capacity >= max_capacity:
            current_cluster += 1
            current_capacity = 0

        labels.append(current_cluster)
        current_capacity += 1

    # Create label mapping back to original order
    result_labels = [-1] * len(user_df)
    for i, orig_idx in enumerate(sorted_df.index):
        result_labels[orig_idx] = labels[i]

    return result_labels

def polar_sector_clustering(user_df, office_lat, office_lon, config):
    """Partition into angular sectors then cluster within sectors"""
    n_sectors = config.get('angular_sectors', 8)
    sector_angle = 360.0 / n_sectors

    # Assign users to sectors based on bearing
    user_df_copy = user_df.copy()
    user_df_copy['sector'] = (user_df_copy['bearing_from_office'] // sector_angle).astype(int)

    labels = [-1] * len(user_df)
    current_cluster = 0

    # Cluster within each sector
    for sector in range(n_sectors):
        sector_users = user_df_copy[user_df_copy['sector'] == sector]
        if len(sector_users) == 0:
            continue

        if len(sector_users) <= 3:
            # Small sectors get single cluster
            for idx in sector_users.index:
                labels[idx] = current_cluster
            current_cluster += 1
        else:
            # Use spatial clustering within sector
            eps_km = config.get('DBSCAN_EPS_KM', 1.5)  # Smaller for within-sector
            sector_labels = dbscan_clustering(sector_users, eps_km, 2)

            for i, idx in enumerate(sector_users.index):
                if sector_labels[i] == -1:
                    labels[idx] = current_cluster
                    current_cluster += 1
                else:
                    labels[idx] = current_cluster + sector_labels[i]

            current_cluster += max(sector_labels) + 1 if len(sector_labels) > 0 else 1

    return labels


# STEP 2: DIRECTION-AWARE CAPACITY SUB-CLUSTERING
def create_capacity_subclusters(user_df, office_lat, office_lon, config):
    """Split geographic clusters by capacity and bearing constraints with direction awareness"""
    if len(user_df) == 0:
        return user_df

    print("  🚗 Creating direction-aware capacity-based sub-clusters...")

    user_df['sub_cluster'] = -1
    sub_cluster_counter = 0
    max_bearing_diff = config.get('max_bearing_difference', 15)  # Reduced for better direction consistency

    for geo_cluster in user_df['geo_cluster'].unique():
        if geo_cluster == -1:
            continue

        geo_cluster_users = user_df[user_df['geo_cluster'] == geo_cluster]

        if len(geo_cluster_users) <= config.get('max_users_per_cluster', 7):
            user_df.loc[geo_cluster_users.index, 'sub_cluster'] = sub_cluster_counter
            sub_cluster_counter += 1
        else:
            # Use bearing-weighted clustering for sub-clusters
            sub_cluster_counter = create_bearing_aware_subclusters(
                geo_cluster_users, user_df, sub_cluster_counter, config, max_bearing_diff
            )

    print(f"  ✅ Created {user_df['sub_cluster'].nunique()} direction-aware capacity-based sub-clusters")
    return user_df

def create_bearing_aware_subclusters(geo_cluster_users, user_df, sub_cluster_counter, config, max_bearing_diff):
    """Create subclusters that maintain directional consistency"""
    # Sort by bearing to maintain direction
    sorted_users = geo_cluster_users.sort_values('bearing_from_office')
    max_users_per_cluster = config.get('max_users_per_cluster', 7)

    current_cluster_users = []

    for idx, (user_idx, user) in enumerate(sorted_users.iterrows()):
        # Check if adding this user would violate bearing constraints
        if current_cluster_users:
            bearing_spread = calculate_bearing_spread([u[1] for u in current_cluster_users] + [user])
            if len(current_cluster_users) >= max_users_per_cluster or bearing_spread > max_bearing_diff:
                # Assign current cluster
                for cluster_user_idx, _ in current_cluster_users:
                    user_df.loc[cluster_user_idx, 'sub_cluster'] = sub_cluster_counter
                sub_cluster_counter += 1
                current_cluster_users = []

        current_cluster_users.append((user_idx, user))

    # Assign remaining users
    if current_cluster_users:
        for cluster_user_idx, _ in current_cluster_users:
            user_df.loc[cluster_user_idx, 'sub_cluster'] = sub_cluster_counter
        sub_cluster_counter += 1

    return sub_cluster_counter

def calculate_bearing_spread(users):
    """Calculate the angular spread of users"""
    if len(users) <= 1:
        return 0

    bearings = [user['bearing_from_office'] for user in users]
    bearings.sort()

    # Handle circular nature of bearings
    max_gap = 0
    for i in range(len(bearings)):
        gap = bearings[(i + 1) % len(bearings)] - bearings[i]
        if gap < 0:
            gap += 360
        max_gap = max(max_gap, gap)

    # Return the complement of the largest gap (actual spread)
    return 360 - max_gap if max_gap > 180 else max_gap


# STEP 3: SEQUENCE-AWARE DRIVER ASSIGNMENT
def assign_drivers_by_priority(user_df, driver_df, office_lat, office_lon):
    """
    Step 3: Assign drivers based on priority and proximity using sequence-aware cost
    """
    print("🚗 Step 3: Assigning drivers by priority...")

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

        # Check bearing coherence and split if necessary
        bearings = unassigned_in_cluster['bearing_from_office'].values
        if len(bearings) > 1:
            bearing_diffs = []
            for i in range(len(bearings)):
                for j in range(i + 1, len(bearings)):
                    bearing_diffs.append(bearing_difference(bearings[i], bearings[j]))
            
            if bearing_diffs and max(bearing_diffs) > MAX_BEARING_DIFFERENCE:
                print(
                    f"  📍 Splitting sub-cluster {sub_cluster_id} due to bearing spread ({max(bearing_diffs):.1f}°)"
                )
                # Split into 2 sub-groups based on bearing
                coords_with_bearing = np.column_stack([
                    unassigned_in_cluster[['latitude', 'longitude']].values,
                    unassigned_in_cluster[['bearing_sin',
                                           'bearing_cos']].values
                ])
                kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
                split_labels = kmeans.fit_predict(coords_with_bearing)

                # Process each split separately
                for split_id in range(2):
                    split_users = unassigned_in_cluster[split_labels == split_id]
                    if len(split_users) > 0:
                        route = assign_best_driver_to_cluster(
                            split_users, available_drivers, used_driver_ids,
                            office_lat, office_lon)
                        if route:
                            routes.append(route)
                            assigned_user_ids.update(
                                u['user_id'] for u in route['assigned_users'])
                continue

        # Assign best driver to this cluster if not split
        route = assign_best_driver_to_cluster(unassigned_in_cluster,
                                              available_drivers,
                                              used_driver_ids, office_lat,
                                              office_lon)

        if route:
            routes.append(route)
            assigned_user_ids.update(u['user_id']
                                     for u in route['assigned_users'])

    # Apply route splitting for poor quality routes
    routes = apply_route_splitting(routes, available_drivers, used_driver_ids, office_lat, office_lon)

    print(
        f"  ✅ Created {len(routes)} initial routes with priority-based assignment"
    )
    return routes, assigned_user_ids


def assign_best_driver_to_cluster(cluster_users, available_drivers,
                                  used_driver_ids, office_lat, office_lon):
    """Find and assign the best available driver to a cluster with sequence-aware cost"""
    cluster_size = len(cluster_users)

    best_driver = None
    min_cost = float('inf')
    best_sequence = None

    for _, driver in available_drivers.iterrows():
        if driver['driver_id'] in used_driver_ids:
            continue

        # Check capacity
        if driver['capacity'] < cluster_size:
            continue

        # Calculate sequence-aware route cost
        route_cost, sequence, normalized_turning_score = calculate_sequence_aware_route_cost(
            driver, cluster_users, office_lat, office_lon
        )

        # Priority penalty (lower priority = higher penalty)
        priority_penalty = driver['priority'] * 2.0

        # Utilization bonus (prefer fuller vehicles)
        utilization = cluster_size / driver['capacity']
        utilization_bonus = utilization * 3.0

        # Adaptive zigzag penalty based on route characteristics
        base_zigzag_weight = _config.get('zigzag_penalty_weight', 2.0)
        
        # Scale penalty by route characteristics
        route_length_factor = max(1.0, cluster_size / 5.0)  # Heavier penalty on short routes
        adaptive_zigzag_weight = base_zigzag_weight / route_length_factor
        
        # Normalized turning penalty (now in same scale as km)
        zigzag_penalty = normalized_turning_score * adaptive_zigzag_weight

        total_cost = route_cost + priority_penalty - utilization_bonus + zigzag_penalty

        if total_cost < min_cost:
            min_cost = total_cost
            best_driver = driver
            best_sequence = sequence

    if best_driver is not None:
        # Update used driver ID and sequence
        used_driver_ids.add(best_driver['driver_id'])

        route = {
            'driver_id': str(best_driver['driver_id']),
            'vehicle_id': str(best_driver.get('vehicle_id', '')),
            'vehicle_type': int(best_driver['capacity']),
            'latitude': float(best_driver['latitude']),
            'longitude': float(best_driver['longitude']),
            'assigned_users': []
        }

        # Add users to route in the optimal sequence
        users_to_assign = cluster_users[cluster_users['user_id'].isin(
            [u['user_id'] for u in best_sequence])]
        
        # Ensure users are added in the correct sequence
        ordered_users_to_assign = []
        for seq_user in best_sequence:
            for _, cluster_user in users_to_assign.iterrows():
                if cluster_user['user_id'] == seq_user['user_id']:
                    ordered_users_to_assign.append(cluster_user)
                    break

        for user in ordered_users_to_assign:
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
        update_route_metrics(route, office_lat, office_lon)
        return route

    return None

def calculate_sequence_aware_route_cost(driver, cluster_users, office_lat, office_lon):
    """Calculate actual route cost including pickup sequence and normalized turning penalty"""
    if len(cluster_users) == 0:
        return float('inf'), [], 0

    driver_pos = (driver['latitude'], driver['longitude'])
    office_pos = (office_lat, office_lon)

    # Get optimal pickup sequence
    sequence = calculate_optimal_sequence(driver_pos, cluster_users, office_pos)

    # Calculate total route distance
    total_distance = 0
    turning_angles = []

    # Driver to first pickup
    if sequence:
        first_user = sequence[0]
        total_distance += haversine_distance(
            driver_pos[0], driver_pos[1], 
            first_user['latitude'], first_user['longitude']
        )

    # Between pickups
    for i in range(len(sequence) - 1):
        current_user = sequence[i]
        next_user = sequence[i + 1]

        distance = haversine_distance(
            current_user['latitude'], current_user['longitude'],
            next_user['latitude'], next_user['longitude']
        )
        total_distance += distance

        # Calculate turning angle
        if i == 0:
            # Angle from driver->current->next
            prev_pos = driver_pos
        else:
            prev_pos = (sequence[i-1]['latitude'], sequence[i-1]['longitude'])

        current_pos = (current_user['latitude'], current_user['longitude'])
        next_pos = (next_user['latitude'], next_user['longitude'])

        turning_angle = calculate_turning_angle(prev_pos, current_pos, next_pos)
        # Square large angles to heavily penalize reversals
        angle_penalty = abs(turning_angle)
        if angle_penalty > 90:
            angle_penalty = angle_penalty ** 1.5  # Exponential penalty for large turns
        turning_angles.append(angle_penalty)

    # Last pickup to office
    if sequence:
        last_user = sequence[-1]
        total_distance += haversine_distance(
            last_user['latitude'], last_user['longitude'],
            office_lat, office_lon
        )

    # Calculate normalized turning score
    raw_turning_score = sum(turning_angles) / len(turning_angles) if turning_angles else 0
    
    # Normalize to [0,1] range where 1 = max_allowed_turning_score
    max_allowed_turning = _config.get('max_allowed_turning_score', 40)
    normalized_turning_score = min(raw_turning_score / max_allowed_turning, 2.0)  # Cap at 2x

    return total_distance, sequence, normalized_turning_score

def calculate_optimal_sequence(driver_pos, cluster_users, office_pos):
    """Calculate optimal pickup sequence using geodesic bearing+distance projection"""
    if len(cluster_users) <= 1:
        return cluster_users.to_dict('records') if hasattr(cluster_users, 'to_dict') else list(cluster_users)

    users_list = cluster_users.to_dict('records') if hasattr(cluster_users, 'to_dict') else list(cluster_users)

    # Calculate main route bearing (driver to office)
    main_route_bearing = calculate_bearing(driver_pos[0], driver_pos[1], office_pos[0], office_pos[1])

    # Sort users by geodesic projection along route axis
    def geodesic_projection_score(user):
        # Distance from driver to user
        distance = haversine_distance(driver_pos[0], driver_pos[1], 
                                    user['latitude'], user['longitude'])
        
        # Bearing from driver to user
        user_bearing = calculate_bearing(driver_pos[0], driver_pos[1], 
                                       user['latitude'], user['longitude'])
        
        # Bearing difference from main route direction
        bearing_diff = normalize_bearing_difference(user_bearing - main_route_bearing)
        bearing_diff_rad = math.radians(abs(bearing_diff))
        
        # Geodesic projection: distance * cos(bearing_difference)
        # Users aligned with route direction get higher scores (sorted first)
        return distance * math.cos(bearing_diff_rad)

    users_list.sort(key=geodesic_projection_score, reverse=True)

    # Apply local 2-opt improvement while respecting directional constraints
    return apply_direction_aware_2opt(users_list, driver_pos, office_pos)

def apply_direction_aware_2opt(sequence, driver_pos, office_pos):
    """Apply 2-opt improvements with adaptive turning angle constraints"""
    if len(sequence) <= 2:
        return sequence

    improved = True
    
    # Stricter adaptive max turning angle based on route length
    base_max_turning_angle = _config.get('max_turning_angle_base', 45)  # Reduced from 60
    route_length = len(sequence)
    
    # Much tighter constraints for shorter routes where backtracking is avoidable
    if route_length <= 3:
        max_turning_angle = base_max_turning_angle * 0.6  # ~27 degrees
    elif route_length <= 5:
        max_turning_angle = base_max_turning_angle * 0.75  # ~34 degrees
    else:
        max_turning_angle = base_max_turning_angle  # 45 degrees

    # Calculate initial quality to determine strictness
    initial_turning = calculate_sequence_turning_score(sequence, driver_pos, office_pos)
    is_poor_quality = initial_turning > _config.get('max_allowed_turning_score', 35)

    while improved:
        improved = False
        best_distance = calculate_sequence_distance(sequence, driver_pos, office_pos)
        best_turning_score = calculate_sequence_turning_score(sequence, driver_pos, office_pos)

        for i in range(len(sequence) - 1):
            for j in range(i + 2, len(sequence)):
                # Try 2-opt swap
                new_sequence = sequence[:i+1] + sequence[i+1:j+1][::-1] + sequence[j+1:]

                # Calculate new metrics
                new_distance = calculate_sequence_distance(new_sequence, driver_pos, office_pos)
                new_turning_score = calculate_sequence_turning_score(new_sequence, driver_pos, office_pos)
                
                # Stricter acceptance criteria for poor quality routes
                if is_poor_quality:
                    # For poor routes, require both distance AND turning improvement
                    distance_improved = new_distance < best_distance * 0.98  # At least 2% improvement
                    turning_improved = new_turning_score < best_turning_score * 0.95  # At least 5% improvement
                    
                    if distance_improved and turning_improved:
                        sequence = new_sequence
                        best_distance = new_distance
                        best_turning_score = new_turning_score
                        improved = True
                        break
                else:
                    # For good routes, use relaxed criteria
                    distance_improved = new_distance < best_distance
                    turning_acceptable = (new_turning_score <= best_turning_score + 3.0 or  # Less degradation allowed
                                        new_turning_score <= max_turning_angle)
                    
                    if distance_improved and turning_acceptable:
                        sequence = new_sequence
                        best_distance = new_distance
                        best_turning_score = new_turning_score
                        improved = True
                        break
            if improved:
                break

    return sequence


def calculate_sequence_turning_score(sequence, driver_pos, office_pos):
    """Calculate average turning angle for a sequence"""
    if len(sequence) <= 1:
        return 0

    users_format = []
    for user in sequence:
        if 'lat' in user:
            users_format.append(user)
        else:
            users_format.append({'lat': user['latitude'], 'lng': user['longitude']})

    return calculate_route_turning_score(users_format, driver_pos, office_pos)

def calculate_sequence_distance(sequence, driver_pos, office_pos):
    """Calculate total distance for a pickup sequence"""
    if not sequence:
        return 0

    total = haversine_distance(driver_pos[0], driver_pos[1], 
                              sequence[0]['latitude'], sequence[0]['longitude'])

    for i in range(len(sequence) - 1):
        total += haversine_distance(
            sequence[i]['latitude'], sequence[i]['longitude'],
            sequence[i+1]['latitude'], sequence[i+1]['longitude']
        )

    total += haversine_distance(
        sequence[-1]['latitude'], sequence[-1]['longitude'],
        office_pos[0], office_pos[1]
    )

    return total

def has_acceptable_turns(sequence, driver_pos, office_pos, max_angle):
    """Check if sequence has acceptable turning angles"""
    if len(sequence) <= 1:
        return True

    for i in range(len(sequence)):
        if i == 0:
            prev_pos = driver_pos
        else:
            prev_pos = (sequence[i-1]['latitude'], sequence[i-1]['longitude'])

        current_pos = (sequence[i]['latitude'], sequence[i]['longitude'])

        if i == len(sequence) - 1:
            next_pos = office_pos
        else:
            next_pos = (sequence[i+1]['latitude'], sequence[i+1]['longitude'])

        turning_angle = abs(calculate_turning_angle(prev_pos, current_pos, next_pos))
        if turning_angle > max_angle:
            return False

    return True

def calculate_turning_angle(prev_pos, current_pos, next_pos):
    """Calculate turning angle at current position"""
    import math

    # Vectors
    v1 = (current_pos[0] - prev_pos[0], current_pos[1] - prev_pos[1])
    v2 = (next_pos[0] - current_pos[0], next_pos[1] - current_pos[1])

    # Calculate angle between vectors
    dot_product = v1[0] * v2[0] + v1[1] * v2[1]
    mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
    mag2 = math.sqrt(v2[0]**2 + v2[1]**2)

    if mag1 == 0 or mag2 == 0:
        return 0

    cos_angle = dot_product / (mag1 * mag2)
    cos_angle = max(-1, min(1, cos_angle))  # Clamp to valid range

    angle_rad = math.acos(cos_angle)
    angle_deg = math.degrees(angle_rad)

    return angle_deg


def apply_route_splitting(routes, available_drivers, used_driver_ids, office_lat, office_lon):
    """Split routes with poor quality metrics into better sub-routes"""
    split_routes = []
    
    turning_threshold = _config.get('route_split_turning_threshold', 30)
    consistency_threshold = _config.get('route_split_consistency_threshold', 0.7)
    
    for route in routes:
        if len(route['assigned_users']) <= 2:
            split_routes.append(route)
            continue
            
        # Calculate route quality metrics
        driver_pos = (route['latitude'], route['longitude'])
        office_pos = (office_lat, office_lon)
        
        turning_score = calculate_route_turning_score(route['assigned_users'], driver_pos, office_pos)
        direction_consistency = calculate_direction_consistency(route['assigned_users'], driver_pos, office_pos)
        
        # Check if route needs splitting
        needs_split = (turning_score > turning_threshold or 
                      direction_consistency < consistency_threshold)
        
        if needs_split and len(route['assigned_users']) >= 4:
            print(f"  🔄 Splitting route {route['driver_id']} - turning: {turning_score:.1f}°, consistency: {direction_consistency:.2f}")
            
            # Split route by bearing sectors
            sub_routes = split_route_by_bearing(route, available_drivers, used_driver_ids, office_lat, office_lon)
            split_routes.extend(sub_routes)
        else:
            split_routes.append(route)
    
    return split_routes


def split_route_by_bearing(route, available_drivers, used_driver_ids, office_lat, office_lon):
    """Split a route into bearing-consistent sub-routes"""
    users = route['assigned_users']
    if len(users) <= 2:
        return [route]
    
    # Calculate bearings from office to each user
    user_bearings = []
    for user in users:
        bearing = calculate_bearing(office_lat, office_lon, user['lat'], user['lng'])
        user_bearings.append((user, bearing))
    
    # Sort by bearing and split into 2 groups
    user_bearings.sort(key=lambda x: x[1])
    mid_point = len(user_bearings) // 2
    
    group1_users = [ub[0] for ub in user_bearings[:mid_point]]
    group2_users = [ub[0] for ub in user_bearings[mid_point:]]
    
    sub_routes = []
    
    # Create route for group 1
    if group1_users:
        route1 = create_sub_route(route, group1_users, available_drivers, used_driver_ids, office_lat, office_lon)
        if route1:
            sub_routes.append(route1)
    
    # Create route for group 2
    if group2_users:
        route2 = create_sub_route(route, group2_users, available_drivers, used_driver_ids, office_lat, office_lon)
        if route2:
            sub_routes.append(route2)
    
    # If couldn't split (no available drivers), return original
    if not sub_routes:
        return [route]
    
    return sub_routes


def create_sub_route(original_route, users, available_drivers, used_driver_ids, office_lat, office_lon):
    """Create a new sub-route from split users"""
    # Try to find an available driver
    for _, driver in available_drivers.iterrows():
        if driver['driver_id'] in used_driver_ids:
            continue
        if driver['capacity'] < len(users):
            continue
            
        # Create new route
        new_route = {
            'driver_id': str(driver['driver_id']),
            'vehicle_id': str(driver.get('vehicle_id', '')),
            'vehicle_type': int(driver['capacity']),
            'latitude': float(driver['latitude']),
            'longitude': float(driver['longitude']),
            'assigned_users': users
        }
        
        # Optimize sequence and update metrics
        new_route = optimize_route_sequence(new_route, office_lat, office_lon)
        update_route_metrics(new_route, office_lat, office_lon)
        
        # Mark driver as used
        used_driver_ids.add(driver['driver_id'])
        
        return new_route
    
    # If no driver available, return None
    return None


# STEP 4: LOCAL OPTIMIZATION
def local_optimization(routes, office_lat, office_lon):
    """
    Step 4: Local optimization within routes and between nearby routes
    """
    print("🔧 Step 4: Local optimization...")

    improved = True
    iterations = 0

    while improved and iterations < MAX_SWAP_ITERATIONS:
        improved = False
        iterations += 1

        # Optimize user sequence within each route
        for route in routes:
            original_sequence = route['assigned_users'].copy()
            route = optimize_route_sequence(route, office_lat, office_lon)
            if route['assigned_users'] != original_sequence:
                improved = True
        
        # Try swapping users between nearby routes
        if try_user_swap(routes, office_lat, office_lon, _config):
            improved = True

    print(f"  ✅ Local optimization completed in {iterations} iterations")
    return routes

def calculate_route_cost_with_quality(route, office_lat, office_lon):
    """Calculate route cost including distance and directional quality metrics"""
    if not route['assigned_users']:
        return 0, {'turning_score': 0, 'tortuosity': 1.0}

    # Calculate basic distance cost
    distance_cost = calculate_route_cost(route, office_lat, office_lon)

    # Calculate quality metrics
    driver_pos = (route['latitude'], route['longitude'])
    office_pos = (office_lat, office_lon)

    turning_score = calculate_route_turning_score(route['assigned_users'], driver_pos, office_pos)
    tortuosity = calculate_tortuosity_ratio(route['assigned_users'], driver_pos, office_pos)

    quality_metrics = {
        'turning_score': turning_score,
        'tortuosity': tortuosity
    }

    return distance_cost, quality_metrics


def calculate_route_cost(route, office_lat, office_lon):
    """Calculate cost of a route based on distances"""
    if not route['assigned_users']:
        return 0

    total_cost = 0
    driver_pos = (route['latitude'], route['longitude'])
    office_pos = (office_lat, office_lon)

    # Cost from driver to users (in sequence)
    current_pos = driver_pos
    for user in route['assigned_users']:
        user_pos = (user['lat'], user['lng'])
        total_cost += haversine_distance(current_pos[0], current_pos[1],
                                         user_pos[0], user_pos[1])
        current_pos = user_pos

    # Cost from last user to office
    total_cost += haversine_distance(current_pos[0], current_pos[1],
                                     office_pos[0], office_pos[1])
    
    # Add penalty for low utilization
    utilization = len(route['assigned_users']) / route['vehicle_type']
    if utilization < _config['LOW_UTILIZATION_THRESHOLD']:
        total_cost += (_config['LOW_UTILIZATION_THRESHOLD'] - utilization) * 5.0 # Penalty for very low utilization

    return total_cost


def update_route_metrics(route, office_lat, office_lon):
    """Update route metrics after modifications (sequence, assignments)"""
    if route['assigned_users']:
        # Recalculate sequence and metrics if users exist
        driver_pos = (route['latitude'], route['longitude'])
        office_pos = (office_lat, office_lon)
        
        # Ensure users are in a format suitable for sequencing
        users_for_sequencing = [{'latitude': u['lat'], 'longitude': u['lng'], 'user_id': u['user_id']} 
                                for u in route['assigned_users']]

        # Re-optimize sequence and update route with new metrics
        optimized_sequence = calculate_optimal_sequence(driver_pos, users_for_sequencing, office_pos)
        
        final_sequence = []
        for seq_user in optimized_sequence:
            for orig_user in route['assigned_users']:
                if orig_user['user_id'] == seq_user['user_id']:
                    final_sequence.append(orig_user)
                    break
        route['assigned_users'] = final_sequence

        route['centroid'] = calculate_users_center(route['assigned_users'])
        route['utilization'] = len(route['assigned_users']) / route['vehicle_type']
        route['turning_score'] = calculate_route_turning_score(route['assigned_users'], driver_pos, office_pos)
        route['tortuosity_ratio'] = calculate_tortuosity_ratio(route['assigned_users'], driver_pos, office_pos)
        route['direction_consistency'] = calculate_direction_consistency(route['assigned_users'], driver_pos, office_pos)
    else:
        # If route becomes empty, reset metrics
        route['centroid'] = [route['latitude'], route['longitude']]
        route['utilization'] = 0
        route['turning_score'] = 0
        route['tortuosity_ratio'] = 1.0
        route['direction_consistency'] = 1.0


def optimize_route_sequence(route, office_lat, office_lon):
    """Optimize pickup sequence using direction-aware sorting and local optimization"""
    if not route['assigned_users'] or len(route['assigned_users']) <= 1:
        return route

    users = route['assigned_users'].copy()
    driver_pos = (route['latitude'], route['longitude'])
    office_pos = (office_lat, office_lon)

    # Convert users to consistent format for sequencing
    users_for_sequencing = []
    for user in users:
        users_for_sequencing.append({
            'latitude': user['lat'],
            'longitude': user['lng'],
            'user_id': user['user_id']
        })

    # Calculate optimal sequence using direction-aware method
    optimized_sequence = calculate_optimal_sequence(driver_pos, users_for_sequencing, office_pos)

    # Convert back to original format
    final_sequence = []
    for seq_user in optimized_sequence:
        for orig_user in users:
            if orig_user['user_id'] == seq_user['user_id']:
                final_sequence.append(orig_user)
                break

    route['assigned_users'] = final_sequence

    # Add route quality metrics
    route['turning_score'] = calculate_route_turning_score(final_sequence, driver_pos, office_pos)
    route['tortuosity_ratio'] = calculate_tortuosity_ratio(final_sequence, driver_pos, office_pos)
    route['direction_consistency'] = calculate_direction_consistency(final_sequence, driver_pos, office_pos)

    return route

def calculate_route_turning_score(users, driver_pos, office_pos):
    """Calculate the average turning angle for a route with input validation"""
    if len(users) <= 1:
        return 0

    # Validate user format
    for user in users:
        if 'lat' not in user or 'lng' not in user:
            raise ValueError(f"User missing required keys 'lat'/'lng': {user.keys()}")

    turning_angles = []

    for i in range(len(users)):
        if i == 0:
            prev_pos = driver_pos
        else:
            prev_pos = (users[i-1]['lat'], users[i-1]['lng'])

        current_pos = (users[i]['lat'], users[i]['lng'])

        if i == len(users) - 1:
            next_pos = office_pos
        else:
            next_pos = (users[i+1]['lat'], users[i+1]['lng'])

        angle = calculate_turning_angle(prev_pos, current_pos, next_pos)
        turning_angles.append(abs(angle))

    return sum(turning_angles) / len(turning_angles) if turning_angles else 0

def calculate_tortuosity_ratio(users, driver_pos, office_pos):
    """Calculate ratio of actual route length to straight line distance with validation"""
    if not users:
        return 1.0

    # Validate user format
    for user in users:
        if 'lat' not in user or 'lng' not in user:
            raise ValueError(f"User missing required keys 'lat'/'lng': {user.keys()}")

    # Actual route distance
    actual_distance = calculate_sequence_distance(
        [{'latitude': u['lat'], 'longitude': u['lng']} for u in users],
        driver_pos, office_pos
    )

    # Straight line distance (driver to centroid to office)
    if users:
        centroid_lat = sum(u['lat'] for u in users) / len(users)
        centroid_lng = sum(u['lng'] for u in users) / len(users)

        straight_distance = (
            haversine_distance(driver_pos[0], driver_pos[1], centroid_lat, centroid_lng) +
            haversine_distance(centroid_lat, centroid_lng, office_pos[0], office_pos[1])
        )
    else:
        straight_distance = haversine_distance(driver_pos[0], driver_pos[1], office_pos[0], office_pos[1])

    return actual_distance / straight_distance if straight_distance > 0 else 1.0

def calculate_direction_consistency(users, driver_pos, office_pos):
    """Calculate fraction of route segments going in consistent direction"""
    if len(users) <= 1:
        return 1.0

    # Calculate main route bearing (driver to office)
    main_bearing = calculate_bearing(driver_pos[0], driver_pos[1], office_pos[0], office_pos[1])

    consistent_segments = 0
    total_segments = 0

    # Check each segment
    for i in range(len(users)):
        if i == 0:
            prev_pos = driver_pos
        else:
            prev_pos = (users[i-1]['lat'], users[i-1]['lng'])

        current_pos = (users[i]['lat'], users[i]['lng'])

        segment_bearing = calculate_bearing(prev_pos[0], prev_pos[1], current_pos[0], current_pos[1])
        bearing_diff = abs(normalize_bearing_difference(segment_bearing - main_bearing))

        if bearing_diff <= 45:  # Within 45 degrees of main direction
            consistent_segments += 1
        total_segments += 1

    return consistent_segments / total_segments if total_segments > 0 else 1.0

def normalize_bearing_difference(diff):
    """Normalize bearing difference to [-180, 180] range"""
    while diff > 180:
        diff -= 360
    while diff < -180:
        diff += 360
    return diff


def try_user_swap(routes, office_lat, office_lon, config):
    """Try swapping users between routes with direction-aware evaluation and nearby driver preference"""
    improvements = 0
    threshold = config.get('SWAP_IMPROVEMENT_THRESHOLD', 0.3)  # Stricter threshold
    max_turning_penalty = config.get('zigzag_penalty_weight', 2.5)

    for i in range(len(routes)):
        for j in range(i + 1, len(routes)):
            route1, route2 = routes[i], routes[j]

            if not route1['assigned_users'] or not route2['assigned_users']:
                continue

            # Calculate distance between route centers (prefer nearby swaps)
            center1 = calculate_route_center(route1)
            center2 = calculate_route_center(route2)
            route_distance = haversine_distance(center1[0], center1[1], center2[0], center2[1])
            
            # Skip distant route swaps to prefer reassignment to nearby drivers
            max_swap_distance = config.get('MERGE_DISTANCE_KM', 1.5) * 1.5  # Allow slightly more than merge distance
            if route_distance > max_swap_distance:
                continue

            # Try swapping each user from route1 to route2
            for user1 in route1['assigned_users'][:]:
                if (len(route2['assigned_users']) + 1 <= route2['vehicle_type']
                        and len(route1['assigned_users']) > 1):

                    # Calculate current costs and quality metrics
                    cost1_before, quality1_before = calculate_route_cost_with_quality(route1, office_lat, office_lon)
                    cost2_before, quality2_before = calculate_route_cost_with_quality(route2, office_lat, office_lon)
                    total_before = cost1_before + cost2_before
                    quality_before = quality1_before['turning_score'] + quality2_before['turning_score']

                    # Perform swap
                    route1['assigned_users'].remove(user1)
                    route2['assigned_users'].append(user1)

                    # Recalculate with optimized sequences
                    route1_optimized = optimize_route_sequence(route1, office_lat, office_lon)
                    route2_optimized = optimize_route_sequence(route2, office_lat, office_lon)

                    cost1_after, quality1_after = calculate_route_cost_with_quality(route1_optimized, office_lat, office_lon)
                    cost2_after, quality2_after = calculate_route_cost_with_quality(route2_optimized, office_lat, office_lon)
                    total_after = cost1_after + cost2_after
                    quality_after = quality1_after['turning_score'] + quality2_after['turning_score']

                    # Combined improvement check (distance + direction quality)
                    distance_improvement = total_before - total_after
                    quality_improvement = quality_before - quality_after

                    # Add proximity bonus (prefer swaps between nearby routes)
                    proximity_bonus = max(0, (max_swap_distance - route_distance) / max_swap_distance) * 0.5
                    
                    total_improvement = distance_improvement + (quality_improvement * max_turning_penalty) + proximity_bonus

                    if total_improvement > threshold:
                        improvements += 1
                        # Keep the optimized sequences
                        routes[i] = route1_optimized
                        routes[j] = route2_optimized
                    else:
                        # Revert swap
                        route2['assigned_users'].remove(user1)
                        route1['assigned_users'].append(user1)

    return improvements


# STEP 5: GLOBAL OPTIMIZATION
def global_optimization(routes, user_df, assigned_user_ids, driver_df, office_lat, office_lon):
    """
    Enhanced Step 5: Global optimization with quality-aware merging and proactive splitting
    """
    print("🌍 Step 5: Enhanced Global optimization...")

    # PHASE 1: Proactive route splitting for poor quality routes
    print("  🔄 Phase 1: Proactive route quality improvement...")
    routes = proactive_route_splitting(routes, driver_df, office_lat, office_lon)
    
    # PHASE 2: Fill underutilized routes with quality checks
    print("  📈 Phase 2: Quality-aware route filling...")
    unassigned_users_df = user_df[~user_df['user_id'].isin(assigned_user_ids)].copy()
    routes = quality_aware_route_filling(routes, unassigned_users_df, assigned_user_ids, office_lat, office_lon)

    # PHASE 3: Conservative quality-first merging
    print("  🔗 Phase 3: Quality-first route merging...")
    routes = conservative_quality_merging(routes, driver_df, office_lat, office_lon)

    # PHASE 4: Final cleanup and optimization
    print("  🧹 Phase 4: Final optimization pass...")
    routes = final_optimization_cleanup(routes, office_lat, office_lon)

    # Handle remaining unassigned users
    remaining_unassigned_users_df = user_df[~user_df['user_id'].isin(assigned_user_ids)]
    unassigned_list = handle_remaining_users(remaining_unassigned_users_df)

    print("  ✅ Enhanced global optimization completed")
    return routes, unassigned_list


def merge_compatible_routes(routes, office_lat, office_lon, config):
    """Merge compatible underutilized routes with multiple passes"""
    current_routes = routes.copy()
    merged_count = 0
    max_passes = 3  # Multiple passes to catch indirect merges

    for pass_num in range(max_passes):
        merged_routes_this_pass = []
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

                # Check if routes can be merged
                if can_merge_routes(route_a, route_b, office_lat, office_lon, config):
                    # Calculate merge score (lower is better)
                    center_a = calculate_route_center(route_a)
                    center_b = calculate_route_center(route_b)
                    distance = haversine_distance(center_a[0], center_a[1],
                                                  center_b[0], center_b[1])

                    # Prefer merging with closer, less utilized routes
                    util_penalty = (route_b.get('utilization', 0) * 2
                                    )  # Prefer lower utilization
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
                update_route_metrics(merged_route, office_lat, office_lon)

                merged_routes_this_pass.append(merged_route)
                used_route_indices.add(orig_i)
                used_route_indices.add(best_candidate_index)
                pass_merges += 1
                merged_count += 1

                # print(
                #     f"    🔗 Pass {pass_num + 1}: Merged routes with utilizations {route_a.get('utilization', 0):.1%} + {best_merge_candidate.get('utilization', 0):.1%} → {merged_route.get('utilization', 0):.1%}"
                # )
            else:
                # No merge candidate found, keep original route
                merged_routes_this_pass.append(route_a)
                used_route_indices.add(orig_i)

        current_routes = merged_routes_this_pass
        # print(f"    📊 Pass {pass_num + 1}: {pass_merges} merges completed")

        # If no merges happened this pass, we're done
        if pass_merges == 0:
            break

    if merged_count > 0:
        print(
            f"  ✅ Total merges: {merged_count}, Final routes: {len(current_routes)}"
        )

    return current_routes


def can_merge_routes(route1, route2, office_lat, office_lon, config):
    """Check if two routes can be merged with strict direction-aware constraints"""
    total_users = len(route1['assigned_users']) + len(route2['assigned_users'])
    max_capacity = max(route1['vehicle_type'], route2['vehicle_type'])

    if total_users > max_capacity:
        return False

    # Stricter distance constraint
    center1 = calculate_route_center(route1)
    center2 = calculate_route_center(route2)
    distance = haversine_distance(center1[0], center1[1], center2[0], center2[1])

    merge_distance = config.get('MERGE_DISTANCE_KM', 1.5)  # Further reduced
    if distance > merge_distance:
        return False

    # Stricter bearing constraint
    bearing1 = calculate_average_bearing(route1, office_lat, office_lon)
    bearing2 = calculate_average_bearing(route2, office_lat, office_lon)
    bearing_diff = abs(bearing1 - bearing2)
    bearing_diff = min(bearing_diff, 360 - bearing_diff)

    max_bearing_diff = config.get('max_bearing_difference', 12)  # Further reduced
    if bearing_diff > max_bearing_diff:
        return False

    # Check current tortuosity and require improvement if merging high-tortuosity routes
    tortuosity1 = route1.get('tortuosity_ratio', 1.0)
    tortuosity2 = route2.get('tortuosity_ratio', 1.0)
    max_current_tortuosity = max(tortuosity1, tortuosity2)
    
    # Simulate merged route to check quality
    merged_quality = simulate_merge_quality(route1, route2, office_lat, office_lon, config)
    if not merged_quality['acceptable']:
        return False
    
    # If either route has high tortuosity, require merged route to improve it
    require_tortuosity_improvement = config.get('merge_tortuosity_improvement_required', True)
    if require_tortuosity_improvement and max_current_tortuosity > 1.3:
        if merged_quality['tortuosity'] >= max_current_tortuosity * 0.95:  # Must improve by at least 5%
            return False

    return True

def simulate_merge_quality(route1, route2, office_lat, office_lon, config):
    """Simulate merging routes and check if result is acceptable with conservative merge criteria"""
    # Create temporary merged route
    merged_users = route1['assigned_users'] + route2['assigned_users']

    # Choose better positioned driver
    driver1_pos = (route1['latitude'], route1['longitude'])
    driver2_pos = (route2['latitude'], route2['longitude'])
    office_pos = (office_lat, office_lon)

    # Calculate which driver position would be better for merged route
    users_center = calculate_users_center(merged_users)

    dist1 = haversine_distance(driver1_pos[0], driver1_pos[1], users_center[0], users_center[1])
    dist2 = haversine_distance(driver2_pos[0], driver2_pos[1], users_center[0], users_center[1])

    better_driver_pos = driver1_pos if dist1 <= dist2 else driver2_pos

    # Normalize user objects to consistent format
    normalized_users = [normalize_user_format(u) for u in merged_users]

    sequence = calculate_optimal_sequence(better_driver_pos, 
                                        [{'latitude': u['lat'], 'longitude': u['lng'], 'user_id': u['user_id']} 
                                         for u in normalized_users], office_pos)

    # Calculate quality metrics with normalized format
    normalized_sequence = [normalize_user_format(u) for u in sequence]
    
    turning_score = calculate_route_turning_score(normalized_sequence, better_driver_pos, office_pos)
    tortuosity = calculate_tortuosity_ratio(normalized_sequence, better_driver_pos, office_pos)

    # Calculate original route metrics for comparison
    orig1_turning = route1.get('turning_score', 0)
    orig2_turning = route2.get('turning_score', 0)
    max_orig_turning = max(orig1_turning, orig2_turning)

    # Conservative merge criteria - must improve or maintain quality
    max_turning_score = config.get('max_allowed_turning_score', 40)  # Stricter default
    max_tortuosity = config.get('max_tortuosity_ratio', 1.6)  # Stricter default
    
    # Require merged route to not degrade turning score significantly
    turning_degradation_allowed = 5.0  # degrees
    
    quality_acceptable = (
        turning_score <= max_turning_score and 
        tortuosity <= max_tortuosity and
        turning_score <= (max_orig_turning + turning_degradation_allowed)
    )

    return {
        'acceptable': quality_acceptable,
        'turning_score': turning_score,
        'tortuosity': tortuosity,
        'user_count': len(merged_users),
        'max_original_turning': max_orig_turning
    }


def normalize_user_format(user):
    """Normalize user object to consistent {'lat', 'lng', 'user_id'} format"""
    if 'lat' in user and 'lng' in user:
        return user
    elif 'latitude' in user and 'longitude' in user:
        return {
            'lat': user['latitude'],
            'lng': user['longitude'],
            'user_id': user.get('user_id', ''),
            **{k: v for k, v in user.items() if k not in ['latitude', 'longitude']}
        }
    else:
        raise ValueError(f"User object missing coordinate keys: {user.keys()}")

def calculate_users_center(users):
    """Calculate center point of a list of users"""
    if not users:
        return (0, 0)

    avg_lat = sum(u['lat'] for u in users) / len(users)
    avg_lng = sum(u['lng'] for u in users) / len(users)
    return (avg_lat, avg_lng)


def calculate_route_center(route):
    """Calculate the geometric center of users in a route"""
    if not route['assigned_users']:
        return (route['latitude'], route['longitude'])
    
    lats = [u['lat'] for u in route['assigned_users']]
    lngs = [u['lng'] for u in route['assigned_users']]
    return (np.mean(lats), np.mean(lngs))


def calculate_average_bearing(route, office_lat, office_lon):
    """Calculate the average bearing of users in a route towards the office"""
    if not route['assigned_users']:
        return calculate_bearing(route['latitude'], route['longitude'], office_lat, office_lon)

    avg_lat = np.mean([u['lat'] for u in route['assigned_users']])
    avg_lng = np.mean([u['lng'] for u in route['assigned_users']])
    
    return calculate_bearing(avg_lat, avg_lng, office_lat, office_lon)


def proactive_route_splitting(routes, driver_df, office_lat, office_lon):
    """Proactively split routes with poor quality metrics"""
    improved_routes = []
    available_drivers = driver_df[~driver_df['driver_id'].isin([r['driver_id'] for r in routes])].copy()
    
    turning_threshold = _config.get('route_split_turning_threshold', 30)
    tortuosity_threshold = _config.get('max_tortuosity_ratio', 1.4)
    consistency_threshold = _config.get('route_split_consistency_threshold', 0.7)
    
    for route in routes:
        if len(route['assigned_users']) < 3:  # Can't meaningfully split small routes
            improved_routes.append(route)
            continue
            
        # Calculate current quality metrics
        driver_pos = (route['latitude'], route['longitude'])
        turning_score = route.get('turning_score', 0)
        tortuosity = route.get('tortuosity_ratio', 1.0)
        consistency = route.get('direction_consistency', 1.0)
        
        # Check if route needs splitting
        needs_split = (
            turning_score > turning_threshold or 
            tortuosity > tortuosity_threshold or 
            consistency < consistency_threshold
        )
        
        if needs_split and len(available_drivers) > 0:
            print(f"    🔄 Splitting poor quality route - turning: {turning_score:.1f}°, tortuosity: {tortuosity:.2f}")
            split_routes = intelligent_route_splitting(route, available_drivers, office_lat, office_lon)
            improved_routes.extend(split_routes)
            
            # Remove used drivers from available pool
            used_driver_ids = [sr['driver_id'] for sr in split_routes if sr['driver_id'] != route['driver_id']]
            available_drivers = available_drivers[~available_drivers['driver_id'].isin(used_driver_ids)]
        else:
            improved_routes.append(route)
    
    return improved_routes


def intelligent_route_splitting(route, available_drivers, office_lat, office_lon):
    """Split a route intelligently based on geographical and directional patterns"""
    users = route['assigned_users']
    if len(users) < 3:
        return [route]
    
    # Calculate bearing spread to determine split strategy
    bearings = []
    for user in users:
        bearing = calculate_bearing(office_lat, office_lon, user['lat'], user['lng'])
        bearings.append(bearing)
    
    bearing_spread = calculate_bearing_spread_from_list(bearings)
    
    # Strategy 1: If large bearing spread, split by direction
    if bearing_spread > 45:
        return split_by_bearing_clusters(route, available_drivers, office_lat, office_lon)
    
    # Strategy 2: If high tortuosity, split by geographical distance
    tortuosity = route.get('tortuosity_ratio', 1.0)
    if tortuosity > 1.5:
        return split_by_geographical_distance(route, available_drivers, office_lat, office_lon)
    
    # Strategy 3: Default - split into balanced groups
    return split_balanced_groups(route, available_drivers, office_lat, office_lon)


def split_by_bearing_clusters(route, available_drivers, office_lat, office_lon):
    """Split route by clustering users with similar bearings"""
    users = route['assigned_users']
    
    # Calculate bearings and cluster
    user_bearings = []
    for user in users:
        bearing = calculate_bearing(office_lat, office_lon, user['lat'], user['lng'])
        user_bearings.append((user, bearing))
    
    # Sort by bearing and find natural break point
    user_bearings.sort(key=lambda x: x[1])
    
    # Find the largest gap in bearings to split
    max_gap = 0
    split_index = len(user_bearings) // 2  # Default to middle
    
    for i in range(len(user_bearings) - 1):
        gap = abs(user_bearings[i+1][1] - user_bearings[i][1])
        if gap > 180:  # Handle circular nature
            gap = 360 - gap
        if gap > max_gap:
            max_gap = gap
            split_index = i + 1
    
    group1 = [ub[0] for ub in user_bearings[:split_index]]
    group2 = [ub[0] for ub in user_bearings[split_index:]]
    
    return create_split_routes(route, [group1, group2], available_drivers, office_lat, office_lon)


def split_by_geographical_distance(route, available_drivers, office_lat, office_lon):
    """Split route by geographical clustering to reduce tortuosity"""
    users = route['assigned_users']
    
    # Use k-means clustering on coordinates
    coords = np.array([[u['lat'], u['lng']] for u in users])
    
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    labels = kmeans.fit_predict(coords)
    
    group1 = [users[i] for i in range(len(users)) if labels[i] == 0]
    group2 = [users[i] for i in range(len(users)) if labels[i] == 1]
    
    return create_split_routes(route, [group1, group2], available_drivers, office_lat, office_lon)


def split_balanced_groups(route, available_drivers, office_lat, office_lon):
    """Split into balanced groups maintaining sequence order"""
    users = route['assigned_users']
    mid_point = len(users) // 2
    
    group1 = users[:mid_point]
    group2 = users[mid_point:]
    
    return create_split_routes(route, [group1, group2], available_drivers, office_lat, office_lon)


def create_split_routes(original_route, user_groups, available_drivers, office_lat, office_lon):
    """Create new routes from split user groups"""
    split_routes = []
    
    for i, group in enumerate(user_groups):
        if not group:
            continue
            
        if i == 0:
            # Keep original driver for first group
            new_route = original_route.copy()
            new_route['assigned_users'] = group
            split_routes.append(new_route)
        else:
            # Find new driver for additional groups
            suitable_driver = find_suitable_driver_for_group(group, available_drivers, office_lat, office_lon)
            if suitable_driver is not None:
                new_route = {
                    'driver_id': str(suitable_driver['driver_id']),
                    'vehicle_id': str(suitable_driver.get('vehicle_id', '')),
                    'vehicle_type': int(suitable_driver['capacity']),
                    'latitude': float(suitable_driver['latitude']),
                    'longitude': float(suitable_driver['longitude']),
                    'assigned_users': group
                }
                split_routes.append(new_route)
            else:
                # If no driver available, merge back with first group
                if split_routes:
                    split_routes[0]['assigned_users'].extend(group)
    
    # Optimize sequences and update metrics for all split routes
    for route in split_routes:
        route = optimize_route_sequence(route, office_lat, office_lon)
        update_route_metrics(route, office_lat, office_lon)
    
    return split_routes if len(split_routes) > 1 else [original_route]


def find_suitable_driver_for_group(user_group, available_drivers, office_lat, office_lon):
    """Find the most suitable driver for a group of users"""
    if available_drivers.empty or len(user_group) == 0:
        return None
    
    group_center = calculate_users_center(user_group)
    best_driver = None
    best_score = float('inf')
    
    for _, driver in available_drivers.iterrows():
        if driver['capacity'] < len(user_group):
            continue
            
        # Calculate distance to group center
        distance = haversine_distance(driver['latitude'], driver['longitude'], 
                                    group_center[0], group_center[1])
        
        # Calculate utilization bonus
        utilization = len(user_group) / driver['capacity']
        utilization_bonus = utilization * 2.0
        
        score = distance - utilization_bonus
        
        if score < best_score:
            best_score = score
            best_driver = driver
    
    return best_driver


def quality_aware_route_filling(routes, unassigned_users_df, assigned_user_ids, office_lat, office_lon):
    """Fill routes with quality degradation checks"""
    for route in routes:
        if len(route['assigned_users']) >= route['vehicle_type'] or unassigned_users_df.empty:
            continue
            
        original_quality = {
            'turning_score': route.get('turning_score', 0),
            'tortuosity': route.get('tortuosity_ratio', 1.0),
            'consistency': route.get('direction_consistency', 1.0)
        }
        
        route_center = route.get('centroid', [route['latitude'], route['longitude']])
        
        # Find compatible users within distance
        compatible_users = []
        for _, user in unassigned_users_df.iterrows():
            distance = haversine_distance(route_center[0], route_center[1],
                                        user['latitude'], user['longitude'])
            
            if distance <= MAX_FILL_DISTANCE_KM:
                # Check bearing compatibility
                if route['assigned_users']:
                    route_bearing = calculate_average_bearing(route, office_lat, office_lon)
                else:
                    route_bearing = calculate_bearing(route['latitude'], route['longitude'], office_lat, office_lon)
                
                user_bearing = user['bearing_from_office']
                bearing_diff = abs(normalize_bearing_difference(user_bearing - route_bearing))
                
                if bearing_diff <= MAX_BEARING_DIFFERENCE:
                    compatible_users.append((user, distance))
        
        # Sort by distance and try adding users one by one with quality checks
        compatible_users.sort(key=lambda x: x[1])
        slots_available = route['vehicle_type'] - len(route['assigned_users'])
        
        for user, _ in compatible_users[:slots_available]:
            # Test adding this user
            test_user_data = {
                'user_id': str(user['user_id']),
                'lat': float(user['latitude']),
                'lng': float(user['longitude']),
                'office_distance': float(user.get('office_distance', 0))
            }
            
            # Temporarily add user and check quality
            test_route = route.copy()
            test_route['assigned_users'] = route['assigned_users'] + [test_user_data]
            test_route = optimize_route_sequence(test_route, office_lat, office_lon)
            
            new_turning = test_route.get('turning_score', 0)
            new_tortuosity = test_route.get('tortuosity_ratio', 1.0)
            
            # Only add if quality doesn't degrade significantly
            turning_degradation = new_turning - original_quality['turning_score']
            tortuosity_degradation = new_tortuosity - original_quality['tortuosity']
            
            max_turning_degradation = _config.get('max_allowed_turning_score', 40) * 0.2  # 20% of max
            max_tortuosity_degradation = 0.2  # Absolute degradation limit
            
            if (turning_degradation <= max_turning_degradation and 
                tortuosity_degradation <= max_tortuosity_degradation):
                
                # Add user permanently
                if pd.notna(user.get('first_name')):
                    test_user_data['first_name'] = str(user['first_name'])
                if pd.notna(user.get('email')):
                    test_user_data['email'] = str(user['email'])
                
                route['assigned_users'].append(test_user_data)
                assigned_user_ids.add(user['user_id'])
                unassigned_users_df = unassigned_users_df[unassigned_users_df['user_id'] != user['user_id']]
                
                # Update route with new metrics
                route = optimize_route_sequence(route, office_lat, office_lon)
                update_route_metrics(route, office_lat, office_lon)
                
                # Update original quality for next iteration
                original_quality = {
                    'turning_score': route.get('turning_score', 0),
                    'tortuosity': route.get('tortuosity_ratio', 1.0),
                    'consistency': route.get('direction_consistency', 1.0)
                }
    
    return routes


def conservative_quality_merging(routes, driver_df, office_lat, office_lon):
    """Conservative merging with strict quality preservation"""
    current_routes = routes.copy()
    merged_count = 0
    max_passes = 2  # Reduced passes for more conservative approach
    
    for pass_num in range(max_passes):
        merged_routes_this_pass = []
        used_route_indices = set()
        pass_merges = 0
        
        # Only consider low utilization routes for merging
        low_util_routes = [(i, r) for i, r in enumerate(current_routes) 
                          if r.get('utilization', 1) < 0.7 and len(r['assigned_users']) > 0]
        
        for orig_i, route_a in low_util_routes:
            if orig_i in used_route_indices:
                continue
                
            best_merge_candidate = None
            best_candidate_index = None
            best_quality_improvement = -float('inf')
            
            for orig_j, route_b in low_util_routes:
                if orig_j in used_route_indices or orig_j == orig_i:
                    continue
                
                # Strict compatibility check
                if strict_merge_compatibility(route_a, route_b, office_lat, office_lon):
                    # Calculate quality improvement
                    quality_improvement = calculate_merge_quality_improvement(route_a, route_b, office_lat, office_lon)
                    
                    if quality_improvement > best_quality_improvement and quality_improvement > 0:
                        best_quality_improvement = quality_improvement
                        best_merge_candidate = route_b
                        best_candidate_index = orig_j
            
            if best_merge_candidate is not None:
                # Perform merge
                merged_route = perform_quality_merge(route_a, best_merge_candidate, office_lat, office_lon)
                merged_routes_this_pass.append(merged_route)
                used_route_indices.add(orig_i)
                used_route_indices.add(best_candidate_index)
                pass_merges += 1
                merged_count += 1
            else:
                merged_routes_this_pass.append(route_a)
                used_route_indices.add(orig_i)
        
        # Add routes that weren't considered for merging
        for i, route in enumerate(current_routes):
            if i not in used_route_indices:
                merged_routes_this_pass.append(route)
        
        current_routes = merged_routes_this_pass
        
        if pass_merges == 0:
            break
    
    if merged_count > 0:
        print(f"    🔗 Conservative merges: {merged_count}, Final routes: {len(current_routes)}")
    
    return current_routes


def strict_merge_compatibility(route1, route2, office_lat, office_lon):
    """Very strict compatibility check for merging"""
    total_users = len(route1['assigned_users']) + len(route2['assigned_users'])
    max_capacity = max(route1['vehicle_type'], route2['vehicle_type'])
    
    if total_users > max_capacity:
        return False
    
    # Stricter distance constraint
    center1 = calculate_route_center(route1)
    center2 = calculate_route_center(route2)
    distance = haversine_distance(center1[0], center1[1], center2[0], center2[1])
    
    if distance > _config.get('MERGE_DISTANCE_KM', 1.5) * 0.8:  # 20% stricter
        return False
    
    # Stricter bearing constraint
    bearing1 = calculate_average_bearing(route1, office_lat, office_lon)
    bearing2 = calculate_average_bearing(route2, office_lat, office_lon)
    bearing_diff = abs(normalize_bearing_difference(bearing1 - bearing2))
    
    if bearing_diff > _config.get('max_bearing_difference', 12) * 0.7:  # 30% stricter
        return False
    
    # Both routes must be underutilized
    util1 = route1.get('utilization', 1)
    util2 = route2.get('utilization', 1)
    if util1 > 0.8 or util2 > 0.8:
        return False
    
    return True


def calculate_merge_quality_improvement(route1, route2, office_lat, office_lon):
    """Calculate expected quality improvement from merging"""
    # Current combined metrics
    current_turning = route1.get('turning_score', 0) + route2.get('turning_score', 0)
    current_tortuosity = (route1.get('tortuosity_ratio', 1) + route2.get('tortuosity_ratio', 1)) / 2
    
    # Simulate merge
    merged_simulation = simulate_merge_quality(route1, route2, office_lat, office_lon, _config)
    
    if not merged_simulation['acceptable']:
        return -float('inf')
    
    # Calculate improvement
    turning_improvement = current_turning - merged_simulation['turning_score']
    tortuosity_improvement = current_tortuosity - merged_simulation['tortuosity']
    
    # Weight improvements
    total_improvement = turning_improvement * 0.6 + tortuosity_improvement * 0.4
    
    return total_improvement


def perform_quality_merge(route1, route2, office_lat, office_lon):
    """Perform merge with quality optimization"""
    # Choose better positioned driver
    center1 = calculate_route_center(route1)
    center2 = calculate_route_center(route2)
    
    all_users = route1['assigned_users'] + route2['assigned_users']
    combined_center = calculate_users_center(all_users)
    
    dist1 = haversine_distance(route1['latitude'], route1['longitude'], 
                              combined_center[0], combined_center[1])
    dist2 = haversine_distance(route2['latitude'], route2['longitude'], 
                              combined_center[0], combined_center[1])
    
    better_route = route1 if dist1 <= dist2 else route2
    
    merged_route = better_route.copy()
    merged_route['assigned_users'] = all_users
    merged_route['vehicle_type'] = max(route1['vehicle_type'], route2['vehicle_type'])
    
    # Optimize and update metrics
    merged_route = optimize_route_sequence(merged_route, office_lat, office_lon)
    update_route_metrics(merged_route, office_lat, office_lon)
    
    return merged_route


def final_optimization_cleanup(routes, office_lat, office_lon):
    """Final cleanup and optimization pass"""
    for route in routes:
        if len(route['assigned_users']) > 1:
            # Re-optimize sequence one final time
            route = optimize_route_sequence(route, office_lat, office_lon)
            update_route_metrics(route, office_lat, office_lon)
    
    return routes


def calculate_bearing_spread_from_list(bearings):
    """Calculate the spread of bearings in a list"""
    if len(bearings) <= 1:
        return 0
    
    bearings = sorted(bearings)
    max_gap = 0
    
    for i in range(len(bearings)):
        gap = bearings[(i + 1) % len(bearings)] - bearings[i]
        if gap < 0:
            gap += 360
        max_gap = max(max_gap, gap)
    
    return 360 - max_gap if max_gap > 180 else max_gap


def handle_remaining_users(unassigned_users_df):
    """Handle users that couldn't be assigned to any route"""
    unassigned_list = []
    for _, user in unassigned_users_df.iterrows():
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


# MAIN STRAIGHTFORWARD ASSIGNMENT FUNCTION
def run_assignment(source_id: str, parameter: int = 1, string_param: str = ""):
    """
    Main assignment function with enhanced direction-aware clustering,
    sequence-aware driver assignment, and optimization steps.
    """
    start_time = time.time()

    try:
        print(
            f"🚀 Starting assignment for source_id: {source_id}"
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
        
        print(
            f"📊 DataFrames prepared - Users: {len(user_df)}, Drivers: {len(driver_df)}"
        )

        # STEP 1: Geographic clustering
        user_df = create_geographic_clusters(user_df, office_lat, office_lon, _config)
        clustering_results = {"method": _config['clustering_method'], "clusters": user_df['geo_cluster'].nunique()}


        # STEP 2: Capacity-based sub-clustering (direction-aware)
        user_df = create_capacity_subclusters(user_df, office_lat, office_lon, _config)

        # STEP 3: Priority-based driver assignment (sequence-aware)
        routes, assigned_user_ids = assign_drivers_by_priority(
            user_df, driver_df, office_lat, office_lon)

        # STEP 4: Local optimization (sequence and swap)
        routes = local_optimization(routes, office_lat, office_lon)

        # STEP 5: Global optimization (fill and merge)
        routes, unassigned_users = global_optimization(routes, user_df,
                                                       assigned_user_ids, driver_df, office_lat, office_lon)

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

        # Final metrics update for all routes
        for route in routes:
            update_route_metrics(route, office_lat, office_lon)

        execution_time = time.time() - start_time

        print(
            f"✅ Assignment complete in {execution_time:.2f}s"
        )
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
    turning_scores = []
    tortuosity_ratios = []
    direction_consistencies = []

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
            
            # Collect quality metrics
            turning_scores.append(route.get('turning_score', 0))
            tortuosity_ratios.append(route.get('tortuosity_ratio', 1.0))
            direction_consistencies.append(route.get('direction_consistency', 1.0))

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
        "avg_turning_score":
        round(np.mean(turning_scores), 1) if turning_scores else 0,
        "avg_tortuosity":
        round(np.mean(tortuosity_ratios), 2) if tortuosity_ratios else 1.0,
        "avg_direction_consistency":
        round(np.mean(direction_consistencies) * 100, 1) if direction_consistencies else 100.0,
        "distance_issues":
        distance_issues,
        "clustering_method":
        result.get("clustering_analysis", {}).get("method", "Unknown")
    }

    return analysis