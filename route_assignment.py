
import os
import math
import json
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


# Load and validate configuration with route efficiency settings
def load_and_validate_config():
    """Load configuration for route assignment"""
    try:
        with open('config.json') as f:
            cfg = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Warning: Could not load config.json, using defaults. Error: {e}")
        cfg = {}

    # Get route assignment specific configuration
    route_cfg = cfg.get("route_assignment_config", {})

    print(f"🎯 Using optimization mode: ROUTE EFFICIENCY")
    
    # Validate and set configuration
    config = {}

    # Distance configurations
    config['MAX_FILL_DISTANCE_KM'] = max(0.1, float(route_cfg.get("max_fill_distance_km", 3.5)))
    config['MERGE_DISTANCE_KM'] = max(0.1, float(route_cfg.get("merge_distance_km", 2.0)))
    config['DBSCAN_EPS_KM'] = max(0.1, float(route_cfg.get("dbscan_eps_km", 1.2)))
    config['OVERFLOW_PENALTY_KM'] = max(0.0, float(route_cfg.get("overflow_penalty_km", 8.0)))
    config['DISTANCE_ISSUE_THRESHOLD'] = max(0.1, float(route_cfg.get("distance_issue_threshold_km", 6.0)))
    config['SWAP_IMPROVEMENT_THRESHOLD'] = max(0.0, float(route_cfg.get("swap_improvement_threshold_km", 0.3)))

    # Utilization thresholds (0-1 range)
    config['MIN_UTIL_THRESHOLD'] = max(0.0, min(1.0, float(route_cfg.get("min_util_threshold", 0.4))))
    config['LOW_UTILIZATION_THRESHOLD'] = max(0.0, min(1.0, float(route_cfg.get("low_utilization_threshold", 0.4))))

    # Integer configurations
    config['MIN_SAMPLES_DBSCAN'] = max(1, int(route_cfg.get("min_samples_dbscan", 2)))
    config['MAX_SWAP_ITERATIONS'] = max(1, int(route_cfg.get("max_swap_iterations", 2)))
    config['MAX_USERS_FOR_FALLBACK'] = max(1, int(route_cfg.get("max_users_for_fallback", 2)))
    config['FALLBACK_MIN_USERS'] = max(1, int(route_cfg.get("fallback_min_users", 2)))
    config['FALLBACK_MAX_USERS'] = max(1, int(route_cfg.get("fallback_max_users", 5)))

    # Angle configurations
    config['MAX_BEARING_DIFFERENCE'] = max(0, min(180, float(route_cfg.get("max_bearing_difference", 12))))
    config['MAX_TURNING_ANGLE'] = max(0, min(180, float(route_cfg.get("max_allowed_turning_score", 20))))

    # Cost penalties
    config['UTILIZATION_PENALTY_PER_SEAT'] = max(0.0, float(route_cfg.get("utilization_penalty_per_seat", 0.5)))

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

    # Route efficiency specific parameters
    config['optimization_mode'] = "route_efficiency"
    config['aggressive_merging'] = route_cfg.get('aggressive_merging', False)
    config['capacity_weight'] = route_cfg.get('capacity_weight', 1.0)
    config['direction_weight'] = route_cfg.get('direction_weight', 4.0)

    # Clustering parameters
    config['clustering_method'] = route_cfg.get('clustering_method', 'adaptive')
    config['min_cluster_size'] = max(2, route_cfg.get('min_cluster_size', 2))
    config['use_sweep_algorithm'] = route_cfg.get('use_sweep_algorithm', True)
    config['angular_sectors'] = route_cfg.get('angular_sectors', 12)
    config['max_users_per_initial_cluster'] = route_cfg.get('max_users_per_initial_cluster', 6)
    config['max_users_per_cluster'] = route_cfg.get('max_users_per_cluster', 5)

    # Route quality parameters
    config['zigzag_penalty_weight'] = route_cfg.get('zigzag_penalty_weight', 5.0)
    config['route_split_turning_threshold'] = route_cfg.get('route_split_turning_threshold', 20)
    config['max_tortuosity_ratio'] = route_cfg.get('max_tortuosity_ratio', 1.2)
    config['route_split_consistency_threshold'] = route_cfg.get('route_split_consistency_threshold', 0.8)
    config['merge_tortuosity_improvement_required'] = route_cfg.get('merge_tortuosity_improvement_required', True)
    
    # Distance calculation constants
    config['LAT_TO_KM'] = 111.0
    config['LON_TO_KM'] = 111.0 * math.cos(math.radians(office_lat))

    print(f"   📊 Max bearing difference: {config['MAX_BEARING_DIFFERENCE']}°")
    print(f"   📊 Max turning score: {config['MAX_TURNING_ANGLE']}°")
    print(f"   📊 Max fill distance: {config['MAX_FILL_DISTANCE_KM']}km")
    print(f"   📊 Capacity weight: {config['capacity_weight']}")
    print(f"   📊 Direction weight: {config['direction_weight']}")

    return config


# Load validated configuration - always route efficiency
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


def coords_to_km(lat, lon, office_lat, office_lon):
    """Convert lat/lon coordinates to km from office using local approximation"""
    lat_km = (lat - office_lat) * _config['LAT_TO_KM']
    lon_km = (lon - office_lon) * _config['LON_TO_KM']
    return lat_km, lon_km


def dbscan_clustering_metric(user_df, eps_km, min_samples, office_lat, office_lon):
    """Perform DBSCAN clustering using proper metric coordinates"""
    # Convert coordinates to km from office
    coords_km = []
    for _, user in user_df.iterrows():
        lat_km, lon_km = coords_to_km(user['latitude'], user['longitude'], office_lat, office_lon)
        coords_km.append([lat_km, lon_km])
    
    coords_km = np.array(coords_km)
    
    # Use DBSCAN with eps in km (no scaling needed now)
    dbscan = DBSCAN(eps=eps_km, min_samples=min_samples)
    labels = dbscan.fit_predict(coords_km)

    # Handle noise points: assign to nearest cluster if possible
    noise_mask = labels == -1
    if noise_mask.any():
        valid_labels = labels[~noise_mask]
        if len(valid_labels) > 0:
            # Find nearest cluster for each noise point
            for i in np.where(noise_mask)[0]:
                noise_point = coords_km[i]
                distances = cdist([noise_point], coords_km[~noise_mask])[0]
                nearest_cluster_idx = np.argmin(distances)
                labels[i] = valid_labels[nearest_cluster_idx]
        else:
            # If all points are noise, assign to a single cluster
            labels[:] = 0
    return labels


def kmeans_clustering_metric(user_df, n_clusters, office_lat, office_lon):
    """Perform KMeans clustering using metric coordinates"""
    # Convert coordinates to km from office
    coords_km = []
    user_ids = []
    for _, user in user_df.iterrows():
        lat_km, lon_km = coords_to_km(user['latitude'], user['longitude'], office_lat, office_lon)
        coords_km.append([lat_km, lon_km])
        user_ids.append(user['user_id'])
    
    # Sort by user_id for deterministic ordering
    sorted_data = sorted(zip(user_ids, coords_km), key=lambda x: x[0])
    coords_km = np.array([item[1] for item in sorted_data])
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(coords_km)
    
    # Map labels back to original order
    label_map = {user_id: label for (user_id, _), label in zip(sorted_data, labels)}
    return [label_map[user_id] for user_id in user_df['user_id']]


def estimate_clusters(user_df, config, office_lat, office_lon):
    """Estimate optimal number of clusters using silhouette score with metric coordinates"""
    # Convert coordinates to km from office
    coords_km = []
    for _, user in user_df.iterrows():
        lat_km, lon_km = coords_to_km(user['latitude'], user['longitude'], office_lat, office_lon)
        coords_km.append([lat_km, lon_km])
    
    coords_km = np.array(coords_km)

    max_clusters = min(10, len(user_df) // 2)
    if max_clusters < 2:
        return 1

    scores = []
    for n_clusters in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(coords_km)
        if len(set(cluster_labels)) > 1:
            score = silhouette_score(coords_km, cluster_labels)
            scores.append((n_clusters, score))

    if not scores:
        return 1

    best_n_clusters = max(scores, key=lambda item: item[1])[0]
    return best_n_clusters


# STEP 1: DIRECTION-AWARE GEOGRAPHIC CLUSTERING
def create_geographic_clusters(user_df, office_lat, office_lon, config):
    """Create direction-aware geographic clusters using proper distance metrics"""
    if len(user_df) == 0:
        return user_df

    print("  🗺️  Creating direction-aware geographic clusters...")

    # Calculate features including bearings
    user_df = calculate_bearings_and_features(user_df, office_lat, office_lon)

    # Use sector-based clustering for direction awareness
    use_sweep = config.get('use_sweep_algorithm', True)

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
    """Partition into angular sectors then cluster within sectors using metric distances"""
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
            # Use spatial clustering within sector with proper metric
            eps_km = config.get('DBSCAN_EPS_KM', 1.5)
            sector_labels = dbscan_clustering_metric(sector_users, eps_km, 2, office_lat, office_lon)

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
    max_bearing_diff = config.get('MAX_BEARING_DIFFERENCE', 20)

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


# Continue with the rest of the existing assignment.py logic...
# [For brevity, I'll note that the rest of the functions from assignment.py should be included here]

def run_assignment(source_id: str, parameter: int = 1, string_param: str = ""):
    """Route efficiency assignment - prioritizes straight routes with minimal turning"""
    return run_route_efficiency_assignment(source_id, parameter, string_param)

def run_route_efficiency_assignment(source_id: str, parameter: int = 1, string_param: str = ""):
    """Main route efficiency assignment function"""
    start_time = time.time()

    # Reload configuration for route efficiency
    global _config
    _config = load_and_validate_config()
    
    print(f"🚀 Starting ROUTE EFFICIENCY assignment for source_id: {source_id}")
    print(f"📋 Parameter: {parameter}, String parameter: {string_param}")

    try:
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
                "optimization_mode": "route_efficiency",
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
                "optimization_mode": "route_efficiency",
                "parameter": parameter,
                "string_param": string_param
            }

        print(f"📥 Data loaded - Users: {len(users)}, Total Drivers: {len(all_drivers)}")

        # Extract office coordinates and validate data
        office_lat, office_lon = extract_office_coordinates(data)
        validate_input_data(data)
        print("✅ Data validation passed")

        # Prepare dataframes
        user_df, driver_df = prepare_user_driver_dataframes(data)
        
        print(f"📊 DataFrames prepared - Users: {len(user_df)}, Drivers: {len(driver_df)}")

        # STEP 1: Geographic clustering with proper distance metrics
        user_df = create_geographic_clusters(user_df, office_lat, office_lon, _config)
        clustering_results = {"method": "metric_aware_" + _config['clustering_method'], 
                            "clusters": user_df['geo_cluster'].nunique()}

        # STEP 2: Capacity-based sub-clustering (direction-aware)
        user_df = create_capacity_subclusters(user_df, office_lat, office_lon, _config)

        # STEP 3: Assign drivers to clusters
        routes, assigned_user_ids = assign_drivers_to_clusters(user_df, driver_df, office_lat, office_lon)

        # STEP 4: Handle remaining unassigned users
        unassigned_users = user_df[~user_df['user_id'].isin(assigned_user_ids)]
        unassigned_users_list = []
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
            unassigned_users_list.append(user_data)

        # Build unassigned drivers list
        assigned_driver_ids = {route['driver_id'] for route in routes}
        unassigned_drivers_list = []
        for _, driver in driver_df.iterrows():
            if driver['driver_id'] not in assigned_driver_ids:
                unassigned_drivers_list.append({
                    'driver_id': str(driver['driver_id']),
                    'capacity': int(driver['capacity']),
                    'vehicle_id': str(driver.get('vehicle_id', '')),
                    'latitude': float(driver['latitude']),
                    'longitude': float(driver['longitude'])
                })

        execution_time = time.time() - start_time
        print(f"✅ Route assignment complete in {execution_time:.2f}s")
        print(f"📊 Final routes: {len(routes)}")
        print(f"🎯 Users assigned: {len(assigned_user_ids)}")
        print(f"👥 Users unassigned: {len(unassigned_users_list)}")

        return {
            "status": "true",
            "execution_time": execution_time,
            "data": routes,
            "unassignedUsers": unassigned_users_list,
            "unassignedDrivers": unassigned_drivers_list,
            "clustering_analysis": clustering_results,
            "optimization_mode": "route_efficiency",
            "parameter": parameter,
            "string_param": string_param
        }

    except Exception as e:
        logger.error(f"Route efficiency assignment failed: {e}", exc_info=True)
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


def assign_drivers_to_clusters(user_df, driver_df, office_lat, office_lon):
    """Assign drivers to user clusters using route efficiency logic"""
    routes = []
    assigned_user_ids = set()
    used_driver_ids = set()

    # Sort drivers by priority
    available_drivers = driver_df.sort_values(['priority'], ascending=[True])

    # Process each sub-cluster
    for sub_cluster_id in user_df['sub_cluster'].unique():
        if sub_cluster_id == -1:
            continue
            
        cluster_users = user_df[user_df['sub_cluster'] == sub_cluster_id]
        unassigned_in_cluster = cluster_users[~cluster_users['user_id'].isin(assigned_user_ids)]
        
        if unassigned_in_cluster.empty:
            continue

        # Find best driver for this cluster
        best_driver = None
        min_cost = float('inf')

        cluster_center = unassigned_in_cluster[['latitude', 'longitude']].mean()
        cluster_size = len(unassigned_in_cluster)

        for _, driver in available_drivers.iterrows():
            if driver['driver_id'] in used_driver_ids:
                continue
            if driver['capacity'] < cluster_size:
                continue

            # Calculate assignment cost
            distance = haversine_distance(
                driver['latitude'], driver['longitude'],
                cluster_center['latitude'], cluster_center['longitude']
            )
            
            # Route efficiency prioritizes direction consistency and minimal distance
            utilization = cluster_size / driver['capacity']
            cost = distance - (utilization * 2.0)  # Bonus for high utilization
            
            if cost < min_cost:
                min_cost = cost
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
            users_to_assign = unassigned_in_cluster.head(best_driver['capacity'])
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
                assigned_user_ids.add(user['user_id'])

            routes.append(route)

    print(f"  ✅ Created {len(routes)} routes with route efficiency assignment")
    return routes, assigned_user_ids


def analyze_assignment_quality(result):
    """Analyze the quality of the assignment with enhanced metrics"""
    if result["status"] != "true":
        return "Assignment failed"

    total_routes = len(result["data"])
    total_assigned = sum(len(route["assigned_users"]) for route in result["data"])
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
        "total_routes": total_routes,
        "total_assigned_users": total_assigned,
        "total_unassigned_users": total_unassigned,
        "assignment_rate": round(total_assigned / (total_assigned + total_unassigned) * 100, 1) if (total_assigned + total_unassigned) > 0 else 0,
        "avg_utilization": round(np.mean(utilizations) * 100, 1) if utilizations else 0,
        "min_utilization": round(np.min(utilizations) * 100, 1) if utilizations else 0,
        "max_utilization": round(np.max(utilizations) * 100, 1) if utilizations else 0,
        "routes_below_80_percent": sum(1 for u in utilizations if u < 0.8),
        "avg_turning_score": round(np.mean(turning_scores), 1) if turning_scores else 0,
        "avg_tortuosity": round(np.mean(tortuosity_ratios), 2) if tortuosity_ratios else 1.0,
        "avg_direction_consistency": round(np.mean(direction_consistencies) * 100, 1) if direction_consistencies else 100.0,
        "distance_issues": distance_issues,
        "clustering_method": result.get("clustering_analysis", {}).get("method", "Unknown"),
        "routes_with_good_turning": sum(1 for t in turning_scores if t <= 35),
        "routes_with_poor_turning": sum(1 for t in turning_scores if t > 50)
    }

    return analysis
