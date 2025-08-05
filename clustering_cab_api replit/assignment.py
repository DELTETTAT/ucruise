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

# Load thresholds from config with error handling
try:
    with open('config.json') as f:
        cfg = json.load(f)
except (FileNotFoundError, json.JSONDecodeError) as e:
    print(
        f"Warning: Could not load config.json, using default values. Error: {e}"
    )
    cfg = {}

# Configuration constants - Simplified
MAX_FILL_DISTANCE_KM = cfg.get("max_fill_distance_km", 5.0)
MERGE_DISTANCE_KM = cfg.get("merge_distance_km", 3.0)
MIN_UTIL_THRESHOLD = cfg.get("min_util_threshold", 0.5)
DBSCAN_EPS_KM = cfg.get("dbscan_eps_km", 3.0)
MIN_SAMPLES_DBSCAN = cfg.get("min_samples_dbscan", 2)
MAX_BEARING_DIFFERENCE = cfg.get("max_bearing_difference", 45)
SWAP_IMPROVEMENT_THRESHOLD = cfg.get("swap_improvement_threshold_km", 0.5)
MAX_SWAP_ITERATIONS = cfg.get("max_swap_iterations", 3)  # Reduced from 10
UTILIZATION_PENALTY_PER_SEAT = cfg.get("utilization_penalty_per_seat", 3.0)
OVERFLOW_PENALTY_KM = cfg.get("overflow_penalty_km", 10.0)
DISTANCE_ISSUE_THRESHOLD = cfg.get("distance_issue_threshold_km", 8.0)
LOW_UTILIZATION_THRESHOLD = cfg.get("low_utilization_threshold", 0.5)
MAX_USERS_FOR_FALLBACK = cfg.get("max_users_for_fallback", 3)
FALLBACK_MIN_USERS = cfg.get("fallback_min_users", 2)
FALLBACK_MAX_USERS = cfg.get("fallback_max_users", 7)
OFFICE_LAT = cfg.get("office_latitude", 30.6810489)
OFFICE_LON = cfg.get("office_longitude", 76.7260711)

# Simplified advanced parameters
PARALLEL_PROCESSING = cfg.get("parallel_processing", True)
ENHANCED_CLUSTERING = cfg.get("enhanced_clustering", True)


def validate_input_data(data):
    """Validate that the API response has the required structure"""
    if not isinstance(data, dict):
        raise ValueError("API response must be a dictionary")

    # Check for users
    users = data.get("users", [])
    if not users:
        raise ValueError("No users found in API response")

    if not isinstance(users, list):
        raise ValueError("Users must be a list")

    # Special handling for single user
    if len(users) == 1:
        print("ℹ️  Single user detected - will create single route assignment")

    # Validate each user has required fields
    for i, user in enumerate(users):
        if not isinstance(user, dict):
            raise ValueError(f"User {i} must be a dictionary")

        required_fields = ["id", "latitude", "longitude"]
        for field in required_fields:
            if field not in user:
                raise ValueError(f"User {i} missing required field: {field}")
            if not user[field]:
                raise ValueError(f"User {i} has empty {field}")

    # Check for drivers (either format)
    drivers_found = False

    # Check nested format first
    if "drivers" in data:
        drivers_data = data["drivers"]
        if isinstance(drivers_data, dict):
            drivers_unassigned = drivers_data.get("driversUnassigned", [])
            drivers_assigned = drivers_data.get("driversAssigned", [])
            if drivers_unassigned or drivers_assigned:
                drivers_found = True

    # Check flat format
    if not drivers_found:
        drivers_unassigned = data.get("driversUnassigned", [])
        drivers_assigned = data.get("driversAssigned", [])
        if drivers_unassigned or drivers_assigned:
            drivers_found = True

    if not drivers_found:
        raise ValueError("No drivers found in API response")

    print("✅ Input data validation passed")


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
    # Corrected: Removed duplicate path segment
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


def prepare_user_driver_dataframes(data):
    """Prepare and clean user and driver dataframes with priority system"""
    users_list = data.get("users", [])
    drivers_unassigned_list = data.get("driversUnassigned", [])
    drivers_assigned_list = data.get("driversAssigned", [])

    df_users = pd.DataFrame(users_list)

    # Combine driversUnassigned and driversAssigned with priority flags
    all_drivers = []

    # Priority 1: driversUnassigned with shift_type_id 1 and 3 (highest priority)
    priority_1_count = 0
    priority_2_count = 0
    for driver in drivers_unassigned_list:
        driver_copy = driver.copy()
        shift_type_id = int(driver.get('shift_type_id', 2))
        if shift_type_id in [1, 3]:
            driver_copy['priority'] = 1
            priority_1_count += 1
        else:
            driver_copy['priority'] = 2
            priority_2_count += 1
        driver_copy['is_assigned'] = False
        all_drivers.append(driver_copy)

    # Priority 3: driversAssigned with shift_type_id 1 and 3 (lower priority)
    # Priority 4: driversAssigned with shift_type_id 2 (lowest priority)
    priority_3_count = 0
    priority_4_count = 0
    for driver in drivers_assigned_list:
        driver_copy = driver.copy()
        shift_type_id = int(driver.get('shift_type_id', 2))
        if shift_type_id in [1, 3]:
            driver_copy['priority'] = 3
            priority_3_count += 1
        else:
            driver_copy['priority'] = 4
            priority_4_count += 1
        driver_copy['is_assigned'] = True
        all_drivers.append(driver_copy)

    print(f"🔍 Driver Priority Assignment Debug:")
    print(f"   Priority 1 (driversUnassigned ST:1,3): {priority_1_count}")
    print(f"   Priority 2 (driversUnassigned ST:2): {priority_2_count}")
    print(f"   Priority 3 (driversAssigned ST:1,3): {priority_3_count}")
    print(f"   Priority 4 (driversAssigned ST:2): {priority_4_count}")

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
    """Calculate haversine distance between two points in kilometers (cached)"""
    R = 6371.0  # Earth radius in kilometers
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c


def calculate_bearing(lat1, lon1, lat2, lon2):
    """Calculate bearing from point A to B in degrees"""
    lat1, lat2 = map(math.radians, [lat1, lat2])
    dlon = math.radians(lon2 - lon1)
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(
        lat2) * math.cos(dlon)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def calculate_bearings_and_features(user_df, office_lat, office_lon):
    """Calculate bearings and add geographical features"""
    user_df = user_df.copy()
    user_df[['office_latitude', 'office_longitude']] = office_lat, office_lon

    def calculate_bearing_vectorized(lat1, lon1, lat2, lon2):
        lat1, lat2 = np.radians(lat1), np.radians(lat2)
        dlon = np.radians(lon2 - lon1)
        x = np.sin(dlon) * np.cos(lat2)
        y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
        return (np.degrees(np.arctan2(x, y)) + 360) % 360

    user_df['bearing'] = calculate_bearing_vectorized(user_df['latitude'],
                                                      user_df['longitude'],
                                                      office_lat, office_lon)
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


# ENHANCED BUT SIMPLIFIED CLUSTERING
def enhanced_geographical_clustering(user_df, office_lat, office_lon):
    """
    Enhanced clustering that handles single user edge case
    """
    coords = user_df[['latitude', 'longitude']].values

    # Handle single user case
    if len(user_df) == 1:
        user_df = user_df.copy()
        user_df['geo_cluster'] = 0
        return user_df, {
            'method': 'Single User Assignment',
            'clusters': 1
        }

    # Enhanced DBSCAN clustering with bearing consideration
    coords_rad = np.radians(coords)
    eps_rad = DBSCAN_EPS_KM / 6371.0

    # Primary clustering using geographical coordinates
    clustering = DBSCAN(eps=eps_rad,
                        min_samples=MIN_SAMPLES_DBSCAN,
                        metric='haversine').fit(coords_rad)
    user_df = user_df.copy()
    user_df['geo_cluster'] = clustering.labels_

    # Post-process clusters to avoid users in opposite directions
    refined_clusters = []
    cluster_id_counter = 0

    for cluster_id in user_df['geo_cluster'].unique():
        if cluster_id == -1:  # Skip noise points for now
            continue

        cluster_users = user_df[user_df['geo_cluster'] == cluster_id]

        if len(cluster_users) <= 1:
            # Single user clusters are fine
            refined_clusters.append((cluster_users.index, cluster_id_counter))
            cluster_id_counter += 1
            continue

        # Check bearing spread within cluster
        bearings = cluster_users['bearing'].values
        max_bearing_diff = 0
        for i in range(len(bearings)):
            for j in range(i + 1, len(bearings)):
                diff = bearing_difference(bearings[i], bearings[j])
                max_bearing_diff = max(max_bearing_diff, diff)

        # If users are too spread in bearing (>90 degrees), split the cluster
        if max_bearing_diff > 90 and len(cluster_users) > 2:
            # Split based on bearing similarity
            bearings_for_clustering = np.column_stack([
                np.sin(np.radians(bearings)),
                np.cos(np.radians(bearings))
            ])

            sub_clusters = min(2, len(cluster_users))  # Split into max 2 sub-clusters
            kmeans = KMeans(n_clusters=sub_clusters, random_state=42)
            sub_labels = kmeans.fit_predict(bearings_for_clustering)

            for sub_label in range(sub_clusters):
                sub_cluster_indices = cluster_users.index[sub_labels == sub_label]
                if len(sub_cluster_indices) > 0:
                    refined_clusters.append((sub_cluster_indices, cluster_id_counter))
                    cluster_id_counter += 1
        else:
            # Cluster is geographically and directionally compact
            refined_clusters.append((cluster_users.index, cluster_id_counter))
            cluster_id_counter += 1

    # Apply refined clustering
    for indices, new_cluster_id in refined_clusters:
        user_df.loc[indices, 'geo_cluster'] = new_cluster_id

    # Handle noise points by assigning to nearest cluster
    noise_mask = user_df['geo_cluster'] == -1
    if noise_mask.any():
        noise_points = user_df[noise_mask][['latitude', 'longitude']].values
        valid_clusters = user_df[user_df['geo_cluster'] != -1]

        if not valid_clusters.empty:
            cluster_centers = valid_clusters.groupby('geo_cluster')[[
                'latitude', 'longitude'
            ]].mean().values
            distances = cdist(noise_points,
                              cluster_centers,
                              metric='euclidean')
            nearest_clusters = valid_clusters['geo_cluster'].unique()[
                np.argmin(distances, axis=1)]
            user_df.loc[noise_mask, 'geo_cluster'] = nearest_clusters
        else:
            user_df.loc[noise_mask, 'geo_cluster'] = 0

    # If DBSCAN fails completely or produces too few clusters, use bearing-enhanced K-means as fallback
    n_clusters_after_dbscan = user_df['geo_cluster'].nunique()
    if n_clusters_after_dbscan <= 1:
        print("📊 DBSCAN failed or produced too few clusters, using bearing-enhanced K-means fallback...")
        coords_features = user_df[[
            'latitude', 'longitude', 'bearing_sin', 'bearing_cos'
        ]].values
        # Weight coordinates more heavily than bearing
        weights = np.array([1.0, 1.0, 0.3, 0.3])
        weighted_features = coords_features * weights
        scaled_features = StandardScaler().fit_transform(weighted_features)

        # Fix: Handle edge case where we have very few users
        # Determine optimal number of clusters based on data size, capped by available drivers
        optimal_clusters = min(6, max(1, len(user_df) // 3))

        # Use KMeans as fallback with optimal cluster count
        # Handle single user case in fallback
        if len(user_df) == 1:
            user_df['geo_cluster'] = 0
            n_clusters = 1
            method_used = "Single User Assignment"
        else:
            n_clusters = min(optimal_clusters, len(user_df))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            user_df['geo_cluster'] = kmeans.fit_predict(scaled_features)
            method_used = "Enhanced K-means"
    else:
        method_used = 'DBSCAN'

    print(
        f"🗂️ Clustering complete: {user_df['geo_cluster'].nunique()} clusters using {method_used}"
    )

    return user_df, {
        'method': method_used,
        'clusters': user_df['geo_cluster'].nunique()
    }


# SIMPLIFIED COST FUNCTION
def calculate_enhanced_cost(route, driver_pos, user_positions, office_lat,
                            office_lon):
    """Enhanced cost function with bearing and directional efficiency"""
    if not user_positions:
        return float('inf')

    # Distance component (primary factor)
    centroid = np.mean(user_positions, axis=0)
    distance_cost = haversine_distance(driver_pos[0], driver_pos[1],
                                       centroid[0], centroid[1])

    # Utilization component (secondary factor)
    utilization = len(user_positions) / route.get('capacity', 1)
    utilization_penalty = (1 - utilization) * UTILIZATION_PENALTY_PER_SEAT

    # Bearing and directional efficiency penalty (CRITICAL factor)
    bearing_penalty = 0
    if len(user_positions) > 1:
        # Calculate bearings from office to each user
        user_bearings = []
        for pos in user_positions:
            bearing = calculate_bearing(office_lat, office_lon, pos[0], pos[1])
            user_bearings.append(bearing)

        # Calculate bearing spread - heavy penalty for users in opposite directions
        bearing_differences = []
        for i in range(len(user_bearings)):
            for j in range(i + 1, len(user_bearings)):
                diff = bearing_difference(user_bearings[i], user_bearings[j])
                bearing_differences.append(diff)

        if bearing_differences:
            max_bearing_diff = max(bearing_differences)
            # EXTREME penalty for users in opposite directions (>90 degrees apart)
            if max_bearing_diff > 90:
                bearing_penalty = 50.0 + (max_bearing_diff - 90) * 2.0  # Very high base penalty + 2km per degree
            # Heavy penalty for widely spread users (>60 degrees apart)
            elif max_bearing_diff > 60:
                bearing_penalty = (max_bearing_diff - 60) * 0.5  # 0.5 km penalty per degree over 60
            # Moderate penalty for moderately spread users (>30 degrees apart)
            elif max_bearing_diff > 30:
                bearing_penalty = (max_bearing_diff - 30) * 0.2  # 0.2 km penalty per degree over 30

        # Compactness penalty - users should be geographically close
        distances = [
            haversine_distance(pos[0], pos[1], centroid[0], centroid[1])
            for pos in user_positions
        ]
        compactness_penalty = np.std(distances) * 0.3

        # Route efficiency penalty - total travel distance within route
        total_route_distance = 0
        for i in range(len(user_positions)):
            for j in range(i + 1, len(user_positions)):
                total_route_distance += haversine_distance(
                    user_positions[i][0], user_positions[i][1],
                    user_positions[j][0], user_positions[j][1]
                )

        # Average inter-user distance penalty
        if len(user_positions) > 1:
            avg_inter_distance = total_route_distance / (len(user_positions) * (len(user_positions) - 1) / 2)
            distance_spread_penalty = avg_inter_distance * 0.2  # Penalty for spread out users
        else:
            distance_spread_penalty = 0
    else:
        compactness_penalty = 0
        distance_spread_penalty = 0

    total_cost = (distance_cost + utilization_penalty + bearing_penalty + 
                  compactness_penalty + distance_spread_penalty)
    return total_cost


def optimize_driver_assignment(user_df, driver_df):
    """
    Optimized driver assignment with priority system
    Priority: 1 (driversUnassigned shift_type 1,3) > 2 (driversUnassigned shift_type 2) > 3 (driversAssigned shift_type 1,3) > 4 (driversAssigned shift_type 2)
    """
    drivers_and_routes = []
    # Sort by priority first, then by capacity (descending)
    available_drivers = driver_df.sort_values(['priority', 'capacity'],
                                              ascending=[True, False]).copy()
    assigned_user_ids = set()

    print(f"🚗 Driver priority distribution:")
    for priority in [1, 2, 3, 4]:
        count = len(available_drivers[available_drivers['priority'] == priority])
        priority_desc = {
            1: "driversUnassigned (shift_type 1,3)",
            2: "driversUnassigned (shift_type 2)",
            3: "driversAssigned (shift_type 1,3)",
            4: "driversAssigned (shift_type 2)"
        }
        print(f"   Priority {priority}: {count} drivers - {priority_desc[priority]}")

    # Debug: Print first few drivers to verify sorting
    print(f"🔍 First 5 drivers after priority sorting:")
    for i, (_, driver) in enumerate(available_drivers.head(5).iterrows()):
        is_assigned_text = "driversAssigned" if driver['is_assigned'] else "driversUnassigned"
        print(f"   {i+1}. Driver {driver['driver_id']}: Priority {driver['priority']}, {is_assigned_text}, Capacity {driver['capacity']}")

    used_driver_indices = set()

    for cluster_id, cluster_users in user_df.groupby('geo_cluster'):
        unassigned_in_cluster = cluster_users[~cluster_users['user_id'].
                                              isin(assigned_user_ids)].copy()

        if unassigned_in_cluster.empty or len(used_driver_indices) >= len(available_drivers):
            continue

        total_users = len(unassigned_in_cluster)
        available_driver_indices = [i for i in available_drivers.index if i not in used_driver_indices]

        if not available_driver_indices:
            continue

        remaining_drivers = available_drivers.loc[available_driver_indices]
        avg_capacity = remaining_drivers['capacity'].mean()
        max_capacity = remaining_drivers['capacity'].max()

        if total_users <= max_capacity:
            sub_clusters = 1
        else:
            sub_clusters = min(max(1, round(total_users / avg_capacity)),
                               len(available_driver_indices), total_users)

        if sub_clusters > 1 and len(unassigned_in_cluster) > 1:
            coords = unassigned_in_cluster[['latitude', 'longitude']].values
            # Ensure we don't try to create more clusters than data points
            actual_clusters = min(sub_clusters, len(unassigned_in_cluster))
            if actual_clusters > 1:
                kmeans = KMeans(n_clusters=actual_clusters, random_state=42)
                unassigned_in_cluster['sub_cluster'] = kmeans.fit_predict(coords)
            else:
                unassigned_in_cluster['sub_cluster'] = 0
        else:
            unassigned_in_cluster['sub_cluster'] = 0

        for sub_id, sub_users in unassigned_in_cluster.groupby('sub_cluster'):
            if sub_users.empty or len(used_driver_indices) >= len(available_drivers):
                continue

            sub_centroid = sub_users[['latitude', 'longitude']].mean().values
            users_needed = len(sub_users)

            best_driver_idx = None
            min_cost = float('inf')
            best_priority = float('inf')

            # Get remaining available drivers
            available_driver_indices = [i for i in available_drivers.index if i not in used_driver_indices]

            # Find the best driver considering priority, bearing, and distance
            for driver_idx in available_driver_indices:
                driver = available_drivers.loc[driver_idx]
                driver_pos = (driver['latitude'], driver['longitude'])

                # Calculate basic distance cost
                distance = haversine_distance(driver_pos[0], driver_pos[1],
                                              sub_centroid[0], sub_centroid[1])

                # Calculate bearing alignment between driver and users relative to office
                user_positions = [(row['latitude'], row['longitude']) for _, row in sub_users.iterrows()]

                # Enhanced cost calculation with bearing considerations
                mock_route = {'capacity': driver['capacity']}
                cost = calculate_enhanced_cost(mock_route, driver_pos, user_positions, 
                                               user_df.iloc[0]['office_latitude'],
                                               user_df.iloc[0]['office_longitude'])

                # Apply overflow penalty if capacity exceeded
                if users_needed > driver['capacity']:
                    cost += OVERFLOW_PENALTY_KM

                # Additional bearing alignment bonus for driver position
                if len(user_positions) > 0:
                    # Calculate average user bearing from office
                    avg_user_bearing = np.mean([
                        calculate_bearing(user_df.iloc[0]['office_latitude'], 
                                        user_df.iloc[0]['office_longitude'],
                                        pos[0], pos[1]) for pos in user_positions
                    ])

                    # Calculate driver bearing from office
                    driver_bearing = calculate_bearing(user_df.iloc[0]['office_latitude'], 
                                                     user_df.iloc[0]['office_longitude'],
                                                     driver_pos[0], driver_pos[1])

                    # Bonus for driver aligned in same direction as users
                    bearing_alignment = bearing_difference(avg_user_bearing, driver_bearing)
                    if bearing_alignment <= 30:  # Driver in same direction as users
                        cost *= 0.9  # 10% bonus
                    elif bearing_alignment <= 60:  # Driver moderately aligned
                        cost *= 0.95  # 5% bonus
                    elif bearing_alignment > 120:  # Driver in opposite direction
                        cost *= 1.2  # 20% penalty

                # STRICT PRIORITY: Always choose lower priority number first, but consider bearing within same priority
                if (driver['priority'] < best_priority or
                    (driver['priority'] == best_priority and (best_driver_idx is None or cost < min_cost))):
                    min_cost = cost
                    best_driver_idx = driver_idx
                    best_priority = driver['priority']

            # Debug: Print selected driver info
            if best_driver_idx is not None:
                best_driver = available_drivers.loc[best_driver_idx]
                is_assigned_text = "driversAssigned" if best_driver['is_assigned'] else "driversUnassigned"
                print(f"   🎯 Selected Driver {best_driver['driver_id']} (Priority {best_driver['priority']}, {is_assigned_text}) for {users_needed} users")

                capacity = best_driver['capacity']
                users_to_assign = sub_users.head(capacity)

                route = create_route_from_users(
                    best_driver.to_dict(), users_to_assign.to_dict('records'),
                    user_df.iloc[0]['office_latitude'],
                    user_df.iloc[0]['office_longitude'])

                for _, user in users_to_assign.iterrows():
                    assigned_user_ids.add(user['user_id'])

                drivers_and_routes.append(route)
                used_driver_indices.add(best_driver_idx)

    return drivers_and_routes, assigned_user_ids


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

    # Calculate route metrics
    if route['assigned_users']:
        lats = [u['lat'] for u in route['assigned_users']]
        lngs = [u['lng'] for u in route['assigned_users']]
        route['centroid'] = [np.mean(lats), np.mean(lngs)]
        route['utilization'] = len(
            route['assigned_users']) / route['vehicle_type']
        route['stops'] = [[u['lat'], u['lng']]
                          for u in route['assigned_users']]
        route['bearing'] = calculate_bearing(route['centroid'][0],
                                             route['centroid'][1], office_lat,
                                             office_lon)
    else:
        route['centroid'] = [route['latitude'], route['longitude']]
        route['utilization'] = 0
        route['stops'] = []
        route['bearing'] = 0

    return route


# SIMPLIFIED LOCAL SEARCH
def simplified_local_search(routes, max_iterations=MAX_SWAP_ITERATIONS):
    """Simplified but effective local search optimization"""
    print("🔧 Starting simplified local search optimization...")

    improved = True
    iterations = 0

    while improved and iterations < max_iterations:
        improved = False
        iterations += 1

        for i, route_a in enumerate(routes):
            for j, route_b in enumerate(routes[i + 1:], start=i + 1):
                if not route_a['assigned_users'] or not route_b[
                        'assigned_users']:
                    continue

                # Try swapping users between routes
                for user_a in route_a['assigned_users'][:]:
                    for user_b in route_b['assigned_users'][:]:
                        if calculate_swap_improvement(
                                route_a, route_b, user_a,
                                user_b) > SWAP_IMPROVEMENT_THRESHOLD:
                            # Perform swap
                            route_a['assigned_users'].remove(user_a)
                            route_b['assigned_users'].remove(user_b)
                            route_a['assigned_users'].append(user_b)
                            route_b['assigned_users'].append(user_a)
                            improved = True
                            update_route_metrics([route_a, route_b])
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break

        if improved:
            print(f"  🔄 Improvement found in iteration {iterations}")

    print(f"✅ Local search complete after {iterations} iterations")
    return routes


def calculate_swap_improvement(route_a, route_b, user_a, user_b):
    """Calculate improvement from swapping two users"""
    centroid_a = route_a.get('centroid',
                             [route_a['latitude'], route_a['longitude']])
    centroid_b = route_b.get('centroid',
                             [route_b['latitude'], route_b['longitude']])

    current_dist_a = haversine_distance(centroid_a[0], centroid_a[1],
                                        user_a['lat'], user_a['lng'])
    current_dist_b = haversine_distance(centroid_b[0], centroid_b[1],
                                        user_b['lat'], user_b['lng'])

    swap_dist_a = haversine_distance(centroid_a[0], centroid_a[1],
                                     user_b['lat'], user_b['lng'])
    swap_dist_b = haversine_distance(centroid_b[0], centroid_b[1],
                                     user_a['lat'], user_a['lng'])

    return (current_dist_a + current_dist_b) - (swap_dist_a + swap_dist_b)


def update_route_metrics(routes):
    """Update route metrics after modifications"""
    for route in routes:
        if route['assigned_users']:
            lats = [u['lat'] for u in route['assigned_users']]
            lngs = [u['lng'] for u in route['assigned_users']]
            route['centroid'] = [np.mean(lats), np.mean(lngs)]
            route['utilization'] = len(
                route['assigned_users']) / route['vehicle_type']
            route['stops'] = [[u['lat'], u['lng']]
                              for u in route['assigned_users']]
        else:
            route['centroid'] = [route['latitude'], route['longitude']]
            route['utilization'] = 0
            route['stops'] = []
            route['bearing'] = 0


# Keep the original utility functions
def fill_underutilized_routes(drivers_and_routes, user_df, assigned_user_ids):
    """Fill underutilized routes with nearby unassigned users"""
    unassigned_users = user_df[~user_df['user_id'].isin(assigned_user_ids
                                                        )].copy()

    for route in drivers_and_routes:
        capacity = route['vehicle_type']
        current_size = len(route['assigned_users'])

        if current_size < capacity and not unassigned_users.empty:
            if route['assigned_users']:
                centroid_lat = np.mean(
                    [u['lat'] for u in route['assigned_users']])
                centroid_lng = np.mean(
                    [u['lng'] for u in route['assigned_users']])
            else:
                centroid_lat, centroid_lng = route['latitude'], route[
                    'longitude']

            distances = []
            for _, user in unassigned_users.iterrows():
                dist = haversine_distance(centroid_lat, centroid_lng,
                                          user['latitude'], user['longitude'])
                if dist <= MAX_FILL_DISTANCE_KM:
                    distances.append((user, dist))

            distances.sort(key=lambda x: x[1])
            slots_available = capacity - current_size

            for user, dist in distances[:slots_available]:
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
                # Remove user from unassigned_users to avoid re-assignment
                unassigned_users = unassigned_users[unassigned_users['user_id']
                                                    != user['user_id']]

    return assigned_user_ids


def merge_underutilized_routes(drivers_and_routes):
    """Merge underutilized routes with nearby routes - Fixed duplicacy issue"""
    if not drivers_and_routes:
        return []

    # Return original routes if only one route exists
    if len(drivers_and_routes) <= 1:
        return drivers_and_routes

    merged_routes = []
    used_indices = set()

    for i, route_a in enumerate(drivers_and_routes):
        if i in used_indices:
            continue

        # Check if route_a is underutilized
        is_route_a_underutilized = route_a.get('utilization', 0) < MIN_UTIL_THRESHOLD

        if not is_route_a_underutilized:
            merged_routes.append(route_a)
            used_indices.add(i)
            continue

        merged = False
        centroid_a = route_a.get('centroid', [route_a['latitude'], route_a['longitude']])

        for j, route_b in enumerate(drivers_and_routes):
            if j <= i or j in used_indices:  # Changed from j == i to j <= i to avoid duplicate processing
                continue

            centroid_b = route_b.get('centroid', [route_b['latitude'], route_b['longitude']])
            dist = haversine_distance(centroid_a[0], centroid_a[1], centroid_b[0], centroid_b[1])

            total_users = len(route_a['assigned_users']) + len(route_b['assigned_users'])
            max_capacity = route_b['vehicle_type']

            # Check bearing compatibility before merging
            bearing_compatible = True
            if route_a['assigned_users'] and route_b['assigned_users']:
                # Calculate bearing spread after potential merge
                all_users = route_a['assigned_users'] + route_b['assigned_users']
                if len(all_users) > 1:
                    # Use office coordinates from config or default
                    office_lat_merge = OFFICE_LAT
                    office_lon_merge = OFFICE_LON

                    bearings = [calculate_bearing(office_lat_merge, office_lon_merge, 
                                                u['lat'], u['lng']) for u in all_users]

                    max_bearing_diff = 0
                    for i in range(len(bearings)):
                        for j in range(i + 1, len(bearings)):
                            diff = bearing_difference(bearings[i], bearings[j])
                            max_bearing_diff = max(max_bearing_diff, diff)

                    # Don't merge if it creates routes with users in opposite directions
                    if max_bearing_diff > MAX_BEARING_DIFFERENCE * 2:  # More strict for merging
                        bearing_compatible = False

            if dist <= MERGE_DISTANCE_KM and total_users <= max_capacity and bearing_compatible:
                # Merge route_a into route_b
                route_b['assigned_users'].extend(route_a['assigned_users'])

                # Update route_b metrics
                if route_b['assigned_users']:
                    lats = [u['lat'] for u in route_b['assigned_users']]
                    lngs = [u['lng'] for u in route_b['assigned_users']]
                    route_b['centroid'] = [np.mean(lats), np.mean(lngs)]
                    route_b['utilization'] = len(route_b['assigned_users']) / route_b['vehicle_type']
                    route_b['stops'] = [[u['lat'], u['lng']] for u in route_b['assigned_users']]

                # Mark route_a as used (it's been merged into route_b)
                used_indices.add(i)
                merged = True
                break

        # If route_a wasn't merged, keep it
        if not merged:
            merged_routes.append(route_a)
            used_indices.add(i)

    # Add any remaining routes that haven't been processed yet
    for i, route in enumerate(drivers_and_routes):
        if i not in used_indices:
            merged_routes.append(route)

    return merged_routes


def handle_remaining_users(user_df, drivers_and_routes, assigned_user_ids,
                           driver_df):
    """Handle remaining unassigned users"""
    unassigned_users = user_df[~user_df['user_id'].isin(assigned_user_ids
                                                        )].copy()
    available_drivers = driver_df[~driver_df['driver_id'].isin(
        [route['driver_id'] for route in drivers_and_routes])].copy()

    if unassigned_users.empty:
        return []

    # Try to fit into existing routes first
    for _, user in unassigned_users.iterrows():
        best_route = None
        min_distance = float('inf')

        for route in drivers_and_routes:
            if len(route['assigned_users']) < route['vehicle_type']:
                if route['assigned_users']:
                    centroid_lat = np.mean(
                        [u['lat'] for u in route['assigned_users']])
                    centroid_lng = np.mean(
                        [u['lng'] for u in route['assigned_users']])
                else:
                    centroid_lat, centroid_lng = route['latitude'], route[
                        'longitude']

                dist = haversine_distance(centroid_lat, centroid_lng,
                                          user['latitude'], user['longitude'])

                if dist < min_distance and dist <= MAX_FILL_DISTANCE_KM:
                    min_distance = dist
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
            # Remove user from unassigned_users to avoid re-assignment in this loop
            unassigned_users = unassigned_users[unassigned_users['user_id']
                                                != user['user_id']]


    # Create new routes for remaining users
    remaining_unassigned = user_df[~user_df['user_id'].isin(assigned_user_ids
                                                            )].copy()

    if not remaining_unassigned.empty and not available_drivers.empty:
        if len(remaining_unassigned) > 1:
            coords = remaining_unassigned[['latitude', 'longitude']].values
            # Determine number of clusters, limited by available drivers and number of users
            n_clusters = min(len(available_drivers), len(remaining_unassigned))
            if n_clusters > 1 and len(remaining_unassigned) >= n_clusters:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                remaining_unassigned['cluster'] = kmeans.fit_predict(coords)
            else:
                remaining_unassigned['cluster'] = 0 # Single cluster if conditions not met
        else:
            remaining_unassigned['cluster'] = 0 # Single cluster for single user

        available_drivers_list = available_drivers.to_dict('records')

        for cluster_id, cluster_users in remaining_unassigned.groupby(
                'cluster'):
            if not available_drivers_list:
                break

            centroid = cluster_users[['latitude', 'longitude']].mean().values

            # Find the closest available driver for this cluster
            closest_driver = min(
                available_drivers_list,
                key=lambda d: haversine_distance(centroid[0], centroid[1], d[
                    'latitude'], d['longitude']))

            capacity = closest_driver['capacity']
            users_to_assign = cluster_users.head(capacity)

            route = create_route_from_users(
                closest_driver, users_to_assign.to_dict('records'),
                user_df.iloc[0]['office_latitude'],
                user_df.iloc[0]['office_longitude'])

            for _, user in users_to_assign.iterrows():
                assigned_user_ids.add(user['user_id'])

            drivers_and_routes.append(route)
            available_drivers_list.remove(closest_driver) # Remove driver from list

    # Return final unassigned users
    final_unassigned = user_df[~user_df['user_id'].isin(assigned_user_ids
                                                        )].copy()
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


def calculate_route_bearing(route, office_lat, office_lon):
    """Calculate representative bearing for a route"""
    if not route['assigned_users']:
        return 0
    avg_lat = sum(u['lat'] for u in route['assigned_users']) / len(
        route['assigned_users'])
    avg_lng = sum(u['lng'] for u in route['assigned_users']) / len(
        route['assigned_users'])
    return calculate_bearing(avg_lat, avg_lng, office_lat, office_lon)


def reassign_underutilized_users_to_nearby_routes(drivers_and_routes,
                                                  unassigned_drivers,
                                                  office_lat, office_lon):
    """Reassign users from underutilized routes to nearby better routes"""
    reassigned_users = []
    routes_to_remove = set()

    # Merge underfilled nearby routes
    merged_routes = []
    routes_used = set()

    for i, route_a in enumerate(drivers_and_routes):
        if i in routes_used:
            continue

        # Check if route_a is underutilized
        is_route_a_underutilized = route_a.get('utilization', 0) < MIN_UTIL_THRESHOLD

        if not is_route_a_underutilized:
            merged_routes.append(route_a)
            routes_used.add(i)
            continue

        merged = False
        for j, route_b in enumerate(drivers_and_routes[i + 1:], start=i + 1):
            if j in routes_used or route_b.get('utilization', 0) >= 1.0: # Don't merge with full routes
                continue

            combined_users = route_a['assigned_users'] + route_b[
                'assigned_users']
            # Check if combined users fit into the max capacity of either vehicle
            if len(combined_users) <= max(route_a['vehicle_type'],
                                          route_b['vehicle_type']):
                centroid_a = route_a.get(
                    'centroid', [route_a['latitude'], route_a['longitude']])
                centroid_b = route_b.get(
                    'centroid', [route_b['latitude'], route_b['longitude']])
                dist = haversine_distance(centroid_a[0], centroid_a[1],
                                          centroid_b[0], centroid_b[1])

                bearing_a = calculate_route_bearing(route_a, office_lat,
                                                    office_lon)
                bearing_b = calculate_route_bearing(route_b, office_lat,
                                                    office_lon)
                bearing_diff = bearing_difference(bearing_a, bearing_b)

                if dist <= MERGE_DISTANCE_KM and bearing_diff <= MAX_BEARING_DIFFERENCE:
                    # Merge route_a into route_b (or a new combined route)
                    route_b['assigned_users'] = combined_users
                    route_b['vehicle_type'] = max(route_a['vehicle_type'], route_b['vehicle_type']) # Update capacity if needed
                    route_b['utilization'] = len(combined_users) / route_b['vehicle_type']
                    routes_used.add(j) # Mark route_b as merged
                    merged = True
                    break # Stop searching for merges for route_a

        if not merged:
            merged_routes.append(route_a) # Keep route_a if not merged
        else:
            merged_routes.append(route_b) # Add the updated route_b
        routes_used.add(i) # Mark route_a as processed

    # Add any routes that were not merged
    for i, route in enumerate(drivers_and_routes):
        if i not in routes_used:
            merged_routes.append(route)

    drivers_and_routes = merged_routes

    # Fallback reassignment for low utilization routes
    low_util_routes = [
        r for r in drivers_and_routes
        if r.get('utilization', 0) < LOW_UTILIZATION_THRESHOLD
        and len(r['assigned_users']) <= MAX_USERS_FOR_FALLBACK
    ]
    combined_users = []
    fallback_route_indices = []

    for route in low_util_routes:
        combined_users.extend(route['assigned_users'])
        idx = drivers_and_routes.index(route)
        fallback_route_indices.append(idx)

    # Check if the combined users fit the criteria for a fallback assignment
    if FALLBACK_MIN_USERS <= len(combined_users) <= FALLBACK_MAX_USERS:
        fallback_driver = None
        # Find a suitable unassigned driver
        for i, drv in enumerate(unassigned_drivers):
            if drv['capacity'] >= len(combined_users):
                fallback_driver = unassigned_drivers.pop(i) # Take driver from list
                break

        if fallback_driver:
            # Remove the original low-utilization routes
            for idx in fallback_route_indices:
                routes_to_remove.add(idx)

            # Create a new route with the fallback driver and combined users
            new_route = {
                'driver_id': fallback_driver['driver_id'],
                'vehicle_id': fallback_driver['vehicle_id'],
                'vehicle_type': fallback_driver['capacity'],
                'latitude': fallback_driver['latitude'],
                'longitude': fallback_driver['longitude'],
                'assigned_users': combined_users
            }
            drivers_and_routes.append(new_route)

    # Individual reassignment for remaining underutilized routes
    for i, source_route in enumerate(drivers_and_routes):
        if i in routes_to_remove or source_route.get('utilization',
                                                     0) >= MIN_UTIL_THRESHOLD:
            continue

        source_bearing = calculate_route_bearing(source_route, office_lat,
                                                 office_lon)
        remaining_users_in_source = []

        for user in source_route['assigned_users']:
            best_target_route_index = -1
            best_target_cost = float('inf')

            # Find the best target route for the user
            for j, target_route in enumerate(drivers_and_routes):
                # Skip self, full routes, or already removed routes
                if j == i or j in routes_to_remove or len(target_route['assigned_users']
                                                          ) >= target_route['vehicle_type']:
                    continue

                # Calculate distance to target route's centroid
                centroid = target_route.get(
                    'centroid',
                    [target_route['latitude'], target_route['longitude']])
                distance = haversine_distance(centroid[0], centroid[1],
                                              user['lat'], user['lng'])

                # Calculate bearing difference
                target_bearing = calculate_route_bearing(
                    target_route, office_lat, office_lon)
                bearing_diff = bearing_difference(source_bearing,
                                                  target_bearing)

                # Check if within merge distance and bearing difference constraints
                if distance <= MERGE_DISTANCE_KM and bearing_diff <= 45:
                    cost = distance + (bearing_diff / 90.0) # Weighted cost
                    if cost < best_target_cost:
                        best_target_cost = cost
                        best_target_route_index = j

            # If a suitable target route was found, reassign the user
            if best_target_route_index != -1:
                drivers_and_routes[best_target_route_index]['assigned_users'].append(user)
                reassigned_users.append(user['user_id'])
            else:
                remaining_users_in_source.append(user) # User could not be reassigned

        # Update the source route with remaining users
        source_route['assigned_users'] = remaining_users_in_source
        # If no users left in source route, mark for removal
        if not source_route['assigned_users']:
            routes_to_remove.add(i)

    # Filter out routes marked for removal
    drivers_and_routes = [
        r for idx, r in enumerate(drivers_and_routes)
        if idx not in routes_to_remove
    ]
    drivers_and_routes = finalize_routes(drivers_and_routes, office_lat,
                                         office_lon)
    return drivers_and_routes


def finalize_routes(drivers_and_routes, office_lat, office_lon):
    """Finalize routes with updated metrics and validate driver uniqueness"""
    # Validate driver uniqueness to prevent duplicacy
    driver_ids_seen = set()
    unique_routes = []

    for route in drivers_and_routes:
        driver_id = route['driver_id']
        if driver_id in driver_ids_seen:
            print(f"⚠️ WARNING: Duplicate driver {driver_id} detected and removed!")
            continue  # Skip duplicate driver routes

        driver_ids_seen.add(driver_id)

        # Update route metrics
        if route['assigned_users']:
            lats = [u['lat'] for u in route['assigned_users']]
            lngs = [u['lng'] for u in route['assigned_users']]
            route['centroid'] = [np.mean(lats), np.mean(lngs)]
            route['utilization'] = len(route['assigned_users']) / route['vehicle_type']
            route['stops'] = [[u['lat'], u['lng']] for u in route['assigned_users']]
            route['bearing'] = calculate_bearing(route['centroid'][0], route['centroid'][1], office_lat, office_lon)
        else:
            route['centroid'] = [route['latitude'], route['longitude']]
            route['utilization'] = 0
            route['stops'] = []
            route['bearing'] = 0

        unique_routes.append(route)

    print(f"✅ Driver uniqueness validated - {len(unique_routes)} unique routes finalized")
    return unique_routes


# MAIN SIMPLIFIED ENHANCED ASSIGNMENT FUNCTION
def run_assignment(source_id: str, parameter: int = 1, string_param: str = ""):
    """
    Simplified enhanced assignment that keeps what works
    """
    start_time = time.time()

    try:
        print(
            f"🚀 Starting simplified enhanced assignment for source_id: {source_id}, parameter: {parameter}, string_param: {string_param}"
        )

        # Load and validate data
        data = load_env_and_fetch_data(source_id, parameter, string_param)
        total_drivers = len(data.get('driversUnassigned', [])) + len(
            data.get('driversAssigned', []))
        print(
            f"📥 Data loaded - Users: {len(data.get('users', []))}, Total Drivers: {total_drivers}"
        )
        print(
            f"   - driversUnassigned: {len(data.get('driversUnassigned', []))}"
        )
        print(f"   - driversAssigned: {len(data.get('driversAssigned', []))}")

        # Extract dynamic office coordinates
        office_lat, office_lon = extract_office_coordinates(data)
        print(f"🏢 Office coordinates - Lat: {office_lat}, Lon: {office_lon}")

        validate_input_data(data)
        print("✅ Data validation passed")

        user_df, driver_df = prepare_user_driver_dataframes(data)
        print(
            f"📊 DataFrames prepared - Users: {len(user_df)}, Drivers: {len(driver_df)}"
        )

        user_df = calculate_bearings_and_features(user_df, office_lat,
                                                  office_lon)
        print("🧭 Bearings and geographical features calculated")

        # Enhanced but practical clustering
        user_df, clustering_results = enhanced_geographical_clustering(
            user_df, office_lat, office_lon)

        clusters_found = user_df['geo_cluster'].nunique()
        print(
            f"🗂️ Enhanced clustering complete - {clusters_found} clusters found using {clustering_results['method']}"
        )

        # Optimized assignment
        drivers_and_routes, assigned_user_ids = optimize_driver_assignment(
            user_df, driver_df)
        print(
            f"🚗 Initial assignment complete - {len(drivers_and_routes)} routes, {len(assigned_user_ids)} users assigned"
        )

        # Fill underutilized routes
        assigned_user_ids = fill_underutilized_routes(drivers_and_routes,
                                                      user_df,
                                                      assigned_user_ids)
        print(
            f"📈 Underutilized routes filled - {len(assigned_user_ids)} total users assigned"
        )

        # Simplified local search optimization
        drivers_and_routes = simplified_local_search(drivers_and_routes)

        # Handle remaining users
        unassigned_users = handle_remaining_users(user_df, drivers_and_routes,
                                                  assigned_user_ids, driver_df)
        print(
            f"👥 Remaining users handled - {len(unassigned_users)} unassigned")

        # Merge underutilized routes
        drivers_and_routes = merge_underutilized_routes(drivers_and_routes)
        print("🔄 Route merging optimization complete")

        # Build unassigned drivers list
        assigned_driver_ids = {
            route['driver_id']
            for route in drivers_and_routes
        }
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

        # Final reassignment optimization
        drivers_and_routes = reassign_underutilized_users_to_nearby_routes(
            drivers_and_routes, unassigned_drivers, office_lat, office_lon)

        # Validate user uniqueness before finalizing
        all_assigned_user_ids = set()
        validated_routes = []

        for route in drivers_and_routes:
            unique_users = []
            for user in route['assigned_users']:
                user_id = user['user_id']
                if user_id not in all_assigned_user_ids:
                    all_assigned_user_ids.add(user_id)
                    unique_users.append(user)
                else:
                    print(f"⚠️ WARNING: Duplicate user {user_id} detected and removed from route!")

            route['assigned_users'] = unique_users
            validated_routes.append(route)

        drivers_and_routes = validated_routes

        # Finalize routes with all metrics
        drivers_and_routes = finalize_routes(drivers_and_routes, office_lat, office_lon)
        print("✅ Routes finalized with metrics and uniqueness validated")

        execution_time = time.time() - start_time

        print(
            f"✅ Simplified enhanced assignment complete in {execution_time:.2f}s"
        )
        print(f"📊 Final routes: {len(drivers_and_routes)}")
        print(f"🎯 Clustering method: {clustering_results['method']}")

        # Use the provided parameter
        print(f"📋 Parameter value: {parameter}")
        print(f"📋 String parameter value: {string_param}")

        return {
            "status": "true",
            "execution_time": execution_time,
            "data": drivers_and_routes,
            "unassignedUsers": unassigned_users,
            "unassignedDrivers": unassigned_drivers,
            "clustering_analysis": clustering_results,
            "parameter": parameter,
            "string_param": string_param
        }

    except requests.exceptions.RequestException as req_err:
        return {"status": "false", "details": str(req_err), "data": []}
    except Exception as e:
        logger.error(f"Assignment failed: {e}")
        return {"status": "false", "details": str(e), "data": []}


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