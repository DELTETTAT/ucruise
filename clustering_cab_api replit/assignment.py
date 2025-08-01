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
    print(f"Warning: Could not load config.json, using default values. Error: {e}")
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
    """Validate input data structure and content"""
    if not isinstance(data, dict):
        raise ValueError("Data must be a dictionary")

    if "users" not in data or "drivers" not in data:
        raise ValueError("Data must contain 'users' and 'drivers' keys")

    users = data.get("users", [])
    drivers = data.get("drivers", [])

    if not users:
        raise ValueError("No users provided")
    if not drivers:
        raise ValueError("No drivers provided")

    # Validate user data structure
    required_user_fields = ['id', 'latitude', 'longitude']
    for i, user in enumerate(users):
        for field in required_user_fields:
            if field not in user:
                raise ValueError(f"User {i} missing required field: {field}")

        # Validate coordinates
        try:
            lat, lon = float(user['latitude']), float(user['longitude'])
            if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
                raise ValueError(f"User {i} has invalid coordinates: {lat}, {lon}")
        except (ValueError, TypeError):
            raise ValueError(f"User {i} has invalid coordinate values")

    # Validate driver data structure
    required_driver_fields = ['id', 'capacity', 'latitude', 'longitude']
    for i, driver in enumerate(drivers):
        for field in required_driver_fields:
            if field not in driver:
                raise ValueError(f"Driver {i} missing required field: {field}")

        # Validate capacity
        try:
            capacity = int(driver['capacity'])
            if capacity <= 0:
                raise ValueError(f"Driver {i} has invalid capacity: {capacity}")
        except (ValueError, TypeError):
            raise ValueError(f"Driver {i} has invalid capacity value")

        # Validate coordinates
        try:
            lat, lon = float(driver['latitude']), float(driver['longitude'])
            if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
                raise ValueError(f"Driver {i} has invalid coordinates: {lat}, {lon}")
        except (ValueError, TypeError):
            raise ValueError(f"Driver {i} has invalid coordinate values")

def load_env_and_fetch_data(source_id: str):
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

    API_URL = f"{BASE_API_URL.rstrip('/')}/{source_id}"
    headers = {
        "Authorization": f"Bearer {API_AUTH_TOKEN}",
        "Content-Type": "application/json"
    }
    resp = requests.get(API_URL, headers=headers)
    resp.raise_for_status()
    payload = resp.json()

    if not payload.get("status") or "data" not in payload:
        raise ValueError("Unexpected response format: 'status' or 'data' missing")

    return payload["data"]

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
    """Prepare and clean user and driver dataframes"""
    users_list = data.get("users", [])
    drivers_list = data.get("drivers", [])

    df_users = pd.DataFrame(users_list)
    df_drivers = pd.DataFrame(drivers_list)

    numeric_cols = ["office_distance", "latitude", "longitude", "capacity"]
    for col in numeric_cols:
        if col in df_users.columns:
            df_users[col] = pd.to_numeric(df_users[col], errors="coerce")
        if col in df_drivers.columns:
            df_drivers[col] = pd.to_numeric(df_drivers[col], errors="coerce")

    user_df = df_users.rename(columns={"id": "user_id"})[
        ['user_id','latitude','longitude','office_distance','shift_type','first_name','email']
    ]
    user_df['user_id'] = user_df['user_id'].astype(str)

    driver_df = df_drivers.rename(columns={"id": "driver_id"})[
        ['driver_id','capacity','vehicle_id','latitude','longitude']
    ]
    driver_df['driver_id'] = driver_df['driver_id'].astype(str)

    for col in ['latitude','longitude','office_distance']:
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
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c

def calculate_bearing(lat1, lon1, lat2, lon2):
    """Calculate bearing from point A to B in degrees"""
    lat1, lat2 = map(math.radians, [lat1, lat2])
    dlon = math.radians(lon2 - lon1)
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    return (math.degrees(math.atan2(x, y)) + 360) % 360

def calculate_bearings_and_features(user_df, office_lat, office_lon):
    """Calculate bearings and add geographical features"""
    user_df = user_df.copy()
    user_df[['office_latitude','office_longitude']] = office_lat, office_lon

    def calculate_bearing_vectorized(lat1, lon1, lat2, lon2):
        lat1, lat2 = np.radians(lat1), np.radians(lat2)
        dlon = np.radians(lon2 - lon1)
        x = np.sin(dlon) * np.cos(lat2)
        y = np.cos(lat1)*np.sin(lat2) - np.sin(lat1)*np.cos(lat2)*np.cos(dlon)
        return (np.degrees(np.arctan2(x,y)) + 360) % 360

    user_df['bearing'] = calculate_bearing_vectorized(
        user_df['latitude'],user_df['longitude'],office_lat,office_lon
    )
    user_df['bearing_sin'] = np.sin(np.radians(user_df['bearing']))
    user_df['bearing_cos'] = np.cos(np.radians(user_df['bearing']))

    # Geographic quadrant analysis - simplified
    user_df['quadrant'] = 'CENTER'
    user_df.loc[(user_df['latitude'] > office_lat) & (user_df['longitude'] > office_lon), 'quadrant'] = 'NE'
    user_df.loc[(user_df['latitude'] > office_lat) & (user_df['longitude'] <= office_lon), 'quadrant'] = 'NW'
    user_df.loc[(user_df['latitude'] <= office_lat) & (user_df['longitude'] > office_lon), 'quadrant'] = 'SE'
    user_df.loc[(user_df['latitude'] <= office_lat) & (user_df['longitude'] <= office_lon), 'quadrant'] = 'SW'

    return user_df

# ENHANCED BUT SIMPLIFIED CLUSTERING
def enhanced_geographical_clustering(user_df, office_lat, office_lon):
    """
    Enhanced clustering that's more practical than the original advanced version
    """
    coords = user_df[['latitude', 'longitude']].values

    # Primary: DBSCAN clustering (proven to work well)
    coords_rad = np.radians(coords)
    eps_rad = DBSCAN_EPS_KM / 6371.0

    clustering = DBSCAN(eps=eps_rad, min_samples=MIN_SAMPLES_DBSCAN, metric='haversine').fit(coords_rad)
    user_df = user_df.copy()
    user_df['geo_cluster'] = clustering.labels_

    # Handle noise points by assigning to nearest cluster
    noise_mask = user_df['geo_cluster'] == -1
    if noise_mask.any():
        noise_points = user_df[noise_mask][['latitude', 'longitude']].values
        valid_clusters = user_df[user_df['geo_cluster'] != -1]

        if not valid_clusters.empty:
            cluster_centers = valid_clusters.groupby('geo_cluster')[['latitude', 'longitude']].mean().values
            distances = cdist(noise_points, cluster_centers, metric='euclidean')
            nearest_clusters = valid_clusters['geo_cluster'].unique()[np.argmin(distances, axis=1)]
            user_df.loc[noise_mask, 'geo_cluster'] = nearest_clusters
        else:
            user_df.loc[noise_mask, 'geo_cluster'] = 0

    # If DBSCAN fails completely, use bearing-enhanced K-means as fallback
    if user_df['geo_cluster'].nunique() <= 1:
        print("📊 DBSCAN failed, using bearing-enhanced K-means fallback...")
        coords_features = user_df[['latitude', 'longitude', 'bearing_sin', 'bearing_cos']].values
        # Weight coordinates more heavily than bearing
        weights = np.array([1.0, 1.0, 0.3, 0.3])
        weighted_features = coords_features * weights
        scaled_features = StandardScaler().fit_transform(weighted_features)

        n_clusters = min(6, max(2, len(user_df) // 4))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        user_df['geo_cluster'] = kmeans.fit_predict(scaled_features)
        clustering_method = 'Enhanced K-means'
    else:
        clustering_method = 'DBSCAN'

    print(f"🗂️ Clustering complete: {user_df['geo_cluster'].nunique()} clusters using {clustering_method}")

    return user_df, {'method': clustering_method, 'clusters': user_df['geo_cluster'].nunique()}

# SIMPLIFIED COST FUNCTION
def calculate_enhanced_cost(route, driver_pos, user_positions, office_lat, office_lon):
    """Simplified but effective cost function"""
    if not user_positions:
        return float('inf')

    # Distance component (primary factor)
    centroid = np.mean(user_positions, axis=0)
    distance_cost = haversine_distance(driver_pos[0], driver_pos[1], centroid[0], centroid[1])

    # Utilization component (secondary factor)
    utilization = len(user_positions) / route.get('capacity', 1)
    utilization_penalty = (1 - utilization) * UTILIZATION_PENALTY_PER_SEAT

    # Simple compactness penalty
    if len(user_positions) > 1:
        distances = [haversine_distance(pos[0], pos[1], centroid[0], centroid[1]) for pos in user_positions]
        compactness_penalty = np.std(distances) * 0.5  # Reduced weight
    else:
        compactness_penalty = 0

    total_cost = distance_cost + utilization_penalty + compactness_penalty
    return total_cost

def optimize_driver_assignment(user_df, driver_df):
    """
    Original optimized driver assignment with minor enhancements
    """
    drivers_and_routes = []
    available_drivers = driver_df.sort_values('capacity', ascending=False).to_dict('records')
    assigned_user_ids = set()

    for cluster_id, cluster_users in user_df.groupby('geo_cluster'):
        unassigned_in_cluster = cluster_users[~cluster_users['user_id'].isin(assigned_user_ids)].copy()

        if unassigned_in_cluster.empty or not available_drivers:
            continue

        total_users = len(unassigned_in_cluster)
        if not available_drivers:
            continue

        avg_capacity = np.mean([d['capacity'] for d in available_drivers])
        max_capacity = max([d['capacity'] for d in available_drivers])

        if total_users <= max_capacity:
            sub_clusters = 1
        else:
            sub_clusters = min(
                max(1, round(total_users / avg_capacity)),
                len(available_drivers),
                total_users
            )

        if sub_clusters > 1:
            coords = unassigned_in_cluster[['latitude', 'longitude']].values
            kmeans = KMeans(n_clusters=sub_clusters, random_state=42)
            unassigned_in_cluster['sub_cluster'] = kmeans.fit_predict(coords)
        else:
            unassigned_in_cluster['sub_cluster'] = 0

        for sub_id, sub_users in unassigned_in_cluster.groupby('sub_cluster'):
            if sub_users.empty or not available_drivers:
                continue

            sub_centroid = sub_users[['latitude', 'longitude']].mean().values
            users_needed = len(sub_users)

            best_driver = None
            min_cost = float('inf')

            for driver in available_drivers:
                driver_pos = (driver['latitude'], driver['longitude'])
                distance = haversine_distance(
                    driver_pos[0], driver_pos[1], 
                    sub_centroid[0], sub_centroid[1]
                )

                if users_needed <= driver['capacity']:
                    utilization = users_needed / driver['capacity']
                    utilization_penalty = (1 - utilization) * UTILIZATION_PENALTY_PER_SEAT
                    cost = distance + utilization_penalty
                else:
                    cost = distance + OVERFLOW_PENALTY_KM

                if cost < min_cost:
                    min_cost = cost
                    best_driver = driver

            if not best_driver:
                continue

            capacity = best_driver['capacity']
            users_to_assign = sub_users.head(capacity)

            route = create_route_from_users(best_driver, users_to_assign.to_dict('records'), 
                                          user_df.iloc[0]['office_latitude'], user_df.iloc[0]['office_longitude'])

            for _, user in users_to_assign.iterrows():
                assigned_user_ids.add(user['user_id'])

            drivers_and_routes.append(route)
            available_drivers.remove(best_driver)

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
        route['utilization'] = len(route['assigned_users']) / route['vehicle_type']
        route['stops'] = [[u['lat'], u['lng']] for u in route['assigned_users']]
        route['bearing'] = calculate_bearing(route['centroid'][0], route['centroid'][1], office_lat, office_lon)

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
            for j, route_b in enumerate(routes[i+1:], start=i+1):
                if not route_a['assigned_users'] or not route_b['assigned_users']:
                    continue

                # Try swapping users between routes
                for user_a in route_a['assigned_users'][:]:
                    for user_b in route_b['assigned_users'][:]:
                        if calculate_swap_improvement(route_a, route_b, user_a, user_b) > SWAP_IMPROVEMENT_THRESHOLD:
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
    centroid_a = route_a.get('centroid', [route_a['latitude'], route_a['longitude']])
    centroid_b = route_b.get('centroid', [route_b['latitude'], route_b['longitude']])

    current_dist_a = haversine_distance(centroid_a[0], centroid_a[1], user_a['lat'], user_a['lng'])
    current_dist_b = haversine_distance(centroid_b[0], centroid_b[1], user_b['lat'], user_b['lng'])

    swap_dist_a = haversine_distance(centroid_a[0], centroid_a[1], user_b['lat'], user_b['lng'])
    swap_dist_b = haversine_distance(centroid_b[0], centroid_b[1], user_a['lat'], user_a['lng'])

    return (current_dist_a + current_dist_b) - (swap_dist_a + swap_dist_b)

def update_route_metrics(routes):
    """Update route metrics after modifications"""
    for route in routes:
        if route['assigned_users']:
            lats = [u['lat'] for u in route['assigned_users']]
            lngs = [u['lng'] for u in route['assigned_users']]
            route['centroid'] = [np.mean(lats), np.mean(lngs)]
            route['utilization'] = len(route['assigned_users']) / route['vehicle_type']
            route['stops'] = [[u['lat'], u['lng']] for u in route['assigned_users']]

# Keep the original utility functions
def fill_underutilized_routes(drivers_and_routes, user_df, assigned_user_ids):
    """Fill underutilized routes with nearby unassigned users"""
    unassigned_users = user_df[~user_df['user_id'].isin(assigned_user_ids)].copy()

    for route in drivers_and_routes:
        capacity = route['vehicle_type']
        current_size = len(route['assigned_users'])

        if current_size < capacity and not unassigned_users.empty:
            if route['assigned_users']:
                centroid_lat = np.mean([u['lat'] for u in route['assigned_users']])
                centroid_lng = np.mean([u['lng'] for u in route['assigned_users']])
            else:
                centroid_lat, centroid_lng = route['latitude'], route['longitude']

            distances = []
            for _, user in unassigned_users.iterrows():
                dist = haversine_distance(
                    centroid_lat, centroid_lng,
                    user['latitude'], user['longitude']
                )
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
                unassigned_users = unassigned_users[unassigned_users['user_id'] != user['user_id']]

    return assigned_user_ids

def merge_underutilized_routes(drivers_and_routes):
    """Merge underutilized routes with nearby routes"""
    merged_routes = []
    used_indices = set()

    for i, route_a in enumerate(drivers_and_routes):
        if i in used_indices:
            continue

        if route_a.get('utilization', 0) >= MIN_UTIL_THRESHOLD:
            merged_routes.append(route_a)
            continue

        merged = False
        centroid_a = route_a.get('centroid', [route_a['latitude'], route_a['longitude']])

        for j, route_b in enumerate(drivers_and_routes):
            if j == i or j in used_indices:
                continue

            centroid_b = route_b.get('centroid', [route_b['latitude'], route_b['longitude']])
            dist = haversine_distance(
                centroid_a[0], centroid_a[1], centroid_b[0], centroid_b[1]
            )

            total_users = len(route_a['assigned_users']) + len(route_b['assigned_users'])
            max_capacity = route_b['vehicle_type']

            if dist <= MERGE_DISTANCE_KM and total_users <= max_capacity:
                route_b['assigned_users'].extend(route_a['assigned_users'])
                lats = [u['lat'] for u in route_b['assigned_users']]
                lngs = [u['lng'] for u in route_b['assigned_users']]
                route_b['centroid'] = [np.mean(lats), np.mean(lngs)]
                route_b['utilization'] = len(route_b['assigned_users']) / route_b['vehicle_type']
                route_b['stops'] = [[u['lat'], u['lng']] for u in route_b['assigned_users']]
                used_indices.add(i)
                merged = True
                break

        if not merged:
            merged_routes.append(route_a)

    return merged_routes

def handle_remaining_users(user_df, drivers_and_routes, assigned_user_ids, driver_df):
    """Handle remaining unassigned users"""
    unassigned_users = user_df[~user_df['user_id'].isin(assigned_user_ids)].copy()
    available_drivers = driver_df[~driver_df['driver_id'].isin(
        [route['driver_id'] for route in drivers_and_routes]
    )].copy()

    if unassigned_users.empty:
        return []

    # Try to fit into existing routes first
    for _, user in unassigned_users.iterrows():
        best_route = None
        min_distance = float('inf')

        for route in drivers_and_routes:
            if len(route['assigned_users']) < route['vehicle_type']:
                if route['assigned_users']:
                    centroid_lat = np.mean([u['lat'] for u in route['assigned_users']])
                    centroid_lng = np.mean([u['lng'] for u in route['assigned_users']])
                else:
                    centroid_lat, centroid_lng = route['latitude'], route['longitude']

                dist = haversine_distance(
                    centroid_lat, centroid_lng,
                    user['latitude'], user['longitude']
                )

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

    # Create new routes for remaining users
    remaining_unassigned = user_df[~user_df['user_id'].isin(assigned_user_ids)].copy()

    if not remaining_unassigned.empty and not available_drivers.empty:
        if len(remaining_unassigned) > 1:
            coords = remaining_unassigned[['latitude', 'longitude']].values
            n_clusters = min(len(available_drivers), len(remaining_unassigned))
            if n_clusters > 1:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                remaining_unassigned['cluster'] = kmeans.fit_predict(coords)
            else:
                remaining_unassigned['cluster'] = 0
        else:
            remaining_unassigned['cluster'] = 0

        available_drivers_list = available_drivers.to_dict('records')

        for cluster_id, cluster_users in remaining_unassigned.groupby('cluster'):
            if not available_drivers_list:
                break

            centroid = cluster_users[['latitude', 'longitude']].mean().values

            closest_driver = min(
                available_drivers_list,
                key=lambda d: haversine_distance(
                    centroid[0], centroid[1], d['latitude'], d['longitude']
                )
            )

            capacity = closest_driver['capacity']
            users_to_assign = cluster_users.head(capacity)

            route = create_route_from_users(closest_driver, users_to_assign.to_dict('records'), 
                                          user_df.iloc[0]['office_latitude'], user_df.iloc[0]['office_longitude'])

            for _, user in users_to_assign.iterrows():
                assigned_user_ids.add(user['user_id'])

            drivers_and_routes.append(route)
            available_drivers_list.remove(closest_driver)

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

def calculate_route_bearing(route, office_lat, office_lon):
    """Calculate representative bearing for a route"""
    if not route['assigned_users']:
        return 0
    avg_lat = sum(u['lat'] for u in route['assigned_users']) / len(route['assigned_users'])
    avg_lng = sum(u['lng'] for u in route['assigned_users']) / len(route['assigned_users'])
    return calculate_bearing(avg_lat, avg_lng, office_lat, office_lon)

def reassign_underutilized_users_to_nearby_routes(drivers_and_routes, unassigned_drivers, office_lat, office_lon):
    """Reassign users from underutilized routes to nearby better routes"""
    reassigned_users = []
    routes_to_remove = set()

    # Merge underfilled nearby routes
    merged_routes = []
    routes_used = set()

    for i, route_a in enumerate(drivers_and_routes):
        if i in routes_used:
            continue

        if route_a.get('utilization', 0) >= 1.0:
            merged_routes.append(route_a)
            routes_used.add(i)
            continue

        merged = False
        for j, route_b in enumerate(drivers_and_routes[i+1:], start=i+1):
            if j in routes_used or route_b.get('utilization', 0) >= 1.0:
                continue

            combined_users = route_a['assigned_users'] + route_b['assigned_users']
            if len(combined_users) <= max(route_a['vehicle_type'], route_b['vehicle_type']):
                centroid_a = route_a.get('centroid', [route_a['latitude'], route_a['longitude']])
                centroid_b = route_b.get('centroid', [route_b['latitude'], route_b['longitude']])
                dist = haversine_distance(centroid_a[0], centroid_a[1], centroid_b[0], centroid_b[1])

                bearing_a = calculate_route_bearing(route_a, office_lat, office_lon)
                bearing_b = calculate_route_bearing(route_b, office_lat, office_lon)
                bearing_diff = bearing_difference(bearing_a, bearing_b)

                if dist <= MERGE_DISTANCE_KM and bearing_diff <= MAX_BEARING_DIFFERENCE:
                    route_a['assigned_users'] = combined_users
                    route_a['vehicle_type'] = max(route_a['vehicle_type'], route_b['vehicle_type'])
                    route_a['utilization'] = len(combined_users) / route_a['vehicle_type']
                    routes_used.add(j)
                    merged = True
                    break

        if not merged:
            merged_routes.append(route_a)
        else:
            merged_routes.append(route_a)
        routes_used.add(i)

    drivers_and_routes = merged_routes

    # Fallback reassignment for low utilization routes
    low_util_routes = [r for r in drivers_and_routes if r.get('utilization', 0) < LOW_UTILIZATION_THRESHOLD and len(r['assigned_users']) <= MAX_USERS_FOR_FALLBACK]
    combined_users = []
    fallback_route_indices = []

    for route in low_util_routes:
        combined_users.extend(route['assigned_users'])
        idx = drivers_and_routes.index(route)
        fallback_route_indices.append(idx)

    if FALLBACK_MIN_USERS <= len(combined_users) <= FALLBACK_MAX_USERS:
        fallback_driver = None
        for i, drv in enumerate(unassigned_drivers):
            if drv['capacity'] >= len(combined_users):
                fallback_driver = unassigned_drivers.pop(i)
                break

        if fallback_driver:
            for idx in fallback_route_indices:
                routes_to_remove.add(idx)

            new_route = {
                'driver_id': fallback_driver['driver_id'],
                'vehicle_id': fallback_driver['vehicle_id'],
                'vehicle_type': fallback_driver['capacity'],
                'latitude': fallback_driver['latitude'],
                'longitude': fallback_driver['longitude'],
                'assigned_users': combined_users
            }
            drivers_and_routes.append(new_route)

    # Individual reassignment
    for i, source_route in enumerate(drivers_and_routes):
        if i in routes_to_remove or source_route.get('utilization', 0) >= MIN_UTIL_THRESHOLD:
            continue

        source_bearing = calculate_route_bearing(source_route, office_lat, office_lon)
        remaining_users = []

        for user in source_route['assigned_users']:
            best_target = None
            best_cost = float('inf')

            for j, target_route in enumerate(drivers_and_routes):
                if j == i or len(target_route['assigned_users']) >= target_route['vehicle_type']:
                    continue

                centroid = target_route.get('centroid', [target_route['latitude'], target_route['longitude']])
                distance = haversine_distance(centroid[0], centroid[1], user['lat'], user['lng'])
                target_bearing = calculate_route_bearing(target_route, office_lat, office_lon)
                bearing_diff = bearing_difference(source_bearing, target_bearing)

                if distance <= MERGE_DISTANCE_KM and bearing_diff <= 45:
                    cost = distance + (bearing_diff / 90.0)
                    if cost < best_cost:
                        best_cost = cost
                        best_target = target_route

            if best_target:
                best_target['assigned_users'].append(user)
                reassigned_users.append(user['user_id'])
            else:
                remaining_users.append(user)

        if not remaining_users:
            routes_to_remove.add(i)
        else:
            source_route['assigned_users'] = remaining_users

    drivers_and_routes = [r for idx, r in enumerate(drivers_and_routes) if idx not in routes_to_remove]
    drivers_and_routes = finalize_routes(drivers_and_routes, office_lat, office_lon)
    return drivers_and_routes

def finalize_routes(drivers_and_routes, office_lat, office_lon):
    """Finalize routes with updated metrics"""
    for route in drivers_and_routes:
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
    return drivers_and_routes

# MAIN SIMPLIFIED ENHANCED ASSIGNMENT FUNCTION
def run_assignment(source_id: str):
    """
    Simplified enhanced assignment that keeps what works
    """
    start_time = time.time()

    try:
        print(f"🚀 Starting simplified enhanced assignment for source_id: {source_id}")

        # Load and validate data
        data = load_env_and_fetch_data(source_id)
        print(f"📥 Data loaded - Users: {len(data.get('users', []))}, Drivers: {len(data.get('drivers', []))}")

        # Extract dynamic office coordinates
        office_lat, office_lon = extract_office_coordinates(data)
        print(f"🏢 Office coordinates - Lat: {office_lat}, Lon: {office_lon}")

        validate_input_data(data)
        print("✅ Data validation passed")

        user_df, driver_df = prepare_user_driver_dataframes(data)
        print(f"📊 DataFrames prepared - Users: {len(user_df)}, Drivers: {len(driver_df)}")

        user_df = calculate_bearings_and_features(user_df, office_lat, office_lon)
        print("🧭 Bearings and geographical features calculated")

        # Enhanced but practical clustering
        if ENHANCED_CLUSTERING:
            user_df, clustering_results = enhanced_geographical_clustering(user_df, office_lat, office_lon)
        else:
            user_df, clustering_results = enhanced_geographical_clustering(user_df, office_lat, office_lon)

        clusters_found = user_df['geo_cluster'].nunique()
        print(f"🗂️ Enhanced clustering complete - {clusters_found} clusters found using {clustering_results['method']}")

        # Optimized assignment
        drivers_and_routes, assigned_user_ids = optimize_driver_assignment(user_df, driver_df)
        print(f"🚗 Initial assignment complete - {len(drivers_and_routes)} routes, {len(assigned_user_ids)} users assigned")

        # Fill underutilized routes
        assigned_user_ids = fill_underutilized_routes(drivers_and_routes, user_df, assigned_user_ids)
        print(f"📈 Underutilized routes filled - {len(assigned_user_ids)} total users assigned")

        # Simplified local search optimization
        drivers_and_routes = simplified_local_search(drivers_and_routes)

        # Handle remaining users
        unassigned_users = handle_remaining_users(user_df, drivers_and_routes, assigned_user_ids, driver_df)
        print(f"👥 Remaining users handled - {len(unassigned_users)} unassigned")

        # Merge underutilized routes
        drivers_and_routes = merge_underutilized_routes(drivers_and_routes)
        print("🔄 Route merging optimization complete")

        # Build unassigned drivers list
        assigned_driver_ids = {route['driver_id'] for route in drivers_and_routes}
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

        # Final reassignment optimization
        drivers_and_routes = reassign_underutilized_users_to_nearby_routes(drivers_and_routes, unassigned_drivers, office_lat, office_lon)

        # Finalize routes with all metrics
        drivers_and_routes = finalize_routes(drivers_and_routes, office_lat, office_lon)
        print("✅ Routes finalized with metrics")

        execution_time = time.time() - start_time

        print(f"✅ Simplified enhanced assignment complete in {execution_time:.2f}s")
        print(f"📊 Final routes: {len(drivers_and_routes)}")
        print(f"🎯 Clustering method: {clustering_results['method']}")

        return {
            "status": "true",
            "execution_time": execution_time,
            "data": drivers_and_routes,
            "unassignedUsers": unassigned_users,
            "unassignedDrivers": unassigned_drivers,
            "clustering_analysis": clustering_results
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
    total_assigned = sum(len(route["assigned_users"]) for route in result["data"])
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
        "min_utilization": round(np.min(utilizations) * 100, 1) if utilizations else 0,
        "max_utilization": round(np.max(utilizations) * 100, 1) if utilizations else 0,
        "routes_below_80_percent": sum(1 for u in utilizations if u < 0.8),
        "distance_issues": distance_issues,
        "clustering_method": result.get("clustering_analysis", {}).get("method", "Unknown")
    }

    return analysis