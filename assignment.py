import os
import math
import json
import requests
import numpy as np
import pandas as pd
import time
from functools import lru_cache
from sklearn.cluster import DBSCAN, KMeans
from dotenv import load_dotenv
import warnings
import logging
from road_network import RoadNetwork

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_and_validate_config():
    """Load configuration with validation and environment fallbacks"""
    try:
        with open('config.json') as f:
            cfg = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Warning: Could not load config.json, using defaults. Error: {e}")
        cfg = {}

    config = {}

    # Distance configurations
    config['MAX_FILL_DISTANCE_KM'] = max(0.1, float(cfg.get("max_fill_distance_km", 5.0)))
    config['min_road_coherence_score'] = max(0.0, min(1.0, float(cfg.get("min_road_coherence_score", 0.7))))
    config['max_extra_distance_km'] = max(1.0, float(cfg.get("max_extra_distance_km", 2.0)))
    config['max_deviation_angle'] = max(15, min(60, int(cfg.get("max_deviation_angle", 35))))
    config['max_route_deviation_ratio'] = max(1.2, float(cfg.get("max_route_deviation_ratio", 1.4)))

    # Office coordinates
    office_lat = float(os.getenv("OFFICE_LAT", cfg.get("office_latitude", 30.6810489)))
    office_lon = float(os.getenv("OFFICE_LON", cfg.get("office_longitude", 76.7260711)))

    if not (-90 <= office_lat <= 90):
        office_lat = 30.6810489
    if not (-180 <= office_lon <= 180):
        office_lon = 76.7260711

    config['OFFICE_LAT'] = office_lat
    config['OFFICE_LON'] = office_lon

    # Road network configuration
    config['use_road_network'] = cfg.get("use_road_network", False)
    config['road_network_path'] = cfg.get("road_network_path", 'tricity_main_roads.graphml')

    return config

_config = load_and_validate_config()

# Initialize road network
ROAD_NETWORK = None
if _config.get('use_road_network', False):
    road_network_path = _config.get('road_network_path', 'tricity_main_roads.graphml')
    if os.path.exists(road_network_path):
        try:
            ROAD_NETWORK = RoadNetwork(road_network_path)
            print("✅ Road network loaded successfully")
        except Exception as e:
            print(f"⚠️ Failed to load road network: {e}")
            _config['use_road_network'] = False
    else:
        print(f"⚠️ Road network file not found: {road_network_path}")
        _config['use_road_network'] = False

MAX_FILL_DISTANCE_KM = _config['MAX_FILL_DISTANCE_KM']
OFFICE_LAT = _config['OFFICE_LAT']
OFFICE_LON = _config['OFFICE_LON']

def validate_input_data(data):
    """Comprehensive data validation with bounds checking"""
    if not isinstance(data, dict):
        raise ValueError("API response must be a dictionary")

    users = data.get("users", [])
    if not users:
        raise ValueError("No users found in API response")

    # Validate users
    for i, user in enumerate(users):
        if not isinstance(user, dict):
            raise ValueError(f"User {i} must be a dictionary")

        required_fields = ["id", "latitude", "longitude"]
        for field in required_fields:
            if field not in user:
                raise ValueError(f"User {i} missing required field: {field}")
            if user[field] is None or user[field] == "":
                raise ValueError(f"User {i} has null/empty {field}")

        try:
            lat = float(user["latitude"])
            lon = float(user["longitude"])
            if not (-90 <= lat <= 90):
                raise ValueError(f"User {i} invalid latitude: {lat}")
            if not (-180 <= lon <= 180):
                raise ValueError(f"User {i} invalid longitude: {lon}")
        except (ValueError, TypeError) as e:
            raise ValueError(f"User {i} invalid coordinates: {e}")

    # Get all drivers
    all_drivers = []
    if "drivers" in data:
        drivers_data = data["drivers"]
        all_drivers.extend(drivers_data.get("driversUnassigned", []))
        all_drivers.extend(drivers_data.get("driversAssigned", []))

    if not all_drivers:
        all_drivers.extend(data.get("driversUnassigned", []))
        all_drivers.extend(data.get("driversAssigned", []))

    if not all_drivers:
        raise ValueError("No drivers found in API response")

    # Validate drivers
    for i, driver in enumerate(all_drivers):
        if not isinstance(driver, dict):
            raise ValueError(f"Driver {i} must be a dictionary")

        required_fields = ["id", "capacity", "latitude", "longitude"]
        for field in required_fields:
            if field not in driver:
                raise ValueError(f"Driver {i} missing required field: {field}")
            if driver[field] is None or driver[field] == "":
                raise ValueError(f"Driver {i} has null/empty {field}")

        try:
            lat = float(driver["latitude"])
            lon = float(driver["longitude"])
            if not (-90 <= lat <= 90):
                raise ValueError(f"Driver {i} invalid latitude: {lat}")
            if not (-180 <= lon <= 180):
                raise ValueError(f"Driver {i} invalid longitude: {lon}")
        except (ValueError, TypeError) as e:
            raise ValueError(f"Driver {i} invalid coordinates: {e}")

        try:
            capacity = int(driver["capacity"])
            if capacity <= 0:
                raise ValueError(f"Driver {i} invalid capacity: {capacity}")
        except (ValueError, TypeError) as e:
            raise ValueError(f"Driver {i} invalid capacity: {e}")

    print(f"✅ Input data validation passed - {len(users)} users, {len(all_drivers)} drivers")

def load_env_and_fetch_data(source_id: str, parameter: int = 1, string_param: str = ""):
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

    API_URL = f"{BASE_API_URL.rstrip('/')}/{source_id}/{parameter}/{string_param}"
    headers = {
        "Authorization": f"Bearer {API_AUTH_TOKEN}",
        "Content-Type": "application/json"
    }

    print(f"📡 Making API call to: {API_URL}")
    resp = requests.get(API_URL, headers=headers)
    resp.raise_for_status()

    if len(resp.text.strip()) == 0:
        raise ValueError(f"API returned empty response body")

    try:
        payload = resp.json()
    except json.JSONDecodeError as e:
        raise ValueError(f"API returned invalid JSON: {str(e)}")

    if not payload.get("status") or "data" not in payload:
        raise ValueError("Unexpected response format: 'status' or 'data' missing")

    data = payload["data"]
    data["_parameter"] = parameter
    data["_string_param"] = string_param

    # Handle nested drivers structure
    if "drivers" in data:
        drivers_data = data["drivers"]
        data["driversUnassigned"] = drivers_data.get("driversUnassigned", [])
        data["driversAssigned"] = drivers_data.get("driversAssigned", [])
    else:
        data["driversUnassigned"] = data.get("driversUnassigned", [])
        data["driversAssigned"] = data.get("driversAssigned", [])

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
    """Prepare and clean user and driver dataframes with priority system and deduplication"""
    users_list = data.get("users", [])
    drivers_unassigned_list = data.get("driversUnassigned", [])
    drivers_assigned_list = data.get("driversAssigned", [])

    df_users = pd.DataFrame(users_list)

    # Combine drivers with priority flags and deduplication
    all_drivers = []
    seen_drivers = {}

    # Priority 1: driversUnassigned with shift_type_id 1 and 3
    for driver in drivers_unassigned_list:
        driver_copy = driver.copy()
        driver_id = str(driver.get('id', ''))
        shift_type_id = int(driver.get('shift_type_id', 2))

        if shift_type_id in [1, 3]:
            priority = 1
        else:
            priority = 2

        driver_copy['priority'] = priority
        driver_copy['is_assigned'] = False

        if driver_id not in seen_drivers or seen_drivers[driver_id]['priority'] > priority:
            seen_drivers[driver_id] = driver_copy

    # Priority 3: driversAssigned with shift_type_id 1 and 3
    # Priority 4: driversAssigned with shift_type_id 2
    for driver in drivers_assigned_list:
        driver_copy = driver.copy()
        driver_id = str(driver.get('id', ''))
        shift_type_id = int(driver.get('shift_type_id', 2))

        if shift_type_id in [1, 3]:
            priority = 3
        else:
            priority = 4

        driver_copy['priority'] = priority
        driver_copy['is_assigned'] = True

        if driver_id not in seen_drivers or seen_drivers[driver_id]['priority'] > priority:
            seen_drivers[driver_id] = driver_copy

    all_drivers = list(seen_drivers.values())

    # Count priorities after deduplication
    final_priority_counts = {1: 0, 2: 0, 3: 0, 4: 0}
    for driver in all_drivers:
        final_priority_counts[driver['priority']] += 1

    print(f"🔍 Driver Priority Assignment:")
    print(f"   Priority 1 (driversUnassigned ST:1,3): {final_priority_counts[1]}")
    print(f"   Priority 2 (driversUnassigned ST:2): {final_priority_counts[2]}")
    print(f"   Priority 3 (driversAssigned ST:1,3): {final_priority_counts[3]}")
    print(f"   Priority 4 (driversAssigned ST:2): {final_priority_counts[4]}")

    df_drivers = pd.DataFrame(all_drivers)

    # Convert numeric columns
    numeric_cols = ["office_distance", "latitude", "longitude", "capacity", "shift_type_id", "priority"]
    for col in numeric_cols:
        if col in df_users.columns:
            df_users[col] = pd.to_numeric(df_users[col], errors="coerce")
        if col in df_drivers.columns:
            df_drivers[col] = pd.to_numeric(df_drivers[col], errors="coerce")

    # Prepare final dataframes with robust field handling
    expected_user_cols = ['id', 'latitude', 'longitude', 'office_distance', 'shift_type', 'first_name', 'email']
    df_users = df_users.reindex(columns=expected_user_cols)
    df_users = df_users.rename(columns={'id': 'user_id'})
    
    # Type conversions with fallbacks
    df_users['user_id'] = df_users['user_id'].astype(str).fillna('')
    df_users['latitude'] = pd.to_numeric(df_users['latitude'], errors='coerce')
    df_users['longitude'] = pd.to_numeric(df_users['longitude'], errors='coerce')
    df_users['office_distance'] = pd.to_numeric(df_users['office_distance'], errors='coerce').fillna(0.0)
    df_users['first_name'] = df_users['first_name'].fillna('').astype(str)
    df_users['email'] = df_users['email'].fillna('').astype(str)
    df_users['shift_type'] = df_users['shift_type'].fillna('').astype(str)
    
    user_df = df_users[['user_id', 'latitude', 'longitude', 'office_distance', 'shift_type', 'first_name', 'email']]

    expected_driver_cols = ['id', 'capacity', 'vehicle_id', 'latitude', 'longitude', 'shift_type_id', 'priority', 'is_assigned']
    df_drivers = pd.DataFrame(all_drivers).reindex(columns=expected_driver_cols)
    df_drivers = df_drivers.rename(columns={'id': 'driver_id'})
    
    # Type conversions with fallbacks
    df_drivers['driver_id'] = df_drivers['driver_id'].astype(str).fillna('')
    df_drivers['capacity'] = pd.to_numeric(df_drivers['capacity'], errors='coerce').fillna(0).astype(int)
    df_drivers['latitude'] = pd.to_numeric(df_drivers['latitude'], errors='coerce')
    df_drivers['longitude'] = pd.to_numeric(df_drivers['longitude'], errors='coerce')
    df_drivers['priority'] = pd.to_numeric(df_drivers['priority'], errors='coerce').fillna(2).astype(int)
    df_drivers['is_assigned'] = df_drivers['is_assigned'].fillna(False).astype(bool)
    df_drivers['shift_type_id'] = pd.to_numeric(df_drivers['shift_type_id'], errors='coerce').fillna(2).astype(int)
    df_drivers['vehicle_id'] = df_drivers['vehicle_id'].fillna('').astype(str)
    
    driver_df = df_drivers[['driver_id', 'capacity', 'vehicle_id', 'latitude', 'longitude', 'shift_type_id', 'priority', 'is_assigned']]

    # Sort drivers by priority
    driver_df = driver_df.sort_values(['priority', 'capacity'], ascending=[True, False])

    return user_df, driver_df

@lru_cache(maxsize=2000)
def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two points - uses road network if available, otherwise haversine"""
    if lat1 == lat2 and lon1 == lon2:
        return 0.0

    if ROAD_NETWORK is not None:
        try:
            return ROAD_NETWORK.get_road_distance(lat1, lon1, lat2, lon2)
        except Exception:
            pass

    # Fallback to haversine
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    a = min(1.0, abs(a))
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return abs(R * c)

def calculate_distance(lat1, lon1, lat2, lon2):
    """Alias for haversine_distance for clarity in route building"""
    return haversine_distance(lat1, lon1, lat2, lon2)


def calculate_bearing(lat1, lon1, lat2, lon2):
    """Calculate bearing from point A to B in degrees"""
    lat1, lat2 = map(math.radians, [lat1, lat2])
    dlon = math.radians(lon2 - lon1)
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    return (math.degrees(math.atan2(x, y)) + 360) % 360

def angular_difference(a, b):
    """Calculate the smallest angular difference between two bearings"""
    diff = abs(a - b) % 360
    return diff if diff <= 180 else 360 - diff

def create_geographic_clusters(user_df, office_lat, office_lon):
    """Create geographic clusters using sector-based and distance-band approach"""
    print("🗺️ Creating geographic clusters...")

    # Calculate bearings and distances from office
    user_df = user_df.copy()
    user_df['office_bearing'] = user_df.apply(
        lambda row: calculate_bearing(office_lat, office_lon, row['latitude'], row['longitude']), 
        axis=1
    )
    user_df['office_distance_calc'] = user_df.apply(
        lambda row: haversine_distance(office_lat, office_lon, row['latitude'], row['longitude']), 
        axis=1
    )

    # Create sectors (8 directional sectors of 45 degrees each)
    user_df['sector'] = ((user_df['office_bearing'] + 22.5) % 360 // 45).astype(int)

    # Create distance bands
    def assign_distance_band(distance):
        if distance <= 3:
            return 'near'
        elif distance <= 7:
            return 'medium'
        else:
            return 'far'

    user_df['distance_band'] = user_df['office_distance_calc'].apply(assign_distance_band)

    # Create geographic clusters by combining sector and distance band
    user_df['geo_cluster'] = user_df['sector'].astype(str) + '_' + user_df['distance_band']

    print(f"   Created {user_df['geo_cluster'].nunique()} geographic clusters")
    for cluster in sorted(user_df['geo_cluster'].unique()):
        count = len(user_df[user_df['geo_cluster'] == cluster])
        sector = cluster.split('_')[0]
        band = cluster.split('_')[1]
        print(f"   Cluster {cluster}: {count} users (Sector {sector}, {band} distance)")

    return user_df

def assign_drivers_to_corridors(user_df, driver_df, office_lat, office_lon):
    """Assign drivers to geographic corridors/clusters (less punitive coherence weighting)."""
    print("🚗 Assigning drivers to geographic corridors...")

    driver_df = driver_df.copy()
    driver_df['office_bearing'] = driver_df.apply(
        lambda row: calculate_bearing(office_lat, office_lon, row['latitude'], row['longitude']),
        axis=1
    )
    driver_df['office_distance'] = driver_df.apply(
        lambda row: haversine_distance(row['latitude'], row['longitude'], office_lat, office_lon),
        axis=1
    )

    cluster_groups = user_df.groupby('geo_cluster')
    cluster_info = []
    for cluster_name, cluster_users in cluster_groups:
        center_lat = cluster_users['latitude'].mean()
        center_lon = cluster_users['longitude'].mean()
        cluster_bearing = calculate_bearing(office_lat, office_lon, center_lat, center_lon)
        cluster_distance = haversine_distance(office_lat, office_lon, center_lat, center_lon)
        cluster_info.append({
            'cluster_name': cluster_name,
            'user_count': len(cluster_users),
            'center_lat': center_lat,
            'center_lon': center_lon,
            'bearing': cluster_bearing,
            'distance': cluster_distance,
            'users': cluster_users
        })

    cluster_info.sort(key=lambda x: x['user_count'], reverse=True)

    assignments = []
    used_drivers = set()

    for cluster in cluster_info:
        if len(used_drivers) >= len(driver_df):
            break

        best_driver = None
        best_score = float('inf')

        # Precompute cluster user positions for a full coherence sample
        cluster_user_positions = [(row['latitude'], row['longitude']) for _, row in cluster['users'].iterrows()]

        for _, driver in driver_df.iterrows():
            if driver['driver_id'] in used_drivers:
                continue

            if ROAD_NETWORK is not None:
                try:
                    road_dist = ROAD_NETWORK.get_road_distance(
                        driver['latitude'], driver['longitude'],
                        cluster['center_lat'], cluster['center_lon']
                    )
                except Exception:
                    road_dist = haversine_distance(
                        driver['latitude'], driver['longitude'],
                        cluster['center_lat'], cluster['center_lon']
                    )

                # compute coherence on the whole cluster (safer than just first 3)
                coherence = ROAD_NETWORK.get_route_coherence_score(
                    (driver['latitude'], driver['longitude']),
                    cluster_user_positions,
                    (office_lat, office_lon)
                )
                # Reduce the punitive factor and make road distance the main term
                coherence_penalty_weight = 5.0
                score = road_dist + (1.0 - coherence) * coherence_penalty_weight + (driver['priority'] * 0.2)
            else:
                bearing_diff = angular_difference(driver['office_bearing'], cluster['bearing'])
                driver_to_center = haversine_distance(
                    driver['latitude'], driver['longitude'],
                    cluster['center_lat'], cluster['center_lon']
                )
                score = (bearing_diff / 45.0) * 2 + (driver_to_center / 5.0) + (driver['priority'] * 0.2)

            if score < best_score:
                best_score = score
                best_driver = driver

        if best_driver is not None:
            assignments.append({
                'driver': best_driver,
                'cluster': cluster,
                'score': best_score
            })
            used_drivers.add(best_driver['driver_id'])
            print(f"   Assigned driver {best_driver['driver_id']} to cluster {cluster['cluster_name']} ({cluster['user_count']} users)")

    return assignments

def build_optimized_routes(assignments, office_lat, office_lon):
    """Build routes within assigned corridors with incremental user validation."""
    print("🛤️ Building optimized routes within corridors...")

    routes = []
    max_deviation_angle = _config.get('max_deviation_angle', 35)
    max_route_deviation = _config.get('max_route_deviation_ratio', 1.4)
    max_fill_km = MAX_FILL_DISTANCE_KM

    for assignment in assignments:
        driver = assignment['driver']
        cluster = assignment['cluster']
        cluster_users = cluster['users']

        print(f"   Building route for driver {driver['driver_id']} in cluster {cluster['cluster_name']}")

        route = {
            'driver': driver.to_dict(),
            'driver_id': str(driver['driver_id']),
            'latitude': float(driver['latitude']),
            'longitude': float(driver['longitude']),
            'assigned_users': [],
            'vehicle_type': int(driver['capacity']),
            'utilization': 0,
            'priority_group': driver.get('priority', 2)
        }

        # Precompute candidate list and sort by a normalized score
        candidates = []
        for _, user in cluster_users.iterrows():
            if ROAD_NETWORK is not None:
                try:
                    dist = ROAD_NETWORK.get_road_distance(
                        driver['latitude'], driver['longitude'],
                        user['latitude'], user['longitude']
                    )
                except Exception:
                    dist = haversine_distance(
                        driver['latitude'], driver['longitude'],
                        user['latitude'], user['longitude']
                    )
            else:
                dist = haversine_distance(
                    driver['latitude'], driver['longitude'],
                    user['latitude'], user['longitude']
                )

            # bearing penalty relative to driver->office direction (small normalized)
            driver_to_office_bearing = calculate_bearing(driver['latitude'], driver['longitude'], office_lat, office_lon)
            user_bearing = calculate_bearing(driver['latitude'], driver['longitude'], user['latitude'], user['longitude'])
            bearing_penalty = angular_difference(user_bearing, driver_to_office_bearing) / 180.0
            score = dist + 2.0 * bearing_penalty
            candidates.append((score, user))

        candidates.sort(key=lambda x: x[0])

        # Greedily add users one-by-one validating against current assigned list
        for score, user in candidates:
            if len(route['assigned_users']) >= driver['capacity']:
                break

            candidate_pos = (float(user['latitude']), float(user['longitude']))
            driver_pos = (route['latitude'], route['longitude'])
            office_pos = (office_lat, office_lon)
            existing_users_positions = [(u['lat'], u['lng']) for u in route['assigned_users']]

            accept = False

            if ROAD_NETWORK is not None:
                try:
                    accept = ROAD_NETWORK.is_user_on_route_path(
                        driver_pos, existing_users_positions, candidate_pos, office_pos,
                        max_detour_ratio=_config.get('max_route_deviation_ratio', 1.4)
                    )
                except Exception:
                    accept = False

                # allow if near even if is_user_on_route_path rejected (soft fallback)
                if not accept:
                    d = haversine_distance(driver_pos[0], driver_pos[1], candidate_pos[0], candidate_pos[1])
                    if d <= max_fill_km:
                        accept = True
            else:
                # fallback bearing/distance logic using config max fill distance
                user_to_office_bearing = calculate_bearing(user['latitude'], user['longitude'], office_lat, office_lon)
                driver_to_office_bearing = calculate_bearing(driver['latitude'], driver['longitude'], office_lat, office_lon)
                bearing_diff = angular_difference(user_to_office_bearing, driver_to_office_bearing)
                driver_to_user_dist = calculate_distance(driver['latitude'], driver['longitude'], user['latitude'], user['longitude'])

                if bearing_diff <= max_deviation_angle or driver_to_user_dist <= max_fill_km:
                    accept = True

            if accept:
                # Validate full route coherence before committing
                temp_users = route['assigned_users'] + [{
                    'user_id': str(user['user_id']),
                    'lat': float(user['latitude']),
                    'lng': float(user['longitude'])
                }]

                if validate_route_coherence(route['latitude'], route['longitude'], temp_users, office_lat, office_lon):
                    route['assigned_users'].append({
                        'user_id': str(user['user_id']),
                        'lat': float(user['latitude']),
                        'lng': float(user['longitude']),
                        'office_distance': float(user.get('office_distance', 0)),
                        'first_name': str(user.get('first_name', '')),
                        'email': str(user.get('email', ''))
                    })

        route['utilization'] = len(route['assigned_users']) / (driver['capacity'] if driver['capacity'] > 0 else 1)

        if route['assigned_users']:
            route = optimize_pickup_sequence(route, office_lat, office_lon)
            routes.append(route)
            print(f"     ✅ Route created: {len(route['assigned_users'])} users, utilization: {route['utilization']:.1%}")
        else:
            print(f"     ❌ No valid users for driver {driver['driver_id']}")

    return routes

def validate_route_coherence(driver_lat, driver_lon, users, office_lat, office_lon):
    """Validate that a route maintains coherence and doesn't exceed deviation limits with relaxed margin"""
    if not users:
        return True

    # Handle edge case: driver at office
    direct_distance = haversine_distance(driver_lat, driver_lon, office_lat, office_lon)
    if direct_distance < 0.001:  # Driver essentially at office
        route_distance = sum(haversine_distance(users[i-1]['lat'] if i > 0 else driver_lat, 
                                               users[i-1]['lng'] if i > 0 else driver_lon,
                                               users[i]['lat'], users[i]['lng']) for i in range(len(users)))
        return route_distance <= _config.get('max_extra_distance_km', 2.0)

    # Use road network coherence if available
    if ROAD_NETWORK is not None:
        driver_pos = (driver_lat, driver_lon)
        user_positions = [(u['lat'], u['lng']) for u in users]
        office_pos = (office_lat, office_lon)
        coherence = ROAD_NETWORK.get_route_coherence_score(driver_pos, user_positions, office_pos)
        min_coherence = _config.get('min_road_coherence_score', 0.7)
        
        # Allow a small coherence margin for practical acceptance
        if coherence + 0.05 >= min_coherence:
            return True
            
        # Additional check: if coherence is close but route distance is reasonable
        route_distance = 0
        current_lat, current_lon = driver_lat, driver_lon
        for user in users:
            route_distance += haversine_distance(current_lat, current_lon, user['lat'], user['lng'])
            current_lat, current_lon = user['lat'], user['lng']
        route_distance += haversine_distance(current_lat, current_lon, office_lat, office_lon)
        
        deviation_ratio = route_distance / max(direct_distance, 0.1)
        if deviation_ratio <= _config.get('max_route_deviation_ratio', 1.4) * 1.1:  # 10% more lenient
            return True
            
        return coherence >= min_coherence
    
    # Fallback to distance-based validation
    max_deviation_ratio = _config.get('max_route_deviation_ratio', 1.4)
    
    # Calculate actual route distance
    route_distance = 0
    current_lat, current_lon = driver_lat, driver_lon

    for user in users:
        route_distance += haversine_distance(current_lat, current_lon, user['lat'], user['lng'])
        current_lat, current_lon = user['lat'], user['lng']

    # Add final leg to office
    route_distance += haversine_distance(current_lat, current_lon, office_lat, office_lon)

    # Check if route deviation is acceptable
    deviation_ratio = route_distance / direct_distance if direct_distance > 0 else 1
    return deviation_ratio <= max_deviation_ratio

def optimize_pickup_sequence(route, office_lat, office_lon):
    """Optimize pickup sequence for minimal total distance"""
    if len(route['assigned_users']) <= 1:
        return route

    users = route['assigned_users']
    driver_pos = (route['latitude'], route['longitude'])

    # Use nearest neighbor heuristic for TSP-like problem
    unvisited = users.copy()
    optimized_sequence = []
    current_pos = driver_pos

    while unvisited:
        nearest_user = None
        nearest_distance = float('inf')

        for user in unvisited:
            distance = haversine_distance(current_pos[0], current_pos[1], user['lat'], user['lng'])

            # Proportional progress bonus towards office
            current_to_office = haversine_distance(current_pos[0], current_pos[1], office_lat, office_lon)
            user_to_office = haversine_distance(user['lat'], user['lng'], office_lat, office_lon)
            
            progress_fraction = max(0, (current_to_office - user_to_office) / max(current_to_office, 1e-6))
            office_direction_bonus = -progress_fraction * 0.5  # Max -0.5 km equivalent

            adjusted_distance = distance + office_direction_bonus

            if adjusted_distance < nearest_distance:
                nearest_distance = adjusted_distance
                nearest_user = user

        if nearest_user:
            optimized_sequence.append(nearest_user)
            unvisited.remove(nearest_user)
            current_pos = (nearest_user['lat'], nearest_user['lng'])

    route['assigned_users'] = optimized_sequence
    return route

def handle_unassigned_users(user_df, routes, driver_df, office_lat, office_lon):
    """Handle remaining unassigned users with relaxed constraints and create new routes with idle drivers."""
    print("🔄 Handling remaining unassigned users...")

    assigned_user_ids = set()
    for route in routes:
        for user in route['assigned_users']:
            assigned_user_ids.add(user['user_id'])

    unassigned_users = []
    for _, user in user_df.iterrows():
        if user['user_id'] not in assigned_user_ids:
            unassigned_users.append({
                'user_id': str(user['user_id']),
                'lat': float(user['latitude']),
                'lng': float(user['longitude']),
                'office_distance': float(user.get('office_distance', 0)),
                'first_name': str(user.get('first_name', '')),
                'email': str(user.get('email', ''))
            })

    print(f"   Processing {len(unassigned_users)} unassigned users...")

    # Collect idle drivers (not used in routes)
    used_driver_ids = {route['driver']['driver_id'] for route in routes}
    idle_drivers = [d for _, d in driver_df.iterrows() if str(d['driver_id']) not in used_driver_ids]

    remaining_unassigned = []

    # Try simple greedy assignment: for each unassigned user, try to place in existing route; otherwise use an idle driver to create a new route
    for user in sorted(unassigned_users, key=lambda x: x['office_distance']):  # prioritize nearer-to-office users
        best_route = None
        best_score = float('inf')

        # try inserting into existing routes first
        for route in routes:
            if len(route['assigned_users']) >= route['driver']['capacity']:
                continue
            temp_users = route['assigned_users'] + [user]
            if validate_route_coherence(route['latitude'], route['longitude'], temp_users, office_lat, office_lon):
                insertion_cost = calculate_insertion_cost_simple(route, user)
                if insertion_cost < best_score:
                    best_score = insertion_cost
                    best_route = route

        if best_route:
            best_route['assigned_users'].append(user)
            best_route['utilization'] = len(best_route['assigned_users']) / best_route['driver']['capacity']
            print(f"     ✅ Added user {user['user_id']} to route {best_route['driver_id']}")
            continue

        # if not assigned to an existing route, try to create a new (relaxed) route with an idle driver
        if idle_drivers:
            driver_row = idle_drivers.pop(0)
            new_route = {
                'driver': driver_row.to_dict(),
                'driver_id': str(driver_row['driver_id']),
                'latitude': float(driver_row['latitude']),
                'longitude': float(driver_row['longitude']),
                'assigned_users': [],
                'vehicle_type': int(driver_row['capacity']),
                'utilization': 0,
                'priority_group': driver_row.get('priority', 2)
            }

            # Add this user as the seed; then try to greedily fill from nearby unassigned users
            new_route['assigned_users'].append(user)
            # try to fill more from remaining unassigned users up to capacity
            others = [u for u in unassigned_users if u['user_id'] != user['user_id']]
            for candidate in sorted(others, key=lambda x: haversine_distance(user['lat'], user['lng'], x['lat'], x['lng'])):
                if len(new_route['assigned_users']) >= driver_row['capacity']:
                    break
                temp_users = new_route['assigned_users'] + [candidate]
                if validate_route_coherence(new_route['latitude'], new_route['longitude'], temp_users, office_lat, office_lon):
                    new_route['assigned_users'].append(candidate)
            new_route['utilization'] = len(new_route['assigned_users']) / (driver_row['capacity'] if driver_row['capacity']>0 else 1)
            routes.append(new_route)
            print(f"     🆕 Created relaxed route {new_route['driver_id']} with {len(new_route['assigned_users'])} users")
            # remove assigned users from the unassigned pool
            assigned_ids = {u['user_id'] for u in new_route['assigned_users']}
            unassigned_users = [u for u in unassigned_users if u['user_id'] not in assigned_ids]
        else:
            # no idle drivers left and couldn't insert into existing routes
            remaining_unassigned.append(user)

    print(f"   ⚠️ {len(remaining_unassigned)} users remain unassigned after relaxed assignment")
    return remaining_unassigned

def calculate_insertion_cost_simple(route, new_user):
    """Calculate simple insertion cost for user assignment"""
    if not route['assigned_users']:
        return haversine_distance(
            route['latitude'], route['longitude'],
            new_user['lat'], new_user['lng']
        )

    # Find cheapest insertion position
    min_cost = float('inf')

    for i in range(len(route['assigned_users']) + 1):
        cost = 0

        # Cost from previous position to new user
        if i == 0:
            prev_lat, prev_lon = route['latitude'], route['longitude']
        else:
            prev_user = route['assigned_users'][i-1]
            prev_lat, prev_lon = prev_user['lat'], prev_user['lng']

        cost += haversine_distance(prev_lat, prev_lon, new_user['lat'], new_user['lng'])

        # Cost from new user to next position
        if i < len(route['assigned_users']):
            next_user = route['assigned_users'][i]
            cost += haversine_distance(new_user['lat'], new_user['lng'], next_user['lat'], next_user['lng'])

            # Subtract original cost between prev and next
            if i == 0:
                cost -= haversine_distance(route['latitude'], route['longitude'], next_user['lat'], next_user['lng'])
            else:
                cost -= haversine_distance(prev_lat, prev_lon, next_user['lat'], next_user['lng'])

        min_cost = min(min_cost, cost)

    return min_cost

def build_final_results(routes, unassigned_users, driver_df, execution_time, parameter, string_param):
    """Build the final result structure"""
    assigned_driver_ids = {route['driver']['driver_id'] for route in routes}
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

    return {
        "status": "true",
        "execution_time": execution_time,
        "data": routes,
        "unassignedUsers": unassigned_users,
        "unassignedDrivers": unassigned_drivers,
        "clustering_analysis": {
            "method": "Geographic Multi-Stage Optimization",
            "clusters": len(routes)
        },
        "parameter": parameter,
        "string_param": string_param
    }

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

def run_road_coherent_assignment(source_id: str, parameter: int = 1, string_param: str = ""):
    """Multi-stage geographic assignment with strict route validation"""
    start_time = time.time()

    try:
        print(f"🚀 Starting geographic multi-stage assignment for source_id: {source_id}")
        print(f"📋 Parameter: {parameter}, String parameter: {string_param}")

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
                "clustering_analysis": {"method": "No Users", "clusters": 0},
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
                "clustering_analysis": {"method": "No Drivers", "clusters": 0},
                "parameter": parameter,
                "string_param": string_param
            }

        office_lat, office_lon = extract_office_coordinates(data)
        validate_input_data(data)

        user_df, driver_df = prepare_user_driver_dataframes(data)

        print(f"📊 Data prepared - Users: {len(user_df)}, Drivers: {len(driver_df)}")

        # Stage 1: Geographic clustering
        user_df = create_geographic_clusters(user_df, office_lat, office_lon)

        # Stage 2: Driver-to-corridor assignment
        assignments = assign_drivers_to_corridors(user_df, driver_df, office_lat, office_lon)

        # Stage 3: Route building within corridors
        routes = build_optimized_routes(assignments, office_lat, office_lon)

        # Stage 4: Handle remaining users
        unassigned_users = handle_unassigned_users(user_df, routes, driver_df, office_lat, office_lon)

        # Build final results
        execution_time = time.time() - start_time
        return build_final_results(routes, unassigned_users, driver_df, execution_time, parameter, string_param)

    except Exception as e:
        logger.error(f"Assignment failed: {e}", exc_info=True)
        return {"status": "false", "details": str(e), "data": []}

# Keep the old function name for compatibility
def run_assignment(source_id: str, parameter: int = 1, string_param: str = ""):
    """Legacy function - redirects to road coherent assignment"""
    return run_road_coherent_assignment(source_id, parameter, string_param)

def analyze_assignment_quality(result):
    """Analyze the quality of the assignment with enhanced metrics"""
    if result["status"] != "true":
        return "Assignment failed"

    total_routes = len(result["data"])
    total_assigned = sum(len(route["assigned_users"]) for route in result["data"])
    total_unassigned = len(result["unassignedUsers"])

    utilizations = []
    total_capacity = 0
    for route in result["data"]:
        total_capacity += route["driver"]["capacity"]
        if route["assigned_users"]:
            util = len(route["assigned_users"]) / route["driver"]["capacity"]
            utilizations.append(util)

    utilization_pct = (total_assigned/total_capacity*100) if total_capacity > 0 else 0
    print(f"   📊 Overall Capacity Utilization: {utilization_pct:.1f}%")

    analysis = {
        "total_routes": total_routes,
        "total_assigned_users": total_assigned,
        "total_unassigned_users": total_unassigned,
        "assignment_rate": round(total_assigned / (total_assigned + total_unassigned) * 100, 1) if (total_assigned + total_unassigned) > 0 else 0,
        "avg_utilization": round(np.mean(utilizations) * 100, 1) if utilizations else 0,
        "min_utilization": round(np.min(utilizations) * 100, 1) if utilizations else 0,
        "max_utilization": round(np.max(utilizations) * 100, 1) if utilizations else 0,
        "routes_below_80_percent": sum(1 for u in utilizations if u < 0.8),
        "clustering_method": result.get("clustering_analysis", {}).get("method", "Unknown")
    }

    return analysis