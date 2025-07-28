import os
import math
import json
import requests
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
from dotenv import load_dotenv

# Load thresholds from config
with open('config.json') as f:
    cfg = json.load(f)
MAX_FILL_DISTANCE_KM = cfg.get("max_fill_distance_km", 5.0)
MERGE_DISTANCE_KM = cfg.get("merge_distance_km", 3.0)
MIN_UTIL_THRESHOLD = cfg.get("min_util_threshold", 0.5)


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


def calculate_bearings_and_features(user_df, office_lat=30.6810489, office_lon=76.7260711):
    """Calculate bearings and add geographical features"""
    user_df[['office_latitude','office_longitude']] = office_lat, office_lon

    def calculate_bearing(lat1, lon1, lat2, lon2):
        lat1, lat2 = np.radians(lat1), np.radians(lat2)
        dlon = np.radians(lon2 - lon1)
        x = np.sin(dlon) * np.cos(lat2)
        y = np.cos(lat1)*np.sin(lat2) - np.sin(lat1)*np.cos(lat2)*np.cos(dlon)
        return (np.degrees(np.arctan2(x,y)) + 360) % 360

    user_df['bearing'] = calculate_bearing(
        user_df['latitude'],user_df['longitude'],office_lat,office_lon
    )
    user_df['bearing_sin'] = np.sin(np.radians(user_df['bearing']))
    user_df['bearing_cos'] = np.cos(np.radians(user_df['bearing']))

    return user_df


def calculate_bearing(lat1, lon1, lat2, lon2):
    """Calculate bearing from point A to B in degrees"""
    lat1, lat2 = map(math.radians, [lat1, lat2])
    dlon = math.radians(lon2 - lon1)
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    return (math.degrees(math.atan2(x, y)) + 360) % 360

def calculate_route_bearing(route, office_lat=30.6810489, office_lon=76.7260711):
    """Assign a representative bearing to the route"""
    if not route['assigned_users']:
        return 0
    avg_lat = sum(u['lat'] for u in route['assigned_users']) / len(route['assigned_users'])
    avg_lng = sum(u['lng'] for u in route['assigned_users']) / len(route['assigned_users'])
    return calculate_bearing(avg_lat, avg_lng, office_lat, office_lon)

def bearing_difference(b1, b2):
    diff = abs(b1 - b2) % 360
    return min(diff, 360 - diff)


def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate haversine distance between two points in kilometers"""
    R = 6371.0  # Earth radius in kilometers
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c


def improved_geographical_clustering(user_df):
    """
    Improved clustering using DBSCAN for natural geographical grouping
    """
    coords = user_df[['latitude', 'longitude']].values
    
    # Use DBSCAN for density-based clustering
    coords_rad = np.radians(coords)
    eps_km = 3.0  # 3km radius for clustering
    eps_rad = eps_km / 6371.0  # Convert to radians
    
    clustering = DBSCAN(eps=eps_rad, min_samples=2, metric='haversine').fit(coords_rad)
    user_df['geo_cluster'] = clustering.labels_
    
    # Handle noise points (label -1) by assigning to nearest cluster
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
            # If no valid clusters, create single cluster
            user_df.loc[noise_mask, 'geo_cluster'] = 0
    
    # If DBSCAN didn't find any clusters, fall back to K-means
    if user_df['geo_cluster'].nunique() <= 1:
        coords_features = user_df[['latitude', 'longitude', 'bearing_sin', 'bearing_cos']].values
        scaled_features = StandardScaler().fit_transform(coords_features)
        n_clusters = min(5, len(user_df))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        user_df['geo_cluster'] = kmeans.fit_predict(scaled_features)
    
    return user_df


def optimize_driver_assignment(user_df, driver_df):
    """
    Optimized driver assignment considering both distance and capacity utilization
    """
    drivers_and_routes = []
    available_drivers = driver_df.sort_values('capacity', ascending=False).to_dict('records')
    assigned_user_ids = set()
    
    # Process each geographical cluster
    for cluster_id, cluster_users in user_df.groupby('geo_cluster'):
        unassigned_in_cluster = cluster_users[~cluster_users['user_id'].isin(assigned_user_ids)].copy()
        
        if unassigned_in_cluster.empty or not available_drivers:
            continue
        
        # Calculate optimal sub-clustering
        total_users = len(unassigned_in_cluster)
        if not available_drivers:
            continue
            
        # Determine number of sub-clusters needed
        avg_capacity = np.mean([d['capacity'] for d in available_drivers])
        max_capacity = max([d['capacity'] for d in available_drivers])
        
        if total_users <= max_capacity:
            # Single route can handle all users
            sub_clusters = 1
        else:
            sub_clusters = min(
                max(1, round(total_users / avg_capacity)),
                len(available_drivers),
                total_users
            )
        
        # Create sub-clusters if needed
        if sub_clusters > 1:
            coords = unassigned_in_cluster[['latitude', 'longitude']].values
            kmeans = KMeans(n_clusters=sub_clusters, random_state=42)
            unassigned_in_cluster['sub_cluster'] = kmeans.fit_predict(coords)
        else:
            unassigned_in_cluster['sub_cluster'] = 0
        
        # Assign drivers to sub-clusters optimally
        for sub_id, sub_users in unassigned_in_cluster.groupby('sub_cluster'):
            if sub_users.empty or not available_drivers:
                continue
            
            sub_centroid = sub_users[['latitude', 'longitude']].mean().values
            users_needed = len(sub_users)
            
            # Find best driver using cost function
            best_driver = None
            min_cost = float('inf')
            
            for driver in available_drivers:
                driver_pos = (driver['latitude'], driver['longitude'])
                distance = haversine_distance(
                    driver_pos[0], driver_pos[1], 
                    sub_centroid[0], sub_centroid[1]
                )
                
                # Cost function: distance + utilization penalty
                if users_needed <= driver['capacity']:
                    utilization = users_needed / driver['capacity']
                    # Prefer high utilization, penalize low utilization
                    utilization_penalty = (1 - utilization) * 3  # 3km penalty per unused seat
                    cost = distance + utilization_penalty
                else:
                    # Penalize if driver can't fit all users
                    cost = distance + 10  # 10km penalty for overflow
                
                if cost < min_cost:
                    min_cost = cost
                    best_driver = driver
            
            if not best_driver:
                continue
            
            # Create route
            capacity = best_driver['capacity']
            users_to_assign = sub_users.head(capacity)  # Take only what fits
            
            route = {
                'driver_id': str(best_driver['driver_id']),
                'vehicle_id': str(best_driver.get('vehicle_id', '')),
                'vehicle_type': capacity,
                'latitude': float(best_driver['latitude']),
                'longitude': float(best_driver['longitude']),
                'assigned_users': []
            }
            
            # Add users to route
            for _, user in users_to_assign.iterrows():
                user_data = {
                    'user_id': str(user['user_id']),
                    'lat': float(user['latitude']),
                    'lng': float(user['longitude']),
                    'office_distance': float(user.get('office_distance', 0))
                }
                
                # Add optional fields if they exist
                if pd.notna(user.get('first_name')):
                    user_data['first_name'] = str(user['first_name'])
                if pd.notna(user.get('email')):
                    user_data['email'] = str(user['email'])
                
                route['assigned_users'].append(user_data)
                assigned_user_ids.add(user['user_id'])
            
            # Calculate route metrics
            if route['assigned_users']:
                lats = [u['lat'] for u in route['assigned_users']]
                lngs = [u['lng'] for u in route['assigned_users']]
                route['centroid'] = [np.mean(lats), np.mean(lngs)]
                route['util'] = len(route['assigned_users']) / route['vehicle_type']
                route['stops'] = [[u['lat'], u['lng']] for u in route['assigned_users']]
                route['utilization'] = route['util']
            
            drivers_and_routes.append(route)
            available_drivers.remove(best_driver)
    
    return drivers_and_routes, assigned_user_ids


def fill_underutilized_routes(drivers_and_routes, user_df, assigned_user_ids):
    """
    Fill underutilized routes with nearby unassigned users
    """
    unassigned_users = user_df[~user_df['user_id'].isin(assigned_user_ids)].copy()
    
    for route in drivers_and_routes:
        capacity = route['vehicle_type']
        current_size = len(route['assigned_users'])
        
        if current_size < capacity and not unassigned_users.empty:
            # Calculate route centroid
            if route['assigned_users']:
                centroid_lat = np.mean([u['lat'] for u in route['assigned_users']])
                centroid_lng = np.mean([u['lng'] for u in route['assigned_users']])
            else:
                centroid_lat, centroid_lng = route['latitude'], route['longitude']
            
            # Find nearby unassigned users
            distances = []
            for _, user in unassigned_users.iterrows():
                dist = haversine_distance(
                    centroid_lat, centroid_lng,
                    user['latitude'], user['longitude']
                )
                if dist <= MAX_FILL_DISTANCE_KM:
                    distances.append((user, dist))
            
            # Add closest users up to capacity
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


def swap_optimization(drivers_and_routes):
    """
    Optimize routes by swapping users between routes
    """
    improved = True
    iterations = 0
    max_iterations = 5
    
    while improved and iterations < max_iterations:
        improved = False
        iterations += 1
        
        for i, route_a in enumerate(drivers_and_routes):
            for j, route_b in enumerate(drivers_and_routes[i+1:], start=i+1):
                # Calculate current route centroids
                if route_a['assigned_users']:
                    centroid_a = (
                        np.mean([u['lat'] for u in route_a['assigned_users']]),
                        np.mean([u['lng'] for u in route_a['assigned_users']])
                    )
                else:
                    centroid_a = (route_a['latitude'], route_a['longitude'])
                
                if route_b['assigned_users']:
                    centroid_b = (
                        np.mean([u['lat'] for u in route_b['assigned_users']]),
                        np.mean([u['lng'] for u in route_b['assigned_users']])
                    )
                else:
                    centroid_b = (route_b['latitude'], route_b['longitude'])
                
                # Try swapping users
                for user_a in route_a['assigned_users'][:]:  # Copy to avoid modification during iteration
                    for user_b in route_b['assigned_users'][:]:
                        # Calculate current distances
                        dist_a_current = haversine_distance(
                            centroid_a[0], centroid_a[1], user_a['lat'], user_a['lng']
                        )
                        dist_b_current = haversine_distance(
                            centroid_b[0], centroid_b[1], user_b['lat'], user_b['lng']
                        )
                        
                        # Calculate distances after swap
                        dist_a_swap = haversine_distance(
                            centroid_a[0], centroid_a[1], user_b['lat'], user_b['lng']
                        )
                        dist_b_swap = haversine_distance(
                            centroid_b[0], centroid_b[1], user_a['lat'], user_a['lng']
                        )
                        
                        # If swap improves total distance by at least 200m
                        if dist_a_swap + dist_b_swap + 0.2 < dist_a_current + dist_b_current:
                            # Perform swap
                            route_a['assigned_users'].remove(user_a)
                            route_b['assigned_users'].remove(user_b)
                            route_a['assigned_users'].append(user_b)
                            route_b['assigned_users'].append(user_a)
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break


def merge_underutilized_routes(drivers_and_routes):
    """
    Merge underutilized routes with nearby routes if capacity and distance allow
    """
    merged_routes = []
    used_indices = set()

    for i, route_a in enumerate(drivers_and_routes):
        if i in used_indices:
            continue

        if route_a['utilization'] >= MIN_UTIL_THRESHOLD:
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
                # Merge A into B
                route_b['assigned_users'].extend(route_a['assigned_users'])
                lats = [u['lat'] for u in route_b['assigned_users']]
                lngs = [u['lng'] for u in route_b['assigned_users']]
                route_b['centroid'] = [np.mean(lats), np.mean(lngs)]
                route_b['util'] = len(route_b['assigned_users']) / route_b['vehicle_type']
                route_b['utilization'] = route_b['util']
                route_b['stops'] = [[u['lat'], u['lng']] for u in route_b['assigned_users']]
                used_indices.add(i)
                merged = True
                break

        if not merged:
            merged_routes.append(route_a)

    return merged_routes


def handle_remaining_users(user_df, drivers_and_routes, assigned_user_ids, driver_df):
    """
    Handle remaining unassigned users with additional clustering
    """
    unassigned_users = user_df[~user_df['user_id'].isin(assigned_user_ids)].copy()
    available_drivers = driver_df[~driver_df['driver_id'].isin(
        [route['driver_id'] for route in drivers_and_routes]
    )].copy()
    
    if unassigned_users.empty:
        return []
    
    # Try to fit unassigned users into existing routes first
    for _, user in unassigned_users.iterrows():
        best_route = None
        min_distance = float('inf')
        
        for route in drivers_and_routes:
            if len(route['assigned_users']) < route['vehicle_type']:
                # Calculate distance to route centroid
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
    
    # Create new routes for remaining users if available drivers exist
    remaining_unassigned = user_df[~user_df['user_id'].isin(assigned_user_ids)].copy()
    
    if not remaining_unassigned.empty and not available_drivers.empty:
        # Cluster remaining users
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
            
            # Find closest driver
            closest_driver = min(
                available_drivers_list,
                key=lambda d: haversine_distance(
                    centroid[0], centroid[1], d['latitude'], d['longitude']
                )
            )
            
            # Create new route
            capacity = closest_driver['capacity']
            users_to_assign = cluster_users.head(capacity)
            
            route = {
                'driver_id': str(closest_driver['driver_id']),
                'vehicle_id': str(closest_driver.get('vehicle_id', '')),
                'vehicle_type': capacity,
                'latitude': float(closest_driver['latitude']),
                'longitude': float(closest_driver['longitude']),
                'assigned_users': []
            }
            
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
            
            # Add route metrics
            if route['assigned_users']:
                lats = [u['lat'] for u in route['assigned_users']]
                lngs = [u['lng'] for u in route['assigned_users']]
                route['centroid'] = [np.mean(lats), np.mean(lngs)]
                route['util'] = len(route['assigned_users']) / route['vehicle_type']
                route['stops'] = [[u['lat'], u['lng']] for u in route['assigned_users']]
                route['utilization'] = route['util']
            
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

def reassign_underutilized_users_to_nearby_routes(drivers_and_routes, unassigned_drivers):
    reassigned_users = []
    routes_to_remove = set()

    # STEP 0: Merge underfilled nearby routes like 2534 & 2577
    i = 0
    while i < len(drivers_and_routes):
        route_a = drivers_and_routes[i]
        if route_a['utilization'] >= 1.0:
            i += 1
            continue

        j = i + 1
        while j < len(drivers_and_routes):
            route_b = drivers_and_routes[j]
            if route_b['utilization'] >= 1.0:
                j += 1
                continue

            combined_users = route_a['assigned_users'] + route_b['assigned_users']
            if len(combined_users) <= max(route_a['vehicle_type'], route_b['vehicle_type']):
                centroid_a = route_a.get('centroid', [route_a['latitude'], route_a['longitude']])
                centroid_b = route_b.get('centroid', [route_b['latitude'], route_b['longitude']])
                dist = haversine_distance(centroid_a[0], centroid_a[1], centroid_b[0], centroid_b[1])

                bearing_a = calculate_route_bearing(route_a)
                bearing_b = calculate_route_bearing(route_b)
                bearing_diff = bearing_difference(bearing_a, bearing_b)

                if dist <= MERGE_DISTANCE_KM and bearing_diff <= 45:
                    route_a['assigned_users'] = combined_users
                    route_a['vehicle_type'] = max(route_a['vehicle_type'], route_b['vehicle_type'])
                    drivers_and_routes.pop(j)
                    continue
            j += 1
        i += 1

    # STEP 1: Fallback reassignment for <50% utilization, ≤3 users
    low_util_routes = [r for r in drivers_and_routes if r.get('utilization', 0) < 0.5 and len(r['assigned_users']) <= 3]
    combined_users = []
    for route in low_util_routes:
        combined_users.extend(route['assigned_users'])
        idx = drivers_and_routes.index(route)
        routes_to_remove.add(idx)

    if 2 <= len(combined_users) <= 7:
        fallback_driver = None
        for i, drv in enumerate(unassigned_drivers):
            if drv['capacity'] >= len(combined_users):
                fallback_driver = unassigned_drivers.pop(i)
                break

        if fallback_driver:
            new_route = {
                'driver_id': fallback_driver['driver_id'],
                'vehicle_id': fallback_driver['vehicle_id'],
                'vehicle_type': fallback_driver['capacity'],
                'latitude': fallback_driver['latitude'],
                'longitude': fallback_driver['longitude'],
                'assigned_users': combined_users
            }
            drivers_and_routes.append(new_route)

    # STEP 2: Reassign individuals from low-util routes
    for i, source_route in enumerate(drivers_and_routes):
        if i in routes_to_remove or source_route.get('utilization', 0) >= MIN_UTIL_THRESHOLD:
            continue

        source_bearing = calculate_route_bearing(source_route)
        remaining_users = []

        for user in source_route['assigned_users']:
            best_target = None
            best_cost = float('inf')

            for j, target_route in enumerate(drivers_and_routes):
                if j == i or len(target_route['assigned_users']) >= target_route['vehicle_type']:
                    continue

                centroid = target_route.get('centroid', [target_route['latitude'], target_route['longitude']])
                distance = haversine_distance(centroid[0], centroid[1], user['lat'], user['lng'])
                target_bearing = calculate_route_bearing(target_route)
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
    drivers_and_routes = finalize_routes(drivers_and_routes)
    return drivers_and_routes

def finalize_routes(drivers_and_routes):
    for route in drivers_and_routes:
        if route['assigned_users']:
            lats = [u['lat'] for u in route['assigned_users']]
            lngs = [u['lng'] for u in route['assigned_users']]
            route['centroid'] = [np.mean(lats), np.mean(lngs)]
            route['util'] = len(route['assigned_users']) / route['vehicle_type']
            route['utilization'] = route['util']
            route['stops'] = [[u['lat'], u['lng']] for u in route['assigned_users']]
            route['bearing'] = calculate_bearing(route['centroid'][0], route['centroid'][1], 30.6810489, 76.7260711)
        else:
            route['centroid'] = [route['latitude'], route['longitude']]
            route['util'] = 0
            route['utilization'] = 0
            route['stops'] = []
            route['bearing'] = 0
    return drivers_and_routes


def run_assignment(source_id: str):
    """
    Main function to run the complete optimized assignment
    """
    try:
        # Load data
        data = load_env_and_fetch_data(source_id)
        user_df, driver_df = prepare_user_driver_dataframes(data)
        user_df = calculate_bearings_and_features(user_df)
        
        # Improved clustering
        user_df = improved_geographical_clustering(user_df)
        
        # Optimized assignment
        drivers_and_routes, assigned_user_ids = optimize_driver_assignment(user_df, driver_df)
        
        # Fill underutilized routes
        assigned_user_ids = fill_underutilized_routes(drivers_and_routes, user_df, assigned_user_ids)
        
        # Swap optimization
        swap_optimization(drivers_and_routes)
        
        # Handle remaining users
        unassigned_users = handle_remaining_users(user_df, drivers_and_routes, assigned_user_ids, driver_df)
        
        # Finalize routes
        drivers_and_routes = finalize_routes(drivers_and_routes)

        # 🔁 Merge small clusters where possible
        drivers_and_routes = merge_underutilized_routes(drivers_and_routes)

        # ✅ Build unassigned drivers list BEFORE using it
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

        # 🔄 Now safe to call with required argument
        drivers_and_routes = reassign_underutilized_users_to_nearby_routes(drivers_and_routes, unassigned_drivers)
        
        return {
            "status": "true",
            "data": drivers_and_routes,
            "unassignedUsers": unassigned_users,
            "unassignedDrivers": unassigned_drivers
        }
        
    except requests.exceptions.RequestException as req_err:
        return {"status": "false", "details": str(req_err), "data": []}
    except Exception as e:
        return {"status": "false", "details": str(e), "data": []}


def analyze_assignment_quality(result):
    """
    Analyze the quality of the assignment
    """
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
                if dist > 8:  # Flag distances > 8km
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
        "distance_issues": distance_issues
    }
    
    return analysis