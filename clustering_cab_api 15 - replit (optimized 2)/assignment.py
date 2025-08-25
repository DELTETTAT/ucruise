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
import pickle
import hashlib

# Try to import OSMnx for road snapping, fallback to PCA if not available
try:
    import osmnx as ox
    import geopandas as gpd
    from shapely.geometry import Point
    HAS_OSMNX = True

    # Configure OSMnx settings (compatible with v2.0+)
    try:
        # For OSMnx v2.0+
        ox.settings.use_cache = True
        ox.settings.log_console = False
        ox.settings.log_file = False
    except AttributeError:
        try:
            # For OSMnx v1.x
            ox.config(use_cache=True, log_console=False, log_file=False)
        except Exception:
            # If all fails, continue without configuration
            pass

    print("✅ OSMnx successfully loaded and configured")

except ImportError:
    HAS_OSMNX = False
    print("⚠️ OSMnx not available, using PCA fallback for road approximation")

# Global graph cache and spatial index cache
_GRAPH_CACHE = {}
_EDGES_CACHE = {}
_SNAP_CACHE = {}

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

    # Road-snapping specific config
    config = {}
    config['SNAP_RADIUS'] = max(10, min(100, float(cfg.get("snap_radius_m", 30))))  # meters
    config['MAX_GAP_DISTANCE'] = max(50, float(cfg.get("max_gap_distance_m", 200)))  # meters
    config['BEARING_TOLERANCE'] = max(5, min(30, float(cfg.get("bearing_tolerance", 15))))  # degrees
    config['MAX_DETOUR_KM'] = max(0.5, float(cfg.get("max_detour_km", 1.5)))  # km

    # Office coordinates
    office_lat = float(os.getenv("OFFICE_LAT", cfg.get("office_latitude", 30.6810489)))
    office_lon = float(os.getenv("OFFICE_LON", cfg.get("office_longitude", 76.7260711)))

    if not (-90 <= office_lat <= 90):
        office_lat = 30.6810489
    if not (-180 <= office_lon <= 180):
        office_lon = 76.7260711

    config['OFFICE_LAT'] = office_lat
    config['OFFICE_LON'] = office_lon

    return config


# Load validated configuration
_config = load_and_validate_config()
SNAP_RADIUS = _config['SNAP_RADIUS']
MAX_GAP_DISTANCE = _config['MAX_GAP_DISTANCE']
BEARING_TOLERANCE = _config['BEARING_TOLERANCE']
MAX_DETOUR_KM = _config['MAX_DETOUR_KM']
OFFICE_LAT = _config['OFFICE_LAT']
OFFICE_LON = _config['OFFICE_LON']


def validate_input_data(data):
    """Basic data validation"""
    if not isinstance(data, dict):
        raise ValueError("API response must be a dictionary")

    users = data.get("users", [])
    if not users:
        raise ValueError("No users found in API response")

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
        raise ValueError("No drivers found in API response")

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
        raise ValueError("API returned empty response body")

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

    return data


def extract_office_coordinates(data):
    """Extract dynamic office coordinates from API data"""
    company_data = data.get("company", {})
    office_lat = float(company_data.get("latitude", OFFICE_LAT))
    office_lon = float(company_data.get("longitude", OFFICE_LON))
    return office_lat, office_lon


def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate the great circle distance between two points on the earth"""
    from math import radians, cos, sin, asin, sqrt

    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371  # Radius of earth in kilometers

    return c * r


def get_osm_graph_with_cache(north, south, east, west):
    """Get OSM graph with persistent caching"""
    bbox_str = f"{north:.4f}_{south:.4f}_{east:.4f}_{west:.4f}"
    cache_file = f"graph_{bbox_str}.graphml"

    if bbox_str in _GRAPH_CACHE:
        return _GRAPH_CACHE[bbox_str], _EDGES_CACHE[bbox_str]

    # Try to load from disk cache
    if os.path.exists(cache_file):
        try:
            print(f"   📁 Loading cached graph from {cache_file}")
            G = ox.load_graphml(cache_file)
            # Convert to GeoDataFrame and create spatial index
            edges = ox.graph_to_gdfs(G, nodes=False, edges=True)
            edges = edges.to_crs(epsg=3857)  # Project to metric CRS

            _GRAPH_CACHE[bbox_str] = G
            _EDGES_CACHE[bbox_str] = edges
            print(f"   ✅ Loaded {len(edges)} edges from cache with spatial index")
            return G, edges
        except Exception as e:
            print(f"   ⚠️ Cache load failed: {e}, downloading fresh")

    # Download new graph
    try:
        print(f"   🌐 Downloading new OSM graph for bbox: ({north}, {south}, {east}, {west})")
        # Fixed: OSMnx v2.0+ requires bbox as separate parameters
        G = ox.graph_from_bbox(north=north, south=south, east=east, west=west, network_type='drive')

        # Save to cache
        ox.save_graphml(G, cache_file)

        # Convert to GeoDataFrame and create spatial index
        edges = ox.graph_to_gdfs(G, nodes=False, edges=True)
        edges = edges.to_crs(epsg=3857)  # Project to metric CRS

        _GRAPH_CACHE[bbox_str] = G
        _EDGES_CACHE[bbox_str] = edges
        print(f"   ✅ Downloaded and cached {len(edges)} edges with spatial index")
        return G, edges

    except Exception as e:
        raise Exception(f"Failed to download OSM graph: {str(e)}")


def calculate_bearing(lat1, lon1, lat2, lon2):
    """Calculate bearing between two points"""
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    dlon = lon2 - lon1
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)

    bearing = math.atan2(x, y)
    bearing = math.degrees(bearing)
    bearing = (bearing + 360) % 360

    return bearing


def bearing_difference(b1, b2):
    """Compute minimum difference between two bearings"""
    diff = abs(b1 - b2) % 360
    return min(diff, 360 - diff)


def snap_users_to_roads_optimized(user_df, edges_gdf):
    """Optimized snapping using spatial index and metric CRS"""
    start_time = time.time()

    # Convert user points to metric CRS GeoDataFrame
    user_gdf = gpd.GeoDataFrame(
        user_df.copy(),
        geometry=gpd.points_from_xy(user_df.longitude, user_df.latitude),
        crs='EPSG:4326'
    ).to_crs('EPSG:3857')  # Project to Web Mercator (meters)

    # Build spatial index for edges
    sindex = edges_gdf.sindex

    snapped_users = []
    cache_hits = 0

    for idx, user in user_gdf.iterrows():
        user_point = user.geometry

        # Check cache first (for users who don't move much)
        cache_key = (round(user.latitude, 6), round(user.longitude, 6))
        if cache_key in _SNAP_CACHE:
            cached_result = _SNAP_CACHE[cache_key].copy()
            cached_result.update({
                'user_id': user['user_id'],
                'latitude': user['latitude'],
                'longitude': user['longitude'],
                'first_name': user['first_name'],
                'email': user['email'],
                'office_distance': user['office_distance']
            })
            snapped_users.append(cached_result)
            cache_hits += 1
            continue

        # Use spatial index to get nearest candidates (much faster than full scan)
        possible_matches = list(sindex.nearest(user_point.bounds, 5))  # Get 5 nearest candidates

        if not possible_matches:
            # Fallback: expand search
            possible_matches = list(sindex.nearest(user_point.bounds, 20))

        if possible_matches:
            candidates = edges_gdf.iloc[possible_matches]

            # Compute precise distances only on candidates (not all edges)
            distances = candidates.geometry.distance(user_point)  # In meters now
            nearest_idx = distances.idxmin()
            nearest_edge = candidates.loc[nearest_idx]

            # Project point onto edge and calculate distance along edge (in meters)
            line_geom = nearest_edge.geometry
            point_on_line = line_geom.interpolate(line_geom.project(user_point))
            proj_distance_m = line_geom.project(point_on_line)  # Distance along line in meters
            snap_distance_m = user_point.distance(point_on_line)  # Snap distance in meters

            # Calculate road heading from line geometry
            coords = list(line_geom.coords)
            if len(coords) > 1:
                # Find appropriate segment for heading calculation
                total_length = line_geom.length
                if proj_distance_m < total_length * 0.1:
                    p1, p2 = coords[0], coords[1]
                elif proj_distance_m > total_length * 0.9:
                    p1, p2 = coords[-2], coords[-1]
                else:
                    segment_ratio = proj_distance_m / total_length
                    segment_idx = min(len(coords) - 2, int(segment_ratio * (len(coords) - 1)))
                    p1, p2 = coords[segment_idx], coords[segment_idx + 1]

                # Convert back to lat/lon for bearing calculation
                p1_lonlat = gpd.GeoSeries([Point(p1)], crs='EPSG:3857').to_crs('EPSG:4326').iloc[0]
                p2_lonlat = gpd.GeoSeries([Point(p2)], crs='EPSG:3857').to_crs('EPSG:4326').iloc[0]
                road_heading = calculate_bearing(p1_lonlat.y, p1_lonlat.x, p2_lonlat.y, p2_lonlat.x)
            else:
                road_heading = 0.0

            # Create result
            snapped_result = {
                'way_id': str(nearest_idx),  # Use the actual edge index
                'proj_distance': proj_distance_m,  # In meters
                'road_heading': road_heading,
                'snap_distance': snap_distance_m  # In meters
            }

            # Cache the snapping result (without user-specific data)
            _SNAP_CACHE[cache_key] = snapped_result.copy()

            # Add user-specific data
            snapped_user = user.copy()
            snapped_user.update(snapped_result)
            snapped_users.append(snapped_user)
        else:
            # Fallback if no candidates found
            snapped_user = user.copy()
            snapped_user.update({
                'way_id': 'unknown',
                'proj_distance': 0.0,
                'road_heading': 0.0,
                'snap_distance': float('inf')
            })
            snapped_users.append(snapped_user)

    result_df = pd.DataFrame(snapped_users)
    snap_time = time.time() - start_time

    print(f"   ✅ Snapped {len(result_df)} users in {snap_time:.2f}s")
    print(f"   📋 Cache hits: {cache_hits}/{len(user_df)} ({cache_hits/len(user_df)*100:.1f}%)")

    return result_df


def snap_to_road_network(user_df, config):
    """Enhanced road snapping with caching and spatial indexing"""
    if not HAS_OSMNX:
        return fallback_pca_road_approximation(user_df)

    try:
        # Calculate bounding box with padding
        padding = 0.01  # degrees
        north = user_df['latitude'].max() + padding
        south = user_df['latitude'].min() - padding
        east = user_df['longitude'].max() + padding
        west = user_df['longitude'].min() - padding

        # Get cached graph and edges
        G, edges = get_osm_graph_with_cache(north, south, east, west)

        snap_radius = config.get('snap_radius_m', 30)

        # Batch snap all users using spatial index
        snapped_users = []
        cache_hits = 0

        # Convert all user points to metric CRS for accurate distance calculations
        user_points_gdf = gpd.GeoDataFrame(
            user_df,
            geometry=gpd.points_from_xy(user_df['longitude'], user_df['latitude']),
            crs='EPSG:4326'
        ).to_crs('EPSG:3857')

        for idx, user in user_points_gdf.iterrows():
            cache_key = f"{user_df.loc[idx, 'latitude']:.6f}_{user_df.loc[idx, 'longitude']:.6f}"

            # Check snap cache
            if cache_key in _SNAP_CACHE:
                snapped_data = _SNAP_CACHE[cache_key].copy()
                cache_hits += 1
            else:
                user_point_metric = user.geometry

                # Use spatial index to find candidate edges efficiently
                buffer = user_point_metric.buffer(snap_radius)
                possible_matches_index = list(edges.sindex.intersection(buffer.bounds))

                if possible_matches_index:
                    possible_matches = edges.iloc[possible_matches_index]

                    # Calculate distances in metric CRS (meters)
                    distances = possible_matches.geometry.distance(user_point_metric)
                    nearest_idx = distances.idxmin()
                    min_distance = distances.loc[nearest_idx]

                    if min_distance <= snap_radius:
                        nearest_edge = possible_matches.loc[nearest_idx]

                        # Project point onto the edge
                        snapped_point = nearest_edge.geometry.interpolate(
                            nearest_edge.geometry.project(user_point_metric)
                        )

                        # Convert back to lat/lon
                        snapped_gdf = gpd.GeoDataFrame([1], geometry=[snapped_point], crs='EPSG:3857').to_crs('EPSG:4326')
                        snapped_lon, snapped_lat = snapped_gdf.geometry[0].coords[0]

                        snapped_data = {
                            'user_id': user_df.loc[idx, 'user_id'],
                            'original_lat': user_df.loc[idx, 'latitude'],
                            'original_lon': user_df.loc[idx, 'longitude'],
                            'snapped_lat': snapped_lat,
                            'snapped_lon': snapped_lon,
                            'snap_distance': min_distance,
                            'way_id': str(nearest_edge.name if hasattr(nearest_edge, 'name') else f"edge_{nearest_idx}"),
                            'road_segment': f"seg_{hash(str(nearest_edge.geometry)) % 10000}"
                        }
                    else:
                        # No nearby road found within snap radius
                        snapped_data = {
                            'user_id': user_df.loc[idx, 'user_id'],
                            'original_lat': user_df.loc[idx, 'latitude'],
                            'original_lon': user_df.loc[idx, 'longitude'],
                            'snapped_lat': user_df.loc[idx, 'latitude'],
                            'snapped_lon': user_df.loc[idx, 'longitude'],
                            'snap_distance': float('inf'),
                            'way_id': 'no_road',
                            'road_segment': 'no_segment'
                        }
                else:
                    # No edges found in search area
                    snapped_data = {
                        'user_id': user_df.loc[idx, 'user_id'],
                        'original_lat': user_df.loc[idx, 'latitude'],
                        'original_lon': user_df.loc[idx, 'longitude'],
                        'snapped_lat': user_df.loc[idx, 'latitude'],
                        'snapped_lon': user_df.loc[idx, 'longitude'],
                        'snap_distance': float('inf'),
                        'way_id': 'no_road',
                        'road_segment': 'no_segment'
                    }

                # Cache the result for future use
                _SNAP_CACHE[cache_key] = snapped_data.copy()

            snapped_users.append(snapped_data)

        print(f"   ✅ Snapped {len(snapped_users)} users to road network")
        print(f"   💾 Cache hits: {cache_hits}/{len(user_df)} ({cache_hits/len(user_df)*100:.1f}%)")
        print(f"   🗂️ Total cached mappings: {len(_SNAP_CACHE)}")

        return pd.DataFrame(snapped_users)

    except Exception as e:
        print(f"   ⚠️ OSM snapping failed: {e}, falling back to PCA")
        return fallback_pca_road_approximation(user_df)


def fallback_pca_road_approximation(user_df):
    """Fallback: fit a line through user points and project them onto it"""
    from sklearn.decomposition import PCA
    import numpy as np

    # Fit PCA to get principal direction
    points = user_df[['latitude', 'longitude']].values
    pca = PCA(n_components=2)
    pca.fit(points)

    # Get the main direction (first principal component)
    main_direction = pca.components_[0]

    # Project all points onto the main line
    mean_point = points.mean(axis=0)

    snapped_users = []
    for idx, user in user_df.iterrows():
        user_point = np.array([user['latitude'], user['longitude']])

        # Project onto line
        centered_point = user_point - mean_point
        proj_distance = np.dot(centered_point, main_direction)

        # Calculate road heading from PCA direction
        road_heading = calculate_bearing(
            mean_point[0], mean_point[1],
            mean_point[0] + main_direction[0], mean_point[1] + main_direction[1]
        )

        snapped_user = user.copy()
        snapped_user['way_id'] = 'pca_line'  # Single line for all users
        snapped_user['proj_distance'] = proj_distance
        snapped_user['road_heading'] = road_heading
        snapped_user['snap_distance'] = 0.0  # No actual snapping distance

        snapped_users.append(snapped_user)

    result_df = pd.DataFrame(snapped_users)
    print(f"   ✅ Fitted PCA line for {len(result_df)} users")
    return result_df


def calculate_circular_mean_heading(headings):
    """Calculate circular mean for headings to handle 0/360 wraparound"""
    angles = np.deg2rad(headings)
    mean_x = np.mean(np.cos(angles))
    mean_y = np.mean(np.sin(angles))
    mean_angle_deg = (np.degrees(np.arctan2(mean_y, mean_x)) + 360) % 360
    return mean_angle_deg


def create_road_clusters(user_df):
    """Group users by road way and direction, then create contiguous clusters"""
    print("🛣️ Creating road-aligned clusters...")

    clusters = []
    cluster_id = 0

    # Group by way_id and direction
    for way_id, way_users in user_df.groupby('way_id'):

        # Split by travel direction if needed (for two-way streets)
        direction_groups = []
        if len(way_users) > 1:
            # Check if users have significantly different headings (opposite directions)
            headings = way_users['road_heading'].values

            # Use circular statistics for headings
            mean_heading = calculate_circular_mean_heading(headings)
            heading_diffs = [bearing_difference(h, mean_heading) for h in headings]
            max_heading_diff = max(heading_diffs)

            if max_heading_diff > BEARING_TOLERANCE * 2:
                # Split into two direction groups based on circular mean
                group1_mask = way_users['road_heading'].apply(
                    lambda h: bearing_difference(h, mean_heading) <= BEARING_TOLERANCE
                )

                group1 = way_users[group1_mask]
                group2 = way_users[~group1_mask]

                if len(group1) > 0:
                    direction_groups.append(group1)
                if len(group2) > 0:
                    direction_groups.append(group2)
            else:
                direction_groups.append(way_users)
        else:
            direction_groups.append(way_users)

        # Create contiguous clusters along each direction
        for direction_group in direction_groups:
            if len(direction_group) == 0:
                continue

            # Sort by projection distance along the road (already in meters)
            sorted_users = direction_group.sort_values('proj_distance').reset_index(drop=True)

            # Split into contiguous clusters based on gaps (using metric distances)
            current_cluster = [0]  # Start with first user

            for i in range(1, len(sorted_users)):
                prev_user = sorted_users.iloc[i-1]
                curr_user = sorted_users.iloc[i]

                # Calculate gap distance (proj_distance is already in meters)
                gap_distance_m = abs(curr_user['proj_distance'] - prev_user['proj_distance'])

                if gap_distance_m <= MAX_GAP_DISTANCE:
                    # Continue current cluster
                    current_cluster.append(i)
                else:
                    # Start new cluster
                    if current_cluster:
                        cluster_users = sorted_users.iloc[current_cluster].copy()
                        cluster_users['cluster'] = cluster_id
                        clusters.append(cluster_users)
                        cluster_id += 1

                    current_cluster = [i]

            # Don't forget the last cluster
            if current_cluster:
                cluster_users = sorted_users.iloc[current_cluster].copy()
                cluster_users['cluster'] = cluster_id
                clusters.append(cluster_users)
                cluster_id += 1

    if clusters:
        result_df = pd.concat(clusters, ignore_index=True)
    else:
        # Fallback: single cluster
        result_df = user_df.copy()
        result_df['cluster'] = 0

    print(f"   ✅ Created {result_df['cluster'].nunique()} road-aligned clusters")
    return result_df


def assign_drivers_to_road_clusters(user_df, driver_df):
    """Assign drivers to road clusters with along-road optimization and partial assignment support"""
    print("🚗 Assigning drivers to road clusters...")

    routes = []
    assigned_user_ids = set()
    used_driver_ids = set()

    # Sort drivers by priority and capacity
    available_drivers = driver_df.sort_values(['priority', 'capacity'], ascending=[True, False])

    # Pre-compute driver positions as arrays for vectorized operations
    driver_lats = available_drivers['latitude'].values
    driver_lons = available_drivers['longitude'].values
    driver_ids = available_drivers['driver_id'].values

    # Process each cluster (including partial assignment)
    for cluster_id, cluster_users in user_df.groupby('cluster'):
        unassigned_in_cluster = cluster_users[~cluster_users['user_id'].isin(assigned_user_ids)]

        if unassigned_in_cluster.empty:
            continue

        # Calculate cluster characteristics
        cluster_center_lat = unassigned_in_cluster['latitude'].mean()
        cluster_center_lon = unassigned_in_cluster['longitude'].mean()
        cluster_road_heading = calculate_circular_mean_heading(unassigned_in_cluster['road_heading'].values)

        # Vectorized distance calculation to all drivers
        cluster_lat_rad = np.radians(cluster_center_lat)
        cluster_lon_rad = np.radians(cluster_center_lon)
        driver_lats_rad = np.radians(driver_lats)
        driver_lons_rad = np.radians(driver_lons)

        # Vectorized haversine distance
        dlat = driver_lats_rad - cluster_lat_rad
        dlon = driver_lons_rad - cluster_lon_rad
        a = np.sin(dlat/2)**2 + np.cos(cluster_lat_rad) * np.cos(driver_lats_rad) * np.sin(dlon/2)**2
        distances = 2 * 6371 * np.arcsin(np.sqrt(a))  # km

        # Get top 5 nearest drivers for detailed evaluation
        nearest_indices = np.argsort(distances)[:5]

        best_driver = None
        min_cost = float('inf')

        for idx in nearest_indices:
            driver = available_drivers.iloc[idx]

            if driver['driver_id'] in used_driver_ids:
                continue

            # Allow partial assignment: driver can take up to their capacity
            cluster_size = len(unassigned_in_cluster)
            users_can_take = min(driver['capacity'], cluster_size)

            if users_can_take == 0:
                continue

            # Road alignment score using circular bearing difference
            driver_bearing = calculate_bearing(
                OFFICE_LAT, OFFICE_LON,
                driver['latitude'], driver['longitude']
            )
            road_alignment_diff = bearing_difference(driver_bearing, cluster_road_heading)
            road_penalty = (road_alignment_diff / 90.0) * 2.0  # Max 2.0 penalty

            # Distance penalty (already computed)
            distance_penalty = distances[idx] * 0.5

            # Priority penalty
            priority_penalty = driver['priority'] * 0.3

            # Utilization bonus (based on how many users they can actually take)
            utilization = users_can_take / driver['capacity']
            utilization_bonus = utilization * 1.5

            # Partial assignment penalty (prefer drivers who can take more users)
            partial_penalty = (cluster_size - users_can_take) * 0.2

            total_cost = road_penalty + distance_penalty + priority_penalty + partial_penalty - utilization_bonus

            if total_cost < min_cost:
                min_cost = total_cost
                best_driver = driver

        if best_driver is not None:
            # Take users up to driver capacity, ordered by projection distance
            users_can_assign = min(best_driver['capacity'], len(unassigned_in_cluster))
            users_to_assign = unassigned_in_cluster.nsmallest(users_can_assign, 'proj_distance')

            # Create route
            route = {
                'driver_id': str(best_driver['driver_id']),
                'vehicle_id': str(best_driver.get('vehicle_id', '')),
                'vehicle_type': int(best_driver['capacity']),
                'latitude': float(best_driver['latitude']),
                'longitude': float(best_driver['longitude']),
                'assigned_users': [],
                'road_way_id': users_to_assign.iloc[0]['way_id'],
                'road_heading': cluster_road_heading
            }

            # Add users in along-road order
            for pickup_order, (_, user) in enumerate(users_to_assign.iterrows()):
                user_data = {
                    'user_id': str(user['user_id']),
                    'lat': float(user['latitude']),
                    'lng': float(user['longitude']),
                    'office_distance': float(user.get('office_distance', 0)),
                    'pickup_order': pickup_order + 1,
                    'proj_distance': float(user['proj_distance'])
                }

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
                route['utilization'] = len(route['assigned_users']) / route['vehicle_type']

            routes.append(route)
            used_driver_ids.add(best_driver['driver_id'])

            # Continue with remaining users in cluster if any
            # (This allows multiple drivers to serve the same large cluster)

    print(f"   ✅ Created {len(routes)} road-aligned routes with partial assignment support")
    return routes, assigned_user_ids


def prepare_user_driver_dataframes(data):
    """Prepare pandas dataframes from the API data"""
    # Prepare users dataframe
    users = data.get('users', [])
    user_records = []

    for user in users:
        user_record = {
            'user_id': str(user.get('id', '')),
            'latitude': float(user.get('latitude', 0.0)),
            'longitude': float(user.get('longitude', 0.0)),
            'first_name': str(user.get('first_name', '')),
            'email': str(user.get('email', ''))
        }

        # Calculate office distance
        user_record['office_distance'] = haversine_distance(
            user_record['latitude'], user_record['longitude'],
            OFFICE_LAT, OFFICE_LON
        )

        user_records.append(user_record)

    user_df = pd.DataFrame(user_records)

    # Prepare drivers dataframe
    all_drivers = []
    if "drivers" in data:
        drivers_data = data["drivers"]
        all_drivers.extend(drivers_data.get("driversUnassigned", []))
        all_drivers.extend(drivers_data.get("driversAssigned", []))
    else:
        all_drivers.extend(data.get("driversUnassigned", []))
        all_drivers.extend(data.get("driversAssigned", []))

    driver_records = []
    for driver in all_drivers:
        shift_type = driver.get('shift_type', 2)
        priority = 1 if shift_type in [1, 3] else 2

        driver_record = {
            'driver_id': str(driver.get('id', '')),
            'vehicle_id': str(driver.get('vehicle_id', '')),
            'capacity': int(driver.get('capacity', 4)),
            'latitude': float(driver.get('latitude', 0.0)),
            'longitude': float(driver.get('longitude', 0.0)),
            'shift_type': int(shift_type),
            'priority': priority
        }

        driver_records.append(driver_record)

    driver_df = pd.DataFrame(driver_records)

    return user_df, driver_df


def handle_unassigned_users(user_df, assigned_user_ids):
    """Handle users that couldn't be assigned to any route"""
    unassigned_users = user_df[~user_df['user_id'].isin(assigned_user_ids)]
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


def save_snap_cache():
    """Save the _SNAP_CACHE to disk"""
    cache_file = "snap_cache.pkl"
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(_SNAP_CACHE, f)
        print(f"   💾 Snap cache saved to {cache_file}")
    except Exception as e:
        print(f"   ⚠️ Failed to save snap cache: {e}")

def load_snap_cache():
    """Load the _SNAP_CACHE from disk"""
    cache_file = "snap_cache.pkl"
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'rb') as f:
                loaded_cache = pickle.load(f)
                _SNAP_CACHE.update(loaded_cache)
            print(f"   💾 Snap cache loaded from {cache_file}")
        except Exception as e:
            print(f"   ⚠️ Failed to load snap cache: {e}")


def run_assignment(source_id: str, parameter: int = 1, string_param: str = ""):
    """
    Road-snapping assignment function:
    1. Snap users to road network (OSM or PCA fallback)
    2. Group by road way and direction
    3. Create contiguous clusters along roads
    4. Assign drivers with along-road optimization
    """
    start_time = time.time()

    try:
        print(f"🚀 Starting road-snapping assignment for source_id: {source_id}")
        print(f"📋 Parameter: {parameter}, String parameter: {string_param}")

        # Load and validate data
        data = load_env_and_fetch_data(source_id, parameter, string_param)

        # Edge case handling
        users = data.get('users', [])
        if not users:
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

        all_drivers = []
        if "drivers" in data:
            drivers_data = data["drivers"]
            all_drivers.extend(drivers_data.get("driversUnassigned", []))
            all_drivers.extend(drivers_data.get("driversAssigned", []))
        else:
            all_drivers.extend(data.get("driversUnassigned", []))
            all_drivers.extend(data.get("driversAssigned", []))

        if not all_drivers:
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

        print(f"📥 Data loaded - Users: {len(users)}, Total Drivers: {len(all_drivers)}")

        # Extract office coordinates and validate data
        office_lat, office_lon = extract_office_coordinates(data)
        validate_input_data(data)
        print("✅ Data validation passed")

        # Prepare dataframes
        user_df, driver_df = prepare_user_driver_dataframes(data)
        print(f"📊 DataFrames prepared - Users: {len(user_df)}, Drivers: {len(driver_df)}")

        # Load snap cache at the beginning of the assignment
        load_snap_cache()

        # Step 1: Snap users to road network
        user_df = snap_to_road_network(user_df, _config)

        # Step 2: Create road-aligned clusters
        user_df = create_road_clusters(user_df)

        # Step 3: Assign drivers to road clusters
        routes, assigned_user_ids = assign_drivers_to_road_clusters(user_df, driver_df)

        # Handle unassigned users
        unassigned_users = handle_unassigned_users(user_df, assigned_user_ids)

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

        # Save cache for future runs
        save_snap_cache()

        execution_time = time.time() - start_time

        print(f"✅ Road-snapping assignment complete in {execution_time:.2f}s")
        print(f"📊 Final routes: {len(routes)}")
        print(f"🎯 Users assigned: {len(assigned_user_ids)}")
        print(f"👥 Users unassigned: {len(user_df) - len(assigned_user_ids)}")

        method_name = "OSM Road Snapping" if HAS_OSMNX else "PCA Line Approximation"

        return {
            "status": "true",
            "execution_time": execution_time,
            "data": routes,
            "unassignedUsers": unassigned_users,
            "unassignedDrivers": unassigned_drivers,
            "clustering_analysis": {
                "method": method_name,
                "clusters": user_df['cluster'].nunique() if not user_df.empty else 0
            },
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
    """Analyze the quality of the assignment"""
    if result["status"] != "true":
        return "Assignment failed"

    total_routes = len(result["data"])
    total_assigned = sum(len(route["assigned_users"]) for route in result["data"])
    total_unassigned = len(result["unassignedUsers"])

    utilizations = []
    for route in result["data"]:
        if route["assigned_users"]:
            util = len(route["assigned_users"]) / route["vehicle_type"]
            utilizations.append(util)

    analysis = {
        "total_routes": total_routes,
        "total_assigned_users": total_assigned,
        "total_unassigned_users": total_unassigned,
        "assignment_rate": round(total_assigned / (total_assigned + total_unassigned) * 100, 1) if (total_assigned + total_unassigned) > 0 else 0,
        "avg_utilization": round(np.mean(utilizations) * 100, 1) if utilizations else 0,
        "clustering_method": result.get("clustering_analysis", {}).get("method", "Unknown")
    }

    return analysis