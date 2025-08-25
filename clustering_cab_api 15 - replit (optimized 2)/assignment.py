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
    from shapely.geometry import Point, box
    import networkx as nx
    HAS_OSMNX = True

    # Configure OSMnx settings (compatible with different versions)
    try:
        # For OSMnx v1.6+
        ox.settings.use_cache = True
        ox.settings.log_console = False
        ox.settings.log_file = False
    except AttributeError:
        try:
            # For OSMnx v1.3-1.5
            ox.config(use_cache=True, log_console=False, log_file=False)
        except Exception:
            try:
                # For older versions
                ox.config(use_cache=True, log_console=False)
            except Exception:
                # If all fails, continue without configuration
                pass

    print("✅ OSMnx successfully loaded and configured")

except ImportError:
    HAS_OSMNX = False
    print("⚠️ OSMnx not available, using PCA fallback for road approximation")

# Global configuration and road graph for TSP
_config = {}
road_graph = None

# Global graph cache and spatial index cache
_GRAPH_CACHE = {}
_EDGES_CACHE = {}
_SNAP_CACHE = {}

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.WARNING)
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
            try:
                # For newer OSMnx versions
                _, edges = ox.graph_to_gdfs(G)
            except ValueError:
                # For older versions
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
        # Try different OSMnx versions - newer versions use different parameter names
        try:
            # For newer OSMnx versions (v1.6+) - use positional arguments
            G = ox.graph_from_bbox(north, south, east, west, network_type='drive')
        except Exception as e1:
            try:
                # Alternative syntax for some versions
                G = ox.graph_from_bbox(north=north, south=south, east=east, west=west, network_type='drive')
            except Exception as e2:
                try:
                    # Final fallback - use polygon method
                    from shapely.geometry import box
                    polygon = box(west, south, east, north)
                    G = ox.graph_from_polygon(polygon, network_type='drive')
                except Exception as e3:
                    print(f"   ❌ All OSMnx methods failed: {e1}, {e2}, {e3}")
                    raise Exception("Could not download OSM graph with any method")

        # Save to cache
        ox.save_graphml(G, cache_file)

        # Convert to GeoDataFrame and create spatial index
        try:
            # For newer OSMnx versions
            _, edges = ox.graph_to_gdfs(G)
        except ValueError:
            # For older versions
            edges = ox.graph_to_gdfs(G, nodes=False, edges=True)

        edges = edges.to_crs(epsg=3857)  # Project to metric CRS

        _GRAPH_CACHE[bbox_str] = G
        _EDGES_CACHE[bbox_str] = edges
        print(f"   ✅ Downloaded and cached {len(edges)} edges with spatial index")
        return G, edges

    except Exception as e:
        print(f"   ⚠️ OSM download failed or timed out: {e}")
        print(f"   🔄 Falling back to PCA road approximation")
        raise Exception(f"OSM download timeout, using PCA fallback")


def calculate_bearing(lat1, lon1, lat2, lon2):
    """Calculate bearing between two points"""
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    dlon = lon2 - lon1
    x = math.sin(dlon) * math.cos(lat2)
    y = math.sin(lat1) * math.cos(lat2) - math.cos(lat1) * math.sin(lat2) * math.cos(dlon)

    bearing = math.atan2(y, x)
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
    try:
        sindex = edges_gdf.sindex
    except Exception:
        sindex = None

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
        if sindex is not None:
            try:
                possible_matches = list(sindex.nearest(user_point.bounds, 5))  # Get 5 nearest candidates
            except Exception:
                # Fallback to distance-based search
                distances = edges_gdf.geometry.distance(user_point)
                possible_matches = distances.nsmallest(5).index.tolist()
        else:
            # Fallback to distance-based search
            distances = edges_gdf.geometry.distance(user_point)
            possible_matches = distances.nsmallest(5).index.tolist()

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


def load_tricity_graph():
    """Load pre-saved tricity graph if available"""
    graph_file = "tricity_main_roads.graphml"
    if os.path.exists(graph_file):
        print(f"   ✅ Loading local graph: {graph_file}")
        G = ox.load_graphml(graph_file)
        return G
    else:
        raise FileNotFoundError(f"{graph_file} not found")


def snap_with_osmnx(user_df, config):
    """Enhanced OSMnx snapping with caching and performance optimizations"""
    global road_graph  # Make road graph globally accessible for TSP

    try:
        print("   🗺️  Using OSMnx for precise road network snapping...")

        # First, try to load existing cached graph file
        graph_file = "tricity_main_roads.graphml"
        if os.path.exists(graph_file):
            print(f"   📁 Loading existing cached graph: {graph_file}")
            G = ox.load_graphml(graph_file)
            road_graph = G
            print("   ✅ Cached graph loaded and stored globally.")
        else:
            # Calculate bounding box for all users
            lat_min, lat_max = user_df['latitude'].min(), user_df['latitude'].max()
            lon_min, lon_max = user_df['longitude'].min(), user_df['longitude'].max()

            # Add buffer (in degrees)
            buffer = 0.01
            bbox = (lat_min - buffer, lat_max + buffer, lon_min - buffer, lon_max + buffer)

            print(f"   📍 Fetching road network for bbox: {bbox}")

            # Fetch road network - fix parameter order and naming
            try:
                # Try newer OSMnx API (v1.6+)
                G = ox.graph_from_bbox(
                    bbox=(bbox[1], bbox[0], bbox[3], bbox[2]),  # (north, south, east, west)
                    network_type='drive',
                    simplify=True
                )
            except TypeError:
                try:
                    # Try older OSMnx API (v1.3-1.5)
                    G = ox.graph_from_bbox(
                        north=bbox[1], south=bbox[0], east=bbox[3], west=bbox[2],
                        network_type='drive',
                        simplify=True
                    )
                except Exception:
                    # Final fallback using individual parameters
                    G = ox.graph_from_bbox(
                        bbox[1], bbox[0], bbox[3], bbox[2],  # north, south, east, west
                        network_type='drive',
                        simplify=True
                    )

            # Store globally for TSP access
            road_graph = G
            print("   🌐 Road graph downloaded and stored globally.")

        # Convert graph to GeoDataFrames with edges projected to metric CRS
        try:
            _, edges = ox.graph_to_gdfs(G)
        except ValueError:
            edges = ox.graph_to_gdfs(G, nodes=False, edges=True)
        edges = edges.to_crs('EPSG:3857') # Project to metric CRS

        return snap_users_to_roads_optimized(user_df, edges)

    except Exception as e:
        print(f"   ⚠️ OSMnx snapping failed: {e}")
        return fallback_pca_road_approximation(user_df)


def snap_to_road_network(user_df, config):
    """
    Snap user coordinates to nearest road using OSMnx or PCA fallback
    """
    print("🛣️  Step 1: Snapping users to road network...")

    if HAS_OSMNX:
        return snap_with_osmnx(user_df, config)
    else:
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
    MIN_CLUSTER_SIZE = 2  # Minimum users per cluster to improve utilization

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

        # More aggressive clustering to reduce unassigned users
        cluster_sizes = result_df.groupby('cluster').size()
        small_clusters = cluster_sizes[cluster_sizes < MIN_CLUSTER_SIZE].index
        large_clusters = cluster_sizes[cluster_sizes >= MIN_CLUSTER_SIZE].index

        if len(small_clusters) > 0:
            if len(large_clusters) > 0:
                # Merge small clusters into the nearest large cluster
                for small_cluster_id in small_clusters:
                    small_cluster_users = result_df[result_df['cluster'] == small_cluster_id]
                    small_cluster_center = (
                        small_cluster_users['latitude'].mean(),
                        small_cluster_users['longitude'].mean()
                    )
                    
                    # Find nearest large cluster
                    best_large_cluster = None
                    min_distance = float('inf')
                    
                    for large_cluster_id in large_clusters:
                        large_cluster_users = result_df[result_df['cluster'] == large_cluster_id]
                        large_cluster_center = (
                            large_cluster_users['latitude'].mean(),
                            large_cluster_users['longitude'].mean()
                        )
                        
                        distance = haversine_distance(
                            small_cluster_center[0], small_cluster_center[1],
                            large_cluster_center[0], large_cluster_center[1]
                        )
                        
                        if distance < min_distance:
                            min_distance = distance
                            best_large_cluster = large_cluster_id
                    
                    if best_large_cluster is not None:
                        result_df.loc[result_df['cluster'] == small_cluster_id, 'cluster'] = best_large_cluster
            else:
                # If no large clusters, merge all small clusters together
                merge_target = small_clusters[0]
                for small_cluster_id in small_clusters[1:]:
                    result_df.loc[result_df['cluster'] == small_cluster_id, 'cluster'] = merge_target

            print(f"   🔗 Merged {len(small_clusters)} small clusters to improve utilization")

        # Re-index clusters to be sequential
        unique_clusters = sorted(result_df['cluster'].unique())
        cluster_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_clusters)}
        result_df['cluster'] = result_df['cluster'].map(cluster_mapping)

    else:
        # Fallback: single cluster
        result_df = user_df.copy()
        result_df['cluster'] = 0

    print(f"   ✅ Created {result_df['cluster'].nunique()} road-aligned clusters")
    return result_df


def calculate_road_distance(G, lat1, lon1, lat2, lon2):
    """Calculate shortest path distance between two points using road network"""
    try:
        # Get nearest nodes for both points
        node1 = ox.nearest_nodes(G, lon1, lat1)
        node2 = ox.nearest_nodes(G, lon2, lat2)

        # Calculate shortest path length in meters
        distance = ox.shortest_path_length(G, node1, node2, weight='length')
        return distance / 1000.0  # Convert to km
    except Exception:
        # Fallback to haversine if routing fails
        return haversine_distance(lat1, lon1, lat2, lon2)


def reorder_cluster_by_roads(G, driver_lat, driver_lon, users_to_assign_df):
    """
    Reorders a cluster of users based on road network distance using a TSP-like greedy approach.
    This function aims to optimize the pickup sequence for a given driver and a set of users.
    """
    if len(users_to_assign_df) <= 1:
        # If only one user or no users, return as is
        for idx, user in enumerate(users_to_assign_df):
            user['pickup_order'] = idx + 1
        return users_to_assign_df

    # Ensure users_to_assign_df is a DataFrame for easier manipulation
    if not isinstance(users_to_assign_df, pd.DataFrame):
        users_to_assign_df = pd.DataFrame(users_to_assign_df)

    # Sort users by projection distance to provide a sensible initial order if graph fails
    users_to_assign_df = users_to_assign_df.sort_values('proj_distance').reset_index(drop=True)

    try:
        # Use the globally available road_graph if G is None
        if G is None and HAS_OSMNX and road_graph is not None:
            G = road_graph
        elif G is None:
            print("   ⚠️ No road graph available for TSP ordering, using projection distance fallback.")
            for idx, user in enumerate(users_to_assign_df):
                user['pickup_order'] = idx + 1
            return users_to_assign_df.to_dict('records')

        # Convert DataFrame to a list of dictionaries for easier manipulation
        remaining_users_list = users_to_assign_df.to_dict('records')
        ordered_users = []

        # Start from the driver's location
        current_lat, current_lon = driver_lat, driver_lon

        # Greedy nearest neighbor algorithm using road distances
        while remaining_users_list:
            best_user_idx = -1
            min_distance = float('inf')

            # Find the nearest user by road distance
            for idx, user in enumerate(remaining_users_list):
                road_dist = calculate_road_distance(
                    G, current_lat, current_lon,
                    user['latitude'], user['longitude']
                )

                if road_dist < min_distance:
                    min_distance = road_dist
                    best_user_idx = idx

            # Add the nearest user to the ordered list
            best_user = remaining_users_list.pop(best_user_idx)
            ordered_users.append(best_user)

            # Update current location to the picked user's location
            current_lat, current_lon = best_user['latitude'], best_user['longitude']

        # Assign pickup order based on the optimized sequence
        for i, user in enumerate(ordered_users):
            user['pickup_order'] = i + 1

        print(f"   🛣️ Optimized pickup order for {len(ordered_users)} users using road network.")
        return ordered_users

    except Exception as e:
        print(f"   ⚠️ Road ordering failed ({e}), using projection fallback.")
        # Fallback to projection distance if TSP calculation fails
        for idx, user in enumerate(users_to_assign_df):
            user['pickup_order'] = idx + 1
        return users_to_assign_df.to_dict('records')


def order_users_by_road_distance(driver_lat, driver_lon, users_to_assign, G=None):
    """Order users by shortest road path from driver location using greedy nearest neighbor"""
    if len(users_to_assign) <= 1:
        return users_to_assign.reset_index(drop=True)

    # If no graph available, fallback to projection distance ordering
    if G is None or not HAS_OSMNX:
        return users_to_assign.sort_values('proj_distance').reset_index(drop=True)

    try:
        # Convert to list for manipulation
        remaining_users = users_to_assign.copy().reset_index(drop=True)
        ordered_users = []

        # Start from driver location
        current_lat, current_lon = driver_lat, driver_lon

        # Greedy nearest neighbor algorithm using road distances
        while len(remaining_users) > 0:
            best_idx = 0
            min_distance = float('inf')

            # Find nearest user by road distance
            for idx, user in remaining_users.iterrows():
                road_dist = calculate_road_distance(G, current_lat, current_lon,
                                                  user['latitude'], user['longitude'])

                if road_dist < min_distance:
                    min_distance = road_dist
                    best_idx = idx

            # Add best user to ordered list
            best_user = remaining_users.iloc[best_idx]
            ordered_users.append(best_user)

            # Update current position and remove user from remaining
            current_lat, current_lon = best_user['latitude'], best_user['longitude']
            remaining_users = remaining_users.drop(remaining_users.index[best_idx]).reset_index(drop=True)

        # Convert back to DataFrame with sequential index
        result_df = pd.DataFrame(ordered_users).reset_index(drop=True)
        print(f"   🛣️ Ordered {len(result_df)} users by road distance (TSP-like)")
        return result_df

    except Exception as e:
        print(f"   ⚠️ Road ordering failed ({e}), using projection fallback")
        return users_to_assign.sort_values('proj_distance').reset_index(drop=True)


def assign_drivers_to_road_clusters(user_df, driver_df):
    """Assign drivers to road clusters with road-aware route ordering"""
    print("🚗 Assigning drivers to road clusters...")

    routes = []
    assigned_user_ids = set()
    used_driver_ids = set()

    # Try to get the road graph for route ordering
    road_graph_for_tsp = None
    try:
        if HAS_OSMNX:
            # Try to load the cached graph
            if 'tricity_main_roads.graphml' in os.listdir():
                road_graph_for_tsp = ox.load_graphml('tricity_main_roads.graphml')
                print("   🗺️ Using pre-saved road graph for optimal pickup ordering")
            # Check if we have a cached graph from earlier operations
            elif _GRAPH_CACHE:
                road_graph_for_tsp = list(_GRAPH_CACHE.values())[0]
                print("   🗺️ Using cached road graph for pickup ordering")
            # If global road_graph is set (from snap_with_osmnx)
            elif road_graph is not None:
                road_graph_for_tsp = road_graph
                print("   🗺️ Using globally stored road graph for pickup ordering")
    except Exception as e:
        print(f"   ⚠️ Could not load road graph for ordering: {e}")

    # Sort drivers by priority and capacity
    available_drivers = driver_df.sort_values(['priority', 'capacity'], ascending=[True, False])

    # First pass: assign drivers to clusters normally
    for cluster_id, cluster_users in user_df.groupby('cluster'):
        unassigned_in_cluster = cluster_users[~cluster_users['user_id'].isin(assigned_user_ids)]

        if unassigned_in_cluster.empty:
            continue

        # Try to assign multiple drivers to larger clusters
        while len(unassigned_in_cluster) > 0 and len(used_driver_ids) < len(available_drivers):
            # Calculate cluster characteristics
            cluster_center_lat = unassigned_in_cluster['latitude'].mean()
            cluster_center_lon = unassigned_in_cluster['longitude'].mean()
            cluster_road_heading = calculate_circular_mean_heading(unassigned_in_cluster['road_heading'].values)

            best_driver = None
            min_cost = float('inf')

            # Find best available driver for remaining users
            for _, driver in available_drivers.iterrows():
                if driver['driver_id'] in used_driver_ids:
                    continue

                # Calculate distance to cluster
                distance = haversine_distance(
                    cluster_center_lat, cluster_center_lon,
                    driver['latitude'], driver['longitude']
                )

                # Relaxed distance threshold - allow up to 5km
                if distance > 5.0:
                    continue

                # Allow partial assignment: driver can take up to their capacity
                cluster_size = len(unassigned_in_cluster)
                users_can_take = min(driver['capacity'], cluster_size)

                if users_can_take == 0:
                    continue

                # Simplified cost calculation for better assignment rate
                distance_penalty = distance * 0.3  # Reduced distance penalty
                priority_penalty = driver['priority'] * 0.2  # Reduced priority penalty
                utilization_bonus = (users_can_take / driver['capacity']) * 2.0  # Increased utilization bonus

                total_cost = distance_penalty + priority_penalty - utilization_bonus

                if total_cost < min_cost:
                    min_cost = total_cost
                    best_driver = driver

            if best_driver is not None:
                # Take users up to driver capacity
                users_can_assign = min(best_driver['capacity'], len(unassigned_in_cluster))
                candidate_users_df = unassigned_in_cluster.head(users_can_assign)

                # Convert DataFrame to list of dicts for reorder_cluster_by_roads
                selected_users = candidate_users_df.to_dict('records')

                # Apply road-aware TSP ordering to optimize pickup sequence
                if HAS_OSMNX and road_graph_for_tsp is not None:
                    selected_users = reorder_cluster_by_roads(
                        road_graph_for_tsp,
                        best_driver['latitude'],
                        best_driver['longitude'],
                        selected_users
                    )
                else:
                    # Fallback: assign sequential pickup order
                    for idx, user in enumerate(selected_users, 1):
                        user['pickup_order'] = idx

                route_data = {
                    'driver_id': str(best_driver['driver_id']),
                    'latitude': float(best_driver['latitude']),
                    'longitude': float(best_driver['longitude']),
                    'vehicle_type': int(best_driver['capacity']),
                    'vehicle_id': str(best_driver.get('vehicle_id', '')),
                    'assigned_users': selected_users,
                    'road_way_id': selected_users[0]['way_id'] if selected_users else 'N/A',
                    'road_heading': cluster_road_heading
                }

                routes.append(route_data)
                used_driver_ids.add(best_driver['driver_id'])

                # Add assigned users to the set
                for user in selected_users:
                    assigned_user_ids.add(user['user_id'])

                # Update unassigned users in cluster
                unassigned_in_cluster = unassigned_in_cluster[~unassigned_in_cluster['user_id'].isin(assigned_user_ids)]
            else:
                break  # No suitable driver found, move to next cluster

    # Second pass: fallback assignment for remaining unassigned users
    remaining_users = user_df[~user_df['user_id'].isin(assigned_user_ids)]
    if len(remaining_users) > 0 and len(used_driver_ids) < len(available_drivers):
        print(f"   🔄 Fallback assignment for {len(remaining_users)} remaining users...")
        
        for _, user in remaining_users.iterrows():
            # Find nearest available driver within 8km
            best_fallback_driver = None
            min_fallback_distance = float('inf')
            
            for _, driver in available_drivers.iterrows():
                if driver['driver_id'] in used_driver_ids:
                    continue
                    
                distance = haversine_distance(
                    user['latitude'], user['longitude'],
                    driver['latitude'], driver['longitude']
                )
                
                if distance < min_fallback_distance and distance <= 8.0:  # 8km max
                    min_fallback_distance = distance
                    best_fallback_driver = driver
            
            if best_fallback_driver is not None:
                # Create individual route for this user
                user_data = user.to_dict()
                user_data['pickup_order'] = 1
                
                route_data = {
                    'driver_id': str(best_fallback_driver['driver_id']),
                    'latitude': float(best_fallback_driver['latitude']),
                    'longitude': float(best_fallback_driver['longitude']),
                    'vehicle_type': int(best_fallback_driver['capacity']),
                    'vehicle_id': str(best_fallback_driver.get('vehicle_id', '')),
                    'assigned_users': [user_data],
                    'road_way_id': user_data.get('way_id', 'N/A'),
                    'road_heading': user_data.get('road_heading', 0.0)
                }
                
                routes.append(route_data)
                used_driver_ids.add(best_fallback_driver['driver_id'])
                assigned_user_ids.add(user['user_id'])

    print(f"   ✅ Created {len(routes)} road-aligned routes with optimal pickup sequences")
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
            'latitude': float(user['latitude']),
            'longitude': float(user['longitude']),
            'lat': float(user['latitude']),  # Add both formats for compatibility
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