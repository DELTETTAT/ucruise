
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
from scipy.optimize import linear_sum_assignment
from dotenv import load_dotenv
import warnings
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
import logging
import pickle
import hashlib
from itertools import combinations

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

    # Enhanced configuration for multi-constraint optimization
    config = {}
    config['SNAP_RADIUS'] = max(10, min(150, float(cfg.get("snap_radius_m", 100))))
    config['MAX_GAP_DISTANCE'] = max(100, float(cfg.get("max_gap_distance_m", 5000)))
    config['BEARING_TOLERANCE'] = max(10, min(45, float(cfg.get("bearing_tolerance", 45))))
    config['MAX_DETOUR_KM'] = max(1.0, float(cfg.get("max_detour_km", 8.0)))
    
    # Advanced clustering parameters
    config['MAX_EXTRA_DISTANCE_KM'] = float(cfg.get("max_extra_distance_km", 4.0))
    config['LOAD_BALANCE'] = cfg.get("load_balance", True)
    config['MIN_DRIVER_CAPACITY_UTIL'] = float(cfg.get("min_driver_capacity_util", 1.0))
    config['ALLOW_EXTRA_FILL'] = cfg.get("allow_extra_fill", True)
    config['MAX_FILL_DISTANCE_KM'] = float(cfg.get("max_fill_distance_km", 3.0))
    config['FALLBACK_ASSIGN_ENABLED'] = cfg.get("fallback_assign_enabled", True)
    config['PREFER_NEAREST_DRIVER'] = cfg.get("prefer_nearest_driver", False)
    config['OVERASSIGN_IF_NEEDED'] = cfg.get("overassign_if_needed", True)
    config['AUTO_BALANCE_SMALL_CLUSTERS'] = cfg.get("auto_balance_small_clusters", True)
    config['MIN_CLUSTER_SIZE'] = int(cfg.get("min_cluster_size", 2))
    config['FALLBACK_MAX_DIST_KM'] = float(cfg.get("fallback_max_dist_km", 4.0))
    config['N_JOBS'] = int(cfg.get("n_jobs", 2))

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
            except (TypeError, ValueError):
                # For older versions
                edges = ox.graph_to_gdfs(G, nodes=False, edges=True)

            edges = edges.to_crs(epsg=3857)  # Project to metric CRS
            _GRAPH_CACHE[bbox_str] = G
            _EDGES_CACHE[bbox_str] = edges
            print(f"   ✅ Loaded {len(edges)} edges from cache with spatial index")
            return G, edges
        except Exception as e:
            print(f"   ⚠️ Cache load failed: {e}, downloading fresh")

    # Download new graph with better error handling
    try:
        print(f"   🌐 Downloading new OSM graph for bbox: ({north}, {south}, {east}, {west})")
        G = None

        # Method 1: Try modern OSMnx API with bbox
        try:
            G = ox.graph_from_bbox(
                bbox=(north, south, east, west), 
                network_type='drive', 
                simplify=True,
                truncate_by_edge=True
            )
            print("   ✅ Used modern bbox API")
        except Exception as e1:
            print(f"   ⚠️ Modern bbox API failed: {e1}")

            # Method 2: Try older parameter style
            try:
                G = ox.graph_from_bbox(
                    north, south, east, west, 
                    network_type='drive', 
                    simplify=True
                )
                print("   ✅ Used legacy parameter style")
            except Exception as e2:
                print(f"   ⚠️ Legacy parameter style failed: {e2}")

                # Method 3: Try polygon method
                try:
                    from shapely.geometry import box
                    polygon = box(west, south, east, north)
                    G = ox.graph_from_polygon(
                        polygon, 
                        network_type='drive', 
                        simplify=True
                    )
                    print("   ✅ Used polygon method")
                except Exception as e3:
                    print(f"   ❌ All OSMnx methods failed: {e1}, {e2}, {e3}")
                    raise Exception(f"Could not download OSM graph: {e3}")

        if G is None:
            raise Exception("Failed to create graph with any method")

        # Save to cache
        try:
            ox.save_graphml(G, cache_file)
            print(f"   💾 Graph saved to {cache_file}")
        except Exception as save_err:
            print(f"   ⚠️ Could not save graph: {save_err}")

        # Convert to GeoDataFrame and create spatial index
        try:
            # Try different methods for graph_to_gdfs
            try:
                nodes, edges = ox.graph_to_gdfs(G)
            except TypeError:
                # For some OSMnx versions that have different signatures
                edges = ox.graph_to_gdfs(G, nodes=False, edges=True, node_geometry=False)
            except Exception:
                # Final fallback
                gdf_nodes, gdf_edges = ox.graph_to_gdfs(G, nodes=True, edges=True)
                edges = gdf_edges

            # Project to metric CRS
            if hasattr(edges, 'crs') and edges.crs is not None:
                edges = edges.to_crs(epsg=3857)
            else:
                # If no CRS, assume WGS84 and set it
                edges = edges.set_crs(epsg=4326, allow_override=True).to_crs(epsg=3857)

            _GRAPH_CACHE[bbox_str] = G
            _EDGES_CACHE[bbox_str] = edges
            print(f"   ✅ Downloaded and cached {len(edges)} edges with spatial index")
            return G, edges

        except Exception as convert_err:
            print(f"   ❌ Failed to convert graph to GeoDataFrame: {convert_err}")
            raise Exception(f"Graph conversion failed: {convert_err}")

    except Exception as e:
        print(f"   ⚠️ OSM download failed or timed out: {e}")
        print(f"   🔄 Falling back to PCA road approximation")
        raise Exception(f"OSM download timeout, using PCA fallback")


def calculate_bearing(lat1, lon1, lat2, lon2):
    """Calculate bearing between two points"""
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    dlon = lon2 - lon1
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)

    bearing = math.atan2(x, y)  # Fixed: corrected x, y order
    bearing = math.degrees(bearing)
    bearing = (bearing + 360) % 360

    return bearing


def bearing_difference(b1, b2):
    """Compute minimum difference between two bearings"""
    diff = abs(b1 - b2) % 360
    return min(diff, 360 - diff)


def snap_users_to_roads_optimized(user_df, edges_gdf):
    """Optimized snapping using spatial index and metric CRS (robust sindex handling)."""
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

        # Use spatial index to get nearest candidates (robust)
        possible_matches = []
        if sindex is not None:
            try:
                # sindex.nearest might return different shapes depending on versions
                raw_matches = list(sindex.nearest(user_point.bounds, 5))
                # flatten if list of pairs/tuples
                flattened = []
                for m in raw_matches:
                    if hasattr(m, '__iter__') and not isinstance(m, (int, np.integer)):
                        flattened.extend(list(m))
                    else:
                        flattened.append(m)
                # convert to labels that exist in edges_gdf.index
                possible_matches = [m for m in flattened if m in edges_gdf.index]
                # if still empty, fallback to distance-based selection below
                if not possible_matches:
                    raise Exception("sindex returned no usable candidates")
            except Exception:
                # Fallback to distance-based search
                distances = edges_gdf.geometry.distance(user_point)
                possible_matches = distances.nsmallest(5).index.tolist()
        else:
            distances = edges_gdf.geometry.distance(user_point)
            possible_matches = distances.nsmallest(5).index.tolist()

        if possible_matches:
            candidates = edges_gdf.loc[possible_matches] if set(possible_matches) <= set(edges_gdf.index) else edges_gdf.iloc[possible_matches]

            # Compute precise distances only on candidates
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
                    north=bbox[1], south=bbox[0], east=bbox[3], west=bbox[2],
                    network_type='drive',
                    simplify=True
                )
            except Exception as e1:
                try:
                    # Final fallback using polygon method
                    from shapely.geometry import box
                    polygon = box(bbox[2], bbox[0], bbox[3], bbox[1])  # west, south, east, north
                    G = ox.graph_from_polygon(polygon, network_type='drive', simplify=True)
                except Exception as e2:
                    print(f"   ❌ All OSMnx methods failed: {e1}, {e2}")
                    raise Exception("Could not download OSM graph with any method")

            # Store globally for TSP access
            road_graph = G
            print("   🌐 Road graph downloaded and stored globally.")

        # Convert graph to GeoDataFrames with edges projected to metric CRS
        try:
            _, edges = ox.graph_to_gdfs(G)
        except (TypeError, ValueError):
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


def adaptive_dbscan_clustering(user_df, driver_capacities):
    """
    Advanced multi-constraint clustering using adaptive DBSCAN with capacity awareness
    """
    print("🎯 Multi-constraint clustering with adaptive DBSCAN...")
    
    # Calculate data density for adaptive parameters
    coords = user_df[['latitude', 'longitude']].values
    scaler = StandardScaler()
    coords_scaled = scaler.fit_transform(coords)
    
    # Estimate optimal eps using k-distance
    k = min(4, len(user_df) - 1)
    if k > 0:
        distances = cdist(coords_scaled, coords_scaled)
        k_distances = np.sort(distances, axis=1)[:, k]
        eps_candidate = np.percentile(k_distances, 75)
    else:
        eps_candidate = 0.5
    
    # Convert back to geographic scale (roughly)
    lat_std = user_df['latitude'].std()
    lon_std = user_df['longitude'].std()
    geo_scale = max(lat_std, lon_std) if max(lat_std, lon_std) > 0 else 0.01
    eps_geographic = eps_candidate * geo_scale
    
    # Adaptive clustering with multiple trials
    best_clustering = None
    best_score = -1
    
    # Try different eps values around the estimated optimal
    eps_values = [eps_geographic * f for f in [0.5, 0.75, 1.0, 1.25, 1.5]]
    min_samples_values = [2, 3, max(2, len(user_df) // 20)]
    
    for eps in eps_values:
        for min_samples in min_samples_values:
            try:
                # Enhanced feature matrix including road information
                features = []
                for _, user in user_df.iterrows():
                    feature_vector = [
                        user['latitude'],
                        user['longitude'],
                        math.cos(math.radians(user.get('road_heading', 0))) * 0.1,  # Directional component
                        math.sin(math.radians(user.get('road_heading', 0))) * 0.1,
                        user.get('office_distance', 0) * 0.1  # Office distance component
                    ]
                    features.append(feature_vector)
                
                features_scaled = scaler.fit_transform(features)
                
                # Apply DBSCAN
                clustering = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=1)
                cluster_labels = clustering.fit_predict(features_scaled)
                
                n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
                
                if n_clusters == 0:
                    continue
                
                # Calculate clustering quality score
                score = evaluate_clustering_quality(user_df, cluster_labels, driver_capacities)
                
                if score > best_score:
                    best_score = score
                    best_clustering = cluster_labels
                    
            except Exception as e:
                continue
    
    # Fallback if DBSCAN fails
    if best_clustering is None:
        print("   ⚠️ DBSCAN failed, using geographic fallback clustering")
        best_clustering = geographic_fallback_clustering(user_df, driver_capacities)
    
    # Assign cluster labels
    user_df = user_df.copy()
    user_df['cluster'] = best_clustering
    
    # Post-process: merge noise points and optimize cluster sizes
    user_df = optimize_cluster_sizes(user_df, driver_capacities)
    
    n_final_clusters = user_df['cluster'].nunique()
    avg_cluster_size = len(user_df) / n_final_clusters if n_final_clusters > 0 else 0
    
    print(f"   ✅ Created {n_final_clusters} capacity-optimized clusters")
    print(f"   📊 Average cluster size: {avg_cluster_size:.1f} users")
    
    return user_df


def evaluate_clustering_quality(user_df, cluster_labels, driver_capacities):
    """
    Evaluate clustering quality based on multiple criteria
    """
    if len(set(cluster_labels)) <= 1:
        return -1
    
    # Geographic compactness (silhouette score)
    coords = user_df[['latitude', 'longitude']].values
    try:
        geo_score = silhouette_score(coords, cluster_labels)
    except:
        geo_score = 0
    
    # Capacity utilization efficiency
    cluster_counts = pd.Series(cluster_labels).value_counts()
    noise_points = (cluster_labels == -1).sum()
    
    capacity_scores = []
    avg_capacity = np.mean(driver_capacities) if driver_capacities else 5
    
    for cluster_size in cluster_counts.values:
        if cluster_size <= avg_capacity:
            capacity_scores.append(cluster_size / avg_capacity)
        else:
            # Penalty for oversized clusters
            capacity_scores.append(avg_capacity / cluster_size)
    
    capacity_score = np.mean(capacity_scores) if capacity_scores else 0
    
    # Noise penalty
    noise_penalty = noise_points / len(cluster_labels) if len(cluster_labels) > 0 else 1
    
    # Combined score
    combined_score = (geo_score * 0.4) + (capacity_score * 0.5) - (noise_penalty * 0.3)
    
    return combined_score


def geographic_fallback_clustering(user_df, driver_capacities):
    """
    Simple geographic clustering fallback when DBSCAN fails
    """
    coords = user_df[['latitude', 'longitude']].values
    n_drivers = len(driver_capacities)
    
    # Use KMeans with driver count
    if len(coords) > n_drivers:
        n_clusters = min(n_drivers, len(coords) // 2)
    else:
        n_clusters = max(1, len(coords) // 3)
    
    try:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(coords)
    except:
        # Ultimate fallback: assign all to one cluster
        cluster_labels = np.zeros(len(coords), dtype=int)
    
    return cluster_labels


def optimize_cluster_sizes(user_df, driver_capacities):
    """
    Optimize cluster sizes to better match driver capacities
    """
    if driver_capacities:
        avg_capacity = np.mean(driver_capacities)
        max_capacity = max(driver_capacities)
    else:
        avg_capacity = 5
        max_capacity = 7
    
    # Handle noise points (-1 labels) by assigning to nearest cluster
    noise_mask = user_df['cluster'] == -1
    if noise_mask.any():
        noise_users = user_df[noise_mask]
        valid_clusters = user_df[~noise_mask]
        
        if not valid_clusters.empty:
            for idx, noise_user in noise_users.iterrows():
                # Find nearest valid cluster
                distances = []
                for cluster_id in valid_clusters['cluster'].unique():
                    cluster_users = valid_clusters[valid_clusters['cluster'] == cluster_id]
                    cluster_center_lat = cluster_users['latitude'].mean()
                    cluster_center_lon = cluster_users['longitude'].mean()
                    
                    dist = haversine_distance(
                        noise_user['latitude'], noise_user['longitude'],
                        cluster_center_lat, cluster_center_lon
                    )
                    distances.append((dist, cluster_id))
                
                if distances:
                    _, nearest_cluster = min(distances)
                    user_df.loc[idx, 'cluster'] = nearest_cluster
        else:
            # If no valid clusters, create new ones
            user_df.loc[noise_mask, 'cluster'] = 0
    
    # Split oversized clusters
    cluster_sizes = user_df.groupby('cluster').size()
    oversized_clusters = cluster_sizes[cluster_sizes > max_capacity]
    
    next_cluster_id = user_df['cluster'].max() + 1
    
    for cluster_id, size in oversized_clusters.items():
        if size <= max_capacity:
            continue
            
        cluster_users = user_df[user_df['cluster'] == cluster_id].copy()
        
        # Split into smaller clusters using KMeans
        n_splits = math.ceil(size / avg_capacity)
        coords = cluster_users[['latitude', 'longitude']].values
        
        try:
            kmeans = KMeans(n_clusters=n_splits, random_state=42, n_init=10)
            sub_labels = kmeans.fit_predict(coords)
            
            # Reassign cluster labels
            for i, (idx, _) in enumerate(cluster_users.iterrows()):
                if sub_labels[i] == 0:
                    continue  # Keep original cluster_id
                else:
                    user_df.loc[idx, 'cluster'] = next_cluster_id + sub_labels[i] - 1
            
            next_cluster_id += n_splits - 1
            
        except:
            # If KMeans fails, simple splitting
            chunk_size = max(2, int(avg_capacity))
            for i, (idx, _) in enumerate(cluster_users.iterrows()):
                if i >= chunk_size:
                    user_df.loc[idx, 'cluster'] = next_cluster_id
                    if (i + 1) % chunk_size == 0:
                        next_cluster_id += 1
    
    # Merge undersized clusters
    cluster_sizes = user_df.groupby('cluster').size()
    undersized_clusters = cluster_sizes[cluster_sizes < _config['MIN_CLUSTER_SIZE']]
    
    for small_cluster_id, _ in undersized_clusters.items():
        small_cluster_users = user_df[user_df['cluster'] == small_cluster_id]
        if small_cluster_users.empty:
            continue
            
        small_center_lat = small_cluster_users['latitude'].mean()
        small_center_lon = small_cluster_users['longitude'].mean()
        
        # Find nearest larger cluster
        best_merge_cluster = None
        min_merge_distance = float('inf')
        
        for other_cluster_id in user_df['cluster'].unique():
            if other_cluster_id == small_cluster_id:
                continue
                
            other_cluster_users = user_df[user_df['cluster'] == other_cluster_id]
            if len(other_cluster_users) >= max_capacity:
                continue  # Don't merge into already full clusters
                
            other_center_lat = other_cluster_users['latitude'].mean()
            other_center_lon = other_cluster_users['longitude'].mean()
            
            distance = haversine_distance(
                small_center_lat, small_center_lon,
                other_center_lat, other_center_lon
            )
            
            if distance < min_merge_distance and distance <= _config['MAX_EXTRA_DISTANCE_KM']:
                min_merge_distance = distance
                best_merge_cluster = other_cluster_id
        
        # Perform merge if suitable cluster found
        if best_merge_cluster is not None:
            user_df.loc[user_df['cluster'] == small_cluster_id, 'cluster'] = best_merge_cluster
    
    # Re-index clusters to be sequential
    unique_clusters = sorted(user_df['cluster'].unique())
    cluster_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_clusters)}
    user_df['cluster'] = user_df['cluster'].map(cluster_mapping)
    
    return user_df


def calculate_road_distance(G, lat1, lon1, lat2, lon2):
    """Calculate shortest path distance between two points using road network; robust to OSMnx API differences."""
    try:
        # Try different nearest nodes signatures
        try:
            node1 = ox.nearest_nodes(G, lon1, lat1)
            node2 = ox.nearest_nodes(G, lon2, lat2)
        except TypeError:
            # newer OSMnx uses distance module
            try:
                from osmnx import distance as ox_distance  # newer packaging
                node1 = ox_distance.nearest_nodes(G, lon1, lat1)
                node2 = ox_distance.nearest_nodes(G, lon2, lat2)
            except Exception:
                # fallback to networkx approach if available
                node1 = ox.nearest_nodes(G, X=lon1, Y=lat1)
                node2 = ox.nearest_nodes(G, X=lon2, Y=lat2)

        # Calculate shortest path length in meters
        try:
            distance_m = ox.shortest_path_length(G, node1, node2, weight='length')
        except Exception:
            # networkx alternate call
            distance_m = nx.shortest_path_length(G, node1, node2, weight='length')

        return distance_m / 1000.0  # Convert to km
    except Exception:
        # Fallback to haversine if routing fails
        return haversine_distance(lat1, lon1, lat2, lon2)


def advanced_tsp_solver(driver_lat, driver_lon, users_df, G=None):
    """
    Advanced TSP solver using nearest neighbor + 2-opt improvements
    """
    if len(users_df) <= 1:
        users_list = users_df.to_dict('records') if hasattr(users_df, 'to_dict') else users_df
        for idx, user in enumerate(users_list):
            user['pickup_order'] = idx + 1
        return users_list
    
    # Convert to DataFrame if needed
    if not isinstance(users_df, pd.DataFrame):
        users_df = pd.DataFrame(users_df)
    
    users_list = users_df.to_dict('records')
    
    # Distance calculation function
    def get_distance(point1, point2):
        if G is not None and HAS_OSMNX:
            try:
                return calculate_road_distance(G, point1[0], point1[1], point2[0], point2[1])
            except:
                pass
        return haversine_distance(point1[0], point1[1], point2[0], point2[1])
    
    # Create distance matrix
    points = [(driver_lat, driver_lon)] + [(u['latitude'], u['longitude']) for u in users_list]
    n = len(points)
    
    distance_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                distance_matrix[i][j] = get_distance(points[i], points[j])
    
    # Nearest neighbor heuristic starting from driver (index 0)
    unvisited = set(range(1, n))  # Exclude driver position
    tour = [0]  # Start from driver
    current = 0
    
    while unvisited:
        nearest = min(unvisited, key=lambda x: distance_matrix[current][x])
        tour.append(nearest)
        unvisited.remove(nearest)
        current = nearest
    
    # 2-opt improvements
    def two_opt_distance(tour):
        return sum(distance_matrix[tour[i]][tour[i + 1]] for i in range(len(tour) - 1))
    
    improved = True
    best_tour = tour[:]
    best_distance = two_opt_distance(tour)
    
    max_iterations = min(100, len(users_list) * 2)  # Limit iterations for performance
    iterations = 0
    
    while improved and iterations < max_iterations:
        improved = False
        iterations += 1
        
        for i in range(1, len(tour) - 2):  # Skip driver position
            for j in range(i + 1, len(tour)):
                if j - i == 1:
                    continue  # Skip adjacent edges
                
                # Create new tour by reversing segment
                new_tour = tour[:i] + tour[i:j][::-1] + tour[j:]
                new_distance = two_opt_distance(new_tour)
                
                if new_distance < best_distance:
                    best_tour = new_tour
                    best_distance = new_distance
                    improved = True
        
        tour = best_tour[:]
    
    # Convert back to user list with pickup orders
    ordered_users = []
    for i in range(1, len(best_tour)):  # Skip driver position
        user_idx = best_tour[i] - 1  # Adjust for driver offset
        user = users_list[user_idx].copy()
        user['pickup_order'] = i
        ordered_users.append(user)
    
    print(f"   🛣️ TSP optimization: {best_distance:.2f}km total distance, {iterations} iterations")
    return ordered_users


def hungarian_driver_assignment(user_clusters, drivers_df):
    """
    Optimal driver-cluster assignment using Hungarian algorithm concepts
    """
    print("🎯 Multi-objective driver assignment with Hungarian optimization...")
    
    clusters = []
    for cluster_id in user_clusters['cluster'].unique():
        cluster_users = user_clusters[user_clusters['cluster'] == cluster_id]
        clusters.append({
            'id': cluster_id,
            'users': cluster_users,
            'size': len(cluster_users),
            'center_lat': cluster_users['latitude'].mean(),
            'center_lon': cluster_users['longitude'].mean(),
            'center_heading': calculate_circular_mean_heading(cluster_users['road_heading'].values)
        })
    
    # Create cost matrix: drivers x clusters
    n_drivers = len(drivers_df)
    n_clusters = len(clusters)
    
    if n_clusters == 0:
        return [], set()
    
    # Pad matrix to be square for Hungarian algorithm
    matrix_size = max(n_drivers, n_clusters)
    cost_matrix = np.full((matrix_size, matrix_size), 1e6)  # High cost for dummy assignments
    
    # Fill real costs
    for i, (_, driver) in enumerate(drivers_df.iterrows()):
        for j, cluster in enumerate(clusters):
            # Multi-objective cost calculation
            distance = haversine_distance(
                driver['latitude'], driver['longitude'],
                cluster['center_lat'], cluster['center_lon']
            )
            
            # Distance cost (normalized)
            distance_cost = min(distance / 10.0, 1.0)  # Cap at 10km
            
            # Capacity utilization cost (prefer full utilization)
            if cluster['size'] <= driver['capacity']:
                utilization_cost = 1.0 - (cluster['size'] / driver['capacity'])
            else:
                utilization_cost = 2.0  # Heavy penalty for overloading
            
            # Priority cost (lower priority = lower cost)
            priority_cost = (driver['priority'] - 1) * 0.1
            
            # Combined cost
            total_cost = distance_cost + utilization_cost + priority_cost
            cost_matrix[i][j] = total_cost
    
    # Apply Hungarian algorithm (using scipy's implementation)
    try:
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        # Extract valid assignments
        assignments = []
        assigned_driver_ids = set()
        assigned_cluster_ids = set()
        
        for driver_idx, cluster_idx in zip(row_indices, col_indices):
            if (driver_idx < n_drivers and cluster_idx < n_clusters and 
                cost_matrix[driver_idx][cluster_idx] < 1e5):  # Valid assignment
                
                driver = drivers_df.iloc[driver_idx]
                cluster = clusters[cluster_idx]
                
                # Check capacity constraint
                if cluster['size'] <= driver['capacity']:
                    assignments.append((driver, cluster))
                    assigned_driver_ids.add(driver['driver_id'])
                    assigned_cluster_ids.add(cluster['id'])
        
        print(f"   ✅ Hungarian algorithm: {len(assignments)} optimal assignments")
        return assignments, assigned_driver_ids
        
    except Exception as e:
        print(f"   ⚠️ Hungarian algorithm failed: {e}, using greedy fallback")
        return greedy_driver_assignment(clusters, drivers_df)


def greedy_driver_assignment(clusters, drivers_df):
    """
    Fallback greedy assignment when Hungarian algorithm fails
    """
    assignments = []
    assigned_driver_ids = set()
    
    # Sort clusters by size (largest first) and drivers by capacity (largest first)
    sorted_clusters = sorted(clusters, key=lambda c: c['size'], reverse=True)
    available_drivers = drivers_df.sort_values(['capacity', 'priority'], ascending=[False, True])
    
    for cluster in sorted_clusters:
        best_driver = None
        best_score = float('inf')
        
        for _, driver in available_drivers.iterrows():
            if driver['driver_id'] in assigned_driver_ids or cluster['size'] > driver['capacity']:
                continue
            
            distance = haversine_distance(
                driver['latitude'], driver['longitude'],
                cluster['center_lat'], cluster['center_lon']
            )
            
            # Simple scoring for greedy selection
            score = distance + (1.0 - cluster['size'] / driver['capacity']) * 5.0
            
            if score < best_score:
                best_score = score
                best_driver = driver
        
        if best_driver is not None:
            assignments.append((best_driver, cluster))
            assigned_driver_ids.add(best_driver['driver_id'])
    
    return assignments, assigned_driver_ids


def dynamic_route_balancing(routes):
    """
    Post-assignment route balancing to improve overall efficiency
    """
    print("⚖️ Dynamic route balancing for optimal utilization...")
    
    if len(routes) < 2:
        return routes
    
    # Identify underutilized and overutilized routes
    balancing_opportunities = []
    
    for i, route in enumerate(routes):
        capacity = route['vehicle_type']
        assigned = len(route['assigned_users'])
        utilization = assigned / capacity
        
        route['utilization'] = utilization
        route['available_capacity'] = capacity - assigned
        
        if utilization < 0.6 and route['available_capacity'] > 0:
            # Underutilized route
            balancing_opportunities.append(('underutilized', i, route))
        elif utilization > 0.9:
            # Well-utilized route (potential source for balancing)
            balancing_opportunities.append(('well_utilized', i, route))
    
    # Try to move users from well-utilized routes to underutilized ones
    transfers_made = 0
    
    underutilized_routes = [x for x in balancing_opportunities if x[0] == 'underutilized']
    
    for _, route_idx, under_route in underutilized_routes:
        if under_route['available_capacity'] <= 0:
            continue
            
        # Find nearby routes with users to potentially transfer
        route_lat = under_route['latitude']
        route_lon = under_route['longitude']
        
        for other_idx, other_route in enumerate(routes):
            if other_idx == route_idx or len(other_route['assigned_users']) <= 1:
                continue
            
            # Look for users in other routes that are closer to this underutilized route
            for user_idx, user in enumerate(other_route['assigned_users']):
                if under_route['available_capacity'] <= 0:
                    break
                
                # Calculate distances
                dist_to_under = haversine_distance(
                    user['latitude'], user['longitude'],
                    route_lat, route_lon
                )
                dist_to_current = haversine_distance(
                    user['latitude'], user['longitude'],
                    other_route['latitude'], other_route['longitude']
                )
                
                # Transfer if significantly closer and within reasonable distance
                if (dist_to_under < dist_to_current * 0.8 and 
                    dist_to_under <= _config['MAX_FILL_DISTANCE_KM']):
                    
                    # Remove from current route
                    transferred_user = other_route['assigned_users'].pop(user_idx)
                    
                    # Add to underutilized route
                    transferred_user['pickup_order'] = len(under_route['assigned_users']) + 1
                    under_route['assigned_users'].append(transferred_user)
                    under_route['available_capacity'] -= 1
                    
                    transfers_made += 1
                    break
    
    # Re-optimize pickup orders for routes that had transfers
    if transfers_made > 0:
        print(f"   🔄 Made {transfers_made} user transfers for better balance")
        
        # Re-optimize pickup orders for affected routes
        for route in routes:
            if len(route['assigned_users']) > 1:
                try:
                    # Get road graph for TSP
                    G = road_graph if HAS_OSMNX and road_graph is not None else None
                    
                    # Re-optimize pickup order
                    optimized_users = advanced_tsp_solver(
                        route['latitude'], route['longitude'],
                        route['assigned_users'], G
                    )
                    route['assigned_users'] = optimized_users
                    
                except Exception as e:
                    # Fallback: simple distance-based ordering
                    route['assigned_users'].sort(
                        key=lambda u: haversine_distance(
                            route['latitude'], route['longitude'],
                            u['latitude'], u['longitude']
                        )
                    )
                    for idx, user in enumerate(route['assigned_users']):
                        user['pickup_order'] = idx + 1
    
    # Remove utilization metadata
    for route in routes:
        route.pop('utilization', None)
        route.pop('available_capacity', None)
    
    return routes


def advanced_driver_cluster_assignment(user_df, driver_df):
    """
    Advanced assignment combining Hungarian optimization with dynamic balancing
    """
    print("🚗 Advanced multi-objective driver assignment...")
    
    # Step 1: Optimal assignment using Hungarian algorithm
    assignments, assigned_driver_ids = hungarian_driver_assignment(user_df, driver_df)
    
    # Step 2: Create initial routes
    routes = []
    assigned_user_ids = set()
    
    for driver, cluster in assignments:
        cluster_users = cluster['users']
        
        # Convert users to proper format
        users_list = []
        for _, user in cluster_users.iterrows():
            lat = float(user['latitude'])
            lng = float(user['longitude'])
            
            if not (-90 <= lat <= 90):
                lat = 0.0
            if not (-180 <= lng <= 180):
                lng = 0.0
            
            user_dict = user.to_dict()
            user_dict.update({
                'lat': lat,
                'lng': lng,
                'latitude': lat,
                'longitude': lng
            })
            users_list.append(user_dict)
        
        # Apply advanced TSP optimization
        try:
            G = road_graph if HAS_OSMNX and road_graph is not None else None
            optimized_users = advanced_tsp_solver(
                driver['latitude'], driver['longitude'],
                users_list, G
            )
        except Exception as e:
            print(f"   ⚠️ TSP optimization failed: {e}, using distance fallback")
            # Fallback: simple distance ordering
            users_list.sort(
                key=lambda u: haversine_distance(
                    driver['latitude'], driver['longitude'],
                    u['latitude'], u['longitude']
                )
            )
            for idx, user in enumerate(users_list):
                user['pickup_order'] = idx + 1
            optimized_users = users_list
        
        # Create route
        route_data = {
            'driver_id': str(driver['driver_id']),
            'latitude': float(driver['latitude']),
            'longitude': float(driver['longitude']),
            'vehicle_type': int(driver['capacity']),
            'vehicle_id': str(driver.get('vehicle_id', '')),
            'assigned_users': optimized_users,
            'road_way_id': optimized_users[0].get('way_id', 'N/A') if optimized_users else 'N/A',
            'road_heading': cluster['center_heading']
        }
        
        routes.append(route_data)
        
        # Track assigned users
        for user in optimized_users:
            assigned_user_ids.add(user['user_id'])
    
    # Step 3: Handle remaining unassigned users with enhanced fallback
    remaining_users = user_df[~user_df['user_id'].isin(assigned_user_ids)]
    remaining_drivers = driver_df[~driver_df['driver_id'].isin(assigned_driver_ids)]
    
    if len(remaining_users) > 0:
        routes, assigned_user_ids = enhanced_fallback_assignment(
            routes, remaining_users, remaining_drivers, assigned_user_ids, assigned_driver_ids
        )
    
    # Step 4: Apply dynamic route balancing
    routes = dynamic_route_balancing(routes)
    
    return routes, assigned_user_ids


def enhanced_fallback_assignment(routes, remaining_users, remaining_drivers, assigned_user_ids, assigned_driver_ids):
    """
    Enhanced fallback assignment for remaining users
    """
    print(f"   🔄 Enhanced fallback assignment for {len(remaining_users)} users...")
    
    # Strategy 1: Fill existing routes with available capacity
    for route in routes:
        if len(remaining_users) == 0:
            break
            
        available_capacity = route['vehicle_type'] - len(route['assigned_users'])
        if available_capacity <= 0:
            continue
        
        route_lat = route['latitude']
        route_lon = route['longitude']
        
        # Find nearest remaining users
        user_distances = []
        for _, user in remaining_users.iterrows():
            if user['user_id'] in assigned_user_ids:
                continue
                
            distance = haversine_distance(
                user['latitude'], user['longitude'],
                route_lat, route_lon
            )
            user_distances.append((distance, user))
        
        # Sort by distance and add closest users
        user_distances.sort(key=lambda x: x[0])
        users_to_add = min(available_capacity, len(user_distances))
        
        for i in range(users_to_add):
            distance, user = user_distances[i]
            
            if distance <= _config['MAX_FILL_DISTANCE_KM']:
                # Add user to route
                lat = float(user['latitude'])
                lng = float(user['longitude'])
                
                user_data = user.to_dict()
                user_data.update({
                    'lat': lat,
                    'lng': lng,
                    'latitude': lat,
                    'longitude': lng,
                    'pickup_order': len(route['assigned_users']) + 1
                })
                
                route['assigned_users'].append(user_data)
                assigned_user_ids.add(user['user_id'])
    
    # Strategy 2: Create new routes with remaining drivers
    remaining_users_after_fill = remaining_users[~remaining_users['user_id'].isin(assigned_user_ids)]
    
    if len(remaining_users_after_fill) > 0 and len(remaining_drivers) > 0:
        # Simple nearest assignment for remaining users
        for _, user in remaining_users_after_fill.iterrows():
            if user['user_id'] in assigned_user_ids:
                continue
            
            # Find nearest available driver
            best_driver = None
            min_distance = float('inf')
            
            for _, driver in remaining_drivers.iterrows():
                if driver['driver_id'] in assigned_driver_ids:
                    continue
                
                distance = haversine_distance(
                    user['latitude'], user['longitude'],
                    driver['latitude'], driver['longitude']
                )
                
                if distance < min_distance and distance <= _config['FALLBACK_MAX_DIST_KM']:
                    min_distance = distance
                    best_driver = driver
            
            if best_driver is not None:
                # Create new single-user route
                lat = float(user['latitude'])
                lng = float(user['longitude'])
                
                user_data = user.to_dict()
                user_data.update({
                    'lat': lat,
                    'lng': lng,
                    'latitude': lat,
                    'longitude': lng,
                    'pickup_order': 1
                })
                
                route_data = {
                    'driver_id': str(best_driver['driver_id']),
                    'latitude': float(best_driver['latitude']),
                    'longitude': float(best_driver['longitude']),
                    'vehicle_type': int(best_driver['capacity']),
                    'vehicle_id': str(best_driver.get('vehicle_id', '')),
                    'assigned_users': [user_data],
                    'road_way_id': user_data.get('way_id', 'N/A'),
                    'road_heading': user_data.get('road_heading', 0.0)
                }
                
                routes.append(route_data)
                assigned_user_ids.add(user['user_id'])
                assigned_driver_ids.add(best_driver['driver_id'])
    
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
        # Validate coordinates before adding
        lat = float(user['latitude']) if pd.notna(user['latitude']) else 0.0
        lng = float(user['longitude']) if pd.notna(user['longitude']) else 0.0
        
        # Ensure coordinates are within valid ranges
        if not (-90 <= lat <= 90):
            lat = 0.0
        if not (-180 <= lng <= 180):
            lng = 0.0
            
        user_data = {
            'user_id': str(user['user_id']),
            'latitude': lat,
            'longitude': lng,
            'lat': lat,  # Add both formats for compatibility
            'lng': lng,
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
    Advanced assignment function with multi-constraint clustering, optimal assignment, and dynamic balancing
    """
    start_time = time.time()

    try:
        print(f"🚀 Starting advanced multi-objective assignment for source_id: {source_id}")
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

        # Step 2: Advanced multi-constraint clustering
        driver_capacities = driver_df['capacity'].tolist()
        user_df = adaptive_dbscan_clustering(user_df, driver_capacities)

        # Step 3: Advanced driver assignment with Hungarian optimization
        routes, assigned_user_ids = advanced_driver_cluster_assignment(user_df, driver_df)

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

        # Calculate final statistics
        total_capacity = sum(route['vehicle_type'] for route in routes)
        total_assigned = sum(len(route['assigned_users']) for route in routes)
        overall_utilization = (total_assigned / total_capacity * 100) if total_capacity > 0 else 0

        print(f"✅ Advanced assignment complete in {execution_time:.2f}s")
        print(f"📊 Final routes: {len(routes)}")
        print(f"🎯 Users assigned: {len(assigned_user_ids)}")
        print(f"👥 Users unassigned: {len(user_df) - len(assigned_user_ids)}")
        print(f"⚡ Overall capacity utilization: {overall_utilization:.1f}%")

        method_name = "Advanced Multi-Constraint Optimization"
        if HAS_OSMNX:
            method_name += " with OSM Road Network"
        else:
            method_name += " with PCA Approximation"

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
    """Analyze the quality of the assignment with enhanced metrics"""
    if result["status"] != "true":
        return "Assignment failed"

    total_routes = len(result["data"])
    total_assigned = sum(len(route["assigned_users"]) for route in result["data"])
    total_unassigned = len(result["unassignedUsers"])
    total_capacity = sum(route["vehicle_type"] for route in result["data"])

    utilizations = []
    route_distances = []
    
    for route in result["data"]:
        if route["assigned_users"]:
            util = len(route["assigned_users"]) / route["vehicle_type"]
            utilizations.append(util)
            
            # Calculate average route distance
            if len(route["assigned_users"]) > 1:
                total_dist = 0
                for i in range(len(route["assigned_users"]) - 1):
                    user1 = route["assigned_users"][i]
                    user2 = route["assigned_users"][i + 1]
                    dist = haversine_distance(
                        user1.get('latitude', 0), user1.get('longitude', 0),
                        user2.get('latitude', 0), user2.get('longitude', 0)
                    )
                    total_dist += dist
                route_distances.append(total_dist)

    # Enhanced analysis metrics
    analysis = {
        "total_routes": total_routes,
        "total_assigned_users": total_assigned,
        "total_unassigned_users": total_unassigned,
        "total_capacity": total_capacity,
        "assignment_rate": round(total_assigned / (total_assigned + total_unassigned) * 100, 1) if (total_assigned + total_unassigned) > 0 else 0,
        "capacity_utilization": round(total_assigned / total_capacity * 100, 1) if total_capacity > 0 else 0,
        "avg_route_utilization": round(np.mean(utilizations) * 100, 1) if utilizations else 0,
        "avg_route_distance": round(np.mean(route_distances), 2) if route_distances else 0,
        "routes_above_80_percent": sum(1 for u in utilizations if u >= 0.8),
        "routes_below_50_percent": sum(1 for u in utilizations if u < 0.5),
        "clustering_method": result.get("clustering_analysis", {}).get("method", "Unknown"),
        "optimization_features": [
            "Multi-Constraint DBSCAN Clustering",
            "Hungarian Algorithm Assignment",
            "Advanced TSP with 2-opt",
            "Dynamic Route Balancing"
        ]
    }

    return analysis
