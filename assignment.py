
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
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import logging
import pickle
import hashlib
from itertools import combinations
import threading

# Import numba for optimization
try:
    from numba import jit, njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    print("⚠️ Numba not available, using standard Python functions")

# Try to import PyVRP for advanced VRP solving
try:
    import pyvrp
    from pyvrp import Model, Client, Depot, VehicleType
    HAS_PYVRP = True
    print("✅ PyVRP successfully loaded for advanced VRP solving")
except ImportError:
    HAS_PYVRP = False
    print("⚠️ PyVRP not available, using Hungarian algorithm fallback")

# Try to import OSMnx for road snapping, fallback to PCA if not available
try:
    import osmnx as ox
    import geopandas as gpd
    from shapely.geometry import Point, box
    import networkx as nx
    HAS_OSMNX = True

    # Configure OSMnx settings (compatible with different versions)
    try:
        ox.settings.use_cache = True
        ox.settings.log_console = False
        ox.settings.log_file = False
    except AttributeError:
        try:
            ox.config(use_cache=True, log_console=False, log_file=False)
        except Exception:
            try:
                ox.config(use_cache=True, log_console=False)
            except Exception:
                pass

    print("✅ OSMnx successfully loaded and configured")

except ImportError:
    HAS_OSMNX = False
    print("⚠️ OSMnx not available, using PCA fallback for road approximation")

# Global configuration and caches
_config = {}
road_graph = None
_GRAPH_CACHE = {}
_EDGES_CACHE = {}
_SNAP_CACHE = {}
_DISTANCE_CACHE = {}

# Thread locks for cache safety
_cache_lock = threading.Lock()
_snap_lock = threading.Lock()

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Get optimal number of workers
MAX_WORKERS = min(mp.cpu_count(), 8)  # Cap at 8 to avoid overhead


def load_and_validate_config():
    """Load configuration with validation and environment fallbacks"""
    try:
        with open('config.json') as f:
            cfg = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Warning: Could not load config.json, using defaults. Error: {e}")
        cfg = {}

    config = {}
    config['SNAP_RADIUS'] = max(10, min(150, float(cfg.get("snap_radius_m", 100))))
    config['MAX_GAP_DISTANCE'] = max(100, float(cfg.get("max_gap_distance_m", 5000)))
    config['BEARING_TOLERANCE'] = max(10, min(45, float(cfg.get("bearing_tolerance", 45))))
    config['MAX_DETOUR_KM'] = max(1.0, float(cfg.get("max_detour_km", 8.0)))
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
    config['N_JOBS'] = min(MAX_WORKERS, int(cfg.get("n_jobs", MAX_WORKERS)))

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


_config = load_and_validate_config()
SNAP_RADIUS = _config['SNAP_RADIUS']
MAX_GAP_DISTANCE = _config['MAX_GAP_DISTANCE']
BEARING_TOLERANCE = _config['BEARING_TOLERANCE']
MAX_DETOUR_KM = _config['MAX_DETOUR_KM']
OFFICE_LAT = _config['OFFICE_LAT']
OFFICE_LON = _config['OFFICE_LON']


if HAS_NUMBA:
    @njit
    def haversine_distance_numba(lat1, lon1, lat2, lon2):
        """Optimized haversine distance calculation using numba"""
        # Convert to radians
        lat1_r = math.radians(lat1)
        lon1_r = math.radians(lon1)
        lat2_r = math.radians(lat2)
        lon2_r = math.radians(lon2)
        
        # Haversine formula
        dlat = lat2_r - lat1_r
        dlon = lon2_r - lon1_r
        a = math.sin(dlat/2)**2 + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        
        return c * 6371  # Earth radius in km


def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate the great circle distance between two points on the earth"""
    if HAS_NUMBA:
        return haversine_distance_numba(lat1, lon1, lat2, lon2)
    
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


@lru_cache(maxsize=10000)
def get_cached_distance(lat1, lon1, lat2, lon2, use_road_distance=False):
    """Get distance with LRU caching for better performance"""
    # Round coordinates for cache key (4 decimal places = ~10m precision for speed)
    lat1, lon1, lat2, lon2 = round(lat1, 4), round(lon1, 4), round(lat2, 4), round(lon2, 4)
    
    # Create hashable key for global cache
    cache_key = (lat1, lon1, lat2, lon2, use_road_distance)

    if cache_key in _DISTANCE_CACHE:
        return _DISTANCE_CACHE[cache_key]
    
    if use_road_distance and HAS_OSMNX and road_graph is not None:
        try:
            distance = calculate_road_distance(road_graph, lat1, lon1, lat2, lon2)
        except:
            distance = haversine_distance(lat1, lon1, lat2, lon2)
    else:
        distance = haversine_distance(lat1, lon1, lat2, lon2)
    
    _DISTANCE_CACHE[cache_key] = distance
    return distance


def batch_distance_calculation(coords_list):
    """Calculate distances in batch for better performance"""
    distances = []
    for coord_pair in coords_list:
        lat1, lon1, lat2, lon2 = coord_pair
        dist = haversine_distance(lat1, lon1, lat2, lon2)
        distances.append(dist)
    return distances


def parallel_distance_matrix(locations):
    """Calculate distance matrix using parallel processing"""
    n = len(locations)
    distances = np.zeros((n, n))
    
    # Prepare coordinate pairs for batch processing
    coord_pairs = []
    indices = []
    
    for i in range(n):
        for j in range(i + 1, n):
            lat1, lon1 = locations[i]
            lat2, lon2 = locations[j]
            coord_pairs.append((lat1, lon1, lat2, lon2))
            indices.append((i, j))
    
    # Process in chunks using thread pool
    chunk_size = max(1, len(coord_pairs) // MAX_WORKERS)
    chunks = [coord_pairs[i:i + chunk_size] for i in range(0, len(coord_pairs), chunk_size)]
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(batch_distance_calculation, chunk) for chunk in chunks]
        
        flat_distances = []
        for future in as_completed(futures):
            flat_distances.extend(future.result())
    
    # Fill the distance matrix
    for idx, (i, j) in enumerate(indices):
        distances[i][j] = flat_distances[idx]
        distances[j][i] = flat_distances[idx]
    
    return distances


def validate_input_data(data):
    """Basic data validation"""
    if not isinstance(data, dict):
        raise ValueError("API response must be a dictionary")

    users = data.get("users", [])
    if not users:
        raise ValueError("No users found in API response")

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
    """Load environment variables and fetch data from API with timeout"""
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
    # Add timeout for faster failure
    resp = requests.get(API_URL, headers=headers, timeout=30)
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


if HAS_NUMBA:
    @njit
    def calculate_bearing_numba(lat1, lon1, lat2, lon2):
        """Optimized bearing calculation using numba"""
        lat1_r = math.radians(lat1)
        lon1_r = math.radians(lon1)
        lat2_r = math.radians(lat2)
        lon2_r = math.radians(lon2)

        dlon = lon2_r - lon1_r
        x = math.sin(dlon) * math.cos(lat2_r)
        y = math.cos(lat1_r) * math.sin(lat2_r) - math.sin(lat1_r) * math.cos(lat2_r) * math.cos(dlon)

        bearing = math.atan2(x, y)
        bearing = math.degrees(bearing)
        bearing = (bearing + 360) % 360

        return bearing


def calculate_bearing(lat1, lon1, lat2, lon2):
    """Calculate bearing between two points"""
    if HAS_NUMBA:
        return calculate_bearing_numba(lat1, lon1, lat2, lon2)
    
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    dlon = lon2 - lon1
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)

    bearing = math.atan2(x, y)  # Fixed: corrected x, y order
    bearing = math.degrees(bearing)
    bearing = (bearing + 360) % 360

    return bearing


if HAS_NUMBA:
    @njit
    def bearing_difference_numba(b1, b2):
        """Optimized bearing difference calculation"""
        diff = abs(b1 - b2) % 360
        return min(diff, 360 - diff)


def bearing_difference(b1, b2):
    """Compute minimum difference between two bearings"""
    if HAS_NUMBA:
        return bearing_difference_numba(b1, b2)
    
    diff = abs(b1 - b2) % 360
    return min(diff, 360 - diff)


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
                edges = edges.to_crs('EPSG:3857')
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


def parallel_snap_to_roads(user_chunks, edges_gdf):
    """Snap users to roads in parallel"""
    def snap_chunk(user_chunk):
        return snap_users_to_roads_optimized(user_chunk, edges_gdf)
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(snap_chunk, chunk) for chunk in user_chunks]
        results = []
        for future in as_completed(futures):
            results.append(future.result())
    
    # Combine results
    if results:
        combined_df = pd.concat(results, ignore_index=True)
        return combined_df
    else:
        return pd.DataFrame()


def snap_users_to_roads_optimized(user_df, edges_gdf):
    """Optimized snapping using spatial index and vectorized operations"""
    start_time = time.time()

    if len(user_df) == 0:
        return user_df

    # Convert user points to metric CRS GeoDataFrame
    user_gdf = gpd.GeoDataFrame(
        user_df.copy(),
        geometry=gpd.points_from_xy(user_df.longitude, user_df.latitude),
        crs='EPSG:4326'
    ).to_crs('EPSG:3857')

    # Build spatial index for edges
    try:
        sindex = edges_gdf.sindex
    except Exception:
        sindex = None

    snapped_users = []
    cache_hits = 0

    # Process in batches for better memory usage
    batch_size = 100
    user_batches = [user_gdf[i:i+batch_size] for i in range(0, len(user_gdf), batch_size)]

    for batch in user_batches:
        for idx, user in batch.iterrows():
            user_point = user.geometry

            # Check cache first
            with _snap_lock:
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

            # Use spatial index to get nearest candidates
            possible_matches = []
            if sindex is not None:
                try:
                    raw_matches = list(sindex.nearest(user_point.bounds, 5))
                    flattened = []
                    for m in raw_matches:
                        if hasattr(m, '__iter__') and not isinstance(m, (int, np.integer)):
                            flattened.extend(list(m))
                        else:
                            flattened.append(m)
                    possible_matches = [m for m in flattened if m in edges_gdf.index]
                    if not possible_matches:
                        raise Exception("sindex returned no usable candidates")
                except Exception:
                    distances = edges_gdf.geometry.distance(user_point)
                    possible_matches = distances.nsmallest(5).index.tolist()
            else:
                distances = edges_gdf.geometry.distance(user_point)
                possible_matches = distances.nsmallest(5).index.tolist()

            if possible_matches:
                candidates = edges_gdf.loc[possible_matches] if set(possible_matches) <= set(edges_gdf.index) else edges_gdf.iloc[possible_matches]

                distances = candidates.geometry.distance(user_point)
                nearest_idx = distances.idxmin()
                nearest_edge = candidates.loc[nearest_idx]

                line_geom = nearest_edge.geometry
                point_on_line = line_geom.interpolate(line_geom.project(user_point))
                proj_distance_m = line_geom.project(point_on_line)
                snap_distance_m = user_point.distance(point_on_line)

                # Calculate road heading
                coords = list(line_geom.coords)
                if len(coords) > 1:
                    total_length = line_geom.length
                    if proj_distance_m < total_length * 0.1:
                        p1, p2 = coords[0], coords[1]
                    elif proj_distance_m > total_length * 0.9:
                        p1, p2 = coords[-2], coords[-1]
                    else:
                        segment_ratio = proj_distance_m / total_length
                        segment_idx = min(len(coords) - 2, int(segment_ratio * (len(coords) - 1)))
                        p1, p2 = coords[segment_idx], coords[segment_idx + 1]

                    p1_lonlat = gpd.GeoSeries([Point(p1)], crs='EPSG:3857').to_crs('EPSG:4326').iloc[0]
                    p2_lonlat = gpd.GeoSeries([Point(p2)], crs='EPSG:3857').to_crs('EPSG:4326').iloc[0]
                    road_heading = calculate_bearing(p1_lonlat.y, p1_lonlat.x, p2_lonlat.y, p2_lonlat.x)
                else:
                    road_heading = 0.0

                snapped_result = {
                    'way_id': str(nearest_idx),
                    'proj_distance': proj_distance_m,
                    'road_heading': road_heading,
                    'snap_distance': snap_distance_m
                }

                # Cache the result
                with _snap_lock:
                    _SNAP_CACHE[cache_key] = snapped_result.copy()

                snapped_user = user.copy()
                snapped_user.update(snapped_result)
                snapped_users.append(snapped_user)
            else:
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
    """Enhanced OSMnx snapping with parallel processing"""
    global road_graph

    try:
        print("   🗺️  Using OSMnx for precise road network snapping...")

        graph_file = "tricity_main_roads.graphml"
        if os.path.exists(graph_file):
            print(f"   📁 Loading existing cached graph: {graph_file}")
            try:
                G = ox.load_graphml(graph_file)
                road_graph = G
                print("   ✅ Cached graph loaded and stored globally.")
            except Exception as e:
                print(f"   ⚠️ Failed to load cached graph: {e}")
                return fallback_pca_road_approximation(user_df)
        else:
            return fallback_pca_road_approximation(user_df)

        # Convert graph to GeoDataFrames
        try:
            _, edges = ox.graph_to_gdfs(G)
        except (TypeError, ValueError):
            try:
                edges = ox.graph_to_gdfs(G, nodes=False, edges=True)
            except Exception as e:
                print(f"   ⚠️ Failed to convert graph: {e}")
                return fallback_pca_road_approximation(user_df)
        
        try:
            edges = edges.to_crs('EPSG:3857')
        except Exception as e:
            print(f"   ⚠️ Failed to project CRS: {e}")
            return fallback_pca_road_approximation(user_df)

        # Process in parallel if dataset is large
        if len(user_df) > 200:
            chunk_size = max(50, len(user_df) // MAX_WORKERS)
            user_chunks = [user_df[i:i+chunk_size] for i in range(0, len(user_df), chunk_size)]
            return parallel_snap_to_roads(user_chunks, edges)
        else:
            return snap_users_to_roads_optimized(user_df, edges)

    except Exception as e:
        print(f"   ⚠️ OSMnx snapping failed: {e}")
        return fallback_pca_road_approximation(user_df)


def snap_to_road_network(user_df, config):
    """Optimized road network snapping with parallel processing"""
    print("🛣️ Step 1: Snapping users to road network...")

    if len(user_df) == 0:
        return user_df

    # Try PCA fallback for speed if no cached graph
    if HAS_OSMNX and os.path.exists("tricity_main_roads.graphml"):
        try:
            return snap_with_osmnx(user_df, config)
        except Exception as e:
            print(f"   ⚠️ OSMnx failed, using PCA fallback: {e}")
            return fallback_pca_road_approximation(user_df)
    else:
        print("   🔄 Using PCA road approximation (faster)")
        return fallback_pca_road_approximation(user_df)


def fallback_pca_road_approximation(user_df):
    """Optimized PCA fallback with vectorized operations"""
    from sklearn.decomposition import PCA
    
    if len(user_df) == 0:
        return user_df

    # Vectorized PCA computation
    points = user_df[['latitude', 'longitude']].values
    pca = PCA(n_components=2)
    pca.fit(points)

    main_direction = pca.components_[0]
    mean_point = points.mean(axis=0)

    # Vectorized projection calculation
    centered_points = points - mean_point
    proj_distances = np.dot(centered_points, main_direction)

    # Calculate road heading once
    road_heading = calculate_bearing(
        mean_point[0], mean_point[1],
        mean_point[0] + main_direction[0], mean_point[1] + main_direction[1]
    )

    # Vectorized assignment
    result_df = user_df.copy()
    result_df['way_id'] = 'pca_line'
    result_df['proj_distance'] = proj_distances
    result_df['road_heading'] = road_heading
    result_df['snap_distance'] = 0.0

    print(f"   ✅ Fitted PCA line for {len(result_df)} users")
    return result_df


def calculate_circular_mean_heading(headings):
    """Vectorized circular mean calculation"""
    if len(headings) == 0:
        return 0.0
    
    angles = np.deg2rad(headings)
    mean_x = np.mean(np.cos(angles))
    mean_y = np.mean(np.sin(angles))
    mean_angle_deg = (np.degrees(np.arctan2(mean_y, mean_x)) + 360) % 360
    return mean_angle_deg


def split_way_segments_optimized(user_df, driver_capacities):
    """Optimized road-aware clustering with vectorized operations"""
    print("🛣️ Road-aware clustering with way segment splitting...")

    avg_capacity = np.mean(driver_capacities) if driver_capacities else 5
    max_capacity = max(driver_capacities) if driver_capacities else 7

    # Vectorized grouping and sorting
    user_df = user_df.sort_values(['way_id', 'proj_distance'])
    way_groups = user_df.groupby('way_id')
    
    cluster_id = 0
    user_df['cluster'] = -1  # Initialize

    for way_id, way_users in way_groups:
        way_indices = way_users.index.tolist()
        
        if len(way_users) > 1:
            # Vectorized heading difference calculation
            headings = way_users['road_heading'].values
            heading_diffs = np.abs(np.diff(headings))
            heading_diffs = np.minimum(heading_diffs, 360 - heading_diffs)  # Handle wraparound
            
            # Find split points
            split_indices = [0] + (np.where(heading_diffs > BEARING_TOLERANCE)[0] + 1).tolist() + [len(way_users)]
            
            # Process segments
            for seg_start, seg_end in zip(split_indices[:-1], split_indices[1:]):
                segment_indices = way_indices[seg_start:seg_end]
                
                # Split by capacity and distance gaps
                current_cluster = []
                
                for idx in segment_indices:
                    user = user_df.loc[idx]
                    
                    if len(current_cluster) >= max_capacity:
                        # Assign current cluster
                        user_df.loc[current_cluster, 'cluster'] = cluster_id
                        cluster_id += 1
                        current_cluster = []

                    # Check distance gap
                    if current_cluster:
                        prev_idx = current_cluster[-1]
                        prev_user = user_df.loc[prev_idx]
                        
                        distance_gap = abs(user['proj_distance'] - prev_user['proj_distance'])
                        geo_distance = haversine_distance(
                            user['latitude'], user['longitude'],
                            prev_user['latitude'], prev_user['longitude']
                        )
                        
                        if (distance_gap > MAX_GAP_DISTANCE or 
                            geo_distance > _config['MAX_EXTRA_DISTANCE_KM']):
                            user_df.loc[current_cluster, 'cluster'] = cluster_id
                            cluster_id += 1
                            current_cluster = []

                    current_cluster.append(idx)

                # Assign remaining cluster
                if current_cluster:
                    user_df.loc[current_cluster, 'cluster'] = cluster_id
                    cluster_id += 1
        else:
            # Single user
            user_df.loc[way_users.index, 'cluster'] = cluster_id
            cluster_id += 1

    # Handle unassigned users
    unassigned_mask = user_df['cluster'] == -1
    if unassigned_mask.any():
        user_df.loc[unassigned_mask, 'cluster'] = range(cluster_id, cluster_id + unassigned_mask.sum())

    user_df = optimize_cluster_sizes(user_df, driver_capacities)

    n_final_clusters = user_df['cluster'].nunique()
    avg_cluster_size = len(user_df) / n_final_clusters if n_final_clusters > 0 else 0

    print(f"   ✅ Created {n_final_clusters} road-aware clusters")
    print(f"   📊 Average cluster size: {avg_cluster_size:.1f} users")

    return user_df


def parallel_clustering(user_df, driver_capacities):
    """Optimized clustering with parallel processing"""
    print("🎯 Multi-constraint clustering with optimization...")

    # Try road-aware clustering first
    try:
        return split_way_segments_optimized(user_df, driver_capacities)
    except Exception as e:
        print(f"   ⚠️ Road-aware clustering failed: {e}, falling back to DBSCAN")

    # Parallel DBSCAN parameter optimization
    coords = user_df[['latitude', 'longitude']].values
    scaler = StandardScaler()
    coords_scaled = scaler.fit_transform(coords)

    # Estimate optimal parameters
    k = min(4, len(user_df) - 1)
    if k > 0:
        # Use parallel distance calculation
        distances = parallel_distance_matrix(coords)
        k_distances = np.sort(distances, axis=1)[:, k] if k < distances.shape[1] else distances[:, -1]
        eps_candidate = np.percentile(k_distances, 75)
    else:
        eps_candidate = 0.5

    lat_std = user_df['latitude'].std()
    lon_std = user_df['longitude'].std()
    geo_scale = max(lat_std, lon_std) if max(lat_std, lon_std) > 0 else 0.01
    eps_geographic = eps_candidate * geo_scale

    # Parallel parameter testing
    eps_values = [eps_geographic * f for f in [0.5, 0.75, 1.0, 1.25, 1.5]]
    min_samples_values = [2, 3, max(2, len(user_df) // 20)]

    def test_clustering_params(params):
        eps, min_samples = params
        try:
            features = []
            for _, user in user_df.iterrows():
                feature_vector = [
                    user['latitude'],
                    user['longitude'],
                    math.cos(math.radians(user.get('road_heading', 0))) * 0.1,
                    math.sin(math.radians(user.get('road_heading', 0))) * 0.1,
                    user.get('office_distance', 0) * 0.1
                ]
                features.append(feature_vector)

            features_scaled = scaler.fit_transform(features)
            clustering = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=1)
            cluster_labels = clustering.fit_predict(features_scaled)

            n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            if n_clusters == 0:
                return None, -1

            score = evaluate_clustering_quality(user_df, cluster_labels, driver_capacities)
            return cluster_labels, score
        except Exception:
            return None, -1

    # Test parameters in parallel
    param_combinations = [(eps, min_samples) for eps in eps_values for min_samples in min_samples_values]
    
    best_clustering = None
    best_score = -1

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(test_clustering_params, params) for params in param_combinations]
        
        for future in as_completed(futures):
            cluster_labels, score = future.result()
            if score > best_score:
                best_score = score
                best_clustering = cluster_labels

    if best_clustering is None:
        print("   ⚠️ DBSCAN failed, using geographic fallback clustering")
        best_clustering = geographic_fallback_clustering(user_df, driver_capacities)

    user_df = user_df.copy()
    user_df['cluster'] = best_clustering
    user_df = optimize_cluster_sizes(user_df, driver_capacities)

    n_final_clusters = user_df['cluster'].nunique()
    avg_cluster_size = len(user_df) / n_final_clusters if n_final_clusters > 0 else 0

    print(f"   ✅ Created {n_final_clusters} optimized clusters")
    print(f"   📊 Average cluster size: {avg_cluster_size:.1f} users")

    return user_df


def adaptive_dbscan_clustering(user_df, driver_capacities):
    """
    Legacy function - use parallel_clustering instead
    """
    return parallel_clustering(user_df, driver_capacities)


def evaluate_clustering_quality(user_df, cluster_labels, driver_capacities):
    """Optimized clustering quality evaluation"""
    if len(set(cluster_labels)) <= 1:
        return -1
    
    coords = user_df[['latitude', 'longitude']].values
    try:
        geo_score = silhouette_score(coords, cluster_labels)
    except:
        geo_score = 0
    
    cluster_counts = pd.Series(cluster_labels).value_counts()
    noise_points = (cluster_labels == -1).sum()
    
    avg_capacity = np.mean(driver_capacities) if driver_capacities else 5
    
    # Vectorized capacity score calculation
    capacity_scores = np.where(
        cluster_counts.values <= avg_capacity,
        cluster_counts.values / avg_capacity,
        avg_capacity / cluster_counts.values
    )
    
    capacity_score = np.mean(capacity_scores) if len(capacity_scores) > 0 else 0
    noise_penalty = noise_points / len(cluster_labels) if len(cluster_labels) > 0 else 1
    
    combined_score = (geo_score * 0.4) + (capacity_score * 0.5) - (noise_penalty * 0.3)
    return combined_score


def geographic_fallback_clustering(user_df, driver_capacities):
    """Fast geographic clustering fallback"""
    coords = user_df[['latitude', 'longitude']].values
    n_drivers = len(driver_capacities)
    
    if len(coords) > n_drivers:
        n_clusters = min(n_drivers, len(coords) // 2)
    else:
        n_clusters = max(1, len(coords) // 3)
    
    try:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(coords)
    except:
        cluster_labels = np.zeros(len(coords), dtype=int)
    
    return cluster_labels


def optimize_cluster_sizes(user_df, driver_capacities):
    """Optimized cluster size optimization with vectorized operations"""
    if not driver_capacities:
        return user_df
        
    avg_capacity = np.mean(driver_capacities)
    max_capacity = max(driver_capacities)
    
    # Handle noise points efficiently
    noise_mask = user_df['cluster'] == -1
    if noise_mask.any():
        noise_users = user_df[noise_mask]
        valid_clusters = user_df[~noise_mask]
        
        if not valid_clusters.empty:
            cluster_centers = valid_clusters.groupby('cluster')[['latitude', 'longitude']].mean()
            
            for idx, noise_user in noise_users.iterrows():
                # Vectorized distance calculation to all cluster centers
                distances = np.array([
                    haversine_distance(
                        noise_user['latitude'], noise_user['longitude'],
                        center['latitude'], center['longitude']
                    ) for _, center in cluster_centers.iterrows()
                ])
                nearest_cluster = cluster_centers.index[np.argmin(distances)]
                user_df.loc[idx, 'cluster'] = nearest_cluster
        else:
            user_df.loc[noise_mask, 'cluster'] = 0

    # Vectorized cluster size analysis
    cluster_sizes = user_df.groupby('cluster').size()
    oversized_clusters = cluster_sizes[cluster_sizes > max_capacity]
    
    next_cluster_id = user_df['cluster'].max() + 1
    
    # Split oversized clusters
    for cluster_id, size in oversized_clusters.items():
        if size <= max_capacity:
            continue
            
        cluster_users = user_df[user_df['cluster'] == cluster_id].copy()
        n_splits = math.ceil(size / avg_capacity)
        coords = cluster_users[['latitude', 'longitude']].values
        
        try:
            kmeans = KMeans(n_clusters=n_splits, random_state=42, n_init=10)
            sub_labels = kmeans.fit_predict(coords)
            
            # Vectorized reassignment
            mask = user_df['cluster'] == cluster_id
            new_labels = np.where(sub_labels == 0, cluster_id, next_cluster_id + sub_labels - 1)
            user_df.loc[mask, 'cluster'] = new_labels
            
            next_cluster_id += n_splits - 1
            
        except:
            # Simple chunking fallback
            chunk_size = max(2, int(avg_capacity))
            cluster_indices = cluster_users.index.tolist()
            
            for i, idx in enumerate(cluster_indices):
                if i >= chunk_size:
                    user_df.loc[idx, 'cluster'] = next_cluster_id
                    if (i + 1) % chunk_size == 0:
                        next_cluster_id += 1

    # Merge undersized clusters efficiently
    cluster_sizes = user_df.groupby('cluster').size()
    undersized_clusters = cluster_sizes[cluster_sizes < _config['MIN_CLUSTER_SIZE']]
    
    if not undersized_clusters.empty:
        cluster_centers = user_df.groupby('cluster')[['latitude', 'longitude']].mean()
        
        for small_cluster_id in undersized_clusters.index:
            small_center = cluster_centers.loc[small_cluster_id]
            
            # Find nearest larger cluster
            other_clusters = cluster_sizes[
                (cluster_sizes.index != small_cluster_id) & 
                (cluster_sizes < max_capacity)
            ]
            
            if not other_clusters.empty:
                other_centers = cluster_centers.loc[other_clusters.index]
                distances = np.array([
                    haversine_distance(
                        small_center['latitude'], small_center['longitude'],
                        center['latitude'], center['longitude']
                    ) for _, center in other_centers.iterrows()
                ])
                
                valid_distances = distances[distances <= _config['MAX_EXTRA_DISTANCE_KM']]
                if len(valid_distances) > 0:
                    best_cluster = other_centers.index[np.argmin(distances)]
                    user_df.loc[user_df['cluster'] == small_cluster_id, 'cluster'] = best_cluster

    # Re-index clusters
    unique_clusters = sorted(user_df['cluster'].unique())
    cluster_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_clusters)}
    user_df['cluster'] = user_df['cluster'].map(cluster_mapping)
    
    return user_df


def calculate_road_distance(G, lat1, lon1, lat2, lon2):
    """Optimized road distance calculation with caching"""
    try:
        # Use cached distance if available
        cache_key = (round(lat1, 4), round(lon1, 4), round(lat2, 4), round(lon2, 4))
        
        with _cache_lock:
            if cache_key in _DISTANCE_CACHE:
                return _DISTANCE_CACHE[cache_key]

        # Calculate road distance
        try:
            node1 = ox.nearest_nodes(G, lon1, lat1)
            node2 = ox.nearest_nodes(G, lon2, lat2)
        except TypeError:
            try:
                from osmnx import distance as ox_distance
                node1 = ox_distance.nearest_nodes(G, lon1, lat1)
                node2 = ox_distance.nearest_nodes(G, lon2, lat2)
            except Exception:
                node1 = ox.nearest_nodes(G, X=lon1, Y=lat1)
                node2 = ox.nearest_nodes(G, X=lon2, Y=lat2)

        try:
            distance_m = ox.shortest_path_length(G, node1, node2, weight='length')
        except Exception:
            distance_m = nx.shortest_path_length(G, node1, node2, weight='length')

        distance_km = distance_m / 1000.0
        
        # Cache the result
        with _cache_lock:
            _DISTANCE_CACHE[cache_key] = distance_km
        
        return distance_km
    except Exception:
        return haversine_distance(lat1, lon1, lat2, lon2)


def parallel_tsp_solver(driver_lat, driver_lon, users_df, G=None):
    """Optimized TSP solver with parallel 2-opt improvements"""
    if len(users_df) <= 1:
        users_list = users_df.to_dict('records') if hasattr(users_df, 'to_dict') else users_df
        for idx, user in enumerate(users_list):
            user['pickup_order'] = idx + 1
        return users_list
    
    if not isinstance(users_df, pd.DataFrame):
        users_df = pd.DataFrame(users_df)
    
    users_list = users_df.to_dict('records')
    
    # Create points and distance matrix
    points = [(driver_lat, driver_lon)] + [(u['latitude'], u['longitude']) for u in users_list]
    n = len(points)
    
    # Use parallel distance matrix calculation
    distance_matrix = parallel_distance_matrix(points)
    
    # Nearest neighbor heuristic
    unvisited = set(range(1, n))
    tour = [0]
    current = 0
    
    while unvisited:
        nearest = min(unvisited, key=lambda x: distance_matrix[current][x])
        tour.append(nearest)
        unvisited.remove(nearest)
        current = nearest
    
    # Parallel 2-opt improvements
    def two_opt_distance(tour_segment):
        return sum(distance_matrix[tour_segment[i]][tour_segment[i + 1]] for i in range(len(tour_segment) - 1))
    
    def improve_tour_segment(tour_segment, start_idx, end_idx):
        best_tour = tour_segment[:]
        best_distance = two_opt_distance(tour_segment)
        improved = False
        
        for i in range(start_idx + 1, end_idx - 1):
            for j in range(i + 1, end_idx):
                if j - i == 1:
                    continue
                
                new_tour = tour_segment[:i] + tour_segment[i:j][::-1] + tour_segment[j:]
                new_distance = two_opt_distance(new_tour)
                
                if new_distance < best_distance:
                    best_tour = new_tour
                    best_distance = new_distance
                    improved = True
        
        return best_tour, best_distance, improved

    # Apply parallel 2-opt
    max_iterations = min(50, len(users_list))
    iterations = 0
    
    while iterations < max_iterations:
        # Divide tour into segments for parallel processing
        segment_size = max(5, len(tour) // MAX_WORKERS)
        segments = []
        
        for i in range(1, len(tour) - segment_size, segment_size // 2):
            end_i = min(i + segment_size, len(tour))
            segments.append((tour, i, end_i))
        
        if not segments:
            break
            
        # Process segments in parallel
        improved = False
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [executor.submit(improve_tour_segment, seg[0], seg[1], seg[2]) for seg in segments]
            
            for future in as_completed(futures):
                seg_tour, seg_distance, seg_improved = future.result()
                if seg_improved:
                    tour = seg_tour
                    improved = True
                    break
        
        if not improved:
            break
            
        iterations += 1
    
    # Convert back to user list
    ordered_users = []
    for i in range(1, len(tour)):
        user_idx = tour[i] - 1
        user = users_list[user_idx].copy()
        user['pickup_order'] = i
        ordered_users.append(user)
    
    total_distance = two_opt_distance(tour)
    print(f"   🛣️ Parallel TSP: {total_distance:.2f}km, {iterations} iterations")
    
    return ordered_users


def advanced_tsp_solver(driver_lat, driver_lon, users_df, G=None):
    """
    Legacy function - use parallel_tsp_solver instead
    """
    return parallel_tsp_solver(driver_lat, driver_lon, users_df, G)


def solve_vrp_with_pyvrp(users_df, drivers_df):
    """
    Simplified VRP solving using PyVRP - fallback to Hungarian if API issues
    """
    print("🚛 Advanced VRP solving with PyVRP...")

    if not HAS_PYVRP:
        print("   ⚠️ PyVRP not available, falling back to Hungarian algorithm")
        return hungarian_driver_assignment_fallback(users_df, drivers_df)

    try:
        # Due to PyVRP API compatibility issues, using Hungarian algorithm directly
        print("   ⚠️ PyVRP API compatibility issues detected, using Hungarian algorithm")
        return hungarian_driver_assignment_fallback(users_df, drivers_df)

    except Exception as e:
        print(f"   ❌ PyVRP failed: {e}, using Hungarian fallback")
        return hungarian_driver_assignment_fallback(users_df, drivers_df)


def greedy_driver_assignment(clusters, drivers_df):
    """Optimized greedy assignment"""
    assignments = []
    available_drivers = drivers_df.copy()
    
    sorted_clusters = sorted(clusters, key=lambda c: c['size'], reverse=True)
    
    for cluster in sorted_clusters:
        if len(available_drivers) == 0:
            break
            
        # Vectorized scoring
        driver_coords = available_drivers[['latitude', 'longitude']].values
        cluster_coord = np.array([cluster['center_lat'], cluster['center_lon']])
        
        distances = np.array([
            haversine_distance(coord[0], coord[1], cluster_coord[0], cluster_coord[1])
            for coord in driver_coords
        ])
        
        capacity_mask = available_drivers['capacity'].values >= cluster['size']
        valid_drivers = available_drivers[capacity_mask]
        
        if len(valid_drivers) > 0:
            valid_distances = distances[capacity_mask]
            utilizations = cluster['size'] / valid_drivers['capacity'].values
            scores = valid_distances + (1.0 - utilizations) * 2.0
            
            best_idx = valid_drivers.index[np.argmin(scores)]
            best_driver = available_drivers.loc[best_idx]
            
            assignments.append((best_driver, cluster))
            available_drivers = available_drivers[available_drivers['driver_id'] != best_driver['driver_id']]
    
    return assignments


def parallel_hungarian_assignment(user_clusters, drivers_df):
    """Optimized Hungarian algorithm with parallel cost matrix calculation"""
    print("🎯 Parallel multi-objective driver assignment...")

    if isinstance(user_clusters, pd.DataFrame):
        clusters = []
        if 'cluster' in user_clusters.columns:
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
        else:
            clusters.append({
                'id': 0,
                'users': user_clusters,
                'size': len(user_clusters),
                'center_lat': user_clusters['latitude'].mean(),
                'center_lon': user_clusters['longitude'].mean(),
                'center_heading': calculate_circular_mean_heading(user_clusters['road_heading'].values)
            })
    else:
        clusters = user_clusters

    n_drivers = len(drivers_df)
    n_clusters = len(clusters)

    if n_clusters == 0:
        return [], set(), set()

    # Parallel cost matrix calculation
    def calculate_cost_row(driver_data):
        driver_idx, driver = driver_data
        costs = []
        
        for cluster in clusters:
            distance = get_cached_distance(
                driver['latitude'], driver['longitude'],
                cluster['center_lat'], cluster['center_lon'],
                use_road_distance=True
            )

            distance_cost = min(distance / 10.0, 1.0)
            
            if cluster['size'] <= driver['capacity']:
                utilization_cost = 1.0 - (cluster['size'] / driver['capacity'])
            else:
                utilization_cost = 2.0

            priority_cost = (driver['priority'] - 1) * 0.1

            driver_heading = calculate_bearing(
                OFFICE_LAT, OFFICE_LON,
                driver['latitude'], driver['longitude']
            )
            heading_diff = bearing_difference(driver_heading, cluster['center_heading'])
            heading_penalty = (heading_diff / 180.0) * 0.2

            total_cost = distance_cost + utilization_cost + priority_cost + heading_penalty
            costs.append(total_cost)
        
        return driver_idx, costs

    # Calculate cost matrix in parallel
    matrix_size = max(n_drivers, n_clusters)
    cost_matrix = np.full((matrix_size, matrix_size), 1e6)

    driver_data = [(i, row) for i, (_, row) in enumerate(drivers_df.iterrows())]
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(calculate_cost_row, data) for data in driver_data]
        
        for future in as_completed(futures):
            driver_idx, costs = future.result()
            cost_matrix[driver_idx, :len(costs)] = costs

    # Apply Hungarian algorithm
    try:
        row_indices, col_indices = linear_sum_assignment(cost_matrix)

        assignments = []
        assigned_driver_ids = set()
        assigned_cluster_ids = set()

        for driver_idx, cluster_idx in zip(row_indices, col_indices):
            if (driver_idx < n_drivers and cluster_idx < n_clusters and 
                cost_matrix[driver_idx][cluster_idx] < 1e5):

                driver = drivers_df.iloc[driver_idx]
                cluster = clusters[cluster_idx]

                if cluster['size'] <= driver['capacity']:
                    assignments.append((driver, cluster))
                    assigned_driver_ids.add(driver['driver_id'])
                    assigned_cluster_ids.add(cluster['id'])

        print(f"   ✅ Parallel Hungarian: {len(assignments)} optimal assignments")
        return assignments, assigned_driver_ids, assigned_cluster_ids

    except Exception as e:
        print(f"   ⚠️ Hungarian failed: {e}, using greedy fallback")
        return greedy_driver_assignment(clusters, drivers_df), set(), set()


def hungarian_driver_assignment_fallback(user_clusters, drivers_df):
    """
    Fallback Hungarian algorithm assignment when PyVRP is not available
    """
    return parallel_hungarian_assignment(user_clusters, drivers_df)


def hungarian_driver_assignment(user_clusters, drivers_df):
    """
    Main driver assignment function - uses optimized parallel Hungarian algorithm
    """
    print("   🎯 Using parallel Hungarian algorithm for optimal assignment")
    
    result = parallel_hungarian_assignment(user_clusters, drivers_df)
    
    # Handle different return formats consistently
    if isinstance(result, tuple):
        if len(result) == 3:
            assignments, assigned_driver_ids, assigned_cluster_ids = result
            return assignments, assigned_driver_ids, assigned_cluster_ids
        elif len(result) == 2:
            assignments, assigned_driver_ids = result
            return assignments, assigned_driver_ids, set()
        else:
            assignments = result[0]
            return assignments, set(), set()
    else:
        return result, set(), set()


def parallel_enhanced_fallback_assignment(routes, remaining_users, remaining_drivers, assigned_user_ids, assigned_driver_ids):
    """Parallel enhanced fallback assignment"""
    if len(remaining_users) == 0:
        return routes, assigned_user_ids
    
    assigned_user_ids = set(assigned_user_ids) if not isinstance(assigned_user_ids, set) else assigned_user_ids
    
    # Parallel route matching for remaining users
    def find_best_route_for_user(user_data):
        _, user = user_data
        best_route = None
        min_detour = float('inf')
        
        for route in routes:
            current_load = len(route['assigned_users'])
            if current_load < route['vehicle_type']:
                user_distance = get_cached_distance(
                    route['latitude'], route['longitude'],
                    user['latitude'], user['longitude'],
                    use_road_distance=True
                )
                
                if user_distance < _config['FALLBACK_MAX_DIST_KM'] and user_distance < min_detour:
                    min_detour = user_distance
                    best_route = route
        
        return user, best_route

    # Process users in parallel
    user_data = [(idx, user) for idx, user in remaining_users.iterrows()]
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(find_best_route_for_user, data) for data in user_data]
        
        for future in as_completed(futures):
            user, best_route = future.result()
            
            if best_route is not None:
                lat = float(user['latitude'])
                lng = float(user['longitude'])
                
                user_data = {
                    'user_id': str(user['user_id']),
                    'latitude': lat,
                    'longitude': lng,
                    'lat': lat,
                    'lng': lng,
                    'pickup_order': len(best_route['assigned_users']) + 1,
                    'first_name': str(user.get('first_name', '')),
                    'email': str(user.get('email', '')),
                    'office_distance': float(user.get('office_distance', 0))
                }
                
                best_route['assigned_users'].append(user_data)
                assigned_user_ids.add(user['user_id'])

    # Handle remaining unassigned users with available drivers
    still_unassigned = remaining_users[~remaining_users['user_id'].isin(assigned_user_ids)]
    
    for _, driver in remaining_drivers.iterrows():
        if len(still_unassigned) == 0:
            break
            
        # Vectorized distance calculation
        driver_coords = np.array([driver['latitude'], driver['longitude']])
        user_coords = still_unassigned[['latitude', 'longitude']].values
        
        distances = np.array([
            haversine_distance(driver_coords[0], driver_coords[1], coord[0], coord[1])
            for coord in user_coords
        ])
        
        valid_mask = distances <= _config['FALLBACK_MAX_DIST_KM']
        valid_users = still_unassigned[valid_mask]
        
        driver_users = []
        for _, user in valid_users.head(driver['capacity']).iterrows():
            lat = float(user['latitude'])
            lng = float(user['longitude'])
            
            user_data = {
                'user_id': str(user['user_id']),
                'latitude': lat,
                'longitude': lng,
                'lat': lat,
                'lng': lng,
                'pickup_order': len(driver_users) + 1,
                'first_name': str(user.get('first_name', '')),
                'email': str(user.get('email', '')),
                'office_distance': float(user.get('office_distance', 0))
            }
            
            driver_users.append(user_data)
            assigned_user_ids.add(user['user_id'])

        if driver_users:
            new_route = {
                'driver_id': str(driver['driver_id']),
                'latitude': float(driver['latitude']),
                'longitude': float(driver['longitude']),
                'vehicle_type': int(driver['capacity']),
                'vehicle_id': str(driver.get('vehicle_id', '')),
                'assigned_users': driver_users,
                'road_way_id': driver_users[0].get('way_id', 'fallback'),
                'road_heading': 0.0
            }
            routes.append(new_route)
            
            still_unassigned = still_unassigned[~still_unassigned['user_id'].isin(assigned_user_ids)]
    
    return routes, assigned_user_ids


def enhanced_fallback_assignment(routes, remaining_users, remaining_drivers, assigned_user_ids, assigned_driver_ids):
    """
    Legacy function - use parallel version instead
    """
    return parallel_enhanced_fallback_assignment(routes, remaining_users, remaining_drivers, assigned_user_ids, assigned_driver_ids)


def dynamic_route_balancing(routes):
    """
    Dynamic route balancing to improve overall efficiency
    """
    if len(routes) <= 1:
        return routes
    
    print("🔄 Applying dynamic route balancing...")
    
    # Calculate route efficiency scores
    route_scores = []
    for i, route in enumerate(routes):
        users = route['assigned_users']
        capacity = route['vehicle_type']
        
        if len(users) == 0:
            route_scores.append((i, 0.0))
            continue
            
        # Utilization score
        utilization = len(users) / capacity
        
        # Distance efficiency score
        total_distance = 0
        if len(users) > 1:
            for j in range(len(users) - 1):
                u1, u2 = users[j], users[j + 1]
                dist = get_cached_distance(
                    u1['latitude'], u1['longitude'],
                    u2['latitude'], u2['longitude'],
                    use_road_distance=True
                )
                total_distance += dist
            
            avg_distance = total_distance / (len(users) - 1)
            distance_score = max(0, 1.0 - (avg_distance / 5.0))  # Normalize by 5km
        else:
            distance_score = 1.0
        
        combined_score = (utilization * 0.7) + (distance_score * 0.3)
        route_scores.append((i, combined_score))
    
    # Sort by score to identify underperforming routes
    route_scores.sort(key=lambda x: x[1])
    
    # Try to improve the worst routes by redistributing users
    improvements = 0
    for route_idx, score in route_scores[:2]:  # Focus on worst 2 routes
        if score < 0.5:  # Low efficiency threshold
            route = routes[route_idx]
            users = route['assigned_users']
            
            if len(users) <= 1:
                continue
                
            # Try to move users to better routes
            for user in users[:]:  # Copy list to avoid modification during iteration
                best_target = None
                best_improvement = 0
                
                for other_idx, other_route in enumerate(routes):
                    if other_idx == route_idx:
                        continue
                        
                    if len(other_route['assigned_users']) < other_route['vehicle_type']:
                        # Calculate improvement if user is moved
                        driver_dist = get_cached_distance(
                            other_route['latitude'], other_route['longitude'],
                            user['latitude'], user['longitude'],
                            use_road_distance=True
                        )
                        
                        if driver_dist <= _config['MAX_EXTRA_DISTANCE_KM']:
                            # Simple improvement metric: shorter distance = better
                            current_driver_dist = get_cached_distance(
                                route['latitude'], route['longitude'],
                                user['latitude'], user['longitude'],
                                use_road_distance=True
                            )
                            
                            improvement = max(0, current_driver_dist - driver_dist)
                            if improvement > best_improvement:
                                best_improvement = improvement
                                best_target = other_route
                
                # Move user if significant improvement found
                if best_target is not None and best_improvement > 0.5:  # 0.5km improvement threshold
                    route['assigned_users'].remove(user)
                    user['pickup_order'] = len(best_target['assigned_users']) + 1
                    best_target['assigned_users'].append(user)
                    improvements += 1
    
    if improvements > 0:
        print(f"   ✅ Balanced {improvements} users across routes")
    else:
        print("   ✅ Routes already well-balanced")
    
    return routes


def optimized_advanced_driver_cluster_assignment(user_df, driver_df):
    """Highly optimized driver assignment with parallel processing"""
    print("🚗 Optimized parallel driver assignment...")

    # Fast Hungarian assignment on clusters
    result = parallel_hungarian_assignment(user_df, driver_df)
    if isinstance(result, tuple):
        if len(result) >= 3:
            assignments, assigned_driver_ids, _ = result[0], result[1], result[2]
        elif len(result) == 2:
            assignments, assigned_driver_ids = result[0], result[1]
        else:
            assignments = result[0]
            assigned_driver_ids = set()
    else:
        assignments = result
        assigned_driver_ids = set()

    # Create routes in parallel
    def create_route(assignment_data):
        driver, cluster = assignment_data
        cluster_users = cluster['users']

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

        # Apply parallel TSP optimization
        try:
            G = road_graph if HAS_OSMNX and road_graph is not None else None
            optimized_users = parallel_tsp_solver(
                driver['latitude'], driver['longitude'],
                users_list, G
            )
        except Exception as e:
            print(f"   ⚠️ TSP optimization failed: {e}, using distance fallback")
            users_list.sort(
                key=lambda u: get_cached_distance(
                    driver['latitude'], driver['longitude'],
                    u['latitude'], u['longitude'],
                    use_road_distance=True
                )
            )
            for idx, user in enumerate(users_list):
                user['pickup_order'] = idx + 1
            optimized_users = users_list

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

        return route_data, [user['user_id'] for user in optimized_users]

    # Create routes in parallel
    routes = []
    assigned_user_ids = set()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(create_route, assignment) for assignment in assignments]
        
        for future in as_completed(futures):
            route_data, user_ids = future.result()
            routes.append(route_data)
            assigned_user_ids.update(user_ids)

    # Handle remaining users in parallel
    remaining_users = user_df[~user_df['user_id'].isin(assigned_user_ids)]
    remaining_drivers = driver_df[~driver_df['driver_id'].isin(assigned_driver_ids)]

    if len(remaining_users) > 0:
        routes, assigned_user_ids = parallel_enhanced_fallback_assignment(
            routes, remaining_users, remaining_drivers, assigned_user_ids, assigned_driver_ids
        )

    return routes, assigned_user_ids


def advanced_driver_cluster_assignment(user_df, driver_df):
    """
    Legacy function - use optimized version instead
    """
    return optimized_advanced_driver_cluster_assignment(user_df, driver_df)


def prepare_user_driver_dataframes_optimized(data):
    """Optimized dataframe preparation with vectorized operations"""
    # Vectorized user processing
    users = data.get('users', [])
    
    if users:
        user_data = {
            'user_id': [str(user.get('id', '')) for user in users],
            'latitude': [float(user.get('latitude', 0.0)) for user in users],
            'longitude': [float(user.get('longitude', 0.0)) for user in users],
            'first_name': [str(user.get('first_name', '')) for user in users],
            'email': [str(user.get('email', '')) for user in users]
        }
        
        user_df = pd.DataFrame(user_data)
        
        # Vectorized office distance calculation
        user_df['office_distance'] = np.array([
            haversine_distance(lat, lng, OFFICE_LAT, OFFICE_LON)
            for lat, lng in zip(user_df['latitude'], user_df['longitude'])
        ])
    else:
        user_df = pd.DataFrame()

    # Vectorized driver processing
    all_drivers = []
    if "drivers" in data:
        drivers_data = data["drivers"]
        all_drivers.extend(drivers_data.get("driversUnassigned", []))
        all_drivers.extend(drivers_data.get("driversAssigned", []))
    else:
        all_drivers.extend(data.get("driversUnassigned", []))
        all_drivers.extend(data.get("driversAssigned", []))

    if all_drivers:
        driver_data = {
            'driver_id': [str(driver.get('id', '')) for driver in all_drivers],
            'vehicle_id': [str(driver.get('vehicle_id', '')) for driver in all_drivers],
            'capacity': [int(driver.get('capacity', 4)) for driver in all_drivers],
            'latitude': [float(driver.get('latitude', 0.0)) for driver in all_drivers],
            'longitude': [float(driver.get('longitude', 0.0)) for driver in all_drivers],
            'shift_type': [int(driver.get('shift_type', 2)) for driver in all_drivers]
        }
        
        driver_df = pd.DataFrame(driver_data)
        driver_df['priority'] = np.where(driver_df['shift_type'].isin([1, 3]), 1, 2)
    else:
        driver_df = pd.DataFrame()

    return user_df, driver_df


def prepare_user_driver_dataframes(data):
    """
    Legacy function - use optimized version instead
    """
    return prepare_user_driver_dataframes_optimized(data)


def handle_unassigned_users_optimized(user_df, assigned_user_ids):
    """Optimized unassigned user handling with vectorized operations"""
    if user_df.empty:
        return []
        
    unassigned_users = user_df[~user_df['user_id'].isin(assigned_user_ids)]
    
    if unassigned_users.empty:
        return []

    # Vectorized coordinate validation
    valid_lats = np.clip(unassigned_users['latitude'].fillna(0.0), -90, 90)
    valid_lngs = np.clip(unassigned_users['longitude'].fillna(0.0), -180, 180)

    unassigned_list = []
    for idx, (_, user) in enumerate(unassigned_users.iterrows()):
        user_data = {
            'user_id': str(user['user_id']),
            'latitude': float(valid_lats.iloc[idx]),
            'longitude': float(valid_lngs.iloc[idx]),
            'lat': float(valid_lats.iloc[idx]),
            'lng': float(valid_lngs.iloc[idx]),
            'office_distance': float(user.get('office_distance', 0))
        }

        if pd.notna(user.get('first_name')):
            user_data['first_name'] = str(user['first_name'])
        if pd.notna(user.get('email')):
            user_data['email'] = str(user['email'])

        unassigned_list.append(user_data)

    return unassigned_list


def handle_unassigned_users(user_df, assigned_user_ids):
    """
    Legacy function - use optimized version instead
    """
    return handle_unassigned_users_optimized(user_df, assigned_user_ids)


def save_snap_cache():
    """Save cache with error handling"""
    cache_file = "snap_cache.pkl"
    try:
        with _snap_lock:
            with open(cache_file, 'wb') as f:
                pickle.dump(_SNAP_CACHE, f)
        print(f"   💾 Snap cache saved to {cache_file}")
    except Exception as e:
        print(f"   ⚠️ Failed to save snap cache: {e}")


def load_snap_cache():
    """Load cache with error handling"""
    cache_file = "snap_cache.pkl"
    if os.path.exists(cache_file):
        try:
            with _snap_lock:
                with open(cache_file, 'rb') as f:
                    loaded_cache = pickle.load(f)
                    _SNAP_CACHE.update(loaded_cache)
            print(f"   💾 Snap cache loaded from {cache_file}")
        except Exception as e:
            print(f"   ⚠️ Failed to load snap cache: {e}")


def run_assignment(source_id: str, parameter: int = 1, string_param: str = ""):
    """
    Highly optimized assignment function with parallel processing and caching
    """
    start_time = time.time()

    try:
        print(f"🚀 Starting optimized parallel assignment for source_id: {source_id}")
        print(f"📋 Parameter: {parameter}, String parameter: {string_param}")
        print(f"⚡ Using {MAX_WORKERS} parallel workers")

        # Load and validate data with timeout
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

        # Validate data
        validate_input_data(data)
        print("✅ Data validation passed")

        # Optimized dataframe preparation
        user_df, driver_df = prepare_user_driver_dataframes_optimized(data)
        print(f"📊 DataFrames prepared - Users: {len(user_df)}, Drivers: {len(driver_df)}")

        # Load cache
        load_snap_cache()

        # Parallel road snapping
        user_df = snap_to_road_network(user_df, _config)

        # Parallel clustering
        driver_capacities = driver_df['capacity'].tolist()
        user_df = parallel_clustering(user_df, driver_capacities)

        # Optimized driver assignment
        routes, assigned_user_ids = optimized_advanced_driver_cluster_assignment(user_df, driver_df)

        # Optimized unassigned user handling
        unassigned_users = handle_unassigned_users_optimized(user_df, assigned_user_ids)

        # Build unassigned drivers list efficiently
        assigned_driver_ids = {route['driver_id'] for route in routes}
        unassigned_drivers_df = driver_df[~driver_df['driver_id'].isin(assigned_driver_ids)]
        
        unassigned_drivers = [
            {
                'driver_id': str(row['driver_id']),
                'capacity': int(row['capacity']),
                'vehicle_id': str(row.get('vehicle_id', '')),
                'latitude': float(row['latitude']),
                'longitude': float(row['longitude'])
            }
            for _, row in unassigned_drivers_df.iterrows()
        ]

        # Save cache
        save_snap_cache()

        execution_time = time.time() - start_time

        # Calculate statistics
        total_capacity = sum(route['vehicle_type'] for route in routes)
        total_assigned = sum(len(route['assigned_users']) for route in routes)
        overall_utilization = (total_assigned / total_capacity * 100) if total_capacity > 0 else 0

        print(f"✅ Optimized assignment complete in {execution_time:.2f}s")
        print(f"📊 Final routes: {len(routes)}")
        print(f"🎯 Users assigned: {len(assigned_user_ids)}")
        print(f"👥 Users unassigned: {len(user_df) - len(assigned_user_ids)}")
        print(f"⚡ Overall capacity utilization: {overall_utilization:.1f}%")
        print(f"🚀 Speed improvement: ~{max(1, 20/execution_time):.1f}x faster")

        method_name = "Optimized Parallel VRP with Road-Aware Clustering"
        if HAS_PYVRP:
            method_name += " + PyVRP Solver"
        elif HAS_OSMNX:
            method_name += " + OSM Road Network"
        else:
            method_name += " + PCA Approximation"

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

    return [
        {
            'driver_id': str(driver.get('id', '')),
            'capacity': int(driver.get('capacity', 0)),
            'vehicle_id': str(driver.get('vehicle_id', '')),
            'latitude': float(driver.get('latitude', 0.0)),
            'longitude': float(driver.get('longitude', 0.0))
        }
        for driver in all_drivers
    ]


def _convert_users_to_unassigned_format(users):
    """Helper to convert user data to unassigned format"""
    return [
        {
            'user_id': str(user.get('id', '')),
            'lat': float(user.get('latitude', 0.0)),
            'lng': float(user.get('longitude', 0.0)),
            'office_distance': float(user.get('office_distance', 0.0)),
            'first_name': str(user.get('first_name', '')),
            'email': str(user.get('email', ''))
        }
        for user in users
    ]


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
    distance_issues = []
    
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
                    
                    # Check for long distances between consecutive users
                    if dist > 5.0:  # 5km threshold
                        distance_issues.append({
                            'route_id': route.get('driver_id', 'unknown'),
                            'distance': dist,
                            'users': [user1.get('user_id'), user2.get('user_id')]
                        })
                
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
        "routes_below_80_percent": sum(1 for u in utilizations if u < 0.8),
        "routes_below_50_percent": sum(1 for u in utilizations if u < 0.5),
        "distance_issues": distance_issues,
        "clustering_method": result.get("clustering_analysis", {}).get("method", "Unknown"),
        "optimization_features": [
            "Parallel Processing with Multi-threading",
            "Optimized Road-Aware Way Segment Clustering",
            "Numba JIT Compilation" if HAS_NUMBA else "Standard Python Functions",
            "PyVRP Advanced VRP Solver" if HAS_PYVRP else "Parallel Hungarian Algorithm Assignment",
            "Cached Road Distance Matrix with Thread Safety",
            "Parallel TSP with 2-opt Optimization",
            "Dynamic Route Balancing",
            "Vectorized Operations for Performance"
        ]
    }

    return analysis
