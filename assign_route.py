import json
import time
import math
import os
import pandas as pd
import numpy as np
import requests
from road_network import RoadNetwork
import networkx as nx
import traceback
from functools import lru_cache
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist
from dotenv import load_dotenv
import warnings
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
from logger_config import get_logger

# Import shared functions from assignment.py
from assignment import (
    load_env_and_fetch_data, 
    validate_input_data,
    extract_office_coordinates,
    prepare_user_driver_dataframes
)

warnings.filterwarnings('ignore')

# Setup logging
logger = get_logger()

# Global variables
ROAD_NETWORK = None
OPTIMIZATION_CONFIGS = {
    "efficiency": {"max_detour_ratio": 1.1},
    "balanced": {"max_detour_ratio": 1.3},
    "capacity": {"max_detour_ratio": 1.5}
}

def load_and_validate_config():
    """Load configuration with road-aware balanced optimization settings"""
    try:
        with open('config.json') as f:
            cfg = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.warning(f"Could not load config.json, using defaults. Error: {e}")
        cfg = {}

    # Always use balanced mode for road-aware
    current_mode = "road_aware_balanced"

    # Get balanced optimization configuration
    mode_configs = cfg.get("mode_configs", {})
    mode_config = mode_configs.get("balanced_optimization", {})

    logger.info(f"🗺️ Using optimization mode: ROAD-AWARE BALANCED")

    config = {}

    # Distance configurations optimized for road-aware routing - using balanced values
    config['MAX_FILL_DISTANCE_KM'] = max(0.1, float(mode_config.get("max_fill_distance_km", cfg.get("max_fill_distance_km", 6.0))))
    config['MERGE_DISTANCE_KM'] = max(0.1, float(mode_config.get("merge_distance_km", cfg.get("merge_distance_km", 3.5))))
    config['DBSCAN_EPS_KM'] = max(0.1, float(cfg.get("dbscan_eps_km", 2.0)))
    config['OVERFLOW_PENALTY_KM'] = max(0.0, float(cfg.get("overflow_penalty_km", 7.0)))
    config['DISTANCE_ISSUE_THRESHOLD'] = max(0.1, float(cfg.get("distance_issue_threshold_km", 10.0)))
    config['SWAP_IMPROVEMENT_THRESHOLD'] = max(0.0, float(cfg.get("swap_improvement_threshold_km", 0.7)))

    # Road-aware specific parameters
    config['ROAD_DETOUR_PENALTY'] = 2.0  # Penalty for road detours
    config['SAME_ROAD_BONUS'] = 1.5      # Bonus for users on same road
    config['ROAD_SPLIT_THRESHOLD'] = 0.3  # Road connectivity threshold

    # Utilization thresholds - balanced values
    config['MIN_UTIL_THRESHOLD'] = max(0.0, min(1.0, float(cfg.get("min_util_threshold", 0.65))))
    config['LOW_UTILIZATION_THRESHOLD'] = max(0.0, min(1.0, float(cfg.get("low_utilization_threshold", 0.6))))

    # Integer configurations
    config['MIN_SAMPLES_DBSCAN'] = max(1, int(cfg.get("min_samples_dbscan", 2)))
    config['MAX_SWAP_ITERATIONS'] = max(1, int(cfg.get("max_swap_iterations", 4)))
    config['MAX_USERS_FOR_FALLBACK'] = max(1, int(cfg.get("max_users_for_fallback", 4)))
    config['FALLBACK_MIN_USERS'] = max(1, int(cfg.get("fallback_min_users", 2)))
    config['FALLBACK_MAX_USERS'] = max(1, int(cfg.get("fallback_max_users", 8)))

    # Angle configurations for road-aware routing - balanced values
    config['MAX_BEARING_DIFFERENCE'] = max(0, min(180, float(mode_config.get("max_bearing_difference", cfg.get("max_bearing_difference", 30)))))
    config['MAX_TURNING_ANGLE'] = max(0, min(180, float(mode_config.get("max_allowed_turning_score", cfg.get("max_allowed_turning_score", 40)))))

    # Cost penalties - balanced values
    config['UTILIZATION_PENALTY_PER_SEAT'] = max(0.0, float(mode_config.get("utilization_penalty_per_seat", cfg.get("utilization_penalty_per_seat", 3.0))))

    # Office coordinates with environment variable fallbacks
    office_lat = float(os.getenv("OFFICE_LAT", cfg.get("office_latitude", 30.6810489)))
    office_lon = float(os.getenv("OFFICE_LON", cfg.get("office_longitude", 76.7260711)))

    # Validate coordinate bounds
    if not (-90 <= office_lat <= 90):
        logger.warning(f"Invalid office latitude {office_lat}, using default")
        office_lat = 30.6810489
    if not (-180 <= office_lon <= 180):
        logger.warning(f"Invalid office longitude {office_lon}, using default")
        office_lon = 76.7260711

    config['OFFICE_LAT'] = office_lat
    config['OFFICE_LON'] = office_lon

    # Balanced optimization parameters with road network enhancement
    config['optimization_mode'] = "road_aware_balanced"
    config['capacity_weight'] = mode_config.get('capacity_weight', 2.5)    # TRUE balanced: equal weight
    config['direction_weight'] = mode_config.get('direction_weight', 2.5)  # TRUE balanced: equal weight
    config['road_coherence_weight'] = 3.0  # Weight for road network coherence

    # Clustering and optimization parameters
    config['clustering_method'] = 'road_aware_adaptive'
    config['min_cluster_size'] = max(2, cfg.get('min_cluster_size', 2))
    config['use_sweep_algorithm'] = cfg.get('use_sweep_algorithm', True)
    config['angular_sectors'] = cfg.get('angular_sectors', 10)  # Balanced value
    config['max_users_per_initial_cluster'] = cfg.get('max_users_per_initial_cluster', 8)
    config['max_users_per_cluster'] = cfg.get('max_users_per_cluster', 7)

    # Balanced optimization parameters
    config['zigzag_penalty_weight'] = mode_config.get('zigzag_penalty_weight', cfg.get('zigzag_penalty_weight', 2.0))
    config['route_split_turning_threshold'] = cfg.get('route_split_turning_threshold', 45)
    config['max_tortuosity_ratio'] = cfg.get('max_tortuosity_ratio', 1.5)
    config['route_split_consistency_threshold'] = cfg.get('route_split_consistency_threshold', 0.6)

    # Latitude conversion factor for distance normalization
    config['LAT_TO_KM'] = 111.0
    config['LON_TO_KM'] = 111.0 * math.cos(math.radians(office_lat))

    logger.info(f"   🗺️ Road-aware max bearing difference: {config['MAX_BEARING_DIFFERENCE']}°")
    logger.info(f"   🗺️ Road-aware max turning score: {config['MAX_TURNING_ANGLE']}°")
    logger.info(f"   📏 Max fill distance: {config['MAX_FILL_DISTANCE_KM']}km")
    logger.info(f"   ⚖️ Capacity weight: {config['capacity_weight']} (TRUE BALANCED)")
    logger.info(f"   ⚖️ Direction weight: {config['direction_weight']} (TRUE BALANCED)")

    # Configuration for distance validation
    config['road_network.distance_validation.max_reasonable_ratio'] = cfg.get('road_network.distance_validation.max_reasonable_ratio', 4.0)
    config['road_network.distance_validation.min_road_efficiency'] = cfg.get('road_network.distance_validation.min_road_efficiency', 0.3)

    return config

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate the great-circle distance between two points on Earth"""
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    r = 6371
    return c * r

def calculate_bearing(lat1, lon1, lat2, lon2):
    """Calculate bearing between two points"""
    lat1, lat2 = map(math.radians, [lat1, lat2])
    dlon = math.radians(lon2 - lon1)
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    return (math.degrees(math.atan2(x, y)) + 360) % 360

def normalize_bearing_difference(diff):
    """Normalize bearing difference to [-180, 180] range"""
    while diff > 180:
        diff -= 360
    while diff < -180:
        diff += 360
    return diff

def bearing_difference(bearing1, bearing2):
    """Calculate the absolute difference between two bearings"""
    diff = abs(bearing1 - bearing2)
    return min(diff, 360 - diff)

def coords_to_km(lat, lon, office_lat, office_lon):
    """Convert coordinates to km from office"""
    lat_km = (lat - office_lat) * 111.0
    lon_km = (lon - office_lon) * 111.0 * math.cos(math.radians(office_lat))
    return lat_km, lon_km

# BALANCED OPTIMIZATION GEOGRAPHIC CLUSTERING (core from assign_balance.py)
def balanced_geographic_clustering(user_df, office_lat, office_lon, config):
    """Balanced geographic clustering enhanced with road network awareness"""
    logger.info("🗺️ Step 1: Balanced geographic clustering with road network enhancement...")

    if len(user_df) == 0:
        return user_df

    # Calculate features including bearings (from assign_balance.py)
    user_df = calculate_bearings_and_features(user_df, office_lat, office_lon)

    # Enhanced clustering: road-aware if network available, balanced otherwise
    if ROAD_NETWORK is not None:
        logger.info("   🗺️ Using road network enhanced balanced clustering")
        labels = road_enhanced_balanced_clustering(user_df, office_lat, office_lon, config)
    else:
        logger.info("   📍 Using pure balanced clustering (no road network)")
        labels = pure_balanced_clustering(user_df, office_lat, office_lon, config)

    user_df['geo_cluster'] = labels
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    logger.info(f"   ✅ Created {n_clusters} balanced geographic clusters")
    return user_df

def calculate_bearings_and_features(user_df, office_lat, office_lon):
    """Calculate bearings and add geographical features - OFFICE TO USER direction"""
    user_df = user_df.copy()

    # Calculate bearing FROM OFFICE TO USER
    user_df['bearing_from_office'] = calculate_bearing_vectorized(
        office_lat, office_lon, user_df['latitude'], user_df['longitude'])
    user_df['bearing_sin'] = np.sin(np.radians(user_df['bearing_from_office']))
    user_df['bearing_cos'] = np.cos(np.radians(user_df['bearing_from_office']))

    return user_df

def calculate_bearing_vectorized(lat1, lon1, lat2, lon2):
    """Vectorized calculation of bearing from point A to B in degrees"""
    lat1, lat2 = np.radians(lat1), np.radians(lat2)
    dlon = np.radians(lon2 - lon1)
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    return (np.degrees(np.arctan2(x, y)) + 360) % 360

def road_enhanced_balanced_clustering(user_df, office_lat, office_lon, config):
    """Road network enhanced balanced clustering - road awareness only as tie-breaker"""
    try:
        # Start with pure balanced clustering as the foundation
        balanced_labels = pure_balanced_clustering(user_df, office_lat, office_lon, config)
        
        # If no road network, return balanced clustering
        if ROAD_NETWORK is None:
            return balanced_labels
        
        # Apply light road-awareness refinement only for tie-breaking
        refined_labels = balanced_labels.copy()
        
        # Group users by their balanced clusters
        cluster_groups = {}
        for i, label in enumerate(balanced_labels):
            if label not in cluster_groups:
                cluster_groups[label] = []
            cluster_groups[label].append(i)
        
        # For larger clusters, check if road connectivity can improve balance
        next_cluster_id = max(balanced_labels) + 1 if balanced_labels else 0
        
        for cluster_id, user_indices in cluster_groups.items():
            if len(user_indices) > config.get('max_users_per_initial_cluster', 8):
                cluster_users = user_df.iloc[user_indices]
                
                # Try light road-aware refinement
                road_refined = apply_light_road_refinement(cluster_users, office_lat, office_lon, config)
                
                if road_refined and len(road_refined) > 1:  # Only if it creates meaningful splits
                    for i, subgroup in enumerate(road_refined):
                        for user_idx in subgroup:
                            original_idx = user_indices[user_idx]
                            if i == 0:
                                refined_labels[original_idx] = cluster_id  # Keep original ID for first group
                            else:
                                refined_labels[original_idx] = next_cluster_id + i - 1
                    next_cluster_id += len(road_refined) - 1
        
        return refined_labels

    except Exception as e:
        logger.warning(f"🗺️ Road enhanced clustering failed: {e}, using pure balanced")
        return pure_balanced_clustering(user_df, office_lat, office_lon, config)

def road_connectivity_clustering(user_df, office_lat, office_lon):
    """Cluster users based on road network connectivity"""
    user_positions = [(row['latitude'], row['longitude']) for _, row in user_df.iterrows()]

    # Find nearest road nodes for each user
    user_road_nodes = []
    for pos in user_positions:
        try:
            node, distance = ROAD_NETWORK.find_nearest_road_node(pos[0], pos[1])
            if distance <= 2.0:  # Within 2km of road
                user_road_nodes.append(node)
            else:
                user_road_nodes.append(None)
        except:
            user_road_nodes.append(None)

    # Group users by road connectivity
    clusters = []
    cluster_id = 0
    processed = set()

    for i, node in enumerate(user_road_nodes):
        if i in processed or node is None:
            clusters.append(-1)  # Unconnected user
            continue

        # Find all users connected to this road segment
        connected_users = [i]
        processed.add(i)

        for j, other_node in enumerate(user_road_nodes):
            if j in processed or other_node is None:
                continue

            # Check if users are on connected road segments
            if are_road_segments_connected(node, other_node):
                connected_users.append(j)
                processed.add(j)

        # Assign cluster ID to all connected users
        for user_idx in connected_users:
            if len(clusters) <= user_idx:
                clusters.extend([-1] * (user_idx - len(clusters) + 1))
            clusters[user_idx] = cluster_id

        cluster_id += 1

    return clusters

def are_road_segments_connected(node1, node2):
    """Check if two road nodes are on connected segments"""
    if ROAD_NETWORK is None or node1 is None or node2 is None:
        return False

    try:
        # Check if there's a path between nodes within reasonable distance
        path_length = ROAD_NETWORK._get_shortest_path_distance(node1, node2)
        return path_length is not None and path_length <= 10.0  # 10km max connection
    except:
        return False

def apply_light_road_refinement(cluster_users, office_lat, office_lon, config):
    """Apply light road-aware refinement - only splits if clearly beneficial"""
    if len(cluster_users) <= 4 or ROAD_NETWORK is None:
        return None
    
    try:
        # Find road nodes for users
        user_road_info = []
        for idx, (_, user) in enumerate(cluster_users.iterrows()):
            node, distance = ROAD_NETWORK.find_nearest_road_node(user['latitude'], user['longitude'])
            if distance <= 1.5:  # Within 1.5km of road
                user_road_info.append((idx, node))
        
        # If less than half users are near roads, don't refine
        if len(user_road_info) < len(cluster_users) * 0.5:
            return None
        
        # Group by road connectivity
        road_groups = {}
        for user_idx, node in user_road_info:
            # Simple grouping by road node proximity
            assigned = False
            for group_key, group_users in road_groups.items():
                group_node = group_key
                if are_road_segments_connected(node, group_node):
                    group_users.append(user_idx)
                    assigned = True
                    break
            
            if not assigned:
                road_groups[node] = [user_idx]
        
        # Only split if we get 2-3 reasonable groups
        if len(road_groups) < 2 or len(road_groups) > 3:
            return None
        
        # Convert to list of groups
        refined_groups = []
        for group_users in road_groups.values():
            if len(group_users) >= 2:  # Minimum group size
                refined_groups.append(group_users)
        
        # Add any remaining users to closest group
        assigned_users = set()
        for group in refined_groups:
            assigned_users.update(group)
        
        remaining = [i for i in range(len(cluster_users)) if i not in assigned_users]
        if remaining:
            # Add to largest group
            largest_group = max(refined_groups, key=len) if refined_groups else []
            largest_group.extend(remaining)
        
        return refined_groups if len(refined_groups) >= 2 else None
        
    except Exception:
        return None

def pure_balanced_clustering(user_df, office_lat, office_lon, config):
    """Pure balanced clustering (from assign_balance.py logic)"""
    # Use sector-based clustering for direction awareness
    use_sweep = config.get('use_sweep_algorithm', True)

    if use_sweep and len(user_df) > 3:
        labels = sweep_clustering(user_df, config)
    else:
        labels = polar_sector_clustering(user_df, office_lat, office_lon, config)

    return labels

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
    n_sectors = config.get('angular_sectors', 10)
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
            eps_km = config.get('DBSCAN_EPS_KM', 2.0)
            sector_labels = dbscan_clustering_metric(sector_users, eps_km, 2, office_lat, office_lon)

            for i, idx in enumerate(sector_users.index):
                if sector_labels[i] == -1:
                    labels[idx] = current_cluster
                    current_cluster += 1
                else:
                    labels[idx] = current_cluster + sector_labels[i]

            current_cluster += max(sector_labels) + 1 if len(sector_labels) > 0 else 1

    return labels

def dbscan_clustering_metric(user_df, eps_km, min_samples, office_lat, office_lon):
    """Perform DBSCAN clustering using proper metric coordinates"""
    # Convert coordinates to km from office
    coords_km = []
    for _, user in user_df.iterrows():
        lat_km, lon_km = coords_to_km(user['latitude'], user['longitude'], office_lat, office_lon)
        coords_km.append([lat_km, lon_km])

    coords_km = np.array(coords_km)

    # Use DBSCAN with eps in km
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

def balanced_subcluster_large_group(cluster_users, office_lat, office_lon, config):
    """Apply balanced subclustering to large groups"""
    max_cluster_size = config.get('max_users_per_cluster', 7)

    if len(cluster_users) <= max_cluster_size:
        return [0] * len(cluster_users)

    # Use balanced K-means clustering
    n_subclusters = math.ceil(len(cluster_users) / max_cluster_size)

    coords_km = []
    for _, user in cluster_users.iterrows():
        lat_km, lon_km = coords_to_km(user['latitude'], user['longitude'], office_lat, office_lon)
        coords_km.append([lat_km, lon_km])

    coords_km = np.array(coords_km)

    if n_subclusters >= len(coords_km):
        return list(range(len(coords_km)))

    kmeans = KMeans(n_clusters=n_subclusters, random_state=42, n_init=10)
    subcluster_labels = kmeans.fit_predict(coords_km)

    return subcluster_labels.tolist()

# BALANCED CAPACITY SUBCLUSTERING (core from assign_balance.py)
def balanced_capacity_subclustering(user_df, office_lat, office_lon, config):
    """Create capacity-aware subclusters with balanced constraints and road enhancement"""
    logger.info("🚗 Step 2: Balanced capacity subclustering with road enhancement...")

    max_cluster_size = config.get('max_users_per_cluster', 7)

    new_clusters = []
    cluster_counter = 0

    for cluster_id in user_df['geo_cluster'].unique():
        cluster_users = user_df[user_df['geo_cluster'] == cluster_id]

        if len(cluster_users) <= max_cluster_size:
            # Small cluster, no need to split
            for idx in cluster_users.index:
                new_clusters.append((idx, cluster_counter))
            cluster_counter += 1
        else:
            # Large cluster, needs splitting with balanced approach and road awareness
            subclusters = balanced_split_large_cluster(cluster_users, office_lat, office_lon, max_cluster_size)

            for subcluster in subclusters:
                for idx in subcluster:
                    new_clusters.append((idx, cluster_counter))
                cluster_counter += 1

    # Apply new cluster assignments
    cluster_map = dict(new_clusters)
    user_df['capacity_cluster'] = user_df.index.map(cluster_map)

    logger.info(f"   🎯 Balanced capacity subclustering: {len(user_df['geo_cluster'].unique())} → {len(user_df['capacity_cluster'].unique())} clusters")
    return user_df

def balanced_split_large_cluster(cluster_users, office_lat, office_lon, max_size):
    """Split large clusters using balanced approach with road awareness"""
    if len(cluster_users) <= max_size:
        return [cluster_users.index.tolist()]

    # Try road-aware splitting first if network available
    if ROAD_NETWORK is not None:
        try:
            return road_aware_balanced_split(cluster_users, office_lat, office_lon, max_size)
        except Exception as e:
            logger.warning(f"🗺️ Road-aware splitting failed: {e}, using balanced geometric split")

    # Fallback to balanced geometric splitting
    return balanced_geometric_split(cluster_users, office_lat, office_lon, max_size)

def road_aware_balanced_split(cluster_users, office_lat, office_lon, max_size):
    """Split using road network information with balanced constraints"""
    user_positions = [(row['latitude'], row['longitude']) for _, row in cluster_users.iterrows()]

    # Find road segments for users
    user_segments = []
    for pos in user_positions:
        node, distance = ROAD_NETWORK.find_nearest_road_node(pos[0], pos[1])
        if distance <= 1.0:  # Within 1km of road
            segment = ROAD_NETWORK.get_road_segment_info(node) if hasattr(ROAD_NETWORK, 'get_road_segment_info') else {'way_id': str(node)}
            user_segments.append(segment)
        else:
            user_segments.append(None)

    # Group users by road segments with balanced size constraints
    segment_groups = {}
    for i, segment in enumerate(user_segments):
        if segment is not None:
            segment_key = segment.get('way_id', f'segment_{i}')
            if segment_key not in segment_groups:
                segment_groups[segment_key] = []
            segment_groups[segment_key].append(cluster_users.index.tolist()[i])

    # Create balanced subclusters from road segments
    subclusters = []
    current_subcluster = []

    for segment_key, user_indices in segment_groups.items():
        if len(current_subcluster) + len(user_indices) <= max_size:
            current_subcluster.extend(user_indices)
        else:
            if current_subcluster:
                subclusters.append(current_subcluster)

            # Split large segment groups if necessary
            if len(user_indices) > max_size:
                for i in range(0, len(user_indices), max_size):
                    subclusters.append(user_indices[i:i + max_size])
            else:
                current_subcluster = user_indices

    if current_subcluster:
        subclusters.append(current_subcluster)

    return subclusters if subclusters else [cluster_users.index.tolist()]

def balanced_geometric_split(cluster_users, office_lat, office_lon, max_size):
    """Split clusters using balanced geometric methods"""
    if len(cluster_users) <= max_size:
        return [cluster_users.index.tolist()]

    # Use balanced K-means to split
    n_subclusters = math.ceil(len(cluster_users) / max_size)

    coords_km = []
    for _, user in cluster_users.iterrows():
        lat_km, lon_km = coords_to_km(user['latitude'], user['longitude'], office_lat, office_lon)
        coords_km.append([lat_km, lon_km])

    coords_km = np.array(coords_km)

    if n_subclusters >= len(coords_km):
        return [[idx] for idx in cluster_users.index.tolist()]

    kmeans = KMeans(n_clusters=n_subclusters, random_state=42, n_init=10)
    subcluster_labels = kmeans.fit_predict(coords_km)

    subclusters = []
    for i in range(n_subclusters):
        subcluster_indices = [cluster_users.index.tolist()[j] for j, label in enumerate(subcluster_labels) if label == i]
        if subcluster_indices:
            subclusters.append(subcluster_indices)

    return subclusters

# BALANCED DRIVER ASSIGNMENT (core from assign_balance.py)
def balanced_driver_assignment_with_road_awareness(user_df, driver_df, office_lat, office_lon):
    """
    TRUE balanced driver assignment with road network enhancement:
    50/50 between capacity utilization and route efficiency, enhanced by road network
    """
    logger.info("🚗 Step 3: TRUE balanced driver assignment with road enhancement...")

    routes = []
    assigned_user_ids = set()
    used_driver_ids = set()

    # Balanced sorting: capacity and priority equally weighted
    available_drivers = driver_df.sort_values(['capacity', 'priority'], ascending=[False, True])

    # Process each capacity cluster
    for cluster_id in user_df['capacity_cluster'].unique():
        cluster_users = user_df[user_df['capacity_cluster'] == cluster_id]
        unassigned_in_cluster = cluster_users[~cluster_users['user_id'].isin(assigned_user_ids)]

        if unassigned_in_cluster.empty:
            continue

        # Find best driver for this cluster with TRUE balanced scoring and road enhancement
        route = assign_best_driver_to_cluster_balanced_road_aware(
            unassigned_in_cluster, available_drivers, used_driver_ids, office_lat, office_lon
        )

        if route:
            routes.append(route)
            assigned_user_ids.update(u['user_id'] for u in route['assigned_users'])

            # Log route creation with balanced metrics
            utilization = len(route['assigned_users']) / route['vehicle_type']
            road_coherence = calculate_road_coherence(route) if ROAD_NETWORK else 0.5

            logger.info(f"  ⚖️ Driver {route['driver_id']}: {len(route['assigned_users'])}/{route['vehicle_type']} seats ({utilization*100:.1f}%) - Road coherence: {road_coherence:.2f}")

    # Second pass: Fill remaining seats with compatible users (from assign_balance.py approach)
    remaining_users = user_df[~user_df['user_id'].isin(assigned_user_ids)]

    for route in routes:
        if remaining_users.empty:
            break

        available_seats = route['vehicle_type'] - len(route['assigned_users'])
        if available_seats <= 0:
            continue

        # Balanced seat filling with light road awareness
        route_bearing = calculate_average_bearing_improved(route, office_lat, office_lon)
        route_center = calculate_route_center_improved(route)

        compatible_users = []
        for _, user in remaining_users.iterrows():
            distance = haversine_distance(route_center[0], route_center[1],
                                        user['latitude'], user['longitude'])

            user_bearing = calculate_bearing(office_lat, office_lon, user['latitude'], user['longitude'])
            bearing_diff = bearing_difference(route_bearing, user_bearing)

            # Balanced criteria for seat filling (same as assign_balance.py)
            if distance <= MAX_FILL_DISTANCE_KM * 1.8 and bearing_diff <= 40:
                score = distance * 0.6 + bearing_diff * 0.08
                compatible_users.append((score, user))

        # Fill available seats
        compatible_users.sort(key=lambda x: x[0])
        users_to_add = []

        for score, user in compatible_users[:available_seats]:
            users_to_add.append(user)

        # Add users to route
        for user in users_to_add:
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

        if users_to_add:
            assigned_ids = {u['user_id'] for u in users_to_add}
            remaining_users = remaining_users[~remaining_users['user_id'].isin(assigned_ids)]

            route = optimize_route_sequence_balanced_road_aware(route, office_lat, office_lon)
            utilization = len(route['assigned_users']) / route['vehicle_type'] * 100
            logger.info(f"  ⚖️ Route {route['driver_id']}: Added {len(users_to_add)} users, now {len(route['assigned_users'])}/{route['vehicle_type']} ({utilization:.1f}%)")

    logger.info(f"  ✅ Balanced road-aware assignment: {len(routes)} routes created")
    return routes, assigned_user_ids

def assign_best_driver_to_cluster_balanced_road_aware(cluster_users, available_drivers, used_driver_ids, office_lat, office_lon):
    """Find and assign the best available driver with TRUE 50/50 balanced optimization with light road bonus"""
    cluster_size = len(cluster_users)

    best_driver = None
    best_score = float('inf')
    best_sequence = None

    # TRUE balanced weights - exactly equal importance (copied from assign_balance.py)
    capacity_weight = 2.5  # Equal weight
    direction_weight = 2.5  # Equal weight

    for _, driver in available_drivers.iterrows():
        if driver['driver_id'] in used_driver_ids:
            continue

        # Capacity check
        if driver['capacity'] < cluster_size:
            continue

        # Calculate route metrics (simplified)
        route_cost, sequence, mean_turning_degrees = calculate_route_cost_balanced(
            driver, cluster_users, office_lat, office_lon
        )

        # TRUE balanced scoring approach (exact copy from assign_balance.py)
        utilization = cluster_size / driver['capacity']

        # Distance component (efficiency factor)
        distance_score = route_cost * 0.5  # 50% weight on distance

        # Direction component (efficiency factor)
        direction_score = mean_turning_degrees * direction_weight * 0.02  # 50% weight on direction

        # Capacity component (capacity factor) - inverted to prefer higher utilization
        capacity_score = (1.0 - utilization) * capacity_weight * 5.0  # 50% weight on capacity

        # Priority component (small tiebreaker)
        priority_score = driver['priority'] * 0.2

        # Balanced total score: equal emphasis on efficiency and capacity
        efficiency_component = distance_score + direction_score  # Route efficiency
        capacity_component = capacity_score  # Capacity utilization

        total_score = efficiency_component + capacity_component + priority_score

        # Apply small road coherence bonus (max 10%)
        if ROAD_NETWORK:
            try:
                user_positions = [(u['latitude'], u['longitude']) for _, u in cluster_users.iterrows()]
                driver_pos = (driver['latitude'], driver['longitude'])
                office_pos = (office_lat, office_lon)
                coherence = ROAD_NETWORK.get_route_coherence_score(driver_pos, user_positions, office_pos)
                total_score *= (1 - 0.1 * coherence)  # Max 10% bonus for good coherence
            except Exception:
                pass  # No road bonus if calculation fails

        if total_score < best_score:
            best_score = total_score
            best_driver = driver
            best_sequence = sequence

    if best_driver is not None:
        used_driver_ids.add(best_driver['driver_id'])

        route = {
            'driver_id': str(best_driver['driver_id']),
            'vehicle_id': str(best_driver.get('vehicle_id', '')),
            'vehicle_type': int(best_driver['capacity']),
            'latitude': float(best_driver['latitude']),
            'longitude': float(best_driver['longitude']),
            'assigned_users': []
        }

        # Add all users from cluster
        if hasattr(cluster_users, 'iterrows'):
            users_to_add = list(cluster_users.iterrows())
        else:
            users_to_add = [(i, user) for i, user in enumerate(cluster_users)]

        for _, user in users_to_add:
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

        # Balanced optimization of sequence with road enhancement
        route = optimize_route_sequence_balanced_road_aware(route, office_lat, office_lon)
        update_route_metrics_improved(route, office_lat, office_lon)

        utilization = len(route['assigned_users']) / route['vehicle_type']
        logger.info(f"    ⚖️ TRUE balanced assignment - Driver {best_driver['driver_id']}: {len(route['assigned_users'])}/{route['vehicle_type']} seats ({utilization*100:.1f}%)")

        return route

    return None

def calculate_route_cost_balanced(driver, cluster_users, office_lat, office_lon):
    """Calculate route cost with TRUE balanced optimization - simplified version"""
    if len(cluster_users) == 0:
        return float('inf'), [], 0

    driver_pos = (driver['latitude'], driver['longitude'])
    office_pos = (office_lat, office_lon)

    # Get optimal pickup sequence with balanced focus (no road complexity)
    sequence = calculate_optimal_sequence_balanced(driver_pos, cluster_users, office_pos)

    # Calculate total route distance using road network if available, haversine otherwise
    total_distance = 0
    bearing_differences = []

    # Driver to first pickup
    if sequence:
        first_user = sequence[0]
        if ROAD_NETWORK:
            total_distance += ROAD_NETWORK.get_road_distance(
                driver_pos[0], driver_pos[1], 
                first_user['latitude'], first_user['longitude']
            )
        else:
            total_distance += haversine_distance(
                driver_pos[0], driver_pos[1], 
                first_user['latitude'], first_user['longitude']
            )

    # Between pickups - calculate bearing differences
    for i in range(len(sequence) - 1):
        current_user = sequence[i]
        next_user = sequence[i + 1]

        if ROAD_NETWORK:
            distance = ROAD_NETWORK.get_road_distance(
                current_user['latitude'], current_user['longitude'],
                next_user['latitude'], next_user['longitude']
            )
        else:
            distance = haversine_distance(
                current_user['latitude'], current_user['longitude'],
                next_user['latitude'], next_user['longitude']
            )
        total_distance += distance

        # Calculate bearing difference between segments
        if i == 0:
            prev_bearing = calculate_bearing(driver_pos[0], driver_pos[1], 
                                           current_user['latitude'], current_user['longitude'])
        else:
            prev_pos = (sequence[i-1]['latitude'], sequence[i-1]['longitude'])
            prev_bearing = calculate_bearing(prev_pos[0], prev_pos[1],
                                           current_user['latitude'], current_user['longitude'])

        next_bearing = calculate_bearing(current_user['latitude'], current_user['longitude'],
                                       next_user['latitude'], next_user['longitude'])

        bearing_diff = bearing_difference(prev_bearing, next_bearing)
        bearing_differences.append(bearing_diff)

    # Last pickup to office
    if sequence:
        last_user = sequence[-1]
        if ROAD_NETWORK:
            total_distance += ROAD_NETWORK.get_road_distance(
                last_user['latitude'], last_user['longitude'],
                office_lat, office_lon
            )
        else:
            total_distance += haversine_distance(
                last_user['latitude'], last_user['longitude'],
                office_lat, office_lon
            )

    # Calculate mean turning angle - balanced weight
    mean_turning_degrees = sum(bearing_differences) / len(bearing_differences) if bearing_differences else 0

    return total_distance, sequence, mean_turning_degrees

def calculate_optimal_sequence_balanced_road_aware(driver_pos, cluster_users, office_pos):
    """Calculate sequence with TRUE balanced optimization enhanced by road network"""
    if len(cluster_users) <= 1:
        users_list = cluster_users.to_dict('records') if hasattr(cluster_users, 'to_dict') else list(cluster_users)
        road_metrics = {'coherence': 1.0, 'method': 'single_user'}
        return users_list, road_metrics

    users_list = cluster_users.to_dict('records') if hasattr(cluster_users, 'to_dict') else list(cluster_users)

    # Calculate main route bearing (driver to office)
    main_route_bearing = calculate_bearing(driver_pos[0], driver_pos[1], office_pos[0], office_pos[1])

    # TRUE balanced scoring with road enhancement: exactly 50% distance, 50% direction
    def true_balanced_road_aware_score(user):
        # Distance component (50%)
        if ROAD_NETWORK:
            distance = ROAD_NETWORK.get_road_distance(driver_pos[0], driver_pos[1], 
                                                    user['latitude'], user['longitude'])
        else:
            distance = haversine_distance(driver_pos[0], driver_pos[1], 
                                        user['latitude'], user['longitude'])

        # Direction component (50%)
        user_bearing = calculate_bearing(driver_pos[0], driver_pos[1], 
                                       user['latitude'], user['longitude'])

        bearing_diff = normalize_bearing_difference(user_bearing - main_route_bearing)
        bearing_diff_rad = math.radians(abs(bearing_diff))

        # Equal weight to distance and bearing alignment
        distance_score = distance  # Raw distance
        direction_score = distance * (1 - math.cos(bearing_diff_rad))  # Direction penalty in distance units

        # TRUE 50/50 balance
        combined_score = distance_score * 0.5 + direction_score * 0.5

        return (combined_score, user['user_id'])

    users_list.sort(key=true_balanced_road_aware_score)

    # Apply truly balanced 2-opt optimization with road enhancement
    optimized_sequence = apply_balanced_2opt_road_aware(users_list, driver_pos, office_pos)

    # Calculate road metrics if network available
    road_metrics = {}
    if ROAD_NETWORK:
        try:
            user_positions = [(u['latitude'], u['longitude']) for u in optimized_sequence]
            coherence = ROAD_NETWORK.get_route_coherence_score(driver_pos, user_positions, office_pos)
            road_metrics = {'coherence': coherence, 'method': 'road_network'}
        except Exception:
            road_metrics = {'coherence': 0.5, 'method': 'fallback'}
    else:
        road_metrics = {'coherence': 0.5, 'method': 'no_network'}

    return optimized_sequence, road_metrics

def apply_balanced_2opt_road_aware(sequence, driver_pos, office_pos):
    """Apply TRUE balanced 2-opt improvements with road enhancement"""
    if len(sequence) <= 2:
        return sequence

    improved = True
    max_iterations = 3
    iteration = 0

    # Calculate main bearing from driver to office
    main_bearing = calculate_bearing(driver_pos[0], driver_pos[1], office_pos[0], office_pos[1])

    # Balanced turning angle threshold (between efficiency and capacity)
    max_turning_threshold = 47  # Balanced value

    while improved and iteration < max_iterations:
        improved = False
        iteration += 1

        best_distance = calculate_sequence_distance_road_aware(sequence, driver_pos, office_pos)
        best_turning_score = calculate_sequence_turning_score_improved(sequence, driver_pos, office_pos)

        for i in range(len(sequence) - 1):
            for j in range(i + 2, len(sequence)):
                # Try 2-opt swap
                new_sequence = sequence[:i+1] + sequence[i+1:j+1][::-1] + sequence[j+1:]

                # Calculate new metrics
                new_distance = calculate_sequence_distance_road_aware(new_sequence, driver_pos, office_pos)
                new_turning_score = calculate_sequence_turning_score_improved(new_sequence, driver_pos, office_pos)

                # TRUE balanced acceptance criteria - equal weight to both factors
                distance_improvement = (best_distance - new_distance) / best_distance
                turning_improvement = (best_turning_score - new_turning_score)

                # Convert turning improvement to same scale as distance (percentage)
                turning_improvement_normalized = turning_improvement / max(best_turning_score, 1.0)

                # Equal weight to both improvements
                combined_improvement = distance_improvement * 0.5 + turning_improvement_normalized * 0.5

                # Accept if combined improvement is positive and turning stays reasonable
                if (combined_improvement > 0.005 and new_turning_score <= max_turning_threshold):
                    sequence = new_sequence
                    best_distance = new_distance
                    best_turning_score = new_turning_score
                    improved = True
                    break
            if improved:
                break

    return sequence

def calculate_sequence_distance_road_aware(sequence, driver_pos, office_pos):
    """Calculate total distance for a pickup sequence with road awareness"""
    if not sequence:
        return 0

    if ROAD_NETWORK:
        total = ROAD_NETWORK.get_road_distance(driver_pos[0], driver_pos[1],
                                             sequence[0]['latitude'], sequence[0]['longitude'])

        for i in range(len(sequence) - 1):
            total += ROAD_NETWORK.get_road_distance(sequence[i]['latitude'], sequence[i]['longitude'],
                                                  sequence[i + 1]['latitude'], sequence[i + 1]['longitude'])

        total += ROAD_NETWORK.get_road_distance(sequence[-1]['latitude'], sequence[-1]['longitude'], 
                                              office_pos[0], office_pos[1])
    else:
        total = haversine_distance(driver_pos[0], driver_pos[1],
                                 sequence[0]['latitude'], sequence[0]['longitude'])

        for i in range(len(sequence) - 1):
            total += haversine_distance(sequence[i]['latitude'], sequence[i]['longitude'],
                                      sequence[i + 1]['latitude'], sequence[i + 1]['longitude'])

        total += haversine_distance(sequence[-1]['latitude'], sequence[-1]['longitude'], 
                                  office_pos[0], office_pos[1])

    return total

def calculate_sequence_turning_score_improved(sequence, driver_pos, office_pos):
    """Calculate average bearing difference for a sequence"""
    if len(sequence) <= 1:
        return 0

    bearing_differences = []
    prev_bearing = None

    for i in range(len(sequence)):
        if i == 0:
            # Bearing from driver to first user
            current_bearing = calculate_bearing(driver_pos[0], driver_pos[1],
                                                sequence[i]['latitude'],
                                                sequence[i]['longitude'])
            if len(sequence) == 1:
                # Single user: bearing from user to office
                next_bearing = calculate_bearing(sequence[i]['latitude'],
                                                 sequence[i]['longitude'],
                                                 office_pos[0], office_pos[1])
                bearing_diff = bearing_difference(current_bearing, next_bearing)
                bearing_differences.append(bearing_diff)
            prev_bearing = current_bearing
            continue

        # Bearing from previous to current
        prev_pos = (sequence[i - 1]['latitude'], sequence[i - 1]['longitude'])
        current_pos = (sequence[i]['latitude'], sequence[i]['longitude'])
        current_bearing = calculate_bearing(prev_pos[0], prev_pos[1],
                                            current_pos[0], current_pos[1])

        if i == len(sequence) - 1:
            # Last user: bearing from current to office
            next_bearing = calculate_bearing(current_pos[0], current_pos[1],
                                             office_pos[0], office_pos[1])
        else:
            # Bearing from current to next
            next_pos = (sequence[i + 1]['latitude'], sequence[i + 1]['longitude'])
            next_bearing = calculate_bearing(current_pos[0], current_pos[1],
                                             next_pos[0], next_pos[1])

        bearing_diff = bearing_difference(prev_bearing, current_bearing)
        if bearing_diff > 0:
            bearing_differences.append(bearing_diff)

        prev_bearing = current_bearing

    return sum(bearing_differences) / len(bearing_differences) if bearing_differences else 0

def optimize_route_sequence_balanced_road_aware(route, office_lat, office_lon):
    """Optimize route sequence with balanced approach and road enhancement"""
    if len(route['assigned_users']) <= 1:
        return route

    driver_pos = (route['latitude'], route['longitude'])
    office_pos = (office_lat, office_lon)
    user_positions = [(u['lat'], u['lng']) for u in route['assigned_users']]

    # Use road network optimization if available
    if ROAD_NETWORK:
        try:
            optimized_indices = ROAD_NETWORK.get_optimal_pickup_sequence(driver_pos, user_positions, office_pos)
            optimized_users = [route['assigned_users'][i] for i in optimized_indices]
            route['assigned_users'] = optimized_users
        except Exception as e:
            logger.warning(f"🗺️ Road sequence optimization failed: {e}, using balanced distance sort")
            # Fallback to balanced distance-based sorting
            route['assigned_users'].sort(key=lambda u: u['office_distance'])
    else:
        # Fallback to balanced distance-based sorting
        route['assigned_users'].sort(key=lambda u: u['office_distance'])

    return route

def calculate_road_coherence(route):
    """Calculate road coherence score for a route"""
    if ROAD_NETWORK is None or not route['assigned_users']:
        return 0.5

    try:
        driver_pos = (route['latitude'], route['longitude'])
        user_positions = [(u['lat'], u['lng']) for u in route['assigned_users']]
        office_pos = (30.6810489, 76.7260711)  # Default office position

        coherence = ROAD_NETWORK.get_route_coherence_score(driver_pos, user_positions, office_pos)
        return coherence
    except:
        return 0.5

def update_route_metrics_improved(route, office_lat, office_lon):
    """Update route metrics with road-aware enhancements"""
    if not route['assigned_users']:
        route['total_distance'] = 0
        route['estimated_time'] = 0
        route['road_coherence'] = 0.0
        return

    # Calculate total distance through all pickup points
    route_points = [(route['latitude'], route['longitude'])]

    for user in route['assigned_users']:
        route_points.append((user['lat'], user['lng']))

    route_points.append((office_lat, office_lon))

    total_distance = 0
    for i in range(len(route_points) - 1):
        if ROAD_NETWORK:
            distance = ROAD_NETWORK.get_road_distance(route_points[i][0], route_points[i][1], 
                                                    route_points[i+1][0], route_points[i+1][1])
        else:
            distance = haversine_distance(route_points[i][0], route_points[i][1], 
                                        route_points[i+1][0], route_points[i+1][1])
        total_distance += distance

    route['total_distance'] = round(total_distance, 2)
    route['estimated_time'] = round(total_distance / 25, 2)  # Assuming 25 km/h average

    # Add road coherence if available
    route['road_coherence'] = calculate_road_coherence(route)

    # Calculate turning score
    route['turning_score'] = calculate_route_turning_score(route, office_lat, office_lon)

def calculate_route_turning_score(route, office_lat, office_lon):
    """Calculate turning score for a route"""
    if len(route['assigned_users']) <= 1:
        return 0.0

    driver_pos = (route['latitude'], route['longitude'])
    office_pos = (office_lat, office_lon)
    user_positions = [(u['lat'], u['lng']) for u in route['assigned_users']]

    path_points = [driver_pos] + user_positions + [office_pos]

    total_turning = 0.0
    for i in range(len(path_points) - 2):
        bearing1 = calculate_bearing(path_points[i][0], path_points[i][1], 
                                   path_points[i+1][0], path_points[i+1][1])
        bearing2 = calculate_bearing(path_points[i+1][0], path_points[i+1][1], 
                                   path_points[i+2][0], path_points[i+2][1])

        turning_angle = abs(normalize_bearing_difference(bearing2 - bearing1))
        total_turning += turning_angle

    return total_turning / max(1, len(path_points) - 2)

# BALANCED LOCAL OPTIMIZATION (from assign_balance.py)
def balanced_local_optimization(routes, office_lat, office_lon):
    """Local optimization with balanced approach and road enhancement"""
    logger.info("🔧 Step 4: Balanced local optimization with road enhancement...")

    optimized_routes = []
    improvements = 0

    for route in routes:
        if not route['assigned_users']:
            optimized_routes.append(route)
            continue

        # Re-optimize sequence with balanced approach and road enhancement
        original_distance = route.get('total_distance', 0)
        route = optimize_route_sequence_balanced_road_aware(route, office_lat, office_lon)
        update_route_metrics_improved(route, office_lat, office_lon)

        new_distance = route.get('total_distance', 0)
        if new_distance < original_distance:
            improvements += 1

        optimized_routes.append(route)

    logger.info(f"  🔧 Balanced local optimization: {improvements} routes improved")
    return optimized_routes

# BALANCED GLOBAL OPTIMIZATION (from assign_balance.py)
def balanced_global_optimization(routes, user_df, assigned_user_ids, driver_df, office_lat, office_lon):
    """Global optimization with balanced approach and road enhancement"""
    logger.info("🌍 Step 5: Balanced global optimization with road enhancement...")

    # Handle remaining unassigned users
    remaining_users = user_df[~user_df['user_id'].isin(assigned_user_ids)]
    unassigned_users = []

    if not remaining_users.empty:
        logger.info(f"  👥 Processing {len(remaining_users)} unassigned users...")

        # Try to assign to existing routes with balanced approach and road awareness
        for _, user in remaining_users.iterrows():
            best_route = None
            best_score = float('inf')

            for route in routes:
                if len(route['assigned_users']) >= route['vehicle_type']:
                    continue

                # Calculate balanced compatibility with road enhancement
                compatibility_score = calculate_balanced_road_compatibility(user, route, office_lat, office_lon)

                if compatibility_score < best_score:
                    best_score = compatibility_score
                    best_route = route

            if best_route and best_score < 12.0:  # Balanced compatibility threshold
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

                # Re-optimize route with balanced approach
                best_route = optimize_route_sequence_balanced_road_aware(best_route, office_lat, office_lon)
                update_route_metrics_improved(best_route, office_lat, office_lon)

                logger.info(f"    ✅ Balanced-assigned user {user['user_id']} to route {best_route['driver_id']}")
            else:
                # Add to unassigned
                user_data = {
                    'user_id': str(user['user_id']),
                    'lat': float(user['latitude']),
                    'lng': float(user['longitude']),
                    'office_distance': float(user.get('office_distance', 0)),
                    'reason': 'No compatible route found'
                }

                if pd.notna(user.get('first_name')):
                    user_data['first_name'] = str(user['first_name'])
                if pd.notna(user.get('email')):
                    user_data['email'] = str(user['email'])

                unassigned_users.append(user_data)

    logger.info(f"  ✅ Balanced global optimization complete: {len(unassigned_users)} users unassigned")
    return routes, unassigned_users

def calculate_balanced_road_compatibility(user, route, office_lat, office_lon):
    """Calculate balanced compatibility between user and route with road enhancement"""
    user_pos = (user['latitude'], user['longitude'])
    driver_pos = (route['latitude'], route['longitude'])

    # Base distance score (balanced weight)
    if ROAD_NETWORK:
        distance = ROAD_NETWORK.get_road_distance(user_pos[0], user_pos[1], driver_pos[0], driver_pos[1])
    else:
        distance = haversine_distance(user_pos[0], user_pos[1], driver_pos[0], driver_pos[1])

    distance_score = distance

    # Road coherence score if available (balanced enhancement)
    road_score = 0.0
    if ROAD_NETWORK and route['assigned_users']:
        try:
            test_users = [(u['lat'], u['lng']) for u in route['assigned_users']] + [user_pos]
            coherence = ROAD_NETWORK.get_route_coherence_score(driver_pos, test_users, (office_lat, office_lon))
            road_score = (1.0 - coherence) * 4.0  # Balanced penalty for low coherence
        except:
            road_score = 2.0  # Default balanced penalty

    # Direction consistency score (balanced weight)
    if route['assigned_users']:
        route_bearing = calculate_route_bearing(route, office_lat, office_lon)
        user_bearing = calculate_bearing(driver_pos[0], driver_pos[1], user_pos[0], user_pos[1])
        bearing_diff = bearing_difference(route_bearing, user_bearing)
        direction_score = bearing_diff / 8.0  # Balanced normalization
    else:
        direction_score = 0.0

    # TRUE balanced combination
    return distance_score + road_score + direction_score

def calculate_route_bearing(route, office_lat, office_lon):
    """Calculate average bearing of a route"""
    if not route['assigned_users']:
        return calculate_bearing(route['latitude'], route['longitude'], office_lat, office_lon)

    bearings = []
    driver_pos = (route['latitude'], route['longitude'])

    for user in route['assigned_users']:
        bearing = calculate_bearing(driver_pos[0], driver_pos[1], user['lat'], user['lng'])
        bearings.append(bearing)

    # Calculate circular mean
    x = sum(math.cos(math.radians(b)) for b in bearings)
    y = sum(math.sin(math.radians(b)) for b in bearings)

    mean_bearing = math.degrees(math.atan2(y, x))
    return (mean_bearing + 360) % 360

# BALANCED FINAL MERGE (from assign_balance.py)
def balanced_final_merge_with_road_awareness(routes, config, office_lat, office_lon):
    """
    TRUE balanced final-pass merge with road network enhancement:
    Equal focus on capacity utilization and route efficiency, enhanced by road coherence
    """
    logger.info("🔄 Step 6: TRUE balanced final-pass merge with road enhancement...")

    merged_routes = []
    used = set()

    # Balanced thresholds (exactly between route efficiency and capacity)
    MERGE_BEARING_THRESHOLD = 32  # Balanced value
    MERGE_DISTANCE_KM = config.get("MERGE_DISTANCE_KM", 3.5) * 1.2  # Balanced expansion
    MERGE_TURNING_THRESHOLD = 50  # Balanced value
    MERGE_TORTUOSITY_THRESHOLD = 1.65  # Balanced value
    MERGE_COHERENCE_THRESHOLD = 0.5  # Road coherence threshold

    for i, r1 in enumerate(routes):
        if i in used:
            continue

        best_merge = None
        best_balanced_score = float('inf')

        for j, r2 in enumerate(routes):
            if j <= i or j in used:
                continue

            # 1. Direction compatibility check (balanced)
            b1 = calculate_route_bearing(r1, office_lat, office_lon)
            b2 = calculate_route_bearing(r2, office_lat, office_lon)
            bearing_diff = bearing_difference(b1, b2)

            if bearing_diff > MERGE_BEARING_THRESHOLD:
                continue

            # 2. Distance compatibility check (balanced)
            c1 = calculate_route_center(r1)
            c2 = calculate_route_center(r2)
            centroid_distance = haversine_distance(c1[0], c1[1], c2[0], c2[1])

            if centroid_distance > MERGE_DISTANCE_KM:
                continue

            # 3. Capacity check
            total_users = len(r1['assigned_users']) + len(r2['assigned_users'])
            max_capacity = max(r1['vehicle_type'], r2['vehicle_type'])

            if total_users > max_capacity:
                continue

            # 4. Quality assessment with balanced criteria and road enhancement
            combined_center = calculate_combined_route_center(r1, r2)
            dist1 = haversine_distance(r1['latitude'], r1['longitude'], combined_center[0], combined_center[1])
            dist2 = haversine_distance(r2['latitude'], r2['longitude'], combined_center[0], combined_center[1])

            better_route = r1 if dist1 <= dist2 else r2

            # Create test merged route
            test_route = better_route.copy()
            test_route['assigned_users'] = r1['assigned_users'] + r2['assigned_users']
            test_route['vehicle_type'] = max_capacity

            # Optimize sequence for merged route
            test_route = optimize_route_sequence_balanced_road_aware(test_route, office_lat, office_lon)

            # Calculate balanced quality metrics
            turning_score = calculate_route_turning_score(test_route, office_lat, office_lon)
            tortuosity = calculate_tortuosity_ratio(test_route, office_lat, office_lon)
            utilization = total_users / max_capacity

            # Road coherence check (if available)
            road_coherence_acceptable = True
            if ROAD_NETWORK:
                coherence = calculate_road_coherence(test_route)
                if coherence < MERGE_COHERENCE_THRESHOLD:
                    road_coherence_acceptable = False

            # Balanced acceptance criteria - both efficiency and capacity matter equally
            efficiency_acceptable = (turning_score <= MERGE_TURNING_THRESHOLD and 
                                   tortuosity <= MERGE_TORTUOSITY_THRESHOLD)
            capacity_acceptable = utilization >= 0.6  # Balanced utilization requirement

            # Only accept if BOTH efficiency and capacity criteria are met AND road coherence is acceptable
            if efficiency_acceptable and capacity_acceptable and road_coherence_acceptable:
                # TRUE balanced scoring: 50% efficiency, 50% capacity
                efficiency_score = turning_score * 0.5 + (tortuosity - 1.0) * 20  # Route quality
                capacity_score = (1.0 - utilization) * 100  # Underutilization penalty

                # Equal weight to both components
                balanced_score = efficiency_score * 0.5 + capacity_score * 0.5

                if balanced_score < best_balanced_score:
                    best_balanced_score = balanced_score
                    best_merge = (j, test_route)

        if best_merge is not None:
            j, merged_route = best_merge
            merged_routes.append(merged_route)
            used.add(i)
            used.add(j)

            utilization_pct = len(merged_route['assigned_users']) / merged_route['vehicle_type'] * 100
            turning = merged_route.get('turning_score', 0)
            coherence = merged_route.get('road_coherence', 0.5)
            logger.info(f"  ⚖️ Balanced merge: {r1['driver_id']} + {routes[j]['driver_id']} = {len(merged_route['assigned_users'])}/{merged_route['vehicle_type']} seats ({utilization_pct:.1f}%, {turning:.1f}° turn, {coherence:.2f} coherence)")
        else:
            merged_routes.append(r1)
            used.add(i)

    # Additional balanced optimization pass - full implementation from assign_balance.py
    logger.info("  ⚖️ Additional balanced optimization pass...")

    # Try to redistribute for better overall balance
    optimization_made = True
    optimization_passes = 0
    max_optimization_passes = 3

    while optimization_made and optimization_passes < max_optimization_passes:
        optimization_made = False
        optimization_passes += 1

        # Sort routes by efficiency score for balancing
        routes_with_scores = []
        for route in merged_routes:
            utilization = len(route['assigned_users']) / route['vehicle_type']
            efficiency = route.get('turning_score', 0)

            # Look for opportunities to balance
            if utilization < 0.7 and efficiency < 35:  # Good efficiency, poor capacity
                routes_with_scores.append(('low_util', route))
            elif utilization > 0.9 and efficiency > 45:  # Good capacity, poor efficiency  
                routes_with_scores.append(('high_util', route))
            else:
                routes_with_scores.append(('balanced', route))

        # Try to swap users between unbalanced routes for better overall balance
        low_util_routes = [r for t, r in routes_with_scores if t == 'low_util']
        high_util_routes = [r for t, r in routes_with_scores if t == 'high_util']

        # Cross-route optimization: move users between routes for better balance
        for low_route in low_util_routes:
            for high_route in high_util_routes:
                if len(high_route['assigned_users']) <= 1:
                    continue

                # Try moving one user from high-util to low-util route
                available_capacity = low_route['vehicle_type'] - len(low_route['assigned_users'])
                if available_capacity > 0:
                    # Find user in high_route that's closest to low_route with compatibility check
                    low_center = calculate_route_center(low_route)
                    low_bearing = calculate_route_bearing(low_route, office_lat, office_lon)

                    best_user = None
                    best_score = float('inf')

                    for user in high_route['assigned_users']:
                        distance = haversine_distance(low_center[0], low_center[1], user['lat'], user['lng'])
                        user_bearing = calculate_bearing(office_lat, office_lon, user['lat'], user['lng'])
                        bearing_diff = bearing_difference(low_bearing, user_bearing)

                        # Balanced compatibility score
                        compatibility_score = distance * 0.6 + bearing_diff * 0.05

                        if (distance <= MERGE_DISTANCE_KM * 1.5 and 
                            bearing_diff <= 35 and 
                            compatibility_score < best_score):
                            best_score = compatibility_score
                            best_user = user

                    if best_user:
                        # Calculate quality before and after move
                        old_low_util = len(low_route['assigned_users']) / low_route['vehicle_type']
                        old_high_util = len(high_route['assigned_users']) / high_route['vehicle_type']

                        new_low_util = (len(low_route['assigned_users']) + 1) / low_route['vehicle_type']
                        new_high_util = (len(high_route['assigned_users']) - 1) / high_route['vehicle_type']

                        # Check if move improves overall balance
                        old_balance_score = abs(0.8 - old_low_util) + abs(0.8 - old_high_util)
                        new_balance_score = abs(0.8 - new_low_util) + abs(0.8 - new_high_util)

                        if new_balance_score < old_balance_score * 0.9:  # Significant improvement
                            # Move the user
                            high_route['assigned_users'].remove(best_user)
                            low_route['assigned_users'].append(best_user)

                            # Re-optimize both routes
                            low_route = optimize_route_sequence_balanced_road_aware(low_route, office_lat, office_lon)
                            high_route = optimize_route_sequence_balanced_road_aware(high_route, office_lat, office_lon)
                            update_route_metrics_improved(low_route, office_lat, office_lon)
                            update_route_metrics_improved(high_route, office_lat, office_lon)

                            optimization_made = True
                            logger.info(f"    ⚖️ Balanced redistribution pass {optimization_passes}: User moved for better balance")
                            break
                if optimization_made:
                    break
            if optimization_made:
                break

        # Try user swaps between similar routes for efficiency improvements
        if not optimization_made:
            for i, route1 in enumerate(merged_routes):
                for j, route2 in enumerate(merged_routes):
                    if i >= j or not route1['assigned_users'] or not route2['assigned_users']:
                        continue

                    # Check if routes are similar enough for user swapping
                    center1 = calculate_route_center(route1)
                    center2 = calculate_route_center(route2)
                    route_distance = haversine_distance(center1[0], center1[1], center2[0], center2[1])

                    if route_distance <= MERGE_DISTANCE_KM * 2.0:
                        # Try swapping users between similar routes
                        for user1 in route1['assigned_users']:
                            for user2 in route2['assigned_users']:
                                # Calculate improvement from swap
                                dist1_to_route2 = haversine_distance(user1['lat'], user1['lng'], center2[0], center2[1])
                                dist2_to_route1 = haversine_distance(user2['lat'], user2['lng'], center1[0], center1[1])

                                current_cost = (haversine_distance(user1['lat'], user1['lng'], center1[0], center1[1]) +
                                              haversine_distance(user2['lat'], user2['lng'], center2[0], center2[1]))
                                new_cost = dist1_to_route2 + dist2_to_route1

                                if new_cost < current_cost * 0.85:  # Significant improvement
                                    # Perform swap
                                    route1['assigned_users'].remove(user1)
                                    route2['assigned_users'].remove(user2)
                                    route1['assigned_users'].append(user2)
                                    route2['assigned_users'].append(user1)

                                    # Re-optimize both routes
                                    route1 = optimize_route_sequence_balanced_road_aware(route1, office_lat, office_lon)
                                    route2 = optimize_route_sequence_balanced_road_aware(route2, office_lat, office_lon)
                                    update_route_metrics_improved(route1, office_lat, office_lon)
                                    update_route_metrics_improved(route2, office_lat, office_lon)

                                    optimization_made = True
                                    logger.info(f"    ⚖️ User swap optimization pass {optimization_passes}: Improved route efficiency")
                                    break
                            if optimization_made:
                                break
                        if optimization_made:
                            break
                    if optimization_made:
                        break

    # Final statistics
    total_seats = sum(r['vehicle_type'] for r in merged_routes)
    total_users = sum(len(r['assigned_users']) for r in merged_routes)
    avg_utilization = (total_users / total_seats * 100) if total_seats > 0 else 0
    avg_turning = np.mean([r.get('turning_score', 0) for r in merged_routes])
    avg_coherence = np.mean([r.get('road_coherence', 0.5) for r in merged_routes]) if merged_routes else 0.5

    logger.info(f"  ⚖️ TRUE balanced merge: {len(routes)} → {len(merged_routes)} routes")
    logger.info(f"  ⚖️ Final balance: {avg_utilization:.1f}% utilization, {avg_turning:.1f}° avg turning, {avg_coherence:.2f} avg coherence")

    return merged_routes

def calculate_route_center(route):
    """Calculate center point of route's users"""
    if not route['assigned_users']:
        return (route['latitude'], route['longitude'])

    avg_lat = sum(u['lat'] for u in route['assigned_users']) / len(route['assigned_users'])
    avg_lng = sum(u['lng'] for u in route['assigned_users']) / len(route['assigned_users'])
    return (avg_lat, avg_lng)

def calculate_combined_route_center(route1, route2):
    """Calculate the center point of users from two routes combined"""
    all_users = route1['assigned_users'] + route2['assigned_users']
    if not all_users:
        return (0, 0)

    avg_lat = sum(u['lat'] for u in all_users) / len(all_users)
    avg_lng = sum(u['lng'] for u in all_users) / len(all_users)
    return (avg_lat, avg_lng)

def calculate_tortuosity_ratio(route, office_lat, office_lon):
    """Calculate tortuosity ratio for a route"""
    if not route['assigned_users']:
        return 1.0

    # Actual route distance
    driver_pos = (route['latitude'], route['longitude'])
    office_pos = (office_lat, office_lon)

    route_points = [driver_pos]
    for user in route['assigned_users']:
        route_points.append((user['lat'], user['lng']))
    route_points.append(office_pos)

    actual_distance = 0.0
    for i in range(len(route_points) - 1):
        if ROAD_NETWORK:
            distance = ROAD_NETWORK.get_road_distance(route_points[i][0], route_points[i][1], 
                                                    route_points[i+1][0], route_points[i+1][1])
        else:
            distance = haversine_distance(route_points[i][0], route_points[i][1], 
                                        route_points[i+1][0], route_points[i+1][1])
        actual_distance += distance

    # Straight line distance (driver to centroid to office)
    if route['assigned_users']:
        centroid_lat = sum(u['lat'] for u in route['assigned_users']) / len(route['assigned_users'])
        centroid_lng = sum(u['lng'] for u in route['assigned_users']) / len(route['assigned_users'])

        if ROAD_NETWORK:
            straight_distance = (ROAD_NETWORK.get_road_distance(driver_pos[0], driver_pos[1], centroid_lat, centroid_lng) +
                                ROAD_NETWORK.get_road_distance(centroid_lat, centroid_lng, office_pos[0], office_pos[1]))
        else:
            straight_distance = (haversine_distance(driver_pos[0], driver_pos[1], centroid_lat, centroid_lng) +
                               haversine_distance(centroid_lat, centroid_lng, office_pos[0], office_pos[1]))
    else:
        if ROAD_NETWORK:
            straight_distance = ROAD_NETWORK.get_road_distance(driver_pos[0], driver_pos[1], office_pos[0], office_pos[1])
        else:
            straight_distance = haversine_distance(driver_pos[0], driver_pos[1], office_pos[0], office_pos[1])

    return actual_distance / straight_distance if straight_distance > 0 else 1.0

def _get_all_drivers_as_unassigned(data):
    """Convert all drivers to unassigned format"""
    drivers_data = data.get('drivers', {})
    all_drivers = []
    all_drivers.extend(drivers_data.get('driversUnassigned', []))
    all_drivers.extend(drivers_data.get('driversAssigned', []))

    unassigned_drivers = []
    for driver in all_drivers:
        driver_data = {
            'driver_id': str(driver.get('id', driver.get('driver_id', ''))),
            'capacity': int(driver.get('capacity', 0)),
            'vehicle_id': str(driver.get('vehicle_id', '')),
            'latitude': float(driver.get('latitude', driver.get('lat', 0.0))),
            'longitude': float(driver.get('longitude', driver.get('lng', 0.0)))
        }
        unassigned_drivers.append(driver_data)

    return unassigned_drivers

def _convert_users_to_unassigned_format(users):
    """Convert users to unassigned format"""
    unassigned_users = []
    for user in users:
        user_data = {
            'user_id': str(user.get('id', user.get('user_id', ''))),
            'lat': float(user.get('latitude', user.get('lat', 0.0))),
            'lng': float(user.get('longitude', user.get('lng', 0.0))),
            'office_distance': float(user.get('office_distance', 0)),
            'reason': 'No drivers available'
        }
        if user.get('first_name'):
            user_data['first_name'] = str(user['first_name'])
        if user.get('email'):
            user_data['email'] = str(user['email'])

        unassigned_users.append(user_data)

    return unassigned_users

def run_road_aware_assignment(source_id: str, parameter: int = 1, string_param: str = ""):
    """Main road-aware assignment function using TRUE balanced optimization logic enhanced by road network"""
    start_time = time.time()

    # Clear any cached data files
    cache_files = [
        "drivers_and_routes.json",
        "drivers_and_routes_capacity.json", 
        "drivers_and_routes_balance.json",
        "drivers_and_routes_road_aware.json"
    ]

    for cache_file in cache_files:
        if os.path.exists(cache_file):
            os.remove(cache_file)

    # Load configuration
    config = load_and_validate_config()

    logger.info(f"🗺️ Starting ROAD-AWARE BALANCED assignment for source_id: {source_id}")
    logger.info(f"📋 Parameter: {parameter}, String parameter: {string_param}")

    try:
        # Initialize road network
        global ROAD_NETWORK
        try:
            logger.info("🗺️ Loading road network...")
            ROAD_NETWORK = RoadNetwork('tricity_main_roads.graphml')
            logger.info(f"✅ Road network loaded: {len(ROAD_NETWORK.graph.nodes)} nodes")
        except Exception as e:
            logger.warning(f"⚠️ Road network unavailable: {e}")
            ROAD_NETWORK = None
            logger.info("📍 Using enhanced balanced fallback")

        # Load and validate data with detailed error logging
        logger.info(f"🌐 Making API request to fetch data for source_id: {source_id}")
        try:
            data = load_env_and_fetch_data(source_id, parameter, string_param)
        except requests.exceptions.HTTPError as http_err:
            logger.error(f"HTTP Error in assign_route: {http_err}")
            logger.error(f"Response status code: {http_err.response.status_code}")
            logger.error(f"Response content: {http_err.response.text}")
            raise
        except Exception as fetch_err:
            logger.error(f"Data fetch error in assign_route: {fetch_err}")
            raise

        # Edge case handling
        users = data.get('users', [])
        if not users:
            logger.warning("⚠️ No users found - returning empty assignment")
            return {
                "status": "true",
                "execution_time": time.time() - start_time,
                "data": [],
                "unassignedUsers": [],
                "unassignedDrivers": _get_all_drivers_as_unassigned(data),
                "clustering_analysis": {"method": "No Users", "clusters": 0},
                "optimization_mode": "road_aware_balanced",
                "parameter": parameter,
                "road_network_enabled": ROAD_NETWORK is not None
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
            logger.warning("⚠️ No drivers available - all users unassigned")
            unassigned_users = _convert_users_to_unassigned_format(users)
            return {
                "status": "true",
                "execution_time": time.time() - start_time,
                "data": [],
                "unassignedUsers": unassigned_users,
                "unassignedDrivers": [],
                "clustering_analysis": {"method": "No Drivers", "clusters": 0},
                "optimization_mode": "road_aware_balanced",
                "parameter": parameter,
                "road_network_enabled": ROAD_NETWORK is not None
            }

        logger.info(f"📥 Data loaded - Users: {len(users)}, Drivers: {len(all_drivers)}")

        # Extract office coordinates and validate
        office_lat, office_lon = extract_office_coordinates(data)
        validate_input_data(data)
        logger.info("✅ Data validation passed")

        # Prepare dataframes
        user_df, driver_df = prepare_user_driver_dataframes(data)
        logger.info(f"📊 DataFrames prepared - Users: {len(user_df)}, Drivers: {len(driver_df)}")

        # STEP 1: Balanced geographic clustering enhanced with road network
        user_df = balanced_geographic_clustering(user_df, office_lat, office_lon, config)
        clustering_method = "road_enhanced_balanced" if ROAD_NETWORK else "pure_balanced"
        clustering_results = {
            "method": clustering_method,
            "clusters": user_df['geo_cluster'].nunique(),
            "road_network_enabled": ROAD_NETWORK is not None
        }

        # STEP 2: Balanced capacity subclustering with road awareness
        user_df = balanced_capacity_subclustering(user_df, office_lat, office_lon, config)

        # STEP 3: TRUE balanced driver assignment with road enhancement
        routes, assigned_user_ids = balanced_driver_assignment_with_road_awareness(user_df, driver_df, office_lat, office_lon)

        # STEP 4: Balanced local optimization with road enhancement
        routes = balanced_local_optimization(routes, office_lat, office_lon)

        # STEP 5: Balanced global optimization with road enhancement
        routes, unassigned_users = balanced_global_optimization(
            routes, user_df, assigned_user_ids, driver_df, office_lat, office_lon)

        # STEP 6: Balanced final merge with road enhancement
        routes = balanced_final_merge_with_road_awareness(routes, config, office_lat, office_lon)

        # Filter out empty routes
        filtered_routes = []
        empty_route_driver_ids = set()

        for route in routes:
            if route['assigned_users'] and len(route['assigned_users']) > 0:
                filtered_routes.append(route)
            else:
                empty_route_driver_ids.add(route['driver_id'])
                logger.info(f"  📋 Moving driver {route['driver_id']} with no users to unassigned")

        routes = filtered_routes

        # Build unassigned drivers list
        assigned_driver_ids = {route['driver_id'] for route in routes}
        unassigned_drivers_df = driver_df[~driver_df['driver_id'].isin(assigned_driver_ids)]
        unassigned_drivers = []

        for _, driver in unassigned_drivers_df.iterrows():
            driver_data = {
                'driver_id': str(driver.get('driver_id', '')),
                'capacity': int(driver.get('capacity', 0)),
                'vehicle_id': str(driver.get('vehicle_id', '')),
                'latitude': float(driver.get('latitude', 0.0)),
                'longitude': float(driver.get('longitude', 0.0))
            }
            unassigned_drivers.append(driver_data)

        # Final metrics update
        for route in routes:
            update_route_metrics_improved(route, office_lat, office_lon)

        execution_time = time.time() - start_time

        # Final statistics
        total_users_in_api = len(users)
        users_assigned = sum(len(r['assigned_users']) for r in routes)
        users_unassigned = len(unassigned_users)
        total_capacity = sum(r['vehicle_type'] for r in routes)
        utilization = (users_assigned / total_capacity * 100) if total_capacity > 0 else 0
        avg_road_coherence = np.mean([r.get('road_coherence', 0.5) for r in routes]) if routes else 0

        logger.info(f"✅ Road-aware balanced assignment complete in {execution_time:.2f}s")
        logger.info(f"🗺️ Road network: {'ENABLED' if ROAD_NETWORK else 'DISABLED (pure balanced mode)'}")
        logger.info(f"📊 Final routes: {len(routes)}")
        logger.info(f"🎯 Users assigned: {users_assigned}/{total_users_in_api} ({users_assigned/total_users_in_api*100:.1f}%)")
        logger.info(f"📊 Overall utilization: {utilization:.1f}%")
        logger.info(f"🛣️ Average road coherence: {avg_road_coherence:.2f}")
        logger.info(f"⚖️ TRUE BALANCED optimization: Equal weight to capacity and efficiency")

        return {
            "status": "true",
            "execution_time": execution_time,
            "data": routes,
            "unassignedUsers": unassigned_users,
            "unassignedDrivers": unassigned_drivers,
            "clustering_analysis": clustering_results,
            "optimization_mode": "road_aware_balanced",
            "parameter": parameter,
            "road_network_enabled": ROAD_NETWORK is not None,
            "road_coherence_avg": avg_road_coherence,
            "utilization": utilization,
            "balanced_optimization": True
        }

    except requests.exceptions.RequestException as req_err:
        logger.error(f"API request failed: {req_err}")
        return {"status": "false", "details": str(req_err), "data": []}
    except ValueError as val_err:
        logger.error(f"Data validation error: {val_err}")
        return {"status": "false", "details": str(val_err), "data": []}
    except Exception as e:
        logger.error(f"Road-aware balanced assignment failed: {e}", exc_info=True)
        return {"status": "false", "details": str(e), "data": []}

# Create an alias for external calling compatibility
def run_assignment_road_aware(source_id: str, parameter: int = 1, string_param: str = ""):
    """Alias for external calling compatibility"""
    return run_road_aware_assignment(source_id, parameter, string_param)

def main():
    """Test the road-aware balanced assignment with sample data"""
    sample_data = {
        "users": [
            {"id": "1", "latitude": 30.6840, "longitude": 76.7300, "first_name": "User1", "email": "user1@test.com"},
            {"id": "2", "latitude": 30.6820, "longitude": 76.7280, "first_name": "User2", "email": "user2@test.com"},
            {"id": "3", "latitude": 30.6860, "longitude": 76.7320, "first_name": "User3", "email": "user3@test.com"},
            {"id": "4", "latitude": 30.6780, "longitude": 76.7200, "first_name": "User4", "email": "user4@test.com"},
            {"id": "5", "latitude": 30.6900, "longitude": 76.7400, "first_name": "User5", "email": "user5@test.com"}
        ],
        "drivers": {
            "driversUnassigned": [
                {"id": "101", "capacity": 4, "latitude": 30.6800, "longitude": 76.7250, "vehicle_id": "V101"},
                {"id": "102", "capacity": 6, "latitude": 30.6830, "longitude": 76.7290, "vehicle_id": "V102"}
            ],
            "driversAssigned": []
        },
        "company": {
            "latitude": 30.6810489,
            "longitude": 76.7260711,
            "name": "Test Company"
        }
    }

    result = run_road_aware_assignment("test", 1, "")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()