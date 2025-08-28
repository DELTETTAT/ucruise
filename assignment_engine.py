
import math
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist
from utils import (
    config_manager, haversine_distance, calculate_bearing, calculate_bearing_vectorized,
    bearing_difference, validate_input_data, load_env_and_fetch_data,
    extract_office_coordinates, prepare_dataframes, calculate_bearings_and_features,
    create_standardized_error_response, validate_route_constraints,
    ValidationError
)
import logging

logger = logging.getLogger(__name__)

class AssignmentEngine:
    """Unified assignment engine for both route and balance assignments"""
    
    def __init__(self, assignment_type: str):
        self.assignment_type = assignment_type
        self.config = config_manager.get_config(assignment_type)
        
    def run_assignment(self, source_id: str, parameter: int = 1, string_param: str = ""):
        """Main assignment function"""
        start_time = pd.Timestamp.now()
        
        try:
            logger.info(f"Starting {self.assignment_type} for source_id: {source_id}")
            
            # Load and validate data
            data = load_env_and_fetch_data(source_id, parameter, string_param)
            validate_input_data(data)
            
            # Handle edge cases
            users = data.get('users', [])
            if not users:
                return self._empty_assignment_response(start_time, parameter, string_param, data)
            
            all_drivers = []
            all_drivers.extend(data.get("driversUnassigned", []))
            all_drivers.extend(data.get("driversAssigned", []))
            
            if not all_drivers:
                return self._no_drivers_response(start_time, parameter, string_param, users)
            
            logger.info(f"Data loaded - Users: {len(users)}, Drivers: {len(all_drivers)}")
            
            # Extract office coordinates and prepare dataframes
            office_lat, office_lon = extract_office_coordinates(data)
            user_df, driver_df = prepare_dataframes(data, self.assignment_type)
            user_df = calculate_bearings_and_features(user_df, office_lat, office_lon)
            
            logger.info(f"DataFrames prepared - Users: {len(user_df)}, Drivers: {len(driver_df)}")
            
            # Run assignment algorithm
            routes, unassigned_users = self._run_assignment_algorithm(
                user_df, driver_df, office_lat, office_lon)
            
            # Validate route constraints
            routes = validate_route_constraints(routes, self.config)
            
            # Build unassigned drivers list
            assigned_driver_ids = {route['driver_id'] for route in routes}
            unassigned_drivers = self._build_unassigned_drivers_list(driver_df, assigned_driver_ids)
            
            execution_time = (pd.Timestamp.now() - start_time).total_seconds()
            
            logger.info(f"Assignment complete in {execution_time:.2f}s")
            logger.info(f"Routes: {len(routes)}, Assigned users: {sum(len(r['assigned_users']) for r in routes)}")
            
            return {
                "status": "true",
                "execution_time": execution_time,
                "data": routes,
                "unassignedUsers": unassigned_users,
                "unassignedDrivers": unassigned_drivers,
                "clustering_analysis": {"method": self.assignment_type, "clusters": len(routes)},
                "optimization_mode": self.config['optimization_mode'],
                "parameter": parameter,
                "string_param": string_param
            }
            
        except ValidationError as e:
            logger.error(f"Validation error: {e}")
            return create_standardized_error_response(
                "Data validation failed", str(e), parameter, string_param)
        except Exception as e:
            logger.error(f"Assignment failed: {e}", exc_info=True)
            return create_standardized_error_response(
                "Assignment failed", str(e), parameter, string_param)
    
    def _run_assignment_algorithm(self, user_df, driver_df, office_lat, office_lon):
        """Run the specific assignment algorithm based on type"""
        if self.assignment_type == "route_assignment":
            return self._route_efficiency_assignment(user_df, driver_df, office_lat, office_lon)
        else:
            return self._balance_assignment(user_df, driver_df, office_lat, office_lon)
    
    def _route_efficiency_assignment(self, user_df, driver_df, office_lat, office_lon):
        """Route efficiency assignment focusing on straight routes"""
        # Step 1: Direction-aware geographic clustering
        user_df = self._create_direction_aware_clusters(user_df, office_lat, office_lon)
        
        # Step 2: Capacity-based sub-clustering
        user_df = self._create_capacity_subclusters(user_df)
        
        # Step 3: Assign drivers with route efficiency optimization
        routes, assigned_user_ids = self._assign_drivers_route_efficiency(
            user_df, driver_df, office_lat, office_lon)
        
        # Step 4: Handle unassigned users
        unassigned_users = self._handle_unassigned_users(user_df, assigned_user_ids)
        
        return routes, unassigned_users
    
    def _balance_assignment(self, user_df, driver_df, office_lat, office_lon):
        """Balance assignment focusing on utilization"""
        # Step 1: Geographic clustering
        user_df = self._create_geographic_clusters(user_df, office_lat, office_lon)
        
        # Step 2: Capacity-based sub-clustering
        user_df = self._create_simple_subclusters(user_df, driver_df)
        
        # Step 3: Assign drivers with balance optimization
        routes, assigned_user_ids = self._assign_drivers_balance(
            user_df, driver_df, office_lat, office_lon)
        
        # Step 4: Global optimization for balance
        routes, unassigned_users = self._balance_global_optimization(
            routes, user_df, assigned_user_ids, driver_df, office_lat, office_lon)
        
        return routes, unassigned_users
    
    def _create_direction_aware_clusters(self, user_df, office_lat, office_lon):
        """Create direction-aware clusters for route efficiency"""
        if len(user_df) <= 1:
            user_df['geo_cluster'] = 0
            return user_df
        
        # Use angular sectors based clustering
        n_sectors = 12
        sector_angle = 360.0 / n_sectors
        
        user_df['sector'] = (user_df['bearing_from_office'] // sector_angle).astype(int)
        
        labels = [-1] * len(user_df)
        current_cluster = 0
        max_users_per_cluster = 6
        
        for sector in range(n_sectors):
            sector_users = user_df[user_df['sector'] == sector]
            if len(sector_users) == 0:
                continue
                
            if len(sector_users) <= max_users_per_cluster:
                for idx in sector_users.index:
                    labels[idx] = current_cluster
                current_cluster += 1
            else:
                # Split large sectors
                coords_km = self._convert_to_metric_coords(sector_users, office_lat, office_lon)
                n_subclusters = math.ceil(len(sector_users) / max_users_per_cluster)
                
                kmeans = KMeans(n_clusters=n_subclusters, random_state=42, n_init=5)
                sublabels = kmeans.fit_predict(coords_km)
                
                for i, idx in enumerate(sector_users.index):
                    labels[idx] = current_cluster + sublabels[i]
                
                current_cluster += n_subclusters
        
        user_df['geo_cluster'] = labels
        return user_df
    
    def _create_geographic_clusters(self, user_df, office_lat, office_lon):
        """Create simple geographic clusters for balance assignment"""
        if len(user_df) <= 1:
            user_df['geo_cluster'] = 0
            return user_df
        
        coords = user_df[['latitude', 'longitude']].values
        scaler = StandardScaler()
        scaled_coords = scaler.fit_transform(coords)
        
        # Try DBSCAN first
        clustering = DBSCAN(eps=0.3, min_samples=self.config['MIN_SAMPLES_DBSCAN']).fit(scaled_coords)
        user_df['geo_cluster'] = clustering.labels_
        
        # Handle noise points
        noise_mask = user_df['geo_cluster'] == -1
        if noise_mask.any():
            for idx in user_df[noise_mask].index:
                user_pos = coords[user_df.index.get_loc(idx)]
                min_dist = float('inf')
                best_cluster = 0
                
                for cluster_id in user_df[user_df['geo_cluster'] != -1]['geo_cluster'].unique():
                    cluster_center = user_df[user_df['geo_cluster'] == cluster_id][
                        ['latitude', 'longitude']].mean()
                    dist = haversine_distance(user_pos[0], user_pos[1],
                                            cluster_center['latitude'], cluster_center['longitude'])
                    if dist < min_dist:
                        min_dist = dist
                        best_cluster = cluster_id
                
                user_df.loc[idx, 'geo_cluster'] = best_cluster
        
        # Fallback to K-means if too few clusters
        n_clusters = user_df['geo_cluster'].nunique()
        if n_clusters <= 1:
            optimal_clusters = min(6, max(2, len(user_df) // 4))
            kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
            user_df['geo_cluster'] = kmeans.fit_predict(scaled_coords)
        
        return user_df
    
    def _create_capacity_subclusters(self, user_df):
        """Create capacity-based subclusters for route efficiency"""
        user_df['sub_cluster'] = -1
        sub_cluster_counter = 0
        max_users_per_cluster = 5
        max_bearing_diff = self.config['MAX_BEARING_DIFFERENCE']
        
        for geo_cluster in user_df['geo_cluster'].unique():
            geo_cluster_users = user_df[user_df['geo_cluster'] == geo_cluster]
            
            if len(geo_cluster_users) <= max_users_per_cluster:
                user_df.loc[geo_cluster_users.index, 'sub_cluster'] = sub_cluster_counter
                sub_cluster_counter += 1
            else:
                # Sort by bearing and create smaller clusters
                sorted_users = geo_cluster_users.sort_values('bearing_from_office')
                current_cluster_users = []
                
                for idx, (user_idx, user) in enumerate(sorted_users.iterrows()):
                    if current_cluster_users:
                        bearing_spread = self._calculate_bearing_spread(
                            [u[1] for u in current_cluster_users] + [user])
                        if len(current_cluster_users) >= max_users_per_cluster or bearing_spread > max_bearing_diff:
                            for cluster_user_idx, _ in current_cluster_users:
                                user_df.loc[cluster_user_idx, 'sub_cluster'] = sub_cluster_counter
                            sub_cluster_counter += 1
                            current_cluster_users = []
                    
                    current_cluster_users.append((user_idx, user))
                
                if current_cluster_users:
                    for cluster_user_idx, _ in current_cluster_users:
                        user_df.loc[cluster_user_idx, 'sub_cluster'] = sub_cluster_counter
                    sub_cluster_counter += 1
        
        return user_df
    
    def _create_simple_subclusters(self, user_df, driver_df):
        """Create simple capacity-based subclusters for balance assignment"""
        avg_capacity = int(driver_df['capacity'].mean())
        max_capacity = int(driver_df['capacity'].max())
        
        user_df['sub_cluster'] = 0
        sub_cluster_counter = 0
        
        for geo_cluster_id, geo_cluster_users in user_df.groupby('geo_cluster'):
            cluster_size = len(geo_cluster_users)
            
            if cluster_size <= max_capacity:
                user_df.loc[geo_cluster_users.index, 'sub_cluster'] = sub_cluster_counter
                sub_cluster_counter += 1
            else:
                coords = geo_cluster_users[['latitude', 'longitude']].values
                n_subclusters = min(math.ceil(cluster_size / avg_capacity), cluster_size)
                
                if n_subclusters > 1:
                    kmeans = KMeans(n_clusters=n_subclusters, random_state=42)
                    sub_labels = kmeans.fit_predict(coords)
                    
                    for i, sub_label in enumerate(sub_labels):
                        user_idx = geo_cluster_users.index[i]
                        user_df.loc[user_idx, 'sub_cluster'] = sub_cluster_counter + sub_label
                    
                    sub_cluster_counter += n_subclusters
                else:
                    user_df.loc[geo_cluster_users.index, 'sub_cluster'] = sub_cluster_counter
                    sub_cluster_counter += 1
        
        return user_df
    
    def _assign_drivers_route_efficiency(self, user_df, driver_df, office_lat, office_lon):
        """Assign drivers with route efficiency optimization"""
        routes = []
        assigned_user_ids = set()
        used_driver_ids = set()
        
        available_drivers = driver_df.sort_values(['priority', 'capacity'], ascending=[True, False])
        
        for sub_cluster_id in sorted(user_df['sub_cluster'].unique()):
            if sub_cluster_id == -1:
                continue
                
            cluster_users = user_df[user_df['sub_cluster'] == sub_cluster_id]
            unassigned_in_cluster = cluster_users[~cluster_users['user_id'].isin(assigned_user_ids)]
            
            if unassigned_in_cluster.empty:
                continue
            
            route = self._assign_best_driver_to_cluster(
                unassigned_in_cluster, available_drivers, used_driver_ids, office_lat, office_lon)
            
            if route:
                routes.append(route)
                assigned_user_ids.update(u['user_id'] for u in route['assigned_users'])
        
        return routes, assigned_user_ids
    
    def _assign_drivers_balance(self, user_df, driver_df, office_lat, office_lon):
        """Assign drivers with balance optimization"""
        routes = []
        assigned_user_ids = set()
        used_driver_ids = set()
        
        available_drivers = driver_df.sort_values(['priority', 'capacity'], ascending=[True, False])
        
        for sub_cluster_id, cluster_users in user_df.groupby('sub_cluster'):
            unassigned_in_cluster = cluster_users[~cluster_users['user_id'].isin(assigned_user_ids)]
            
            if unassigned_in_cluster.empty:
                continue
            
            route = self._assign_best_driver_to_cluster_balance(
                unassigned_in_cluster, available_drivers, used_driver_ids, office_lat, office_lon)
            
            if route:
                routes.append(route)
                assigned_user_ids.update(u['user_id'] for u in route['assigned_users'])
        
        return routes, assigned_user_ids
    
    def _assign_best_driver_to_cluster(self, cluster_users, available_drivers, used_driver_ids, office_lat, office_lon):
        """Find best driver for cluster with route efficiency optimization"""
        cluster_center = cluster_users[['latitude', 'longitude']].mean()
        cluster_size = len(cluster_users)
        
        best_driver = None
        min_cost = float('inf')
        
        for _, driver in available_drivers.iterrows():
            if driver['driver_id'] in used_driver_ids or driver['capacity'] < cluster_size:
                continue
            
            distance = haversine_distance(
                driver['latitude'], driver['longitude'],
                cluster_center['latitude'], cluster_center['longitude'])
            
            # Route efficiency cost calculation
            priority_penalty = driver['priority'] * 0.5
            utilization = cluster_size / driver['capacity']
            utilization_bonus = utilization * self.config['capacity_weight'] * 2.0
            
            total_cost = distance + priority_penalty - utilization_bonus
            
            if total_cost < min_cost:
                min_cost = total_cost
                best_driver = driver
        
        if best_driver is not None:
            used_driver_ids.add(best_driver['driver_id'])
            return self._create_route(best_driver, cluster_users.head(best_driver['capacity']))
        
        return None
    
    def _assign_best_driver_to_cluster_balance(self, cluster_users, available_drivers, used_driver_ids, office_lat, office_lon):
        """Find best driver for cluster with balance optimization"""
        cluster_center = cluster_users[['latitude', 'longitude']].mean()
        cluster_size = len(cluster_users)
        
        best_driver = None
        min_cost = float('inf')
        
        for _, driver in available_drivers.iterrows():
            if driver['driver_id'] in used_driver_ids or driver['capacity'] < cluster_size:
                continue
            
            distance = haversine_distance(
                driver['latitude'], driver['longitude'],
                cluster_center['latitude'], cluster_center['longitude'])
            
            # Balance optimization cost calculation
            priority_penalty = driver['priority'] * 1.0
            utilization = cluster_size / driver['capacity']
            utilization_bonus = utilization * 4.0  # Higher bonus for balance
            
            total_cost = distance + priority_penalty - utilization_bonus
            
            if total_cost < min_cost:
                min_cost = total_cost
                best_driver = driver
        
        if best_driver is not None:
            used_driver_ids.add(best_driver['driver_id'])
            return self._create_route(best_driver, cluster_users.head(best_driver['capacity']))
        
        return None
    
    def _create_route(self, driver, users_to_assign):
        """Create a route from driver and users"""
        route = {
            'driver_id': str(driver['driver_id']),
            'vehicle_id': str(driver.get('vehicle_id', '')),
            'vehicle_type': int(driver['capacity']),
            'latitude': float(driver['latitude']),
            'longitude': float(driver['longitude']),
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
        
        return route
    
    def _balance_global_optimization(self, routes, user_df, assigned_user_ids, driver_df, office_lat, office_lon):
        """Global optimization for balance assignment"""
        # Fill underutilized routes
        unassigned_users = user_df[~user_df['user_id'].isin(assigned_user_ids)]
        
        for route in routes:
            if len(route['assigned_users']) < route['vehicle_type'] and not unassigned_users.empty:
                route_center = [route['latitude'], route['longitude']]
                if route['assigned_users']:
                    lats = [u['lat'] for u in route['assigned_users']]
                    lngs = [u['lng'] for u in route['assigned_users']]
                    route_center = [np.mean(lats), np.mean(lngs)]
                
                # Find nearby users
                for _, user in unassigned_users.iterrows():
                    if len(route['assigned_users']) >= route['vehicle_type']:
                        break
                    
                    distance = haversine_distance(
                        route_center[0], route_center[1],
                        user['latitude'], user['longitude'])
                    
                    if distance <= self.config['MAX_FILL_DISTANCE_KM'] * 1.5:
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
        
        # Create unassigned users list
        remaining_unassigned = user_df[~user_df['user_id'].isin(assigned_user_ids)]
        unassigned_list = []
        for _, user in remaining_unassigned.iterrows():
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
        
        return routes, unassigned_list
    
    def _handle_unassigned_users(self, user_df, assigned_user_ids):
        """Handle unassigned users for route efficiency"""
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
    
    def _convert_to_metric_coords(self, users_df, office_lat, office_lon):
        """Convert coordinates to metric system"""
        coords_km = []
        for _, user in users_df.iterrows():
            lat_km = (user['latitude'] - office_lat) * self.config['LAT_TO_KM']
            lon_km = (user['longitude'] - office_lon) * self.config['LON_TO_KM']
            coords_km.append([lat_km, lon_km])
        return np.array(coords_km)
    
    def _calculate_bearing_spread(self, users):
        """Calculate angular spread of users"""
        if len(users) <= 1:
            return 0
        
        bearings = [user['bearing_from_office'] for user in users]
        bearings.sort()
        
        max_gap = 0
        for i in range(len(bearings)):
            gap = bearings[(i + 1) % len(bearings)] - bearings[i]
            if gap < 0:
                gap += 360
            max_gap = max(max_gap, gap)
        
        return 360 - max_gap if max_gap > 180 else max_gap
    
    def _empty_assignment_response(self, start_time, parameter, string_param, data):
        """Handle empty users case"""
        return {
            "status": "true",
            "execution_time": (pd.Timestamp.now() - start_time).total_seconds(),
            "data": [],
            "unassignedUsers": [],
            "unassignedDrivers": self._get_all_drivers_as_unassigned(data),
            "clustering_analysis": {"method": "No Users", "clusters": 0},
            "optimization_mode": self.config['optimization_mode'],
            "parameter": parameter,
            "string_param": string_param
        }
    
    def _no_drivers_response(self, start_time, parameter, string_param, users):
        """Handle no drivers case"""
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
        
        return {
            "status": "true",
            "execution_time": (pd.Timestamp.now() - start_time).total_seconds(),
            "data": [],
            "unassignedUsers": unassigned_users,
            "unassignedDrivers": [],
            "clustering_analysis": {"method": "No Drivers", "clusters": 0},
            "optimization_mode": self.config['optimization_mode'],
            "parameter": parameter,
            "string_param": string_param
        }
    
    def _get_all_drivers_as_unassigned(self, data):
        """Get all drivers in unassigned format"""
        all_drivers = []
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
    
    def _build_unassigned_drivers_list(self, driver_df, assigned_driver_ids):
        """Build list of unassigned drivers"""
        unassigned_drivers = []
        for _, driver in driver_df.iterrows():
            if driver['driver_id'] not in assigned_driver_ids:
                unassigned_drivers.append({
                    'driver_id': str(driver['driver_id']),
                    'capacity': int(driver['capacity']),
                    'vehicle_id': str(driver.get('vehicle_id', '')),
                    'latitude': float(driver['latitude']),
                    'longitude': float(driver['longitude'])
                })
        return unassigned_drivers

def run_assignment(source_id: str, parameter: int = 1, string_param: str = "", assignment_type: str = "route_assignment"):
    """Main assignment function"""
    engine = AssignmentEngine(assignment_type)
    return engine.run_assignment(source_id, parameter, string_param)
