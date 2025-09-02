import json
import time
import math
import os
import pandas as pd
from road_network import RoadNetwork
import networkx as nx
import traceback

# Global variables that will be set during assignment
ROAD_NETWORK = None
OPTIMIZATION_CONFIGS = {
    "efficiency": {"max_detour_ratio": 1.1},
    "balanced": {"max_detour_ratio": 1.3},
    "capacity": {"max_detour_ratio": 1.5}
}

class SimpleRouteAssigner:
    """Simple route assignment utility for fallback scenarios"""

    def __init__(self, config):
        self.config = config
        self.road_network = None

    def assign_routes(self, users, drivers, office_pos):
        """Simple route assignment method"""
        routes = []
        assigned_user_ids = set()
        used_driver_ids = set()

        for driver in drivers:
            if len(assigned_user_ids) >= len(users):
                break

            driver_route = {
                'driver_id': str(driver['id']),
                'vehicle_id': str(driver.get('vehicle_id', '')),
                'vehicle_type': int(driver['capacity']),
                'latitude': float(driver['latitude']),
                'longitude': float(driver['longitude']),
                'assigned_users': []
            }

            # Assign closest users to this driver
            driver_pos = (driver['latitude'], driver['longitude'])
            available_users = [u for u in users if str(u['id']) not in assigned_user_ids]

            # Sort by distance to driver
            available_users.sort(key=lambda u: haversine_distance(
                driver_pos[0], driver_pos[1], 
                float(u['latitude']), float(u['longitude'])
            ))

            users_assigned = 0
            for user in available_users:
                if users_assigned >= driver['capacity']:
                    break

                user_data = {
                    'user_id': str(user['id']),
                    'lat': float(user['latitude']),
                    'lng': float(user['longitude']),
                    'office_distance': haversine_distance(
                        float(user['latitude']), float(user['longitude']),
                        office_pos[0], office_pos[1]
                    )
                }
                if user.get('first_name'):
                    user_data['first_name'] = str(user['first_name'])
                if user.get('email'):
                    user_data['email'] = str(user['email'])

                driver_route['assigned_users'].append(user_data)
                assigned_user_ids.add(str(user['id']))
                users_assigned += 1

            if driver_route['assigned_users']:
                routes.append(driver_route)
                used_driver_ids.add(str(driver['id']))

        return routes, assigned_user_ids, used_driver_ids


class RoadCorridorAnalyzer:
    """Analyzes road network to identify natural travel corridors from office"""

    def __init__(self, road_network, office_pos):
        self.road_network = road_network
        self.office_pos = office_pos
        office_node, office_dist = road_network.find_nearest_road_node(office_pos[0], office_pos[1])
        if office_dist > 3.0:  # Office too far from road network
            print(f"Warning: Office is {office_dist:.2f}km from nearest road node")
            self.office_node = None
        else:
            self.office_node = office_node
        self.corridors = self._identify_corridors()

    def _identify_corridors(self):
        """Identify major road corridors radiating from office"""
        if not self.office_node:
            return {}

        corridors = {}
        # Define 8 cardinal directions
        directions = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
        bearings = [0, 45, 90, 135, 180, 225, 270, 315]

        for direction, bearing in zip(directions, bearings):
            corridor_nodes = self._find_corridor_nodes(bearing)
            if corridor_nodes:
                corridors[direction] = {
                    'bearing': bearing,
                    'nodes': corridor_nodes,
                    'users': []
                }

        return corridors

    def _find_corridor_nodes(self, target_bearing, tolerance=22.5):
        """Find road nodes that extend in a specific direction from office"""
        corridor_nodes = []
        visited = set()
        queue = [self.office_node]
        max_iterations = 1000  # Prevent infinite loops

        iteration_count = 0
        while queue and len(corridor_nodes) < 50 and iteration_count < max_iterations:
            iteration_count += 1
            current_node = queue.pop(0)
            if current_node in visited:
                continue
            visited.add(current_node)

            # Check if this node is in the target direction
            if current_node != self.office_node:
                try:
                    current_pos = self.road_network.node_positions.get(current_node)
                    if current_pos:
                        bearing = self.road_network._calculate_bearing(
                            self.office_pos[0], self.office_pos[1],
                            current_pos[0], current_pos[1]
                        )
                        bearing_diff = abs(self._normalize_bearing_difference(bearing - target_bearing))

                        if bearing_diff <= tolerance:
                            corridor_nodes.append(current_node)
                except Exception as e:
                    continue  # Skip problematic nodes

            # Add neighbors to queue with limit
            try:
                if hasattr(self.road_network.graph, 'neighbors'):
                    neighbors = list(self.road_network.graph.neighbors(current_node))
                    # Limit neighbors to prevent explosion
                    for neighbor in neighbors[:10]:  # Max 10 neighbors per node
                        if neighbor not in visited and len(queue) < 200:  # Limit queue size
                            queue.append(neighbor)
            except Exception as e:
                continue  # Skip problematic neighbors

        print(f"    Found {len(corridor_nodes)} nodes for bearing {target_bearing}° after {iteration_count} iterations")
        return corridor_nodes

    def _normalize_bearing_difference(self, diff):
        """Normalize bearing difference to [-180, 180] range"""
        while diff > 180:
            diff -= 360
        while diff < -180:
            diff += 360
        return diff

    def assign_users_to_corridors(self, users):
        """Assign users to the most appropriate corridor"""
        for user in users:
            user_pos = (float(user['latitude']), float(user['longitude']))
            user_bearing = self.road_network._calculate_bearing(
                self.office_pos[0], self.office_pos[1],
                user_pos[0], user_pos[1]
            )

            # Find closest corridor by bearing
            best_corridor = None
            min_bearing_diff = float('inf')

            for direction, corridor in self.corridors.items():
                bearing_diff = abs(self._normalize_bearing_difference(
                    user_bearing - corridor['bearing']
                ))
                if bearing_diff < min_bearing_diff:
                    min_bearing_diff = bearing_diff
                    best_corridor = direction

            if best_corridor and min_bearing_diff <= 45:  # Within 45 degrees
                user_data = {
                    'user_id': str(user['id']),
                    'lat': float(user['latitude']),
                    'lng': float(user['longitude']),
                    'office_distance': haversine_distance(
                        user_pos[0], user_pos[1], 
                        self.office_pos[0], self.office_pos[1]
                    )
                }
                if user.get('first_name'):
                    user_data['first_name'] = str(user['first_name'])
                if user.get('email'):
                    user_data['email'] = str(user['email'])

                self.corridors[best_corridor]['users'].append(user_data)

    def get_corridor_routes(self):
        """Generate route backbones for each corridor with users"""
        corridor_routes = []

        for direction, corridor in self.corridors.items():
            if corridor['users']:
                # Sort users by distance from office (closest first for pickup sequence)
                corridor['users'].sort(key=lambda u: u['office_distance'])

                corridor_routes.append({
                    'direction': direction,
                    'bearing': corridor['bearing'],
                    'users': corridor['users']
                })

        # Fallback corridors would be handled by the calling function
        return corridor_routes


    def assign_users_to_routes(self, users, drivers, office_pos, routes, assigned_user_ids, used_driver_ids, mode):
        """Assigns users to existing routes based on the specified mode."""

        available_drivers = [d for d in drivers if str(d['id']) not in used_driver_ids]

        # Sort users by their distance to the office for efficient processing
        # This helps in prioritizing users closer to the final destination
        sorted_users = sorted(users, key=lambda u: haversine_distance(float(u['latitude']), float(u['longitude']), office_pos[0], office_pos[1]))

        for user in sorted_users:
            if str(user['id']) in assigned_user_ids:
                continue

            best_route_for_user = None
            best_route_score = -float('inf')

            # Try to assign user to an existing route
            for route in routes:
                if len(route['assigned_users']) >= route['vehicle_type']:
                    continue

                driver_pos = (route['latitude'], route['longitude'])
                route_users = route['assigned_users']
                user_pos = (float(user['latitude']), float(user['longitude']))

                # ENHANCED DISTANCE CONSTRAINTS - Prevent scattered routes
                # 1. Maximum distance from driver to user
                driver_to_user_distance = haversine_distance(driver_pos[0], driver_pos[1], user_pos[0], user_pos[1])
                MAX_DRIVER_USER_DISTANCE = 8.0  # Tighter constraint
                if driver_to_user_distance > MAX_DRIVER_USER_DISTANCE:
                    continue

                # 2. Maximum distance between any two users in the route
                if route_users:
                    max_user_to_user_distance = 0.0
                    for existing_user in route_users:
                        existing_pos = (existing_user['lat'], existing_user['lng'])
                        distance = haversine_distance(user_pos[0], user_pos[1], existing_pos[0], existing_pos[1])
                        max_user_to_user_distance = max(max_user_to_user_distance, distance)
                    
                    MAX_USER_USER_DISTANCE = 6.0  # Users shouldn't be more than 6km apart
                    if max_user_to_user_distance > MAX_USER_USER_DISTANCE:
                        continue

                # 3. DIRECTIONAL COHERENCE CHECK - Prevent zigzag routes
                if route_users and self.road_network:
                    # Calculate main direction from driver to office
                    main_bearing = self.road_network._calculate_bearing(driver_pos[0], driver_pos[1], office_pos[0], office_pos[1])
                    
                    # Check if adding this user maintains directional coherence
                    user_bearing = self.road_network._calculate_bearing(driver_pos[0], driver_pos[1], user_pos[0], user_pos[1])
                    bearing_difference = abs(self.road_network._normalize_bearing_difference(user_bearing - main_bearing))
                    
                    MAX_BEARING_DEVIATION = 45.0  # Allow maximum 45° deviation from main direction
                    if bearing_difference > MAX_BEARING_DEVIATION:
                        print(f"    ✖ user {user['id']} rejected: bearing deviation {bearing_difference:.1f}° > {MAX_BEARING_DEVIATION}°")
                        continue

                # 4. ROUTE COMPACTNESS CHECK - Ensure route doesn't become too spread out
                if route_users:
                    # Calculate route bounding box diagonal
                    all_lats = [u['lat'] for u in route_users] + [user_pos[0]]
                    all_lngs = [u['lng'] for u in route_users] + [user_pos[1]]
                    
                    lat_range = max(all_lats) - min(all_lats)
                    lng_range = max(all_lngs) - min(all_lngs)
                    
                    # Convert to approximate distance
                    lat_distance = lat_range * 111.0  # 1° lat ≈ 111km
                    lng_distance = lng_range * 111.0 * math.cos(math.radians(sum(all_lats) / len(all_lats)))
                    diagonal_distance = math.sqrt(lat_distance**2 + lng_distance**2)
                    
                    MAX_ROUTE_DIAGONAL = 12.0  # Route shouldn't span more than 12km diagonal
                    if diagonal_distance > MAX_ROUTE_DIAGONAL:
                        print(f"    ✖ user {user['id']} rejected: route diagonal {diagonal_distance:.1f}km > {MAX_ROUTE_DIAGONAL}km")
                        continue

                # Determine route type for road network functions
                route_type = "straight" if mode == "efficiency" else "balanced" if mode == "balanced" else "capacity"

                # Check if user is on route path with road network awareness
                config = OPTIMIZATION_CONFIGS.get(mode, OPTIMIZATION_CONFIGS["balanced"])
                max_detour = config.get('max_detour_ratio', 1.2)  # Tighter detour ratio

                # Enhanced coherence and path check
                if self.road_network:
                    test_users = [(u['lat'], u['lng']) for u in route_users] + [user_pos]
                    coherence = self.road_network.get_route_coherence_score(
                        driver_pos, test_users, office_pos
                    )

                    # Stricter coherence threshold
                    if coherence < 0.6:  # Increased from 0.4
                        print(f"    ✖ user {user['id']} rejected: low coherence {coherence:.2f}")
                        continue

                    # Check if user is on route path
                    on_path = self.road_network.is_user_on_route_path(
                        driver_pos, [(u['lat'], u['lng']) for u in route_users],
                        user_pos, office_pos, max_detour_ratio=max_detour, route_type=route_type
                    )

                    if not on_path:
                        print(f"    ✖ user {user['id']} not on route path for route {route['driver_id']}")
                        continue

                # Sequential routing - no backtracking toward office
                if route_users:
                    user_office_distance = haversine_distance(user_pos[0], user_pos[1], office_pos[0], office_pos[1])
                    last_user_office_distance = min([
                        haversine_distance(u['lat'], u['lng'], office_pos[0], office_pos[1]) 
                        for u in route_users
                    ])
                    # Strict check - no tolerance for backtracking
                    if user_office_distance > last_user_office_distance + 0.1:  # 100m tolerance only
                        continue  # Skip users that force backtracking

                # Check capacity - get capacity from route
                if len(route_users) >= route['vehicle_type']:
                    continue

                # Use copy for testing instead of mutating the original list
                test_route_users = route_users.copy()
                test_user_data = {
                    'user_id': str(user['id']),
                    'lat': float(user['latitude']),
                    'lng': float(user['longitude']),
                    'office_distance': haversine_distance(
                        float(user['latitude']), float(user['longitude']), 
                        office_pos[0], office_pos[1]
                    )
                }
                if user.get('first_name'):
                    test_user_data['first_name'] = str(user['first_name'])
                if user.get('email'):
                    test_user_data['email'] = str(user['email'])

                test_route_users.append(test_user_data)

                # Calculate office distances for all users if not present
                for u in test_route_users:
                    if 'office_distance' not in u or u['office_distance'] == 0:
                        u['office_distance'] = haversine_distance(
                            u['lat'], u['lng'], office_pos[0], office_pos[1]
                        )

                # Create test route for scoring
                test_route = route.copy()
                test_route['assigned_users'] = test_route_users

                # For straight routes, use sequential ordering by office distance (no backtracking)
                # For other modes, use the road network optimizer
                if route_type == "straight":
                    # Sort by office distance to ensure monotonic progression toward office
                    test_route_users.sort(key=lambda u: u['office_distance'])
                    optimized_sequence = list(range(len(test_route_users)))
                else:
                    # Use road network optimization for balanced/capacity modes
                    optimized_sequence = self.road_network.get_optimal_pickup_sequence(
                        driver_pos,
                        [(u['lat'], u['lng']) for u in test_route_users],
                        office_pos
                    ) if self.road_network else list(range(len(test_route_users)))

                current_route_score = self.calculate_route_score(test_route, driver_pos, office_pos, optimized_sequence)

                if current_route_score > best_route_score:
                    best_route_score = current_route_score
                    best_route_for_user = route


            # If a suitable existing route is found, assign the user
            if best_route_for_user:
                route_users = best_route_for_user['assigned_users']
                driver_pos = (best_route_for_user['latitude'], best_route_for_user['longitude'])

                # Recalculate office distances and re-sort/optimize
                for u in route_users:
                    if 'office_distance' not in u or u['office_distance'] == 0:
                        u['office_distance'] = haversine_distance(
                            u['lat'], u['lng'], office_pos[0], office_pos[1]
                        )

                route_type = "straight" if mode == "efficiency" else "balanced" if mode == "balanced" else "capacity"
                if route_type == "straight":
                    route_users.sort(key=lambda u: u['office_distance'])
                    optimized_sequence = list(range(len(route_users)))
                else:
                    optimized_sequence = self.road_network.get_optimal_pickup_sequence(
                        driver_pos,
                        [(float(user['latitude']), float(user['longitude'])) for user in route_users],
                        office_pos
                    ) if self.road_network else list(range(len(route_users)))

                # Add user to the route with optimized sequence
                final_user_data = {
                    'user_id': str(user['id']),
                    'lat': float(user['latitude']),
                    'lng': float(user['longitude']),
                    'office_distance': haversine_distance(
                        float(user['latitude']), float(user['longitude']), office_pos[0], office_pos[1]
                    )
                }
                if user.get('first_name'):
                    final_user_data['first_name'] = str(user['first_name'])
                if user.get('email'):
                    final_user_data['email'] = str(user['email'])

                best_route_for_user['assigned_users'].append(final_user_data)
                assigned_user_ids.add(str(user['id']))


        # Try to fill existing routes with users from other corridors
        # (This part needs to be integrated with the main assignment logic)

        return routes, assigned_user_ids


    def calculate_route_score(self, route, driver_pos, office_pos, optimized_sequence):
        """Calculates a score for a given route with heavy penalty for scattered routes."""
        score = 0
        num_users = len(route['assigned_users'])
        
        if not route['assigned_users']:
            return score

        # Base score for number of users
        score += num_users * 10

        # Calculate route metrics
        total_route_dist = 0
        if optimized_sequence:
            path = [driver_pos] + [(route['assigned_users'][i]['lat'], route['assigned_users'][i]['lng']) for i in optimized_sequence]
            
            # Calculate total route distance
            for i in range(len(path) - 1):
                total_route_dist += haversine_distance(path[i][0], path[i][1], path[i+1][0], path[i+1][1])

            # Add final leg to office
            if path:
                total_route_dist += haversine_distance(path[-1][0], path[-1][1], office_pos[0], office_pos[1])

            direct_dist = haversine_distance(driver_pos[0], driver_pos[1], office_pos[0], office_pos[1])
            
            if direct_dist > 0:
                detour_ratio = total_route_dist / direct_dist
                # Heavy penalty for high detour ratios (scattered routes)
                if detour_ratio > 2.0:
                    score -= detour_ratio * 50  # Severe penalty for very scattered routes
                elif detour_ratio > 1.5:
                    score -= detour_ratio * 30  # Strong penalty for scattered routes
                else:
                    score -= detour_ratio * 15  # Normal penalty

            # COMPACTNESS PENALTY - penalize routes that are geographically spread out
            if len(route['assigned_users']) > 1:
                all_lats = [u['lat'] for u in route['assigned_users']]
                all_lngs = [u['lng'] for u in route['assigned_users']]
                
                lat_range = max(all_lats) - min(all_lats)
                lng_range = max(all_lngs) - min(all_lngs)
                
                # Convert to approximate distance
                lat_distance = lat_range * 111.0  # 1° lat ≈ 111km
                avg_lat = sum(all_lats) / len(all_lats)
                lng_distance = lng_range * 111.0 * math.cos(math.radians(avg_lat))
                
                route_spread = math.sqrt(lat_distance**2 + lng_distance**2)
                
                # Heavy penalty for spread-out routes
                if route_spread > 8.0:
                    score -= route_spread * 10  # Heavy penalty for very spread routes
                elif route_spread > 5.0:
                    score -= route_spread * 5   # Moderate penalty

            # DIRECTIONAL CONSISTENCY BONUS/PENALTY
            if len(route['assigned_users']) > 1 and self.road_network:
                main_bearing = self.road_network._calculate_bearing(driver_pos[0], driver_pos[1], office_pos[0], office_pos[1])
                
                bearing_deviations = []
                for user in route['assigned_users']:
                    user_bearing = self.road_network._calculate_bearing(driver_pos[0], driver_pos[1], user['lat'], user['lng'])
                    deviation = abs(self.road_network._normalize_bearing_difference(user_bearing - main_bearing))
                    bearing_deviations.append(deviation)
                
                avg_deviation = sum(bearing_deviations) / len(bearing_deviations)
                
                # Bonus for good directional consistency
                if avg_deviation < 30:
                    score += 15  # Good directional consistency
                elif avg_deviation > 60:
                    score -= 25  # Poor directional consistency

            # Add coherence score if available
            if self.road_network:
                user_positions = [(u['lat'], u['lng']) for u in route['assigned_users']]
                coherence = self.road_network.get_route_coherence_score(driver_pos, user_positions, office_pos)
                score += coherence * 15  # Increased weight for coherence

            # EFFICIENCY BONUS - prefer routes with good distance efficiency
            if num_users > 1:
                avg_distance_per_user = total_route_dist / num_users
                if avg_distance_per_user < 5.0:  # Very efficient
                    score += 10
                elif avg_distance_per_user > 15.0:  # Very inefficient
                    score -= 20

        return score

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate the great-circle distance between two points on the Earth using built-in math."""
    # Convert latitude and longitude from degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))

    # Radius of earth in kilometers
    r = 6371

    return c * r


def update_route_metrics_improved(route, office_lat, office_lon):
    """Updates route metrics like total distance and number of users."""
    if not route['assigned_users']:
        route['total_distance'] = 0
        route['estimated_time'] = 0
        return

    # Ensure users are sorted by their intended pickup order
    # This assumes the 'assigned_users' list is already in the correct sequence
    # If not, a re-sorting step based on the final sequence would be needed here.

    route_points = [(route['latitude'], route['longitude'])]  # Start with driver's initial position

    # Add user pickup locations in order
    # For simplicity, using the order they appear in assigned_users.
    # A more robust solution would use the optimized_sequence from assignment.
    for user in route['assigned_users']:
        route_points.append((user['lat'], user['lng']))

    # Add office location as the final destination
    route_points.append((office_lat, office_lon))

    total_distance = 0
    for i in range(len(route_points) - 1):
        total_distance += haversine_distance(route_points[i][0], route_points[i][1], route_points[i+1][0], route_points[i+1][1])

    route['total_distance'] = round(total_distance, 2)
    # Placeholder for estimated time calculation
    route['estimated_time'] = round(total_distance / 30, 2) # Assuming average speed of 30 km/h

def assign_routes_road_aware(data):
    """Performs road-aware route assignment using real road network data."""
    # Normalize user data structure
    raw_users = data.get('users', [])
    users = []
    for user in raw_users:
        normalized_user = {
            'id': str(user.get('id', user.get('user_id', ''))),
            'latitude': float(user.get('latitude', user.get('lat', 0.0))),
            'longitude': float(user.get('longitude', user.get('lng', 0.0))),
            'first_name': str(user.get('first_name', '')),
            'email': str(user.get('email', '')),
            'office_distance': float(user.get('office_distance', 0.0))
        }
        users.append(normalized_user)

    # Normalize driver data structure
    drivers_data = data.get('drivers', {})
    drivers_unassigned = drivers_data.get('driversUnassigned', [])
    drivers_assigned = drivers_data.get('driversAssigned', [])
    raw_drivers = drivers_unassigned + drivers_assigned
    all_drivers = []
    for driver in raw_drivers:
        normalized_driver = {
            'id': str(driver.get('id', driver.get('driver_id', ''))),
            'latitude': float(driver.get('latitude', driver.get('lat', 0.0))),
            'longitude': float(driver.get('longitude', driver.get('lng', 0.0))),
            'capacity': int(driver.get('capacity', 1)),
            'vehicle_id': str(driver.get('vehicle_id', '')),
            'shift_type_id': driver.get('shift_type_id', 'Unknown')
        }
        all_drivers.append(normalized_driver)

    office_lat = float(data['company']['latitude'])
    office_lon = float(data['company']['longitude'])
    office_pos = (office_lat, office_lon)
    company = data.get('company', {}).get('name', 'Unknown')
    total_drivers = len(all_drivers)

    print("📊 DETAILED API DATA BREAKDOWN:")
    print(f"   👥 Total Users from API: {len(users)}")
    print(f"   🚗 driversUnassigned: {len(drivers_unassigned)}")
    print(f"   🚙 driversAssigned: {len(drivers_assigned)}")
    print(f"   🚛 Total Drivers Available: {total_drivers}")
    print(f"   🏢 Office Location: ({office_lat}, {office_lon})")
    print(f"   📍 Company: {company}")

    # Detailed user breakdown
    if users:
        print(f"   📋 User ID Range: {users[0]['id']} to {users[-1]['id']}")
        user_locations = [(u.get('lat', 0), u.get('lng', 0)) for u in users]
        print(f"   📍 User Geographic Spread: {len(set(user_locations))} unique locations")
    else:
        print("   📋 Users: No users found in API response")

    # Detailed driver breakdown  
    if all_drivers:
        print(f"   🔑 Driver ID Range: {all_drivers[0]['id']} to {all_drivers[-1]['id']}")
        shift_types = {}
        for driver in all_drivers:
            st = driver.get('shift_type_id', 'Unknown')
            shift_types[st] = shift_types.get(st, 0) + 1
        print(f"   🕐 Shift Type Distribution: {dict(shift_types)}")
    else:
        print("   🔑 Drivers: No drivers found in API response")

    print(f"   📊 Initial Processing Status: READY TO ASSIGN")

    routes = []
    assigned_user_ids = set()
    used_driver_ids = set()

    try:
        # Initialize real RoadNetwork with GraphML file
        print("🗺️ Loading road network from GraphML...")
        global ROAD_NETWORK
        ROAD_NETWORK = RoadNetwork('tricity_main_roads.graphml')
        print(f"✅ Road network loaded successfully with {len(ROAD_NETWORK.graph.nodes)} nodes")

        # Phase 1: Analyze road corridors from office
        print("📍 Analyzing road corridors from office...")
        print(f"   🎯 Office location: ({office_pos[0]:.6f}, {office_pos[1]:.6f})")
        print(f"   🔍 Road network nodes available: {len(ROAD_NETWORK.node_positions)}")
        corridor_analyzer = RoadCorridorAnalyzer(ROAD_NETWORK, office_pos)

        # Phase 2: Assign users to corridors
        print("👥 Assigning users to road corridors...")
        corridor_analyzer.assign_users_to_corridors(users)

        # Phase 3: Apply road-aware clustering within corridors
        print("🛣️ Applying road-aware clustering within corridors...")
        for direction, corridor in corridor_analyzer.corridors.items():
            if len(corridor['users']) > 2:  # Only cluster if enough users
                try:
                    from sklearn.cluster import DBSCAN
                    import numpy as np

                    # Create road distance matrix
                    user_positions = [(u['lat'], u['lng']) for u in corridor['users']]
                    n_users = len(user_positions)

                    if n_users > 1:
                        # Use road distances for clustering
                        distance_matrix = np.zeros((n_users, n_users))
                        for i in range(n_users):
                            for j in range(i+1, n_users):
                                road_dist = ROAD_NETWORK.get_road_distance(
                                    user_positions[i][0], user_positions[i][1],
                                    user_positions[j][0], user_positions[j][1]
                                )
                                distance_matrix[i][j] = road_dist
                                distance_matrix[j][i] = road_dist

                        # Apply DBSCAN with road distances
                        clustering = DBSCAN(eps=2.0, min_samples=2, metric='precomputed')
                        cluster_labels = clustering.fit_predict(distance_matrix)

                        # Reorganize users by clusters
                        clusters = {}
                        for idx, label in enumerate(cluster_labels):
                            if label not in clusters:
                                clusters[label] = []
                            clusters[label].append(corridor['users'][idx])

                        # Update corridor with clustered users (keep largest cluster)
                        if clusters:
                            largest_cluster = max(clusters.values(), key=len)
                            corridor['users'] = largest_cluster
                            print(f"   📍 {direction}: Clustered to {len(largest_cluster)} users")

                except ImportError:
                    print(f"   📍 {direction}: sklearn not available, using geometric clustering")
                except Exception as e:
                    print(f"   📍 {direction}: Clustering failed ({e}), using original assignment")

        # Phase 4: Get corridor-based routes
        corridor_routes = corridor_analyzer.get_corridor_routes()
        print(f"🛣️ Identified {len(corridor_routes)} corridors with users")
        for corridor in corridor_routes:
            print(f"   📍 {corridor['direction']}: {len(corridor['users'])} users")

    except Exception as e:
        print(f"❌ ROAD NETWORK INITIALIZATION FAILED:")
        print(f"   🔍 Error Type: {type(e).__name__}")
        print(f"   📝 Error Message: {str(e)}")
        print(f"   📂 GraphML File: {'EXISTS' if os.path.exists('tricity_main_roads.graphml') else 'MISSING'}")
        print(f"   🔧 Detailed Error Analysis:")
        print(f"      - Error occurred during: Network preparation")
        print(f"      - NetworkX version: {nx.__version__}")
        try:
            import scipy
            print(f"      - Scipy available: {scipy.__version__}")
        except:
            print(f"      - Scipy available: Not installed")
        print(f"   📋 Full traceback: {traceback.format_exc()}")
        print("📍 SWITCHING TO GEOMETRIC FALLBACK MODE...")
        ROAD_NETWORK = None

        # Fallback to simple assignment without road network
        corridor_routes = _create_fallback_corridors(users, office_pos)


    # Mode selection
    mode = "efficiency"
    route_assigner = SimpleRouteAssigner(OPTIMIZATION_CONFIGS[mode])
    if ROAD_NETWORK:
        route_assigner.road_network = ROAD_NETWORK

    # PHASE 1: Create routes based on corridor analysis
    print("🚗 Creating routes along identified corridors...")

    # Sort drivers by capacity (larger capacity first for flexibility)
    sorted_drivers = sorted(all_drivers, key=lambda d: d['capacity'], reverse=True)

    # Create routes for each corridor with users
    for corridor_route in corridor_routes:
        corridor_users = corridor_route['users'].copy()  # Make a copy to avoid modification issues
        if not corridor_users:
            continue

        print(f"  📍 Processing corridor {corridor_route['direction']} with {len(corridor_users)} users")

        # Assign drivers to this corridor
        corridor_assigned = False

        # For each corridor, try to assign drivers until all users are assigned or no drivers left
        corridor_iterations = 0
        max_corridor_iterations = 100  # Increased limit for better coverage

        while corridor_users and corridor_iterations < max_corridor_iterations:
            corridor_iterations += 1
            print(f"    📍 Corridor {corridor_route['direction']} iteration {corridor_iterations}, {len(corridor_users)} users remaining")

            # Find next available driver
            available_driver = None
            for driver in sorted_drivers:
                if driver is None:
                    continue

                driver_id = str(driver['id'])
                if driver_id not in used_driver_ids:
                    available_driver = driver
                    break

            if available_driver is None:
                print(f"    ⚠️ No more drivers available for corridor {corridor_route['direction']}")
                break

            driver = available_driver
            driver_id = str(driver['id'])

            # Create route for this driver along the corridor
            try:
                current_route = {
                    'driver_id': driver_id,
                    'vehicle_id': str(driver.get('vehicle_id', '')),
                    'vehicle_type': int(driver['capacity']),
                    'latitude': float(driver['latitude']),
                    'longitude': float(driver['longitude']),
                    'assigned_users': [],
                    'corridor_direction': corridor_route['direction']
                }

                # Try to fill this driver to capacity with users from this corridor
                users_assigned_to_this_driver = 0
                remaining_corridor_users = []

                user_processing_count = 0

                for user in corridor_users:
                    user_processing_count += 1

                    if user is None or users_assigned_to_this_driver >= driver['capacity']:
                        if user is not None:
                            remaining_corridor_users.append(user)
                        continue

                    try:
                        user_assigned = False

                        if ROAD_NETWORK and route_assigner and route_assigner.road_network:
                            # Use more lenient road network validation
                            driver_pos = (current_route['latitude'], current_route['longitude'])
                            user_pos = (user['lat'], user['lng'])

                            # Simplified validation to prevent hanging
                            try:
                                # Quick distance check first
                                distance = haversine_distance(driver_pos[0], driver_pos[1], user_pos[0], user_pos[1])
                                if distance > 15.0:  # Skip very distant users
                                    remaining_corridor_users.append(user)
                                    continue

                                # Simple road validation without complex path checking
                                current_route['assigned_users'].append(user)
                                assigned_user_ids.add(user['user_id'])
                                users_assigned_to_this_driver += 1
                                user_assigned = True
                            except Exception as road_error:
                                # Fallback to simple assignment
                                current_route['assigned_users'].append(user)
                                assigned_user_ids.add(user['user_id'])
                                users_assigned_to_this_driver += 1
                                user_assigned = True
                        else:
                            # Simple assignment without road validation
                            current_route['assigned_users'].append(user)
                            assigned_user_ids.add(user['user_id'])
                            users_assigned_to_this_driver += 1
                            user_assigned = True

                        if not user_assigned:
                            remaining_corridor_users.append(user)

                    except Exception as user_assign_error:
                        print(f"    ⚠️ Error assigning user {user.get('user_id', 'unknown')}: {user_assign_error}")
                        remaining_corridor_users.append(user)

                # Update corridor_users for next iteration
                corridor_users = remaining_corridor_users

                # Only create route if users were assigned
                if current_route['assigned_users']:
                    try:
                        # Optimize pickup sequence using road network if available
                        if ROAD_NETWORK and route_assigner and route_assigner.road_network:
                            driver_pos = (current_route['latitude'], current_route['longitude'])
                            user_positions = [(u['lat'], u['lng']) for u in current_route['assigned_users']]

                            optimal_sequence = ROAD_NETWORK.get_optimal_pickup_sequence(
                                driver_pos, user_positions, office_pos
                            )

                            # Reorder users based on optimal sequence
                            if optimal_sequence and len(optimal_sequence) == len(current_route['assigned_users']):
                                reordered_users = [current_route['assigned_users'][i] for i in optimal_sequence]
                                current_route['assigned_users'] = reordered_users
                        else:
                            # Simple ordering by distance to office
                            current_route['assigned_users'].sort(key=lambda u: u.get('office_distance', 0))

                        # Update route metrics
                        update_route_metrics_improved(current_route, office_lat, office_lon)
                        routes.append(current_route)

                        print(f"    ✅ Created route for driver {driver_id} with {len(current_route['assigned_users'])} users")
                    except Exception as route_finalize_error:
                        print(f"    ⚠️ Error finalizing route for driver {driver_id}: {route_finalize_error}")

            except Exception as route_create_error:
                print(f"    ⚠️ Error creating route for driver {driver_id}: {route_create_error}")

            # Only mark driver as used if they have any users assigned
            if current_route['assigned_users']:
                used_driver_ids.add(driver_id)
            else:
                # Driver didn't get any users, keep them available for other corridors
                pass

        if not corridor_assigned and corridor_users:
            print(f"    ⚠️ Could not assign {len(corridor_users)} users from corridor {corridor_route['direction']}")

    # PHASE 2: Cross-corridor filling - try to fill existing routes with users from other corridors
    print("🔄 Phase 2: Cross-corridor route filling...")

    remaining_users = [u for u in users if str(u['id']) not in assigned_user_ids]

    # Try to fill existing routes with users from any corridor
    for route in routes:
        if not remaining_users:
            break

        available_capacity = route['vehicle_type'] - len(route['assigned_users'])
        if available_capacity <= 0:
            continue

        route_center = calculate_route_center(route)
        users_added_to_route = 0
        users_to_remove = []

        for user in remaining_users:
            if users_added_to_route >= available_capacity:
                break

            user_pos = (float(user['latitude']), float(user['longitude']))
            distance_to_route = haversine_distance(route_center[0], route_center[1], user_pos[0], user_pos[1])

            # Tighter distance constraint for cross-corridor filling
            if distance_to_route <= 3.0:  # Tighter constraint for compact routes
                # Additional road-aware check
                if ROAD_NETWORK:
                    driver_pos = (route['latitude'], route['longitude'])
                    existing_users = [(u['lat'], u['lng']) for u in route['assigned_users']]
                    user_pos = (float(user['latitude']), float(user['longitude']))

                    # Must pass road path check
                    if not ROAD_NETWORK.is_user_on_route_path(
                        driver_pos, existing_users, user_pos, office_pos,
                        max_detour_ratio=1.2, route_type="balanced"
                    ):
                        continue  # Skip if not on road path

                user_data = {
                    'user_id': str(user['id']),
                    'lat': float(user['latitude']),
                    'lng': float(user['longitude']),
                    'office_distance': haversine_distance(user_pos[0], user_pos[1], office_pos[0], office_pos[1])
                }
                if user.get('first_name'):
                    user_data['first_name'] = str(user['first_name'])
                if user.get('email'):
                    user_data['email'] = str(user['email'])

                route['assigned_users'].append(user_data)
                assigned_user_ids.add(str(user['id']))
                users_added_to_route += 1
                users_to_remove.append(user)

                print(f"    ✅ Cross-corridor: Added user {user['id']} to route {route['driver_id']}")

        # Remove assigned users from remaining list
        for user in users_to_remove:
            remaining_users.remove(user)

        # Re-optimize route if users were added
        if users_added_to_route > 0:
            update_route_metrics_improved(route, office_lat, office_lon)

    # PHASE 3: Handle truly remaining unassigned users with available drivers
    final_remaining_users = [u for u in users if str(u['id']) not in assigned_user_ids]
    final_available_drivers = [d for d in sorted_drivers if str(d['id']) not in used_driver_ids]

    if final_remaining_users and final_available_drivers:
        print(f"🔄 Assigning {len(final_remaining_users)} remaining users to {len(final_available_drivers)} available drivers...")

        for user in final_remaining_users:
            if not final_available_drivers:
                break

            # Find best available driver for this user
            best_driver = None
            best_distance = float('inf')

            for driver in final_available_drivers:
                if str(driver['id']) not in used_driver_ids:
                    user_pos = (float(user['latitude']), float(user['longitude']))
                    driver_pos = (float(driver['latitude']), float(driver['longitude']))

                    if ROAD_NETWORK:
                        distance = ROAD_NETWORK.get_road_distance(
                            driver_pos[0], driver_pos[1], user_pos[0], user_pos[1]
                        )
                    else:
                        distance = haversine_distance(
                            driver_pos[0], driver_pos[1], user_pos[0], user_pos[1]
                        )

                    if distance < best_distance and distance < 15.0:  # Max 15km
                        best_distance = distance
                        best_driver = driver

            if best_driver:
                # Create single-user route
                user_data = {
                    'user_id': str(user['id']),
                    'lat': float(user['latitude']),
                    'lng': float(user['longitude']),
                    'office_distance': haversine_distance(
                        float(user['latitude']), float(user['longitude']),
                        office_lat, office_lon
                    )
                }
                if user.get('first_name'):
                    user_data['first_name'] = str(user['first_name'])
                if user.get('email'):
                    user_data['email'] = str(user['email'])

                single_user_route = {
                    'driver_id': str(best_driver['id']),
                    'vehicle_id': str(best_driver.get('vehicle_id', '')),
                    'vehicle_type': int(best_driver['capacity']),
                    'latitude': float(best_driver['latitude']),
                    'longitude': float(best_driver['longitude']),
                    'assigned_users': [user_data]
                }

                update_route_metrics_improved(single_user_route, office_lat, office_lon)
                routes.append(single_user_route)
                used_driver_ids.add(str(best_driver['id']))

                # Remove this driver from available pool
                final_available_drivers = [d for d in final_available_drivers if str(d['id']) != str(best_driver['id'])]

                print(f"    ✅ Assigned remaining user {user['id']} to driver {best_driver['id']}")

    # PHASE 4: Global optimization pass
    print("🔧 Starting advanced optimization phases...")

    # Phase 1: Merging underutilized routes
    routes, used_driver_ids = merge_underutilized_routes(routes, all_drivers, office_pos, ROAD_NETWORK)

    # Phase 2: Split high-detour routes
    routes, used_driver_ids = split_high_detour_routes(routes, all_drivers, office_pos, ROAD_NETWORK)

    # Phase 3: Post-assignment geographic consolidation
    routes, used_driver_ids = post_assignment_geographic_consolidation(routes, all_drivers, office_pos, ROAD_NETWORK)

    # Phase 4: Local route optimization
    routes = local_route_optimization(routes, [], office_pos, ROAD_NETWORK) # Pass empty unassigned_users as it's handled differently

    # Phase 5: Global optimization pass
    routes = global_optimization_pass(routes, office_pos, ROAD_NETWORK)

    # COMPREHENSIVE FINAL ACCOUNTING

    unassigned_users = []
    for user in users:
        if str(user['id']) not in assigned_user_ids:
            user_data = {
                'user_id': str(user['id']),
                'lat': float(user['latitude']),
                'lng': float(user['longitude']),
                'office_distance': haversine_distance(
                    float(user['latitude']), float(user['longitude']),
                    office_lat, office_lon
                ),
                'reason': 'Not assigned to any route'
            }
            if user.get('first_name'):
                user_data['first_name'] = str(user['first_name'])
            if user.get('email'):
                user_data['email'] = str(user['email'])
            unassigned_users.append(user_data)

    unused_drivers = []
    for driver in all_drivers:
        if str(driver['id']) not in used_driver_ids:
            driver_data = {
                'driver_id': str(driver['id']),
                'capacity': int(driver['capacity']),
                'vehicle_id': str(driver.get('vehicle_id', '')),
                'latitude': float(driver['latitude']),
                'longitude': float(driver['longitude']),
                'reason': 'Not assigned any users',
                'shift_type_id': driver.get('shift_type_id', 'Unknown')
            }
            unused_drivers.append(driver_data)

    total_assigned_users = sum(len(r['assigned_users']) for r in routes)
    total_capacity = sum(r['vehicle_type'] for r in routes)
    utilization = (total_assigned_users / total_capacity * 100) if total_capacity > 0 else 0

    print(f"📊 FINAL ASSIGNMENT SUMMARY:")
    print(f"   🗺️ Road Network Status: {'✅ ACTIVE' if ROAD_NETWORK else '❌ FAILED - Using Geometric Fallback'}")
    print(f"   📍 Routes Created: {len(routes)}")
    print(f"   👥 Users Successfully Assigned: {total_assigned_users}")
    print(f"   🔢 Unique User IDs in Routes: {len(assigned_user_ids)}")
    print(f"   📊 SUCCESS RATE:")
    print(f"      • Total API Users: {len(users)}")
    print(f"      • Successfully Assigned: {len(assigned_user_ids)}")
    print(f"      • Assignment Rate: {(len(assigned_user_ids)/len(users)*100):.1f}%")

    missing_users = set(str(u['id']) for u in users) - assigned_user_ids
    if missing_users:
        print(f"   ❌ LOST USERS ({len(missing_users)}):")
        for user_id in list(missing_users)[:5]:  # Show first 5
            user_data = next((u for u in users if str(u['id']) == user_id), {})
            print(f"      • User {user_id}: {user_data.get('first_name', 'Unknown')} at ({user_data.get('lat', 'N/A')}, {user_data.get('lng', 'N/A')})")
        if len(missing_users) > 5:
            print(f"      ... and {len(missing_users)-5} more users")

    print(f"   🚗 Original API Drivers: {total_drivers}")
    print(f"   ✅ Drivers Used: {len(used_driver_ids)}")
    print(f"   💤 Drivers Unused: {len(unused_drivers)}")

    if unused_drivers:
        print(f"   💤 UNUSED DRIVERS ({len(unused_drivers)}):")
        for driver in unused_drivers[:3]:  # Show first 3
            print(f"      • Driver {driver['driver_id']}: ST:{driver.get('shift_type_id', 'N/A')}, Cap:{driver.get('capacity', 'N/A')}")
        if len(unused_drivers) > 3:
            print(f"      ... and {len(unused_drivers)-3} more drivers")

    total_assigned_users = sum(len(r['assigned_users']) for r in routes)
    total_capacity = sum(r['vehicle_type'] for r in routes)
    utilization = (total_assigned_users / total_capacity * 100) if total_capacity > 0 else 0

    print(f"🎯 Road-aware assignment complete: {len(routes)} routes, {total_assigned_users} users assigned")
    print(f"📊 Overall utilization: {total_assigned_users}/{total_capacity} seats ({(total_assigned_users/total_capacity)*100:.1f}%)")
    print(f"👥 Final Unassigned: {len(unassigned_users)} users, {len(unused_drivers)} drivers")

    users_accounted = len(assigned_user_ids) + len(unassigned_users)
    drivers_accounted = len(used_driver_ids) + len(unused_drivers)
    print(f"✅ User Accounting: {users_accounted}/{len(users)} ({'✅ COMPLETE' if users_accounted == len(users) else '❌ INCOMPLETE'})")
    print(f"✅ Driver Accounting: {drivers_accounted}/{total_drivers} ({'✅ COMPLETE' if drivers_accounted == total_drivers else '❌ INCOMPLETE'})")

    return {
        "status": "true",
        "data": routes,
        "unassignedUsers": unassigned_users,
        "unassignedDrivers": unused_drivers,
        "message": f"Road-aware assignment completed with {len(routes)} routes, {utilization:.1f}% utilization"
    }

def _create_fallback_corridors(users, office_pos):
    """Create simple geometric corridors when road network is unavailable"""
    corridors = {
        'N': {'bearing': 0, 'users': []},
        'NE': {'bearing': 45, 'users': []},
        'E': {'bearing': 90, 'users': []},
        'SE': {'bearing': 135, 'users': []},
        'S': {'bearing': 180, 'users': []},
        'SW': {'bearing': 225, 'users': []},
        'W': {'bearing': 270, 'users': []},
        'NW': {'bearing': 315, 'users': []}
    }

    for user in users:
        user_pos = (float(user['latitude']), float(user['longitude']))
        user_bearing = calculate_bearing(
            office_pos[0], office_pos[1], user_pos[0], user_pos[1]
        )

        # Find closest corridor
        best_corridor = None
        min_diff = float('inf')

        for direction, corridor in corridors.items():
            diff = abs(normalize_bearing_difference(user_bearing - corridor['bearing']))
            if diff < min_diff:
                min_diff = diff
                best_corridor = direction

        if best_corridor and min_diff <= 45:
            user_data = {
                'user_id': str(user['id']),
                'lat': float(user['latitude']),
                'lng': float(user['longitude']),
                'office_distance': haversine_distance(
                    user_pos[0], user_pos[1], office_pos[0], office_pos[1]
                )
            }
            if user.get('first_name'):
                user_data['first_name'] = str(user['first_name'])
            if user.get('email'):
                user_data['email'] = str(user['email'])

            corridors[best_corridor]['users'].append(user_data)

    # Convert to corridor routes format
    corridor_routes = []
    for direction, corridor in corridors.items():
        if corridor['users']:
            corridor['users'].sort(key=lambda u: u['office_distance'])
            corridor_routes.append({
                'direction': direction,
                'bearing': corridor['bearing'],
                'users': corridor['users']
            })

    return corridor_routes

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

def calculate_route_center(route):
    """Calculate center point of a route's users"""
    if not route['assigned_users']:
        return (route['latitude'], route['longitude'])

    avg_lat = sum(u['lat'] for u in route['assigned_users']) / len(route['assigned_users'])
    avg_lng = sum(u['lng'] for u in route['assigned_users']) / len(route['assigned_users'])
    return (avg_lat, avg_lng)


def assign_routes_fallback(data):
    """Fallback assignment function when road network fails"""
    users = data.get('users', [])
    drivers_data = data.get('drivers', {})
    all_drivers = drivers_data.get('driversUnassigned', []) + drivers_data.get('driversAssigned', [])
    office_lat = data['company']['latitude']
    office_lon = data['company']['longitude']
    office_pos = (office_lat, office_lon)

    routes = []
    assigned_user_ids = set()
    used_driver_ids = set()
    available_users = users.copy()

    mode = "efficiency"
    route_assigner = SimpleRouteAssigner(OPTIMIZATION_CONFIGS[mode])

    for driver in all_drivers:
        driver_id = str(driver['id'])
        if driver_id in used_driver_ids:
            continue

        current_route = {
            'driver_id': driver_id,
            'vehicle_id': str(driver.get('vehicle_id', '')),
            'vehicle_type': int(driver['capacity']),
            'latitude': float(driver['latitude']),
            'longitude': float(driver['longitude']),
            'assigned_users': []
        }

        current_route_drivers_pos = [(u['lat'], u['lng']) for u in current_route['assigned_users']]
        driver_pos = (current_route['latitude'], current_route['longitude'])

        # Filter users that can potentially be assigned to this driver
        potential_users = [u for u in available_users if str(u['id']) not in assigned_user_ids]

        # Sort potential users by distance to office for better assignment
        potential_users.sort(key=lambda u: haversine_distance(float(u['latitude']), float(u['longitude']), office_lat, office_lon))

        assigned_to_this_route = []

        for user in potential_users:
            user_id = str(user['id'])
            if user_id in assigned_user_ids:
                continue

            # Check capacity
            if len(current_route['assigned_users']) >= driver['capacity']:
                break # Route is full

            # --- Apply Assignment Constraints ---

            # 1. Directional Shortest Path / Road Network Constraints
            # Check if user is on route path with road network awareness
            if route_assigner.road_network:
                current_route_user_positions = [(u['lat'], u['lng']) for u in current_route['assigned_users']]

                # Using a simplified check for demonstration
                if not route_assigner.road_network.is_user_on_route_path(
                    driver_pos, current_route_user_positions,
                    (float(user['latitude']), float(user['longitude'])),
                    office_pos, max_detour_ratio=1.15, route_type="straight" # Using the modified parameter
                ):
                    continue # Skip users not aligned with the straight path

            # 2. Hard Coherence Filter at Assignment
            if route_assigner.road_network:
                test_users_for_coherence = [(u['lat'], u['lng']) for u in current_route['assigned_users']] + [(float(user['latitude']), float(user['longitude']))]
                coherence = route_assigner.road_network.get_route_coherence_score(
                    driver_pos, test_users_for_coherence, office_pos
                )
                if coherence < 0.7:
                    continue # Reject users that break corridor coherence

            # 3. Sequential Routing (No Backtracking)
            if current_route['assigned_users']:
                user_office_distance = haversine_distance(
                    float(user['latitude']), float(user['longitude']), office_pos[0], office_pos[1]
                )
                # Find the minimum office distance among currently assigned users
                last_user_office_distance = min([
                    haversine_distance(u['lat'], u['lng'], office_pos[0], office_pos[1]) 
                    for u in current_route['assigned_users']
                ])
                if user_office_distance > last_user_office_distance:
                    continue # Skip users that force backtracking

            # If all checks pass, tentatively add the user
            user_data = {
                'user_id': str(user['id']),
                'lat': float(user['latitude']),
                'lng': float(user['longitude']),
                'office_distance': haversine_distance(float(user['latitude']), float(user['longitude']), office_pos[0], office_pos[1])
            }
            if user.get('first_name'):
                user_data['first_name'] = str(user['first_name'])
            if user.get('email'):
                user_data['email'] = str(user['email'])

            current_route['assigned_users'].append(user_data)
            assigned_to_this_route.append(user) # Keep track of users added in this iteration
            assigned_user_ids.add(user_id)


        # If any users were assigned to this driver, finalize the route
        if current_route['assigned_users']:
            # Fix Pickup Sequencing (for 'straight' routes, use sorted-by-office-distance)
            route_type = "straight" if mode == "efficiency" else "balanced" if mode == "balanced" else "capacity"
            if route_type == "straight":
                # Sort users by their office distance to ensure monotonic progression
                current_route['assigned_users'].sort(key=lambda u: u['office_distance'])
            else:
                # For other modes, use the road network's optimal sequence calculation
                # (This part might require the road_network object to be fully functional)
                if route_assigner.road_network:
                    driver_location = (current_route['latitude'], current_route['longitude'])
                    user_positions = [(u['lat'], u['lng']) for u in current_route['assigned_users']]
                    office = office_pos

                    optimized_sequence_indices = route_assigner.road_network.get_optimal_pickup_sequence(
                        driver_location, user_positions, office
                    )

                    # Reorder the assigned_users list based on the optimized sequence
                    current_route['assigned_users'] = [current_route['assigned_users'][i] for i in optimized_sequence_indices]


            # Update route metrics
            update_route_metrics_improved(current_route, office_lat, office_lon)
            routes.append(current_route)
            used_driver_ids.add(driver_id)
            print(f"✅ Created road-aware route for driver {driver_id} with {len(current_route['assigned_users'])} users")

            # Remove assigned users from the available pool for the next driver
            for assigned_user in assigned_to_this_route:
                if assigned_user in available_users:
                    available_users.remove(assigned_user)

    # FALLBACK PHASE: Try to assign remaining users with relaxed constraints
    remaining_users = [u for u in users if str(u['id']) not in assigned_user_ids]
    if remaining_users:
        print(f"🔄 Fallback assignment for {len(remaining_users)} unassigned users...")
        for user in remaining_users:
            # Find route with available capacity and reasonable distance
            best_route = None
            best_distance = float('inf')

            for route in routes:
                if len(route['assigned_users']) < route['vehicle_type']:
                    user_pos = (float(user['latitude']), float(user['longitude']))
                    driver_pos = (route['latitude'], route['longitude'])
                    distance = haversine_distance(user_pos[0], user_pos[1], driver_pos[0], driver_pos[1])

                    # More relaxed constraints: up to 25km distance for distant users
                    if distance < 25.0 and distance < best_distance:
                        best_route = route

            if best_route:
                user_data = {
                    'user_id': str(user['id']),
                    'lat': float(user['latitude']),
                    'lng': float(user['longitude']),
                    'office_distance': float(user.get('office_distance', 0)) # Ensure office_distance is present
                }
                if user.get('first_name'):
                    user_data['first_name'] = str(user['first_name'])
                if user.get('email'):
                    user_data['email'] = str(user['email'])

                best_route['assigned_users'].append(user_data)
                assigned_user_ids.add(str(user['id']))
                print(f"    ✅ Fallback assigned user {user['id']} to route {best_route['driver_id']} (distance: {best_distance:.1f}km)")

    # EMERGENCY PHASE: Create new routes for truly unassigned users
    still_unassigned = [u for u in users if str(u['id']) not in assigned_user_ids]
    if still_unassigned:
        print(f"🚨 Emergency route creation for {len(still_unassigned)} distant users...")

        # Find unused drivers
        used_driver_ids_set = {r['driver_id'] for r in routes}
        available_emergency_drivers = [d for d in all_drivers if str(d['id']) not in used_driver_ids_set]

        for user in still_unassigned:
            if not available_emergency_drivers:
                break

            # Use the first available driver for emergency routes
            emergency_driver = available_emergency_drivers.pop(0)

            emergency_route = {
                'driver_id': str(emergency_driver['id']),
                'vehicle_id': str(emergency_driver.get('vehicle_id', '')),
                'vehicle_type': int(emergency_driver['capacity']),
                'latitude': float(emergency_driver['latitude']),
                'longitude': float(emergency_driver['longitude']),
                'assigned_users': []
            }

            user_data = {
                'user_id': str(user['id']),
                'lat': float(user['latitude']),
                'lng': float(user['longitude']),
                'office_distance': float(user.get('office_distance', 0)) # Ensure office_distance is present
            }

            if user.get('first_name'):
                user_data['first_name'] = str(user['first_name'])
            if user.get('email'):
                user_data['email'] = str(user['email'])

            emergency_route['assigned_users'].append(user_data)
            assigned_user_ids.add(str(user['id']))

            # Update route metrics
            update_route_metrics_improved(emergency_route, office_lat, office_lon)
            routes.append(emergency_route)
            used_driver_ids.add(str(emergency_driver['id']))

            distance = haversine_distance(float(user['latitude']), float(user['longitude']),
                                        float(emergency_driver['latitude']), float(emergency_driver['longitude']))
            print(f"    🚨 Emergency route created for user {user['id']} with driver {emergency_driver['id']} (distance: {distance:.1f}km)")

    # Handle unassigned users and drivers
    unassigned_users = []
    for user in users:
        if str(user['id']) not in assigned_user_ids:
            user_data = {
                'user_id': str(user['id']),
                'lat': float(user['latitude']),
                'lng': float(user['longitude']),
                'office_distance': float(user.get('office_distance', 0))
            }
            if user.get('first_name'):
                user_data['first_name'] = str(user['first_name'])
            if user.get('email'):
                user_data['email'] = str(user['email'])
            unassigned_users.append(user_data)

    unassigned_drivers = []
    for driver in all_drivers: # Direct iteration over list
        if str(driver['id']) not in used_driver_ids:
            driver_data = {
                'driver_id': str(driver['id']),
                'capacity': int(driver['capacity']),
                'vehicle_id': str(driver.get('vehicle_id', '')),
                'latitude': float(driver['latitude']),
                'longitude': float(driver['longitude'])
            }
            unassigned_drivers.append(driver_data)

    total_assigned = sum(len(r['assigned_users']) for r in routes)
    total_capacity = sum(r['vehicle_type'] for r in routes)
    utilization = (total_assigned / total_capacity * 100) if total_capacity > 0 else 0

    print(f"🎯 Road-aware assignment complete: {len(routes)} routes, {total_assigned} users assigned")
    print(f"📊 Overall utilization: {total_assigned}/{total_capacity} seats ({utilization:.1f}%)")
    print(f"👥 Unassigned: {len(unassigned_users)} users, {len(unassigned_drivers)} drivers")

    return {
        "status": "true",
        "data": routes,
        "unassignedUsers": unassigned_users,
        "unassignedDrivers": unassigned_drivers,
        "message": f"Road-aware assignment completed with {len(routes)} routes, {utilization:.1f}% utilization"
    }


def run_road_aware_assignment(source_id: str, parameter: int = 1, string_param: str = ""):
    """Main entry point for road-aware assignment"""
    start_time = time.time()

    print(f"🗺️ Starting road-aware assignment for source_id: {source_id}")
    print(f"📋 Parameter: {parameter}, String parameter: {string_param}")

    try:
        # Load data from API - import the real data loading function
        from assignment import load_env_and_fetch_data

        data = load_env_and_fetch_data(source_id, parameter, string_param)

        # Perform road-aware assignment
        result = assign_routes_road_aware(data)

        # Add execution metadata
        result["execution_time"] = time.time() - start_time
        result["optimization_mode"] = "road_aware"
        result["parameter"] = parameter
        result["string_param"] = string_param
        result["road_network_enabled"] = ROAD_NETWORK is not None

        return result

    except Exception as e:
        print(f"ERROR: Road-aware assignment execution failed: {e}")
        return {
            "status": "false",
            "execution_time": time.time() - start_time,
            "message": f"Execution failed: {str(e)}",
            "data": [],
            "unassignedUsers": [],
            "unassignedDrivers": []
        }


def merge_underutilized_routes(routes, all_drivers, office_pos, road_network):
    """Merge underutilized routes (especially 1-2 user routes) into nearby routes.
    Relaxed thresholds and corridor-preferring logic to avoid many small parallel routes.
    """
    print("  🔗 Phase 1: Merging underutilized routes (relaxed thresholds).")

    merged_routes = []
    retired_driver_ids = set()
    merges_made = 0

    # Sort routes by utilization (lowest first for merging)
    routes_by_util = sorted(routes, key=lambda r: len(r.get('assigned_users', [])) / max(1, r.get('vehicle_type', 1)))

    # Parameters (tweak these if you still see over-splitting)
    MAX_CENTER_DIST_KM = 12.0         # relaxed from 8.0
    MAX_DETOUR_PER_USER_KM = 6.0      # allow ~6km detour per merged user (total scaled)
    PATH_CHECK_MAX_DETOUR = 1.4       # allow slightly looser detour ratio for is_user_on_route_path

    for i, route_a in enumerate(routes_by_util):
        if route_a['driver_id'] in retired_driver_ids:
            continue

        # Target underutilized routes: <=2 users or <50% capacity (same logic as before)
        utilization_a = len(route_a.get('assigned_users', [])) / max(1, route_a.get('vehicle_type', 1))
        if len(route_a.get('assigned_users', [])) > 2 and utilization_a > 0.5:
            merged_routes.append(route_a)
            continue

        # Look for a suitable route to merge into
        best_merge_route = None
        best_merge_score = float('inf')

        for j, route_b in enumerate(routes_by_util):
            if i == j or route_b['driver_id'] in retired_driver_ids:
                continue

            # Check capacity
            available_capacity = route_b.get('vehicle_type', 1) - len(route_b.get('assigned_users', []))
            if available_capacity < len(route_a.get('assigned_users', [])):
                # Not enough capacity
                # debug:
                # print(f"Skip {route_b['driver_id']} — not enough capacity")
                continue

            # Distance between centers
            center_a = calculate_route_center(route_a)
            center_b = calculate_route_center(route_b)
            center_distance = haversine_distance(center_a[0], center_a[1], center_b[0], center_b[1])
            if center_distance > MAX_CENTER_DIST_KM:
                # too far apart
                # debug:
                # print(f"Skip {route_b['driver_id']} — center_distance {center_distance:.2f}km > {MAX_CENTER_DIST_KM}")
                continue

            # Test merge feasibility
            merge_feasible = True
            total_detour = 0.0
            for user in route_a.get('assigned_users', []):
                user_pos = (user['lat'], user['lng'])
                driver_b_pos = (route_b['latitude'], route_b['longitude'])

                if road_network:
                    # Looser path check: allow PATH_CHECK_MAX_DETOUR ratio
                    existing_users_b = [(u['lat'], u['lng']) for u in route_b.get('assigned_users', [])]
                    try:
                        ok = road_network.is_user_on_route_path(
                            driver_b_pos, existing_users_b, user_pos, office_pos,
                            max_detour_ratio=PATH_CHECK_MAX_DETOUR, route_type="balanced"
                        )
                    except Exception as e:
                        ok = False

                    # Fallback: if path check fails but users are geographically very close, allow merge
                    if not ok:
                        user_to_user_distance = min([
                            haversine_distance(user_pos[0], user_pos[1], existing_user[0], existing_user[1])
                            for existing_user in existing_users_b
                        ]) if existing_users_b else float('inf')

                        # If users are within 2km of each other, override path check
                        if user_to_user_distance <= 2.0:
                            print(f"    🔄 Override: user {user['user_id']} close to existing users ({user_to_user_distance:.2f}km), allowing merge")
                            ok = True
                        else:
                            # Debug print for merge failures
                            print(f"    ✖ user {user['user_id']} not on route path for candidate merge into {route_b['driver_id']}")
                            # Log distances for debugging
                            d1 = road_network.get_road_distance(driver_b_pos[0], driver_b_pos[1], user_pos[0], user_pos[1])
                            d2 = road_network.get_road_distance(user_pos[0], user_pos[1], office_pos[0], office_pos[1])
                            direct = road_network.get_road_distance(driver_b_pos[0], driver_b_pos[1], office_pos[0], office_pos[1])
                            print(f"      detour if added = {d1 + d2:.2f} km vs direct {direct:.2f} km")
                            merge_feasible = False
                            break

                    # Compute detour in road-distance terms and accumulate
                    direct_dist = road_network.get_road_distance(driver_b_pos[0], driver_b_pos[1], office_pos[0], office_pos[1])
                    via_user_dist = (road_network.get_road_distance(driver_b_pos[0], driver_b_pos[1], user_pos[0], user_pos[1]) +
                                     road_network.get_road_distance(user_pos[0], user_pos[1], office_pos[0], office_pos[1]))
                    detour = max(0.0, via_user_dist - direct_dist)
                    total_detour += detour
                else:
                    # fallback: straight-line check but scaled a bit
                    distance = haversine_distance(driver_b_pos[0], driver_b_pos[1], user_pos[0], user_pos[1])
                    if distance > (MAX_DETOUR_PER_USER_KM * 1.5):  # looser fallback
                        merge_feasible = False
                        break
                    total_detour += distance

            # Acceptable total detour threshold scales with number of users being merged
            allowed_total_detour = max(MAX_DETOUR_PER_USER_KM, MAX_DETOUR_PER_USER_KM * len(route_a.get('assigned_users', [])))

            # Debug info (very useful to keep while testing)
            print(f"    → Trying merge {route_a['driver_id']} -> {route_b['driver_id']}: center_dist={center_distance:.2f} km, total_detour={total_detour:.2f} km, allowed={allowed_total_detour:.2f} km, path_ok={merge_feasible}")

            if not merge_feasible or total_detour > allowed_total_detour:
                # skip candidate
                continue

            # cost-based score (prefer close centers, low detour, and prefer same corridor)
            merge_score = center_distance + total_detour + (len(route_b.get('assigned_users', [])) * 0.5)

            # Prefer merging routes in same corridor if available
            if route_a.get('corridor_direction') and route_b.get('corridor_direction') and route_a.get('corridor_direction') == route_b.get('corridor_direction'):
                merge_score -= 2.0  # bonus to prefer same-corridor merges

            if merge_score < best_merge_score:
                best_merge_score = merge_score
                best_merge_route = route_b

        # perform the merge
        if best_merge_route:
            for user in route_a.get('assigned_users', []):
                best_merge_route.setdefault('assigned_users', []).append(user)

            # Re-optimize the merged route metrics
            update_route_metrics_improved(best_merge_route, office_pos[0], office_pos[1])
            retired_driver_ids.add(route_a['driver_id'])
            merges_made += 1
            print(f"    ✅ Merged route {route_a['driver_id']} ({len(route_a.get('assigned_users', []))} users) into route {best_merge_route['driver_id']}")
        else:
            merged_routes.append(route_a)

    # Add non-retired routes (make sure we include all remaining routes)
    for route in routes_by_util:
        if route['driver_id'] not in retired_driver_ids and route not in merged_routes:
            merged_routes.append(route)

    print(f"    🔗 Completed {merges_made} route merges, {len(routes)} → {len(merged_routes)} routes")

    # Build used_driver_ids from final merged routes (strings)
    used_driver_ids = {str(r['driver_id']) for r in merged_routes}

    return merged_routes, used_driver_ids



def split_high_detour_routes(routes, all_drivers, office_pos, road_network):
    """Split routes with high detour ratios into more efficient sub-routes"""
    print("  ✂️ Phase 2: Splitting high-detour routes...")

    optimized_routes = []
    used_driver_ids = {r['driver_id'] for r in routes}
    available_drivers = [d for d in all_drivers if str(d['id']) not in used_driver_ids]
    splits_made = 0

    for route in routes:
        # Skip routes with ≤2 users (can't split effectively)
        if len(route['assigned_users']) <= 2:
            optimized_routes.append(route)
            continue

        # Calculate detour ratio
        detour_ratio = calculate_route_detour_ratio(route, office_pos, road_network)

        if detour_ratio > 1.4 and len(route['assigned_users']) >= 3 and available_drivers:
            print(f"    🔍 High detour route {route['driver_id']}: {detour_ratio:.2f} ratio, attempting split...")

            # Cluster users into 2 groups using road network awareness
            split_routes = split_route_into_clusters(route, available_drivers, office_pos, road_network)

            if len(split_routes) > 1:
                # Successfully split
                optimized_routes.extend(split_routes)

                # Update available drivers
                for split_route in split_routes:
                    if split_route['driver_id'] != route['driver_id']:  # New driver used
                        available_drivers = [d for d in available_drivers if str(d['id']) != split_route['driver_id']]
                        used_driver_ids.add(split_route['driver_id'])

                splits_made += 1
                print(f"    ✅ Split route {route['driver_id']} into {len(split_routes)} routes")
            else:
                optimized_routes.append(route)
        else:
            optimized_routes.append(route)

    print(f"    ✂️ Completed {splits_made} route splits")
    return optimized_routes, used_driver_ids


def local_route_optimization(routes, unassigned_users, office_pos, road_network):
    """Local optimization: improve each route individually"""
    print("  🔧 Phase 3: Local route optimization...")

    improvements_made = 0

    for route in routes:
        original_user_count = len(route['assigned_users'])

        # Try adding nearby unassigned users
        available_capacity = route['vehicle_type'] - len(route['assigned_users'])
        if available_capacity > 0 and unassigned_users:
            driver_pos = (route['latitude'], route['longitude'])

            for user in unassigned_users[:]:  # Copy list to avoid modification during iteration
                if available_capacity <= 0:
                    break

                user_pos = (user['lat'], user['lng'])

                # Check if user fits well in this route
                if road_network:
                    existing_users = [(u['lat'], u['lng']) for u in route['assigned_users']]
                    if road_network.is_user_on_route_path(
                        driver_pos, existing_users, user_pos, office_pos,
                        max_detour_ratio=1.2, route_type="straight"
                    ):
                        # Test route quality after adding user
                        test_route = route.copy()
                        test_route['assigned_users'] = route['assigned_users'] + [user]

                        test_detour = calculate_route_detour_ratio(test_route, office_pos, road_network)
                        if test_detour < 1.3:  # Acceptable detour
                            route['assigned_users'].append(user)
                            unassigned_users.remove(user)
                            available_capacity -= 1
                            improvements_made += 1
                            print(f"    ✅ Added unassigned user {user['user_id']} to route {route['driver_id']}")
                else:
                    # Fallback: simple distance check
                    distance = haversine_distance(driver_pos[0], driver_pos[1], user_pos[0], user_pos[1])
                    if distance < 3.0:  # Within 3km
                        route['assigned_users'].append(user)
                        unassigned_users.remove(user)
                        available_capacity -= 1
                        improvements_made += 1

        # Re-optimize pickup sequence
        if road_network and len(route['assigned_users']) > 1:
            driver_pos = (route['latitude'], route['longitude'])
            user_positions = [(u['lat'], u['lng']) for u in route['assigned_users']]
            optimal_sequence = road_network.get_optimal_pickup_sequence(driver_pos, user_positions, office_pos)

            if optimal_sequence and len(optimal_sequence) == len(route['assigned_users']):
                reordered_users = [route['assigned_users'][i] for i in optimal_sequence]
                route['assigned_users'] = reordered_users

        # Update route metrics
        update_route_metrics_improved(route, office_pos[0], office_pos[1])

        final_user_count = len(route['assigned_users'])
        if final_user_count > original_user_count:
            improvements_made += 1

    print(f"    🔧 Local optimization: {improvements_made} improvements made")
    return routes


def global_optimization_pass(routes, office_pos, road_network):
    """Conservative global optimization: only make swaps that significantly improve route quality"""
    print("  🌍 Phase 4: Conservative global optimization pass...")

    max_iterations = 3  # Reduced iterations
    swaps_made = 0
    MIN_IMPROVEMENT_THRESHOLD = 1.0  # Require larger improvement

    for iteration in range(max_iterations):
        iteration_swaps = 0

        # Try swapping users between every pair of routes
        for i, route_a in enumerate(routes):
            for j, route_b in enumerate(routes):
                if i >= j or not route_a['assigned_users'] or not route_b['assigned_users']:
                    continue

                # Skip if either route already has good utilization and quality
                util_a = len(route_a['assigned_users']) / route_a['vehicle_type']
                util_b = len(route_b['assigned_users']) / route_b['vehicle_type']
                
                # Don't mess with well-utilized routes unless they have quality issues
                if util_a > 0.8 and route_a.get('turning_score', 0) < 35:
                    continue
                if util_b > 0.8 and route_b.get('turning_score', 0) < 35:
                    continue

                # Try moving one user from route_a to route_b
                for user_idx, user in enumerate(route_a['assigned_users']):
                    # Check if route_b has capacity
                    if len(route_b['assigned_users']) >= route_b['vehicle_type']:
                        continue

                    # Calculate current metrics for both routes
                    current_distance_a = calculate_route_total_distance(route_a, office_pos, road_network)
                    current_distance_b = calculate_route_total_distance(route_b, office_pos, road_network)
                    current_turning_a = route_a.get('turning_score', 0)
                    current_turning_b = route_b.get('turning_score', 0)

                    # Test the swap
                    test_route_a = route_a.copy()
                    test_route_b = route_b.copy()
                    test_route_a['assigned_users'] = [u for k, u in enumerate(route_a['assigned_users']) if k != user_idx]
                    test_route_b['assigned_users'] = route_b['assigned_users'] + [user]

                    # Skip if route_a becomes empty
                    if not test_route_a['assigned_users']:
                        continue

                    # Strict path checking with road network
                    if road_network:
                        driver_b_pos = (route_b['latitude'], route_b['longitude'])
                        existing_users_b = [(u['lat'], u['lng']) for u in route_b['assigned_users']]
                        user_pos = (user['lat'], user['lng'])

                        # More strict path checking
                        if not road_network.is_user_on_route_path(
                            driver_b_pos, existing_users_b, user_pos, office_pos,
                            max_detour_ratio=1.15, route_type="straight"  # Stricter
                        ):
                            continue
                        
                        # Check coherence of resulting route
                        test_coherence_b = road_network.get_route_coherence_score(
                            driver_b_pos, [(u['lat'], u['lng']) for u in test_route_b['assigned_users']], office_pos
                        )
                        if test_coherence_b < 0.7:  # Require good coherence
                            continue

                    # Calculate new metrics
                    new_distance_a = calculate_route_total_distance(test_route_a, office_pos, road_network)
                    new_distance_b = calculate_route_total_distance(test_route_b, office_pos, road_network)
                    
                    # Calculate turning scores for test routes
                    new_turning_a = calculate_route_turning_score_improved(
                        test_route_a['assigned_users'],
                        (test_route_a['latitude'], test_route_a['longitude']),
                        office_pos
                    ) if test_route_a['assigned_users'] else 0
                    
                    new_turning_b = calculate_route_turning_score_improved(
                        test_route_b['assigned_users'],
                        (test_route_b['latitude'], test_route_b['longitude']),
                        office_pos
                    )

                    # Calculate comprehensive improvement
                    distance_improvement = (current_distance_a + current_distance_b) - (new_distance_a + new_distance_b)
                    turning_improvement = (current_turning_a + current_turning_b) - (new_turning_a + new_turning_b)
                    
                    # Require both distance AND turning improvement, or significant improvement in one
                    total_improvement = distance_improvement + (turning_improvement * 0.1)  # Weight turning at ~10km per degree
                    
                    # Stricter acceptance criteria
                    quality_acceptable = (new_turning_a <= 40 and new_turning_b <= 40)  # Both routes must have good quality
                    significant_improvement = total_improvement > MIN_IMPROVEMENT_THRESHOLD
                    
                    # Additional check: don't create routes with too many users going in different directions
                    if road_network and len(test_route_b['assigned_users']) > 3:
                        # Check if the route becomes too scattered
                        max_bearing_spread = 0
                        bearings = []
                        for u in test_route_b['assigned_users']:
                            bearing = road_network._calculate_bearing(office_pos[0], office_pos[1], u['lat'], u['lng'])
                            bearings.append(bearing)
                        
                        if len(bearings) > 1:
                            bearings.sort()
                            for k in range(len(bearings)):
                                spread = abs(bearings[(k + 1) % len(bearings)] - bearings[k])
                                if spread > 180:
                                    spread = 360 - spread
                                max_bearing_spread = max(max_bearing_spread, spread)
                        
                        if max_bearing_spread > 60:  # Don't allow too much directional spread
                            continue

                    if quality_acceptable and significant_improvement:
                        route_a['assigned_users'] = test_route_a['assigned_users']
                        route_b['assigned_users'] = test_route_b['assigned_users']

                        # Update route metrics
                        update_route_metrics_improved(route_a, office_pos[0], office_pos[1])
                        update_route_metrics_improved(route_b, office_pos[0], office_pos[1])

                        iteration_swaps += 1
                        swaps_made += 1
                        print(f"    🔄 Conservative swap: user {user['user_id']} from route {route_a['driver_id']} to {route_b['driver_id']} (improvement: {total_improvement:.1f})")
                        break  # Only one swap per route pair per iteration

        if iteration_swaps == 0:
            break  # No improvements found, exit early

    print(f"    🌍 Conservative global optimization: {swaps_made} quality-preserving swaps made")
    return routes


def calculate_route_detour_ratio(route, office_pos, road_network):
    """Calculate the detour ratio for a route"""
    if not route['assigned_users']:
        return 1.0

    driver_pos = (route['latitude'], route['longitude'])

    # Calculate actual route distance
    actual_distance = calculate_route_total_distance(route, office_pos, road_network)

    # Calculate direct distance (driver to office)
    if road_network:
        direct_distance = road_network.get_road_distance(driver_pos[0], driver_pos[1], office_pos[0], office_pos[1])
    else:
        direct_distance = haversine_distance(driver_pos[0], driver_pos[1], office_pos[0], office_pos[1])

    return actual_distance / max(direct_distance, 0.1)


def calculate_route_total_distance(route, office_pos, road_network):
    """Calculate total distance for a route"""
    if not route['assigned_users']:
        return 0.0

    total_distance = 0.0
    current_pos = (route['latitude'], route['longitude'])

    # Distance through all users
    for user in route['assigned_users']:
        user_pos = (user['lat'], user['lng'])
        if road_network:
            distance = road_network.get_road_distance(current_pos[0], current_pos[1], user_pos[0], user_pos[1])
        else:
            distance = haversine_distance(current_pos[0], current_pos[1], user_pos[0], user_pos[1])
        total_distance += distance
        current_pos = user_pos

    # Distance from last user to office
    if road_network:
        final_distance = road_network.get_road_distance(current_pos[0], current_pos[1], office_pos[0], office_pos[1])
    else:
        final_distance = haversine_distance(current_pos[0], current_pos[1], office_pos[0], office_pos[1])
    total_distance += final_distance

    return total_distance


def post_assignment_geographic_consolidation(routes, all_drivers, office_pos, road_network):
    """
    Post-assignment phase: Aggressively consolidate users who are geographically close
    but ended up in different routes due to timing of assignment
    """
    print("  🌍 Phase 3: Post-assignment geographic consolidation...")

    consolidations_made = 0
    max_iterations = 3
    GEOGRAPHIC_THRESHOLD_KM = 2.0  # Users within 2km should be in same route
    MAX_ROUTE_DETOUR = 1.3  # Allow up to 30% detour for consolidation

    for iteration in range(max_iterations):
        iteration_consolidations = 0
        
        # Find all single-user routes and underutilized routes
        single_user_routes = [r for r in routes if len(r['assigned_users']) == 1]
        underutilized_routes = [r for r in routes if len(r['assigned_users']) <= 2 and r['vehicle_type'] >= 5]
        
        print(f"    📍 Iteration {iteration + 1}: {len(single_user_routes)} single-user routes, {len(underutilized_routes)} underutilized routes")
        
        # For each single-user route, try to find a better home
        routes_to_remove = []
        
        for single_route in single_user_routes:
            single_user = single_route['assigned_users'][0]
            single_pos = (single_user['lat'], single_user['lng'])
            best_target_route = None
            best_consolidation_score = float('inf')
            
            # Look for target routes that could absorb this user
            for target_route in routes:
                if target_route['driver_id'] == single_route['driver_id']:
                    continue
                
                # Check capacity
                if len(target_route['assigned_users']) >= target_route['vehicle_type']:
                    continue
                
                # Calculate geographic proximity to target route users
                min_user_distance = float('inf')
                for target_user in target_route['assigned_users']:
                    target_pos = (target_user['lat'], target_user['lng'])
                    distance = haversine_distance(single_pos[0], single_pos[1], target_pos[0], target_pos[1])
                    min_user_distance = min(min_user_distance, distance)
                
                # Only consider if users are geographically close
                if min_user_distance > GEOGRAPHIC_THRESHOLD_KM:
                    continue
                
                # Test route quality after adding user
                test_route = target_route.copy()
                test_route['assigned_users'] = target_route['assigned_users'] + [single_user]
                
                # Calculate detour ratio
                original_distance = calculate_route_total_distance(target_route, office_pos, road_network)
                new_distance = calculate_route_total_distance(test_route, office_pos, road_network)
                
                # Check if addition creates reasonable route
                if road_network:
                    driver_pos = (target_route['latitude'], target_route['longitude'])
                    existing_users = [(u['lat'], u['lng']) for u in target_route['assigned_users']]
                    
                    # Use more lenient path check for post-assignment consolidation
                    on_path = road_network.is_user_on_route_path(
                        driver_pos, existing_users, single_pos, office_pos,
                        max_detour_ratio=MAX_ROUTE_DETOUR, route_type="balanced"
                    )
                    
                    if not on_path:
                        continue
                
                # Calculate consolidation score (lower is better)
                route_center = calculate_route_center(target_route)
                center_distance = haversine_distance(route_center[0], route_center[1], single_pos[0], single_pos[1])
                utilization_bonus = len(target_route['assigned_users']) / target_route['vehicle_type']
                
                consolidation_score = (center_distance + min_user_distance - utilization_bonus * 2.0)
                
                if consolidation_score < best_consolidation_score:
                    best_consolidation_score = consolidation_score
                    best_target_route = target_route
            
            # Perform consolidation if good target found
            if best_target_route:
                print(f"    🔄 Consolidating user {single_user['user_id']} from driver {single_route['driver_id']} to driver {best_target_route['driver_id']} (distance: {best_consolidation_score:.2f}km)")
                
                # Move user to target route
                best_target_route['assigned_users'].append(single_user)
                
                # Re-optimize target route
                update_route_metrics_improved(best_target_route, office_pos[0], office_pos[1])
                
                # Mark single route for removal
                routes_to_remove.append(single_route)
                iteration_consolidations += 1
                consolidations_made += 1
        
        # Remove empty routes
        routes = [r for r in routes if r not in routes_to_remove]
        
        # Phase 2: Merge remaining underutilized routes with each other
        remaining_underutilized = [r for r in routes if len(r['assigned_users']) <= 2 and r['vehicle_type'] >= 5]
        
        while len(remaining_underutilized) >= 2:
            route_a = remaining_underutilized.pop(0)
            best_merge_route = None
            best_merge_distance = float('inf')
            
            for route_b in remaining_underutilized:
                # Check if they can be merged
                if len(route_a['assigned_users']) + len(route_b['assigned_users']) <= max(route_a['vehicle_type'], route_b['vehicle_type']):
                    # Calculate distance between route centers
                    center_a = calculate_route_center(route_a)
                    center_b = calculate_route_center(route_b)
                    center_distance = haversine_distance(center_a[0], center_a[1], center_b[0], center_b[1])
                    
                    if center_distance <= 5.0 and center_distance < best_merge_distance:  # Within 5km
                        best_merge_distance = center_distance
                        best_merge_route = route_b
            
            if best_merge_route:
                print(f"    🔗 Merging underutilized routes {route_a['driver_id']} and {best_merge_route['driver_id']} (distance: {best_merge_distance:.2f}km)")
                
                # Use the route with higher capacity
                if route_a['vehicle_type'] >= best_merge_route['vehicle_type']:
                    target_route = route_a
                    source_route = best_merge_route
                else:
                    target_route = best_merge_route
                    source_route = route_a
                
                # Move all users from source to target
                for user in source_route['assigned_users']:
                    target_route['assigned_users'].append(user)
                
                # Update route metrics
                update_route_metrics_improved(target_route, office_pos[0], office_pos[1])
                
                # Remove source route
                routes = [r for r in routes if r['driver_id'] != source_route['driver_id']]
                remaining_underutilized = [r for r in remaining_underutilized if r['driver_id'] != source_route['driver_id']]
                
                iteration_consolidations += 1
                consolidations_made += 1
                
                break  # Restart the merging process
        
        if iteration_consolidations == 0:
            break  # No more consolidations possible
    
    print(f"    🌍 Geographic consolidation complete: {consolidations_made} consolidations made")
    
    # Update used_driver_ids
    used_driver_ids = {r['driver_id'] for r in routes}
    
    return routes, used_driver_ids


def split_route_into_clusters(route, available_drivers, office_pos, road_network):
    """Split a route into 2 clusters using road network awareness"""
    if len(route['assigned_users']) < 3 or not available_drivers:
        return [route]

    users = route['assigned_users']
    driver_pos = (route['latitude'], route['longitude'])

    # Use DBSCAN clustering based on road distances
    if road_network:
        # Create distance matrix using road network
        n_users = len(users)
        distance_matrix = []

        for i, user_a in enumerate(users):
            user_a_pos = (user_a['lat'], user_a['lng'])
            distances = []
            for j, user_b in enumerate(users):
                if i == j:
                    distances.append(0.0)
                else:
                    user_b_pos = (user_b['lat'], user_b['lng'])
                    dist = road_network.get_road_distance(user_a_pos[0], user_a_pos[1], user_b_pos[0], user_b_pos[1])
                    distances.append(dist)
            distance_matrix.append(distances)

        # Simple clustering: split by distance from driver
        user_distances = []
        for user in users:
            user_pos = (user['lat'], user['lng'])
            dist = road_network.get_road_distance(driver_pos[0], driver_pos[1], user_pos[0], user_pos[1])
            user_distances.append((user, dist))

        # Sort by distance and split roughly in half
        user_distances.sort(key=lambda x: x[1])
        mid_point = len(user_distances) // 2

        cluster_1 = [item[0] for item in user_distances[:mid_point]]
        cluster_2 = [item[0] for item in user_distances[mid_point:]]
    else:
        # Fallback: simple geographic clustering
        mid_point = len(users) // 2
        cluster_1 = users[:mid_point]
        cluster_2 = users[mid_point:]

    # Create route for cluster 1 (keep original driver)
    route_1 = route.copy()
    route_1['assigned_users'] = cluster_1

    # Create route for cluster 2 (assign new driver)
    if available_drivers and cluster_2:
        # Find best driver for cluster 2
        cluster_2_center = (
            sum(u['lat'] for u in cluster_2) / len(cluster_2),
            sum(u['lng'] for u in cluster_2) / len(cluster_2)
        )

        best_driver = None
        best_distance = float('inf')

        for driver in available_drivers:
            if driver['capacity'] >= len(cluster_2):
                distance = haversine_distance(
                    driver['latitude'], driver['longitude'],
                    cluster_2_center[0], cluster_2_center[1]
                )
                if distance < best_distance:
                    best_distance = distance
                    best_driver = driver

        if best_driver:
            route_2 = {
                'driver_id': str(best_driver['id']),
                'vehicle_id': str(best_driver.get('vehicle_id', '')),
                'vehicle_type': int(best_driver['capacity']),
                'latitude': float(best_driver['latitude']),
                'longitude': float(best_driver['longitude']),
                'assigned_users': cluster_2
            }

            # Update route metrics for both routes
            update_route_metrics_improved(route_1, office_pos[0], office_pos[1])
            update_route_metrics_improved(route_2, office_pos[0], office_pos[1])

            return [route_1, route_2]

    # If splitting failed, return original route
    return [route]


def main():
    """Test the road-aware assignment with sample data"""
    # For testing purposes - you can replace this with actual API data
    sample_data = {
        "users": [
            {"id": "1", "latitude": 30.6840, "longitude": 76.7300, "first_name": "User1", "email": "user1@test.com"},
            {"id": "2", "latitude": 30.6820, "longitude": 76.7280, "first_name": "User2", "email": "user2@test.com"},
            {"id": "3", "latitude": 30.6860, "longitude": 76.7320, "first_name": "User3", "email": "user3@test.com"},
            {"id": "4", "latitude": 30.6780, "longitude": 76.7200, "first_name": "User4", "email": "user4@test.com"}, # Further away
            {"id": "5", "latitude": 30.6900, "longitude": 76.7400, "first_name": "User5", "email": "user5@test.com"}  # Another direction
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
            "longitude": 76.7260711
        }
    }

    # Initialize ROAD_NETWORK and OPTIMIZATION_CONFIGS if they are not globally defined
    global ROAD_NETWORK, OPTIMIZATION_CONFIGS
    try:
        if ROAD_NETWORK is None:
            ROAD_NETWORK = RoadNetwork('tricity_main_roads.graphml')
            print("✅ Road network loaded for testing")
    except Exception as e:
        print(f"⚠️ Could not load road network for testing: {e}")
        ROAD_NETWORK = None

    OPTIMIZATION_CONFIGS = {
        "efficiency": {"max_detour_ratio": 1.1},
        "balanced": {"max_detour_ratio": 1.3},
        "capacity": {"max_detour_ratio": 1.5}
    }


    result = assign_routes_road_aware(sample_data)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()