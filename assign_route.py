import json
import time
import math
import pandas as pd
from road_network import RoadNetwork

# Global variables that will be set during assignment
ROAD_NETWORK = None
OPTIMIZATION_CONFIGS = {
    "efficiency": {"max_detour_ratio": 1.1},
    "balanced": {"max_detour_ratio": 1.3},
    "capacity": {"max_detour_ratio": 1.5}
}

class RoadCorridorAnalyzer:
    """Analyzes road network to identify natural travel corridors from office"""

    def __init__(self, road_network, office_pos):
        self.road_network = road_network
        self.office_pos = office_pos
        self.office_node = road_network.find_nearest_road_node(office_pos[0], office_pos[1])
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

        while queue and len(corridor_nodes) < 50:  # Limit search
            current_node = queue.pop(0)
            if current_node in visited:
                continue
            visited.add(current_node)

            # Check if this node is in the target direction
            if current_node != self.office_node:
                current_pos = self.road_network.node_positions.get(current_node)
                if current_pos:
                    bearing = self.road_network._calculate_bearing(
                        self.office_pos[0], self.office_pos[1],
                        current_pos[0], current_pos[1]
                    )
                    bearing_diff = abs(self._normalize_bearing_difference(bearing - target_bearing))

                    if bearing_diff <= tolerance:
                        corridor_nodes.append(current_node)

            # Add neighbors to queue
            if hasattr(self.road_network.graph, 'neighbors'):
                for neighbor in self.road_network.graph.neighbors(current_node):
                    if neighbor not in visited:
                        queue.append(neighbor)

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

        return corridor_routes


class SimpleRouteAssigner:
    def __init__(self, config):
        self.config = config
        self.road_network = None

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
                route_users_pos = [(u['lat'], u['lng']) for u in route_users]

                # Get optimization parameters from mode config
                config = OPTIMIZATION_CONFIGS.get(mode, OPTIMIZATION_CONFIGS["balanced"])

                # Determine route type for road network functions
                route_type = "straight" if mode == "efficiency" else "balanced" if mode == "balanced" else "capacity"

                # Check if user is on route path with road network awareness
                if self.road_network and not self.road_network.is_user_on_route_path(
                    driver_pos, [(u['lat'], u['lng']) for u in route_users],
                    (float(user['latitude']), float(user['longitude'])),
                    office_pos, max_detour_ratio=1.15, route_type="straight"
                ):
                    continue  # Skip users not aligned with the straight path

                # Hard coherence filter - test adding this user
                if self.road_network:
                    test_users = [(u['lat'], u['lng']) for u in route_users] + [(float(user['latitude']), float(user['longitude']))]
                    coherence = self.road_network.get_route_coherence_score(
                        driver_pos, test_users, office_pos
                    )
                    if coherence < 0.7:
                        continue  # Reject users that break corridor coherence

                # Sequential routing - no backtracking toward office
                if route_users:
                    user_office_distance = haversine_distance(
                        float(user['latitude']), float(user['longitude']), office_pos[0], office_pos[1]
                    )
                    last_user_office_distance = min([
                        haversine_distance(u['lat'], u['lng'], office_pos[0], office_pos[1]) 
                        for u in route_users
                    ])
                    if user_office_distance > last_user_office_distance:
                        continue  # Skip users that force backtracking

                # Check capacity - get capacity from route
                if len(route_users) >= route['vehicle_type']:
                    continue

                route_users.append(user)
                assigned_user_ids.add(str(user['id']))

                # Re-optimize sequence for the updated route
                # Calculate office distances for all users if not present
                for u in route_users:
                    if 'office_distance' not in u or u['office_distance'] == 0:
                        u['office_distance'] = haversine_distance(
                            u['lat'], u['lng'], office_pos[0], office_pos[1]
                        )

                # For straight routes, use sequential ordering by office distance (no backtracking)
                # For other modes, use the road network optimizer
                if route_type == "straight":
                    # Sort by office distance to ensure monotonic progression toward office
                    route_users.sort(key=lambda u: u['office_distance'])
                    optimized_sequence = list(range(len(route_users)))
                else:
                    # Use road network optimization for balanced/capacity modes
                    optimized_sequence = self.road_network.get_optimal_pickup_sequence(
                        driver_pos,
                        [(float(user['latitude']), float(user['longitude'])) for user in route_users],
                        office_pos
                    ) if self.road_network else list(range(len(route_users)))

                current_route_score = self.calculate_route_score(route, driver_pos, office_pos, optimized_sequence)

                if current_route_score > best_route_score:
                    best_route_score = current_route_score
                    best_route_for_user = route

                # Backtrack: remove user if it was added for score calculation
                route_users.remove(user)
                assigned_user_ids.remove(str(user['id']))


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


        # Try to assign remaining users to new routes if capacity allows
        # (This part needs to be integrated with the main assignment logic)

        return routes, assigned_user_ids


    def calculate_route_score(self, route, driver_pos, office_pos, optimized_sequence):
        """Calculates a score for a given route."""
        # Example scoring: prioritize routes with more users, shorter total distance, better coherence
        score = 0
        num_users = len(route['assigned_users'])
        score += num_users * 10

        # Add a penalty for deviation from a direct path to the office
        total_route_dist = 0
        if optimized_sequence:
            path = [driver_pos] + [(route['assigned_users'][i]['lat'], route['assigned_users'][i]['lng']) for i in optimized_sequence]
            for i in range(len(path) - 1):
                total_route_dist += haversine_distance(path[i][0], path[i][1], path[i+1][0], path[i+1][1])

            direct_dist = haversine_distance(driver_pos[0], driver_pos[1], office_pos[0], office_pos[1])
            if direct_dist > 0:
                detour_ratio = total_route_dist / direct_dist
                score -= detour_ratio * 5 # Penalize longer detours

            # Add coherence score if available
            if self.road_network:
                user_positions = [(u['lat'], u['lng']) for u in route['assigned_users']]
                coherence = self.road_network.get_route_coherence_score(driver_pos, user_positions, office_pos)
                score += coherence * 5

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
    users = data.get('users', [])
    drivers_data = data.get('drivers', {})
    all_drivers = drivers_data.get('driversUnassigned', []) + drivers_data.get('driversAssigned', [])
    office_lat = data['company']['latitude']
    office_lon = data['company']['longitude']
    office_pos = (office_lat, office_lon)

    routes = []
    assigned_user_ids = set()
    used_driver_ids = set()

    try:
        # Initialize real RoadNetwork with GraphML file
        print("🗺️ Loading road network from GraphML...")
        global ROAD_NETWORK
        ROAD_NETWORK = RoadNetwork('tricity_main_roads.graphml')
        print(f"✅ Road network loaded successfully")

        # Phase 1: Analyze road corridors from office
        print("📍 Analyzing road corridors from office...")
        corridor_analyzer = RoadCorridorAnalyzer(ROAD_NETWORK, office_pos)

        # Phase 2: Assign users to corridors
        print("👥 Assigning users to road corridors...")
        corridor_analyzer.assign_users_to_corridors(users)

        # Phase 3: Get corridor-based routes
        corridor_routes = corridor_analyzer.get_corridor_routes()
        print(f"🛣️ Identified {len(corridor_routes)} corridors with users")

    except Exception as e:
        print(f"⚠️ Could not load road network: {e}")
        print("📍 Falling back to simple geometric assignment...")
        # Fallback to simple assignment without road network
        corridor_routes = _create_fallback_corridors(users, office_pos)
        ROAD_NETWORK = None

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
        while corridor_users:
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

                # Assign users to this route (up to capacity)
                users_for_this_route = corridor_users[:driver['capacity']]
                remaining_corridor_users = corridor_users[driver['capacity']:]

                # Add users to route with road-aware validation
                successfully_assigned_users = []
                for user in users_for_this_route:
                    if user is None:
                        continue
                        
                    try:
                        if ROAD_NETWORK and route_assigner and route_assigner.road_network:
                            # Use road network validation
                            driver_pos = (current_route['latitude'], current_route['longitude'])
                            user_pos = (user['lat'], user['lng'])

                            # Check if user is reasonably accessible via road network
                            if ROAD_NETWORK.is_user_on_route_path(
                                driver_pos, 
                                [(u['lat'], u['lng']) for u in current_route['assigned_users']],
                                user_pos, 
                                office_pos, 
                                max_detour_ratio=1.3, 
                                route_type="efficiency"
                            ):
                                current_route['assigned_users'].append(user)
                                assigned_user_ids.add(user['user_id'])
                                successfully_assigned_users.append(user)
                            else:
                                remaining_corridor_users.append(user)  # Put back for next driver
                        else:
                            # Simple assignment without road validation
                            current_route['assigned_users'].append(user)
                            assigned_user_ids.add(user['user_id'])
                            successfully_assigned_users.append(user)
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
                        used_driver_ids.add(driver_id)
                        corridor_assigned = True

                        print(f"    ✅ Created route for driver {driver_id} with {len(current_route['assigned_users'])} users")
                    except Exception as route_finalize_error:
                        print(f"    ⚠️ Error finalizing route for driver {driver_id}: {route_finalize_error}")

            except Exception as route_create_error:
                print(f"    ⚠️ Error creating route for driver {driver_id}: {route_create_error}")

            # Mark this driver as used so it won't be selected again
            used_driver_ids.add(driver_id)

        if not corridor_assigned and corridor_users:
            print(f"    ⚠️ Could not assign {len(corridor_users)} users from corridor {corridor_route['direction']}")

    # PHASE 2: Handle remaining unassigned users with available drivers
    remaining_users = [u for u in users if str(u['id']) not in assigned_user_ids]
    available_drivers_remaining = [d for d in sorted_drivers if str(d['id']) not in used_driver_ids]
    
    if remaining_users and available_drivers_remaining:
        print(f"🔄 Assigning {len(remaining_users)} remaining users to {len(available_drivers_remaining)} available drivers...")

        for user in remaining_users:
            if not available_drivers_remaining:
                break

            # Find best available driver for this user
            best_driver = None
            best_distance = float('inf')

            for driver in available_drivers_remaining:
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
                assigned_user_ids.add(str(user['id']))
                
                # Remove this driver from available pool
                available_drivers_remaining = [d for d in available_drivers_remaining if str(d['id']) != str(best_driver['id'])]

                print(f"    ✅ Assigned remaining user {user['id']} to driver {best_driver['id']}")

    # Handle unassigned users and drivers
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
                )
            }
            if user.get('first_name'):
                user_data['first_name'] = str(user['first_name'])
            if user.get('email'):
                user_data['email'] = str(user['email'])
            unassigned_users.append(user_data)

    unassigned_drivers = []
    for driver in all_drivers:
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
    print(f"🎯 Road-aware assignment complete: {len(routes)} routes, {total_assigned} users assigned")

    return {
        "status": "true",
        "data": routes,
        "unassignedUsers": unassigned_users,
        "unassignedDrivers": unassigned_drivers,
        "message": f"Road-aware assignment completed with {len(routes)} routes"
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
                    float(user['latitude']), float(user['longitude']), office_lat, office_lon
                )
                # Find the minimum office distance among currently assigned users
                last_user_office_distance = min([
                    haversine_distance(u['lat'], u['lng'], office_lat, office_lon) 
                    for u in current_route['assigned_users']
                ])
                if user_office_distance > last_user_office_distance:
                    continue # Skip users that force backtracking

            # If all checks pass, tentatively add the user
            user_data = {
                'user_id': str(user['id']),
                'lat': float(user['latitude']),
                'lng': float(user['longitude']),
                'office_distance': haversine_distance(float(user['latitude']), float(user['longitude']), office_lat, office_lon)
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
            # 4. Fix Pickup Sequencing (for 'straight' routes, use sorted-by-office-distance)
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

                    # More relaxed constraints: up to 15km distance for distant users
                    if distance < 15.0 and distance < best_distance:
                        best_distance = distance
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
            used_driver_ids.add(emergency_driver['id'])

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
    for _, driver in pd.DataFrame(all_drivers).iterrows(): # Use DataFrame for easier filtering
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
    print(f"🎯 Road-aware assignment complete: {len(routes)} routes, {total_assigned} users assigned")

    return {
        "status": "true",
        "data": routes,
        "unassignedUsers": unassigned_users,
        "unassignedDrivers": unassigned_drivers,
        "message": f"Road-aware assignment completed with {len(routes)} routes"
    }


def run_road_aware_assignment(source_id: str, parameter: int = 1, string_param: str = ""):
    """Main entry point for road-aware assignment"""
    start_time = time.time()

    print(f"🗺️ Starting road-aware assignment for source_id: {source_id}")
    print(f"📋 Parameter: {parameter}, String parameter: {string_param}")

    try:
        # Load data from API
        # Replace this with actual data loading logic
        def load_env_and_fetch_data(source_id, parameter, string_param):
            # Mock data loading
            return {
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