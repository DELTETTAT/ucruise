import json
import time
import pandas as pd
from loguru import logger
from haversine import haversine, Unit

# Assuming these are defined elsewhere and accessible
# from .route_assigner import SimpleRouteAssigner
# from .road_network import RoadNetwork
# from .config import OPTIMIZATION_CONFIGS, ROAD_NETWORK

# Mock implementations for demonstration purposes
class MockRoadNetwork:
    def is_user_on_route_path(self, driver_pos, route_users_pos, candidate_user_pos, office_pos, max_detour_ratio, route_type):
        # Simplified logic: just check if candidate user is generally in the direction of the office
        # A real implementation would involve pathfinding and angle calculations
        if not route_users_pos:
            return True

        # Basic check: candidate user should not be significantly further away from the office than the last user
        candidate_dist = haversine(candidate_user_pos[0], candidate_user_pos[1], office_pos[0], office_pos[1])
        last_user_dist = min([haversine(u[0], u[1], office_pos[0], office_pos[1]) for u in route_users_pos])

        return candidate_dist <= last_user_dist * 1.5 # Allow some detour


    def get_route_coherence_score(self, driver_pos, test_users_pos, office_pos):
        # Simplified coherence score: measures how well the path segments align
        # A score of 1.0 means a perfect straight line, lower means more deviation
        if len(test_users_pos) < 2:
            return 1.0

        total_dist = 0
        straight_dist = haversine(driver_pos[0], driver_pos[1], test_users_pos[-1][0], test_users_pos[-1][1])
        
        for i in range(len(test_users_pos) - 1):
            total_dist += haversine(test_users_pos[i][0], test_users_pos[i][1], test_users_pos[i+1][0], test_users_pos[i+1][1])
        
        coherence = straight_dist / (total_dist + 1e-6) # Add epsilon to avoid division by zero
        return max(0.0, min(1.0, coherence * 1.5)) # Scale to be between 0 and 1

    def get_optimal_pickup_sequence(self, driver_pos, user_positions, office_pos):
        # Placeholder for a more sophisticated TSP solver or heuristic
        # For now, just return the order as is, or sorted by distance to office
        
        # Simple greedy approach: sort by distance to office
        user_dist_pairs = []
        for i, user_pos in enumerate(user_positions):
            dist = haversine(user_pos[0], user_pos[1], office_pos[0], office_pos[1])
            user_dist_pairs.append((dist, i))
        
        user_dist_pairs.sort()
        return [idx for dist, idx in user_dist_pairs]


class SimpleRouteAssigner:
    def __init__(self, config):
        self.config = config
        self.road_network = None

    def assign_users_to_routes(self, users, drivers, office_pos, routes, assigned_user_ids, used_driver_ids, mode):
        """Assigns users to existing routes based on the specified mode."""
        
        available_drivers = [d for d in drivers if str(d['id']) not in used_driver_ids]
        
        # Sort users by their distance to the office for efficient processing
        # This helps in prioritizing users closer to the final destination
        sorted_users = sorted(users, key=lambda u: haversine(float(u['latitude']), float(u['longitude']), office_pos[0], office_pos[1]))

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
                        float(user['latitude']), float(user['longitude']), office_lat, office_lon
                    )
                    last_user_office_distance = min([
                        haversine_distance(u['lat'], u['lng'], office_lat, office_lon) 
                        for u in route_users
                    ])
                    if user_office_distance > last_user_office_distance:
                        continue  # Skip users that force backtracking

                # Check capacity
                if len(route_users) >= driver['capacity']:
                    continue

                route_users.append(user)
                assigned_user_ids.add(str(user['id']))

                # Re-optimize sequence for the updated route
                # Calculate office distances for all users if not present
                for u in route_users:
                    if 'office_distance' not in u or u['office_distance'] == 0:
                        u['office_distance'] = haversine_distance(
                            float(u['latitude']), float(u['longitude']), office_lat, office_lon
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
                            float(u['latitude']), float(u['longitude']), office_pos[0], office_pos[1]
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
                    'office_distance': user['office_distance'] 
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
                total_route_dist += haversine(path[i][0], path[i][1], path[i+1][0], path[i+1][1])
            
            direct_dist = haversine(driver_pos[0], driver_pos[1], office_pos[0], office_pos[1])
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
    """Calculate the great-circle distance between two points on the Earth."""
    return haversine(lat1, lon1, lat2, lon2, unit=Unit.KILOMETERS)


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
    """Performs road-aware route assignment."""
    users = data.get('users', [])
    drivers_data = data.get('drivers', {})
    all_drivers = drivers_data.get('driversUnassigned', []) + drivers_data.get('driversAssigned', [])
    office_lat = data['company']['latitude']
    office_lon = data['company']['longitude']
    office_pos = (office_lat, office_lon)

    routes = []
    assigned_user_ids = set()
    used_driver_ids = set()

    # Mock RoadNetwork and RouteAssigner for demonstration
    # In a real scenario, these would be properly initialized and configured
    ROAD_NETWORK = MockRoadNetwork() # Replace with actual RoadNetwork initialization
    OPTIMIZATION_CONFIGS = {
        "efficiency": {"max_detour_ratio": 1.1},
        "balanced": {"max_detour_ratio": 1.3},
        "capacity": {"max_detour_ratio": 1.5}
    }
    
    # Mode selection (e.g., "efficiency", "balanced", "capacity")
    # This should ideally come from configuration or user input
    mode = "efficiency" # Defaulting to efficiency for demonstration

    route_assigner = SimpleRouteAssigner(OPTIMIZATION_CONFIGS[mode])
    if ROAD_NETWORK:
        route_assigner.road_network = ROAD_NETWORK

    # PHASE 1: Assign users to available drivers, prioritizing efficient routes
    
    # Sort drivers by capacity (e.g., larger capacity first for flexibility) or other criteria
    sorted_drivers = sorted(all_drivers, key=lambda d: d['capacity'], reverse=True)
    
    available_users = list(users) # Keep track of users not yet assigned

    for driver in sorted_drivers:
        if len(assigned_user_ids) == len(users):
            break # All users assigned

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
        logger.error(f"Road-aware assignment execution failed: {e}", exc_info=True)
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

    # Mock ROAD_NETWORK and OPTIMIZATION_CONFIGS if they are not globally defined
    global ROAD_NETWORK, OPTIMIZATION_CONFIGS
    ROAD_NETWORK = MockRoadNetwork()
    OPTIMIZATION_CONFIGS = {
        "efficiency": {"max_detour_ratio": 1.1},
        "balanced": {"max_detour_ratio": 1.3},
        "capacity": {"max_detour_ratio": 1.5}
    }


    result = assign_routes_road_aware(sample_data)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()