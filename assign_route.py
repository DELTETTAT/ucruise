import json
import time
import math
import os
import pandas as pd
from road_network import RoadNetwork
import networkx as nx
import traceback
from logger_config import get_logger

# Global variables that will be set during assignment
ROAD_NETWORK = None
OPTIMIZATION_CONFIGS = {
    "efficiency": {"max_detour_ratio": 1.1},
    "balanced": {"max_detour_ratio": 1.3},
    "capacity": {"max_detour_ratio": 1.5}
}

def _ensure_assigned_users(route):
    """Defensive helper to ensure route has assigned_users list"""
    if route is None:
        return False
    if 'assigned_users' not in route or route['assigned_users'] is None:
        route['assigned_users'] = []
    return True

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

            if best_corridor and min_bearing_diff <= 22.5:  # Tightened bearing tolerance
                self.corridors[best_corridor]['users'].append({
                    'user_id': str(user['id']),
                    'lat': float(user['latitude']),
                    'lng': float(user['longitude']),
                    'office_distance': haversine_distance(
                        user_pos[0], user_pos[1],
                        self.office_pos[0], self.office_pos[1]
                    )
                })

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
        """Assigns users to existing routes with strict quality validation and driver fallback."""

        available_drivers = [d for d in drivers if str(d['id']) not in used_driver_ids]

        # Sort users by their distance to the office for efficient processing
        sorted_users = sorted(users, key=lambda u: haversine_distance(float(u['latitude']), float(u['longitude']), office_pos[0], office_pos[1]))

        for user in sorted_users:
            if str(user['id']) in assigned_user_ids:
                continue

            best_route_for_user = None
            best_route_score = -float('inf')
            route_quality_acceptable = False

            # Try to assign user to an existing route with STRICT quality checks
            for route in routes:
                logger = get_logger()
                user_pos = (float(user['latitude']), float(user['longitude']))
                driver_pos = (route['latitude'], route['longitude'])

                logger.debug(f"🔍 Evaluating user {user['id']} for route {route['driver_id']}")
                logger.debug(f"   User position: ({user_pos[0]:.6f}, {user_pos[1]:.6f})")
                logger.debug(f"   Driver position: ({driver_pos[0]:.6f}, {driver_pos[1]:.6f})")

                # Check capacity first
                if len(route['assigned_users']) >= route['vehicle_type']:
                    logger.debug(f"   ❌ Route {route['driver_id']} rejected: At capacity ({len(route['assigned_users'])}/{route['vehicle_type']})")
                    continue

                route_users = route['assigned_users']

                # Determine route type for road network functions
                route_type = "straight" if mode == "efficiency" else "balanced" if mode == "balanced" else "capacity"
                logger.debug(f"   📍 Route type: {route_type}, Mode: {mode}")

                # Check if user is on route path with road network awareness
                config = OPTIMIZATION_CONFIGS.get(mode, OPTIMIZATION_CONFIGS["balanced"])
                max_detour = config.get('max_detour_ratio', 1.25)
                logger.debug(f"   ⚙️ Max detour ratio: {max_detour}")

                # STRICT route quality validation - no compromises on quality
                route_quality_metrics = self._validate_route_quality_strict(
                    route, user_pos, office_pos, mode, max_detour
                )

                if route_quality_metrics is None or not route_quality_metrics.get('acceptable', False):
                    rejection_reason = route_quality_metrics.get('rejection_reason', 'Unknown validation failure') if route_quality_metrics else 'Validation returned None'
                    logger.debug(f"   ❌ Route {route['driver_id']} rejected: {rejection_reason}")
                    continue

                logger.debug(f"   ✅ Route {route['driver_id']} quality validation passed")
                route_quality_acceptable = True

                # Sequential routing - no backtracking toward office
                backtracking_check_passed = True
                if route_users:
                    user_office_distance = haversine_distance(
                        user_pos[0], user_pos[1], office_pos[0], office_pos[1]
                    )
                    last_user_office_distance = max([
                        haversine_distance(u['lat'], u['lng'], office_pos[0], office_pos[1])
                        for u in route_users
                    ])

                    logger.debug(f"   🏢 User office distance: {user_office_distance:.3f}km")
                    logger.debug(f"   🏢 Closest route user to office: {last_user_office_distance:.3f}km")

                    # Strict check - no tolerance for backtracking
                    if user_office_distance > last_user_office_distance + 0.1:  # 100m tolerance only
                        logger.debug(f"   ❌ Route {route['driver_id']} rejected: Backtracking detected ({user_office_distance:.3f} > {last_user_office_distance + 0.1:.3f})")
                        backtracking_check_passed = False
                        continue  # Skip users that force backtracking
                    else:
                        logger.debug(f"   ✅ No backtracking detected")

                # Check capacity - get capacity from route
                if len(route_users) >= route['vehicle_type']:
                    logger.debug(f"   ❌ Route {route['driver_id']} rejected: At capacity")
                    continue

                logger.debug(f"   ✅ Route {route['driver_id']} available: {len(route_users)}/{route['vehicle_type']} capacity")

                # Create test route with new user for scoring
                test_route = route.copy()
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

                # Safe assignment
                if not _ensure_assigned_users(test_route):
                    logger.error(f"   ❌ Route {route['driver_id']} has invalid structure")
                    continue

                test_route['assigned_users'] = route['assigned_users'] + [test_user_data]

                # Calculate optimized sequence for scoring
                if len(test_route['assigned_users']) > 1 and self.road_network:
                    driver_pos_for_seq = (test_route['latitude'], test_route['longitude'])
                    user_positions_for_seq = [(u['lat'], u['lng']) for u in test_route['assigned_users']]
                    optimized_sequence = self.road_network.get_optimal_pickup_sequence(
                        driver_pos_for_seq, user_positions_for_seq, office_pos
                    )
                else:
                    optimized_sequence = list(range(len(test_route['assigned_users'])))

                current_route_score = self.calculate_route_score(test_route, driver_pos, office_pos, optimized_sequence)

                logger.debug(f"   📊 Route score for {route['driver_id']}: {current_route_score:.3f}")
                logger.debug(f"   📊 Previous best score: {best_route_score:.3f}")

                if current_route_score > best_route_score:
                    logger.debug(f"   🎯 New best route: {route['driver_id']} (score: {current_route_score:.3f})")
                    best_route_score = current_route_score
                    best_route_for_user = route
                else:
                    logger.debug(f"   📊 Route {route['driver_id']} score {current_route_score:.3f} not better than current best {best_route_score:.3f}")

            # If a suitable HIGH-QUALITY route is found, assign the user
            if best_route_for_user and route_quality_acceptable:
                logger.info(f"🎯 QUALITY ASSIGNMENT: User {user['id']} to route {best_route_for_user['driver_id']} (score: {best_route_score:.3f})")

                self._assign_user_to_route(user, best_route_for_user, office_pos, mode)
                assigned_user_ids.add(str(user['id']))

            # QUALITY FALLBACK: Assign new driver instead of compromising route quality
            elif available_drivers:
                logger.info(f"🚗 QUALITY FALLBACK: Creating new route for user {user['id']} to maintain quality standards")

                new_route = self._create_quality_route_for_user(user, available_drivers, office_pos, used_driver_ids)
                if new_route:
                    routes.append(new_route)
                    assigned_user_ids.add(str(user['id']))
                    used_driver_ids.add(new_route['driver_id'])
                    available_drivers = [d for d in available_drivers if str(d['id']) != new_route['driver_id']]
                    logger.info(f"   ✅ Created high-quality route with driver {new_route['driver_id']}")
                else:
                    logger.warning(f"❌ Failed to create new route for user {user['id']}")
            else:
                logger.warning(f"❌ No quality assignment possible for user {user['id']} - no available drivers for fallback")

        return routes, assigned_user_ids

    def _validate_route_quality_strict(self, route, user_pos, office_pos, mode, max_detour):
        """Strict route quality validation with no compromises."""

        driver_pos = (route['latitude'], route['longitude'])
        route_users = route['assigned_users']

        # Create test route with new user
        test_users = [(u['lat'], u['lng']) for u in route_users] + [user_pos]

        # Quality metrics
        metrics = {
            'acceptable': False,
            'rejection_reason': '',
            'coherence_score': 0.0,
            'directional_consistency': 0.0,
            'detour_ratio': float('inf'),
            'zigzag_penalty': 0.0
        }

        if not self.road_network:
            # Without road network, use geometric validation
            return self._validate_geometric_quality(driver_pos, test_users, office_pos, metrics)

        try:
            # 1. Coherence score must be high (>= 0.7)
            coherence = self.road_network.get_route_coherence_score(driver_pos, test_users, office_pos)
            metrics['coherence_score'] = coherence

            if coherence < 0.7:
                metrics['rejection_reason'] = f"Low coherence: {coherence:.3f} < 0.7"
                return metrics

            # 2. User must be on route path
            on_path = self.road_network.is_user_on_route_path(
                driver_pos, [(u['lat'], u['lng']) for u in route_users],
                user_pos, office_pos, max_detour_ratio=max_detour, route_type=mode
            )

            if not on_path:
                metrics['rejection_reason'] = f"User not on optimal route path (detour > {max_detour}x)"
                return metrics

            # 3. Directional consistency check
            directional_consistency = self._calculate_directional_consistency(driver_pos, test_users, office_pos)
            metrics['directional_consistency'] = directional_consistency

            if directional_consistency < 0.8:
                metrics['rejection_reason'] = f"Poor directional consistency: {directional_consistency:.3f} < 0.8"
                return metrics

            # 4. Zigzag penalty check
            zigzag_penalty = self._calculate_zigzag_penalty(driver_pos, test_users, office_pos)
            metrics['zigzag_penalty'] = zigzag_penalty

            if zigzag_penalty > 15.0:  # Max 15 degrees average turning
                metrics['rejection_reason'] = f"Excessive zigzag: {zigzag_penalty:.1f}° > 15°"
                return metrics

            # 5. No backtracking check
            if self._has_backtracking(test_users, office_pos):
                metrics['rejection_reason'] = "Route involves backtracking away from office"
                return metrics

            metrics['acceptable'] = True
            return metrics

        except Exception as e:
            metrics['rejection_reason'] = f"Validation error: {str(e)}"
            return metrics

    def _validate_geometric_quality(self, driver_pos, test_users, office_pos, metrics):
        """Geometric quality validation when road network unavailable."""

        # Basic directional consistency using bearings
        main_bearing = self._calculate_bearing(driver_pos[0], driver_pos[1], office_pos[0], office_pos[1])

        inconsistent_segments = 0
        total_segments = len(test_users)

        current_pos = driver_pos
        for user_pos in test_users:
            segment_bearing = self._calculate_bearing(current_pos[0], current_pos[1], user_pos[0], user_pos[1])
            bearing_diff = abs(self._normalize_bearing_difference(segment_bearing - main_bearing))

            if bearing_diff > 45:  # More than 45° deviation
                inconsistent_segments += 1
            current_pos = user_pos

        consistency_ratio = 1.0 - (inconsistent_segments / max(1, total_segments))
        metrics['directional_consistency'] = consistency_ratio

        if consistency_ratio < 0.7:
            metrics['rejection_reason'] = f"Poor geometric consistency: {consistency_ratio:.3f} < 0.7"
            return metrics

        # Check for excessive detour
        total_distance = sum(haversine_distance(
            test_users[i-1][0] if i > 0 else driver_pos[0],
            test_users[i-1][1] if i > 0 else driver_pos[1],
            test_users[i][0], test_users[i][1]
        ) for i in range(len(test_users)))

        direct_distance = haversine_distance(driver_pos[0], driver_pos[1], office_pos[0], office_pos[1])
        detour_ratio = total_distance / max(direct_distance, 0.1)
        metrics['detour_ratio'] = detour_ratio

        if detour_ratio > 2.0:  # More than 2x direct distance
            metrics['rejection_reason'] = f"Excessive detour: {detour_ratio:.2f}x > 2.0x"
            return metrics

        metrics['acceptable'] = True
        return metrics

    def _calculate_directional_consistency(self, driver_pos, user_positions, office_pos):
        """Calculate how consistently the route heads toward office."""

        if len(user_positions) < 2:
            return 1.0

        main_bearing = self._calculate_bearing(driver_pos[0], driver_pos[1], office_pos[0], office_pos[1])

        consistent_segments = 0
        total_segments = 0

        route_points = [driver_pos] + user_positions
        for i in range(len(route_points) - 1):
            segment_bearing = self._calculate_bearing(
                route_points[i][0], route_points[i][1],
                route_points[i+1][0], route_points[i+1][1]
            )
            bearing_diff = abs(self._normalize_bearing_difference(segment_bearing - main_bearing))

            if bearing_diff <= 30:  # Within 30 degrees is consistent
                consistent_segments += 1
            total_segments += 1

        return consistent_segments / max(1, total_segments)

    def _calculate_zigzag_penalty(self, driver_pos, user_positions, office_pos):
        """Calculate zigzag penalty based on turning angles."""

        if len(user_positions) < 2:
            return 0.0

        route_points = [driver_pos] + user_positions + [office_pos]

        if len(route_points) < 3:
            return 0.0

        total_turning = 0.0
        turning_count = 0

        for i in range(1, len(route_points) - 1):
            bearing1 = self._calculate_bearing(
                route_points[i-1][0], route_points[i-1][1],
                route_points[i][0], route_points[i][1]
            )
            bearing2 = self._calculate_bearing(
                route_points[i][0], route_points[i][1],
                route_points[i+1][0], route_points[i+1][1]
            )

            turning_angle = abs(self._normalize_bearing_difference(bearing2 - bearing1))
            total_turning += turning_angle
            turning_count += 1

        return total_turning / max(1, turning_count)

    def _has_backtracking(self, user_positions, office_pos):
        """Check if route involves backtracking away from office."""

        if len(user_positions) < 2:
            return False

        # Check if distance to office increases significantly at any point
        prev_distance = haversine_distance(user_positions[0][0], user_positions[0][1], office_pos[0], office_pos[1])

        for i in range(1, len(user_positions)):
            current_distance = haversine_distance(user_positions[i][0], user_positions[i][1], office_pos[0], office_pos[1])

            # Allow 500m tolerance for minor variations
            if current_distance > prev_distance + 0.5:
                return True

            prev_distance = current_distance

        return False

    def _assign_user_to_route(self, user, route, office_pos, mode):
        """Assign user to route with optimal sequencing."""

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

        if _ensure_assigned_users(route):
            route['assigned_users'].append(final_user_data)

            # Re-optimize sequence for quality
            if len(route['assigned_users']) > 1:
                self._optimize_route_sequence_quality(route, office_pos, mode)
        else:
            logger = get_logger()
            logger.error("Failed to assign user: route structure invalid")

    def _create_quality_route_for_user(self, user, available_drivers, office_pos, used_driver_ids):
        """Create a new high-quality route for a single user."""

        user_pos = (float(user['latitude']), float(user['longitude']))
        best_driver = None
        best_distance = float('inf')

        # Sort drivers by ID for deterministic selection in case of ties
        sorted_drivers = sorted(available_drivers, key=lambda d: d['id'])

        for driver in sorted_drivers:
            if str(driver['id']) in used_driver_ids:
                continue

            driver_pos = (float(driver['latitude']), float(driver['longitude']))
            distance = haversine_distance(driver_pos[0], driver_pos[1], user_pos[0], user_pos[1])

            # Only consider drivers within reasonable distance (10km max)
            if distance <= 10.0 and distance < best_distance:
                best_distance = distance
                best_driver = driver

        if not best_driver:
            return None

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

        new_route = {
            'driver_id': str(best_driver['id']),
            'vehicle_id': str(best_driver.get('vehicle_id', '')),
            'vehicle_type': int(best_driver['capacity']),
            'latitude': float(best_driver['latitude']),
            'longitude': float(best_driver['longitude']),
            'assigned_users': [user_data]
        }

        update_route_metrics_improved(new_route, office_pos[0], office_pos[1])
        return new_route

    def _optimize_route_sequence_quality(self, route, office_pos, mode):
        """Optimize route sequence prioritizing quality over other factors."""

        if len(route['assigned_users']) <= 1:
            return

        driver_pos = (route['latitude'], route['longitude'])
        user_positions = [(u['lat'], u['lng']) for u in route['assigned_users']]

        if self.road_network and mode != "straight":
            # Use road-aware optimization
            optimal_sequence = self.road_network.get_optimal_pickup_sequence(
                driver_pos, user_positions, office_pos
            )

            # Reorder users according to optimal sequence
            original_users = route['assigned_users'].copy()
            route['assigned_users'] = [original_users[i] for i in optimal_sequence]
        else:
            # Simple distance-based ordering for straight mode or no road network
            route['assigned_users'].sort(key=lambda u: u.get('office_distance', 0))

        update_route_metrics_improved(route, office_pos[0], office_pos[1])

    def _calculate_bearing(self, lat1, lon1, lat2, lon2):
        """Calculate bearing between two points."""
        lat1_rad, lon1_rad, lat2_rad, lon2_rad = map(math.radians, [lat1, lon1, lat2, lon2])
        dlon = lon2_rad - lon1_rad
        x = math.sin(dlon) * math.cos(lat2_rad)
        y = math.cos(lat1_rad) * math.sin(lat2_rad) - math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(dlon)
        initial_bearing = math.atan2(x, y)
        return (math.degrees(initial_bearing) + 360) % 360

    def _normalize_bearing_difference(self, diff):
        """Normalize bearing difference to [-180, 180] range."""
        while diff > 180:
            diff -= 360
        while diff < -180:
            diff += 360
        return diff

    def calculate_route_score(self, route, driver_pos, office_pos, optimized_sequence):
        """Calculates a score for a given route."""
        logger = get_logger()

        # Example scoring: prioritize routes with more users, shorter total distance, better coherence
        score = 0
        num_users = len(route['assigned_users'])
        user_score = num_users * 10
        score += user_score

        logger.debug(f"   📊 Route scoring for {route['driver_id']}:")
        logger.debug(f"     👥 Users: {num_users} (+{user_score} points)")

        # Add a penalty for deviation from a direct path to the office
        total_route_dist = 0
        detour_penalty = 0
        coherence_bonus = 0

        if optimized_sequence:
            path = [driver_pos] + [(route['assigned_users'][i]['lat'], route['assigned_users'][i]['lng']) for i in optimized_sequence]
            for i in range(len(path) - 1):
                total_route_dist += haversine_distance(path[i][0], path[i][1], path[i+1][0], path[i+1][1])

            direct_dist = haversine_distance(driver_pos[0], driver_pos[1], office_pos[0], office_pos[1])
            if direct_dist > 0:
                detour_ratio = total_route_dist / direct_dist
                detour_penalty = detour_ratio * 15  # Stronger penalty for detours
                score -= detour_penalty

                logger.debug(f"     🛣️ Total route distance: {total_route_dist:.3f}km")
                logger.debug(f"     📏 Direct distance: {direct_dist:.3f}km")
                logger.debug(f"     📊 Detour ratio: {detour_ratio:.3f} (-{detour_penalty:.1f} points)")

            # Add coherence score if available
            if self.road_network:
                user_positions = [(u['lat'], u['lng']) for u in route['assigned_users']]
                coherence = self.road_network.get_route_coherence_score(driver_pos, user_positions, office_pos)
                coherence_bonus = coherence * 5
                score += coherence_bonus

                logger.debug(f"     🎯 Coherence: {coherence:.3f} (+{coherence_bonus:.1f} points)")

        final_score = score
        logger.debug(f"     🏆 Final score: {final_score:.3f}")

        return final_score

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
    if not route or not route.get('assigned_users'):
        if route:
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
    logger = get_logger()
    logger.info("🗺️ Starting road-aware route assignment")

    # Defensive validation
    if not data or not isinstance(data, dict):
        raise ValueError("assign_routes_road_aware called with invalid data (None or not a dict)")
    company = data.get('company', {})
    if not company or company.get('latitude') is None or company.get('longitude') is None:
        raise ValueError("Missing company latitude/longitude in input data")

    # Initialize office coordinates early
    office_lat = float(company['latitude'])
    office_lon = float(company['longitude'])
    office_pos = (office_lat, office_lon)

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

    # Sort users by ID for deterministic processing
    users.sort(key=lambda u: u['id'])

    logger.log_data_validation(len(users), 0, (0, 0))

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

    # Sort drivers by ID for deterministic processing
    all_drivers.sort(key=lambda d: d['id'])

    logger.log_data_validation(len(users), len(all_drivers), (0, 0))

    company = data.get('company', {}).get('name', 'Unknown')
    total_drivers = len(all_drivers)

    print("📊 DETAILED API DATA BREAKDOWN:")
    print(f"   👥 Total Users from API: {len(users)}")
    print(f"   🚗 driversUnassigned: {len(drivers_unassigned)}")
    print(f"   🚙 driversAssigned: {len(drivers_assigned)}")
    print(f"   🚛 Total Drivers Available: {total_drivers}")
    print(f"   🏢 Office Location: ({office_lat}, {office_lon})")
    print(f"   📍 Company: {company}")

    routes = []
    assigned_user_ids = set()
    used_driver_ids = set()

    try:
        # Initialize real RoadNetwork with GraphML file
        logger.info("🗺️ Loading road network from GraphML...")
        global ROAD_NETWORK
        ROAD_NETWORK = RoadNetwork('tricity_main_roads.graphml')
        logger.info(f"✅ Road network loaded successfully with {len(ROAD_NETWORK.graph.nodes)} nodes")

        # Phase 1: Analyze road corridors from office
        logger.info("📍 Analyzing road corridors from office...")
        corridor_analyzer = RoadCorridorAnalyzer(ROAD_NETWORK, office_pos)

        # Phase 2: Assign users to corridors
        logger.info("👥 Assigning users to road corridors...")
        corridor_analyzer.assign_users_to_corridors(users)

        # Phase 3: Get corridor-based routes
        corridor_routes = corridor_analyzer.get_corridor_routes()
        logger.info(f"🛣️ Identified {len(corridor_routes)} corridors with users")
        logger.log_clustering_decision("road_corridor", len(users), len(corridor_routes),
                                     {"corridors": [f"{c['direction']}: {len(c['users'])} users" for c in corridor_routes]})

        for corridor in corridor_routes:
            logger.debug(f"   📍 {corridor['direction']}: {len(corridor['users'])} users")

    except Exception as e:
        logger.error(f"❌ ROAD NETWORK INITIALIZATION FAILED: {e}", exc_info=True)
        ROAD_NETWORK = None
        corridor_routes = _create_fallback_corridors(users, office_pos)
        logger.warning("Using geometric fallback corridors due to road network failure")

    # Mode selection
    mode = "efficiency"
    route_assigner = SimpleRouteAssigner(OPTIMIZATION_CONFIGS[mode])
    if ROAD_NETWORK:
        route_assigner.road_network = ROAD_NETWORK

    # PHASE 1: Create routes based on corridor analysis
    print("🚗 Creating routes along identified corridors...")

    # Sort drivers by capacity (larger capacity first for flexibility), then by ID for determinism
    sorted_drivers = sorted(all_drivers, key=lambda d: (d['capacity'], d['id']), reverse=True)

    # Create routes for each corridor with users
    for corridor_route in corridor_routes:
        corridor_users = corridor_route['users'].copy()
        if not corridor_users:
            continue

        print(f"  📍 Processing corridor {corridor_route['direction']} with {len(corridor_users)} users")

        # Assign drivers to this corridor
        corridor_iterations = 0
        max_corridor_iterations = 100

        while corridor_users and corridor_iterations < max_corridor_iterations:
            corridor_iterations += 1

            # Find next available driver
            available_driver = None
            for driver in sorted_drivers:
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

            for user in corridor_users:
                if users_assigned_to_this_driver >= driver['capacity']:
                    remaining_corridor_users.append(user)
                    continue

                try:
                    # Quick distance check first
                    driver_pos = (current_route['latitude'], current_route['longitude'])
                    user_pos = (user['lat'], user['lng'])
                    distance = haversine_distance(driver_pos[0], driver_pos[1], user_pos[0], user_pos[1])

                    if distance <= 4.0:  # Tightened corridor assignment distance
                        logger.info(f"      ✅ CORRIDOR ASSIGNMENT: User {user['user_id']} assigned to driver {driver_id}")
                        logger.debug(f"         📍 Distance to driver: {distance:.3f}km")
                        logger.debug(f"         🛣️ Corridor: {corridor_route['direction']}")
                        logger.debug(f"         🎯 User position: ({user_pos[0]:.6f}, {user_pos[1]:.6f})")

                        # Log detailed assignment reasoning
                        corridor_assignment_details = {
                            'corridor_direction': corridor_route['direction'],
                            'distance_to_driver': distance,
                            'corridor_bearing': corridor_route['bearing'],
                            'assignment_phase': 'corridor_initial_assignment',
                            'driver_capacity_used': f"{users_assigned_to_this_driver + 1}/{driver['capacity']}"
                        }

                        logger.log_user_assignment(
                            user['user_id'],
                            driver_id,
                            corridor_assignment_details
                        )

                        if _ensure_assigned_users(current_route):
                            current_route['assigned_users'].append(user)
                            assigned_user_ids.add(user['user_id'])
                            users_assigned_to_this_driver += 1
                        else:
                            logger.error(f"Failed to assign user {user['user_id']}: route structure invalid")
                    else:
                        logger.debug(f"      ❌ User {user['user_id']} too far from driver {driver_id}: {distance:.3f}km > 15km")
                        remaining_corridor_users.append(user)

                except Exception as user_assign_error:
                    logger.error(f"    ⚠️ Error assigning user {user.get('user_id', 'unknown')} to driver {driver_id}: {user_assign_error}", exc_info=True)
                    remaining_corridor_users.append(user)

            # Update corridor_users for next iteration
            corridor_users = remaining_corridor_users

            # Only create route if users were assigned
            if current_route['assigned_users']:
                try:
                    # Simple ordering by distance to office
                    current_route['assigned_users'].sort(key=lambda u: u.get('office_distance', 0))

                    # Update route metrics
                    update_route_metrics_improved(current_route, office_lat, office_lon)

                    # Log detailed route creation
                    quality_metrics = {
                        'total_distance': current_route.get('total_distance', 0),
                        'estimated_time': current_route.get('estimated_time', 0),
                        'utilization': len(current_route['assigned_users']) / current_route['vehicle_type'],
                        'corridor': corridor_route['direction']
                    }

                    logger.log_route_creation(
                        driver_id,
                        current_route['assigned_users'],
                        f"Created along {corridor_route['direction']} corridor with {users_assigned_to_this_driver} users",
                        quality_metrics
                    )

                    # Log each user assignment
                    for user in current_route['assigned_users']:
                        logger.log_user_assignment(
                            user['user_id'],
                            driver_id,
                            {
                                'pickup_order': current_route['assigned_users'].index(user) + 1,
                                'distance_from_driver': haversine_distance(
                                    current_route['latitude'], current_route['longitude'],
                                    user['lat'], user['lng']
                                ),
                                'corridor': corridor_route['direction']
                            }
                        )

                    routes.append(current_route)
                    used_driver_ids.add(driver_id)

                    logger.info(f"    ✅ Created route for driver {driver_id} with {len(current_route['assigned_users'])} users")
                except Exception as route_finalize_error:
                    logger.error(f"    ⚠️ Error finalizing route for driver {driver_id}: {route_finalize_error}", exc_info=True)

        if corridor_users:
            logger.warning(f"    ⚠️ Could not assign {len(corridor_users)} users from corridor {corridor_route['direction']}")
            for unassigned_user in corridor_users:
                logger.log_user_unassigned(
                    unassigned_user['user_id'],
                    f"No available driver in {corridor_route['direction']} corridor",
                    [d['id'] for d in sorted_drivers if str(d['id']) not in used_driver_ids][:3]
                )

    # PHASE 2: Cross-corridor filling
    logger.info("🔄 Phase 2: Cross-corridor route filling...")
    logger.log_optimization_step(
        "Cross-corridor filling",
        {"routes": len(routes), "assigned_users": len(assigned_user_ids)},
        {},
        f"Starting cross-corridor filling with {len([u for u in users if str(u['id']) not in assigned_user_ids])} remaining users"
    )

    remaining_users = [u for u in users if str(u['id']) not in assigned_user_ids]
    # Sort remaining users by ID for deterministic processing
    remaining_users.sort(key=lambda u: u['id'])

    # Sort routes by driver_id for deterministic processing
    routes_sorted = sorted(routes, key=lambda r: r['driver_id'])
    for route in routes_sorted:
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

            if distance_to_route <= 1.0:  # Tightened cross-corridor distance
                logger.info(f"🔄 CROSS-CORRIDOR ASSIGNMENT: User {user['id']} assigned to route {route['driver_id']}")
                logger.debug(f"   📍 Distance to route center: {distance_to_route:.3f}km")
                logger.debug(f"   🎯 Route center: ({route_center[0]:.6f}, {route_center[1]:.6f})")
                logger.debug(f"   👤 User position: ({user_pos[0]:.6f}, {user_pos[1]:.6f})")

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

                # Log cross-corridor assignment details
                cross_corridor_details = {
                    'distance_to_route_center': distance_to_route,
                    'route_center_coords': route_center,
                    'available_capacity_before': route['vehicle_type'] - len(route['assigned_users']),
                    'assignment_phase': 'cross_corridor_filling',
                    'original_corridor': route.get('corridor_direction', 'unknown')
                }

                logger.log_user_assignment(
                    str(user['id']),
                    route['driver_id'],
                    cross_corridor_details
                )

                if _ensure_assigned_users(route):
                    route['assigned_users'].append(user_data)
                    assigned_user_ids.add(str(user['id']))
                    users_added_to_route += 1
                    users_to_remove.append(user)
                else:
                    logger.error(f"Failed to assign user {user['id']}: route structure invalid")
            else:
                logger.debug(f"   ❌ User {user['id']} too far from route center: {distance_to_route:.3f}km > 5km")

        for user in users_to_remove:
            remaining_users.remove(user)

        if users_added_to_route > 0:
            update_route_metrics_improved(route, office_lat, office_lon)

    # PHASE 3: Emergency assignment for remaining users
    final_remaining_users = [u for u in users if str(u['id']) not in assigned_user_ids]
    final_available_drivers = [d for d in sorted_drivers if str(d['id']) not in used_driver_ids]

    if final_remaining_users and final_available_drivers:
        logger.warning(f"🚨 Emergency assignment for {len(final_remaining_users)} remaining users...")
        logger.log_optimization_step(
            "Emergency assignment",
            {"unassigned_users": len(final_remaining_users), "available_drivers": len(final_available_drivers)},
            {},
            "Starting emergency single-user route assignments"
        )

        for user in final_remaining_users:
            if not final_available_drivers:
                break

            best_driver = None
            best_distance = float('inf')

            # Sort available drivers by ID for deterministic processing
            sorted_available_drivers = sorted(final_available_drivers, key=lambda d: d['id'])
            for driver in sorted_available_drivers:
                if str(driver['id']) not in used_driver_ids:
                    user_pos = (float(user['latitude']), float(user['longitude']))
                    driver_pos = (float(driver['latitude']), float(driver['longitude']))
                    distance = haversine_distance(driver_pos[0], driver_pos[1], user_pos[0], user_pos[1])

                    logger.debug(f"     🚗 Evaluating driver {driver['id']} for emergency assignment")
                    logger.debug(f"       📏 Distance: {distance:.3f}km (threshold: 20km)")
                    logger.debug(f"       🎯 Driver position: ({driver_pos[0]:.6f}, {driver_pos[1]:.6f})")

                    # Use driver ID for tie-breaking when distances are equal
                    if (distance < best_distance and distance < 20.0) or \
                       (abs(distance - best_distance) < 0.001 and distance < 20.0 and driver['id'] < best_driver['id']):
                        logger.debug(f"       🎯 New best emergency driver: {driver['id']} (distance: {distance:.3f}km)")
                        best_distance = distance
                        best_driver = driver

            if best_driver is not None:
                logger.info(f"🚨 EMERGENCY ASSIGNMENT: User {user['id']} -> Driver {best_driver['id']}")
                logger.debug(f"   📏 Distance: {best_distance:.3f}km")
                logger.debug(f"   🎯 Best driver details: ID={best_driver['id']}, Capacity={best_driver['capacity']}")

                # Double-check driver isn't already used (safety check)
                if str(best_driver['id']) in used_driver_ids:
                    logger.warning(f"❌ Driver {best_driver['id']} already used - skipping to prevent duplication")
                    logger.warning(f"   🔍 Used driver IDs: {list(used_driver_ids)}")
                    continue

                used_driver_ids.add(str(best_driver['id']))

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

                logger.debug(f"   📋 Emergency route user data: {user_data}")

                single_user_route = {
                    'driver_id': str(best_driver['id']),
                    'vehicle_id': str(best_driver.get('vehicle_id', '')),
                    'vehicle_type': int(best_driver['capacity']),
                    'latitude': float(best_driver['latitude']),
                    'longitude': float(best_driver['longitude']),
                    'assigned_users': [user_data]
                }

                logger.debug(f"   🚗 Emergency route created: {single_user_route['driver_id']} with user {user_data['user_id']}")

                update_route_metrics_improved(single_user_route, office_lat, office_lon)

                # Log emergency route creation
                emergency_quality_metrics = {
                    'total_distance': single_user_route.get('total_distance', 0),
                    'estimated_time': single_user_route.get('estimated_time', 0),
                    'utilization': 1.0 / single_user_route['vehicle_type'],
                    'assignment_reason': 'emergency_single_user',
                    'distance_to_user': best_distance
                }

                logger.log_route_creation(
                    str(best_driver['id']),
                    [user_data],
                    f"Emergency single-user route (distance: {best_distance:.2f}km)",
                    emergency_quality_metrics
                )

                logger.log_user_assignment(
                    str(user['id']),
                    str(best_driver['id']),
                    {
                        'pickup_order': 1,
                        'distance_from_driver': best_distance,
                        'assignment_type': 'emergency'
                    }
                )

                routes.append(single_user_route)

                final_available_drivers = [d for d in final_available_drivers if str(d['id']) != str(best_driver['id'])]

            else:
                logger.error(f"🚨 EMERGENCY ASSIGNMENT FAILED: No suitable driver found for user {user['id']}")
                logger.error(f"   📍 User position: ({float(user['latitude']):.6f}, {float(user['longitude']):.6f})")
                logger.error(f"   🚗 Available drivers: {[d['id'] for d in final_available_drivers]}")
                logger.error(f"   📏 Distances to available drivers: {[haversine_distance(float(user['latitude']), float(user['longitude']), float(d['latitude']), float(d['longitude'])) for d in final_available_drivers]}")

    # PHASE 4: Advanced optimization phases
    print("🔧 Starting advanced optimization phases...")

    if routes is None:
        routes = []

    routes, used_driver_ids = merge_underutilized_routes(routes, all_drivers, office_pos, ROAD_NETWORK)
    routes, used_driver_ids = split_high_detour_routes(routes, all_drivers, office_pos, ROAD_NETWORK)
    routes = local_route_optimization(routes, [], office_pos, ROAD_NETWORK)
    routes = global_optimization_pass(routes, office_pos, ROAD_NETWORK)

    # Final accounting
    unassigned_users = []
    for user in users:
        if str(user['id']) not in assigned_user_ids:
            user_pos = (float(user['latitude']), float(user['longitude']))
            office_distance = haversine_distance(user_pos[0], user_pos[1], office_lat, office_lon)

            # Analyze why user wasn't assigned
            reasons = []

            # Check distances to all drivers
            min_driver_distance = float('inf')
            closest_driver = None
            for driver in all_drivers:
                driver_pos = (float(driver['latitude']), float(driver['longitude']))
                dist = haversine_distance(user_pos[0], user_pos[1], driver_pos[0], driver_pos[1])
                if dist < min_driver_distance:
                    min_driver_distance = dist
                    closest_driver = driver['id']

            if min_driver_distance > 20.0:
                reasons.append(f"Too far from nearest driver ({min_driver_distance:.1f}km > 20km)")

            # Check if all drivers were at capacity
            drivers_at_capacity = 0
            for route in routes:
                if len(route['assigned_users']) >= route['vehicle_type']:
                    drivers_at_capacity += 1

            if drivers_at_capacity == len(routes):
                reasons.append("All routes at capacity")

            # Check office distance
            if office_distance > 50.0:
                reasons.append(f"Very far from office ({office_distance:.1f}km)")

            detailed_reason = "; ".join(reasons) if reasons else "Unknown assignment failure"

            logger.error(f"🚨 UNASSIGNED USER ANALYSIS: {user['id']}")
            logger.error(f"   📍 Position: ({user_pos[0]:.6f}, {user_pos[1]:.6f})")
            logger.error(f"   🏢 Office distance: {office_distance:.3f}km")
            logger.error(f"   🚗 Closest driver: {closest_driver} ({min_driver_distance:.3f}km)")
            logger.error(f"   ❌ Reasons: {detailed_reason}")

            user_data = {
                'user_id': str(user['id']),
                'lat': float(user['latitude']),
                'lng': float(user['longitude']),
                'office_distance': office_distance,
                'reason': detailed_reason
            }
            if user.get('first_name'):
                user_data['first_name'] = str(user['first_name'])
            if user.get('email'):
                user_data['email'] = str(user['email'])

            unassigned_users.append(user_data)

            # Log to tracking system
            logger.log_user_unassigned(
                str(user['id']),
                detailed_reason,
                [str(d['id']) for d in all_drivers][:5]  # Sample of driver IDs
            )

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

    # Log final summary with detailed analysis
    logger.log_final_summary(
        len(users), len(assigned_user_ids), unassigned_users,
        total_drivers, len(used_driver_ids), unused_drivers, routes
    )

    logger.log_accounting_check(
        len(users), len(assigned_user_ids), len(unassigned_users),
        len(users) - (len(assigned_user_ids) + len(unassigned_users))
    )

    print(f"📊 FINAL ASSIGNMENT SUMMARY:")
    print(f"   🗺️ Road Network Status: {'✅ ACTIVE' if ROAD_NETWORK else '❌ FAILED - Using Geometric Fallback'}")
    print(f"   📍 Routes Created: {len(routes)}")
    print(f"   👥 Users Successfully Assigned: {total_assigned_users}")
    print(f"   📊 Assignment Rate: {(len(assigned_user_ids)/len(users)*100):.1f}%")
    print(f"   📊 Overall utilization: {utilization:.1f}%")

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
    if not route or not route.get('assigned_users'):
        return (route.get('latitude', 0), route.get('longitude', 0))

    avg_lat = sum(u['lat'] for u in route['assigned_users']) / len(route['assigned_users'])
    avg_lng = sum(u['lng'] for u in route['assigned_users']) / len(route['assigned_users'])
    return (avg_lat, avg_lng)

def merge_underutilized_routes(routes, all_drivers, office_pos, road_network):
    """Merge underutilized routes (especially 1-2 user routes) into nearby routes"""
    print("  🔗 Phase 1: Merging underutilized routes...")

    merged_routes = []
    retired_driver_ids = set()
    merges_made = 0

    # Sort routes by utilization (lowest first for merging)
    routes_by_util = sorted(routes, key=lambda r: r['driver_id']) # Sort by driver_id for determinism

    for i, route_a in enumerate(routes_by_util):
        if route_a['driver_id'] in retired_driver_ids:
            continue

        # Target underutilized routes (≤2 users or <50% capacity)
        utilization_a = len(route_a['assigned_users']) / route_a['vehicle_type']
        if len(route_a['assigned_users']) > 2 and utilization_a > 0.5:
            merged_routes.append(route_a)
            continue

        # Look for a suitable route to merge into
        best_merge_route = None
        best_merge_score = float('inf')

        for j, route_b in enumerate(routes_by_util):
            if i == j or route_b['driver_id'] in retired_driver_ids:
                continue

            # Check if route_b has capacity
            available_capacity = route_b['vehicle_type'] - len(route_b['assigned_users'])
            if available_capacity < len(route_a['assigned_users']):
                continue

            # Check distance between route centers
            center_a = calculate_route_center(route_a)
            center_b = calculate_route_center(route_b)
            center_distance = haversine_distance(center_a[0], center_a[1], center_b[0], center_b[1])

            if center_distance > 8.0:  # Max 8km between route centers
                continue

            merge_score = center_distance
            if merge_score < best_merge_score:
                best_merge_score = merge_score
                best_merge_route = route_b

        if best_merge_route:
            # Perform the merge
            if _ensure_assigned_users(best_merge_route):
                for user in route_a['assigned_users']:
                    best_merge_route['assigned_users'].append(user)

                # Re-optimize the merged route
                update_route_metrics_improved(best_merge_route, office_pos[0], office_pos[1])
                retired_driver_ids.add(route_a['driver_id'])
                merges_made += 1

                print(f"    ✅ Merged route {route_a['driver_id']} into route {best_merge_route['driver_id']}")
            else:
                merged_routes.append(route_a)
        else:
            merged_routes.append(route_a)

    # Add non-retired routes
    for route in routes_by_util:
        if route['driver_id'] not in retired_driver_ids and route not in merged_routes:
            merged_routes.append(route)

    print(f"    🔗 Completed {merges_made} route merges")

    # Update used_driver_ids by removing retired drivers
    used_driver_ids = {r['driver_id'] for r in merged_routes}

    return merged_routes, used_driver_ids

def split_high_detour_routes(routes, all_drivers, office_pos, road_network):
    """Split routes with high detour ratios into more efficient sub-routes"""
    print("  ✂️ Phase 2: Splitting high-detour routes...")

    optimized_routes = []
    # Get a list of drivers not currently assigned to any route in the 'routes' list
    current_route_driver_ids = {r['driver_id'] for r in routes}
    available_drivers_for_split = [d for d in all_drivers if str(d['id']) not in current_route_driver_ids]

    splits_made = 0

    for route in routes:
        # Skip routes with ≤2 users (can't split effectively)
        if len(route['assigned_users']) <= 2:
            optimized_routes.append(route)
            continue

        # Calculate detour ratio
        detour_ratio = calculate_route_detour_ratio(route, office_pos, road_network)

        if detour_ratio > 1.4 and len(route['assigned_users']) >= 3 and available_drivers_for_split:
            # Split the route
            split_routes, newly_assigned_drivers = split_route_into_clusters(route, available_drivers_for_split, office_pos, road_network)

            if len(split_routes) > 1:
                optimized_routes.extend(split_routes)
                # Remove the drivers used for splitting from the available list
                for assigned_driver_id in newly_assigned_drivers:
                    available_drivers_for_split = [d for d in available_drivers_for_split if str(d['id']) != assigned_driver_id]
                splits_made += 1
            else:
                optimized_routes.append(route)
        else:
            optimized_routes.append(route)

    print(f"    ✂️ Completed {splits_made} route splits")

    # Update used_driver_ids to reflect newly created routes from splits
    newly_used_driver_ids = {r['driver_id'] for r in optimized_routes}
    return optimized_routes, newly_used_driver_ids

def local_route_optimization(routes, unassigned_users, office_pos, road_network):
    """Local optimization: improve each route individually"""
    print("  🔧 Phase 3: Local route optimization...")

    improvements_made = 0

    for route in routes:
        # Re-optimize pickup sequence
        if len(route['assigned_users']) > 1:
            route['assigned_users'].sort(key=lambda u: u.get('office_distance', 0))

        # Update route metrics
        update_route_metrics_improved(route, office_pos[0], office_pos[1])
        improvements_made += 1

    print(f"    🔧 Local optimization: {improvements_made} improvements made")
    return routes

def global_optimization_pass(routes, office_pos, road_network):
    """Quality-focused global optimization with road awareness"""
    print("  🌍 Phase 4: Quality-focused global optimization...")

    improvements_made = 0
    quality_analyzer = RoadCorridorAnalyzer(road_network, office_pos) if road_network else None

    # Phase 1: Route sequence optimization with road awareness
    for route in routes:
        if len(route['assigned_users']) > 1:
            original_distance = route.get('total_distance', 0)

            if road_network:
                # Road-aware sequence optimization
                driver_pos = (route['latitude'], route['longitude'])
                user_positions = [(u['lat'], u['lng']) for u in route['assigned_users']]

                optimal_sequence = road_network.get_optimal_pickup_sequence(
                    driver_pos, user_positions, office_pos
                )

                # Reorder users according to optimal sequence
                original_users = route['assigned_users'].copy()
                route['assigned_users'] = [original_users[i] for i in optimal_sequence]
            else:
                # Distance-based ordering as fallback
                route['assigned_users'].sort(key=lambda u: u.get('office_distance', 0))

            update_route_metrics_improved(route, office_pos[0], office_pos[1])

            new_distance = route.get('total_distance', 0)
            if new_distance < original_distance:
                improvements_made += 1
                print(f"    ✅ Route {route['driver_id']}: Improved by {original_distance - new_distance:.2f}km")

    # Phase 2: Quality-preserving user swaps between routes
    if road_network:
        swap_improvements = quality_preserving_route_swaps(routes, office_pos, road_network)
        improvements_made += swap_improvements

    # Phase 3: Route coherence validation and correction
    for route in routes:
        if road_network and len(route['assigned_users']) > 1:
            driver_pos = (route['latitude'], route['longitude'])
            user_positions = [(u['lat'], u['lng']) for u in route['assigned_users']]

            coherence = road_network.get_route_coherence_score(driver_pos, user_positions, office_pos)

            if coherence < 0.6:  # Low coherence route needs attention
                print(f"    ⚠️ Route {route['driver_id']} has low coherence: {coherence:.3f}")
                # Could implement route splitting here if needed

    print(f"    🌍 Global optimization: {improvements_made} quality improvements made")
    return routes

def calculate_route_detour_ratio(route, office_pos, road_network):
    """Calculate the detour ratio for a route"""
    if not route or not route.get('assigned_users'):
        return 1.0

    driver_pos = (route['latitude'], route['longitude'])

    # Calculate actual route distance
    actual_distance = calculate_route_total_distance(route, office_pos, road_network)

    # Calculate direct distance (driver to office)
    direct_distance = haversine_distance(driver_pos[0], driver_pos[1], office_pos[0], office_pos[1])

    return actual_distance / max(direct_distance, 0.1)

def calculate_route_total_distance(route, office_pos, road_network):
    """Calculate total distance for a route"""
    if not route or not route.get('assigned_users'):
        return 0.0

    total_distance = 0.0
    current_pos = (route['latitude'], route['longitude'])

    # Distance through all users
    for user in route['assigned_users']:
        user_pos = (user['lat'], user['lng'])
        distance = haversine_distance(current_pos[0], current_pos[1], user_pos[0], user_pos[1])
        total_distance += distance
        current_pos = user_pos

    # Distance from last user to office
    final_distance = haversine_distance(current_pos[0], current_pos[1], office_pos[0], office_pos[1])
    total_distance += final_distance

    return total_distance

def split_route_into_clusters(route, available_drivers, office_pos, road_network):
    """Split a route into 2 clusters and assign a new driver if available"""
    if len(route['assigned_users']) < 3 or not available_drivers:
        return [route], [] # Return original route and no new drivers assigned

    users = route['assigned_users']

    # Simple clustering: split by distance from driver
    mid_point = len(users) // 2
    cluster_1_users = users[:mid_point]
    cluster_2_users = users[mid_point:]

    # Create route for cluster 1 (keep original driver)
    route_1 = route.copy()
    route_1['assigned_users'] = cluster_1_users
    update_route_metrics_improved(route_1, office_pos[0], office_pos[1])

    assigned_driver_ids = []

    # If we have available drivers and cluster 2 has users, create new route
    if available_drivers and cluster_2_users:
        # Take the first available driver
        best_driver_for_split = None
        min_driver_dist = float('inf')

        # Find the closest available driver to the route's center for the new route
        route_center = calculate_route_center(route)
        for driver in available_drivers:
            driver_pos = (driver['latitude'], driver['longitude'])
            dist = haversine_distance(route_center[0], route_center[1], driver_pos[0], driver_pos[1])
            if dist < min_driver_dist:
                min_driver_dist = dist
                best_driver_for_split = driver

        if best_driver_for_split:
            # IMMEDIATELY mark driver as used to prevent duplicates
            assigned_driver_ids.append(str(best_driver_for_split['id']))

            route_2 = {
                'driver_id': str(best_driver_for_split['id']),
                'vehicle_id': str(best_driver_for_split.get('vehicle_id', '')),
                'vehicle_type': int(best_driver_for_split['capacity']),
                'latitude': float(best_driver_for_split['latitude']),
                'longitude': float(best_driver_for_split['longitude']),
                'assigned_users': cluster_2_users
            }

            update_route_metrics_improved(route_2, office_pos[0], office_pos[1])

            return [route_1, route_2], assigned_driver_ids

    # If splitting failed (e.g., no available drivers for cluster 2), return original route
    return [route], []

def quality_preserving_route_swaps(routes, office_pos, road_network):
    """Perform user swaps between routes while maintaining high quality standards."""

    swap_improvements = 0
    max_swap_attempts = 50  # Limit to prevent excessive computation

    print("    🔄 Attempting quality-preserving route swaps...")

    for attempt in range(max_swap_attempts):
        improvement_found = False

        # Sort routes by driver_id for deterministic processing
        sorted_routes = sorted(routes, key=lambda r: r['driver_id'])

        for i, route_a in enumerate(sorted_routes):
            if len(route_a['assigned_users']) == 0:
                continue

            for j, route_b in enumerate(sorted_routes[i+1:], i+1):
                if len(route_b['assigned_users']) == 0:
                    continue

                # Try swapping users between routes
                swap_result = try_quality_swap(route_a, route_b, office_pos, road_network)

                if swap_result['improved']:
                    swap_improvements += 1
                    improvement_found = True
                    print(f"      ✅ Quality swap between routes {route_a['driver_id']} and {route_b['driver_id']}")
                    break

            if improvement_found:
                break

        if not improvement_found:
            break  # No more improvements possible

    return swap_improvements

def try_quality_swap(route_a, route_b, office_pos, road_network):
    """Try swapping users between two routes to improve overall quality."""

    result = {'improved': False, 'swap_type': None}

    # Calculate current quality scores
    driver_a_pos = (route_a['latitude'], route_a['longitude'])
    driver_b_pos = (route_b['latitude'], route_b['longitude'])

    users_a = [(u['lat'], u['lng']) for u in route_a['assigned_users']]
    users_b = [(u['lat'], u['lng']) for u in route_b['assigned_users']]

    if not road_network:
        return result

    current_score_a = road_network.get_route_coherence_score(driver_a_pos, users_a, office_pos)
    current_score_b = road_network.get_route_coherence_score(driver_b_pos, users_b, office_pos)
    current_total_score = current_score_a + current_score_b

    # Try different swap scenarios
    best_improvement = 0
    best_swap = None

    # 1. Try single user swaps
    for i, user_a in enumerate(route_a['assigned_users']):
        for j, user_b in enumerate(route_b['assigned_users']):

            # Check if swap maintains capacity constraints
            if (len(route_a['assigned_users']) <= route_a['vehicle_type'] and 
                len(route_b['assigned_users']) <= route_b['vehicle_type']):

                # Create test configurations
                test_users_a = users_a.copy()
                test_users_b = users_b.copy()

                test_users_a[i] = (user_b['lat'], user_b['lng'])
                test_users_b[j] = (user_a['lat'], user_a['lng'])

                # Calculate new scores
                new_score_a = road_network.get_route_coherence_score(driver_a_pos, test_users_a, office_pos)
                new_score_b = road_network.get_route_coherence_score(driver_b_pos, test_users_b, office_pos)
                new_total_score = new_score_a + new_score_b

                improvement = new_total_score - current_total_score

                # Only accept swaps that significantly improve quality
                if improvement > 0.05 and improvement > best_improvement:
                    # Additional quality checks
                    if (new_score_a >= 0.6 and new_score_b >= 0.6 and  # Maintain minimum quality
                        not would_create_zigzag(driver_a_pos, test_users_a, office_pos) and
                        not would_create_zigzag(driver_b_pos, test_users_b, office_pos)):

                        best_improvement = improvement
                        best_swap = {
                            'type': 'single_swap',
                            'user_a_idx': i,
                            'user_b_idx': j,
                            'improvement': improvement
                        }

    # Execute best swap if found
    if best_swap and best_improvement > 0.05:
        if best_swap['type'] == 'single_swap':
            # Perform the swap
            user_a = route_a['assigned_users'][best_swap['user_a_idx']]
            user_b = route_b['assigned_users'][best_swap['user_b_idx']]

            route_a['assigned_users'][best_swap['user_a_idx']] = user_b
            route_b['assigned_users'][best_swap['user_b_idx']] = user_a

            # Update route metrics
            update_route_metrics_improved(route_a, office_pos[0], office_pos[1])
            update_route_metrics_improved(route_b, office_pos[0], office_pos[1])

            result['improved'] = True
            result['swap_type'] = 'single_swap'

    return result

def would_create_zigzag(driver_pos, user_positions, office_pos):
    """Check if a route configuration would create excessive zigzag pattern."""

    if len(user_positions) < 2:
        return False

    route_points = [driver_pos] + user_positions + [office_pos]

    total_turning = 0
    turning_count = 0

    for i in range(1, len(route_points) - 1):
        bearing1 = calculate_bearing(
            route_points[i-1][0], route_points[i-1][1],
            route_points[i][0], route_points[i][1]
        )
        bearing2 = calculate_bearing(
            route_points[i][0], route_points[i][1],
            route_points[i+1][0], route_points[i+1][1]
        )

        turning_angle = abs(normalize_bearing_difference(bearing2 - bearing1))
        total_turning += turning_angle
        turning_count += 1

    avg_turning = total_turning / max(1, turning_count)
    return avg_turning > 20.0  # More than 20 degrees average turning is considered zigzag

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

def main():
    """Test the road-aware assignment with sample data"""
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