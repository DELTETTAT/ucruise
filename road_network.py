import networkx as nx
import numpy as np
from functools import lru_cache
from typing import List, Tuple, Optional
import math


class RoadNetwork:

    def __init__(self, graphml_path: str):
        """Initialize road network from GraphML file"""
        try:
            self.graph = nx.read_graphml(graphml_path)
            # Convert to undirected if needed
            if self.graph.is_directed():
                self.graph = self.graph.to_undirected()

            # Store node positions for quick access
            self.node_positions = {}
            for node, data in self.graph.nodes(data=True):
                if 'x' in data and 'y' in data:
                    self.node_positions[node] = (float(data['y']), float(data['x']))  # lat, lon
                elif 'lat' in data and 'lon' in data:
                    self.node_positions[node] = (float(data['lat']), float(data['lon']))

            print(f"Road network loaded: {len(self.graph.nodes)} nodes, {len(self.graph.edges)} edges")

        except Exception as e:
            raise Exception(f"Failed to load road network: {e}")

    def find_shortest_path(self, start_pos, end_pos):
        """Find shortest path between two geographic positions"""
        try:
            # Find nearest nodes to start and end positions
            start_node = self.find_nearest_road_node(start_pos[0], start_pos[1])
            end_node = self.find_nearest_road_node(end_pos[0], end_pos[1])

            if start_node is None or end_node is None:
                return None

            # Use NetworkX to find shortest path
            path = nx.shortest_path(self.graph, start_node, end_node, weight='weight')
            return path

        except Exception as e:
            print(f"Path finding error: {e}")
            return None

    def find_directional_path(self, start_pos, end_pos, main_direction_bias=0.8):
        """
        Find path that strongly considers adherence to main direction.
        Uses modified Dijkstra with strong directional bias toward main route (driver → office).
        """
        try:
            start_node = self.find_nearest_road_node(start_pos[0], start_pos[1])
            end_node = self.find_nearest_road_node(end_pos[0], end_pos[1])

            if start_node is None or end_node is None:
                return None

            # Calculate main direction bearing
            main_bearing = self._calculate_bearing(start_pos[0], start_pos[1], end_pos[0], end_pos[1])

            # Modified edge weights considering strong directional bias
            def directional_weight(u, v, data):
                base_weight = data.get('weight', 1.0)
                
                # Get positions of nodes
                if u in self.node_positions and v in self.node_positions:
                    u_pos = self.node_positions[u]
                    v_pos = self.node_positions[v]
                    
                    # Calculate bearing of this edge
                    edge_bearing = self._calculate_bearing(u_pos[0], u_pos[1], v_pos[0], v_pos[1])
                    
                    # Calculate deviation from main direction
                    bearing_diff = abs(self._normalize_bearing_difference(edge_bearing - main_bearing))
                    
                    # Strong penalty for edges deviating > 30° from main direction
                    if bearing_diff > 30:
                        direction_penalty = 2.0 + (bearing_diff / 60.0)  # 2x-5x penalty for side roads
                    else:
                        direction_penalty = 1.0 + (main_direction_bias * (bearing_diff / 180.0))
                    
                    return base_weight * direction_penalty
                
                return base_weight

            # Use Dijkstra with modified weights
            path = nx.shortest_path(self.graph, start_node, end_node, weight=directional_weight)
            return path

        except Exception as e:
            print(f"Directional path finding error: {e}")
            return self.find_shortest_path(start_pos, end_pos)  # Fallback to regular path

    def _normalize_bearing_difference(self, diff):
        """Normalize bearing difference to [-180, 180] range"""
        while diff > 180:
            diff -= 360
        while diff < -180:
            diff += 360
        return diff

    def _prepare_network(self):
        """Prepare the network for routing calculations"""
        # Convert node positions to float and create spatial index
        self.node_positions = {}
        for node_id, data in self.graph.nodes(data=True):
            try:
                lat = float(data.get('d4', data.get('y', 0)))
                lon = float(data.get('d5', data.get('x', 0)))
                self.node_positions[node_id] = (lat, lon)
            except (ValueError, TypeError):
                continue

        # Add edge weights based on distance if not present
        for u, v, data in self.graph.edges(data=True):
            if 'weight' not in data:
                if u in self.node_positions and v in self.node_positions:
                    pos1 = self.node_positions[u]
                    pos2 = self.node_positions[v]
                    distance = self._haversine_distance(
                        pos1[0], pos1[1], pos2[0], pos2[1])
                    data['weight'] = distance

    @staticmethod
    def _haversine_distance(lat1: float, lon1: float, lat2: float,
                            lon2: float) -> float:
        """Calculate haversine distance between two points"""
        R = 6371.0  # Earth radius in kilometers
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(
            dlon / 2)**2
        c = 2 * math.asin(math.sqrt(a))
        return R * c

    def find_nearest_road_node(self, lat: float, lon: float) -> Optional[str]:
        """Find the nearest road network node to given coordinates"""
        if not self.node_positions:
            return None

        min_distance = float('inf')
        nearest_node = None

        for node_id, (node_lat, node_lon) in self.node_positions.items():
            distance = self._haversine_distance(lat, lon, node_lat, node_lon)
            if distance < min_distance:
                min_distance = distance
                nearest_node = node_id

        return nearest_node if min_distance < 1.0 else None  # Max 1km to nearest road

    @lru_cache(maxsize=1000)
    def get_road_distance(self, lat1: float, lon1: float, lat2: float,
                          lon2: float) -> float:
        """Get road network distance between two points"""
        try:
            node1 = self.find_nearest_road_node(lat1, lon1)
            node2 = self.find_nearest_road_node(lat2, lon2)

            if not node1 or not node2:
                # Fallback to straight-line distance if no road nodes found
                return self._haversine_distance(lat1, lon1, lat2, lon2)

            if node1 == node2:
                return self._haversine_distance(lat1, lon1, lat2, lon2)

            # Calculate shortest path on road network
            try:
                path_length = nx.shortest_path_length(self.graph,
                                                      node1,
                                                      node2,
                                                      weight='weight')

                # Add distance from points to nearest road nodes
                dist_to_road1 = self._haversine_distance(
                    lat1, lon1, *self.node_positions[node1])
                dist_to_road2 = self._haversine_distance(
                    lat2, lon2, *self.node_positions[node2])

                return path_length + dist_to_road1 + dist_to_road2

            except nx.NetworkXNoPath:
                # No path found, use straight-line distance
                return self._haversine_distance(lat1, lon1, lat2, lon2)

        except Exception:
            # Any error, fallback to straight-line distance
            return self._haversine_distance(lat1, lon1, lat2, lon2)

    def get_route_coherence_score(self, driver_pos: Tuple[float, float],
                                  user_positions: List[Tuple[float, float]],
                                  office_pos: Tuple[float, float]) -> float:
        """
        Calculate route coherence score based on road network.
        Higher score means more coherent route along roads.
        """
        if not user_positions:
            return 1.0

        total_road_distance = 0.0
        total_straight_distance = 0.0

        # Calculate route: driver -> users -> office
        route_points = [driver_pos] + user_positions + [office_pos]

        for i in range(len(route_points) - 1):
            lat1, lon1 = route_points[i]
            lat2, lon2 = route_points[i + 1]

            road_dist = self.get_road_distance(lat1, lon1, lat2, lon2)
            straight_dist = self._haversine_distance(lat1, lon1, lat2, lon2)

            total_road_distance += road_dist
            total_straight_distance += straight_dist

        # Base coherence score: closer to 1.0 means route follows roads well
        if total_straight_distance == 0:
            return 1.0

        base_coherence = total_straight_distance / total_road_distance

        # Apply directional penalty for zig-zag routes
        directional_penalty = self._calculate_directional_penalty(route_points)

        # Apply backtracking penalty
        backtrack_penalty = self._calculate_backtrack_penalty(
            route_points, office_pos)

        # Enhanced bonus for corridor-like routes
        corridor_bonus = 0.0
        if directional_penalty < 0.15:  # Very few direction changes
            road_efficiency = total_straight_distance / total_road_distance
            if road_efficiency > 0.85:  # Road distance very close to straight line
                corridor_bonus = 0.2  # 20% bonus for true corridor routes
            elif road_efficiency > 0.75:
                corridor_bonus = 0.1  # 10% bonus for good corridor routes

        # Penalty for non-sequential users (office distance not decreasing)
        sequence_penalty = 0.0
        if len(user_positions) > 1:
            office_distances = [self._haversine_distance(pos[0], pos[1], office_pos[0], office_pos[1]) 
                              for pos in user_positions]
            # Check if distances are roughly decreasing (allowing some tolerance)
            for i in range(len(office_distances) - 1):
                if office_distances[i] < office_distances[i + 1] - 0.5:  # 500m tolerance
                    sequence_penalty += 0.1

        # Final score with enhanced corridor detection
        final_score = (base_coherence * (1.0 - directional_penalty) * (1.0 - backtrack_penalty) 
                      + corridor_bonus - sequence_penalty)
        return min(1.0, max(0.0, final_score))

    def is_user_on_route_path(self,
                              driver_pos: Tuple[float, float],
                              existing_users: List[Tuple[float, float]],
                              candidate_user: Tuple[float, float],
                              office_pos: Tuple[float, float],
                              max_detour_ratio: float = 1.2,
                              route_type: str = "balanced") -> bool:
        """
        Check if adding a candidate user creates a reasonable detour
        based on road network distances with route-type-specific validation.
        """
        # Apply route-type-specific detour ratios
        if route_type == "straight":
            max_detour_ratio = min(max_detour_ratio, 1.15)  # 10-15% max detour
            max_path_deviation = 0.2  # Very strict path deviation
        elif route_type == "balanced":
            max_detour_ratio = min(max_detour_ratio, 1.25)  # 20-25% max detour
            max_path_deviation = 0.3  # Moderate path deviation
        elif route_type == "capacity":
            max_detour_ratio = min(max_detour_ratio, 1.4)   # 40% max detour (more lenient)
            max_path_deviation = 0.5  # Allow more deviation for capacity optimization
        else:
            max_path_deviation = 0.3  # Default moderate

        if not existing_users:
            # First user - use directional path finding for better validation
            direct_path = self.find_directional_path(driver_pos, office_pos)
            if direct_path:
                direct_distance = self._calculate_path_distance_from_nodes(direct_path)
            else:
                direct_distance = self.get_road_distance(*driver_pos, *office_pos)
            
            # Calculate distance via user using directional paths
            user_path1 = self.find_directional_path(driver_pos, candidate_user)
            user_path2 = self.find_directional_path(candidate_user, office_pos)
            
            if user_path1 and user_path2:
                via_user_distance = (
                    self._calculate_path_distance_from_nodes(user_path1) +
                    self._calculate_path_distance_from_nodes(user_path2)
                )
            else:
                via_user_distance = (
                    self.get_road_distance(*driver_pos, *candidate_user) +
                    self.get_road_distance(*candidate_user, *office_pos)
                )

            detour_ratio = via_user_distance / max(direct_distance, 0.1)

            # Strict path deviation check
            if self._point_to_line_distance(
                    candidate_user, driver_pos,
                    office_pos) > self._haversine_distance(
                        *driver_pos, *office_pos) * max_path_deviation:
                return False

            return detour_ratio <= max_detour_ratio

        # For existing routes, be more strict about coherence
        current_route = [driver_pos] + existing_users + [office_pos]
        current_distance = 0.0
        for i in range(len(current_route) - 1):
            current_distance += self.get_road_distance(*current_route[i],
                                                       *current_route[i + 1])

        # Calculate new route distance with candidate user
        # Try inserting user at best position
        best_distance = float('inf')
        best_position = -1

        for insert_pos in range(1, len(current_route)):
            new_route = current_route.copy()
            new_route.insert(insert_pos, candidate_user)

            new_distance = 0.0
            for i in range(len(new_route) - 1):
                new_distance += self.get_road_distance(*new_route[i],
                                                       *new_route[i + 1])

            if new_distance < best_distance:
                best_distance = new_distance
                best_position = insert_pos

        # Check if detour is reasonable
        detour_ratio = best_distance / max(current_distance, 0.1)

        # Additional coherence check - ensure the new route makes sense directionally
        if best_position > 0:
            # Check if the insertion creates logical geographical progression
            prev_pos = current_route[best_position - 1]
            next_pos = current_route[best_position] if best_position < len(
                current_route) else office_pos

            # Calculate bearings to ensure logical progression
            bearing_to_candidate = self._calculate_bearing(
                *prev_pos, *candidate_user)
            bearing_to_next = self._calculate_bearing(*candidate_user,
                                                      *next_pos)
            bearing_direct = self._calculate_bearing(*prev_pos, *next_pos)

            # Check if the path makes geographical sense
            bearing_diff = abs(bearing_to_candidate - bearing_direct)
            if bearing_diff > 180:
                bearing_diff = 360 - bearing_diff

            # Reject if requires too much directional change (>45 degrees)
            if bearing_diff > 45:
                return False

        return detour_ratio <= max_detour_ratio

    def _point_to_line_distance(self, point: Tuple[float, float],
                                line_start: Tuple[float, float],
                                line_end: Tuple[float, float]) -> float:
        """Calculate perpendicular distance from point to line segment"""
        import math

        # Convert to approximate Cartesian coordinates for distance calculation
        x0, y0 = point
        x1, y1 = line_start
        x2, y2 = line_end

        # Convert lat/lng to meters (approximate)
        lat_to_m = 111320.0  # meters per degree latitude
        lng_to_m = 111320.0 * math.cos(math.radians(
            point[0]))  # longitude varies by latitude

        x0_m, y0_m = x0 * lat_to_m, y0 * lng_to_m
        x1_m, y1_m = x1 * lat_to_m, y1 * lng_to_m
        x2_m, y2_m = x2 * lat_to_m, y2 * lng_to_m

        # Calculate perpendicular distance
        A = x0_m - x1_m
        B = y0_m - y1_m
        C = x2_m - x1_m
        D = y2_m - y1_m

        dot = A * C + B * D
        len_sq = C * C + D * D

        if len_sq == 0:
            # Line start and end are the same point
            return math.sqrt(A * A + B * B) / 1000.0  # Convert to km

        param = dot / len_sq

        if param < 0:
            xx, yy = x1_m, y1_m
        elif param > 1:
            xx, yy = x2_m, y2_m
        else:
            xx = x1_m + param * C
            yy = y1_m + param * D

        dx = x0_m - xx
        dy = y0_m - yy
        return math.sqrt(dx * dx + dy * dy) / 1000.0  # Convert to km

    def _calculate_bearing(self, lat1: float, lon1: float, lat2: float,
                           lon2: float) -> float:
        """Calculate bearing between two points"""
        import math
        lat1, lat2 = map(math.radians, [lat1, lat2])
        dlon = math.radians(lon2 - lon1)
        x = math.sin(dlon) * math.cos(lat2)
        y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(
            lat2) * math.cos(dlon)
        return (math.degrees(math.atan2(x, y)) + 360) % 360

    def _calculate_directional_penalty(
            self, route_points: List[Tuple[float, float]]) -> float:
        """Calculate penalty for routes with excessive directional changes (zig-zag)"""
        if len(route_points) < 3:
            return 0.0

        bearings = []
        for i in range(len(route_points) - 1):
            bearing = self._calculate_bearing(*route_points[i],
                                              *route_points[i + 1])
            bearings.append(bearing)

        if len(bearings) < 2:
            return 0.0

        # Calculate bearing changes
        bearing_changes = []
        for i in range(len(bearings) - 1):
            change = abs(bearings[i + 1] - bearings[i])
            if change > 180:
                change = 360 - change
            bearing_changes.append(change)

        # Heavy penalty for sharp directional changes (zig-zag pattern)
        total_penalty = 0.0
        for change in bearing_changes:
            if change > 45:  # Sharp turn
                total_penalty += (change - 45) / 135  # Normalize to 0-1 scale

        # Normalize by number of segments
        return min(1.0, total_penalty / max(1, len(bearing_changes)))

    def _calculate_backtrack_penalty(self, route_points: List[Tuple[float,
                                                                    float]],
                                     office_pos: Tuple[float, float]) -> float:
        """Calculate penalty for routes that backtrack away from office"""
        if len(route_points) < 2:
            return 0.0

        driver_pos = route_points[0]
        penalty = 0.0

        # Calculate initial distance to office
        prev_distance_to_office = self._haversine_distance(
            *driver_pos, *office_pos)

        for i in range(1,
                       len(route_points) - 1):  # Exclude final office position
            current_pos = route_points[i]
            current_distance_to_office = self._haversine_distance(
                *current_pos, *office_pos)

            # Penalty if we're moving further from office
            if current_distance_to_office > prev_distance_to_office:
                backtrack_amount = current_distance_to_office - prev_distance_to_office
                penalty += min(0.5, backtrack_amount /
                               10.0)  # Max 0.5 penalty per point

            prev_distance_to_office = current_distance_to_office

        return min(1.0, penalty)

    def _calculate_path_distance_from_nodes(self, path_nodes):
        """Calculate total distance for a path given as node list"""
        if not path_nodes or len(path_nodes) < 2:
            return 0.0
        
        total_distance = 0.0
        for i in range(len(path_nodes) - 1):
            if path_nodes[i] in self.node_positions and path_nodes[i+1] in self.node_positions:
                pos1 = self.node_positions[path_nodes[i]]
                pos2 = self.node_positions[path_nodes[i+1]]
                total_distance += self._haversine_distance(pos1[0], pos1[1], pos2[0], pos2[1])
        
        return total_distance

    def get_optimal_pickup_sequence(
            self, driver_pos: Tuple[float,
                                    float], user_positions: List[Tuple[float,
                                                                       float]],
            office_pos: Tuple[float, float]) -> List[int]:
        """Find optimal pickup sequence using improved heuristics with 2-opt optimization"""
        if len(user_positions) <= 1:
            return list(range(len(user_positions)))

        # Step 1: Greedy nearest neighbor construction
        remaining_users = list(range(len(user_positions)))
        pickup_sequence = []
        current_pos = driver_pos

        while remaining_users:
            best_user_idx = None
            best_score = float('inf')

            for user_idx in remaining_users:
                user_pos = user_positions[user_idx]

                # Enhanced scoring with directional bias
                distance_score = self.get_road_distance(*current_pos, *user_pos)

                # Progress toward office (stronger bias)
                current_to_office = self._haversine_distance(*current_pos, *office_pos)
                user_to_office = self._haversine_distance(*user_pos, *office_pos)
                progress_score = max(0, current_to_office - user_to_office) * -3  # Increased bias

                # Directional alignment bonus
                if len(pickup_sequence) == 0:  # First user
                    main_bearing = self._calculate_bearing(*driver_pos, *office_pos)
                    user_bearing = self._calculate_bearing(*current_pos, *user_pos)
                    bearing_diff = abs(self._normalize_bearing_difference(user_bearing - main_bearing))
                    direction_penalty = (bearing_diff / 180.0) * 2.0  # Penalty for off-direction
                else:
                    direction_penalty = 0

                total_score = distance_score + progress_score + direction_penalty

                if total_score < best_score:
                    best_score = total_score
                    best_user_idx = user_idx

            if best_user_idx is not None:
                pickup_sequence.append(best_user_idx)
                remaining_users.remove(best_user_idx)
                current_pos = user_positions[best_user_idx]

        # Step 2: Apply 2-opt improvement
        pickup_sequence = self._apply_2opt_improvement(pickup_sequence, user_positions, driver_pos, office_pos)

        return pickup_sequence

    def _apply_2opt_improvement(self, sequence, user_positions, driver_pos, office_pos, max_iterations=10):
        """Apply 2-opt local search to improve the pickup sequence"""
        if len(sequence) <= 2:
            return sequence

        current_sequence = sequence.copy()
        improved = True
        iteration = 0

        while improved and iteration < max_iterations:
            improved = False
            iteration += 1
            
            current_distance = self._calculate_sequence_total_distance(
                current_sequence, user_positions, driver_pos, office_pos
            )

            for i in range(len(current_sequence) - 1):
                for j in range(i + 2, len(current_sequence)):
                    # Create new sequence by reversing the segment between i+1 and j
                    new_sequence = (current_sequence[:i+1] + 
                                   current_sequence[i+1:j+1][::-1] + 
                                   current_sequence[j+1:])

                    new_distance = self._calculate_sequence_total_distance(
                        new_sequence, user_positions, driver_pos, office_pos
                    )

                    if new_distance < current_distance:
                        current_sequence = new_sequence
                        current_distance = new_distance
                        improved = True
                        break
                
                if improved:
                    break

        return current_sequence

    def _calculate_sequence_total_distance(self, sequence, user_positions, driver_pos, office_pos):
        """Calculate total distance for a given pickup sequence"""
        total_distance = 0.0
        current_pos = driver_pos

        # Distance to pickup all users in sequence
        for user_idx in sequence:
            user_pos = user_positions[user_idx]
            total_distance += self.get_road_distance(*current_pos, *user_pos)
            current_pos = user_pos

        # Distance from last user to office
        total_distance += self.get_road_distance(*current_pos, *office_pos)

        return total_distance