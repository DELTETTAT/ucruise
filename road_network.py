import networkx as nx
import numpy as np
from functools import lru_cache
from typing import List, Tuple, Optional
import math


import math
import networkx as nx
from functools import lru_cache
from typing import List, Tuple
import os

class RoadNetwork:

    def __init__(self, graphml_path: str):
        """Initialize road network from GraphML file"""
        try:
            self.graph = nx.read_graphml(graphml_path)
            
            # Convert to undirected if needed
            if self.graph.is_directed():
                self.graph = self.graph.to_undirected()
            
            # Convert MultiGraph to simple Graph to avoid edge key issues
            if self.graph.is_multigraph():
                print("🔄 Converting MultiGraph to simple Graph...")
                simple_graph = nx.Graph()
                simple_graph.add_nodes_from(self.graph.nodes(data=True))
                
                # Add edges, keeping only the first edge if multiple exist between same nodes
                for u, v, data in self.graph.edges(data=True):
                    if not simple_graph.has_edge(u, v):
                        simple_graph.add_edge(u, v, **data)
                
                self.graph = simple_graph
                print("✅ MultiGraph converted to simple Graph")

            print(f"✅ Road network loaded: {len(self.graph.nodes)} nodes, {self.graph.number_of_edges()} edges")

            # Build node positions dictionary for faster lookups
            self.node_positions = {}
            for node, data in self.graph.nodes(data=True):
                try:
                    # Try different coordinate attribute names
                    lat = data.get('lat') or data.get('y') or data.get('d4')
                    lon = data.get('lon') or data.get('x') or data.get('d5')

                    if lat is not None and lon is not None:
                        self.node_positions[node] = (float(lat), float(lon))
                except (ValueError, TypeError):
                    continue

            print(f"✅ Node positions extracted: {len(self.node_positions)} nodes with coordinates")

            # Initialize edge weights and spatial index
            self._prepare_network()

            # Add caching for expensive computations
            self._path_length_cache = {}  # Cache for (node1, node2) -> distance
            self._node_cache = {}  # Cache for (lat, lon) -> (node, distance)

        except Exception as e:
            raise Exception(f"Failed to load road network: {e}")

    def find_shortest_path(self, start_pos, end_pos):
        """Find shortest path between two geographic positions"""
        try:
            # Find nearest nodes to start and end positions
            start_node, _ = self.find_nearest_road_node(start_pos[0], start_pos[1])
            end_node, _ = self.find_nearest_road_node(end_pos[0], end_pos[1])

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
            start_node, _ = self.find_nearest_road_node(start_pos[0], start_pos[1])
            end_node, _ = self.find_nearest_road_node(end_pos[0], end_pos[1])

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
        """Prepare the network by adding weights and building spatial index"""
        print("🔧 Preparing road network indices...")

        # Add distance-based weights to edges
        edge_weights = {}
        for u, v, data in self.graph.edges(data=True):
            if 'weight' not in data:
                if u in self.node_positions and v in self.node_positions:
                    node1 = self.node_positions[u]
                    node2 = self.node_positions[v]
                    distance = self._calculate_distance(node1[0], node1[1], node2[0], node2[1])
                    edge_weights[(u, v)] = distance
                else:
                    # Handle cases where node positions might be missing for an edge
                    edge_weights[(u, v)] = 1.0  # Default weight
        
        # Set edge weights using NetworkX method compatible with all versions
        for (u, v), weight in edge_weights.items():
            self.graph.edges[u, v]['weight'] = weight

        print(f"✅ Edge weights set for {self.graph.number_of_edges()} edges")

        # Build spatial index for fast nearest neighbor queries
        try:
            from scipy.spatial import cKDTree
            self._node_list = []
            self._node_coords = []

            for node_id, (lat, lon) in self.node_positions.items():
                self._node_list.append(node_id)
                self._node_coords.append((lat, lon))

            if self._node_coords:
                self._kdtree = cKDTree(self._node_coords)
                print(f"✅ Spatial index built with {len(self._node_coords)} nodes")
            else:
                self._kdtree = None
        except ImportError:
            print("⚠️ scipy not available, using linear search for nearest nodes")
            self._kdtree = None
        except Exception as e:
            print(f"⚠️ Failed to build spatial index: {e}")
            self._kdtree = None
        
        print(f"✅ Network preparation complete")


    @staticmethod
    def _calculate_distance(lat1: float, lon1: float, lat2: float,
                            lon2: float) -> float:
        """Calculate haversine distance between two points in km"""
        R = 6371.0  # Earth radius in kilometers
        lat1_rad, lon1_rad, lat2_rad, lon2_rad = map(math.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(
            dlon / 2)**2
        c = 2 * math.asin(math.sqrt(a))
        return R * c

    def find_nearest_road_node(self, lat, lon, max_search_km=2.0):
        """Find the nearest road network node to a given position
        Returns: (node_id, distance_km) or (None, float('inf')) if too far
        """
        if not hasattr(self, '_kdtree') or not self._kdtree or not self.node_positions:
            print("⚠️ Spatial index (KDTree) unavailable — find_nearest_road_node will always return None")
            return None, float('inf')

        # Query the KDTree for the nearest node
        distance, idx = self._kdtree.query([lat, lon])

        if idx < len(self._node_list):
            node_id = self._node_list[idx]
            # Approximate conversion from KDTree distance to km.
            # This assumes KDTree uses Euclidean distance on lat/lon, which is a rough approximation.
            # For more accuracy, one might need to transform coordinates or use a geographic KDTree.
            distance_km = distance * 111.0  # Rough conversion to km (1 degree lat/lon approx 111km)

            if distance_km > max_search_km:
                print(f"Warning: Nearest node {node_id} is {distance_km:.2f}km away (max: {max_search_km}km)")
                return None, distance_km

            return node_id, distance_km

        return None, float('inf')


    def get_road_distance(self, lat1: float, lon1: float, lat2: float,
                          lon2: float, max_distance_km=50):
        """Calculate road distance between two points using shortest path"""
        try:
            node1, dist1 = self.find_nearest_road_node(lat1, lon1)
            node2, dist2 = self.find_nearest_road_node(lat2, lon2)

            # Reject if either node is too far from road network or no nodes found
            if not node1 or not node2 or dist1 > 3.0 or dist2 > 3.0:
                # Fallback to haversine if no road path found or points are too far from roads
                return self._calculate_distance(lat1, lon1, lat2, lon2)

            if node1 == node2:
                # If start and end points map to the same road node, use haversine distance
                # or a direct road segment distance if available and more accurate.
                # For simplicity, we use haversine here.
                return self._calculate_distance(lat1, lon1, lat2, lon2)

            # Calculate shortest path on road network using cached function
            cache_key = (min(node1, node2), max(node1, node2))
            if cache_key in self._path_length_cache:
                path_length = self._path_length_cache[cache_key]
            else:
                try:
                    path_length = nx.shortest_path_length(self.graph,
                                                      node1,
                                                      node2,
                                                      weight='weight')
                    # Cache the result
                    self._path_length_cache[cache_key] = path_length
                    # Limit cache size to prevent memory issues
                    if len(self._path_length_cache) > 10000:
                        # Clear oldest entries (simple FIFO)
                        old_keys = list(self._path_length_cache.keys())[:5000]
                        for old_key in old_keys:
                            del self._path_length_cache[old_key]
                except nx.NetworkXNoPath:
                    # No path found, fallback to haversine
                    path_length = self._calculate_distance(lat1, lon1, lat2, lon2)
                    self._path_length_cache[cache_key] = path_length # Cache fallback result

            # Add distance from points to nearest road nodes
            dist_to_road1 = self._calculate_distance(
                lat1, lon1, *self.node_positions[node1])
            dist_to_road2 = self._calculate_distance(
                lat2, lon2, *self.node_positions[node2])

            return path_length + dist_to_road1 + dist_to_road2

        except Exception:
            # Any error, fallback to straight-line distance
            return self._calculate_distance(lat1, lon1, lat2, lon2)

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
            straight_dist = self._calculate_distance(lat1, lon1, lat2, lon2)

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

        # Enhanced bonus for corridor-like routes (more lenient)
        corridor_bonus = 0.0
        if directional_penalty < 0.25:  # More lenient direction changes
            road_efficiency = total_straight_distance / total_road_distance
            if road_efficiency > 0.75:  # More lenient road efficiency
                corridor_bonus = 0.2  # 20% bonus for good routes
            elif road_efficiency > 0.6:  # Even more lenient
                corridor_bonus = 0.1  # 10% bonus for acceptable routes

        # Much more lenient penalty for non-sequential users
        sequence_penalty = 0.0
        if len(user_positions) > 1:
            office_distances = [self._calculate_distance(pos[0], pos[1], office_pos[0], office_pos[1]) 
                              for pos in user_positions]
            # Check if distances are roughly decreasing (allowing much more tolerance)
            for i in range(len(office_distances) - 1):
                if office_distances[i] < office_distances[i + 1] - 2.0:  # 2km tolerance (was 0.5km)
                    sequence_penalty += 0.05  # Smaller penalty (was 0.1)

        # Final score with enhanced corridor detection
        final_score = (base_coherence * (1.0 - directional_penalty) * (1.0 - backtrack_penalty) 
                      + corridor_bonus - sequence_penalty)
        return min(1.0, max(0.0, final_score))

    def is_user_on_route_path(self,
                              driver_pos: Tuple[float, float],
                              existing_users: List[Tuple[float, float]],
                              candidate_pos: Tuple[float, float],
                              office_pos: Tuple[float, float],
                              max_detour_ratio=1.25,
                              route_type: str = "balanced") -> bool:
        """
        Check if a candidate user fits well along the existing route path.
        """
        try:
            # Get road distance from driver to candidate and candidate to office
            driver_to_candidate = self.get_road_distance(driver_pos[0], driver_pos[1], candidate_pos[0], candidate_pos[1])
            candidate_to_office = self.get_road_distance(candidate_pos[0], candidate_pos[1], office_pos[0], office_pos[1])

            # Get direct driver to office distance
            driver_to_office = self.get_road_distance(driver_pos[0], driver_pos[1], office_pos[0], office_pos[1])

            # Calculate detour ratio
            total_route_distance = driver_to_candidate + candidate_to_office
            detour_ratio = total_route_distance / max(driver_to_office, 0.1)

            # Tighter thresholds for compact routes
            if route_type == "straight":
                max_allowed_detour = 1.15  # Tighter
            elif route_type == "capacity":
                max_allowed_detour = 1.3   # Tighter
            else:  # balanced
                max_allowed_detour = 1.2   # Tighter

            # Primary detour check
            if detour_ratio <= max_allowed_detour:
                return True

            # Fallback: Check proximity to route path nodes (road-aware proximity)
            candidate_node, candidate_dist = self.find_nearest_road_node(candidate_pos[0], candidate_pos[1])
            if candidate_node and candidate_dist <= 1.0:  # Within 1km of road network
                # Check if candidate is near the path between driver and office
                driver_node, _ = self.find_nearest_road_node(driver_pos[0], driver_pos[1])
                office_node, _ = self.find_nearest_road_node(office_pos[0], office_pos[1])

                if driver_node and office_node:
                    try:
                        # Get the shortest path nodes
                        path = nx.shortest_path(self.graph, driver_node, office_node, weight='weight')

                        # Check if candidate is close to any node in the path
                        for path_node in path:
                            if path_node in self.node_positions:
                                path_pos = self.node_positions[path_node]
                                path_distance = self._calculate_distance(
                                    candidate_pos[0], candidate_pos[1], 
                                    path_pos[0], path_pos[1]
                                )
                                if path_distance <= 0.8:  # Within 800m of path node
                                    return True
                    except nx.NetworkXNoPath:
                        pass # No path, continue to next check
                    except Exception as e:
                        print(f"Error during path proximity check: {e}") # Log other errors

            return False

        except Exception as e:
            # More conservative fallback: check if candidate is within a reasonable distance
            # from the driver's position, as a last resort.
            straight_line_distance = self._calculate_distance(driver_pos[0], driver_pos[1], candidate_pos[0], candidate_pos[1])
            return straight_line_distance <= 3.0  # Within 3km (tighter)

    def _point_to_line_distance(self, point: Tuple[float, float],
                                line_start: Tuple[float, float],
                                line_end: Tuple[float, float]) -> float:
        """Calculate perpendicular distance from point to line segment in km"""
        import math

        # Convert to approximate Cartesian coordinates for distance calculation
        x0, y0 = point
        x1, y1 = line_start
        x2, y2 = line_end

        # Convert lat/lng to meters (approximate)
        lat_to_m = 111320.0  # meters per degree latitude
        # Use average latitude for longitude conversion for better approximation
        avg_lat = (y1 + y2) / 2.0 if line_start != line_end else y1
        lng_to_m = 111320.0 * math.cos(math.radians(avg_lat))

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
            # Closest point is line_start
            xx, yy = x1_m, y1_m
        elif param > 1:
            # Closest point is line_end
            xx, yy = x2_m, y2_m
        else:
            # Closest point is projection on the line segment
            xx = x1_m + param * C
            yy = y1_m + param * D

        dx = x0_m - xx
        dy = y0_m - yy
        return math.sqrt(dx * dx + dy * dy) / 1000.0  # Convert to km

    def _calculate_bearing(self, lat1: float, lon1: float, lat2: float,
                           lon2: float) -> float:
        """Calculate bearing between two points in degrees"""
        import math
        lat1_rad, lon1_rad, lat2_rad, lon2_rad = map(math.radians, [lat1, lon1, lat2, lon2])
        dlon = lon2_rad - lon1_rad
        x = math.sin(dlon) * math.cos(lat2_rad)
        y = math.cos(lat1_rad) * math.sin(lat2_rad) - math.sin(lat1_rad) * math.cos(
            lat2_rad) * math.cos(dlon)
        initial_bearing = math.atan2(x, y)
        # Normalize to 0-360 degrees
        return (math.degrees(initial_bearing) + 360) % 360

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
                total_penalty += (change - 45) / 135  # Normalize to 0-1 scale (max 180->135)

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

        # Calculate initial distance to office from driver's start position
        prev_distance_to_office = self._calculate_distance(
            *driver_pos, *office_pos)

        for i in range(1,
                       len(route_points) - 1):  # Exclude final office position
            current_pos = route_points[i]
            current_distance_to_office = self._calculate_distance(
                *current_pos, *office_pos)

            # Penalty if we're moving further from office
            if current_distance_to_office > prev_distance_to_office:
                backtrack_amount = current_distance_to_office - prev_distance_to_office
                # Apply a scaled penalty, capped to avoid excessive penalties
                penalty += min(0.5, backtrack_amount / 10.0)  # Max 0.5 penalty per point, scaled by distance

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
                total_distance += self._calculate_distance(pos1[0], pos1[1], pos2[0], pos2[1])
            else:
                # Handle cases where node positions might be missing
                print(f"Warning: Missing position for node in path calculation: {path_nodes[i]} or {path_nodes[i+1]}")
                # Could fallback to a default distance or skip
                pass

        return total_distance

    def get_optimal_pickup_sequence(
            self, driver_pos: Tuple[float,
                                    float], user_positions: List[Tuple[float,
                                                                       float]],
            office_pos: Tuple[float,
                              float]) -> List[int]:
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
                current_to_office = self._calculate_distance(*current_pos, *office_pos)
                user_to_office = self._calculate_distance(*user_pos, *office_pos)
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
            else:
                # Should not happen if remaining_users is not empty, but as a safeguard
                break

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
                for j in range(i + 1, len(current_sequence)): # Start j from i+1 for valid swaps
                    # Create new sequence by reversing the segment between i and j (inclusive)
                    # The segment to reverse is from index i to j (inclusive)
                    # new_sequence = current_sequence[:i] + current_sequence[i:j+1][::-1] + current_sequence[j+1:]
                    
                    # Correct 2-opt swap: reverse segment between i+1 and j
                    new_sequence = current_sequence[:i+1] + current_sequence[i+1:j+1][::-1] + current_sequence[j+1:]

                    new_distance = self._calculate_sequence_total_distance(
                        new_sequence, user_positions, driver_pos, office_pos
                    )

                    if new_distance < current_distance:
                        current_sequence = new_sequence
                        current_distance = new_distance
                        improved = True
                        # Break inner loop and restart outer loop to ensure best improvement per iteration
                        break  
                if improved:
                    break # Restart outer loop if improvement found

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