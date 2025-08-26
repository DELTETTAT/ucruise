
import networkx as nx
import numpy as np
from functools import lru_cache
from typing import List, Tuple, Optional
import math

class RoadNetwork:
    def __init__(self, graphml_path: str):
        """Load and initialize the road network from GraphML file"""
        self.graph = nx.read_graphml(graphml_path)
        self._prepare_network()
    
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
                    distance = self._haversine_distance(pos1[0], pos1[1], pos2[0], pos2[1])
                    data['weight'] = distance
    
    @staticmethod
    def _haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate haversine distance between two points"""
        R = 6371.0  # Earth radius in kilometers
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
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
    def get_road_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
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
                path_length = nx.shortest_path_length(
                    self.graph, node1, node2, weight='weight'
                )
                
                # Add distance from points to nearest road nodes
                dist_to_road1 = self._haversine_distance(
                    lat1, lon1, *self.node_positions[node1]
                )
                dist_to_road2 = self._haversine_distance(
                    lat2, lon2, *self.node_positions[node2]
                )
                
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
        backtrack_penalty = self._calculate_backtrack_penalty(route_points, office_pos)
        
        # Final score combines all factors
        final_score = base_coherence * (1.0 - directional_penalty) * (1.0 - backtrack_penalty)
        return min(1.0, max(0.0, final_score))
    
    def is_user_on_route_path(self, driver_pos: Tuple[float, float],
                             existing_users: List[Tuple[float, float]],
                             candidate_user: Tuple[float, float],
                             office_pos: Tuple[float, float],
                             max_detour_ratio: float = 1.2) -> bool:
        """
        Check if adding a candidate user creates a reasonable detour
        based on road network distances with enhanced validation.
        """
        if not existing_users:
            # First user - check if generally on path to office with stricter criteria
            direct_distance = self.get_road_distance(*driver_pos, *office_pos)
            via_user_distance = (
                self.get_road_distance(*driver_pos, *candidate_user) +
                self.get_road_distance(*candidate_user, *office_pos)
            )
            
            detour_ratio = via_user_distance / max(direct_distance, 0.1)
            
            # Additional check: ensure user is not too far from direct path
            straight_line_distance = self._haversine_distance(*driver_pos, *office_pos)
            user_to_path_distance = self._point_to_line_distance(
                candidate_user, driver_pos, office_pos
            )
            
            # Reject if user is too far off the main path
            if user_to_path_distance > straight_line_distance * 0.3:  # 30% of path length
                return False
                
            return detour_ratio <= max_detour_ratio
        
        # For existing routes, be more strict about coherence
        current_route = [driver_pos] + existing_users + [office_pos]
        current_distance = 0.0
        for i in range(len(current_route) - 1):
            current_distance += self.get_road_distance(
                *current_route[i], *current_route[i + 1]
            )
        
        # Calculate new route distance with candidate user
        # Try inserting user at best position
        best_distance = float('inf')
        best_position = -1
        
        for insert_pos in range(1, len(current_route)):
            new_route = current_route.copy()
            new_route.insert(insert_pos, candidate_user)
            
            new_distance = 0.0
            for i in range(len(new_route) - 1):
                new_distance += self.get_road_distance(
                    *new_route[i], *new_route[i + 1]
                )
            
            if new_distance < best_distance:
                best_distance = new_distance
                best_position = insert_pos
        
        # Check if detour is reasonable
        detour_ratio = best_distance / max(current_distance, 0.1)
        
        # Additional coherence check - ensure the new route makes sense directionally
        if best_position > 0:
            # Check if the insertion creates logical geographical progression
            prev_pos = current_route[best_position - 1]
            next_pos = current_route[best_position] if best_position < len(current_route) else office_pos
            
            # Calculate bearings to ensure logical progression
            bearing_to_candidate = self._calculate_bearing(*prev_pos, *candidate_user)
            bearing_to_next = self._calculate_bearing(*candidate_user, *next_pos)
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
        lng_to_m = 111320.0 * math.cos(math.radians(point[0]))  # longitude varies by latitude
        
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
    
    def _calculate_bearing(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate bearing between two points"""
        import math
        lat1, lat2 = map(math.radians, [lat1, lat2])
        dlon = math.radians(lon2 - lon1)
        x = math.sin(dlon) * math.cos(lat2)
        y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
        return (math.degrees(math.atan2(x, y)) + 360) % 360
    
    def _calculate_directional_penalty(self, route_points: List[Tuple[float, float]]) -> float:
        """Calculate penalty for routes with excessive directional changes (zig-zag)"""
        if len(route_points) < 3:
            return 0.0
        
        bearings = []
        for i in range(len(route_points) - 1):
            bearing = self._calculate_bearing(*route_points[i], *route_points[i + 1])
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
    
    def _calculate_backtrack_penalty(self, route_points: List[Tuple[float, float]], 
                                   office_pos: Tuple[float, float]) -> float:
        """Calculate penalty for routes that backtrack away from office"""
        if len(route_points) < 2:
            return 0.0
        
        driver_pos = route_points[0]
        penalty = 0.0
        
        # Calculate initial distance to office
        prev_distance_to_office = self._haversine_distance(*driver_pos, *office_pos)
        
        for i in range(1, len(route_points) - 1):  # Exclude final office position
            current_pos = route_points[i]
            current_distance_to_office = self._haversine_distance(*current_pos, *office_pos)
            
            # Penalty if we're moving further from office
            if current_distance_to_office > prev_distance_to_office:
                backtrack_amount = current_distance_to_office - prev_distance_to_office
                penalty += min(0.5, backtrack_amount / 10.0)  # Max 0.5 penalty per point
            
            prev_distance_to_office = current_distance_to_office
        
        return min(1.0, penalty)
    
    def get_optimal_pickup_sequence(self, driver_pos: Tuple[float, float],
                                  user_positions: List[Tuple[float, float]],
                                  office_pos: Tuple[float, float]) -> List[int]:
        """Find optimal pickup sequence to minimize zig-zag and backtracking"""
        if len(user_positions) <= 1:
            return list(range(len(user_positions)))
        
        # Use a greedy approach: always pick the next user that minimizes detour
        remaining_users = list(range(len(user_positions)))
        pickup_sequence = []
        current_pos = driver_pos
        
        while remaining_users:
            best_user_idx = None
            best_score = float('inf')
            
            for user_idx in remaining_users:
                user_pos = user_positions[user_idx]
                
                # Calculate score based on:
                # 1. Distance from current position
                # 2. Progress toward office
                # 3. Road network distance
                
                distance_score = self.get_road_distance(*current_pos, *user_pos)
                
                # Progress toward office (negative if moving away)
                current_to_office = self._haversine_distance(*current_pos, *office_pos)
                user_to_office = self._haversine_distance(*user_pos, *office_pos)
                progress_score = max(0, current_to_office - user_to_office) * -2  # Bonus for progress
                
                total_score = distance_score + progress_score
                
                if total_score < best_score:
                    best_score = total_score
                    best_user_idx = user_idx
            
            if best_user_idx is not None:
                pickup_sequence.append(best_user_idx)
                remaining_users.remove(best_user_idx)
                current_pos = user_positions[best_user_idx]
        
        return pickup_sequence
