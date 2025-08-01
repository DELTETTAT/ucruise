import json
import time
import os
import requests
from datetime import datetime
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from assignment import run_assignment, load_env_and_fetch_data, analyze_assignment_quality, haversine_distance, validate_input_data

def test_api_connection_debug(source_id: str):
    """Test API connection with detailed debugging"""
    print("🔧 TESTING API CONNECTION")
    print("-" * 40)

    # Load environment
    if not os.path.exists(".env"):
        print("❌ .env file not found")
        return False

    load_dotenv(".env")

    BASE_API_URL = os.getenv("API_URL")
    API_AUTH_TOKEN = os.getenv("API_AUTH_TOKEN")

    print(f"🔧 API_URL: {BASE_API_URL}")
    print(f"🔧 API_AUTH_TOKEN: {'*' * len(API_AUTH_TOKEN) if API_AUTH_TOKEN else 'Not set'}")

    if not BASE_API_URL or not API_AUTH_TOKEN:
        print("❌ Missing API_URL or API_AUTH_TOKEN in .env file")
        return False

    API_URL = f"{BASE_API_URL.rstrip('/')}/{source_id}"

    headers = {
        "Authorization": f"Bearer {API_AUTH_TOKEN}",
        "Content-Type": "application/json"
    }

    print(f"\n🌐 Testing API endpoint: {API_URL}")

    try:
        response = requests.get(API_URL, headers=headers, timeout=30)
        print(f"📊 Status Code: {response.status_code}")
        print(f"📊 Content Length: {len(response.text)}")
        print(f"📊 Content Type: {response.headers.get('content-type', 'unknown')}")

        if response.status_code == 200:
            print("✅ API request successful")

            if response.text.strip():
                print(f"📊 Raw response (first 200 chars): {response.text[:200]}")
                try:
                    data = response.json()
                    print(f"✅ JSON parsing successful")
                    print(f"📊 Response keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")
                    return True
                except Exception as e:
                    print(f"❌ JSON parsing failed: {e}")
                    return False
            else:
                print("❌ Empty response received")
                return False
        else:
            print(f"❌ API request failed with status {response.status_code}")
            print(f"📊 Error response: {response.text[:200]}")
            return False

    except Exception as e:
        print(f"❌ Request failed: {e}")
        return False

def comprehensive_analysis(source_id: str, output_file: str = "analysis_report.json"):
    """
    Comprehensive analysis that fetches data, runs assignment, and provides detailed statistics
    """
    print(f"🔍 Starting comprehensive analysis for source_id: {source_id}")

    analysis_report = {
        "metadata": {
            "source_id": source_id,
            "timestamp": datetime.now().isoformat(),
            "analysis_version": "1.0"
        },
        "raw_data_analysis": {},
        "api_response": {},
        "assignment_results": {},
        "performance_metrics": {},
        "geographical_analysis": {},
        "optimization_insights": {},
        "recommendations": []
    }

    try:
        # Test API connection first
        print("🔍 Testing API connection...")
        if not test_api_connection_debug(source_id):
            analysis_report["error"] = "API connection test failed"
            with open(output_file, 'w') as f:
                json.dump(analysis_report, f, indent=2, default=str)
            return analysis_report

        # 1. RAW DATA ANALYSIS
        print("📊 Analyzing raw data from API...")
        start_time = time.time()

        try:
            print(f"🌐 Fetching data for source_id: {source_id}")
            raw_data = load_env_and_fetch_data(source_id)
            data_fetch_time = time.time() - start_time

            print(f"📥 API Response type: {type(raw_data)}")
            if isinstance(raw_data, dict):
                print(f"📥 API Response keys: {list(raw_data.keys())}")

                # Log office coordinates being used
                if '_extracted_office_lat' in raw_data and '_extracted_office_lon' in raw_data:
                    print(f"📍 Office coordinates: {raw_data['_extracted_office_lat']}, {raw_data['_extracted_office_lon']}")

            if not raw_data or not isinstance(raw_data, dict):
                raise ValueError("API returned empty or invalid data structure")
            
            # Validate the data structure
            try:
                validate_input_data(raw_data)
                print("✅ Data validation passed")
            except Exception as validation_error:
                print(f"⚠️ Data validation warning: {validation_error}")
                # Continue with analysis but note the validation issue
                analysis_report["data_validation_warning"] = str(validation_error)

        except Exception as api_error:
            analysis_report["error"] = f"API fetch failed: {str(api_error)}"
            print(f"❌ API fetch failed: {api_error}")
            with open(output_file, 'w') as f:
                json.dump(analysis_report, f, indent=2, default=str)
            return analysis_report

        users_data = raw_data.get("users", [])
        drivers_data = raw_data.get("drivers", [])

        print(f"🔍 Raw data - Users: {len(users_data)}, Drivers: {len(drivers_data)}")

        # Check for data quality issues that might cause user loss
        users_with_coords = [u for u in users_data if u.get('latitude') and u.get('longitude')]
        users_missing_coords = len(users_data) - len(users_with_coords)

        print(f"🔍 Users with coordinates: {len(users_with_coords)}")
        print(f"⚠️  Users missing coordinates: {users_missing_coords}")

        # Raw data statistics
        analysis_report["raw_data_analysis"] = {
            "total_users": len(users_data),
            "total_drivers": len(drivers_data),
            "users_with_valid_coordinates": len(users_with_coords),
            "users_missing_coordinates": users_missing_coords,
            "data_fetch_time_seconds": round(data_fetch_time, 3),
            "users_analysis": analyze_users_data(users_data),
            "drivers_analysis": analyze_drivers_data(drivers_data),
            "capacity_vs_demand": analyze_capacity_demand(users_data, drivers_data)
        }

        # 2. API RESPONSE STRUCTURE
        analysis_report["api_response"] = {
            "response_structure": {
                "has_status": "status" in raw_data,
                "has_users": "users" in raw_data,
                "has_drivers": "drivers" in raw_data,
                "additional_fields": [k for k in raw_data.keys() if k not in ["users", "drivers", "status"]]
            },
            "data_quality": assess_data_quality(users_data, drivers_data)
        }

        # 3. RUN ASSIGNMENT
        print("🚗 Running driver assignment algorithm...")
        assignment_start = time.time()

        assignment_result = run_assignment(source_id)
        assignment_time = time.time() - assignment_start

        # DEBUG: Track all users through the assignment process
        debug_user_tracking = debug_user_assignment(users_data, assignment_result)
        analysis_report["user_tracking_debug"] = debug_user_tracking

        analysis_report["assignment_results"] = {
            "execution_time_seconds": round(assignment_time, 3),
            "status": assignment_result["status"],
            "basic_stats": analyze_assignment_quality(assignment_result) if assignment_result["status"] == "true" else None
        }

        if assignment_result["status"] == "true":
            # 4. IDENTIFY LOST USERS
            lost_users_analysis = identify_lost_users(users_data, assignment_result)
            analysis_report["lost_users_analysis"] = lost_users_analysis

            if lost_users_analysis["lost_users_count"] > 0:
                print(f"⚠️  FOUND {lost_users_analysis['lost_users_count']} LOST USERS!")
                for user_detail in lost_users_analysis["lost_users_details"]:
                    print(f"   - User ID {user_detail['id']}: lat={user_detail['latitude']}, lng={user_detail['longitude']}")

            # 5. DETAILED ASSIGNMENT ANALYSIS
            routes = assignment_result["data"]
            unassigned_users = assignment_result.get("unassignedUsers", [])
            unassigned_drivers = assignment_result.get("unassignedDrivers", [])

            analysis_report["performance_metrics"] = calculate_performance_metrics(routes, unassigned_users, unassigned_drivers)
            analysis_report["geographical_analysis"] = geographical_analysis(routes)
            analysis_report["optimization_insights"] = optimization_insights(routes, users_data, drivers_data)
            analysis_report["recommendations"] = generate_recommendations(analysis_report)

            # 6. ROUTE-BY-ROUTE BREAKDOWN
            analysis_report["route_breakdown"] = []
            for i, route in enumerate(routes):
                route_analysis = analyze_individual_route(route, i)
                analysis_report["route_breakdown"].append(route_analysis)

        # 6. SAVE RESULTS
        print(f"💾 Saving analysis to {output_file}")
        with open(output_file, 'w') as f:
            json.dump(analysis_report, f, indent=2, default=str)

        # 7. PRINT SUMMARY
        print_analysis_summary(analysis_report)

        return analysis_report

    except Exception as e:
        analysis_report["error"] = str(e)
        print(f"❌ Analysis failed: {e}")
        with open(output_file, 'w') as f:
            json.dump(analysis_report, f, indent=2, default=str)
        return analysis_report

def analyze_users_data(users_data):
    """Analyze users data structure and statistics"""
    if not users_data:
        return {"error": "No users data available"}

    df = pd.DataFrame(users_data)

    analysis = {
        "total_count": len(users_data),
        "fields_available": list(df.columns),
        "data_types": df.dtypes.to_dict(),
        "missing_data": df.isnull().sum().to_dict(),
        "statistics": {}
    }

    # Geographical stats
    if 'latitude' in df.columns and 'longitude' in df.columns:
        df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
        df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')

        analysis["statistics"]["geographical"] = {
            "lat_range": [float(df['latitude'].min()), float(df['latitude'].max())],
            "lng_range": [float(df['longitude'].min()), float(df['longitude'].max())],
            "center_point": [float(df['latitude'].mean()), float(df['longitude'].mean())]
        }

    # Office distance stats
    if 'office_distance' in df.columns:
        df['office_distance'] = pd.to_numeric(df['office_distance'], errors='coerce')
        analysis["statistics"]["office_distance"] = {
            "min": float(df['office_distance'].min()),
            "max": float(df['office_distance'].max()),
            "mean": float(df['office_distance'].mean()),
            "median": float(df['office_distance'].median())
        }

    # Shift type distribution
    if 'shift_type' in df.columns:
        analysis["statistics"]["shift_distribution"] = df['shift_type'].value_counts().to_dict()

    return analysis

def analyze_drivers_data(drivers_data):
    """Analyze drivers data structure and statistics"""
    if not drivers_data:
        return {"error": "No drivers data available"}

    df = pd.DataFrame(drivers_data)

    analysis = {
        "total_count": len(drivers_data),
        "fields_available": list(df.columns),
        "data_types": df.dtypes.to_dict(),
        "missing_data": df.isnull().sum().to_dict(),
        "statistics": {}
    }

    # Capacity analysis
    if 'capacity' in df.columns:
        df['capacity'] = pd.to_numeric(df['capacity'], errors='coerce')
        analysis["statistics"]["capacity"] = {
            "total_capacity": int(df['capacity'].sum()),
            "min_capacity": int(df['capacity'].min()),
            "max_capacity": int(df['capacity'].max()),
            "avg_capacity": float(df['capacity'].mean()),
            "capacity_distribution": df['capacity'].value_counts().to_dict()
        }

    # Geographical stats
    if 'latitude' in df.columns and 'longitude' in df.columns:
        df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
        df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')

        analysis["statistics"]["geographical"] = {
            "lat_range": [float(df['latitude'].min()), float(df['latitude'].max())],
            "lng_range": [float(df['longitude'].min()), float(df['longitude'].max())],
            "center_point": [float(df['latitude'].mean()), float(df['longitude'].mean())]
        }

    return analysis

def analyze_capacity_demand(users_data, drivers_data):
    """Analyze capacity vs demand relationship"""
    total_users = len(users_data)
    total_capacity = sum(int(d.get('capacity', 0)) for d in drivers_data)

    return {
        "total_demand": total_users,
        "total_capacity": total_capacity,
        "capacity_utilization_potential": round((total_users / total_capacity * 100), 2) if total_capacity > 0 else 0,
        "surplus_capacity": max(0, total_capacity - total_users),
        "demand_shortage": max(0, total_users - total_capacity),
        "capacity_adequacy": "adequate" if total_capacity >= total_users else "insufficient"
    }

def assess_data_quality(users_data, drivers_data):
    """Assess quality of data from API"""
    issues = []

    # Check users data quality
    for i, user in enumerate(users_data):
        if not user.get('latitude') or not user.get('longitude'):
            issues.append(f"User {i}: Missing coordinates")
        if not user.get('id'):
            issues.append(f"User {i}: Missing ID")

    # Check drivers data quality
    for i, driver in enumerate(drivers_data):
        if not driver.get('latitude') or not driver.get('longitude'):
            issues.append(f"Driver {i}: Missing coordinates")
        if not driver.get('capacity'):
            issues.append(f"Driver {i}: Missing capacity")
        if not driver.get('id'):
            issues.append(f"Driver {i}: Missing ID")

    return {
        "issues_found": len(issues),
        "issues_list": issues[:10],  # Limit to first 10 issues
        "data_completeness_score": max(0, 100 - len(issues) * 5)
    }

def calculate_performance_metrics(routes, unassigned_users, unassigned_drivers):
    """Calculate detailed performance metrics"""
    total_assigned = sum(len(route["assigned_users"]) for route in routes)
    total_unassigned = len(unassigned_users)
    total_users = total_assigned + total_unassigned

    utilizations = []
    route_distances = []

    for route in routes:
        if route["assigned_users"]:
            util = len(route["assigned_users"]) / route["vehicle_type"]
            utilizations.append(util)

            # Calculate average distance from driver to users
            driver_pos = (route["latitude"], route["longitude"])
            distances = []
            for user in route["assigned_users"]:
                dist = haversine_distance(driver_pos[0], driver_pos[1], user["lat"], user["lng"])
                distances.append(dist)
            route_distances.extend(distances)

    return {
        "assignment_efficiency": {
            "total_users": total_users,
            "assigned_users": total_assigned,
            "unassigned_users": total_unassigned,
            "assignment_rate_percent": round((total_assigned / total_users * 100), 2) if total_users > 0 else 0
        },
        "utilization_metrics": {
            "average_utilization_percent": round(np.mean(utilizations) * 100, 2) if utilizations else 0,
            "min_utilization_percent": round(np.min(utilizations) * 100, 2) if utilizations else 0,
            "max_utilization_percent": round(np.max(utilizations) * 100, 2) if utilizations else 0,
            "utilization_distribution": {
                "below_50_percent": sum(1 for u in utilizations if u < 0.5),
                "50_to_80_percent": sum(1 for u in utilizations if 0.5 <= u < 0.8),
                "80_to_100_percent": sum(1 for u in utilizations if u >= 0.8),
                "full_capacity": sum(1 for u in utilizations if u >= 1.0)
            }
        },
        "distance_metrics": {
            "average_distance_km": round(np.mean(route_distances), 2) if route_distances else 0,
            "max_distance_km": round(np.max(route_distances), 2) if route_distances else 0,
            "distances_over_5km": sum(1 for d in route_distances if d > 5),
            "distances_over_10km": sum(1 for d in route_distances if d > 10)
        },
        "resource_utilization": {
            "total_routes_created": len(routes),
            "drivers_utilized": len(routes),
            "drivers_unused": len(unassigned_drivers),
            "driver_utilization_rate_percent": round((len(routes) / (len(routes) + len(unassigned_drivers)) * 100), 2) if (len(routes) + len(unassigned_drivers)) > 0 else 0
        }
    }

def geographical_analysis(routes):
    """Analyze geographical distribution of routes"""
    route_centers = []
    for route in routes:
        if route["assigned_users"]:
            lats = [u["lat"] for u in route["assigned_users"]]
            lngs = [u["lng"] for u in route["assigned_users"]]
            route_centers.append([np.mean(lats), np.mean(lngs)])

    if not route_centers:
        return {"error": "No routes with assigned users"}

    centers_df = pd.DataFrame(route_centers, columns=['lat', 'lng'])

    return {
        "route_distribution": {
            "number_of_routes": len(routes),
            "geographical_spread": {
                "lat_range": [float(centers_df['lat'].min()), float(centers_df['lat'].max())],
                "lng_range": [float(centers_df['lng'].min()), float(centers_df['lng'].max())],
                "center_of_operations": [float(centers_df['lat'].mean()), float(centers_df['lng'].mean())]
            }
        },
        "clustering_effectiveness": analyze_clustering_effectiveness(routes)
    }

def analyze_clustering_effectiveness(routes):
    """Analyze how well users are clustered geographically"""
    cluster_tightness = []

    for route in routes:
        if len(route["assigned_users"]) > 1:
            coords = [(u["lat"], u["lng"]) for u in route["assigned_users"]]
            distances = []
            for i, coord1 in enumerate(coords):
                for coord2 in coords[i+1:]:
                    dist = haversine_distance(coord1[0], coord1[1], coord2[0], coord2[1])
                    distances.append(dist)

            if distances:
                avg_intra_cluster_distance = np.mean(distances)
                cluster_tightness.append(avg_intra_cluster_distance)

    return {
        "average_intra_cluster_distance_km": round(np.mean(cluster_tightness), 2) if cluster_tightness else 0,
        "tightest_cluster_km": round(np.min(cluster_tightness), 2) if cluster_tightness else 0,
        "loosest_cluster_km": round(np.max(cluster_tightness), 2) if cluster_tightness else 0,
        "well_clustered_routes": sum(1 for t in cluster_tightness if t < 3.0)  # Routes with avg distance < 3km
    }

def optimization_insights(routes, users_data, drivers_data):
    """Generate optimization insights and suggestions"""
    insights = {
        "efficiency_opportunities": [],
        "resource_optimization": [],
        "geographical_optimization": []
    }

    # Efficiency opportunities
    underutilized_routes = [r for r in routes if len(r["assigned_users"]) / r["vehicle_type"] < 0.8]
    if underutilized_routes:
        insights["efficiency_opportunities"].append({
            "type": "underutilized_vehicles",
            "count": len(underutilized_routes),
            "potential_capacity_savings": sum(r["vehicle_type"] - len(r["assigned_users"]) for r in underutilized_routes)
        })

    # Resource optimization
    total_capacity = sum(int(d.get('capacity', 0)) for d in drivers_data)
    total_users = len(users_data)
    if total_capacity > total_users * 1.2:  # 20% overcapacity
        insights["resource_optimization"].append({
            "type": "excess_capacity",
            "excess_seats": total_capacity - total_users,
            "recommendation": "Consider reducing fleet size or expanding service area"
        })

    return insights

def analyze_individual_route(route, route_index):
    """Analyze individual route performance"""
    if not route["assigned_users"]:
        return {
            "route_id": route_index,
            "driver_id": route["driver_id"],
            "status": "empty",
            "utilization_percent": 0
        }

    driver_pos = (route["latitude"], route["longitude"])
    distances = []
    for user in route["assigned_users"]:
        dist = haversine_distance(driver_pos[0], driver_pos[1], user["lat"], user["lng"])
        distances.append(dist)

    return {
        "route_id": route_index,
        "driver_id": route["driver_id"],
        "vehicle_capacity": route["vehicle_type"],
        "assigned_users_count": len(route["assigned_users"]),
        "utilization_percent": round((len(route["assigned_users"]) / route["vehicle_type"]) * 100, 2),
        "distance_metrics": {
            "avg_distance_to_users_km": round(np.mean(distances), 2),
            "max_distance_to_user_km": round(np.max(distances), 2),
            "min_distance_to_user_km": round(np.min(distances), 2)
        },
        "geographical_center": route.get("centroid", [route["latitude"], route["longitude"]]),
        "efficiency_rating": calculate_route_efficiency_rating(route, distances)
    }

def calculate_route_efficiency_rating(route, distances):
    """Calculate efficiency rating for a route (1-10 scale)"""
    utilization = len(route["assigned_users"]) / route["vehicle_type"]
    avg_distance = np.mean(distances) if distances else 0

    # Base score from utilization (40% weight)
    utilization_score = utilization * 4

    # Distance penalty (60% weight) - lower distances = higher score
    if avg_distance <= 2:
        distance_score = 6
    elif avg_distance <= 5:
        distance_score = 4
    elif avg_distance <= 8:
        distance_score = 2
    else:
        distance_score = 0

    total_score = utilization_score + distance_score
    return round(min(10, max(1, total_score)), 1)

def generate_recommendations(analysis_report):
    """Generate actionable recommendations based on analysis"""
    recommendations = []

    # Check assignment rate
    if analysis_report["assignment_results"]["basic_stats"]:
        assignment_rate = analysis_report["assignment_results"]["basic_stats"]["assignment_rate"]
        if assignment_rate < 90:
            recommendations.append({
                "priority": "high",
                "category": "assignment_efficiency",
                "issue": f"Low assignment rate: {assignment_rate}%",
                "recommendation": "Increase driver coverage in underserved areas or adjust capacity allocation"
            })

    # Check utilization
    perf_metrics = analysis_report.get("performance_metrics", {})
    util_metrics = perf_metrics.get("utilization_metrics", {})
    if util_metrics.get("average_utilization_percent", 0) < 70:
        recommendations.append({
            "priority": "medium",
            "category": "resource_optimization",
            "issue": "Low average vehicle utilization",
            "recommendation": "Consider route consolidation or fleet size reduction"
        })

    # Check distance efficiency
    distance_metrics = perf_metrics.get("distance_metrics", {})
    if distance_metrics.get("distances_over_5km", 0) > 5:
        recommendations.append({
            "priority": "medium",
            "category": "geographical_optimization",
            "issue": "Many long-distance assignments",
            "recommendation": "Review driver positioning or add drivers in high-demand areas"
        })

    return recommendations

def identify_lost_users(users_data, assignment_result):
    """Identify which specific users were lost during assignment"""
    if assignment_result["status"] != "true":
        return {"error": "Assignment failed"}

    # Get all user IDs from assignment result
    assigned_user_ids = set()
    for route in assignment_result["data"]:
        for user in route["assigned_users"]:
            assigned_user_ids.add(str(user["user_id"]))

    unassigned_user_ids = set()
    for user in assignment_result.get("unassignedUsers", []):
        unassigned_user_ids.add(str(user["user_id"]))

    processed_user_ids = assigned_user_ids | unassigned_user_ids

    # Get original user IDs
    original_user_ids = set(str(user.get("id", "")) for user in users_data)

    # Find lost users
    lost_user_ids = original_user_ids - processed_user_ids

    lost_users_details = []
    for user in users_data:
        if str(user.get("id", "")) in lost_user_ids:
            lost_users_details.append({
                "id": user.get("id"),
                "has_latitude": bool(user.get("latitude")),
                "has_longitude": bool(user.get("longitude")),
                "latitude": user.get("latitude"),
                "longitude": user.get("longitude"),
                "office_distance": user.get("office_distance"),
                "shift_type": user.get("shift_type")
            })

    return {
        "original_users": len(users_data),
        "processed_users": len(processed_user_ids),
        "lost_users_count": len(lost_user_ids),
        "lost_user_ids": list(lost_user_ids),
        "lost_users_details": lost_users_details
    }

def print_analysis_summary(analysis_report):
    """Print a formatted summary of the analysis"""
    print("\n" + "="*60)
    print("📊 COMPREHENSIVE ANALYSIS SUMMARY")
    print("="*60)

    # Basic info
    metadata = analysis_report["metadata"]
    print(f"Source ID: {metadata['source_id']}")
    print(f"Analysis Time: {metadata['timestamp']}")

    # Raw data
    raw_data = analysis_report["raw_data_analysis"]
    print(f"\n📋 RAW DATA:")
    print(f"  Users: {raw_data['total_users']}")
    print(f"  Drivers: {raw_data['total_drivers']}")
    print(f"  Data Fetch Time: {raw_data['data_fetch_time_seconds']}s")

    # Assignment results
    if analysis_report["assignment_results"]["status"] == "true":
        basic_stats = analysis_report["assignment_results"]["basic_stats"]
        print(f"\n🚗 ASSIGNMENT RESULTS:")
        print(f"  Routes Created: {basic_stats['total_routes']}")
        print(f"  Users Assigned: {basic_stats['total_assigned_users']}")
        print(f"  Assignment Rate: {basic_stats['assignment_rate']}%")
        print(f"  Average Utilization: {basic_stats['avg_utilization']}%")

        # Performance metrics
        perf = analysis_report["performance_metrics"]
        print(f"\n📈 PERFORMANCE METRICS:")
        print(f"  Driver Utilization: {perf['resource_utilization']['driver_utilization_rate_percent']}%")
        print(f"  Average Distance: {perf['distance_metrics']['average_distance_km']}km")

        # Recommendations
        recommendations = analysis_report["recommendations"]
        if recommendations:
            print(f"\n💡 TOP RECOMMENDATIONS:")
            for i, rec in enumerate(recommendations[:3], 1):
                print(f"  {i}. [{rec['priority'].upper()}] {rec['recommendation']}")
    else:
        print(f"\n❌ ASSIGNMENT FAILED")

    print("="*60)

def debug_user_assignment(users_data, assignment_result):
    """
    Debug function to track users' assignment status.
    """
    user_tracking = {}
    for user in users_data:
        user_id = str(user.get("id", "N/A"))
        user_tracking[user_id] = {"status": "original"}  # Initial status

    if assignment_result["status"] == "true":
        # Mark assigned users
        for route in assignment_result["data"]:
            for assigned_user in route["assigned_users"]:
                user_id = str(assigned_user.get("user_id", "N/A"))
                if user_id in user_tracking:
                    user_tracking[user_id]["status"] = "assigned"

        # Mark unassigned users
        for unassigned_user in assignment_result.get("unassignedUsers", []):
            user_id = str(unassigned_user.get("user_id", "N/A"))
            if user_id in user_tracking:
                user_tracking[user_id]["status"] = "unassigned"

        # Identify lost users (not assigned or unassigned)
        assigned_user_ids = set()
        for route in assignment_result["data"]:
            for user in route["assigned_users"]:
                assigned_user_ids.add(str(user["user_id"]))

        unassigned_user_ids = set()
        for user in assignment_result.get("unassignedUsers", []):
            unassigned_user_ids.add(str(user["user_id"]))

        processed_user_ids = assigned_user_ids | unassigned_user_ids

        # Get original user IDs
        original_user_ids = set(str(user.get("id", "")) for user in users_data)

        # Find lost users
        lost_user_ids = original_user_ids - processed_user_ids

        for user_id in lost_user_ids:
            if user_id in user_tracking:
                user_tracking[user_id]["status"] = "lost"
    else:
        print("Assignment failed, user tracking not available.")

    # Summarize user status
    status_summary = {}
    for user_id, tracking_info in user_tracking.items():
        status = tracking_info["status"]
        if status not in status_summary:
            status_summary[status] = 0
        status_summary[status] += 1

    return {
        "user_statuses": user_tracking,
        "status_summary": status_summary
    }

if __name__ == "__main__":
    # Example usage
    SOURCE_ID = "UC_healthcarellp"  # Replace with your actual source_id

    print("🔍 Starting comprehensive analysis...")
    print("This will fetch data, run assignment, and generate detailed analysis")
    print(f"Using source_id: {SOURCE_ID}")

    result = comprehensive_analysis(SOURCE_ID)

    print(f"\n✅ Analysis complete! Check 'analysis_report.json' for full details.")
