import subprocess
import threading
import time
import webbrowser
import sys
from assignment import run_assignment, analyze_assignment_quality

SOURCE_ID = "UC_logisticllp"  # <-- Replace with your real source_id

def start_fastapi():
    subprocess.run(["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5000", "--reload"])

def launch_browser():
    time.sleep(5)  # Wait longer for server to start
    try:
        webbrowser.open("http://localhost:5000/visualize")
        print("🌐 Browser opened at: http://localhost:5000/visualize")
    except Exception as e:
        print(f"⚠️  Could not auto-open browser: {e}")
        print("   Please manually visit: http://localhost:5000/visualize")

def display_detailed_analytics(result):
    """Display comprehensive analytics in terminal with enhanced formatting"""
    print("\n" + "🎯" + "="*78 + "🎯")
    print("📊 ROUTEFLOW - INTELLIGENT ASSIGNMENT ANALYTICS DASHBOARD")
    print("🎯" + "="*78 + "🎯")
    
    if result["status"] != "true":
        print("❌ Assignment failed - no analytics available")
        return
    
    # Basic metrics
    routes = result["data"]
    unassigned_users = result.get("unassignedUsers", [])
    unassigned_drivers = result.get("unassignedDrivers", [])
    
    total_assigned = sum(len(route["assigned_users"]) for route in routes)
    total_capacity = sum(route["vehicle_type"] for route in routes)
    
    # Enhanced Overview Section
    print(f"\n📈 SYSTEM OVERVIEW")
    print("─" * 50)
    print(f"   🚗 Active Routes Created: {len(routes)}")
    print(f"   👥 Users Successfully Assigned: {total_assigned}")
    print(f"   ⚠️  Users Unassigned: {len(unassigned_users)}")
    print(f"   🚙 Drivers Deployed: {len(routes)}")
    print(f"   💤 Drivers Available: {len(unassigned_drivers)}")
    print(f"   🏆 Total Fleet Capacity: {total_capacity} passengers")
    print(f"   📊 Overall Capacity Utilization: {(total_assigned/total_capacity*100):.1f}%")
    
    # Enhanced Route Performance Analysis
    utilizations = []
    high_efficiency = 0
    medium_efficiency = 0
    low_efficiency = 0
    total_route_distance = 0
    
    print(f"\n🚗 DETAILED ROUTE PERFORMANCE ANALYSIS")
    print("─" * 50)
    
    for i, route in enumerate(routes):
        assigned = len(route["assigned_users"])
        capacity = route["vehicle_type"]
        utilization = assigned / capacity
        utilizations.append(utilization)
        
        # Calculate route distance metrics
        route_distances = []
        for user in route["assigned_users"]:
            dist = haversine_distance(float(route["latitude"]), float(route["longitude"]), 
                                    float(user["lat"]), float(user["lng"]))
            route_distances.append(dist)
        
        avg_route_distance = sum(route_distances) / len(route_distances) if route_distances else 0
        max_route_distance = max(route_distances) if route_distances else 0
        total_route_distance += avg_route_distance
        
        # Efficiency categorization
        if utilization >= 0.8:
            efficiency_icon = "🟢"
            efficiency_label = "EXCELLENT"
            high_efficiency += 1
        elif utilization >= 0.6:
            efficiency_icon = "🟡"
            efficiency_label = "GOOD"
            medium_efficiency += 1
        elif utilization >= 0.4:
            efficiency_icon = "🟠"
            efficiency_label = "FAIR"
            medium_efficiency += 1
        else:
            efficiency_icon = "🔴"
            efficiency_label = "NEEDS OPTIMIZATION"
            low_efficiency += 1
        
        print(f"   Route {i+1:2d}: {efficiency_icon} {efficiency_label:15} | "
              f"{assigned}/{capacity} users ({utilization*100:5.1f}%) | "
              f"Avg Dist: {avg_route_distance:4.1f}km | Max: {max_route_distance:4.1f}km")
    
    avg_utilization = sum(utilizations) / len(utilizations) if utilizations else 0
    avg_distance_per_route = total_route_distance / len(routes) if routes else 0
    
    # Performance Summary with Advanced Metrics
    print(f"\n📊 ADVANCED EFFICIENCY METRICS")
    print("─" * 50)
    print(f"   🎯 Average Route Utilization: {avg_utilization*100:.1f}%")
    print(f"   🟢 High Efficiency Routes (≥80%): {high_efficiency} ({high_efficiency/len(routes)*100:.1f}%)")
    print(f"   🟡 Medium Efficiency Routes (40-79%): {medium_efficiency} ({medium_efficiency/len(routes)*100:.1f}%)")
    print(f"   🔴 Low Efficiency Routes (<40%): {low_efficiency} ({low_efficiency/len(routes)*100:.1f}%)")
    print(f"   📏 Average Distance per Route: {avg_distance_per_route:.1f} km")
    print(f"   ⭐ System Efficiency Score: {calculate_system_efficiency_score(utilizations, avg_distance_per_route)}/10")
    
    # Resource Optimization Insights
    print(f"\n💡 OPTIMIZATION INSIGHTS")
    print("─" * 50)
    if avg_utilization < 0.6:
        print("   ⚠️  RECOMMENDATION: Consider reducing fleet size or expanding service area")
    if avg_distance_per_route > 8:
        print("   ⚠️  RECOMMENDATION: Review geographical clustering - routes may be too spread out")
    if low_efficiency > len(routes) * 0.3:
        print("   ⚠️  RECOMMENDATION: Reassign users to optimize capacity utilization")
    if high_efficiency >= len(routes) * 0.7:
        print("   ✅ EXCELLENT: Route assignments are highly optimized!")
    
    # Cost & Environmental Impact Estimates
    fuel_cost_per_km = 0.08  # Example: $0.08 per km
    co2_per_km = 0.12  # Example: 0.12 kg CO2 per km
    total_distance_estimate = total_route_distance * 2  # Round trip estimate
    
    print(f"\n🌍 ENVIRONMENTAL & COST IMPACT ESTIMATES")
    print("─" * 50)
    print(f"   ⛽ Estimated Daily Fuel Cost: ${total_distance_estimate * fuel_cost_per_km:.2f}")
    print(f"   🌱 Estimated CO2 Emissions: {total_distance_estimate * co2_per_km:.1f} kg/day")
    print(f"   🚗 Distance Efficiency: {total_assigned/total_distance_estimate:.1f} users per km")
    
    # Geographical insights
    print(f"\n🗺️  GEOGRAPHICAL DISTRIBUTION ANALYSIS")
    print("─" * 50)
    clustering_method = result.get("clustering_analysis", {}).get("method", "Intelligent Auto-Assignment")
    print(f"   🧠 Clustering Algorithm: {clustering_method}")
    print(f"   📍 Service Area Coverage: {len(routes)} distinct zones")
    
    if unassigned_users:
        print(f"\n⚠️  UNASSIGNED USERS REQUIRING ATTENTION")
        print("─" * 50)
        for i, user in enumerate(unassigned_users[:5]):  # Show first 5
            office_dist = user.get('office_distance', 'N/A')
            print(f"   {i+1}. User {user['user_id']}: Location ({user.get('lat', 'N/A')}, {user.get('lng', 'N/A')}) | "
                  f"Office Distance: {office_dist} km")
        if len(unassigned_users) > 5:
            print(f"   ... and {len(unassigned_users) - 5} more users need manual assignment")
    
    print("\n" + "🎯" + "="*78 + "🎯")
    print("🌐 ACCESS FULL INTERACTIVE DASHBOARD: http://localhost:5000/visualize")
    print("📊 Real-time analytics, route optimization, and performance monitoring available")
    print("🎯" + "="*78 + "🎯\n")

def calculate_system_efficiency_score(utilizations, avg_distance):
    """Calculate overall system efficiency score (0-10)"""
    if not utilizations:
        return 0
    
    # Utilization score (0-5 points)
    avg_util = sum(utilizations) / len(utilizations)
    util_score = min(avg_util * 5, 5)
    
    # Distance efficiency score (0-3 points) 
    distance_score = max(0, 3 - (avg_distance / 10) * 3)
    
    # Consistency score (0-2 points)
    util_variance = sum((u - avg_util) ** 2 for u in utilizations) / len(utilizations)
    consistency_score = max(0, 2 - util_variance * 4)
    
    return round(util_score + distance_score + consistency_score, 1)

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate the great circle distance between two points on the earth"""
    from math import radians, cos, sin, asin, sqrt
    
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371  # Radius of earth in kilometers
    
    return c * r

if __name__ == "__main__":
    print("🚀 Starting Driver Assignment Dashboard...")
    print(f"📍 Source ID: {SOURCE_ID}")
    print("-" * 50)
    
    try:
        print("⏳ Running assignment algorithm...")
        result = run_assignment(SOURCE_ID)

        if result["status"] == "true":
            print("✅ Assignment completed successfully!")
            
            # Save results
            import json
            with open("drivers_and_routes.json", "w") as f:
                json.dump(result["data"], f, indent=2)
            print("💾 Results saved to drivers_and_routes.json")
            
            # Display detailed analytics
            display_detailed_analytics(result)
            
            # Additional quality analysis
            quality_analysis = analyze_assignment_quality(result)
            if isinstance(quality_analysis, dict):
                print(f"\n🎯 QUALITY METRICS:")
                print(f"   Assignment Rate: {quality_analysis.get('assignment_rate', 0)}%")
                print(f"   Routes Below 80% Utilization: {quality_analysis.get('routes_below_80_percent', 0)}")
                if quality_analysis.get('distance_issues'):
                    print(f"   Distance Issues: {len(quality_analysis['distance_issues'])}")
            
        else:
            print("❌ Assignment failed:")
            print(f"   Error: {result.get('details', 'Unknown error')}")
            print(f"   Please check your configuration and API credentials")
            exit(1)

        print("\n🚀 Launching Dashboard...")
        print("   - Starting FastAPI server on port 5000")
        print("   - Opening browser automatically")

        # Start server in background
        server_thread = threading.Thread(target=start_fastapi, daemon=True)
        server_thread.start()
        
        # Launch browser
        browser_thread = threading.Thread(target=launch_browser, daemon=True)
        browser_thread.start()

        print("\n✅ Dashboard is starting up...")
        print("📱 Manual URL: http://localhost:5000/visualize")
        print("📊 API Endpoint: http://localhost:5000/routes")
        print("\n⌨️  Press Ctrl+C to stop the server")
        
        try:
            # Keep the main thread alive
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 Shutting down dashboard...")
            print("👋 Goodbye!")
            
    except KeyboardInterrupt:
        print("\n🛑 Assignment interrupted by user")
        exit(0)
    except Exception as e:
        print(f"❌ Error running assignment: {e}")
        print(f"📋 Error type: {type(e).__name__}")
        exit(1)
