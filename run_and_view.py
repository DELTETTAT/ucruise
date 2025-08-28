import subprocess
import time
import threading
import webbrowser
import json
import os
from assignment_engine import run_assignment
from utils import ValidationError
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_assignment_quality(result):
    """Analyze assignment quality with dynamic driver analysis"""
    if result["status"] != "true":
        return "Assignment failed"

    total_routes = len(result["data"])
    total_assigned = sum(len(route["assigned_users"]) for route in result["data"])
    total_unassigned = len(result["unassignedUsers"])

    utilizations = []
    distance_issues = []

    for route in result["data"]:
        if route["assigned_users"]:
            util = len(route["assigned_users"]) / route["vehicle_type"]
            utilizations.append(util)

            # Check distances (using dynamic threshold)
            driver_pos = (route["latitude"], route["longitude"])
            for user in route["assigned_users"]:
                from utils import haversine_distance
                dist = haversine_distance(driver_pos[0], driver_pos[1],
                                         user["lat"], user["lng"])
                # Dynamic threshold based on assignment type
                threshold = 8.0 if result.get("optimization_mode") == "balance_assignment" else 6.0
                if dist > threshold:
                    distance_issues.append({
                        "driver_id": route["driver_id"],
                        "user_id": user["user_id"],
                        "distance_km": round(dist, 2)
                    })

    analysis = {
        "total_routes": total_routes,
        "total_assigned_users": total_assigned,
        "total_unassigned_users": total_unassigned,
        "assignment_rate": round(total_assigned / (total_assigned + total_unassigned) * 100, 1) if (total_assigned + total_unassigned) > 0 else 0,
        "avg_utilization": round(sum(utilizations) / len(utilizations) * 100, 1) if utilizations else 0,
        "min_utilization": round(min(utilizations) * 100, 1) if utilizations else 0,
        "max_utilization": round(max(utilizations) * 100, 1) if utilizations else 0,
        "routes_below_80_percent": sum(1 for u in utilizations if u < 0.8),
        "distance_issues": distance_issues,
        "optimization_mode": result.get("optimization_mode", "unknown")
    }

    return analysis

def print_detailed_analytics(result):
    """Print detailed analytics with dynamic data"""
    print(f"\n📊 DETAILED ANALYTICS:")
    print(f"{'═' * 60}")

    if result["status"] != "true":
        print(f"❌ Assignment failed: {result.get('details', 'Unknown error')}")
        return

    # Basic metrics
    total_routes = len(result["data"])
    total_assigned = sum(len(route["assigned_users"]) for route in result["data"])
    total_unassigned = len(result["unassignedUsers"])
    total_drivers = len(result["unassignedDrivers"]) + total_routes

    print(f"🎯 ASSIGNMENT SUMMARY:")
    print(f"   Total Routes Created: {total_routes}")
    print(f"   Users Assigned: {total_assigned}")
    print(f"   Users Unassigned: {total_unassigned}")
    print(f"   Available Drivers: {len(result['unassignedDrivers'])}")
    print(f"   Assignment Rate: {(total_assigned / (total_assigned + total_unassigned) * 100):.1f}%")

    # Route utilization analysis
    if result["data"]:
        utilizations = [len(route["assigned_users"]) / route["vehicle_type"] for route in result["data"]]
        print(f"\n🚗 ROUTE UTILIZATION:")
        print(f"   Average: {(sum(utilizations) / len(utilizations) * 100):.1f}%")
        print(f"   Range: {(min(utilizations) * 100):.1f}% - {(max(utilizations) * 100):.1f}%")
        print(f"   Routes >80%: {sum(1 for u in utilizations if u > 0.8)}")
        print(f"   Routes <60%: {sum(1 for u in utilizations if u < 0.6)}")

    # Driver analysis - dynamic based on actual data
    if result["data"]:
        print(f"\n👨‍💼 DRIVER ANALYSIS:")
        driver_capacities = {}
        for route in result["data"]:
            capacity = route["vehicle_type"]
            if capacity not in driver_capacities:
                driver_capacities[capacity] = []
            driver_capacities[capacity].append(len(route["assigned_users"]))

        for capacity, assignments in driver_capacities.items():
            avg_assigned = sum(assignments) / len(assignments)
            utilization = avg_assigned / capacity * 100
            print(f"   Capacity {capacity}: {len(assignments)} drivers, {utilization:.1f}% avg utilization")

    # System metrics
    print(f"\n⚙️ SYSTEM METRICS:")
    print(f"   Execution Time: {result.get('execution_time', 0):.2f}s")
    print(f"   Optimization Mode: {result.get('optimization_mode', 'unknown').upper()}")
    print(f"   Clustering Method: {result.get('clustering_analysis', {}).get('method', 'unknown')}")

def start_fastapi():
    """Start FastAPI server"""
    try:
        subprocess.run(["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "3000"], check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to start FastAPI server: {e}")
    except KeyboardInterrupt:
        logger.info("FastAPI server stopped by user")

def launch_browser():
    """Launch browser to visualization page"""
    time.sleep(3)  # Wait for server to start
    try:
        webbrowser.open("http://0.0.0.0:3000/visualize")
        logger.info("Browser launched successfully")
    except Exception as e:
        logger.warning(f"Could not launch browser: {e}")

def main():
    """Main function with improved error handling"""
    print("🚀 DRIVER ASSIGNMENT SYSTEM v2.0")
    print("=" * 50)

    # Get input parameters with hardcoded defaults
    source_id = input("Enter source_id (default: UC_unify_dev): ").strip() or "UC_unify_dev"
    
    try:
        parameter_input = input("Enter parameter (default 1): ").strip()
        parameter = int(parameter_input) if parameter_input else 1
    except ValueError:
        parameter = 1

    string_param = input("Enter string parameter (default: Evening shift): ").strip() or "Evening shift"

    print(f"\n📋 Configuration:")
    print(f"   Source ID: {source_id}")
    print(f"   Parameter: {parameter}")
    print(f"   String Parameter: '{string_param}'")

    print(f"\n🔄 Running assignment...")

    try:
        # Run assignment with automatic type detection
        result = run_assignment(source_id, parameter, string_param)

        if result["status"] == "true":
            print(f"\n✅ Assignment completed successfully!")
            print_detailed_analytics(result)

            # Save route data for visualization
            with open("drivers_and_routes.json", "w") as f:
                json.dump(result["data"], f, indent=2)
            print(f"📁 Route data saved to drivers_and_routes.json")

            # Quality analysis
            quality_analysis = analyze_assignment_quality(result)
            if isinstance(quality_analysis, dict):
                print(f"\n🎯 QUALITY METRICS:")
                print(f"   Assignment Rate: {quality_analysis.get('assignment_rate', 0)}%")
                print(f"   Routes Below 80% Utilization: {quality_analysis.get('routes_below_80_percent', 0)}")
                if quality_analysis.get('distance_issues'):
                    print(f"   Distance Issues: {len(quality_analysis['distance_issues'])}")

        else:
            print("❌ Assignment failed:")
            print(f"   Error: {result.get('error', 'Unknown error')}")
            print(f"   Details: {result.get('details', 'No details available')}")
            return

        print("\n🚀 Launching Dashboard...")
        print("   - Starting FastAPI server on port 3000")
        print("   - Opening browser automatically")

        # Start server in background
        server_thread = threading.Thread(target=start_fastapi, daemon=True)
        server_thread.start()

        # Launch browser
        browser_thread = threading.Thread(target=launch_browser, daemon=True)
        browser_thread.start()

        print("\n✅ Dashboard is starting up...")
        print("📱 Manual URL: http://localhost:3000/visualize")
        print("📊 API Endpoint: http://localhost:3000/routes")
        print("\n🔄 Server running... Press Ctrl+C to stop")

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n\n👋 Shutting down gracefully...")

    except ValidationError as e:
        print(f"❌ Data validation error: {e}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    main()