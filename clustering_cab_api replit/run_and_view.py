import subprocess
import threading
import time
import webbrowser
from assignment import run_assignment

SOURCE_ID = "UC_logisticllp"  # <-- Replace with your real source_id

def start_fastapi():
    subprocess.run(["uvicorn", "main:app", "--reload"])

def start_html_server():
    subprocess.run(["python", "-m", "http.server", "8080"])

def launch_browser():
    time.sleep(3)  # Wait for servers to start
    webbrowser.open("http://localhost:8080/visualize.html")

if __name__ == "__main__":
    print("✅ Running assignment...")
    try:
        result = run_assignment(SOURCE_ID)

        if result["status"] == "true":
            print("✅ Assignment successful. Saving drivers_and_routes.json...")
            import json
            with open("drivers_and_routes.json", "w") as f:
                json.dump(result["data"], f, indent=2)

            print(f"📊 Assignment Summary:")
            print(f"   Routes created: {len(result['data'])}")
            print(f"   Unassigned users: {len(result.get('unassignedUsers', []))}")
            print(f"   Unassigned drivers: {len(result.get('unassignedDrivers', []))}")
        else:
            print("❌ Assignment failed:")
            print(f"   Error: {result.get('details', 'Unknown error')}")
            exit(1)

        print("🚀 Launching servers and browser...")

        threading.Thread(target=start_fastapi, daemon=True).start()
        threading.Thread(target=start_html_server, daemon=True).start()
        threading.Thread(target=launch_browser).start()

        input("Press Enter to exit...\n")
    except Exception as e:
        print(f"❌ Error running assignment: {e}")
        exit(1)