import subprocess
import threading
import time
import webbrowser
from assignment import run_assignment

SOURCE_ID = "UC_healthcarellp"  # <-- Replace with your real source_id

def start_fastapi():
    subprocess.run(["uvicorn", "main:app", "--reload"])

def start_html_server():
    subprocess.run(["python", "-m", "http.server", "8080"])

def launch_browser():
    time.sleep(3)  # Wait for servers to start
    webbrowser.open("http://localhost:8080/visualize.html")

if __name__ == "__main__":
    print("✅ Running assignment...")
    result = run_assignment(SOURCE_ID)

    if result["status"] == "true":
        print("✅ Assignment successful. Saving drivers_and_routes.json...")
        import json
        with open("drivers_and_routes.json", "w") as f:
            json.dump(result["data"], f, indent=2)

        print("🚀 Launching servers and browser...")

        threading.Thread(target=start_fastapi, daemon=True).start()
        threading.Thread(target=start_html_server, daemon=True).start()
        threading.Thread(target=launch_browser).start()

        input("Press Enter to exit...\n")
    else:
        print("❌ Assignment failed:")
        print(result["details"])
