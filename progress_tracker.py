
import sys
import time
from datetime import datetime

class ProgressTracker:
    def __init__(self):
        self.start_time = None
        self.current_stage = ""
        self.stages = [
            "Data Loading & Validation",
            "Geographic Clustering", 
            "Capacity Sub-clustering",
            "Driver Assignment",
            "Local Optimization",
            "Global Optimization",
            "Final Merge & Validation"
        ]
        self.current_stage_index = 0
        self.stage_details = {}
        
    def start_assignment(self, source_id, mode):
        self.start_time = datetime.now()
        print(f"\n🚀 Starting {mode} Assignment")
        print(f"📋 Source ID: {source_id}")
        print(f"⏰ Started at: {self.start_time.strftime('%H:%M:%S')}")
        print("="*60)
    
    def start_stage(self, stage_name, details=""):
        self.current_stage = stage_name
        if stage_name in self.stages:
            self.current_stage_index = self.stages.index(stage_name) + 1
        
        # Progress bar
        progress = (self.current_stage_index / len(self.stages)) * 100
        bar_length = 30
        filled_length = int(bar_length * self.current_stage_index // len(self.stages))
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        
        print(f"\n📊 Stage {self.current_stage_index}/{len(self.stages)}: {stage_name}")
        print(f"[{bar}] {progress:.1f}%")
        if details:
            print(f"   {details}")
        
        self.stage_details[stage_name] = {
            'start_time': datetime.now(),
            'details': details
        }
    
    def update_stage_progress(self, message):
        elapsed = datetime.now() - self.stage_details.get(self.current_stage, {}).get('start_time', datetime.now())
        print(f"   ⏳ {message} (Elapsed: {elapsed.total_seconds():.1f}s)")
    
    def complete_stage(self, summary):
        if self.current_stage in self.stage_details:
            elapsed = datetime.now() - self.stage_details[self.current_stage]['start_time']
            print(f"   ✅ {summary} (Completed in {elapsed.total_seconds():.1f}s)")
    
    def show_final_summary(self, result):
        total_time = datetime.now() - self.start_time
        
        print("\n" + "="*60)
        print("🎯 ASSIGNMENT COMPLETED")
        print(f"⏱️  Total Time: {total_time.total_seconds():.1f} seconds")
        
        if result.get("status") == "true":
            routes = result.get("data", [])
            unassigned_users = result.get("unassignedUsers", [])
            unassigned_drivers = result.get("unassignedDrivers", [])
            
            total_users = sum(len(r.get('assigned_users', [])) for r in routes) + len(unassigned_users)
            total_assigned = sum(len(r.get('assigned_users', [])) for r in routes)
            total_routes = len(routes)
            
            print(f"📈 Routes Created: {total_routes}")
            print(f"👥 Users Assigned: {total_assigned}/{total_users}")
            print(f"🚗 Drivers Used: {len(routes)}")
            
            if unassigned_users:
                print(f"⚠️  Unassigned Users: {len(unassigned_users)}")
            if unassigned_drivers:
                print(f"⚠️  Unused Drivers: {len(unassigned_drivers)}")
            
            # Calculate utilization
            total_capacity = sum(r.get('vehicle_type', 0) for r in routes)
            utilization = (total_assigned / total_capacity * 100) if total_capacity > 0 else 0
            print(f"📊 Overall Utilization: {utilization:.1f}%")
            
        else:
            print(f"❌ Assignment Failed: {result.get('details', 'Unknown error')}")
        
        print("="*60)
        print(f"📄 Detailed logs saved in logs/assignment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

# Global progress tracker
progress_tracker = ProgressTracker()

def get_progress_tracker():
    return progress_tracker
