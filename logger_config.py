import logging
import os
from datetime import datetime
import json

class RouteAssignmentLogger:
    def __init__(self, log_dir="logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        # Create timestamp for this session with milliseconds for uniqueness
        self.session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]

        # Setup main logger
        self.logger = logging.getLogger('route_assignment')
        self.logger.setLevel(logging.DEBUG)

        # Clear existing handlers
        self.logger.handlers.clear()

        # Create file handler with UTF-8 encoding
        log_file = os.path.join(log_dir, f"assignment_{self.session_timestamp}.log")
        file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)

        # Create detailed formatter
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)8s | %(funcName)20s:%(lineno)4d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        # All tracking will go to the main log file
        self.tracking_file = log_file

        self.log_session_start()

    def log_session_start(self):
        self.logger.info("="*80)
        self.logger.info("ROUTE ASSIGNMENT SESSION STARTED")
        self.logger.info(f"Session ID: {self.session_timestamp}")
        self.logger.info("="*80)

    def info(self, message):
        """Add info method for compatibility"""
        self.logger.info(message)

    def warning(self, message):
        """Add warning method for compatibility"""
        self.logger.warning(message)

    def error(self, message, exc_info=False):
        """Add error method for compatibility"""
        self.logger.error(message, exc_info=exc_info)

    def critical(self, message):
        """Add critical method for compatibility"""
        self.logger.critical(message)

    def debug(self, message):
        """Add debug method for compatibility"""
        self.logger.debug(message)

    def log_data_validation(self, users_count, drivers_count, office_coords):
        self.logger.info(f"DATA VALIDATION - Users: {users_count}, Drivers: {drivers_count}")
        self.logger.info(f"Office coordinates: {office_coords}")

    def log_clustering_decision(self, method, user_count, cluster_count, details):
        self.logger.info(f"CLUSTERING - Method: {method}, Users: {user_count}, Clusters: {cluster_count}")
        self.logger.debug(f"Clustering details: {details}")

    def log_route_creation(self, driver_id, users, reason, quality_metrics):
        self.logger.info(f"ROUTE CREATED - Driver: {driver_id}, Users: {len(users)}")
        self.logger.info(f"Creation reason: {reason}")
        for user in users:
            self.logger.debug(f"  User {user.get('user_id', 'N/A')} at ({user.get('lat', 'N/A')}, {user.get('lng', 'N/A')})")
        self.logger.debug(f"Quality metrics: {quality_metrics}")

    def log_route_rejection(self, driver_id, users, reason):
        self.logger.warning(f"ROUTE REJECTED - Driver: {driver_id}, Reason: {reason}")
        for user in users:
            self.logger.debug(f"  Rejected user {user.get('user_id', 'N/A')}")

    def log_user_assignment(self, user_id, driver_id, route_details):
        self.logger.info(f"USER ASSIGNED - User: {user_id} -> Driver: {driver_id}")
        self.logger.debug(f"Route details: {route_details}")

    def log_user_unassigned(self, user_id, reason, attempted_drivers):
        self.logger.warning(f"USER UNASSIGNED - User: {user_id}, Reason: {reason}")
        self.logger.debug(f"Attempted drivers: {attempted_drivers}")

        # Additional tracking info in main log
        self.logger.info(f"TRACKING | UNASSIGNED_USER | {user_id} | {reason}")

    def log_driver_unused(self, driver_id, reason, capacity, location):
        self.logger.warning(f"DRIVER UNUSED - Driver: {driver_id}, Reason: {reason}")
        self.logger.debug(f"Capacity: {capacity}, Location: {location}")

        # Additional tracking info in main log
        self.logger.info(f"TRACKING | UNUSED_DRIVER | {driver_id} | {reason} | {capacity} | {location}")

    def log_optimization_step(self, step_name, before_state, after_state, changes):
        self.logger.info(f"OPTIMIZATION - {step_name}")
        self.logger.debug(f"Before: {before_state}")
        self.logger.debug(f"After: {after_state}")
        self.logger.info(f"Changes made: {changes}")

    def log_final_summary(self, total_users, assigned_users, unassigned_users, 
                         total_drivers, used_drivers, unused_drivers, routes):
        self.logger.info("="*80)
        self.logger.info("FINAL ASSIGNMENT SUMMARY")
        self.logger.info(f"Users: {assigned_users}/{total_users} assigned ({len(unassigned_users)} unassigned)")
        self.logger.info(f"Drivers: {used_drivers}/{total_drivers} used ({len(unused_drivers)} unused)")
        self.logger.info(f"Routes created: {len(routes)}")

        # Detailed unassigned analysis
        if unassigned_users:
            self.logger.warning(f"UNASSIGNED USERS ANALYSIS ({len(unassigned_users)} users):")
            for user in unassigned_users:
                self.logger.warning(f"  User {user.get('user_id', 'N/A')} at ({user.get('lat', 'N/A')}, {user.get('lng', 'N/A')})")

        if unused_drivers:
            self.logger.warning(f"UNUSED DRIVERS ANALYSIS ({len(unused_drivers)} drivers):")
            for driver in unused_drivers:
                self.logger.warning(f"  Driver {driver.get('driver_id', 'N/A')} capacity {driver.get('capacity', 'N/A')}")

        self.logger.logger.info("="*80)

    def log_accounting_check(self, api_users, final_assigned, final_unassigned, discrepancy):
        self.logger.critical("USER ACCOUNTING CHECK")
        self.logger.critical(f"API Users: {api_users}")
        self.logger.critical(f"Final Assigned: {final_assigned}")
        self.logger.critical(f"Final Unassigned: {final_unassigned}")
        self.logger.critical(f"Total Accounted: {final_assigned + final_unassigned}")
        if discrepancy:
            self.logger.critical(f"DISCREPANCY DETECTED: {discrepancy} users missing!")
        else:
            self.logger.info("User accounting is correct")

# Global logger instance and session tracking
route_logger = None
current_session_id = None

def get_logger():
    global route_logger
    if route_logger is None:
        route_logger = RouteAssignmentLogger()
    return route_logger

def reset_logger():
    """Reset logger for new session - only if not in an active session"""
    global route_logger, current_session_id
    
    # Don't reset if we're in the middle of a session
    if current_session_id is not None:
        return route_logger
    
    if route_logger is not None:
        # Close existing handlers before resetting
        for handler in route_logger.logger.handlers[:]:
            handler.close()
            route_logger.logger.removeHandler(handler)
    route_logger = None
    return None

def start_session():
    """Start a new logging session"""
    global current_session_id
    current_session_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    return get_logger()

def end_session():
    """End the current logging session"""
    global current_session_id
    current_session_id = None