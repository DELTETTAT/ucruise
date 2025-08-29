
import os
import json
import requests
from datetime import datetime
from dotenv import load_dotenv
from assignment import run_assignment

def capture_api_response(source_id, parameter, string_param):
    """
    Capture the raw API response when fetching data
    """
    print("🌐 CAPTURING API REQUEST/RESPONSE")
    print("="*60)
    
    # Load environment variables
    if not os.path.exists(".env"):
        print("❌ ERROR: .env file not found!")
        return None
    
    load_dotenv(".env")
    BASE_API_URL = os.getenv("API_URL")
    API_AUTH_TOKEN = os.getenv("API_AUTH_TOKEN")
    
    if not BASE_API_URL or not API_AUTH_TOKEN:
        print("❌ Missing API_URL or API_AUTH_TOKEN in .env file")
        return None
    
    # Construct API URL
    API_URL = f"{BASE_API_URL.rstrip('/')}/{source_id}/{parameter}/{string_param}"
    
    headers = {
        "Authorization": f"Bearer {API_AUTH_TOKEN}",
        "Content-Type": "application/json"
    }
    
    print(f"📡 API Request Details:")
    print(f"   URL: {API_URL}")
    print(f"   Headers: {json.dumps(headers, indent=2)}")
    print(f"   Method: GET")
    print(f"   Timestamp: {datetime.now().isoformat()}")
    
    try:
        print("\n⏳ Making API request...")
        response = requests.get(API_URL, headers=headers, timeout=30)
        
        print(f"\n📊 API Response Details:")
        print(f"   Status Code: {response.status_code}")
        print(f"   Response Time: {response.elapsed.total_seconds():.2f} seconds")
        print(f"   Content-Type: {response.headers.get('content-type', 'unknown')}")
        print(f"   Content-Length: {len(response.text)} characters")
        
        if response.status_code == 200:
            try:
                response_data = response.json()
                print(f"   JSON Structure: Valid")
                
                # Analyze response structure
                if isinstance(response_data, dict):
                    print(f"   Top-level keys: {list(response_data.keys())}")
                    
                    # Check for users
                    users = response_data.get('users', [])
                    print(f"   Users count: {len(users)}")
                    
                    # Check for drivers structure
                    if 'drivers' in response_data:
                        drivers = response_data['drivers']
                        drivers_unassigned = drivers.get('driversUnassigned', [])
                        drivers_assigned = drivers.get('driversAssigned', [])
                        print(f"   Drivers Unassigned: {len(drivers_unassigned)}")
                        print(f"   Drivers Assigned: {len(drivers_assigned)}")
                    else:
                        drivers_unassigned = response_data.get('driversUnassigned', [])
                        drivers_assigned = response_data.get('driversAssigned', [])
                        print(f"   Drivers Unassigned (flat): {len(drivers_unassigned)}")
                        print(f"   Drivers Assigned (flat): {len(drivers_assigned)}")
                
                return {
                    "request": {
                        "url": API_URL,
                        "method": "GET",
                        "headers": headers,
                        "timestamp": datetime.now().isoformat(),
                        "source_id": source_id,
                        "parameter": parameter,
                        "string_param": string_param
                    },
                    "response": {
                        "status_code": response.status_code,
                        "response_time_seconds": response.elapsed.total_seconds(),
                        "content_type": response.headers.get('content-type', 'unknown'),
                        "content_length": len(response.text),
                        "data": response_data
                    }
                }
                
            except json.JSONDecodeError as e:
                print(f"   JSON Structure: Invalid - {e}")
                return {
                    "request": {
                        "url": API_URL,
                        "method": "GET",
                        "headers": headers,
                        "timestamp": datetime.now().isoformat()
                    },
                    "response": {
                        "status_code": response.status_code,
                        "response_time_seconds": response.elapsed.total_seconds(),
                        "content_type": response.headers.get('content-type', 'unknown'),
                        "content_length": len(response.text),
                        "raw_text": response.text,
                        "json_error": str(e)
                    }
                }
        else:
            print(f"   Error Response: {response.text[:500]}")
            return {
                "request": {
                    "url": API_URL,
                    "method": "GET", 
                    "headers": headers,
                    "timestamp": datetime.now().isoformat()
                },
                "response": {
                    "status_code": response.status_code,
                    "response_time_seconds": response.elapsed.total_seconds(),
                    "content_type": response.headers.get('content-type', 'unknown'),
                    "error_text": response.text
                }
            }
            
    except Exception as e:
        print(f"❌ Request failed: {e}")
        return {
            "request": {
                "url": API_URL,
                "method": "GET",
                "headers": headers,
                "timestamp": datetime.now().isoformat()
            },
            "response": {
                "error": str(e),
                "error_type": type(e).__name__
            }
        }

def main():
    """
    Test script to capture API request/response and assignment results
    """
    print("🧪 API REQUEST/RESPONSE CAPTURE TEST")
    print("="*60)
    
    # Configuration
    source_id = "UC_frontdev"  # Update this to match your API format
    parameter = 1
    string_param = "Evening%20shift"  # Use plain text, let requests handle URL encoding
    
    print(f"📋 Test Configuration:")
    print(f"   Source ID: {source_id}")
    print(f"   Parameter: {parameter}")
    print(f"   String Parameter: {string_param}")
    print(f"   Current Directory: {os.getcwd()}")
    print(f"   .env File Exists: {os.path.exists('.env')}")
    
    # Step 1: Capture API request/response
    print(f"\n" + "="*60)
    print("STEP 1: CAPTURING RAW API DATA")
    print("="*60)
    
    api_capture = capture_api_response(source_id, parameter, string_param)
    
    if not api_capture:
        print("❌ Failed to capture API data")
        return
    
    # Step 2: Run assignment and capture result
    print(f"\n" + "="*60)
    print("STEP 2: RUNNING ASSIGNMENT ALGORITHM")
    print("="*60)
    
    print("🚀 Starting assignment algorithm...")
    assignment_start_time = datetime.now()
    
    try:
        assignment_result = run_assignment(source_id, parameter, string_param)
        assignment_end_time = datetime.now()
        assignment_duration = (assignment_end_time - assignment_start_time).total_seconds()
        
        print(f"✅ Assignment completed in {assignment_duration:.2f} seconds")
        print(f"📊 Assignment Status: {assignment_result.get('status', 'unknown')}")
        
        if assignment_result.get('status') == 'true':
            routes = assignment_result.get('data', [])
            unassigned_users = assignment_result.get('unassignedUsers', [])
            unassigned_drivers = assignment_result.get('unassignedDrivers', [])
            
            print(f"   Routes Created: {len(routes)}")
            print(f"   Users Assigned: {sum(len(route.get('assigned_users', [])) for route in routes)}")
            print(f"   Users Unassigned: {len(unassigned_users)}")
            print(f"   Drivers Used: {len(routes)}")
            print(f"   Drivers Unused: {len(unassigned_drivers)}")
        else:
            print(f"   Assignment Error: {assignment_result.get('details', 'Unknown error')}")
    
    except Exception as e:
        assignment_end_time = datetime.now()
        assignment_duration = (assignment_end_time - assignment_start_time).total_seconds()
        print(f"❌ Assignment failed after {assignment_duration:.2f} seconds: {e}")
        
        assignment_result = {
            "status": "false",
            "error": str(e),
            "error_type": type(e).__name__,
            "data": []
        }
    
    # Step 3: Compile complete test report
    print(f"\n" + "="*60)
    print("STEP 3: GENERATING COMPLETE TEST REPORT")
    print("="*60)
    
    complete_report = {
        "test_metadata": {
            "test_name": "API Request/Response Capture Test",
            "timestamp": datetime.now().isoformat(),
            "source_id": source_id,
            "parameter": parameter,
            "string_param": string_param,
            "test_duration_seconds": (datetime.now() - assignment_start_time).total_seconds()
        },
        "api_data_capture": api_capture,
        "assignment_execution": {
            "start_time": assignment_start_time.isoformat(),
            "end_time": assignment_end_time.isoformat(),
            "duration_seconds": assignment_duration,
            "result": assignment_result
        },
        "summary": {
            "api_request_successful": api_capture.get('response', {}).get('status_code') == 200,
            "assignment_successful": assignment_result.get('status') == 'true',
            "total_api_response_size": api_capture.get('response', {}).get('content_length', 0),
            "routes_generated": len(assignment_result.get('data', [])),
            "users_from_api": len(api_capture.get('response', {}).get('data', {}).get('users', [])),
            "drivers_from_api": (
                len(api_capture.get('response', {}).get('data', {}).get('driversUnassigned', [])) +
                len(api_capture.get('response', {}).get('data', {}).get('driversAssigned', []))
            )
        }
    }
    
    # Save to JSON file
    output_filename = f"api_capture_test_{source_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    try:
        with open(output_filename, 'w') as f:
            json.dump(complete_report, f, indent=2, default=str)
        
        print(f"✅ Complete test report saved to: {output_filename}")
        print(f"📄 File size: {os.path.getsize(output_filename)} bytes")
        
        # Display summary
        print(f"\n📊 TEST SUMMARY:")
        print(f"   API Status: {'✅ Success' if complete_report['summary']['api_request_successful'] else '❌ Failed'}")
        print(f"   Assignment Status: {'✅ Success' if complete_report['summary']['assignment_successful'] else '❌ Failed'}")
        print(f"   API Response Size: {complete_report['summary']['total_api_response_size']} characters")
        print(f"   Users from API: {complete_report['summary']['users_from_api']}")
        print(f"   Drivers from API: {complete_report['summary']['drivers_from_api']}")
        print(f"   Routes Generated: {complete_report['summary']['routes_generated']}")
        
        print(f"\n🎯 REPORT SECTIONS:")
        print(f"   • test_metadata: General test information")
        print(f"   • api_data_capture: Raw API request/response data")
        print(f"   • assignment_execution: Assignment algorithm results")
        print(f"   • summary: Key metrics and status")
        
    except Exception as e:
        print(f"❌ Failed to save report: {e}")
        print(f"📋 Report data available in memory but not saved")

if __name__ == "__main__":
    main()
