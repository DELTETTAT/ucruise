
import os
import requests
import json
from dotenv import load_dotenv
from datetime import datetime

def dump_api_response(source_id: str, parameter: int = 1, string_param: str = "Evening%20shift"):
    """Dump the raw API response to a file for inspection"""
    
    # Load environment variables
    env_path = ".env"
    if os.path.exists(env_path):
        load_dotenv(env_path)
    else:
        print("❌ .env file not found")
        return
    
    BASE_API_URL = os.getenv("API_URL")
    API_AUTH_TOKEN = os.getenv("API_AUTH_TOKEN")
    
    if not BASE_API_URL or not API_AUTH_TOKEN:
        print("❌ API_URL and API_AUTH_TOKEN must be set in .env")
        return
    
    # Construct API URL
    API_URL = f"{BASE_API_URL.rstrip('/')}/{source_id}/{parameter}/{string_param}"
    
    headers = {
        "Authorization": f"Bearer {API_AUTH_TOKEN}",
        "Content-Type": "application/json"
    }
    
    print(f"📡 Making API call to: {API_URL}")
    print(f"🔑 Using auth token: {API_AUTH_TOKEN[:10]}...")
    
    try:
        # Make the API call
        resp = requests.get(API_URL, headers=headers)
        
        # Create timestamp for filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"api_response_dump_{timestamp}.json"
        
        # Prepare response data for dumping
        response_data = {
            "request_info": {
                "url": API_URL,
                "method": "GET",
                "headers": {
                    "Authorization": f"Bearer {API_AUTH_TOKEN[:10]}...",
                    "Content-Type": "application/json"
                },
                "parameters": {
                    "source_id": source_id,
                    "parameter": parameter,
                    "string_param": string_param
                },
                "timestamp": datetime.now().isoformat()
            },
            "response_info": {
                "status_code": resp.status_code,
                "headers": dict(resp.headers),
                "content_type": resp.headers.get('content-type', 'unknown'),
                "content_length": len(resp.text)
            }
        }
        
        # Try to parse JSON response
        try:
            json_response = resp.json()
            response_data["response_body"] = json_response
            response_data["json_parse_success"] = True
        except json.JSONDecodeError as e:
            response_data["response_body_raw"] = resp.text
            response_data["json_parse_success"] = False
            response_data["json_parse_error"] = str(e)
        
        # Save to file
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(response_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ API response dumped to: {filename}")
        print(f"📊 Status Code: {resp.status_code}")
        print(f"📏 Content Length: {len(resp.text)} characters")
        print(f"🗂️ Content Type: {resp.headers.get('content-type', 'unknown')}")
        
        if response_data["json_parse_success"]:
            json_data = response_data["response_body"]
            if isinstance(json_data, dict):
                print(f"📋 Response Structure:")
                for key in json_data.keys():
                    if isinstance(json_data[key], list):
                        print(f"   - {key}: {len(json_data[key])} items")
                    elif isinstance(json_data[key], dict):
                        print(f"   - {key}: {len(json_data[key])} keys")
                    else:
                        print(f"   - {key}: {type(json_data[key]).__name__}")
        else:
            print(f"⚠️ JSON parsing failed - raw response saved")
        
        return filename
        
    except requests.exceptions.RequestException as e:
        print(f"❌ API request failed: {e}")
        
        # Still dump the error information
        error_data = {
            "request_info": {
                "url": API_URL,
                "method": "GET",
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            },
            "error_type": type(e).__name__
        }
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        error_filename = f"api_error_dump_{timestamp}.json"
        
        with open(error_filename, "w") as f:
            json.dump(error_data, f, indent=2)
        
        print(f"❌ Error details saved to: {error_filename}")
        return error_filename

if __name__ == "__main__":
    # Use the same parameters as in run_and_view.py
    SOURCE_ID = "UC_unify_dev"
    PARAMETER = 1
    STRING_PARAM = "Evening%20shift"
    
    print("🔍 API Response Dumper")
    print("=" * 50)
    print(f"Source ID: {SOURCE_ID}")
    print(f"Parameter: {PARAMETER}")
    print(f"String Param: {STRING_PARAM}")
    print("=" * 50)
    
    dump_api_response(SOURCE_ID, PARAMETER, STRING_PARAM)
