import os
from analysis import comprehensive_analysis

def main():
    """
    Test the analysis with different source IDs
    """
    # Check if .env file exists
    if not os.path.exists(".env"):
        print("❌ ERROR: .env file not found!")
        print("Please create a .env file with:")
        print("API_URL=your_api_url_here")
        print("API_AUTH_TOKEN=your_token_here")
        return

    # Replace with your actual values
    source_id = "UC_unify_test"  # Update this to match your API format
    parameter = 1  # Update this to your parameter value
    string_param = "Morning%20shift"  # Update this to your string parameter

    print("🧪 TESTING COMPREHENSIVE ANALYSIS")
    print("="*50)
    print(f"Testing with source_id: {source_id}")
    print(f"Parameter: {parameter}")
    print(f"String parameter: {string_param}")
    print(f"Current directory: {os.getcwd()}")
    print(f".env file exists: {os.path.exists('.env')}")

    # Run comprehensive analysis
    result = comprehensive_analysis(source_id, parameter, string_param, f"test_analysis_{source_id}.json")

    if result.get("error"):
        print(f"❌ Test failed: {result['error']}")
    else:
        print("✅ Test completed successfully!")
        print("📄 Analysis report saved to:", f"test_analysis_{source_id}.json")

if __name__ == "__main__":
    main()