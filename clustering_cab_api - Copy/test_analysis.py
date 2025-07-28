
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
    
    # Replace with your actual source_id
    source_id = "UC_healthcarellp"
    
    print("🧪 TESTING COMPREHENSIVE ANALYSIS")
    print("="*50)
    print(f"Testing with source_id: {source_id}")
    print(f"Current directory: {os.getcwd()}")
    print(f".env file exists: {os.path.exists('.env')}")
    
    # Run comprehensive analysis
    result = comprehensive_analysis(source_id, f"test_analysis_{source_id}.json")
    
    if result.get("error"):
        print(f"❌ Test failed: {result['error']}")
        print("\n🔧 Troubleshooting steps:")
        print("1. Check if your .env file has correct API_URL and API_AUTH_TOKEN")
        print("2. Verify the source_id is correct")
        print("3. Test the API endpoint manually")
        print("4. Check if the API is returning valid JSON")
    else:
        print("✅ Test completed successfully!")
        print("📄 Analysis report saved to:", f"test_analysis_{source_id}.json")

if __name__ == "__main__":
    main()
