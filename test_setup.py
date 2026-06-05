#!/usr/bin/env python3
"""
Test script to verify API keys and basic functionality
"""
import os
from dotenv import load_dotenv

load_dotenv()

def test_api_keys():
    """Check if required API keys are configured"""
    print("🔍 Checking API Keys...\n")

    # Required keys
    gemini_key = os.getenv("GEMINI_API_KEY")
    serpapi_key = os.getenv("SERPAPI_KEY")

    # Optional keys
    ieee_key = os.getenv("IEEE_API_KEY")
    searchapi_key = os.getenv("SEARCHAPI_KEY")
    admin_pwd = os.getenv("NOVELTY_ADMIN_PASSWORD")

    all_good = True

    # Check required keys
    if gemini_key and gemini_key != "your_gemini_api_key_here":
        print("✅ GEMINI_API_KEY is configured")
    else:
        print("❌ GEMINI_API_KEY is missing or not configured")
        all_good = False

    if serpapi_key and serpapi_key != "your_serpapi_key_here":
        print("✅ SERPAPI_KEY is configured")
    else:
        print("❌ SERPAPI_KEY is missing or not configured")
        all_good = False

    # Check optional keys
    print("\n📋 Optional API Keys:")
    if ieee_key and ieee_key != "your_ieee_api_key_here":
        print("✅ IEEE_API_KEY is configured")
    else:
        print("⚠️  IEEE_API_KEY not configured (IEEE papers will be skipped)")

    if searchapi_key and searchapi_key != "your_searchapi_key_here":
        print("✅ SEARCHAPI_KEY is configured")
    else:
        print("⚠️  SEARCHAPI_KEY not configured (will use SerpAPI for patents)")

    if admin_pwd and admin_pwd != "your_admin_password_here":
        print("✅ NOVELTY_ADMIN_PASSWORD is configured")
    else:
        print("⚠️  NOVELTY_ADMIN_PASSWORD not configured (admin dashboard disabled)")

    return all_good

def test_imports():
    """Test if all required packages are installed"""
    print("\n\n📦 Checking Python Packages...\n")

    packages = [
        ("streamlit", "Streamlit"),
        ("pandas", "Pandas"),
        ("numpy", "NumPy"),
        ("requests", "Requests"),
        ("dotenv", "python-dotenv"),
        ("google.generativeai", "google-generativeai"),
        ("langchain", "LangChain"),
        ("langchain_google_genai", "langchain-google-genai"),
        ("sentence_transformers", "sentence-transformers"),
        ("faiss", "faiss-cpu"),
        ("reportlab", "ReportLab"),
    ]

    all_good = True
    for module_name, package_name in packages:
        try:
            __import__(module_name)
            print(f"✅ {package_name} is installed")
        except ImportError:
            print(f"❌ {package_name} is NOT installed")
            all_good = False

    return all_good

def test_gemini_connection():
    """Test Gemini API connection"""
    print("\n\n🤖 Testing Gemini AI Connection...\n")

    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key or api_key == "your_gemini_api_key_here":
            print("❌ GEMINI_API_KEY not configured, skipping connection test")
            return False

        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.3,
            google_api_key=api_key
        )

        response = llm.invoke("Say 'API test successful' in 3 words")
        print(f"✅ Gemini API is working!")
        print(f"   Response: {response.content}")
        return True

    except Exception as e:
        print(f"❌ Gemini API test failed: {str(e)}")
        return False

def test_serpapi_connection():
    """Test SerpAPI connection"""
    print("\n\n🔎 Testing SerpAPI Connection...\n")

    try:
        import requests
        api_key = os.getenv("SERPAPI_KEY")

        if not api_key or api_key == "your_serpapi_key_here":
            print("❌ SERPAPI_KEY not configured, skipping connection test")
            return False

        # Test with a simple search
        params = {
            "engine": "google",
            "q": "test",
            "api_key": api_key,
            "num": 1
        }

        r = requests.get("https://serpapi.com/search.json", params=params, timeout=10)

        if r.status_code == 200:
            data = r.json()
            print("✅ SerpAPI is working!")

            # Check account info if available
            if "search_metadata" in data:
                print(f"   Search ID: {data['search_metadata'].get('id', 'N/A')}")

            return True
        else:
            print(f"❌ SerpAPI returned status code: {r.status_code}")
            print(f"   Response: {r.text[:200]}")
            return False

    except Exception as e:
        print(f"❌ SerpAPI test failed: {str(e)}")
        return False

def main():
    print("=" * 60)
    print("   Patent & Research Novelty Checker - Setup Test")
    print("=" * 60)

    results = []

    # Test 1: API Keys
    results.append(test_api_keys())

    # Test 2: Package imports
    results.append(test_imports())

    # Test 3: Gemini connection
    results.append(test_gemini_connection())

    # Test 4: SerpAPI connection
    results.append(test_serpapi_connection())

    # Summary
    print("\n" + "=" * 60)
    print("   SUMMARY")
    print("=" * 60 + "\n")

    if all(results[:2]):  # At least keys and packages must be good
        if all(results[2:]):  # All API tests passed
            print("✅ All tests passed! Your setup is complete.")
            print("\n🚀 You can now run: streamlit run app.py")
        else:
            print("⚠️  Basic setup is complete, but some API tests failed.")
            print("   Check your API keys and network connection.")
            print("\n   You can still try running: streamlit run app.py")
    else:
        print("❌ Setup is incomplete. Please fix the issues above.")
        print("\n   1. Install missing packages: pip install -r requirements.txt")
        print("   2. Configure your .env file with API keys")

    print("\n" + "=" * 60 + "\n")

if __name__ == "__main__":
    main()
