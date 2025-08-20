#!/usr/bin/env python3
"""
Test Setup Script

This script tests if your Gemini AI Data Science Agent is properly configured
and all API keys are working.
"""

import os
import sys

def test_api_keys():
    """Test if API keys are properly set"""
    print("🔑 Testing API Keys...")
    
    # Check if API keys are set
    gemini_key = os.getenv('GEMINI_API_KEY')
    openai_key = os.getenv('OPENAI_API_KEY')
    huggingface_key = os.getenv('HUGGINGFACE_API_KEY')
    
    if gemini_key:
        print("✅ Gemini API Key: Found")
    else:
        print("❌ Gemini API Key: Not found")
        return False
    
    if openai_key:
        print("✅ OpenAI API Key: Found")
    else:
        print("⚠️  OpenAI API Key: Not found (optional)")
    
    if huggingface_key:
        print("✅ Hugging Face API Key: Found")
    else:
        print("⚠️  Hugging Face API Key: Not found (optional)")
    
    return True

def test_imports():
    """Test if all required packages can be imported"""
    print("\n📦 Testing Package Imports...")
    
    try:
        import pandas as pd
        print("✅ pandas: Imported")
    except ImportError:
        print("❌ pandas: Failed to import")
        return False
    
    try:
        import numpy as np
        print("✅ numpy: Imported")
    except ImportError:
        print("❌ numpy: Failed to import")
        return False
    
    try:
        import streamlit as st
        print("✅ streamlit: Imported")
    except ImportError:
        print("❌ streamlit: Failed to import")
        return False
    
    try:
        import plotly.express as px
        print("✅ plotly: Imported")
    except ImportError:
        print("❌ plotly: Failed to import")
        return False
    
    try:
        import google.generativeai as genai
        print("✅ google-generativeai: Imported")
    except ImportError:
        print("❌ google-generativeai: Failed to import")
        return False
    
    try:
        from sklearn.datasets import load_iris
        print("✅ scikit-learn: Imported")
    except ImportError:
        print("❌ scikit-learn: Failed to import")
        return False
    
    return True

def test_gemini_connection():
    """Test if Gemini API is working"""
    print("\n🤖 Testing Gemini API Connection...")
    
    try:
        import google.generativeai as genai
        
        # Configure Gemini
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            print("❌ No Gemini API key found")
            return False
        
        genai.configure(api_key=api_key)
        
        # Test with a simple prompt
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        response = model.generate_content("Hello! Can you respond with 'Gemini is working!'?")
        
        if response.text:
            print("✅ Gemini API: Working")
            print(f"   Response: {response.text}")
            return True
        else:
            print("❌ Gemini API: No response received")
            return False
            
    except Exception as e:
        print(f"❌ Gemini API: Error - {str(e)}")
        return False

def test_sample_data():
    """Test if sample datasets can be loaded"""
    print("\n📊 Testing Sample Data Loading...")
    
    try:
        from sklearn.datasets import load_iris, load_diabetes
        
        # Load Iris dataset
        iris = load_iris()
        print(f"✅ Iris Dataset: Loaded ({iris.data.shape})")
        
        # Load Diabetes dataset
        diabetes = load_diabetes()
        print(f"✅ Diabetes Dataset: Loaded ({diabetes.data.shape})")
        
        return True
        
    except Exception as e:
        print(f"❌ Sample Data: Error - {str(e)}")
        return False

def main():
    """Main test function"""
    print("🧪 Gemini AI Data Science Agent - Setup Test")
    print("=" * 50)
    
    # Test API keys
    if not test_api_keys():
        print("\n❌ API keys not found. Please run 'python api_keys_config.py' first.")
        return
    
    # Test imports
    if not test_imports():
        print("\n❌ Some packages failed to import. Please run 'pip install -r requirements.txt'")
        return
    
    # Test Gemini connection
    if not test_gemini_connection():
        print("\n❌ Gemini API connection failed. Please check your API key.")
        return
    
    # Test sample data
    if not test_sample_data():
        print("\n❌ Sample data loading failed.")
        return
    
    print("\n" + "=" * 50)
    print("🎉 All tests passed! Your setup is complete!")
    print("=" * 50)
    print("\n🚀 You can now:")
    print("   1. Run the web interface: streamlit run app.py")
    print("   2. Test the example: python example_usage.py")
    print("   3. Start analyzing your data!")
    print("\n📚 For more information, check the README.md file")

if __name__ == "__main__":
    main()
