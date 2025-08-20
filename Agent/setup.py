#!/usr/bin/env python3
"""
Setup script for Gemini AI Data Science Agent

This script helps users set up the environment and install dependencies.
"""

import os
import sys
import subprocess
import platform

def check_python_version():
    """Check if Python version is compatible"""
    print("🐍 Checking Python version...")
    
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    
    print(f"✅ Python {sys.version.split()[0]} is compatible")
    return True

def install_dependencies():
    """Install required dependencies"""
    print("\n📦 Installing dependencies...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    print("\n📁 Creating directories...")
    
    directories = ["outputs", "uploads", "cache"]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"✅ Created {directory}/ directory")
        else:
            print(f"✅ {directory}/ directory already exists")

def create_env_file():
    """Create .env file template"""
    print("\n🔧 Creating .env file template...")
    
    env_content = """# Gemini API Configuration
GEMINI_API_KEY=your_gemini_api_key_here

# Optional: OpenAI API Key (for comparison features)
OPENAI_API_KEY=your_openai_api_key_here

# Optional: Hugging Face API Key (for additional ML models)
HUGGINGFACE_API_KEY=your_huggingface_api_key_here
"""
    
    if not os.path.exists(".env"):
        with open(".env", "w") as f:
            f.write(env_content)
        print("✅ Created .env file template")
        print("📝 Please edit .env file and add your Gemini API key")
    else:
        print("✅ .env file already exists")

def check_gemini_api():
    """Check if Gemini API key is configured"""
    print("\n🔑 Checking Gemini API configuration...")
    
    try:
        from dotenv import load_dotenv
        load_dotenv()
        
        api_key = os.getenv('GEMINI_API_KEY')
        if api_key and api_key != 'your_gemini_api_key_here':
            print("✅ Gemini API key is configured")
            return True
        else:
            print("⚠️  Gemini API key not configured")
            print("📝 Please add your API key to the .env file")
            return False
    except ImportError:
        print("⚠️  python-dotenv not installed, checking environment directly")
        api_key = os.getenv('GEMINI_API_KEY')
        if api_key:
            print("✅ Gemini API key found in environment")
            return True
        else:
            print("⚠️  Gemini API key not found")
            return False

def test_installation():
    """Test if the installation works"""
    print("\n🧪 Testing installation...")
    
    try:
        # Test imports
        import pandas as pd
        import numpy as np
        import streamlit as st
        import plotly.express as px
        import google.generativeai as genai
        from sklearn.datasets import load_iris
        
        print("✅ All required packages imported successfully")
        
        # Test data loading
        iris = load_iris()
        print(f"✅ Sample dataset loaded: {iris.data.shape}")
        
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def print_next_steps():
    """Print next steps for the user"""
    print("\n" + "="*60)
    print("🎉 Setup completed successfully!")
    print("="*60)
    print("\n📋 Next steps:")
    print("1. Get your Gemini API key from https://aistudio.google.com/")
    print("2. Add your API key to the .env file")
    print("3. Run the web application: streamlit run app.py")
    print("4. Or test the example: python example_usage.py")
    print("\n📚 Documentation:")
    print("- README.md: Complete setup and usage guide")
    print("- example_usage.py: Programmatic usage examples")
    print("\n🔗 Useful links:")
    print("- Google AI Studio: https://aistudio.google.com/")
    print("- Streamlit: https://streamlit.io/")
    print("- Gemini API Docs: https://ai.google.dev/docs")

def main():
    """Main setup function"""
    print("🚀 Gemini AI Data Science Agent - Setup")
    print("="*50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        print("❌ Setup failed during dependency installation")
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Create .env file
    create_env_file()
    
    # Check API configuration
    api_configured = check_gemini_api()
    
    # Test installation
    if not test_installation():
        print("❌ Setup failed during testing")
        sys.exit(1)
    
    # Print next steps
    print_next_steps()
    
    if not api_configured:
        print("\n⚠️  Remember to configure your Gemini API key before using the application!")

if __name__ == "__main__":
    main()
