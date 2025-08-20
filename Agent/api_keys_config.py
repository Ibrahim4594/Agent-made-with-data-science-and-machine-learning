#!/usr/bin/env python3
"""
API Keys Configuration File

This file contains your API keys for the Gemini AI Data Science Agent.
You can either:
1. Set these as environment variables
2. Use this file directly (not recommended for production)
3. Create a .env file with these values

For security, it's recommended to use environment variables or a .env file.
"""

import os

# Set your API keys here
GEMINI_API_KEY = "Enter your Gemini API key here"
OPENAI_API_KEY = "Enter your OpenAI API key here"
HUGGINGFACE_API_KEY = "Enter your Hugging Face API key here"

# Set environment variables
os.environ['GEMINI_API_KEY'] = GEMINI_API_KEY
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
os.environ['HUGGINGFACE_API_KEY'] = HUGGINGFACE_API_KEY

print("✅ API keys have been set in environment variables!")
print("🔑 Gemini API Key: Configured")
print("🔑 OpenAI API Key: Configured")
print("🔑 Hugging Face API Key: Configured")
print("\n🚀 You can now run the application!")
print("   - Web interface: streamlit run app.py")
print("   - Example usage: python example_usage.py")
