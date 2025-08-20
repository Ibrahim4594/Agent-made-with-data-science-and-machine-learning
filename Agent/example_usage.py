#!/usr/bin/env python3
"""
Example usage of the Gemini AI Data Science Agent

This script demonstrates how to use the agent programmatically
for data analysis, machine learning, and AI-powered insights.
"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris, load_diabetes
import os
import json
from datetime import datetime

# Import our custom modules
from gemini_agent import GeminiDataScienceAgent
from ml_utils import MLUtils
from config import Config

def setup_environment():
    """Setup environment and check API key"""
    print("🔧 Setting up environment...")
    
    # Check if API key is set
    if not Config.GEMINI_API_KEY:
        print("❌ GEMINI_API_KEY not found in environment variables")
        print("Please set your Gemini API key in the .env file or environment")
        return False
    
    print("✅ Environment setup complete")
    return True

def load_sample_datasets():
    """Load sample datasets for demonstration"""
    print("📊 Loading sample datasets...")
    
    datasets = {}
    
    # Load Iris dataset (classification)
    iris = load_iris()
    iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
    iris_df['target'] = iris.target
    iris_df['species'] = iris.target_names[iris.target]
    datasets['iris'] = iris_df
    
    # Load Diabetes dataset (regression)
    diabetes = load_diabetes()
    diabetes_df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
    diabetes_df['target'] = diabetes.target
    datasets['diabetes'] = diabetes_df
    
    print(f"✅ Loaded {len(datasets)} datasets")
    return datasets

def demonstrate_ai_analysis(agent, data, dataset_name):
    """Demonstrate AI-powered data analysis"""
    print(f"\n🤖 Running AI analysis on {dataset_name} dataset...")
    
    # Comprehensive analysis
    print("📊 Performing comprehensive analysis...")
    analysis = agent.analyze_data(data, "comprehensive")
    
    if "error" not in analysis:
        print("✅ Analysis completed successfully!")
        print("\n📋 Analysis Summary:")
        print(f"Dataset shape: {analysis['data_shape']}")
        print(f"Columns: {analysis['columns']}")
        print(f"Timestamp: {analysis['timestamp']}")
        
        # Save analysis results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(f"outputs/{dataset_name}_analysis_{timestamp}.json", "w") as f:
            json.dump(analysis, f, indent=2, default=str)
        print(f"💾 Analysis saved to outputs/{dataset_name}_analysis_{timestamp}.json")
    else:
        print(f"❌ Analysis failed: {analysis['error']}")

def demonstrate_business_insights(agent, data, dataset_name):
    """Demonstrate business insights generation"""
    print(f"\n💼 Generating business insights for {dataset_name} dataset...")
    
    focus_areas = ["Sales", "Marketing", "Operations"]
    insights = agent.generate_insights(data, focus_areas)
    
    if "error" not in insights:
        print("✅ Business insights generated successfully!")
        print("\n💡 Key Insights:")
        print(insights['insights'][:500] + "..." if len(insights['insights']) > 500 else insights['insights'])
        
        # Save insights
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(f"outputs/{dataset_name}_insights_{timestamp}.json", "w") as f:
            json.dump(insights, f, indent=2, default=str)
        print(f"💾 Insights saved to outputs/{dataset_name}_insights_{timestamp}.json")
    else:
        print(f"❌ Insights generation failed: {insights['error']}")

def demonstrate_ml_modeling(ml_utils, data, dataset_name, target_column):
    """Demonstrate machine learning model training and evaluation"""
    print(f"\n🧠 Training ML models on {dataset_name} dataset...")
    
    # Detect problem type
    problem_type = ml_utils.detect_problem_type(data[target_column])
    print(f"📊 Detected problem type: {problem_type}")
    
    # Get available models
    available_models = list(ml_utils.get_available_models(problem_type).keys())
    print(f"🤖 Available models: {available_models}")
    
    # Train and evaluate a few models
    models_to_test = available_models[:3]  # Test first 3 models
    
    for model_name in models_to_test:
        print(f"\n🚀 Training {model_name}...")
        
        # Prepare data
        X_train, X_test, y_train, y_test = ml_utils.prepare_data(data, target_column)
        
        # Train model
        train_result = ml_utils.train_model(model_name, X_train, y_train, problem_type)
        
        if "error" not in train_result:
            print(f"✅ {model_name} trained successfully!")
            
            # Evaluate model
            eval_result = ml_utils.evaluate_model(model_name, X_test, y_test, problem_type)
            
            if "error" not in eval_result:
                print(f"📊 {model_name} Performance:")
                for metric, value in eval_result['metrics'].items():
                    print(f"  - {metric}: {value:.4f}")
                
                # Save model results
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                model_results = {
                    "model_name": model_name,
                    "dataset": dataset_name,
                    "problem_type": problem_type,
                    "target_column": target_column,
                    "metrics": eval_result['metrics'],
                    "feature_importance": eval_result['feature_importance'],
                    "timestamp": timestamp
                }
                
                with open(f"outputs/{dataset_name}_{model_name}_{timestamp}.json", "w") as f:
                    json.dump(model_results, f, indent=2, default=str)
                print(f"💾 Model results saved to outputs/{dataset_name}_{model_name}_{timestamp}.json")
            else:
                print(f"❌ Model evaluation failed: {eval_result['error']}")
        else:
            print(f"❌ Model training failed: {train_result['error']}")

def demonstrate_chat_interface(agent, data, dataset_name):
    """Demonstrate chat interface with the AI agent"""
    print(f"\n💬 Testing chat interface with {dataset_name} dataset...")
    
    # Sample questions
    questions = [
        f"What are the key characteristics of the {dataset_name} dataset?",
        "What patterns do you see in this data?",
        "What would be good features for machine learning?",
        "Are there any data quality issues I should be aware of?"
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\n❓ Question {i}: {question}")
        
        try:
            response = agent.chat(question)
            print(f"🤖 Answer: {response[:300]}..." if len(response) > 300 else f"🤖 Answer: {response}")
        except Exception as e:
            print(f"❌ Chat error: {str(e)}")

def demonstrate_visualizations(agent, data, dataset_name):
    """Demonstrate visualization generation"""
    print(f"\n📈 Generating visualizations for {dataset_name} dataset...")
    
    chart_types = ["distribution", "correlation"]
    
    try:
        visualizations = agent.generate_visualizations(data, chart_types)
        
        if "error" not in visualizations:
            print(f"✅ Generated {len(visualizations)} visualizations:")
            for chart_type in visualizations.keys():
                print(f"  - {chart_type} chart")
            
            # Save visualization info
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            viz_info = {
                "dataset": dataset_name,
                "chart_types": list(visualizations.keys()),
                "timestamp": timestamp
            }
            
            with open(f"outputs/{dataset_name}_visualizations_{timestamp}.json", "w") as f:
                json.dump(viz_info, f, indent=2, default=str)
            print(f"💾 Visualization info saved to outputs/{dataset_name}_visualizations_{timestamp}.json")
        else:
            print(f"❌ Visualization failed: {visualizations['error']}")
    except Exception as e:
        print(f"❌ Visualization error: {str(e)}")

def main():
    """Main demonstration function"""
    print("🚀 Gemini AI Data Science Agent - Example Usage")
    print("=" * 60)
    
    # Setup environment
    if not setup_environment():
        return
    
    # Initialize agent and ML utilities
    print("\n🤖 Initializing Gemini AI Agent...")
    try:
        agent = GeminiDataScienceAgent()
        ml_utils = MLUtils()
        print("✅ Agent initialized successfully!")
    except Exception as e:
        print(f"❌ Failed to initialize agent: {str(e)}")
        return
    
    # Load sample datasets
    datasets = load_sample_datasets()
    
    # Create outputs directory if it doesn't exist
    os.makedirs("outputs", exist_ok=True)
    
    # Demonstrate features for each dataset
    for dataset_name, data in datasets.items():
        print(f"\n{'='*60}")
        print(f"📊 Working with {dataset_name.upper()} dataset")
        print(f"{'='*60}")
        
        print(f"Dataset shape: {data.shape}")
        print(f"Columns: {list(data.columns)}")
        print(f"Data types: {dict(data.dtypes)}")
        
        # AI Analysis
        demonstrate_ai_analysis(agent, data, dataset_name)
        
        # Business Insights
        demonstrate_business_insights(agent, data, dataset_name)
        
        # ML Modeling
        target_column = 'target'  # Both datasets have 'target' column
        demonstrate_ml_modeling(ml_utils, data, dataset_name, target_column)
        
        # Chat Interface
        demonstrate_chat_interface(agent, data, dataset_name)
        
        # Visualizations
        demonstrate_visualizations(agent, data, dataset_name)
    
    print(f"\n{'='*60}")
    print("🎉 Demonstration completed successfully!")
    print("📁 Check the 'outputs' directory for generated files")
    print("🌐 Run 'streamlit run app.py' to use the web interface")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
