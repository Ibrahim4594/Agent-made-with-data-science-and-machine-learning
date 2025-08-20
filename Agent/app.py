import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import base64
import io
import json
from datetime import datetime
import os

from gemini_agent import GeminiDataScienceAgent
from ml_utils import MLUtils
from config import Config

# Page configuration
st.set_page_config(
    page_title="Gemini AI Data Science Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #c3e6cb;
    }
    .error-message {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #f5c6cb;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'agent' not in st.session_state:
    st.session_state.agent = None
if 'ml_utils' not in st.session_state:
    st.session_state.ml_utils = None
if 'current_data' not in st.session_state:
    st.session_state.current_data = None
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'trained_models' not in st.session_state:
    st.session_state.trained_models = {}

def initialize_agent():
    """Initialize the Gemini AI agent"""
    try:
        if st.session_state.agent is None:
            st.session_state.agent = GeminiDataScienceAgent()
            st.session_state.ml_utils = MLUtils()
        return True
    except Exception as e:
        st.error(f"Failed to initialize agent: {str(e)}")
        return False

def load_sample_data():
    """Load sample datasets for demonstration"""
    sample_data = {}
    
    # Sample 1: Iris dataset
    from sklearn.datasets import load_iris
    iris = load_iris()
    iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
    iris_df['target'] = iris.target
    iris_df['species'] = iris.target_names[iris.target]
    sample_data['Iris Dataset'] = iris_df
    
    # Sample 2: Diabetes dataset
    from sklearn.datasets import load_diabetes
    diabetes = load_diabetes()
    diabetes_df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
    diabetes_df['target'] = diabetes.target
    sample_data['Diabetes Dataset'] = diabetes_df
    
    # Sample 3: Breast cancer dataset
    from sklearn.datasets import load_breast_cancer
    cancer = load_breast_cancer()
    cancer_df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
    cancer_df['target'] = cancer.target
    sample_data['Breast Cancer Dataset'] = cancer_df
    
    return sample_data

def main():
    """Main application function"""
    
    # Header
    st.markdown('<h1 class="main-header">🤖 Gemini AI Data Science Agent</h1>', unsafe_allow_html=True)
    st.markdown("### Powered by Google's Gemini 2.0 Flash - Your AI Partner for Data Science & Machine Learning")
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # API Key input
        api_key = st.text_input("Gemini API Key", value="AIzaSyCe9Yk344zlvW78DWV60cJscC9--tbNwP0", type="password", help="Enter your Gemini API key")
        if api_key:
            os.environ['GEMINI_API_KEY'] = api_key
        
        # Initialize agent
        if st.button("🚀 Initialize Agent", type="primary"):
            if initialize_agent():
                st.success("Agent initialized successfully!")
            else:
                st.error("Failed to initialize agent. Check your API key.")
        
        st.divider()
        
        # Data upload section
        st.header("📊 Data Management")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload your dataset",
            type=['csv', 'xlsx', 'json'],
            help="Upload CSV, Excel, or JSON files"
        )
        
        # Sample data
        st.subheader("Or try sample data:")
        sample_data = load_sample_data()
        selected_sample = st.selectbox("Choose sample dataset", [""] + list(sample_data.keys()))
        
        if selected_sample:
            st.session_state.current_data = sample_data[selected_sample]
            st.success(f"Loaded {selected_sample}")
    
    # Main content area
    if st.session_state.agent is None:
        st.info("👈 Please initialize the agent in the sidebar to get started!")
        return
    
    # Load data
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                data = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.xlsx'):
                data = pd.read_excel(uploaded_file)
            elif uploaded_file.name.endswith('.json'):
                data = pd.read_json(uploaded_file)
            
            st.session_state.current_data = data
            st.success(f"Successfully loaded {uploaded_file.name}")
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
    
    # Display current data
    if st.session_state.current_data is not None:
        data = st.session_state.current_data
        
        # Data overview
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Rows", len(data))
        with col2:
            st.metric("Columns", len(data.columns))
        with col3:
            st.metric("Missing Values", data.isnull().sum().sum())
        with col4:
            st.metric("Memory Usage", f"{data.memory_usage(deep=True).sum() / 1024:.1f} KB")
        
        # Data preview
        st.subheader("📋 Data Preview")
        st.dataframe(data.head(10), use_container_width=True)
        
        # Data info
        with st.expander("📊 Data Information"):
            buffer = io.StringIO()
            data.info(buf=buffer)
            st.text(buffer.getvalue())
        
        # Main tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🤖 AI Analysis", "📈 Visualizations", "🧠 ML Models", "💬 Chat", "📋 Reports"
        ])
        
        with tab1:
            st.header("🤖 AI-Powered Data Analysis")
            
            analysis_type = st.selectbox(
                "Choose analysis type",
                ["comprehensive", "exploratory", "business_insights", "custom"]
            )
            
            if st.button("🔍 Run AI Analysis", type="primary"):
                with st.spinner("AI is analyzing your data..."):
                    try:
                        if analysis_type == "business_insights":
                            focus_areas = st.multiselect(
                                "Focus areas (optional)",
                                ["Sales", "Marketing", "Operations", "Finance", "Customer Behavior"]
                            )
                            results = st.session_state.agent.generate_insights(data, focus_areas)
                        else:
                            results = st.session_state.agent.analyze_data(data, analysis_type)
                        
                        st.session_state.analysis_results = results
                        
                        if "error" not in results:
                            st.success("Analysis completed successfully!")
                            
                            # Display results
                            st.subheader("📊 Analysis Results")
                            st.markdown(results.get("raw_response", results.get("insights", "")))
                            
                            # Store results
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            with open(f"outputs/analysis_{timestamp}.json", "w") as f:
                                json.dump(results, f, indent=2, default=str)
                        else:
                            st.error(f"Analysis failed: {results['error']}")
                            
                    except Exception as e:
                        st.error(f"Error during analysis: {str(e)}")
        
        with tab2:
            st.header("📈 Data Visualizations")
            
            # Chart type selection
            chart_types = st.multiselect(
                "Select chart types",
                ["distribution", "correlation", "scatter", "boxplot", "heatmap"],
                default=["distribution", "correlation"]
            )
            
            if st.button("📊 Generate Visualizations"):
                with st.spinner("Generating visualizations..."):
                    try:
                        visualizations = st.session_state.agent.generate_visualizations(data, chart_types)
                        
                        if "error" not in visualizations:
                            for chart_type, img_base64 in visualizations.items():
                                st.subheader(f"📈 {chart_type.title()} Chart")
                                st.image(f"data:image/png;base64,{img_base64}", use_column_width=True)
                        else:
                            st.error(f"Visualization failed: {visualizations['error']}")
                            
                    except Exception as e:
                        st.error(f"Error generating visualizations: {str(e)}")
            
            # Interactive plots
            st.subheader("🎯 Interactive Plots")
            
            # Scatter plot
            if len(data.select_dtypes(include=[np.number]).columns) >= 2:
                num_cols = data.select_dtypes(include=[np.number]).columns
                x_col = st.selectbox("X-axis", num_cols)
                y_col = st.selectbox("Y-axis", num_cols)
                
                fig = px.scatter(data, x=x_col, y=y_col, title=f"{x_col} vs {y_col}")
                st.plotly_chart(fig, use_container_width=True)
            
            # Distribution plot
            if len(num_cols) > 0:
                dist_col = st.selectbox("Select column for distribution", num_cols)
                fig = px.histogram(data, x=dist_col, title=f"Distribution of {dist_col}")
                st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.header("🧠 Machine Learning Models")
            
            # Target column selection
            target_col = st.selectbox("Select target column", data.columns)
            
            if target_col:
                # Detect problem type
                problem_type = st.session_state.ml_utils.detect_problem_type(data[target_col])
                st.info(f"Detected problem type: {problem_type}")
                
                # Model selection
                available_models = list(st.session_state.ml_utils.get_available_models(problem_type).keys())
                selected_model = st.selectbox("Select model", available_models)
                
                col1, col2 = st.columns(2)
                with col1:
                    hyperparameter_tuning = st.checkbox("Enable hyperparameter tuning")
                with col2:
                    cross_validation = st.checkbox("Enable cross-validation")
                
                if st.button("🚀 Train Model", type="primary"):
                    with st.spinner("Training model..."):
                        try:
                            # Prepare data
                            X_train, X_test, y_train, y_test = st.session_state.ml_utils.prepare_data(
                                data, target_col
                            )
                            
                            # Train model
                            train_result = st.session_state.ml_utils.train_model(
                                selected_model, X_train, y_train, problem_type, hyperparameter_tuning
                            )
                            
                            if "error" not in train_result:
                                st.success(f"Model {selected_model} trained successfully!")
                                
                                # Evaluate model
                                eval_result = st.session_state.ml_utils.evaluate_model(
                                    selected_model, X_test, y_test, problem_type
                                )
                                
                                if "error" not in eval_result:
                                    # Display metrics
                                    st.subheader("📊 Model Performance")
                                    
                                    metrics = eval_result["metrics"]
                                    col1, col2, col3, col4 = st.columns(4)
                                    
                                    for i, (metric, value) in enumerate(metrics.items()):
                                        with [col1, col2, col3, col4][i % 4]:
                                            st.metric(metric.upper(), f"{value:.4f}")
                                    
                                    # Feature importance
                                    if eval_result["feature_importance"]:
                                        st.subheader("🎯 Feature Importance")
                                        importance_df = pd.DataFrame(
                                            list(eval_result["feature_importance"].items()),
                                            columns=["Feature", "Importance"]
                                        ).sort_values("Importance", ascending=False)
                                        
                                        fig = px.bar(
                                            importance_df.head(10),
                                            x="Importance",
                                            y="Feature",
                                            orientation="h",
                                            title="Top 10 Most Important Features"
                                        )
                                        st.plotly_chart(fig, use_container_width=True)
                                    
                                    # Store model
                                    st.session_state.trained_models[selected_model] = {
                                        "problem_type": problem_type,
                                        "target_column": target_col,
                                        "metrics": metrics,
                                        "timestamp": datetime.now().isoformat()
                                    }
                                else:
                                    st.error(f"Evaluation failed: {eval_result['error']}")
                            else:
                                st.error(f"Training failed: {train_result['error']}")
                                
                        except Exception as e:
                            st.error(f"Error during model training: {str(e)}")
                
                # Cross-validation
                if cross_validation and st.button("🔄 Run Cross-Validation"):
                    with st.spinner("Running cross-validation..."):
                        try:
                            X = data.drop(columns=[target_col])
                            y = data[target_col]
                            
                            cv_result = st.session_state.ml_utils.cross_validate_model(
                                selected_model, X, y, problem_type
                            )
                            
                            if "error" not in cv_result:
                                st.subheader("🔄 Cross-Validation Results")
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    st.metric("Mean Score", f"{cv_result['mean_score']:.4f}")
                                with col2:
                                    st.metric("Std Score", f"{cv_result['std_score']:.4f}")
                                with col3:
                                    st.metric("CV Folds", cv_result['cv_folds'])
                                
                                # CV scores plot
                                fig = px.line(
                                    y=cv_result['cv_scores'],
                                    title="Cross-Validation Scores",
                                    labels={'x': 'Fold', 'y': 'Score'}
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.error(f"Cross-validation failed: {cv_result['error']}")
                                
                        except Exception as e:
                            st.error(f"Error during cross-validation: {str(e)}")
        
        with tab4:
            st.header("💬 Chat with AI Agent")
            
            # Chat interface
            user_message = st.text_area("Ask me anything about your data or request analysis:", height=100)
            
            if st.button("💬 Send Message", type="primary"):
                if user_message:
                    with st.spinner("AI is thinking..."):
                        try:
                            response = st.session_state.agent.chat(user_message)
                            st.markdown("### 🤖 AI Response:")
                            st.markdown(response)
                        except Exception as e:
                            st.error(f"Error in chat: {str(e)}")
            
            # Conversation history
            if st.session_state.agent.conversation_history:
                st.subheader("📝 Conversation History")
                for i, message in enumerate(st.session_state.agent.conversation_history[-10:]):  # Show last 10 messages
                    if message["role"] == "user":
                        st.markdown(f"**You:** {message['parts'][0]}")
                    else:
                        st.markdown(f"**AI:** {message['parts'][0]}")
                    st.divider()
        
        with tab5:
            st.header("📋 Analysis Reports")
            
            # Generate comprehensive report
            if st.button("📄 Generate Report", type="primary"):
                with st.spinner("Generating comprehensive report..."):
                    try:
                        # Data summary
                        st.subheader("📊 Data Summary")
                        st.write(f"Dataset shape: {data.shape}")
                        st.write(f"Columns: {list(data.columns)}")
                        st.write(f"Data types: {dict(data.dtypes)}")
                        
                        # Missing values
                        missing_data = data.isnull().sum()
                        if missing_data.sum() > 0:
                            st.subheader("⚠️ Missing Values")
                            st.write(missing_data[missing_data > 0])
                        
                        # Statistical summary
                        st.subheader("📈 Statistical Summary")
                        st.dataframe(data.describe(), use_container_width=True)
                        
                        # Correlation matrix
                        if len(data.select_dtypes(include=[np.number]).columns) > 1:
                            st.subheader("🔗 Correlation Matrix")
                            corr_matrix = data.select_dtypes(include=[np.number]).corr()
                            fig = px.imshow(
                                corr_matrix,
                                title="Correlation Matrix",
                                color_continuous_scale="RdBu"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # AI insights
                        if st.session_state.analysis_results:
                            st.subheader("🤖 AI Insights")
                            st.markdown(st.session_state.analysis_results.get("raw_response", ""))
                        
                        # Model performance
                        if st.session_state.trained_models:
                            st.subheader("🧠 Model Performance")
                            for model_name, model_info in st.session_state.trained_models.items():
                                st.write(f"**{model_name}** ({model_info['problem_type']})")
                                for metric, value in model_info['metrics'].items():
                                    st.write(f"  - {metric}: {value:.4f}")
                                st.write(f"  - Trained: {model_info['timestamp']}")
                                st.divider()
                        
                        # Export report
                        report_data = {
                            "timestamp": datetime.now().isoformat(),
                            "data_shape": data.shape,
                            "columns": list(data.columns),
                            "analysis_results": st.session_state.analysis_results,
                            "trained_models": st.session_state.trained_models
                        }
                        
                        # Save report
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        with open(f"outputs/report_{timestamp}.json", "w") as f:
                            json.dump(report_data, f, indent=2, default=str)
                        
                        st.success("Report generated and saved!")
                        
                    except Exception as e:
                        st.error(f"Error generating report: {str(e)}")

if __name__ == "__main__":
    main()
