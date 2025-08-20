import google.generativeai as genai
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import json
import io
import base64
from typing import Dict, List, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

from config import Config

class GeminiDataScienceAgent:
    """AI Agent specialized in Data Science and Machine Learning using Gemini API"""
    
    def __init__(self):
        """Initialize the Gemini AI Agent"""
        self.config = Config()
        self.config.validate_config()
        
        # Configure Gemini API
        genai.configure(api_key=self.config.GEMINI_API_KEY)
        
        # Initialize models
        self.model = genai.GenerativeModel(
            model_name=self.config.GEMINI_MODEL,
            generation_config=genai.types.GenerationConfig(
                temperature=self.config.TEMPERATURE,
                top_p=self.config.TOP_P,
                max_output_tokens=self.config.MAX_TOKENS,
            )
        )
        
        # Initialize conversation history
        self.conversation_history = []
        self.current_dataset = None
        self.current_analysis = {}
        
    def chat(self, message: str, context: str = "") -> str:
        """Chat with the Gemini AI agent"""
        try:
            # Build context with conversation history
            full_context = self._build_context(context)
            
            # Create chat session
            chat = self.model.start_chat(history=self.conversation_history)
            
            # Send message
            response = chat.send_message(f"{full_context}\n\nUser: {message}")
            
            # Update conversation history
            self.conversation_history.append({
                "role": "user",
                "parts": [message]
            })
            self.conversation_history.append({
                "role": "model",
                "parts": [response.text]
            })
            
            return response.text
            
        except Exception as e:
            return f"Error in chat: {str(e)}"
    
    def analyze_data(self, data: pd.DataFrame, analysis_type: str = "comprehensive") -> Dict[str, Any]:
        """Analyze dataset using Gemini AI"""
        try:
            # Prepare data summary
            data_summary = self._get_data_summary(data)
            
            # Create analysis prompt
            prompt = self._create_analysis_prompt(data_summary, analysis_type)
            
            # Get AI analysis
            response = self.model.generate_content(prompt)
            
            # Parse and structure the analysis
            analysis = self._parse_analysis_response(response.text, data)
            
            # Store current analysis
            self.current_analysis = analysis
            self.current_dataset = data
            
            return analysis
            
        except Exception as e:
            return {"error": f"Analysis failed: {str(e)}"}
    
    def generate_insights(self, data: pd.DataFrame, focus_areas: List[str] = None) -> Dict[str, Any]:
        """Generate business insights from data"""
        try:
            data_summary = self._get_data_summary(data)
            
            prompt = f"""
            Analyze the following dataset and provide business insights:
            
            {data_summary}
            
            Focus areas: {focus_areas if focus_areas else 'General business insights'}
            
            Please provide:
            1. Key trends and patterns
            2. Business opportunities
            3. Risk factors
            4. Recommendations for action
            5. Data quality assessment
            """
            
            response = self.model.generate_content(prompt)
            
            return {
                "insights": response.text,
                "data_summary": data_summary,
                "focus_areas": focus_areas
            }
            
        except Exception as e:
            return {"error": f"Insight generation failed: {str(e)}"}
    
    def suggest_ml_models(self, data: pd.DataFrame, target_column: str) -> Dict[str, Any]:
        """Suggest appropriate ML models for the dataset"""
        try:
            data_summary = self._get_data_summary(data)
            
            prompt = f"""
            Based on the following dataset, suggest appropriate machine learning models:
            
            {data_summary}
            Target column: {target_column}
            
            Please provide:
            1. Problem type (classification/regression)
            2. Recommended models with reasoning
            3. Feature engineering suggestions
            4. Evaluation metrics to use
            5. Potential challenges and solutions
            """
            
            response = self.model.generate_content(prompt)
            
            return {
                "suggestions": response.text,
                "target_column": target_column,
                "data_summary": data_summary
            }
            
        except Exception as e:
            return {"error": f"Model suggestion failed: {str(e)}"}
    
    def explain_model(self, model, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """Explain model predictions and performance"""
        try:
            # Get predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            if hasattr(model, 'predict_proba'):
                # Classification model
                accuracy = accuracy_score(y_test, y_pred)
                report = classification_report(y_test, y_pred, output_dict=True)
                
                prompt = f"""
                Explain the following classification model results:
                
                Accuracy: {accuracy:.4f}
                Classification Report: {json.dumps(report, indent=2)}
                
                Please provide:
                1. Model performance interpretation
                2. Key strengths and weaknesses
                3. Recommendations for improvement
                4. Business implications
                """
            else:
                # Regression model
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                prompt = f"""
                Explain the following regression model results:
                
                Mean Squared Error: {mse:.4f}
                R² Score: {r2:.4f}
                
                Please provide:
                1. Model performance interpretation
                2. Key strengths and weaknesses
                3. Recommendations for improvement
                4. Business implications
                """
            
            response = self.model.generate_content(prompt)
            
            return {
                "explanation": response.text,
                "predictions": y_pred.tolist(),
                "actual": y_test.tolist()
            }
            
        except Exception as e:
            return {"error": f"Model explanation failed: {str(e)}"}
    
    def generate_visualizations(self, data: pd.DataFrame, chart_types: List[str] = None) -> Dict[str, str]:
        """Generate various visualizations for the dataset"""
        try:
            visualizations = {}
            
            if not chart_types:
                chart_types = ['distribution', 'correlation', 'scatter', 'boxplot']
            
            for chart_type in chart_types:
                if chart_type == 'distribution':
                    fig = self._create_distribution_plot(data)
                elif chart_type == 'correlation':
                    fig = self._create_correlation_plot(data)
                elif chart_type == 'scatter':
                    fig = self._create_scatter_plot(data)
                elif chart_type == 'boxplot':
                    fig = self._create_boxplot(data)
                else:
                    continue
                
                # Convert to base64 for web display
                img_bytes = fig.to_image(format="png")
                img_base64 = base64.b64encode(img_bytes).decode()
                visualizations[chart_type] = img_base64
            
            return visualizations
            
        except Exception as e:
            return {"error": f"Visualization generation failed: {str(e)}"}
    
    def _build_context(self, additional_context: str = "") -> str:
        """Build context for the AI agent"""
        base_context = """
        You are an expert Data Science and Machine Learning AI Agent powered by Google's Gemini API. 
        You specialize in:
        - Data analysis and exploration
        - Statistical modeling and inference
        - Machine learning model development
        - Business intelligence and insights
        - Data visualization
        - Predictive analytics
        
        Always provide clear, actionable insights and explain your reasoning.
        """
        
        if additional_context:
            return f"{base_context}\n\nAdditional Context: {additional_context}"
        
        return base_context
    
    def _get_data_summary(self, data: pd.DataFrame) -> str:
        """Generate a comprehensive data summary"""
        summary = f"""
        Dataset Shape: {data.shape}
        
        Column Information:
        {data.info()}
        
        First 5 rows:
        {data.head().to_string()}
        
        Basic Statistics:
        {data.describe().to_string()}
        
        Missing Values:
        {data.isnull().sum().to_string()}
        
        Data Types:
        {data.dtypes.to_string()}
        """
        return summary
    
    def _create_analysis_prompt(self, data_summary: str, analysis_type: str) -> str:
        """Create analysis prompt based on type"""
        if analysis_type == "comprehensive":
            return f"""
            Perform a comprehensive analysis of the following dataset:
            
            {data_summary}
            
            Please provide:
            1. Data quality assessment
            2. Statistical summary
            3. Key patterns and trends
            4. Outlier detection
            5. Correlation analysis
            6. Recommendations for further analysis
            """
        elif analysis_type == "exploratory":
            return f"""
            Perform exploratory data analysis on:
            
            {data_summary}
            
            Focus on:
            1. Data distribution patterns
            2. Relationships between variables
            3. Interesting findings
            4. Questions for deeper investigation
            """
        else:
            return f"""
            Analyze the dataset with focus on {analysis_type}:
            
            {data_summary}
            """
    
    def _parse_analysis_response(self, response: str, data: pd.DataFrame) -> Dict[str, Any]:
        """Parse AI response into structured analysis"""
        return {
            "raw_response": response,
            "data_shape": data.shape,
            "columns": list(data.columns),
            "timestamp": pd.Timestamp.now().isoformat()
        }
    
    def _create_distribution_plot(self, data: pd.DataFrame) -> go.Figure:
        """Create distribution plots for numerical columns"""
        numerical_cols = data.select_dtypes(include=[np.number]).columns
        
        if len(numerical_cols) == 0:
            return go.Figure()
        
        fig = go.Figure()
        
        for col in numerical_cols[:5]:  # Limit to 5 columns
            fig.add_trace(go.Histogram(
                x=data[col].dropna(),
                name=col,
                opacity=0.7
            ))
        
        fig.update_layout(
            title="Distribution of Numerical Variables",
            xaxis_title="Value",
            yaxis_title="Frequency",
            barmode='overlay'
        )
        
        return fig
    
    def _create_correlation_plot(self, data: pd.DataFrame) -> go.Figure:
        """Create correlation heatmap"""
        numerical_data = data.select_dtypes(include=[np.number])
        
        if numerical_data.shape[1] < 2:
            return go.Figure()
        
        corr_matrix = numerical_data.corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0
        ))
        
        fig.update_layout(
            title="Correlation Matrix",
            width=600,
            height=600
        )
        
        return fig
    
    def _create_scatter_plot(self, data: pd.DataFrame) -> go.Figure:
        """Create scatter plot matrix"""
        numerical_data = data.select_dtypes(include=[np.number])
        
        if numerical_data.shape[1] < 2:
            return go.Figure()
        
        # Select first 4 numerical columns for scatter matrix
        cols = numerical_data.columns[:4]
        
        fig = px.scatter_matrix(
            numerical_data[cols],
            title="Scatter Plot Matrix"
        )
        
        return fig
    
    def _create_boxplot(self, data: pd.DataFrame) -> go.Figure:
        """Create box plots for numerical columns"""
        numerical_data = data.select_dtypes(include=[np.number])
        
        if numerical_data.shape[1] == 0:
            return go.Figure()
        
        fig = go.Figure()
        
        for col in numerical_data.columns[:5]:  # Limit to 5 columns
            fig.add_trace(go.Box(
                y=data[col].dropna(),
                name=col
            ))
        
        fig.update_layout(
            title="Box Plots of Numerical Variables",
            yaxis_title="Value"
        )
        
        return fig
