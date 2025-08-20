# 🤖 Gemini AI Data Science Agent

A powerful AI-powered data science and machine learning platform built with Google's Gemini 2.0 Flash API. This application provides an intuitive web interface for data analysis, visualization, machine learning model training, and AI-powered insights.

## ✨ Features

### 🧠 AI-Powered Analysis
- **Comprehensive Data Analysis**: AI-driven insights into your datasets
- **Business Intelligence**: Generate actionable business insights
- **Natural Language Chat**: Ask questions about your data in plain English
- **Automated Insights**: Discover patterns, trends, and anomalies

### 📊 Data Visualization
- **Interactive Charts**: Distribution plots, correlation matrices, scatter plots
- **Custom Visualizations**: Generate charts based on your data
- **Real-time Plotting**: Dynamic and responsive visualizations
- **Export Capabilities**: Save charts and visualizations

### 🚀 Machine Learning
- **Auto Model Selection**: Intelligent model recommendation based on data
- **Multiple Algorithms**: Support for classification and regression
- **Hyperparameter Tuning**: Automated optimization of model parameters
- **Cross-Validation**: Robust model evaluation
- **Feature Importance**: Understand what drives your predictions

### 📋 Reporting & Export
- **Comprehensive Reports**: Detailed analysis reports
- **Model Performance**: Track and compare model metrics
- **Export Functionality**: Save results and models
- **Historical Tracking**: Maintain analysis history

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- Gemini API key from Google AI Studio

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd website
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   Create a `.env` file in the root directory:
   ```env
   GEMINI_API_KEY=your_gemini_api_key_here
   ```

4. **Get your Gemini API key**
   - Visit [Google AI Studio](https://aistudio.google.com/)
   - Create a new API key
   - Copy the key to your `.env` file

## 🚀 Usage

### Starting the Application

1. **Run the Streamlit app**
   ```bash
   streamlit run app.py
   ```

2. **Open your browser**
   Navigate to `http://localhost:8501`

### Getting Started

1. **Initialize the Agent**
   - Enter your Gemini API key in the sidebar
   - Click "Initialize Agent"

2. **Load Your Data**
   - Upload a CSV, Excel, or JSON file
   - Or try the sample datasets provided

3. **Explore Your Data**
   - View data overview and statistics
   - Check data quality and missing values

4. **Run AI Analysis**
   - Choose analysis type (comprehensive, exploratory, business insights)
   - Get AI-powered insights and recommendations

5. **Create Visualizations**
   - Generate various chart types
   - Create interactive plots

6. **Train ML Models**
   - Select target variable
   - Choose appropriate models
   - Enable hyperparameter tuning and cross-validation

7. **Chat with AI**
   - Ask questions about your data
   - Request specific analyses
   - Get explanations and recommendations

## 📁 Project Structure

```
website/
├── app.py                 # Main Streamlit application
├── gemini_agent.py        # Core Gemini AI agent
├── ml_utils.py           # Machine learning utilities
├── config.py             # Configuration management
├── requirements.txt      # Python dependencies
├── README.md            # This file
├── uploads/             # Uploaded files directory
├── outputs/             # Generated reports and results
└── cache/               # Cached data and models
```

## 🔧 Configuration

### Environment Variables

- `GEMINI_API_KEY`: Your Google Gemini API key (required)
- `OPENAI_API_KEY`: Optional OpenAI API key for comparison features
- `HUGGINGFACE_API_KEY`: Optional Hugging Face API key for additional models

### Model Configuration

The application uses Gemini 2.0 Flash by default, but you can modify the model in `config.py`:

```python
GEMINI_MODEL = "gemini-2.0-flash-exp"  # Latest Gemini 2.0 Flash
GEMINI_PRO_MODEL = "gemini-1.5-pro"    # Gemini 1.5 Pro for complex tasks
```

## 📊 Supported Data Formats

- **CSV files**: Comma-separated values
- **Excel files**: .xlsx and .xls formats
- **JSON files**: JavaScript Object Notation
- **Sample datasets**: Built-in datasets for testing

## 🧠 Supported Machine Learning Models

### Classification Models
- Random Forest
- Gradient Boosting
- Logistic Regression
- Support Vector Machine (SVM)
- Decision Tree
- K-Nearest Neighbors (KNN)
- Naive Bayes

### Regression Models
- Random Forest
- Gradient Boosting
- Linear Regression
- Ridge Regression
- Lasso Regression
- Support Vector Regression (SVR)
- Decision Tree
- K-Nearest Neighbors (KNN)

## 📈 Features in Detail

### AI Analysis Types

1. **Comprehensive Analysis**
   - Data quality assessment
   - Statistical summary
   - Pattern recognition
   - Outlier detection
   - Correlation analysis
   - Recommendations

2. **Exploratory Analysis**
   - Data distribution patterns
   - Variable relationships
   - Interesting findings
   - Questions for investigation

3. **Business Insights**
   - Key trends and patterns
   - Business opportunities
   - Risk factors
   - Actionable recommendations
   - Data quality assessment

### Visualization Types

- **Distribution Plots**: Histograms and density plots
- **Correlation Matrix**: Heatmap of variable correlations
- **Scatter Plots**: Relationship between variables
- **Box Plots**: Distribution and outliers
- **Interactive Plots**: Dynamic visualizations

### Model Evaluation Metrics

**Classification Metrics:**
- Accuracy
- Precision
- Recall
- F1-Score

**Regression Metrics:**
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R² Score

## 🔒 Security & Privacy

- API keys are stored securely in environment variables
- No data is sent to external servers except for AI analysis
- All processing is done locally
- Generated reports are saved locally

## 🐛 Troubleshooting

### Common Issues

1. **API Key Error**
   - Ensure your Gemini API key is valid
   - Check that the key is properly set in the environment

2. **Import Errors**
   - Install all dependencies: `pip install -r requirements.txt`
   - Ensure Python version is 3.8 or higher

3. **Memory Issues**
   - Reduce dataset size for large files
   - Close other applications to free memory

4. **Model Training Errors**
   - Check data quality and missing values
   - Ensure target variable is properly selected
   - Verify data types are appropriate

### Performance Tips

- Use smaller datasets for faster processing
- Enable caching for repeated operations
- Close unused browser tabs
- Use appropriate model complexity for your data size

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Google AI for providing the Gemini API
- Streamlit for the web framework
- Scikit-learn for machine learning capabilities
- Plotly for interactive visualizations
- Pandas for data manipulation

## 📞 Support

For support and questions:
- Create an issue in the repository
- Check the troubleshooting section
- Review the documentation

---

**Made with ❤️ using Google's Gemini 2.0 Flash API**
