Python AI & Data Project

This repository contains a Python-based project leveraging AI, data analysis, and visualization tools. It integrates advanced AI capabilities, including generative AI, machine learning, and data science workflows, along with interactive visualization and web apps.

Features

Generative AI: Powered by Google Generative AI and OpenAI APIs.

Data Analysis & Visualization: Using pandas, numpy, matplotlib, seaborn, plotly.

Machine Learning: Implement ML models with scikit-learn.

Interactive Web Apps: Built with streamlit.

Environment Management: Use .env files with python-dotenv.

Workflow Automation: Supports integration with LangChain and other AI frameworks.

Jupyter Support: Fully compatible with jupyter notebooks for interactive experimentation.

Installation

Clone the repository

git clone <repository_url>
cd <repository_folder>


Create a virtual environment

python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate


Install dependencies

pip install -r requirements.txt


Required Python packages:

google-generativeai>=0.3.2
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
plotly>=5.17.0
python-dotenv>=1.0.0
requests>=2.31.0
openai>=1.3.0
langchain>=0.0.350
langchain-google-genai>=0.0.5
jupyter>=1.0.0
ipykernel>=6.25.0

Usage

Streamlit Web App

streamlit run app.py


Jupyter Notebook

jupyter notebook


Import AI & ML modules

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import google.generativeai as genai
import openai

Environment Variables

Create a .env file in the root directory to store your API keys:

OPENAI_API_KEY=your_openai_api_key
GOOGLE_API_KEY=your_google_api_key

Contributing

Contributions are welcome! Please open an issue or submit a pull request.

License

This project is licensed under the MIT License.
