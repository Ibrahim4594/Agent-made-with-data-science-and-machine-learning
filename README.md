Agent Made with Data Science and Machine Learning

This repository showcases a Python-based agent that leverages advanced data science and machine learning techniques to automate various tasks. The agent integrates generative AI, data analysis, visualization, and interactive applications to provide a comprehensive solution for data-driven challenges.

🔧 Features

Generative AI: Utilizes Google Generative AI and OpenAI APIs for advanced AI capabilities.

Data Analysis & Visualization: Employs pandas, numpy, matplotlib, seaborn, and plotly for data manipulation and visualization.

Machine Learning: Implements machine learning models using scikit-learn.

Interactive Web Applications: Developed with streamlit for user-friendly interfaces.

Environment Management: Configured with .env files using python-dotenv for secure handling of environment variables.

Workflow Automation: Supports integration with LangChain and other AI frameworks for streamlined workflows.

Jupyter Notebook Support: Fully compatible with jupyter notebooks for interactive experimentation.

📦 Installation

To set up the project locally, follow these steps:

Clone the repository:

git clone https://github.com/Ibrahim4594/Agent-made-with-data-science-and-machine-learning.git
cd Agent-made-with-data-science-and-machine-learning


Create and activate a virtual environment:

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install dependencies:

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

🚀 Usage

Streamlit Web Application:

streamlit run app.py


Jupyter Notebook:

jupyter notebook


Importing Modules:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import google.generativeai as genai
import openai

🔐 Environment Variables

Create a .env file in the root directory to store your API keys:

OPENAI_API_KEY=your_openai_api_key
GOOGLE_API_KEY=your_google_api_key

🤝 Contributing

Contributions are welcome! Please fork the repository, make your changes, and submit a pull request. Ensure that your code adheres to the project's coding standards and includes appropriate tests.

📄 License

This project is licensed under the MIT License.
