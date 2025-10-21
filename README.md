# ExplainAI 🧠

**ExplainAI** is an interactive Streamlit web application designed to explain and visualize machine learning models using SHAP (SHapley Additive exPlanations).  
It helps users understand how each feature impacts predictions, remove feature dependencies, and debug model interpretability issues.

## 🚀 Features
- Upload datasets and train models (e.g., Keras, XGBoost, LightGBM)
- Generate SHAP explanations for feature importance
- Visualize local and global explanations interactively
- Streamlit-based UI for instant feedback and analysis
- Supports model optimization and fairness insights

## 🧰 Tech Stack
- **Python** (Keras, scikit-learn, SHAP)
- **Streamlit** for UI
- **Pandas / NumPy** for data handling
- **Matplotlib / Plotly** for visualization

## 📦 Setup
```bash
# clone the repo
git clone https://github.com/sushika/explainai.git

# go inside
cd explainai

# create virtual environment
python -m venv .venv
.venv\Scripts\activate

# install dependencies
pip install -r requirements.txt

# run the app
streamlit run app.py
