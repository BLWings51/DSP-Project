# Fraud Detection System

An interactive Streamlit application for exploring transaction fraud detection using machine learning models and explainability tools. This project provides:

- **Fraud-Guessing Tutorial**: A mini-game where you spot which transaction is fraudulent.
- **Model Demo**: Compare different classifiers (Random Forest, Neural Network, Ensembles) on live inputs.
- **Explainability**: Visualize model decisions with **SHAP** (force, waterfall, bar plots) and **LIME**.

---

## 🚀 Features

- **Load & Preprocess**  
  - Clean raw BankSim CSV  
  - Handle missing / `'U'` ages, single-quote strings, categorical encodings  
  - Normalize numerical features to [0,1]

- **Models**  
  - Random Forest with class balancing  
  - Feed-forward Neural Network  
  - Voting, AdaBoost & Stacking ensembles  
  - **KerasBinaryClassifier** wrapper for scikit-learn compatibility

- **Explainability**  
  - **SHAP** TreeExplainer & KernelExplainer  
  - **LIME** TabularExplainer  

- **Interactive UI**  
  - Streamlit pages: Home, Tutorial, Model Demo  
  - Sidebar controls for input, model/method selection  
  - Dynamic, selection-driven UI (valid feature combinations, human-friendly labels)

---

## 📦 Installation

1. **Clone the repository**  
   ```bash
   git clone https://github.com/your-org/fraud-detection-app.git
   cd fraud-detection-app
   
2. **Create & activate a virtual environment**  
   ```bash
   python3 -m venv venv
    source venv/bin/activate   # Linux/macOS
    venv\Scripts\activate

3. **Install dependencies**  
   ```bash
   pip install -r requirements.txt