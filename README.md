# Fraud Detection Interactive App

An educational and exploratory application for detecting fraudulent transactions using machine learning. Users can play a guessing tutorial, configure custom transactions for multiple models (Random Forest, Neural Network, Ensembles), and view explainable AI visualizations (SHAP).

## Features

- Tutorial Mode: Guess the fraudulent transaction among samples, with score tracking.
- Model Demo Mode: Configure transactions, select model, and view predictions.
- Explainability: SHAP visualizations (force, waterfall, bar plots).
- Interactive UI: Streamlit interface with sidebar navigation.

## Installation

1. Clone the repository and enter its folder.
2. Create and activate a Python virtual environment.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the application with Streamlit:
   ```bash
   streamlit run app.py
   ```
2. Use the sidebar to navigate among Home, Tutorial, and Model Demo.
3. In Tutorial, guess the fraudulent transaction and view your score.
4. In Model Demo, configure a transaction, click Predict & Explain, and view visualizations.

## Project Structure

```
app.py                  – Streamlit main application
data_loader.py          – Data loading, preprocessing, train/test split
utils.py                – Model save/load, single-row preprocessing
model_explainer.py      – SHAP explanation functions
preload_explainers.py   – Build and cache SHAP/Keras explainers
banksimData.csv         – Sample transaction dataset
models/                 – Trained model files and explainer cache
diagrams/               – PlantUML diagram files (.puml)
tests/                  – Unit and integration tests
requirements.txt        – Python dependencies
README.md               – This document
```

## Diagrams

- System Flow: Data moves from CSV → preprocessing → model inference → explanation → UI rendering.
- Class Diagram: Core classes and relationships.
- Sequence Diagram: Request flow from user input to plot rendering.
- System Architecture: Components and interactions.
- Environment Diagram: Runtime dependencies and execution context.

## Testing

- Unit tests for data preprocessing, single-row transformation, and model wrappers.
- Integration tests for end-to-end preprocessing and prediction consistency.
- Run all tests with:
  ```bash
  pytest -q
  ```


