```markdown
# 1. FRAUD DETECTION INTERACTIVE APP

An educational and exploratory console-style application for detecting fraudulent transactions using machine learning. Allows users to play a guessing tutorial, configure custom transactions for multiple models (Random Forest, Neural Network, Ensembles), and view explainable AI visualizations (SHAP and LIME).

# 2. FEATURES

1. **Tutorial Mode**  
   - Presents five sample transactions (one fraudulent) in a table  
   - User selects which row is fraudulent  
   - Tracks score and allows repeated rounds  

2. **Model Demo Mode**  
   - Sidebar inputs for customer, merchant, amount, age, gender, etc.  
   - Supports three model types:  
     1. Random Forest  
     2. Neural Network  
     3. Ensembles (Voting, AdaBoost, Stacking)  
   - Shows predicted probability and verdict  

3. **Explainability**  
   - SHAP visualizations: force plot, waterfall plot, bar plot  
   - LIME explanations for single-instance predictions  

4. **Interactive UI**  
   - Streamlit interface with sidebar navigation  
   - Lazy loading of models and explainers for responsive performance  

# 3. INSTALLATION

1. Clone repository and enter its folder  
2. Create and activate a Python virtual environment  
3. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```  

# 4. USAGE

1. Run the application with Streamlit:  
   ```bash
   streamlit run app.py
   ```  
2. Use the sidebar to navigate among **Home**, **Tutorial**, and **Model Demo**  
3. In **Tutorial**, guess the fraudulent transaction and view your score  
4. In **Model Demo**, configure a transaction, click **Predict & Explain**, and view visualizations  

# 5. PROJECT STRUCTURE

```
app.py                     – Streamlit main application  
data_loader.py             – Data loading, preprocessing, train/test split  
utils.py                   – Model save/load, single-row preprocessing  
model_explainer.py         – SHAP and LIME explanation functions  
preload_explainers.py      – Build and cache SHAP/Keras explainers  
banksimData.csv            – Sample transaction dataset  
models/                    – Trained model files and explainer cache  
diagrams/                  – PlantUML diagram files (.puml)  
tests/                     – Unit tests and integration tests  
requirements.txt           – Pinned Python dependencies  
README.md                  – This document  
```

# 6. DESIGN AND ARCHITECTURE

## 6.1 High-Level System Flow  
*(diagrams/system_flow.puml)*  
Shows how data moves through the app—from CSV → preprocessing → model inference → explanation → UI rendering.

## 6.2 Class Diagram  
*(diagrams/class_diagram.puml)*  
Illustrates core classes and relationships: `KerasBinaryClassifier`, `PredictProbaWrapper`, explainer cache.

## 6.3 Sequence Diagram  
*(diagrams/sequence_diagram.puml)*  
Details request flow: user input → preprocessing → `model.predict` → SHAP/LIME call → plot rendering.

## 6.4 System Architecture  
*(diagrams/architecture.puml)*  
Depicts components and interactions: Streamlit UI, data layer, model layer, explainer layer.

## 6.5 Environment Diagram  
*(diagrams/environment.puml)*  
Outlines runtime dependencies and execution context.

# 7. TESTING

- **Unit tests** for data preprocessing, single-row transformation, and model wrappers  
- **Integration tests** verifying end-to-end preprocessing and prediction consistency  
- Run all tests with `pytest`:
  ```bash
  pytest -q
  ```

# 8. CONTRIBUTING

1. Fork the repository and create a feature branch  
2. Write clear, modular code and corresponding tests  
3. Submit a pull request against the `main` branch  

# 9. LICENSE

Distributed under the MIT License. See [LICENSE](LICENSE) for details.  
```