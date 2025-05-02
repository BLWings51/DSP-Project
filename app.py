import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from preload_explainers import PredictProbaWrapper, EnsembleProbaWrapper
from data_loader import load_transaction_data, preprocess_data, KerasBinaryClassifier
from utils import load_column_mapping, preprocess_single_transaction
from model_explainer import explain_prediction, explain_with_lime, load_explainers_from_cache
from faker import Faker
import os

fake = Faker()

# ─── 1) Load & preprocess data + maps + combos ────────────────────────────────────
@st.cache_data
def load_and_preprocess_data():
    raw_df = load_transaction_data()
    df_proc, col_map = preprocess_data(raw_df)
    feature_names = [c for c in df_proc.columns if c != 'fraud']
    cust_map  = {c: fake.name()    for c in raw_df['customer'].unique()}
    merch_map = {m: fake.company() for m in raw_df['merchant'].unique()}
    inv_cust  = {v:k for k,v in cust_map.items()}
    inv_merch = {v:k for k,v in merch_map.items()}
    m2c = raw_df.groupby('merchant')['category'].unique().apply(list).to_dict()
    m2z = raw_df.groupby('merchant')['zipMerchant'].unique().apply(list).to_dict()
    return raw_df, df_proc, feature_names, col_map, cust_map, merch_map, inv_cust, inv_merch, m2c, m2z

# ─── 2) Load trained models ───────────────────────────────────────────────────────
@st.cache_resource
def load_models(models_dir='models'):
    rf = joblib.load(os.path.join(models_dir, 'rf_model.joblib'))
    nfeat = rf.n_features_in_
    net = load_model(os.path.join(models_dir, 'nn_model.h5'))
    nn  = KerasBinaryClassifier(model=net, n_features=nfeat)
    nn.fit(np.zeros((1, nfeat)))
    ens = {}
    for kind in ['voting','adaboost','stacking']:
        path = os.path.join(models_dir, f'ensemble_{kind}.joblib')
        m = joblib.load(path)
        if hasattr(m, '_n_features_in'):
            m._n_features_in = nfeat
        ens[f"{kind.title()} Ensemble"] = m
    return {'Random Forest': rf, 'Neural Network': nn, **ens}

# ─── 3) Load SHAP explainers ──────────────────────────────────────────────────────
@st.cache_resource
def load_explainers():
    return load_explainers_from_cache('models/explainers_cache.pkl')

# ─── 4) Sidebar transaction widget (for Model Demo) ──────────────────────────────
def create_transaction_input(
    raw_df, df_proc, col_map,
    cust_map, merch_map, inv_cust, inv_merch,
    m2c, m2z
):
    st.sidebar.header("Configure Transaction")

    tx = {}
    for col in df_proc.columns:
        if col == 'fraud':
            continue

        if col == 'customer':
            name = st.sidebar.selectbox("Customer", list(cust_map.values()))
            tx[col] = inv_cust[name]

        elif col == 'merchant':
            name = st.sidebar.selectbox("Merchant", list(merch_map.values()))
            tx[col] = inv_merch[name]

        elif col == 'category':
            raw_opts = m2c.get(tx['merchant'], [])
            # strip "es_" for display
            disp_map = {opt.replace("es_",""): opt for opt in raw_opts}
            pick = st.sidebar.selectbox("Category", list(disp_map.keys()))
            tx[col] = disp_map[pick]

        elif col == 'zipMerchant':
            raw_opts = m2z.get(tx['merchant'], [])
            pick = st.sidebar.selectbox("Merchant ZIP", raw_opts)
            tx[col] = pick

        elif col == 'zipcodeOri':
            raw_opts = raw_df['zipcodeOri'].unique().tolist()
            pick = st.sidebar.selectbox("Origin ZIP", raw_opts)
            tx[col] = pick

        elif col == 'step':
            # raw slider in [min_step, max_step]
            min_step = int(raw_df['step'].min())
            max_step = int(raw_df['step'].max())
            raw_step = st.sidebar.slider("Transaction Step", min_step, max_step, int(raw_df['step'].median()))
            # normalize to [0,1] for the model
            tx[col] = (raw_step - min_step) / (max_step - min_step)

        elif col == 'age':
            # raw age slider 0–100
            min_age = 0
            max_age = 100
            raw_age = st.sidebar.slider("Customer Age", min_age, max_age, int(df_proc['age'].median()) * 100)
            # normalize to [0,1]
            tx[col] = (raw_age - min_age) / (max_age - min_age)

        elif col == 'gender':
            # four categories: M,F,E,U
            gender_map = {'M':'Male','F':'Female','E':'Enterprise','U':'Unknown'}
            pick = st.sidebar.selectbox("Gender", list(gender_map.values()))
            # reverse map
            tx[col] = next(k for k,v in gender_map.items() if v==pick)

        elif col == 'amount':
            mn, mx = float(raw_df['amount'].min()), float(raw_df['amount'].max())
            val = st.sidebar.slider("Amount (£)", mn, mx, float(raw_df['amount'].median()))
            # normalize back to [0,1] for model
            tx[col] = (val - mn) / (mx - mn)

        elif col in col_map:
            # other encoded categorical
            opts = list(col_map[col].keys())
            tx[col] = st.sidebar.selectbox(col, opts)

        else:
            mn, mx = float(df_proc[col].min()), float(df_proc[col].max())
            tx[col] = st.sidebar.slider(col, mn, mx, float(df_proc[col].median()))

    return pd.DataFrame([tx])

# ─── 5) Home ──────────────────────────────────────────────────────────────────────
def show_home():
    st.title("💰 Fraud Detection Tutorial")
    st.markdown("""
    **Learn by doing**:  
    1. **Guess** fraud in the Tutorial.  
    2. **Explore** ML models & explanations in Model Demo.  
    """)
    st.image(
        "https://images.unsplash.com/photo-1581090700227-0df3c23d1b00",
        use_container_width=True
    )
    st.markdown("_Use the sidebar to switch pages._")

# ─── 6) Tutorial ─────────────────────────────────────────────────────────────────
def show_tutorial(raw_df, cust_map, merch_map):
    st.title("🎓 Fraud-Guessing Tutorial")
    st.markdown("One of these five transactions is fraudulent. Spot the odd one!")

    state = st.session_state
    state.setdefault('score', 0)
    state.setdefault('total', 0)
    state.setdefault('next_round', True)
    state.setdefault('answered', False)

    if state.next_round:
        legit = raw_df[raw_df.fraud==0].sample(4)
        fraud = raw_df[raw_df.fraud==1].sample(1)
        pool  = pd.concat([legit, fraud]).sample(frac=1).reset_index(drop=True)
        state.pool = pool
        state.answered = False
        state.next_round = False
        state.tutorial_choice = 1

    pool = state.pool
    disp = pd.DataFrame({
        "Customer":   [cust_map[c] for c in pool.customer],
        "Age":        [f"{int(str(a).strip("\"'\""))*10}s" for a in pool.age],
        "Gender":     ["Male" if g=="M" else "Female" for g in pool.gender],
        "Merchant":   [merch_map[m] for m in pool.merchant],
        "Category":   [cat.replace("es_","") for cat in pool.category],
        "Amount (£)": [round(v,2) for v in pool.amount]
    })
    st.table(disp)

    choice = st.radio("Which row is fraud?", list(range(1,6)),
                      format_func=lambda x: f"Row {x}", key="tutorial_choice")

    if not state.answered:
        if st.button("Submit Guess"):
            correct_idx = int(pool[pool.fraud==1].index[0])
            state.total += 1
            if (choice-1) == correct_idx:
                state.score += 1
                st.success("✅ Correct!")
            else:
                st.error(f"❌ Wrong. It was Row {correct_idx+1}.")
            st.info(f"Score: {state.score}/{state.total}")
            state.answered = True
    else:
        if st.button("Next Round"):
            state.next_round = True

# ─── 7) Model Demo ────────────────────────────────────────────────────────────────
def show_model_demo(raw_df, df_proc, features, col_map,
                    cust_map, merch_map, inv_cust, inv_merch,
                    m2c, m2z, models, explainers):
    st.title("🔍 Model Demo & Explanations")

    tx_df = create_transaction_input(
        raw_df, df_proc, col_map,
        cust_map, merch_map, inv_cust, inv_merch,
        m2c, m2z
    )
    st.subheader("Transaction")
    st.dataframe(tx_df)

    model_choice = st.sidebar.selectbox("Model", list(models.keys()), key="demo_model")
    method       = st.sidebar.selectbox("Explanation", ["SHAP","LIME"], key="demo_method")
    if not st.sidebar.button("Predict & Explain"):
        return

    raw = tx_df.iloc[0].to_dict()
    proc = preprocess_single_transaction(raw, col_map)
    X = np.array(proc, dtype=np.float32).reshape(1,-1)
    mdl = models[model_choice]

    if model_choice=="Neural Network":
        p     = float(mdl.predict(X)[0]); probs=[1-p,p]; y=int(p>0.5)
    else:
        probs = mdl.predict_proba(X)[0]; y=int(mdl.predict(X)[0])

    if y: st.error(f"⚠️ Fraud (P={probs[1]:.1%})")
    else: st.success(f"✅ Legitimate (P={probs[0]:.1%})")

    st.subheader(f"{method} Explanation for {model_choice}")
    if method=="SHAP":
        key_map = {
            'Random Forest':'random_forest',
            'Neural Network':'neural_network',
            'Voting Ensemble':'voting',
            'Adaboost Ensemble':'adaboost',
            'Stacking Ensemble':'stacking'
        }
        expl_key = key_map[model_choice]
        expl     = explainers.get(expl_key)
        tabs     = st.tabs(["Force","Waterfall","Bar"])
        for tab, pt in zip(tabs,["force","waterfall","bar"]):
            with tab:
                fig = explain_prediction(
                    model_name    = expl_key,
                    transaction   = X,
                    feature_names = features,
                    explainers    = {expl_key:expl},
                    plot_type     = pt
                )
                st.pyplot(fig)
    else:
        lime_exp = explain_with_lime(mdl,X,features)
        fig, ax = plt.subplots(figsize=(8,6))
        lime_exp.as_pyplot_figure()
        st.pyplot(fig)

# ─── 8) Main ──────────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(page_title="Fraud Detection Tutorial", layout="wide")
    page = st.sidebar.radio("Navigate to", ["Home","Tutorial","Model Demo"])

    raw_df, df_proc, features, col_map, \
    cust_map, merch_map, inv_cust, inv_merch, \
    m2c, m2z = load_and_preprocess_data()

    if page=="Home":
        show_home()
    elif page=="Tutorial":
        show_tutorial(raw_df,cust_map,merch_map)
    else:
        models     = load_models()
        explainers = load_explainers()
        show_model_demo(
            raw_df, df_proc, features, col_map,
            cust_map, merch_map, inv_cust, inv_merch,
            m2c, m2z, models, explainers
        )

if __name__=="__main__":
    main()
