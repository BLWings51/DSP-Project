import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from preload_explainers import PredictProbaWrapper, EnsembleProbaWrapper
from data_loader import load_transaction_data, preprocess_data, KerasBinaryClassifier
from utils import load_column_mapping, preprocess_single_transaction
from model_explainer import explain_prediction, load_explainers_from_cache
from faker import Faker
import os

import torch
import torch.nn as nn

fake = Faker()


MODEL_METRICS = {
    "Random Forest": {
        "F1": 0.99,
        "Precision": 0.89,
        "Recall": 0.75,
        "Accuracy": 0.99,
        "Brier": 0.003,
        "ROC": "images/randomForestROC.png",
    },
    "Stacked Ensemble": {
        "F1": 0.81,
        "Precision": 0.89,
        "Recall": 0.75,
        "Accuracy": 0.99,
        "Brier": 0.003,
        "ROC": "images/stackedEnsembleROC.png"
    },
    "Adaboost Ensemble": {
        "F1": 0.78,
        "Precision": 0.87,
        "Recall": 0.70,
        "Accuracy": 0.99,
        "Brier": 0.167,
        "ROC": "images/adaboostEnsembleROC.png",
    },
    "Voting Ensemble": {
        "F1": 0.92,
        "Precision": 0.91,
        "Recall": 0.93,
        "Accuracy": 0.96,
        "Brier": 0.007,
    },
    "Neural Network": {
        "F1": 0.81,
        "Precision": 0.89,
        "Recall": 0.75,
        "Accuracy": 0.99,
        "Brier": 0.010,
    }
}


@st.cache_data
def load_and_preprocess_data():
    # grab raw data and prep it, then build friendly maps and valid combos
    raw_df = load_transaction_data()
    df_proc, col_map = preprocess_data(raw_df)
    feature_names = [c for c in df_proc.columns if c != 'fraud']
    # fake human names/companies for display
    cust_map  = {c: fake.name()    for c in raw_df['customer'].unique()}
    merch_map = {m: fake.company() for m in raw_df['merchant'].unique()}
    inv_cust  = {v:k for k,v in cust_map.items()}
    inv_merch = {v:k for k,v in merch_map.items()}
    # valid category and zip combos per merchant
    m2c = raw_df.groupby('merchant')['category'].unique().apply(list).to_dict()
    m2z = raw_df.groupby('merchant')['zipMerchant'].unique().apply(list).to_dict()
    return raw_df, df_proc, feature_names, col_map, cust_map, merch_map, inv_cust, inv_merch, m2c, m2z

@st.cache_resource
def load_models(models_dir='models'):
    # load RF, NN, and ensemble models once
    rf = joblib.load(os.path.join(models_dir, 'rf_model.joblib'))
    nfeat = rf.n_features_in_
    net = load_model(os.path.join(models_dir, 'nn_model.h5'))
    nn  = KerasBinaryClassifier(model=net, n_features=nfeat)
    nn.fit(np.zeros((1, nfeat)))
    ens = {}
    for kind in ['voting','adaboost','stacking']:
        m = joblib.load(os.path.join(models_dir, f'ensemble_{kind}.joblib'))
        if hasattr(m, '_n_features_in'):
            m._n_features_in = nfeat
        ens[f"{kind.title()} Ensemble"] = m
    return {'Random Forest': rf, 'Neural Network': nn, **ens}

@st.cache_resource
def load_explainers():
    # pull in cached SHAP explainers
    return load_explainers_from_cache('models/explainers_cache.pkl')


class Generator(nn.Module):
    """Generator network must match the one you trained for fraud synthesis."""
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.input_dim = input_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh()  # each feature in [-1, 1]
        )

    def forward(self, x):
        return self.net(x)

@st.cache_resource(show_spinner="Loading GAN generator…")
def load_gan_generator(model_path: str, input_dim: int, output_dim: int, device: str = 'cpu'):
    """Load *only* the trained Generator weights for inference."""
    gen = Generator(input_dim, output_dim)
    state = torch.load(model_path, map_location=device)
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    gen.load_state_dict(state)
    gen.eval()
    return gen.to(device)

# Helper for denormalising a single generated vector -> readable transaction

def _vector_to_transaction(vec: np.ndarray, df_proc: pd.DataFrame, raw_df: pd.DataFrame,
                           cust_map: dict, merch_map: dict) -> dict:
    """Convert a [-1,1] GAN vector back to a limited set of interpretable fields."""
    cols = df_proc.columns.tolist()
    col_idx = {c: i for i, c in enumerate(cols)}
    out = {}
    template = raw_df[raw_df.fraud == 1].sample(1).iloc[0]

    out["Customer"] = cust_map[template.customer]
    out["Gender"]   = "Male" if template.gender == "M" else "Female"
    out["Merchant"] = merch_map[template.merchant]
    out["Category"] = template.category.replace("es_", "")

    def denorm(val, orig_min, orig_max):
        return ((val + 1) / 2) * (orig_max - orig_min) + orig_min

    age_idx   = col_idx.get("age")
    amt_idx   = col_idx.get("amount")
    step_idx  = col_idx.get("step")
    if age_idx is not None:
        age_raw = denorm(vec[age_idx], 0, 1)
        out["Age"] = f"{int(age_raw * 100) // 10 * 10}s"
    if amt_idx is not None:
        mn, mx = raw_df["amount"].min(), raw_df["amount"].max()
        out["Amount (£)"] = round(denorm(vec[amt_idx], mn, mx), 2)
    if step_idx is not None:
        mn, mx = raw_df["step"].min(), raw_df["step"].max()
        out["Step"] = int(denorm(vec[step_idx], mn, mx))

    out["fraud"] = 1
    return out


def create_transaction_input(raw_df, df_proc, col_map,
                             cust_map, merch_map, inv_cust, inv_merch,
                             m2c, m2z):
    """Original helper reused unchanged."""
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
            opts = m2c.get(tx['merchant'], [])
            disp = {o.replace("es_", ""): o for o in opts}
            pick = st.sidebar.selectbox("Category", list(disp.keys()))
            tx[col] = disp[pick]
        elif col == 'zipMerchant':
            opts = m2z.get(tx['merchant'], [])
            tx[col] = st.sidebar.selectbox("Merchant ZIP", opts)
        elif col == 'zipcodeOri':
            opts = raw_df['zipcodeOri'].unique().tolist()
            tx[col] = st.sidebar.selectbox("Origin ZIP", opts)
        elif col == 'step':
            mn, mx = int(raw_df['step'].min()), int(raw_df['step'].max())
            val = st.sidebar.slider("Transaction Step", mn, mx, int(raw_df['step'].median()))
            tx[col] = (val - mn) / (mx - mn)
        elif col == 'age':
            val = st.sidebar.slider("Customer Age", 0, 100, int(df_proc['age'].median()*100))
            tx[col] = (val - 0) / 100
        elif col == 'gender':
            gm = {'M':'Male','F':'Female','E':'Enterprise','U':'Unknown'}
            pick = st.sidebar.selectbox("Gender", list(gm.values()))
            tx[col] = next(k for k,v in gm.items() if v == pick)
        elif col == 'amount':
            mn, mx = raw_df['amount'].min(), raw_df['amount'].max()
            val = st.sidebar.slider("Amount (£)", float(mn), float(mx), float(raw_df['amount'].median()))
            tx[col] = (val - mn) / (mx - mn)
        elif col in col_map:
            tx[col] = st.sidebar.selectbox(col, list(col_map[col].keys()))
        else:
            mn, mx = df_proc[col].min(), df_proc[col].max()
            tx[col] = st.sidebar.slider(col, float(mn), float(mx), float(df_proc[col].median()))
    return pd.DataFrame([tx])


def show_home():
    """Homepage now shows a compact dashboard of model metrics & ROC curves."""
    st.title("Fraud Detection Teaching Tool")

    st.markdown("""
    **Welcome!**
    Financial fraud costs the global economy billions annually, and detecting it is both critical and challenging. Fraudulent transactions are rare but damaging, often mimicking normal behaviour to avoid detection.
    In this tutorial app, you'll learn how fraud detection works by:

    1. Trying to spot fraud yourself

    2. Seeing how machine learning models make decisions

    3. Understanding how a GAN can be trained to generate convincing fake fraud

    Use the navigation panel to explore and learn by doing.
    """)

    st.subheader("Model Performance Overview")

    for model_name, vals in MODEL_METRICS.items():
        with st.container():
            st.markdown(f"### {model_name}")
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("F1",       f"{vals['F1']:.2f}")
            c2.metric("Precision", f"{vals['Precision']:.2f}")
            c3.metric("Recall",    f"{vals['Recall']:.2f}")
            c4.metric("Accuracy",  f"{vals['Accuracy']:.2f}")
            c5.metric("Brier",     f"{vals['Brier']:.2f}")

            roc_path = vals.get("ROC")
            if roc_path and os.path.exists(roc_path):
                st.image(roc_path, caption="ROC Curve", width=800)


            st.markdown("---")


def show_tutorial(raw_df, cust_map, merch_map):
    """Original tutorial (unchanged)"""
    st.title("Guessing Tutorial")
    st.markdown("""
    In this challenge, four transactions are real, and one is fraudulent. Can you spot the odd one out?

    This is how fraud detection often begins: with human analysts scanning patterns in amounts, merchants, categories, and customer behaviours.
    But fraudsters get smarter—patterns evolve—so relying only on human intuition isn't enough.

    After each round, you'll see whether you were right. Try to beat your own score and observe what makes fraud stand out to you.
    """)
    state = st.session_state
    state.setdefault('score', 0)
    state.setdefault('total', 0)
    state.setdefault('next_round', True)
    state.setdefault('answered', False)
    if state.next_round:
        legit = raw_df[raw_df.fraud==0].sample(4)
        fraud = raw_df[raw_df.fraud==1].sample(1)
        pool = pd.concat([legit,fraud]).sample(frac=1).reset_index(drop=True)
        state.pool = pool
        state.next_round = False
        state.answered = False
        state.tutorial_choice = 1
    pool = state.pool
    disp = pd.DataFrame({
        "Customer":[cust_map[c] for c in pool.customer],
        "Age":[f"{int(str(a).strip('\"\'\"'))*10}s" for a in pool.age],
        "Gender":["Male" if g=="M" else "Female" for g in pool.gender],
        "Merchant":[merch_map[m] for m in pool.merchant],
        "Category":[cat.replace("es_","") for cat in pool.category],
        "Amount (£)":[round(v,2) for v in pool.amount]
    })
    st.table(disp)
    choice = st.radio("Which row is fraud?", range(0,5),
                      format_func=lambda x: f"Row {x}", key="tutorial_choice")
    choice = choice + 1
    if not state.answered:
        if st.button("Submit Guess"):
            idx = int(pool[pool.fraud==1].index[0])
            state.total += 1
            if choice-1 == idx:
                state.score += 1
                st.success("✅ Correct!")
            else:
                st.error(f"❌ Wrong. It was Row {idx}.")
            st.info(f"Score: {state.score}/{state.total}")
            state.answered = True
    else:
        if st.button("Next Round"):
            state.next_round = True


def show_model_demo(raw_df, df_proc, features, col_map,
                    cust_map, merch_map, inv_cust, inv_merch,
                    m2c, m2z, models, explainers):
    """Original model demo (unchanged)"""
    st.title("Model Demo & Explanations")
    st.markdown("""
    Here, you can simulate a fraud detection model at work. Build a custom transaction and choose a machine learning model to classify it.

    Prediction shows whether the transaction is likely fraud, along with the probability.

    Explanation visualizes how each feature (like amount, zip code, or age) influences the decision.

    Fraud detection models are trained on thousands (or millions) of labelled transactions. They learn patterns and use those to assign a fraud probability to new data.

    Real-world models use ensembles, neural nets, and explainability tools (like SHAP) to improve accuracy and transparency.
    """)
    tx_df = create_transaction_input(raw_df, df_proc, col_map,
                                     cust_map, merch_map, inv_cust, inv_merch,
                                     m2c, m2z)
    st.subheader("Transaction")
    st.dataframe(tx_df)
    model_choice = st.sidebar.selectbox("Model", list(models.keys()), key="demo_model")
    method       = st.sidebar.selectbox("Explanation", ["SHAP"], key="demo_method")
    if not st.sidebar.button("Predict & Explain"):
        return
    raw = tx_df.iloc[0].to_dict()
    proc = preprocess_single_transaction(raw, col_map)
    X = np.array(proc, dtype=np.float32).reshape(1,-1)
    mdl = models[model_choice]
    if model_choice=="Neural Network":
        p = float(mdl.predict(X)[0]); probs=[1-p,p]; pred=int(p>0.5)
    else:
        probs = mdl.predict_proba(X)[0]; pred=int(mdl.predict(X)[0])
    if pred:
        st.error(f"Fraud (P={probs[1]:.1%})")
    else:
        st.success(f"Legitimate (P={probs[0]:.1%})")
    st.subheader(f"{method} Explanation for {model_choice}")
    if method=="SHAP":
        key_map = {
            'Random Forest':'random_forest',
            'Neural Network':'neural_network',
            'Voting Ensemble':'voting',
            'Adaboost Ensemble':'adaboost',
            'Stacking Ensemble':'stacking'
        }
        expl = explainers[key_map[model_choice]]
        tabs = st.tabs(["Force","Waterfall","Bar"])
        for tab,pt in zip(tabs,["force","waterfall","bar"]):
            with tab:
                fig = explain_prediction(
                    model_name    = key_map[model_choice],
                    transaction   = X,
                    feature_names = features,
                    explainers    = {key_map[model_choice]:expl},
                    plot_type     = pt
                )
                st.pyplot(fig)

def show_gan_guess(raw_df: pd.DataFrame, df_proc: pd.DataFrame,
                   cust_map: dict, merch_map: dict, gan_gen: Generator,
                   noise_dim: int = 100):
    """A mini-game where the user acts as the GAN discriminator."""
    st.title("GAN Guess - Be the Discriminator!")
    st.markdown("""
    A Generative Adversarial Network (GAN) has learned what fraud looks like—and is now trying to fool you with synthetic transactions.

    Your job: act as the discriminator and spot the fake.

    This mirrors how GANs work:

    1. The generator creates fake fraud samples.

    2. The discriminator tries to distinguish fake from real.

    Over time, the generator gets better at deception, and the discriminator improves at detection—this adversarial loop leads to realistic synthetic data, which can help train more robust fraud models.

    Think you can outsmart the generator?

    """)

    state = st.session_state
    state.setdefault('gan_score', 0)
    state.setdefault('gan_total', 0)
    state.setdefault('gan_next', True)
    state.setdefault('gan_answered', False)

    if state.gan_next:
        # --- generate synthetic fraud vector & map to readable transaction ---
        with torch.no_grad():
            noise = torch.randn(1, noise_dim)
            vec   = gan_gen(noise).cpu().numpy()[0]
        synth_tx = _vector_to_transaction(vec, df_proc, raw_df, cust_map, merch_map)
        synth_df = pd.DataFrame([synth_tx])

        # --- sample 4 *real* legitimate transactions ---
        legit_df = raw_df[raw_df.fraud == 0].sample(4)
        legit_disp = pd.DataFrame({
            "Customer":   [cust_map[c] for c in legit_df.customer],
            "Age":        [f"{int(str(a).strip("\"'\""))*10}s" for a in legit_df.age],
            "Gender":     ["Male" if g == "M" else "Female" for g in legit_df.gender],
            "Merchant":   [merch_map[m] for m in legit_df.merchant],
            "Category":   [cat.replace("es_", "") for cat in legit_df.category],
            "Amount (£)": [round(v,2) for v in legit_df.amount],
            "fraud":      legit_df.fraud  # all zeros
        })
        synth_df = synth_df.drop(columns=["Step"])

        # --- combine & shuffle ---
        pool = pd.concat([legit_disp, synth_df], ignore_index=True).sample(frac=1, random_state=np.random.randint(0,1e6)).reset_index(drop=True)
        state.gan_pool = pool
        state.gan_next = False
        state.gan_answered = False
        state.gan_choice = 1

    pool = state.gan_pool
    # Hide the fraud column when displaying
    disp = pool.drop(columns=["fraud"])
    st.table(disp)

    choice = st.radio("Which row is the *synthetic* fraud?", range(0,5),
                      format_func=lambda i: f"Row {i}", key="gan_choice")
    choice = choice + 1
    if not state.gan_answered:
        if st.button("Submit Guess", key="gan_submit"):
            # the generated row is the one with fraud==1
            idx = int(pool[pool.fraud == 1].index[0])
            state.gan_total += 1
            if choice - 1 == idx:
                state.gan_score += 1
                st.success("Spot on! You fooled the generator.")
            else:
                st.error(f"Not quite - the synthetic transaction was Row {idx}.")
            st.info(f"Score: {state.gan_score}/{state.gan_total}")
            state.gan_answered = True
    else:
        if st.button("Next Round", key="gan_next_round"):
            state.gan_next = True


def main():
    st.set_page_config(page_title="Fraud Detection Tutorial", layout="wide")

    page = st.sidebar.radio("Navigate to", ["Home", "Tutorial", "Model Demo", "GAN Guess"], index=0)

    (raw_df, df_proc, features, col_map,
     cust_map, merch_map, inv_cust, inv_merch, m2c, m2z) = load_and_preprocess_data()

    if page == "Home":
        show_home()
    elif page == "Tutorial":
        show_tutorial(raw_df, cust_map, merch_map)
    elif page == "Model Demo":
        models     = load_models()
        explainers = load_explainers()
        show_model_demo(raw_df, df_proc, features, col_map,
                        cust_map, merch_map, inv_cust, inv_merch,
                        m2c, m2z, models, explainers)
    else:  # GAN Guess
        # lazily load the generator only when this page is used to avoid GPU overhead
        noise_dim = 100
        out_dim   = len([c for c in df_proc.columns if c != 'fraud'])
        gen_path  = os.path.join('models', 'gan_generator.pth')
        if not os.path.exists(gen_path):
            st.error("GAN generator weights not found - please train the GAN and place 'gan_generator.pth' into the models/ folder.")
            return
        gan_gen = load_gan_generator(gen_path, input_dim=noise_dim, output_dim=out_dim)
        show_gan_guess(raw_df, df_proc, cust_map, merch_map, gan_gen, noise_dim)


if __name__ == "__main__":
    main()
