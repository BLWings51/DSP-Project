# tests/test_integration.py

import numpy as np
import pandas as pd
import pytest

from data_loader import load_transaction_data, preprocess_data, split_data
from utils import preprocess_single_transaction, KerasBinaryClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

@pytest.fixture
def small_csv(tmp_path):
    # Create a tiny 2-row CSV: one legit, one fraud
    df = pd.DataFrame({
        'step':        [0,    1],
        'customer':    ['C1', 'C2'],
        'age':         ['1',  '2'],
        'gender':      ['M',  'F'],
        'zipcodeOri':  ['28007','28008'],
        'merchant':    ['M1', 'M2'],
        'zipMerchant': ['28007','28008'],
        'category':    ['es_a','es_b'],
        'amount':      [10.0, 20.0],
        'fraud':       [0,    1]
    })
    path = tmp_path / "test.csv"
    df.to_csv(path, index=False)
    return str(path)

def test_preprocess_data_and_mapping(small_csv):
    # Load and preprocess
    df = load_transaction_data(file_path=small_csv)
    proc, col_map = preprocess_data(df)

    # 1) All non-fraud columns in proc are normalized to [0,1]
    for col in proc.columns:
        if col == 'fraud':
            continue
        mn, mx = proc[col].min(), proc[col].max()
        assert 0.0 <= mn <= mx <= 1.0, f"{col} out of [0,1]: {mn},{mx}"

    # 2) column_mapping must have exactly age_median + the object-dtype columns in the raw CSV
    raw = pd.read_csv(small_csv)
    obj_cols = raw.select_dtypes(include=['object']).columns.tolist()
    obj_cols = [c for c in obj_cols if c != 'fraud']
    expected_keys = set(obj_cols + ['age_median'])
    assert set(col_map.keys()) == expected_keys

    # 3) Each mapping (except age_median) is a dict mapping original -> int
    for k,v in col_map.items():
        if k == 'age_median':
            assert isinstance(v, float)
        else:
            assert isinstance(v, dict)
            # all values in that dict are ints
            assert all(isinstance(code, (int, np.integer)) for code in v.values())

def test_preprocess_single_transaction_roundtrip(small_csv):
    # Round-trip each row through preprocess_single_transaction
    raw_df = pd.read_csv(small_csv)
    proc_df, col_map = preprocess_data(raw_df)

    for idx, raw_row in raw_df.iterrows():
        arr = preprocess_single_transaction(raw_row.to_dict(), col_map).flatten()
        # must have exactly one entry per column in proc_df
        assert arr.shape[0] == proc_df.shape[1]

        # check each column
        for i, col in enumerate(proc_df.columns):
            got = arr[i]
            raw_val = raw_row[col]
            if col == 'age':
                # age was taken raw (no normalization) in preprocess_single_transaction
                # handle quoted or 'U' if needed (not present here)
                exp_age = float(str(raw_val).replace("'", "")) \
                          if str(raw_val).replace("'", "") != 'U' \
                          else col_map['age_median']
                assert got == exp_age
            elif col in ('step', 'amount', 'zipcodeOri', 'zipMerchant'):
                # these numeric fields pass through unchanged
                # pandas read zipcodeOri, zipMerchant as ints
                assert got == raw_val
            elif col == 'fraud':
                assert got == raw_val
            else:
                # categorical columns: must match the mapping
                mapping = col_map[col]
                assert got == mapping[raw_val]

def test_split_data_small_raises(small_csv):
    # With only two samples (one per class), stratify will fail
    df = load_transaction_data(file_path=small_csv)
    proc, _ = preprocess_data(df)
    with pytest.raises(Exception) as ei:
        split_data(proc, target_column='fraud', test_size=0.5, random_state=0)
    msg = str(ei.value)
    assert 'least populated class' in msg or 'too few' in msg

def test_voting_classifier_with_keras_wrapper(small_csv):
    # Build X,y
    df = load_transaction_data(file_path=small_csv)
    proc, _ = preprocess_data(df)
    X = proc.drop('fraud', axis=1).values
    y = proc['fraud'].values

    # Train a tiny RandomForest
    rf = RandomForestClassifier(n_estimators=3, random_state=0)
    rf.fit(X, y)

    # Train a minimal Keras binary model
    nn_model = Sequential([Dense(1, activation='sigmoid', input_shape=(X.shape[1],))])
    nn_model.compile(loss='binary_crossentropy', optimizer='adam')
    # wrap and no real training needed for predict_proba stub
    nn_clf = KerasBinaryClassifier(model=nn_model, n_features=X.shape[1])
    nn_clf.fit(X, y)

    # Ensemble
    vc = VotingClassifier([('rf', rf), ('nn', nn_clf)], voting='soft')
    vc.fit(X, y)

    probs = vc.predict_proba(X)
    preds = vc.predict(X)

    # shapes and valid ranges
    assert probs.shape == (2, 2)
    assert preds.shape == (2,)
    assert np.all((probs >= 0) & (probs <= 1))
    assert set(preds) <= {0, 1}
