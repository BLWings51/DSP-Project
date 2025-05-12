import pandas as pd
import numpy as np
import pytest

from data_loader import (
    load_transaction_data,
    preprocess_data,
    split_data,
    KerasBinaryClassifier
)
from utils import preprocess_single_transaction


@pytest.fixture
def small_csv(tmp_path):
    # Small CSV for load_transaction_data
    p = tmp_path / "test.csv"
    df = pd.DataFrame({
        'step': [0, 1],
        'customer': ["C1", "C2"],
        'age': ["1", "2"],
        'gender': ["M", "F"],
        'zipcodeOri': ["28007", "28008"],
        'merchant': ["M1", "M2"],
        'zipMerchant': ["28007", "28008"],
        'category': ["es_a", "es_b"],
        'amount': [10.0, 20.0],
        'fraud': [0, 1]
    })
    df.to_csv(p, index=False)
    return str(p)



def test_load_transaction_data(small_csv):
    df = load_transaction_data(file_path=small_csv)
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (2, 10)
    expected_cols = [
        'step', 'customer', 'age', 'gender',
        'zipcodeOri', 'merchant', 'zipMerchant',
        'category', 'amount', 'fraud'
    ]
    assert list(df.columns) == expected_cols


def test_preprocess_data_and_mapping():
    df = pd.DataFrame({
        'step': [0, 1, 2],
        'customer': ['C1', 'C2', 'C3'],
        'age': ["1", "U", "'3'"],
        'gender': ['M', 'F', 'M'],
        'zipcodeOri': ['28007', '28007', '28007'],
        'merchant': ['M1', 'M1', 'M1'],
        'zipMerchant': ['28007', '28007', '28007'],
        'category': ['es_a', 'es_b', 'es_a'],
        'amount': [5.0, 15.0, 25.0],
        'fraud': [0, 1, 0]
    })
    proc, col_map = preprocess_data(df)

    # Check age normalization: raw ages [1,2,3] → median=2 → (2-1)/(3-1)=0.5
    assert np.isclose(proc.loc[1, 'age'], 0.5)

    # Numeric columns remain in [0,1]
    for col in ['step', 'age', 'amount']:
        assert 0.0 <= proc[col].min() <= proc[col].max() <= 1.0

    for col in [
        'customer', 'gender', 'zipcodeOri',
        'merchant', 'zipMerchant', 'category'
    ]:
        assert col in col_map
        assert proc[col].nunique() == len(col_map[col])


def test_preprocess_single_transaction_roundtrip():
    df = pd.DataFrame({
        'step': [0, 1],
        'customer': ['C1', 'C2'],
        'age': ["1", "2"],
        'gender': ['M', 'F'],
        'zipcodeOri': ['28007', '28008'],
        'merchant': ['M1', 'M2'],
        'zipMerchant': ['28007', '28008'],
        'category': ['es_a', 'es_b'],
        'amount': [10.0, 20.0],
        'fraud': [0, 1]
    })
    proc, col_map = preprocess_data(df)
    raw_row = df.iloc[1].to_dict()
    arr = preprocess_single_transaction(raw_row, col_map)

    flat = np.array(arr).flatten()

    # Length matches number of columns in raw_row
    assert flat.shape[0] == len(raw_row)

    # All entries are numeric types (no strings)
    for v in flat:
        assert isinstance(v, (int, float, np.integer, np.floating))


def test_split_data_stratify():
    df = pd.DataFrame({
        'x': list(range(100)),
        'fraud': [0]*90 + [1]*10
    })
    X_train, X_test, y_train, y_test = split_data(
        df, target_column='fraud',
        test_size=0.2, random_state=0
    )
    assert len(X_train) == 80
    assert len(X_test) == 20
    # 10% fraud in each
    assert y_train.sum() == 8
    assert y_test.sum() == 2


def test_keras_binary_classifier_dummy():
    class DummyModel:
        def predict(self, X, verbose=0):
            # always returns 0.7
            return np.full((X.shape[0],), 0.7)

    dummy = DummyModel()
    clf = KerasBinaryClassifier(model=dummy, n_features=3)
    clf.fit(np.zeros((1, 3)))

    X = np.arange(6).reshape(2, 3)
    proba = clf.predict_proba(X)
    assert proba.shape == (2, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)

    pred = clf.predict(X)
    assert np.all(pred == 1)
