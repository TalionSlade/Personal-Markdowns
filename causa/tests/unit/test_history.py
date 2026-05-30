"""Tests for the pandas-backed ObservationHistory adapter."""

from __future__ import annotations

import pytest

from causa.adapters.history.pandas_history import PandasObservationHistory


def test_empty_history_returns_empty_dataframe():
    h = PandasObservationHistory(schema=("a", "b"))
    df = h.as_dataframe()
    assert df.shape == (0, 2)


def test_append_records_observation():
    h = PandasObservationHistory()
    h.append({"x": 1, "y": "ok"})
    df = h.as_dataframe()
    assert df.shape == (1, 2)


def test_schema_mismatch_rejected():
    h = PandasObservationHistory()
    h.append({"x": 1, "y": "ok"})
    with pytest.raises(ValueError):
        h.append({"x": 1})  # missing y
    with pytest.raises(ValueError):
        h.append({"x": 1, "y": "ok", "z": 2})  # extra z


def test_window_returns_last_k_rows():
    h = PandasObservationHistory()
    for i in range(5):
        h.append({"x": i})
    df = h.window(k=2)
    assert df["x"].tolist() == [3, 4]


def test_window_rejects_non_positive_k():
    h = PandasObservationHistory()
    h.append({"x": 1})
    with pytest.raises(ValueError):
        h.window(k=0)


def test_clear_resets_history():
    h = PandasObservationHistory()
    h.append({"x": 1})
    h.clear()
    assert h.n_observations == 0
