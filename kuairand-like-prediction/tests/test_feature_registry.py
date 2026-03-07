import pandas as pd
import pytest

from src.feature_registry import (
    FeatureRegistry,
    get_training_columns,
    validate_no_banned_columns,
    TARGET_COLUMN,
    ID_COLUMNS,
)


def test_get_training_columns_basics():
    df = pd.DataFrame({
        "user_id": [1, 2],
        "video_id": [10, 20],
        "is_like": [0, 1],
        "user_age": [25, 30],
        "video_category": ["cat1", "cat2"],
    })
    cols = get_training_columns(df)
    # should not include ids or target
    assert "user_id" not in cols
    assert "video_id" not in cols
    assert "is_like" not in cols
    # should include the feature columns
    assert "user_age" in cols
    assert "video_category" in cols


def test_validate_detects_banned_column():
    df = pd.DataFrame({
        "user_id": [1],
        "video_id": [2],
        "is_like": [0],
        "clicked": [1],
    })
    with pytest.raises(ValueError) as exc:
        validate_no_banned_columns(df)
    assert "clicked" in str(exc.value)


def test_missing_columns_are_handled_gracefully():
    # missing many expected columns; policy functions should not crash
    df = pd.DataFrame({"is_like": [0, 1]})
    cols = get_training_columns(df)
    assert isinstance(cols, list)
    # validate should be fine since no banned columns present
    validate_no_banned_columns(df)
