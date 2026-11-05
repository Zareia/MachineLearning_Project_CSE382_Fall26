import random

import numpy as np
import pandas as pd
import pytest

from src.data.load import load_raw, validate_schema
from src.utils.config import load_config, set_seed


def test_original_h2_and_team_config():
    df = load_raw()
    validate_schema(df)
    assert df.shape == (79330, 31)
    assert load_config()["random_seed"] == 42
    assert df["Agent"].dropna().map(lambda x: isinstance(x, str)).all()


def test_reject_other_schema_or_population():
    with pytest.raises(ValueError, match="schema"):
        validate_schema(pd.DataFrame({"hotel": ["City Hotel"]}))
    with pytest.raises(ValueError, match="H2 rows"):
        validate_schema(load_raw().iloc[:3])


def test_reproducible_random_seeds():
    set_seed(42)
    first = random.random(), np.random.random()
    set_seed(42)
    assert first == (random.random(), np.random.random())
