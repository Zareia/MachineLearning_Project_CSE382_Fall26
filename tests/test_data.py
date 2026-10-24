from src.data.load import EXPECTED_ROWS, SOURCE_COLUMNS, load_raw, validate_schema


def test_raw_h2_schema():
    df = load_raw()
    assert df.shape == (EXPECTED_ROWS, len(SOURCE_COLUMNS))
    validate_schema(df)
