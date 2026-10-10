import pytest

from src.data import load_dataset, split_dataset


@pytest.fixture(scope="module")
def df():
    return load_dataset()


@pytest.fixture(scope="module")
def splits(df):
    return split_dataset(df)


def test_schema_and_labels(df):
    assert list(df.columns) == ["label", "text"]
    assert set(df["label"].unique()) == {0, 1}
    assert df["text"].is_unique
    assert not df["text"].isna().any()


def test_no_control_characters(df):
    assert not df["text"].str.contains(r"[\x80-\x9f]", regex=True).any()


def test_split_sizes(df, splits):
    total = len(splits.X_train) + len(splits.X_val) + len(splits.X_test)
    assert total == len(df)
    assert len(splits.X_test) == pytest.approx(0.2 * len(df), abs=2)
    assert len(splits.X_val) == pytest.approx(0.1 * len(df), abs=2)


def test_splits_do_not_overlap(splits):
    train, val, test = set(splits.X_train), set(splits.X_val), set(splits.X_test)
    assert not train & val
    assert not train & test
    assert not val & test


def test_stratification(df, splits):
    overall = df["label"].mean()
    for y in (splits.y_train, splits.y_val, splits.y_test):
        assert y.mean() == pytest.approx(overall, abs=0.01)


def test_split_is_deterministic(df):
    a, b = split_dataset(df), split_dataset(df)
    assert (a.X_test == b.X_test).all()
    assert (a.y_test == b.y_test).all()