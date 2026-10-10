import pytest

from src.config import ARTIFACTS_DIR
from src.predict import predict

pytestmark = pytest.mark.skipif(
    not (ARTIFACTS_DIR / "model.keras").exists()
    or not (ARTIFACTS_DIR / "metrics.json").exists(),
    reason="Run src.train and src.evaluate first",
)


def test_obvious_spam():
    result = predict(["WINNER!! You have won a $1000 prize. Call now to claim."])
    assert result[0]["label"] == "spam"


def test_obvious_ham():
    result = predict(["Ok, see you at 6 at the cafe."])
    assert result[0]["label"] == "ham"


def test_empty_list():
    assert predict([]) == []


def test_rejects_non_list():
    with pytest.raises(TypeError):
        predict("not a list")


def test_rejects_non_string_items():
    with pytest.raises(TypeError):
        predict(["ok", 123])


def test_output_schema():
    result = predict(["Free entry in a weekly competition, text WIN to 80086"])[0]
    assert set(result) == {"text", "label", "spam_probability"}
    assert result["label"] in {"spam", "ham"}
    assert 0.0 <= result["spam_probability"] <= 1.0


def test_preserves_order():
    texts = ["Ok, see you at 6 at the cafe.", "WINNER!! Call now to claim your prize."]
    result = predict(texts)
    assert [r["text"] for r in result] == texts