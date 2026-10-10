# SMS Spam Detector

End-to-end spam classifier for English SMS: data pipeline, three models compared on the same splits, validation-tuned decision threshold, tests, and a live demo.

**Live demo:** https://sms-spam-detector-czk4kgsl9ayaczxsutbjpa.streamlit.app/

## Results

Held-out test set (20%), spam is the positive class. Thresholds were chosen on a separate validation set (10%), never on the test set.

| Model | Spam precision | Spam recall | Spam F1 | Macro F1 | Errors |
|---|---|---|---|---|---|
| TF-IDF + Naive Bayes | 0.9828 | 0.8906 | 0.9344 | 0.9628 | 16 |
| TF-IDF + Logistic Regression | 0.9730 | 0.8438 | 0.9038 | 0.9456 | 23 |
| BiLSTM (TensorFlow/Keras) | 0.9256 | 0.8750 | 0.8996 | 0.9429 | 25 |

**Takeaway:** simple TF-IDF models match or beat the BiLSTM on this dataset. Retraining the BiLSTM changed its spam F1 by about 3 points, so differences of that size between models should not be read as a real ranking.

## Data

UCI SMS Spam Collection (Kaggle: `uciml/sms-spam-collection-dataset`). Duplicates are removed and non-ASCII characters are stripped so the saved model loads identically on any OS. Splits are stratified 70/10/20 (train/validation/test) with a fixed seed. Spam is a minority class (about 13%), handled with class weights (BiLSTM) and a tuned threshold.

## Project structure

```
src/        data, model, train, evaluate, baselines, predict, error_analysis
tests/      data and inference tests (pytest)
artifacts/  trained model, metrics, confusion matrix, error list
streamlit_app.py   demo UI
```

## Run locally

Python 3.12 (TensorFlow 2.21 does not install on Python 3.14).

```
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m src.train
python -m src.evaluate
python -m src.baselines
python -m pytest -q
streamlit run streamlit_app.py
```

## Limitations

- Trained on English SMS only. It is not intended for email, phishing, or non-English text.
- The dataset is small (roughly 650 spam messages after deduplication), so metrics have noticeable variance and the validation-chosen threshold is sensitive.
- The data is from a 2011-era collection and does not reflect current spam patterns.