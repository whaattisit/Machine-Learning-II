import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression
import joblib
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score
from preprocessing import TextCleaner
from pathlib import Path
import mlflow
import mlflow.sklearn
import os
from mlflow.models import infer_signature


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "filter.joblib"
dataset_eng = pd.read_csv(BASE_DIR / 'reviews_en.csv')

EXPERIMENT_NAME = "review-scoring"
REGISTERED_MODEL_NAME = "review-scorer"

PARAMS = {
    "test_size" : 0.2,
    "random_state" : 42,
    "word_max_features" : 70000,
    "word_ngram_range" : "1-3",
    "char_max_features" : 30000,
    "char_ngram_range" : "3-5",
    "C" : 5,
    "max_iter" : 3000 
}

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
mlflow.set_experiment(EXPERIMENT_NAME)

print('Started train.py')

print('Initiliazing dataset')
X, y = dataset_eng[['review_text']], dataset_eng['score']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=PARAMS["test_size"], random_state=PARAMS["random_state"])

features = FeatureUnion(
    transformer_list=[
        ('tfidfwords', TfidfVectorizer(max_features=PARAMS["word_max_features"], ngram_range=(1, 3), dtype=np.float32, sublinear_tf=True)),
        ('tfidfchars', TfidfVectorizer(max_features=PARAMS["char_max_features"], ngram_range=(3, 5), dtype=np.float32, sublinear_tf=True, analyzer='char_wb'))
    ]
)
pipeline = Pipeline(
    steps=[
        ('cleaning', TextCleaner()),
        ('features', features),
        ('regression', LogisticRegression(C=PARAMS["C"], max_iter=PARAMS["max_iter"]))
    ]
)

with mlflow.start_run(run_name="tfidf-logreg") as run:
    mlflow.log_params(PARAMS)

    print('Fitting model...')
    pipeline.fit(X_train, y_train)

    y_proba = pipeline.predict_proba(X_test)
    y_pred = np.rint(y_proba @ pipeline.classes_)

    metrics = {
        "accuracy" : accuracy_score(y_test, y_pred),
        "MAE" : mean_absolute_error(y_test, y_pred),
        "R_2" : r2_score(y_test, y_pred)
    }
    mlflow.log_metrics(metrics)
    print(metrics)

    signature = infer_signature(X_test.head(5), y_pred[:5])
    model_info = mlflow.sklearn.log_model(
        sk_model=pipeline,
        name="model",
        signature=signature,
        input_example=X_test.head(3),
        serialization_format="cloudpickle",
        registered_model_name=REGISTERED_MODEL_NAME,
    )

    joblib.dump(pipeline, MODEL_PATH)
    mlflow.log_artifact(str(MODEL_PATH))

    print(f"RUN ID: {run.info.run_id}")
    print(f"MODEL URI: {model_info.model_uri}")