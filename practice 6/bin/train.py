import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression
import joblib
from preprocessing import TextCleaner
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
dataset_eng = pd.read_csv(BASE_DIR / 'reviews_en.csv')


print('Started train.py')

print('Initiliazing dataset')
X, y = dataset_eng[['review_text']], dataset_eng['score']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

features = FeatureUnion(
    transformer_list=[
        ('tfidfwords', TfidfVectorizer(max_features=70000, ngram_range=(1, 3), dtype=np.float32, sublinear_tf=True)),
        ('tfidfchars', TfidfVectorizer(max_features=30000, ngram_range=(3, 5), dtype=np.float32, sublinear_tf=True, analyzer='char_wb'))
    ]
)
pipeline = Pipeline(
    steps=[
        ('cleaning', TextCleaner()),
        ('features', features),
        ('regression', LogisticRegression(C=5, max_iter=3000))
    ]
)
print('Fitting model...')
pipeline.fit(X_train, y_train)

joblib.dump(pipeline, BASE_DIR / 'filter.joblib')