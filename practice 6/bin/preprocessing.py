from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import re
import numpy as np

class TextCleaner(BaseEstimator, TransformerMixin):
    def __init__(self, punctuations=r"""@#!?+&*[]-%.:/();$=><|{}^'`"""):
        self.punctuations = punctuations

    def fit(self, X, y=None):
        return self

    def _to_1d_strings(self, X):
        if isinstance(X, pd.DataFrame):
            series = X.iloc[:, 0]
        elif isinstance(X, pd.Series):
            series = X
        else:
            series = pd.Series(np.asarray(X).ravel())
        return series.fillna("").astype(str).tolist()

    def transform(self, X):
        texts = self._to_1d_strings(X)
        trans_table = str.maketrans("", "", self.punctuations)

        converted = []
        for text in texts:
            text = text.lower()
            text = re.sub(r"[^a-zA-Z?.!,¿0-9/']", " ", text) # replace everything with space except (a-z, 0-9, A-Z, ".", "?", "!", ",")
            text = re.sub(r"[\n\r\t]", " ", text)
            text = re.sub(r"[’`]", "'", text)
            text = text.translate(trans_table)
            converted.append(text)
        return converted