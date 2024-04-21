import pickle
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, TransformerMixin

class BinaryClassifier(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.model = LogisticRegression(max_iter=1000,penalty='l2', C=0.000003)  

    def fit(self, X, y):
        self.model.fit(X, y)
        return self
    
    def transform(self, X):
        return self.model.predict_proba(X)

    def predict(self, X):
        return self.model.predict(X)
    
with open('combined_pipeline.pkl', 'rb') as file:
    loaded_combined_pipeline = pickle.load(file)

arr = ['acquired','closed','ipo','operating']
def prediction(input_arr):
    np_arr = np.array(input_arr).reshape(-1,4)
    df_d = pd.DataFrame(np_arr,columns=['founded_at','funding_rounds','funding_total_usd','country_code'])
    df_d['founded_at'] = pd.to_datetime(df_d['founded_at']).dt.year
    y = loaded_combined_pipeline.predict(df_d)[0]
    res = arr[y]
    return res

x = [['17-11-2008','2444','33333','USA']]

print(prediction(x))