from flask import Flask, render_template, request
from flask_cors import CORS
from cp import prediction
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
app = Flask(__name__)
CORS(app) 
@app.route('/',methods=['get'])
def help():
    return "this is working"
@app.route('/predict', methods=['POST'])
def submit():
    data = request.json  # Access submitted form data
    # Process the data as neededs
    temp  = prediction([[data['founded_at'],data['closed_at'],data['funding_rounds'],data['funding_total_usd'],data['country_code']]])
    
    return temp

if __name__ == '__main__':
    app.run(debug=True)
