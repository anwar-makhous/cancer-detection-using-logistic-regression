import numpy as np
from flask import Flask, render_template, url_for, request
import pandas as pd
from scipy.linalg._solve_toeplitz import float64
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.metrics import jaccard_similarity_score

app = Flask(__name__)

cell_df = pd.read_csv("data/cell_samples.csv")
cell_df = cell_df[pd.to_numeric(cell_df['BareNuc'], errors='coerce').notnull()]
cell_df['BareNuc'] = cell_df['BareNuc'].astype('int')
feature_df = cell_df[['Clump', 'UnifSize', 'UnifShape', 'MargAdh', 'SingEpiSize', 'BareNuc', 'BlandChrom', 'NormNucl', 'Mit']]
X = np.asarray(feature_df)
cell_df['Class'] = cell_df['Class'].astype('int')
y = np.asarray(cell_df['Class'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
sscaler = preprocessing.StandardScaler()
X_test1 = sscaler.fit(X_test).transform(X_test)
X_train = sscaler.fit(X_train).transform(X_train)
LR = LogisticRegression(C=0.01, solver='liblinear').fit(X_train, y_train)


@app.route('/')
def index():
    return render_template('index.html')



@app.route('/lab', methods=['POST'])
def lab():
    if request.method == 'POST':
        yhat = LR.predict(X_test1)
        acc = jaccard_similarity_score(y_test, yhat)
        return render_template('results.html', XX=X_test, X=X_test1, ytest=y_test, prediction=yhat, acc=acc)


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':

        f1 = float64(request.form['f1'])
        f2 = float64(request.form['f2'])
        f3 = float64(request.form['f3'])
        f4 = float64(request.form['f4'])
        f5 = float64(request.form['f5'])
        f6 = float64(request.form['f6'])
        f7 = float64(request.form['f7'])
        f8 = float64(request.form['f8'])
        f9 = float64(request.form['f9'])
        features = np.asarray([[f1, f2, f3, f4, f5, f6, f7, f8, f9]])
        my_scaler = preprocessing.StandardScaler()
        features = my_scaler.fit(X).transform(features)
        my_prediction = LR.predict(features)

        return render_template('myresult.html', prediction=my_prediction)


if __name__ == '__main__':
    app.run(debug=True)
