import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

bnotes = pd.read_csv("PhishingData.csv")
X = bnotes.drop('Result', axis=1)
y = bnotes['Result']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

mlp = MLPClassifier(max_iter=500, activation='relu')
mlp.fit(X_train, y_train)

filename = 'finalized_model.pkl'
joblib.dump(mlp, filename)

pred = mlp.predict(X_test)

confusion_matrix(y_test, pred)
print(classification_report(y_test, pred))