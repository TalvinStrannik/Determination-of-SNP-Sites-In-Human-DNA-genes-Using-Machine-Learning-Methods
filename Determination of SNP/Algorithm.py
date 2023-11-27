import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

snp_data = pd.read_csv('snp_data.csv')

X = snp_data.drop('label', axis=1)
y = snp_data['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

def predict_snp(model, snp):
    snp_array = np.array(snp).reshape(1, -1)
    prediction = model.predict(snp_array)
    return prediction[0]

snp_example = [0, 1, 0, 0, 1, 1, 0, 1]
prediction_example = predict_snp(model, snp_example)
print(f'Prediction: {prediction_example}')